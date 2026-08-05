//! Recursive goal evaluation with probes and cycle detection.
//!
//! Phase 2 of the solver rewrite: instead of returning nested obligations
//! to the flat fulfillment loop, goals are resolved recursively within
//! nested transactions, with explicit cycle detection and coinductive
//! classification.
//!
//! ## Cycle detection
//!
//! Goal identity is determined by a non-allocating structural canonical
//! key (`CanonTy`), built on the stack by recursively resolving inference
//! variable bindings through the type structure.  This avoids arena
//! allocation during cycle detection and ensures that `TypeId` volatility
//! does not affect key equality.
//!
//! ## Known limitations
//!
//! ### Alpha-equivalence is not canonicalized
//!
//! `CanonTy` preserves concrete binder indices (`param_index` values)
//! rather than using de Bruijn depth.  This means alpha-equivalent
//! quantified types with different binder indices (e.g. `∀X0. X0` versus
//! `∀X1. X1`) produce different `GoalKey` values and are not recognised
//! as the same goal by cycle detection.
//!
//! This is a completeness limitation, not a soundness bug: it can only
//! cause a genuinely cyclic goal to be missed and fall through to a
//! depth-limit overflow.  It cannot cause an obligation to be falsely
//! proven (false coinductive success) or falsely rejected.
//!
//! If alpha-complete cycle detection is needed in the future (e.g. for
//! higher-ranked recursive obligations), the canonical key should be
//! switched to de Bruijn depth representation, where bound variables
//! are identified by their nesting depth rather than a global index.

use crate::ast::OverflowPolicy;
use crate::hir::traits::solver::delegate::SolverDelegate;
use crate::hir::traits::solver::eval_ctxt::EvalCtxt;
use crate::hir::traits::solver::obligation::{
    BuiltinImplSource, ImplSource, Obligation, Predicate, SolveError,
};
use crate::hir::traits::solver::search_graph::{
    AvailableDepth, CanonicalPredicate, CanonicalResponse, FIXPOINT_STEP_LIMIT,
    GoalKey as SgGoalKey, GoalKind as SgGoalKind, PathKind, SearchGraph, canonicalize_goal_key,
    canonicalize_response,
};
use crate::hir::traits::solver::select::MAX_RECURSION_DEPTH;
use crate::hir::types::{AdtKind, DefId, TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;

// ── Canonical type for cycle detection ──

/// A non-allocating structural canonical type for goal identity.
///
/// Built on the stack by recursively resolving inference variable bindings.
/// Does NOT allocate into the arena, so it is safe for the read-only cycle
/// detection path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CanonTy {
    // ── Primitives ──
    Int {
        bits: u8,
        signed: bool,
        overflow_policy: OverflowPolicy,
    },
    UInt {
        bits: u8,
        overflow_policy: OverflowPolicy,
    },
    Float {
        bits: u8,
    },
    Bool,
    Char,
    Byte,
    USize,
    Unit,
    Never,
    // ── Variables (not resolved further) ──
    InferVar {
        id: usize,
    },
    GenericParam {
        index: usize,
    },
    SkolemVar {
        id: usize,
        universe_num: usize,
    },
    // ── Composites ──
    Adt {
        kind: AdtKind,
        def_id: DefId,
        args: Vec<CanonTy>,
    },
    Tuple(Vec<CanonTy>),
    Ref {
        ty: Box<CanonTy>,
        mutable: bool,
    },
    Pointer(Box<CanonTy>),
    Ptr {
        size: Box<CanonTy>,
        pointee: Box<CanonTy>,
    },
    Fn {
        params: Vec<CanonTy>,
        ret: Box<CanonTy>,
    },
    Array {
        elem: Box<CanonTy>,
        size: u64,
    },
    Slice(Box<CanonTy>),
    // Binders: the bound variable is preserved as GenericParam, not resolved.
    Forall {
        param_index: usize,
        body: Box<CanonTy>,
    },
    Exists {
        param_index: usize,
        base: Box<CanonTy>,
    },
    Mu {
        param_index: usize,
        body: Box<CanonTy>,
    },
    Nu {
        param_index: usize,
        body: Box<CanonTy>,
    },
    Poly {
        quantifiers: Vec<usize>,
        body: Box<CanonTy>,
    },
    Coproduct(Vec<CanonTy>),
    AssociatedType {
        trait_id: DefId,
        name: Symbol,
        self_ty: Box<CanonTy>,
    },
    DynTrait {
        traits: Vec<DefId>,
    },
    Rational {
        int_bits: u8,
        frac_bits: u8,
    },
    Error,
    // Sentinel for types we can't canonicalize (should not appear in practice).
    Unknown,
}

/// Canonicalize a type under current bindings, building a non-allocating
/// structural key on the stack.
///
/// `bound` tracks the `param_index` values of enclosing binders so that
/// bound `GenericParam` variables are preserved rather than resolved
/// through the binding table.
///
/// IMPORTANT: bound-variable detection must happen BEFORE binding resolution.
/// We use `ctx.get_raw(ty)` to inspect the raw type first, and only call
/// `ctx.resolve_binding(ty)` for non-bound variables.
fn canonicalize_type(ctx: &TypeContext, ty: TypeId, bound: &mut Vec<usize>) -> CanonTy {
    // Check the raw type for bound GenericParam BEFORE resolving bindings.
    // If we resolved first, a bound variable bound to a concrete type
    // (e.g. GenericParam(0) ↦ Int) would lose its binder identity.
    match ctx.get_raw(ty) {
        TypeData::GenericParam { index, .. } if bound.contains(index) => {
            return CanonTy::GenericParam { index: *index };
        }
        _ => {}
    }

    // Now resolve bindings for non-bound variables.
    let resolved = ctx.resolve_binding(ty);
    match ctx.get(resolved) {
        // ── Primitives ──
        TypeData::Int {
            bits,
            signed,
            overflow_policy,
        } => CanonTy::Int {
            bits: *bits,
            signed: *signed,
            overflow_policy: *overflow_policy,
        },
        TypeData::UInt {
            bits,
            overflow_policy,
        } => CanonTy::UInt {
            bits: *bits,
            overflow_policy: *overflow_policy,
        },
        TypeData::Float { bits } => CanonTy::Float { bits: *bits },
        TypeData::Bool => CanonTy::Bool,
        TypeData::Char => CanonTy::Char,
        TypeData::Byte => CanonTy::Byte,
        TypeData::USize => CanonTy::USize,
        TypeData::Unit => CanonTy::Unit,
        TypeData::Never => CanonTy::Never,
        TypeData::Error => CanonTy::Error,

        // ── Variables ──
        // Bound GenericParam: preserve without resolving.
        TypeData::GenericParam { index, .. } if bound.contains(index) => {
            CanonTy::GenericParam { index: *index }
        }
        TypeData::InferVar { id } => CanonTy::InferVar { id: *id },
        TypeData::GenericParam { index, .. } => CanonTy::GenericParam { index: *index },
        TypeData::SkolemVar { id, universe_num } => CanonTy::SkolemVar {
            id: *id,
            universe_num: *universe_num,
        },

        // ── Composites: recurse into children ──
        TypeData::Adt { kind, def_id, args } => CanonTy::Adt {
            kind: *kind,
            def_id: *def_id,
            args: args
                .iter()
                .map(|&a| canonicalize_type(ctx, a, bound))
                .collect(),
        },
        TypeData::Tuple { elems } => CanonTy::Tuple(
            elems
                .iter()
                .map(|&e| canonicalize_type(ctx, e, bound))
                .collect(),
        ),
        TypeData::Ref { ty, mutable } => CanonTy::Ref {
            ty: Box::new(canonicalize_type(ctx, *ty, bound)),
            mutable: *mutable,
        },
        TypeData::Pointer { ty } => CanonTy::Pointer(Box::new(canonicalize_type(ctx, *ty, bound))),
        TypeData::Ptr { size, pointee } => CanonTy::Ptr {
            size: Box::new(canonicalize_type(ctx, *size, bound)),
            pointee: Box::new(canonicalize_type(ctx, *pointee, bound)),
        },
        TypeData::Fn { params, ret } => CanonTy::Fn {
            params: params
                .iter()
                .map(|&p| canonicalize_type(ctx, p, bound))
                .collect(),
            ret: Box::new(canonicalize_type(ctx, *ret, bound)),
        },
        TypeData::Array { elem, size } => CanonTy::Array {
            elem: Box::new(canonicalize_type(ctx, *elem, bound)),
            size: *size,
        },
        TypeData::Slice { elem } => CanonTy::Slice(Box::new(canonicalize_type(ctx, *elem, bound))),

        // ── Binders: push the bound variable, recurse, pop ──
        TypeData::Forall {
            param_index, body, ..
        } => {
            bound.push(*param_index);
            let body = canonicalize_type(ctx, *body, bound);
            bound.pop();
            CanonTy::Forall {
                param_index: *param_index,
                body: Box::new(body),
            }
        }
        TypeData::Exists {
            param_index, base, ..
        } => {
            bound.push(*param_index);
            let base = canonicalize_type(ctx, *base, bound);
            bound.pop();
            CanonTy::Exists {
                param_index: *param_index,
                base: Box::new(base),
            }
        }
        TypeData::Mu {
            param_index, body, ..
        } => {
            bound.push(*param_index);
            let body = canonicalize_type(ctx, *body, bound);
            bound.pop();
            CanonTy::Mu {
                param_index: *param_index,
                body: Box::new(body),
            }
        }
        TypeData::Nu {
            param_index, body, ..
        } => {
            bound.push(*param_index);
            let body = canonicalize_type(ctx, *body, bound);
            bound.pop();
            CanonTy::Nu {
                param_index: *param_index,
                body: Box::new(body),
            }
        }
        TypeData::Poly { quantifiers, body } => {
            for (idx, _) in quantifiers {
                bound.push(*idx);
            }
            let body = canonicalize_type(ctx, *body, bound);
            for _ in quantifiers {
                bound.pop();
            }
            CanonTy::Poly {
                quantifiers: quantifiers.iter().map(|(idx, _)| *idx).collect(),
                body: Box::new(body),
            }
        }

        // ── Other composites ──
        TypeData::Coproduct { alternatives } => CanonTy::Coproduct(
            alternatives
                .iter()
                .map(|&a| canonicalize_type(ctx, a, bound))
                .collect(),
        ),
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => CanonTy::AssociatedType {
            trait_id: *trait_id,
            name: *name,
            self_ty: Box::new(canonicalize_type(ctx, *self_ty, bound)),
        },
        TypeData::DynTrait { traits } => CanonTy::DynTrait {
            traits: traits.clone(),
        },
        TypeData::Rational {
            int_bits,
            frac_bits,
        } => CanonTy::Rational {
            int_bits: *int_bits,
            frac_bits: *frac_bits,
        },
        // Regex types are opaque leaf types — they contain no nested TypeIds
        // and cannot participate in trait solving.
        TypeData::Regex { .. } => CanonTy::Unknown,
        TypeData::Type => CanonTy::Unknown,
        TypeData::Opaque { .. } => CanonTy::Unknown,
    }
}

// ── Goal key ── (now in search_graph.rs — GoalKey, GoalKind)

// ── Evaluate a goal and all its nested obligations recursively. ──
pub fn evaluate_goal<D: SolverDelegate>(
    ecx: &mut EvalCtxt<D>,
    goal: &Obligation,
) -> Result<ImplSource, SolveError> {
    // Fixpoint iteration: re-evaluate if the goal was ambiguous and the
    // search graph detected changes (e.g. new inference variables were
    // introduced that may resolve ambiguity in a nested goal).
    ecx.search_graph.begin_fixpoint();
    for iteration in 0..FIXPOINT_STEP_LIMIT {
        let result = evaluate_goal_inner(ecx, goal, 0);

        match &result {
            Ok(ImplSource::Deferred { .. }) if ecx.search_graph.has_changed() => {
                // The goal was ambiguous but changes occurred — retry.
                // Reset the changed flag for the next iteration.
                ecx.search_graph.begin_fixpoint();
                continue;
            }
            _ => return result,
        }
    }

    // Fixpoint limit reached — treat as overflow.
    Err(SolveError::Overflow {
        obligation: Box::new(goal.clone()),
        depth: FIXPOINT_STEP_LIMIT,
    })
}

/// Inner recursive evaluation with explicit depth, using SearchGraph
/// for cycle detection (now embedded in EvalCtxt).
fn evaluate_goal_inner<D: SolverDelegate>(
    ecx: &mut EvalCtxt<D>,
    goal: &Obligation,
    depth: usize,
) -> Result<ImplSource, SolveError> {
    if depth >= MAX_RECURSION_DEPTH {
        return Err(SolveError::Overflow {
            obligation: Box::new(goal.clone()),
            depth,
        });
    }

    // ── Cycle detection via SearchGraph (embedded in EvalCtxt) ──
    // This MUST happen BEFORE canonical cache lookup, because a cache hit
    // that bypasses cycle detection can cause infinite loops or staleness.
    let goal_key = SgGoalKey::from_obligation(goal, ecx.ctx());
    if let Some(ref key) = goal_key {
        match ecx.search_graph.try_entry(key, ecx.delegate) {
            Ok(()) => {
                // Push the goal onto the stack for evaluation.
                let step_kind = if let Some(trait_id) = key.trait_id {
                    if ecx.delegate.trait_is_coinductive(trait_id) {
                        crate::hir::traits::solver::search_graph::PathKind::Coinductive
                    } else {
                        crate::hir::traits::solver::search_graph::PathKind::Inductive
                    }
                } else {
                    match key.kind {
                        SgGoalKind::Sized | SgGoalKind::CopyLike | SgGoalKind::AutoTrait => {
                            crate::hir::traits::solver::search_graph::PathKind::Coinductive
                        }
                        _ => crate::hir::traits::solver::search_graph::PathKind::Inductive,
                    }
                };
                ecx.search_graph.push_goal(key.clone(), step_kind, true);
            }
            Err(path_kind) => {
                return ecx.search_graph.handle_cycle(key, goal, path_kind);
            }
        }
    }

    // ── Canonical cache lookup ──
    // AFTER cycle detection — cache hits can skip evaluation but must not
    // bypass the search graph's cycle/stack safety invariants.
    let max_universe = ecx.max_input_universe;
    let canonical_key = canonicalize_goal_key(goal, ecx.ctx(), max_universe);
    let cache_hit = canonical_key.as_ref().and_then(|ck| {
        ecx.search_graph
            .lookup_global_cache(ck, AvailableDepth(depth))
    });
    if let Some(cached_response) = cache_hit {
        // Pop the goal from the search graph (it was pushed above).
        if goal_key.is_some() {
            ecx.search_graph.pop_goal();
        }
        // Instantiate canonical variables with fresh inference variables.
        // SAFETY: The canonicalization guard rejects goals with free InferVars,
        // so `var_values` is always empty and `infer` is never actually used.
        let impl_source = crate::hir::traits::solver::search_graph::instantiate_canonical(
            &cached_response,
            None,
            ecx.ctx(),
        );
        return Ok(impl_source);
    }

    // ── Evaluate using the new builder-pattern probe API ──
    let result = ecx
        .probe(crate::hir::traits::solver::eval_ctxt::ProbeKind::TraitGoal)
        .enter(|ecx| {
            // Use the GoalKind-based assembly engine instead of delegate.select().
            // The assembly engine handles candidate assembly, winnowing, and
            // confirmation via the GoalKind trait (see assembly/mod.rs).
            let impl_source =
                crate::hir::traits::solver::assembly::assemble_and_evaluate_candidates(ecx, goal)?;

            if matches!(&impl_source, ImplSource::Deferred { .. }) {
                return Ok(impl_source);
            }

            for nested in impl_source.nested_obligations() {
                match evaluate_goal_inner(ecx, &nested, depth + 1) {
                    Ok(ImplSource::Deferred { stalled_on }) => {
                        return Ok(ImplSource::Deferred { stalled_on });
                    }
                    Ok(_) => {}
                    Err(e) => {
                        return Err(e);
                    }
                }
            }

            Ok(impl_source)
        });

    // Exit the search graph (pop from active path).
    if goal_key.is_some() {
        ecx.search_graph.pop_goal();
    }

    // ── Canonical cache store ──
    // SAFETY: The `predicate_has_unresolved_vars` check in
    // `canonicalize_goal_key` already ensures we only cache goals
    // with NO free inference variables.  All ImplSource variants
    // are safe to cache because their TypeId values are fully
    // resolved (no InferVars / SkolemVars).
    if let Some(ck) = canonical_key {
        // Collect GoalKeys from nested obligations for cycle detection.
        let nested_goals: Vec<SgGoalKey> = match &result {
            Ok(src) => src
                .nested_obligations()
                .iter()
                .filter_map(|o| SgGoalKey::from_obligation(o, ecx.ctx()))
                .collect(),
            Err(_) => vec![],
        };
        if let Some(response) =
            canonicalize_response(&result, ecx.ctx(), max_universe, depth, false, nested_goals)
        {
            ecx.search_graph.insert_global_cache(ck, response);
        }
    }

    result
}
