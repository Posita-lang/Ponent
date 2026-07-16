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
use crate::hir::traits::solver::obligation::{BuiltinImplSource, ImplSource, Obligation, Predicate, SolveError};
use crate::hir::traits::solver::select::{MAX_RECURSION_DEPTH, SelectionContext};
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
    Int { bits: u8, signed: bool, overflow_policy: OverflowPolicy },
    UInt { bits: u8, overflow_policy: OverflowPolicy },
    Float { bits: u8 },
    Bool,
    Char,
    Byte,
    USize,
    Unit,
    Never,
    // ── Variables (not resolved further) ──
    InferVar { id: usize },
    GenericParam { index: usize },
    SkolemVar { id: usize, universe_num: usize },
    // ── Composites ──
    Adt { kind: AdtKind, def_id: DefId, args: Vec<CanonTy> },
    Tuple(Vec<CanonTy>),
    Ref { ty: Box<CanonTy>, mutable: bool },
    Pointer(Box<CanonTy>),
    Ptr { size: Box<CanonTy>, pointee: Box<CanonTy> },
    Fn { params: Vec<CanonTy>, ret: Box<CanonTy> },
    Array { elem: Box<CanonTy>, size: u64 },
    Slice(Box<CanonTy>),
    // Binders: the bound variable is preserved as GenericParam, not resolved.
    Forall { param_index: usize, body: Box<CanonTy> },
    Exists { param_index: usize, base: Box<CanonTy> },
    Mu { param_index: usize, body: Box<CanonTy> },
    Nu { param_index: usize, body: Box<CanonTy> },
    Poly { quantifiers: Vec<usize>, body: Box<CanonTy> },
    Coproduct(Vec<CanonTy>),
    AssociatedType { trait_id: DefId, name: Symbol, self_ty: Box<CanonTy> },
    DynTrait { traits: Vec<DefId> },
    Rational { int_bits: u8, frac_bits: u8 },
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
        TypeData::Int { bits, signed, overflow_policy } => {
            CanonTy::Int { bits: *bits, signed: *signed, overflow_policy: *overflow_policy }
        }
        TypeData::UInt { bits, overflow_policy } => {
            CanonTy::UInt { bits: *bits, overflow_policy: *overflow_policy }
        }
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
        TypeData::SkolemVar { id, universe_num } => {
            CanonTy::SkolemVar { id: *id, universe_num: *universe_num }
        }

        // ── Composites: recurse into children ──
        TypeData::Adt { kind, def_id, args } => CanonTy::Adt {
            kind: *kind,
            def_id: *def_id,
            args: args.iter().map(|&a| canonicalize_type(ctx, a, bound)).collect(),
        },
        TypeData::Tuple { elems } => {
            CanonTy::Tuple(elems.iter().map(|&e| canonicalize_type(ctx, e, bound)).collect())
        }
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
            params: params.iter().map(|&p| canonicalize_type(ctx, p, bound)).collect(),
            ret: Box::new(canonicalize_type(ctx, *ret, bound)),
        },
        TypeData::Array { elem, size } => CanonTy::Array {
            elem: Box::new(canonicalize_type(ctx, *elem, bound)),
            size: *size,
        },
        TypeData::Slice { elem } => CanonTy::Slice(Box::new(canonicalize_type(ctx, *elem, bound))),

        // ── Binders: push the bound variable, recurse, pop ──
        TypeData::Forall { param_index, body, .. } => {
            bound.push(*param_index);
            let body = canonicalize_type(ctx, *body, bound);
            bound.pop();
            CanonTy::Forall { param_index: *param_index, body: Box::new(body) }
        }
        TypeData::Exists { param_index, base, .. } => {
            bound.push(*param_index);
            let base = canonicalize_type(ctx, *base, bound);
            bound.pop();
            CanonTy::Exists { param_index: *param_index, base: Box::new(base) }
        }
        TypeData::Mu { param_index, body, .. } => {
            bound.push(*param_index);
            let body = canonicalize_type(ctx, *body, bound);
            bound.pop();
            CanonTy::Mu { param_index: *param_index, body: Box::new(body) }
        }
        TypeData::Nu { param_index, body, .. } => {
            bound.push(*param_index);
            let body = canonicalize_type(ctx, *body, bound);
            bound.pop();
            CanonTy::Nu { param_index: *param_index, body: Box::new(body) }
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
        TypeData::Coproduct { alternatives } => {
            CanonTy::Coproduct(alternatives.iter().map(|&a| canonicalize_type(ctx, a, bound)).collect())
        }
        TypeData::AssociatedType { trait_id, name, self_ty } => CanonTy::AssociatedType {
            trait_id: *trait_id,
            name: name.clone(),
            self_ty: Box::new(canonicalize_type(ctx, *self_ty, bound)),
        },
        TypeData::DynTrait { traits } => CanonTy::DynTrait { traits: traits.clone() },
        TypeData::Rational { int_bits, frac_bits } => {
            CanonTy::Rational { int_bits: *int_bits, frac_bits: *frac_bits }
        }
    }
}

// ── Goal key ──

/// The predicate kind for cycle-detection identity.
///
/// Distinguishes `Trait`, `AutoTrait`, and `Sized` so that a cycle
/// detected for one predicate kind is not falsely identified as a
/// cycle for a different kind (which would cause false coinductive
/// success or false overflow).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum GoalKind {
    Trait,
    AutoTrait,
    Sized,
}

/// The resolved identity of a goal for cycle detection.
///
/// Uses `CanonTy` (non-allocating structural keys) instead of raw `TypeId`s,
/// so equality is collision-free and stable even when the type contains
/// unresolved inference variables (which `TypeContext::alloc` does not cache).
///
/// Includes `kind` to distinguish `Trait`, `AutoTrait`, and `Sized` —
/// these are semantically different even when they share the same
/// `trait_id`, `self_ty`, and `args`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GoalKey {
    kind: GoalKind,
    trait_id: Option<DefId>,
    self_ty: CanonTy,
    args: Vec<CanonTy>,
}

/// An active frame on the cycle-detection stack.
struct ActiveGoal {
    predicate: Predicate,
}

/// Evaluate a goal and all its nested obligations recursively.
pub fn evaluate_goal(
    selcx: &mut SelectionContext,
    goal: &Obligation,
) -> Result<ImplSource, SolveError> {
    let mut stack: Vec<ActiveGoal> = Vec::new();
    evaluate_goal_inner(selcx, goal, &mut stack, 0)
}

/// Inner recursive evaluation with explicit depth and cycle stack.
fn evaluate_goal_inner(
    selcx: &mut SelectionContext,
    goal: &Obligation,
    stack: &mut Vec<ActiveGoal>,
    depth: usize,
) -> Result<ImplSource, SolveError> {
    if depth >= MAX_RECURSION_DEPTH {
        return Err(SolveError::Overflow {
            obligation: Box::new(goal.clone()),
            depth,
        });
    }

    // ── Cycle detection ──
    // Compute the current goal's canonical key under current bindings.
    let goal_key = compute_goal_key(goal, selcx.ctx);
    let is_keyed = goal_key.is_some();

    // For each active frame, re-canonicalize under current bindings and compare.
    if let Some(ref key) = goal_key {
        for active in stack.iter() {
            let active_key = compute_goal_key_for_predicate(&active.predicate, selcx.ctx);
            if let Some(ref ak) = active_key {
                if ak == key {
                    match &goal.predicate {
                        Predicate::AutoTrait { .. } => {
                            return Ok(ImplSource::Auto { nested: vec![] });
                        }
                        Predicate::Sized { .. } => {
                            return Ok(ImplSource::Builtin(BuiltinImplSource::Sized));
                        }
                        _ => {
                            return Err(SolveError::Overflow {
                                obligation: Box::new(goal.clone()),
                                depth,
                            });
                        }
                    }
                }
            }
        }
    }

    if is_keyed {
        stack.push(ActiveGoal {
            predicate: goal.predicate.clone(),
        });
    }

    // ── Evaluate ──
    selcx.ctx.begin_transaction();

    let result = selcx.select(goal);

    let pop_goal = |stack: &mut Vec<ActiveGoal>, pushed: bool| {
        if pushed {
            stack.pop();
        }
    };

    match result {
        Ok(impl_source) => {
            if matches!(&impl_source, ImplSource::Deferred { .. }) {
                selcx.ctx.rollback_transaction();
                pop_goal(stack, is_keyed);
                return Ok(impl_source);
            }
            for nested in impl_source.nested_obligations() {
                match evaluate_goal_inner(selcx, &nested, stack, depth + 1) {
                    Ok(ImplSource::Deferred { stalled_on }) => {
                        selcx.ctx.rollback_transaction();
                        pop_goal(stack, is_keyed);
                        return Ok(ImplSource::Deferred { stalled_on });
                    }
                    Ok(_) => {}
                    Err(e) => {
                        selcx.ctx.rollback_transaction();
                        pop_goal(stack, is_keyed);
                        return Err(e);
                    }
                }
            }
            selcx.ctx.commit_transaction();
            pop_goal(stack, is_keyed);
            Ok(impl_source)
        }
        Err(e) => {
            selcx.ctx.rollback_transaction();
            pop_goal(stack, is_keyed);
            Err(e)
        }
    }
}

/// Compute the canonical goal key for an obligation under current bindings.
fn compute_goal_key(obligation: &Obligation, ctx: &TypeContext) -> Option<GoalKey> {
    compute_goal_key_for_predicate(&obligation.predicate, ctx)
}

/// Compute the canonical goal key for a predicate under current bindings.
fn compute_goal_key_for_predicate(predicate: &Predicate, ctx: &TypeContext) -> Option<GoalKey> {
    let mut bound = Vec::new();
    match predicate {
        Predicate::Trait { trait_id, self_ty, args } => {
            Some(GoalKey {
                kind: GoalKind::Trait,
                trait_id: Some(*trait_id),
                self_ty: canonicalize_type(ctx, *self_ty, &mut bound),
                args: args.iter().map(|a| canonicalize_type(ctx, *a, &mut bound)).collect(),
            })
        }
        Predicate::AutoTrait { trait_id, self_ty } => {
            Some(GoalKey {
                kind: GoalKind::AutoTrait,
                trait_id: Some(*trait_id),
                self_ty: canonicalize_type(ctx, *self_ty, &mut bound),
                args: vec![],
            })
        }
        Predicate::Sized { ty } => {
            Some(GoalKey {
                kind: GoalKind::Sized,
                trait_id: None,
                self_ty: canonicalize_type(ctx, *ty, &mut bound),
                args: vec![],
            })
        }
        _ => None,
    }
}