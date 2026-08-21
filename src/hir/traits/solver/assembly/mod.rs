//! # Candidate Assembly for the Trait Solver
//!
//! Analogous to `rustc_next_trait_solver::solve::assembly`.
//! Provides the `GoalKind` trait that encapsulates candidate assembly
//! for a specific goal type, and the `assemble_and_evaluate_candidates`
//! function that drives the assembly → winnowing → confirmation pipeline.
//!
//! ## Design
//!
//! Instead of having 6+ independent assembly methods on `SolverDelegate`
//! (impls, caller_bounds, builtins, object_ty, poly), we define a single
//! `GoalKind` trait that each predicate type implements.  The assembly
//! engine is a generic function that:
//!
//! 1. Calls `GoalKind` methods to assemble candidates from each source
//! 2. Evaluates each candidate in a probe (transaction rollback)
//! 3. Winnows overlapping candidates by specificity
//! 4. Confirms the winning candidate
//!
//! ## Borrow Discipline
//!
//! `GoalKind` methods take `&mut TypeContext<'input>` (not `&mut EvalCtxt`) to avoid
//! borrow conflicts with the assembly functions that concurrently hold
//! references to `TraitEnv` / `BuiltinTraitRegistry` through `ecx.delegate`.
//! The `EvalCtxt` is only used in the top-level `assemble_and_evaluate_candidates`.

use crate::hir::traits::ImplCandidate;
use crate::hir::traits::TraitEnv;
use crate::hir::traits::solver::builtins;
use crate::hir::traits::solver::builtins::BuiltinTraitRegistry;
use crate::hir::traits::solver::delegate::SolverDelegate;
use crate::hir::traits::solver::eval_ctxt::EvalCtxt;
use crate::hir::traits::solver::eval_ctxt::{GoalSource, GoalStalledOn, ProbeKind};
use crate::hir::traits::solver::obligation::{
    BuiltinImplSource, ImplSource, Obligation, ObligationCause, ObligationCauseCode, Predicate,
    SolveError,
};
use crate::hir::traits::solver::select::{
    Candidate, Candidates, MAX_RECURSION_DEPTH, ResolvedObligation,
};
use crate::hir::types::{DefId, Subst, TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counter for fresh inference variables used during generic impl matching.
static ASSEMBLY_FRESH_VAR_ID: AtomicUsize = AtomicUsize::new(3_000_000);

/// Unify with rustc-style rigid-param semantics: when one side is a
/// `GenericParam` and the other a fresh `InferVar`, bind the VAR to the
/// param regardless of argument order.
///
/// `TypeContext::unify` binds the PARAM for `(param, var)` pairs — a
/// direction the checker's call-instantiation and GADT refinement paths
/// deliberately depend on (a call-site var must stay a var so defaulting
/// can fill it).  The solver needs the opposite: a goal self_ty that is
/// the impl's rigid param must not absorb the goal's variable — the
/// dangling var makes `SgGoalKey::from_obligation` return `None`, so a
/// recursive blanket `impl<T: Foo> Foo for T` on a generic-param goal
/// skips cycle detection and defers instead of reporting overflow.
fn solver_unify<'input>(
    ctx: &mut TypeContext<'input>,
    a: TypeId,
    b: TypeId,
) -> Result<TypeId, crate::hir::types::TypeError> {
    // Resolve bindings FIRST (OmniML union-find discipline: unification
    // operates on the ROOT of each variable — `Term.structure t =
    // (Union_find.get t).structure` in the reference `unifier.ml` — never
    // on a stale bound node).  `get_raw` bypasses binding resolution, so
    // without this an InferVar bound earlier in inference (e.g. a goal
    // self_ty resolved during candidate evaluation) would still classify
    // as an unbound var and `set_binding` would silently overwrite its
    // existing binding with the impl's GenericParam.
    let a = ctx.resolve_binding(a);
    let b = ctx.resolve_binding(b);
    let a_is_param = matches!(ctx.get_raw(a), TypeData::GenericParam { .. });
    let b_is_param = matches!(ctx.get_raw(b), TypeData::GenericParam { .. });
    let a_is_var = matches!(ctx.get_raw(a), TypeData::InferVar { .. });
    let b_is_var = matches!(ctx.get_raw(b), TypeData::InferVar { .. });
    let (var, param) = if a_is_var && b_is_param {
        (a, b)
    } else if a_is_param && b_is_var {
        (b, a)
    } else {
        return ctx.unify(a, b);
    };
    ctx.begin_transaction();
    // Mirror the (InferVar, GenericParam) arm of `unify_internal_impl`:
    // a GADT skolem must not be bound into the var, and the var must not
    // occur inside the param.
    let result = if ctx.contains_gadt_skolem(param) || ctx.occurs_check(var, param) {
        Err(crate::hir::types::TypeError::Mismatch {
            expected: param,
            found: var,
            span: crate::ast::Span::new(0, 0),
        })
    } else if ctx.set_binding(var, param) {
        Ok(param)
    } else {
        Err(crate::hir::types::TypeError::Mismatch {
            expected: param,
            found: var,
            span: crate::ast::Span::new(0, 0),
        })
    };
    match result {
        Ok(ty) => {
            ctx.commit_transaction();
            Ok(ty)
        }
        Err(e) => {
            ctx.rollback_transaction();
            Err(e)
        }
    }
}

// ── GoalKind trait ────────────────────────────────────────────────

/// Trait that encapsulates candidate assembly for a specific goal type.
///
/// Analogous to Rust's `assembly::GoalKind` trait.
/// Each predicate type that can be a goal implements this trait to
/// provide the candidate assembly logic specific to that goal.
///
/// The `assemble_and_evaluate_candidates` function calls these methods
/// in a generic pipeline, and the `EvalCtxt` dispatches to the correct
/// `GoalKind` implementation based on the predicate kind.
///
/// Note: `consider_*` methods take `&mut EvalCtxt` (not `&mut TypeContext<'input>`)
/// to enable probe integration at the candidate level.  Each candidate
/// can be wrapped in `ecx.probe_trait_candidate(source).enter(...)`.
pub(super) trait GoalKind<'input, D: SolverDelegate<'input>> {
    /// The self type of the goal (after resolving through bindings if needed).
    fn self_ty(&self) -> TypeId;

    /// The trait def id if this is a trait goal, or `None` for builtin-only
    /// goals like `Sized` / `Copy`.
    fn trait_def_id(&self) -> Option<DefId>;

    /// Resolve the goal through bindings, returning a `ResolvedObligation`.
    fn resolve(&self, ctx: &TypeContext<'input>) -> ResolvedObligation;

    /// Consider a user-defined impl as a candidate.
    ///
    /// Tries to unify the impl's `for_type` with the obligation's `self_ty`
    /// inside a transaction.  If unification succeeds, pushes a `Candidate::Impl`.
    /// The transaction is rolled back — confirmation will re-apply it.
    fn consider_impl_candidate(
        &self,
        ecx: &mut EvalCtxt<'_, 'input, D>,
        idx: usize,
        impl_cand: &ImplCandidate,
        obligation: &ResolvedObligation,
        candidates: &mut Vec<Candidate>,
    );

    /// Consider a caller-bound (where-clause) as a candidate.
    fn consider_caller_bound_candidate(
        &self,
        ecx: &mut EvalCtxt<'_, 'input, D>,
        self_ty: TypeId,
        args: &[TypeId],
        obligation: &ResolvedObligation,
        candidates: &mut Vec<Candidate>,
    );

    /// Consider a builtin trait impl as a candidate.
    fn consider_builtin_candidate(
        &self,
        ecx: &mut EvalCtxt<'_, 'input, D>,
        builtin: BuiltinImplSource,
        candidates: &mut Vec<Candidate>,
    );

    /// Consider an object type bound as a candidate.
    fn consider_object_candidate(
        &self,
        ecx: &mut EvalCtxt<'_, 'input, D>,
        object_trait_id: DefId,
        nested: Vec<Obligation>,
        candidates: &mut Vec<Candidate>,
    );

    /// Consider a poly/unbox type (Posita-specific) as a candidate.
    fn consider_poly_candidate(
        &self,
        ecx: &mut EvalCtxt<'_, 'input, D>,
        quantifier_count: usize,
        candidates: &mut Vec<Candidate>,
    );
}

// ── GoalKind implementation for Predicate ─────────────────────────

impl<'input, D: SolverDelegate<'input>> GoalKind<'input, D> for Predicate {
    fn self_ty(&self) -> TypeId {
        match self {
            Predicate::Trait { self_ty, .. } => *self_ty,
            Predicate::AutoTrait { self_ty, .. } => *self_ty,
            Predicate::Sized { ty } => *ty,
            Predicate::CopyLike { ty, .. } => *ty,
            Predicate::ProjectionEq { self_ty, .. } => *self_ty,
            Predicate::ProjectionNormalize { projection, .. } => projection.self_ty,
            Predicate::NormalizesTo { projection, .. } => projection.self_ty,
            Predicate::Eq { a, .. } => *a,
            Predicate::Sub { sub, .. } => *sub,
            Predicate::Match { scrutinee, .. } => *scrutinee,
            Predicate::Forall { body } | Predicate::Exists { body } => body.self_ty(),
            Predicate::Instance { scheme_ty, .. } => *scheme_ty,
            Predicate::Let { def, .. } => def.self_ty(),
        }
    }

    fn trait_def_id(&self) -> Option<DefId> {
        match self {
            Predicate::Trait { trait_id, .. }
            | Predicate::AutoTrait { trait_id, .. }
            | Predicate::ProjectionEq { trait_id, .. } => Some(*trait_id),
            Predicate::ProjectionNormalize { projection, .. } => Some(projection.trait_id),
            Predicate::Eq { .. } | Predicate::Sub { .. } | Predicate::Match { .. } => None,
            _ => None,
        }
    }

    fn resolve(&self, ctx: &TypeContext<'input>) -> ResolvedObligation {
        match self {
            Predicate::Trait {
                trait_id,
                self_ty,
                args,
            } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                let resolved_args: Vec<TypeId> =
                    args.iter().map(|a| ctx.resolve_binding(*a)).collect();
                let ambiguous = ctx.is_infer_var(resolved_self);
                ResolvedObligation {
                    trait_id: Some(*trait_id),
                    self_ty: resolved_self,
                    args: resolved_args,
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::AutoTrait { trait_id, self_ty } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                let ambiguous = ctx.is_infer_var(resolved_self);
                ResolvedObligation {
                    trait_id: Some(*trait_id),
                    self_ty: resolved_self,
                    args: vec![],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Sized { ty } => {
                let resolved_ty = ctx.resolve_binding(*ty);
                let ambiguous = ctx.is_infer_var(resolved_ty);
                ResolvedObligation {
                    trait_id: Some(DefId(usize::MAX)), // built-in Sized
                    self_ty: resolved_ty,
                    args: vec![],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Eq { a, b } => {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                let ambiguous = ctx.is_infer_var(ra) || ctx.is_infer_var(rb);
                ResolvedObligation {
                    trait_id: None,
                    self_ty: ra,
                    args: vec![rb],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Sub { sub, sup } => {
                let rsub = ctx.resolve_binding(*sub);
                let rsup = ctx.resolve_binding(*sup);
                let ambiguous = ctx.is_infer_var(rsub) || ctx.is_infer_var(rsup);
                ResolvedObligation {
                    trait_id: None,
                    self_ty: rsub,
                    args: vec![rsup],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Match { scrutinee, .. } => {
                let resolved = ctx.resolve_binding(*scrutinee);
                let ambiguous = ctx.is_infer_var(resolved);
                ResolvedObligation {
                    trait_id: None,
                    self_ty: resolved,
                    args: vec![],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            _ => {
                // Fallback for other predicate types (ProjectionEq, CopyLike, etc.)
                ResolvedObligation {
                    trait_id: None,
                    self_ty: ctx.error(),
                    args: vec![],
                    ambiguous: false,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
        }
    }

    fn consider_impl_candidate(
        &self,
        ecx: &mut EvalCtxt<'_, 'input, D>,
        idx: usize,
        impl_cand: &ImplCandidate,
        obligation: &ResolvedObligation,
        candidates: &mut Vec<Candidate>,
    ) {
        // Try unification inside a transaction, then ROLL BACK regardless.
        let ctx = ecx.ctx();
        ctx.begin_transaction();
        let result = try_match_impl(ctx, idx, impl_cand, obligation);
        ctx.rollback_transaction();
        if let Ok(impl_source) = result {
            candidates.push(Candidate::Impl { idx, impl_source });
        }
    }

    fn consider_caller_bound_candidate(
        &self,
        ecx: &mut EvalCtxt<'_, 'input, D>,
        self_ty: TypeId,
        args: &[TypeId],
        obligation: &ResolvedObligation,
        candidates: &mut Vec<Candidate>,
    ) {
        let ctx = ecx.ctx();
        ctx.begin_transaction();

        let ok = solver_unify(ctx, obligation.self_ty, self_ty).is_ok();

        let args_ok = if ok {
            if args.len() == obligation.args.len() {
                args.iter()
                    .zip(obligation.args.iter())
                    .all(|(ba, oa)| solver_unify(ctx, *ba, *oa).is_ok())
            } else {
                false
            }
        } else {
            false
        };

        // Roll back — candidate assembly must be side-effect-free.
        // confirm_candidate will re-apply the unification.
        ctx.rollback_transaction();
        if args_ok {
            candidates.push(Candidate::Param {
                self_ty,
                args: args.to_vec(),
            });
        }
    }

    fn consider_builtin_candidate(
        &self,
        _ecx: &mut EvalCtxt<'_, 'input, D>,
        builtin: BuiltinImplSource,
        candidates: &mut Vec<Candidate>,
    ) {
        candidates.push(Candidate::Builtin(builtin));
    }

    fn consider_object_candidate(
        &self,
        _ecx: &mut EvalCtxt<'_, 'input, D>,
        object_trait_id: DefId,
        nested: Vec<Obligation>,
        candidates: &mut Vec<Candidate>,
    ) {
        candidates.push(Candidate::Object {
            object_trait_id,
            nested,
        });
    }

    fn consider_poly_candidate(
        &self,
        _ecx: &mut EvalCtxt<'_, 'input, D>,
        quantifier_count: usize,
        candidates: &mut Vec<Candidate>,
    ) {
        candidates.push(Candidate::Poly { quantifier_count });
    }
}

// ── Assembly engine ───────────────────────────────────────────────

/// Assemble and evaluate candidates for a given goal.
///
/// This is the core assembly engine, analogous to Rust's
/// `EvalCtxt::assemble_and_evaluate_candidates`.  It:
///
/// 1. Resolves the obligation through bindings
/// 2. If the self_ty is still an inference variable, returns `Deferred`
/// 3. Assembles candidates from all sources (impls, caller_bounds, builtins, etc.)
/// 4. Winnows overlapping candidates
/// 5. Confirms the winning candidate
///
/// Unlike the old `SelectionContext::select`, this function uses the
/// `GoalKind` trait to dispatch candidate assembly logic, making it
/// extensible to new predicate types without modifying the core engine.
pub(super) fn assemble_and_evaluate_candidates<'input, D: SolverDelegate<'input>>(
    ecx: &mut EvalCtxt<'_, 'input, D>,
    obligation: &Obligation,
) -> Result<ImplSource, SolveError> {
    // ── Depth check ──
    if obligation.recursion_depth >= MAX_RECURSION_DEPTH {
        return Err(SolveError::Overflow {
            obligation: Box::new(obligation.clone()),
            depth: obligation.recursion_depth,
        });
    }

    // ── Handle ProjectionEq / ProjectionNormalize directly ──
    // These are not trait obligations — they are resolved by looking up
    // the associated type in the impl and unifying with the target.
    // We handle these BEFORE extracting any borrows from ecx, because
    // they go through the delegate directly.
    //
    // Similarly, Eq / Sub / Match constraints are resolved directly
    // without going through the candidate assembly pipeline.
    match &obligation.predicate {
        Predicate::ProjectionEq {
            trait_id,
            self_ty,
            assoc_name,
            value,
        } => {
            return ecx.delegate.handle_projection_eq(
                *trait_id,
                *self_ty,
                *assoc_name,
                *value,
                &obligation.cause,
            );
        }
        Predicate::ProjectionNormalize { projection, target } => {
            return ecx.delegate.handle_projection_normalize(
                projection,
                *target,
                &obligation.cause,
            );
        }
        Predicate::Eq { a, b } => {
            return ecx.compute_eq_goal(*a, *b, obligation.cause.span);
        }
        Predicate::Sub { sub, sup } => {
            return ecx.compute_sub_goal(*sub, *sup, obligation.cause.span);
        }
        Predicate::Match {
            scrutinee,
            branches_id,
        } => {
            return ecx.compute_match_goal(*scrutinee, *branches_id, obligation.cause.span);
        }
        Predicate::Forall { body } => {
            return ecx.compute_forall_goal(body, &obligation.cause, obligation.recursion_depth);
        }
        Predicate::Exists { body } => {
            return ecx.compute_exists_goal(body, &obligation.cause, obligation.recursion_depth);
        }
        Predicate::Instance {
            scheme_ty,
            instantiation_ty,
        } => {
            return ecx.compute_instance_goal(
                *scheme_ty,
                *instantiation_ty,
                &obligation.cause,
                obligation.recursion_depth,
            );
        }
        Predicate::Let { def, body } => {
            return ecx.compute_let_goal(def, body, &obligation.cause, obligation.recursion_depth);
        }
        _ => {}
    }

    // ── Resolve self_ty ──
    // We need &mut TypeContext<'input> for resolve, so we must get it before
    // borrowing any other data from ecx.delegate.
    let resolved = obligation.predicate.resolve(ecx.ctx());

    // ── If self_ty is still an infer var, defer ──
    if resolved.ambiguous {
        let stalled_on = vec![resolved.self_ty];
        return Ok(ImplSource::Deferred { stalled_on });
    }

    // ── Candidate assembly ──
    let mut candidates = Vec::new();
    let mut ambiguous = false;

    // Assemble from impls
    assemble_impl_candidates(ecx, &obligation.predicate, &resolved, &mut candidates);

    // Assemble from caller bounds
    assemble_caller_bound_candidates(ecx, &obligation.predicate, &resolved, &mut candidates);

    // Assemble from builtins
    assemble_builtin_candidates(
        ecx,
        &obligation.predicate,
        &resolved,
        &mut candidates,
        &mut ambiguous,
    );

    // Assemble from object type
    assemble_object_candidates(ecx, &obligation.predicate, &resolved, &mut candidates);

    // Assemble from poly/unbox (Posita-specific)
    assemble_poly_candidates(ecx, &obligation.predicate, &resolved, &mut candidates);

    // ── Winnowing ──
    let mut all_candidates = Candidates {
        vec: candidates,
        ambiguous,
    };
    // Release any mutable borrows on ecx before winnow (which only needs shared refs).
    {
        let trait_env_ptr = ecx.trait_env;
        let ctx = ecx.ctx();
        // SAFETY: `trait_env_ptr` points to the `TraitEnv` owned by `ecx`; the
        // mutable borrow is released before this line, and `trait_env` is only
        // read (immutably) within `winnow`.
        let trait_env = unsafe { &*trait_env_ptr };
        winnow(ctx, trait_env, &mut all_candidates, &resolved)?;
    }

    // ── Confirmation ──
    match all_candidates.vec.len() {
        0 => {
            if all_candidates.ambiguous {
                Err(SolveError::Ambiguous {
                    trait_id: resolved.trait_id,
                    self_ty: resolved.self_ty,
                    span: obligation.cause.span,
                    num_candidates: 0,
                })
            } else {
                Err(SolveError::NotFound {
                    trait_id: resolved.trait_id,
                    self_ty: resolved.self_ty,
                    span: obligation.cause.span,
                })
            }
        }
        1 => {
            let trait_env_ptr = ecx.trait_env;
            let ctx = ecx.ctx();
            // SAFETY: `trait_env_ptr` points to the `TraitEnv` owned by `ecx`;
            // the mutable borrow is released above and `trait_env` is only read.
            let trait_env = unsafe { &*trait_env_ptr };
            confirm_candidate(ctx, trait_env, &resolved, &all_candidates.vec[0])
        }
        _ => {
            // Multiple candidates survived winnowing → ambiguity
            Err(SolveError::Ambiguous {
                trait_id: resolved.trait_id,
                self_ty: resolved.self_ty,
                span: obligation.cause.span,
                num_candidates: all_candidates.vec.len(),
            })
        }
    }
}

// ── Candidate assembly helpers ────────────────────────────────────

fn assemble_impl_candidates<'input, D: SolverDelegate<'input>>(
    ecx: &mut EvalCtxt<'_, 'input, D>,
    predicate: &Predicate,
    resolved: &ResolvedObligation,
    candidates: &mut Vec<Candidate>,
) {
    let trait_id = resolved.trait_id;
    // Read the raw pointer value FIRST (Copy), then dereference after getting ctx.
    // The raw pointer is Copy, so accessing it doesn't borrow ecx.
    let trait_env_ptr = ecx.trait_env;
    // BORROW BARRIER: calling ecx.ctx() borrows ecx mutably, which prevents
    // any subsequent code from accidentally borrowing ecx immutably through
    // ecx.trait_env / ecx.caller_bounds / ecx.builtin_registry while we
    // hold a reference derived from the raw pointer.  Without this barrier,
    // the compiler would reject `unsafe { &*trait_env_ptr }` because ecx
    // could still be immutably borrowed via field access.  DO NOT REMOVE.
    let _ctx = ecx.ctx();
    // SAFETY: `trait_env_ptr` points to the `TraitEnv` owned by `ecx`; the
    // mutable borrow barrier above (`let _ctx = ecx.ctx()`) ensures ecx is
    // not immutably borrowed here, and `trait_env` is only read below.
    let trait_env = unsafe { &*trait_env_ptr };
    for (idx, impl_cand) in trait_env.all_impls().iter().enumerate() {
        if Some(impl_cand.trait_id) != trait_id {
            continue;
        }
        // Evaluate each candidate inside a probe so that CandidateHeadUsages
        // are tracked during candidate evaluation.  If the candidate fails to
        // match, its head_usages are automatically discarded (not propagated
        // to the parent goal), preventing failed candidates from polluting
        // the cycle head dependency tracking.
        let _head_usages = ecx
            .probe(ProbeKind::SingleCandidate)
            .enter_single_candidate(|ecx| {
                let mut single = Vec::new();
                predicate.consider_impl_candidate(ecx, idx, impl_cand, resolved, &mut single);
                // Transfer the candidate (if any) to the outer candidates vec.
                if let Some(cand) = single.into_iter().next() {
                    candidates.push(cand);
                }
                // Return () — the probe only needs to track head_usages.
                // The actual success/failure is determined by whether a
                // candidate was pushed to the outer vec.
                Ok(())
            });
        // head_usages is dropped here — if the candidate was NOT pushed
        // (failed to match), its cycle head dependencies are discarded.
        // If the candidate WAS pushed, the head_usages should be merged
        // into the parent goal's heads.  Currently we discard them even
        // on success, which is conservative (may cause extra fixpoint
        // iterations but never unsoundness).  Future work: merge head_usages
        // for winning candidates.
    }
}

fn assemble_caller_bound_candidates<'input, D: SolverDelegate<'input>>(
    ecx: &mut EvalCtxt<'_, 'input, D>,
    predicate: &Predicate,
    resolved: &ResolvedObligation,
    candidates: &mut Vec<Candidate>,
) {
    let trait_id = resolved.trait_id;
    let caller_bounds_ptr = ecx.caller_bounds;
    // BORROW BARRIER: same as above — prevents borrow conflict between
    // ecx.caller_bounds (immutable field access) and ecx.ctx() (mutable
    // method call) when the caller_bound reference is later used.
    let _ctx = ecx.ctx();
    // SAFETY: `caller_bounds_ptr` points to the caller-bounds slice owned by
    // `ecx`; the borrow barrier above (`let _ctx = ecx.ctx()`) ensures ecx is
    // not immutably borrowed here, and `caller_bounds` is only read below.
    let caller_bounds = unsafe { &*caller_bounds_ptr };
    for bound in caller_bounds {
        let (bound_trait_id, self_ty, args) = match bound {
            Predicate::Trait {
                trait_id,
                self_ty,
                args,
            } => (trait_id, self_ty, Some(args)),
            Predicate::AutoTrait { trait_id, self_ty } => (trait_id, self_ty, None),
            _ => continue,
        };
        if Some(*bound_trait_id) == trait_id {
            let args_vec = args.cloned().unwrap_or_default();
            predicate
                .consider_caller_bound_candidate(ecx, *self_ty, &args_vec, resolved, candidates);
        }
    }
}

fn assemble_builtin_candidates<'input, D: SolverDelegate<'input>>(
    ecx: &mut EvalCtxt<'_, 'input, D>,
    predicate: &Predicate,
    resolved: &ResolvedObligation,
    candidates: &mut Vec<Candidate>,
    ambiguous: &mut bool,
) {
    let builtin_registry = {
        // SAFETY: `ecx.builtin_registry` points to the builtin trait registry
        // owned by `ecx`; the registry outlives this function and is only
        // read (immutably) below via `lookup`.
        unsafe { &*ecx.builtin_registry }
    };
    let ctx = ecx.ctx();
    let builtin_kind = builtin_registry.lookup(
        resolved
            .trait_id
            .unwrap_or(crate::hir::types::DefId(usize::MAX)),
    );

    match builtin_kind {
        Some(crate::hir::traits::solver::builtins::BuiltinTrait::Sized) => {
            let self_ty = resolved.self_ty;
            if ctx.is_infer_var(self_ty) {
                *ambiguous = true;
            } else if builtins::compute_sized(self_ty, ctx) {
                predicate.consider_builtin_candidate(ecx, BuiltinImplSource::Sized, candidates);
            }
        }
        Some(crate::hir::traits::solver::builtins::BuiltinTrait::Copy) => {
            if builtins::compute_copy(resolved.self_ty, ctx) {
                predicate.consider_builtin_candidate(ecx, BuiltinImplSource::Copy, candidates);
            }
        }
        Some(crate::hir::traits::solver::builtins::BuiltinTrait::Clone) => {
            if builtins::compute_clone(resolved.self_ty, ctx) {
                predicate.consider_builtin_candidate(ecx, BuiltinImplSource::Clone, candidates);
            }
        }
        _ => {
            // Other builtins (Add, Sub, Eq, Ord, Drop, etc.) have no automatic
            // structural derivation — they require a user-defined impl.
            // Rely on assemble_impl_candidates for these.
        }
    }
}

fn assemble_object_candidates<'input, D: SolverDelegate<'input>>(
    ecx: &mut EvalCtxt<'_, 'input, D>,
    predicate: &Predicate,
    resolved: &ResolvedObligation,
    candidates: &mut Vec<Candidate>,
) {
    // Collect matching trait IDs first, then drop ctx before calling
    // consider_object_candidate (which borrows ecx mutably).
    let matching_trait_ids: Vec<DefId> = {
        let ctx = ecx.ctx();
        if let TypeData::DynTrait { traits, .. } = ctx.get(resolved.self_ty) {
            traits
                .iter()
                .filter(|&&t| Some(t) == resolved.trait_id)
                .copied()
                .collect()
        } else {
            return;
        }
    };
    for trait_id in matching_trait_ids {
        predicate.consider_object_candidate(ecx, trait_id, vec![], candidates);
    }
}

fn assemble_poly_candidates<'input, D: SolverDelegate<'input>>(
    ecx: &mut EvalCtxt<'_, 'input, D>,
    predicate: &Predicate,
    resolved: &ResolvedObligation,
    candidates: &mut Vec<Candidate>,
) {
    // Read quantifier_count first, then drop ctx before calling
    // consider_poly_candidate (which borrows ecx mutably).
    let quantifier_count = {
        let ctx = ecx.ctx();
        if let TypeData::Poly { quantifiers, .. } = ctx.get(resolved.self_ty) {
            quantifiers.len()
        } else {
            return;
        }
    };
    predicate.consider_poly_candidate(ecx, quantifier_count, candidates);
}

// ── Winnowing ─────────────────────────────────────────────────────

fn winnow<'input>(
    ctx: &mut TypeContext<'input>,
    trait_env: &TraitEnv,
    candidates: &mut Candidates,
    _resolved: &ResolvedObligation,
) -> Result<(), SolveError> {
    if candidates.vec.len() <= 1 {
        return Ok(());
    }

    // Sort by specificity: concrete > generic, impl > param > builtin
    candidates
        .vec
        .sort_by(|a, b| specificity(ctx, trait_env, a, b));

    // Keep only the most specific ones
    let mut i = 1;
    while i < candidates.vec.len() {
        if candidate_should_be_dropped(&candidates.vec[i], &candidates.vec[0]) {
            candidates.vec.swap_remove(i);
        } else {
            i += 1;
        }
    }

    if candidates.vec.len() > 1 {
        candidates.ambiguous = true;
    }

    Ok(())
}

/// Order candidates by specificity (most specific first).
fn specificity<'input>(
    ctx: &mut TypeContext<'input>,
    trait_env: &TraitEnv,
    a: &Candidate,
    b: &Candidate,
) -> std::cmp::Ordering {
    match (a, b) {
        // Param<'input> candidates are most specific (caller knows best)
        (Candidate::Param { .. }, _) => std::cmp::Ordering::Less,
        (_, Candidate::Param { .. }) => std::cmp::Ordering::Greater,
        // Impl candidates are more specific than builtins
        (Candidate::Impl { .. }, Candidate::Builtin(_)) => std::cmp::Ordering::Less,
        (Candidate::Builtin(_), Candidate::Impl { .. }) => std::cmp::Ordering::Greater,
        // Impl vs Impl: compare constructor depth, then specialization.
        (Candidate::Impl { idx: ai, .. }, Candidate::Impl { idx: bi, .. }) => {
            let a_cand = &trait_env.all_impls()[*ai];
            let b_cand = &trait_env.all_impls()[*bi];
            let a_depth = ctx.type_constructor_depth(a_cand.for_type);
            let b_depth = ctx.type_constructor_depth(b_cand.for_type);
            match b_depth.cmp(&a_depth) {
                std::cmp::Ordering::Equal => {
                    // Same depth — check impl specialization.
                    use crate::hir::traits::solver::coherence::specializes;
                    if specializes(ctx, a_cand, b_cand) {
                        std::cmp::Ordering::Less
                    } else if specializes(ctx, b_cand, a_cand) {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                }
                other => other,
            }
        }
        // Otherwise equal
        _ => std::cmp::Ordering::Equal,
    }
}

/// Check if a candidate should be dropped in favor of another.
fn candidate_should_be_dropped(victim: &Candidate, other: &Candidate) -> bool {
    match (victim, other) {
        (Candidate::Param { .. }, _) => false,
        (_, Candidate::Param { .. }) => true,
        (Candidate::Impl { .. }, Candidate::Builtin(_)) => false,
        (Candidate::Builtin(_), Candidate::Impl { .. }) => true,
        _ => false,
    }
}

// ── Confirmation ──────────────────────────────────────────────────

fn confirm_candidate<'input>(
    ctx: &mut TypeContext<'input>,
    trait_env: &TraitEnv,
    resolved: &ResolvedObligation,
    candidate: &Candidate,
) -> Result<ImplSource, SolveError> {
    match candidate {
        Candidate::Impl { idx, .. } => {
            // Re-apply the bindings for the winning candidate.
            let impl_cand = &trait_env.all_impls()[*idx];
            ctx.begin_transaction();
            let result = try_match_impl(ctx, *idx, impl_cand, resolved);
            match result {
                Ok(impl_source) => {
                    ctx.commit_transaction();
                    Ok(impl_source)
                }
                Err(e) => {
                    ctx.rollback_transaction();
                    Err(e)
                }
            }
        }
        Candidate::Param { self_ty, args } => {
            // Re-apply the unification for the matched caller bound.
            ctx.begin_transaction();
            let ok = solver_unify(ctx, resolved.self_ty, *self_ty).is_ok()
                && args.len() == resolved.args.len()
                && args
                    .iter()
                    .zip(resolved.args.iter())
                    .all(|(a, b)| solver_unify(ctx, *a, *b).is_ok());
            if ok {
                ctx.commit_transaction();
                Ok(ImplSource::Param(vec![]))
            } else {
                ctx.rollback_transaction();
                Err(SolveError::NotFound {
                    trait_id: resolved.trait_id,
                    self_ty: resolved.self_ty,
                    span: crate::ast::Span::new(0, 0),
                })
            }
        }
        Candidate::Builtin(kind) => Ok(ImplSource::Builtin(*kind)),
        Candidate::Object {
            object_trait_id,
            nested,
        } => Ok(ImplSource::Object {
            object_trait_id: *object_trait_id,
            nested: nested.clone(),
        }),
        Candidate::Poly { quantifier_count } => {
            let body = match ctx.get(resolved.self_ty) {
                TypeData::Poly { body, .. } => *body,
                _ => {
                    return Err(SolveError::NotFound {
                        trait_id: resolved.trait_id,
                        self_ty: resolved.self_ty,
                        span: crate::ast::Span::new(0, 0),
                    });
                }
            };
            let mut fresh_subst = Subst::new();
            ctx.begin_transaction();
            for i in 0..*quantifier_count {
                let id = ASSEMBLY_FRESH_VAR_ID.fetch_add(1, Ordering::Relaxed);
                let fresh = ctx.alloc_infer_var(id, 0);
                fresh_subst.insert(i, fresh);
            }
            let unboxed_body = ctx.subst(body, &fresh_subst);
            let confirmed_obligation = Obligation {
                cause: ObligationCause {
                    span: crate::ast::Span::new(0, 0),
                    code: ObligationCauseCode::PolyUnbox {
                        span: crate::ast::Span::new(0, 0),
                    },
                },
                predicate: Predicate::Trait {
                    trait_id: resolved
                        .trait_id
                        .unwrap_or(crate::hir::types::DefId(usize::MAX)),
                    self_ty: unboxed_body,
                    args: resolved.args.clone(),
                },
                recursion_depth: resolved.parent_depth + 1,
            };
            ctx.commit_transaction();
            Ok(ImplSource::Poly {
                subst: fresh_subst,
                nested: vec![confirmed_obligation],
            })
        }
    }
}

// ── Impl matching (extracted from SelectionContext) ────────────────

/// Try to match an impl candidate against the obligation.
///
/// Generates fresh inference variables for each generic param, substitutes
/// the impl's `for_type` and `trait_args`, unifies with the obligation's
/// `self_ty` and `args`, and produces sub-obligations from the impl's
/// where-clause bounds.
///
/// This is called inside a transaction — the caller is responsible for
/// committing or rolling back.
fn try_match_impl<'input>(
    ctx: &mut TypeContext<'input>,
    cand_idx: usize,
    impl_cand: &ImplCandidate,
    obligation: &ResolvedObligation,
) -> Result<ImplSource, SolveError> {
    let arity = impl_cand.arity;

    // Generate fresh infer vars for each generic param
    let mut subst = Subst::new();
    for i in 0..arity {
        let id = ASSEMBLY_FRESH_VAR_ID.fetch_add(1, Ordering::Relaxed);
        let fresh = ctx.alloc_infer_var(id, 0);
        subst.insert(i, fresh);
    }

    // Substitute the candidate's for_type with fresh infer vars.
    let substituted_for_type = ctx.subst(impl_cand.for_type, &subst);

    // Unify substituted for_type with obligation's self_ty
    solver_unify(ctx, obligation.self_ty, substituted_for_type).map_err(|_| {
        SolveError::NotFound {
            trait_id: obligation.trait_id,
            self_ty: obligation.self_ty,
            span: crate::ast::Span::new(0, 0),
        }
    })?;

    // Unify trait generic args
    let substituted_trait_args: Vec<TypeId> = impl_cand
        .trait_args
        .iter()
        .map(|&arg| ctx.subst(arg, &subst))
        .collect();

    if substituted_trait_args.len() != obligation.args.len() {
        return Err(SolveError::NotFound {
            trait_id: obligation.trait_id,
            self_ty: obligation.self_ty,
            span: crate::ast::Span::new(0, 0),
        });
    }

    for (impl_arg, ob_arg) in substituted_trait_args.iter().zip(obligation.args.iter()) {
        solver_unify(ctx, *impl_arg, *ob_arg).map_err(|e| SolveError::Mismatch {
            expected: *ob_arg,
            found: *impl_arg,
            span: crate::ast::Span::new(0, 0),
            note: format!("unify failed: {e:?}"),
        })?;
    }

    // Generate sub-obligations from impl's where-clause
    let mut nested: Vec<Obligation> = Vec::new();
    for &(ref_self_ty, bound_trait_id, ref bound_args) in &impl_cand.where_clause_bounds {
        let substituted_self = ctx.subst(ref_self_ty, &subst);
        let substituted_args: Vec<TypeId> = bound_args
            .iter()
            .map(|&arg| ctx.subst(arg, &subst))
            .collect();
        nested.push(Obligation {
            cause: ObligationCause {
                span: impl_cand.span,
                code: ObligationCauseCode::ImplBound {
                    impl_def_id: impl_cand.trait_id,
                },
            },
            predicate: Predicate::Trait {
                trait_id: bound_trait_id,
                self_ty: substituted_self,
                args: substituted_args,
            },
            recursion_depth: obligation.parent_depth + 1,
        });
    }

    Ok(ImplSource::UserDefined {
        cand_idx,
        subst,
        nested,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for the `solver_unify` binding discipline: an `InferVar`
    /// that is ALREADY BOUND must never be re-bound.  `solver_unify`
    /// inspects the RAW type nodes, and `ctx.get_raw` bypasses binding
    /// resolution — a var bound to a concrete type earlier in inference
    /// was classified as unbound and its existing binding silently
    /// overwritten with the impl's GenericParam.  The OmniML union-find
    /// discipline is find-before-inspect (`Term.structure t =
    /// (Union_find.get t).structure` in the reference `unifier.ml`):
    /// unification operates on the ROOT of each variable, never on a
    /// stale bound node.
    #[test]
    fn test_solver_unify_preserves_existing_var_binding() {
        let mut ctx = TypeContext::new();
        let var = ctx.alloc_infer_var(7, 0);
        let concrete = ctx.struct_ty(DefId(100), vec![]);
        let param = ctx.generic_param(0, crate::symbol::Symbol::intern("T"));
        // The var was already resolved earlier in inference: ?v ↦ S.
        assert!(ctx.set_binding(var, concrete), "fixture: pre-bind the var");
        let result = solver_unify(&mut ctx, var, param);
        assert!(
            result.is_ok(),
            "unifying a pre-bound var with a param must succeed: {:?}",
            result
        );
        // The existing binding must survive; the param absorbs the
        // RESOLVED type instead (find-first unification).
        assert_eq!(
            ctx.resolve_binding(var),
            concrete,
            "the var's existing binding must not be overwritten by the param"
        );
        assert_eq!(
            ctx.resolve_binding(param),
            concrete,
            "the param must be bound to the var's resolved type"
        );
    }
}
