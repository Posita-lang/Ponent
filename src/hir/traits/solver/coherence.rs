use crate::ast::Span;
use crate::hir::traits::ImplCandidate;
use crate::hir::types::{Subst, TypeContext, TypeData, TypeId};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Static counter for generating fresh inference variable IDs during
/// overlap detection normalization.  We use a large base offset to
/// avoid collisions with inference variables created by the main
/// inference context (which typically uses lower IDs).
static OVERLAP_FRESH_VAR_ID: AtomicUsize = AtomicUsize::new(1_000_000);

/// Result of an overlap check between two impls.
#[derive(Clone, Debug)]
pub struct OverlapConflict {
    /// Index of the existing impl in TraitEnv::impls.
    pub existing_idx: usize,
    /// Span of the existing impl declaration.
    pub existing_span: Span,
    /// Span of the new (conflicting) impl declaration.
    pub new_span: Span,
    /// Description of the conflict.
    pub kind: OverlapKind,
}

#[derive(Clone, Debug)]
pub enum OverlapKind {
    /// Both impls match the same (trait, type) combination.
    DirectOverlap,
    /// The impls are equivalent after unification.
    Equivalent,
}

/// Check whether a new impl overlaps with any existing impl.
///
/// Returns `Some(OverlapConflict)` if overlap is detected.
///
/// Uses a two-phase approach:
/// 1. Fast structural comparison: if both types are concrete (no GenericParams),
///    compare their TypeData directly.  Different TypeData → no overlap.
/// 2. Unification: if either type is a GenericParam, use `try_unify` to check
///    if there exists a substitution that makes them equal.
///    The caller must wrap this in `begin_transaction`/`rollback_transaction`.
pub fn check_overlap(
    existing_impls: &[ImplCandidate],
    new_impl: &ImplCandidate,
    ctx: &mut TypeContext,
) -> Option<OverlapConflict> {
    for (existing_idx, existing) in existing_impls.iter().enumerate() {
        if existing.trait_id != new_impl.trait_id {
            continue;
        }

        // Fast path: structural comparison of concrete types.
        // Also compare trait_args — two impls of the same trait with
        // different generic arguments on the same type are NOT overlapping
        // (e.g., `impl Add<Int<32>> for T` vs `impl Add<Int<64>> for T`).
        // NOTE: We use contains_generic_param (recursive) instead of a
        // shallow check so that composite types like `Tuple([GenericParam(0)])`
        // are correctly identified as non-concrete and sent to the slow path.
        let new_data = ctx.get(new_impl.for_type);
        let existing_data = ctx.get(existing.for_type);

        let both_concrete =
            !contains_generic_param(new_data, ctx) && !contains_generic_param(existing_data, ctx);
        let args_concrete = new_impl
            .trait_args
            .iter()
            .all(|a| !contains_generic_param(ctx.get(*a), ctx))
            && existing
                .trait_args
                .iter()
                .all(|a| !contains_generic_param(ctx.get(*a), ctx));

        if both_concrete && args_concrete {
            // All types are concrete: compare for_type AND trait_args structurally.
            if new_data == existing_data && new_impl.trait_args == existing.trait_args {
                return Some(OverlapConflict {
                    existing_idx,
                    existing_span: existing.span,
                    new_span: new_impl.span,
                    kind: OverlapKind::DirectOverlap,
                });
            }
            // Both for_type and all trait_args are concrete and at least one
            // differs — no overlap possible.
            continue;
        }

        // Slow path: normalize all GenericParam to fresh inference variables,
        // then unify.  This handles alpha-equivalence across different
        // GenericParam indices (e.g., `impl<T> Trait for (T,)` and
        // `impl<U> Trait for (U,)`) because both T and U are replaced with
        // the same fresh inference variable (same index → same fresh var),
        // and different indices → different fresh vars that can still unify
        // via the inference variable binding mechanism.
        //
        // This approach is more robust than calling try_unify directly on
        // the original types, because try_unify treats GenericParam(0) and
        // GenericParam(1) as distinct and would not unify them without
        // binding side effects that may not survive the caller's transaction
        // rollback.  By normalizing to fresh inference vars first, we ensure
        // that all GenericParam are treated as universally quantified
        // variables that can be instantiated to any type.
        //
        // NOTE: The caller wraps this in begin_transaction/rollback_transaction,
        // so any bindings created by try_unify below are automatically undone.
        //
        // Step 1: Collect all unique GenericParam indices from both impls.
        let mut all_indices = Vec::new();
        collect_generic_param_indices(new_impl.for_type, ctx, &mut all_indices);
        collect_generic_param_indices(existing.for_type, ctx, &mut all_indices);
        for a in &new_impl.trait_args {
            collect_generic_param_indices(*a, ctx, &mut all_indices);
        }
        for a in &existing.trait_args {
            collect_generic_param_indices(*a, ctx, &mut all_indices);
        }
        all_indices.sort();
        all_indices.dedup();

        // Step 2: Build a substitution mapping each GenericParam index to a
        // fresh inference variable.  Same index → same fresh var, so both
        // types are normalized with the same bindings for corresponding
        // parameters (they are treated as the same universally quantified
        // variable).
        let mut subst = Subst::new();
        for &idx in &all_indices {
            let fresh_id = OVERLAP_FRESH_VAR_ID.fetch_add(1, Ordering::Relaxed);
            let fresh_var = ctx.alloc_infer_var(fresh_id);
            subst.insert(idx, fresh_var);
        }

        // Step 3: Normalize both for_type and trait_args with the substitution.
        let new_for_ty = ctx.subst(new_impl.for_type, &subst);
        let existing_for_ty = ctx.subst(existing.for_type, &subst);
        let new_trait_args: Vec<TypeId> = new_impl
            .trait_args
            .iter()
            .map(|a| ctx.subst(*a, &subst))
            .collect();
        let existing_trait_args: Vec<TypeId> = existing
            .trait_args
            .iter()
            .map(|a| ctx.subst(*a, &subst))
            .collect();

        // Step 4: Unify the normalized for_type and trait_args.
        let for_type_ok = ctx.try_unify(new_for_ty, existing_for_ty).is_ok();
        let trait_args_ok = new_trait_args.len() == existing_trait_args.len()
            && new_trait_args
                .iter()
                .zip(existing_trait_args.iter())
                .all(|(a, b)| ctx.try_unify(*a, *b).is_ok());

        if for_type_ok && trait_args_ok {
            return Some(OverlapConflict {
                existing_idx,
                existing_span: existing.span,
                new_span: new_impl.span,
                kind: OverlapKind::DirectOverlap,
            });
        }
    }
    None
}

/// Recursively check if a TypeData contains any GenericParam anywhere.
/// This is used by the overlap fast path to decide whether structural
/// comparison is sufficient.  A composite type like `Tuple([GenericParam(0)])`
/// contains a generic parameter and must be checked via unification, not
/// structural comparison.
fn contains_generic_param(data: &TypeData, ctx: &TypeContext) -> bool {
    match data {
        TypeData::GenericParam { .. } => true,
        TypeData::Adt { args, .. } => args.iter().any(|a| contains_generic_param_by_id(*a, ctx)),
        TypeData::Tuple { elems } => elems.iter().any(|e| contains_generic_param_by_id(*e, ctx)),
        TypeData::Fn { params, ret } => {
            params.iter().any(|p| contains_generic_param_by_id(*p, ctx))
                || contains_generic_param_by_id(*ret, ctx)
        }
        TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
            contains_generic_param_by_id(*ty, ctx)
        }
        TypeData::Ptr { pointee, .. } => contains_generic_param_by_id(*pointee, ctx),
        TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
            contains_generic_param_by_id(*elem, ctx)
        }
        TypeData::Forall { body, .. }
        | TypeData::Exists { base: body, .. }
        | TypeData::Mu { body, .. }
        | TypeData::Nu { body, .. }
        | TypeData::Poly { body, .. } => contains_generic_param_by_id(*body, ctx),
        TypeData::AssociatedType { self_ty, .. } => contains_generic_param_by_id(*self_ty, ctx),
        TypeData::Coproduct { alternatives } => alternatives
            .iter()
            .any(|a| contains_generic_param_by_id(*a, ctx)),
        // All other types (Int, Bool, etc.) have no GenericParams.
        _ => false,
    }
}

/// Check if a TypeId refers to a type that contains any GenericParam.
/// Resolves the TypeId via ctx.get() and recurses into contains_generic_param.
fn contains_generic_param_by_id(ty: TypeId, ctx: &TypeContext) -> bool {
    contains_generic_param(ctx.get(ty), ctx)
}

/// Collect all GenericParam indices from a type, recursing through
/// composite types.  Uses type data resolved via ctx.get().
/// This is used by the overlap slow path to build a substitution that maps
/// each GenericParam index to a fresh inference variable.
fn collect_generic_param_indices(ty: TypeId, ctx: &TypeContext, out: &mut Vec<usize>) {
    collect_generic_param_indices_data(ctx.get(ty), ctx, out)
}

/// Internal recursive helper that operates on resolved TypeData.
fn collect_generic_param_indices_data(data: &TypeData, ctx: &TypeContext, out: &mut Vec<usize>) {
    match data {
        TypeData::GenericParam { index, .. } => out.push(*index),
        TypeData::Adt { args, .. } => {
            for &a in args {
                collect_generic_param_indices(a, ctx, out);
            }
        }
        TypeData::Tuple { elems } => {
            for &e in elems {
                collect_generic_param_indices(e, ctx, out);
            }
        }
        TypeData::Fn { params, ret } => {
            for &p in params {
                collect_generic_param_indices(p, ctx, out);
            }
            collect_generic_param_indices(*ret, ctx, out);
        }
        TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
            collect_generic_param_indices(*ty, ctx, out);
        }
        TypeData::Ptr { pointee, .. } => collect_generic_param_indices(*pointee, ctx, out),
        TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
            collect_generic_param_indices(*elem, ctx, out);
        }
        TypeData::Forall { body, .. }
        | TypeData::Exists { base: body, .. }
        | TypeData::Mu { body, .. }
        | TypeData::Nu { body, .. }
        | TypeData::Poly { body, .. } => collect_generic_param_indices(*body, ctx, out),
        TypeData::AssociatedType { self_ty, .. } => {
            collect_generic_param_indices(*self_ty, ctx, out)
        }
        TypeData::Coproduct { alternatives } => {
            for &a in alternatives {
                collect_generic_param_indices(a, ctx, out);
            }
        }
        // All other types (Int, Bool, etc.) have no GenericParams.
        _ => {}
    }
}

/// Check whether a new impl's for_type overlaps with any existing impl's for_type.
///
/// This is a lighter check than `check_overlap` — it only checks if the
/// *head types* (for_type) unify, without checking trait_id. This is used
/// for inherent impl overlap detection.
pub fn check_inherent_overlap(
    existing_impls: &[ImplCandidate],
    new_for_type: TypeId,
    ctx: &mut TypeContext,
) -> Option<OverlapConflict> {
    for (existing_idx, existing) in existing_impls.iter().enumerate() {
        ctx.begin_transaction();
        let unification = ctx.try_unify(new_for_type, existing.for_type);
        ctx.rollback_transaction();

        if unification.is_ok() {
            return Some(OverlapConflict {
                existing_idx,
                existing_span: existing.span,
                new_span: Span::new(0, 0), // caller should fill this
                kind: OverlapKind::DirectOverlap,
            });
        }
    }
    None
}
