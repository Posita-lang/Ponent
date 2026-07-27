use crate::ast::Span;
use crate::hir::traits::ImplCandidate;
use crate::hir::types::{Subst, TypeContext, TypeData, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Static counter for generating fresh inference variable IDs during
/// overlap detection normalization.  We use a large base offset to
/// avoid collisions with inference variables created by the main
/// inference context (which typically uses lower IDs).
static OVERLAP_FRESH_VAR_ID: AtomicUsize = AtomicUsize::new(1_000_000);

/// Check whether `more_specific` is a specialization of `more_general`.
///
/// An impl `A` specializes impl `B` if every type that matches `A` also
/// matches `B`.  Equivalently, `A.for_type` unifies with `B.for_type`
/// under some substitution.  Returns `true` if specialization is detected.
///
/// Uses a transaction that is rolled back — this function is side-effect-free.
/// Walk a type tree, replacing every `GenericParam(i)` with a freshly
/// allocated inference variable.  Uses `OVERLAP_FRESH_VAR_ID` (base 1,000,000)
/// to guarantee IDs never collide with the main inference context.
/// Every `TypeData` variant containing `TypeId` children is handled,
/// ensuring that nested GenericParams in composite types (Adt, Tuple, Ref,
/// Fn, Array, Slice, Pointer, Ptr, AssociatedType, Coproduct, Forall, Exists,
/// Mu, Nu, Opaque, DynTrait, Poly) are all properly freshened.
/// See rustc's `fresh_subst` in coherence (rustc_hir_analysis/src/coherence).
fn freshen_generics(ty: TypeId, ctx: &mut TypeContext, map: &mut HashMap<usize, TypeId>) -> TypeId {
    let resolved = ctx.resolve_binding(ty);
    match ctx.get(resolved).clone() {
        TypeData::GenericParam { index, .. } => *map.entry(index).or_insert_with(|| {
            let fresh_id = OVERLAP_FRESH_VAR_ID.fetch_add(1, Ordering::Relaxed);
            ctx.alloc_infer_var(fresh_id)
        }),
        TypeData::Adt { kind, def_id, args } => {
            let new_args = args
                .iter()
                .map(|a| freshen_generics(*a, ctx, map))
                .collect();
            ctx.find_type(&TypeData::Adt {
                kind,
                def_id,
                args: new_args,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Tuple { elems } => {
            let new_elems = elems
                .iter()
                .map(|e| freshen_generics(*e, ctx, map))
                .collect();
            ctx.find_type(&TypeData::Tuple { elems: new_elems })
                .unwrap_or(ctx.error())
        }
        TypeData::Ref { ty, mutable } => {
            let new_ty = freshen_generics(ty, ctx, map);
            ctx.find_type(&TypeData::Ref {
                ty: new_ty,
                mutable,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Fn { params, ret, .. } => {
            let new_params = params
                .iter()
                .map(|p| freshen_generics(*p, ctx, map))
                .collect();
            let new_ret = freshen_generics(ret, ctx, map);
            ctx.find_type(&TypeData::Fn {
                params: new_params,
                ret: new_ret,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Array { elem, size } => {
            let new_elem = freshen_generics(elem, ctx, map);
            ctx.find_type(&TypeData::Array {
                elem: new_elem,
                size,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Slice { elem } => {
            let new_elem = freshen_generics(elem, ctx, map);
            ctx.find_type(&TypeData::Slice { elem: new_elem })
                .unwrap_or(ctx.error())
        }
        TypeData::Pointer { ty } => {
            let new_ty = freshen_generics(ty, ctx, map);
            ctx.find_type(&TypeData::Pointer { ty: new_ty })
                .unwrap_or(ctx.error())
        }
        TypeData::Ptr { size, pointee } => {
            let new_size = freshen_generics(size, ctx, map);
            let new_pointee = freshen_generics(pointee, ctx, map);
            ctx.find_type(&TypeData::Ptr {
                size: new_size,
                pointee: new_pointee,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => {
            let new_self = freshen_generics(self_ty, ctx, map);
            ctx.find_type(&TypeData::AssociatedType {
                trait_id,
                name,
                self_ty: new_self,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Coproduct { alternatives } => {
            let new_alts = alternatives
                .iter()
                .map(|a| freshen_generics(*a, ctx, map))
                .collect();
            ctx.find_type(&TypeData::Coproduct {
                alternatives: new_alts,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Forall {
            param_index,
            param_name,
            body,
            ..
        } => {
            let new_body = freshen_generics(body, ctx, map);
            ctx.find_type(&TypeData::Forall {
                param_index,
                param_name,
                body: new_body,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Exists {
            param_index,
            name,
            base,
        } => {
            let new_base = freshen_generics(base, ctx, map);
            ctx.find_type(&TypeData::Exists {
                param_index,
                name,
                base: new_base,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Mu {
            param_index,
            param_name,
            body,
        } => {
            let new_body = freshen_generics(body, ctx, map);
            ctx.find_type(&TypeData::Mu {
                param_index,
                param_name,
                body: new_body,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Nu {
            param_index,
            param_name,
            body,
        } => {
            let new_body = freshen_generics(body, ctx, map);
            ctx.find_type(&TypeData::Nu {
                param_index,
                param_name,
                body: new_body,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Poly { quantifiers, body } => {
            let new_body = freshen_generics(body, ctx, map);
            ctx.find_type(&TypeData::Poly {
                quantifiers,
                body: new_body,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Opaque { def_id, hidden } => match hidden {
            Some(hidden_ty) => {
                let new_hidden = freshen_generics(hidden_ty, ctx, map);
                ctx.find_type(&TypeData::Opaque {
                    def_id,
                    hidden: Some(new_hidden),
                })
                .unwrap_or(ctx.error())
            }
            None => ty,
        },
        TypeData::DynTrait { traits } => {
            // DynTrait may contain bound GenericParams in its trait refs.
            // For now, treat as leaf since DynTrait bounds aren't freshened.
            let ty = ty;
            ty
        }
        // Leaf types — no TypeId children to recurse into.
        TypeData::InferVar { .. }
        | TypeData::SkolemVar { .. }
        | TypeData::Int { .. }
        | TypeData::UInt { .. }
        | TypeData::Float { .. }
        | TypeData::Rational { .. }
        | TypeData::Bool
        | TypeData::Char
        | TypeData::Byte
        | TypeData::USize
        | TypeData::Never
        | TypeData::Unit
        | TypeData::Error
        | TypeData::Type
        | TypeData::Regex { .. } => ty,
    }
}

pub fn specializes(
    ctx: &mut TypeContext,
    more_specific: &ImplCandidate,
    more_general: &ImplCandidate,
) -> bool {
    if more_specific.trait_id != more_general.trait_id {
        return false;
    }

    ctx.begin_transaction();

    // Step 1: Create fresh inference variables for the general impl's type
    // parameters by walking its for_type, trait_args, AND where_clause_bounds.
    // A SINGLE shared fresh_map is used across all three, so the same
    // GenericParam(i) always maps to the same fresh InferVar everywhere.
    let mut fresh_map: HashMap<usize, TypeId> = HashMap::default();
    let fresh_for_type = freshen_generics(more_general.for_type, ctx, &mut fresh_map);
    let fresh_args: Vec<TypeId> = more_general
        .trait_args
        .iter()
        .map(|a| freshen_generics(*a, ctx, &mut fresh_map))
        .collect();

    // Step 2: Unify specific.for_type with the freshened general.for_type.
    let for_type_ok = ctx
        .try_unify(more_specific.for_type, fresh_for_type)
        .is_ok();

    // Step 3: Check trait args unify.
    let args_ok = if for_type_ok {
        more_specific.trait_args.len() == fresh_args.len()
            && more_specific
                .trait_args
                .iter()
                .zip(&fresh_args)
                .all(|(a, b)| ctx.try_unify(*a, *b).is_ok())
    } else {
        false
    };

    // Step 4: Where-clause check using the SAME fresh_map so that the fresh
    // InferVars in for_type/args are consistent with those in the bounds.
    let where_ok = if args_ok {
        more_general.where_clause_bounds.len() <= more_specific.where_clause_bounds.len()
            && more_general
                .where_clause_bounds
                .iter()
                .all(|(g_self_ty, g_trait_id, g_args)| {
                    // Freshen this where-clause bound's types with the SAME fresh_map.
                    let g_fresh_self = freshen_generics(*g_self_ty, ctx, &mut fresh_map);
                    let g_fresh_args: Vec<TypeId> = g_args
                        .iter()
                        .map(|a| freshen_generics(*a, ctx, &mut fresh_map))
                        .collect();
                    more_specific.where_clause_bounds.iter().any(
                        |(s_self_ty, s_trait_id, s_args)| {
                            s_trait_id == g_trait_id
                                && ctx.try_unify(*s_self_ty, g_fresh_self).is_ok()
                                && s_args.len() == g_fresh_args.len()
                                && s_args
                                    .iter()
                                    .zip(&g_fresh_args)
                                    .all(|(a, b)| ctx.try_unify(*a, *b).is_ok())
                        },
                    )
                })
    } else {
        false
    };

    // Step 5: Directionality check — AFTER all unifications (for_type, trait_args,
    // AND where_clause_bounds).  Verify that the specific impl's GenericParams
    // were NOT bound by any of the above steps.  This catches GenericParams that
    // appear only in where_clause_bounds, which were not checked by earlier passes.
    let direction_ok = where_ok && generic_params_untouched_all(more_specific, ctx);

    ctx.rollback_transaction();
    direction_ok
}

/// Check that the specific impl's GenericParams were not bound during
/// unification.  Walks for_type, trait_args, AND where_clause_bounds.
fn generic_params_untouched_all(specific: &ImplCandidate, ctx: &TypeContext) -> bool {
    fn gp_untouched(ty: TypeId, ctx: &TypeContext) -> bool {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved) {
            TypeData::GenericParam { .. } => resolved == ty,
            TypeData::Adt { args, .. } => args.iter().all(|a| gp_untouched(*a, ctx)),
            TypeData::Tuple { elems } => elems.iter().all(|e| gp_untouched(*e, ctx)),
            TypeData::Ref { ty, .. } => gp_untouched(*ty, ctx),
            TypeData::Fn { params, ret, .. } => {
                params.iter().all(|p| gp_untouched(*p, ctx)) && gp_untouched(*ret, ctx)
            }
            TypeData::Array { elem, .. } => gp_untouched(*elem, ctx),
            TypeData::Slice { elem } => gp_untouched(*elem, ctx),
            TypeData::Pointer { ty } => gp_untouched(*ty, ctx),
            TypeData::Ptr { size, pointee } => {
                gp_untouched(*size, ctx) && gp_untouched(*pointee, ctx)
            }
            TypeData::AssociatedType { self_ty, .. } => gp_untouched(*self_ty, ctx),
            TypeData::Coproduct { alternatives } => {
                alternatives.iter().all(|a| gp_untouched(*a, ctx))
            }
            TypeData::Forall { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => gp_untouched(*body, ctx),
            TypeData::Exists { base, .. } => gp_untouched(*base, ctx),
            TypeData::Poly { body, .. } => gp_untouched(*body, ctx),
            TypeData::Opaque {
                hidden: Some(hidden),
                ..
            } => gp_untouched(*hidden, ctx),
            // DynTrait contains DefIds, not TypeIds — no generic params to check.
            TypeData::DynTrait { .. } => true,
            _ => true,
        }
    }
    gp_untouched(specific.for_type, ctx)
        && specific.trait_args.iter().all(|a| gp_untouched(*a, ctx))
        && specific
            .where_clause_bounds
            .iter()
            .all(|(self_ty, _, args)| {
                gp_untouched(*self_ty, ctx) && args.iter().all(|a| gp_untouched(*a, ctx))
            })
}
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
