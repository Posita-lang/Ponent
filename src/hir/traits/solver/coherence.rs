use crate::ast::Span;
use crate::hir::traits::ImplCandidate;
use crate::hir::types::{TypeContext, TypeData, TypeId};
use std::collections::HashMap;

/// Check whether `more_specific` is a specialization of `more_general`.
///
/// An impl `A` specializes impl `B` if every type that matches `A` also
/// matches `B`.  Equivalently, `A.for_type` unifies with `B.for_type`
/// under some substitution.  Returns `true` if specialization is detected.
///
/// Uses a transaction that is rolled back — this function is side-effect-free.
/// Walk a type tree, replacing every FREE `GenericParam(i)` with a freshly
/// allocated inference variable.  Fresh IDs come from the TypeContext-local
/// overlap counter (`TypeContext::alloc_overlap_fresh_var`), which starts
/// at a 1_000_000 base offset so IDs never collide with the main inference
/// context.  Every `TypeData` variant containing `TypeId` children is
/// handled, ensuring that nested GenericParams in composite types (Adt, Tuple, Ref,
/// Fn, Array, Slice, Pointer, Ptr, AssociatedType, Coproduct, Forall, Exists,
/// Mu, Nu, Opaque, DynTrait, Poly) are all properly freshened.
/// BOUND variables under Forall/Exists/Mu/Nu/Poly are NOT freshened (they
/// are quantifier-bound, not free) — same binder-shadowing semantics as
/// `TypeContext::subst`.
/// See rustc's `fresh_subst` in coherence (rustc_hir_analysis/src/coherence).
fn freshen_generics<'input>(
    ty: TypeId,
    ctx: &mut TypeContext<'input>,
    map: &mut HashMap<usize, TypeId>,
) -> TypeId {
    freshen_generics_bound(ty, ctx, map, &mut Vec::new())
}

/// Recursive worker for `freshen_generics`; `bound` is the stack of
/// quantifier-bound `param_index` values currently in scope.
fn freshen_generics_bound<'input>(
    ty: TypeId,
    ctx: &mut TypeContext<'input>,
    map: &mut HashMap<usize, TypeId>,
    bound: &mut Vec<usize>,
) -> TypeId {
    let resolved = ctx.resolve_binding(ty);
    match ctx.get(resolved).clone() {
        TypeData::GenericParam { index, .. } => {
            if bound.contains(&index) {
                // Bound variable under an enclosing quantifier — keep the
                // GenericParam as-is (it is NOT free, so it must not be
                // freshened into an inference variable).
                ty
            } else {
                *map.entry(index)
                    .or_insert_with(|| ctx.alloc_overlap_fresh_var(0))
            }
        }
        // NOTE: these branches build types through the REAL constructors
        // (`ctx.tuple` / `ctx.reference_with_lifetime` / `ctx.alloc` …),
        // NOT the retired `ctx.find_type` stub (which always returns
        // `None`, so every composite type used to fall to `ctx.error()`).
        // `reference_with_lifetime` also PRESERVES the Ref lifetime — two
        // `&'a T` / `&'b T` impls stay distinguishable after freshening
        // (unify_internal_impl treats distinct explicit regions as a
        // mismatch), aligning with `TypeContext::subst`.
        TypeData::Adt { kind, def_id, args } => {
            let new_args = args
                .iter()
                .map(|a: &TypeId| freshen_generics_bound(*a, ctx, map, bound))
                .collect();
            ctx.alloc(TypeData::Adt {
                kind,
                def_id,
                args: new_args,
            })
        }
        TypeData::Tuple { elems } => {
            let new_elems = elems
                .iter()
                .map(|e| freshen_generics_bound(*e, ctx, map, bound))
                .collect();
            ctx.tuple(new_elems)
        }
        TypeData::Ref {
            ty,
            mutable,
            lifetime,
        } => {
            let new_ty = freshen_generics_bound(ty, ctx, map, bound);
            ctx.reference_with_lifetime(new_ty, mutable, lifetime)
        }
        TypeData::Fn { params, ret, .. } => {
            let new_params = params
                .iter()
                .map(|p| freshen_generics_bound(*p, ctx, map, bound))
                .collect();
            let new_ret = freshen_generics_bound(ret, ctx, map, bound);
            ctx.function(new_params, new_ret)
        }
        TypeData::Array { elem, size } => {
            let new_elem = freshen_generics_bound(elem, ctx, map, bound);
            ctx.array(new_elem, size)
        }
        TypeData::Slice { elem } => {
            let new_elem = freshen_generics_bound(elem, ctx, map, bound);
            ctx.slice(new_elem)
        }
        TypeData::Pointer { ty } => {
            let new_ty = freshen_generics_bound(ty, ctx, map, bound);
            ctx.pointer(new_ty)
        }
        TypeData::Ptr { size, pointee } => {
            let new_size = freshen_generics_bound(size, ctx, map, bound);
            let new_pointee = freshen_generics_bound(pointee, ctx, map, bound);
            ctx.ptr(new_size, new_pointee)
        }
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => {
            let new_self = freshen_generics_bound(self_ty, ctx, map, bound);
            ctx.associated_type(trait_id, name, new_self)
        }
        TypeData::Coproduct { alternatives } => {
            let new_alts = alternatives
                .iter()
                .map(|a| freshen_generics_bound(*a, ctx, map, bound))
                .collect();
            ctx.coproduct(new_alts)
        }
        TypeData::Forall {
            param_index,
            param_name,
            body,
            ..
        } => {
            bound.push(param_index);
            let new_body: TypeId = freshen_generics_bound(body, ctx, map, bound);
            bound.pop();
            ctx.alloc(TypeData::Forall {
                param_index,
                param_name,
                body: new_body,
            })
        }
        TypeData::Exists {
            param_index,
            name,
            base,
        } => {
            bound.push(param_index);
            let new_base: TypeId = freshen_generics_bound(base, ctx, map, bound);
            bound.pop();
            ctx.alloc(TypeData::Exists {
                param_index,
                name,
                base: new_base,
            })
        }
        TypeData::Mu {
            param_index,
            param_name,
            body,
        } => {
            bound.push(param_index);
            let new_body: TypeId = freshen_generics_bound(body, ctx, map, bound);
            bound.pop();
            ctx.alloc(TypeData::Mu {
                param_index,
                param_name,
                body: new_body,
            })
        }
        TypeData::Nu {
            param_index,
            param_name,
            body,
        } => {
            bound.push(param_index);
            let new_body: TypeId = freshen_generics_bound(body, ctx, map, bound);
            bound.pop();
            ctx.alloc(TypeData::Nu {
                param_index,
                param_name,
                body: new_body,
            })
        }
        TypeData::Poly { quantifiers, body } => {
            let shadowed: Vec<usize> = quantifiers.iter().map(|(idx, _)| *idx).collect();
            bound.extend(shadowed.iter());
            let new_body = freshen_generics_bound(body, ctx, map, bound);
            for _ in &shadowed {
                bound.pop();
            }
            ctx.poly(quantifiers, new_body)
        }
        TypeData::Opaque { def_id, hidden } => match hidden {
            Some(hidden_ty) => {
                let new_hidden = freshen_generics_bound(hidden_ty, ctx, map, bound);
                ctx.alloc(TypeData::Opaque {
                    def_id,
                    hidden: Some(new_hidden),
                })
            }
            None => ty,
        },
        // DynTrait holds only `Vec<DefId>` (trait refs) — no TypeId
        // children to freshen, so treating it as a leaf is correct.
        TypeData::DynTrait { .. } => ty,
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

pub fn specializes<'input>(
    ctx: &mut TypeContext<'input>,
    more_specific: &ImplCandidate,
    more_general: &ImplCandidate,
) -> bool {
    if more_specific.trait_id != more_general.trait_id {
        return false;
    }

    // Two explicit, balanced transactions: the OUTER one covers freshening
    // (fresh InferVar allocation must not leak into the caller's context),
    // the INNER one covers the unification conjunction.  Both are popped
    // with symmetric `rollback_transaction()` calls below — never a
    // depth-based `rollback_to` that relies on an implicit stack-depth
    // assumption (which would silently break if a `commit_transaction`
    // were ever inserted between them).
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
    // Use one transaction for the whole conjunction so shared substitutions
    // across for_type, trait_args, and where-clause bounds are preserved.
    ctx.begin_transaction();

    let for_type_ok = ctx
        .try_unify(more_specific.for_type, fresh_for_type, None)
        .is_ok();

    // Step 3: Check trait args unify.
    let args_ok = if for_type_ok {
        more_specific.trait_args.len() == fresh_args.len()
            && more_specific
                .trait_args
                .iter()
                .zip(&fresh_args)
                .all(|(a, b)| ctx.try_unify(*a, *b, None).is_ok())
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
                                && ctx.try_unify(*s_self_ty, g_fresh_self, None).is_ok()
                                && s_args.len() == g_fresh_args.len()
                                && s_args
                                    .iter()
                                    .zip(&g_fresh_args)
                                    .all(|(a, b)| ctx.try_unify(*a, *b, None).is_ok())
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

    // Pop BOTH transactions symmetrically: the inner conjunction frame,
    // then the outer freshening frame.  Explicitly balanced against the
    // two `begin_transaction` calls above (no depth-based `rollback_to`,
    // so the pairing is self-evident and survives future edits).
    ctx.rollback_transaction(); // pop inner (unification conjunction)
    ctx.rollback_transaction(); // pop outer (freshening frame)
    direction_ok
}

/// Check that the specific impl's GenericParams were not bound during
/// unification.  Walks for_type, trait_args, AND where_clause_bounds.
fn generic_params_untouched_all<'input>(
    specific: &ImplCandidate,
    ctx: &TypeContext<'input>,
) -> bool {
    fn gp_untouched<'input>(ty: TypeId, ctx: &TypeContext<'input>) -> bool {
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

/// Check whether two impls' where-clause bounds are mutually exclusive.
///
/// `where_clause_bounds: Vec<(self_ty, trait_id, args)>`.  Two bounds are
/// EXCLUSIVE when they require the SAME trait on the SAME self type with
/// INCOMPATIBLE arguments: `MyType: Foo<Int<32>>` vs `MyType: Foo<Int<64>>`
/// cannot both hold, so the two impls can never both apply → NOT
/// overlapping.  This mirrors the fast-path rule for impl heads ("two
/// impls of the same trait with different generic arguments on the same
/// type are NOT overlapping") applied to bounds.
///
/// Different trait bounds (`Display` vs `Debug`) are NOT exclusive — a
/// type can satisfy both — so such impls still overlap (matches rustc
/// coherence).  Same-trait bounds with unifiable args are the same
/// requirement → compatible.
///
/// Conservative direction: only PROVEN exclusivity returns `false`; any
/// uncertainty keeps overlap (fail-closed — never silently accept a
/// possibly-overlapping pair).
///
/// Self-contained: probes with `try_unify` inside its own transaction, so
/// bindings are rolled back before returning (safe in both the fast path,
/// which otherwise performs no unification, and the slow path).
fn where_clauses_compatible<'input>(
    existing: &ImplCandidate<'input>,
    new: &ImplCandidate<'input>,
    ctx: &mut TypeContext<'input>,
) -> bool {
    for (e_self, e_trait, e_args) in &existing.where_clause_bounds {
        for (n_self, n_trait, n_args) in &new.where_clause_bounds {
            if e_trait != n_trait {
                continue; // different traits can co-exist on one type.
            }
            ctx.begin_transaction();
            let self_ok = ctx.try_unify(*e_self, *n_self, None).is_ok();
            let args_ok = e_args.len() == n_args.len()
                && e_args
                    .iter()
                    .zip(n_args)
                    .all(|(a, b)| ctx.try_unify(*a, *b, None).is_ok());
            ctx.rollback_transaction();
            if self_ok {
                // Same trait on the same self: if the args do NOT unify,
                // the bounds are mutually exclusive → the impls cannot
                // both apply → NOT overlapping.  If they DO unify, this
                // pair is the same requirement — keep looking.
                if !args_ok {
                    return false;
                }
            }
            // Different self types: the bounds constrain different types,
            // both satisfiable — compatible.
        }
    }
    true
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
///
/// NOTE (where-clause): `where_clause_bounds` are NOT consulted here, unlike
/// `specializes` (which checks them).  This is a DELIBERATE conservative
/// choice for overlap detection: two impls whose where-clauses are disjoint
/// on paper (e.g. `impl<T: Display> Tr for T` vs `impl<T: Debug> Tr for T`)
/// could still both apply to a type satisfying both bounds, so reporting
/// overlap is the sound (fail-closed) direction — false positives are
/// preferred over false negatives (accepting a genuinely overlapping pair).
/// Each slow-path iteration additionally runs its own transaction so that
/// partial-unification bindings never leak across iterations.
pub fn check_overlap<'input>(
    existing_impls: &[ImplCandidate<'input>],
    new_impl: &ImplCandidate<'input>,
    ctx: &mut TypeContext<'input>,
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
                // where-clause check: mutually exclusive bounds (same
                // trait, same self, incompatible args) mean the two impls
                // can never both apply — NOT overlapping.
                if where_clauses_compatible(existing, new_impl, ctx) {
                    return Some(OverlapConflict {
                        existing_idx,
                        existing_span: existing.span,
                        new_span: new_impl.span,
                        kind: OverlapKind::DirectOverlap,
                    });
                }
                continue;
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
        // Step 1+2+3: normalize all FREE GenericParams to fresh inference
        // variables via the SAME freshening used by `specializes`
        // (`freshen_generics` over a SHARED fresh_map).  Same index → same
        // fresh var, so both types are normalized with the same bindings
        // for corresponding parameters (they are treated as the same
        // universally quantified variable); quantifier-BOUND variables are
        // left untouched (binder shadowing).  This is the SINGLE
        // freshening mechanism in this file — the former
        // `collect_generic_param_indices` + `Subst` + `ctx.subst` path is
        // removed so the two impls never drift.
        let mut fresh_map: HashMap<usize, TypeId> = HashMap::default();
        let new_for_ty = freshen_generics(new_impl.for_type, ctx, &mut fresh_map);
        let existing_for_ty = freshen_generics(existing.for_type, ctx, &mut fresh_map);
        let new_trait_args: Vec<TypeId> = new_impl
            .trait_args
            .iter()
            .map(|a| freshen_generics(*a, ctx, &mut fresh_map))
            .collect();
        let existing_trait_args: Vec<TypeId> = existing
            .trait_args
            .iter()
            .map(|a| freshen_generics(*a, ctx, &mut fresh_map))
            .collect();

        // Step 4: Unify the normalized for_type and trait_args.
        // Use try_unify (not can_unify) because the caller wraps this
        // in begin_transaction/rollback_transaction.  Using can_unify
        // would create a nested inner transaction that rolls back before
        // the caller's outer transaction, losing shared substitutions.
        //
        // Per-iteration transaction isolation: each iteration runs its own
        // begin/rollback so that bindings created by a PARTIAL unification
        // (e.g. for_type unifies but a trait_arg does not) do not leak into
        // the NEXT iteration's judgement.  Without this, iteration i's
        // leftover bindings would pollute iteration i+1's `try_unify`
        // (the caller's outer transaction only rolls back once, at the
        // end of the whole function).  Same discipline as
        // `check_inherent_overlap` (which rolls back per iteration).
        ctx.begin_transaction();
        let for_type_ok = ctx.try_unify(new_for_ty, existing_for_ty, None).is_ok();
        let trait_args_ok = new_trait_args.len() == existing_trait_args.len()
            && new_trait_args
                .iter()
                .zip(existing_trait_args.iter())
                .all(|(a, b)| ctx.try_unify(*a, *b, None).is_ok());
        ctx.rollback_transaction();

        if for_type_ok && trait_args_ok {
            // where-clause check: mutually exclusive bounds (same
            // trait, same self, incompatible args) mean the two impls can
            // never both apply — NOT overlapping.
            if where_clauses_compatible(existing, new_impl, ctx) {
                return Some(OverlapConflict {
                    existing_idx,
                    existing_span: existing.span,
                    new_span: new_impl.span,
                    kind: OverlapKind::DirectOverlap,
                });
            }
        }
    }
    None
}

/// Recursively check if a TypeData contains any GenericParam anywhere.
/// This is used by the overlap fast path to decide whether structural
/// comparison is sufficient.  A composite type like `Tuple([GenericParam(0)])`
/// contains a generic parameter and must be checked via unification, not
/// structural comparison.
fn contains_generic_param<'input>(data: &TypeData, ctx: &TypeContext<'input>) -> bool {
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
        // DynTrait holds only `Vec<DefId>` (trait refs) — no TypeId
        // children, so it can never contain a GenericParam.  Explicit leaf
        // (same convention as `freshen_generics_bound` and
        // `generic_params_untouched_all`).
        TypeData::DynTrait { .. } => false,
        // All other types (Int, Bool, etc.) have no GenericParams.
        _ => false,
    }
}

/// Check if a TypeId refers to a type that contains any GenericParam.
/// Resolves the TypeId via ctx.get() and recurses into contains_generic_param.
fn contains_generic_param_by_id<'input>(ty: TypeId, ctx: &TypeContext<'input>) -> bool {
    contains_generic_param(ctx.get(ty), ctx)
}

/// Check whether a new impl's for_type overlaps with any existing impl's for_type.
///
/// This is a lighter check than `check_overlap` — it only checks if the
/// *head types* (for_type) unify, without checking trait_id. This is used
/// for inherent impl overlap detection.
pub fn check_inherent_overlap<'input>(
    existing_impls: &[ImplCandidate],
    new_for_type: TypeId,
    ctx: &mut TypeContext<'input>,
) -> Option<OverlapConflict> {
    for (existing_idx, existing) in existing_impls.iter().enumerate() {
        ctx.begin_transaction();
        let unification = ctx.try_unify(new_for_type, existing.for_type, None);
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
