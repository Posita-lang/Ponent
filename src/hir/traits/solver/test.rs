//! Tests for the trait solver's overlap detection (`check_overlap`).
//!
//! These tests verify that `check_overlap` correctly detects semantically
//! identical impls (including those with GenericParam inside composite types)
//! while correctly allowing non-overlapping impls.

use crate::ast::Span;
use crate::hir::traits::ImplCandidate;
use crate::hir::traits::solver::coherence::check_overlap;
use crate::hir::types::{DefId, TypeContext};
use crate::symbol::Symbol;

/// Helper: create a minimal ImplCandidate for testing.
/// `for_type` is the type the impl is for; `trait_args` are the trait's
/// generic arguments (e.g. `Int<32>` in `impl Add<Int<32>> for MyType`).
fn make_candidate(
    trait_id: DefId,
    for_type: crate::hir::types::TypeId,
    trait_args: Vec<crate::hir::types::TypeId>,
) -> ImplCandidate {
    ImplCandidate {
        trait_id,
        for_type,
        methods: vec![],
        resolved_methods: vec![],
        assoc_tys: vec![],
        span: Span::new(0, 0),
        has_auto_deref: false,
        context: vec![],
        where_clause_bounds: vec![],
        arity: 0,
        trait_args,
    }
}

// ── Concrete types (no GenericParam) ───────────────────────────────

#[test]
fn test_concrete_same_type_overlaps() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let type_a = DefId(100);
    let for_ty = ctx.struct_ty(type_a, vec![]);

    let existing = make_candidate(trait_id, for_ty, vec![]);
    let new = make_candidate(trait_id, for_ty, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "identical concrete types should overlap"
    );
}

#[test]
fn test_concrete_different_types_no_overlap() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let type_a = DefId(100);
    let type_b = DefId(101);

    let for_ty_a = ctx.struct_ty(type_a, vec![]);
    let for_ty_b = ctx.struct_ty(type_b, vec![]);

    let existing = make_candidate(trait_id, for_ty_a, vec![]);
    let new = make_candidate(trait_id, for_ty_b, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_none(),
        "different concrete types should NOT overlap"
    );
}

#[test]
fn test_concrete_different_trait_no_overlap() {
    let mut ctx = TypeContext::new();
    let trait_a = DefId(42);
    let trait_b = DefId(99);
    let type_id = DefId(100);
    let for_ty = ctx.struct_ty(type_id, vec![]);

    let existing = make_candidate(trait_a, for_ty, vec![]);
    let new = make_candidate(trait_b, for_ty, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(conflict.is_none(), "different traits should NOT overlap");
}

// ── GenericParam at top level ─────────────────────────────────────

#[test]
fn test_generic_param_top_level_overlaps() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);

    // impl<T> Trait for T  (GenericParam(0, "T"))
    let gp_t = ctx.generic_param(0, Symbol::intern("T"));
    // impl<U> Trait for U  (GenericParam(0, "U")) — same index, different name
    let gp_u = ctx.generic_param(0, Symbol::intern("U"));

    let existing = make_candidate(trait_id, gp_t, vec![]);
    let new = make_candidate(trait_id, gp_u, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "top-level GenericParam with same index should overlap"
    );
}

#[test]
fn test_generic_param_top_level_different_index_overlaps() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);

    // impl<T> Trait for T  (GenericParam(0, "T"))
    let gp_0 = ctx.generic_param(0, Symbol::intern("T"));
    // impl<U> Trait for U  (GenericParam(1, "U")) — different index
    let gp_1 = ctx.generic_param(1, Symbol::intern("U"));

    let existing = make_candidate(trait_id, gp_0, vec![]);
    let new = make_candidate(trait_id, gp_1, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "top-level GenericParam with different indices should overlap (alpha-equivalent)"
    );
}

// ── GenericParam inside composite types (the original bug) ─────────

#[test]
fn test_tuple_with_generic_param_overlaps_same_index() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);

    // impl<T> Trait for (T,)
    let gp_t = ctx.generic_param(0, Symbol::intern("T"));
    let tuple_t = ctx.tuple(vec![gp_t]);

    // impl<U> Trait for (U,)
    let gp_u = ctx.generic_param(0, Symbol::intern("U"));
    let tuple_u = ctx.tuple(vec![gp_u]);

    let existing = make_candidate(trait_id, tuple_t, vec![]);
    let new = make_candidate(trait_id, tuple_u, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "Tuple([GenericParam]) with same index should overlap"
    );
}

#[test]
fn test_tuple_with_generic_param_overlaps_different_index() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);

    // impl<T> Trait for (T,)
    let gp_0 = ctx.generic_param(0, Symbol::intern("T"));
    let tuple_0 = ctx.tuple(vec![gp_0]);

    // impl<U> Trait for (U,) — but U is index 1 (different)
    let gp_1 = ctx.generic_param(1, Symbol::intern("U"));
    let tuple_1 = ctx.tuple(vec![gp_1]);

    let existing = make_candidate(trait_id, tuple_0, vec![]);
    let new = make_candidate(trait_id, tuple_1, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "Tuple([GenericParam]) with different indices should overlap (alpha-equivalent)"
    );
}

#[test]
fn test_adt_with_generic_param_overlaps() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let vec_def_id = DefId(100);

    // impl<T> Trait for Vec<T>
    let gp_t = ctx.generic_param(0, Symbol::intern("T"));
    let vec_t = ctx.struct_ty(vec_def_id, vec![gp_t]);

    // impl<U> Trait for Vec<U>
    let gp_u = ctx.generic_param(0, Symbol::intern("U"));
    let vec_u = ctx.struct_ty(vec_def_id, vec![gp_u]);

    let existing = make_candidate(trait_id, vec_t, vec![]);
    let new = make_candidate(trait_id, vec_u, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "Adt(Vec, [GenericParam]) should overlap — same def_id, same index"
    );
}

#[test]
fn test_adt_with_generic_param_different_index_overlaps() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let vec_def_id = DefId(100);

    // impl<T> Trait for Vec<T>  (index 0)
    let gp_0 = ctx.generic_param(0, Symbol::intern("T"));
    let vec_0 = ctx.struct_ty(vec_def_id, vec![gp_0]);

    // impl<U> Trait for Vec<U>  (index 1 — different parameter position)
    let gp_1 = ctx.generic_param(1, Symbol::intern("U"));
    let vec_1 = ctx.struct_ty(vec_def_id, vec![gp_1]);

    let existing = make_candidate(trait_id, vec_0, vec![]);
    let new = make_candidate(trait_id, vec_1, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "Adt(Vec, [GenericParam]) with different indices should overlap (alpha-equivalent)"
    );
}

// ── Non-overlap: different trait_args ──────────────────────────────

#[test]
fn test_same_head_type_different_trait_args_no_overlap() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let int32 = ctx.int(32, true);
    let int64 = ctx.int(64, true);

    // impl<T> Add<Int<32>> for T
    let gp_t = ctx.generic_param(0, Symbol::intern("T"));
    let existing = make_candidate(trait_id, gp_t, vec![int32]);

    // impl<T> Add<Int<64>> for T
    let gp_u = ctx.generic_param(0, Symbol::intern("U"));
    let new = make_candidate(trait_id, gp_u, vec![int64]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_none(),
        "same head type but different trait args should NOT overlap"
    );
}

// ── Non-overlap: different concrete parts inside composite types ───

#[test]
fn test_composite_different_concrete_elem_no_overlap() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let int32 = ctx.int(32, true);
    let int64 = ctx.int(64, true);

    // impl<T> Trait for (T, Int<32>)
    let gp_t = ctx.generic_param(0, Symbol::intern("T"));
    let tuple_a = ctx.tuple(vec![gp_t, int32]);

    // impl<U> Trait for (U, Int<64>)
    let gp_u = ctx.generic_param(0, Symbol::intern("U"));
    let tuple_b = ctx.tuple(vec![gp_u, int64]);

    let existing = make_candidate(trait_id, tuple_a, vec![]);
    let new = make_candidate(trait_id, tuple_b, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_none(),
        "different concrete elements in composite should NOT overlap"
    );
}

// ── Overlap: generic vs concrete (param can be instantiated) ───────

#[test]
fn test_generic_param_vs_concrete_overlaps() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let type_id = DefId(100);

    // impl<T> Trait for T
    let gp_t = ctx.generic_param(0, Symbol::intern("T"));

    // impl Trait for MyStruct
    let for_ty = ctx.struct_ty(type_id, vec![]);

    let existing = make_candidate(trait_id, gp_t, vec![]);
    let new = make_candidate(trait_id, for_ty, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "generic param vs concrete type should overlap (param can be instantiated)"
    );
}

// ── Multiple existing impls ───────────────────────────────────────

#[test]
fn test_overlap_against_multiple_existing() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let type_a = DefId(100);
    let type_b = DefId(101);
    let type_c = DefId(102);

    let for_ty_a = ctx.struct_ty(type_a, vec![]);
    let for_ty_b = ctx.struct_ty(type_b, vec![]);
    let for_ty_c = ctx.struct_ty(type_c, vec![]);

    let existing_a = make_candidate(trait_id, for_ty_a, vec![]);
    let existing_b = make_candidate(trait_id, for_ty_b, vec![]);

    // New impl matches type_c which is NOT in the existing list
    let new = make_candidate(trait_id, for_ty_c, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing_a.clone(), existing_b.clone()], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_none(),
        "different concrete type against multiple existing"
    );

    // Now try a new impl that matches one of the existing ones
    let new_matching = make_candidate(trait_id, for_ty_a, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing_a, existing_b], &new_matching, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "overlap against one of multiple existing should be detected"
    );
}

// ── Edge: different GenericParam indices in nested ADT ─────────────

#[test]
fn test_adt_nested_generic_param_different_index_overlaps() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let outer_def_id = DefId(100);
    let inner_def_id = DefId(101);

    // impl<T> Trait for Outer<Inner<T>>
    let gp_0 = ctx.generic_param(0, Symbol::intern("T"));
    let inner_t = ctx.struct_ty(inner_def_id, vec![gp_0]);
    let outer_t = ctx.struct_ty(outer_def_id, vec![inner_t]);

    // impl<U> Trait for Outer<Inner<U>>  (U is index 1)
    let gp_1 = ctx.generic_param(1, Symbol::intern("U"));
    let inner_u = ctx.struct_ty(inner_def_id, vec![gp_1]);
    let outer_u = ctx.struct_ty(outer_def_id, vec![inner_u]);

    let existing = make_candidate(trait_id, outer_t, vec![]);
    let new = make_candidate(trait_id, outer_u, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "nested ADT with different GenericParam indices should overlap"
    );
}

// ── Binder types (Forall, Exists, Mu, Nu, Poly) ───────────────────
// These have bound variables inside that should NOT be treated as
// impl-level generic parameters.  The normalization in check_overlap
// currently recurses into binder bodies, which is conceptually wrong
// (bound variables are not impl-level generic params), but in practice
// it works because the transaction is rolled back and try_unify has
// its own alpha-conversion logic.

#[test]
fn test_forall_with_generic_param_body_overlaps() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);

    // ∀X. X   — Forall with bound variable at index 0 referencing itself
    // This is the identity type: ∀X. X
    let gp_0 = ctx.generic_param(0, Symbol::intern("X"));
    let forall_a = ctx.forall(0, Symbol::intern("X"), gp_0);

    // ∀Y. Y   — same type, alpha-equivalent
    let gp_0b = ctx.generic_param(0, Symbol::intern("Y"));
    let forall_b = ctx.forall(0, Symbol::intern("Y"), gp_0b);

    // Both are ∀X.X — must overlap
    let existing = make_candidate(trait_id, forall_a, vec![]);
    let new = make_candidate(trait_id, forall_b, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "alpha-equivalent Forall types should overlap"
    );
}

#[test]
fn test_forall_different_param_index_overlaps() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);

    // ∀X. X at index 0
    let gp_0 = ctx.generic_param(0, Symbol::intern("X"));
    let forall_a = ctx.forall(0, Symbol::intern("X"), gp_0);

    // ∀Y. Y at index 1 — different param_index but same structure
    // This is alpha-equivalent to ∀X. X
    let gp_1 = ctx.generic_param(1, Symbol::intern("Y"));
    let forall_b = ctx.forall(1, Symbol::intern("Y"), gp_1);

    let existing = make_candidate(trait_id, forall_a, vec![]);
    let new = make_candidate(trait_id, forall_b, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "alpha-equivalent Forall with different param_index should overlap"
    );
}

#[test]
fn test_forall_vs_concrete_no_overlap() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);

    // ∀X. X
    let gp_0 = ctx.generic_param(0, Symbol::intern("X"));
    let forall = ctx.forall(0, Symbol::intern("X"), gp_0);

    // Int<32>
    let int32 = ctx.int(32, true);

    // These are different types — the normalization correctly distinguishes
    // them because the GenericParam inside the Forall body is replaced with
    // a fresh InferVar, and Forall(0, $v0) ≠ Int<32> structurally.
    // This is an IMPROVEMENT over the old code, which used try_unify directly
    // and would incorrectly bind the bound variable to Int<32>.
    let existing = make_candidate(trait_id, forall, vec![]);
    let new = make_candidate(trait_id, int32, vec![]);

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    // The new normalization code correctly reports NO overlap here.
    // Forall(0, $v0) and Int<32> are structurally different types.
    assert!(
        conflict.is_none(),
        "Forall vs concrete: normalization correctly distinguishes binder types from concrete types"
    );
}
