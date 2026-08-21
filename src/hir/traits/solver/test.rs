//! Tests for the trait solver's overlap detection (`check_overlap`).
//!
//! These tests verify that `check_overlap` correctly detects semantically
//! identical impls (including those with GenericParam inside composite types)
//! while correctly allowing non-overlapping impls.

use crate::ast::Span;
use crate::hir::infer::TypeVariableKind;
use crate::hir::traits::ImplCandidate;
use crate::hir::traits::solver::coherence::{check_overlap, specializes};
use crate::hir::traits::solver::obligation::{
    Obligation, ObligationCause, ObligationCauseCode, Predicate,
};
use crate::hir::traits::solver::search_graph::canonicalize_goal_key;
use crate::hir::types::{DefId, TypeContext};
use crate::symbol::Symbol;

/// Helper: create a minimal ImplCandidate for testing.
/// `for_type` is the type the impl is for; `trait_args` are the trait's
/// generic arguments (e.g. `Int<32>` in `impl Add<Int<32>> for MyType`).
fn make_candidate(
    trait_id: DefId,
    for_type: crate::hir::types::TypeId,
    trait_args: Vec<crate::hir::types::TypeId>,
) -> ImplCandidate<'static> {
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

/// Like `make_candidate`, but with where-clause bounds
/// `(self_ty, trait_id, args)` — for the where-clause overlap tests.
fn make_candidate_with_bounds(
    trait_id: DefId,
    for_type: crate::hir::types::TypeId,
    trait_args: Vec<crate::hir::types::TypeId>,
    where_clause_bounds: Vec<(
        crate::hir::types::TypeId,
        DefId,
        Vec<crate::hir::types::TypeId>,
    )>,
) -> ImplCandidate<'static> {
    ImplCandidate {
        trait_id,
        for_type,
        methods: vec![],
        resolved_methods: vec![],
        assoc_tys: vec![],
        span: Span::new(0, 0),
        has_auto_deref: false,
        context: vec![],
        where_clause_bounds,
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

// ── Canonical cache correctness tests ──

#[test]
fn test_canonicalize_goal_key_caches_concrete_goals() {
    let mut ctx = TypeContext::new();

    // Create a goal without InferVars — should be cacheable.
    let int_ty = ctx.int(32, true);
    let sized = Predicate::Sized { ty: int_ty };
    let obligation = Obligation {
        cause: ObligationCause {
            span: Span::new(0, 0),
            code: ObligationCauseCode::Misc,
        },
        predicate: sized,
        recursion_depth: 0,
    };

    let key = canonicalize_goal_key(&obligation, &ctx, 0);
    assert!(
        key.is_some(),
        "goal without InferVar should be cacheable (key is Some)"
    );
}

#[test]
fn test_same_goal_same_canonical_key() {
    let mut ctx = TypeContext::new();

    let int_ty = ctx.int(64, true);
    let sized1 = Predicate::Sized { ty: int_ty };
    let ob1 = Obligation {
        cause: ObligationCause {
            span: Span::new(0, 0),
            code: ObligationCauseCode::Misc,
        },
        predicate: sized1,
        recursion_depth: 0,
    };

    let int_ty2 = ctx.int(64, true);
    let sized2 = Predicate::Sized { ty: int_ty2 };
    let ob2 = Obligation {
        cause: ObligationCause {
            span: Span::new(0, 0),
            code: ObligationCauseCode::Misc,
        },
        predicate: sized2,
        recursion_depth: 0,
    };

    let key1 = canonicalize_goal_key(&ob1, &ctx, 0);
    let key2 = canonicalize_goal_key(&ob2, &ctx, 0);
    assert_eq!(
        key1, key2,
        "structurally identical goals should produce identical canonical keys"
    );
}

#[test]
fn test_different_goal_different_canonical_key() {
    let mut ctx = TypeContext::new();

    let int32 = ctx.int(32, true);
    let int64 = ctx.int(64, true);

    let sized32 = Predicate::Sized { ty: int32 };
    let ob32 = Obligation {
        cause: ObligationCause {
            span: Span::new(0, 0),
            code: ObligationCauseCode::Misc,
        },
        predicate: sized32,
        recursion_depth: 0,
    };

    let sized64 = Predicate::Sized { ty: int64 };
    let ob64 = Obligation {
        cause: ObligationCause {
            span: Span::new(0, 0),
            code: ObligationCauseCode::Misc,
        },
        predicate: sized64,
        recursion_depth: 0,
    };

    let key32 = canonicalize_goal_key(&ob32, &ctx, 0);
    let key64 = canonicalize_goal_key(&ob64, &ctx, 0);
    assert_ne!(
        key32, key64,
        "structurally different goals should produce different canonical keys"
    );
}

// ── Regression: Ref lifetime preservation in freshening ────────────

/// The `specializes` freshening must PRESERVE explicit Ref lifetimes.
/// Two impls `impl<T> Tr for &'a T` and `impl<T> Tr for &'b T` with
/// DIFFERENT explicit lifetimes must NOT specialize: the unifier treats
/// two distinct explicit regions as a mismatch.  Regression: freshening
/// dropped the lifetime to `None`, making both sides "elided" and unifying
/// — a false-positive specialization.
#[test]
fn test_specializes_distinguishes_ref_lifetimes() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let gp = ctx.generic_param(0, Symbol::intern("T"));
    let ref_a = ctx.reference_with_lifetime(gp, false, Some(Symbol::intern("'a")));
    let ref_b = ctx.reference_with_lifetime(gp, false, Some(Symbol::intern("'b")));
    let specific = make_candidate(trait_id, ref_a, vec![]);
    let general = make_candidate(trait_id, ref_b, vec![]);
    assert!(
        !specializes(&mut ctx, &specific, &general),
        "impls differing only in an explicit Ref lifetime must not specialize"
    );
}

/// The `specializes` freshening must ALSO preserve the shared-lifetime
/// case: `&'a T` vs `&'a T` (same explicit lifetime) DOES specialize —
/// the fix must not over-reject.
#[test]
fn test_specializes_same_ref_lifetime_specializes() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let gp = ctx.generic_param(0, Symbol::intern("T"));
    let ref_a1 = ctx.reference_with_lifetime(gp, false, Some(Symbol::intern("'a")));
    let ref_a2 = ctx.reference_with_lifetime(gp, false, Some(Symbol::intern("'a")));
    let specific = make_candidate(trait_id, ref_a1, vec![]);
    let general = make_candidate(trait_id, ref_a2, vec![]);
    assert!(
        specializes(&mut ctx, &specific, &general),
        "impls with the SAME explicit Ref lifetime must specialize"
    );
}

// ── Regression: check_overlap slow-path iteration isolation ────────

/// The slow path must be self-contained per iteration: a PARTIAL
/// unification (for_type unifies but a trait_arg does not) must not leak
/// bindings into the next iteration's judgement.  Regression: the slow
/// path had no per-iteration transaction (only the caller's outer one),
/// so iteration i's leftover bindings could pollute iteration i+1.
#[test]
fn test_overlap_slow_path_iteration_isolation() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    // existing[0]: impl Tr for (T,) with trait_arg Int<32> — the FIRST
    // iteration's for_type (both contain GenericParam(0)) unifies, but the
    // trait_arg Int<32> vs Int<64> does NOT: a partial failure that, before
    // the fix, left fresh-var bindings behind.
    let gp = ctx.generic_param(0, Symbol::intern("T"));
    let tuple = ctx.tuple(vec![gp]);
    let int32 = ctx.int(32, true);
    let int64 = ctx.int(64, true);
    let existing0 = make_candidate(trait_id, tuple, vec![int32]);
    // existing[1]: impl Tr for (Int<64>,) — unrelated, must NOT overlap.
    let tuple64 = ctx.tuple(vec![int64]);
    let existing1 = make_candidate(trait_id, tuple64, vec![]);
    let new = make_candidate(trait_id, ctx.tuple(vec![gp]), vec![int64]);

    let existing = vec![existing0, existing1];
    ctx.begin_transaction();
    let conflict = check_overlap(&existing, &new, &mut ctx);
    ctx.rollback_transaction();

    // new (T,) with Int<64> vs existing0 (T,) with Int<32>: for_type
    // unifies but trait args differ → NOT a conflict; existing1 (Int<64>,)
    // is unrelated → overall no conflict.  A leaked binding from iteration 0
    // (fresh var T := (T,)) would corrupt iteration 1's unification.
    assert!(
        conflict.is_none(),
        "the slow path must isolate iterations — no overlap expected here: {:?}",
        conflict
    );
}

// ── where-clause overlap detection ─────────────────────────────────

/// Two impls whose where-clause bounds are MUTUALLY EXCLUSIVE (same trait,
/// same self type, but incompatible concrete args — `MyType: Foo<Int<32>>`
/// vs `MyType: Foo<Int<64>>`) must NOT overlap.  This mirrors the existing
/// fast-path rule ("two impls of the same trait with different generic
/// arguments on the same type are NOT overlapping") applied to bounds.
/// Regression: `check_overlap` ignored `where_clause_bounds` entirely.
#[test]
fn test_where_clause_exclusive_bounds_no_overlap_fast_path() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let foo_trait = DefId(200);
    let for_ty = ctx.struct_ty(DefId(100), vec![]);
    let int32 = ctx.int(32, true);
    let int64 = ctx.int(64, true);
    // impl Tr for MyType where MyType: Foo<Int<32>>
    let existing = make_candidate_with_bounds(
        trait_id,
        for_ty,
        vec![],
        vec![(for_ty, foo_trait, vec![int32])],
    );
    // impl Tr for MyType where MyType: Foo<Int<64>>
    let new = make_candidate_with_bounds(
        trait_id,
        for_ty,
        vec![],
        vec![(for_ty, foo_trait, vec![int64])],
    );

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_none(),
        "mutually exclusive where-clause bounds must NOT overlap (fast path): {:?}",
        conflict
    );
}

/// Same as above, but through the SLOW path (for_type contains a
/// GenericParam, so unification is used).
#[test]
fn test_where_clause_exclusive_bounds_no_overlap_slow_path() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let foo_trait = DefId(200);
    let gp = ctx.generic_param(0, Symbol::intern("T"));
    let tuple = ctx.tuple(vec![gp]);
    let int32 = ctx.int(32, true);
    let int64 = ctx.int(64, true);
    // impl<T: Foo<Int<32>>> Tr for (T,)
    let existing =
        make_candidate_with_bounds(trait_id, tuple, vec![], vec![(gp, foo_trait, vec![int32])]);
    // impl<T: Foo<Int<64>>> Tr for (T,)
    let new = make_candidate_with_bounds(
        trait_id,
        ctx.tuple(vec![gp]),
        vec![],
        vec![(gp, foo_trait, vec![int64])],
    );

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_none(),
        "mutually exclusive where-clause bounds must NOT overlap (slow path): {:?}",
        conflict
    );
}

/// Two impls with DIFFERENT trait bounds (`MyType: Display` vs
/// `MyType: Debug`) DO overlap — a type can satisfy both, so the bounds
/// are not mutually exclusive.  Matches rustc coherence.
#[test]
fn test_where_clause_different_traits_still_overlap() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let display_trait = DefId(200);
    let debug_trait = DefId(201);
    let for_ty = ctx.struct_ty(DefId(100), vec![]);
    let existing = make_candidate_with_bounds(
        trait_id,
        for_ty,
        vec![],
        vec![(for_ty, display_trait, vec![])],
    );
    let new = make_candidate_with_bounds(
        trait_id,
        for_ty,
        vec![],
        vec![(for_ty, debug_trait, vec![])],
    );

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "different trait bounds can hold simultaneously — still overlapping: {:?}",
        conflict
    );
}

/// Two impls with IDENTICAL where-clause bounds DO overlap.
#[test]
fn test_where_clause_same_bounds_still_overlap() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let foo_trait = DefId(200);
    let for_ty = ctx.struct_ty(DefId(100), vec![]);
    let int32 = ctx.int(32, true);
    let existing = make_candidate_with_bounds(
        trait_id,
        for_ty,
        vec![],
        vec![(for_ty, foo_trait, vec![int32])],
    );
    let new = make_candidate_with_bounds(
        trait_id,
        for_ty,
        vec![],
        vec![(for_ty, foo_trait, vec![int32])],
    );

    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();

    assert!(
        conflict.is_some(),
        "identical where-clause bounds — still overlapping: {:?}",
        conflict
    );
}

// ── fresh-var counter is TypeContext-local (regression: was process-global) ──

/// The overlap-fresh-var counter must be per-`TypeContext`: two contexts
/// each start their own counter at the base offset, so (a) parallel tests
/// and long-running processes never share a monotonic global, and (b) the
/// first fresh var of every context has the SAME id (deterministic).
#[test]
fn test_overlap_fresh_var_counter_is_context_local() {
    let mut ctx1 = TypeContext::new();
    let mut ctx2 = TypeContext::new();
    let fv1 = ctx1.alloc_overlap_fresh_var(0);
    let fv2 = ctx2.alloc_overlap_fresh_var(0);
    let id1 = match ctx1.get(fv1) {
        crate::hir::types::TypeData::InferVar { id, .. } => *id,
        other => panic!("expected InferVar, got {:?}", other),
    };
    let id2 = match ctx2.get(fv2) {
        crate::hir::types::TypeData::InferVar { id, .. } => *id,
        other => panic!("expected InferVar, got {:?}", other),
    };
    assert_eq!(
        id1, id2,
        "each TypeContext must start its own fresh-var counter at the same base"
    );
    // And within one context the counter is monotonic.
    let fv3 = ctx1.alloc_overlap_fresh_var(0);
    let id3 = match ctx1.get(fv3) {
        crate::hir::types::TypeData::InferVar { id, .. } => *id,
        other => panic!("expected InferVar, got {:?}", other),
    };
    assert!(
        id3 > id1,
        "within one context the counter must be monotonic"
    );
}

// ── slow path reuses freshen_generics (with binder shadowing) ─────

/// The slow path now normalizes via the SAME `freshen_generics` as
/// `specializes` (binder shadowing included): a free GenericParam outside
/// a Forall is freshened, while the quantifier-BOUND variable inside is
/// kept — alpha-equivalent composite types still overlap.
#[test]
fn test_overlap_slow_path_freshen_with_binder_shadowing() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    // ∀X. (X,) paired with a free GenericParam: Tuple([Forall(0,X,X), GP(0)])
    let gp0 = ctx.generic_param(0, Symbol::intern("X"));
    let forall_a = ctx.forall(0, Symbol::intern("X"), gp0);
    let outer_a = ctx.tuple(vec![forall_a, gp0]);
    // Alpha-equivalent: ∀Y. (Y,) with a free GP at a different index.
    let gp1 = ctx.generic_param(1, Symbol::intern("Y"));
    let forall_b = ctx.forall(1, Symbol::intern("Y"), gp1);
    let outer_b = ctx.tuple(vec![forall_b, gp1]);

    let existing = make_candidate(trait_id, outer_a, vec![]);
    let new = make_candidate(trait_id, outer_b, vec![]);
    ctx.begin_transaction();
    let conflict = check_overlap(&[existing], &new, &mut ctx);
    ctx.rollback_transaction();
    assert!(
        conflict.is_some(),
        "alpha-equivalent Forall+free-GP composites must overlap: {:?}",
        conflict
    );
}

// ── DynTrait explicit leaf (fast-path structural comparison) ────────

/// DynTrait holds only `Vec<DefId>` — no GenericParam — so it is a leaf
/// for `contains_generic_param` and overlaps are decided by structural
/// comparison: identical dyn types overlap, different ones do not.
#[test]
fn test_overlap_dyn_trait_structural_leaf() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let dyn_a1 = ctx.alloc(crate::hir::types::TypeData::DynTrait {
        traits: vec![DefId(100)],
    });
    let dyn_a2 = ctx.alloc(crate::hir::types::TypeData::DynTrait {
        traits: vec![DefId(100)],
    });
    let dyn_b = ctx.alloc(crate::hir::types::TypeData::DynTrait {
        traits: vec![DefId(101)],
    });

    // Identical dyn types → overlap (fast path).
    ctx.begin_transaction();
    let same = check_overlap(
        &[make_candidate(trait_id, dyn_a1, vec![])],
        &make_candidate(trait_id, dyn_a2, vec![]),
        &mut ctx,
    );
    ctx.rollback_transaction();
    assert!(
        same.is_some(),
        "identical dyn trait types must overlap: {:?}",
        same
    );

    // Different dyn types → no overlap (fast path structural difference).
    ctx.begin_transaction();
    let diff = check_overlap(
        &[make_candidate(trait_id, dyn_a1, vec![])],
        &make_candidate(trait_id, dyn_b, vec![]),
        &mut ctx,
    );
    ctx.rollback_transaction();
    assert!(
        diff.is_none(),
        "different dyn trait types must NOT overlap: {:?}",
        diff
    );
}

// ── specializes explicit balanced transactions (no frame leak) ─────

/// `specializes` opens TWO transactions (freshening + unification) and must
/// pop BOTH symmetrically: after any call the transaction depth returns to
/// its entry value, so repeated calls never grow the stack.
#[test]
fn test_specializes_balanced_transactions_no_depth_leak() {
    let mut ctx = TypeContext::new();
    let trait_id = DefId(42);
    let gp = ctx.generic_param(0, Symbol::intern("T"));
    let tuple = ctx.tuple(vec![gp]);
    let specific = make_candidate(trait_id, tuple, vec![]);
    let gp_u = ctx.generic_param(0, Symbol::intern("U"));
    let general = make_candidate(trait_id, ctx.tuple(vec![gp_u]), vec![]);

    let depth_before = ctx.transaction_depth();
    let _ = specializes(&mut ctx, &specific, &general);
    assert_eq!(
        ctx.transaction_depth(),
        depth_before,
        "specializes must not leak transaction frames"
    );
    // And a second call is equally clean (no accumulated growth).
    let _ = specializes(&mut ctx, &specific, &general);
    assert_eq!(
        ctx.transaction_depth(),
        depth_before,
        "repeated specializes calls must not grow the transaction stack"
    );
}
