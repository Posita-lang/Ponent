use crate::ast::FileId;
use super::*;

fn new_ctx() -> TypeContext<'static> {
    TypeContext::new()
}

/// `find_type` looks up an ALREADY-INTERNED logical type in the
/// factory's dedup table: the same `TypeData` built from the same
/// children must return the `TypeId` that `alloc` originally
/// produced.  (The bottom-up intern invariant — children allocated
/// before the parent — is what makes identical keys hit.)
#[test]
fn test_find_type_hit_and_dedup() {
    let mut ctx = TypeContext::new();
    let a = ctx.int(32, true);
    let b = ctx.bool();
    let t1 = ctx.tuple(smallvec![a, b]);
    // Same logical type, rebuilt from the same interned children:
    // find_type must hit and agree with the allocated TypeId.
    let found = ctx.find_type(&TypeData::Tuple {
        elems: smallvec![a, b],
    });
    assert_eq!(found, Some(t1));
}

/// Volatile types (`InferVar` / `SkolemVar` and any composite that
/// embeds them) are scope-sensitive and NEVER interned — `alloc`
/// skips caching them, so `find_type` must return `None` rather than
/// fabricating a stale intern for a one-shot variable.
#[test]
fn test_find_type_skips_volatile() {
    let mut ctx = TypeContext::new();
    let var = ctx.alloc(TypeData::InferVar { id: 0, universe: 0 });
    assert_eq!(
        ctx.find_type(&TypeData::InferVar { id: 0, universe: 0 }),
        None,
        "a raw InferVar is never interned"
    );
    // A composite embedding the InferVar is volatile too — no table
    // entry, so find_type declines.
    let tup = TypeData::Tuple {
        elems: smallvec![var],
    };
    assert_eq!(
        ctx.find_type(&tup),
        None,
        "a composite with an InferVar is volatile"
    );
}

/// Pin: `place_is_prefix_of` implements "target is a PREFIX
/// of frozen" (target equals some ancestor along frozen's base chain) —
/// NOT a structure-aligned match.  `a.b` is a prefix of `a.b.c`;
/// `a.b` is NOT a prefix of `a.c` (sibling fields).  A proposed
/// "simultaneous decomposition" rewrite would break the first case
/// (field names at the same depth differ: `b` vs `c`), so this test
/// pins the correct semantics against misrepairs.
#[test]
fn test_place_is_prefix_of() {
    use crate::hir::place::place_is_prefix_of;
    let root_a = FrozenPlace::Root(Symbol::intern("a"));
    let ab = FrozenPlace::Field(
        Box::new(FrozenPlace::Root(Symbol::intern("a"))),
        Symbol::intern("b"),
    );
    let abc = FrozenPlace::Field(
        Box::new(FrozenPlace::Field(
            Box::new(FrozenPlace::Root(Symbol::intern("a"))),
            Symbol::intern("b"),
        )),
        Symbol::intern("c"),
    );
    let ac = FrozenPlace::Field(
        Box::new(FrozenPlace::Root(Symbol::intern("a"))),
        Symbol::intern("c"),
    );
    // `a` is a prefix of `a.b` and `a.b.c`.
    assert!(place_is_prefix_of(&root_a, &ab));
    assert!(place_is_prefix_of(&root_a, &abc));
    // `a.b` is a prefix of `a.b.c`.
    assert!(place_is_prefix_of(&ab, &abc));
    // `a.b` is NOT a prefix of `a.c` (siblings) — in either direction.
    assert!(!place_is_prefix_of(&ab, &ac));
    assert!(!place_is_prefix_of(&ac, &ab));
    // Equality is a prefix.
    assert!(place_is_prefix_of(&abc, &abc));
}

/// Constant-index granularity (`FrozenPlace::ConstIndex`, mirroring
/// rustc's `ProjectionElem::ConstantIndex`): `a[0]` and `a[1]` are
/// DISTINCT places — freezing `a[0]` does NOT freeze `a[1]`; a DYNAMIC
/// index (`a[i]`) conservatively overlaps every constant index on the
/// same base, in both directions.
#[test]
fn test_place_is_prefix_of_const_index() {
    use crate::hir::place::place_is_prefix_of;
    let root_a = FrozenPlace::Root(Symbol::intern("a"));
    let a0 = FrozenPlace::ConstIndex(Box::new(root_a.clone()), 0);
    let a1 = FrozenPlace::ConstIndex(Box::new(root_a.clone()), 1);
    let ai = FrozenPlace::Index(Box::new(root_a.clone()));

    // `a` is a prefix of `a[0]` / `a[1]` (writing the whole array
    // touches every element).
    assert!(place_is_prefix_of(&root_a, &a0));
    assert!(place_is_prefix_of(&root_a, &a1));
    // `a[0]` is a prefix of itself (equality).
    assert!(place_is_prefix_of(&a0, &a0));
    // `a[0]` and `a[1]` are DISTINCT constant elements — neither is a
    // prefix of the other (freezing `a[0]` must NOT freeze `a[1]`).
    assert!(!place_is_prefix_of(&a0, &a1));
    assert!(!place_is_prefix_of(&a1, &a0));
    // Dynamic `a[i]` conservatively overlaps constant `a[0]` — in
    // BOTH directions (`a[i]` may equal `a[0]`).
    assert!(place_is_prefix_of(&a0, &ai));
    assert!(place_is_prefix_of(&ai, &a0));
    // Dynamic `a[i]` overlaps itself (equality).
    assert!(place_is_prefix_of(&ai, &ai));
}

/// Raw pointers (`Pointer`/`Ptr`) are Copy — a raw
/// pointer is a usize-sized integer that owns nothing and has no
/// `Drop`; copying an address has no memory cost or ownership
/// consequence.  The pointee does not matter.
#[test]
fn test_type_is_copy_raw_pointer() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let ptr = ctx.alloc(TypeData::Pointer { ty: int_ty });
    assert!(
        ctx.type_is_copy(ptr),
        "raw `Pointer` must be Copy"
    );
    let usize_ty = ctx.alloc(TypeData::USize);
    let ptr2 = ctx.alloc(TypeData::Ptr {
        size: usize_ty,
        pointee: int_ty,
    });
    assert!(
        ctx.type_is_copy(ptr2),
        "raw `Ptr` must be Copy"
    );
}

// -- TypeId tag --
#[test]
fn test_typeid_tag_encode_decode() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    assert_eq!(int_ty.tag(), TypeTag::Int);

    let fn_ty = ctx.function(smallvec![int_ty], ctx.bool());
    assert_eq!(fn_ty.tag(), TypeTag::Fn);
}

#[test]
fn test_typeid_tag_index_roundtrip() {
    let mut ctx = new_ctx();
    let b = ctx.bool();
    let idx = b.index();
    assert_eq!(**ctx.slot(idx), TypeData::Bool);
}

// -- Variance --
#[test]
fn test_variance_fn_param_contravariant() {
    let ctx = TypeContext::new();
    assert_eq!(ctx.check_variance(0, ctx.bool(), -1), true);
}

#[test]
fn test_variance_invariant_ref() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "T".into());
    let ref_ty = ctx.reference(p0, false);
    // Ref is invariant: param inside Ref cannot be at covariant or contravariant position
    assert!(!ctx.check_variance(0, ref_ty, 1));
    assert!(!ctx.check_variance(0, ref_ty, -1));
    // A covariant tuple containing the param works
    let tup_ty = ctx.tuple(smallvec![p0]);
    assert!(ctx.check_variance(0, tup_ty, 1));
}

/// The region solver's unification consistency: `&'a T` must NOT
/// unify with `&'b T` when both explicit lifetimes differ (SYNTAX.md
/// §Explicit Lifetime Parameters — "mismatches cause compile
/// errors"; rustc's "lifetime mismatch").
#[test]
fn test_unify_explicit_lifetime_mismatch_rejected() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let a = ctx.reference_with_lifetime(int_ty, false, Some(Symbol::intern("a")));
    let b = ctx.reference_with_lifetime(int_ty, false, Some(Symbol::intern("b")));
    assert!(
        ctx.unify(a, b).is_err(),
        "&'a T must not unify with &'b T (different explicit lifetimes)"
    );
}

/// The SAME explicit lifetime unifies with itself.
#[test]
fn test_unify_same_explicit_lifetime_ok() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let a = ctx.reference_with_lifetime(int_ty, false, Some(Symbol::intern("a")));
    assert!(ctx.unify(a, a).is_ok(), "&'a T must unify with itself");
}

/// An ELIDED lifetime (`&T` — `None`) is compatible with an explicit
/// one (`&'a T`): the elided side does not constrain the region.
#[test]
fn test_unify_elided_with_explicit_ok() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let elided = ctx.reference(int_ty, false);
    let explicit = ctx.reference_with_lifetime(int_ty, false, Some(Symbol::intern("a")));
    assert!(
        ctx.unify(elided, explicit).is_ok(),
        "elided &T must unify with explicit &'a T"
    );
}

/// SUBTYPE-level explicit-lifetime consistency (rustc `regions()`'s
/// invariant branch): `&'a T` is NOT a subtype of `&'b T` when the
/// two explicit regions differ — the pure-type relation stays
/// invariant; the region solver verifies `'a: 'b` at the signature
/// level (SYNTAX.md §Explicit Lifetime Parameters).
#[test]
fn test_subtype_explicit_lifetime_mismatch_rejected() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let a = ctx.reference_with_lifetime(int_ty, false, Some(Symbol::intern("a")));
    let b = ctx.reference_with_lifetime(int_ty, false, Some(Symbol::intern("b")));
    assert!(
        !ctx.subtype(a, b),
        "&'a T must NOT be a subtype of &'b T (different explicit lifetimes)"
    );
    assert!(
        !ctx.subtype(b, a),
        "&'b T must NOT be a subtype of &'a T (different explicit lifetimes)"
    );
}

/// The SAME explicit lifetime is trivially in subtype relation.
#[test]
fn test_subtype_same_explicit_lifetime_ok() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let a = ctx.reference_with_lifetime(int_ty, false, Some(Symbol::intern("a")));
    assert!(ctx.subtype(a, a), "&'a T must be a subtype of itself");
}

/// An ELIDED lifetime is compatible with an explicit one in BOTH
/// directions at the subtype level (the `None` side does not
/// constrain the region).
#[test]
fn test_subtype_elided_with_explicit_ok() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let elided = ctx.reference(int_ty, false);
    let explicit = ctx.reference_with_lifetime(int_ty, false, Some(Symbol::intern("a")));
    assert!(
        ctx.subtype(elided, explicit) && ctx.subtype(explicit, elided),
        "elided &T and explicit &'a T must be mutually subtypeable"
    );
}

#[test]
fn test_variance_tuple_covariant() {
    let ctx = TypeContext::new();
    assert_eq!(ctx.check_variance(0, ctx.bool(), 1), true);
}

#[test]
fn test_variance_nested_fn() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "T".into());
    let bool_ty = ctx.bool();
    let int_ty = ctx.int(32, true);
    let inner = ctx.function(smallvec![p0], bool_ty);
    let outer = ctx.function(smallvec![int_ty], inner);
    assert!(ctx.type_contains_param(0, outer));
}

// -- Characteristic κ --
#[test]
fn test_characteristic_bool() {
    let mut ctx = TypeContext::new();
    assert_eq!(
        ctx.characteristic(ctx.bool()),
        Characteristic::FiniteExhaustible(2)
    );
}

#[test]
fn test_characteristic_int32() {
    let mut ctx = TypeContext::new();
    let i32 = ctx.int(32, true);
    assert_eq!(
        ctx.characteristic(i32),
        Characteristic::FiniteExhaustible(2u64.pow(32) as usize)
    );
}

#[test]
fn test_characteristic_unit() {
    let mut ctx = TypeContext::new();
    assert_eq!(
        ctx.characteristic(ctx.unit()),
        Characteristic::FiniteExhaustible(1)
    );
}

#[test]
fn test_characteristic_fn() {
    let mut ctx = TypeContext::new();
    let bool_ty = ctx.bool();
    let fn_ty = ctx.function(smallvec![bool_ty], bool_ty);
    // Bool → Bool has 2^2 = 4 inhabitants.
    assert_eq!(
        ctx.characteristic(fn_ty),
        Characteristic::FiniteExhaustible(4)
    );
}

#[test]
fn test_characteristic_slice() {
    let mut ctx = TypeContext::new();
    let bool_ty = ctx.bool();
    let slice_ty = ctx.slice(bool_ty);
    assert_eq!(
        ctx.characteristic(slice_ty),
        Characteristic::InfiniteEnumerable
    );
}

#[test]
fn test_characteristic_ksp_convergence() {
    // Simulate a recursive type pattern where KSP-style iteration
    // is needed to reach convergence:
    //   μX. Bool × X  — a recursive type representing an infinite
    //   stream of Bools.  We encode it using Forall/GenericParam.
    let mut ctx = TypeContext::new();
    let bool_ty = ctx.bool();
    let p0 = ctx.generic_param(0, "X".into());
    // Simulate μX: Bool × X  via Forall(0, "X", (Bool, X) ⇒ X)
    let body = {
        let tup = ctx.tuple(smallvec![bool_ty, p0]);
        ctx.function(smallvec![tup], p0)
    };
    let ty = ctx.forall(0, "X".into(), body);
    // With axiom links connecting GPIO occurrences as bidirectional edges,
    // the remaining cycle includes a contravariant edge (Fn param → Tuple),
    // so the result is Undecidable.
    let kappa = ctx.characteristic(ty);
    assert_eq!(
        kappa,
        Characteristic::Undecidable,
        "recursive stream type with contravariant path should be Undecidable"
    );
}

// -- B5a: ADT recursion detection via variant-payload expansion --
#[test]
fn kappa_recursive_enum_covariant_infinite() {
    // A monomorphic self-recursive enum with a covariant payload
    // (shape: `type Tree = enum { Node(Tree, Tree) }`): the expansion
    // back-edge makes a covariant cycle → InfiniteEnumerable (the
    // pre-B5a verdict was FiniteExhaustible(usize::MAX) — recursion was
    // invisible to the graph).
    let mut ctx = TypeContext::new();
    let tree = ctx.enum_ty(DefId(9001), smallvec![]);
    let pair = ctx.tuple(smallvec![tree, tree]);
    ctx.register_adt_expansion(tree, vec![pair]);
    assert_eq!(
        ctx.characteristic(tree),
        Characteristic::InfiniteEnumerable,
        "covariant self-referential enum must be InfiniteEnumerable"
    );
}

#[test]
fn kappa_recursive_enum_invariant_undecidable() {
    // A self-reference under a contravariant position (shape:
    // `type Evil = enum { E(Evil -> ()) }`): the cycle carries a
    // contravariant edge → Undecidable.
    let mut ctx = TypeContext::new();
    let evil = ctx.enum_ty(DefId(9002), smallvec![]);
    let fn_ty = ctx.function(smallvec![evil], ctx.unit());
    ctx.register_adt_expansion(evil, vec![fn_ty]);
    assert_eq!(
        ctx.characteristic(evil),
        Characteristic::Undecidable,
        "contravariant self-referential enum must be Undecidable"
    );
}

#[test]
fn kappa_per_scc_refinement_covariant_root() {
    // B5b: the root sits in a COVARIANT cycle while a DISJOINT
    // non-covariant cycle is also reachable:
    //   wrap = enum { V(wrap, evil) }   (covariant self-cycle)
    //   evil = enum { E(evil -> ()) }   (contravariant self-cycle)
    // The pre-B5b global-flag classifier reported the root
    // `Undecidable` (worst SCC's verdict); per-SCC classification
    // gives the root its own SCC's verdict — InfiniteEnumerable.
    let mut ctx = TypeContext::new();
    let evil = ctx.enum_ty(DefId(9003), smallvec![]);
    let evil_fn = ctx.function(smallvec![evil], ctx.unit());
    ctx.register_adt_expansion(evil, vec![evil_fn]);
    let wrap = ctx.enum_ty(DefId(9004), smallvec![]);
    let pair = ctx.tuple(smallvec![wrap, evil]);
    ctx.register_adt_expansion(wrap, vec![pair]);
    assert_eq!(
        ctx.characteristic(wrap),
        Characteristic::InfiniteEnumerable,
        "per-SCC: covariant root SCC must not inherit the distant SCC's verdict"
    );
}

#[test]
fn kappa_generic_recursive_enum_instantiation() {
    // The shape the resolver hook (B5a) registers for a generic
    // self-recursive enum instantiation:
    //   `type List<T> = enum { Cons(T, List<T>), Nil }` with T = Int<32>
    //   → payloads [Tuple{Int<32>, List[Int<32>]}, Unit]
    let mut ctx = TypeContext::new();
    let arg = ctx.int(32, true);
    let list = ctx.enum_ty(DefId(9005), smallvec![arg]);
    let cons_payload = ctx.tuple(smallvec![arg, list]);
    ctx.register_adt_expansion(list, vec![cons_payload, ctx.unit()]);
    assert_eq!(
        ctx.characteristic(list),
        Characteristic::InfiniteEnumerable,
        "self-referential generic enum instantiation must be InfiniteEnumerable"
    );
}

// -- Transaction --
#[test]
fn test_transaction_commit() {
    let mut ctx = TypeContext::new();
    let a = ctx.int(32, true);
    let b = ctx.int(64, false);
    assert!(ctx.unify(a, b).is_err());
}

#[test]
fn test_transaction_rollback() {
    let mut ctx = TypeContext::new();
    let a = ctx.int(32, true);
    let bool_ty = ctx.bool();
    ctx.begin_transaction();
    ctx.set_binding(a, bool_ty);
    ctx.rollback_transaction();
    assert!(ctx.resolve_binding(a) == a);
}

#[test]
fn test_transaction_nested() {
    let mut ctx = TypeContext::new();
    let a = ctx.int(32, true);
    let bool_ty = ctx.bool();
    let unit_ty = ctx.unit();
    ctx.begin_transaction();
    ctx.set_binding(a, bool_ty);
    ctx.begin_transaction();
    ctx.set_binding(a, unit_ty);
    ctx.rollback_transaction();
    assert_eq!(ctx.resolve_binding(a), bool_ty);
    ctx.commit_transaction();
}

#[test]
/// Inner commit + outer rollback: verify that the outer transaction's undo log
/// correctly absorbs the inner transaction's changes on commit.  Without the
/// log-merge in `commit_transaction`, the outer rollback would leave the inner
/// transaction's modifications in place, breaking the atomicity semantics.
fn test_transaction_nested_commit_outer_rollback() {
    let mut ctx = TypeContext::new();
    let a = ctx.int(32, true);
    let bool_ty = ctx.bool();
    let unit_ty = ctx.unit();

    // Outer: set a → Bool
    ctx.begin_transaction();
    ctx.set_binding(a, bool_ty);
    assert_eq!(ctx.resolve_binding(a), bool_ty);

    // Inner: set a → Unit, then commit
    ctx.begin_transaction();
    ctx.set_binding(a, unit_ty);
    assert_eq!(ctx.resolve_binding(a), unit_ty);
    ctx.commit_transaction();

    // After inner commit, a is still Unit
    assert_eq!(ctx.resolve_binding(a), unit_ty);

    // Outer rollback: should restore a to its state BEFORE the outer began
    ctx.rollback_transaction();
    assert_eq!(ctx.resolve_binding(a), a);
}

#[test]
/// Three-level nested transaction with commit/rollback at each layer.
/// Verifies that the undo-log merge across three levels correctly restores
/// the outermost state when the outermost rolls back.
///
/// Layer-3 commit → merge into Layer-2 log.
/// Layer-2 commit → merge (Layer-3 ∪ Layer-2) into Layer-1 log.
/// Layer-1 rollback → reverse-apply the combined log → initial state.
fn test_transaction_nested_three_level() {
    let mut ctx = TypeContext::new();
    let a = ctx.int(32, true);
    let bool_ty = ctx.bool();
    let unit_ty = ctx.unit();
    let int64 = ctx.int(64, true);

    // L1: a → Bool
    ctx.begin_transaction();
    ctx.set_binding(a, bool_ty);

    // L2: a → Unit
    ctx.begin_transaction();
    ctx.set_binding(a, unit_ty);

    // L3: a → Int64, then commit L3
    ctx.begin_transaction();
    ctx.set_binding(a, int64);
    assert_eq!(ctx.resolve_binding(a), int64);
    ctx.commit_transaction(); // L3 log merged into L2

    // L2 commit → merged log (L3+L2) merged into L1
    assert_eq!(ctx.resolve_binding(a), int64);
    ctx.commit_transaction();

    // After L2 commit, a is still Int64
    assert_eq!(ctx.resolve_binding(a), int64);

    // L1 rollback → should undo everything
    ctx.rollback_transaction();
    assert_eq!(ctx.resolve_binding(a), a);
}

// -- replace_generic --
#[test]
fn test_replace_generic_fn_ret() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "T".into());
    let p1 = ctx.generic_param(1, "U".into());
    let int_ty = ctx.int(32, true);
    let fn_ty = ctx.function(smallvec![p0], p1);
    let replaced = ctx.replace_generic(fn_ty, 0, int_ty);
    let expected = ctx.function(smallvec![int_ty], p1);
    assert_eq!(replaced, expected);
}

#[test]
fn test_replace_generic_noop() {
    let mut ctx = TypeContext::new();
    let bool_ty = ctx.bool();
    let int_ty = ctx.int(32, true);
    let replaced = ctx.replace_generic(bool_ty, 0, int_ty);
    assert_eq!(replaced, bool_ty);
}

// -- Yoneda reduction --
#[test]
fn test_yoneda_single_param_case1() {
    // ∀X.(Int⇒X)⇒X → Int
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let int_ty = ctx.int(32, true);
    let inner_fn = ctx.function(smallvec![int_ty], p0);
    let outer_fn = ctx.function(smallvec![inner_fn], p0);
    let forall_id = ctx.forall(0, "X".into(), outer_fn);
    let forall = ctx.try_yoneda_reduce(forall_id);
    assert_eq!(forall, int_ty, "∀X.(Int⇒X)⇒X should reduce to Int");
}

#[test]
fn test_yoneda_single_param_case2() {
    // ∀X.(X⇒Int)⇒(X⇒Bool) → Int⇒Bool  (co-Yoneda)
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let int_ty = ctx.int(32, true);
    let bool_ty = ctx.bool();
    let inner_fn = ctx.function(smallvec![p0], int_ty);
    let outer_fn = ctx.function(smallvec![p0], bool_ty);
    let combined = ctx.function(smallvec![inner_fn], outer_fn);
    let forall_id = ctx.forall(0, "X".into(), combined);
    let forall = ctx.try_yoneda_reduce(forall_id);
    assert_eq!(
        forall,
        ctx.function(smallvec![int_ty], bool_ty),
        "∀X.(X⇒Int)⇒(X⇒Bool) should reduce to Int⇒Bool"
    );
}

#[test]
fn test_yoneda_no_reduction() {
    // ∀X.Int⇒Int should not reduce
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let fn_ty = ctx.function(smallvec![int_ty], int_ty);
    let forall = ctx.forall(0, "X".into(), fn_ty);
    assert!(matches!(ctx.get(forall), TypeData::Forall { .. }));
}

#[test]
fn test_yoneda_partial_match_no_reduction() {
    // Regression: a PARTIAL Yoneda match must NOT reduce.
    // `∀X. (A → X) → Int → X` — `A → X` matches the schema but `Int`
    // does not; silently dropping `Int` would reduce the type to `A`
    // (mathematically it is `Int → A`).  The reduction must abort and
    // return the type unchanged (fail-closed).
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let int_ty = ctx.int(32, true);
    let branch = ctx.function(smallvec![int_ty], p0); // A → X (A = Int)
    let outer_fn = ctx.function(smallvec![branch, int_ty], p0); // (A → X) → Int → X
    let forall_id = ctx.forall(0, "X".into(), outer_fn);
    let reduced = ctx.try_yoneda_reduce(forall_id);
    assert_eq!(
        reduced, forall_id,
        "a partial Yoneda match must NOT reduce (Int must not be dropped)"
    );
    assert!(
        matches!(ctx.get(reduced), TypeData::Forall { .. }),
        "the type stays a forall after a rejected reduction"
    );
}

#[test]
fn test_yoneda_multi_param_inner_fn() {
    // ∀X.(Int⇒Bool⇒X)⇒X → (Int, Bool)  (single branch, product of params)
    // The inner Fn has params [Int, Bool] and ret=X.
    // With a single branch, Σₖ = Πⱼ Aⱼ = (Int, Bool).
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let int_ty = ctx.int(32, true);
    let bool_ty = ctx.bool();
    let inner_fn = ctx.function(smallvec![int_ty, bool_ty], p0);
    let outer_fn = ctx.function(smallvec![inner_fn], p0);
    let forall = ctx.forall(0, "X".into(), outer_fn);
    let reduced = ctx.try_yoneda_reduce(forall);
    let expected = ctx.tuple(smallvec![int_ty, bool_ty]);
    assert_eq!(
        reduced, expected,
        "∀X.(Int⇒Bool⇒X)⇒X should reduce to (Int,Bool)"
    );
}

/// PIN: the implicit Fn-encoded pattern (the removed "Case B") must
/// NOT reduce — `(Int⇒Bool⇒X)⇒X` with NO enclosing ∀X binder stays
/// unchanged. (Replaces the old `test_yoneda_distributed_case_b`,
/// which asserted the reduction.)
///
/// Without the ∀X, X is a FREE type variable, and reducing would
/// assert an isomorphism
///     ∀X. ((Int⇒Bool⇒X) ⇒ X)  ≅  (Int, Bool)
/// holding for EVERY instantiation of X with a single X-independent
/// pair. Hand counterexample at X := 1 (unit):
///     LHS = (Int⇒Bool⇒1)⇒1 ≅ 1⇒1 ≅ 1
///     RHS = (Int, Bool)     = 2^33
/// — not isomorphic. The continuation type (A⇒X)⇒X is the DOUBLE
/// DUAL of A; the Yoneda lemma collapses it to A only under the ∀X
/// (Nat(Hom(A,−), K) ≅ K(A) needs the whole natural transformation —
/// Mac Lane §III.2; cf. Pistone & Tranchini §2, where every schema is
/// ∀X-headed). The legal form of this reduction is the ∀-bound one,
/// covered by test_yoneda_multi_param_inner_fn
/// (∀X.(Int⇒Bool⇒X)⇒X → (Int,Bool)).
#[test]
fn test_yoneda_fn_encoded_no_reduction() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let int_ty = ctx.int(32, true);
    let bool_ty = ctx.bool();

    // (Int ⇒ Bool ⇒ X) ⇒ X   — X free, no Forall binder anywhere.
    let inner_fn = ctx.function(smallvec![int_ty, bool_ty], p0);
    let ty = ctx.function(smallvec![inner_fn], p0);

    assert_eq!(
        ctx.try_yoneda_reduce(ty),
        ty,
        "no ∀X binder ⇒ X is free ⇒ fail-closed: no reduction"
    );
}

/// co-Yoneda, multi-param branch tail: `∀X.(X⇒Int⇒Float)⇒(X⇒Bool)` →
/// `(Int⇒Float)⇒Bool`.
///
/// Isomorphism (⋆)II / (2) of Pistone & Tranchini (2022) §1–2:
///   ∀X.(X⇒A₀)⇒B⟨X⟩  ≡  B⟨X ↦ A₀⟩
/// The branch is curried as `X ⇒ (Int⇒Float)` (Fig. 4(a):
/// A⇒(B⇒C) ≡ B⇒(A⇒C) — the tail after X IS A₀), so A₀ = Int⇒Float,
/// and B⟨X⟩ = X⇒Bool with X in a NEGATIVE position — exactly the
/// contravariant reading ≡^X requires (Notation 2.1: the
/// co-representable functor X ↦ (X⇒A₀) is contravariant, so B must
/// match). Instantiating: B[X ↦ A₀] = (Int⇒Float)⇒Bool.
///
/// Hand computation of the isomorphism (semantic sanity check):
///   → : t ↦ t[Int⇒Float](id)
///       t[A₀] : (A₀⇒A₀)⇒(A₀⇒Bool) — applying the identity yields
///       A₀⇒Bool directly.
///   ← : f ↦ ΛX. λ(g:X⇒Int⇒Float). λ(x:X). f (g x)
///       (g x : Int⇒Float = A₀, f applied : Bool).
///
/// WHY THE OLD SHAPE WAS WRONG: the previous test used
/// `∀X.(X⇒Int⇒Float)⇒X`, i.e. B⟨X⟩ = X — an all-POSITIVE B under the
/// co-Yoneda (contravariant) schema. The source is EMPTY: at X := 0,
///   t[0] : (0⇒Int⇒Float)⇒0 ≅ 1⇒0 ≅ 0,
/// while the claimed reduct Int⇒Float is inhabited (λi. 0.0). The old
/// assertion therefore pinned a FALSE isomorphism (∅ ≅ |Int⇒Float|);
/// the variance gate now rejects it. This rewrite moves the test
/// inside the schema's domain (B negative) while preserving the
/// original regression intent: the multi-parameter tail (Int⇒Float,
/// NOT just Int) must survive the reduction.
#[test]
fn test_coyoneda_multi_param_preserves_return() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let int_ty = ctx.int(32, true);
    let float_ty = ctx.float(64);
    let bool_ty = ctx.bool();

    // Branch: X ⇒ Int ⇒ Float   (the single k-branch of the ≡^X schema)
    let inner_fn = ctx.function(smallvec![p0, int_ty], float_ty);
    // B⟨X⟩ = X ⇒ Bool   (X negative: function-parameter position)
    let b_ty = ctx.function(smallvec![p0], bool_ty);
    // ∀X.(X⇒Int⇒Float)⇒(X⇒Bool)
    let outer = ctx.function(smallvec![inner_fn], b_ty);
    let forall = ctx.forall(0, "X".into(), outer);

    // Expected: B⟨X ↦ Int⇒Float⟩ = (Int⇒Float)⇒Bool
    let int_to_float = ctx.function(smallvec![int_ty], float_ty);
    let expected = ctx.function(smallvec![int_to_float], bool_ty);

    assert_eq!(
        ctx.try_yoneda_reduce(forall),
        expected,
        "∀X.(X⇒Int⇒Float)⇒(X⇒Bool) should reduce to (Int⇒Float)⇒Bool, \
             preserving the full multi-param tail"
    );
}

// ── Yoneda / co-Yoneda with inner quantifiers (∀Z⃗ₖ) ────────
//
// These test the fix for a binding-maintenance bug where inner-Forall
// GenericParam references became dangling after Yoneda reduction
// peeled the quantifier layers.

#[test]
fn test_yoneda_inner_quantifier_one() {
    // ∀X. (∀Z. Z ⇒ X) ⇒ X  →  ∃Z. Z
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let gp_z = ctx.generic_param(1, "Z".into());

    // Expected: ∃Z. Z
    let expected = ctx.alloc(TypeData::Exists {
        param_index: 1,
        name: "Z".into(),
        base: gp_z,
    });

    let inner_fn = ctx.function(smallvec![gp_z], p0);
    let inner_forall = ctx.alloc(TypeData::Forall {
        param_index: 1,
        param_name: "Z".into(),
        body: inner_fn,
    });
    let outer_fn = ctx.function(smallvec![inner_forall], p0);
    let forall_id = ctx.forall(0, "X".into(), outer_fn);
    let result = ctx.try_yoneda_reduce(forall_id);
    assert_eq!(result, expected, "∀X.(∀Z.Z⇒X)⇒X should reduce to ∃Z.Z");
}

#[test]
fn test_yoneda_inner_quantifier_x_in_body() {
    // ∀X. (∀Z. (Z, X) ⇒ X) ⇒ X  →  μX. ∃Z. (Z, X)
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let gp_z = ctx.generic_param(1, "Z".into());
    let int_ty = ctx.int(32, true);

    // Expected: μX. ∃Z. (Z, X)
    let tup = ctx.tuple(smallvec![gp_z, p0]);
    let inner_exists = ctx.alloc(TypeData::Exists {
        param_index: 1,
        name: "Z".into(),
        base: tup,
    });
    let expected = ctx.alloc(TypeData::Mu {
        param_index: 0,
        param_name: "X".into(),
        body: inner_exists,
    });

    let tup = ctx.tuple(smallvec![gp_z, p0]);
    let inner_fn = ctx.function(smallvec![tup], p0);
    let inner_forall = ctx.alloc(TypeData::Forall {
        param_index: 1,
        param_name: "Z".into(),
        body: inner_fn,
    });
    let outer_fn = ctx.function(smallvec![inner_forall], p0);
    let forall_id = ctx.forall(0, "X".into(), outer_fn);
    let result = ctx.try_yoneda_reduce(forall_id);
    assert_eq!(
        result, expected,
        "∀X.(∀Z.(Z,X)⇒X)⇒X should reduce to μX.∃Z.(Z,X)"
    );
}

#[test]
fn test_yoneda_two_inner_quantifiers() {
    // ∀X. (∀Z₁. ∀Z₂. (Z₁, Z₂, Int) ⇒ X) ⇒ X  →  ∃Z₂. ∃Z₁. (Z₁, Z₂, Int)
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let gp_z1 = ctx.generic_param(1, "Z₁".into());
    let gp_z2 = ctx.generic_param(2, "Z₂".into());
    let int_ty = ctx.int(32, true);

    // Expected: ∃Z₂. ∃Z₁. (Z₁, Z₂, Int)
    let tup = ctx.tuple(smallvec![gp_z1, gp_z2, int_ty]);
    let inner_ex = ctx.alloc(TypeData::Exists {
        param_index: 1,
        name: "Z₁".into(),
        base: tup,
    });
    let expected = ctx.alloc(TypeData::Exists {
        param_index: 2,
        name: "Z₂".into(),
        base: inner_ex,
    });

    let tup = ctx.tuple(smallvec![gp_z1, gp_z2, int_ty]);
    let inner_fn = ctx.function(smallvec![tup], p0);
    let inner_forall2 = ctx.alloc(TypeData::Forall {
        param_index: 2,
        param_name: "Z₂".into(),
        body: inner_fn,
    });
    let inner_forall1 = ctx.alloc(TypeData::Forall {
        param_index: 1,
        param_name: "Z₁".into(),
        body: inner_forall2,
    });
    let outer_fn = ctx.function(smallvec![inner_forall1], p0);
    let forall_id = ctx.forall(0, "X".into(), outer_fn);
    let result = ctx.try_yoneda_reduce(forall_id);
    assert_eq!(
        result, expected,
        "∀X.(∀Z₁.∀Z₂.(Z₁,Z₂,Int)⇒X)⇒X should reduce to ∃Z₂.∃Z₁.(Z₁,Z₂,Int)"
    );
}

/// co-Yoneda with an inner quantifier, B in negative position:
/// `∀X.(∀Z.X⇒Z)⇒(X⇒Int)` → `(∀Z.Z)⇒Int`.
///
/// Fig. 3, second schema (≡^X), k = 1 branch:
///   ∀X.⟨∀Z⃗. X ⇒ A⟨X⟩⟩ ⇒ C⟨X⟩  ≡  C⟨X ↦ {νX.}∀Z⃗. A⟨X⟩⟩
/// The branch is `∀Z.X⇒Z`: A = Z with X NOT occurring in A, so the
/// {νX.} wrapper is elided — the paper derives isomorphism (2) from
/// (4) exactly this way ("μX.A ≡ νX.A ≡ A when X does not occur in
/// A"), leaving the substitution X ↦ ∀Z.Z. C⟨X⟩ = X⇒Int (negative
/// B ✓), so the reduct is C[X ↦ ∀Z.Z] = (∀Z.Z)⇒Int.
///
/// Hand computation of the extraction (→ direction):
///   t : ∀X.(∀Z.X⇒Z)⇒(X⇒Int)  ↦  λ(h:∀Z.Z). t[∀Z.Z](ΛZ.λ_. h[Z])
/// Instantiate X := ∀Z.Z:
///   t[∀Z.Z] : (∀Z.(∀Z.Z)⇒Z) ⇒ ((∀Z.Z)⇒Int).
/// From h : ∀Z.Z build g' = ΛZ.λ_. h[Z] : ∀Z.(∀Z.Z)⇒Z (ignore the
/// argument, return h[Z] : Z), so t[∀Z.Z](g') : (∀Z.Z)⇒Int. Note
/// ∀Z.Z is EMPTY (instantiate Z := 0), so the reduct has exactly ONE
/// inhabitant — the vacuous function from an empty domain (κ = 1).
///
/// The old shape `∀X.(∀Z.X⇒Z)⇒X` only "worked" because BOTH sides
/// were empty (0 ≅ 0 — source: at X := 0, t[0] : (∀Z.1)⇒0 ≅ 0), an
/// accidental isomorphism outside the schema's domain; the variance
/// gate now rejects it.
///
/// The branch's ∀Z⃗ is PRESERVED as ∀ — the dual of the Yoneda side's
/// ∃Z⃗ (Fig. 3 keeps ∀Z⃗ₖ under the ν/∀ combination). NB: the expected
/// TypeId equality relies on the fresh binder index landing on 1 —
/// deterministic here because fresh_param_index() first yields 0
/// (which collides with pi = 0 and is skipped), then 1, matching the
/// peeled index before any other fresh allocation in this reduction.
#[test]
fn test_coyoneda_inner_quantifier_one() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let gp_z = ctx.generic_param(1, "Z".into());
    let int_ty = ctx.int(32, true);

    // Expected: (∀Z.Z) ⇒ Int
    let forall_z = ctx.forall(1, "Z".into(), gp_z);
    let expected = ctx.function(smallvec![forall_z], int_ty);

    // Branch: ∀Z. X ⇒ Z
    let inner_fn = ctx.function(smallvec![p0], gp_z);
    let inner_forall = ctx.forall(1, "Z".into(), inner_fn);
    // B⟨X⟩ = X ⇒ Int   (X negative)
    let b_ty = ctx.function(smallvec![p0], int_ty);
    // ∀X.(∀Z.X⇒Z)⇒(X⇒Int)
    let outer_fn = ctx.function(smallvec![inner_forall], b_ty);
    let forall_id = ctx.forall(0, "X".into(), outer_fn);

    assert_eq!(
        ctx.try_yoneda_reduce(forall_id),
        expected,
        "∀X.(∀Z.X⇒Z)⇒(X⇒Int) should reduce to (∀Z.Z)⇒Int, keeping the \
             inner ∀ as ∀ (the co-Yoneda dual of the Yoneda ∃)"
    );
}

/// co-Yoneda with X occurring in the branch body — schema (4), the ν
/// case: `∀X.(∀Z.X⇒(Z,X))⇒(X⇒Int)` → `(νX.∀Z.(Z,X))⇒Int`.
///
/// Isomorphism (4) of Pistone & Tranchini (2022) §2:
///   ∀X.(X⇒A⟨X⟩)⇒C⟨X⟩  ≡  C⟨X ↦ νX.A⟨X⟩⟩
/// with the branch's inner ∀Z⃗ folded into the ν body (Fig. 3, ≡^X).
/// Preconditions, both satisfied here:
///   - C⟨X⟩ = X⇒Int: every X occurrence NEGATIVE (the contravariant
///     reading ≡^X requires — Notation 2.1);
///   - A⟨X⟩ = (Z,X): every X occurrence POSITIVE (Forall body and
///     Tuple are both covariant positions) — the schemas only form
///     μ/ν over strictly positive bodies.
/// Since X DOES occur in A, the {νX.} wrapper is NOT elided (contrast
/// test_coyoneda_inner_quantifier_one), and the reduct is
/// C[X ↦ νX.∀Z.(Z,X)] = (νX.∀Z.(Z,X))⇒Int — νX.∀Z.(Z,X) being the
/// final coalgebra of F(Y) = ∀Z.(Z,Y).
///
/// Hand computation (why the old shape was outside the schema): the
/// old test used B⟨X⟩ = X — positive. Its source is again EMPTY: at
/// X := 0, g : ∀Z.0⇒(Z,0) ≅ ∀Z.1 is inhabited, so
///   t[0] : (∀Z.1)⇒0 ≅ 0,
/// while the claimed reduct νX.∀Z.(Z,X) IS inhabited — the canonical
/// final-coalgebra element h = ΛZ.λz.(z, h) (unfolding
/// νX.F(X) ≅ F(νX.F(X))) is well-typed. The old assertion therefore
/// pinned ∅ ≅ |νX.∀Z.(Z,X)| > 0 — a false isomorphism the variance
/// gate now rejects.
#[test]
fn test_coyoneda_inner_quantifier_x_in_body() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let gp_z = ctx.generic_param(1, "Z".into());
    let int_ty = ctx.int(32, true);

    // Expected: (νX.∀Z.(Z,X)) ⇒ Int
    let tup = ctx.tuple(smallvec![gp_z, p0]);
    let inner_forall = ctx.forall(1, "Z".into(), tup);
    let nu = ctx.alloc(TypeData::Nu {
        param_index: 0,
        param_name: "X".into(),
        body: inner_forall,
    });
    let expected = ctx.function(smallvec![nu], int_ty);

    // Branch: ∀Z. X ⇒ (Z, X)
    let i_tup = ctx.tuple(smallvec![gp_z, p0]);
    let inner_fn = ctx.function(smallvec![p0], i_tup);
    let inner_forall_wrap = ctx.forall(1, "Z".into(), inner_fn);
    // B⟨X⟩ = X ⇒ Int   (X negative)
    let b_ty = ctx.function(smallvec![p0], int_ty);
    // ∀X.(∀Z.X⇒(Z,X))⇒(X⇒Int)
    let outer_fn = ctx.function(smallvec![inner_forall_wrap], b_ty);
    let forall_id = ctx.forall(0, "X".into(), outer_fn);

    assert_eq!(
        ctx.try_yoneda_reduce(forall_id),
        expected,
        "∀X.(∀Z.X⇒(Z,X))⇒(X⇒Int) should reduce to (νX.∀Z.(Z,X))⇒Int — \
             X occurs in the branch body, so the νX wrapper is kept"
    );
}

#[test]
fn test_yoneda_two_branches_with_inner_quantifiers() {
    // ∀X. (∀Z₁. Z₁ ⇒ X) ⇒ (∀Z₂. Z₂ ⇒ X) ⇒ X  →  ∃Z₁.Z₁ + ∃Z₂.Z₂
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let gp_z1 = ctx.generic_param(1, "Z₁".into());
    let gp_z2 = ctx.generic_param(2, "Z₂".into());

    // Expected: Coproduct(∃Z₁.Z₁, ∃Z₂.Z₂)
    let ex_z1 = ctx.alloc(TypeData::Exists {
        param_index: 1,
        name: "Z₁".into(),
        base: gp_z1,
    });
    let ex_z2 = ctx.alloc(TypeData::Exists {
        param_index: 2,
        name: "Z₂".into(),
        base: gp_z2,
    });
    let expected = ctx.coproduct(smallvec![ex_z1, ex_z2]);

    // Branch 1: ∀Z₁. Z₁ ⇒ X
    let inner_fn1 = ctx.function(smallvec![gp_z1], p0);
    let forall1 = ctx.alloc(TypeData::Forall {
        param_index: 1,
        param_name: "Z₁".into(),
        body: inner_fn1,
    });
    // Branch 2: ∀Z₂. Z₂ ⇒ X
    let inner_fn2 = ctx.function(smallvec![gp_z2], p0);
    let forall2 = ctx.alloc(TypeData::Forall {
        param_index: 2,
        param_name: "Z₂".into(),
        body: inner_fn2,
    });
    let outer_fn = ctx.function(smallvec![forall1, forall2], p0);
    let forall_id = ctx.forall(0, "X".into(), outer_fn);
    let result = ctx.try_yoneda_reduce(forall_id);
    assert_eq!(
        result, expected,
        "∀X.(∀Z₁.Z₁⇒X)⇒(∀Z₂.Z₂⇒X)⇒X should reduce to ∃Z₁.Z₁ + ∃Z₂.Z₂"
    );
}

#[test]
fn test_yoneda_inner_quantifier_no_x_ref() {
    // ∀X. (∀Z. (Int ⇒ Z) ⇒ X) ⇒ X  →  ∃Z. (Int ⇒ Z)
    // Here A = Int ⇒ Z, and there is no X reference inside A.
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let gp_z = ctx.generic_param(1, "Z".into());
    let int_ty = ctx.int(32, true);

    // Expected: ∃Z. (Int ⇒ Z)
    let arrow = ctx.function(smallvec![int_ty], gp_z);
    let expected = ctx.alloc(TypeData::Exists {
        param_index: 1,
        name: "Z".into(),
        base: arrow,
    });

    let inner_fn = ctx.function(smallvec![arrow], p0);
    let inner_forall = ctx.alloc(TypeData::Forall {
        param_index: 1,
        param_name: "Z".into(),
        body: inner_fn,
    });
    let outer_fn = ctx.function(smallvec![inner_forall], p0);
    let forall_id = ctx.forall(0, "X".into(), outer_fn);
    let result = ctx.try_yoneda_reduce(forall_id);
    assert_eq!(
        result, expected,
        "∀X.(∀Z.(Int⇒Z)⇒X)⇒X should reduce to ∃Z.(Int⇒Z)"
    );
}

#[test]
fn test_yoneda_x_to_x_does_not_duplicate_branch() {
    // ∀X.(X→X)→X  — branch X→X matches BOTH Yoneda (ret=X) and co-Yoneda
    // (first param=X).  Must NOT push two copies into branch_replacements.
    //
    // Paper (Pistone & Tranchini 2022 §2): the ≡_X schema matches when the
    // branch's return is X (the bound variable).  If the first parameter is
    // also X, the branch is interpreted as the Yoneda case A⟨X⟩ = X, giving
    // Σₖ A⟨X⟩ = X and therefore μX.X.
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let branch = ctx.function(smallvec![p0], p0); // X → X
    let outer = ctx.function(smallvec![branch], p0); // (X→X) → X
    let forall_id = ctx.forall(0, "X".into(), outer); // ∀X.(X→X)→X
    let ty = ctx.try_yoneda_reduce(forall_id);

    // Should be µX.X — a single branch, not a coproduct with two entries.
    match ctx.get(ty) {
        TypeData::Mu {
            param_index, body, ..
        } => {
            assert_eq!(*param_index, 0, "mu binds the outer X index");
            match ctx.get(*body) {
                TypeData::GenericParam { index, .. } => {
                    assert_eq!(
                        *index, 0,
                        "mu body should be X (GenericParam(0)), not a coproduct"
                    );
                }
                other => panic!("expected GenericParam(0) inside Mu, got {other:?}"),
            }
        }
        other => panic!("expected Mu, got {other:?}"),
    }
}

// ── Forall subtype with α-conversion ──────────────────────────

#[test]
fn test_subtype_forall_alpha_equiv_gp() {
    // ∀X.{0} X <: ∀Y.{7} Y  → true (alpha-equivalent after renaming Y→X)
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let p7 = ctx.generic_param(7, "Y".into());
    let fx = ctx.forall(0, "X".into(), p0);
    let fy = ctx.forall(7, "Y".into(), p7);
    assert!(
        ctx.subtype(fx, fy),
        "∀X.X <: ∀Y.Y should hold under alpha-conversion"
    );
    assert!(
        ctx.subtype(fy, fx),
        "∀Y.Y <: ∀X.X should hold symmetrically"
    );
}

#[test]
fn test_subtype_forall_alpha_equiv_fn() {
    // ∀X.{0} (X → Int) <: ∀Y.{7} (Y → Int)  → true
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let p0 = ctx.generic_param(0, "X".into());
    let p7 = ctx.generic_param(7, "Y".into());
    let fn_x = ctx.function(smallvec![p0], int32);
    let fn_y = ctx.function(smallvec![p7], int32);
    let fx = ctx.forall(0, "X".into(), fn_x);
    let fy = ctx.forall(7, "Y".into(), fn_y);
    assert!(
        ctx.subtype(fx, fy),
        "∀X.(X→Int) <: ∀Y.(Y→Int) should hold under alpha-conversion"
    );
}

#[test]
fn test_subtype_forall_alpha_equiv_fails_on_body_diff() {
    // ∀X.{0} (X → Int) <: ∀Y.{7} (Int → Y)  → false (different structure)
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let p0 = ctx.generic_param(0, "X".into());
    let p7 = ctx.generic_param(7, "Y".into());
    let fn_x = ctx.function(smallvec![p0], int32); // X → Int
    let fn_y = ctx.function(smallvec![int32], p7); // Int → Y
    let fx = ctx.forall(0, "X".into(), fn_x);
    let fy = ctx.forall(7, "Y".into(), fn_y);
    assert!(
        !ctx.subtype(fx, fy),
        "∀X.(X→Int) <: ∀Y.(Int→Y) should be false"
    );
}

#[test]
fn test_subtype_forall_alpha_same_index_still_works() {
    // ∀X.{0} X <: ∀X.{0} X  → true (same index, no renaming needed)
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let fx = ctx.forall(0, "X".into(), p0);
    assert!(
        ctx.subtype(fx, fx),
        "∀X.X <: ∀X.X with same index should hold"
    );
}

#[test]
fn test_subtype_forall_no_capture_bug() {
    // Regression test: α-conversion must NOT capture a free GenericParam
    // that happens to share the same index as sub's binder.
    //
    // Context: free variable X (index 0) from outer scope.
    //   sub = ∀X.(X → X)   — binds index 0
    //   sup = ∀Y.(X → X)   — binds index 1, body has free GenericParam(0)
    //
    // Without capture-avoidance, renaming Y→X in sup's body would capture
    // the free X, making both bodies (X→X) == (X→X) and incorrectly
    // returning true.
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let p1 = ctx.generic_param(1, "Y".into());

    // Build sub: ∀X.{0} (X → X) — binder index 0
    let sub_fn = ctx.function(smallvec![p0], p0);
    let sub = ctx.forall(0, "X".into(), sub_fn);

    // Build sup: ∀Y.{1} (X → X) — binder index 1, body has free GP(0)
    let sup_fn = ctx.function(smallvec![p0], p0);
    let sup = ctx.forall(1, "Y".into(), sup_fn);

    // ∀X.(X→X) <: ∀Y.(X→X) must be FALSE:
    // the body of sup contains a FREE X (GP{0}) which is NOT the
    // bound Y (GP{1}).  After α-conversion with capture avoidance,
    // sub's X(0) → fresh(2), sup's Y(1) → fresh(2),
    // sup's free X(0) STAYS AS 0, giving bodies (GP(2)→GP(2)) vs
    // (GP(0)→GP(0)) — structurally different → false.
    assert!(
        !ctx.subtype(sub, sup),
        "∀X.(X→X) <: ∀Y.(X→X) must NOT hold — free X in sup would be captured"
    );
}

// ── HRTB / Forall subtype tests ──────────────────────────────

#[test]
fn test_subtype_forall_identical() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let fn_ty = ctx.function(smallvec![p0], p0);
    let forall = ctx.forall(0, "X".into(), fn_ty);
    assert!(ctx.subtype(forall, forall));
}

#[test]
fn test_subtype_forall_body_subtype() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let never = ctx.never();
    let int32 = ctx.int(32, true);
    let sub_fn = ctx.function(smallvec![p0], never);
    let sup_fn = ctx.function(smallvec![p0], int32);
    let sub_forall = ctx.forall(0, "X".into(), sub_fn);
    let sup_forall = ctx.forall(0, "X".into(), sup_fn);
    assert!(ctx.subtype(sub_forall, sup_forall));
}

#[test]
fn test_subtype_forall_peel_sup() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let forall_ty = ctx.forall(0, "X".into(), int_ty);
    assert!(ctx.subtype(int_ty, forall_ty));
}

#[test]
fn test_normalize_associated_type_concrete_self() {
    let mut ctx = TypeContext::new();
    let def_id = DefId(42);
    let int_ty = ctx.int(32, true);
    let adt_ty = ctx.alloc(TypeData::Adt {
        kind: AdtKind::Struct,
        def_id,
        args: smallvec![int_ty],
    });
    assert_eq!(
        ctx.try_normalize_associated_type_def_id(adt_ty),
        Some(def_id)
    );
}

#[test]
fn test_normalize_associated_type_abstract_self() {
    let mut ctx = TypeContext::new();
    let var_id = ctx.alloc(TypeData::InferVar { id: 0, universe: 0 });
    assert_eq!(ctx.try_normalize_associated_type_def_id(var_id), None);
}

// ── Transaction + path compression ──────────────────────────

#[test]
fn test_transaction_rollback_preserves_binding_chain() {
    // Verify that a binding change made inside a transaction is correctly
    // undone on rollback (Fix 1), and that the surrounding chain is intact
    // afterwards.
    //
    // `resolve_binding` no longer path-compresses (see the note on
    // `TypeContext::resolve_binding`): `bindings` is a plain `HashMap`
    // because interior mutability had to go for `TypeContext` to become
    // `Sync`, and a `&self` walk cannot rewrite the map.  So this test
    // now pins the *observable* contract — resolution and rollback —
    // rather than the compression side effect.
    let mut ctx = TypeContext::new();
    let a = ctx.alloc(TypeData::InferVar { id: 1, universe: 0 });
    let b = ctx.alloc(TypeData::InferVar { id: 2, universe: 0 });
    let c = ctx.alloc(TypeData::InferVar { id: 3, universe: 0 });
    let d = ctx.alloc(TypeData::InferVar { id: 4, universe: 0 });

    // Build a binding chain: a → b → c
    ctx.set_binding(a, b);
    ctx.set_binding(b, c);
    assert_eq!(ctx.bindings.get(&a).copied(), Some(b));
    assert_eq!(ctx.bindings.get(&b).copied(), Some(c));
    assert_eq!(ctx.resolve_binding(a), c);

    // Inside a transaction, re-point `a` at `d`.  The undo log must
    // record the old value (`b`).
    ctx.begin_transaction();
    ctx.set_binding(a, d);
    assert_eq!(ctx.resolve_binding(a), d);

    // Rollback — `a` must point at `b` again, and the chain must still
    // resolve to `c`.
    ctx.rollback_transaction();
    assert_eq!(ctx.bindings.get(&a).copied(), Some(b));
    assert_eq!(ctx.resolve_binding(a), c);
}

#[test]
fn test_resolve_binding_mut_compresses_chain() {
    let mut ctx = TypeContext::new();
    let a = ctx.alloc(TypeData::InferVar { id: 1, universe: 0 });
    let b = ctx.alloc(TypeData::InferVar { id: 2, universe: 0 });
    let c = ctx.alloc(TypeData::InferVar { id: 3, universe: 0 });

    ctx.set_binding(a, b);
    ctx.set_binding(b, c);
    assert_eq!(ctx.bindings.get(&a).copied(), Some(b));
    assert_eq!(ctx.resolve_binding(a), c);

    assert_eq!(ctx.resolve_binding_mut(a), Some(c));
    assert_eq!(ctx.bindings.get(&a).copied(), Some(c), "head rewired");
    assert_eq!(ctx.bindings.get(&b).copied(), Some(c), "mid rewired");

    // Resolution is a pure memo of the same walk: identical before/after.
    assert_eq!(ctx.resolve_binding(a), c);
    assert_eq!(ctx.resolve_binding(b), c);
}

#[test]
fn test_resolve_binding_compressed_agrees_with_resolve_binding() {
    let mut ctx = TypeContext::new();
    let a = ctx.alloc(TypeData::InferVar { id: 1, universe: 0 });
    let b = ctx.alloc(TypeData::InferVar { id: 2, universe: 0 });
    let c = ctx.alloc(TypeData::InferVar { id: 3, universe: 0 });
    let x = ctx.alloc(TypeData::InferVar { id: 4, universe: 0 });
    let y = ctx.alloc(TypeData::InferVar { id: 5, universe: 0 });
    let z = ctx.alloc(TypeData::InferVar { id: 6, universe: 0 });

    ctx.set_binding(a, b);
    ctx.set_binding(b, c);
    ctx.set_binding(x, y);
    ctx.set_binding(y, z);

    let compressed = ctx.resolve_binding_compressed(a);
    let shared = ctx.resolve_binding(a);
    assert_eq!(compressed, c);
    assert_eq!(compressed, shared, "compression must not change resolution");

    let via_shared = ctx.resolve_binding(x);
    let via_compressed = ctx.resolve_binding_compressed(x);
    assert_eq!(via_shared, z);
    assert_eq!(via_shared, via_compressed, "agrees on an untouched chain");
}

#[test]
fn test_resolve_binding_mut_declines_inside_gadt_arm() {
    let mut ctx = TypeContext::new();
    let a = ctx.alloc(TypeData::InferVar { id: 1, universe: 0 });
    let b = ctx.alloc(TypeData::InferVar { id: 2, universe: 0 });
    let c = ctx.alloc(TypeData::InferVar { id: 3, universe: 0 });

    ctx.set_binding(a, b);
    ctx.set_binding(b, c);

    ctx.push_gadt_arm();
    assert_eq!(
        ctx.resolve_binding_mut(a),
        None,
        "compression must not run while a GADT arm can shadow chain nodes"
    );
    assert_eq!(ctx.bindings.get(&a).copied(), Some(b), "chain untouched");
    // The fallback path still resolves correctly while the arm is live.
    assert_eq!(ctx.resolve_binding_compressed(a), c);
    ctx.pop_gadt_arm();

    // Once the arm closes the registry is empty again, so compression
    // is permitted.
    assert_eq!(ctx.resolve_binding_mut(a), Some(c));
    assert_eq!(ctx.bindings.get(&a).copied(), Some(c));
    assert_eq!(ctx.resolve_binding(a), c);
}

#[test]
fn test_compression_write_is_rolled_back_by_transaction() {
    let mut ctx = TypeContext::new();
    let a = ctx.alloc(TypeData::InferVar { id: 1, universe: 0 });
    let b = ctx.alloc(TypeData::InferVar { id: 2, universe: 0 });
    let c = ctx.alloc(TypeData::InferVar { id: 3, universe: 0 });

    ctx.set_binding(a, b);
    ctx.set_binding(b, c);

    ctx.begin_transaction();
    assert_eq!(ctx.resolve_binding_mut(a), Some(c));
    assert_eq!(ctx.bindings.get(&a).copied(), Some(c));
    ctx.rollback_transaction();

    // The undo log must have captured the pre-compression pointer, so the
    // original chain shape is restored...
    assert_eq!(ctx.bindings.get(&a).copied(), Some(b));
    assert_eq!(ctx.resolve_binding(a), c);
}

#[test]
fn test_compression_does_not_relocate_generic_binding_origin() {
    let mut ctx = TypeContext::new();
    let param = ctx.generic_param(0, "T".into());
    let mid = ctx.alloc(TypeData::InferVar { id: 7, universe: 0 });
    let root = ctx.alloc(TypeData::InferVar { id: 8, universe: 0 });

    let original_site = crate::ast::Span::new(FileId::DUMMY, 10, 20);
    ctx.current_unify_span = Some(original_site);
    ctx.set_binding(param, mid);
    ctx.current_unify_span = None;
    ctx.set_binding(mid, root);
    assert_eq!(
        ctx.generic_binding_origins.get(&param).copied(),
        Some(original_site)
    );

    // A later span-carrying unification walks this chain.  Compression
    // must not record IT as the binding site.
    let later_site = crate::ast::Span::new(FileId::DUMMY, 900, 950);
    ctx.current_unify_span = Some(later_site);
    assert_eq!(ctx.resolve_binding_mut(param), Some(root));
    ctx.current_unify_span = None;

    assert_eq!(ctx.bindings.get(&param).copied(), Some(root), "compressed");
    assert_eq!(
        ctx.generic_binding_origins.get(&param).copied(),
        Some(original_site),
        "compression must not relocate the E104 binding-site origin"
    );
}

// ── characteristic resolves bindings ─────────────────────────

#[test]
fn test_characteristic_resolves_binding() {
    // Verify that characteristic resolves bindings before computing κ
    // (Fix 3).  If an InferVar is bound to Bool, characteristic should
    // return Bool's κ (2), not the κ of InferVar (usize::MAX fallback).
    let mut ctx = TypeContext::new();
    let bool_ty = ctx.bool();
    let infer = ctx.alloc(TypeData::InferVar {
        id: 42,
        universe: 0,
    });

    // Bind infer → Bool
    ctx.set_binding(infer, bool_ty);

    // characteristic should resolve the binding and compute κ(Bool) = 2
    assert_eq!(
        ctx.characteristic(infer),
        Characteristic::FiniteExhaustible(2),
        "κ(InferVar bound to Bool) should be 2, not a fallback"
    );
}

// ── checked_shl overflow safety ─────────────────────────────

#[test]
fn test_characteristic_int_overflow_safe() {
    // Verify that Int with bits >= usize::BITS saturates instead of
    // panicking (Fix 4).
    let mut ctx = TypeContext::new();
    // usize::BITS is 64 on 64-bit, 32 on 32-bit.  bits=64 is valid
    // for Int<64>.  This should not panic.
    let large = ctx.int(64, true);
    let k = ctx.characteristic(large);
    // Should saturate to usize::MAX or wrap around, not panic.
    assert!(k != Characteristic::Undecidable);
}

// ── def_id_to_type_id prototype preservation ────────────────

#[test]
fn test_def_id_preserves_prototype() {
    // Verify that struct_ty with different generic args does NOT
    // overwrite the prototype mapping (Fix 5).
    let mut ctx = TypeContext::new();
    let def_id = DefId(99);
    let int32 = ctx.int(32, true);
    let bool_ty = ctx.bool();

    // Create generic instances in various orders
    let vec_i32 = ctx.struct_ty(def_id, smallvec![int32]);
    let vec_bool = ctx.struct_ty(def_id, smallvec![bool_ty]);

    // These should be different TypeIds (different args)
    assert_ne!(vec_i32, vec_bool, "Vec<i32> and Vec<bool> should differ");

    // get_type_id_for_def_id should return the FIRST registered
    // (the prototype), NOT the last instance.
    let registered = ctx.get_type_id_for_def_id(def_id);
    assert_eq!(
        registered,
        Some(vec_i32),
        "get_type_id_for_def_id should return the first registered instance (the prototype)"
    );
}

// ── α-conversion with capture avoidance ──────────────────────

#[test]
fn test_alpha_conv_forall_different_indices() {
    // ∀X{0}.X <: ∀Y{7}.Y — structurally identical, different indices
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let p7 = ctx.generic_param(7, "Y".into());
    let fx = ctx.forall(0, "X".into(), p0);
    let fy = ctx.forall(7, "Y".into(), p7);

    assert!(ctx.subtype(fx, fy));
    assert!(ctx.subtype(fy, fx));
    assert!(ctx.unify(fx, fy).is_ok());
}

#[test]
fn test_alpha_conv_forall_no_capture() {
    // Forall(2, "X", body=GP(2)) vs Forall(0, "Y", body=GP(2)).
    // Without capture avoidance, renaming Y→X captures the free GP(2).
    let mut ctx = TypeContext::new();
    let gp2 = ctx.generic_param(2, "X".into());
    let fsub = ctx.forall(2, "X".into(), gp2);
    let fsup = ctx.forall(0, "Y".into(), gp2);
    assert!(
        !ctx.subtype(fsub, fsup),
        "∀X(2).X <: ∀Y(0).X(free) must NOT hold — capture would be incorrect"
    );
}

#[test]
fn test_alpha_conv_mu_different_indices() {
    let mut ctx = TypeContext::new();
    let int_ty = ctx.int(32, true);
    let mu0 = ctx.alloc(TypeData::Mu {
        param_index: 0,
        param_name: "X".into(),
        body: int_ty,
    });
    let mu5 = ctx.alloc(TypeData::Mu {
        param_index: 5,
        param_name: "Y".into(),
        body: int_ty,
    });
    assert!(ctx.unify(mu0, mu5).is_ok());
}

#[test]
fn test_alpha_conv_poly_unify_and_subtype() {
    let mut ctx = TypeContext::new();
    let p0 = ctx.generic_param(0, "X".into());
    let p3 = ctx.generic_param(3, "Z".into());
    let poly1 = ctx.poly(smallvec![(0, "X".into())], p0);
    let poly2 = ctx.poly(smallvec![(3, "Z".into())], p3);
    assert!(ctx.subtype(poly1, poly2));
    assert!(ctx.unify(poly1, poly2).is_ok());
}

#[test]
fn test_occurs_check_through_binding() {
    let mut ctx = TypeContext::new();
    let param = ctx.alloc(TypeData::InferVar { id: 0, universe: 0 });
    let mid = ctx.alloc(TypeData::InferVar { id: 1, universe: 0 });
    let ty = ctx.alloc(TypeData::InferVar { id: 2, universe: 0 });
    ctx.set_binding(ty, mid);
    ctx.set_binding(mid, param);
    assert!(
        ctx.occurs_check(param, ty),
        "occurs_check should find param through binding chain ty→mid→param"
    );
}

/// Witness solving is opt-in and explicit: `resolve_existential_witness`
/// is the single observation point — an INERT existential equation
/// registered via `register_existential_equation` is returned to
/// consumers that explicitly ask, and never leaks into `resolve_binding`.
#[test]
fn test_resolve_existential_witness_opt_in() {
    let mut ctx = TypeContext::new();
    ctx.push_gadt_arm();
    let skolem = ctx.generic_param(0, Symbol::intern("X"));
    let concrete = ctx.int(32, true);
    ctx.register_existential_equation(skolem, concrete);
    // Explicit opt-in observation:
    assert_eq!(ctx.resolve_existential_witness(skolem), Some(concrete));
    // Opacity is the default — resolve_binding must NOT follow it:
    assert_eq!(ctx.resolve_binding(skolem), skolem);
    // Unknown skolem: no witness.
    let other = ctx.generic_param(1, Symbol::intern("Y"));
    assert_eq!(ctx.resolve_existential_witness(other), None);
}

// ── Rigid / Wobbly distinction tests (PLDI'06 §2,§4,§6) ──────────

/// Rigidity enum: Rigid and Wobbly are distinct and hash correctly.
#[test]
fn test_rigidity_distinct() {
    use crate::hir::types::Rigidity;
    assert_ne!(Rigidity::Rigid, Rigidity::Wobbly);
    assert_eq!(Rigidity::Rigid, Rigidity::Rigid);
    assert_eq!(Rigidity::Wobbly, Rigidity::Wobbly);
}

/// `unify_biased`: rigid-vs-rigid unification behaves like standard unify.
#[test]
fn test_unify_biased_rigid_rigid() {
    let mut ctx = TypeContext::new();
    let a = ctx.int(32, true);
    let b = ctx.int(32, true);
    let result = ctx.unify_biased(a, b, Rigidity::Rigid, Rigidity::Rigid);
    assert!(
        result.is_ok(),
        "rigid-vs-rigid should unify like standard unify"
    );
}

/// `unify_biased`: rigid-vs-rigid mismatch is an error.
#[test]
fn test_unify_biased_rigid_rigid_mismatch() {
    let mut ctx = TypeContext::new();
    let a = ctx.int(32, true);
    let b = ctx.int(64, true);
    let result = ctx.unify_biased(a, b, Rigidity::Rigid, Rigidity::Rigid);
    assert!(
        result.is_err(),
        "rigid-vs-rigid mismatch should be an error"
    );
}

/// `unify_biased`: rigid-vs-wobbly binds the wobbly variable to the rigid type.
#[test]
fn test_unify_biased_rigid_wobbly_binds() {
    let mut ctx = TypeContext::new();
    let wobbly = ctx.alloc(TypeData::InferVar { id: 0, universe: 0 });
    let rigid = ctx.int(32, true);
    let result = ctx.unify_biased(wobbly, rigid, Rigidity::Wobbly, Rigidity::Rigid);
    assert!(result.is_ok(), "rigid-vs-wobbly should succeed");
    // The wobbly variable should be bound to the rigid type.
    assert_eq!(ctx.resolve_binding(wobbly), rigid);
}

/// `unify_biased`: wobbly-vs-rigid binds the wobbly variable (symmetric).
#[test]
fn test_unify_biased_wobbly_rigid_binds() {
    let mut ctx = TypeContext::new();
    let rigid = ctx.int(32, true);
    let wobbly = ctx.alloc(TypeData::InferVar { id: 0, universe: 0 });
    let result = ctx.unify_biased(rigid, wobbly, Rigidity::Rigid, Rigidity::Wobbly);
    assert!(result.is_ok(), "wobbly-vs-rigid should succeed");
    // The wobbly variable should be bound to the rigid type.
    assert_eq!(ctx.resolve_binding(wobbly), rigid);
}

/// `unify_biased`: wobbly-vs-wobbly uses standard MGU (equality constraint).
#[test]
fn test_unify_biased_wobbly_wobbly() {
    let mut ctx = TypeContext::new();
    let a = ctx.alloc(TypeData::InferVar { id: 0, universe: 0 });
    let b = ctx.alloc(TypeData::InferVar { id: 1, universe: 0 });
    let result = ctx.unify_biased(a, b, Rigidity::Wobbly, Rigidity::Wobbly);
    assert!(result.is_ok(), "wobbly-vs-wobbly should succeed");
    // Both should resolve to the same thing (equality constraint).
    assert_eq!(ctx.resolve_binding(a), ctx.resolve_binding(b));
}

/// `bias_fresh_vars`: biases fresh InferVars in a compound type.
#[test]
fn test_bias_fresh_vars_tuple() {
    let mut ctx = TypeContext::new();
    let fresh0 = ctx.alloc(TypeData::InferVar { id: 0, universe: 0 });
    let fresh1 = ctx.alloc(TypeData::InferVar { id: 1, universe: 0 });
    let var_tuple = ctx.tuple(smallvec![fresh0, fresh1]);
    let int32 = ctx.int(32, true);
    let bool_ty = ctx.bool();
    let target_tuple = ctx.tuple(smallvec![int32, bool_ty]);
    let result = ctx.bias_fresh_vars(var_tuple, target_tuple);
    assert!(result.is_ok(), "bias_fresh_vars on tuple should succeed");
    // The fresh variables inside the tuple should be bound.
    assert_eq!(ctx.resolve_binding(fresh0), int32);
    assert_eq!(ctx.resolve_binding(fresh1), bool_ty);
}

/// `bias_fresh_vars`: occurs check prevents recursive binding.
#[test]
fn test_bias_fresh_vars_occurs_check() {
    let mut ctx = TypeContext::new();
    let fresh = ctx.alloc(TypeData::InferVar { id: 0, universe: 0 });
    // Try to bind fresh to itself (occurs check should reject).
    let result = ctx.bias_fresh_vars(fresh, fresh);
    assert!(
        result.is_err(),
        "bias_fresh_vars should reject recursive binding"
    );
}

/// Stage 5a: yoneda_equiv_check — canonical-form (βη) equivalence fallback
/// with parametricity gate (fail-closed).
#[test]
fn test_yoneda_equiv_check_basic() {
    let mut ctx = TypeContext::new();
    let i = ctx.int(32, true);
    let u = ctx.unit();
    // Structurally identical fns unify anyway; the canonical query answers true.
    let f1 = ctx.function(smallvec![i], u);
    let f2 = ctx.function(smallvec![i], u);
    assert_eq!(ctx.yoneda_equiv_check(f1, f2), Some(true));

    // Non-equivalent types must answer false.
    let f3 = ctx.function(smallvec![i], i);
    assert_eq!(ctx.yoneda_equiv_check(f1, f3), Some(false));
}

#[test]
fn test_yoneda_equiv_check_gate_closed() {
    // A `dyn Trait` (or direct break_parametricity) closes the gate:
    // fail-closed None for equivalence-strength consumers.
    let mut ctx = TypeContext::new();
    let i = ctx.int(32, true);
    let u = ctx.unit();
    let f1 = ctx.function(smallvec![i], u);
    let f2 = ctx.function(smallvec![i], u);

    ctx.parametricity.break_parametricity();
    assert_eq!(ctx.yoneda_equiv_check(f1, f2), None);
}

#[test]
fn test_yoneda_equiv_check_beta_eta_pair() {
    // ∀X.(Int⇒X)⇒Unit vs (Int⇒Unit)⇒Unit?  Not Yoneda-iso under the
    // partial Fig.4a axiomatization; but the vacuous-binder law gives
    // ∀X.Unit ≡ Unit.  Check the βη pair: Int⇒∀X.Unit vs ∀X.(Int⇒Unit).
    let mut ctx = TypeContext::new();
    let i = ctx.int(32, true);
    let u = ctx.unit();
    let forall_u = ctx.forall(0, "X".into(), u);
    let lifted = ctx.function(smallvec![i], forall_u);
    let fn_i_u = ctx.function(smallvec![i], u);
    let hoisted = ctx.forall(0, "X".into(), fn_i_u);
    // These do NOT unify structurally (different TypeData shapes).
    assert_ne!(lifted, hoisted);
    // But they are βη-isomorphic (A⇒∀X.B ≡ ∀X.A⇒B).
    assert_eq!(ctx.yoneda_equiv_check(lifted, hoisted), Some(true));
}

/// Regression: `TypeContext::get`/`get_raw` must assert arena-slot
/// existence, never `index < types.count()`.
///
/// `boxcar::Vec::count()` counts *completed* pushes, and completion order
/// is not claim order: a thread stalled between `next_index()` and its
/// `count.fetch_add` leaves `count` below `max_claimed_index + 1` even
/// though every already-returned index has a fully written slot.  The
/// assertion below drives that interleaving deterministically via
/// `push_with` (its closure runs after the claim, before the bump) and
/// shows a perfectly valid pushed index observing `index >= count()` —
/// i.e. the old predicate was a load-dependent false positive that made
/// `cargo test` fail ~1 run in N under full-suite parallelism.
#[test]
fn boxcar_count_lags_in_flight_push_so_index_lt_count_is_unsound() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let arena: boxcar::Vec<u32> = boxcar::Vec::new();
    let claimed = AtomicBool::new(false);
    let release = AtomicBool::new(false);

    std::thread::scope(|s| {
        s.spawn(|| {
            // Claims index 0, then stalls inside the closure — i.e.
            // between the claim and the `count` increment in `write`.
            arena.push_with(|_| {
                claimed.store(true, Ordering::SeqCst);
                while !release.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                1
            });
        });

        while !claimed.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        // Completes a push at index 1 while index 0 is still in flight.
        let idx = arena.push(2);
        let count = arena.count();

        release.store(true, Ordering::SeqCst);

        assert_eq!(idx, 1);
        assert_eq!(
            count, 1,
            "count must lag the completed push at index 1 while index 0 is in flight"
        );
        assert!(
            idx >= count,
            "a fully pushed index must be able to observe index >= count() \
             — otherwise the old get() precondition would hold"
        );
    });
}
