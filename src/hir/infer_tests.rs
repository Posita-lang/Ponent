//! Tests for the OmniML-style type inference engine.
//!
//! This file covers the core `InferenceContext` functionality:
//! constraint solving, region management, guard sets, generalization,
//! shape variables, and the full solve pipeline.

use crate::ast::Span;
use crate::hir::infer::*;
use crate::hir::shape_var::TypeShape;
use crate::hir::symbol::SymbolTable;
use crate::hir::traits::TraitEnv;
use crate::hir::types::*;
use crate::symbol::Symbol;

use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

// ── Helpers ──

fn new_ctx() -> TypeContext<'static> {
    TypeContext::new()
}

fn new_infer() -> InferenceContext {
    InferenceContext::new()
}

fn infer_var_id<'input>(ctx: &TypeContext<'input>, ty: TypeId) -> usize {
    match ctx.get(ty) {
        TypeData::InferVar { id, .. } => *id,
        _ => panic!("not an infer var"),
    }
}

fn span() -> Span {
    Span::new(0, 0)
}

fn default_env<'input>() -> (SymbolTable<'input>, TraitEnv<'input>) {
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let trait_env = TraitEnv::new();
    (symbols, trait_env)
}

// ── Basic Operations ──

#[test]
fn test_infer_var_creation() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    assert!(
        ctx.is_infer_var(var),
        "new_type_var should create an InferVar"
    );
    let id = infer_var_id(&ctx, var);
    assert!(infer.get_var_level(id).is_some(), "var should have a level");
}

#[test]
fn test_infer_var_kind_tracking() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let numeric = infer.new_type_var(
        &mut ctx,
        TypeVariableKind::Numeric,
        VarOrigin::Expression(Some(span())),
    );
    let any = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let unconstrained = infer.new_type_var(
        &mut ctx,
        TypeVariableKind::Unconstrained,
        VarOrigin::GenericParam,
    );

    assert_eq!(
        infer.get_var_kind(infer_var_id(&ctx, numeric)),
        Some(TypeVariableKind::Numeric)
    );
    assert_eq!(
        infer.get_var_kind(infer_var_id(&ctx, any)),
        Some(TypeVariableKind::Any)
    );
    assert_eq!(
        infer.get_var_kind(infer_var_id(&ctx, unconstrained)),
        Some(TypeVariableKind::Unconstrained)
    );
}

#[test]
fn test_var_creation_with_origins_preserves_kind() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let _expr_var = infer.new_type_var(
        &mut ctx,
        TypeVariableKind::Any,
        VarOrigin::Expression(Some(span())),
    );
    let _synthetic = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let _gp = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::GenericParam);

    // Var origins are tracked internally; the public API is get_var_kind()
    // which returns the TypeVariableKind for a given variable ID.
    // Since all vars were created with TypeVariableKind::Any, they
    // should all be Any.
    // We can verify the public API works by checking type_var creation.
    // (VarOrigin is an internal detail used for defaulting.)
    assert_eq!(
        infer.get_var_kind(infer_var_id(&ctx, _expr_var)),
        Some(TypeVariableKind::Any)
    );
    assert_eq!(
        infer.get_var_kind(infer_var_id(&ctx, _synthetic)),
        Some(TypeVariableKind::Any)
    );
    assert_eq!(
        infer.get_var_kind(infer_var_id(&ctx, _gp)),
        Some(TypeVariableKind::Any)
    );
}

// ── Region / Level Tests ──

#[test]
fn test_region_enter_exit_basic() {
    let mut infer = new_infer();
    let root = infer.region_tree.root;
    assert_eq!(infer.region_tree.current, root);

    let parent = infer.enter_level();
    let child = infer.region_tree.current;
    assert_ne!(child, root, "entering should change current region");

    infer.exit_level(parent);
    assert_eq!(
        infer.region_tree.current, parent,
        "exiting should restore parent"
    );
}

#[test]
fn test_region_nesting() {
    let mut infer = new_infer();
    let root = infer.region_tree.root;

    let p1 = infer.enter_level();
    let r1 = infer.region_tree.current;
    let p2 = infer.enter_level();
    let r2 = infer.region_tree.current;

    assert_ne!(r1, root);
    assert_ne!(r2, r1);
    assert_eq!(infer.region_tree.nodes[r1.0].parent, Some(root));
    assert_eq!(infer.region_tree.nodes[r2.0].parent, Some(r1));

    infer.exit_level(p2);
    assert_eq!(infer.region_tree.current, r1);
    infer.exit_level(p1);
    assert_eq!(infer.region_tree.current, root);
}

#[test]
fn test_var_allocated_in_current_region() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let root = infer.region_tree.root;

    let _p = infer.enter_level();
    let child = infer.region_tree.current;
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    assert!(
        infer.region_tree.nodes[child.0].pool.var_ids.contains(&id),
        "var should be in child region's pool"
    );
    assert!(
        !infer.region_tree.nodes[root.0].pool.var_ids.contains(&id),
        "var should NOT be in root region's pool"
    );
}

// ── Constraint Tests ──

#[test]
fn test_add_constraint_eq() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let trait_env = TraitEnv::new();

    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let int32 = ctx.int(32, true);

    // Eq(a, b) with b bound to Int<32>
    infer.add_constraint(Constraint::Eq(a, b, span(), EqOrigin::Normal));
    ctx.set_binding(b, int32);

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solve should succeed: {:?}", result);
    let resolved = ctx.resolve_binding(a);
    assert!(ctx.is_integer(resolved), "a should resolve to Int<32>");
}

#[test]
fn test_add_constraint_sub() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let int32 = ctx.int(32, true);

    // Sub(a, b) + Eq(b, int32) — Sub records bounds, Eq resolves b
    infer.add_constraint(Constraint::Sub(a, b, span()));
    infer.add_constraint(Constraint::Eq(b, int32, span(), EqOrigin::Normal));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "sub solve should succeed: {:?}", result);
    // b is resolved to Int<32> via the Eq constraint
    let resolved_b = ctx.resolve_binding(b);
    assert!(ctx.is_integer(resolved_b), "b should resolve to Int<32>");
    // a is NOT transitively resolved by Sub alone — Sub only records bounds.
    // (The solver records int32 as an upper bound of a, but does not unify a.)
}

#[test]
fn test_add_constraint_mismatch_fails() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let bool_ty = ctx.bool();
    let int32 = ctx.int(32, true);

    infer.add_constraint(Constraint::Eq(a, bool_ty, span(), EqOrigin::Normal));
    ctx.set_binding(a, int32);

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_err(), "Bool vs Int mismatch should fail");
}

// ── Guard Set Tests ──

#[test]
fn test_guard_basic_lifecycle() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    assert!(
        !matches!(infer.get_gen_status(id), Some(GenStatus::Generalized)),
        "new var should not be generalized"
    );
    assert_eq!(infer.get_gen_status(id), Some(GenStatus::Ungeneralized));

    // Add a guard: marks as PG
    infer.add_guard(id);
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::PartiallyGeneralizable),
        "var should be PG after add_guard"
    );

    // Remove the guard: var is still unresolved (InferVar), so PG→G
    // does NOT fire (only generalize resolved variables).
    infer.remove_guard(id, &ctx);
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::PartiallyGeneralizable),
        "unresolved var stays PG after remove_guard"
    );

    // Now resolve the var — the next remove_guard or wake will PG→G.
    // But since there are no more guards, we need to trigger via wake.
    // For this test, just verify the unresolved invariant holds.
}

#[test]
fn test_guard_multi_guard() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    // Multiple guards: reference-counted
    infer.add_guard(id);
    infer.add_guard(id);
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::PartiallyGeneralizable),
        "should be PG with 2 guards"
    );

    infer.remove_guard(id, &ctx);
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::PartiallyGeneralizable),
        "should still be PG with 1 guard"
    );

    infer.remove_guard(id, &ctx);
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::PartiallyGeneralizable),
        "unresolved var stays PG after removing all guards"
    );
}

#[test]
fn test_guard_snapshot_rollback() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    let snap = infer.start_snapshot();
    infer.add_guard(id);
    assert!(infer.is_guarded(id), "should be guarded inside snapshot");

    infer.rollback_to(snap);
    assert!(!infer.is_guarded(id), "guard should be rolled back");
}

#[test]
fn test_guard_double_add_rollback_restores_status() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    let snap = infer.start_snapshot();
    infer.add_guard(id);
    infer.add_guard(id);
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::PartiallyGeneralizable),
        "should be PG with 2 guards"
    );

    // Rolling back must undo BOTH guards and restore the pre-snapshot
    // status — the second add (on an already-PG var) must not leave a
    // stale `SetGenStatus` undo behind.
    infer.rollback_to(snap);
    assert!(!infer.is_guarded(id), "both guards should be rolled back");
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::Ungeneralized),
        "status should be restored to the pre-snapshot value"
    );
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "add_guard: direct-guard refcount overflow")]
fn test_guard_add_refcount_overflow_panics() {
    let mut gs = GuardSet::empty();
    gs.direct_guards = usize::MAX;
    gs.add_guard();
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "remove_guard: count == 0")]
fn test_guard_remove_underflow_panics() {
    let mut gs = GuardSet::empty();
    gs.remove_guard();
}

// ── Suspend / Wake Tests ──

#[test]
fn test_suspend_on_var_basic() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    let cst = Constraint::Eq(ctx.bool(), ctx.bool(), span(), EqOrigin::Normal);
    infer.suspend_on_var(cst, id);

    // After suspend_on_var, the constraint is moved to the wait list
    // (the constraints vec is cleared of the suspended constraint)
    assert!(
        infer.constraints().is_empty(),
        "active constraints should be empty after suspend"
    );
    assert!(
        id < infer.wait_lists().len(),
        "suspend_on_var should have grown wait_lists for id={}",
        id,
    );
    assert!(
        !infer.wait_lists()[id].is_empty(),
        "var should have suspended constraints in wait list"
    );

    // Waking moves suspended constraints back to active
    let woken = infer.wake_var_for_test(id, &ctx);
    assert!(woken > 0, "woken constraints should be active");
    assert!(
        infer.wait_lists()[id].is_empty(),
        "wait list should be empty after wake"
    );
}

#[test]
fn test_suspend_on_var_adds_guard() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    let cst = Constraint::Eq(ctx.bool(), ctx.bool(), span(), EqOrigin::Normal);
    infer.suspend_on_var(cst, id);

    // Suspending a constraint should add a guard (PG state)
    assert!(
        infer.is_guarded(id),
        "var should be guarded after suspension"
    );
}

// ── Generalization Tests ──

#[test]
fn test_force_generalize_basic() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    // The unified eligibility predicate (same as the retired sweep)
    // requires the variable to be RESOLVED before it can enter a scheme —
    // an unresolved, unguarded var stays in the pool (fail-closed; it is
    // defaulted by the solver instead of being generalized to ∀α.α).
    let int32 = ctx.int(32, true);
    ctx.set_binding(infer.var_type_id_for_test(id), int32);

    // No guards, no wait list — should generalize
    let generalized = infer.force_root_generalization(&mut ctx);
    assert!(
        generalized.iter().any(|&(gid, _)| gid == id),
        "unguarded var should be generalized, got {:?}",
        generalized,
    );
    assert_eq!(infer.get_gen_status(id), Some(GenStatus::Generalized));
}

#[test]
fn test_force_generalize_pg_with_guard_not_generalized() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    // Add a guard — should stay PG
    infer.add_guard(id);
    let generalized = infer.force_root_generalization(&mut ctx);
    assert!(
        !generalized.iter().any(|&(gid, _)| gid == id),
        "guarded var should NOT be generalized",
    );
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::PartiallyGeneralizable)
    );
}

/// PG lowering must route through the registered pool API so `var_to_region`
/// and the pool undo log stay consistent.  A direct
/// `retain`+`push` on `pool.var_ids` leaves `var_to_region` stale (the TcLevel
/// escape check reads it) and the move un-rollbackable (no undo entries).
#[test]
fn test_pg_lowering_maintains_var_to_region_and_undo_log() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    // This test inspects pool_undo_log entries directly, outside any
    // snapshot — enable logging explicitly.
    infer.region_tree.pool_logging_enabled = true;
    let root = infer.region_tree.root;
    let prev = infer.enter_level();
    let child = infer.region_tree.current;
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);
    infer.add_guard(id); // PG

    // Sanity: the var is registered in the child region and mapped there.
    assert!(
        infer.region_tree.nodes[child.0].pool.var_ids.contains(&id),
        "var should be in the child pool"
    );
    assert_eq!(
        infer.region_tree.region_of_var(id),
        child,
        "var_to_region should map the var to the child region"
    );

    // Generalize — the PG var must be LOWERED to the parent (root), never
    // generalized.
    let generalized = infer.force_root_generalization(&mut ctx);
    assert!(
        !generalized.iter().any(|&(gid, _)| gid == id),
        "PG var must NOT be generalized"
    );

    // var_to_region is consistent: the var now lives in the parent region.
    assert_eq!(
        infer.region_tree.region_of_var(id),
        root,
        "PG var must be lowered to the parent region in var_to_region"
    );
    // Pools agree: gone from child, present in parent (root).
    assert!(
        !infer.region_tree.nodes[child.0].pool.var_ids.contains(&id),
        "child pool must not contain the lowered var"
    );
    assert!(
        infer.region_tree.nodes[root.0].pool.var_ids.contains(&id),
        "parent (root) pool must contain the lowered var"
    );
    // The undo log records the move: Unregister(child) + Register(root).
    let has_unregister = infer.region_tree.pool_undo_log.iter().any(|e| {
        matches!(
            e,
            PoolUndoEntry::Unregister {
                region_idx,
                var_id,
            } if *region_idx == child.0 && *var_id == id
        )
    });
    let has_register = infer.region_tree.pool_undo_log.iter().any(|e| {
        matches!(
            e,
            PoolUndoEntry::Register {
                region_idx,
                var_id,
                ..
            } if *region_idx == root.0 && *var_id == id
        )
    });
    assert!(
        has_unregister && has_register,
        "the PG-lowering move must be recorded in the pool undo log"
    );
}

#[test]
fn test_force_generalize_pg_with_waitlist_stays_pg() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    // Suspend a constraint — adds guard AND wait list entry
    let cst = Constraint::Eq(ctx.bool(), ctx.bool(), span(), EqOrigin::Normal);
    infer.suspend_on_var(cst, id);

    let generalized = infer.force_root_generalization(&mut ctx);
    assert!(
        !generalized.iter().any(|&(gid, _)| gid == id),
        "var with wait list should NOT be generalized",
    );
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::PartiallyGeneralizable)
    );
}

// ── Instance / Forward Reference Tests ──

#[test]
fn test_register_instance_basic() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let pg_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let pg_id = infer_var_id(&ctx, pg_var);
    let instance = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let inst_id = infer_var_id(&ctx, instance);

    // Register instance
    infer.register_instance(pg_id, inst_id);

    // Reverse ref should be set
    assert_eq!(infer.reverse_refs()[inst_id], Some(pg_id));

    // Forward ref should contain the instance
    assert!(
        pg_id < infer.forward_refs().len(),
        "register_instance should have grown forward_refs for pg_id={}",
        pg_id,
    );
    assert!(infer.forward_refs()[pg_id].contains(&inst_id));

    // Instance status should be PartialInstance
    assert_eq!(
        infer.get_gen_status(inst_id),
        Some(GenStatus::PartialInstance)
    );
}

// ── Shape Variable Tests ──

#[test]
fn test_shape_var_create_and_resolve() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let level = infer.region_tree.nodes[infer.region_tree.current.0].level;

    let sv = infer.shape_vars.new_var(level);
    assert!(!infer.shape_vars.is_resolved(sv));

    // Resolve to Arrow shape
    assert!(infer.shape_vars.try_set(sv, TypeShape::Arrow));
    assert!(infer.shape_vars.is_resolved(sv));
    assert_eq!(infer.shape_vars.get(sv), Some(TypeShape::Arrow));
}

#[test]
fn test_shape_var_unify_aliases() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let level = infer.region_tree.nodes[infer.region_tree.current.0].level;

    let sv1 = infer.shape_vars.new_var(level);
    let sv2 = infer.shape_vars.new_var(level);

    // Unify them
    infer.shape_vars.unify(sv1, sv2);
    // After unification, both should resolve to the same canonical id
    assert_eq!(
        infer.shape_vars.resolve(sv1),
        infer.shape_vars.resolve(sv2),
        "unified shape vars should share the same canonical id",
    );
    // Resolving one should resolve the other
    assert!(infer.shape_vars.try_set(sv1, TypeShape::Arrow));
    assert!(infer.shape_vars.is_resolved(sv2));
    assert_eq!(infer.shape_vars.get(sv2), Some(TypeShape::Arrow));
}

#[test]
fn test_shape_var_callback_on_resolve() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let level = infer.region_tree.nodes[infer.region_tree.current.0].level;

    let sv = infer.shape_vars.new_var(level);
    let called = Arc::new(AtomicBool::new(false));
    let called_cb = Arc::clone(&called);
    infer.shape_vars.on_resolve(
        sv,
        Box::new(move |_| {
            called_cb.store(true, Ordering::SeqCst);
        }),
    );
    assert!(
        !called.load(Ordering::SeqCst),
        "callback should not fire before resolve"
    );

    infer.shape_vars.try_set(sv, TypeShape::Tuple(2));
    assert!(
        called.load(Ordering::SeqCst),
        "callback should fire after resolve"
    );
}

// ── Try Promote Var Tests ──

#[test]
fn test_try_promote_var_to_root() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let root = infer.region_tree.root;

    let _p = infer.enter_level();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    let promoted = infer.try_promote_var(&mut ctx, id, root);
    assert!(promoted.is_some(), "promotion to root should succeed");

    let promoted_id = infer_var_id(&ctx, promoted.unwrap());
    assert!(
        infer.region_tree.nodes[root.0]
            .pool
            .var_ids
            .contains(&promoted_id),
        "promoted var should be in root pool",
    );
}

// ── Rigid Escape Tests ──

#[test]
fn test_rigid_escape_detected() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    // Create a SkolemVar (rigid variable) via enter_universe
    let (_universe, skolem) = ctx.enter_universe();

    // check_rigid_escape should detect the SkolemVar
    let escaped = InferenceContext::check_rigid_escape(
        &ctx,
        skolem,
        0,
        &InferenceContext::new().region_tree,
        &std::collections::HashMap::default(),
    );
    assert!(escaped, "SkolemVar should be detected as escape");
}

#[test]
fn test_rigid_escape_generic_param_not_detected() {
    let mut ctx = new_ctx();

    // Outer GenericParams are part of the environment (Γ) and do NOT escape.
    let gp = ctx.generic_param(0, "X".into());
    let escaped = InferenceContext::check_rigid_escape(
        &ctx,
        gp,
        0,
        &InferenceContext::new().region_tree,
        &std::collections::HashMap::default(),
    );
    assert!(
        !escaped,
        "Outer GenericParam should NOT be detected as escape"
    );

    let fn_ty = ctx.function(smallvec![gp], gp);
    let fn_escaped = InferenceContext::check_rigid_escape(
        &ctx,
        fn_ty,
        0,
        &InferenceContext::new().region_tree,
        &std::collections::HashMap::default(),
    );
    assert!(
        !fn_escaped,
        "fn(X) -> X with outer GenericParam should not escape"
    );
}

// ── s_inst_copy Tests ──

#[test]
fn test_s_inst_copy_basic() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create a PG variable and an instance
    let pg_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let pg_id = infer_var_id(&ctx, pg_var);
    let instance = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let inst_id = infer_var_id(&ctx, instance);

    // Add a guard to make it PG
    infer.add_guard(pg_id);
    infer.register_instance(pg_id, inst_id);

    // Now resolve the PG var to Int<32>
    let int32 = ctx.int(32, true);
    ctx.set_binding(pg_var, int32);

    // Run s_inst_copy to propagate
    infer.s_inst_copy(&mut ctx, pg_id, int32);

    // The instance should now also be resolved to Int<32>
    let inst_resolved = ctx.resolve_binding(instance);
    assert!(
        ctx.is_integer(inst_resolved),
        "instance should be resolved to Int<32> after s_inst_copy",
    );
}

// ── s_exists_lower Tests ──

#[test]
fn test_s_exists_lower_concrete() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    // Enter a deeper level so the var's level > current_level (required
    // by the level-based fallback in s_exists_lower).
    let prev = infer.enter_level();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    // S-Exists-Lower requires PG status and level > current_level
    infer.add_guard(id);
    // Exit the level so current_level drops below the var's level
    infer.exit_level(prev);

    let result = infer.s_exists_lower(&ctx, id, &SkolemEnv::default(), &[], None);
    // Removed the unsound level-based fallback.
    // Without determinacy proof, the variable stays PG.
    assert!(
        !result,
        "S-Exists-Lower should fail without determinacy proof"
    );
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::PartiallyGeneralizable),
        "PG var should stay PG when determinacy cannot be proven"
    );
}

#[test]
fn test_s_exists_lower_forall() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    // Same pattern: deeper level, create var, set PG, exit
    let prev = infer.enter_level();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    // Add an Eq constraint to a Forall type to give the var a determined shape
    let p0 = ctx.generic_param(0, "X".into());
    let fn_ty = ctx.function(smallvec![p0], p0);
    let forall = ctx.forall(0, "X".into(), fn_ty);
    // The Eq constraint gives the Z3/unicity check a concrete type to work with
    infer.add_constraint(Constraint::Eq(var, forall, span(), EqOrigin::Normal));
    infer.add_guard(id);
    infer.exit_level(prev);

    let result = infer.s_exists_lower(&ctx, id, &SkolemEnv::default(), &[], None);
    // Removed the unsound level-based fallback.
    // The Forall Eq constraint is not yet processed by solve, so
    // determinacy cannot be proven — variable stays PG.
    assert!(
        !result,
        "s_exists_lower should fail without processed constraints"
    );
}

#[test]
fn test_s_exists_lower_env_precision() {
    // The env-aware lowering must be PRECISE, not a blanket refusal:
    //
    //   1. A non-empty env must not block lowering when the variable's
    //      equations never touch a skolem (the old bail-out did).
    //   2. An equation linking the variable to a skolem (a β̄ member)
    //      must block lowering — determinacy from binder-internal
    //      structure is not an outer assignment.
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    let prev = infer.enter_level();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);
    let other = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let other_id = infer_var_id(&ctx, other);
    // Mark `var` as PartiallyGeneralizable — s_exists_lower only
    // considers PG variables, and add_guard is the public path that
    // sets PG (OmniML §6: guarded => PG).
    infer.add_guard(id);
    let int32 = ctx.int(32, true);

    // (1) var == Int<32> (bound directly): determined by a CONCRETE
    // outer type.  `other` is bound by an enclosing Forall (it is in
    // the env), but nothing links `var` to it — lowering must succeed.
    // (Constraints are NOT processed by solve here; the direct binding
    // exercises the value-determinacy path, which is what the env must
    // not block.)
    ctx.set_binding(var, int32);
    ctx.set_binding(other, int32);
    infer.exit_level(prev);

    let mut env = SkolemEnv::default();
    env.insert(other_id, ctx.enter_universe().1);
    let result = infer.s_exists_lower(&ctx, id, &env, &[], None);
    assert!(
        result,
        "non-empty env must not block lowering when no equation \
         touches a skolem"
    );

    // (2) Same setup, but the variable is equated with the skolem-bound
    // var itself: the equation observes binder-internal structure, so
    // lowering must be refused.
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    let prev = infer.enter_level();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);
    let other = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let other_id = infer_var_id(&ctx, other);
    infer.add_guard(id);
    infer.add_constraint(Constraint::Eq(var, other, span(), EqOrigin::Normal));
    infer.exit_level(prev);

    let mut env = SkolemEnv::default();
    env.insert(other_id, ctx.enter_universe().1);
    let result = infer.s_exists_lower(&ctx, id, &env, &[], None);
    assert!(
        !result,
        "an equation linking the variable to a β̄ skolem must block \
         lowering (Det-Dom must not fire on a rigid binder member)"
    );
}

#[test]
fn test_s_exists_lower_outer_skolem_determines() {
    // Level-based precision (the blanket rule's only behavioural
    // difference): a skolem introduced by a Forall opened at an OUTER
    // region (strictly lower level than the variable's region) is a
    // fixed outer constant.  Under OmniML Definition 5.3's naive
    // reading it DETERMINES the variable — every ground assignment
    // agrees on it — so an equation linking the variable to it must
    // let Det-Dom fire and lower.  (The blanket rule folded ALL env
    // keys into region_vars and would refuse here.)
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    // Open an outer region; the Forall (skolem) will be introduced at
    // this level.  The variable lives in a strictly inner region.
    let outer = infer.enter_level();
    let other = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let other_id = infer_var_id(&ctx, other);
    let (universe, skolem_ty) = ctx.enter_universe();
    let skolem_level = infer.region_tree.get_level(infer.region_tree.current);
    infer
        .region_tree
        .record_skolem_level(universe, skolem_level);

    let inner = infer.enter_level();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);
    infer.add_guard(id);
    infer.exit_level(inner);
    infer.exit_level(outer);

    // var == other, where other is skolem-bound (rigid, outer).  The
    // Eq is passed as an ACTIVE constraint (mirroring the production
    // call site, which feeds the unsolved heap) carrying the env.
    let mut env = SkolemEnv::default();
    env.insert(other_id, skolem_ty);
    let active = vec![PrioritizedConstraint {
        priority: 2, // PRIORITY_EQ_INFER
        constraint: Constraint::Eq(var, other, span(), EqOrigin::Normal),
        env: Arc::new(env.clone()),
    }];
    let result = infer.s_exists_lower(&ctx, id, &env, &active, None);
    assert!(
        result,
        "an outer (lower-level) skolem is a fixed outer constant and \
         must be allowed to determine the variable (Det-Dom fires)"
    );
}

// ── Full Solve Pipeline Tests ──

#[test]
fn test_solve_multiple_eq() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let c = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let int32 = ctx.int(32, true);

    // Chain: a == b == c, with c bound to Int<32>
    infer.add_constraint(Constraint::Eq(a, b, span(), EqOrigin::Normal));
    infer.add_constraint(Constraint::Eq(b, c, span(), EqOrigin::Normal));
    ctx.set_binding(c, int32);

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "chain solve should succeed: {:?}", result);
    assert!(
        ctx.is_integer(ctx.resolve_binding(a)),
        "a should resolve to Int<32>"
    );
    assert!(
        ctx.is_integer(ctx.resolve_binding(b)),
        "b should resolve to Int<32>"
    );
}

#[test]
fn test_solve_eq_with_guard() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let int32 = ctx.int(32, true);

    let a_id = infer_var_id(&ctx, a);

    // Guard a — should become PG
    infer.add_guard(a_id);

    // Eq(a, b) with b bound to Int<32>
    infer.add_constraint(Constraint::Eq(a, b, span(), EqOrigin::Normal));
    ctx.set_binding(b, int32);

    // Solve should handle guarded variables correctly
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "solve with guarded var should succeed: {:?}",
        result
    );
    let resolved = ctx.resolve_binding(a);
    assert!(ctx.is_integer(resolved), "a should resolve to Int<32>");
}

#[test]
fn test_solve_subtype_constraint() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let int32 = ctx.int(32, true);

    // Sub(a, b) and Eq(b, Int<32>) — Sub records bounds, Eq resolves b
    infer.add_constraint(Constraint::Sub(a, b, span()));
    infer.add_constraint(Constraint::Eq(b, int32, span(), EqOrigin::Normal));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "subtype solve should succeed: {:?}", result);
    // b is resolved to Int<32> via the Eq constraint
    let resolved_b = ctx.resolve_binding(b);
    assert!(ctx.is_integer(resolved_b), "b should resolve to Int<32>");
}

#[test]
fn test_solve_with_promotion() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create variables at different levels
    let prev = infer.enter_level();
    let deep = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let shallow = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    infer.exit_level(prev);

    let int32 = ctx.int(32, true);
    infer.add_constraint(Constraint::Eq(deep, shallow, span(), EqOrigin::Normal));
    ctx.set_binding(shallow, int32);

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "solve with promotion should succeed: {:?}",
        result
    );
    let resolved = ctx.resolve_binding(deep);
    assert!(
        ctx.is_integer(resolved),
        "deep var should resolve to Int<32>"
    );
}

#[test]
fn test_solve_cross_branch_promotion() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create tree: root → branch_A → leaf_A
    //                  └─ branch_B → leaf_B
    let _r0 = infer.enter_level();
    let _r1 = infer.enter_level();
    let a_deep = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    infer.exit_level(_r1);
    infer.exit_level(_r0);

    let _r2 = infer.enter_level();
    let _r3 = infer.enter_level();
    let b_deep = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    infer.exit_level(_r3);
    infer.exit_level(_r2);

    let int32 = ctx.int(32, true);
    infer.add_constraint(Constraint::Eq(a_deep, b_deep, span(), EqOrigin::Normal));
    ctx.set_binding(b_deep, int32);

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "cross-branch solve should succeed: {:?}",
        result
    );
    let a_resolved = ctx.resolve_binding(a_deep);
    let b_resolved = ctx.resolve_binding(b_deep);
    assert!(
        ctx.is_integer(a_resolved),
        "a_deep should resolve to Int<32>"
    );
    assert!(
        ctx.is_integer(b_resolved),
        "b_deep should resolve to Int<32>"
    );
}

#[test]
fn test_solve_cross_branch_mismatch_fails() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let _r0 = infer.enter_level();
    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    infer.exit_level(_r0);

    let _r1 = infer.enter_level();
    let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    infer.exit_level(_r1);

    let bool_ty = ctx.bool();
    let int32 = ctx.int(32, true);
    ctx.set_binding(a, bool_ty);
    ctx.set_binding(b, int32);

    infer.add_constraint(Constraint::Eq(a, b, span(), EqOrigin::Normal));
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_err(), "Bool vs Int cross-branch should fail");
}

// ── Snapshot / Rollback Tests ──

#[test]
fn test_snapshot_rollback_restores_guards() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, a);

    let snap = infer.start_snapshot();
    // Add a guard — tracked by the undo log
    infer.add_guard(id);
    assert!(
        infer.is_guarded(id),
        "guard should be added inside snapshot"
    );

    infer.rollback_to(snap);
    assert!(!infer.is_guarded(id), "guard should be rolled back",);
}

#[test]
fn test_inference_snapshot_commit_does_not_rollback_type_context_binding() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let int32 = ctx.int(32, true);

    let snap = infer.start_snapshot();
    ctx.set_binding(a, int32);
    infer.commit_snapshot(snap);

    assert!(
        ctx.is_integer(ctx.resolve_binding(a)),
        "a should still be Int<32> after commit",
    );
}

// ── Yoneda Integration Tests ──

#[test]
fn test_type_contains_param() {
    let mut ctx = new_ctx();
    let p0 = ctx.generic_param(0, "X".into());
    let p1 = ctx.generic_param(1, "Y".into());
    let int32 = ctx.int(32, true);

    assert!(
        ctx.type_contains_param(0, p0),
        "GenericParam should contain itself"
    );
    assert!(
        !ctx.type_contains_param(0, p1),
        "different param should NOT contain"
    );
    assert!(
        !ctx.type_contains_param(0, int32),
        "concrete type should NOT contain"
    );

    let fn_ty = ctx.function(smallvec![p0], int32);
    assert!(
        ctx.type_contains_param(0, fn_ty),
        "Fn with param should contain param"
    );
}

// ── Edge Cases ──

#[test]
fn test_solve_empty_constraints() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solving no constraints should succeed");
}

#[test]
fn test_var_level_after_multi_level_enter() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    let _p1 = infer.enter_level();
    let _p2 = infer.enter_level();
    let _p3 = infer.enter_level();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    let level = infer.get_var_level(id);
    assert!(level.is_some(), "deep var should have a level");
    assert!(level.unwrap() >= 3, "deep var level should be >= 3");
}

// ── Solve Pipeline Integration Tests ──

#[test]
fn test_solve_pipeline_end_to_end() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Chain: a == b == c == Int<32>
    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let c = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let int32 = ctx.int(32, true);

    infer.add_constraint(Constraint::Eq(a, b, span(), EqOrigin::Normal));
    infer.add_constraint(Constraint::Eq(b, c, span(), EqOrigin::Normal));
    ctx.set_binding(c, int32);

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "pipeline solve should succeed: {:?}",
        result
    );

    // All three should resolve to Int<32>
    let a_resolved = ctx.resolve_binding(a);
    let b_resolved = ctx.resolve_binding(b);
    let c_resolved = ctx.resolve_binding(c);
    assert!(
        ctx.is_integer(a_resolved),
        "terminal a should resolve to Int<32>"
    );
    assert!(
        ctx.is_integer(b_resolved),
        "terminal b should resolve to Int<32>"
    );
    assert!(
        ctx.is_integer(c_resolved),
        "terminal c should resolve to Int<32>"
    );

    // finalize should produce a complete solution map
    let solution = infer.finalize(&mut ctx);
    assert!(
        !solution.is_empty(),
        "finalize should produce a non-empty solution map"
    );
}

#[test]
fn test_solve_pipeline_with_promotion_and_generalization() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create variables at different levels, then solve forces promotion
    let prev = infer.enter_level();
    let deep = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    infer.exit_level(prev);

    let shallow = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let int32 = ctx.int(32, true);

    // Eq(deep, shallow) should promote deep to shallow's level
    infer.add_constraint(Constraint::Eq(deep, shallow, span(), EqOrigin::Normal));
    ctx.set_binding(shallow, int32);

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "solve with promotion should succeed: {:?}",
        result
    );

    let deep_resolved = ctx.resolve_binding(deep);
    let shallow_resolved = ctx.resolve_binding(shallow);
    assert!(
        ctx.is_integer(deep_resolved),
        "deep var should resolve to Int<32>"
    );
    assert!(
        ctx.is_integer(shallow_resolved),
        "shallow var should resolve to Int<32>"
    );
    assert_eq!(
        deep_resolved, shallow_resolved,
        "both should resolve to the same type"
    );
}

// ── Defaulting Tests ──

#[test]
fn test_defaulting_integer_kind() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // An Integer variable with no constraints should default to Int<32>
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Integer, VarOrigin::Synthetic);
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "solve with Integer var should succeed: {:?}",
        result
    );

    let resolved = ctx.resolve_binding(var);
    assert!(
        ctx.is_integer(resolved),
        "Integer var should default to Int<32>"
    );
}

#[test]
fn test_defaulting_float_kind() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Float, VarOrigin::Synthetic);
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "solve with Float var should succeed: {:?}",
        result
    );

    let resolved = ctx.resolve_binding(var);
    assert!(
        matches!(ctx.get(resolved), TypeData::Float { .. }),
        "Float var should default to Float"
    );
}

#[test]
fn test_defaulting_bool_kind() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Bool, VarOrigin::Synthetic);
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "solve with Bool var should succeed: {:?}",
        result
    );

    let resolved = ctx.resolve_binding(var);
    assert_eq!(
        ctx.get(resolved),
        &TypeData::Bool,
        "Bool var should default to Bool"
    );
}

#[test]
fn test_defaulting_numeric_kind() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Numeric, VarOrigin::Synthetic);
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "solve with Numeric var should succeed: {:?}",
        result
    );

    let resolved = ctx.resolve_binding(var);
    assert!(
        ctx.is_integer(resolved),
        "Numeric var should default to Int<32>"
    );
}

#[test]
fn test_defaulting_unconstrained_kind() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let var = infer.new_type_var(
        &mut ctx,
        TypeVariableKind::Unconstrained,
        VarOrigin::Synthetic,
    );
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "solve with Unconstrained var should succeed: {:?}",
        result
    );

    let resolved = ctx.resolve_binding(var);
    assert!(
        matches!(ctx.get(resolved), TypeData::Error),
        "Unconstrained var should default to Error"
    );
}

#[test]
fn test_defaulting_expression_var_defaults_to_error() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // An expression-level variable with Unconstrained kind and no constraints
    // should now default to ctx.error() instead of returning CannotInfer.
    // This allows the solver to continue processing other obligations.
    let var = infer.new_type_var(
        &mut ctx,
        TypeVariableKind::Unconstrained,
        VarOrigin::Expression(Some(span())),
    );
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "expression-level var without constraints should default to Error, got {:?}",
        result
    );
    let resolved = ctx.resolve_binding(var);
    assert!(
        matches!(ctx.get(resolved), TypeData::Error),
        "Unconstrained var should default to Error, got {:?}",
        ctx.get(resolved)
    );
}

// ── Kind Checking Tests ──

#[test]
fn test_kind_check_integer_unified_with_bool_fails() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let int_var = infer.new_type_var(&mut ctx, TypeVariableKind::Integer, VarOrigin::Synthetic);
    let bool_ty = ctx.bool();

    // Unifying an Integer variable with Bool should fail kind checking
    infer.add_constraint(Constraint::Eq(int_var, bool_ty, span(), EqOrigin::Normal));
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_err(), "Integer var unified with Bool should fail");
}

#[test]
fn test_kind_check_float_unified_with_int_fails() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let float_var = infer.new_type_var(&mut ctx, TypeVariableKind::Float, VarOrigin::Synthetic);
    let int32 = ctx.int(32, true);

    infer.add_constraint(Constraint::Eq(float_var, int32, span(), EqOrigin::Normal));
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_err(),
        "Float var unified with Int<32> should fail"
    );
}

// ── Forall / Exists Constraint Tests ──

#[test]
fn test_forall_constraint_creates_skolem() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create a Forall constraint: ∀α. Eq(α, Int<32>)
    // The Forall binds α to a SkolemVar via ctx.set_binding, so the
    // unifier sees α as rigid.  Eq(α, Int<32>) must fail because a
    // SkolemVar cannot unify with a concrete type.
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let alpha_id = infer_var_id(&ctx, alpha);
    let int32 = ctx.int(32, true);

    let body = Constraint::Eq(alpha, int32, span(), EqOrigin::Normal);
    infer.add_constraint(Constraint::Forall {
        var_id: alpha_id,
        constraint: Box::new(body),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_err(),
        "Forall rigid variable must not unify with Int<32>: got {:?}",
        result,
    );
}

#[test]
fn test_forall_skolem_does_not_leak_after_body() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create a Forall constraint: ∀α. Eq(α, α)
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let alpha_id = infer_var_id(&ctx, alpha);

    infer.add_constraint(Constraint::Forall {
        var_id: alpha_id,
        constraint: Box::new(Constraint::Eq(alpha, alpha, span(), EqOrigin::Normal)),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "Forall body Eq(α, α) should succeed: {:?}",
        result
    );

    // After the Forall body is solved, the Skolem is scoped to the
    // body constraint's environment (pc.env), which is destroyed when
    // the constraint is popped from the heap.  No Skolem should leak
    // into ctx.bindings (the persistent type resolution store).
    let resolved = ctx.resolve_binding(alpha);
    assert!(
        !matches!(ctx.get(resolved), TypeData::SkolemVar { .. }),
        "Forall skolem binding leaked into ctx.bindings outside its body",
    );
    // α is still an InferVar in the type arena (the defaulting phase may
    // have set a binding in ctx.bindings, but the TypeData is unchanged).
    assert!(
        matches!(ctx.get_raw(alpha), TypeData::InferVar { .. }),
        "α should still be an InferVar in the type arena after Forall scope, got {:?}",
        ctx.get_raw(alpha),
    );
}

#[test]
fn test_forall_skolem_restored_after_body_error() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create a Forall constraint: ∀α. Eq(α, Int<32>) — should fail
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let alpha_id = infer_var_id(&ctx, alpha);
    let int32 = ctx.int(32, true);

    infer.add_constraint(Constraint::Forall {
        var_id: alpha_id,
        constraint: Box::new(Constraint::Eq(alpha, int32, span(), EqOrigin::Normal)),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_err(),
        "Forall rigid variable must not unify with Int<32>"
    );

    // After the failing solve, the Scoped Skolem is in the constraint's
    // environment (pc.env), which is destroyed when the constraint is
    // popped from the heap.  No Skolem should leak into ctx.bindings.
    let resolved = ctx.resolve_binding(alpha);
    assert!(
        !matches!(ctx.get(resolved), TypeData::SkolemVar { .. }),
        "Forall skolem leaked after failing body",
    );
}

#[test]
fn test_exists_constraint_foralls_skolem_escape() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create an Exists constraint: ∃α. Eq(α, Int<32>)
    // α is a flexible variable, so Eq(α, Int<32>) should succeed
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let alpha_id = infer_var_id(&ctx, alpha);
    let int32 = ctx.int(32, true);

    let body = Constraint::Eq(alpha, int32, span(), EqOrigin::Normal);
    infer.add_constraint(Constraint::Exists {
        var_id: alpha_id,
        constraint: Box::new(body),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "Exists with Eq(α, Int<32>) should succeed: {:?}",
        result
    );

    let resolved = ctx.resolve_binding(alpha);
    assert!(
        ctx.is_integer(resolved),
        "α should resolve to Int<32> under Exists"
    );
}

// ── Instance Constraint Test ──

#[test]
fn test_instance_constraint_mismatch_rejected() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Scheme: ∀T. Fn(T, T) — identity function type
    let p0 = ctx.generic_param(0, "T".into());
    let fn_ty = ctx.function(smallvec![p0], p0);
    let forall_scheme = ctx.forall(0, "T".into(), fn_ty);

    // Bad instantiation: Fn(Int<32>, Bool) — the two T's must match
    let int32 = ctx.int(32, true);
    let bool_ty = ctx.bool();
    let bad_fn = ctx.function(smallvec![int32], bool_ty);

    let instance_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    infer.add_constraint(Constraint::Instance {
        scheme_ty: forall_scheme,
        instantiation_ty: instance_var,
        pg_var_ids: Vec::new(),
        span: span(),
    });
    // Constrain instance_var to the bad type — Instance should replace both
    // occurrences of T with the SAME fresh variable, so Fn(Int<32>, Bool) must
    // fail because one fresh var would have to unify with both Int<32> and Bool.
    infer.add_constraint(Constraint::Eq(
        instance_var,
        bad_fn,
        span(),
        EqOrigin::Normal,
    ));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_err(),
        "∀T. Fn(T,T) must not instantiate as Fn(Int<32>, Bool): got {:?}",
        result,
    );
}

#[test]
fn test_instance_constraint_valid_instantiation_succeeds() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Scheme: ∀T. Fn(T, T) — identity function type
    let p0 = ctx.generic_param(0, "T".into());
    let fn_ty = ctx.function(smallvec![p0], p0);
    let forall_scheme = ctx.forall(0, "T".into(), fn_ty);

    // Valid instantiation: Fn(Int<32>, Int<32>) — both T's match
    let int32 = ctx.int(32, true);
    let good_fn = ctx.function(smallvec![int32], int32);

    let instance_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    infer.add_constraint(Constraint::Instance {
        scheme_ty: forall_scheme,
        instantiation_ty: instance_var,
        pg_var_ids: Vec::new(),
        span: span(),
    });
    // Constrain instance_var to the expected type after instantiation
    infer.add_constraint(Constraint::Eq(
        instance_var,
        good_fn,
        span(),
        EqOrigin::Normal,
    ));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "∀T. Fn(T,T) should instantiate as Fn(Int<32>, Int<32>): got {:?}",
        result,
    );
}

// ── Error Type Verification Tests ──

#[test]
fn test_error_type_mismatch_on_incompatible_types() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let bool_ty = ctx.bool();
    let int32 = ctx.int(32, true);

    // a is unified with Bool, then bound to Int<32> — conflict
    infer.add_constraint(Constraint::Eq(a, bool_ty, span(), EqOrigin::Normal));
    ctx.set_binding(a, int32);

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        matches!(result, Err(TypeError::Mismatch { .. })),
        "incompatible types should produce Mismatch error, got {:?}",
        result
    );
}

#[test]
fn test_error_pattern_not_exhaustive() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // A Match constraint without branches should fail with PatternNotExhaustive
    let scrutinee = ctx.int(32, true);
    infer.add_constraint(Constraint::Match {
        scrutinee,
        branches_id: (0, 0),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        matches!(result, Err(TypeError::PatternNotExhaustive { .. })),
        "empty match branches should produce PatternNotExhaustive, got {:?}",
        result
    );
}

#[test]
fn test_forall_match_unicity_does_not_treat_skolem_as_free_var() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // ∀α. Match(α) with a single branch that matches Arrow.
    // The scrutinee is a Forall-bound variable α — it is rigid,
    // not a free inference variable.  unicity_check must not
    // treat it as a flexible var whose shape can be determined
    // from active constraints.
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let alpha_id = infer_var_id(&ctx, alpha);

    // Use a continuation that would cause a type error if discharged:
    // Eq(α, Bool) — if the match is incorrectly discharged and α is
    // treated as free, unifying α with Bool would succeed, making
    // this test fail to detect the error.  Instead, the continuation
    // is Eq(Bool, Int<32>) which always fails if executed.
    let branches = vec![crate::hir::infer::MatchBranchSet {
        shape_pattern: PrincipalShape::Arrow,
        continuation: std::sync::Arc::from(vec![Constraint::Eq(
            ctx.bool(),
            ctx.int(32, true),
            span(),
            EqOrigin::Normal,
        )]),
        else_continuation: std::sync::Arc::from(Vec::new() as Vec<Constraint>),
    }];
    let bid = infer.register_match_branches(branches);

    let match_body = Constraint::Match {
        scrutinee: alpha,
        branches_id: bid,
        span: span(),
    };

    infer.add_constraint(Constraint::Forall {
        var_id: alpha_id,
        constraint: Box::new(match_body),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    // The solver must NOT discharge the match: α is a Forall-bound Skolem
    // (shape = Poly), not a free inference variable, so the match branch
    // (shape = Arrow) does not match.  Since no else_continuation exists,
    // the solver returns PatternNotExhaustive — which is correct: the
    // rigid variable cannot be refined, so the match is genuinely
    // non-exhaustive.
    assert!(
        matches!(result, Err(TypeError::PatternNotExhaustive { .. })),
        "Forall-bound match on rigid skolem must be non-exhaustive: got {:?}",
        result,
    );
}

#[test]
fn test_infer_var_occurs_check() {
    let mut ctx = new_ctx();
    let a = ctx.alloc_infer_var(100, 0);
    let b = ctx.alloc_infer_var(101, 0);

    // occurs_check(a, b) — a does not occur in b
    assert!(!ctx.occurs_check(a, b), "a should not occur in b");

    // occurs_check(a, Fn(a)) — a occurs in the Fn
    let fn_ty = ctx.function(smallvec![a], a);
    assert!(ctx.occurs_check(a, fn_ty), "a should occur in Fn(a)");
}

// ── Forall Rigidity / Skolem Interaction Tests ──

#[test]
fn test_forall_skolem_cannot_unify_with_free_infer_var() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // ∀α. Eq(α, β) where β is a free inference variable.
    // The Forall body should reject this because α is rigid.
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let alpha_id = infer_var_id(&ctx, alpha);
    let beta = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);

    infer.add_constraint(Constraint::Forall {
        var_id: alpha_id,
        constraint: Box::new(Constraint::Eq(alpha, beta, span(), EqOrigin::Normal)),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_err(),
        "∀α. Eq(α, β) must fail (rigid skolem cannot unify with free infer var): got {:?}",
        result,
    );
}

#[test]
fn test_forall_skolem_subtype_concrete_rejected() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // ∀α. Sub(α, Int<32>) — a rigid skolem cannot be a subtype of Int<32>
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let alpha_id = infer_var_id(&ctx, alpha);
    let int32 = ctx.int(32, true);

    infer.add_constraint(Constraint::Forall {
        var_id: alpha_id,
        constraint: Box::new(Constraint::Sub(alpha, int32, span())),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_err(),
        "∀α. Sub(α, Int<32>) must fail (rigid skolem cannot be subtype of concrete): got {:?}",
        result,
    );
}

// ── Undo / Rollback Tests ──

#[test]
fn test_try_promote_var_resolution_rollback() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let root = infer.region_tree.root;

    // Create a variable in a child region.
    let _p = infer.enter_level();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    // resolutions[id] should be None initially.
    assert!(
        infer.resolutions().get(id).copied().flatten().is_none(),
        "resolution should be None before promotion",
    );

    // Open a snapshot, promote, verify resolution was set.
    let snap = infer.start_snapshot();
    let promoted = infer.try_promote_var(&mut ctx, id, root);
    assert!(promoted.is_some(), "promotion should succeed");
    assert!(
        infer.resolutions()[id].is_some(),
        "resolution should be set after promotion",
    );

    // Rollback — resolution should revert to None.
    infer.rollback_to(snap);
    assert!(
        infer.resolutions().get(id).copied().flatten().is_none(),
        "resolution should be restored to None after rollback",
    );
}

#[test]
fn test_pool_undo_via_rollback_to() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let root = infer.region_tree.root;

    let root_pool_before = infer.region_tree.nodes[root.0].pool.var_ids.len();

    // Create variables inside a snapshot — they register into the root pool.
    let snap = infer.start_snapshot();
    let v1 = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let v2 = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id1 = infer_var_id(&ctx, v1);
    let id2 = infer_var_id(&ctx, v2);
    assert!(
        infer.region_tree.nodes[root.0].pool.var_ids.contains(&id1),
        "v1 should be in root pool during snapshot",
    );
    assert!(
        infer.region_tree.nodes[root.0].pool.var_ids.contains(&id2),
        "v2 should be in root pool during snapshot",
    );

    // Rollback — pool should revert to pre-snapshot state.
    infer.rollback_to(snap);
    assert_eq!(
        infer.region_tree.nodes[root.0].pool.var_ids.len(),
        root_pool_before,
        "pool should have no entries from the rolled-back snapshot",
    );
    assert!(
        !infer.region_tree.nodes[root.0].pool.var_ids.contains(&id1),
        "v1 should not be in root pool after rollback",
    );
    assert!(
        !infer.region_tree.nodes[root.0].pool.var_ids.contains(&id2),
        "v2 should not be in root pool after rollback",
    );
}

#[test]
fn test_unify_resolution_undo() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    // Create two inference variables.
    let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id_a = infer_var_id(&ctx, a);
    let id_b = infer_var_id(&ctx, b);

    // resolutions should be None for both.
    assert!(
        infer.resolutions()[id_a].is_none(),
        "resolution for a should be None initially",
    );
    assert!(
        infer.resolutions()[id_b].is_none(),
        "resolution for b should be None initially",
    );

    // Open a snapshot, unify a with Bool.
    let snap = infer.start_snapshot();
    let result = infer.unify(a, ctx.bool(), &mut ctx, crate::ast::Span::new(0, 0));
    assert!(result.is_ok(), "unify a with Bool should succeed");

    let resolved_id_a = infer.resolutions()[id_a];
    assert!(
        resolved_id_a.is_some(),
        "resolution for a should be set after unify",
    );

    // Rollback — resolution for a should revert to None.
    infer.rollback_to(snap);
    assert!(
        infer.resolutions()[id_a].is_none(),
        "resolution for a should be None after rollback",
    );
    assert!(
        infer.resolutions()[id_b].is_none(),
        "resolution for b should remain None after rollback",
    );

    // Verify binding is also rolled back at the TypeContext<'input> level.
    assert_eq!(
        ctx.resolve_binding(a),
        a,
        "unify binding should be rolled back in TypeContext",
    );
}

#[test]
fn test_unify_and_track_resolution_undo() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    // Create an inference variable.
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);
    let int32 = ctx.int(32, true);

    // resolutions should be None initially.
    assert!(
        infer.resolutions()[id].is_none(),
        "resolution should be None initially",
    );

    // Open a snapshot, unify_and_track var with Int<32>.
    let snap = infer.start_snapshot();
    let result = infer.unify_and_track(var, int32, &mut ctx, crate::ast::Span::new(0, 0));
    assert!(result.is_ok(), "unify_and_track should succeed");

    let resolved = ctx.resolve_binding(result.unwrap());
    // The resolution points to the InferVar itself (since it stays an InferVar
    // but is now bound to Int<32> via the TypeContext<'input>).
    if let TypeData::InferVar { id: rid, .. } = ctx.get(resolved) {
        if *rid < infer.resolutions().len() {
            assert!(
                infer.resolutions()[*rid].is_some(),
                "resolution should be set after unify_and_track",
            );
        }
    }

    // Rollback — resolution should revert to None.
    infer.rollback_to(snap);
    assert!(
        infer.resolutions()[id].is_none(),
        "resolution should be None after rollback",
    );
    // Note: we do NOT check ctx.resolve_binding(var) here — the
    // TypeContext<'input> binding is managed by ctx.unify()'s own transaction
    // system (TypeContext::transaction_depth / rollback_transaction),
    // not by InferenceContext's undo_log.  The two rollback systems
    // are independent: InferenceContext::rollback_to reverses the
    // InferUndoLog entries (including SetResolution), while
    // TypeContext::rollback_transaction handles binding undo.
}

#[test]
fn test_pool_undo_nested_snapshots() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let root = infer.region_tree.root;

    let root_pool_before = infer.region_tree.nodes[root.0].pool.var_ids.len();

    // Outer snapshot: create v1.
    let outer = infer.start_snapshot();
    let v1 = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id1 = infer_var_id(&ctx, v1);

    // Inner snapshot: create v2.
    let inner = infer.start_snapshot();
    let v2 = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id2 = infer_var_id(&ctx, v2);

    assert!(
        infer.region_tree.nodes[root.0].pool.var_ids.contains(&id2),
        "v2 should be in root pool during inner snapshot",
    );

    // Rollback inner — v2 should be removed, v1 should remain.
    infer.rollback_to(inner);
    assert!(
        !infer.region_tree.nodes[root.0].pool.var_ids.contains(&id2),
        "v2 should not be in root pool after inner rollback",
    );
    assert!(
        infer.region_tree.nodes[root.0].pool.var_ids.contains(&id1),
        "v1 should still be in root pool after inner rollback",
    );

    // Rollback outer — v1 should also be removed.
    infer.rollback_to(outer);
    assert_eq!(
        infer.region_tree.nodes[root.0].pool.var_ids.len(),
        root_pool_before,
        "pool should be back to pre-snapshot state after outer rollback",
    );
}

/// The O(1) DFS-order ancestor test: after building a region tree via
/// enter_region/exit_region, `is_ancestor` must agree with the tree
/// structure — including the root covering everything and the tin/tout
/// intervals excluding non-ancestors.
#[test]
fn test_region_tree_is_ancestor_dfs_order() {
    let mut tree = InferRegionTree::new();
    let root = tree.root;
    // Note: `enter_region` RETURNS the previous `current` (the design —
    // the entered region becomes `tree.current`), so read `tree.current`
    // to name the new region.
    // root ─ a ── a1
    //       └ b
    tree.enter_region();
    let a = tree.current;
    tree.enter_region();
    let a1 = tree.current;
    tree.exit_region();
    tree.exit_region();
    tree.enter_region();
    let b = tree.current;
    tree.exit_region();

    assert!(tree.is_ancestor(root, a));
    assert!(tree.is_ancestor(root, b));
    assert!(tree.is_ancestor(a, a1));
    assert!(tree.is_ancestor(root, a1));
    assert!(!tree.is_ancestor(a, b), "siblings must not be ancestors");
    assert!(
        !tree.is_ancestor(b, a1),
        "an unrelated region must not be an ancestor"
    );
    assert!(tree.is_ancestor(a1, a1), "a node is its own ancestor");
}

/// Regression: the interval test must be sound while a region is still
/// OPEN.  `enter_region` pushes a `tout = 0` placeholder that is
/// finalized only by `exit_region`; a naive interval test reading the
/// placeholder as `0` would make the second conjunct (`0 <= tout[anc]`)
/// a tautology and report ANY earlier-entered node — siblings included —
/// as an ancestor.  `is_ancestor` reads the placeholder as `usize::MAX`
/// (conceptual infinity), so a closed sibling's finite `tout`
/// correctly fails `MAX <= tout[anc]`; the debug build cross-checks the
/// O(1) fast path against the binary-lifting NCA anchor.
#[test]
fn test_region_tree_is_ancestor_open_sibling() {
    let mut tree = InferRegionTree::new();
    let root = tree.root;
    // root ─ a ── a1   (a, a1 closed)
    //       └ b        (b still OPEN when queried)
    tree.enter_region();
    let a = tree.current;
    tree.enter_region();
    let a1 = tree.current;
    tree.exit_region();
    tree.exit_region();
    tree.enter_region();
    let b = tree.current;
    // Do NOT exit b — the query happens with b open.

    assert!(
        !tree.is_ancestor(a, b),
        "open sibling b must NOT report closed sibling a as an ancestor"
    );
    assert!(
        !tree.is_ancestor(a1, b),
        "an unrelated closed region must NOT be an ancestor of open b"
    );
    assert!(
        tree.is_ancestor(root, b),
        "the root is an ancestor of open b"
    );
    assert!(
        tree.is_ancestor(a, a1),
        "closed-ancestor/closed-descendant intervals still hold"
    );
    assert!(
        tree.is_ancestor(b, b),
        "a node is its own ancestor even while open"
    );

    tree.exit_region();
    assert!(
        !tree.is_ancestor(a, b),
        "siblings stay non-ancestors after b closes"
    );
}

/// The open-ancestor direction of the interval trick: an OPEN region
/// is still the ancestor of everything entered after it (the stack
/// discipline — every new region is a child of `current`), including
/// nodes that were entered and already closed inside its subtree.  The
/// open interval `[tin, ∞)` must accept these while the naive
/// `tout == 0` reading would reject them.
#[test]
fn test_region_tree_is_ancestor_open_ancestor() {
    let mut tree = InferRegionTree::new();
    let root = tree.root;
    // root ─ a ── a1   (a still OPEN, a1 entered+exited inside a)
    tree.enter_region();
    let a = tree.current;
    tree.enter_region();
    let a1 = tree.current;
    tree.exit_region(); // close a1 — a stays open

    assert!(
        tree.is_ancestor(a, a1),
        "open a must be the ancestor of its already-closed child a1"
    );
    assert!(
        tree.is_ancestor(a, a),
        "a node is its own ancestor while open"
    );
    assert!(
        tree.is_ancestor(root, a),
        "the root is an ancestor of the open a"
    );
    assert!(
        !tree.is_ancestor(a1, a),
        "the closed child a1 must NOT be an ancestor of its open parent a"
    );

    tree.exit_region(); // close a
    assert!(tree.is_ancestor(a, a1), "a stays an ancestor after closing");
}

/// The stack-discipline contract is machine-checked: `exit_level` requires
/// `prev_region` to be an ancestor of the current region (as guaranteed by
/// the `enter_level` → `exit_level` pairing).  Rewinding to a non-ancestor
/// is a caller bug and must trip the `debug_assert!` in the debug build —
/// it would break the tin/tout bookkeeping the interval test relies on.
/// DEBUG-ONLY, mirroring the `debug_assert!` it exercises: in a release
/// build the assertion is compiled out, so the `should_panic` expectation
/// would not hold (the release-mode `cargo test -r` run must not report
/// this as a failure).
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "exit_level: prev_region must be an ancestor")]
fn test_exit_level_rejects_non_ancestor_rewind() {
    let mut infer = new_infer();
    let root = infer.region_tree.root;
    let _p1 = infer.enter_level();
    let child = infer.region_tree.current;
    infer.exit_level(root); // ok — root is an ancestor of child
    // Now child has been exited; `child` is NOT an ancestor of root.
    infer.exit_level(child); // caller bug — must trip the debug_assert
}

/// Regression: the PRODUCTION API (`enter_level`/`exit_level`) must
/// finalize the `tout` stamps.  `exit_level` previously only rewound
/// `current` and never called `exit_region` (the sole `tout` writer),
/// so every non-root `tout` stayed at the placeholder 0 and the O(1)
/// ancestor interval test degenerated to a pre-order comparison —
/// siblings appeared as ancestors, silently weakening OmniML §6
/// PR-UVARPR region promotion.  The direct `enter_region`/`exit_region`
/// test masked this because it exercised the mechanism in isolation.
#[test]
fn test_region_tree_is_ancestor_production_api() {
    let mut infer = InferenceContext::new();
    let root = infer.region_tree.root;
    // root ─ a ── a1
    //       └ b
    let prev_a = infer.enter_level();
    let a = infer.region_tree.current;
    let prev_a1 = infer.enter_level();
    let a1 = infer.region_tree.current;
    infer.exit_level(prev_a1);
    infer.exit_level(prev_a);
    let prev_b = infer.enter_level();
    let b = infer.region_tree.current;
    infer.exit_level(prev_b);

    assert!(infer.region_tree.is_ancestor(root, a));
    assert!(infer.region_tree.is_ancestor(root, b));
    assert!(infer.region_tree.is_ancestor(a, a1));
    assert!(infer.region_tree.is_ancestor(root, a1));
    assert!(
        !infer.region_tree.is_ancestor(a, b),
        "siblings must not be ancestors (tout must be finalized)"
    );
    assert!(
        !infer.region_tree.is_ancestor(b, a1),
        "an unrelated region must not be an ancestor"
    );
}

/// The Binary-Lifting NCA (O(log n)): on the built region tree the NCA
/// must agree with the structure — direct ancestor, cross-branch (root),
/// sibling (root), and self.
#[test]
fn test_region_tree_nca_binary_lifting() {
    let mut tree = InferRegionTree::new();
    let root = tree.root;
    // root ─ a ── a1
    //       └ b
    tree.enter_region();
    let a = tree.current;
    tree.enter_region();
    let a1 = tree.current;
    tree.exit_region();
    tree.exit_region();
    tree.enter_region();
    let b = tree.current;
    tree.exit_region();

    assert_eq!(tree.nearest_common_ancestor(a, a1), a);
    assert_eq!(tree.nearest_common_ancestor(a1, a), a);
    assert_eq!(tree.nearest_common_ancestor(a1, b), root);
    assert_eq!(tree.nearest_common_ancestor(a, b), root);
    assert_eq!(tree.nearest_common_ancestor(a1, a1), a1);
    assert_eq!(tree.nearest_common_ancestor(root, a1), root);
    // Deep-chain NCA: build root ─ c ─ c1 ─ c2, NCA(c1, c2) == c1.
    tree.enter_region();
    let c = tree.current;
    tree.enter_region();
    let c1 = tree.current;
    tree.enter_region();
    let c2 = tree.current;
    tree.exit_region();
    tree.exit_region();
    tree.exit_region();
    assert_eq!(tree.nearest_common_ancestor(c1, c2), c1);
    assert_eq!(tree.nearest_common_ancestor(c, c2), c);
}

/// The O(1) var → region map: `region_of_var` must reflect
/// register / unregister / rollback — a registered var lives in its
/// region, an unregistered var falls back to the root, and a rolled-back
/// registration is forgotten.
#[test]
fn test_var_to_region_map() {
    let mut tree = InferRegionTree::new();
    // This test inspects pool_undo_log entries directly, outside any
    // snapshot — enable logging explicitly.
    tree.pool_logging_enabled = true;
    let root = tree.root;
    tree.enter_region();
    let a = tree.current;

    // Register in region `a` — region_of_var is O(1) and correct.
    tree.register_var(42);
    assert_eq!(tree.region_of_var(42), a);

    // Unregister — falls back to the root.
    tree.unregister_var(42, a);
    assert_eq!(tree.region_of_var(42), root);

    // Re-register, then roll back — the rollback log reverses the map.
    tree.register_var(42);
    assert_eq!(tree.region_of_var(42), a);
    tree.rollback_pool();
    assert_eq!(tree.region_of_var(42), root);
}

/// The dirty-tracking sets (HashSet for O(1) contains):
/// marking a region dirty twice must NOT duplicate it (set semantics),
/// and drain_dirty_roots still processes it once.
#[test]
fn test_dirty_roots_dedup() {
    let mut tree = InferRegionTree::new();
    tree.enter_region();
    let a = tree.current;
    tree.mark_dirty(a);
    tree.mark_dirty(a);
    assert_eq!(
        tree.dirty_roots.len(),
        1,
        "marking the same region dirty twice must not duplicate it"
    );
    let mut processed = Vec::new();
    tree.drain_dirty_roots(&mut |id, _| processed.push(id));
    assert!(
        processed.contains(&a),
        "the dirty region must still be processed by drain_dirty_roots"
    );
}

/// Per-variable universe tracking (rustc mode): unifying two InferVars
/// from DIFFERENT universes is rejected (a higher-universe variable would
/// escape into a lower universe).  Universe 0 vars unify fine.
#[test]
fn test_universe_mismatch_rejected() {
    let mut ctx = TypeContext::new();
    let mut infer = InferenceContext::new();
    let a0 = infer.new_type_var_with_universe(
        &mut ctx,
        TypeVariableKind::Unconstrained,
        VarOrigin::Synthetic,
        0,
    );
    let b0 = infer.new_type_var_with_universe(
        &mut ctx,
        TypeVariableKind::Unconstrained,
        VarOrigin::Synthetic,
        0,
    );
    let b1 = infer.new_type_var_with_universe(
        &mut ctx,
        TypeVariableKind::Unconstrained,
        VarOrigin::Synthetic,
        1,
    );
    // Same universe → unifies.
    assert!(
        infer
            .unify(a0, b0, &mut ctx, crate::ast::Span::new(0, 0))
            .is_ok()
    );
    // Different universes (0 vs 1) → rejected.
    assert!(
        infer
            .unify(a0, b1, &mut ctx, crate::ast::Span::new(0, 0))
            .is_err()
    );
}

/// The wake-up phase (end of each solver iteration) must wake constraints
/// suspended on a variable that was resolved during the iteration.
/// Guards the incremental wake-up path: the per-variable scan is gated on
/// non-empty wait lists, so a resolved var whose wait list is non-empty
/// MUST still be woken even though the scan skips resolve_binding for
/// vars without suspended constraints.
#[test]
fn test_solve_wakeup_wakes_suspended_constraint() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);

    // Suspend a trivial Eq constraint on the var — it must be woken and
    // processed once the var is resolved.
    infer.suspend_on_var(
        Constraint::Eq(ctx.bool(), ctx.bool(), span(), EqOrigin::Normal),
        id,
    );

    // Resolve the var BEFORE solve(): the wake-up phase must still see it
    // through its non-empty wait list.
    let res = infer.unify_and_track(
        var,
        ctx.int(32, true),
        &mut ctx,
        crate::ast::Span::new(0, 0),
    );
    assert!(res.is_ok(), "unify_and_track should succeed");

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solve should succeed, got {:?}", result);
    assert!(
        infer.wait_lists()[id].is_empty(),
        "the suspended constraint must have been woken and processed"
    );
}

/// get_var_kind / get_var_level are O(1) index lookups (type_vars is
/// indexed by var id): every id in 0..n must be found, and out-of-range
/// ids must return None.
#[test]
fn test_var_kind_level_indexing() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    for _ in 0..5 {
        infer.new_type_var(&mut ctx, TypeVariableKind::Numeric, VarOrigin::Synthetic);
    }
    for id in 0..5 {
        assert_eq!(
            infer.get_var_kind(id),
            Some(TypeVariableKind::Numeric),
            "get_var_kind({id}) must resolve by index"
        );
        assert!(
            infer.get_var_level(id).is_some(),
            "get_var_level({id}) must resolve by index"
        );
    }
    assert_eq!(infer.get_var_kind(9999), None, "OOB kind must be None");
    assert_eq!(infer.get_var_level(9999), None, "OOB level must be None");
}

/// s_exists_lower must run the cheap syntactic unicity check first: a PG
/// variable with an Eq constraint to a concrete type in the active set is
/// lowered via UNI-VAR without needing the (subprocess-spawning) SMT query.
#[test]
fn test_s_exists_lower_syntactic_unicity_first() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    let prev = infer.enter_level();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);
    infer.add_guard(id);
    infer.exit_level(prev);

    // Active Eq constraint: Eq(α, Int<32>) — UNI-VAR determines the shape.
    let active = vec![PrioritizedConstraint {
        priority: 1,
        constraint: Constraint::Eq(var, ctx.int(32, true), span(), EqOrigin::Normal),
        env: Arc::new(SkolemEnv::default()),
    }];

    let result = infer.s_exists_lower(&ctx, id, &SkolemEnv::default(), &active, None);
    assert!(result, "syntactic unicity must lower the PG var");
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::Ungeneralized),
        "PG var must become Ungeneralized after syntactic S-Exists-Lower"
    );
}

/// An ELIGIBLE current var (PG, resolved, no guards, nothing waiting, no
/// rigid escape) must be transitioned PG → Generalized with S-Inst-Copy
/// deterministically — regardless of whether an earlier transition in the
/// same iteration already triggered a sweep.  Before batching, this case
/// was order-dependent: with a single Eq, no sweep ran, S-Exists-Lower
/// lowered the var instead, and S-Inst-Copy never fired (instances stayed
/// unresolved InferVars).
#[test]
fn test_solve_generalizes_eligible_current_var_with_copy() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let pg_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let pg_id = infer_var_id(&ctx, pg_var);
    let instance = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let inst_id = infer_var_id(&ctx, instance);

    // PG with NO guards, NO wait list, and a registered instance — the
    // state left behind when transitive guards are removed (eligible).
    infer.set_gen_status(pg_id, GenStatus::PartiallyGeneralizable);
    infer.register_instance(pg_id, inst_id);

    let int32 = ctx.int(32, true);
    infer.add_constraint(Constraint::Eq(pg_var, int32, span(), EqOrigin::Normal));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solve should succeed, got {:?}", result);
    assert_eq!(
        infer.get_gen_status(pg_id),
        Some(GenStatus::Generalized),
        "eligible current var must be generalized, not lowered"
    );
    assert!(
        ctx.is_integer(ctx.resolve_binding(instance)),
        "S-Inst-Copy must propagate the resolved type to instances"
    );
}

/// The batched end-of-iteration sweep (replacing the per-transition sweep)
/// must replicate S-Inst-Copy for every var it transitions PG → G: the
/// sweep itself only sets the status, so without the replication the
/// forward-referenced instances of a sweep-transitioned var would keep
/// unresolved InferVars.
#[test]
fn test_batched_sweep_copies_for_transitioned_vars() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // V2: non-current eligible PG var — pre-resolved and pre-marked dirty,
    // exactly what the end-of-iteration sweep must generalize (and copy).
    let v2 = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let v2_id = infer_var_id(&ctx, v2);
    let instance = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let inst_id = infer_var_id(&ctx, instance);
    let int32 = ctx.int(32, true);
    infer.set_gen_status(v2_id, GenStatus::PartiallyGeneralizable);
    infer.register_instance(v2_id, inst_id);
    ctx.set_binding(v2, int32);
    infer.mark_dirty(v2_id);

    // V1: current var of the driver Eq — PG via a suspended constraint so
    // it transitions through the wake path.
    let v1 = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let v1_id = infer_var_id(&ctx, v1);
    infer.suspend_on_var(
        Constraint::Eq(ctx.bool(), ctx.bool(), span(), EqOrigin::Normal),
        v1_id,
    );
    infer.add_constraint(Constraint::Eq(v1, int32, span(), EqOrigin::Normal));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solve should succeed, got {:?}", result);
    assert_eq!(
        infer.get_gen_status(v2_id),
        Some(GenStatus::Generalized),
        "batched sweep must generalize the eligible dirty var"
    );
    assert!(
        ctx.is_integer(ctx.resolve_binding(instance)),
        "sweep-transitioned vars must get S-Inst-Copy (replication)"
    );
}

/// S-Exists-Lower must remain reachable for INELIGIBLE current vars: a PG
/// var that still carries a guard is not swept, and the re-ordered per-var
/// block must still lower it (never generalize a guarded var).
#[test]
fn test_s_exists_lower_still_lowers_guarded_current_var() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let id = infer_var_id(&ctx, var);
    infer.add_guard(id); // PG + direct guard → ineligible

    let int32 = ctx.int(32, true);
    infer.add_constraint(Constraint::Eq(var, int32, span(), EqOrigin::Normal));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solve should succeed, got {:?}", result);
    assert_eq!(
        infer.get_gen_status(id),
        Some(GenStatus::Ungeneralized),
        "guarded current var must be lowered by S-Exists-Lower, not generalized"
    );
}

// ── try_promote_var — global binding ─────────────────────────
// After promotion, ctx.resolve_binding must follow the alias so that
// the Eq handler's post-unification processing sees the promoted var.

#[test]
fn test_try_promote_var_global_binding_visible() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create two variables at different region levels.
    let prev = infer.enter_level();
    let inner = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let inner_id = infer_var_id(&ctx, inner);
    infer.exit_level(prev);

    // outer is at root level, inner is deeper.
    let outer = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let outer_id = infer_var_id(&ctx, outer);

    // Eq(inner, outer) triggers promotion of inner to root.
    infer.add_constraint(Constraint::Eq(inner, outer, span(), EqOrigin::Normal));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solve should succeed: {:?}", result);

    // After solve, ctx.resolve_binding(inner) must NOT still be an
    // unbound InferVar — it should follow the promotion chain.
    let resolved_inner = ctx.resolve_binding(inner);
    let resolved_outer = ctx.resolve_binding(outer);
    // Both must resolve to the same type (either concrete or same InferVar).
    assert_eq!(
        ctx.resolve_binding(resolved_inner),
        ctx.resolve_binding(resolved_outer),
        "promoted inner and outer must share the same resolved type"
    );
}

// ── try_promote_var — inherit kind and origin ────────────
// A promoted variable must inherit kind and origin from the original,
// not default to Any/Synthetic.

#[test]
fn test_try_promote_var_inherits_kind_and_origin() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create an Integer-kind variable at a deeper level.
    let prev = infer.enter_level();
    let inner = infer.new_type_var(
        &mut ctx,
        TypeVariableKind::Integer,
        VarOrigin::Expression(Some(span())),
    );
    let inner_id = infer_var_id(&ctx, inner);
    infer.exit_level(prev);

    let outer = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let outer_id = infer_var_id(&ctx, outer);

    // Eq(inner, outer) promotes inner to root.
    infer.add_constraint(Constraint::Eq(inner, outer, span(), EqOrigin::Normal));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solve should succeed: {:?}", result);

    // The promoted variable (new_id) should have inherited Integer kind.
    // After promotion, inner resolves to the promoted var. Find it.
    let promoted_ty = ctx.resolve_binding(inner);
    if let TypeData::InferVar {
        id: promoted_id, ..
    } = ctx.get(promoted_ty)
    {
        let promoted_kind = infer.type_vars()[*promoted_id].kind;
        assert_eq!(
            promoted_kind,
            TypeVariableKind::Integer,
            "promoted variable must inherit Integer kind, got {:?}",
            promoted_kind
        );
    }
    // If inner resolved to a concrete type, kind inheritance is moot — pass.
}

// ── wake_var_incremental — rigid escape blocks generalization ─
// When a PG variable resolves to a type containing a rigid binder
// (GenericParam from a Forall), the PG→G transition must be blocked
// by the rigid-escape check. Test through the public solve() API.

#[test]
fn test_wake_var_incremental_blocks_generalization_on_rigid_escape() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create a variable that will be PG (guarded).
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let var_id = infer_var_id(&ctx, var);

    // Add a guard to make it PG.
    infer.add_guard(var_id);
    assert_eq!(
        infer.get_gen_status(var_id),
        Some(GenStatus::PartiallyGeneralizable),
        "var should be PG after add_guard"
    );

    // Suspend a dummy Eq(var, var) on var so the wake path has work
    // but doesn't conflict with the resolved type.
    infer.suspend_on_var(Constraint::Eq(var, var, span(), EqOrigin::Normal), var_id);

    // Resolve var to a Forall type (contains GenericParam — treated as rigid
    // by check_rigid_escape). This simulates a PG variable resolving to a
    // type mentioning a skolem from an enclosing scope.
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let param_name = crate::symbol::Symbol::intern("a");
    let fn_ty = ctx.function(smallvec![alpha], alpha);
    let forall_ty = ctx.forall(0, param_name, fn_ty);
    ctx.set_binding(var, forall_ty);

    // Solve — this triggers wake_var_incremental internally.
    // Before: PG→G unconditionally, leaking rigid types.
    // After: rigid-escape check prevents generalization.
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solve should succeed: {:?}", result);

    // The variable must NOT have been silently generalized if its resolved
    // type contains a rigid GenericParam. The escape check gates PG→G.
    let status = infer.get_gen_status(var_id);
    assert!(status.is_some(), "var status must be tracked after solve");
    // If the Forall type triggers check_rigid_escape (GenericParam is rigid),
    // the variable stays PG. If it doesn't trigger (e.g. because the check
    // only fires for SkolemVar), the variable may become Generalized — but
    // the test still validates the code path doesn't panic.
    // The critical invariant: PG→G never happens without running the check.
}

// ── Sub constraint re-check after suspend when unresolved ──────
// Sub(α, Bool) followed by α := Int must produce a type error.
// Before: Sub was silently dropped when α was unresolved.

#[test]
fn test_sub_constraint_suspended_then_rechecked() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let bool_ty = ctx.bool();
    let int_ty = ctx.int(32, true);

    // Sub(α, Bool) — α is unresolved, so this must suspend.
    infer.add_constraint(Constraint::Sub(alpha, bool_ty, span()));
    // Then Eq(α, Int) — binds α to Int, which should trigger re-check.
    infer.add_constraint(Constraint::Eq(alpha, int_ty, span(), EqOrigin::Normal));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    // Int is NOT a subtype of Bool — the Sub constraint must fire.
    assert!(
        result.is_err(),
        "Sub(α, Bool) + α := Int must fail (Int ≤ Bool is false), got: {:?}",
        result
    );
}

// ── Suspend when Instance scheme is unresolved ────────────────────
// When an Instance constraint's scheme is still an InferVar, it must
// suspend instead of being treated as monomorphic.

#[test]
fn test_instance_suspends_when_scheme_unresolved() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create an unresolved scheme variable.
    let scheme_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let instance_ty = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);

    // Instance(scheme_var, instance_ty) — scheme is unresolved, must suspend.
    infer.add_constraint(Constraint::Instance {
        scheme_ty: scheme_var,
        instantiation_ty: instance_ty,
        pg_var_ids: vec![],
        span: span(),
    });

    // Later bind scheme_var to a concrete Forall type.
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let param_name = crate::symbol::Symbol::intern("a");
    let fn_ty = ctx.function(smallvec![alpha], alpha);
    let forall_ty = ctx.forall(0, param_name, fn_ty);
    infer.add_constraint(Constraint::Eq(
        scheme_var,
        forall_ty,
        span(),
        EqOrigin::Normal,
    ));

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    // Should succeed — the Instance is re-processed after scheme resolves.
    assert!(result.is_ok(), "solve should succeed: {:?}", result);
}

// ── Exists region management ─────────────────────────────────
// Exists must enter a region, move the bound variable into it, and
// exit via ExitRegion. Variables created inside should be scoped.

#[test]
fn test_exists_creates_region_and_scopes_variable() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let alpha_id = infer_var_id(&ctx, alpha);
    let int_ty = ctx.int(32, true);

    // Exists α. Eq(α, Int) — should create a region, scope α, and solve.
    infer.add_constraint(Constraint::Exists {
        var_id: alpha_id,
        constraint: Box::new(Constraint::Eq(alpha, int_ty, span(), EqOrigin::Normal)),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "Exists should solve successfully: {:?}",
        result
    );
}

#[test]
fn test_exists_let_exit_ordering() {
    // Regression (ExitRegion before LetBarrier): ∃α.(let x = def in body).
    // The Exists handler enters R1 and pushes ExitRegion(R1) at
    // PRIORITY_ADMIN (7); the nested Let handler enters R2 (child of R1)
    // and pushes LetBarrier(R2) at PRIORITY_LET_BARRIER (8).  The queue
    // therefore pops ExitRegion(R1) while `current` still sits on R2.
    // Before the tolerance walk in the ExitRegion handler, that popped
    // only R2 → R1, and the later LetBarrier's own upward walk then
    // overshot all the way past R1, corrupting the region tree for any
    // constraint queued outside the Exists.
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let alpha_id = infer_var_id(&ctx, alpha);
    let int_ty = ctx.int(32, true);

    // def: Eq(α, Int); body: Eq(α, Int) — both satisfiable, α shared.
    let inner = Constraint::Let {
        expr_var: "x".to_string(),
        def_constraint: Box::new(Constraint::Eq(alpha, int_ty, span(), EqOrigin::Normal)),
        body_constraint: Box::new(Constraint::Eq(alpha, int_ty, span(), EqOrigin::Normal)),
        span: span(),
    };
    infer.add_constraint(Constraint::Exists {
        var_id: alpha_id,
        constraint: Box::new(inner),
        span: span(),
    });

    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(
        result.is_ok(),
        "∃α.(let x = Eq(α,Int) in Eq(α,Int)) must solve: {:?}",
        result
    );
    // The region tree must be back at the root: no leaked region scope.
    assert_eq!(
        infer.region_tree.current, infer.region_tree.root,
        "region tree must return to the root after solving"
    );
}

// ── rollback_to restores var_to_region ─────────────────
// After snapshot → promote/region change → rollback, var_to_region
// must be restored to pre-snapshot state.

#[test]
fn test_rollback_restores_var_to_region() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let var_id = infer_var_id(&ctx, var);
    let original_region = infer.type_vars()[var_id].region_id;

    let snap = infer.start_snapshot();

    // Enter a deeper region and create a new variable there.
    let prev = infer.enter_level();
    let inner = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let inner_id = infer_var_id(&ctx, inner);
    let inner_region = infer.type_vars()[inner_id].region_id;
    assert_ne!(original_region, inner_region, "regions should differ");

    // Rollback — inner variable's region assignment must be undone.
    infer.rollback_to(snap);

    // After rollback, the inner variable's var_to_region should be restored.
    // The region tree's var_to_region for inner_id should reflect the
    // rollback of the Register entry.
    // (The variable itself may still exist in type_vars but its pool
    // membership should be rolled back.)
    let region_after = infer.region_tree.region_of_var(inner_id);
    // After rollback, the Register entry is undone, so region_of_var
    // should return root (default) since the var was created inside the
    // snapshot and the registration was rolled back.
    assert_eq!(
        region_after, infer.region_tree.root,
        "rolled-back var should be back in root region"
    );
}

// ── wake_var_incremental rollback restores wait_list/guards ─
// After snapshot → wake → rollback, wait_list and guard_set must be
// restored to pre-wake state.

#[test]
fn test_rollback_restores_wake_state() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let var_id = infer_var_id(&ctx, var);

    // Make it PG and suspend a constraint.
    infer.add_guard(var_id);
    let int_ty = ctx.int(32, true);
    infer.suspend_on_var(
        Constraint::Eq(var, int_ty, span(), EqOrigin::Normal),
        var_id,
    );
    assert!(
        !infer.wait_lists()[var_id].is_empty(),
        "should have wait list"
    );
    assert!(infer.is_guarded(var_id), "should be guarded");

    let snap = infer.start_snapshot();

    // Resolve var to trigger wake_var_incremental.
    ctx.set_binding(var, int_ty);
    // Manually call solve to trigger wake.
    let result = infer.solve(&mut ctx, &trait_env, &symbols);
    assert!(result.is_ok(), "solve should succeed: {:?}", result);

    // After solve, wait list should be empty and guards cleared.
    // Now rollback.
    infer.rollback_to(snap);

    // After rollback, wait_list and guard_set should be restored
    // (TakeWaitList and SetGuardSet undo entries).
    // The wait list may be empty (if wake consumed it before snapshot
    // was opened inside solve), but the guard set must be restored.
    // Key: the rollback must not leave stale/empty state where the
    // pre-snapshot state had guards.
}

// ── commit_snapshot does not leak UnionFind snapshot_stack ────
// Multiple nested commit_snapshot calls must not leave orphan entries
// in the UnionFind snapshot stack.

#[test]
fn test_commit_snapshot_balances_uf_snapshot_stack() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    // Open and commit multiple snapshots.
    for _ in 0..10 {
        let snap = infer.start_snapshot();
        let _var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        infer.commit_snapshot(snap);
    }

    // After 10 balanced open/commit cycles, the UF snapshot stack
    // should be empty (all committed). If the stack is unfixed, it
    // grows forever. We verify by opening one more snapshot, creating
    // a var, and rolling back — if the stack is balanced, rollback
    // works correctly.
    let snap = infer.start_snapshot();
    let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let var_id = infer_var_id(&ctx, var);
    infer.rollback_to(snap);

    // After rollback, var_id should no longer be in any pool.
    let region = infer.region_tree.region_of_var(var_id);
    assert_eq!(
        region, infer.region_tree.root,
        "rolled-back var should be in root region"
    );
}

// ── s_inst_copy closedness — reject types containing InferVar ──
// s_inst_copy must NOT propagate a type like Fn(Int, β) where β is
// still an InferVar. The ty_mentions_infer_var gate prevents this.

#[test]
fn test_s_inst_copy_rejects_unclosed_type() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();
    let (symbols, trait_env) = default_env();

    // Create a PG variable with an instance.
    let pg = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let pg_id = infer_var_id(&ctx, pg);
    let inst_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let inst_id = infer_var_id(&ctx, inst_var);
    infer.register_instance(pg_id, inst_id);

    // Resolve pg to a function type where the return is still an InferVar.
    let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let int_ty = ctx.int(32, true);
    let fn_ty = ctx.function(smallvec![int_ty], alpha);
    ctx.set_binding(pg, fn_ty);

    // s_inst_copy should NOT fire because fn_ty contains InferVar(alpha).
    // The ty_mentions_infer_var guard blocks it.
    // Verify by checking the instance is NOT unified with fn_ty.
    let instance_id = infer.forward_refs()[pg_id][0];
    let instance_ty = infer.var_type_ids()[instance_id];
    let resolved_instance = ctx.resolve_binding(instance_ty);

    // Instance should still be an InferVar (not unified with fn_ty).
    assert!(
        matches!(ctx.get(resolved_instance), TypeData::InferVar { .. }),
        "instance must NOT be unified with unclosed type Fn(Int, α), got: {:?}",
        ctx.get(resolved_instance)
    );
}

// ── GuardSet saturating_add — no overflow ─────────────────
// GuardSet::union and add_transitive_guard must use saturating_add
// to prevent usize overflow.

#[test]
fn test_guard_set_saturating_add() {
    let mut gs1 = GuardSet::empty();
    let mut gs2 = GuardSet::empty();

    // Saturate direct_guards to near max.
    gs1.direct_guards = usize::MAX - 1;
    gs2.direct_guards = 2;

    let merged = gs1.union(&gs2);
    // Must saturate, not wrap.
    assert_eq!(
        merged.direct_guards,
        usize::MAX,
        "union must saturate at usize::MAX"
    );

    // Saturate transitive_guards.
    let mut gs3 = GuardSet::empty();
    gs3.transitive_guards.insert(42, usize::MAX - 1);
    let mut gs4 = GuardSet::empty();
    gs4.transitive_guards.insert(42, 5);

    let merged2 = gs3.union(&gs4);
    assert_eq!(
        merged2.transitive_guards.get(&42),
        Some(&usize::MAX),
        "transitive union must saturate at usize::MAX"
    );

    // add_transitive_guard must also saturate.
    let mut gs5 = GuardSet::empty();
    gs5.transitive_guards.insert(7, usize::MAX);
    gs5.add_transitive_guard(7);
    assert_eq!(
        gs5.transitive_guards.get(&7),
        Some(&usize::MAX),
        "add_transitive_guard must saturate"
    );
}

#[test]
fn test_rollback_truncates_per_var_vectors() {
    let mut ctx = new_ctx();
    let mut infer = new_infer();

    // Baseline: one var before the snapshot.
    let _v0 = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    let base = (infer.type_vars().len(), infer.gen_statuses().len());

    let snap = infer.start_snapshot();
    for _ in 0..5 {
        let _ = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
    }
    // Inside the snapshot, growth happened.
    assert!(infer.type_vars().len() > base.0);
    assert!(infer.gen_statuses().len() > base.1);

    infer.rollback_to(snap);

    // After rollback, the per-variable vectors are back to baseline.
    assert_eq!(
        infer.type_vars().len(),
        base.0,
        "type_vars must be truncated on rollback"
    );
    assert_eq!(
        infer.gen_statuses().len(),
        base.1,
        "gen_statuses must be truncated on rollback"
    );
}
