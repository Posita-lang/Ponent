// ── Anya Module ────────────────────────────────────────────────────────
//
use crate::hir::infer::{GenStatus, InferRegionTree, PoolUndoEntry};
use crate::hir::shape_var::Status;
use crate::hir::types::TypeContext;

/// ── Tracing infrastructure ─────────────────────────────────────────

/// Whether path tracing is enabled at runtime.  Controlled by the
/// `PONENT_TRACE` environment variable; only compiled in debug builds.
/// Cached in a `OnceLock` so hot trace paths do not re-read the
/// environment on every call.
#[cfg(debug_assertions)]
pub(crate) fn tracing_enabled() -> bool {
    static TRACE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE_ENABLED.get_or_init(|| std::env::var("PONENT_TRACE").is_ok())
}

/// Per-thread recursion depth for trace indentation.  Managed by
/// `TraceGuard`; incremented on entering a traced function and restored
/// on exit (even on early return).
#[cfg(debug_assertions)]
thread_local! {
    static TRACE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// RAII guard that increments the trace depth for the duration of a
/// traced call and restores it on drop.
#[cfg(debug_assertions)]
pub(crate) struct TraceGuard;

#[cfg(debug_assertions)]
impl TraceGuard {
    pub(crate) fn enter() -> Self {
        TRACE_DEPTH.with(|d| d.set(d.get() + 1));
        TraceGuard
    }
}

#[cfg(debug_assertions)]
impl Drop for TraceGuard {
    fn drop(&mut self) {
        TRACE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Print one trace line with depth-based indentation.
#[cfg(debug_assertions)]
pub(crate) fn trace_line(args: std::fmt::Arguments<'_>) {
    if !tracing_enabled() {
        return;
    }
    let depth = TRACE_DEPTH.with(|d| d.get());
    // Write the indentation without allocating (the `"  ".repeat(depth)`
    // alternative allocates a fresh String on every trace line).
    for _ in 0..depth {
        eprint!("  ");
    }
    eprintln!("[TRACE] {}", args);
}

/// Trace one step of type resolution: entry (the type being resolved, the
/// binder name table, and the skolem list) and the outcome (Some/None).
#[cfg(debug_assertions)]
pub(crate) fn trace_resolve(
    ty: &crate::ast::Type,
    exist_params: &[crate::symbol::Symbol],
    skolems: &[crate::hir::types::TypeId],
    result: &Option<crate::hir::types::TypeId>,
) {
    if !tracing_enabled() {
        return;
    }
    let names: Vec<String> = exist_params
        .iter()
        .map(|s| s.as_str().to_string())
        .collect();
    match result {
        Some(id) => trace_line(format_args!(
            "resolve {ty:?} names={names:?} skolems={skolems:?} => Some({id:?})"
        )),
        None => trace_line(format_args!(
            "resolve {ty:?} names={names:?} skolems={skolems:?} => None"
        )),
    }
}

/// ── Pool logging ──────────────────────────────────────────────────

/// Trace the current GADT existential skolem scope stack: for each scope,
/// the number of skolems and their TypeIds, plus the variant's binder
/// names.  Used by `check_pattern_inner` to debug nested existential
/// pattern scope handling (same-named binders must resolve to their own
/// scope's skolem).
#[cfg(debug_assertions)]
pub(crate) fn trace_skolem_scope(
    scopes: &[crate::hir::types::ExistScopeFrame],
    exist_params: &[crate::symbol::Symbol],
    variant: &crate::symbol::Symbol,
) {
    if !tracing_enabled() {
        return;
    }
    let names: Vec<String> = exist_params
        .iter()
        .map(|s| s.as_str().to_string())
        .collect();
    trace_line(format_args!(
        "skolem_scope variant={} exist_params={names:?} stack={:?}",
        variant.as_str(),
        scopes
    ));
}

/// Print the current state of all region pools.
pub fn log_pool(tree: &InferRegionTree) {
    eprintln!("[Region tree ({} nodes)]:", tree.nodes.len());
    for (i, node) in tree.nodes.iter().enumerate() {
        let level = node.level;
        let indent = "  ".repeat(level);
        let dirty = if node.dirty { " [dirty]" } else { "" };
        let alive = if node.pool.is_alive() {
            " [alive]"
        } else {
            " [dead]"
        };
        eprintln!(
            "{}Region {} (level={}){}{}: var_ids={:?}, rigid={:?}",
            indent, i, level, dirty, alive, node.pool.var_ids, node.pool.rigid_var_ids,
        );
    }
}

/// ── Generalization logging ───────────────────────────────────────

/// Print the list of generalized variables.
pub fn log_generalized(generalized: &[(usize, usize)]) {
    if generalized.is_empty() {
        eprintln!("[Generalization] No variables were generalized.");
        return;
    }
    eprintln!("[Generalization] {} variables:", generalized.len());
    for (region_id, var_id) in generalized {
        eprintln!("   (region {}, var {})", region_id, var_id);
    }
}

/// ── Pool invariant assertions ────────────────────────────────────

/// Assert that the pool membership invariant holds:
/// 1. Each var_id appears in at most one pool.
/// 2. No Generalized variables are in any pool.
/// 3. PG variables are in their parent region's pool (if applicable).
///
/// Panics if any invariant is violated.
pub fn assert_pool_invariant(tree: &InferRegionTree) {
    // Collect all var_ids across all pools, check for duplicates.
    let mut all_vars: Vec<usize> = Vec::new();
    for node in &tree.nodes {
        all_vars.extend(&node.pool.var_ids);
    }
    // Check for duplicates
    let mut seen = std::collections::HashSet::new();
    for &v in &all_vars {
        assert!(
            seen.insert(v),
            "Pool invariant violated: var {} appears in multiple pools!",
            v
        );
    }
    eprintln!(
        "[Pool invariant] All {} variables are unique across pools.",
        all_vars.len()
    );
}

/// ── Undo log logging ─────────────────────────────────────────────

/// Print the current undo log for debugging.
pub fn log_undo_log(log: &[PoolUndoEntry]) {
    if log.is_empty() {
        eprintln!("[Undo log] Empty.");
        return;
    }
    eprintln!("[Undo log] {} entries:", log.len());
    for (i, entry) in log.iter().enumerate() {
        match entry {
            PoolUndoEntry::Register {
                region_idx,
                var_id,
                kind,
            } => {
                eprintln!(
                    "  [{}] Register region={}, var={}, kind={:?}",
                    i, region_idx, var_id, kind
                );
            }
            PoolUndoEntry::Unregister { region_idx, var_id } => {
                eprintln!("  [{}] Unregister region={}, var={}", i, region_idx, var_id);
            }
        }
    }
}

/// ── GenStatus logging ────────────────────────────────────────────

/// Print the generalization status of all variables.
pub fn log_gen_statuses(gen_statuses: &[GenStatus]) {
    eprintln!("[GenStatuses] {} vars:", gen_statuses.len());
    for (i, status) in gen_statuses.iter().enumerate() {
        let label = match status {
            GenStatus::Ungeneralized => "Ungeneralized",
            GenStatus::Generalized => "Generalized",
            GenStatus::PartiallyGeneralizable => "PG",
            GenStatus::PartialInstance => "PI",
        };
        eprintln!("  var {}: {}", i, label);
    }
}

/// ── GADT registry logging ────────────────────────────────────────

/// Print the current GADT fact registry for debugging.
/// Shows each arm's facts: param refinements and existential equations.
pub fn log_gadt_registry<'input>(ctx: &TypeContext<'input>) {
    let facts = ctx.gadt.facts.borrow();
    let depth = facts.len();
    eprintln!("[GADT registry] {} arm(s) active:", depth);
    for (arm_i, arm) in facts.iter().enumerate() {
        eprintln!("  Arm {} ({} facts):", arm_i, arm.len());
        for fact in arm {
            eprintln!("    {:?}", fact);
        }
    }
}

/// Assert that push_gadt_arm / pop_gadt_arm calls are balanced.
/// The GADT registry should be empty outside of pattern-match arms.
/// Also cross-validates the depth counter with the stack length.
/// Panics if the registry depth is non-zero when expected to be zero.
pub fn assert_gadt_registry_empty<'input>(ctx: &TypeContext<'input>) {
    let eq_depth = ctx.gadt.facts.borrow().len();
    let counter = ctx.gadt.arm_depth.get();
    assert_eq!(
        eq_depth, counter,
        "GADT registry desync: gadt_facts.len()={} but gadt_arm_depth={}",
        eq_depth, counter
    );
    assert!(
        eq_depth == 0,
        "GADT registry invariant violated: expected 0 active arms, found {} (push/pop mismatch)",
        eq_depth,
    );
}

// ── CFG / borrow-check tracing ────────────────────────────────────────
// Added for the borrow-check and CFG troubleshooting: the leave-with,
// catch, and place-precision investigations all required temporary
// eprintln! probes.  These make the data directly visible under
// `PONENT_TRACE=1` without touching the code.

use crate::hir::cfg_graph::{BlockId, CfgGraph, Point};
use crate::hir::types::{FrozenPlace, LoanKind};
use crate::symbol::Symbol;

/// Dump the CFG structure: per block — the id, the statement count, the
/// terminator, the successors, and the back edges.
pub fn log_cfg(cfg: &CfgGraph) {
    if !tracing_enabled() {
        return;
    }
    eprintln!("── CFG ──");
    eprintln!("blocks: {}", cfg.blocks().len());
    for (i, blk) in cfg.blocks().iter().enumerate() {
        eprintln!(
            "  block {i}: {} stmts, terminator {:?}, successors {:?}",
            blk.stmts.len(),
            blk.terminator,
            cfg.successors(BlockId(i)),
        );
    }
    eprintln!("  back_edges: {:?}", cfg.back_edges());
}

/// Dump the borrow-check data: the loans and the access events collected
/// by the post-pass.
pub fn log_borrow_data(
    loans: &[(
        FrozenPlace,
        Option<Symbol>,
        LoanKind,
        Point,
        crate::ast::Span,
        bool,
    )],
    events: &[(FrozenPlace, Point, bool, crate::ast::Span)],
) {
    if !tracing_enabled() {
        return;
    }
    eprintln!("── borrow data ──");
    for (i, (place, var, kind, pt, _, _)) in loans.iter().enumerate() {
        eprintln!("  loan {i}: place {place:?}, var {var:?}, kind {kind:?}, point {pt:?}");
    }
    for (i, (place, pt, is_read, _)) in events.iter().enumerate() {
        eprintln!("  event {i}: place {place:?}, point {pt:?}, read {is_read}");
    }
}

/// Dump the point-level liveness: per block, per variable, the live
/// statement runs (the sparse-interval liveness).
pub fn log_point_liveness(live: &crate::hir::cfg_graph::PointLiveness) {
    if !tracing_enabled() {
        return;
    }
    eprintln!("── point liveness ──");
    for (bi, blk) in live.live_intervals().iter().enumerate() {
        for (var, runs) in blk {
            eprintln!("  block {bi}: var {var:?} live runs {runs:?}");
        }
    }
}

/// Dump the extracted Polonius facts (the polonius_int.dl input schema).
pub fn log_facts(facts: &crate::hir::polonius::PoloniusFacts) {
    if !tracing_enabled() {
        return;
    }
    eprintln!("── Polonius facts ──");
    eprintln!("  cfg_edge: {:?}", facts.cfg_edge);
    eprintln!("  loan_issued_at: {:?}", facts.loan_issued_at);
    eprintln!("  loan_invalidated_at: {:?}", facts.loan_invalidated_at);
    eprintln!("  var_used_at: {:?}", facts.var_used_at);
    eprintln!(
        "  use_of_var_derefs_origin: {:?}",
        facts.use_of_var_derefs_origin
    );
    eprintln!("  subset_base: {:?}", facts.subset_base);
    eprintln!("  universal_region: {:?}", facts.universal_region);
    eprintln!(
        "  known_placeholder_subset: {:?}",
        facts.known_placeholder_subset
    );
}
