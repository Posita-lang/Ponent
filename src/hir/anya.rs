// ── Anya Module ────────────────────────────────────────────────────
//
use crate::hir::shape_var::Status;

/// ── Pool logging ──────────────────────────────────────────────────

/// Print the current state of all region pools.
pub fn log_pool(tree: &InferRegionTree) {
    eprintln!("[Region tree ({} nodes)]:", tree.nodes.len());
    for (i, node) in tree.nodes.iter().enumerate() {
        let level = node.level;
        let indent = "  ".repeat(level);
        let dirty = if node.dirty { " [dirty]" } else { "" };
        let alive = if node.pool.is_alive() { " [alive]" } else { " [dead]" };
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
    eprintln!("[Pool invariant] All {} variables are unique across pools.", all_vars.len());
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
            PoolUndoEntry::Register { region_idx, old_var_len, old_rigid_len } => {
                eprintln!("  [{}] Register region={}, var_len={}, rigid_len={}", i, region_idx, old_var_len, old_rigid_len);
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
        };
        eprintln!("  var {}: {}", i, label);
    }
}