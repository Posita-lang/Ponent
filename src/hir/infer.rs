use crate::ast::FileId;
use crate::ast::Span;
use crate::hir::shape_var::{
    ShapeVarContext, ShapeVarId, TypeShape, shapes_compatible, type_data_to_shape,
};
use crate::hir::smt::SmtSolver;
use crate::hir::symbol::SymbolTable;
use crate::hir::traits::TraitEnv;
use crate::hir::types::*;
use ena::unify::{EqUnifyValue, UnificationTableStorage, UnifyKey};
use ena::snapshot_vec as sv;
use ena::undo_log::{Snapshots, UndoLogs};
use ena::unify::Delegate;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet;
use small_map::FxSmallMap;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::sync::Arc;

pub mod defaulting;

/// ── Inline GuardSet (ported from OmniML guard_set.ml) ───────────

/// Reference-counted guard set for tracking when inference variables
/// are captured by suspended constraints (OmniML §6).
#[derive(Debug, Clone)]
pub struct GuardSet {
    pub direct_guards: usize,
    pub transitive_guards: HashMap<usize, usize>,
}

impl Default for GuardSet {
    fn default() -> Self {
        Self::empty()
    }
}

impl GuardSet {
    pub fn empty() -> Self {
        GuardSet {
            direct_guards: 0,
            transitive_guards: HashMap::default(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.direct_guards == 0 && self.transitive_guards.is_empty()
    }
    pub fn add_guard(&mut self) {
        debug_assert!(
            self.direct_guards != usize::MAX,
            "add_guard: direct-guard refcount overflow (usize::MAX) — \
             add/remove imbalance or runaway guard growth"
        );
        // Saturating, never wrapping: a corrupted refcount must keep the
        // variable GUARDED (fail-safe — premature generalization is
        // unsound); a wrap to 0 would make `is_empty()` report unguarded
        // and the variable could be generalized while a constraint still
        // blocks it.
        self.direct_guards = self.direct_guards.saturating_add(1);
    }
    pub fn remove_guard(&mut self) {
        debug_assert!(
            self.direct_guards > 0,
            "remove_guard: count == 0 — removal without matching addition; \
             this is an invariant violation that indicates a guard-set \
             lifecycle bug",
        );
        if self.direct_guards > 0 {
            self.direct_guards -= 1;
        }
    }
    pub fn add_transitive_guard(&mut self, region_id: usize) {
        let count = self.transitive_guards.entry(region_id).or_insert(0);
        *count = count.saturating_add(1);
    }
    pub fn remove_transitive_guard(&mut self, region_id: usize) {
        if let Some(count) = self.transitive_guards.get_mut(&region_id) {
            debug_assert!(
                *count > 0,
                "remove_transitive_guard: count == 0 for region_id={} — \
                 removal without matching addition; this is an invariant violation \
                 that indicates a guard-set lifecycle bug",
                region_id,
            );
            if *count == 0 {
                // Usize underflow guard (release builds): prevent silent state
                // corruption.  The debug_assert! above catches this in debug builds.
                self.transitive_guards.remove(&region_id);
                return;
            }
            *count -= 1;
            if *count == 0 {
                self.transitive_guards.remove(&region_id);
            }
        }
    }
    pub fn clear_transitive_guard(&mut self, region_id: usize) {
        self.transitive_guards.remove(&region_id);
    }
    pub fn is_transitively_guarded(&self, region_id: usize) -> bool {
        self.transitive_guards
            .get(&region_id)
            .map_or(false, |&c| c > 0)
    }
    pub fn union(&self, other: &Self) -> Self {
        let mut transitive = self.transitive_guards.clone();
        for (&region, &count) in &other.transitive_guards {
            transitive
                .entry(region)
                .and_modify(|v| *v = v.saturating_add(count))
                .or_insert(count);
        }
        GuardSet {
            direct_guards: self.direct_guards.saturating_add(other.direct_guards),
            transitive_guards: transitive,
        }
    }
    pub fn clear(&mut self) {
        self.direct_guards = 0;
        self.transitive_guards.clear();
    }
    /// Clear only the direct guards, preserving transitive guards.
    /// Used by wake-up code: `suspend_on_var` adds direct guards, and
    /// waking should remove only those — not transitive guards that may
    /// have been added by cross-region dependency tracking.
    pub fn clear_direct_guards(&mut self) {
        self.direct_guards = 0;
    }
}

/// ── Inline InferRegionTree (ported from OmniML tree.ml + generalization.ml InferPool) ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InferRegionId(pub usize);

#[derive(Debug, Clone)]
pub struct InferPool {
    pub var_ids: Vec<usize>,
    pub rigid_var_ids: Vec<usize>,
}

impl InferPool {
    pub fn new() -> Self {
        InferPool {
            var_ids: Vec::new(),
            rigid_var_ids: Vec::new(),
        }
    }
    pub fn register_var(&mut self, var_id: usize) {
        self.var_ids.push(var_id);
    }
    pub fn register_rigid_var(&mut self, var_id: usize) {
        self.rigid_var_ids.push(var_id);
    }
    pub fn is_alive(&self) -> bool {
        !self.var_ids.is_empty() || !self.rigid_var_ids.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct InferRegionNode {
    pub id: InferRegionId,
    pub level: usize,
    pub parent: Option<InferRegionId>,
    pub children: Vec<InferRegionId>,
    pub pool: InferPool,
    pub dirty: bool,
    pub dirty_children: FxHashSet<InferRegionId>,
    /// Shape variable region node ID (optional — created lazily).
    /// Corresponds to OmniML's `Pool.shape_var_region`.
    pub shape_var_region: Option<usize>,
    /// Parent shape variable region node ID.
    /// Corresponds to OmniML's `Pool.parent_shape_var_region`.
    pub parent_shape_var_region: usize,
}

#[derive(Debug, Clone)]
pub struct InferRegionTree {
    pub nodes: Vec<InferRegionNode>,
    pub root: InferRegionId,
    pub current: InferRegionId,
    /// Roots of the dirty frontier, matching the OmniML `Tree.With_dirty.dirty_roots`
    /// design.  When a region is marked dirty and no dirty ancestor is found, it is
    /// added here so that `drain_dirty_roots` can reach it — otherwise orphaned
    /// dirty regions would be silently skipped.
    pub dirty_roots: FxHashSet<InferRegionId>,
    /// Undo log for pool mutations.
    /// When a transaction is rolled back, pools are restored to their
    /// previous states to prevent zombie entries from abandoned variables.
    pub pool_undo_log: Vec<PoolUndoEntry>,
    /// When `false`, pool mutations are NOT recorded in `pool_undo_log`.
    /// Mirrors `InferenceContext::snapshot_depth > 0`; maintained by
    /// `start_snapshot` / `commit_snapshot` / `rollback_to`.  Without this
    /// gate the log grows without bound: entries pushed outside any
    /// snapshot — and entries committed at depth 0 — are consumed by
    /// nothing and never freed.
    pub pool_logging_enabled: bool,
    /// DFS pre-order timestamps (`tin`) and post-order timestamps
    /// (`tout`), parallel to `nodes` — enable the O(1) ancestor test
    /// `tin[anc] <= tin[node] && tout[node] <= tout[anc]`.
    tin: Vec<usize>,
    tout: Vec<usize>,
    next_dfs: usize,
    /// Binary Lifting ancestor table, parallel to `nodes`:
    /// `ancestor_table[i][k]` = the 2^k-th ancestor of node i.
    ancestor_table: Vec<[InferRegionId; MAX_LOG]>,
    /// O(1) var → region lookup: maintained by
    /// register_var / register_var_in_region / unregister_var and the
    /// rollback log — replaces the linear scan over every region's pool.
    /// Uses Vec indexed by var_id (dense sequential integers) for direct
    /// O(1) indexing without hashing overhead.
    var_to_region: Vec<Option<InferRegionId>>,
    /// Region level at which each Forall-introduced skolem was created, keyed
    /// by its `universe_num`.
    ///
    /// `TypeContext::enter_universe` hands out `universe_num` from a global
    /// monotone counter, so it is an *allocation ordinal*, not a scope depth.
    /// `check_rigid_escape` must compare a skolem's introduction *depth*
    /// against the level of the region being generalized; comparing the raw
    /// counter instead makes every skolem past the Nth Forall look like it
    /// escapes every region of level ≤ N, spuriously blocking generalization.
    /// This map is the missing ordinal → depth translation.
    ///
    /// Monotone by construction: entries are never removed, so no undo
    /// logging is needed — a rolled-back Forall allocates a fresh universe
    /// on retry and the old entry becomes inert.
    skolem_levels: FxSmallMap<16, usize, usize>,
}

/// Which pool list a Register entry modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegisterKind {
    Var,
    RigidVar,
}

/// A single undo entry for a pool mutation.
#[derive(Debug, Clone)]
pub enum PoolUndoEntry {
    /// A variable was registered (pushed into `var_ids` or `rigid_var_ids`).
    /// Reverse: remove the var by value from the appropriate list.
    Register {
        region_idx: usize,
        var_id: usize,
        kind: RegisterKind,
    },
    /// A variable was unregistered (removed by value from `var_ids`).
    /// Reverse: re-insert it.
    Unregister { region_idx: usize, var_id: usize },
}

impl InferRegionTree {
    pub fn new() -> Self {
        InferRegionTree {
            nodes: vec![InferRegionNode {
                id: InferRegionId(0),
                level: 0,
                parent: None,
                children: Vec::new(),
                pool: InferPool::new(),
                dirty: false,
                dirty_children: FxHashSet::default(),
                shape_var_region: None,
                parent_shape_var_region: 0,
            }],
            root: InferRegionId(0),
            current: InferRegionId(0),
            dirty_roots: FxHashSet::default(),
            pool_undo_log: Vec::new(),
            pool_logging_enabled: false,
            tin: vec![0],
            // The root's interval covers the whole tree.
            tout: vec![usize::MAX],
            next_dfs: 1,
            ancestor_table: vec![[NO_ANCESTOR; MAX_LOG]],
            var_to_region: Vec::new(),
            skolem_levels: FxSmallMap::new(),
        }
    }
    pub fn enter_region(&mut self) -> InferRegionId {
        let new_id = InferRegionId(self.nodes.len());
        let current_level = self.nodes[self.current.0].level;
        let parent_shape_var = self.nodes[self.current.0]
            .shape_var_region
            .unwrap_or(self.nodes[self.current.0].parent_shape_var_region);
        self.nodes.push(InferRegionNode {
            id: new_id,
            level: current_level + 1,
            parent: Some(self.current),
            children: Vec::new(),
            pool: InferPool::new(),
            dirty: false,
            dirty_children: FxHashSet::default(),
            shape_var_region: None,
            parent_shape_var_region: parent_shape_var,
        });
        self.nodes[self.current.0].children.push(new_id);
        self.tin.push(self.next_dfs);
        // Placeholder `tout` parallel to `nodes` — exit assigns by id
        // (the post-order exit order differs from the pre-order id order).
        self.tout.push(0);
        self.next_dfs += 1;
        // ── Incremental Binary Lifting table build ──
        // f(u, k) = f(f(u, k-1), k-1); break at the root (NO_ANCESTOR).
        let mut ancestors = [NO_ANCESTOR; MAX_LOG];
        ancestors[0] = self.current;
        for k in 1..MAX_LOG {
            let mid = ancestors[k - 1];
            if mid == NO_ANCESTOR {
                break;
            }
            ancestors[k] = self.ancestor_table[mid.0][k - 1];
        }
        self.ancestor_table.push(ancestors);
        let old = self.current;
        self.current = new_id;
        old
    }
    pub fn exit_region(&mut self) {
        if let Some(parent) = self.nodes[self.current.0].parent {
            // Assign by id — the post-order exit order differs from the
            // pre-order id order, so `push` would misalign `tout` with
            // `nodes`.
            self.tout[self.current.0] = self.next_dfs;
            self.next_dfs += 1;
            self.current = parent;
        }
    }
    pub fn get_level(&self, region_id: InferRegionId) -> usize {
        self.nodes[region_id.0].level
    }
    /// Record the region level at which a Forall-introduced skolem was
    /// created.  See the `skolem_levels` field for why this is needed.
    pub fn record_skolem_level(&mut self, universe: usize, level: usize) {
        self.skolem_levels.insert(universe, level);
    }
    /// Introduction depth of a skolem, if it was allocated by the Forall
    /// handler.  Skolems allocated elsewhere (subtype checking, trait
    /// evaluation, tests) have no recorded level.
    #[inline]
    pub fn skolem_level(&self, universe: usize) -> Option<usize> {
        self.skolem_levels.get(&universe).copied()
    }
    /// Lift `node` up `steps` levels.  Clamps to the root when `steps`
    /// exceeds the node's depth.
    #[inline(always)]
    fn lift(&self, mut node: InferRegionId, mut steps: usize) -> InferRegionId {
        debug_assert!(steps < (1 << MAX_LOG), "lift steps exceed MAX_LOG capacity");
        let mut k = 0;
        while steps > 0 {
            if steps & 1 != 0 {
                let anc = self.ancestor_table[node.0][k];
                if anc == NO_ANCESTOR {
                    return self.root;
                }
                node = anc;
            }
            steps >>= 1;
            k += 1;
        }
        node
    }

    /// Binary Lifting NCA: O(log n) queries.
    ///
    /// 1. Align the depths (lift the deeper one).
    /// 2. If equal after alignment → NCA.
    /// 3. Jump from high to low bits simultaneously (2^k ancestors differ
    ///    and are both valid).
    /// 4. The direct parent is the NCA.
    pub fn nearest_common_ancestor(&self, a: InferRegionId, b: InferRegionId) -> InferRegionId {
        if a == b {
            return a;
        }
        let level_a = self.nodes[a.0].level;
        let level_b = self.nodes[b.0].level;

        // Step 1: align depths.
        let (mut lo, mut hi) = if level_a >= level_b { (b, a) } else { (a, b) };
        let depth_diff = level_a.abs_diff(level_b);
        hi = self.lift(hi, depth_diff);

        // Step 2: aligned equal → NCA.
        if lo == hi {
            return lo;
        }

        // Step 3: simultaneous lifting from the high bit down.  The safety
        // fix over the bare `anc_lo != anc_hi` form: only lift when BOTH
        // ancestors are valid (not NO_ANCESTOR) and differ — otherwise a
        // node could be lifted onto the NO_ANCESTOR sentinel and index out
        // of bounds on the next access.
        for k in (0..MAX_LOG).rev() {
            let anc_lo = self.ancestor_table[lo.0][k];
            let anc_hi = self.ancestor_table[hi.0][k];
            if anc_lo != NO_ANCESTOR && anc_hi != NO_ANCESTOR && anc_lo != anc_hi {
                lo = anc_lo;
                hi = anc_hi;
            }
        }

        // Step 4: the direct parent is the NCA.
        let parent = self.ancestor_table[lo.0][0];
        debug_assert_eq!(
            parent, self.ancestor_table[hi.0][0],
            "NCA invariant: both nodes must share the same parent after lifting"
        );
        parent
    }
    /// O(1) ancestor test via the DFS pre/post-order intervals:
    /// `ancestor` is an ancestor of `node` iff
    /// `tin[ancestor] <= tin[node] && tout[node] <= tout[ancestor]`.
    ///
    /// While a region is still OPEN its `tout` is the `0` placeholder
    /// (assigned only by `exit_region`).  The interval test treats the
    /// placeholder as `usize::MAX` (conceptual infinity): under the stack
    /// discipline — every new region is entered as a CHILD of `current`,
    /// and `current` moves only via `enter_region`/`exit_region` (with
    /// `exit_level`'s rewind being the caller-bug guarded exception) —
    /// everything entered after an open region is its descendant, so the
    /// open interval `[tin, ∞)` correctly rejects closed siblings while
    /// accepting real open ancestors.  The NCA anchor in the debug build
    /// cross-checks this O(1) fast path against the lifting table, which is
    /// correct on every tree state — making the stack-discipline contract
    /// machine-checkable.
    pub fn is_ancestor(&self, ancestor: InferRegionId, node: InferRegionId) -> bool {
        debug_assert!(ancestor.0 < self.nodes.len());
        debug_assert!(node.0 < self.nodes.len());
        // Open-region `tout` placeholders read as infinity: a closed
        // sibling of an open node has a finite `tout`, so `MAX <= tout[anc]`
        // fails; the open ancestor's interval covers everything after it.
        let t_anc = if self.tout[ancestor.0] == 0 {
            usize::MAX
        } else {
            self.tout[ancestor.0]
        };
        let t_node = if self.tout[node.0] == 0 {
            usize::MAX
        } else {
            self.tout[node.0]
        };
        let result = self.tin[ancestor.0] <= self.tin[node.0] && t_node <= t_anc;
        // Debug anchor: NCA is correct for open AND closed regions; the O(1)
        // interval fast path must agree with it on every reachable tree
        // state (this is the machine-checked stack-discipline contract).
        debug_assert_eq!(
            result,
            self.nearest_common_ancestor(ancestor, node) == ancestor,
            "interval fast path must agree with the NCA anchor"
        );
        result
    }
    /// Mark a region as dirty (its pool changed and it may need
    /// re-examination at exit).
    ///
    /// DESIGN NOTE (dormant machinery): the §6 dirty-frontier path —
    /// `mark_dirty` → `drain_dirty_roots` → per-region generalization on
    /// region exit — is currently DORMANT: nothing calls `drain_dirty_roots`
    /// from the constraint handlers, and `force_generalize_for_regions`
    /// approximates the sweep instead via `pg_vars` + `dirty_levels` +
    /// binding-root sharing.  `mark_dirty` IS live (region entry), so the
    /// dirty flags are maintained but never drained by the exit path.
    /// Deliberate, not an oversight: wiring the exit drain changes
    /// generalization timing (a semantic change), and deleting the
    /// machinery would discard the §6 design and its tests.  Pick one when
    /// the per-query context split lands — wire it if exit-time
    /// generalization is needed there, delete it otherwise.
    pub fn mark_dirty(&mut self, region_id: InferRegionId) {
        let node = &mut self.nodes[region_id.0];
        if node.dirty {
            return;
        }
        node.dirty = true;
        // Walk up to find the nearest dirty ancestor (or dirty_roots).
        // This mirrors OmniML's `find_closest_dirty_ancestor` which returns
        // `t.dirty_roots` when no dirty ancestor is found.
        let mut current = node.parent;
        while let Some(pid) = current {
            let parent = &mut self.nodes[pid.0];
            if parent.dirty {
                if !parent.dirty_children.contains(&region_id) {
                    parent.dirty_children.insert(region_id);
                }
                return;
            }
            current = parent.parent;
        }
        // No dirty ancestor found — add to dirty_roots so that
        // drain_dirty_roots can reach this region.
        if !self.dirty_roots.contains(&region_id) {
            self.dirty_roots.insert(region_id);
        }
    }
    pub fn mark_current_dirty(&mut self) {
        self.mark_dirty(self.current);
    }
    // ── Pool membership: register / unregister ──────────────────────────
    //
    // Each type variable belongs to exactly one region's pool at any time.
    // When a variable moves to a higher region (via try_promote_var or
    // region adjustment), unregister_var is called to remove it from the
    // old pool before register_var_in_region adds it to the new one.
    // This invariant is critical for soundness: a variable appearing in
    // multiple pools could be prematurely generalised when the old region
    // is exited — a variable appearing in multiple pools could be
    // prematurely generalised when the old region is processed.

    pub fn register_var(&mut self, var_id: usize) {
        let idx = self.current.0;
        if self.pool_logging_enabled {
            self.pool_undo_log.push(PoolUndoEntry::Register {
                region_idx: idx,
                var_id,
                kind: RegisterKind::Var,
            });
        }
        self.nodes[idx].pool.register_var(var_id);
        // §6 dirty frontier: the pool's membership changed, so the region
        // must be re-examined at exit (its PG variables may have become
        // generalizable, or a generalized var may have landed here).
        self.mark_dirty(InferRegionId(idx));
        if var_id >= self.var_to_region.len() {
            self.var_to_region.resize(var_id + 1, None);
        }
        self.var_to_region[var_id] = Some(self.current);
    }
    pub fn register_rigid_var(&mut self, var_id: usize) {
        let idx = self.current.0;
        if self.pool_logging_enabled {
            self.pool_undo_log.push(PoolUndoEntry::Register {
                region_idx: idx,
                var_id,
                kind: RegisterKind::RigidVar,
            });
        }
        self.nodes[idx].pool.register_rigid_var(var_id);
    }
    pub fn register_var_in_region(&mut self, var_id: usize, region_id: InferRegionId) {
        let idx = region_id.0;
        // Idempotent: a repeated registration (e.g. a PG var lowered twice
        // in one drain cycle) must not push a duplicate pool entry or a
        // spurious undo record.
        if self.nodes[idx].pool.var_ids.contains(&var_id) {
            return;
        }
        if var_id >= self.var_to_region.len() {
            self.var_to_region.resize(var_id + 1, None);
        }
        self.var_to_region[var_id] = Some(region_id);
        if self.pool_logging_enabled {
            self.pool_undo_log.push(PoolUndoEntry::Register {
                region_idx: idx,
                var_id,
                kind: RegisterKind::Var,
            });
        }
        self.nodes[idx].pool.register_var(var_id);
        // §6 dirty frontier: same reasoning as register_var — the target
        // region's membership changed (PG lowering lands vars in parents).
        self.mark_dirty(region_id);
    }
    /// Remove a variable from a specific region's pool.
    /// Called by try_promote_var (line 724) and region adjustment (line 1383)
    /// before re-registering the variable in its new region.
    /// NOTE: This function exists and is actively used.
    /// See try_promote_var (line 776) and region adjustment (line 1383)
    /// for call sites.
    pub fn unregister_var(&mut self, var_id: usize, region_id: InferRegionId) {
        if self.pool_logging_enabled {
            self.pool_undo_log.push(PoolUndoEntry::Unregister {
                region_idx: region_id.0,
                var_id,
            });
        }
        self.nodes[region_id.0]
            .pool
            .var_ids
            .retain(|&v| v != var_id);
        if var_id < self.var_to_region.len() {
            self.var_to_region[var_id] = None;
        }
        // §6 dirty frontier: membership changed here too (a var was
        // promoted out or generalized away) — the region must be
        // re-examined before its next exit.
        self.mark_dirty(region_id);
    }
    pub fn collect_dirty_ids(&self) -> Vec<InferRegionId> {
        self.nodes
            .iter()
            .filter(|n| n.dirty)
            .map(|n| n.id)
            .collect()
    }
    pub fn collect_alive_ids(&self) -> Vec<InferRegionId> {
        self.nodes
            .iter()
            .filter(|n: &&InferRegionNode| n.pool.is_alive())
            .map(|n: &InferRegionNode| n.id)
            .collect()
    }

    /// Find which region a variable was registered in, by searching all
    /// region pools.  Returns the root region if not found.
    /// Used by TcLevel escape checking in try_unify.
    pub fn region_of_var(&self, var_id: usize) -> InferRegionId {
        // O(1) via the var → region map — previously a linear
        // scan over every region's pool, O(total vars).
        self.var_to_region
            .get(var_id)
            .and_then(|v| *v)
            .unwrap_or(InferRegionId(0)) // root
    }

    pub fn drain_dirty<F>(&mut self, node_id: InferRegionId, f: &mut F)
    where
        F: FnMut(InferRegionId, &mut Self),
    {
        // Iterative post-order traversal with an explicit stack.
        // Phase 1: collect the post-order sequence (children before parents).
        // Phase 2: execute `f` on each node in that order, then clean up.
        if !self.nodes[node_id.0].dirty {
            return;
        }
        let mut stack: Vec<InferRegionId> = vec![node_id];
        let mut post_order: Vec<InferRegionId> = Vec::new();
        while let Some(id) = stack.pop() {
            if !self.nodes[id.0].dirty {
                continue;
            }
            post_order.push(id);
            let children: Vec<InferRegionId> =
                self.nodes[id.0].dirty_children.iter().copied().collect();
            for child_id in children {
                if self.nodes[child_id.0].dirty {
                    stack.push(child_id);
                }
            }
        }
        // Phase 2: process in post-order (children before parents), then clean up.
        post_order.reverse(); // stack DFS gives pre-order; reverse for post-order
        for id in post_order {
            f(id, self);
            if self.nodes[id.0].dirty_children.is_empty() {
                self.nodes[id.0].dirty = false;
                if let Some(parent_id) = self.nodes[id.0].parent {
                    self.nodes[parent_id.0].dirty_children.retain(|&c| c != id);
                }
            }
        }
    }
    pub fn drain_dirty_roots<F>(&mut self, f: &mut F)
    where
        F: FnMut(InferRegionId, &mut Self),
    {
        // Drain from dirty_roots first, then fall back to the root for
        // backward compatibility with regions that are reachable from root.
        let roots: Vec<InferRegionId> = self.dirty_roots.drain().collect();
        for root_id in roots {
            self.drain_dirty(root_id, f);
        }
        self.drain_dirty(self.root, f);
    }

    /// Roll back pool mutations recorded in the undo log.
    ///
    /// Uses a single pass in reverse order, applying the inverse of each
    /// operation:
    /// - Register (add var X to pool Y) → remove var X from pool Y by value
    /// - Unregister (remove var X from pool Y) → re-insert var X into pool Y
    ///
    /// # Why single-pass reverse-order is correct
    ///
    /// This is a standard undo-log pattern: the last operation is undone first.
    /// Because each entry records the *identity* of the variable being added or
    /// removed (not just a length), every inverse operation is exact:
    ///
    /// | Operation          | Forward                    | Inverse                   |
    /// |--------------------|----------------------------|---------------------------|
    /// | `Register(var_id)` | `push var_id`              | `retain(!= var_id)`       |
    /// | `Unregister(var_id)`| `retain(!= var_id)`       | `push var_id`             |
    ///
    /// The use of value-based identity (not position) means the inverse of
    /// Register always removes the exact variable that was added, regardless
    /// of how many other variables were added or removed in between.  This
    /// makes the single-pass approach correct for any interleaving pattern.
    ///
    /// # Why the two-pass approach was wrong
    ///
    /// The previous implementation used two passes with length-based truncation:
    ///
    ///   Pass 1: for each Register entry (reverse), `truncate(old_var_len)`
    ///   Pass 2: for each Unregister entry (reverse), `push(var_id)`
    ///
    /// This is unsound because `truncate` restores by *position*, not by value.
    /// When an Unregister removes a variable from the middle of the pool, the
    /// length-based snapshot no longer identifies which elements belong there.
    ///
    /// ## Minimal reproduction of the bug
    ///
    /// Save the following as `bug.rs` and run with `rustc bug.rs && ./bug`:
    ///
    /// ```rust
    /// // Initial pool: [99]
    /// // 1. Register(100) → [99, 100]   records old_var_len=1
    /// // 2. Unregister(99) → [100]      removes 99 by value
    /// // 3. Register(200) → [100, 200]  records old_var_len=1 again
    /// //
    /// // Two-pass rollback trace:
    /// //   Pass 1 (reverse truncate):
    /// //     Register(200) → truncate(1) → [100]     ← wrong, removed 200
    /// //     Unregister(99) → skip
    /// //     Register(100) → truncate(1) → [100]     ← no-op, already 1
    /// //   Pass 2 (reverse re-insert):
    /// //     Unregister(99) → push 99 → [100, 99]    ← 99 is back but 100 survived!
    /// //     (the other entries are Register, skipped)
    /// //   Result: [100, 99]  ❌  Expected: [99]
    /// ```
    ///
    /// The root cause: `old_var_len=1` is ambiguous — it could mean `[99]` or
    /// `[100]`.  Once Unregister removes 99, truncating to length 1 keeps the
    /// *wrong* element (100).  The value-based approach (single-pass reverse)
    /// records `var_id=100` and `var_id=200`, so the inverse is unambiguous:
    ///
    /// ```text
    ///   Register(200) inverse → retain(!=200) → [100]
    ///   Unregister(99) inverse → push 99 → [100, 99]
    ///   Register(100) inverse → retain(!=100) → [99]
    ///   Result: [99]  ✓
    /// ```
    ///
    /// # Contrast with OmniML reference
    ///
    /// The OmniML reference implementation (omniml/lib/constraint_solver/
    /// generalization.ml) avoids this issue entirely by using a union-find
    /// transactional store for rollback (omniml/lib/std/union_find.ml) and
    /// never removing individual variables from pools by value (the pool
    /// only has `register_type` — prepend to list).  Ponent's `unregister_var`
    /// (for `try_promote_var` / region promotion) is a custom addition that
    /// requires value-based rollback.
    pub fn rollback_pool(&mut self) {
        for entry in self.pool_undo_log.iter().rev() {
            match entry {
                PoolUndoEntry::Register {
                    region_idx,
                    var_id,
                    kind,
                } => {
                    if *region_idx < self.nodes.len() {
                        match kind {
                            RegisterKind::Var => {
                                self.nodes[*region_idx]
                                    .pool
                                    .var_ids
                                    .retain(|&v| v != *var_id);
                                if *var_id < self.var_to_region.len() {
                                    self.var_to_region[*var_id] = None;
                                }
                            }
                            RegisterKind::RigidVar => {
                                self.nodes[*region_idx]
                                    .pool
                                    .rigid_var_ids
                                    .retain(|&v| v != *var_id);
                            }
                        }
                    }
                }
                PoolUndoEntry::Unregister { region_idx, var_id } => {
                    if *region_idx < self.nodes.len() {
                        self.nodes[*region_idx].pool.var_ids.push(*var_id);
                        if *var_id >= self.var_to_region.len() {
                            self.var_to_region.resize(*var_id + 1, None);
                        }
                        self.var_to_region[*var_id] = Some(InferRegionId(*region_idx));
                    }
                }
            }
        }
        self.pool_undo_log.clear();
    }
}

// ── UndoLog (rustc-style) ──────────────────────────────────────
// A single enum of reversible operations.  Recorded when a snapshot
// is open; popped and `reverse()`'d on rollback.

/// A single reversible operation on `InferenceContext` state.
#[derive(Debug, Clone)]
pub(crate) enum InferUndoLog {
    PushConstraint,
    PushTypeVar,
    PushVarTypeId,
    /// Batch registration of match branches: carries the pre-extend
    /// length so a rollback truncates ALL of them (a single per-element
    /// entry cannot invert a variable-size `extend`).
    PushMatchBranches(usize),
    PushWaitList,
    /// A constraint was pushed onto `wait_lists[var_id]`: reverse pops the
    /// last element (the log is strict reverse-chronological, every push
    /// appends).  Pairs with the logged `AddGuard` so a rollback restores
    /// both the guard AND the suspension.
    PushWaitListEntry(usize),
    PushGuardSet,
    PushGenStatus,
    PushForwardRef,
    /// An instance was pushed into an existing forward_refs[pg].  Reverse: pop.
    ForwardRefPush(usize),
    /// Undo for `s_inst_copy`'s `forward_refs[pg].clear()`: stores the
    /// cleared entries so a rollback restores the instance registrations
    /// (the plain `ForwardRefPush` entry cannot — it only pops).
    ClearForwardRefs(usize, Vec<usize>),
    PushReverseRef,
    SetReverseRef(usize, Option<usize>),
    PushLowerBound,
    PushUpperBound,
    SetGenStatus(usize, GenStatus),
    AddGuard(usize),
    RemoveGuard(usize),
    AddTransitiveGuard(usize, usize),
    RemoveTransitiveGuard(usize, usize),
    InsertGenericParamBinding(usize, Option<TypeId>),
    PushShapeVar,
    /// Undo entry for `next_var_id` mutations: stores the previous value so
    /// that a rollback spanning variable creation restores the counter.
    /// Without this, `new_type_var` would allocate ids that conflict with
    /// the existing `type_vars`/`var_type_ids` vectors, breaking the
    /// "indexed by var id" invariant.
    SetNextVarId(usize),
    /// Undo for `try_promote_var` wait-list transfer.
    /// Stores the pre-transfer contents of BOTH the source and destination
    /// wait-lists so rollback restores them exactly.
    TransferWaitList {
        from: usize,
        to: usize,
        old_from: Vec<(Constraint, Arc<SkolemEnv>)>,
        old_to: Vec<(Constraint, Arc<SkolemEnv>)>,
    },
    /// Undo for `try_promote_var` guard-set transfer.
    TransferGuards {
        from: usize,
        to: usize,
        old_from: GuardSet,
        old_to: GuardSet,
    },
    /// Undo for `try_promote_var`'s pending-Sub dedup-set transfer.  The
    /// wait list entries moved to the new var keep their dedup keys in the
    /// OLD var's set; without transferring them a re-pushed identical Sub
    /// would pass dedup on the new var and double-suspend.
    TransferPendingSubVars {
        from: usize,
        to: usize,
        old_from: FxHashSet<(TypeId, TypeId)>,
        old_to: FxHashSet<(TypeId, TypeId)>,
    },
    /// Undo for `try_promote_var` forward/reverse reference transfer.
    TransferRefs {
        from: usize,
        to: usize,
        old_from_fwd: Vec<usize>,
        old_to_fwd: Vec<usize>,
        /// (instance_id, old_reverse_value) for every instance whose
        /// reverse-ref was rewritten.
        old_reverse: Vec<(usize, Option<usize>)>,
    },
    /// Undo for `try_promote_var` lower/upper bound transfer.
    TransferBounds {
        from: usize,
        to: usize,
        old_from_lb: Vec<TypeId>,
        old_to_lb: Vec<TypeId>,
        old_from_ub: Vec<TypeId>,
        old_to_ub: Vec<TypeId>,
    },
    /// Undo for `type_vars[var_id].region_id` mutations in
    /// `s_exists_lower` / `try_promote_var` / `generalize_region`.
    SetVarRegion(usize, InferRegionId),
    /// Undo for `wake_var_incremental`'s `mem::take` on wait_list.
    /// Stores the taken entries so rollback restores the suspension.
    TakeWaitList(usize, Vec<(Constraint, Arc<SkolemEnv>)>),
    /// Undo for `wake_var_incremental`'s `clear_direct_guards`.
    /// Stores the pre-clear GuardSet so rollback restores it.
    SetGuardSet(usize, GuardSet),
    /// Undo for `try_promote_var`'s promotions shadow map entry.
    /// Stores the TypeId whose promotion entry should be removed on rollback.
    RemovePromotion(TypeId),
    /// Undo for `try_promote_var`'s `var_promotions` insertion.  Without
    /// this the stale `var_id → promoted` mapping survives a rollback and
    /// `check_rigid_escape` would follow it into a variable whose region
    /// slot has been undone.
    RemoveVarPromotion(usize),
}

/// A multi-equation: the transitive closure of all equality information
/// for a set of variables.  In solved form each multi-equation contains
/// at most one non-variable type (Definition 5.1).
struct MultiEquation {
    /// All InferVar ids in this equivalence class.
    variables: FxHashSet<usize>,
    /// The (at most one) non-variable type in this class, if any.
    concrete: Option<TypeId>,
    /// True if two incompatible concrete types were found for the same
    /// variable — the constraints are unsatisfiable.
    conflicted: bool,
}

impl MultiEquation {
    fn new() -> Self {
        MultiEquation {
            variables: FxHashSet::default(),
            concrete: None,
            conflicted: false,
        }
    }
    /// Set the concrete type, detecting conflicts.
    fn try_set_concrete(&mut self, ty: TypeId) {
        match self.concrete {
            None => self.concrete = Some(ty),
            Some(existing) => {
                if existing != ty {
                    self.conflicted = true;
                }
            }
        }
    }
}

/// Transactional union-find for type variable equivalence classes.
/// Backed by ena's `UnificationTable` (InPlace storage) with an EXTERNAL
/// undo log (`UfUndoLog`): every parent/rank mutation — including path
/// compression and union-by-rank on pre-existing variables — is recorded
/// in the log while a snapshot is open and replayed in reverse on
/// rollback, matching the previous hand-written engine's semantics.
///
/// Since the resolution-authority migration, this table is THE authoritative
/// store for inference-variable resolution: each var's class root carries
/// the bound `TypeId` (see `resolve` / `unify` / `try_promote_var`).
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// ena in-place unification table. Storage-only (`L = ()`): all
    /// mutations flow through `with_log(&mut self.undo_log)` views so ena
    /// records its undo entries in OUR log instead of a private one.
    table: UnificationTableStorage<UfKey>,
    /// External undo log for `table`: ena `sv::UndoLog` entries
    /// (NewElem/SetElem) recorded while a snapshot is open.
    undo_log: UfUndoLog,
    /// Stack of undo_log lengths at each `snapshot()` call.
    snapshot_stack: Vec<usize>,
}

/// ena unification key for Ponent type-inference variables (var ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UfKey(pub(crate) u32);

impl UnifyKey for UfKey {
    type Value = Option<TypeId>;

    fn index(&self) -> u32 {
        self.0
    }

    fn from_index(u: u32) -> Self {
        UfKey(u)
    }

    fn tag() -> &'static str {
        "PonentUfKey"
    }
}

// `Option<TypeId>` inherits `UnifyValue` from this impl: None merges with
// anything, two equal Somes merge, two distinct Somes conflict — mirroring
// the `MultiEquation::try_set_concrete` conflict rule for the value-bearing
// migration that follows this engine swap.
impl EqUnifyValue for TypeId {}

/// External undo log consumed by ena's `InPlace` storage.  Entries are
/// recorded only while a snapshot is open (ena's `in_snapshot()` gate is
/// driven by `open_snapshots`).
#[derive(Debug, Clone, Default)]
struct UfUndoLog {
    log: Vec<sv::UndoLog<Delegate<UfKey>>>,
    open_snapshots: usize,
}

impl UndoLogs<sv::UndoLog<Delegate<UfKey>>> for UfUndoLog {
    fn num_open_snapshots(&self) -> usize {
        self.open_snapshots
    }

    fn push(&mut self, undo: sv::UndoLog<Delegate<UfKey>>) {
        self.log.push(undo);
    }

    fn clear(&mut self) {
        self.log.clear();
        self.open_snapshots = 0;
    }
}

impl Snapshots<sv::UndoLog<Delegate<UfKey>>> for UfUndoLog {
    type Snapshot = usize; // undo-log length at snapshot time

    fn start_snapshot(&mut self) -> usize {
        self.open_snapshots += 1;
        self.log.len()
    }

    fn actions_since_snapshot(&self, snapshot: &usize) -> &[sv::UndoLog<Delegate<UfKey>>] {
        &self.log[*snapshot..]
    }

    fn rollback_to<R>(&mut self, storage: impl FnOnce() -> R, snapshot: usize)
    where
        R: ena::undo_log::Rollback<sv::UndoLog<Delegate<UfKey>>>,
    {
        debug_assert!(
            self.log.len() >= snapshot,
            "UfUndoLog::rollback_to: snapshot token outlives the log"
        );
        debug_assert!(self.open_snapshots > 0);
        if self.log.len() > snapshot {
            let mut values = storage();
            while self.log.len() > snapshot {
                // Fully-qualified call: keeps the `Rollback` trait name out
                // of scope so other `.reverse()` calls (e.g. slice::reverse)
                // are not shadowed by trait method resolution.
                <R as ena::undo_log::Rollback<sv::UndoLog<Delegate<UfKey>>>>::reverse(
                    &mut values,
                    self.log.pop().unwrap(),
                );
            }
        }
        self.open_snapshots -= 1;
    }

    fn commit(&mut self, snapshot: usize) {
        debug_assert!(
            self.log.len() >= snapshot,
            "UfUndoLog::commit: snapshot token outlives the log"
        );
        debug_assert!(self.open_snapshots > 0);
        self.open_snapshots -= 1;
        if self.open_snapshots == 0 {
            // Outermost commit: there is no enclosing snapshot left that
            // could roll back to these entries (matches ena's VecLog and
            // the previous hand-written engine).
            self.log.clear();
        }
    }
}

/// Opaque snapshot token for UnionFind. Only the UnionFind itself
/// interprets it; callers pass it to `rollback_to`.
#[derive(Debug, Clone)]
pub struct UFSnapshot(usize); // undo_log index

impl UnionFind {
    pub fn new() -> Self {
        Self {
            table: UnificationTableStorage::new(),
            undo_log: UfUndoLog::default(),
            snapshot_stack: Vec::new(),
        }
    }

    /// Grow the table so that keys `0..n` exist. Appends fresh unbound
    /// keys; ena records `NewElem` undo entries in the external log while
    /// a snapshot is open, so rollback truncates them.
    pub fn resize_to(&mut self, n: usize) {
        let mut t = self.table.with_log(&mut self.undo_log);
        while t.len() < n {
            t.new_key(None);
        }
    }

    /// Find the class representative with path compression.
    /// Compression mutations are recorded in the external undo log by ena.
    pub fn find(&mut self, x: usize) -> usize {
        let mut t = self.table.with_log(&mut self.undo_log);
        t.find(UfKey(x as u32)).0 as usize
    }

    /// Union by rank. Returns `Some((root_a, root_b))` — the roots as
    /// found BEFORE the merge — if a merge happened, `None` if already in
    /// the same set. Root choice is ena's union-by-rank decision.
    ///
    /// NOT used by the resolution path: resolution relies on the
    /// "classes are single keys" invariant (root-carried values, readable
    /// via `probe_read(&self)`), which a merge would break.  Kept for the
    /// future structural-equivalence use it was originally written for.
    pub fn union_sets(&mut self, a: usize, b: usize) -> Option<(usize, usize)> {
        let mut t = self.table.with_log(&mut self.undo_log);
        let ra = t.find(UfKey(a as u32));
        let rb = t.find(UfKey(b as u32));
        if ra == rb {
            return None;
        }
        match t.unify_var_var(ra, rb) {
            Ok(()) => Some((ra.0 as usize, rb.0 as usize)),
            Err((v1, v2)) => {
                debug_assert!(
                    v1 == v2,
                    "UnionFind::union_sets: conflicting concrete values {v1:?} vs {v2:?} — \
                     value conflicts must be resolved by the caller (MultiEquation semantics)"
                );
                None
            }
        }
    }

    /// Return the class root of `x` (ena exposes the root, not the
    /// immediate parent pointer).  The old immediate-parent variant had no
    /// callers; this keeps the API shape for the value-bearing migration.
    pub fn parent_of(&mut self, x: usize) -> usize {
        self.find(x)
    }

    /// Bind the class root of `x` to a concrete type (value on the root).
    ///
    /// Resolution semantics: this is a LAST-WRITE-WINS overwrite of the
    /// root value (matching the old `resolutions` table, where
    /// `try_promote_var` unconditionally overwrote `resolutions[var_id]`).
    /// The ena error case (root already bound to a DIFFERENT type) is
    /// collapsed to the same overwrite by rebinding — rollback safety is
    /// unaffected because ena's SetElem undo restores the pre-bind value.
    /// The bind is recorded in the external undo log while a snapshot is
    /// open, so rollback restores the previous root value.
    pub fn bind_value(&mut self, x: usize, ty: TypeId) {
        let mut t = self.table.with_log(&mut self.undo_log);
        // ena's unify_var_value merges; on a value conflict re-run the
        // unification against the discarded value to force the overwrite.
        if t.unify_var_value(UfKey(x as u32), Some(ty)).is_err() {
            // Root currently holds a different Some(_).  Overwrite it:
            // unify against the stale value first (no-op on the root value
            // is impossible — Option<TypeId> EqUnifyValue errors on
            // distinct values), then bind.  The stale value is dropped.
            let _ = t.unify_var_value(UfKey(x as u32), None).is_ok();
            t.unify_var_value(UfKey(x as u32), Some(ty))
                .expect("Option unify with None cannot fail after error reset");
        }
    }

    /// Read the concrete type bound to `x`'s class root, if any.
    /// Ena path-compresses on read, so this is amortized O(1) &mut self.
    pub fn probe_value(&mut self, x: usize) -> Option<TypeId> {
        let mut t = self.table.with_log(&mut self.undo_log);
        t.probe_value(UfKey(x as u32))
    }

    /// Read-only probe: returns the value stored at `x`'s node itself.
    ///
    /// Sound as a full-value read ONLY under the "classes are single keys"
    /// invariant (`union_sets` is intentionally unused for resolution):
    /// every node is its own root, so ena's `try_probe_value` (root-only
    /// read) sees the same value a chasing probe would.  Lets the entire
    /// `resolve` family stay `&self` — no `RefCell`, no `&mut` cascade
    /// through ~40 callers.
    pub fn probe_read(&self, x: usize) -> Option<TypeId> {
        if x >= self.table.len() {
            return None;
        }
        self.table
            .try_probe_value(UfKey(x as u32))
            .and_then(|v| v.clone())
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Take a snapshot. The returned token captures the undo log position;
    /// `rollback_to` will replay undo entries in reverse from this point.
    pub fn snapshot(&mut self) -> UFSnapshot {
        // Bump the open-snapshot count so ena starts recording undo
        // entries, and capture the log length as the rollback point.
        let idx = self.undo_log.start_snapshot();
        self.snapshot_stack.push(idx);
        UFSnapshot(idx)
    }

    /// Discard the most recent snapshot without replaying undo entries.
    /// Called by `commit_snapshot` to keep the stack in sync.  Entries
    /// stay in the log for enclosing snapshots; the log is cleared only
    /// when the OUTERMOST snapshot commits (ena `VecLog` semantics).
    pub fn commit(&mut self) {
        if let Some(idx) = self.snapshot_stack.pop() {
            self.undo_log.commit(idx);
        }
    }

    /// Roll back to a snapshot by replaying undo entries in reverse.
    /// This correctly restores path compression, union-by-rank, and
    /// variable allocation mutations (ena replays `NewElem` as a pop and
    /// `SetElem` as a value restore over the shared storage).
    pub fn rollback_to(&mut self, snap: &UFSnapshot) {
        debug_assert!(
            self.undo_log.log.len() >= snap.0,
            "UnionFind::rollback_to: snapshot token outlives the undo log"
        );
        self.undo_log.rollback_to(|| &mut self.table, snap.0);
        // Pop snapshot_stack entries that are at or after this snapshot.
        while self.snapshot_stack.last().is_some_and(|&s| s >= snap.0) {
            self.snapshot_stack.pop();
        }
    }
}

impl Default for UnionFind {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceContext {
    fn reverse(&mut self, undo: InferUndoLog) {
        match undo {
            InferUndoLog::PushConstraint => {
                self.constraints.pop();
            }
            InferUndoLog::PushTypeVar => {
                self.type_vars.pop();
            }
            InferUndoLog::PushVarTypeId => {
                self.var_type_ids.pop();
            }
            InferUndoLog::PushMatchBranches(prev_len) => {
                self.match_branches.truncate(prev_len);
            }
            InferUndoLog::PushWaitList => {
                self.wait_lists.pop();
            }
            InferUndoLog::PushWaitListEntry(v) => {
                self.wait_lists.get_mut(v).map(|w| w.pop());
            }
            InferUndoLog::PushGuardSet => {
                self.guard_sets.pop();
            }
            InferUndoLog::PushGenStatus => {
                self.gen_statuses.pop();
            }
            InferUndoLog::PushForwardRef => {
                self.forward_refs.pop();
            }
            InferUndoLog::ForwardRefPush(pg) => {
                if pg < self.forward_refs.len() {
                    self.forward_refs[pg].pop();
                }
            }
            InferUndoLog::ClearForwardRefs(pg, entries) => {
                if pg < self.forward_refs.len() {
                    self.forward_refs[pg] = entries;
                }
            }
            InferUndoLog::PushReverseRef => {
                self.reverse_refs.pop();
            }
            InferUndoLog::SetReverseRef(i, old) => {
                if i < self.reverse_refs.len() {
                    self.reverse_refs[i] = old;
                }
            }
            InferUndoLog::PushLowerBound => {
                self.lower_bounds.pop();
            }
            InferUndoLog::PushUpperBound => {
                self.upper_bounds.pop();
            }
            InferUndoLog::SetGenStatus(i, old) => {
                if i < self.gen_statuses.len() {
                    let cur = self.gen_statuses[i];
                    self.gen_statuses[i] = old;
                    // Maintain pg_vars index on rollback.
                    if old == GenStatus::PartiallyGeneralizable
                        && cur != GenStatus::PartiallyGeneralizable
                    {
                        self.pg_vars.insert(i);
                    } else if old != GenStatus::PartiallyGeneralizable
                        && cur == GenStatus::PartiallyGeneralizable
                    {
                        self.pg_vars.remove(&i);
                    }
                }
            }
            InferUndoLog::AddGuard(i) => {
                if i < self.guard_sets.len() {
                    self.guard_sets[i].remove_guard();
                }
            }
            InferUndoLog::RemoveGuard(i) => {
                if i < self.guard_sets.len() {
                    self.guard_sets[i].add_guard();
                }
            }
            InferUndoLog::AddTransitiveGuard(i, region_id) => {
                if i < self.guard_sets.len() {
                    self.guard_sets[i].remove_transitive_guard(region_id);
                }
            }
            InferUndoLog::RemoveTransitiveGuard(i, region_id) => {
                if i < self.guard_sets.len() {
                    self.guard_sets[i].add_transitive_guard(region_id);
                }
            }
            InferUndoLog::InsertGenericParamBinding(key, old) => {
                match old {
                    Some(v) => self.generic_param_bindings.insert(key, v),
                    None => self.generic_param_bindings.remove(&key),
                };
            }
            InferUndoLog::PushShapeVar => {
                self.shape_vars
                    .truncate_vars(self.shape_vars.vars_len().saturating_sub(1));
            }
            InferUndoLog::SetNextVarId(old) => {
                self.next_var_id = old;
            }
            InferUndoLog::TransferWaitList {
                from,
                to,
                old_from,
                old_to,
            } => {
                if from < self.wait_lists.len() {
                    self.wait_lists[from] = old_from;
                }
                if to < self.wait_lists.len() {
                    self.wait_lists[to] = old_to;
                }
            }
            InferUndoLog::TransferGuards {
                from,
                to,
                old_from,
                old_to,
            } => {
                if from < self.guard_sets.len() {
                    self.guard_sets[from] = old_from;
                }
                if to < self.guard_sets.len() {
                    self.guard_sets[to] = old_to;
                }
            }
            InferUndoLog::TransferPendingSubVars {
                from,
                to,
                old_from,
                old_to,
            } => {
                if from < self.pending_sub_vars.len() {
                    self.pending_sub_vars[from] = old_from;
                }
                if to < self.pending_sub_vars.len() {
                    self.pending_sub_vars[to] = old_to;
                }
            }
            InferUndoLog::TransferRefs {
                from,
                to,
                old_from_fwd,
                old_to_fwd,
                old_reverse,
            } => {
                if from < self.forward_refs.len() {
                    self.forward_refs[from] = old_from_fwd;
                }
                if to < self.forward_refs.len() {
                    self.forward_refs[to] = old_to_fwd;
                }
                for (inst_id, old_rev) in old_reverse {
                    if inst_id < self.reverse_refs.len() {
                        self.reverse_refs[inst_id] = old_rev;
                    }
                }
            }
            InferUndoLog::TransferBounds {
                from,
                to,
                old_from_lb,
                old_to_lb,
                old_from_ub,
                old_to_ub,
            } => {
                if from < self.lower_bounds.len() {
                    self.lower_bounds[from] = old_from_lb;
                }
                if to < self.lower_bounds.len() {
                    self.lower_bounds[to] = old_to_lb;
                }
                if from < self.upper_bounds.len() {
                    self.upper_bounds[from] = old_from_ub;
                }
                if to < self.upper_bounds.len() {
                    self.upper_bounds[to] = old_to_ub;
                }
            }
            InferUndoLog::SetVarRegion(var_id, old_region) => {
                if var_id < self.type_vars.len() {
                    self.type_vars[var_id].region_id = old_region;
                }
            }
            InferUndoLog::TakeWaitList(i, wl) => {
                if i < self.wait_lists.len() {
                    self.wait_lists[i] = wl;
                }
            }
            InferUndoLog::SetGuardSet(i, gs) => {
                if i < self.guard_sets.len() {
                    self.guard_sets[i] = gs;
                }
            }
            InferUndoLog::RemovePromotion(ty) => {
                self.promotions.remove(&ty);
            }
            InferUndoLog::RemoveVarPromotion(var_id) => {
                self.var_promotions.remove(&var_id);
            }
        }
    }
}

/// Priority wrapper for constraints, enabling BucketQueue-based scheduling.
/// Constraints are processed in order of "determinism":
///   Priority 0: Eq(concrete, concrete) — both sides fully resolved
///   Priority 1: Eq(concrete, infer)    — one side is InferVar
///   Priority 2: Eq(infer, infer)       — both sides are InferVar
///   Priority 3: Sub(concrete, concrete)
///   Priority 4: Sub(concrete, infer) / Sub(infer, concrete)
///   Priority 5: Sub(infer, infer)
///   Priority 6: Impl constraints
//
// Named constants for the priority levels above, so the solver's
// determinism ordering is not buried as magic numbers.
/// Binary Lifting max levels: 2^20 ≈ 10^6, far beyond any real nesting depth.
const MAX_LOG: usize = 20;

/// Sentinel value: "no ancestor" (the root's parent).
const NO_ANCESTOR: InferRegionId = InferRegionId(usize::MAX);

const PRIORITY_EQ_CONCRETE: u8 = 0;
const PRIORITY_EQ_MIXED: u8 = 1;
const PRIORITY_EQ_INFER: u8 = 2;
const PRIORITY_SUB_CONCRETE: u8 = 3;
const PRIORITY_SUB_MIXED: u8 = 4;
const PRIORITY_SUB_INFER: u8 = 5;
const PRIORITY_MATCH_SUSPENDED: u8 = 6;
const PRIORITY_MATCH_RESOLVED: u8 = 3;
const PRIORITY_FORALL: u8 = 6;
const PRIORITY_DEFER: u8 = 6;
const PRIORITY_ADMIN: u8 = 7;
const PRIORITY_LET_BARRIER: u8 = 8;
const PRIORITY_BODY_FLOOR: u8 = 4;
// ── Binder-introduction priorities ────────────────────────────────
// These were previously bare `1` / `2` literals in `Constraint::priority`.
// They deliberately *share* bands with the Eq priorities: there are more
// constraint kinds (15) than buckets (9), so a distinct band per kind is not
// expressible in a `u8`.  `BucketQueue` drains each band FIFO, so a shared
// band is a deterministic insertion-order tie rather than a nondeterministic
// one — naming the constants keeps the sharing intentional and visible
// instead of an accident that silently changes if the Eq bands are renumbered.
const PRIORITY_INSTANCE: u8 = PRIORITY_EQ_MIXED;
const PRIORITY_EXISTS: u8 = PRIORITY_EQ_INFER;
const PRIORITY_LET: u8 = PRIORITY_EQ_INFER;
/// Default bit width for inferred `Int` / `UInt` types.
const DEFAULT_INT_WIDTH: u32 = 32;
/// Default bit width for inferred `Float` types.
const DEFAULT_FLOAT_WIDTH: u32 = 64;
/// A scoped environment carried with each delayed constraint, mapping
/// inference variable IDs to Skolem TypeIds introduced by enclosing
/// Forall binders.  The solver consults this environment before falling
/// through to the global resolution tables, enabling properly scoped
/// rigid variable semantics for universally quantified constraints.
pub(crate) type SkolemEnv = HashMap<usize, TypeId>;

/// Shared empty SkolemEnv — avoids allocating a fresh `HashMap` for every
/// initial constraint and every `suspend_on_var` call that has no enclosing
/// `Forall` scope.
fn empty_skolem_env() -> Arc<SkolemEnv> {
    static EMPTY: std::sync::OnceLock<Arc<SkolemEnv>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(|| Arc::new(SkolemEnv::default())).clone()
}

#[derive(Clone)]
pub(crate) struct PrioritizedConstraint {
    pub(crate) priority: u8,
    pub(crate) constraint: Constraint,
    /// Scoped skolem environment: maps var_id → Skolem TypeId for
    /// Forall-bound rigid variables that apply to this constraint.
    /// Inherited from the parent constraint and extended by new Forall
    /// binders.  Empty unless the constraint is inside a Forall body.
    /// Wrapped in Arc to avoid O(map_size) HashMap clones on every
    /// heap push — most constraints inherit their parent's env unchanged;
    /// only Forall creates a new env.
    pub(crate) env: Arc<SkolemEnv>,
}

impl PartialEq for PrioritizedConstraint {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedConstraint {}

impl PartialOrd for PrioritizedConstraint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // BinaryHeap is a max-heap; BucketQueue gives native min-priority behavior
        other.priority.partial_cmp(&self.priority)
    }
}

impl Ord for PrioritizedConstraint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.priority.cmp(&self.priority)
    }
}

/// Bucket queue for prioritized constraints with small integer priorities (0..=6).
/// Push is O(1) (append to bucket), pop is amortized O(1) (scan from lowest
/// non-empty bucket).  Replaces `BinaryHeap` which has O(log n) push/pop.
///
/// Priority constants range from PRIORITY_EQ_CONCRETE (0) to PRIORITY_LET_BARRIER (8),
/// giving exactly 9 buckets.
pub(crate) struct BucketQueue {
    buckets: [VecDeque<PrioritizedConstraint>; 9],
    min_priority: usize,
    len: usize,
    /// Counters for constraint types, enabling O(1) checks instead of
    /// full-heap scans (`heap.iter().any(Match)`).
    match_count: usize,
    eq_count: usize,
}

impl BucketQueue {
    pub(crate) fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| VecDeque::new()),
            min_priority: 9,
            len: 0,
            match_count: 0,
            eq_count: 0,
        }
    }
    pub(crate) fn push(&mut self, pc: PrioritizedConstraint) {
        let p = pc.priority as usize;
        // Unconditional (not debug_assert): priorities are compile-time
        // constants 0..9 today, but a future constant ≥ 9 must fail loudly
        // in release too rather than index out of bounds there.
        assert!(p < 9, "priority {} out of bucket range 0..9", p);
        if p < self.min_priority {
            self.min_priority = p;
        }
        match &pc.constraint {
            Constraint::Match { .. } => self.match_count += 1,
            Constraint::Eq(..) => self.eq_count += 1,
            _ => {}
        }
        // ExitRegion uses push_front (LIFO) to ensure inner regions exit
        // before outer regions when nested Exists/Forall are processed.
        if matches!(pc.constraint, Constraint::ExitRegion { .. }) {
            self.buckets[p].push_front(pc);
        } else {
            self.buckets[p].push_back(pc);
        }
        self.len += 1;
    }
    pub(crate) fn pop(&mut self) -> Option<PrioritizedConstraint> {
        while self.min_priority < 9 {
            if let Some(pc) = self.buckets[self.min_priority].pop_front() {
                self.len -= 1;
                match &pc.constraint {
                    Constraint::Match { .. } => self.match_count -= 1,
                    Constraint::Eq(..) => self.eq_count -= 1,
                    _ => {}
                }
                return Some(pc);
            }
            self.min_priority += 1;
        }
        None
    }
    pub(crate) fn has_matches(&self) -> bool {
        self.match_count > 0
    }
    pub(crate) fn has_eqs(&self) -> bool {
        self.eq_count > 0
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = &PrioritizedConstraint> {
        self.buckets.iter().flat_map(|b| b.iter())
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    /// Drain all items into a Vec, preserving priority order on subsequent pops.
    pub(crate) fn drain(&mut self) -> Vec<PrioritizedConstraint> {
        let mut out = Vec::with_capacity(self.len);
        for bucket in &mut self.buckets {
            out.extend(bucket.drain(..));
        }
        self.len = 0;
        self.min_priority = 9;
        self.match_count = 0;
        self.eq_count = 0;
        out
    }
    pub(crate) fn extend(&mut self, iter: impl IntoIterator<Item = PrioritizedConstraint>) {
        for pc in iter {
            self.push(pc);
        }
    }
}

/// Generalization state for an inference variable (OmniML §3.2).
/// Controls whether a variable can be generalized (let-polymorphism).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenState {
    /// Not yet generalized (free in current scope).
    Ungeneralized,
    /// Fully generalized (let-bound, can be instantiated arbitrarily).
    Generalized,
    /// Partially generalized — awaiting suspended constraints to resolve.
    PartialGeneralized,
    /// Partially instantiated — an instance of a PG variable.
    PartialInstance(usize), // id of the PG variable
}

/// Origin of a type inference variable, used in the defaulting phase to
/// distinguish between "should have been resolved" and "expected to remain
/// unresolved" variables.  When solver exhaustion leaves a variable of kind
/// `Unconstrained` or `Any` unresolved, the origin determines whether the
/// compiler reports `CannotInfer` or silently defaults to `Error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarOrigin {
    /// Variable created for an expression or binding whose type should be
    /// fully inferred.  Carries the source span of the expression, if known.
    /// Unresolved at solver exhaustion indicates a type inference failure.
    Expression(Option<crate::ast::Span>),
    /// Variable created as a placeholder for a generic parameter
    /// (e.g., `T` in `def foo<T>(x: T)`).  Expected to remain unresolved
    /// until the function is instantiated at a call site.
    GenericParam,
    /// Synthetic variable created during generic instantiation, promotion,
    /// or other internal compiler operations.  Not expected to require
    /// resolution at the current scope.
    Synthetic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeVariableKind {
    Unconstrained,
    Integer,
    Float,
    Numeric,
    Bool,
    Any,
}

/// The principal shape of a type variable (OmniML-inspired).
/// Tracks what "shape" the type is known to have, even before
/// the concrete type is fully resolved. This enables suspended
/// match constraints to determine when they can be discharged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrincipalShape {
    /// Unknown — no shape information yet.
    Unknown,
    /// Integer-family scalar: Int / UInt / USize / Rational.
    Integer,
    /// Floating-point.
    Float,
    /// Boolean.
    Bool,
    /// Character-family scalar: Char / Byte.
    Char,
    /// Function type: τ₁ → τ₂
    Arrow,
    /// Tuple type: τ₁ × τ₂ × ...
    Tuple(usize),
    /// Named constructor: identity (`DefId.0`) and arity.
    /// Two constructors are the same shape only if both id and arity match.
    Constructor(usize, usize),
    /// Polymorphic type: ∀α. τ
    Poly,
    /// InferVar or unresolved
    Var,
}

#[derive(Debug, Clone)]
pub struct TypeVar {
    pub id: usize,
    pub kind: TypeVariableKind,
    pub gen_state: GenState,
    pub shape: PrincipalShape,
    pub region_id: InferRegionId,
}

impl TypeVar {
    /// Backward-compatible level accessor — resolves level from the region tree.
    /// Used by legacy code that hasn't been migrated to the region tree yet.
    pub fn get_level(&self, tree: &InferRegionTree) -> usize {
        tree.get_level(self.region_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqOrigin {
    /// Normal type inference.
    Normal,
}

#[derive(Debug, Clone)]
pub enum Constraint {
    Eq(TypeId, TypeId, Span, EqOrigin),
    Sub(TypeId, TypeId, Span),
    /// OmniML suspended match constraint (O'Brien, Rémy & Scherer §4.1):
    /// `match τ with patterns` — suspends until the shape of τ is known.
    /// When τ resolves to a concrete type, the match is discharged.
    Match {
        /// The type whose shape must be determined.
        scrutinee: TypeId,
        /// (start, count) into the inference context's `match_branches` table.
        /// The count ensures discharge_match only scans this branch set,
        /// preventing cross-contamination from later-registered branches.
        branches_id: (usize, usize),
        span: Span,
    },
    /// OmniML existential: `∃α. C` — bind a fresh flexible type variable.
    Exists {
        var_id: usize,
        constraint: Box<Constraint>,
        span: Span,
    },
    /// OmniML universal: `∀α. C` — bind a fresh rigid (skolem) variable.
    Forall {
        var_id: usize,
        constraint: Box<Constraint>,
        span: Span,
    },
    /// OmniML instance: instantiate a generalized scheme at a type.
    /// `x[τ]` — the scheme for `x` is instantiated with `τ`.
    /// Carries the scheme TypeId directly (not a variable name) so the
    /// solver can instantiate without an external environment lookup.
    Instance {
        /// The polymorphic scheme to instantiate (e.g. `Forall(∀α.τ)`).
        scheme_ty: TypeId,
        /// The type to instantiate at (the instance being checked).
        instantiation_ty: TypeId,
        /// PG variable IDs corresponding to each binder in the scheme,
        /// in the same order as `binder_indices` collected by the handler.
        /// Empty when the caller does not track PG variables (backward
        /// compatible).
        pg_var_ids: Vec<usize>,
        span: Span,
    },
    /// OmniML let-constraint: `let x = λα.∃ᾱ. C₁ in C₂`
    Let {
        expr_var: String,
        def_constraint: Box<Constraint>,
        body_constraint: Box<Constraint>,
        span: Span,
    },
    /// Administrative: exit a region entered by Exists/Forall.
    /// Pushed at PRIORITY_DEFER so it fires after all inner constraints.
    ExitRegion {
        region_id: InferRegionId,
        span: Span,
    },
    /// Administrative: barrier between def and body of a Let constraint.
    /// Ensures def constraints are fully processed before body is pushed.
    LetBarrier {
        expr_var: String,
        body_constraint: Box<Constraint>,
        /// The region entered by the Let handler for the definition scope.
        /// The barrier exits this region and generalizes before pushing body.
        region_id: InferRegionId,
        span: Span,
    },
}

/// Describes the instantiation target of a polymorphic scheme,
/// collected from a TypeData node without holding a borrow on the
/// TypeContext<'input>. Used by the Instance constraint handler to separate
/// the immutable scan phase from the mutable variable-creation phase.
enum InstantiationTarget {
    /// `∀α₁.∀α₂. ... αₙ. body_ty` — Forall binders.
    Forall {
        binder_indices: Vec<usize>,
        body_ty: TypeId,
    },
    /// `∀quantifiers. body_ty` — Poly type.
    Poly {
        binder_indices: Vec<usize>,
        body_ty: TypeId,
    },
    /// A concrete (monomorphic) type — no instantiation needed.
    Concrete(TypeId),
}

impl Constraint {
    /// Compute priority: lower = more deterministic, processed first.
    /// This enables BinaryHeap-based scheduling where concrete-concrete
    /// constraints are resolved before those involving inference variables.
    pub fn priority<'input>(&self, ctx: &TypeContext<'input>) -> u8 {
        match self {
            Constraint::Eq(a, b, _, _) => {
                let a_is_infer =
                    matches!(ctx.get(ctx.resolve_binding(*a)), TypeData::InferVar { .. });
                let b_is_infer =
                    matches!(ctx.get(ctx.resolve_binding(*b)), TypeData::InferVar { .. });
                match (a_is_infer, b_is_infer) {
                    (false, false) => PRIORITY_EQ_CONCRETE, // concrete-concrete: highest priority
                    (true, false) | (false, true) => PRIORITY_EQ_MIXED, // one infer var
                    (true, true) => PRIORITY_EQ_INFER,      // both infer vars
                }
            }
            Constraint::Sub(sub, sup, _) => {
                let sub_is_infer = matches!(
                    ctx.get(ctx.resolve_binding(*sub)),
                    TypeData::InferVar { .. }
                );
                let sup_is_infer = matches!(
                    ctx.get(ctx.resolve_binding(*sup)),
                    TypeData::InferVar { .. }
                );
                match (sub_is_infer, sup_is_infer) {
                    (false, false) => PRIORITY_SUB_CONCRETE,
                    (true, true) => PRIORITY_SUB_INFER,
                    _ => PRIORITY_SUB_MIXED,
                }
            }
            Constraint::Match { scrutinee, .. } => {
                // Match constraints: low priority — they suspend until the
                // scrutinee's shape is resolved.
                let resolved = ctx.resolve_binding(*scrutinee);
                if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                    PRIORITY_MATCH_SUSPENDED // still an infer var → needs resolution first
                } else {
                    PRIORITY_MATCH_RESOLVED // resolved → medium priority
                }
            }
            Constraint::Exists { .. } => PRIORITY_EXISTS,
            Constraint::Forall { .. } => PRIORITY_FORALL, // forall: low priority (skolem)
            Constraint::Instance { .. } => PRIORITY_INSTANCE,
            Constraint::Let { .. } => PRIORITY_LET,
            Constraint::ExitRegion { .. } => PRIORITY_ADMIN,
            Constraint::LetBarrier { .. } => PRIORITY_LET_BARRIER,
        }
    }

    /// Extract the source span from any constraint variant.
    pub fn span(&self) -> Span {
        match self {
            Constraint::Eq(_, _, span, _)
            | Constraint::Sub(_, _, span)
            | Constraint::Match { span, .. }
            | Constraint::Exists { span, .. }
            | Constraint::Forall { span, .. }
            | Constraint::Instance { span, .. }
            | Constraint::Let { span, .. }
            | Constraint::ExitRegion { span, .. }
            | Constraint::LetBarrier { span, .. } => *span,
        }
    }
}

/// Generalization status for an inference variable (OmniML §6).
/// Tracks whether a variable can be fully generalized, partially generalized,
/// or is still waiting for suspended constraints to be resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenStatus {
    /// (I) Initial — not yet generalized
    Ungeneralized,
    /// (G) Fully generalized — safe to copy instances
    Generalized,
    /// (PG) Partially generalizable — guarded by a suspended constraint
    PartiallyGeneralizable,
    /// (PI) Partial instance — previously PG but has been updated; needs re-generalization
    PartialInstance,
}

#[derive(Debug)]
pub struct InferenceContext {
    type_vars: Vec<TypeVar>,
    var_type_ids: Vec<TypeId>,
    constraints: Vec<Constraint>,
    next_var_id: usize,
    /// Region tree replacing the linear level system (OmniML §6).
    /// Tracks nested scopes as a tree with dirty-marking for efficient
    /// incremental generalization. When PG variables keep a region alive,
    /// sibling regions at equivalent depths remain correctly distinguished.
    pub region_tree: InferRegionTree,
    /// Per-variable lower bounds (subtypes that must be ≤ this variable).
    /// lower_bounds[i] contains TypeIds that must be subtypes of variable i.
    lower_bounds: Vec<Vec<TypeId>>,
    /// Per-variable upper bounds (supertypes that this variable must be ≤).
    /// upper_bounds[i] contains TypeIds that variable i must be a subtype of.
    upper_bounds: Vec<Vec<TypeId>>,
    /// Per-variable wait lists (OmniML §3.2): constraints suspended on this var.
    /// When the var is bound (unified with a concrete type), these constraints
    /// are woken and reprocessed, enabling bidirectional type information flow.
    /// Each entry carries the constraint together with its scoped SkolemEnv
    /// so that Forall-bound rigid variables remain rigid after suspension/wake.
    wait_lists: Vec<Vec<(Constraint, Arc<SkolemEnv>)>>,
    /// Guard sets (OmniML §6): reference-counted guards tracking whether a
    /// variable is captured by suspended constraints. A non-empty guard set
    /// means the variable is PG (PartiallyGeneralizable). When all guards are
    /// discharged, the variable can become G (Generalized).
    /// Uses GuardSet with direct_guards and per-region transitive_guards.
    guard_sets: Vec<GuardSet>,
    /// Per-variable generalisation status (I / G / PG / PI).
    gen_statuses: Vec<GenStatus>,
    /// Shape variable context (OmniML §3.3, §6).
    /// Manages shape variables — first-class unifiable variables that represent
    /// not-yet-known principal shapes.  When a shape variable is resolved,
    /// all suspended match constraints waiting on it are woken.
    pub shape_vars: ShapeVarContext,
    /// Match branches table (OmniML §4.1): each SuspendedMatch constraint
    /// references a set of branch patterns by index.
    /// A branch is (label, expected_pattern) — discharged when the scrutinee's
    /// principal shape matches the pattern.
    match_branches: Vec<MatchBranchSet>,
    /// Forward references for incremental instantiation (OmniML §5.2):
    /// Maps a PG variable id to the list of instance variable ids that
    /// were created from it. When the PG var is refined, all instances
    /// are updated.
    forward_refs: Vec<Vec<usize>>,
    /// Reverse of forward_refs: for each instance, which PG var it came from.
    reverse_refs: Vec<Option<usize>>,
    /// Tracks which InferVar ids have been unified since the last
    /// `force_generalize` call, enabling incremental processing.
    /// After a partial generalization (via `force_generalize_for_regions`
    /// with `dirty_levels` or `target_var`), unprocessed entries remain
    /// in the set — the next full `force_generalize` will clean them up.
    /// Do NOT read this field directly; use `mark_dirty` / `is_dirty`.
    dirty_set: FxHashSet<usize>,
    /// Per-InferenceContext resolution table (TypeOrVar pattern).
    /// REPLACED by `union_find`: each variable's class root carries the
    /// bound TypeId (ena `Option<TypeId>` root value); var-var edges (e.g.
    /// `try_promote_var` bindings) are represented by binding the old var's
    /// root value to the new var's TypeId — the class stays a single key,
    /// so reads are pure `try_probe_value(&self)` with no pointer chasing.
    union_find: UnionFind,
    /// Per-variable origin, indexed by the same `id` used in `InferVar { id }`.
    /// Captured at variable creation time so the defaulting phase can
    /// distinguish between expression-level variables that should have been
    /// resolved and generic/synthetic variables that are expected to remain
    /// unresolved.
    var_origins: Vec<VarOrigin>,
    /// Local bindings for GenericParam indices during instantiation.
    generic_param_bindings: HashMap<usize, TypeId>,
    /// Variables resolved since last solver wake-up (avoids O(N) scan).
    resolved_ids: Vec<usize>,
    /// Parallel set for O(1) dedup with `resolved_ids`.
    resolved_set: FxHashSet<usize>,
    /// Variables that currently have non-empty wait lists.
    /// Used for incremental wake-up: only these vars are checked each
    /// outer iteration, avoiding the O(N) scan of all variables.
    suspended_vars: FxHashSet<usize>,
    /// Match constraints already discharged this solve. Prevents reprocessing.
    discharged_matches: FxHashSet<(TypeId, (usize, usize))>,
    /// Sub constraints already discharged this solve (both sides resolved and
    /// subtype-checked).  Prevents redundant processing from duplicate wake-ups.
    discharged_subs: FxHashSet<(TypeId, TypeId)>,
    /// Per-variable set of `(sub, sup)` pairs for Sub constraints currently
    /// registered (suspended) on that variable.  Prevents the same Sub from
    /// being suspended on the same variable twice, which would cause exponential
    /// entry doubling (2^k entries after k wake-suspend cycles).
    /// Cleared in `wake_var_incremental` when a variable's wait_list is taken.
    pending_sub_vars: Vec<FxHashSet<(TypeId, TypeId)>>,
    /// Subset of `gen_statuses` indices that are currently PG (PartiallyGeneralizable).
    /// Maintained incrementally by `set_gen_status` so that `force_generalize_for_regions`
    /// avoids O(total_vars) full-table scans.  Saved/restored in the snapshot stack.
    pg_vars: FxHashSet<usize>,
    /// Reusable buffer for `suspended_vars` drain — avoids allocating a
    /// new `Vec<usize>` every outer iteration of the solve loop.
    wake_buffer: Vec<usize>,
    /// Undo log (rustc-style).  Records reversible operations when a
    /// snapshot is open.  Popped and `reverse()`'d on rollback.
    undo_log: Vec<InferUndoLog>,
    /// Current snapshot nesting depth.  0 = no snapshot open.
    snapshot_depth: usize,
    /// Stack of snapshot states: for each open snapshot, records the
    /// `resolved_ids` length, `resolved_set`, `dirty_set`, `suspended_vars`,
    /// `discharged_matches`, `discharged_subs`, and `pool_undo_log` length
    /// so they can be restored on rollback (not covered by the main undo log).
    resolved_snapshot_stack: Vec<(
        usize,
        FxHashSet<usize>,
        FxHashSet<usize>,
        FxHashSet<usize>,
        FxHashSet<(TypeId, (usize, usize))>,
        usize,
        FxHashSet<usize>,
        UFSnapshot,
        FxHashSet<(usize, usize)>,
        FxHashSet<usize>,
        FxHashSet<(TypeId, TypeId)>,
        [usize; 11],
    )>,
    /// Parallel to `resolved_snapshot_stack`: for each open snapshot, the
    /// `region_tree.current` at snapshot time.  A snapshot opened mid-`Exists`
    /// leaves `current` on a child region; rolling back the solver state
    /// without restoring it would leave the next solve iteration entering
    /// regions under an ABANDONED region, defeating `is_ancestor` and the
    /// TcLevel escape check.  Region NODES are append-only (ids stay valid),
    /// so restoring `current` alone is sufficient.
    region_current_snapshot_stack: Vec<InferRegionId>,
    /// `(var_id, region_id)` pairs that have already been generalized.
    ///
    /// The sweep consults this to skip variables it has already promoted to
    /// `Generalized`, which lets them dodge the two expensive per-variable
    /// walks below (`check_rigid_escape` and the transitive-guard root scan).
    ///
    /// This used to be a `HashMap<(usize, usize), TypeId>` named `scheme_cache`
    /// that mapped a variable to its generalized *scheme* — but nothing ever
    /// read the value: only `contains_key`.  The one constraint form that
    /// would consume a scheme, `Constraint::Instance` (S-Let-AppR), is
    /// currently constructed only by tests, so there is no live reader to
    /// wire it to.  Keeping an unread payload is worse than not having it:
    /// the name promises memoized schemes, which invites reasoning about
    /// invalidation-on-promotion that does not apply to a plain set.
    ///
    /// If the checker starts emitting `Instance`, promote this back to a
    /// real `(var_id, region_id) -> TypeId` memo — that is the shape a
    /// let-polymorphic instantiation hot path wants.
    generalized_in_region: FxHashSet<(usize, usize)>,
    /// PG variables whose `s_inst_copy` was deferred because the resolved
    /// type still contained nested InferVars.  Retried after each outer
    /// iteration once those variables have been resolved, so promotion
    /// does not lose in-flight copies.
    pending_inst_copy: FxHashSet<usize>,
    /// Promotion shadow map: old_var_ty → promoted_ty. Replaces the
    /// non-rollbackable `ctx.set_binding` in `try_promote_var`, keeping
    /// promotion aliases local to the solver and fully rollback-safe.
    promotions: HashMap<TypeId, TypeId>,
    /// Records promotion chains so `check_rigid_escape` can follow
    /// old → new variable mappings after `unregister_var` nullifies
    /// `var_to_region` for the source variable.
    /// entry to `None`.
    var_promotions: HashMap<usize, usize>,
}

// ── Test accessors ────────────────────────────────────────────────
// These are only used by infer_tests.rs; they expose private fields
// to the test module as read-only references, preserving encapsulation
// while allowing test assertions on internal state.
#[cfg(test)]
impl InferenceContext {
    pub(crate) fn forward_refs(&self) -> &[Vec<usize>] {
        &self.forward_refs
    }
    pub(crate) fn reverse_refs(&self) -> &[Option<usize>] {
        &self.reverse_refs
    }
    pub(crate) fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }
    pub(crate) fn wait_lists(&self) -> &[Vec<(Constraint, Arc<SkolemEnv>)>] {
        &self.wait_lists
    }
    /// Test helper: the per-variable TypeId table (tests bind variables to
    /// concrete types via `ctx.set_binding(var_type_id_for_test(id), ..)`).
    pub(crate) fn var_type_id_for_test(&self, var_id: usize) -> TypeId {
        self.var_type_ids[var_id]
    }
    pub(crate) fn match_branches_len_for_test(&self) -> usize {
        self.match_branches.len()
    }
    /// Test helper: the UF root value for a var id (the migrated
    /// resolution state; `None` = unresolved).
    pub(crate) fn resolution_for_test(&mut self, var_id: usize) -> Option<TypeId> {
        self.union_find.probe_value(var_id)
    }
    /// Test helper: wake suspended constraints into a local heap and return
    /// the number of woken constraints.  Does not expose the internal
    /// PrioritizedConstraint type to tests.
    pub(crate) fn wake_var_for_test<'input>(
        &mut self,
        var_id: usize,
        ctx: &TypeContext<'input>,
    ) -> usize {
        let mut heap = BucketQueue::new();
        self.wake_var_incremental(var_id, &mut heap, ctx);
        heap.len()
    }
}

/// A set of pattern alternatives for a suspended match constraint.
#[derive(Debug, Clone)]
pub struct MatchBranchSet {
    /// The pattern label (e.g. "Arrow", "Tuple", "Coproduct", etc.).
    /// When the scrutinee's shape matches this, the branch is taken.
    pub shape_pattern: PrincipalShape,
    /// Continuation constraints to add when this branch matches.
    /// Shared via Arc so that `discharge_match` only clones a pointer,
    /// not the full constraint tree.
    pub continuation: Arc<[Constraint]>,
    /// Fallback constraints to add when no branch matches (else_).
    /// Used as a default when the shape cannot be determined uniquely.
    /// The system emits a diagnostic when else_ is triggered.
    pub else_continuation: Arc<[Constraint]>,
}

impl InferenceContext {
    /// Resolve a type through the scoped environment first, falling back to
    /// ctx.resolve_binding() for ordinary bindings.  Env-mapped (skolemized)
    /// variables are replaced with their Skolem TypeId so that all constraint
    /// forms (Eq, Sub, etc.) operate on the rigid type, not the raw InferVar.
    #[inline]
    fn resolve_with_env<'input>(
        &self,
        ty: TypeId,
        ctx: &TypeContext<'input>,
        env: &SkolemEnv,
    ) -> TypeId {
        let r = self.resolve_promotion_aware(ty, ctx);
        if let TypeData::InferVar { id, .. } = ctx.get(r)
            && let Some(skolem) = env.get(id)
        {
            return *skolem;
        }
        r
    }

    /// If `ty` resolves to an `InferVar` that is NOT bound by an enclosing
    /// `Forall` scope (`env`), return its variable id.  Returns `None` for
    /// concrete types, Skolem-bound variables, and variables already bound
    /// in `ctx.bindings`.
    ///
    /// NOTE: an earlier revision of this comment also described a GADT
    /// arm-exit abstraction (re-binding refinement-bound variables before
    /// cross-arm unification).  That mechanism was never implemented here —
    /// GADT refinements currently stay concrete across arm unification.
    fn free_infer_var_id<'input>(
        &self,
        ty: TypeId,
        ctx: &TypeContext<'input>,
        env: &SkolemEnv,
    ) -> Option<usize> {
        let r = self.resolve_promotion_aware(ty, ctx);
        match ctx.get(r) {
            TypeData::InferVar { id, .. } if !env.contains_key(id) => Some(*id),
            _ => None,
        }
    }

    pub fn new() -> Self {
        InferenceContext {
            type_vars: Vec::new(),
            var_type_ids: Vec::new(),
            constraints: Vec::new(),
            next_var_id: 0,
            region_tree: InferRegionTree::new(),
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
            wait_lists: Vec::new(),
            guard_sets: Vec::new(),
            gen_statuses: Vec::new(),
            shape_vars: ShapeVarContext::new(),
            match_branches: Vec::new(),
            forward_refs: Vec::new(),
            reverse_refs: Vec::new(),
            dirty_set: FxHashSet::default(),
            union_find: UnionFind::new(),
            var_origins: Vec::new(),
            generic_param_bindings: HashMap::default(),
            resolved_ids: Vec::new(),
            resolved_set: FxHashSet::default(),
            suspended_vars: FxHashSet::default(),
            discharged_matches: FxHashSet::default(),
            discharged_subs: FxHashSet::default(),
            pending_sub_vars: Vec::new(),
            pg_vars: FxHashSet::default(),
            wake_buffer: Vec::new(),
            undo_log: Vec::new(),
            snapshot_depth: 0,
            resolved_snapshot_stack: Vec::new(),
            region_current_snapshot_stack: Vec::new(),
            generalized_in_region: FxHashSet::default(),
            pending_inst_copy: FxHashSet::default(),
            promotions: HashMap::default(),
            var_promotions: HashMap::default(),
        }
    }

    pub fn new_type_var<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        kind: TypeVariableKind,
        origin: VarOrigin,
    ) -> TypeId {
        self.new_type_var_with_universe(ctx, kind, origin, 0)
    }

    /// Create a new type variable with a specific universe index.
    /// Currently, the universe is accepted but not stored — the body is
    /// identical to `new_type_var`.  When per-variable universe tracking
    /// is implemented, this function will store `universe` in the
    /// InferVar's metadata for correct HRTB scoping during canonical
    /// instantiation.
    pub fn new_type_var_with_universe<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        kind: TypeVariableKind,
        origin: VarOrigin,
        universe: usize,
    ) -> TypeId {
        let id = self.next_var_id;
        self.next_var_id += 1;
        self.push_undo(InferUndoLog::SetNextVarId(id));
        // Store the per-variable universe (rustc's CanonicalVarKind::Ty { ui }).
        let ty_id = ctx.alloc_infer_var(id, universe);
        self.union_find.resize_to(id + 1);
        if id >= self.var_origins.len() {
            self.var_origins.resize(id + 1, VarOrigin::Synthetic);
        }
        self.var_origins[id] = origin;
        let region_id = self.region_tree.current;
        self.type_vars.push(TypeVar {
            id,
            kind,
            gen_state: GenState::Ungeneralized,
            shape: PrincipalShape::Unknown,
            region_id,
        });
        self.push_undo(InferUndoLog::PushTypeVar);
        self.var_type_ids.push(ty_id);
        self.push_undo(InferUndoLog::PushVarTypeId);
        // Register the variable in the current region's pool
        self.region_tree.register_var(id);
        // Grow bounds vectors to match the new variable id
        let new_len = id + 1;
        self.lower_bounds.resize_with(new_len, Vec::new);
        self.upper_bounds.resize_with(new_len, Vec::new);
        self.wait_lists.resize_with(new_len, Vec::new);
        self.pending_sub_vars
            .resize_with(new_len, FxHashSet::default);
        self.guard_sets.resize_with(new_len, GuardSet::empty);
        self.gen_statuses
            .resize_with(new_len, || GenStatus::Ungeneralized);
        self.forward_refs.resize_with(new_len, Vec::new);
        self.reverse_refs.resize_with(new_len, || None);
        ty_id
    }

    /// Resolve a TypeId through inference variable bindings (TypeOrVar pattern).
    /// Follows the chain of resolutions until a concrete type is found.
    /// Checks the promotions shadow map first.
    ///
    /// Resolution state lives in `union_find`: each variable's class root
    /// carries the bound `TypeId` (var-var bindings, e.g. promotion, bind
    /// the root value to the new var's TypeId).  Reads are pure
    /// `try_probe_value(&self)` — classes stay single keys, so no pointer
    /// chasing and no cycle is possible beyond the promotions map (still
    /// bounded by `max_steps`).
    pub fn resolve<'input>(&self, ty: TypeId, ctx: &TypeContext<'input>) -> TypeId {
        let mut current = ty;
        let mut steps = 0usize;
        // The chain alternates promotion edges (≤ promotions.len()) and UF
        // root-value edges (each var binds at most one value, so ≤ table
        // size hops before repeating a node).  Exceeding this bound means
        // a genuine cycle (solver bug), not user input.
        let max_steps = self.promotions.len() + self.union_find.len() + 1;
        loop {
            steps += 1;
            debug_assert!(
                steps <= max_steps,
                "resolve: cycle detected in promotion chain after {steps} steps"
            );
            if steps > max_steps {
                return current; // fail-safe in release
            }
            // Check promotions shadow map first.
            if let Some(&promoted) = self.promotions.get(&current) {
                current = promoted;
                continue;
            }
            let TypeData::InferVar { id, .. } = ctx.get(current) else {
                return current;
            };
            let Some(resolved) = self.union_find.probe_read(*id) else {
                return current;
            };
            if resolved == current {
                return current;
            }
            current = resolved;
        }
    }

    /// Resolve through the promotions shadow map first, then fall back to
    /// `ctx.resolve_binding`.  Required in wake_var_incremental,
    /// force_generalize_for_regions, unicity_check, and other sites where
    /// a promoted variable's ID may differ from the one stored in pc.env.
    /// and anywhere the result feeds into shape/guard/generalization decisions.
    fn resolve_promotion_aware<'input>(&self, ty: TypeId, ctx: &TypeContext<'input>) -> TypeId {
        let mut current = ty;
        let mut steps = 0usize;
        // Bound the fixed-point iteration.  Each of the three edge kinds
        // below can fire at most (len) times before repeating a node:
        // promotions (≤ promotions.len()), ctx bindings, and UF root-value
        // edges (≤ union_find.len()).  The sum plus one is a sound upper
        // bound: exceeding it means a cycle has formed in one of the maps
        // (a solver bug, not user input).
        let max_steps = self.promotions.len() + self.union_find.len() + 1;
        loop {
            steps += 1;
            if steps > max_steps {
                break; // fail-safe: return the best answer we have
            }
            let mut next = current;
            // 1. Promotions shadow map — solver-local, invisible to `ctx`.
            if let Some(&promoted) = self.promotions.get(&next) {
                next = promoted;
            }
            // 2. Global bindings maintained by `TypeContext::unify`.
            let bound = ctx.resolve_binding(next);
            if bound != next {
                next = bound;
            }
            // 3. Solver-local resolutions recorded by `unify`/`unify_and_track`.
            //    These mirror (2) in the common case, but a bare `ctx.unify`
            //    elsewhere, or a resolution written without a ctx bind, makes
            //    them diverge — both edges must be followed or the unified type
            //    is silently lost.
            if let TypeData::InferVar { id, .. } = ctx.get(next)
                && let Some(r) = self.union_find.probe_read(*id)
                && r != next
            {
                next = r;
            }
            if next == current {
                break;
            }
            current = next;
        }
        current
    }

    /// Promotion-aware priority computation.  `Constraint::priority` is a
    /// free method that can only see `ctx.resolve_binding`, so a variable
    /// promoted by `try_promote_var` (aliased ONLY in the solver-local
    /// `promotions` map) still classifies as `InferVar` there — giving the
    /// constraint the wrong priority band and delaying/deferring
    /// constraints that are in fact fully determined.  This method runs
    /// the same classification through `resolve_promotion_aware`.
    fn constraint_priority<'input>(&self, c: &Constraint, ctx: &TypeContext<'input>) -> u8 {
        let is_infer = |ty: TypeId| -> bool {
            let r = self.resolve_promotion_aware(ty, ctx);
            matches!(ctx.get(r), TypeData::InferVar { .. })
        };
        match c {
            Constraint::Eq(a, b, _, _) => {
                let ai = is_infer(*a);
                let bi = is_infer(*b);
                match (ai, bi) {
                    (false, false) => PRIORITY_EQ_CONCRETE,
                    (true, false) | (false, true) => PRIORITY_EQ_MIXED,
                    (true, true) => PRIORITY_EQ_INFER,
                }
            }
            Constraint::Sub(sub, sup, _) => {
                let si = is_infer(*sub);
                let pi = is_infer(*sup);
                match (si, pi) {
                    (false, false) => PRIORITY_SUB_CONCRETE,
                    (true, true) => PRIORITY_SUB_INFER,
                    _ => PRIORITY_SUB_MIXED,
                }
            }
            Constraint::Match { scrutinee, .. } => {
                if is_infer(*scrutinee) {
                    PRIORITY_MATCH_SUSPENDED
                } else {
                    PRIORITY_MATCH_RESOLVED
                }
            }
            Constraint::Exists { .. } => PRIORITY_EXISTS,
            Constraint::Forall { .. } => PRIORITY_FORALL,
            Constraint::Instance { .. } => PRIORITY_INSTANCE,
            Constraint::Let { .. } => PRIORITY_LET,
            Constraint::ExitRegion { .. } => PRIORITY_ADMIN,
            Constraint::LetBarrier { .. } => PRIORITY_LET_BARRIER,
        }
    }

    /// Unify two types via the global `TypeContext<'input>` AND mirror the
    /// post-unification resolution into `union_find` so that `self.resolve()`
    /// stays consistent with global bindings.
    ///
    /// This must be used instead of bare `ctx.unify()` inside the solver loop
    /// whenever one of the sides is an InferVar owned by this context.
    ///
    /// Universe escape (higher-ranked type) checking is performed inside
    /// `TypeContext::unify` via `ty_contains_foreign_universe`: if the
    /// non-InferVar side contains a variable whose universe exceeds the
    /// InferVar's universe, the bind is rejected.  This covers both the
    /// `(InferVar, _)` and `(_, InferVar)` branches symmetrically, so no
    /// additional check is needed at this call site.
    pub(crate) fn unify_and_track<'input>(
        &mut self,
        a: TypeId,
        b: TypeId,
        ctx: &mut TypeContext<'input>,
        span: Span,
    ) -> Result<TypeId, TypeError> {
        let result = match ctx.unify(a, b) {
            Ok(ty) => ty,
            Err(mut e) => {
                e.set_span(span);
                return Err(e);
            }
        };
        // After unification, mirror the sides' post-unification resolution
        // into union_find so that self.resolve() follows the same chain
        // as ctx.resolve_binding().  `ctx.unify(InferVar, X)` binds the var
        // and returns X — the returned type itself is NOT an InferVar, so
        // only inspecting `result` previously recorded nothing and
        // self.resolve() (which never consults ctx.bindings) desynced from
        // the global unifier.  Mirror BOTH sides: whichever side resolved
        // to something else gets its root value bound.  The bind is
        // undo-logged inside UnionFind (ena SetElem) while a snapshot is
        // open — no per-entry InferUndoLog needed.
        for side in [a, b] {
            let resolved_side = ctx.resolve_binding(side);
            if resolved_side == side {
                continue; // unbound (or a non-var side that won the binding)
            }
            let TypeData::InferVar { id, .. } = ctx.get(side) else {
                continue;
            };
            if self.union_find.probe_read(*id).is_some() {
                continue; // already tracked
            }
            // Last-write-wins overwrite; undo via ena SetElem.
            self.union_find.bind_value(*id, resolved_side);
        }
        Ok(result)
    }

    /// Unify with local InferVar resolution (TypeOrVar pattern).
    /// Records resolutions in `union_find` instead of global bindings.
    pub fn unify<'input>(
        &mut self,
        a: TypeId,
        b: TypeId,
        ctx: &mut TypeContext<'input>,
        span: Span,
    ) -> Result<TypeId, TypeError> {
        let ra = self.resolve(a, ctx);
        let rb = self.resolve(b, ctx);
        if ra == rb {
            return Ok(ra);
        }
        match (ctx.get(ra), ctx.get(rb)) {
            (TypeData::InferVar { id, universe, .. }, _) => {
                // Per-variable universe tracking: binding a HIGHER-universe
                // variable into a LOWER universe lets it escape — reject.
                // (rustc `can_define`: binding is legal iff
                // `max_universe(binding_ty) <= var.universe`; only a
                // strictly-higher binding universe escapes — a lower one
                // is legal.)
                if let TypeData::InferVar { universe: bu, .. } = ctx.get(rb)
                    && *bu > *universe
                {
                    return Err(TypeError::Mismatch {
                        expected: ra,
                        found: rb,
                        span,
                    });
                }
                if ty_contains_foreign_universe(ctx, rb, *universe) {
                    return Err(TypeError::Mismatch {
                        expected: ra,
                        found: rb,
                        span,
                    });
                }
                // Occurs-check: `a := rb` where `rb`
                // contains `a` would create a cyclic binding.
                if ctx.occurs_check(a, rb) {
                    return Err(TypeError::Mismatch {
                        expected: ra,
                        found: rb,
                        span,
                    });
                }
                // Bind the var's UF root value to `rb` (last-write-wins).
                // Undo is handled by ena's SetElem log while a snapshot is open.
                self.union_find.bind_value(*id, rb);
                Ok(rb)
            }
            (_, TypeData::InferVar { id, universe, .. }) => {
                if let TypeData::InferVar { universe: au, .. } = ctx.get(ra)
                    && *au > *universe
                {
                    return Err(TypeError::Mismatch {
                        expected: ra,
                        found: rb,
                        span,
                    });
                }
                // Symmetric recursive universe check:
                // the first arm checks `rb`; `ra` may be a composite type
                // containing a foreign-universe InferVar).
                if ty_contains_foreign_universe(ctx, ra, *universe) {
                    return Err(TypeError::Mismatch {
                        expected: ra,
                        found: rb,
                        span,
                    });
                }
                // Occurs-check: binding `b := ra` where `ra` contains `b`
                // would create a cyclic binding (resolve would loop).
                if ctx.occurs_check(rb, ra) {
                    return Err(TypeError::Mismatch {
                        expected: ra,
                        found: rb,
                        span,
                    });
                }
                // Bind the var's UF root value to `ra` (last-write-wins).
                // Undo is handled by ena's SetElem log while a snapshot is open.
                self.union_find.bind_value(*id, ra);
                Ok(ra)
            }
            _ => match ctx.unify(ra, rb) {
                Ok(ty) => Ok(ty),
                Err(mut e) => {
                    e.set_span(span);
                    Err(e)
                }
            },
        }
    }

    /// Look up the kind of a type variable by its id.
    /// `type_vars` is indexed by var id (sequential allocation in
    /// `new_type_var`, only appended/popped on rollback), so this is an
    /// O(1) index lookup — not a linear scan.
    pub fn get_var_kind(&self, id: usize) -> Option<TypeVariableKind> {
        self.type_vars.get(id).map(|tv| {
            debug_assert_eq!(tv.id, id, "type_vars must be indexed by var id");
            tv.kind
        })
    }

    /// Get the region level of a type variable by its id.
    /// O(1) index lookup (see `get_var_kind`).
    pub fn get_var_level(&self, id: usize) -> Option<usize> {
        self.type_vars.get(id).map(|tv| {
            debug_assert_eq!(tv.id, id, "type_vars must be indexed by var id");
            self.region_tree.get_level(tv.region_id)
        })
    }

    /// Enter a deeper typing scope (let/forall/region).
    /// Creates a new child region in the region tree.
    /// Returns the previous InferRegionId so caller can restore it.
    pub fn enter_level(&mut self) -> InferRegionId {
        self.region_tree.enter_region()
    }

    /// Exit the current typing scope, restoring the previous region.
    /// Finalizes the post-order `tout` stamps for every region exited on
    /// the way back to `prev_region` — `exit_region` is the ONLY writer
    /// of `tout`, and production previously used only `exit_level`, which
    /// just rewound `current` without assigning stamps.  Every non-root
    /// `tout` stayed at the placeholder 0, degenerating the O(1)
    /// ancestor interval test to a pre-order comparison (any region
    /// entered earlier — including siblings — passed as "ancestor").
    pub fn exit_level(&mut self, prev_region: InferRegionId) {
        assert!(
            self.region_tree
                .is_ancestor(prev_region, self.region_tree.current),
            "exit_level: prev_region must be an ancestor of the current region (caller bug)"
        );
        while self.region_tree.current != prev_region {
            self.region_tree.exit_region();
        }
    }

    /// Try to promote a variable to the target region's scope
    /// by creating a new variable at the target region and unifying.
    /// Uses the region tree's ancestor check (OmniML §6 PR-UVARPR).
    pub fn try_promote_var<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        var_id: usize,
        target_region: InferRegionId,
    ) -> Option<TypeId> {
        // Follow the promotion chain first: `promotions` is solver-local
        // (invisible to ctx.resolve_binding), so callers resolve the *ctx*
        // chain and can hand us a var_id that was already promoted.  The
        // stale entry's region still reports the ORIGINAL region, and
        // re-promoting would mint a second fresh variable whose
        // `promotions[var_ty]` write DISCARDS the first promotion —
        // orphaning every wait list / guard / bound / forward ref
        // installed on it.  Forward to the current representative instead
        // (mirrors the "resolve first" discipline of resolve_promotion_aware).
        let var_ty = self.var_type_ids[var_id];
        let current = self.resolve_promotion_aware(var_ty, ctx);
        if let TypeData::InferVar { id: current_id, .. } = ctx.get(current) {
            if *current_id != var_id {
                return self.try_promote_var(ctx, *current_id, target_region);
            }
        }

        let var_region = self.type_vars.get(var_id)?.region_id;

        // If the variable's region is already an ancestor of (or equal to)
        // the target region, it is already at an outer/equal scope — no
        // promotion needed. Otherwise the variable is deeper and needs
        // promotion to the target region.
        if self.region_tree.is_ancestor(var_region, target_region) || var_region == target_region {
            return Some(self.var_type_ids[var_id]);
        }
        // Promotion needed: create a new variable at the target region.
        // Inherit kind and origin from the original variable.
        let kind = self.type_vars[var_id].kind;
        let origin = self
            .var_origins
            .get(var_id)
            .copied()
            .unwrap_or(VarOrigin::Synthetic);
        let new_ty_id = self.new_type_var(ctx, kind, origin);
        let new_id = self.next_var_id - 1;
        let current_region = self.region_tree.current;
        self.region_tree.unregister_var(new_id, current_region);
        self.region_tree
            .register_var_in_region(new_id, target_region);
        self.push_undo(InferUndoLog::SetVarRegion(
            new_id,
            self.type_vars[new_id].region_id,
        ));
        self.type_vars[new_id].region_id = target_region;
        if var_id < self.type_vars.len() {
            self.type_vars[new_id].shape = self.type_vars[var_id].shape;
            self.type_vars[new_id].gen_state = self.type_vars[var_id].gen_state;
        }

        // ── Transfer all suspended-constraint state from the old variable
        // to the new one (OmniML §3.2, §6). Without this transfer, constraints
        // suspended on the old variable are silently lost because the old
        // variable's resolution points to the new InferVar — which is never
        // considered "resolved" (it's still an InferVar) and thus never
        // triggers wake-up. This would drop Match, Impl, and any other
        // constraints queued via suspend_on_var.
        if var_id < self.wait_lists.len() && new_id < self.wait_lists.len() {
            // Snapshot both sides before mutation (lazy — clone only when a
            // snapshot is actually open; push_undo is a no-op otherwise).
            if self.snapshot_depth > 0 {
                self.push_undo(InferUndoLog::TransferWaitList {
                    from: var_id,
                    to: new_id,
                    old_from: self.wait_lists[var_id].clone(),
                    old_to: self.wait_lists[new_id].clone(),
                });
            }
            let old_wait = std::mem::take(&mut self.wait_lists[var_id]);
            self.wait_lists[new_id].extend(old_wait);
            // Transfer the pending-Sub dedup sets: wait list entries moved
            // above keep their dedup keys in the OLD var's set, so without
            // this a re-pushed identical Sub passes dedup on the new var
            // and double-suspends.
            if var_id < self.pending_sub_vars.len() && new_id < self.pending_sub_vars.len() {
                if self.snapshot_depth > 0 {
                    self.push_undo(InferUndoLog::TransferPendingSubVars {
                        from: var_id,
                        to: new_id,
                        old_from: self.pending_sub_vars[var_id].clone(),
                        old_to: self.pending_sub_vars[new_id].clone(),
                    });
                }
                let old_pending = std::mem::take(&mut self.pending_sub_vars[var_id]);
                self.pending_sub_vars[new_id].extend(old_pending);
            }
            // Transfer suspended-vars tracking: the old var is now an alias and
            // will not be checked by the wake loop; the new var carries the
            // constraints and must be tracked instead.
            if self.suspended_vars.remove(&var_id) {
                self.suspended_vars.insert(new_id);
            }
        }
        if var_id < self.guard_sets.len() && new_id < self.guard_sets.len() {
            if self.snapshot_depth > 0 {
                self.push_undo(InferUndoLog::TransferGuards {
                    from: var_id,
                    to: new_id,
                    old_from: self.guard_sets[var_id].clone(),
                    old_to: self.guard_sets[new_id].clone(),
                });
            }
            let old_guards = std::mem::take(&mut self.guard_sets[var_id]);
            self.guard_sets[new_id] = self.guard_sets[new_id].union(&old_guards);
        }
        if var_id < self.gen_statuses.len() && new_id < self.gen_statuses.len() {
            let old_status = self.gen_statuses[var_id];
            if old_status != GenStatus::Ungeneralized {
                self.write_gen_status(new_id, old_status);
            }
        }
        if var_id < self.lower_bounds.len()
            && new_id < self.lower_bounds.len()
            && var_id < self.upper_bounds.len()
            && new_id < self.upper_bounds.len()
        {
            if self.snapshot_depth > 0 {
                self.push_undo(InferUndoLog::TransferBounds {
                    from: var_id,
                    to: new_id,
                    old_from_lb: self.lower_bounds[var_id].clone(),
                    old_to_lb: self.lower_bounds[new_id].clone(),
                    old_from_ub: self.upper_bounds[var_id].clone(),
                    old_to_ub: self.upper_bounds[new_id].clone(),
                });
            }
            let old_lbs = std::mem::take(&mut self.lower_bounds[var_id]);
            self.lower_bounds[new_id].extend(old_lbs);
            let old_ubs = std::mem::take(&mut self.upper_bounds[var_id]);
            self.upper_bounds[new_id].extend(old_ubs);
        }

        // ── Forward references (OmniML §5.2) ──────────────────────
        // If the old variable was a PG var with instances, transfer
        // those instances to the new variable and update their reverse
        // references to point to the new variable.
        if var_id < self.forward_refs.len() && new_id < self.forward_refs.len() {
            // Snapshot forward and reverse refs before mutation (lazy).
            if self.snapshot_depth > 0 {
                let mut old_reverse: Vec<(usize, Option<usize>)> = Vec::new();
                let old_fwd = self.forward_refs[var_id].clone();
                for &inst_id in &old_fwd {
                    if inst_id < self.reverse_refs.len() {
                        old_reverse.push((inst_id, self.reverse_refs[inst_id]));
                    }
                }
                let old_rev_of_old = if var_id < self.reverse_refs.len() {
                    self.reverse_refs[var_id]
                } else {
                    None
                };
                old_reverse.push((var_id, old_rev_of_old));

                self.push_undo(InferUndoLog::TransferRefs {
                    from: var_id,
                    to: new_id,
                    old_from_fwd: self.forward_refs[var_id].clone(),
                    old_to_fwd: self.forward_refs[new_id].clone(),
                    old_reverse,
                });
            }

            let old_fwd = std::mem::take(&mut self.forward_refs[var_id]);
            self.forward_refs[new_id].extend(old_fwd);
            for &inst_id in &self.forward_refs[new_id] {
                if inst_id < self.reverse_refs.len() {
                    self.reverse_refs[inst_id] = Some(new_id);
                }
            }
        }
        // If the old variable was itself an instance of a PG var,
        // transfer that relationship to the new variable and update
        // the PG var's forward_reference list.
        if var_id < self.reverse_refs.len() {
            let old_reverse = self.reverse_refs[var_id];
            self.reverse_refs[new_id] = old_reverse;
            self.reverse_refs[var_id] = None; // old var is now just an alias
            if let Some(pg_id) = old_reverse
                && pg_id < self.forward_refs.len()
                && let Some(pos) = self.forward_refs[pg_id].iter().position(|&r| r == var_id)
            {
                self.forward_refs[pg_id][pos] = new_id;
            }
        }

        // Bind the old variable to the new one (promotion): the old var's
        // UF root value now points at `new_ty_id`, so every resolve family
        // member follows the promotion edge.  Last-write-wins matches the
        // old unconditional `resolutions[var_id] = Some(new_ty_id)`.  Undo
        // is ena SetElem while a snapshot is open.
        self.union_find.bind_value(var_id, new_ty_id);
        // Bind in the solver-local promotions shadow map instead
        // of the non-rollbackable ctx.set_binding. This keeps promotion
        // aliases local to the solver and fully rollback-safe.
        let var_ty = self.var_type_ids[var_id];
        self.promotions.insert(var_ty, new_ty_id);
        self.push_undo(InferUndoLog::RemovePromotion(var_ty));
        // Track var_id → new_id so check_rigid_escape can follow the
        // promotion chain when looking up var_to_region.
        self.var_promotions.insert(var_id, new_id);
        self.push_undo(InferUndoLog::RemoveVarPromotion(var_id));
        Some(new_ty_id)
    }

    pub fn add_constraint(&mut self, c: Constraint) {
        self.constraints.push(c);
        // Logged for rollback symmetry with the internal fallback path in
        // `suspend_on_var_with_env` (which pushes its own PushConstraint).
        self.push_undo(InferUndoLog::PushConstraint);
    }

    /// OmniML-inspired: suspend a constraint on the target InferVar id.
    /// When the var is bound, the constraint will be woken and reprocessed.
    /// Also marks the variable as PartiallyGeneralizable (PG) and adds a
    /// guard entry so the variable stays PG until the guard is released.
    /// Uses an empty SkolemEnv — constraints suspended through the public
    /// API are not inside a Forall scope.
    pub fn suspend_on_var(&mut self, c: Constraint, var_id: usize) {
        self.suspend_on_var_with_env(c, empty_skolem_env(), var_id)
    }

    /// Internal variant: suspend a constraint with a scoped SkolemEnv so
    /// that Forall-bound rigid variables remain rigid after suspension/wake.
    fn suspend_on_var_with_env(&mut self, c: Constraint, env: Arc<SkolemEnv>, var_id: usize) {
        if var_id < self.wait_lists.len() {
            self.wait_lists[var_id].push((c, env));
            // Pair the suspension with its undo entry BEFORE the guard is
            // logged: reverse restores the guard first, then pops the
            // suspension — a rolled-back var is neither guarded nor armed.
            self.push_undo(InferUndoLog::PushWaitListEntry(var_id));
            // Track this variable so the wake loop knows to check it.
            self.suspended_vars.insert(var_id);
            // Add a guard: the variable is now blocked until this constraint
            // is woken and processed.  This is essential for the PG→G lifecycle
            // (OmniML §6).  Uses the reference-counted GuardSet — it also
            // marks the variable PartiallyGeneralizable (PG), so no separate
            // status update is needed here.
            self.add_guard(var_id);
        } else {
            // Invariant violation: `wait_lists` is grown for every created
            // InferVar (extend_infer_vars → resize_with), so a var_id out of
            // range means the constraint generator is broken.  Pushing into
            // `self.constraints` here would SILENTLY LOSE the constraint:
            // solve() takes the initial batch with mem::take before the
            // loop, and nothing drains this vector again.  Fail loudly
            // instead (unwind keeps the query poison path intact).
            panic!(
                "suspend_on_var_with_env: var_id {} out of range (wait_lists.len = {})",
                var_id,
                self.wait_lists.len(),
            );
        }
    }

    /// Extract the InferVar id from a constraint (if any side is an unresolved InferVar).
    /// Returns None if both sides are concrete.
    fn infer_var_from_constraint<'input>(
        &self,
        c: &Constraint,
        ctx: &TypeContext<'input>,
    ) -> Option<usize> {
        match c {
            Constraint::Eq(a, b, _, _) => {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                if let TypeData::InferVar { id, .. } = ctx.get(ra) {
                    return Some(*id);
                }
                if let TypeData::InferVar { id, .. } = ctx.get(rb) {
                    return Some(*id);
                }
                None
            }
            Constraint::Sub(sub, sup, _) => {
                let rs = ctx.resolve_binding(*sub);
                let rsup = ctx.resolve_binding(*sup);
                if let TypeData::InferVar { id, .. } = ctx.get(rs) {
                    return Some(*id);
                }
                if let TypeData::InferVar { id, .. } = ctx.get(rsup) {
                    return Some(*id);
                }
                None
            }
            Constraint::Match { scrutinee, .. } => {
                let r = ctx.resolve_binding(*scrutinee);
                if let TypeData::InferVar { id, .. } = ctx.get(r) {
                    Some(*id)
                } else {
                    None
                }
            }
            // ∃ binds a FLEXIBLE var, but its handler manages region
            // entry/exit itself and the var it binds is not "the InferVar
            // to suspend on" — mirror the rigid Forall case.  (This helper
            // currently has no callers; kept consistent so a future use
            // cannot silently suspend an Exists on its own bound var.)
            Constraint::Exists { .. } => None,
            Constraint::Forall { .. } => None, // rigid var — not an infer var
            Constraint::Instance {
                instantiation_ty, ..
            } => {
                let r = ctx.resolve_binding(*instantiation_ty);
                if let TypeData::InferVar { id, .. } = ctx.get(r) {
                    Some(*id)
                } else {
                    None
                }
            }
            Constraint::Let { .. } => None, // structural — no single infer var
            Constraint::ExitRegion { .. } => None, // administrative — no infer var
            Constraint::LetBarrier { .. } => None, // administrative — no infer var
        }
    }

    /// Determine the principal shape of a resolved type.
    pub fn shape_of_type<'input>(ctx: &TypeContext<'input>, ty: TypeId) -> PrincipalShape {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved) {
            TypeData::Fn { params, .. } => PrincipalShape::Arrow,
            TypeData::Tuple { elems } => PrincipalShape::Tuple(elems.len()),
            TypeData::Adt { def_id, args, .. } => PrincipalShape::Constructor(def_id.0, args.len()),
            TypeData::Forall { .. }
            | TypeData::Exists { .. }
            | TypeData::Poly { .. }
            | TypeData::SkolemVar { .. } => PrincipalShape::Poly,
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::USize
            | TypeData::Rational { .. } => PrincipalShape::Integer,
            TypeData::Float { .. } => PrincipalShape::Float,
            TypeData::Bool => PrincipalShape::Bool,
            TypeData::Char | TypeData::Byte => PrincipalShape::Char,
            TypeData::InferVar { .. } | TypeData::GenericParam { .. } => PrincipalShape::Var,
            _ => PrincipalShape::Unknown,
        }
    }

    /// Try to set the shape of a variable from its resolved type.
    /// Returns true if the shape was updated.
    fn try_set_shape<'input>(&mut self, var_id: usize, ctx: &TypeContext<'input>) -> bool {
        if var_id < self.type_vars.len() && var_id < self.var_type_ids.len() {
            let ty = self.resolve_promotion_aware(self.var_type_ids[var_id], ctx);
            if !matches!(ctx.get(ty), TypeData::InferVar { .. }) {
                let new_shape = Self::shape_of_type(ctx, ty);
                if self.type_vars[var_id].shape != new_shape {
                    self.type_vars[var_id].shape = new_shape;
                    return true;
                }
            }
        }
        false
    }

    /// Incrementally wake constraints for a resolved variable.
    /// Woken constraints are enqueued directly onto the heap.
    /// After waking, if the wait list is empty AND no guards remain,
    /// the variable can be re-generalised (G) — OmniML §6.
    fn wake_var_incremental<'input>(
        &mut self,
        var_id: usize,
        heap: &mut BucketQueue,
        ctx: &TypeContext<'input>,
    ) {
        if var_id < self.wait_lists.len() && !self.wait_lists[var_id].is_empty() {
            // Log wait_list and guard_set before mutation so rollback restores them.
            if self.snapshot_depth > 0 {
                self.push_undo(InferUndoLog::SetGuardSet(
                    var_id,
                    self.guard_sets[var_id].clone(),
                ));
                self.push_undo(InferUndoLog::TakeWaitList(
                    var_id,
                    self.wait_lists[var_id].clone(),
                ));
            }
            let suspended = std::mem::take(&mut self.wait_lists[var_id]);
            // Clear the per-variable Sub dedup set: all entries were just taken,
            // so no Sub is "pending" on this variable anymore.  A re-suspension
            // from the heap will re-add via the dedup guard below.
            if var_id < self.pending_sub_vars.len() {
                self.pending_sub_vars[var_id].clear();
            }
            for (c, env) in suspended {
                let p = self.constraint_priority(&c, ctx);
                heap.push(PrioritizedConstraint {
                    priority: p,
                    constraint: c,
                    env,
                });
            }
            // Clear the DIRECT guards for this variable: every constraint
            // placed in the wait list by `suspend_on_var` had a direct
            // guard added; we must release those guards now that the
            // constraints have been woken to the heap (OmniML §6
            // confirmation: "once a suspended match constraint is solved,
            // it removes the guards it introduced").  TRANSITIVE guards
            // stay: they come from cross-region dependency tracking
            // (`GuardSet::union`) and a sibling region may still depend on
            // this variable — clearing them would allow premature
            // generalization.
            if var_id < self.guard_sets.len() {
                self.guard_sets[var_id].clear_direct_guards();
            }
            // All constraints woken and guards cleared — transition to
            // Generalized if no further guards remain AND no rigid escape.
            if var_id < self.gen_statuses.len()
                && self.gen_statuses[var_id] == GenStatus::PartiallyGeneralizable
            {
                let guards_empty =
                    var_id < self.guard_sets.len() && self.guard_sets[var_id].is_empty();
                let wl_empty = var_id < self.wait_lists.len() && self.wait_lists[var_id].is_empty();
                if guards_empty && wl_empty {
                    let resolved = self.resolve_promotion_aware(self.var_type_ids[var_id], ctx);
                    // Only generalize a *resolved* variable.
                    // When resolved is still InferVar, the old conjunction
                    // `!matches!(..InferVar) && ...` was false, making
                    // `!escapes == true`, silently promoting an unbound var.
                    if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                        let level = self
                            .type_vars
                            .get(var_id)
                            .map(|v| self.region_tree.get_level(v.region_id))
                            .unwrap_or(0);
                        if !Self::check_rigid_escape(
                            ctx,
                            resolved,
                            level,
                            &self.region_tree,
                            &self.var_promotions,
                        ) {
                            self.write_gen_status(var_id, GenStatus::Generalized);
                        }
                    }
                    // If still unresolved, leave as PG: a later wake will retry.
                }
            }
            // Remove from suspended set: either a new suspension will re-add it,
            // or the variable has no more pending work.
            self.suspended_vars.remove(&var_id);
        }
    }

    // ── OmniML: Match branches ───────────────────────────────────

    /// Register a set of match branch patterns. Returns a `(start, count)`
    /// pair identifying the range of branches in `self.match_branches`.
    /// `discharge_match` uses the exact range so later-registered branch sets
    /// are never accidentally scanned.
    pub fn register_match_branches(&mut self, branches: Vec<MatchBranchSet>) -> (usize, usize) {
        let start = self.match_branches.len();
        let count = branches.len();
        self.match_branches.reserve(count);
        self.match_branches.extend(branches);

        // Batch mutation, batch undo: the payload is the pre-extend
        // length so a rollback restores the table exactly (a single
        // per-element pop would leave count − 1 stale branch sets that
        // the next registration would alias — cross-contamination).
        self.push_undo(InferUndoLog::PushMatchBranches(start));
        (start, count)
    }

    /// Try to discharge a Match constraint using a shape variable.
    /// If the scrutinee has a shape variable and it's resolved, discharge
    /// immediately.  If not, register a callback on the shape variable
    /// so the match fires when the shape becomes known.
    /// Returns `true` if the match was handled (either discharged or
    /// registered for later).
    pub fn try_match_via_shape_var<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        scrutinee: TypeId,
        branches_id: (usize, usize),
        heap: &mut BucketQueue,
        span: crate::ast::Span,
        env: &Arc<SkolemEnv>,
    ) -> bool {
        // Use env-aware resolution so that Forall-bound Skolems are not
        // treated as free inference variables.
        let resolved = self.resolve_with_env(scrutinee, ctx, env);
        match ctx.get(resolved) {
            TypeData::InferVar { id, .. } => {
                // Always suspend the Match constraint, regardless of whether
                // this variable already has guards.  A non-empty guard set
                // from a *previous* Match cannot repel a *new* Match with a
                // potentially different branches_id — silently dropping it
                // would lose entire branch sets.
                let match_c = Constraint::Match {
                    scrutinee,
                    branches_id,
                    span,
                };
                // Suspend on this variable only.  Preserve the scoped SkolemEnv
                // so that Forall-bound rigid variables remain rigid after wake.
                self.suspend_on_var_with_env(match_c, Arc::clone(env), *id);
                true
            }
            _ => {
                // Propagate the result — if discharge_match fails (no branch
                // matches and no else_ fallback), the caller falls through to
                // unicity check and re-push logic instead of silently losing
                // the constraint.
                self.discharge_match(ctx, scrutinee, branches_id, heap, env)
            }
        }
    }

    // ── OmniML: Contextual Unicity C[τ!ξ] ────────────────────────
    //
    // From O'Brien, Rémy & Scherer §4.1:
    //   C[τ!ζ] iff ∀φ, φ ⊢ [C[τ = g]] ⇒ shape(g) = ζ
    //
    // Three syntactic rules (decidable approximation):
    //   UNI-TYPE: τ is non-variable → shape(τ) = ξ
    //   UNI-VAR:  τ = α and ∃ equalities α = τ' where τ' is non-variable
    //   UNI-BACKPROP: τ = α and all instances of α share shape ξ

    /// Check whether a type has a unique shape determined by the
    /// constraint context. Returns `Some(shape)` if unicity holds,
    /// `None` if the shape cannot be uniquely determined.
    ///
    /// Implements the ⊆-closed erasure semantics:
    ///   C[τ!ζ] iff ∀φ, φ ⊢ [C[τ = g]] ⇒ shape(g) = ζ
    /// where [C] erases all SuspendedMatch constraints to true.
    pub fn unicity_check<'input>(
        &self,
        ctx: &TypeContext<'input>,
        ty: TypeId,
        env: &SkolemEnv,
        active_constraints: &[PrioritizedConstraint],
    ) -> Option<PrincipalShape> {
        self.unicity_check_with_index(ctx, ty, env, active_constraints, None)
    }

    /// Build an index mapping each InferVar ID to the list of (resolved_a, resolved_b)
    /// pairs from Eq constraints that involve that variable.  This turns the O(n)
    /// linear scan in UNI-VAR into an O(1) HashMap lookup per variable.
    fn build_eq_index<'input>(
        &self,
        ctx: &TypeContext<'input>,
        active_constraints: &[PrioritizedConstraint],
    ) -> HashMap<usize, Vec<(TypeId, TypeId)>> {
        let mut index: HashMap<usize, Vec<(TypeId, TypeId)>> = HashMap::default();
        for pc in active_constraints {
            if let Constraint::Eq(a, b, _, _) = &pc.constraint {
                // resolve_with_env (not the ctx-only resolver) so the index
                // follows solver-local promotions: try_promote_var binds
                // aliases ONLY in `promotions`/`resolutions`, never in
                // ctx.bindings, so a ctx.resolve_binding-based index records
                // keys under stale pre-promotion ids and `build_multi_equation`
                // (which resolves promotion-aware) misses every such equality —
                // Det-Dom/Det-Esc spuriously fail.
                let ra = self.resolve_with_env(*a, ctx, &pc.env);
                let rb = self.resolve_with_env(*b, ctx, &pc.env);
                if let TypeData::InferVar { id, .. } = ctx.get(ra) {
                    index.entry(*id).or_default().push((ra, rb));
                }
                if let TypeData::InferVar { id, .. } = ctx.get(rb) {
                    index.entry(*id).or_default().push((rb, ra));
                }
            }
        }
        index
    }

    /// Core unicity check with optional pre-built Eq index.
    /// When `eq_index` is provided, UNI-VAR uses O(1) lookup instead of O(n) scan.
    fn unicity_check_with_index<'input>(
        &self,
        ctx: &TypeContext<'input>,
        ty: TypeId,
        env: &SkolemEnv,
        active_constraints: &[PrioritizedConstraint],
        eq_index: Option<&HashMap<usize, Vec<(TypeId, TypeId)>>>,
    ) -> Option<PrincipalShape> {
        let resolved = self.resolve_with_env(ty, ctx, env);
        let data = ctx.get(resolved);

        // ── UNI-TYPE: non-variable type ──────────────────────────
        // If τ is already resolved to a concrete type, its shape is known.
        if !matches!(data, TypeData::InferVar { .. }) {
            return Some(Self::shape_of_type(ctx, resolved));
        }

        // τ is an InferVar — extract its id.
        let var_id = match data {
            TypeData::InferVar { id, .. } => *id,
            _ => return None,
        };

        // ── UNI-VAR: α is unified with a concrete type ───────────
        // If a pre-built Eq index is available, use O(1) lookup.
        // Otherwise fall back to the O(n) linear scan over active constraints.
        if let Some(index) = eq_index {
            if let Some(pairs) = index.get(&var_id) {
                for &(ra, rb) in pairs {
                    let other = if ra == resolved { rb } else { ra };
                    // Promotion-aware: the other side may have been
                    // promoted by try_promote_var, whose alias lives only
                    // in the solver-local `promotions` map invisible to
                    // `ctx.resolve_binding`.
                    let other_resolved = self.resolve_promotion_aware(other, ctx);
                    if !matches!(ctx.get(other_resolved), TypeData::InferVar { .. }) {
                        return Some(Self::shape_of_type(ctx, other_resolved));
                    }
                }
            }
        } else {
            for pc in active_constraints {
                if let Constraint::Eq(a, b, _, _) = &pc.constraint {
                    let ra = self.resolve_with_env(*a, ctx, &pc.env);
                    let rb = self.resolve_with_env(*b, ctx, &pc.env);
                    let other = if ra == resolved {
                        Some(rb)
                    } else if rb == resolved {
                        Some(ra)
                    } else {
                        None
                    };
                    if let Some(other_ty) = other {
                        let other_resolved = self.resolve_promotion_aware(other_ty, ctx);
                        if !matches!(ctx.get(other_resolved), TypeData::InferVar { .. }) {
                            return Some(Self::shape_of_type(ctx, other_resolved));
                        }
                    }
                }
            }
        }

        // NOTE: no separate resolutions-table check is needed here.  The
        // entry-point `resolve_with_env` delegates to `resolve_promotion_aware`,
        // which already follows UF root values to a fixpoint — if var_id
        // had a concrete resolution we returned via UNI-TYPE above, and if we
        // reach this point the fixpoint is an unresolved InferVar.  (A former
        // re-check here was therefore unreachable: `res` was always an
        // InferVar and the inner `if` never fired.)

        // ── UNI-BACKPROP: shape from incremental instantiations ──
        // If this variable is a PG variable with forward references,
        // check if all its instances resolve to the same shape.
        if var_id < self.forward_refs.len() && !self.forward_refs[var_id].is_empty() {
            let mut shared_shape: Option<PrincipalShape> = None;
            for &instance_id in &self.forward_refs[var_id] {
                if instance_id < self.var_type_ids.len() {
                    let instance_ty =
                        self.resolve_promotion_aware(self.var_type_ids[instance_id], ctx);
                    let instance_data = ctx.get(instance_ty);
                    if matches!(instance_data, TypeData::InferVar { .. }) {
                        // Instance is still unresolved — can't determine shape.
                        return None;
                    }
                    let inst_shape = Self::shape_of_type(ctx, instance_ty);
                    match shared_shape {
                        None => shared_shape = Some(inst_shape),
                        Some(ref s) if *s != inst_shape => {
                            // Instances disagree on shape — unicity fails.
                            return None;
                        }
                        _ => {}
                    }
                }
            }
            if let Some(shape) = shared_shape {
                return Some(shape);
            }
        }

        // ── Check Sub constraints for upper/lower bounds ─────────
        // If the variable has bounds that all share the same shape,
        // that shape is uniquely determined.
        let mut shape_from_bounds: Option<PrincipalShape> = None;

        // Check upper bounds (supertype constraints)
        if var_id < self.upper_bounds.len() {
            for &bound in &self.upper_bounds[var_id] {
                // Promotion-aware: a bound recorded through try_promote_var
                // aliases only in the solver-local `promotions` map, which
                // `ctx.resolve_binding` cannot follow.
                let bound_resolved = self.resolve_promotion_aware(bound, ctx);
                if !matches!(ctx.get(bound_resolved), TypeData::InferVar { .. }) {
                    let s = Self::shape_of_type(ctx, bound_resolved);
                    match shape_from_bounds {
                        None => shape_from_bounds = Some(s),
                        Some(ref existing) if *existing != s => return None,
                        _ => {}
                    }
                }
            }
        }
        // Check lower bounds (subtype constraints)
        if var_id < self.lower_bounds.len() {
            for &bound in &self.lower_bounds[var_id] {
                let bound_resolved = self.resolve_promotion_aware(bound, ctx);
                if !matches!(ctx.get(bound_resolved), TypeData::InferVar { .. }) {
                    let s = Self::shape_of_type(ctx, bound_resolved);
                    match shape_from_bounds {
                        None => shape_from_bounds = Some(s),
                        Some(ref existing) if *existing != s => return None,
                        _ => {}
                    }
                }
            }
        }

        shape_from_bounds
    }

    /// Z3-based unicity check: delegates to SmtSolver when syntactic
    /// rules (UNI-TYPE/UNI-VAR/UNI-BACKPROP) are insufficient.
    /// Encodes ALL active constraints (Eq, Sub, bindings) as SMT-LIB2
    /// over an uninterpreted sort `Type`, then queries Z3 for whether
    /// exactly one shape is forced by the constraint context.
    ///
    /// This implements the full ⊆-closed erasure semantics:
    ///   C[τ!ζ] iff ∀φ, φ ⊢ [C[τ = g]] ⇒ shape(g) = ζ
    pub fn unicity_check_smt<'input>(
        &self,
        ctx: &TypeContext<'input>,
        ty: TypeId,
        env: &SkolemEnv,
        active: &[PrioritizedConstraint],
    ) -> Option<PrincipalShape> {
        let solver = SmtSolver::new("z3");

        // ── 1. Collect all resolved bindings ─────────────────────
        // Skip variables bound by an enclosing Forall whose binder was
        // opened at or inside this variable's region: their skolem is
        // binder-internal structure, invisible to the outer unicity
        // question.  Bindings for OUTER skolems (binder opened strictly
        // outside) are kept — from here they are fixed constants and
        // legitimately pin the variable's shape (same level rule as
        // `check_determinacy`).  Feeding INNER-skolem information in
        // would let Z3 "prove" a unique shape from what the outer
        // context cannot observe (a spurious-lowering channel).
        // The target type's variable index: unicity is asked about
        // `ty`, which is always `var_type_ids[var_id]` for some var_id.
        let var_level = self
            .var_type_ids
            .iter()
            .position(|&vt| ctx.resolve_binding(vt) == ctx.resolve_binding(ty))
            .and_then(|vid| self.type_vars.get(vid))
            .map(|v| self.region_tree.get_level(v.region_id));
        let in_env = |id: usize| -> bool {
            match env.get(&id) {
                Some(&skolem_ty) => match (var_level, ctx.get(skolem_ty)) {
                    (Some(var_lvl), TypeData::SkolemVar { universe_num, .. }) => {
                        self.region_tree
                            .skolem_level(*universe_num)
                            .unwrap_or(*universe_num)
                            >= var_lvl
                    }
                    _ => true,
                },
                None => false,
            }
        };
        let mut bindings: std::collections::HashMap<usize, TypeId> =
            std::collections::HashMap::default();
        for (i, var_ty) in self.var_type_ids.iter().enumerate() {
            if in_env(i) {
                continue;
            }
            let resolved = ctx.resolve_binding(*var_ty);
            if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                bindings.insert(i, resolved);
            }
        }

        // ── 2. Collect all equality constraints involving InferVars ──
        // Read from the ACTIVE constraint slice, NOT self.constraints:
        // solve() drains self.constraints into the priority queue at
        // startup (std::mem::take), so during solving this vec is EMPTY
        // and a self.constraints read silently fed Z3 a nil-equation
        // problem — it then "proved" unicity from bindings alone, a
        // spurious-lowering channel.
        let mut eq_pairs: Vec<(usize, usize)> = Vec::new();
        for pc in active {
            if let Constraint::Eq(a, b, _, _) = &pc.constraint {
                let ra = self.resolve_with_env(*a, ctx, &pc.env);
                let rb = self.resolve_with_env(*b, ctx, &pc.env);
                if let (TypeData::InferVar { id: aid, .. }, TypeData::InferVar { id: bid, .. }) =
                    (ctx.get(ra), ctx.get(rb))
                {
                    if in_env(*aid) || in_env(*bid) {
                        continue;
                    }
                    eq_pairs.push((*aid, *bid));
                }
            }
        }

        // ── 3. Check unicity via Z3 ──────────────────────────────
        solver.check_unicity(ctx, ty, &bindings, &eq_pairs)
    }

    // ── OmniML: Incremental Instantiation ────────────────────────
    //
    // From O'Brien, Rémy & Scherer §5.2:
    // When a regional abstraction let x = λ∝[∝].C₁ in C₂ contains a
    // suspended constraint, variables in the region are PG. Instances
    // of PG variables must be tracked so that when the PG variable is
    // refined, the instances are updated.
    //
    // This implements the forward-reference mechanism (§6 "From a stack
    // to a tree"): each PG variable has a list of its instances. When
    // the PG variable is unified with a concrete type, all instances
    // are re-unified.

    /// Register that `instance_id` was created as an instance of
    /// `pg_var_id` (a PartiallyGeneralizable variable).
    /// This enables incremental instantiation: when pg_var_id is
    /// refined, instance_id will be updated.
    pub fn register_instance(&mut self, pg_var_id: usize, instance_id: usize) {
        while self.forward_refs.len() <= pg_var_id {
            self.forward_refs.push(Vec::new());
        }
        while self.reverse_refs.len() <= instance_id {
            self.reverse_refs.push(None);
        }
        // The caller guarantees no duplicate registrations. A duplicate
        // would corrupt the incremental instantiation invariant, so we
        // assert in debug and allow the push unconditionally in release.
        debug_assert!(
            !self.forward_refs[pg_var_id].contains(&instance_id),
            "register_instance: duplicate instance {instance_id} for PG var {pg_var_id}",
        );
        self.forward_refs[pg_var_id].push(instance_id);
        self.push_undo(InferUndoLog::ForwardRefPush(pg_var_id));
        let old_rev = self.reverse_refs[instance_id];
        self.reverse_refs[instance_id] = Some(pg_var_id);
        self.push_undo(InferUndoLog::SetReverseRef(instance_id, old_rev));

        // Mark the instance as PI (PartialInstance)
        while self.gen_statuses.len() <= instance_id {
            self.gen_statuses.push(GenStatus::Ungeneralized);
        }
        self.write_gen_status(instance_id, GenStatus::PartialInstance);
    }

    /// ── S-Inst-Copy (OmniML §5.3) ──────────────────────────────────
    ///
    /// Copy solved constraints from a PG abstraction to all its instances.
    /// When a PG variable's multi-equation is resolved (e.g. α = τ),
    /// the equality is propagated to every instance of α.
    /// Returns the number of instances that were updated.
    ///
    /// Guarded by the caller: s_inst_copy is only invoked when
    /// `resolve_ty` contains no InferVars (resolved to a concrete type).
    /// This means nested PG variables cannot appear inside τ, so
    /// alias-sharing via unify_and_track is safe — different PG variables
    /// come from separate let bindings and never share constraints.
    /// Instantiation is transitive: if β is an instance of α and γ is an
    /// instance of β, then γ is also an instance of α.  `forward_refs` is a
    /// forest, so this recursion terminates; the cap is a belt-and-braces
    /// guard against a malformed cycle hanging the solver.
    const S_INST_COPY_MAX_DEPTH: usize = 64;

    pub fn s_inst_copy<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        pg_var_id: usize,
        resolve_ty: TypeId,
    ) -> usize {
        // Pre-resolve through the solver-local promotions map before the
        // closedness check: `ty_mentions_infer_var` only follows
        // `ctx.resolve_binding`, so a promoted leaf would otherwise look
        // unresolved and defer this copy indefinitely.
        let resolve_ty = self.resolve_promotion_aware(resolve_ty, ctx);
        // Enforce closedness in ALL build profiles — a forgotten guard would
        // alias-share types containing unresolved InferVars across instances
        // even in release builds, causing cross-instance contamination.
        if Self::ty_mentions_infer_var(ctx, resolve_ty) {
            self.pending_inst_copy.insert(pg_var_id); // record for retry
            return 0; // leave forward_refs intact for later re-processing
        }
        self.pending_inst_copy.remove(&pg_var_id); // clear on success
        let mut visited = FxHashSet::default();
        self.s_inst_copy_inner(ctx, pg_var_id, resolve_ty, &mut visited, 0)
    }

    fn s_inst_copy_inner<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        pg_var_id: usize,
        resolve_ty: TypeId,
        visited: &mut FxHashSet<usize>,
        depth: usize,
    ) -> usize {
        if depth > Self::S_INST_COPY_MAX_DEPTH || !visited.insert(pg_var_id) {
            return 0;
        }
        if pg_var_id >= self.forward_refs.len() {
            return 0;
        }
        let instances: Vec<usize> = self.forward_refs[pg_var_id].clone();
        let mut updated = 0;
        // Track whether every instance propagated successfully.  A failed
        // `ctx.unify` (the instance's own constraints conflict with the PG
        // variable's resolution — a genuine type error) must NOT clear the
        // forward references: clearing would leave the instance as a
        // dangling, unreachable variable that can never be re-processed.
        // Keeping the association (fail-closed) lets the conflict surface
        // when the instance's other constraints are solved.
        let mut all_propagated = true;
        for inst_id in &instances {
            if *inst_id < self.var_type_ids.len() {
                let instance_ty_id = self.var_type_ids[*inst_id];
                // S-Inst-Copy: bind instance to the resolved type using
                // unify_and_track so that union_find root values stay consistent
                // with ctx.bindings.  Using bare ctx.unify would leave
                // self.resolve() pointing at a stale chain.
                if self
                    .unify_and_track(instance_ty_id, resolve_ty, ctx, Span::new(FileId::DUMMY, 0, 0))
                    .is_err()
                {
                    all_propagated = false;
                } else {
                    // The instance may itself be a PG variable with its own
                    // instances.  Without this recursion the copy stops one
                    // level down and the grandchildren keep unresolved
                    // InferVars — exactly the state the closedness guard
                    // above exists to prevent.
                    updated +=
                        self.s_inst_copy_inner(ctx, *inst_id, resolve_ty, visited, depth + 1);
                    // Count only SUCCESSFUL direct propagations: a failed
                    // unify bound nothing, so counting it would report
                    // successes + failures and mislead callers that use the
                    // return value as a progress signal.
                    updated += 1;
                }
            }
        }
        // Clear forward refs ONLY when every instance propagated.  Log a
        // ClearForwardRefs entry carrying the entries so a rollback restores
        // the registrations.  On a failed propagation the forward refs are
        // left untouched (no undo needed — the state is unchanged).
        if all_propagated && !self.forward_refs[pg_var_id].is_empty() {
            if self.snapshot_depth > 0 {
                let entries = self.forward_refs[pg_var_id].clone();
                self.push_undo(InferUndoLog::ClearForwardRefs(pg_var_id, entries));
            }
            self.forward_refs[pg_var_id].clear();
        }
        updated
    }

    // ── S-Exists-Lower: Z3-backed semantic check (OmniML §5.3) ───
    //
    // The paper's S-Exists-Lower requires:
    //   "C determines β̄ iff every ground assignment φ and φ' that satisfy
    //    (the erasure of) C and coincide outside of β̄ coincide on β̄."
    //
    // We implement this via Z3 (unicity_check_smt). If Z3 determines the
    // variable's shape is uniquely determined by the constraint context, it
    // is safe to lower from PG to monomorphic (Ungeneralized).
    //
    // Falls back to a level-based heuristic when Z3 is unavailable or the
    // query times out, as a conservative over-approximation.

    /// Attempt to lower a variable using the full Z3-backed semantic check
    /// (OmniML §5.3 S-Exists-Lower). If Z3 determines the variable's shape is
    /// uniquely determined by the constraint context, it can be safely lowered
    /// from PG to monomorphic (Ungeneralized).
    ///
    /// The cheap syntactic unicity rules (UNI-TYPE / UNI-VAR / UNI-BACKPROP /
    /// bounds) are tried FIRST — they are a decidable subset of the Z3 check,
    /// so a positive syntactic answer is a proof that Z3 would also accept,
    /// and the (subprocess-spawning) SMT query is only run when they fail.
    ///
    /// Falls back to the level-based heuristic when Z3 is unavailable.
    pub fn s_exists_lower<'input>(
        &mut self,
        ctx: &TypeContext<'input>,
        var_id: usize,
        env: &SkolemEnv,
        active: &[PrioritizedConstraint],
        eq_index: Option<&HashMap<usize, Vec<(TypeId, TypeId)>>>,
    ) -> bool {
        if var_id >= self.type_vars.len() || var_id >= self.gen_statuses.len() {
            return false;
        }
        if self.gen_statuses[var_id] != GenStatus::PartiallyGeneralizable {
            return false;
        }

        // A Forall-scoped skolem is rigid: it stands for a β̄ variable
        // inside the binder, so an equation tying our variable to it is
        // an OUTER observation about binder-internal structure — not a
        // determinate outer assignment.  Rather than bail out wholesale,
        // treat the env's variable ids as rigid members throughout the
        // checks below: `check_determinacy` folds them into every
        // "mentions a variable" set (so Det-Dom/Det-Esc correctly FAIL
        // when the multi-equation touches a skolem, and correctly PASS
        // when it doesn't), and `unicity_check_smt` excludes skolem-
        // bound vars from the free-variable pool it quantifies over.
        // This is strictly more precise than refusing to run the checks
        // at all: lowering still works for variables whose equations
        // never touch the binder.

        // ── Value determinacy check (not shape unicity) ─────────
        // A variable is determined iff it is resolved to a concrete type.
        let resolved = self.resolve_promotion_aware(self.var_type_ids[var_id], ctx);
        if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
            self.set_gen_status(var_id, GenStatus::Ungeneralized);
            return true;
        }

        // ── Det-Dom / Det-Esc syntactic determinacy (OmniML §5.3) ──
        // Sound approximation: proves determinacy when the variable is
        // equated with an outside variable (Det-Dom) or with a type
        // that mentions no region variable (Det-Esc).
        if self.check_determinacy(ctx, var_id, env, active, eq_index) {
            self.set_gen_status(var_id, GenStatus::Ungeneralized);
            return true;
        }

        // ── Z3-backed semantic check ──────────────────────────────
        // Only reached when the syntactic rules above failed.
        // Z3 is strictly more complete but requires a subprocess.
        if self
            .unicity_check_smt(ctx, self.var_type_ids[var_id], env, active)
            .is_some()
        {
            self.set_gen_status(var_id, GenStatus::Ungeneralized);
            return true;
        }

        // Removed the unsound level-based fallback.
        // The old code lowered a PG variable to Ungeneralized based solely
        // on region depth and ancestry, WITHOUT verifying determinacy
        // (OmniML Definition 5.3). A variable in a deep region that is
        // NOT determined by the outer context would get monomorphized,
        // losing principality. The OmniML reference implementation has no
        // such fallback — it simply leaves the variable PG and revisits
        // when more information arrives.
        false
    }

    /// ── S-Generalize / update_and_generalize_generation (OmniML §5.3) ──
    ///
    /// Drains all dirty regions, collects guarded roots, and generalizes
    /// PG variables that are no longer guarded or referenced.
    ///
    /// The optional `target_var_id` restricts processing to just the region
    /// containing that variable (for targeted generalization before instantiation).
    pub fn force_generalize<'input>(&mut self, ctx: &mut TypeContext<'input>) {
        self.force_generalize_for_regions(ctx, &[], None)
    }

    /// Full generation-based generalization.  `dirty_levels` lists region
    /// levels that have been marked dirty.  `target_var_id` (if set) limits
    /// processing to the region containing that specific variable.
    pub fn force_generalize_for_regions<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        dirty_levels: &[usize],
        target_var: Option<usize>,
    ) {
        // Collect PG variables from dirty_set or region levels.
        // Uses the `pg_vars` index to avoid O(total_vars) full-table scans.
        let mut dirty: Vec<usize> = if let Some(tv) = target_var {
            let region = self
                .type_vars
                .get(tv)
                .map(|v| v.region_id)
                .unwrap_or(self.region_tree.root);
            self.pg_vars
                .iter()
                .copied()
                .filter(|i| {
                    self.type_vars
                        .get(*i)
                        .map(|v| v.region_id == region)
                        .unwrap_or(false)
                })
                .collect()
        } else if !self.dirty_set.is_empty() {
            self.dirty_set
                .iter()
                .copied()
                .filter(|i| self.pg_vars.contains(i))
                .collect()
        } else if !dirty_levels.is_empty() {
            // Collect the actual region IDs at the given levels, then
            // filter by ancestry — NOT bare level equality.  Sibling
            // regions at the same depth share a level, so level-based
            // matching would incorrectly generalize PG variables from
            // an unrelated sibling region that only happens to be at the
            // same depth.
            let target_regions: Vec<InferRegionId> = self
                .region_tree
                .nodes
                .iter()
                .filter(|n| dirty_levels.contains(&n.level))
                .map(|n| n.id)
                .collect();
            self.pg_vars
                .iter()
                .copied()
                .filter(|i| {
                    self.type_vars
                        .get(*i)
                        .map(|v| {
                            target_regions
                                .iter()
                                .any(|&r| self.region_tree.is_ancestor(r, v.region_id))
                        })
                        .unwrap_or(false)
                })
                .collect()
        } else {
            self.pg_vars.iter().copied().collect()
        };

        if dirty.is_empty() {
            return;
        }

        // Skip variables already generalized in this region: they have
        // nothing left to do, and dropping them here also spares them the
        // `check_rigid_escape` walk and the transitive-guard root scan.
        // `retain` is O(n) in one pass; the previous `position`+`remove`
        // loop was O(n²) in the number of skips.
        dirty.retain(|&i| {
            !self.type_vars.get(i).is_some_and(|tv| {
                self.gen_statuses.get(i) == Some(&GenStatus::Generalized)
                    && self.generalized_in_region.contains(&(i, tv.region_id.0))
            })
        });

        if dirty.is_empty() {
            return;
        }

        // Ensure guard_sets consistency.
        for &i in &dirty {
            while self.guard_sets.len() <= i {
                self.guard_sets.push(GuardSet::empty());
            }
        }

        // Compute transitive guards via binding-root sharing.
        // Optimization: pre-compute all roots to avoid O(n²) resolve_binding calls.
        let mut trans_guarded: FxHashSet<usize> = FxHashSet::default();
        for &i in &dirty {
            if i < self.guard_sets.len() && !self.guard_sets[i].is_empty() {
                trans_guarded.insert(i);
            }
        }
        // Pre-compute binding roots for all dirty indices.
        let mut roots: HashMap<usize, TypeId> = HashMap::default();
        for &idx in &dirty {
            if idx < self.var_type_ids.len() {
                roots.insert(
                    idx,
                    self.resolve_promotion_aware(self.var_type_ids[idx], ctx),
                );
            }
        }
        // Collect roots of ALL guarded PG variables, not just dirty ones.
        // A guarded variable unified in a previous iteration (not in `dirty`)
        // would otherwise be missed, allowing its unified partner to be
        // incorrectly generalized.
        let mut guarded_roots: FxHashSet<TypeId> = FxHashSet::default();
        for &j in self.pg_vars.iter() {
            if j < self.var_type_ids.len()
                && j < self.guard_sets.len()
                && !self.guard_sets[j].is_empty()
            {
                let root = self.resolve_promotion_aware(self.var_type_ids[j], ctx);
                guarded_roots.insert(root);
            }
        }
        // Any non-guarded item whose root matches a guarded root becomes guarded.
        for &i in &dirty {
            if !trans_guarded.contains(&i)
                && i < self.var_type_ids.len()
                && let Some(&root) = roots.get(&i)
                && guarded_roots.contains(&root)
            {
                trans_guarded.insert(i);
            }
        }

        // Process innermost-first (highest level first = most nested first).
        let mut vars_by_level: Vec<(usize, usize)> = dirty
            .iter()
            .map(|&i| {
                (
                    i,
                    self.type_vars
                        .get(i)
                        .map(|v| self.region_tree.get_level(v.region_id))
                        .unwrap_or(0),
                )
            })
            .collect();
        vars_by_level.sort_by(|a, b| b.1.cmp(&a.1));

        // Rigid scope check per generation: a PG variable whose resolved
        // type contains an escaped skolem (introduced inside the region
        // being generalized) must NOT be generalized — generalizing would
        // copy the skolem into the type scheme (`s_inst_copy`), leaking
        // a Forall-bound variable into an outer scope.
        let mut rigid_escaped: FxHashSet<usize> = FxHashSet::default();
        for &(i, level) in &vars_by_level {
            if i >= self.var_type_ids.len() {
                continue;
            }
            let resolved = self.resolve_promotion_aware(self.var_type_ids[i], ctx);
            if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                continue;
            }
            if Self::check_rigid_escape(
                ctx,
                resolved,
                level,
                &self.region_tree,
                &self.var_promotions,
            ) {
                rigid_escaped.insert(i);
            }
        }

        // Generalize eligible PG → G in level order (generation order).
        for &(i, _level) in &vars_by_level {
            if i >= self.gen_statuses.len()
                || self.gen_statuses[i] != GenStatus::PartiallyGeneralizable
            {
                continue;
            }
            let is_trans_guarded = trans_guarded.contains(&i);
            let has_waiting = i < self.wait_lists.len() && !self.wait_lists[i].is_empty();
            let is_resolved = {
                let ty = self.resolve_promotion_aware(self.var_type_ids[i], ctx);
                !matches!(ctx.get(ty), TypeData::InferVar { .. })
            };
            // The rigid-escape check result is CONSUMED here: a variable
            // whose resolved type escapes a rigid may not be generalized
            // (it stays PartiallyGeneralizable / handled elsewhere).
            if is_resolved && !is_trans_guarded && !has_waiting && !rigid_escaped.contains(&i) {
                self.write_gen_status(i, GenStatus::Generalized);
                // Record the generalization: (var_id, region_id).
                if let Some(tv) = self.type_vars.get(i) {
                    self.generalized_in_region.insert((i, tv.region_id.0));
                }
            }
        }

        // Update dirty_set: remove only the processed levels.
        if dirty_levels.is_empty() && target_var.is_none() {
            self.dirty_set.clear();
        } else {
            // Hash the removal list once: retain against a Vec would be
            // O(dirty_set × |dirty|) on every sweep.
            let processed: FxHashSet<usize> = dirty.iter().copied().collect();
            self.dirty_set.retain(|i| !processed.contains(i));
        }
    }

    /// Whether `var_id` meets the sweep's per-var generalization criterion
    /// (the same checks `force_generalize_for_regions` applies, for a single
    /// variable): resolved, no direct guards, nothing waiting, no rigid
    /// escape from its region level, and not transitively guarded via
    /// binding-root sharing with another dirty guarded var.  Used by the
    /// Eq handler to transition eligible current vars PG → Generalized
    /// deterministically (before S-Exists-Lower), since the sweep itself
    /// now runs once per iteration instead of per transition.
    fn is_sweep_eligible<'input>(&self, ctx: &TypeContext<'input>, var_id: usize) -> bool {
        if var_id >= self.gen_statuses.len()
            || self.gen_statuses[var_id] != GenStatus::PartiallyGeneralizable
        {
            return false;
        }
        if var_id < self.wait_lists.len() && !self.wait_lists[var_id].is_empty() {
            return false;
        }
        let Some(ty_id) = self.var_type_ids.get(var_id) else {
            return false;
        };
        let resolved = self.resolve_promotion_aware(*ty_id, ctx);
        if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
            return false;
        }
        if var_id < self.guard_sets.len() && !self.guard_sets[var_id].is_empty() {
            return false;
        }
        let level = self
            .type_vars
            .get(var_id)
            .map(|v| self.region_tree.get_level(v.region_id))
            .unwrap_or(0);
        if Self::check_rigid_escape(
            ctx,
            resolved,
            level,
            &self.region_tree,
            &self.var_promotions,
        ) {
            return false;
        }
        // Transitive guard: a PG var with direct guards sharing our
        // binding root (mirrors `trans_guarded` + `guarded_roots` in the
        // sweep).  The sweep scans ALL pg_vars, not just the dirty set —
        // a guarded var unified in an earlier iteration is no longer
        // dirty but still blocks generalization of its root partners
        // (the exact miss the sweep's `guarded_roots` comment warns
        // about).
        // Both sides must go through the same resolver, otherwise a
        // promotion-only difference makes the roots compare unequal and the
        // transitive guard is silently dropped.
        let root = resolved;
        for &j in &self.pg_vars {
            if j != var_id
                && j < self.guard_sets.len()
                && !self.guard_sets[j].is_empty()
                && j < self.var_type_ids.len()
                && self.resolve_promotion_aware(self.var_type_ids[j], ctx) == root
            {
                return false;
            }
        }
        true
    }

    /// Check whether a resolved type contains escaped rigid (skolem) variables.
    /// Recursively walks the type tree looking for `GenericParam` or `SkolemVar`
    /// references that would indicate a Forall-bound variable has escaped into
    /// Detect whether `ty` contains a rigid variable that escapes the region
    /// being generalized (max_level).  Outer GenericParams are part of the
    /// environment (Γ) and are always safe; SkolemVars escape only if they
    /// were introduced inside the region being generalized (universe > max_level)
    /// or are GADT existential witnesses (always escape if free).
    pub(crate) fn check_rigid_escape<'input>(
        ctx: &TypeContext<'input>,
        ty: TypeId,
        max_level: usize,
        region_tree: &InferRegionTree,
        var_promotions: &HashMap<usize, usize>,
    ) -> bool {
        Self::check_rigid_escape_inner(ctx, ty, max_level, region_tree, var_promotions)
    }

    fn check_rigid_escape_inner<'input>(
        ctx: &TypeContext<'input>,
        ty: TypeId,
        max_level: usize,
        region_tree: &InferRegionTree,
        var_promotions: &HashMap<usize, usize>,
    ) -> bool {
        // Iterative traversal with an explicit stack.  Each stack entry is
        // either a type to visit or a "post-action" that unbinds a variable
        // after its scope body has been fully processed.  This avoids stack
        // overflow on deeply nested types (e.g. Fn(Fn(Fn(...)))).
        enum Action {
            Visit(TypeId),
        }
        let mut stack: Vec<Action> = vec![Action::Visit(ty)];
        while let Some(action) = stack.pop() {
            match action {
                Action::Visit(ty) => {
                    let resolved = ctx.resolve_binding(ty);
                    match ctx.get(resolved) {
                        TypeData::GenericParam { .. } => {}
                        TypeData::SkolemVar {
                            id: _,
                            universe_num,
                            ..
                        } => {
                            // SkolemVars are NEVER bound by Forall/Exists/Mu/Nu/Poly
                            // binders — they live in a different namespace (dynamically
                            // allocated via enter_universe, not AST param_index).
                            // The old `bound_vars.contains(id)` check falsely matched
                            // when a SkolemVar.id collided with a GenericParam.param_index
                            // (they live in different namespaces).  The correct escape check
                            // is the universe/depth comparison below.
                            // GADT existential witnesses are always escaping:
                            // they are tied to a single match arm and must
                            // never be copied into a type scheme.
                            if *universe_num == TypeContext::GADT_SKOLEM_UNIVERSE {
                                return true;
                            }
                            // Compare *introduction depth*, not the raw
                            // universe ordinal.  Skolems with no recorded
                            // level were allocated outside the Forall
                            // handler; fall back to the ordinal so their
                            // behaviour is unchanged.
                            let depth = region_tree
                                .skolem_level(*universe_num)
                                .unwrap_or(*universe_num);
                            if depth >= max_level {
                                return true;
                            }
                        }
                        TypeData::Fn { params, ret } => {
                            stack.push(Action::Visit(*ret));
                            for &p in params.iter().rev() {
                                stack.push(Action::Visit(p));
                            }
                        }
                        TypeData::Tuple { elems }
                        | TypeData::Coproduct {
                            alternatives: elems,
                        } => {
                            for &e in elems.iter().rev() {
                                stack.push(Action::Visit(e));
                            }
                        }
                        TypeData::Adt { args, .. } => {
                            for &a in args.iter().rev() {
                                stack.push(Action::Visit(a));
                            }
                        }
                        TypeData::Forall {
                            param_index, body, ..
                        }
                        | TypeData::Exists {
                            param_index,
                            base: body,
                            ..
                        }
                        | TypeData::Mu {
                            param_index, body, ..
                        }
                        | TypeData::Nu {
                            param_index, body, ..
                        } => {
                            // Binder bookkeeping removed: bound_vars was
                            // write-only since the old contains() check was
                            // retired (see the note above) — SkolemVar
                            // consults skolem_level/universe_num, GenericParam
                            // is unconditional, so there is nothing to consult.
                            stack.push(Action::Visit(*body));
                        }
                        TypeData::Poly { body, .. } => {
                            stack.push(Action::Visit(*body));
                        }
                        TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                            stack.push(Action::Visit(*ty));
                        }
                        TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                            stack.push(Action::Visit(*elem));
                        }
                        TypeData::Ptr { pointee, .. } => {
                            stack.push(Action::Visit(*pointee));
                        }
                        TypeData::AssociatedType { self_ty, .. } => {
                            stack.push(Action::Visit(*self_ty));
                        }
                        // `hidden` is a TypeId revealed outside the defining
                        // scope — a skolem hidden inside it must block
                        // generalization just like one in an Adt arg.
                        // (DynTrait needs no arm: `traits: Vec<DefId>` holds
                        // no TypeId children.)
                        TypeData::Opaque {
                            hidden: Some(hidden),
                            ..
                        } => {
                            stack.push(Action::Visit(*hidden));
                        }
                        TypeData::Opaque { hidden: None, .. } => {}
                        TypeData::InferVar { id, .. } => {
                            // Follow the promotion chain to find the effective
                            // variable. After try_promote_var, the old variable's
                            // var_to_region is None (unregister_var), so we must
                            // walk promotions to reach the current representative.
                            // but the promoted variable has the correct region.
                            let mut effective_id = *id;
                            // Cycle cap, mirroring `resolve` /
                            // `resolve_promotion_aware`: every other walk
                            // over `var_promotions` bounds its steps, and
                            // this runs on every generalization — a cyclic
                            // map would hang the compiler here.
                            let mut steps = 0usize;
                            let max_steps = var_promotions.len() + 1;
                            while let Some(&next) = var_promotions.get(&effective_id) {
                                steps += 1;
                                if steps > max_steps {
                                    debug_assert!(
                                        false,
                                        "check_rigid_escape: cycle in var_promotions after {steps} steps"
                                    );
                                    break;
                                }
                                effective_id = next;
                            }
                            if region_tree
                                .var_to_region
                                .get(effective_id)
                                .and_then(|v| *v)
                                .map_or(false, |rid| region_tree.get_level(rid) > max_level)
                            {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        false
    }

    /// Check whether `ty` mentions ANY variable in `vars`.
    /// Used by Det-Esc: `fv(τ) # ᾱ, β̄` (τ must not mention any region var).
    fn type_mentions_any_var<'input>(
        ctx: &TypeContext<'input>,
        ty: TypeId,
        vars: &FxHashSet<usize>,
    ) -> bool {
        // Iterative worklist traversal — avoids stack overflow on deep types.
        let mut stack: Vec<TypeId> = vec![ty];
        while let Some(t) = stack.pop() {
            let resolved = ctx.resolve_binding(t);
            match ctx.get(resolved) {
                TypeData::InferVar { id, .. } => {
                    if vars.contains(id) {
                        return true;
                    }
                }
                TypeData::Fn { params, ret } => {
                    stack.push(*ret);
                    stack.extend(params.iter().copied());
                }
                TypeData::Tuple { elems }
                | TypeData::Coproduct {
                    alternatives: elems,
                } => {
                    stack.extend(elems.iter().copied());
                }
                TypeData::Adt { args, .. } => {
                    stack.extend(args.iter().copied());
                }
                TypeData::Forall { body, .. }
                | TypeData::Exists { base: body, .. }
                | TypeData::Poly { body, .. }
                | TypeData::Mu { body, .. }
                | TypeData::Nu { body, .. } => {
                    stack.push(*body);
                }
                TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                    stack.push(*ty);
                }
                TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                    stack.push(*elem);
                }
                TypeData::Ptr { pointee, .. } => {
                    stack.push(*pointee);
                }
                TypeData::AssociatedType { self_ty, .. } => {
                    stack.push(*self_ty);
                }
                _ => {}
            }
        }
        false
    }

    /// Check whether a type contains ANY unresolved InferVar (recursively).
    /// Used to gate `s_inst_copy` — the contract requires `resolve_ty` to be
    /// fully closed (no InferVars anywhere) before alias-sharing into instances.
    fn ty_mentions_infer_var<'input>(ctx: &TypeContext<'input>, ty: TypeId) -> bool {
        let mut stack: Vec<TypeId> = vec![ty];
        while let Some(t) = stack.pop() {
            let r = ctx.resolve_binding(t);
            match ctx.get(r) {
                TypeData::InferVar { .. } => return true,
                TypeData::Fn { params, ret } => {
                    stack.push(*ret);
                    stack.extend(params.iter().copied());
                }
                TypeData::Tuple { elems }
                | TypeData::Coproduct {
                    alternatives: elems,
                } => {
                    stack.extend(elems.iter().copied());
                }
                TypeData::Adt { args, .. } => {
                    stack.extend(args.iter().copied());
                }
                TypeData::Forall { body, .. }
                | TypeData::Exists { base: body, .. }
                | TypeData::Poly { body, .. }
                | TypeData::Mu { body, .. }
                | TypeData::Nu { body, .. } => {
                    stack.push(*body);
                }
                TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                    stack.push(*ty);
                }
                TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                    stack.push(*elem);
                }
                TypeData::Ptr { pointee, .. } => {
                    stack.push(*pointee);
                }
                TypeData::AssociatedType { self_ty, .. } => {
                    stack.push(*self_ty);
                }
                _ => {}
            }
        }
        false
    }

    /// Helper: find the first InferVar in a type tree.
    /// Used by the Instance handler to suspend on the first unresolved
    /// variable when the scheme contains nested InferVars.
    fn find_first_infer_var<'input>(ctx: &TypeContext<'input>, ty: TypeId) -> Option<usize> {
        let mut stack: Vec<TypeId> = vec![ty];
        while let Some(t) = stack.pop() {
            let r = ctx.resolve_binding(t);
            match ctx.get(r) {
                TypeData::InferVar { id, .. } => return Some(*id),
                TypeData::Fn { params, ret } => {
                    stack.push(*ret);
                    stack.extend(params.iter().copied());
                }
                TypeData::Tuple { elems }
                | TypeData::Coproduct {
                    alternatives: elems,
                } => {
                    stack.extend(elems.iter().copied());
                }
                TypeData::Adt { args, .. } => {
                    stack.extend(args.iter().copied());
                }
                TypeData::Forall { body, .. }
                | TypeData::Exists { base: body, .. }
                | TypeData::Poly { body, .. }
                | TypeData::Mu { body, .. }
                | TypeData::Nu { body, .. } => {
                    stack.push(*body);
                }
                TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                    stack.push(*ty);
                }
                TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                    stack.push(*elem);
                }
                TypeData::Ptr { pointee, .. } => {
                    stack.push(*pointee);
                }
                TypeData::AssociatedType { self_ty, .. } => {
                    stack.push(*self_ty);
                }
                _ => {}
            }
        }
        None
    }

    // ── Det-Dom / Det-Esc — COMPLETE (OmniML §5.3, Definition 5.3) ─

    /// Build the full multi-equation containing `var_id` by taking the
    /// transitive closure over ALL equality sources:
    ///   1. union_find root values  (local solved form)
    ///   2. ctx.resolve_binding  (global binding chains)
    ///   3. active Eq constraints  (pending in the heap)
    ///
    /// This produces the solved-form multi-equation required for the
    /// COMPLETE Det-Dom / Det-Esc check (§5.3: "coincides with the
    /// semantic definition whenever C is in solved form").
    fn build_multi_equation<'input>(
        &self,
        ctx: &TypeContext<'input>,
        var_id: usize,
        active: &[PrioritizedConstraint],
        eq_index: Option<&HashMap<usize, Vec<(TypeId, TypeId)>>>,
    ) -> MultiEquation {
        let mut me = MultiEquation::new();
        let mut worklist: Vec<usize> = vec![var_id];
        let mut visited: FxHashSet<usize> = FxHashSet::default();

        while let Some(vid) = worklist.pop() {
            if !visited.insert(vid) {
                continue;
            }
            me.variables.insert(vid);

            // ── Source 1: local resolution table (UF root values) ──
            if let Some(resolved_ty) = self.union_find.probe_read(vid) {
                let resolved = ctx.resolve_binding(resolved_ty);
                match ctx.get(resolved) {
                    TypeData::InferVar { id, .. } => {
                        worklist.push(*id);
                    }
                    _ => {
                        me.try_set_concrete(resolved);
                        // Walk the concrete type for nested InferVars
                        self.collect_nested_vars(ctx, resolved, &mut worklist);
                    }
                }
            }

            // ── Source 2: global binding chain ────────────────────
            if vid < self.var_type_ids.len() {
                let ty = self.var_type_ids[vid];
                let resolved = ctx.resolve_binding(ty);
                match ctx.get(resolved) {
                    TypeData::InferVar { id, .. } if *id != vid => {
                        worklist.push(*id);
                    }
                    TypeData::InferVar { .. } => {} // self-loop
                    _ => {
                        me.try_set_concrete(resolved);
                        self.collect_nested_vars(ctx, resolved, &mut worklist);
                    }
                }
            }

            // ── Source 3: active Eq constraints ───────────────────
            // Use the pre-built eq_index for O(1) lookup when available,
            // falling back to the O(n) linear scan over `active`.
            if let Some(index) = eq_index {
                if let Some(pairs) = index.get(&vid) {
                    // Each pair is (infer_var_ty, partner_ty) — the first
                    // element is always the InferVar matching `vid`, the
                    // second is the other side of the Eq constraint.
                    for &(_var_ty, other) in pairs {
                        // Re-resolve through promotions: the index was built
                        // earlier and further promotions may have aliased
                        // `other` since — pushing the stale variable id
                        // would miss the promoted representative.
                        let other = self.resolve_promotion_aware(other, ctx);
                        if let TypeData::InferVar { id, .. } = ctx.get(other) {
                            worklist.push(*id);
                        } else {
                            if me.concrete.is_none() {
                                me.try_set_concrete(other);
                            }
                            self.collect_nested_vars(ctx, other, &mut worklist);
                        }
                    }
                }
            } else {
                for pc in active {
                    if let Constraint::Eq(a, b, _, _) = &pc.constraint {
                        let ra = self.resolve_with_env(*a, ctx, &pc.env);
                        let rb = self.resolve_with_env(*b, ctx, &pc.env);

                        let mut a_has_vid = false;
                        let mut b_has_vid = false;

                        if let TypeData::InferVar { id, .. } = ctx.get(ra) {
                            if *id == vid {
                                a_has_vid = true;
                            }
                        } else {
                            let mut nested = Vec::new();
                            self.collect_nested_vars(ctx, ra, &mut nested);
                            if nested.contains(&vid) {
                                a_has_vid = true;
                            }
                        }

                        if let TypeData::InferVar { id, .. } = ctx.get(rb) {
                            if *id == vid {
                                b_has_vid = true;
                            }
                        } else {
                            let mut nested = Vec::new();
                            self.collect_nested_vars(ctx, rb, &mut nested);
                            if nested.contains(&vid) {
                                b_has_vid = true;
                            }
                        }

                        if a_has_vid {
                            if let TypeData::InferVar { id, .. } = ctx.get(rb) {
                                worklist.push(*id);
                            } else {
                                me.try_set_concrete(rb);
                                self.collect_nested_vars(ctx, rb, &mut worklist);
                            }
                        }

                        if b_has_vid {
                            if let TypeData::InferVar { id, .. } = ctx.get(ra) {
                                worklist.push(*id);
                            } else {
                                me.try_set_concrete(ra);
                                self.collect_nested_vars(ctx, ra, &mut worklist);
                            }
                        }
                    }
                }
            }
        }

        me
    }

    /// Walk a type and collect all InferVar ids nested inside it.
    /// These are members of the same multi-equation (U-Decomp has
    /// decomposed the shape into sub-component equalities).
    fn collect_nested_vars<'input>(
        &self,
        ctx: &TypeContext<'input>,
        ty: TypeId,
        out: &mut Vec<usize>,
    ) {
        // Iterative worklist traversal — avoids stack overflow on deep types.
        let mut stack: Vec<TypeId> = vec![ty];
        while let Some(t) = stack.pop() {
            let resolved = ctx.resolve_binding(t);
            match ctx.get(resolved) {
                TypeData::InferVar { id, .. } => out.push(*id),
                TypeData::Fn { params, ret } => {
                    stack.push(*ret);
                    stack.extend(params.iter().copied());
                }
                TypeData::Tuple { elems }
                | TypeData::Coproduct {
                    alternatives: elems,
                } => {
                    stack.extend(elems.iter().copied());
                }
                TypeData::Adt { args, .. } => {
                    stack.extend(args.iter().copied());
                }
                TypeData::Forall { body, .. }
                | TypeData::Exists { base: body, .. }
                | TypeData::Poly { body, .. }
                | TypeData::Mu { body, .. }
                | TypeData::Nu { body, .. } => {
                    stack.push(*body);
                }
                TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                    stack.push(*ty);
                }
                TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                    stack.push(*elem);
                }
                TypeData::Ptr { pointee, .. } => {
                    stack.push(*pointee);
                }
                TypeData::AssociatedType { self_ty, .. } => {
                    stack.push(*self_ty);
                }
                _ => {} // concrete leaf — no nested vars
            }
        }
    }

    /// Check whether a concrete type mentions any InferVar that is
    /// NOT in `region_vars`.  Used by the composite Det-Dom check:
    /// α = Fn(β, γ) where γ is outside the region.
    fn concrete_mentions_outside_var<'input>(
        &self,
        ctx: &TypeContext<'input>,
        ty: TypeId,
        region_vars: &FxHashSet<usize>,
    ) -> bool {
        // Iterative worklist traversal — avoids stack overflow on deep types.
        let mut stack: Vec<TypeId> = vec![ty];
        while let Some(t) = stack.pop() {
            let resolved = ctx.resolve_binding(t);
            match ctx.get(resolved) {
                TypeData::InferVar { id, .. } => {
                    if !region_vars.contains(id) {
                        return true;
                    }
                }
                TypeData::Fn { params, ret } => {
                    stack.push(*ret);
                    stack.extend(params.iter().copied());
                }
                TypeData::Tuple { elems }
                | TypeData::Coproduct {
                    alternatives: elems,
                } => {
                    stack.extend(elems.iter().copied());
                }
                TypeData::Adt { args, .. } => {
                    stack.extend(args.iter().copied());
                }
                TypeData::Forall { body, .. }
                | TypeData::Exists { base: body, .. }
                | TypeData::Poly { body, .. }
                | TypeData::Mu { body, .. }
                | TypeData::Nu { body, .. } => {
                    stack.push(*body);
                }
                TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                    stack.push(*ty);
                }
                TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                    stack.push(*elem);
                }
                TypeData::Ptr { pointee, .. } => {
                    stack.push(*pointee);
                }
                TypeData::AssociatedType { self_ty, .. } => {
                    stack.push(*self_ty);
                }
                _ => {} // concrete leaf — no variables
            }
        }
        false
    }

    /// ── Det-Dom / Det-Esc — COMPLETE (OmniML §5.3, Definition 5.3) ──
    ///
    /// Builds the full multi-equation containing `var_id` (transitive
    /// closure over resolutions, bindings, and active Eq constraints),
    /// then applies the two syntactic rules on the entire class.
    ///
    /// §5.3 guarantee: "This syntactic relation coincides with the
    /// semantic definition of determinacy whenever C is in solved form."
    /// The multi-equation builder produces the solved-form structure,
    /// so this check is COMPLETE — not merely a sound approximation.
    ///
    /// **Det-Dom**: ∃ γ in the multi-equation with γ ∉ ᾱ ∪ β̄
    ///              (an outside variable constrains the region)
    /// **Det-Esc**: ∃ τ in the multi-equation with fv(τ) # ᾱ ∪ β̄
    ///              (a type external to the region determines ᾱ)
    fn check_determinacy<'input>(
        &self,
        ctx: &TypeContext<'input>,
        var_id: usize,
        env: &SkolemEnv,
        active: &[PrioritizedConstraint],
        eq_index: Option<&HashMap<usize, Vec<(TypeId, TypeId)>>>,
    ) -> bool {
        // ── Collect region membership: β̄ ∪ ᾱ ─────────────────────
        // OmniML §5.3 defines a region as the full existential
        // prefix including all descendant regions. The old code only collected
        // the immediate node's pool, missing variables in child regions.
        // Det-Dom would spuriously treat child-region vars as "outside".
        // Follow the promotion chain first: after try_promote_var only the
        // NEW variable's region_id is updated; a pre-promotion id would
        // still point at the original (deeper) region.
        let mut effective_id = var_id;
        {
            let mut steps = 0usize;
            let max_steps = self.var_promotions.len() + 1;
            while let Some(&next) = self.var_promotions.get(&effective_id) {
                steps += 1;
                if steps > max_steps {
                    break; // cycle guard — cannot happen, but never hang
                }
                effective_id = next;
            }
        }
        let var_region = self.type_vars.get(effective_id).map(|v| v.region_id);
        let mut region_vars: FxHashSet<usize> = var_region
            .map(|r| {
                self.region_tree
                    .nodes
                    .iter()
                    .filter(|node| self.region_tree.is_ancestor(r, node.id))
                    .flat_map(|node| node.pool.var_ids.iter().copied())
                    .collect()
            })
            .unwrap_or_default();

        // ── Scoped skolems: level-based rigidity ──────────────────
        // OmniML's Det-Dom/Det-Esc are stated purely over existential
        // region variables; ∀-skolems are an extension (no counterpart
        // exists in the paper — its polymorphism lives at let-binders).
        // The guiding principle is OutsideIn's one-way flow: only known
        // information from OUTSIDE a binder may be used inside.  Whether
        // a skolem counts as "outside" is decided by comparing its
        // introduction level (recorded by the Forall handler, same
        // mechanism as `check_rigid_escape`) with the level of the
        // variable's region:
        //
        // - skolem level < var's region level: the Forall was opened
        //   strictly outside the var's region, so from the var's
        //   perspective the skolem is a fixed outer constant.  Under
        //   Definition 5.3's naive reading it DETERMINES the variable
        //   (every ground assignment agrees on it) — leave it out of
        //   `region_vars` so Det-Dom can fire and lower.
        // - skolem level >= var's region level: the Forall is opened at
        //   or inside the var's region scope.  An equation linking the
        //   variable to it is an observation about binder-internal
        //   structure that cannot be expressed once the variable is
        //   lowered out — fold it into `region_vars` so Det-Dom/Det-Esc
        //   correctly refuse to lower.
        //
        // Skolems with no recorded level (allocated outside the Forall
        // handler) fall back to the universe ordinal, mirroring
        // `check_rigid_escape`.
        let var_level = var_region.map(|r| self.region_tree.get_level(r));
        for (&skolem_var, &skolem_ty) in env {
            let blocked = match (var_level, ctx.get(skolem_ty)) {
                (Some(var_lvl), TypeData::SkolemVar { universe_num, .. }) => {
                    let depth = self
                        .region_tree
                        .skolem_level(*universe_num)
                        .unwrap_or(*universe_num);
                    depth >= var_lvl
                }
                // No region (shouldn't happen) or non-skolem mapping:
                // stay conservative.
                _ => true,
            };
            if blocked {
                region_vars.insert(skolem_var);
            }
        }
        // ── Build the full multi-equation (transitive closure) ────
        let me = self.build_multi_equation(ctx, var_id, active, eq_index);

        // A conflicted multi-equation means unsatisfiable constraints.
        // The variable is vacuously "determined" but should be treated as
        // an error rather than silently accepted.
        // NOT lower it — the error surfaces during unification.
        if me.conflicted {
            return false;
        }

        // ── Det-Dom ───────────────────────────────────────────────
        // The multi-equation contains a variable γ OUTSIDE the region.
        // Since γ is monomorphic (not in ᾱ ∪ β̄), it constrains the
        // entire multi-equation — including var_id.
        for &member_id in &me.variables {
            if member_id != var_id && !region_vars.contains(&member_id) {
                return true; // Det-Dom: outside variable in same class
            }
        }

        // ── Det-Esc ───────────────────────────────────────────────
        // The multi-equation contains a concrete type τ whose free
        // variables do NOT include any region variable.
        if let Some(concrete_ty) = me.concrete {
            if !Self::type_mentions_any_var(ctx, concrete_ty, &region_vars) {
                return true; // Det-Esc: external type determines var_id
            }
        }

        // ── Det-Dom (composite form) ──────────────────────────────
        // A concrete type in the multi-equation mentions an OUTSIDE
        // variable.  E.g. α = Fn(β, γ) where γ ∉ region.
        if let Some(concrete_ty) = me.concrete {
            if self.concrete_mentions_outside_var(ctx, concrete_ty, &region_vars) {
                return true;
            }
        }

        false
    }

    /// Mark a variable as dirty for the next `force_generalize` call.
    /// Called when a variable is unified or updated, enabling incremental
    /// processing instead of re-checking all variables.
    pub fn mark_dirty(&mut self, var_id: usize) {
        self.dirty_set.insert(var_id);
    }

    // ── Snapshot / Rollback (rustc-style UndoLog) ────────────────

    /// Open a new snapshot.  While open, reversible operations record
    /// an `InferUndoLog` entry.  Returns the current log length.
    #[must_use = "start_snapshot returns a snapshot token that must be consumed by rollback_to or commit_snapshot"]
    pub fn start_snapshot(&mut self) -> usize {
        self.snapshot_depth += 1;
        // Pool mutations from now on are reversible — record them.
        self.region_tree.pool_logging_enabled = true;
        self.region_current_snapshot_stack
            .push(self.region_tree.current);
        self.resolved_snapshot_stack.push((
            self.resolved_ids.len(),
            self.resolved_set.clone(),
            self.dirty_set.clone(),
            self.suspended_vars.clone(),
            self.discharged_matches.clone(),
            self.region_tree.pool_undo_log.len(),
            self.pg_vars.clone(),
            self.union_find.snapshot(),
            self.generalized_in_region.clone(),
            self.pending_inst_copy.clone(),
            self.discharged_subs.clone(),
            [
                self.type_vars.len(),
                self.var_type_ids.len(),
                self.lower_bounds.len(),
                self.upper_bounds.len(),
                self.wait_lists.len(),
                self.pending_sub_vars.len(),
                self.guard_sets.len(),
                self.gen_statuses.len(),
                self.forward_refs.len(),
                self.reverse_refs.len(),
                self.var_origins.len(),
            ],
        ));
        self.undo_log.len()
    }

    /// Roll back to a snapshot: pop entries and call `reverse()`.
    ///
    /// All `gen_statuses` mutations (guards, lowering, generalization,
    /// wake transitions, instance promotion) are undo-logged, so a rolled
    /// back status always matches the rolled-back `guard_sets` /
    /// `forward_refs` / `resolutions`.  Residual non-reversible state from
    /// the solve loop: `wait_lists` takes (wake paths) and `type_vars[i].
    /// region_id` (S-Exists-Lower level fallback) — a leaked-but-empty
    /// wait list is conservative (a var stays PG until its next wake),
    /// never unsound.
    pub fn rollback_to(&mut self, snapshot_len: usize) {
        while self.undo_log.len() > snapshot_len {
            let undo = self.undo_log.pop().unwrap();
            self.reverse(undo);
        }
        // Restore region_tree.current FIRST: the undo reversals below (pool
        // unregister, var_to_region writes) only mutate metadata, but any
        // later region operation must run relative to the pre-snapshot
        // position, not the abandoned one.
        if let Some(prev_region) = self.region_current_snapshot_stack.pop() {
            self.region_tree.current = prev_region;
        }
        // Restore resolved_ids/resolved_set/dirty_set/suspended_vars/discharged_matches
        // from snapshot stack, and roll back pool mutations that were made inside this
        // snapshot.
        if let Some((
            ids_len,
            set,
            dirty,
            suspended,
            discharged,
            pool_len,
            pg,
            uf_snap,
            old_generalized,
            old_pending,
            old_discharged_subs,
            vec_lens,
        )) = self.resolved_snapshot_stack.pop()
        {
            self.resolved_ids.truncate(ids_len);
            self.resolved_set = set;
            self.dirty_set = dirty;
            self.suspended_vars = suspended;
            self.discharged_matches = discharged;
            self.discharged_subs = old_discharged_subs;
            self.pg_vars = pg;
            self.union_find.rollback_to(&uf_snap);
            self.generalized_in_region = old_generalized;
            self.pending_inst_copy = old_pending;
            // Roll back only pool entries added since this snapshot was opened,
            // then truncate to preserve entries from outer snapshots.
            while self.region_tree.pool_undo_log.len() > pool_len {
                let entry = self.region_tree.pool_undo_log.pop().unwrap();
                match entry {
                    PoolUndoEntry::Register {
                        region_idx,
                        var_id,
                        kind,
                    } => {
                        if let Some(node) = self.region_tree.nodes.get_mut(region_idx) {
                            let vec = match kind {
                                RegisterKind::Var => &mut node.pool.var_ids,
                                RegisterKind::RigidVar => &mut node.pool.rigid_var_ids,
                            };
                            vec.retain(|&v| v != var_id);
                        }
                        // Also restore var_to_region to mirror rollback_pool.
                        if kind == RegisterKind::Var
                            && var_id < self.region_tree.var_to_region.len()
                        {
                            self.region_tree.var_to_region[var_id] = None;
                        }
                    }
                    PoolUndoEntry::Unregister { region_idx, var_id } => {
                        if let Some(node) = self.region_tree.nodes.get_mut(region_idx) {
                            node.pool.var_ids.push(var_id);
                        }
                        // Also restore var_to_region to mirror rollback_pool.
                        if var_id >= self.region_tree.var_to_region.len() {
                            self.region_tree.var_to_region.resize(var_id + 1, None);
                        }
                        self.region_tree.var_to_region[var_id] = Some(InferRegionId(region_idx));
                    }
                }
            }
            // Restore per-variable vector lengths captured at start_snapshot.
            // Without this, new_type_var's `resize_with` growth is not undone,
            // and the invariant `type_vars.len() == gen_statuses.len() == ...`
            // is violated after rollback. Consumers that index by var_id would
            // then read stale slots from a previous, discarded inference scope.
            let [
                tv_len,
                vti_len,
                lb_len,
                ub_len,
                wl_len,
                psv_len,
                gs_len,
                gst_len,
                fr_len,
                rr_len,
                vo_len,
            ] = vec_lens;
            self.type_vars.truncate(tv_len);
            self.var_type_ids.truncate(vti_len);
            self.lower_bounds.truncate(lb_len);
            self.upper_bounds.truncate(ub_len);
            self.wait_lists.truncate(wl_len);
            self.pending_sub_vars.truncate(psv_len);
            self.guard_sets.truncate(gs_len);
            self.gen_statuses.truncate(gst_len);
            self.forward_refs.truncate(fr_len);
            self.reverse_refs.truncate(rr_len);
            self.var_origins.truncate(vo_len);
        }
        self.snapshot_depth -= 1;
        if self.snapshot_depth == 0 {
            // The reversal loop above already consumed the log down to the
            // snapshot's pool_len; nothing committed can remain — clear and
            // gate off so no-transaction pool mutations stop accumulating.
            self.region_tree.pool_undo_log.clear();
            self.region_tree.pool_logging_enabled = false;
        }
    }

    /// Commit a snapshot: discard undo entries without reversing.
    /// At the outermost snapshot, the log is cleared.
    pub fn commit_snapshot(&mut self, snapshot_len: usize) {
        if self.snapshot_depth == 1 {
            self.undo_log.truncate(snapshot_len);
        }
        self.union_find.commit();
        self.resolved_snapshot_stack.pop();
        // Discard the parallel region-current record: the snapshot is being
        // kept, so the (possibly deeper) current region stays live.
        self.region_current_snapshot_stack.pop();
        self.snapshot_depth -= 1;
        if self.snapshot_depth == 0 {
            // Outermost commit: every pool mutation is now permanent —
            // neither rollback path will ever need these entries again.
            self.region_tree.pool_undo_log.clear();
            self.region_tree.pool_logging_enabled = false;
        }
    }

    /// Push an undo entry if a snapshot is currently open.
    fn push_undo(&mut self, undo: InferUndoLog) {
        if self.snapshot_depth > 0 {
            self.undo_log.push(undo);
        }
    }

    /// Check if a variable has any forward references (instances).
    pub fn has_instances(&self, var_id: usize) -> bool {
        var_id < self.forward_refs.len() && !self.forward_refs[var_id].is_empty()
    }

    /// Check if a variable is an instance of a PG variable.
    pub fn is_instance(&self, var_id: usize) -> Option<usize> {
        if var_id < self.reverse_refs.len() {
            self.reverse_refs[var_id]
        } else {
            None
        }
    }

    /// Discharge a suspended Match constraint: when the scrutinee's shape
    /// is known (unicity holds), add the matched branch's continuation
    /// constraints and remove the guard on the scrutinee variable.
    pub fn discharge_match<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        scrutinee_ty: TypeId,
        branches_id: (usize, usize),
        heap: &mut BucketQueue,
        env: &Arc<SkolemEnv>,
    ) -> bool {
        // Use env-aware resolution so that Forall-bound Skolems are not
        // treated as free inference variables when computing shape.
        let resolved = self.resolve_with_env(scrutinee_ty, ctx, env);
        let shape = Self::shape_of_type(ctx, resolved);
        self.discharge_match_with_shape(ctx, scrutinee_ty, branches_id, heap, env, shape)
    }

    /// [`Self::discharge_match`] with an explicitly supplied shape.
    ///
    /// MATCH-CTX discharge (§4.1) commits a branch for a still-unresolved
    /// scrutinee whose shape was decided by the unicity check.  Computing the
    /// shape with `shape_of_type` on an unresolved InferVar would yield
    /// `PrincipalShape::Var` and select the wrong branch (or only the else
    /// fallback) — the unicity-determined shape must be used instead.
    pub(crate) fn discharge_match_with_shape<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        scrutinee_ty: TypeId,
        branches_id: (usize, usize),
        heap: &mut BucketQueue,
        env: &Arc<SkolemEnv>,
        shape: PrincipalShape,
    ) -> bool {
        // Find the branch that matches this shape.
        let (start, count) = branches_id;
        let end = start + count;
        if start < self.match_branches.len() {
            for i in start..end.min(self.match_branches.len()) {
                let branch = &self.match_branches[i];
                let matches_pattern = shape == branch.shape_pattern;

                if matches_pattern {
                    // Enqueue continuation constraints with the scoped env
                    // so that Forall-bound rigid variables remain rigid.
                    for c in branch.continuation.iter() {
                        let p = self.constraint_priority(&c, ctx);
                        heap.push(PrioritizedConstraint {
                            priority: p,
                            constraint: c.clone(),
                            env: Arc::clone(env),
                        });
                    }
                    return true;
                }
            }

            // No exact match — use the set-level fallback, which by
            // convention lives on the FIRST branch.  Only that one is ever
            // consulted; a non-empty else_continuation on any other branch
            // would be silently ignored, so the invariant is asserted.
            //
            // count == 0 means this set registered no branches; a later set's
            // branches now live at `start`, so consulting `match_branches[start]`
            // would use a foreign else_continuation.  There is no fallback here.
            if count == 0 {
                return false;
            }
            // Clamp `end` to the table length: a rollback may have truncated
            // `match_branches` below `end`, and the debug_assert slice must
            // not go out of bounds.
            let clamped_end = end.min(self.match_branches.len());
            debug_assert!(
                self.match_branches[start + 1..clamped_end]
                    .iter()
                    .all(|b| b.else_continuation.is_empty()),
                "only the first branch of a MatchBranchSet may carry an else_continuation"
            );
            let first = &self.match_branches[start];
            if !first.else_continuation.is_empty() {
                for c in first.else_continuation.iter() {
                    let p = self.constraint_priority(&c, ctx);
                    heap.push(PrioritizedConstraint {
                        priority: p,
                        constraint: c.clone(),
                        env: Arc::clone(env),
                    });
                }
                return true;
            }
        }

        false
    }

    /// Get the generalisation status for a variable.
    pub fn get_gen_status(&self, var_id: usize) -> Option<GenStatus> {
        self.gen_statuses.get(var_id).copied()
    }

    /// Test-support setter for the generation status.  Mirrors the direct
    /// field access the in-module unit tests use; guarded by a bounds check
    /// (vars created via `new_type_var` always have an entry).  Undo-logged
    /// like every other gen_statuses mutation, so snapshots stay consistent.
    pub(crate) fn set_gen_status(&mut self, var_id: usize, status: GenStatus) {
        if var_id < self.gen_statuses.len() {
            let old = self.gen_statuses[var_id];
            self.gen_statuses[var_id] = status;
            // Maintain pg_vars index: insert on PG entry, remove on PG exit.
            if status == GenStatus::PartiallyGeneralizable
                && old != GenStatus::PartiallyGeneralizable
            {
                self.pg_vars.insert(var_id);
            } else if status != GenStatus::PartiallyGeneralizable
                && old == GenStatus::PartiallyGeneralizable
            {
                self.pg_vars.remove(&var_id);
            }
            self.push_undo(InferUndoLog::SetGenStatus(var_id, old));
        }
    }

    /// Single writer for `gen_statuses`: keeps the `pg_vars` index and the
    /// undo log in lockstep.  Every site that assigns `self.gen_statuses[i]`
    /// directly MUST go through here — a raw write leaves `pg_vars` holding
    /// a stale index (a PG→G transition whose index entry survives makes the
    /// generalization sweep re-scan a dead variable; worse, a promoted PG
    /// var that was never inserted can NEVER be swept, so its instances
    /// never receive S-Inst-Copy).  Idempotent calls are no-ops: no index
    /// churn, no undo entry.
    pub(crate) fn write_gen_status(&mut self, var_id: usize, status: GenStatus) {
        if var_id >= self.gen_statuses.len() {
            return;
        }
        let old = self.gen_statuses[var_id];
        if old == status {
            return;
        }
        self.gen_statuses[var_id] = status;
        if status == GenStatus::PartiallyGeneralizable {
            self.pg_vars.insert(var_id);
        } else {
            self.pg_vars.remove(&var_id);
        }
        self.push_undo(InferUndoLog::SetGenStatus(var_id, old));
    }

    /// Mark a variable as guarded by a suspended constraint.
    /// Increments the reference-counted guard on the variable.
    pub fn add_guard(&mut self, var_id: usize) {
        // Grow `guard_sets` to cover var_id.  This extension is deliberately
        // NOT undo-logged: on rollback the vec keeps trailing empty entries,
        // which are identity (`is_empty()` on GuardSet::empty()) and every
        // consumer bounds-checks before reading, so the extra capacity is
        // harmless.  `new_type_var` grows guard_sets and gen_statuses in
        // lockstep, so var_id < gen_statuses.len() holds for tracked vars.
        if self.guard_sets.len() <= var_id {
            self.guard_sets.resize_with(var_id + 1, GuardSet::empty);
        }
        self.push_undo(InferUndoLog::AddGuard(var_id));
        self.guard_sets[var_id].add_guard();
        debug_assert!(
            self.is_guarded(var_id),
            "add_guard: variable {var_id} must be guarded after add_guard"
        );
        // A guarded variable cannot be generalized (OmniML §6): reflect the
        // guard in the gen status, and undo-log the status only when it
        // ACTUALLY changes — an unconditional `SetGenStatus` undo would log
        // no-op entries on the second add of an already-PG variable.
        if var_id < self.gen_statuses.len() {
            let old = self.gen_statuses[var_id];
            if old != GenStatus::PartiallyGeneralizable {
                self.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
                self.pg_vars.insert(var_id);
                self.push_undo(InferUndoLog::SetGenStatus(var_id, old));
            }
        }
    }

    /// Check whether a variable has any active guards.
    /// A guarded variable is PartiallyGeneralizable (PG) and cannot be
    /// generalized until all guards are removed.
    #[inline]
    pub(crate) fn is_guarded(&self, var_id: usize) -> bool {
        var_id < self.guard_sets.len() && !self.guard_sets[var_id].is_empty()
    }

    /// Remove a guard from a variable when its suspended constraint is discharged.
    /// Decrements the reference count. Performs PG→G transition only when
    /// rigid-escape check passes.
    pub fn remove_guard<'input>(&mut self, var_id: usize, ctx: &TypeContext<'input>) {
        // Only log/mutate when the variable actually carries a direct
        // guard: `GuardSet::remove_guard` is a no-op at count 0 (its
        // debug_assert fires), while an unconditional `RemoveGuard` undo
        // entry would re-`add_guard` on rollback — re-creating a guard
        // that never existed and permanently blocking generalization.
        if var_id < self.guard_sets.len() && self.guard_sets[var_id].direct_guards > 0 {
            self.push_undo(InferUndoLog::RemoveGuard(var_id));
            self.guard_sets[var_id].remove_guard();
            if self.guard_sets[var_id].is_empty()
                && var_id < self.gen_statuses.len()
                && self.gen_statuses[var_id] == GenStatus::PartiallyGeneralizable
            {
                let resolved = self.resolve_promotion_aware(self.var_type_ids[var_id], ctx);
                // Only generalize a *resolved* variable.
                // When resolved is still InferVar, the old conjunction
                // `!matches!(..InferVar) && ...` was false, making
                // `!escapes == true`, silently promoting an unbound var.
                if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                    let level = self
                        .type_vars
                        .get(var_id)
                        .map(|v| self.region_tree.get_level(v.region_id))
                        .unwrap_or(0);
                    // Escape must be judged independently of where the solver
                    // currently is.  Conjoining `level <= cur_level` made the
                    // guard vacuously false for every variable in an outer
                    // region (level < cur_level), so `!escape` became true and
                    // the variable was generalized even though its type still
                    // mentions a rigid introduced in that region — a
                    // soundness hole (escaping skolem copied into the scheme).
                    if !Self::check_rigid_escape(
                        ctx,
                        resolved,
                        level,
                        &self.region_tree,
                        &self.var_promotions,
                    ) {
                        self.write_gen_status(var_id, GenStatus::Generalized);
                    }
                }
                // If still unresolved, leave as PG: a later wake will retry.
            }
        }
    }

    pub fn solve<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
        trait_env: &TraitEnv,
        symbols: &SymbolTable<'input>,
    ) -> Result<(), TypeError> {
        // ── Build priority queue ────────────────────────────────────
        let mut heap: BucketQueue = BucketQueue::new();
        // Moving constraints out of `self.constraints` avoids cloning each one.
        // This is safe at solve start when no snapshot is open: `self.constraints`
        // is not needed again until new constraints are pushed during solving.
        // When a snapshot is active (e.g. double-solve or rollback retry), keep
        // the original clone so the caller's view of `self.constraints` is intact.
        let constraints = if self.snapshot_depth == 0 {
            std::mem::take(&mut self.constraints)
        } else {
            self.constraints.clone()
        };
        for c in constraints {
            let priority = self.constraint_priority(&c, ctx);
            heap.push(PrioritizedConstraint {
                priority,
                constraint: c,
                env: empty_skolem_env(),
            });
        }

        // ── Process all constraints in priority order ───────────────
        // With incremental wake-up: after each unification, check if an
        // InferVar was resolved and immediately wake its suspended constraints.
        // This follows OmniML's job-queue pattern where unify enqueues jobs
        // that the scheduler runs immediately.
        loop {
            while let Some(pc) = heap.pop() {
                match &pc.constraint {
                    Constraint::Eq(a, b, eq_span, _) => {
                        // Check for InferVars before unifying.
                        // Promotion-aware: after try_promote_var the old
                        // variable aliases to its representative ONLY in the
                        // solver-local promotions/resolutions maps, which
                        // ctx.resolve_binding cannot follow — using it here
                        // would classify a promoted alias as a fresh
                        // InferVar and redo promotion/unification work.
                        let ra = self.resolve_promotion_aware(*a, ctx);
                        let rb = self.resolve_promotion_aware(*b, ctx);
                        // Trivial satisfaction: identical TypeIds (or roots
                        // already unified to the same type) carry no new
                        // information — skip the promotion / mark_dirty /
                        // wake / S-Exists-Lower machinery instead of
                        // self-unifying the variable with itself.
                        if ra == rb {
                            continue;
                        }
                        // Compute var_ids unconditionally from the raw TypeData
                        // so they are available for Skolem env lookups even when
                        // the variable is treated as rigid (not a free InferVar).
                        let (mut a_var_id, auniverse) = match ctx.get(ra) {
                            TypeData::InferVar { id, universe, .. } => (Some(*id), Some(*universe)),
                            _ => (None, None),
                        };
                        let (mut b_var_id, buniverse) = match ctx.get(rb) {
                            TypeData::InferVar { id, universe, .. } => (Some(*id), Some(*universe)),
                            _ => (None, None),
                        };
                        let mut a_is_infer = a_var_id.is_some();
                        let mut b_is_infer = b_var_id.is_some();

                        // If the InferVar is a Forall-bound Skolem in the constraint's
                        // scoped environment (pc.env), it is NOT a free InferVar
                        // — treat it as rigid so the unifier rejects illegal
                        // unification with concrete types.
                        if a_is_infer
                            && let Some(id) = a_var_id
                            && pc.env.contains_key(&id)
                        {
                            a_is_infer = false;
                        }
                        if b_is_infer
                            && let Some(id) = b_var_id
                            && pc.env.contains_key(&id)
                        {
                            b_is_infer = false;
                        }

                        // Per-variable universe tracking: the bidirectional check
                        // was removed here because `ctx.unify` (called via
                        // `unify_and_track`) already enforces the correct
                        // unidirectional rule: binding a HIGHER-universe variable
                        // into a LOWER universe is rejected (escape), while
                        // low→high is allowed.  The old bidirectional check
                        // incorrectly rejected ALL different-universe pairs.

                        // Level-based promotion (Fan, Xu & Xie 2025 §6.2):
                        // If unifying two InferVars at different regions, promote
                        // the deeper variable to the nearest common ancestor (NCA)
                        // of both regions before unifying.  Using the NCA guarantees
                        // the promoted variable lives in a scope that is accessible
                        // from both original branches — critical when the two
                        // variables live in sibling branches, not just on the same
                        // ancestor-descendant line.
                        if a_is_infer && b_is_infer {
                            if let (Some(avid), Some(bvid)) = (a_var_id, b_var_id) {
                                // Reject promotion when either variable is Skolem-bound.
                                // After promotion, a_var_id/b_var_id are overwritten with
                                // new IDs that are NOT in pc.env, so the Skolem check
                                // below would fail to recognize them as rigid.  Instead,
                                // delegate to the rigid-variable path immediately.
                                let a_is_skolem = pc.env.contains_key(&avid);
                                let b_is_skolem = pc.env.contains_key(&bvid);
                                if a_is_skolem || b_is_skolem {
                                    let lhs = pc.env.get(&avid).copied().unwrap_or(*a);
                                    let rhs = pc.env.get(&bvid).copied().unwrap_or(*b);
                                    match (a_is_skolem, b_is_skolem) {
                                        (true, true) => match ctx.unify(lhs, rhs) {
                                            Ok(_) => {}
                                            Err(mut e) => {
                                                e.set_span(*eq_span);
                                                return Err(e);
                                            }
                                        },
                                        (true, false) | (false, true) => {
                                            return Err(TypeError::Mismatch {
                                                expected: lhs,
                                                found: rhs,
                                                span: *eq_span,
                                            });
                                        }
                                        _ => unreachable!(),
                                    }
                                } else {
                                    // Defensive .get(): ids harvested from
                                    // ctx.get() are not guaranteed to have
                                    // passed through new_type_var (e.g. the
                                    // GENERIC_MATCH_VAR_ID vars minted by the
                                    // trait solver) — index those blindly and
                                    // we panic instead of degrading.
                                    let (a_region, b_region) = match (
                                        self.type_vars.get(avid).map(|v| v.region_id),
                                        self.type_vars.get(bvid).map(|v| v.region_id),
                                    ) {
                                        (Some(ar), Some(br)) => (ar, br),
                                        // Unknown-to-solver vars: fall back to
                                        // a plain unification without the NCA
                                        // promotion logic.
                                        _ => {
                                            if let Err(mut e) =
                                                self.unify_and_track(*a, *b, ctx, *eq_span)
                                            {
                                                e.set_span(*eq_span);
                                                return Err(e);
                                            }
                                            continue;
                                        }
                                    };
                                    let a_lvl = self.region_tree.get_level(a_region);
                                    let b_lvl = self.region_tree.get_level(b_region);
                                    let nca = self
                                        .region_tree
                                        .nearest_common_ancestor(a_region, b_region);
                                    debug_assert!(
                                        self.region_tree.is_ancestor(nca, a_region)
                                            && self.region_tree.is_ancestor(nca, b_region),
                                        "NCA must be an ancestor of both regions",
                                    );
                                    // Track effective var IDs after promotion so that
                                    // post-unification processing (mark_dirty, wake, etc.)
                                    // operates on the promoted variable, not the old alias.
                                    let mut eff_a = avid;
                                    let mut eff_b = bvid;
                                    if a_lvl > b_lvl {
                                        // a is deeper — promote a to NCA.
                                        if let Some(promoted) = self.try_promote_var(ctx, avid, nca)
                                        {
                                            if let TypeData::InferVar { id, .. } = ctx.get(promoted)
                                            {
                                                eff_a = *id;
                                            }
                                            self.unify_and_track(*b, promoted, ctx, *eq_span)?;
                                        } else {
                                            self.unify_and_track(*a, *b, ctx, *eq_span)?;
                                        }
                                    } else if b_lvl > a_lvl {
                                        // b is deeper — promote b to NCA.
                                        if let Some(promoted) = self.try_promote_var(ctx, bvid, nca)
                                        {
                                            if let TypeData::InferVar { id, .. } = ctx.get(promoted)
                                            {
                                                eff_b = *id;
                                            }
                                            self.unify_and_track(*a, promoted, ctx, *eq_span)?;
                                        } else {
                                            self.unify_and_track(*a, *b, ctx, *eq_span)?;
                                        }
                                    } else {
                                        // Same level — NCA handles the sibling / ancestor / equal cases.
                                        if nca == a_region && nca == b_region {
                                            self.unify_and_track(*a, *b, ctx, *eq_span)?;
                                        } else if self.region_tree.is_ancestor(a_region, b_region) {
                                            self.unify_and_track(*a, *b, ctx, *eq_span)?;
                                        } else if self.region_tree.is_ancestor(b_region, a_region) {
                                            self.unify_and_track(*a, *b, ctx, *eq_span)?;
                                        } else {
                                            // Sibling regions at the same level: promote both to NCA.
                                            let a_promoted =
                                                self.try_promote_var(ctx, avid, nca).unwrap_or(*a);
                                            let b_promoted =
                                                self.try_promote_var(ctx, bvid, nca).unwrap_or(*b);
                                            if let TypeData::InferVar { id, .. } =
                                                ctx.get(a_promoted)
                                            {
                                                eff_a = *id;
                                            }
                                            if let TypeData::InferVar { id, .. } =
                                                ctx.get(b_promoted)
                                            {
                                                eff_b = *id;
                                            }
                                            self.unify_and_track(
                                                a_promoted, b_promoted, ctx, *eq_span,
                                            )?;
                                        }
                                    }
                                    // Overwrite a_var_id/b_var_id with effective IDs so that
                                    // the post-unification loop below uses the promoted vars.
                                    a_var_id = Some(eff_a);
                                    b_var_id = Some(eff_b);
                                } // close else (non-skolem promotion path)
                            } else {
                                // Both are genuine free InferVars but the inner
                                // pattern didn't match (shouldn't happen).
                                self.unify_and_track(*a, *b, ctx, *eq_span)?;
                            }
                        } else {
                            // At least one side is not a free InferVar.
                            // If either side is a Forall-bound Skolem, resolve
                            // BOTH sides through the scoped environment and
                            // use ctx.unify directly.  This handles nested
                            // Forall (e.g. ∀α. ∀β. Eq(α, β)) correctly:
                            // both sides are resolved to their Skolem TypeIds
                            // before unification, so ctx.unify sees two
                            // distinct Skolems and rejects the unification.
                            let lhs = a_var_id
                                .and_then(|id| pc.env.get(&id).copied())
                                .unwrap_or(*a);
                            let rhs = b_var_id
                                .and_then(|id| pc.env.get(&id).copied())
                                .unwrap_or(*b);
                            if a_var_id.and_then(|id| pc.env.get(&id)).is_some()
                                || b_var_id.and_then(|id| pc.env.get(&id)).is_some()
                            {
                                // At least one side is a scoped skolem.
                                // If exactly one side is a skolem and the other
                                // is a free inference variable (not in the env),
                                // reject the unification — ctx.unify would bind the
                                // free variable to the skolem, leaking it.
                                let a_skolem = a_var_id.and_then(|id| pc.env.get(&id).copied());
                                let b_skolem = b_var_id.and_then(|id| pc.env.get(&id).copied());
                                match (a_skolem, b_skolem) {
                                    (Some(sa), Some(sb)) => {
                                        // Two rigid vars: only identical skolems unify.
                                        match ctx.unify(sa, sb) {
                                            Ok(_) => {}
                                            Err(mut e) => {
                                                e.set_span(*eq_span);
                                                return Err(e);
                                            }
                                        }
                                    }
                                    (Some(_), None) if b_var_id.is_some() => {
                                        // Skolem vs free infer var: reject to prevent
                                        // the free var from being bound to the skolem.
                                        return Err(TypeError::Mismatch {
                                            expected: lhs,
                                            found: rhs,
                                            span: *eq_span,
                                        });
                                    }
                                    (None, Some(_)) if a_var_id.is_some() => {
                                        // Free infer var vs skolem: reject.
                                        return Err(TypeError::Mismatch {
                                            expected: rhs,
                                            found: lhs,
                                            span: *eq_span,
                                        });
                                    }
                                    (Some(sa), None) => match ctx.unify(sa, *b) {
                                        Ok(_) => {}
                                        Err(mut e) => {
                                            e.set_span(*eq_span);
                                            return Err(e);
                                        }
                                    },
                                    (None, Some(sb)) => match ctx.unify(*a, sb) {
                                        Ok(_) => {}
                                        Err(mut e) => {
                                            e.set_span(*eq_span);
                                            return Err(e);
                                        }
                                    },
                                    (None, None) => {
                                        self.unify_and_track(*a, *b, ctx, *eq_span)?;
                                    }
                                }
                            } else {
                                self.unify_and_track(*a, *b, ctx, *eq_span)?;
                            }
                        }

                        // Mark dirty for drain_dirty tracking.
                        for var_id in [a_var_id, b_var_id].iter().flatten() {
                            self.mark_dirty(*var_id);
                        }

                        // S-Exists-Lower needs the Eq constraints currently in
                        // the heap; snapshot + index them lazily so an Eq with
                        // two PG variables builds them once instead of once
                        // per variable.
                        let mut s_exists_cache: Option<(
                            Vec<PrioritizedConstraint>,
                            HashMap<usize, Vec<(TypeId, TypeId)>>,
                        )> = None;

                        // Incremental wake-up: if a variable was just resolved,
                        // immediately enqueue its suspended constraints.
                        for var_id in [a_var_id, b_var_id].iter().flatten() {
                            // Wake on RESOLUTION, not only on shape change:
                            // try_set_shape returns true only when the stored
                            // PrincipalShape actually differs, so a re-unify
                            // that keeps the shape (α already Scalar, or an
                            // Eq chain resolving to an already-recorded
                            // shape) would skip the fast path here and wait
                            // for the outer sweep.  Waking on resolution is
                            // idempotent — an empty wait list just no-ops.
                            let resolved_now = !matches!(
                                ctx.get(
                                    self.resolve_promotion_aware(self.var_type_ids[*var_id], ctx)
                                ),
                                TypeData::InferVar { .. }
                            );
                            let shape_changed = self.try_set_shape(*var_id, ctx);
                            if shape_changed || resolved_now {
                                self.wake_var_incremental(*var_id, &mut heap, ctx);
                            }
                            // ── OmniML §5.2: Incremental instantiation ──
                            // S-Inst-Copy fires on the PG→G *transition*, not while
                            // the var is still PG.  After wake_var_incremental clears
                            // the wait list and sets the status to Generalized (no
                            // guards remain), we propagate the resolved type to all
                            // instances.  This avoids two bugs:
                            //   1. Premature copy: s_inst_copy while still PG would
                            //      clear forward_refs, dropping future refinements.
                            //   2. Missed copy: if wake_var_incremental already set
                            //      Generalized, the old PG guard was dead code.
                            if *var_id < self.gen_statuses.len()
                                && self.gen_statuses[*var_id] == GenStatus::Generalized
                            {
                                let resolved_ty =
                                    self.resolve_promotion_aware(self.var_type_ids[*var_id], ctx);
                                if !matches!(ctx.get(resolved_ty), TypeData::InferVar { .. })
                                    && !Self::ty_mentions_infer_var(ctx, resolved_ty)
                                {
                                    self.s_inst_copy(ctx, *var_id, resolved_ty);
                                }
                            }
                            // ── S-Generalize for eligible current vars ──
                            // The sweep runs once per outer iteration (batched),
                            // so a current var that is eligible for generalization
                            // (resolved, unguarded, nothing waiting, no rigid
                            // escape) is transitioned HERE — before
                            // S-Exists-Lower — keeping the PG→G transition and
                            // S-Inst-Copy deterministic instead of dependent on
                            // whether an earlier transition in this iteration
                            // already triggered a sweep.
                            if *var_id < self.gen_statuses.len()
                                && self.gen_statuses[*var_id] == GenStatus::PartiallyGeneralizable
                                && self.is_sweep_eligible(ctx, *var_id)
                            {
                                self.write_gen_status(*var_id, GenStatus::Generalized);
                                let resolved_ty =
                                    self.resolve_promotion_aware(self.var_type_ids[*var_id], ctx);
                                if !Self::ty_mentions_infer_var(ctx, resolved_ty) {
                                    self.s_inst_copy(ctx, *var_id, resolved_ty);
                                }
                            }
                            // ── S-Exists-Lower (OmniML §5.3) ──
                            // For vars that remain PG (still have guards/waiting),
                            // try lowering if uniquely determined.  The syntactic
                            // unicity check runs on the Eq constraints currently
                            // in the heap (collected lazily — only for PG vars);
                            // if that fails, the Z3-backed check runs internally.
                            if *var_id < self.gen_statuses.len()
                                && self.gen_statuses[*var_id] == GenStatus::PartiallyGeneralizable
                            {
                                let slot = s_exists_cache.get_or_insert_with(|| {
                                    let active: Vec<PrioritizedConstraint> = heap
                                        .iter()
                                        .filter(|pc| matches!(pc.constraint, Constraint::Eq(..)))
                                        .cloned()
                                        .collect();
                                    let index = self.build_eq_index(ctx, &active);
                                    (active, index)
                                });
                                if self.s_exists_lower(
                                    ctx,
                                    *var_id,
                                    &pc.env,
                                    &slot.0,
                                    Some(&slot.1),
                                ) {
                                    // lowering succeeded
                                }
                            }
                        }
                    }
                    Constraint::Sub(sub, sup, _span) => {
                        // Skip Sub constraints already fully discharged
                        // (both sides resolved and subtype-checked).  This
                        // prevents redundant re-processing from duplicate
                        // wake-ups when both sides resolve independently.
                        if self.discharged_subs.contains(&(*sub, *sup)) {
                            continue;
                        }

                        // Resolve both sides through the scoped env so that
                        // Forall-bound Skolems are used instead of raw InferVars.
                        let resolved_sub = self.resolve_with_env(*sub, ctx, &pc.env);
                        let resolved_sup = self.resolve_with_env(*sup, ctx, &pc.env);

                        // Idempotent bound recording — skip if already present.
                        // Without this, repeated wake/suspend cycles grow bounds
                        // without bound, causing O(n²) redundant subtype checks.
                        // InferVar bounds are never recorded: they carry no
                        // information (the exhaustion pass skips them), so
                        // recording them only bloats the O(n) `contains` scan.
                        if let Some(id) = self.free_infer_var_id(*sup, ctx, &pc.env)
                            && id < self.lower_bounds.len()
                            && !matches!(ctx.get(resolved_sub), TypeData::InferVar { .. })
                            && !self.lower_bounds[id].contains(&resolved_sub)
                        {
                            self.lower_bounds[id].push(resolved_sub);
                            self.mark_dirty(id);
                        }
                        if let Some(id) = self.free_infer_var_id(*sub, ctx, &pc.env)
                            && id < self.upper_bounds.len()
                            && !matches!(ctx.get(resolved_sup), TypeData::InferVar { .. })
                            && !self.upper_bounds[id].contains(&resolved_sup)
                        {
                            self.upper_bounds[id].push(resolved_sup);
                            self.mark_dirty(id);
                        }

                        // If either side is still a free InferVar, suspend the
                        // constraint so it is re-checked when the variable resolves.
                        // Without this, the Sub constraint is silently dropped and
                        // a later binding (e.g. α := Int after α ≤ Bool) is never caught.
                        let sub_free = self.free_infer_var_id(*sub, ctx, &pc.env);
                        let sup_free = self.free_infer_var_id(*sup, ctx, &pc.env);
                        match (sub_free, sup_free) {
                            (None, None) => {
                                // Both resolved: discharge.  Record in
                                // discharged_subs so any duplicate copy still
                                // sitting in another wait_list is skipped.
                                self.discharged_subs.insert((*sub, *sup));
                                // Clean up pending_sub_vars entries for this
                                // Sub across all variables.  In practice the
                                // entries are already gone (taken by
                                // wake_var_incremental), but defensive cleanup
                                // guards against edge cases.
                                for slot in self.pending_sub_vars.iter_mut() {
                                    slot.remove(&(*sub, *sup));
                                }
                                if !ctx.subtype(resolved_sub, resolved_sup) {
                                    return Err(TypeError::Mismatch {
                                        expected: resolved_sup,
                                        found: resolved_sub,
                                        span: *_span,
                                    });
                                }
                            }
                            (Some(sub_id), Some(sup_id)) => {
                                // Both sides are free: suspend on BOTH variables,
                                // with per-variable dedup to prevent entry
                                // doubling (2^k entries after k wake-suspend
                                // cycles).  Each live registration carries
                                // exactly one guard; wake_var_incremental
                                // clears all guards on a variable atomically.
                                if sub_id < self.pending_sub_vars.len()
                                    && self.pending_sub_vars[sub_id].insert((*sub, *sup))
                                {
                                    self.suspend_on_var_with_env(
                                        pc.constraint.clone(),
                                        Arc::clone(&pc.env),
                                        sub_id,
                                    );
                                }
                                if sup_id < self.pending_sub_vars.len()
                                    && self.pending_sub_vars[sup_id].insert((*sub, *sup))
                                {
                                    self.suspend_on_var_with_env(
                                        pc.constraint.clone(),
                                        Arc::clone(&pc.env),
                                        sup_id,
                                    );
                                }
                            }
                            (Some(id), None) | (None, Some(id)) => {
                                // Exactly one side is free: suspend on that side,
                                // with dedup.
                                if id < self.pending_sub_vars.len()
                                    && self.pending_sub_vars[id].insert((*sub, *sup))
                                {
                                    self.suspend_on_var_with_env(
                                        pc.constraint.clone(),
                                        Arc::clone(&pc.env),
                                        id,
                                    );
                                }
                            }
                        }
                    }
                    Constraint::Match {
                        scrutinee,
                        branches_id,
                        span: match_span,
                    } => {
                        // Skip matches already discharged in a previous outer iteration.
                        if self
                            .discharged_matches
                            .contains(&(*scrutinee, *branches_id))
                        {
                            continue;
                        }
                        // OmniML §4.1: Try to discharge suspended match constraints.
                        // Check unicity — if the scrutinee's shape is uniquely determined,
                        // discharge the match and enqueue continuation constraints.
                        // Use env-aware resolution so that Forall-bound Skolems are not
                        // treated as free inference variables.
                        let resolved = self.resolve_with_env(*scrutinee, ctx, &pc.env);
                        let resolved_data = ctx.get(resolved);

                        if !matches!(resolved_data, TypeData::InferVar { .. }) {
                            // Scrutinee is resolved — shape is known (UNI-TYPE).
                            if !self.discharge_match(
                                ctx,
                                *scrutinee,
                                *branches_id,
                                &mut heap,
                                &pc.env,
                            ) {
                                return Err(TypeError::PatternNotExhaustive { span: *match_span });
                            }
                            self.discharged_matches.insert((*scrutinee, *branches_id));
                        } else {
                            // Scrutinee is InferVar — suspend Match on the variable.
                            // try_match_via_shape_var always returns true for InferVar
                            // (it suspends via suspend_on_var_with_env).
                            let handled = self.try_match_via_shape_var(
                                ctx,
                                *scrutinee,
                                *branches_id,
                                &mut heap,
                                *match_span,
                                &pc.env,
                            );
                            debug_assert!(
                                handled,
                                "try_match_via_shape_var must return true for InferVar"
                            );
                        }
                    }
                    Constraint::Exists {
                        var_id,
                        constraint,
                        span: _,
                    } => {
                        // OmniML: ∃α. C — bind a fresh flexible variable.
                        // Enter a new region so that variables created inside
                        // the Exists body are properly scoped (OmniML §5.3).
                        _ = self.region_tree.enter_region();
                        let region_id = self.region_tree.current;
                        // Move the Exists-bound variable into the new region
                        // so pool membership and generalization scope are correct.
                        if *var_id < self.type_vars.len() {
                            let old_region = self.type_vars[*var_id].region_id;
                            if old_region != region_id {
                                self.region_tree.unregister_var(*var_id, old_region);
                                self.region_tree.register_var_in_region(*var_id, region_id);
                                self.push_undo(InferUndoLog::SetVarRegion(*var_id, old_region));
                                self.type_vars[*var_id].region_id = region_id;
                            }
                        }
                        let inner = constraint.as_ref().clone();
                        let p = self.constraint_priority(&inner, ctx);
                        heap.push(PrioritizedConstraint {
                            priority: p,
                            constraint: inner,
                            env: Arc::clone(&pc.env),
                        });
                        // ExitRegion fires after all inner constraints at ADMIN priority.
                        heap.push(PrioritizedConstraint {
                            priority: PRIORITY_ADMIN,
                            constraint: Constraint::ExitRegion {
                                region_id,
                                span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                            },
                            env: Arc::clone(&pc.env),
                        });
                    }
                    Constraint::Forall {
                        var_id,
                        constraint,
                        span: _,
                    } => {
                        // OmniML: ∀α. C — bind a fresh rigid (skolem) variable.
                        // Enter a new universe and create a SkolemVar, then bind
                        // the corresponding InferVar to the SkolemVar so that any
                        // subsequent unification with a non-skolem type is rejected
                        // (the unification code's catch-all returns Mismatch for
                        // SkolemVar/non-SkolemVar or SkolemVar/SkolemVar pairs).
                        //
                        // The SkolemVar uses `enter_universe` so that its
                        // `universe_num` can be checked by `check_skolem_escape`
                        // on TypeContext<'input>, preventing the skolem from leaking
                        // into outer scopes via generalization.
                        //
                        // The Skolem mapping is recorded in the constraint's
                        // scoped environment (pc.env) rather than in global
                        // UF root values.  This ensures the rigid binding
                        // applies only to the Forall body and is automatically
                        // cleaned up when the constraint is popped from the heap.
                        debug_assert!(
                            *var_id < self.var_type_ids.len(),
                            "Forall var_id {} out of range (var_type_ids.len = {})",
                            *var_id,
                            self.var_type_ids.len(),
                        );
                        if *var_id < self.var_type_ids.len() {
                            let (universe, skolem_ty) = ctx.enter_universe();
                            // `universe` is an allocation ordinal, not a scope
                            // depth.  Record the depth it was allocated at so
                            // `check_rigid_escape` can compare like with like.
                            let skolem_level = self.region_tree.get_level(self.region_tree.current);
                            self.region_tree.record_skolem_level(universe, skolem_level);
                            let mut child_env = SkolemEnv::clone(&pc.env);
                            child_env.insert(*var_id, skolem_ty);
                            // DESIGN NOTE: Forall does NOT enter a region. It uses
                            // universe-based scoping via SkolemEnv carried with each
                            // constraint. This is critical: entering a region here would
                            // cause the region-level generalization sweep to interact
                            // with skolem bindings, breaking the rigid-escape check.
                            // See OmniML §3.4 (unicity conditions) and §5.3 (S-EXISTS-LOWER).
                            // If you need region-based scoping, use Exists, not Forall.
                            let inner = constraint.as_ref().clone();
                            let p = self.constraint_priority(&inner, ctx);
                            heap.push(PrioritizedConstraint {
                                priority: p,
                                constraint: inner,
                                env: Arc::new(child_env),
                            });
                        } else {
                            // Invariant violated: the Forall-bound variable
                            // has no corresponding InferVar. Pushing the body
                            // without the skolem mapping would treat the
                            // rigid variable as a free InferVar and let it
                            // unify with a concrete type — a soundness hole.
                            // Fail loudly (panic keeps unwind semantics, so
                            // the query system's poison/catch_unwind path
                            // still reports it as a diagnostic).
                            panic!(
                                "Forall var_id {} out of range (var_type_ids.len = {})",
                                var_id,
                                self.var_type_ids.len(),
                            );
                        }
                    }
                    Constraint::Instance {
                        scheme_ty,
                        instantiation_ty,
                        pg_var_ids,
                        span: inst_span,
                    } => {
                        // ── OmniML instantiation (S-Let-AppR) ─────────────
                        // `Instance(scheme_ty, instantiation_ty)` holds when
                        // `instantiation_ty` is a valid instance of `scheme_ty`.
                        // If scheme_ty = ∀α₁...∀αₙ. τ_body, create fresh
                        // InferVars β₁...βₙ and constrain:
                        //   Eq(instantiation_ty, τ_body[α₁:=β₁,...,αₙ:=βₙ])

                        // Phase 1: Scan scheme type (immutable borrow on ctx)
                        // to collect binders without holding any borrow.
                        let resolved_scheme = ctx.resolve_binding(*scheme_ty);
                        let scheme_info = {
                            // Use a block to limit the borrow scope.
                            match ctx.get(resolved_scheme) {
                                TypeData::Forall { .. } => {
                                    // Collect all Forall binders by walking the chain.
                                    let mut indices: Vec<usize> = Vec::new();
                                    let mut inner = resolved_scheme;
                                    loop {
                                        match ctx.get(inner) {
                                            TypeData::Forall {
                                                param_index, body, ..
                                            } => {
                                                indices.push(*param_index);
                                                inner = *body;
                                            }
                                            _ => break,
                                        }
                                    }
                                    // The final `inner` is the body of the innermost Forall.
                                    Some(InstantiationTarget::Forall {
                                        binder_indices: indices,
                                        body_ty: inner,
                                    })
                                }
                                TypeData::Poly { quantifiers, body } => {
                                    let indices: Vec<usize> =
                                        quantifiers.iter().map(|(idx, _)| *idx).collect();
                                    Some(InstantiationTarget::Poly {
                                        binder_indices: indices,
                                        body_ty: *body,
                                    })
                                }
                                TypeData::InferVar { id, .. } => {
                                    // Scheme is still unknown — suspend so the
                                    // Instance is re-processed when the scheme resolves.
                                    // Without this, treating it as monomorphic defeats
                                    // partial type schemes (OmniML §5.2).
                                    self.suspend_on_var_with_env(
                                        pc.constraint.clone(),
                                        Arc::clone(&pc.env),
                                        *id,
                                    );
                                    None
                                }
                                _ if Self::ty_mentions_infer_var(ctx, resolved_scheme) => {
                                    // Scheme contains nested InferVars (e.g. Fn(InferVar, Bool));
                                    // suspend until those variables are resolved.
                                    // InferVar found instead of treating as concrete.
                                    // Without this, fresh vars won't be registered as
                                    // instances of the PG vars inside the scheme.
                                    if let Some(first_var) =
                                        Self::find_first_infer_var(ctx, resolved_scheme)
                                    {
                                        self.suspend_on_var_with_env(
                                            pc.constraint.clone(),
                                            Arc::clone(&pc.env),
                                            first_var,
                                        );
                                    }
                                    None
                                }
                                _ => {
                                    // Concrete (or error) — just unify directly.
                                    Some(InstantiationTarget::Concrete(resolved_scheme))
                                }
                            }
                        };

                        // Phase 2: Create fresh vars and emit Eq (mutable borrow).
                        if let Some(target) = scheme_info {
                            // Collect fresh variables so we can register instances.
                            let mut fresh_vars: Vec<usize> = Vec::new();
                            // Extract binder_indices before the consuming match
                            // so we can derive PG var IDs when pg_var_ids is empty.
                            let binder_indices_copy: Vec<usize> = match &target {
                                InstantiationTarget::Forall { binder_indices, .. }
                                | InstantiationTarget::Poly { binder_indices, .. } => {
                                    binder_indices.clone()
                                }
                                _ => vec![],
                            };
                            let eq_c = match target {
                                InstantiationTarget::Forall {
                                    binder_indices,
                                    body_ty,
                                } => {
                                    // Allocate fresh vars in binder_indices
                                    // order FIRST, then substitute in any
                                    // order.  `fresh_vars` is zipped against
                                    // `effective_pg_ids` (forward binder
                                    // order) below — pushing during a `.rev()`
                                    // substitution loop reversed it, pairing
                                    // the innermost fresh var with the
                                    // outermost PG var.
                                    let mut fv_map: HashMap<usize, TypeId> = HashMap::default();
                                    for &idx in binder_indices.iter() {
                                        let fv = self.new_type_var(
                                            ctx,
                                            TypeVariableKind::Any,
                                            VarOrigin::Synthetic,
                                        );
                                        if let TypeData::InferVar { id, .. } = ctx.get(fv) {
                                            fresh_vars.push(*id);
                                        }
                                        fv_map.insert(idx, fv);
                                    }
                                    let mut instantiated = body_ty;
                                    for &idx in binder_indices.iter().rev() {
                                        let fv = fv_map[&idx];
                                        instantiated = ctx.replace_generic(instantiated, idx, fv);
                                    }
                                    Constraint::Eq(
                                        *instantiation_ty,
                                        instantiated,
                                        *inst_span,
                                        EqOrigin::Normal,
                                    )
                                }
                                InstantiationTarget::Poly {
                                    binder_indices,
                                    body_ty,
                                } => {
                                    let mut instantiated = body_ty;
                                    for &idx in binder_indices.iter() {
                                        let fv = self.new_type_var(
                                            ctx,
                                            TypeVariableKind::Any,
                                            VarOrigin::Synthetic,
                                        );
                                        // Track fresh var for instance registration.
                                        if let TypeData::InferVar { id, .. } = ctx.get(fv) {
                                            fresh_vars.push(*id);
                                        }
                                        instantiated = ctx.replace_generic(instantiated, idx, fv);
                                    }
                                    Constraint::Eq(
                                        *instantiation_ty,
                                        instantiated,
                                        *inst_span,
                                        EqOrigin::Normal,
                                    )
                                }
                                InstantiationTarget::Concrete(target_ty) => Constraint::Eq(
                                    *instantiation_ty,
                                    target_ty,
                                    *inst_span,
                                    EqOrigin::Normal,
                                ),
                            };
                            // Register each fresh var as an instance of the
                            // corresponding PG var. When pg_var_ids is empty (backward
                            // compat), derive from the scheme's binders by checking
                            // if the corresponding GenericParam is tracked as PG.
                            // POSITIONAL, not compressed: fresh_vars[i] was
                            // pushed in binder_indices order, so entry i must
                            // correspond to binder i.  A filter_map here would
                            // compress non-PG binders out and pair every PG
                            // var AFTER the first with the WRONG fresh var.
                            let effective_pg_ids: Vec<Option<usize>> = if pg_var_ids.is_empty() {
                                binder_indices_copy
                                    .iter()
                                    .map(|&idx| {
                                        self.generic_param_bindings.get(&idx).and_then(|ty| {
                                            if let TypeData::InferVar { id, .. } = ctx.get(*ty) {
                                                if *id < self.gen_statuses.len()
                                                    && self.gen_statuses[*id]
                                                        == GenStatus::PartiallyGeneralizable
                                                {
                                                    return Some(*id);
                                                }
                                            }
                                            None
                                        })
                                    })
                                    .collect()
                            } else {
                                let mut v: Vec<Option<usize>> =
                                    pg_var_ids.iter().copied().map(Some).collect();
                                v.resize(fresh_vars.len(), None);
                                v
                            };
                            for (pos, &fresh_id) in fresh_vars.iter().enumerate() {
                                if let Some(Some(pg_id)) = effective_pg_ids.get(pos) {
                                    self.register_instance(*pg_id, fresh_id);
                                }
                            }
                            let p = self.constraint_priority(&eq_c, ctx);
                            heap.push(PrioritizedConstraint {
                                priority: p,
                                constraint: eq_c,
                                env: Arc::clone(&pc.env),
                            });
                        }
                    }
                    Constraint::Let {
                        expr_var,
                        def_constraint,
                        body_constraint,
                        span,
                    } => {
                        // Enter a region for the let-definition scope.
                        // OmniML §5.2: let-constraint introduces a region whose
                        // variables are generalized (possibly partially) before
                        // the body is processed.
                        _ = self.region_tree.enter_region();
                        let let_region = self.region_tree.current;
                        // Push def first, then a barrier that will generalize
                        // the let-region and push body only after def constraints
                        // are fully processed.
                        let def_p = self.constraint_priority(&def_constraint, ctx);
                        heap.push(PrioritizedConstraint {
                            priority: def_p,
                            constraint: def_constraint.as_ref().clone(),
                            env: Arc::clone(&pc.env),
                        });
                        heap.push(PrioritizedConstraint {
                            priority: PRIORITY_LET_BARRIER,
                            constraint: Constraint::LetBarrier {
                                expr_var: expr_var.clone(),
                                body_constraint: body_constraint.clone(),
                                region_id: let_region,
                                span: *span,
                            },
                            env: Arc::clone(&pc.env),
                        });
                    }
                    Constraint::LetBarrier {
                        expr_var: _,
                        body_constraint,
                        region_id,
                        span: _,
                    } => {
                        // Before exiting nested regions, wake any suspended constraints
                        // in the regions being abandoned. Without this, variables
                        // registered in a child region would remain stuck forever.
                        // (e.g. from a suspended Exists) are abandoned without
                        // their constraints being processed.
                        // Ancestry-guarded (mirrors ExitRegion): only unwind
                        // genuine DESCENDANTS of the let-region.  If an
                        // ExitRegion overtook us and already exited past the
                        // let-region, `is_ancestor` is false and we must NOT
                        // keep walking — that would close unrelated outer
                        // regions up to the root.
                        while self.region_tree.current != *region_id
                            && self
                                .region_tree
                                .is_ancestor(*region_id, self.region_tree.current)
                        {
                            let current = self.region_tree.current;
                            let vars: Vec<usize> =
                                self.region_tree.nodes[current.0].pool.var_ids.clone();
                            for vid in vars {
                                if vid < self.wait_lists.len() && !self.wait_lists[vid].is_empty() {
                                    self.wake_var_incremental(vid, &mut heap, ctx);
                                }
                            }
                            self.region_tree.exit_region();
                        }
                        if self.region_tree.current == *region_id {
                            // §6 region-exit generalization, same as
                            // ExitRegion: drain BEFORE leaving so a
                            // non-generalizable PG var lowers into the
                            // parent pool and is revisited at the parent's
                            // exit.
                            self.drain_dirty_generalization(ctx);
                            self.region_tree.exit_region();
                        }
                        // Generalization for the let-region already happened
                        // in the drain above (§6 frontier: registration
                        // marked the region dirty, and drain_dirty_
                        // generalization ran before exit_region).  The old
                        // targeted force_generalize_for_regions call here
                        // was a redundant second pass over the same level.
                        let body_p = self
                            .constraint_priority(&body_constraint, ctx)
                            .max(PRIORITY_BODY_FLOOR);
                        heap.push(PrioritizedConstraint {
                            priority: body_p,
                            constraint: body_constraint.as_ref().clone(),
                            env: Arc::clone(&pc.env),
                        });
                    }
                    Constraint::ExitRegion { region_id, .. } => {
                        // LIFO ordering normally guarantees inner regions exit
                        // before outer ones, but a `Let` nested in the body
                        // pops its LetBarrier (PRIORITY_LET_BARRIER, 8) AFTER
                        // this handler (PRIORITY_ADMIN, 7) — so `current` can
                        // still sit on a descendant region here.  Walk up to
                        // the recorded region before leaving it, mirroring the
                        // tolerance in the LetBarrier handler: exit_region()
                        // only stamps tout and moves `current`, so the walk
                        // has no side effects on guards or pools, and the
                        // LetBarrier's own upward walk is a no-op afterwards
                        // (its generalisation is keyed on the recorded
                        // region_id, not on `current`).
                        //
                        // The guard is ANCESTRY, not "reached the root": if
                        // this region was already exited by an earlier
                        // overtake (a repeated ExitRegion, or a LetBarrier
                        // that unwound past us), `current` is an ANCESTOR
                        // and `is_ancestor(region_id, current)` is false —
                        // stopping at the root instead would keep exiting
                        // UNRELATED outer regions all the way up.
                        while self.region_tree.current != *region_id
                            && self
                                .region_tree
                                .is_ancestor(*region_id, self.region_tree.current)
                        {
                            self.region_tree.exit_region();
                        }
                        // §6 region-exit generalization: drain the dirty
                        // frontier BEFORE leaving — pools are still alive, so
                        // a PG variable that cannot generalize here is
                        // correctly lowered into the parent pool and revisited
                        // when the parent region exits.
                        self.drain_dirty_generalization(ctx);
                        if self.region_tree.current == *region_id {
                            self.region_tree.exit_region();
                        }
                        // If `current` stopped at the root we overshot the
                        // recorded region (a prior ExitRegion already left
                        // it) — nothing to do; exit_region is a no-op there
                        // anyway.
                    }
                }
            }

            // ── OmniML §4.1: MATCH-CTX for suspended Match constraints ──
            // Wrap MATCH-CTX in a fixed-point loop: discharging match A may
            // resolve variable X, which enables match B (suspended on X) to
            // also discharge in the same outer iteration.
            {
                let mut match_ctx_changed = true;
                while match_ctx_changed {
                    match_ctx_changed = false;
                    // Collect active Eq constraints from the heap so
                    // UNI-VAR can discover shapes from equations like
                    // α = Int that have not yet been processed by the main
                    // constraint loop.
                    let active_eqs: Vec<PrioritizedConstraint> = heap
                        .iter()
                        .filter(|pc| matches!(pc.constraint, Constraint::Eq(..)))
                        .cloned()
                        .collect();
                    let mut match_ctx: Vec<(usize, TypeId, (usize, usize), Arc<SkolemEnv>)> =
                        Vec::new();
                    for &vid in self.suspended_vars.iter() {
                        if vid >= self.wait_lists.len() {
                            continue;
                        }
                        for (c, env) in &self.wait_lists[vid] {
                            if let Constraint::Match {
                                scrutinee,
                                branches_id,
                                ..
                            } = c
                            {
                                if !self
                                    .discharged_matches
                                    .contains(&(*scrutinee, *branches_id))
                                {
                                    match_ctx.push((
                                        vid,
                                        *scrutinee,
                                        *branches_id,
                                        Arc::clone(env),
                                    ));
                                }
                            }
                        }
                    }
                    for (vid, scrutinee, branches_id, env) in match_ctx {
                        if self.discharged_matches.contains(&(scrutinee, branches_id)) {
                            continue;
                        }
                        let resolved = self.resolve_promotion_aware(scrutinee, ctx);
                        if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                            // Resolved scrutinee — the wake path below handles it.
                            continue;
                        }
                        if let Some(unicity_shape) =
                            Self::unicity_check(self, ctx, scrutinee, &env, &active_eqs)
                        {
                            if self.discharge_match_with_shape(
                                ctx,
                                scrutinee,
                                branches_id,
                                &mut heap,
                                &env,
                                unicity_shape,
                            ) {
                                match_ctx_changed = true;
                                self.discharged_matches.insert((scrutinee, branches_id));
                                // Remove the discharged entry from the wait list it
                                // was suspended on.
                                if vid < self.wait_lists.len() {
                                    if self.snapshot_depth > 0 {
                                        self.push_undo(InferUndoLog::TakeWaitList(
                                            vid,
                                            self.wait_lists[vid].clone(),
                                        ));
                                    }
                                    self.wait_lists[vid].retain(|(c, _)| {
                                        !matches!(c, Constraint::Match {
                                            scrutinee: s,
                                            branches_id: b,
                                            ..
                                        } if *s == scrutinee && *b == branches_id)
                                    });
                                }
                                self.remove_guard(vid, ctx);
                                let wl_empty =
                                    vid < self.wait_lists.len() && self.wait_lists[vid].is_empty();
                                let guards_empty =
                                    vid < self.guard_sets.len() && self.guard_sets[vid].is_empty();
                                if wl_empty && guards_empty {
                                    self.suspended_vars.remove(&vid);
                                }
                            }
                        }
                    }
                }
            }

            // ── Batch force_generalize (OmniML §5.3) ──────────────────
            // One sweep per outer iteration (after the heap drains and the
            // Match phase has removed guards), replacing the per-transition
            // sweep in the Eq handler.  The sweep only sets the status, so
            // for every var it transitions PG → Generalized we replicate
            // S-Inst-Copy — otherwise forward-referenced instances of a
            // sweep-transitioned var would keep unresolved InferVars.
            if !self.dirty_set.is_empty() {
                let swept: Vec<usize> = self
                    .dirty_set
                    .iter()
                    .copied()
                    .filter(|i| {
                        self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable)
                    })
                    .collect();
                self.force_generalize_for_regions(ctx, &[], None);
                for var_id in swept {
                    if self.gen_statuses.get(var_id) == Some(&GenStatus::Generalized)
                        && let Some(ty_id) = self.var_type_ids.get(var_id)
                    {
                        // promotion-aware: a promoted var's binding lives
                        // only in the solver-local maps, and a ctx-only
                        // resolve here would skip s_inst_copy for it.
                        let resolved_ty = self.resolve_promotion_aware(*ty_id, ctx);
                        if !matches!(ctx.get(resolved_ty), TypeData::InferVar { .. })
                            && !Self::ty_mentions_infer_var(ctx, resolved_ty)
                        {
                            self.s_inst_copy(ctx, var_id, resolved_ty);
                        }
                    }
                }
            }

            // ── Wake-up: reprocess suspended constraints ───────────────
            // After processing all active constraints, check if any variables
            // were resolved. If so, wake their wait-listed constraints and
            // continue solving (OmniML bidirectional flow §3.2).
            // Only vars with non-empty wait lists can be woken, so the
            // (potentially chain-walking) resolve_binding is gated on the
            // cheap wait-list check first — the wake behavior is unchanged.
            let mut woken = 0usize;
            // Only variables with suspended constraints can possibly have wait
            // lists that need waking.  This uses the incremental suspended_vars
            // set rather than scanning every variable.  We drain the set here
            // and re-insert any variable whose wait list is still non-empty
            // after waking (e.g. because a newly-suspended constraint was added
            // during processing of an earlier woken constraint).
            self.wake_buffer.clear();
            let mut tmp = std::mem::take(&mut self.suspended_vars);
            self.wake_buffer.extend(tmp.drain());
            self.suspended_vars = tmp;
            let to_wake: Vec<usize> = self.wake_buffer.drain(..).collect();
            for i in to_wake {
                if i < self.wait_lists.len() && !self.wait_lists[i].is_empty() {
                    let ty_id = self.var_type_ids[i];
                    // promotion-aware: promoted vars bind solver-locally, so
                    // a ctx-only resolve would never wake their wait lists.
                    let resolved = self.resolve_promotion_aware(ty_id, ctx);
                    if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                        // Unified wake path: wake_var_incremental handles
                        // wait_list drain, guard clearing, and PG→G transition.
                        let count = self.wait_lists[i].len();
                        self.wake_var_incremental(i, &mut heap, ctx);
                        woken += count;
                    }
                    // Re-insert only if the wait list is still non-empty
                    // (a new suspension may have been added during processing).
                    if i < self.wait_lists.len() && !self.wait_lists[i].is_empty() {
                        self.suspended_vars.insert(i);
                    }
                }
            }
            // ── Retry deferred s_inst_copy ───────────────────────
            // s_inst_copy returns 0 when resolve_ty still contains
            // nested InferVars.  Now that the outer iteration may
            // have resolved some of those, retry the deferred copies.
            {
                let pending: Vec<usize> = self.pending_inst_copy.drain().collect();
                for pg_id in pending {
                    if pg_id >= self.var_type_ids.len() {
                        continue;
                    }
                    if self.gen_statuses.get(pg_id) != Some(&GenStatus::Generalized) {
                        continue;
                    }
                    let resolved_ty = self.resolve_promotion_aware(self.var_type_ids[pg_id], ctx);
                    if !matches!(ctx.get(resolved_ty), TypeData::InferVar { .. })
                        && !Self::ty_mentions_infer_var(ctx, resolved_ty)
                    {
                        self.s_inst_copy(ctx, pg_id, resolved_ty);
                    }
                    // If still not closed, s_inst_copy re-inserts
                    // into pending_inst_copy for the next iteration.
                }
            }
            if woken == 0 {
                // If MATCH-CTX pushed continuation constraints onto the heap,
                // re-enter the main loop instead of the expensive exhaustion pass.
                if !heap.is_empty() {
                    continue;
                }
                // Solver exhaustion — check for remaining undischarged Match
                // constraints and fire their else_continuation as a fallback.
                let remaining: Vec<PrioritizedConstraint> = heap.drain();
                let mut non_match: Vec<PrioritizedConstraint> = Vec::new();
                let mut match_elses: Vec<(
                    TypeId,
                    (usize, usize),
                    crate::ast::Span,
                    Arc<SkolemEnv>,
                )> = Vec::new();
                for pc in remaining {
                    if let Constraint::Match {
                        scrutinee,
                        branches_id,
                        span,
                    } = &pc.constraint
                    {
                        let resolved = self.resolve_promotion_aware(*scrutinee, ctx);
                        if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                            match_elses.push((
                                *scrutinee,
                                *branches_id,
                                *span,
                                Arc::clone(&pc.env),
                            ));
                        } else {
                            // Unresolved Match — preserve for later wait_list scan.
                            non_match.push(pc);
                        }
                    } else {
                        // Non-Match constraint — must not be silently dropped.
                        non_match.push(pc);
                    }
                }
                // Re-push preserved non-Match constraints.
                for pc in non_match {
                    heap.push(pc);
                }
                let mut else_heap = BucketQueue::new();
                let mut match_errors = Vec::new();
                for (scrutinee, branches_id, match_span, env) in match_elses {
                    if !self.discharge_match(ctx, scrutinee, branches_id, &mut else_heap, &env) {
                        match_errors.push(TypeError::PatternNotExhaustive { span: match_span });
                    } else {
                        // Record the discharge so later passes (wait-list scan,
                        // MATCH-CTX) never re-process the same match.
                        self.discharged_matches.insert((scrutinee, branches_id));
                    }
                }
                if !match_errors.is_empty() {
                    return Err(match_errors.into_iter().next().unwrap());
                }
                // Merge else continuations back into the main heap so they
                // are processed by the solver loop (not silently dropped).
                if !else_heap.is_empty() {
                    heap.extend(else_heap.drain());
                    continue;
                }
                // If non-Match constraints were preserved, re-enter the loop.
                if !heap.is_empty() {
                    continue;
                }

                // ── Scan wait-lists for never-discharged Match constraints ──
                // A Match whose scrutinee was never resolved stayed suspended
                // throughout the entire solve.  Report it instead of silently
                // dropping it.
                let pending_matches: Vec<(usize, (TypeId, (usize, usize), Span, Arc<SkolemEnv>))> =
                    self.wait_lists
                        .iter_mut()
                        .enumerate()
                        .filter_map(|(vid, wl)| {
                            if wl.is_empty() {
                                return None;
                            }
                            // Drain and inspect; rebuild with non-Match constraints.
                            let mut still: Vec<(Constraint, Arc<SkolemEnv>)> = Vec::new();
                            let mut matches_found: Vec<(
                                TypeId,
                                (usize, usize),
                                Span,
                                Arc<SkolemEnv>,
                            )> = Vec::new();
                            for (c, env) in wl.drain(..) {
                                match c {
                                    Constraint::Match {
                                        scrutinee,
                                        branches_id,
                                        span: m_span,
                                    } => matches_found.push((scrutinee, branches_id, m_span, env)),
                                    other => still.push((other, env)),
                                }
                            }
                            *wl = still;
                            if !matches_found.is_empty() {
                                Some(
                                    matches_found
                                        .into_iter()
                                        .map(|m| (vid, m))
                                        .collect::<Vec<_>>(),
                                )
                            } else {
                                None
                            }
                        })
                        .flatten()
                        .collect();
                for (vid, (scrutinee, branches_id, m_span, env)) in pending_matches {
                    let resolved = self.resolve_promotion_aware(scrutinee, ctx);
                    if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                        return Err(TypeError::CannotInfer { span: m_span });
                    }
                    if !self.discharge_match(ctx, scrutinee, branches_id, &mut heap, &env) {
                        return Err(TypeError::PatternNotExhaustive { span: m_span });
                    }
                    self.discharged_matches.insert((scrutinee, branches_id));
                    // Remove exactly one guard per extracted Match constraint
                    // (each suspend_on_var_with_env added one guard).
                    // Do NOT clear_direct_guards — non-Match constraints still
                    // in the wait_list may have their own guards.
                    if vid < self.guard_sets.len() {
                        self.guard_sets[vid].remove_guard();
                    }
                    // Only remove from suspended_vars when wait_list is empty
                    // AND all guards are cleared.
                    let wl_empty = vid < self.wait_lists.len() && self.wait_lists[vid].is_empty();
                    let guards_empty =
                        vid < self.guard_sets.len() && self.guard_sets[vid].is_empty();
                    if wl_empty && guards_empty {
                        self.suspended_vars.remove(&vid);
                    }
                }
                // If any discharged continuations were enqueued, re-enter the
                // loop to process them instead of breaking.
                if !heap.is_empty() {
                    continue;
                }
                break; // converged: no more constraints to wake
            }
            // Continue the loop to process woken constraints
        }

        // ── Verify recorded bounds for never-resolved vars (exhaustion pass) ──
        // A `Sub` constraint with a free InferVar side records the opposing
        // bound (lower_bounds / upper_bounds) and suspends; the wake path
        // re-checks it when the suspended variable resolves.  If the variable
        // *never* resolves, no wake fires and a contradiction between its
        // lower and upper bounds would go undetected — defaulting would
        // silently pick one side (e.g. lower=[Bool], upper=[Int]).  Verify
        // every fully-resolved (lb, ub) pair for still-unresolved variables.
        // Unresolved or Error bounds are skipped: they carry no decision, and
        // error-type propagation is handled elsewhere.
        for var_id in 0..self.lower_bounds.len().max(self.upper_bounds.len()) {
            let ty_id = match self.var_type_ids.get(var_id) {
                Some(&ty_id) => ty_id,
                None => continue,
            };
            let resolved = self.resolve_promotion_aware(ty_id, ctx);
            if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                // Resolved variables had their suspended Sub constraints
                // re-checked by the wake path — nothing new to verify here.
                continue;
            }
            let lbs: &[TypeId] = self
                .lower_bounds
                .get(var_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let ubs: &[TypeId] = self
                .upper_bounds
                .get(var_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            if lbs.is_empty() || ubs.is_empty() {
                continue;
            }
            let var_span = self
                .var_origins
                .get(var_id)
                .and_then(|o| match o {
                    VarOrigin::Expression(sp) => *sp,
                    _ => None,
                })
                .unwrap_or(Span::new(FileId::DUMMY, 0, 0));
            for &lb in lbs {
                // Promotion-aware: bounds were recorded through
                // resolve_with_env (promotion-aware); verifying them with
                // promotion-blind `ctx.resolve_binding` would skip a
                // promoted bound as a stale InferVar and miss a real
                // `lb ≤ ub` violation.
                let lb_resolved = self.resolve_promotion_aware(lb, ctx);
                let lb_data = ctx.get(lb_resolved);
                if matches!(lb_data, TypeData::InferVar { .. } | TypeData::Error) {
                    continue;
                }
                for &ub in ubs {
                    let ub_resolved = self.resolve_promotion_aware(ub, ctx);
                    let ub_data = ctx.get(ub_resolved);
                    if matches!(ub_data, TypeData::InferVar { .. } | TypeData::Error) {
                        continue;
                    }
                    if !ctx.subtype(lb_resolved, ub_resolved) {
                        return Err(TypeError::Mismatch {
                            expected: ub_resolved,
                            found: lb_resolved,
                            span: var_span,
                        });
                    }
                }
            }
        }

        // Kind checking: ensure that solved types respect the variable's kind
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let mut resolved = self.resolve_promotion_aware(ty_id, ctx);
            // A refined type (`exists n: T invariant P(n)`) has the CARRIER's
            // kind — `exists n: Int<32> invariant n != 0` IS an integer type
            // (SYNTAX.md: the binder "is erased at runtime"; the invariant is
            // verified at construction points, not by the kind check).
            if let TypeData::Exists { base, .. } = ctx.get(resolved) {
                resolved = *base;
            }
            let data = ctx.get(resolved);
            if let TypeData::InferVar { .. } = data {
                continue; // will be defaulted below
            }
            if matches!(data, TypeData::Error) {
                continue;
            }
            let kind = self.type_vars[i].kind;
            // Extract the best available span for this variable: prefer the
            // VarOrigin::Expression span, fall back to Span::new(FileId::DUMMY, 0, 0).
            let var_span = self
                .var_origins
                .get(i)
                .and_then(|o| match o {
                    VarOrigin::Expression(sp) => *sp,
                    _ => None,
                })
                .unwrap_or(Span::new(FileId::DUMMY, 0, 0));
            match kind {
                TypeVariableKind::Integer => {
                    if !matches!(
                        data,
                        TypeData::Int { .. }
                            | TypeData::UInt { .. }
                            | TypeData::USize
                            | TypeData::Rational { .. }
                    ) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: var_span,
                        });
                    }
                }
                TypeVariableKind::Float => {
                    if !matches!(data, TypeData::Float { .. }) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: var_span,
                        });
                    }
                }
                TypeVariableKind::Bool => {
                    if !matches!(data, TypeData::Bool) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: var_span,
                        });
                    }
                }
                TypeVariableKind::Numeric => {
                    if !matches!(
                        data,
                        TypeData::Int { .. }
                            | TypeData::UInt { .. }
                            | TypeData::Float { .. }
                            | TypeData::Rational { .. }
                            | TypeData::USize
                    ) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: var_span,
                        });
                    }
                }
                _ => {}
            }
        }

        // Defaulting: delegate to the standalone defaulting module.
        // This function is shared between the old solver (InferenceContext::solve)
        // and the new solver (FulfillmentContext::evaluate_all).
        //
        // Build the type_vars slice: (kind, origin) pairs for each variable.
        let type_vars: Vec<(TypeVariableKind, VarOrigin)> = self
            .type_vars
            .iter()
            .enumerate()
            .map(|(i, tv)| {
                let origin = self
                    .var_origins
                    .get(i)
                    .copied()
                    .unwrap_or(VarOrigin::Synthetic);
                (tv.kind, origin)
            })
            .collect();
        defaulting::default_variables(ctx, &self.var_type_ids, &type_vars, &self.gen_statuses)?;

        Ok(())
    }

    /// Returns the variable type IDs for the defaulting logic.
    pub fn var_type_ids(&self) -> &[TypeId] {
        &self.var_type_ids
    }

    /// Returns the type variables (kind, shape, etc.) for the defaulting logic.
    pub fn type_vars(&self) -> &[TypeVar] {
        &self.type_vars
    }

    /// Returns the generalisation statuses for the defaulting logic.
    pub fn gen_statuses(&self) -> &[GenStatus] {
        &self.gen_statuses
    }

    /// Returns the variable origins for the defaulting logic.
    pub fn var_origins(&self) -> &[VarOrigin] {
        &self.var_origins
    }

    pub fn finalize<'input>(&self, ctx: &mut TypeContext<'input>) -> HashMap<usize, TypeId> {
        let mut solution = HashMap::default();
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = self.resolve_promotion_aware(ty_id, ctx);
            let data = ctx.get(resolved);
            match data {
                TypeData::InferVar { id, .. } => {
                    // Variable is still unbound — try to infer from bounds
                    let var_id = *id;

                    // PG variables must not be silently defaulted.
                    // A PG variable that is still unresolved means a suspended
                    // constraint was never discharged — defaulting it to Int
                    // or Error silently discards the pending constraint.
                    if var_id < self.gen_statuses.len()
                        && self.gen_statuses[var_id] == GenStatus::PartiallyGeneralizable
                    {
                        solution.insert(var_id, ctx.error());
                        if i != var_id {
                            // Mirror the pre-promotion alias: `replace_infer`
                            // consults only `solution` (never `promotions`),
                            // so residual `InferVar(old_id)` leaves would
                            // stay unsubstituted without this entry.
                            solution.insert(self.type_vars[i].id, ctx.error());
                        }
                        continue;
                    }

                    let lbs: &[TypeId] = if var_id < self.lower_bounds.len() {
                        &self.lower_bounds[var_id]
                    } else {
                        &[]
                    };
                    let ubs: &[TypeId] = if var_id < self.upper_bounds.len() {
                        &self.upper_bounds[var_id]
                    } else {
                        &[]
                    };
                    let chosen = if !lbs.is_empty() && !ubs.is_empty() {
                        // Both bounds present: find a lower bound that is a
                        // subtype of all upper bounds.
                        let resolved_lbs: Vec<TypeId> = lbs
                            .iter()
                            .map(|t| ctx.resolve_binding(*t))
                            .filter(|t| {
                                !matches!(ctx.get(*t), TypeData::InferVar { .. } | TypeData::Error)
                            })
                            .collect();
                        let resolved_ubs: Vec<TypeId> = ubs
                            .iter()
                            .map(|t| ctx.resolve_binding(*t))
                            .filter(|t| {
                                !matches!(ctx.get(*t), TypeData::InferVar { .. } | TypeData::Error)
                            })
                            .collect();
                        resolved_lbs
                            .iter()
                            .copied()
                            .find(|lb| resolved_ubs.iter().all(|ub| ctx.subtype(*lb, *ub)))
                            .unwrap_or_else(|| ctx.error())
                    } else if !lbs.is_empty() {
                        // Only lower bounds: pick the first resolved lower bound.
                        let first_resolved = lbs.iter().find(|t| {
                            let r = ctx.resolve_binding(**t);
                            !matches!(ctx.get(r), TypeData::InferVar { .. } | TypeData::Error)
                        });
                        first_resolved.copied().unwrap_or(ctx.error())
                    } else if !ubs.is_empty() {
                        // Only upper bounds: pick the first resolved upper bound.
                        let first_resolved = ubs.iter().find(|t| {
                            let r = ctx.resolve_binding(**t);
                            !matches!(ctx.get(r), TypeData::InferVar { .. } | TypeData::Error)
                        });
                        first_resolved.copied().unwrap_or(ctx.error())
                    } else {
                        // No bounds — default based on kind
                        match self.type_vars[i].kind {
                            TypeVariableKind::Integer => ctx.int(DEFAULT_INT_WIDTH, true),
                            TypeVariableKind::Float => ctx.float(DEFAULT_FLOAT_WIDTH),
                            TypeVariableKind::Bool => ctx.bool(),
                            TypeVariableKind::Numeric => ctx.int(DEFAULT_INT_WIDTH, true),
                            _ => ctx.error(),
                        }
                    };
                    solution.insert(var_id, chosen);
                    if i != var_id {
                        // Mirror the pre-promotion alias so `replace_infer`
                        // can resolve residual `InferVar(old_id)` leaves
                        // (`promotions` is solver-local and invisible to
                        // `replace_infer`).
                        solution.insert(self.type_vars[i].id, chosen);
                    }
                }
                _ => {
                    solution.insert(self.type_vars[i].id, resolved);
                }
            }
        }
        solution
    }

    /// Generalize all regions: process dirty region roots, generalizing
    /// type variables that are no longer guarded.  This is the entry point
    /// for OmniML §6 force_root_generalization.
    ///
    /// After this call, generalized variables will have `Status::Generic`
    /// and are removed from their region's pool.  Unguarded variables
    /// that remain in the pool are left as `Instance` for the next pass.
    ///
    /// Returns a list of (region_id, var_id) pairs for variables that
    /// were successfully generalized.
    /// Drain the §6 dirty frontier and generalize each dirty region's pool.
    ///
    /// This is the production generalization path (OmniML §6): regions whose
    /// pools changed since their last drain are visited deepest-first, and
    /// `generalize_region` runs its per-variable eligibility check on each.
    /// Called from the ExitRegion and LetBarrier handlers at region-exit
    /// time, so outer constraints observe schemes as soon as a region
    /// closes — not only at the end of solving.
    ///
    /// Guard predicate equivalence with the retired sweep is maintained
    /// inside `generalize_region` (resolved + no direct guards + no waiting
    /// constraints + no rigid escape + no binding-root sharing with a
    /// guarded PG var); the region-keyed transitive-guard model
    /// (`add_transitive_guard`) is NOT used — nothing in production fills
    /// it, and binding-root sharing is the functional equivalent.
    pub fn drain_dirty_generalization<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
    ) -> Vec<(usize, usize)> {
        // O(1) gate: nothing dirty → nothing to do.
        if self.region_tree.dirty_roots.is_empty()
            && !self.region_tree.nodes[self.region_tree.root.0].dirty
        {
            return Vec::new();
        }
        let mut generalized = Vec::new();
        // Worklist loop: PG lowering inside generalize_region registers the
        // lowered variable in the parent pool, whose register_var_in_region
        // marks the parent dirty — possibly a region that was NOT dirty when
        // this drain started.  A one-shot snapshot would strand the lowered
        // PG var until the NEXT drain (OmniML §6: it must be revisited at
        // the parent's exit within the same drain).  Re-collect each round.
        loop {
            let mut dirty_ids: Vec<InferRegionId> = self
                .region_tree
                .nodes
                .iter()
                .filter(|n| n.dirty)
                .map(|n| n.id)
                .collect();
            if dirty_ids.is_empty() {
                break;
            }
            // Deepest-first (§6 topological order: child regions before
            // parents, so lowered PG vars are revisited at the parent level
            // in the same drain).
            dirty_ids.sort_by(|a, b| {
                let la = self.region_tree.nodes[a.0].level;
                let lb = self.region_tree.nodes[b.0].level;
                lb.cmp(&la)
            });
            for region_id in dirty_ids {
                // Skip regions already cleared by an earlier iteration of
                // this same drain (a child's PG lowering can dirty the
                // parent, but never re-dirties a cleared region — the guard
                // is cheap insurance).
                if !self.region_tree.nodes[region_id.0].dirty {
                    continue;
                }
                self.generalize_region(region_id, ctx, &mut generalized);
                // Clear the frontier entry — same cleanup contract as
                // `drain_dirty`'s phase 2. Without this the flags accumulate
                // forever and every subsequent drain rescans the whole
                // history (O(n²) over region exits).
                self.region_tree.nodes[region_id.0].dirty = false;
                if let Some(parent_id) = self.region_tree.nodes[region_id.0].parent {
                    self.region_tree.nodes[parent_id.0]
                        .dirty_children
                        .retain(|&c| c != region_id);
                }
                self.region_tree.dirty_roots.remove(&region_id);
            }
        }
        generalized
    }

    pub fn force_root_generalization<'input>(
        &mut self,
        ctx: &mut TypeContext<'input>,
    ) -> Vec<(usize, usize)> {
        let mut generalized = Vec::new();
        // Collect all region IDs that have alive pools or are dirty.
        let mut region_ids: Vec<InferRegionId> = {
            let mut ids = Vec::new();
            for (i, node) in self.region_tree.nodes.iter().enumerate() {
                if node.pool.is_alive() || node.dirty {
                    ids.push(InferRegionId(i));
                }
            }
            ids
        };
        // Process regions from leaves to root (deepest level first).
        // This ensures child regions are generalised before parent regions,
        // matching the OmniML §6 topological ordering requirement.
        region_ids.sort_by(|a, b| {
            let la = self.region_tree.nodes[a.0].level;
            let lb = self.region_tree.nodes[b.0].level;
            lb.cmp(&la) // descending: deepest first
        });
        for region_id in &region_ids {
            self.generalize_region(*region_id, ctx, &mut generalized);
        }
        generalized
    }

    /// Generalize a single region: iterate over its pool's type variables,
    /// check if each is guarded, and if not, mark as Generic and remove
    /// from the pool.
    fn generalize_region<'input>(
        &mut self,
        region_id: InferRegionId,
        ctx: &mut TypeContext<'input>,
        out: &mut Vec<(usize, usize)>,
    ) {
        let var_ids: Vec<usize> = self.region_tree.nodes[region_id.0].pool.var_ids.clone();
        // Variables generalized by THIS call only.  The final pool-removal
        // loop must not iterate the caller-accumulated `out`: it mixes
        // entries from other regions/other calls, making the removal
        // O(total × pool_size) and — if the pool invariant were ever
        // violated — able to remove a foreign variable from this region.
        let mut local_generalized: Vec<usize> = Vec::new();
        for &var_id in &var_ids {
            let ty_id = if var_id < self.var_type_ids.len() {
                self.var_type_ids[var_id]
            } else {
                continue;
            };
            // Check if the variable is guarded (has guards).
            let is_guarded = if var_id < self.guard_sets.len() {
                let gs = &self.guard_sets[var_id];
                !gs.is_empty()
            } else {
                false
            };
            // Check if the variable is PG (PartiallyGeneralizable).
            let is_pg = if var_id < self.gen_statuses.len() {
                self.gen_statuses[var_id] == GenStatus::PartiallyGeneralizable
            } else {
                false
            };
            if is_pg {
                // PG variables are guarded by suspended constraints.
                // They cannot be generalized yet.  Instead, "lower" them
                // to the parent region (OmniML §6 generalize_generation):
                // move the variable to the parent region's pool so that
                // it remains alive and will be revisited when the parent
                // region is exited (or when guards are discharged).
                if let Some(parent_id) = self.region_tree.nodes[region_id.0].parent {
                    let parent = parent_id;
                    // Route through the registered pool API so `var_to_region`
                    // and the pool undo log stay consistent — never mutate
                    // `pool.var_ids` directly (a direct retain+push would
                    // leave `var_to_region` stale and the PG move
                    // un-rollbackable).
                    self.region_tree.unregister_var(var_id, region_id);
                    self.region_tree.register_var_in_region(var_id, parent);
                    self.push_undo(InferUndoLog::SetVarRegion(
                        var_id,
                        self.type_vars[var_id].region_id,
                    ));
                    self.type_vars[var_id].region_id = parent;
                }
                // If there is no parent (root region), the variable stays
                // in the current pool — it will be processed again on the
                // next generalization pass.
                continue;
            }
            if !is_guarded {
                // Full eligibility check — the SAME predicate the sweep
                // applied, so the frontier path is sound-equivalent to the
                // retired approximation: resolved, nothing waiting, no
                // rigid escape from this region's level, and not
                // binding-root-shared with a guarded PG variable.
                let resolved_ty = self.resolve_promotion_aware(self.var_type_ids[var_id], ctx);
                if matches!(ctx.get(resolved_ty), TypeData::InferVar { .. }) {
                    // Unresolved: cannot enter a scheme.  Leave it in the
                    // pool — conservative (fail-closed), the end-of-solve
                    // pass or check_unresolved handles it.
                    continue;
                }
                if var_id < self.wait_lists.len() && !self.wait_lists[var_id].is_empty() {
                    // Suspended constraints still block this variable.
                    continue;
                }
                let level = self.region_tree.nodes[region_id.0].level;
                if Self::check_rigid_escape(
                    ctx,
                    resolved_ty,
                    level,
                    &self.region_tree,
                    &self.var_promotions,
                ) {
                    // A skolem escaping to an outer scope must not be
                    // copied into the generalized scheme.
                    continue;
                }
                // Binding-root sharing (the sweep's `guarded_roots`): a
                // PG variable with guards resolving to the same root
                // blocks our generalization — unifying them later would
                // otherwise observe a scheme where a monotype was
                // expected.
                let shared_with_guarded = self.pg_vars.iter().any(|&j| {
                    j != var_id
                        && j < self.guard_sets.len()
                        && !self.guard_sets[j].is_empty()
                        && j < self.var_type_ids.len()
                        && self.resolve_promotion_aware(self.var_type_ids[j], ctx) == resolved_ty
                });
                if shared_with_guarded {
                    continue;
                }
                // Update the gen_status if it exists.
                if var_id < self.gen_statuses.len() {
                    self.write_gen_status(var_id, GenStatus::Generalized);
                    // Propagate the resolved type to all instances so they
                    // see the now-generic binding (OmniML §5.2 S-Inst-Copy).
                    if !Self::ty_mentions_infer_var(ctx, resolved_ty) {
                        // S-Inst-Copy: the resolved type is fully closed (no InferVars
                        // anywhere in the tree), so alias-sharing via unify_and_track
                        // is safe — no cross-instance contamination possible.
                        self.s_inst_copy(ctx, var_id, resolved_ty);
                    }
                }
                // When a variable is generalized, update its instances.
                // BUT: only mark an instance as Generalized if it is NOT
                // itself still guarded (OmniML §6: a variable can become
                // Generic only when all its constraints are resolved).
                if var_id < self.forward_refs.len() {
                    // Copy the instance list so the (mutable) push_undo
                    // below does not conflict with the iteration borrow.
                    let instance_ids: Vec<usize> = self.forward_refs[var_id].clone();
                    for inst_id in instance_ids {
                        // Check if the instance is still guarded.
                        let inst_guarded = if inst_id < self.guard_sets.len() {
                            !self.guard_sets[inst_id].is_empty()
                        } else {
                            false
                        };
                        let inst_pg = if inst_id < self.gen_statuses.len() {
                            self.gen_statuses[inst_id] == GenStatus::PartiallyGeneralizable
                        } else {
                            false
                        };
                        // Only promote if the instance is not guarded and not PG.
                        if !inst_guarded && !inst_pg {
                            self.write_gen_status(inst_id, GenStatus::Generalized);
                            // Remove the instance from its region's pool.
                            // Generalized variables must not belong to any pool
                            // (OmniML §6: "once a term is generalised, it is
                            // removed from its pool").
                            if inst_id < self.type_vars.len() {
                                let inst_region = self.type_vars[inst_id].region_id;
                                if inst_region.0 < self.region_tree.nodes.len()
                                    && self.region_tree.nodes[inst_region.0]
                                        .pool
                                        .var_ids
                                        .contains(&inst_id)
                                {
                                    self.region_tree.unregister_var(inst_id, inst_region);
                                }
                            }
                        }
                    }
                }
                local_generalized.push(var_id);
                out.push((region_id.0, var_id));
            }
        }
        // Remove generalized variables from the pool — through the
        // registered API so `var_to_region` and the undo log stay
        // consistent.  Only THIS call's generalizations are considered.
        for vid in local_generalized.iter() {
            if self.region_tree.nodes[region_id.0]
                .pool
                .var_ids
                .contains(vid)
            {
                self.region_tree.unregister_var(*vid, region_id);
            }
        }
        // Mark the region as processed.
        self.region_tree.nodes[region_id.0].dirty = false;
    }

    pub fn apply_solution<'input>(
        ty: TypeId,
        solution: &HashMap<usize, TypeId>,
        ctx: &mut TypeContext<'input>,
    ) -> TypeId {
        replace_infer(ty, solution, ctx)
    }

    /// Check for inference variables that remain unresolved and were
    /// defaulted to `error` (unconstrained/any kind). Returns a list of
    /// diagnostic messages describing the ambiguous variables.
    pub fn check_unresolved<'input>(&self, ctx: &TypeContext<'input>) -> Vec<String> {
        let mut results = Vec::new();
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = self.resolve_promotion_aware(ty_id, ctx);
            if matches!(ctx.get(resolved), TypeData::InferVar { .. })
                && i < self.type_vars.len()
                && self.type_vars[i].kind == TypeVariableKind::Any
            {
                // Only report `Any` — `Unconstrained` is defaulted to error
                // by the solver as a normal fallback, not an ambiguity.
                results.push(format!("unresolved type variable #{} (Any)", i));
            }
        }
        results
    }
}

impl Default for InferenceContext {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn replace_infer<'input>(
    ty: TypeId,
    solution: &HashMap<usize, TypeId>,
    ctx: &mut TypeContext<'input>,
) -> TypeId {
    let resolved = ctx.resolve_binding(ty);
    let data = ctx.get(resolved).clone();
    match data {
        TypeData::InferVar { id, .. } => solution.get(&id).copied().unwrap_or(ty),
        TypeData::SkolemVar { .. } => ty,
        TypeData::Int { .. }
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
        // `Poly` is a first-class polymorphic type `[σ]` (a boxed polytope).
        // In the OmniML calculus (§3.1, O'Brien, Rémy & Scherer) a polytope
        // wraps a *closed* type scheme `σ = ∀ᾱ.τ`.  The body is sealed — it
        // must not contain free inference variables, only `GenericParam`
        // references bound by the quantifiers.
        //
        // The reference implementation (omniml/lib/constraint_solver/types.ml)
        // defines it as:
        //   type t = ... | Poly of Type.Scheme.t
        //   module Scheme = struct
        //     type t = { quantifiers : Type.Var.t list; body : Type.t }
        //   end
        // and enforces the closedness invariant in `principal_shape.ml`:
        //   | Poly _ ->
        //     (* Invariant: no occurrences of [Poly]. *)
        //     assert false
        //
        // Contrast with `subst()` which *does* recurse into `Poly`'s body:
        // `subst` replaces `GenericParam` indices, which CAN appear in the
        // body as bound variables (with binder shadowing).  `replace_infer`
        // replaces `InferVar` IDs, which should never appear in a closed
        // polytope body — so treating `Poly` as a leaf is correct.
        //
        // # Defense-in-depth
        // The reference implementation enforces the closedness invariant
        // explicitly via `Poly.invariant` (principal_shape.ml).  Ponent
        // does NOT have a runtime check for this invariant.  If a bug
        // elsewhere in the compiler creates a `Poly` whose body contains
        // an `InferVar`, `replace_infer` will silently fail to replace it,
        // leaving a stale inference variable in the type.  Adding an
        // invariant assertion (e.g. in `TypeContext::poly()`) would catch
        // such violations at the construction site rather than silently
        // propagating them.
        | TypeData::Poly { .. }
        | TypeData::Regex { .. }
        | TypeData::Type => ty,
        TypeData::GenericParam { .. } => ty,
        TypeData::Adt { kind, def_id, args } => {
            let new_args: SmallVec<[TypeId; 4]> = args
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.alloc(TypeData::Adt {
                kind,
                def_id,
                args: new_args,
            })
        }
        TypeData::Tuple { elems } => {
            let new_elems: SmallVec<[TypeId; 4]> = elems
                .iter()
                .map(|&e| replace_infer(e, solution, ctx))
                .collect();
            ctx.alloc(TypeData::Tuple { elems: new_elems })
        }
        TypeData::Array { elem, size } => {
            let new_elem = replace_infer(elem, solution, ctx);
            ctx.alloc(TypeData::Array {
                elem: new_elem,
                size,
            })
        }
        TypeData::Slice { elem } => {
            let new_elem = replace_infer(elem, solution, ctx);
            ctx.alloc(TypeData::Slice { elem: new_elem })
        }
        TypeData::Ref { ty, mutable, .. } => {
            let new_ty = replace_infer(ty, solution, ctx);
            ctx.alloc(TypeData::Ref {
                ty: new_ty,
                mutable,
                lifetime: None,
            })
        }
        TypeData::Pointer { ty } => {
            let new_ty = replace_infer(ty, solution, ctx);
            ctx.alloc(TypeData::Pointer { ty: new_ty })
        }
        TypeData::Ptr { size, pointee } => {
            let new_size = replace_infer(size, solution, ctx);
            let new_pointee = replace_infer(pointee, solution, ctx);
            ctx.alloc(TypeData::Ptr {
                size: new_size,
                pointee: new_pointee,
            })
        }
        TypeData::Fn { params, ret } => {
            let new_params: SmallVec<[TypeId; 4]> = params
                .iter()
                .map(|&p| replace_infer(p, solution, ctx))
                .collect();
            let new_ret = replace_infer(ret, solution, ctx);
            ctx.alloc(TypeData::Fn {
                params: new_params,
                ret: new_ret,
            })
        }
        TypeData::DynTrait { .. } => ty,
        TypeData::Exists {
            param_index,
            name,
            base,
        } => {
            let new_base = replace_infer(base, solution, ctx);
            ctx.alloc(TypeData::Exists {
                param_index,
                name,
                base: new_base,
            })
        }
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => {
            let new_self = replace_infer(self_ty, solution, ctx);
            ctx.alloc(TypeData::AssociatedType {
                trait_id,
                name,
                self_ty: new_self,
            })
        }
        TypeData::Forall {
            param_index,
            param_name,
            body,
        } => {
            let new_body = replace_infer(body, solution, ctx);
            ctx.alloc(TypeData::Forall {
                param_index,
                param_name,
                body: new_body,
            })
        }
        TypeData::Mu {
            param_index,
            param_name,
            body,
        } => {
            let new_body = replace_infer(body, solution, ctx);
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
            let new_body = replace_infer(body, solution, ctx);
            ctx.alloc(TypeData::Nu {
                param_index,
                param_name,
                body: new_body,
            })
        }
        TypeData::Opaque { def_id, hidden } => match hidden {
            Some(hidden_ty) => {
                let new_hidden = replace_infer(hidden_ty, solution, ctx);
                ctx.alloc(TypeData::Opaque {
                    def_id,
                    hidden: Some(new_hidden),
                })
            }
            None => ty,
        },
        TypeData::Coproduct { alternatives } => {
            let new_alts: SmallVec<[TypeId; 4]> = alternatives
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.alloc(TypeData::Coproduct {
                alternatives: new_alts,
            })
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    fn new_ctx() -> TypeContext<'static> {
        TypeContext::new()
    }

    #[test]
    fn test_shape_of_fn() {
        let mut ctx = new_ctx();
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(smallvec![int_ty], bool_ty);
        let shape = InferenceContext::shape_of_type(&ctx, fn_ty);
        assert!(matches!(shape, PrincipalShape::Arrow));
    }

    #[test]
    fn test_shape_of_tuple() {
        let mut ctx = new_ctx();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        let tup = ctx.tuple(smallvec![bool_ty, int_ty]);
        let shape = InferenceContext::shape_of_type(&ctx, tup);
        assert!(matches!(shape, PrincipalShape::Tuple(2)));
    }

    #[test]
    fn test_shape_of_forall() {
        let mut ctx = new_ctx();
        let p0 = ctx.generic_param(0, "X".into());
        let fn_ty = ctx.function(smallvec![p0], p0);
        let forall = ctx.forall(0, "X".into(), fn_ty);
        let shape = InferenceContext::shape_of_type(&ctx, forall);
        assert!(matches!(shape, PrincipalShape::Poly));
    }

    #[test]
    fn test_suspend_and_wake_var() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(
            &mut ctx,
            TypeVariableKind::Unconstrained,
            VarOrigin::Synthetic,
        );
        // Extract the var ID from the InferVar type
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.suspend_on_var(
            Constraint::Eq(
                ctx.bool(),
                ctx.bool(),
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            ),
            var_id,
        );
        // Waking moves suspended constraints back to the active list
        let woken = infer.wake_var_for_test(var_id, &ctx);
        // The heap should now have the woken constraint
        assert!(woken > 0);
    }

    #[test]
    fn test_wake_var_incremental() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(
            &mut ctx,
            TypeVariableKind::Unconstrained,
            VarOrigin::Synthetic,
        );
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.suspend_on_var(
            Constraint::Eq(
                ctx.bool(),
                ctx.bool(),
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            ),
            var_id,
        );
        // wake_var_incremental needs a heap, var_id, and ctx
        let mut heap = BucketQueue::new();
        infer.wake_var_incremental(var_id, &mut heap, &ctx);
        // After waking, the wait list should be empty
        assert!(infer.wait_lists[var_id].is_empty());
    }

    #[test]
    fn test_level_enter_exit() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let prev = infer.enter_level();
        assert!(infer.region_tree.get_level(infer.region_tree.current) > 0);
        infer.exit_level(prev);
        assert_eq!(infer.region_tree.get_level(infer.region_tree.current), 0);
    }

    #[test]
    fn test_level_new_type_var() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let prev = infer.enter_level();
        let var = infer.new_type_var(
            &mut ctx,
            TypeVariableKind::Unconstrained,
            VarOrigin::Synthetic,
        );
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        assert!(infer.get_var_level(var_id).unwrap_or(0) > 0);
        infer.exit_level(prev);
    }

    #[test]
    fn test_eq_concrete_priority() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        let eq = Constraint::Eq(
            bool_ty,
            int_ty,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            EqOrigin::Normal,
        );
        // Both Eq constraints with concrete types have the same priority.
        // This test was originally comparing Eq vs Impl (which was removed
        // in the unified solver migration).  Now it verifies that Eq
        // constraints with concrete types are equal priority.
        assert_eq!(eq.priority(&ctx), eq.priority(&ctx));
    }

    #[test]
    fn test_match_resolved_priority() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        let eq = Constraint::Eq(
            bool_ty,
            int_ty,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            EqOrigin::Normal,
        );
        let match_c = Constraint::Match {
            scrutinee: bool_ty,
            branches_id: (0, 0),
            span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
        };
        // Eq with concrete types has higher priority than Match with unresolved scrutinee
        // (Match has priority 6 when the scrutinee is an infer var).
        // Since the scrutinee is a concrete bool, Match has priority 3, equal to Eq.
        // This test verifies that the remaining constraint types work correctly
        // after the removal of Constraint::Impl (which was previously the lowest
        // priority at 5).  Eq with concrete types has the same priority as Match
        // with a resolved scrutinee (both are priority 3).
        assert!(eq.priority(&ctx) <= match_c.priority(&ctx));
    }

    #[test]
    fn test_match_constraint_priority() {
        let mut ctx = TypeContext::new();
        let infer = InferenceContext::new();
        let var = ctx.alloc_infer_var(0, 0);
        // Match constraint on an InferVar → low priority (6)
        let match_c = Constraint::Match {
            scrutinee: var,
            branches_id: (0, 0),
            span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
        };
        assert_eq!(
            match_c.priority(&ctx),
            6,
            "Match on InferVar should have lowest priority"
        );
        // Eq on concrete types → high priority (0)
        let eq_c = Constraint::Eq(
            ctx.bool(),
            ctx.int(DEFAULT_INT_WIDTH, true),
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            EqOrigin::Normal,
        );
        assert_eq!(
            eq_c.priority(&ctx),
            0,
            "Eq on concrete should have highest priority"
        );
    }

    #[test]
    fn test_unicity_check_non_var() {
        // UNI-TYPE: non-variable type has unique shape
        let mut ctx = TypeContext::new();
        let infer = InferenceContext::new();
        // Use pre-allocated built-in types
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        let shape = InferenceContext::unicity_check(
            &infer,
            &ctx,
            int_ty,
            &Arc::new(HashMap::default()),
            &[],
        );
        assert!(shape.is_some(), "non-variable type should have known shape");
    }

    #[test]
    fn test_unicity_check_fn_type() {
        let mut ctx = TypeContext::new();
        let infer = InferenceContext::new();
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        let fn_ty = ctx.function(smallvec![int_ty], int_ty);
        let shape = InferenceContext::unicity_check(
            &infer,
            &ctx,
            fn_ty,
            &Arc::new(HashMap::default()),
            &[],
        );
        assert!(shape.is_some(), "function type should have known shape");
        assert_eq!(shape.unwrap(), PrincipalShape::Arrow);
    }

    #[test]
    fn test_register_instance_and_propagate() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let pg_var = infer.new_type_var(
            &mut ctx,
            TypeVariableKind::Unconstrained,
            VarOrigin::Synthetic,
        );
        let pg_id = match ctx.get(pg_var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        let inst1 = infer.new_type_var(
            &mut ctx,
            TypeVariableKind::Unconstrained,
            VarOrigin::Synthetic,
        );
        let inst1_id = match ctx.get(inst1) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        // Register the instance
        infer.register_instance(pg_id, inst1_id);
        assert!(infer.has_instances(pg_id), "PG var should have instances");
        assert_eq!(
            infer.is_instance(inst1_id),
            Some(pg_id),
            "instance should track its PG var"
        );
        // Propagate PG resolution via S-Inst-Copy
        let bool_ty = ctx.bool();
        let updated = infer.s_inst_copy(&mut ctx, pg_id, bool_ty);
        assert_eq!(updated, 1, "should have updated 1 instance");
        // Check that the instance was unified with bool
        assert!(
            !infer.has_instances(pg_id),
            "forward refs should be cleared"
        );
    }

    #[test]
    fn test_register_match_branches() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Arc::from(vec![Constraint::Eq(
                ctx.int(DEFAULT_INT_WIDTH, true),
                ctx.int(DEFAULT_INT_WIDTH, true),
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            )]),
            else_continuation: Arc::from(Vec::new() as Vec<Constraint>),
        }];
        let id = infer.register_match_branches(branches);
        assert!(id.0 < infer.match_branches.len());
    }

    #[test]
    fn test_rollback_register_match_branches_batch() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut mk_branch = |shape: PrincipalShape| MatchBranchSet {
            shape_pattern: shape,
            continuation: Arc::from(vec![Constraint::Eq(
                ctx.int(DEFAULT_INT_WIDTH, true),
                ctx.int(DEFAULT_INT_WIDTH, true),
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            )]),
            else_continuation: Arc::from(Vec::new() as Vec<Constraint>),
        };
        let pre_len = infer.match_branches_len_for_test();
        let snap = infer.start_snapshot();
        let (start, count) = infer.register_match_branches(vec![
            mk_branch(PrincipalShape::Arrow),
            mk_branch(PrincipalShape::Tuple(2)),
        ]);
        assert_eq!(count, 2);
        assert_eq!(start, pre_len);
        assert_eq!(
            infer.match_branches_len_for_test(),
            pre_len + 2,
            "both branches registered inside the snapshot"
        );
        infer.rollback_to(snap);
        assert_eq!(
            infer.match_branches_len_for_test(),
            pre_len,
            "rollback must remove ALL branches of the batch, not count−1"
        );
    }

    #[test]
    fn test_match_priority_change_on_resolve() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(
            &mut ctx,
            TypeVariableKind::Unconstrained,
            VarOrigin::Synthetic,
        );
        let match_c = Constraint::Match {
            scrutinee: var,
            branches_id: (0, 0),
            span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
        };
        // Before resolution: low priority
        assert_eq!(match_c.priority(&ctx), 6);
        // After resolution: medium priority
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        ctx.bindings.insert(var, int_ty);
        assert_eq!(
            match_c.priority(&ctx),
            3,
            "match on resolved type should have medium priority"
        );
    }

    // ── else_ fallback ───────────────────────────────────────────────

    #[test]
    fn test_else_continuation_on_mismatch() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = BucketQueue::new();
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);

        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Arc::from(Vec::new() as Vec<Constraint>),
            else_continuation: Arc::from(vec![Constraint::Eq(
                int_ty,
                int_ty,
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            )]),
        }];
        let id = infer.register_match_branches(branches);
        let int_ty2 = ctx.int(64, false);

        let result = infer.discharge_match(
            &mut ctx,
            int_ty2,
            id,
            &mut heap,
            &Arc::new(HashMap::default()),
        );
        assert!(result, "else_ fallback should return true");
        assert!(!heap.is_empty(), "else_ continuation should be enqueued");
    }

    #[test]
    fn test_else_continuation_empty_still_fails() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = BucketQueue::new();

        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Arc::from(Vec::new() as Vec<Constraint>),
            else_continuation: Arc::from(Vec::new() as Vec<Constraint>),
        }];
        let id = infer.register_match_branches(branches);
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);

        let result = infer.discharge_match(
            &mut ctx,
            int_ty,
            id,
            &mut heap,
            &Arc::new(HashMap::default()),
        );
        assert!(!result, "no else_ fallback should still fail");
        assert!(heap.is_empty(), "no constraints should be enqueued");
    }

    /// Regression: an empty branch set (`count == 0`) must not consult
    /// `match_branches[start]` when a later-registered set's branch now
    /// lives at that index.  Before the fix the fallback path either
    /// panicked in `debug_assert!` (`[start+1..start]` is an empty range
    /// with start > end) or, in release builds, wrongly enqueued the later
    /// set's `else_continuation`.
    #[test]
    fn test_discharge_match_empty_branch_set_does_not_borrow_later_set() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = BucketQueue::new();
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);

        let empty_id = infer.register_match_branches(Vec::new());
        assert_eq!(empty_id, (0, 0));

        let later = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Arc::from(Vec::new() as Vec<Constraint>),
            else_continuation: Arc::from(vec![Constraint::Eq(
                int_ty,
                int_ty,
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            )]),
        }];
        let _later_id = infer.register_match_branches(later);

        let result = infer.discharge_match(
            &mut ctx,
            int_ty,
            empty_id,
            &mut heap,
            &Arc::new(HashMap::default()),
        );
        assert!(!result, "an empty branch set matches nothing");
        assert!(
            heap.is_empty(),
            "must not borrow a later set's else_continuation"
        );
    }

    // ── force_generalization ─────────────────────────────────────────

    #[test]
    fn test_force_generalize_pg_with_guard() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        let _var = infer.new_type_var(
            &mut ctx,
            TypeVariableKind::Unconstrained,
            VarOrigin::Synthetic,
        );
        let var_id = 0;
        if var_id < infer.gen_statuses.len() {
            infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
            infer.pg_vars.insert(var_id);
        }
        infer.add_guard(var_id);

        infer.force_generalize(&mut ctx);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::PartiallyGeneralizable,
            "guarded PG var should remain PG after force_generalize"
        );
    }

    #[test]
    fn test_force_generalize_pg_no_guard_resolved() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        let var = infer.new_type_var(
            &mut ctx,
            TypeVariableKind::Unconstrained,
            VarOrigin::Synthetic,
        );
        let var_id = 0;
        if var_id < infer.gen_statuses.len() {
            infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
            infer.pg_vars.insert(var_id);
        }

        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        ctx.bindings.insert(var, int_ty);

        infer.force_generalize(&mut ctx);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Generalized,
            "un-guarded resolved PG var should become Generalized"
        );
    }

    // ── [s] pattern: try_match_via_shape_var callback ─────────────────

    #[test]
    fn test_try_match_via_shape_var_registers_waitlist() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = BucketQueue::new();

        let var = infer.new_type_var(
            &mut ctx,
            TypeVariableKind::Unconstrained,
            VarOrigin::Synthetic,
        );
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Arc::from(Vec::new() as Vec<Constraint>),
            else_continuation: Arc::from(Vec::new() as Vec<Constraint>),
        }];
        let id = infer.register_match_branches(branches);

        let handled = infer.try_match_via_shape_var(
            &mut ctx,
            var,
            id,
            &mut heap,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            &Arc::new(HashMap::default()),
        );
        assert!(handled, "should register the match on the wait list");

        let var_id = 0;
        if var_id < infer.wait_lists.len() {
            assert!(
                !infer.wait_lists[var_id].is_empty(),
                "match should be in the wait list"
            );
        }
    }

    #[test]
    fn test_try_match_via_shape_var_concrete_discharges() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = BucketQueue::new();

        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Integer,
            continuation: Arc::from(Vec::new() as Vec<Constraint>),
            else_continuation: Arc::from(Vec::new() as Vec<Constraint>),
        }];
        let id = infer.register_match_branches(branches);

        let handled = infer.try_match_via_shape_var(
            &mut ctx,
            int_ty,
            id,
            &mut heap,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            &Arc::new(HashMap::default()),
        );
        assert!(
            handled,
            "concrete type with matching shape should discharge"
        );
    }

    #[test]
    fn test_let_constraint_priority() {
        let c = Constraint::Let {
            expr_var: "x".into(),
            def_constraint: Box::new(Constraint::Eq(
                TypeId::from_raw(1),
                TypeId::from_raw(2),
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            )),
            body_constraint: Box::new(Constraint::Eq(
                TypeId::from_raw(1),
                TypeId::from_raw(1),
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            )),
            span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
        };
        let ctx = TypeContext::new();
        assert_eq!(c.priority(&ctx), 2, "Let should have medium-high priority");
    }

    // ═══════════════════════════════════════════════════════════════
    // Shape Variable Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_shape_var_new_and_resolve() {
        let mut svc = ShapeVarContext::new();
        let sva = svc.new_var(0);
        assert_eq!(svc.resolve(sva), sva);
        assert_eq!(svc.get(sva), None);
        assert!(!svc.is_resolved(sva));

        assert!(svc.try_set(sva, TypeShape::Arrow));
        assert_eq!(svc.get(sva), Some(TypeShape::Arrow));
        assert!(svc.is_resolved(sva));
    }

    #[test]
    fn test_shape_var_try_set_idempotent() {
        let mut svc = ShapeVarContext::new();
        let sv = svc.new_var(1);
        assert!(svc.try_set(sv, TypeShape::Tuple(2)));
        // Setting the same shape again succeeds
        assert!(svc.try_set(sv, TypeShape::Tuple(2)));
        // Setting a different shape fails (mismatch)
        assert!(!svc.try_set(sv, TypeShape::Arrow));
        assert_eq!(svc.get(sv), Some(TypeShape::Tuple(2)));
    }

    #[test]
    fn test_shape_var_try_set_fires_callback() {
        let mut svc = ShapeVarContext::new();
        let sv = svc.new_var(0);
        let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f = fired.clone();
        svc.on_resolve(sv, move |_| {
            f.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        assert!(
            !fired.load(std::sync::atomic::Ordering::SeqCst),
            "callback should not fire before resolution"
        );

        assert!(svc.try_set(sv, TypeShape::Arrow));
        assert!(
            fired.load(std::sync::atomic::Ordering::SeqCst),
            "callback should fire on resolution"
        );
    }

    #[test]
    fn test_shape_var_on_resolve_immediate() {
        let mut svc = ShapeVarContext::new();
        let sv = svc.new_var(0);
        svc.try_set(sv, TypeShape::Poly);

        let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f = fired.clone();
        svc.on_resolve(sv, move |_| {
            f.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        assert!(
            fired.load(std::sync::atomic::Ordering::SeqCst),
            "on_resolve should fire immediately if already resolved"
        );
    }

    #[test]
    fn test_shape_var_unify_aliasing() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);
        assert_ne!(svc.resolve(a), svc.resolve(b));

        svc.unify(a, b);
        // After unify, both resolve to the same canonical id
        assert_eq!(svc.resolve(a), svc.resolve(b));
    }

    #[test]
    fn test_shape_var_unify_merges_waitlists() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);

        let fired_a = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fired_b = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fa = fired_a.clone();
        let fb = fired_b.clone();
        svc.on_resolve(a, move |_| {
            fa.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        svc.on_resolve(b, move |_| {
            fb.store(true, std::sync::atomic::Ordering::SeqCst);
        });

        svc.unify(a, b);
        // Resolve the unified target — both callbacks should fire
        let target = svc.resolve(a);
        svc.try_set(target, TypeShape::Arrow);
        assert!(
            fired_a.load(std::sync::atomic::Ordering::SeqCst),
            "callback on a should fire"
        );
        assert!(
            fired_b.load(std::sync::atomic::Ordering::SeqCst),
            "callback on b should fire"
        );
    }

    #[test]
    fn test_shape_var_unify_propagates_resolved() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);
        svc.try_set(a, TypeShape::Constructor(0, 1));

        let fired_b = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fb = fired_b.clone();
        svc.on_resolve(b, move |_| {
            fb.store(true, std::sync::atomic::Ordering::SeqCst);
        });

        svc.unify(a, b);
        assert!(
            fired_b.load(std::sync::atomic::Ordering::SeqCst),
            "b's callback should fire when unified with resolved a"
        );
    }

    #[test]
    fn test_shape_var_get_level() {
        let mut svc = ShapeVarContext::new();
        let sv = svc.new_var(3);
        assert_eq!(svc.get_level(sv), 3);
    }

    #[test]
    fn test_shape_var_num_unresolved() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);
        let c = svc.new_var(0);
        assert_eq!(svc.num_unresolved(), 3);
        svc.try_set(a, TypeShape::Arrow);
        assert_eq!(svc.num_unresolved(), 2);
        svc.try_set(b, TypeShape::Tuple(1));
        assert_eq!(svc.num_unresolved(), 1);
        // c is still unresolved
        assert!(!svc.is_resolved(c));
    }

    #[test]
    fn test_shape_var_unresolved_above_level() {
        let mut svc = ShapeVarContext::new();
        let _l0 = svc.new_var(0);
        let _l1 = svc.new_var(1);
        let _l2 = svc.new_var(2);
        let above1 = svc.unresolved_above_level(1);
        assert_eq!(above1.len(), 1, "only level-2 var should be above level 1");
        assert_eq!(svc.get_level(above1[0]), 2);
    }

    #[test]
    fn test_shapes_compatible() {
        assert!(shapes_compatible(TypeShape::Unknown, TypeShape::Arrow));
        assert!(shapes_compatible(TypeShape::Arrow, TypeShape::Unknown));
        assert!(shapes_compatible(TypeShape::Arrow, TypeShape::Arrow));
        assert!(shapes_compatible(TypeShape::Tuple(3), TypeShape::Tuple(3)));
        assert!(!shapes_compatible(TypeShape::Tuple(2), TypeShape::Tuple(3)));
        // Same constructor identity → compatible.
        assert!(shapes_compatible(
            TypeShape::Constructor(0, 2),
            TypeShape::Constructor(0, 2)
        ));
        // Different constructor identity → NOT compatible.
        assert!(!shapes_compatible(
            TypeShape::Constructor(0, 2),
            TypeShape::Constructor(5, 2)
        ));
        assert!(!shapes_compatible(TypeShape::Arrow, TypeShape::Tuple(1)));
        assert!(shapes_compatible(TypeShape::Poly, TypeShape::Poly));
        assert!(!shapes_compatible(TypeShape::Poly, TypeShape::Arrow));
    }

    #[test]
    fn test_type_data_to_shape_variants() {
        let mut ctx = TypeContext::new();
        // Fn → Arrow
        let fn_ty = ctx.function(smallvec![ctx.bool()], ctx.bool());
        assert_eq!(type_data_to_shape(ctx.get(fn_ty)), TypeShape::Arrow);
        // Tuple(n) → Tuple(n)
        let b = ctx.bool();
        let i = ctx.int(DEFAULT_INT_WIDTH, true);
        let tup = ctx.tuple(smallvec![b, i]);
        assert_eq!(type_data_to_shape(ctx.get(tup)), TypeShape::Tuple(2));
        // Struct → Constructor(n)
        let b2 = ctx.bool();
        let i2 = ctx.int(DEFAULT_INT_WIDTH, true);
        let s = ctx.struct_ty(DefId(42), smallvec![b2, i2]);
        assert_eq!(
            type_data_to_shape(ctx.get(s)),
            TypeShape::Constructor(42, 2)
        );
        // Forall → Poly
        let p0 = ctx.generic_param(0, "X".into());
        let forall = ctx.forall(0, "X".into(), p0);
        assert_eq!(type_data_to_shape(ctx.get(forall)), TypeShape::Poly);
        // Int → Unknown (primitive)
        let int32 = ctx.int(DEFAULT_INT_WIDTH, true);
        assert_eq!(type_data_to_shape(ctx.get(int32)), TypeShape::Unknown);
        // Bool → Unknown
        let bool_ty = ctx.bool();
        assert_eq!(type_data_to_shape(ctx.get(bool_ty)), TypeShape::Unknown);
    }

    // ═══════════════════════════════════════════════════════════════
    // Level-based Promotion Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_try_promote_var_basic() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let _prev = infer.enter_level();
        let _prev2 = infer.enter_level();
        // Create a variable at the current deep level
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        let deep_level = infer.get_var_level(var_id).unwrap();
        assert!(deep_level > 0, "var should be at a deep level");

        // Promote to level 0
        let promoted = infer.try_promote_var(&mut ctx, var_id, InferRegionId(0));
        assert!(promoted.is_some(), "promotion should succeed");
        // The old var should now be bound to the promoted var (via infer.resolve)
        let resolved = infer.resolve(var, &ctx);
        assert!(
            matches!(ctx.get(resolved), TypeData::InferVar { id, .. } if *id != var_id),
            "original var should be bound to a new InferVar"
        );
    }

    #[test]
    fn test_try_promote_var_already_low() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        // Variable at level 0 (default)
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        // Try to promote to level 0 — no-op since already at level 0
        let promoted = infer.try_promote_var(&mut ctx, var_id, InferRegionId(0));
        assert!(promoted.is_some(), "should return the existing var");
        let resolved = ctx.resolve_binding(var);
        assert!(
            matches!(ctx.get(resolved), TypeData::InferVar { id, .. } if *id == var_id),
            "should be unchanged"
        );
    }

    #[test]
    fn test_try_promote_var_transfers_wait_list_and_guards() {
        // ── Verify that promotion transfers wait list, guard set, gen_status,
        // lower_bounds, upper_bounds, and forward/reverse refs to the new
        // variable. Without this transfer, constraints suspended on the old
        // variable are silently lost (the critical vulnerability).
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let _prev = infer.enter_level();
        let _prev2 = infer.enter_level();

        // Create a variable at a deep level
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        let deep_level = infer.get_var_level(var_id).unwrap();
        assert!(deep_level > 0, "var should be at a deep level");

        // Suspend constraints on it (simulating match/impl obligations)
        infer.suspend_on_var(
            Constraint::Eq(
                ctx.bool(),
                ctx.int(DEFAULT_INT_WIDTH, true),
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            ),
            var_id,
        );
        infer.suspend_on_var(
            Constraint::Eq(
                ctx.bool(),
                ctx.bool(),
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            ),
            var_id,
        );

        // Verify suspension was recorded
        assert_eq!(
            infer.wait_lists[var_id].len(),
            2,
            "old var should have 2 suspended constraints"
        );
        assert!(
            !infer.guard_sets[var_id].is_empty(),
            "old var should have guards"
        );
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::PartiallyGeneralizable,
            "old var should be PG"
        );

        // Also set up a forward reference (simulating instance tracking)
        let inst = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let inst_id = match ctx.get(inst) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.register_instance(var_id, inst_id);
        assert!(infer.has_instances(var_id), "old var should have instances");

        // Also add a lower bound
        if var_id < infer.lower_bounds.len() {
            infer.lower_bounds[var_id].push(ctx.bool());
        }

        // ── Promote the variable ──────────────────────────────────
        let promoted = infer.try_promote_var(&mut ctx, var_id, InferRegionId(0));
        assert!(promoted.is_some(), "promotion should succeed");

        // The promoted TypeId corresponds to the new variable
        let new_id = match ctx.get(promoted.unwrap()) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        assert_ne!(new_id, var_id, "new var must be different from old");

        // ── Verify transfer: the new variable now owns the state ──
        assert!(
            infer.wait_lists[var_id].is_empty(),
            "old var's wait list should be cleared"
        );
        assert_eq!(
            infer.wait_lists[new_id].len(),
            2,
            "new var should have the 2 suspended constraints"
        );

        assert!(
            infer.guard_sets[var_id].is_empty(),
            "old var's guard set should be cleared"
        );
        assert!(
            !infer.guard_sets[new_id].is_empty(),
            "new var should have the guards"
        );

        assert_eq!(
            infer.gen_statuses[new_id],
            GenStatus::PartiallyGeneralizable,
            "new var should inherit PG status"
        );

        // The old var is now just a resolution alias, status is irrelevant
        // but the new var must have the right status.

        // ── Verify forward refs transfer ──────────────────────────
        assert!(
            infer.has_instances(new_id),
            "new var should inherit the instances from the old var"
        );
        // Actually, forward_refs[new_id] was extended with old_fwd,
        // which contained inst_id. But new_id might not be < forward_refs.len()
        // let's check more carefully
        if new_id < infer.forward_refs.len() {
            assert!(
                infer.forward_refs[new_id].contains(&inst_id),
                "new var's forward refs should include the instance"
            );
        }
        // The instance should now point to new_id
        if inst_id < infer.reverse_refs.len() {
            assert_eq!(
                infer.reverse_refs[inst_id],
                Some(new_id),
                "instance should now point to the new var"
            );
        }

        // ── Verify lower bound transfer ───────────────────────────
        if var_id < infer.lower_bounds.len() && new_id < infer.lower_bounds.len() {
            assert!(
                infer.lower_bounds[var_id].is_empty(),
                "old var's lower bounds should be cleared"
            );
            assert!(
                !infer.lower_bounds[new_id].is_empty(),
                "new var should have the lower bounds"
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Generalization (PG → G) Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_force_generalize_pg_with_waitlist_stays_pg() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        // Set PG status and add a suspended constraint (wait list not empty)
        if var_id < infer.gen_statuses.len() {
            infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
            infer.pg_vars.insert(var_id);
        }
        let int32_1 = ctx.int(DEFAULT_INT_WIDTH, true);
        infer.suspend_on_var(
            Constraint::Eq(var, int32_1, crate::ast::Span::new(FileId::DUMMY, 0, 0), EqOrigin::Normal),
            var_id,
        );

        // Resolve the variable so it has a concrete type
        let int64 = ctx.int(64, false);
        ctx.bindings.insert(var, int64);

        infer.force_generalize(&mut ctx);
        // Should remain PG because the wait list is not empty
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::PartiallyGeneralizable,
            "PG var with non-empty wait list should stay PG"
        );
    }

    #[test]
    fn test_force_generalize_dirty_set_triggers_generalization() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(var_id);
        // Resolve the variable
        ctx.bindings.insert(var, ctx.bool());
        // Mark dirty
        infer.mark_dirty(var_id);

        infer.force_generalize(&mut ctx);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Generalized,
            "resolved PG var in dirty set should become Generalized"
        );
    }

    #[test]
    fn test_force_generalize_for_regions() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let prev = infer.enter_level();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.exit_level(prev);
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(var_id);
        let unit_ty = ctx.unit();
        ctx.bindings.insert(var, unit_ty);

        // Use dirty_levels to trigger generalization
        let level = infer.get_var_level(var_id).unwrap_or(0);
        infer.force_generalize_for_regions(&mut ctx, &[level], None);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Generalized,
            "PG var in dirty level should generalize"
        );
    }

    #[test]
    fn test_rigid_escape_generic_param_not_detected() {
        let mut ctx = TypeContext::new();
        let gp = ctx.generic_param(0, "T".into());
        let escaped = InferenceContext::check_rigid_escape(
            &ctx,
            gp,
            0,
            &InferenceContext::new().region_tree,
            &std::collections::HashMap::default(),
        );
        assert!(!escaped, "Outer GenericParam should NOT escape");
    }

    #[test]
    fn test_rigid_escape_concrete_not_detected() {
        let mut ctx = TypeContext::new();
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        let not_escaped = InferenceContext::check_rigid_escape(
            &ctx,
            int_ty,
            0,
            &InferenceContext::new().region_tree,
            &std::collections::HashMap::default(),
        );
        assert!(!not_escaped, "Int<32> is not an escape");
    }

    #[test]
    fn test_rigid_escape_fn_with_gp_not_detected() {
        let mut ctx = TypeContext::new();
        let gp = ctx.generic_param(1, "U".into());
        let fn_ty = ctx.function(smallvec![gp], gp);
        let escaped = InferenceContext::check_rigid_escape(
            &ctx,
            fn_ty,
            0,
            &InferenceContext::new().region_tree,
            &std::collections::HashMap::default(),
        );
        assert!(
            !escaped,
            "fn(U) -> U with outer GenericParam should not escape"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // S-Inst-Copy Propagation Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_s_inst_copy_follows_aliases() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let pg = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let pg_id = match ctx.get(pg) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        // Create an instance
        let inst = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let inst_id = match ctx.get(inst) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.register_instance(pg_id, inst_id);

        // Resolve PG → fn(Int) → Bool
        let int32_2 = ctx.int(DEFAULT_INT_WIDTH, true);
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(smallvec![int32_2], bool_ty);
        ctx.bindings.insert(pg, fn_ty);

        // S-Inst-Copy propagates the PG resolution to instances
        let resolved_pg = ctx.resolve_binding(pg);
        let updated = infer.s_inst_copy(&mut ctx, pg_id, resolved_pg);
        assert_eq!(updated, 1, "should have updated 1 instance");

        let inst_resolved = ctx.resolve_binding(inst);
        match ctx.get(inst_resolved) {
            TypeData::Fn { params, ret } => {
                assert_eq!(params.len(), 1, "instance should become a fn type");
                let p0_resolved = ctx.resolve_binding(params[0]);
                assert!(ctx.is_integer(p0_resolved), "param should be Int<32>");
                assert!(
                    ctx.is_bool(ctx.resolve_binding(*ret)),
                    "return should be Bool"
                );
            }
            other => panic!("instance should be fn type, got {:?}", other),
        }
    }

    #[test]
    fn test_s_inst_copy_pg_alias_resolved() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let pg = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let pg_id = match ctx.get(pg) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        let inst = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let inst_id = match ctx.get(inst) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.register_instance(pg_id, inst_id);
        let bool_ty = ctx.bool();
        ctx.bindings.insert(pg, bool_ty);
        let resolved_pg = ctx.resolve_binding(pg);
        let updated = infer.s_inst_copy(&mut ctx, pg_id, resolved_pg);
        assert_eq!(updated, 1, "should have updated the instance");

        let inst_resolved = ctx.resolve_binding(inst);
        assert!(ctx.is_bool(inst_resolved), "instance should now be Bool");
    }

    // ═══════════════════════════════════════════════════════════════
    // S-Exists-Lower Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_s_exists_lower_concrete() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        // Enter a deeper level so the var's level > current_level
        let prev = infer.enter_level();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        // S-Exists-Lower requires PG status and level > current_level
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(var_id);
        // exit_level so current_level drops below the var's level
        infer.exit_level(prev);
        let lowered = infer.s_exists_lower(&mut ctx, var_id, &SkolemEnv::default(), &[], None);
        // Removed the unsound level-based fallback.
        // Without determinacy proof, the variable stays PG.
        assert!(
            !lowered,
            "S-Exists-Lower should fail without determinacy proof"
        );
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::PartiallyGeneralizable,
            "PG var should stay PG when determinacy cannot be proven"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration: Complete Solve with Shape Variables
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_solve_eq_concrete_success() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let symbols = SymbolTable::new(crate::hir::types::CrateId(DefId(0)));
        let trait_env = TraitEnv::new();

        let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        // Add constraint: Eq(a, b)
        infer.add_constraint(Constraint::Eq(
            a,
            b,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            EqOrigin::Normal,
        ));
        // Unify a with Int<32>
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        ctx.bindings.insert(a, int_ty);

        let result = infer.solve(&mut ctx, &trait_env, &symbols);
        assert!(result.is_ok(), "solve should succeed");
        // b should now be resolved to Int<32> too
        let b_resolved = ctx.resolve_binding(b);
        assert!(ctx.is_integer(b_resolved), "b should be Int<32>");
    }

    #[test]
    fn test_solve_level_promotion() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let symbols = SymbolTable::new(crate::hir::types::CrateId(DefId(0)));
        let trait_env = TraitEnv::new();

        // Create a variable at a deeper level
        let prev = infer.enter_level();
        let deep_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let shallow_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        infer.exit_level(prev);

        // Eq(deep, shallow) — should promote deep to shallow's level
        infer.add_constraint(Constraint::Eq(
            deep_var,
            shallow_var,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            EqOrigin::Normal,
        ));
        let bool_ty = ctx.bool();
        ctx.bindings.insert(shallow_var, bool_ty);

        let result = infer.solve(&mut ctx, &trait_env, &symbols);
        assert!(result.is_ok(), "level promotion solve should succeed");
        let deep_resolved = ctx.resolve_binding(deep_var);
        assert!(
            ctx.is_bool(deep_resolved),
            "deep var should resolve to Bool"
        );
    }

    #[test]
    fn test_try_match_via_shape_var_suspend_discharge_roundtrip() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        // Register a match branch: Arrow → Eq(Int, Int)
        let int32 = ctx.int(DEFAULT_INT_WIDTH, true);
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Arc::from(vec![Constraint::Eq(
                int32,
                int32,
                crate::ast::Span::new(FileId::DUMMY, 0, 0),
                EqOrigin::Normal,
            )]),
            else_continuation: Arc::from(Vec::new() as Vec<Constraint>),
        }];
        let bid = infer.register_match_branches(branches);

        // Create an infer var and try_match via shape var (should suspend)
        let infer_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(infer_var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        let mut heap = BucketQueue::new();
        let _handled = infer.try_match_via_shape_var(
            &mut ctx,
            infer_var,
            bid,
            &mut heap,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            &Arc::new(HashMap::default()),
        );
        // For an InferVar, try_match_via_shape_var suspends the Match constraint
        // on the wait list (returns true) rather than discharging immediately.
        // For a concrete (non-InferVar) type, it delegates to discharge_match and
        // returns the actual result — true if a branch pattern matched or else_
        // fallback fired, false if neither.
        assert!(
            var_id < infer.wait_lists.len() && !infer.wait_lists[var_id].is_empty(),
            "match should be suspended on the infer var's wait list"
        );

        // Now resolve the infer var to a concrete fn type and wake it
        let fn_bool = ctx.bool();
        let fn_ty = ctx.function(smallvec![fn_bool], fn_bool);
        ctx.bindings.insert(infer_var, fn_ty);
        infer.wake_var_incremental(var_id, &mut heap, &ctx);

        // The match should now be in the heap
        assert!(!heap.is_empty(), "match should be woken and placed in heap");
        let woken = heap.pop().unwrap();
        assert!(
            matches!(woken.constraint, Constraint::Match { .. }),
            "woken constraint should be Match"
        );

        // Discharge it directly
        let fn_ty2 = ctx.function(smallvec![ctx.bool()], ctx.bool());
        let discharged = infer.discharge_match(
            &mut ctx,
            fn_ty2,
            bid,
            &mut heap,
            &Arc::new(HashMap::default()),
        );
        assert!(discharged, "match on fn type should discharge");
        // The continuation Eq(int32, int32) should be in the heap now
        assert!(
            !heap.is_empty(),
            "continuation constraints should be enqueued"
        );
    }

    #[test]
    fn test_force_generalize_after_solve_completes_pg() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        // Create variables at deeper scope (simulating let-polymorphism)
        let _prev = infer.enter_level();
        let x = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let x_id = match ctx.get(x) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        assert!(
            infer.get_var_level(x_id).unwrap_or(0) > 0,
            "x should be at a deeper level"
        );

        // Bind x to Int<32>
        let int32 = ctx.int(DEFAULT_INT_WIDTH, true);
        ctx.bindings.insert(x, int32);

        // Mark the variable PG and then force_generalize
        infer.gen_statuses[x_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(x_id);
        infer.force_generalize(&mut ctx);

        assert_eq!(
            infer.gen_statuses[x_id],
            GenStatus::Generalized,
            "resolved PG var at inner scope should generalize"
        );
    }

    #[test]
    fn test_solve_cross_branch_nca_promotion() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let symbols = SymbolTable::new(crate::hir::types::CrateId(DefId(0)));
        let trait_env = TraitEnv::new();

        // Create a tree: root → branch_A → leaf_A1 (a_deep)
        //                  └─ branch_B → leaf_B1 (b_deep)
        // Variables in leaf_A1 and leaf_B1 are siblings, not ancestors.
        // Without NCA-based promotion, unifying them would place the
        // promoted variable in the WRONG branch, breaking scoping.
        let _r0 = infer.enter_level(); // → branch_A
        let _r1 = infer.enter_level(); // → leaf_A1
        let a_deep = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let a_deep_id = match ctx.get(a_deep) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.exit_level(_r1); // back to branch_A
        infer.exit_level(_r0); // back to root

        // Now enter branch_B → leaf_B1
        let _r2 = infer.enter_level(); // → branch_B
        let _r3 = infer.enter_level(); // → leaf_B1
        let b_deep = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let b_deep_id = match ctx.get(b_deep) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.exit_level(_r3); // back to branch_B
        infer.exit_level(_r2); // back to root

        // Verify the variables are at different (sibling) regions
        let a_region = infer.type_vars[a_deep_id].region_id;
        let b_region = infer.type_vars[b_deep_id].region_id;
        assert_ne!(
            a_region, b_region,
            "sibling vars must be in different regions"
        );
        let root = infer.region_tree.root;
        assert!(
            infer.region_tree.is_ancestor(root, a_region),
            "root must be ancestor of a's region",
        );
        assert!(
            infer.region_tree.is_ancestor(root, b_region),
            "root must be ancestor of b's region",
        );

        // Eq(a_deep, b_deep) — should promote both to NCA (root) before unifying
        infer.add_constraint(Constraint::Eq(
            a_deep,
            b_deep,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            EqOrigin::Normal,
        ));

        // Bind B to Int<32> so the solver can resolve
        let int32 = ctx.int(DEFAULT_INT_WIDTH, true);
        ctx.bindings.insert(b_deep, int32);

        let result = infer.solve(&mut ctx, &trait_env, &symbols);
        assert!(
            result.is_ok(),
            "cross-branch promotion solve should succeed: {:?}",
            result,
        );

        // Both variables should resolve to Int<32>
        let a_resolved = ctx.resolve_binding(a_deep);
        let b_resolved = ctx.resolve_binding(b_deep);
        assert!(
            ctx.is_integer(a_resolved),
            "a_deep should resolve to Int, got {:?}",
            ctx.get(a_resolved),
        );
        assert!(
            ctx.is_integer(b_resolved),
            "b_deep should resolve to Int, got {:?}",
            ctx.get(b_resolved),
        );
    }

    #[test]
    fn test_solve_cross_branch_no_leak_on_unification_failure() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let symbols = SymbolTable::new(crate::hir::types::CrateId(DefId(0)));
        let trait_env = TraitEnv::new();

        // Sibling branches: promote to NCA, then unify with incompatible
        // types should still fail (smoke test that NCA doesn't paper over
        // real type errors).
        let _r0 = infer.enter_level(); // → branch_A
        let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        infer.exit_level(_r0);

        let _r1 = infer.enter_level(); // → branch_B
        let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        infer.exit_level(_r1);

        // Bind a to Bool, b to Int<32>, then constrain Eq(a, b)
        let bool_ty = ctx.bool();
        let int32 = ctx.int(DEFAULT_INT_WIDTH, true);
        ctx.bindings.insert(a, bool_ty);
        ctx.bindings.insert(b, int32);

        infer.add_constraint(Constraint::Eq(
            a,
            b,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            EqOrigin::Normal,
        ));

        let result = infer.solve(&mut ctx, &trait_env, &symbols);
        assert!(
            result.is_err(),
            "cross-branch unification of Bool and Int should fail, got {:?}",
            result,
        );
    }

    #[test]
    fn test_cross_region_generalize_after_rollback() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        // This test exercises rollback_pool directly, outside any snapshot —
        // enable pool logging explicitly.
        infer.region_tree.pool_logging_enabled = true;

        // enter_level() returns the parent region, sets current to child.
        let _parent = infer.enter_level();
        let child_region = infer.region_tree.current;
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        // The variable should be in the child region's pool
        assert!(
            infer.region_tree.nodes[child_region.0]
                .pool
                .var_ids
                .contains(&var_id),
            "new var should be in child region's pool"
        );

        // Promote the variable to the parent (root) region
        let root = infer.region_tree.root;
        let promoted = infer.try_promote_var(&mut ctx, var_id, root);
        assert!(promoted.is_some(), "try_promote_var should succeed");
        let promoted_id = match ctx.get(promoted.unwrap()) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };

        // The promoted variable should be in the root region's pool
        assert!(
            infer.region_tree.nodes[root.0]
                .pool
                .var_ids
                .contains(&promoted_id),
            "promoted var should be in root region's pool"
        );

        // Simulate a transaction rollback: roll back the pool
        infer.region_tree.rollback_pool();

        // After rollback, the root region's pool should be restored to its
        // pre-registration state (the promoted variable should be removed)
        assert!(
            !infer.region_tree.nodes[root.0]
                .pool
                .var_ids
                .contains(&promoted_id),
            "after rollback, promoted var should NOT be in root region's pool"
        );
        // Both the original variable and the promoted variable were created
        // inside the undo-log scope.  With value-based rollback, every
        // Register inverse removes the exact var that was added, so both
        // variables are correctly removed from the child pool.
        assert!(
            !infer.region_tree.nodes[child_region.0]
                .pool
                .var_ids
                .contains(&promoted_id),
            "after rollback, promoted var should NOT be in child region's pool"
        );
        assert!(
            !infer.region_tree.nodes[child_region.0]
                .pool
                .var_ids
                .contains(&var_id),
            "after rollback, original var should NOT be in child region's pool"
        );
        // Both pools should be empty (no variables existed before the
        // undo-logged operations).
        assert!(
            infer.region_tree.nodes[root.0].pool.var_ids.is_empty(),
            "after rollback, root pool should be empty"
        );
        assert!(
            infer.region_tree.nodes[child_region.0]
                .pool
                .var_ids
                .is_empty(),
            "after rollback, child pool should be empty"
        );

        // Now run force_root_generalization — it should not crash
        // (both pools are empty, so nothing to generalize).
        let generalized = infer.force_root_generalization(&mut ctx);
        assert!(
            generalized.is_empty(),
            "after rollback with empty pools, nothing should be generalized"
        );
    }

    // rollback_pool value-based — value-based rollback using var identity
    // rather than position.
    // Value-based rollback using var identity rather than position.
    // The OmniML reference implementation (omniml/lib/constraint_solver/
    // generalization.ml) avoids this by using a union-find transactional
    // store and never removing individual variables from pools by value.
    // Ponent's `unregister_var` (for `try_promote_var`) requires value-based
    // rollback because the pool is a bag — element order is not semantically
    // meaningful.

    #[test]
    fn test_rollback_pool_register_then_unregister_same_region() {
        // Register → Unregister → Register on the same region.
        // Initial pool: [A]
        // 1. Register(B) → [A, B]
        // 2. Unregister(A) → [B]
        // 3. Register(C) → [B, C]
        // Rollback must restore to [A] — the set of elements, not their order.
        let mut ctx = TypeContext::new();
        let mut tree = InferRegionTree::new();
        // These tests exercise the pool undo machinery directly, outside any
        // snapshot — enable logging explicitly.
        tree.pool_logging_enabled = true;
        let root = tree.root;

        // Seed the root pool with var 99 (simulating a pre-existing variable).
        tree.nodes[root.0].pool.var_ids.push(99);

        let pool_before = tree.nodes[root.0].pool.var_ids.clone();
        assert_eq!(pool_before, vec![99], "seed: pool should start with [99]");

        // Step 1: Register var 100
        tree.register_var_in_region(100, root);
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![99, 100],
            "after register 100: pool should be [99, 100]"
        );

        // Step 2: Unregister var 99 (the original seed)
        tree.unregister_var(99, root);
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![100],
            "after unregister 99: pool should be [100]"
        );

        // Step 3: Register var 200
        tree.register_var_in_region(200, root);
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![100, 200],
            "after register 200: pool should be [100, 200]"
        );

        // Rollback — should restore to [99]
        tree.rollback_pool();
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![99],
            "rollback: pool should be restored to [99], got {:?}",
            tree.nodes[root.0].pool.var_ids,
        );
        assert!(
            tree.pool_undo_log.is_empty(),
            "rollback should clear the undo log"
        );
    }

    #[test]
    fn test_rollback_pool_unregister_then_register_same_region() {
        // Reverse order of operations: Unregister first, then Register.
        // Initial pool state: [A, B]
        // 1. Unregister(A) → [B]
        // 2. Register(C) → [B, C]
        // Rollback should restore to [A, B] (set, not necessarily order).
        //
        // Note: the pool is a bag (Vec used as a set) — element order
        // is not semantically meaningful.  This matches the OmniML
        // reference implementation (omniml/lib/constraint_solver/
        // generalization.ml) where the pool is an unordered list and
        // variables are checked for membership rather than position.
        let mut ctx = TypeContext::new();
        let mut tree = InferRegionTree::new();
        // These tests exercise the pool undo machinery directly, outside any
        // snapshot — enable logging explicitly.
        tree.pool_logging_enabled = true;
        let root = tree.root;

        let mut sort = |v: &mut Vec<usize>| v.sort();

        tree.nodes[root.0].pool.var_ids = vec![10, 20];

        let mut expected = vec![10, 20];
        sort(&mut expected);
        let mut actual = tree.nodes[root.0].pool.var_ids.clone();
        sort(&mut actual);
        assert_eq!(actual, expected, "seed: pool should start with [10, 20]");

        tree.unregister_var(10, root);
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![20],
            "after unregister 10: pool should be [20]"
        );

        tree.register_var_in_region(30, root);
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![20, 30],
            "after register 30: pool should be [20, 30]"
        );

        tree.rollback_pool();
        let mut actual = tree.nodes[root.0].pool.var_ids.clone();
        sort(&mut actual);
        assert_eq!(
            actual, expected,
            "rollback: pool should be restored to [10, 20] (set), got {:?}",
            tree.nodes[root.0].pool.var_ids,
        );
    }

    #[test]
    fn test_rollback_pool_register_only_same_region() {
        // Simple case: only Register entries, no Unregister.
        // Initial: [A]
        // 1. Register(B) → [A, B]
        // 2. Register(C) → [A, B, C]
        // Rollback should restore to [A].
        let mut ctx = TypeContext::new();
        let mut tree = InferRegionTree::new();
        // These tests exercise the pool undo machinery directly, outside any
        // snapshot — enable logging explicitly.
        tree.pool_logging_enabled = true;
        let root = tree.root;

        tree.nodes[root.0].pool.var_ids.push(1);

        tree.register_var_in_region(2, root);
        tree.register_var_in_region(3, root);
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![1, 2, 3],
            "after two registers: pool should be [1, 2, 3]"
        );

        tree.rollback_pool();
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![1],
            "rollback: pool should be restored to [1], got {:?}",
            tree.nodes[root.0].pool.var_ids,
        );
    }

    #[test]
    fn test_rollback_pool_different_regions() {
        // Register and Unregister on DIFFERENT regions should not interfere.
        // Root pool: [A]
        // Child pool: [B]
        // 1. Register(C) in root → root [A, C], child [B]
        // 2. Unregister(B) from child → root [A, C], child []
        // Rollback should restore both pools independently.
        let mut ctx = TypeContext::new();
        let mut tree = InferRegionTree::new();
        // These tests exercise the pool undo machinery directly, outside any
        // snapshot — enable logging explicitly.
        tree.pool_logging_enabled = true;
        let root = tree.root;

        let _parent = tree.enter_region();
        let child = tree.current;

        tree.nodes[root.0].pool.var_ids.push(10);
        tree.nodes[child.0].pool.var_ids.push(20);

        // Register in root
        tree.register_var_in_region(30, root);
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![10, 30],
            "root pool after register 30"
        );
        assert_eq!(
            tree.nodes[child.0].pool.var_ids,
            vec![20],
            "child pool unchanged"
        );

        // Unregister from child
        tree.unregister_var(20, child);
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![10, 30],
            "root pool unchanged after child unregister"
        );
        assert_eq!(
            tree.nodes[child.0].pool.var_ids,
            Vec::<usize>::new(),
            "child pool after unregister 20"
        );

        tree.rollback_pool();
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![10],
            "root pool after rollback: should be [10], got {:?}",
            tree.nodes[root.0].pool.var_ids,
        );
        assert_eq!(
            tree.nodes[child.0].pool.var_ids,
            vec![20],
            "child pool after rollback: should be [20], got {:?}",
            tree.nodes[child.0].pool.var_ids,
        );
    }

    #[test]
    fn test_rollback_pool_rigid_var() {
        // RigidVar registers should also roll back correctly.
        let mut ctx = TypeContext::new();
        let mut tree = InferRegionTree::new();
        // These tests exercise the pool undo machinery directly, outside any
        // snapshot — enable logging explicitly.
        tree.pool_logging_enabled = true;
        let root = tree.root;

        tree.nodes[root.0].pool.rigid_var_ids.push(1);

        tree.register_rigid_var(2);
        assert_eq!(
            tree.nodes[root.0].pool.rigid_var_ids,
            vec![1, 2],
            "after rigid register: should be [1, 2]"
        );

        tree.rollback_pool();
        assert_eq!(
            tree.nodes[root.0].pool.rigid_var_ids,
            vec![1],
            "rigid rollback: should be restored to [1], got {:?}",
            tree.nodes[root.0].pool.rigid_var_ids,
        );
    }

    #[test]
    fn test_rollback_pool_empty_undo_log() {
        // Rollback with no entries should be a no-op.
        let mut ctx = TypeContext::new();
        let mut tree = InferRegionTree::new();
        // These tests exercise the pool undo machinery directly, outside any
        // snapshot — enable logging explicitly.
        tree.pool_logging_enabled = true;
        let root = tree.root;

        tree.nodes[root.0].pool.var_ids.push(42);
        let snapshot = tree.nodes[root.0].pool.var_ids.clone();

        tree.rollback_pool();
        assert_eq!(
            tree.nodes[root.0].pool.var_ids, snapshot,
            "rollback with empty log should not change pool"
        );
    }

    #[test]
    fn test_rollback_pool_interleaved_register_and_unregister() {
        // Complex interleaving: Register, Unregister, Register, Unregister
        // on the same region.  Verifies that the value-based single-pass
        // approach correctly handles multiple alternating operations.
        let mut ctx = TypeContext::new();
        let mut tree = InferRegionTree::new();
        // These tests exercise the pool undo machinery directly, outside any
        // snapshot — enable logging explicitly.
        tree.pool_logging_enabled = true;
        let root = tree.root;

        // Seed: [1, 2]
        tree.nodes[root.0].pool.var_ids = vec![1, 2];

        // 1. Register(3) → [1, 2, 3]
        tree.register_var_in_region(3, root);
        // 2. Unregister(2) → [1, 3]
        tree.unregister_var(2, root);
        // 3. Register(4) → [1, 3, 4]
        tree.register_var_in_region(4, root);
        // 4. Unregister(1) → [3, 4]
        tree.unregister_var(1, root);

        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![3, 4],
            "after all operations: pool should be [3, 4]"
        );

        tree.rollback_pool();
        assert_eq!(
            tree.nodes[root.0].pool.var_ids,
            vec![1, 2],
            "rollback: pool should be restored to [1, 2], got {:?}",
            tree.nodes[root.0].pool.var_ids,
        );
    }

    // ── gen_statuses undo discipline ──────────────────────────────
    // Every gen_statuses mutation must be undo-logged: a rollback after
    // a sweep / lowering / solve mutation must restore the status to its
    // pre-snapshot value, keeping it consistent with the rolled-back
    // guard_sets / forward_refs.

    #[test]
    fn test_rollback_restores_gen_status_after_force_generalize() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        // PG without guards (the state after transitive guard removal) —
        // a guarded var is skipped by the sweep.
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(var_id);
        let int32 = ctx.int(32, true);
        ctx.set_binding(var, int32);

        let snap = infer.start_snapshot();
        infer.force_generalize(&mut ctx);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Generalized,
            "sweep must generalize the eligible var"
        );
        infer.rollback_to(snap);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::PartiallyGeneralizable,
            "rollback must restore the PG status, not leave a stale Generalized"
        );
    }

    #[test]
    fn test_rollback_restores_gen_status_after_s_exists_lower() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.add_guard(var_id); // PG + guard
        let int32 = ctx.int(32, true);
        ctx.set_binding(var, int32);

        let snap = infer.start_snapshot();
        let lowered = infer.s_exists_lower(&ctx, var_id, &SkolemEnv::default(), &[], None);
        assert!(lowered, "resolved PG var must be lowered via UNI-TYPE");
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Ungeneralized,
            "lowering must set Ungeneralized"
        );
        infer.rollback_to(snap);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::PartiallyGeneralizable,
            "rollback must restore the PG status after a lowering"
        );
    }

    /// The user-visible scenario: a snapshot open across a solve() that
    /// generalizes a current var (step 3.5) and copies it to instances.
    /// Rollback must revert the status AND the forward_refs/instance
    /// statuses together — no stale Generalized with restored instances.
    #[test]
    fn test_rollback_after_solve_restores_gen_statuses() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let symbols = SymbolTable::new(crate::hir::types::CrateId(DefId(0)));
        let trait_env = TraitEnv::new();

        let pg_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let pg_id = match ctx.get(pg_var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        let instance = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let inst_id = match ctx.get(instance) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        // PG without guards (the state after transitive guard removal),
        // with a registered instance.  Set BEFORE the snapshot so the
        // rollback has to undo only the solve-loop mutations.
        infer.gen_statuses[pg_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(pg_id);
        infer.register_instance(pg_id, inst_id);
        let int32 = ctx.int(32, true);
        infer.add_constraint(Constraint::Eq(
            pg_var,
            int32,
            crate::ast::Span::new(FileId::DUMMY, 0, 0),
            EqOrigin::Normal,
        ));

        let snap = infer.start_snapshot();
        let result = infer.solve(&mut ctx, &trait_env, &symbols);
        assert!(result.is_ok(), "solve should succeed, got {:?}", result);
        assert_eq!(
            infer.gen_statuses[pg_id],
            GenStatus::Generalized,
            "eligible current var must be generalized during solve"
        );
        assert!(
            infer.forward_refs[pg_id].is_empty(),
            "s_inst_copy clears forward_refs after propagating"
        );

        infer.rollback_to(snap);
        assert_eq!(
            infer.gen_statuses[pg_id],
            GenStatus::PartiallyGeneralizable,
            "rollback must restore PG — a stale Generalized would desync with the restored forward_refs"
        );
        assert!(
            infer.forward_refs[pg_id].contains(&inst_id),
            "rollback must restore the instance registration"
        );
        assert_eq!(
            infer.gen_statuses[inst_id],
            GenStatus::PartialInstance,
            "rollback must restore the instance's PartialInstance status"
        );
    }

    // ── Det-Dom / Det-Esc tests (OmniML §5.3) ──────────────────────

    #[test]
    fn test_det_dom_outside_var() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        // Create an outer variable (level 0).
        let outer = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);

        // Enter a deeper region and create the PG variable.
        let prev = infer.enter_level();
        let inner = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let inner_id = match ctx.get(inner) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[inner_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(inner_id);

        // Eq(inner, outer): the outside variable determines inner.
        let active = vec![PrioritizedConstraint {
            priority: 0,
            constraint: Constraint::Eq(inner, outer, crate::ast::Span::new(FileId::DUMMY, 0, 0), EqOrigin::Normal),
            env: Arc::new(HashMap::default()),
        }];

        let determined =
            infer.check_determinacy(&ctx, inner_id, &empty_skolem_env(), &active, None);
        assert!(
            determined,
            "Det-Dom: outside variable should determine inner PG var"
        );

        infer.exit_level(prev);
    }

    #[test]
    fn test_det_esc_concrete_type() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        let prev = infer.enter_level();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(var_id);

        // Eq(var, Int<32>): concrete type determines var.
        let int_ty = ctx.int(DEFAULT_INT_WIDTH, true);
        let active = vec![PrioritizedConstraint {
            priority: 0,
            constraint: Constraint::Eq(var, int_ty, crate::ast::Span::new(FileId::DUMMY, 0, 0), EqOrigin::Normal),
            env: Arc::new(HashMap::default()),
        }];

        let determined = infer.check_determinacy(&ctx, var_id, &empty_skolem_env(), &active, None);
        assert!(determined, "Det-Esc: concrete type should determine PG var");

        infer.exit_level(prev);
    }

    #[test]
    fn test_det_esc_inside_alias_not_determined() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        let prev = infer.enter_level();
        let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let a_id = match ctx.get(a) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        infer.gen_statuses[a_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(a_id);

        // Eq(a, b): both in the same region → NOT determined.
        let active = vec![PrioritizedConstraint {
            priority: 0,
            constraint: Constraint::Eq(a, b, crate::ast::Span::new(FileId::DUMMY, 0, 0), EqOrigin::Normal),
            env: Arc::new(HashMap::default()),
        }];

        let determined = infer.check_determinacy(&ctx, a_id, &empty_skolem_env(), &active, None);
        assert!(
            !determined,
            "inside-region alias must NOT determine the PG var"
        );

        infer.exit_level(prev);
    }

    #[test]
    fn test_det_dom_composite_type() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        // Outer variable at level 0.
        let outer = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);

        // Inner region.
        let prev = infer.enter_level();
        let inner = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let inner_id = match ctx.get(inner) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[inner_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(inner_id);

        // Eq(outer, Fn(inner, Bool)):
        // outer is outside; the RHS mentions inner → Det-Dom.
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(smallvec![inner], bool_ty);
        let active = vec![PrioritizedConstraint {
            priority: 0,
            constraint: Constraint::Eq(outer, fn_ty, crate::ast::Span::new(FileId::DUMMY, 0, 0), EqOrigin::Normal),
            env: Arc::new(HashMap::default()),
        }];

        let determined =
            infer.check_determinacy(&ctx, inner_id, &empty_skolem_env(), &active, None);
        assert!(
            determined,
            "Det-Dom: outside var equated with fn(inner) should determine inner"
        );

        infer.exit_level(prev);
    }

    #[test]
    fn test_s_exists_lower_via_det_dom() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        let outer = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);

        let prev = infer.enter_level();
        let inner = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let inner_id = match ctx.get(inner) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[inner_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(inner_id);

        let active = vec![PrioritizedConstraint {
            priority: 0,
            constraint: Constraint::Eq(inner, outer, crate::ast::Span::new(FileId::DUMMY, 0, 0), EqOrigin::Normal),
            env: Arc::new(HashMap::default()),
        }];

        let lowered = infer.s_exists_lower(&ctx, inner_id, &SkolemEnv::default(), &active, None);
        assert!(lowered, "s_exists_lower should succeed via Det-Dom");
        assert_eq!(
            infer.gen_statuses[inner_id],
            GenStatus::Ungeneralized,
            "PG var should be lowered to Ungeneralized"
        );

        infer.exit_level(prev);
    }

    /// Transitive Det-Dom: α = β (inside), β = γ (outside).
    /// The approximation misses this; the complete check catches it.
    #[test]
    fn test_det_dom_transitive() {
        let mut ctx = new_ctx();
        let mut infer = InferenceContext::new();

        // Outside variable (level 0).
        let gamma = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);

        // Inner region with two variables.
        let prev = infer.enter_level();
        let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let beta = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let alpha_id = match ctx.get(alpha) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[alpha_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(alpha_id);

        // Eq(α, β) and Eq(β, γ): transitive chain.
        let active = vec![
            PrioritizedConstraint {
                priority: 0,
                constraint: Constraint::Eq(
                    alpha,
                    beta,
                    crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    EqOrigin::Normal,
                ),
                env: Arc::new(HashMap::default()),
            },
            PrioritizedConstraint {
                priority: 0,
                constraint: Constraint::Eq(
                    beta,
                    gamma,
                    crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    EqOrigin::Normal,
                ),
                env: Arc::new(HashMap::default()),
            },
        ];

        let determined =
            infer.check_determinacy(&ctx, alpha_id, &empty_skolem_env(), &active, None);
        assert!(
            determined,
            "transitive Det-Dom: α = β = γ (γ outside) must determine α"
        );

        infer.exit_level(prev);
    }

    /// Transitive Det-Esc: α = β (inside), β = Int<32>.
    #[test]
    fn test_det_esc_transitive() {
        let mut ctx = new_ctx();
        let mut infer = InferenceContext::new();

        let prev = infer.enter_level();
        let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let beta = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let alpha_id = match ctx.get(alpha) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[alpha_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(alpha_id);

        let int32 = ctx.int(DEFAULT_INT_WIDTH, true);
        let active = vec![
            PrioritizedConstraint {
                priority: 0,
                constraint: Constraint::Eq(
                    alpha,
                    beta,
                    crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    EqOrigin::Normal,
                ),
                env: Arc::new(HashMap::default()),
            },
            PrioritizedConstraint {
                priority: 0,
                constraint: Constraint::Eq(
                    beta,
                    int32,
                    crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    EqOrigin::Normal,
                ),
                env: Arc::new(HashMap::default()),
            },
        ];

        let determined =
            infer.check_determinacy(&ctx, alpha_id, &empty_skolem_env(), &active, None);
        assert!(
            determined,
            "transitive Det-Esc: α = β = Int<32> must determine α"
        );

        infer.exit_level(prev);
    }

    /// Det-Dom composite: α = Fn(β, γ) where γ is outside the region.
    #[test]
    fn test_det_dom_composite_outside_var() {
        let mut ctx = new_ctx();
        let mut infer = InferenceContext::new();

        let gamma = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);

        let prev = infer.enter_level();
        let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let beta = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let alpha_id = match ctx.get(alpha) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[alpha_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(alpha_id);

        // α = Fn(β, γ): γ is outside the region.
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(smallvec![beta, gamma], bool_ty);
        let active = vec![PrioritizedConstraint {
            priority: 0,
            constraint: Constraint::Eq(alpha, fn_ty, crate::ast::Span::new(FileId::DUMMY, 0, 0), EqOrigin::Normal),
            env: Arc::new(HashMap::default()),
        }];

        let determined =
            infer.check_determinacy(&ctx, alpha_id, &empty_skolem_env(), &active, None);
        assert!(
            determined,
            "Det-Dom composite: α = Fn(β, γ) with γ outside must determine α"
        );

        infer.exit_level(prev);
    }

    /// End-to-end: s_exists_lower succeeds via transitive Det-Dom.
    #[test]
    fn test_s_exists_lower_transitive_det_dom() {
        let mut ctx = new_ctx();
        let mut infer = InferenceContext::new();

        let gamma = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);

        let prev = infer.enter_level();
        let alpha = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let beta = infer.new_type_var(&mut ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let alpha_id = match ctx.get(alpha) {
            TypeData::InferVar { id, .. } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[alpha_id] = GenStatus::PartiallyGeneralizable;
        infer.pg_vars.insert(alpha_id);

        let active = vec![
            PrioritizedConstraint {
                priority: 0,
                constraint: Constraint::Eq(
                    alpha,
                    beta,
                    crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    EqOrigin::Normal,
                ),
                env: Arc::new(HashMap::default()),
            },
            PrioritizedConstraint {
                priority: 0,
                constraint: Constraint::Eq(
                    beta,
                    gamma,
                    crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    EqOrigin::Normal,
                ),
                env: Arc::new(HashMap::default()),
            },
        ];

        let lowered = infer.s_exists_lower(&ctx, alpha_id, &SkolemEnv::default(), &active, None);
        assert!(
            lowered,
            "s_exists_lower must succeed via transitive Det-Dom"
        );
        assert_eq!(
            infer.gen_statuses[alpha_id],
            GenStatus::Ungeneralized,
            "PG var must be lowered to Ungeneralized"
        );

        infer.exit_level(prev);
    }
}
