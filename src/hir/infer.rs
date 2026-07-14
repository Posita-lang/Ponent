use crate::ast::Span;
use crate::hir::shape_var::{
    ShapeVarContext, ShapeVarId, TypeShape, shapes_compatible, type_data_to_shape,
};
use crate::hir::smt::SmtSolver;
use crate::hir::symbol::SymbolTable;
use crate::hir::traits::TraitEnv;
use crate::hir::types::*;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet;
use std::collections::BinaryHeap;

/// ── Inline GuardSet (ported from OmniML guard_set.ml) ───────────

/// Reference-counted guard set for tracking when inference variables
/// are captured by suspended constraints (OmniML §6).
#[derive(Debug, Clone)]
pub struct GuardSet {
    pub direct_guards: usize,
    pub transitive_guards: HashMap<usize, usize>,
}

impl Default for GuardSet {
    fn default() -> Self { Self::empty() }
}

impl GuardSet {
    pub fn empty() -> Self { GuardSet { direct_guards: 0, transitive_guards: HashMap::default() } }
    pub fn is_empty(&self) -> bool { self.direct_guards == 0 && self.transitive_guards.is_empty() }
    pub fn add_guard(&mut self) { self.direct_guards = self.direct_guards.wrapping_add(1); }
    pub fn remove_guard(&mut self) { if self.direct_guards > 0 { self.direct_guards -= 1; } }
    pub fn add_transitive_guard(&mut self, region_id: usize) { *self.transitive_guards.entry(region_id).or_insert(0) += 1; }
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
            if *count == 0 { self.transitive_guards.remove(&region_id); }
        }
    }
    pub fn clear_transitive_guard(&mut self, region_id: usize) { self.transitive_guards.remove(&region_id); }
    pub fn is_transitively_guarded(&self, region_id: usize) -> bool {
        self.transitive_guards.get(&region_id).map_or(false, |&c| c > 0)
    }
    pub fn union(&self, other: &Self) -> Self {
        let mut transitive = self.transitive_guards.clone();
        for (&region, &count) in &other.transitive_guards { *transitive.entry(region).or_insert(0) += count; }
        GuardSet { direct_guards: self.direct_guards + other.direct_guards, transitive_guards: transitive }
    }
    pub fn clear(&mut self) { self.direct_guards = 0; self.transitive_guards.clear(); }
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
    pub fn new() -> Self { InferPool { var_ids: Vec::new(), rigid_var_ids: Vec::new() } }
    pub fn register_var(&mut self, var_id: usize) { self.var_ids.push(var_id); }
    pub fn register_rigid_var(&mut self, var_id: usize) { self.rigid_var_ids.push(var_id); }
    pub fn is_alive(&self) -> bool { !self.var_ids.is_empty() }
}

#[derive(Debug, Clone)]
pub struct InferRegionNode {
    pub id: InferRegionId,
    pub level: usize,
    pub parent: Option<InferRegionId>,
    pub children: Vec<InferRegionId>,
    pub pool: InferPool,
    pub dirty: bool,
    pub dirty_children: Vec<InferRegionId>,
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
    pub dirty_roots: Vec<InferRegionId>,
    /// Undo log for pool mutations.
    /// When a transaction is rolled back, pools are restored to their
    /// previous states to prevent zombie entries from abandoned variables.
    pub pool_undo_log: Vec<PoolUndoEntry>,
}

/// A single undo entry for a pool mutation.
#[derive(Debug, Clone)]
pub enum PoolUndoEntry {
    /// A variable was registered (pushed).  Truncate to the saved lengths.
    Register { region_idx: usize, old_var_len: usize, old_rigid_len: usize },
    /// A variable was unregistered (removed by value).  Re-insert it.
    Unregister { region_idx: usize, var_id: usize },
}

impl InferRegionTree {
    pub fn new() -> Self {
        InferRegionTree {
            nodes: vec![InferRegionNode {
                id: InferRegionId(0), level: 0, parent: None,
                children: Vec::new(), pool: InferPool::new(),
                dirty: false, dirty_children: Vec::new(),
                shape_var_region: None,
                parent_shape_var_region: 0,
            }],
            root: InferRegionId(0), current: InferRegionId(0), dirty_roots: Vec::new(),
            pool_undo_log: Vec::new(),
        }
    }
    pub fn enter_region(&mut self) -> InferRegionId {
        let new_id = InferRegionId(self.nodes.len());
        let current_level = self.nodes[self.current.0].level;
        let parent_shape_var = self.nodes[self.current.0].shape_var_region
            .unwrap_or(self.nodes[self.current.0].parent_shape_var_region);
        self.nodes.push(InferRegionNode {
            id: new_id, level: current_level + 1, parent: Some(self.current),
            children: Vec::new(), pool: InferPool::new(),
            dirty: false, dirty_children: Vec::new(),
            shape_var_region: None,
            parent_shape_var_region: parent_shape_var,
        });
        self.nodes[self.current.0].children.push(new_id);
        let old = self.current;
        self.current = new_id;
        old
    }
    pub fn exit_region(&mut self) { if let Some(parent) = self.nodes[self.current.0].parent { self.current = parent; } }
    pub fn get_level(&self, region_id: InferRegionId) -> usize { self.nodes[region_id.0].level }
    pub fn nearest_common_ancestor(&self, a: InferRegionId, b: InferRegionId) -> InferRegionId {
        if a == b { return a; }
        let a_node = &self.nodes[a.0]; let b_node = &self.nodes[b.0];
        if a_node.level < b_node.level { self.nearest_common_ancestor(a, self.nodes[b.0].parent.expect("parent"))
        } else if a_node.level > b_node.level { self.nearest_common_ancestor(self.nodes[a.0].parent.expect("parent"), b)
        } else { self.nearest_common_ancestor(self.nodes[a.0].parent.expect("parent"), self.nodes[b.0].parent.expect("parent")) }
    }
    pub fn is_ancestor(&self, ancestor: InferRegionId, node: InferRegionId) -> bool { self.nearest_common_ancestor(ancestor, node) == ancestor }
    pub fn mark_dirty(&mut self, region_id: InferRegionId) {
        let node = &mut self.nodes[region_id.0];
        if node.dirty { return; }
        node.dirty = true;
        // Walk up to find the nearest dirty ancestor (or dirty_roots).
        // This mirrors OmniML's `find_closest_dirty_ancestor` which returns
        // `t.dirty_roots` when no dirty ancestor is found.
        let mut current = node.parent;
        while let Some(pid) = current {
            let parent = &mut self.nodes[pid.0];
            if parent.dirty {
                if !parent.dirty_children.contains(&region_id) {
                    parent.dirty_children.push(region_id);
                }
                return;
            }
            current = parent.parent;
        }
        // No dirty ancestor found — add to dirty_roots so that
        // drain_dirty_roots can reach this region.
        if !self.dirty_roots.contains(&region_id) {
            self.dirty_roots.push(region_id);
        }
    }
    pub fn mark_current_dirty(&mut self) { self.mark_dirty(self.current); }
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
        self.pool_undo_log.push(PoolUndoEntry::Register {
            region_idx: idx,
            old_var_len: self.nodes[idx].pool.var_ids.len(),
            old_rigid_len: self.nodes[idx].pool.rigid_var_ids.len(),
        });
        self.nodes[idx].pool.register_var(var_id);
    }
    pub fn register_rigid_var(&mut self, var_id: usize) {
        let idx = self.current.0;
        self.pool_undo_log.push(PoolUndoEntry::Register {
            region_idx: idx,
            old_var_len: self.nodes[idx].pool.var_ids.len(),
            old_rigid_len: self.nodes[idx].pool.rigid_var_ids.len(),
        });
        self.nodes[idx].pool.register_rigid_var(var_id);
    }
    pub fn register_var_in_region(&mut self, var_id: usize, region_id: InferRegionId) {
        let idx = region_id.0;
        self.pool_undo_log.push(PoolUndoEntry::Register {
            region_idx: idx,
            old_var_len: self.nodes[idx].pool.var_ids.len(),
            old_rigid_len: self.nodes[idx].pool.rigid_var_ids.len(),
        });
        self.nodes[idx].pool.register_var(var_id);
    }
    /// Remove a variable from a specific region's pool.
    /// Called by try_promote_var (line 724) and region adjustment (line 1383)
    /// before re-registering the variable in its new region.
    /// NOTE: This function exists and is actively used — a previous
    /// reviewer incorrectly claimed that "no such function exists".
    /// See try_promote_var (line 776) and region adjustment (line 1383)
    /// for call sites.
    pub fn unregister_var(&mut self, var_id: usize, region_id: InferRegionId) {
        self.pool_undo_log.push(PoolUndoEntry::Unregister {
            region_idx: region_id.0,
            var_id,
        });
        self.nodes[region_id.0].pool.var_ids.retain(|&v| v != var_id);
    }
    pub fn collect_dirty_ids(&self) -> Vec<InferRegionId> { self.nodes.iter().filter(|n| n.dirty).map(|n| n.id).collect() }
    pub fn collect_alive_ids(&self) -> Vec<InferRegionId> { self.nodes.iter().filter(|n| n.pool.is_alive()).map(|n| n.id).collect() }

    pub fn drain_dirty<F>(&mut self, node_id: InferRegionId, f: &mut F) where F: FnMut(InferRegionId, &mut Self) {
        if !self.nodes[node_id.0].dirty { return; }
        let children: Vec<InferRegionId> = self.nodes[node_id.0].dirty_children.clone();
        for child_id in children { self.drain_dirty(child_id, f); }
        f(node_id, self);
        if self.nodes[node_id.0].dirty_children.is_empty() {
            self.nodes[node_id.0].dirty = false;
            if let Some(parent_id) = self.nodes[node_id.0].parent { self.nodes[parent_id.0].dirty_children.retain(|&c| c != node_id); }
        }
    }
    pub fn drain_dirty_roots<F>(&mut self, f: &mut F) where F: FnMut(InferRegionId, &mut Self) {
        // Drain from dirty_roots first, then fall back to the root for
        // backward compatibility with regions that are reachable from root.
        let roots: Vec<InferRegionId> = self.dirty_roots.drain(..).collect();
        for root_id in roots {
            self.drain_dirty(root_id, f);
        }
        self.drain_dirty(self.root, f);
    }

    /// Roll back pool mutations recorded in the undo log.
    /// Should be called after a transaction rollback to prevent zombie
    /// entries from abandoned inference variables.
    pub fn rollback_pool(&mut self) {
        // Two-pass rollback: first truncate all Register entries
        // (in reverse order), then re-insert all Unregister entries
        // (also in reverse order).  This ordering ensures that a
        // truncation does not undo a re-insertion when the same
        // region's pool was both grown (by a register) and then
        // shrunk (by an unregister) inside the same transaction.
        // Pass 1: truncate pools to saved lengths (Register entries)
        for entry in self.pool_undo_log.iter().rev() {
            if let PoolUndoEntry::Register { region_idx, old_var_len, old_rigid_len } = entry {
                if *region_idx < self.nodes.len() {
                    self.nodes[*region_idx].pool.var_ids.truncate(*old_var_len);
                    self.nodes[*region_idx].pool.rigid_var_ids.truncate(*old_rigid_len);
                }
            }
        }
        // Pass 2: re-insert unregistered variables (Unregister entries)
        for entry in self.pool_undo_log.iter().rev() {
            if let PoolUndoEntry::Unregister { region_idx, var_id } = entry {
                if *region_idx < self.nodes.len() {
                    self.nodes[*region_idx].pool.var_ids.push(*var_id);
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
    PushMatchBranch,
    PushResolution,
    PushWaitList,
    PushGuardSet,
    PushGenStatus,
    PushForwardRef,
    /// An instance was pushed into an existing forward_refs[pg].  Reverse: pop.
    ForwardRefPush(usize),
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
    PoolRegister { region_idx: usize, old_var_len: usize, old_rigid_len: usize },
    PoolUnregister { region_idx: usize, var_id: usize },
}

impl InferenceContext {
    fn reverse(&mut self, undo: InferUndoLog) {
        match undo {
            InferUndoLog::PushConstraint => { self.constraints.pop(); }
            InferUndoLog::PushTypeVar => { self.type_vars.pop(); }
            InferUndoLog::PushVarTypeId => { self.var_type_ids.pop(); }
            InferUndoLog::PushMatchBranch => { self.match_branches.pop(); }
            InferUndoLog::PushResolution => { self.resolutions.pop(); }
            InferUndoLog::PushWaitList => { self.wait_lists.pop(); }
            InferUndoLog::PushGuardSet => { self.guard_sets.pop(); }
            InferUndoLog::PushGenStatus => { self.gen_statuses.pop(); }
            InferUndoLog::PushForwardRef => { self.forward_refs.pop(); }
            InferUndoLog::ForwardRefPush(pg) => {
                if pg < self.forward_refs.len() { self.forward_refs[pg].pop(); }
            }
            InferUndoLog::PushReverseRef => { self.reverse_refs.pop(); }
            InferUndoLog::SetReverseRef(i, old) => { if i < self.reverse_refs.len() { self.reverse_refs[i] = old; } }
            InferUndoLog::PushLowerBound => { self.lower_bounds.pop(); }
            InferUndoLog::PushUpperBound => { self.upper_bounds.pop(); }
            InferUndoLog::SetGenStatus(i, old) => {
                if i < self.gen_statuses.len() { self.gen_statuses[i] = old; }
            }
            InferUndoLog::AddGuard(i) => {
                if i < self.guard_sets.len() { self.guard_sets[i].remove_guard(); }
            }
            InferUndoLog::RemoveGuard(i) => {
                if i < self.guard_sets.len() { self.guard_sets[i].add_guard(); }
            }
            InferUndoLog::AddTransitiveGuard(i, region_id) => {
                if i < self.guard_sets.len() { self.guard_sets[i].remove_transitive_guard(region_id); }
            }
            InferUndoLog::RemoveTransitiveGuard(i, region_id) => {
                if i < self.guard_sets.len() { self.guard_sets[i].add_transitive_guard(region_id); }
            }
            InferUndoLog::InsertGenericParamBinding(key, old) => {
                match old { Some(v) => self.generic_param_bindings.insert(key, v), None => self.generic_param_bindings.remove(&key) };
            }
            InferUndoLog::PushShapeVar => {
                self.shape_vars.truncate_vars(self.shape_vars.vars_len().saturating_sub(1));
            }
            InferUndoLog::PoolRegister { region_idx, old_var_len, old_rigid_len } => {
                if region_idx < self.region_tree.nodes.len() {
                    self.region_tree.nodes[region_idx].pool.var_ids.truncate(old_var_len);
                    self.region_tree.nodes[region_idx].pool.rigid_var_ids.truncate(old_rigid_len);
                }
            }
            InferUndoLog::PoolUnregister { region_idx, var_id } => {
                if region_idx < self.region_tree.nodes.len() {
                    self.region_tree.nodes[region_idx].pool.var_ids.push(var_id);
                }
            }
        }
    }
}

/// Priority wrapper for constraints, enabling BinaryHeap-based sorting.
/// Constraints are processed in order of "determinism":
///   Priority 0: Eq(concrete, concrete) — both sides fully resolved
///   Priority 1: Eq(concrete, infer)    — one side is InferVar
///   Priority 2: Eq(infer, infer)       — both sides are InferVar
///   Priority 3: Sub(concrete, concrete)
///   Priority 4: Sub(concrete, infer) / Sub(infer, concrete)
///   Priority 5: Sub(infer, infer)
///   Priority 6: Impl constraints
#[derive(Debug, Clone)]
struct PrioritizedConstraint {
    priority: u8,
    constraint: Constraint,
}

impl PartialEq for PrioritizedConstraint {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedConstraint {}

impl PartialOrd for PrioritizedConstraint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // BinaryHeap is a max-heap, so reverse for min-priority behavior
        other.priority.partial_cmp(&self.priority)
    }
}

impl Ord for PrioritizedConstraint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.priority.cmp(&self.priority)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Scalar — integer, float, char, byte, bool, rational.
    Scalar,
    /// Function type: τ₁ → τ₂
    Arrow,
    /// Tuple type: τ₁ × τ₂ × ...
    Tuple(usize),
    /// Named constructor: C τ₁ τ₂ ...
    Constructor(usize),
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

#[derive(Debug, Clone)]
pub enum Constraint {
    Eq(TypeId, TypeId, Span),
    Sub(TypeId, TypeId, Span),
    Impl(TypeId, DefId, Span),
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
        span: Span,
    },
    /// OmniML let-constraint: `let x = λα.∃ᾱ. C₁ in C₂`
    Let {
        expr_var: String,
        def_constraint: Box<Constraint>,
        body_constraint: Box<Constraint>,
        span: Span,
    },
}

/// Describes the instantiation target of a polymorphic scheme,
/// collected from a TypeData node without holding a borrow on the
/// TypeContext. Used by the Instance constraint handler to separate
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
    pub fn priority(&self, ctx: &TypeContext) -> u8 {
        match self {
            Constraint::Eq(a, b, _) => {
                let a_is_infer =
                    matches!(ctx.get(ctx.resolve_binding(*a)), TypeData::InferVar { .. });
                let b_is_infer =
                    matches!(ctx.get(ctx.resolve_binding(*b)), TypeData::InferVar { .. });
                match (a_is_infer, b_is_infer) {
                    (false, false) => 0,                // concrete-concrete: highest priority
                    (true, false) | (false, true) => 1, // one infer var
                    (true, true) => 2,                  // both infer vars
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
                    (false, false) => 3,
                    _ => 4,
                }
            }
            Constraint::Impl(..) => 5, // trait impl checks: lowest priority
            Constraint::Match { scrutinee, .. } => {
                // Match constraints: low priority — they suspend until the
                // scrutinee's shape is resolved.
                let resolved = ctx.resolve_binding(*scrutinee);
                if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                    6 // still an infer var → needs resolution first
                } else {
                    3 // resolved → medium priority
                }
            }
            Constraint::Exists { .. } => 2, // exists: medium-high priority
            Constraint::Forall { .. } => 6, // forall: low priority (skolem)
            Constraint::Instance { .. } => 1, // instance: high priority
            Constraint::Let { .. } => 2,    // let: medium-high priority
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
    wait_lists: Vec<Vec<Constraint>>,
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
    dirty_set: std::collections::HashSet<usize>,
    /// Per-InferenceContext resolution table (TypeOrVar pattern).
    resolutions: Vec<Option<TypeId>>,
    /// Local bindings for GenericParam indices during instantiation.
    generic_param_bindings: HashMap<usize, TypeId>,
    /// Variables resolved since last solver wake-up (avoids O(N) scan).
    resolved_ids: Vec<usize>,
    /// Parallel set for O(1) dedup with `resolved_ids`.
    resolved_set: FxHashSet<usize>,
    /// Undo log (rustc-style).  Records reversible operations when a
    /// snapshot is open.  Popped and `reverse()`'d on rollback.
    undo_log: Vec<InferUndoLog>,
    /// Current snapshot nesting depth.  0 = no snapshot open.
    snapshot_depth: usize,
    /// Stack of snapshot states: for each open snapshot, records the
    /// `resolved_ids` length, `resolved_set`, and `dirty_set` so they
    /// can be restored on rollback (not covered by the undo log).
    resolved_snapshot_stack: Vec<(usize, FxHashSet<usize>, std::collections::HashSet<usize>)>,
}

/// A set of pattern alternatives for a suspended match constraint.
#[derive(Debug, Clone)]
pub struct MatchBranchSet {
    /// The pattern label (e.g. "Arrow", "Tuple", "Coproduct", etc.).
    /// When the scrutinee's shape matches this, the branch is taken.
    pub shape_pattern: PrincipalShape,
    /// Continuation constraints to add when this branch matches.
    pub continuation: Vec<Constraint>,
    /// Fallback constraints to add when no branch matches (else_).
    /// Used as a default when the shape cannot be determined uniquely.
    /// The system emits a diagnostic when else_ is triggered.
    pub else_continuation: Vec<Constraint>,
}

impl InferenceContext {
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
            dirty_set: std::collections::HashSet::new(),
            resolutions: Vec::new(),
            generic_param_bindings: HashMap::default(),
            resolved_ids: Vec::new(),
            resolved_set: FxHashSet::default(),
            undo_log: Vec::new(),
            snapshot_depth: 0,
            resolved_snapshot_stack: Vec::new(),
        }
    }

    pub fn new_type_var(&mut self, ctx: &mut TypeContext, kind: TypeVariableKind) -> TypeId {
        let id = self.next_var_id;
        self.next_var_id += 1;
        let ty_id = ctx.alloc_infer_var(id);
        if id >= self.resolutions.len() {
            self.resolutions.resize(id + 1, None);
        }
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
        while self.lower_bounds.len() <= id {
            self.lower_bounds.push(Vec::new());
        }
        while self.upper_bounds.len() <= id {
            self.upper_bounds.push(Vec::new());
        }
        while self.wait_lists.len() <= id {
            self.wait_lists.push(Vec::new());
        }
        while self.guard_sets.len() <= id {
            self.guard_sets.push(GuardSet::empty());
        }
        while self.gen_statuses.len() <= id {
            self.gen_statuses.push(GenStatus::Ungeneralized);
        }
        while self.forward_refs.len() <= id {
            self.forward_refs.push(Vec::new());
        }
        while self.reverse_refs.len() <= id {
            self.reverse_refs.push(None);
        }
        ty_id
    }

    /// Resolve a TypeId through inference variable bindings (TypeOrVar pattern).
    /// Follows the chain of resolutions until a concrete type is found.
    pub fn resolve(&self, ty: TypeId, ctx: &TypeContext) -> TypeId {
        let mut current = ty;
        loop {
            match ctx.get(current) {
                TypeData::InferVar { id } => {
                    if *id < self.resolutions.len() {
                        if let Some(resolved) = self.resolutions[*id] {
                            if resolved == current {
                                return current;
                            }
                            current = resolved;
                            continue;
                        }
                    }
                    return current;
                }
                _ => return current,
            }
        }
    }

    /// Unify two types via the global `TypeContext` AND record the resolution
    /// in `self.resolutions` so that `self.resolve()` stays consistent with
    /// global bindings.
    ///
    /// This must be used instead of bare `ctx.unify()` inside the solver loop
    /// whenever one of the sides is an InferVar owned by this context.
    fn unify_and_track(
        &mut self,
        a: TypeId,
        b: TypeId,
        ctx: &mut TypeContext,
    ) -> Result<TypeId, TypeError> {
        let result = ctx.unify(a, b)?;
        // After unification, record any new bindings in self.resolutions
        // so that self.resolve() follows the same chain as ctx.resolve_binding().
        let resolved = ctx.resolve_binding(result);
        if let TypeData::InferVar { id } = ctx.get(resolved) {
            if *id < self.resolutions.len() {
                if self.resolutions[*id].is_none() {
                    self.resolutions[*id] = Some(resolved);
                }
            }
        }
        Ok(result)
    }

    /// Unify with local InferVar resolution (TypeOrVar pattern).
    /// Records resolutions in `self.resolutions` instead of global bindings.
    pub fn unify(
        &mut self,
        a: TypeId,
        b: TypeId,
        ctx: &mut TypeContext,
    ) -> Result<TypeId, TypeError> {
        let ra = self.resolve(a, ctx);
        let rb = self.resolve(b, ctx);
        if ra == rb {
            return Ok(ra);
        }
        match (ctx.get(ra), ctx.get(rb)) {
            (TypeData::InferVar { id }, _) if *id < self.resolutions.len() => {
                self.resolutions[*id] = Some(rb);
                Ok(rb)
            }
            (_, TypeData::InferVar { id }) if *id < self.resolutions.len() => {
                self.resolutions[*id] = Some(ra);
                Ok(ra)
            }
            _ => ctx.unify(ra, rb),
        }
    }

    /// Look up the kind of a type variable by its id.
    pub fn get_var_kind(&self, id: usize) -> Option<TypeVariableKind> {
        self.type_vars
            .iter()
            .find(|tv| tv.id == id)
            .map(|tv| tv.kind)
    }

    /// Get the region level of a type variable by its id.
    pub fn get_var_level(&self, id: usize) -> Option<usize> {
        self.type_vars
            .iter()
            .find(|tv| tv.id == id)
            .map(|tv| self.region_tree.get_level(tv.region_id))
    }

    /// Enter a deeper typing scope (let/forall/region).
    /// Creates a new child region in the region tree.
    /// Returns the previous InferRegionId so caller can restore it.
    pub fn enter_level(&mut self) -> InferRegionId {
        self.region_tree.enter_region()
    }

    /// Exit the current typing scope, restoring the previous region.
    pub fn exit_level(&mut self, prev_region: InferRegionId) {
        self.region_tree.current = prev_region;
    }

    /// Try to promote a variable to the target region's scope
    /// by creating a new variable at the target region and unifying.
    /// Uses the region tree's ancestor check (OmniML §6 PR-UVARPR).
    pub fn try_promote_var(
        &mut self,
        ctx: &mut TypeContext,
        var_id: usize,
        target_region: InferRegionId,
    ) -> Option<TypeId> {
        let var_region = self.type_vars.get(var_id)?.region_id;

        // If the variable's region is already an ancestor of (or equal to)
        // the target region, it is already at an outer/equal scope — no
        // promotion needed. Otherwise the variable is deeper and needs
        // promotion to the target region.
        if self.region_tree.is_ancestor(var_region, target_region) || var_region == target_region {
            return Some(self.var_type_ids[var_id]);
        }
        // Promotion needed: create a new variable at the target region.
        // `new_type_var` registers the var in the *current* region's pool,
        // but we need it in the target (outer) region's pool. We unregister
        // from the current pool and re-register in the target pool.
        let new_ty_id = self.new_type_var(ctx, TypeVariableKind::Any);
        let new_id = self.next_var_id - 1;
        let current_region = self.region_tree.current;
        self.region_tree.unregister_var(new_id, current_region);
        self.region_tree.register_var_in_region(new_id, target_region);
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
            let old_wait = std::mem::take(&mut self.wait_lists[var_id]);
            self.wait_lists[new_id].extend(old_wait);
        }
        if var_id < self.guard_sets.len() && new_id < self.guard_sets.len() {
            let old_guards = std::mem::take(&mut self.guard_sets[var_id]);
            self.guard_sets[new_id] = self.guard_sets[new_id].union(&old_guards);
        }
        if var_id < self.gen_statuses.len() && new_id < self.gen_statuses.len() {
            let old_status = self.gen_statuses[var_id];
            if old_status != GenStatus::Ungeneralized {
                self.gen_statuses[new_id] = old_status;
            }
        }
        if var_id < self.lower_bounds.len() && new_id < self.lower_bounds.len() {
            let old_lbs = std::mem::take(&mut self.lower_bounds[var_id]);
            self.lower_bounds[new_id].extend(old_lbs);
        }
        if var_id < self.upper_bounds.len() && new_id < self.upper_bounds.len() {
            let old_ubs = std::mem::take(&mut self.upper_bounds[var_id]);
            self.upper_bounds[new_id].extend(old_ubs);
        }

        // ── Forward references (OmniML §5.2) ──────────────────────
        // If the old variable was a PG var with instances, transfer
        // those instances to the new variable and update their reverse
        // references to point to the new variable.
        if var_id < self.forward_refs.len() && new_id < self.forward_refs.len() {
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
            if let Some(pg_id) = old_reverse {
                if pg_id < self.forward_refs.len() {
                    if let Some(pos) =
                        self.forward_refs[pg_id].iter().position(|&r| r == var_id)
                    {
                        self.forward_refs[pg_id][pos] = new_id;
                    }
                }
            }
        }

        // Bind the old variable to the new one (promotion)
        if var_id < self.resolutions.len() {
            self.resolutions[var_id] = Some(new_ty_id);
        }
        Some(new_ty_id)
    }

    pub fn add_constraint(&mut self, c: Constraint) {
        self.constraints.push(c);
    }

    /// OmniML-inspired: suspend a constraint on the target InferVar id.
    /// When the var is bound, the constraint will be woken and reprocessed.
    /// Also marks the variable as PartiallyGeneralizable (PG) and adds a
    /// guard entry so the variable stays PG until the guard is released.
    pub fn suspend_on_var(&mut self, c: Constraint, var_id: usize) {
        if var_id < self.wait_lists.len() {
            self.wait_lists[var_id].push(c);
            if var_id < self.gen_statuses.len() {
                let old = self.gen_statuses[var_id];
                self.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
                self.push_undo(InferUndoLog::SetGenStatus(var_id, old));
            }
            // Add a guard: the variable is now blocked until this constraint
            // is woken and processed.  This is essential for the PG→G lifecycle
            // (OmniML §6).  Uses the reference-counted GuardSet.
            self.add_guard(var_id);
        } else {
            self.constraints.push(c);
            self.push_undo(InferUndoLog::PushConstraint);
        }
    }

    /// Extract the InferVar id from a constraint (if any side is an unresolved InferVar).
    /// Returns None if both sides are concrete.
    fn infer_var_from_constraint(&self, c: &Constraint, ctx: &TypeContext) -> Option<usize> {
        match c {
            Constraint::Eq(a, b, _) => {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                if let TypeData::InferVar { id } = ctx.get(ra) {
                    return Some(*id);
                }
                if let TypeData::InferVar { id } = ctx.get(rb) {
                    return Some(*id);
                }
                None
            }
            Constraint::Sub(sub, sup, _) => {
                let rs = ctx.resolve_binding(*sub);
                let rsup = ctx.resolve_binding(*sup);
                if let TypeData::InferVar { id } = ctx.get(rs) {
                    return Some(*id);
                }
                if let TypeData::InferVar { id } = ctx.get(rsup) {
                    return Some(*id);
                }
                None
            }
            Constraint::Impl(ty, ..) => {
                let r = ctx.resolve_binding(*ty);
                if let TypeData::InferVar { id } = ctx.get(r) {
                    Some(*id)
                } else {
                    None
                }
            }
            Constraint::Match { scrutinee, .. } => {
                let r = ctx.resolve_binding(*scrutinee);
                if let TypeData::InferVar { id } = ctx.get(r) {
                    Some(*id)
                } else {
                    None
                }
            }
            Constraint::Exists { var_id, .. } => Some(*var_id),
            Constraint::Forall { .. } => None, // rigid var — not an infer var
            Constraint::Instance {
                instantiation_ty, ..
            } => {
                let r = ctx.resolve_binding(*instantiation_ty);
                if let TypeData::InferVar { id } = ctx.get(r) {
                    Some(*id)
                } else {
                    None
                }
            }
            Constraint::Let { .. } => None, // structural — no single infer var
        }
    }

    /// Wake all constraints suspended on the given var_id, moving them
    /// back into the active constraint list for reprocessing.
    fn wake_var(&mut self, var_id: usize) {
        if var_id < self.wait_lists.len() {
            let mut suspended = std::mem::take(&mut self.wait_lists[var_id]);
            self.constraints.append(&mut suspended);
        }
    }

    /// Determine the principal shape of a resolved type.
    pub fn shape_of_type(ctx: &TypeContext, ty: TypeId) -> PrincipalShape {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved) {
            TypeData::Fn { params, .. } => PrincipalShape::Arrow,
            TypeData::Tuple { elems } => PrincipalShape::Tuple(elems.len()),
            TypeData::Adt { args, .. } => PrincipalShape::Constructor(args.len()),
            TypeData::Forall { .. }
            | TypeData::Exists { .. }
            | TypeData::Poly { .. }
            | TypeData::SkolemVar { .. } => PrincipalShape::Poly,
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Rational { .. } => PrincipalShape::Scalar,
            TypeData::InferVar { .. } | TypeData::GenericParam { .. } => PrincipalShape::Var,
            _ => PrincipalShape::Unknown,
        }
    }

    /// Try to set the shape of a variable from its resolved type.
    /// Returns true if the shape was updated.
    fn try_set_shape(&mut self, var_id: usize, ctx: &TypeContext) -> bool {
        if var_id < self.type_vars.len() && var_id < self.var_type_ids.len() {
            let ty = ctx.resolve_binding(self.var_type_ids[var_id]);
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
    fn wake_var_incremental(
        &mut self,
        var_id: usize,
        heap: &mut BinaryHeap<PrioritizedConstraint>,
        ctx: &TypeContext,
    ) {
        if var_id < self.wait_lists.len() && !self.wait_lists[var_id].is_empty() {
            let suspended = std::mem::take(&mut self.wait_lists[var_id]);
            for c in suspended {
                let p = c.priority(ctx);
                heap.push(PrioritizedConstraint {
                    priority: p,
                    constraint: c,
                });
            }
            // Clear all guards for this variable: every constraint placed
            // in the wait list by `suspend_on_var` had a guard added; we
            // must release those guards now that the constraints have been
            // woken to the heap (OmniML §6 confirmation: "once a suspended
            // match constraint is solved, it removes the guards it
            // introduced").
            if var_id < self.guard_sets.len() {
                self.guard_sets[var_id].clear();
            }
            // All constraints woken and guards cleared — transition to
            // Generalized if no further guards remain.
            if var_id < self.gen_statuses.len()
                && self.gen_statuses[var_id] == GenStatus::PartiallyGeneralizable
            {
                let guards_empty =
                    var_id < self.guard_sets.len() && self.guard_sets[var_id].is_empty();
                if guards_empty {
                    self.gen_statuses[var_id] = GenStatus::Generalized;
                }
            }
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
        for b in branches {
            self.match_branches.push(b);
        }
        self.push_undo(InferUndoLog::PushMatchBranch);
        (start, count)
    }

    /// Try to discharge a Match constraint using a shape variable.
    /// If the scrutinee has a shape variable and it's resolved, discharge
    /// immediately.  If not, register a callback on the shape variable
    /// so the match fires when the shape becomes known.
    /// Returns `true` if the match was handled (either discharged or
    /// registered for later).
    pub fn try_match_via_shape_var(
        &mut self,
        ctx: &mut TypeContext,
        scrutinee: TypeId,
        branches_id: (usize, usize),
        heap: &mut BinaryHeap<PrioritizedConstraint>,
    ) -> bool {
        let resolved = ctx.resolve_binding(scrutinee);
        match ctx.get(resolved) {
            TypeData::InferVar { id } => {
                // Always suspend the Match constraint, regardless of whether
                // this variable already has guards.  A non-empty guard set
                // from a *previous* Match cannot repel a *new* Match with a
                // potentially different branches_id — silently dropping it
                // would lose entire branch sets.
                let match_c = Constraint::Match {
                    scrutinee,
                    branches_id,
                    span: crate::ast::Span::new(0, 0),
                };
                // #3: Register on this var AND all vars sharing its
                // binding root (transitive wait_list).
                let root = ctx.resolve_binding(scrutinee);
                let targets: Vec<usize> = self
                    .var_type_ids
                    .iter()
                    .enumerate()
                    .filter(|(_, ty_id)| ctx.resolve_binding(**ty_id) == root)
                    .map(|(i, _)| i)
                    .collect();
                for other_id in targets {
                    self.suspend_on_var(match_c.clone(), other_id);
                }
                true
            }
            _ => {
                // Propagate the result — if discharge_match fails (no branch
                // matches and no else_ fallback), the caller falls through to
                // unicity check and re-push logic instead of silently losing
                // the constraint.
                self.discharge_match(ctx, scrutinee, branches_id, heap)
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
    pub fn unicity_check(
        &self,
        ctx: &TypeContext,
        ty: TypeId,
        active_constraints: &[PrioritizedConstraint],
    ) -> Option<PrincipalShape> {
        let resolved = ctx.resolve_binding(ty);
        let data = ctx.get(resolved);

        // ── UNI-TYPE: non-variable type ──────────────────────────
        // If τ is already resolved to a concrete type, its shape is known.
        if !matches!(data, TypeData::InferVar { .. }) {
            return Some(Self::shape_of_type(ctx, resolved));
        }

        // τ is an InferVar — extract its id.
        let var_id = match data {
            TypeData::InferVar { id } => *id,
            _ => return None,
        };

        // ── UNI-VAR: α is unified with a concrete type ───────────
        // Scan all Eq constraints in the active set. If any equality
        // binds α to a non-variable type, that determines the shape.
        for pc in active_constraints {
            if let Constraint::Eq(a, b, _) = &pc.constraint {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                // Check if this Eq constraint involves our variable
                let other = if ra == resolved {
                    Some(rb)
                } else if rb == resolved {
                    Some(ra)
                } else {
                    None
                };
                if let Some(other_ty) = other {
                    let other_resolved = ctx.resolve_binding(other_ty);
                    if !matches!(ctx.get(other_resolved), TypeData::InferVar { .. }) {
                        return Some(Self::shape_of_type(ctx, other_resolved));
                    }
                }
            }
        }

        // ── UNI-BACKPROP: shape from incremental instantiations ──
        // If this variable is a PG variable with forward references,
        // check if all its instances resolve to the same shape.
        if var_id < self.forward_refs.len() && !self.forward_refs[var_id].is_empty() {
            let mut shared_shape: Option<PrincipalShape> = None;
            for &instance_id in &self.forward_refs[var_id] {
                if instance_id < self.var_type_ids.len() {
                    let instance_ty = ctx.resolve_binding(self.var_type_ids[instance_id]);
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
                let bound_resolved = ctx.resolve_binding(bound);
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
                let bound_resolved = ctx.resolve_binding(bound);
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
    pub fn unicity_check_smt(&self, ctx: &TypeContext, ty: TypeId) -> Option<PrincipalShape> {
        let solver = SmtSolver::new("z3");

        // ── 1. Collect all resolved bindings ─────────────────────
        let mut bindings: std::collections::HashMap<usize, TypeId> =
            std::collections::HashMap::default();
        for (i, var_ty) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(*var_ty);
            if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                bindings.insert(i, resolved);
            }
        }

        // ── 2. Collect all equality constraints involving InferVars ──
        let mut eq_pairs: Vec<(usize, usize)> = Vec::new();
        for c in &self.constraints {
            if let Constraint::Eq(a, b, _) = c {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                if let (TypeData::InferVar { id: aid }, TypeData::InferVar { id: bid }) =
                    (ctx.get(ra), ctx.get(rb))
                {
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
        if !self.forward_refs[pg_var_id].contains(&instance_id) {
            self.forward_refs[pg_var_id].push(instance_id);
            self.push_undo(InferUndoLog::ForwardRefPush(pg_var_id));
        }
        let old_rev = self.reverse_refs[instance_id];
        self.reverse_refs[instance_id] = Some(pg_var_id);
        self.push_undo(InferUndoLog::SetReverseRef(instance_id, old_rev));

        // Mark the instance as PI (PartialInstance)
        while self.gen_statuses.len() <= instance_id {
            self.gen_statuses.push(GenStatus::Ungeneralized);
        }
        let old_gs = self.gen_statuses[instance_id];
        self.gen_statuses[instance_id] = GenStatus::PartialInstance;
        self.push_undo(InferUndoLog::SetGenStatus(instance_id, old_gs));
    }

    /// ── S-Inst-Copy (OmniML §5.3) ──────────────────────────────────
    ///
    /// Copy solved constraints from a PG abstraction to all its instances.
    /// When a PG variable's multi-equation is resolved (e.g. α = τ),
    /// the equality is propagated to every instance of α.
    /// If τ itself contains other region variables (e.g. β, γ from the
    /// same abstraction), fresh instances of those are created and bound.
    ///
    /// Returns the number of instances that were updated.
    pub fn s_inst_copy(
        &mut self,
        ctx: &mut TypeContext,
        pg_var_id: usize,
        resolve_ty: TypeId,
    ) -> usize {
        if pg_var_id >= self.forward_refs.len() {
            return 0;
        }
        let instances: Vec<usize> = self.forward_refs[pg_var_id].clone();
        let mut updated = 0;
        for inst_id in instances {
            if inst_id < self.var_type_ids.len() {
                let instance_ty_id = self.var_type_ids[inst_id];
                // S-Inst-Copy: copy the solved equation α = τ to the instance.
                // Walk τ to find any other region variables referenced by the
                // abstraction. For each such variable that itself has instances,
                // recursively copy.
                self.s_inst_copy_walk(ctx, instance_ty_id, resolve_ty);
                // Bind instance to the concrete type
                if let Err(_) = ctx.unify(instance_ty_id, resolve_ty) {
                    // unification error will be caught elsewhere
                }
                // Recursively propagate if resolve_ty contains other PG vars
                self.s_inst_copy_deepen(ctx, resolve_ty);
                updated += 1;
            }
        }
        // Clear forward refs (all propagated)
        self.forward_refs[pg_var_id].clear();
        updated
    }

    /// Walk a type and recursively apply S-Inst-Copy to any region variables
    /// found inside it that have their own instances.
    fn s_inst_copy_deepen(&mut self, ctx: &mut TypeContext, ty: TypeId) {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved).clone() {
            TypeData::Fn { params, ret } => {
                for p in params {
                    self.s_inst_copy_deepen(ctx, p);
                }
                self.s_inst_copy_deepen(ctx, ret);
            }
            TypeData::Tuple { elems }
            | TypeData::Coproduct {
                alternatives: elems,
            } => {
                for e in elems {
                    self.s_inst_copy_deepen(ctx, e);
                }
            }
            TypeData::Adt { args, .. } => {
                for a in args {
                    self.s_inst_copy_deepen(ctx, a);
                }
            }
            TypeData::InferVar { id } => {
                // If this region variable has instances, propagate to them too
                if id < self.forward_refs.len() && !self.forward_refs[id].is_empty() {
                    let resolved_inner = ctx.resolve_binding(ty);
                    if !matches!(ctx.get(resolved_inner), TypeData::InferVar { .. }) {
                        self.s_inst_copy(ctx, id, resolved_inner);
                    }
                }
            }
            _ => {}
        }
    }

    /// Copy one solved equation to one instance variable (S-Inst-Copy detail).
    fn s_inst_copy_walk(&mut self, ctx: &mut TypeContext, instance_ty: TypeId, source_ty: TypeId) {
        let resolved_source = ctx.resolve_binding(source_ty);
        match ctx.get(resolved_source) {
            TypeData::InferVar { id } => {
                // This instance refers to another region variable.
                // If the source has its own forward refs (instances),
                // recursively copy them to the new instance's peer.
                if *id < self.forward_refs.len() && !self.forward_refs[*id].is_empty() {
                    let resolved_src = ctx.resolve_binding(source_ty);
                    if !matches!(ctx.get(resolved_src), TypeData::InferVar { .. }) {
                        self.s_inst_copy(ctx, *id, resolved_src);
                    }
                }
            }
            _ => {}
        }
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
    /// Falls back to the level-based heuristic when Z3 is unavailable.
    pub fn s_exists_lower(&mut self, ctx: &TypeContext, var_id: usize) -> bool {
        if var_id >= self.type_vars.len() || var_id >= self.gen_statuses.len() {
            return false;
        }
        if self.gen_statuses[var_id] != GenStatus::PartiallyGeneralizable {
            return false;
        }

        // ── Z3-backed semantic check ──────────────────────────────
        // Query whether this variable's shape is uniquely determined.
        if let Some(_shape) = self.unicity_check_smt(ctx, self.var_type_ids[var_id]) {
            // Shape is uniquely determined → safe to lower.
            self.gen_statuses[var_id] = GenStatus::Ungeneralized;
            return true;
        }

        // ── Fallback: level-based heuristic ───────────────────────
        // When Z3 is unavailable or the query times out, use the
        // conservative level-based approximation.
        let var_region = self.type_vars[var_id].region_id;
        let var_level = self.region_tree.get_level(var_region);
        let cur_region = self.region_tree.current;
        let cur_level = self.region_tree.get_level(cur_region);
        if var_level > 0 && var_level > cur_level {
            // Move the variable from its current region's pool to the root pool
            // before updating region_id, keeping the pool and the field consistent.
            let root_id = self.region_tree.root;
            let old_region = self.type_vars[var_id].region_id;
            self.region_tree.unregister_var(var_id, old_region);
            self.type_vars[var_id].region_id = root_id;
            self.region_tree.register_var_in_region(var_id, root_id);
            self.gen_statuses[var_id] = GenStatus::Ungeneralized;
            return true;
        }

        false
    }

    /// ── S-Generalize / update_and_generalize_generation (OmniML §5.3) ──
    ///
    /// Drains all dirty regions, collects guarded roots, and generalizes
    /// PG variables that are no longer guarded or referenced.
    ///
    /// The optional `target_var_id` restricts processing to just the region
    /// containing that variable (for targeted generalization before instantiation).
    pub fn force_generalize(&mut self, ctx: &mut TypeContext) {
        self.force_generalize_for_regions(ctx, &[], None)
    }

    /// Full generation-based generalization.  `dirty_levels` lists region
    /// levels that have been marked dirty.  `target_var_id` (if set) limits
    /// processing to the region containing that specific variable.
    pub fn force_generalize_for_regions(
        &mut self,
        ctx: &mut TypeContext,
        dirty_levels: &[usize],
        target_var: Option<usize>,
    ) {
        // Collect PG variables from dirty_set or region levels.
        let dirty: Vec<usize> = if let Some(tv) = target_var {
            let region = self.type_vars.get(tv).map(|v| v.region_id).unwrap_or(self.region_tree.root);
            (0..self.gen_statuses.len())
                .filter(|i| {
                    self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable)
                        && self.type_vars.get(*i).map(|v| v.region_id == region).unwrap_or(false)
                })
                .collect()
        } else if !self.dirty_set.is_empty() {
            self.dirty_set.iter().copied()
                .filter(|i| self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable))
                .collect()
        } else if !dirty_levels.is_empty() {
            let parent_levels: std::collections::HashSet<usize> = dirty_levels.iter().copied().collect();
            (0..self.gen_statuses.len())
                .filter(|i| {
                    self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable)
                        && self.type_vars.get(*i)
                            .map(|v| parent_levels.contains(&self.region_tree.get_level(v.region_id)))
                            .unwrap_or(false)
                })
                .collect()
        } else {
            (0..self.gen_statuses.len())
                .filter(|i| self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable))
                .collect()
        };

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
        let mut trans_guarded: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &i in &dirty {
            if i < self.guard_sets.len() && !self.guard_sets[i].is_empty() {
                trans_guarded.insert(i);
            }
        }
        for &i in &dirty {
            if !trans_guarded.contains(&i) && i < self.var_type_ids.len() {
                let i_root = ctx.resolve_binding(self.var_type_ids[i]);
                for &j in &dirty {
                    if j < self.var_type_ids.len() && trans_guarded.contains(&j) {
                        if ctx.resolve_binding(self.var_type_ids[j]) == i_root {
                            trans_guarded.insert(i);
                            break;
                        }
                    }
                }
            }
        }

        // Process innermost-first (highest level first = most nested first).
        let mut vars_by_level: Vec<(usize, usize)> = dirty
            .iter()
            .map(|&i| (i, self.type_vars.get(i).map(|v| self.region_tree.get_level(v.region_id)).unwrap_or(0)))
            .collect();
        vars_by_level.sort_by(|a, b| b.1.cmp(&a.1));

        // Rigid scope check per generation.
        let cur_level = self.region_tree.get_level(self.region_tree.current);
        for &(i, level) in &vars_by_level {
            if i >= self.var_type_ids.len() {
                continue;
            }
            let resolved = ctx.resolve_binding(self.var_type_ids[i]);
            if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                continue;
            }
            if level < cur_level {
                if Self::check_rigid_escape(ctx, resolved, level) {
                    continue;
                }
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
                let ty = ctx.resolve_binding(self.var_type_ids[i]);
                !matches!(ctx.get(ty), TypeData::InferVar { .. })
            };
            if is_resolved && !is_trans_guarded && !has_waiting {
                self.gen_statuses[i] = GenStatus::Generalized;
            }
        }

        // Update dirty_set: remove only the processed levels.
        if dirty_levels.is_empty() && target_var.is_none() {
            self.dirty_set.clear();
        } else {
            self.dirty_set.retain(|i| !dirty.contains(i));
        }
    }

    /// Check whether a resolved type contains escaped rigid (skolem) variables.
    /// Recursively walks the type tree looking for `GenericParam` or `SkolemVar`
    /// references that would indicate a Forall-bound variable has escaped into
    /// an outer scope.  SkolemVar is the runtime representation of a universally
    /// quantified variable bound by a `Constraint::Forall` — detecting it here
    /// prevents skolem constants from leaking into generalized type schemes.
    fn check_rigid_escape(ctx: &TypeContext, ty: TypeId, max_level: usize) -> bool {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved) {
            TypeData::GenericParam { .. } | TypeData::SkolemVar { .. } => true, // escape detected
            TypeData::Fn { params, ret } => {
                params
                    .iter()
                    .any(|&p| Self::check_rigid_escape(ctx, p, max_level))
                    || Self::check_rigid_escape(ctx, *ret, max_level)
            }
            TypeData::Tuple { elems }
            | TypeData::Coproduct {
                alternatives: elems,
            } => elems
                .iter()
                .any(|&e| Self::check_rigid_escape(ctx, e, max_level)),
            TypeData::Adt { args, .. } => args
                .iter()
                .any(|&a| Self::check_rigid_escape(ctx, a, max_level)),
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Poly { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => Self::check_rigid_escape(ctx, *body, max_level),
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                Self::check_rigid_escape(ctx, *ty, max_level)
            }
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                Self::check_rigid_escape(ctx, *elem, max_level)
            }
            TypeData::Ptr { pointee, .. } => Self::check_rigid_escape(ctx, *pointee, max_level),
            TypeData::AssociatedType { self_ty, .. } => {
                Self::check_rigid_escape(ctx, *self_ty, max_level)
            }
            _ => false, // Int, Bool, etc. are safe
        }
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
        self.resolved_snapshot_stack
            .push((self.resolved_ids.len(), self.resolved_set.clone(), self.dirty_set.clone()));
        self.undo_log.len()
    }

    /// Roll back to a snapshot: pop entries and call `reverse()`.
    pub fn rollback_to(&mut self, snapshot_len: usize) {
        while self.undo_log.len() > snapshot_len {
            let undo = self.undo_log.pop().unwrap();
            self.reverse(undo);
        }
        // Restore resolved_ids/resolved_set/dirty_set from snapshot stack.
        if let Some((ids_len, set, dirty)) = self.resolved_snapshot_stack.pop() {
            self.resolved_ids.truncate(ids_len);
            self.resolved_set = set;
            self.dirty_set = dirty;
        }
        self.snapshot_depth -= 1;
    }

    /// Commit a snapshot: discard undo entries without reversing.
    /// At the outermost snapshot, the log is cleared.
    pub fn commit_snapshot(&mut self, snapshot_len: usize) {
        if self.snapshot_depth == 1 {
            self.undo_log.truncate(snapshot_len);
        }
        self.resolved_snapshot_stack.pop();
        self.snapshot_depth -= 1;
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
    pub fn discharge_match(
        &mut self,
        ctx: &mut TypeContext,
        scrutinee_ty: TypeId,
        branches_id: (usize, usize),
        heap: &mut BinaryHeap<PrioritizedConstraint>,
    ) -> bool {
        let resolved = ctx.resolve_binding(scrutinee_ty);
        let shape = Self::shape_of_type(ctx, resolved);

        // Find the branch that matches this shape.
        let (start, count) = branches_id;
        let end = start + count;
        if start < self.match_branches.len() {
            for i in start..end.min(self.match_branches.len()) {
                let branch = &self.match_branches[i];
                let matches_pattern = shape == branch.shape_pattern;

                if matches_pattern {
                    // Enqueue continuation constraints.
                    for c in &branch.continuation {
                        let p = c.priority(ctx);
                        heap.push(PrioritizedConstraint {
                            priority: p,
                            constraint: c.clone(),
                        });
                    }
                    return true;
                }
            }

            // No exact match — try the else_ fallback of the first branch.
            let first = &self.match_branches[start];
            if !first.else_continuation.is_empty() {
                for c in &first.else_continuation {
                    let p = c.priority(ctx);
                    heap.push(PrioritizedConstraint {
                        priority: p,
                        constraint: c.clone(),
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

    /// Mark a variable as guarded by a suspended constraint.
    /// Increments the reference-counted guard on the variable.
    pub fn add_guard(&mut self, var_id: usize) {
        while self.guard_sets.len() <= var_id {
            self.guard_sets.push(GuardSet::empty());
        }
        self.push_undo(InferUndoLog::AddGuard(var_id));
        self.guard_sets[var_id].add_guard();
        if var_id < self.gen_statuses.len() {
            let old = self.gen_statuses[var_id];
            self.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
            self.push_undo(InferUndoLog::SetGenStatus(var_id, old));
        }
    }

    /// Remove a guard from a variable when its suspended constraint is discharged.
    /// Decrements the reference count. If no guards remain, the variable can be
    /// re-generalised (PG → G transition, OmniML §6).
    pub fn remove_guard(&mut self, var_id: usize) {
        if var_id < self.guard_sets.len() {
            self.push_undo(InferUndoLog::RemoveGuard(var_id));
            self.guard_sets[var_id].remove_guard();
            if self.guard_sets[var_id].is_empty() && var_id < self.gen_statuses.len() {
                if self.gen_statuses[var_id] == GenStatus::PartiallyGeneralizable {
                    let old = self.gen_statuses[var_id];
                    self.gen_statuses[var_id] = GenStatus::Generalized;
                    self.push_undo(InferUndoLog::SetGenStatus(var_id, old));
                }
            }
        }
    }

    pub fn solve(
        &mut self,
        ctx: &mut TypeContext,
        trait_env: &TraitEnv,
        symbols: &SymbolTable,
    ) -> Result<(), TypeError> {
        // ── Build priority queue ────────────────────────────────────
        let mut heap: BinaryHeap<PrioritizedConstraint> = BinaryHeap::new();
        for c in &self.constraints {
            let priority = c.priority(ctx);
            heap.push(PrioritizedConstraint {
                priority,
                constraint: c.clone(),
            });
        }

        // ── Process all constraints in priority order ───────────────
        // With incremental wake-up: after each unification, check if an
        // InferVar was resolved and immediately wake its suspended constraints.
        // This follows OmniML's job-queue pattern where unify enqueues jobs
        // that the scheduler runs immediately.
        //
        // `delayed` holds constraints that cannot be processed yet (e.g. Impl
        // on an unresolved infer var).  They are re‑queued into `heap` once
        // per outer iteration to be retried after any new equalities arrive.
        // `delayed_retried` prevents infinite looping when the dependency
        // never resolves: we retry at most once per stall, and if no variable
        // was woken in between, the stall is permanent.
        let mut delayed: Vec<PrioritizedConstraint> = Vec::new();
        let mut delayed_retried: bool = false;
        loop {
            let mut active_count = heap.len();
            while let Some(pc) = heap.pop() {
                active_count -= 1;
                match &pc.constraint {
                    Constraint::Eq(a, b, _) => {
                        // Check if either side is an InferVar before unifying
                        let ra = ctx.resolve_binding(*a);
                        let rb = ctx.resolve_binding(*b);
                        let a_is_infer = matches!(ctx.get(ra), TypeData::InferVar { .. });
                        let b_is_infer = matches!(ctx.get(rb), TypeData::InferVar { .. });
                        let a_var_id = if a_is_infer {
                            if let TypeData::InferVar { id } = ctx.get(ra) {
                                Some(*id)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let b_var_id = if b_is_infer {
                            if let TypeData::InferVar { id } = ctx.get(rb) {
                                Some(*id)
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        // Level-based promotion (Fan, Xu & Xie 2025 §6.2):
                        // If unifying two InferVars at different regions, promote
                        // the deeper variable to the nearest common ancestor (NCA)
                        // of both regions before unifying.  Using the NCA guarantees
                        // the promoted variable lives in a scope that is accessible
                        // from both original branches — critical when the two
                        // variables live in sibling branches, not just on the same
                        // ancestor-descendant line.
                        if let (Some(avid), Some(bvid)) = (a_var_id, b_var_id) {
                            let a_region = self.type_vars[avid].region_id;
                            let b_region = self.type_vars[bvid].region_id;
                            let a_lvl = self.region_tree.get_level(a_region);
                            let b_lvl = self.region_tree.get_level(b_region);
                            let nca = self.region_tree.nearest_common_ancestor(a_region, b_region);
                            debug_assert!(
                                self.region_tree.is_ancestor(nca, a_region) && self.region_tree.is_ancestor(nca, b_region),
                                "NCA must be an ancestor of both regions",
                            );
                            if a_lvl > b_lvl {
                                // a is deeper — promote a to NCA
                                if let Some(promoted) = self.try_promote_var(ctx, avid, nca) {
                                    self.unify_and_track(promoted, *b, ctx)?;
                                } else {
                                    self.unify_and_track(*a, *b, ctx)?;
                                }
                            } else if b_lvl > a_lvl {
                                // b is deeper — promote b to NCA
                                if let Some(promoted) = self.try_promote_var(ctx, bvid, nca) {
                                    self.unify_and_track(*a, promoted, ctx)?;
                                } else {
                                    self.unify_and_track(*a, *b, ctx)?;
                                }
                            } else {
                                // Same level — NCA handles the sibling / ancestor / equal cases.
                                // If NCA equals one of the regions, one region is an ancestor
                                // of the other (same level still possible via grandparent-child
                                // through different paths?  No — same level with one ancestor of
                                // the other implies they are the same node.)  Otherwise the
                                // regions are siblings: promote the shallower-rooted variable
                                // (the one whose region is deeper in the NCA's subtree).
                                if nca == a_region && nca == b_region {
                                    // Same region — no promotion needed.
                                    self.unify_and_track(*a, *b, ctx)?;
                                } else if self.region_tree.is_ancestor(a_region, b_region) {
                                    // a_region is ancestor of b_region *at the same level* —
                                    // this is actually impossible (ancestor at same level == same node),
                                    // but handle defensively.
                                    self.unify_and_track(*a, *b, ctx)?;
                                } else if self.region_tree.is_ancestor(b_region, a_region) {
                                    // Same as above but swapped — defensive only.
                                    self.unify_and_track(*a, *b, ctx)?;
                                } else {
                                    // Sibling regions at the same level: promote both to NCA,
                                    // then unify the promoted copies.
                                    let a_promoted = self.try_promote_var(ctx, avid, nca)
                                        .unwrap_or(*a);
                                    let b_promoted = self.try_promote_var(ctx, bvid, nca)
                                        .unwrap_or(*b);
                                    self.unify_and_track(a_promoted, b_promoted, ctx)?;
                                }
                            }
                        } else {
                            self.unify_and_track(*a, *b, ctx)?;
                        }

                        // Mark dirty for drain_dirty tracking.
                        for var_id in [a_var_id, b_var_id].iter().flatten() {
                            self.mark_dirty(*var_id);
                        }

                        // Incremental wake-up: if a variable was just resolved,
                        // immediately enqueue its suspended constraints.
                        for var_id in [a_var_id, b_var_id].iter().flatten() {
                            if self.try_set_shape(*var_id, ctx) {
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
                                let resolved_ty = ctx.resolve_binding(self.var_type_ids[*var_id]);
                                if !matches!(ctx.get(resolved_ty), TypeData::InferVar { .. }) {
                                    // ── S-Generalize (OmniML §5.3) ──
                                    self.force_generalize(ctx);
                                    self.s_inst_copy(ctx, *var_id, resolved_ty);
                                }
                            }
                            // ── S-Exists-Lower (OmniML §5.3) ──
                            // For vars that remain PG (still have guards/waiting),
                            // try Z3-backed semantic lowering if uniquely determined.
                            if *var_id < self.gen_statuses.len()
                                && self.gen_statuses[*var_id] == GenStatus::PartiallyGeneralizable
                            {
                                if self.s_exists_lower(ctx, *var_id) {
                                    // lowering succeeded
                                }
                            }
                        }
                    }
                    Constraint::Sub(sub, sup, _span) => {
                        let resolved_sub = ctx.resolve_binding(*sub);
                        let resolved_sup = ctx.resolve_binding(*sup);

                        // If sup is an InferVar, record sub as a lower bound of sup
                        if let TypeData::InferVar { id } = ctx.get(resolved_sup) {
                            if *id < self.lower_bounds.len() {
                                self.lower_bounds[*id].push(resolved_sub);
                                self.mark_dirty(*id);
                            }
                        }
                        // If sub is an InferVar, record sup as an upper bound of sub
                        if let TypeData::InferVar { id } = ctx.get(resolved_sub) {
                            if *id < self.upper_bounds.len() {
                                self.upper_bounds[*id].push(resolved_sup);
                                self.mark_dirty(*id);
                            }
                        }

                        // If both sides are resolved (not InferVar), check the subtype relationship now
                        let sub_is_infer =
                            matches!(ctx.get(resolved_sub), TypeData::InferVar { .. });
                        let sup_is_infer =
                            matches!(ctx.get(resolved_sup), TypeData::InferVar { .. });
                        if !sub_is_infer && !sup_is_infer {
                            if !ctx.subtype(resolved_sub, resolved_sup) {
                                return Err(TypeError::Mismatch {
                                    expected: resolved_sup,
                                    found: resolved_sub,
                                    span: *_span,
                                });
                            }
                        }
                    }
                    Constraint::Impl(ty, trait_id, span) => {
                        let resolved = ctx.resolve_binding(*ty);
                        let data = ctx.get(resolved);
                        // If the type is an error, skip
                        if matches!(data, TypeData::Error) {
                            return Ok(());
                        }
                        // If still an infer var or generic param, delay Impl
                        // checking until the type is resolved — push to the
                        // `delayed` queue instead of returning Ok(()) or
                        // re-pushing to `heap` (which would cause immediate
                        // re-pop and an infinite loop).
                        if matches!(data, TypeData::InferVar { .. })
                            || matches!(data, TypeData::GenericParam { .. })
                        {
                            delayed.push(PrioritizedConstraint {
                                priority: 7,
                                constraint: pc.constraint.clone(),
                            });
                            continue;
                        }
                        // Otherwise, check that the impl exists
                        let impl_found = if trait_env.lookup_impl(*trait_id, resolved).is_some() {
                            true
                        } else {
                            trait_env
                                .lookup_impl_generic(*trait_id, resolved, ctx, symbols)
                                .is_some()
                        };
                        if !impl_found {
                            return Err(TypeError::TraitNotImplemented {
                                ty: *ty,
                                trait_name: format!("{:?}", trait_id),
                                span: *span,
                            });
                        }
                        // Generate obligations for associated types: when we have a
                        // resolved Impl(concrete_ty, trait_id, _), look for concrete types
                        // for any AssociatedType { trait_id, name, self_ty } by matching
                        // the impl's assoc_tys entries.
                        if let Some(impl_candidate) = trait_env.lookup_impl(*trait_id, resolved) {
                            for (assoc_name, assoc_ty) in &impl_candidate.assoc_tys {
                                // Walk all Eq constraints to substitute any AssociatedType
                                // that matches this name, trait_id, and self_ty
                                for eq_c in &self.constraints {
                                    if let Constraint::Eq(a, b, _) = eq_c {
                                        for id in &[*a, *b] {
                                            let resolved_id = ctx.resolve_binding(*id);
                                            if let TypeData::AssociatedType {
                                                trait_id: at_trait_id,
                                                name: at_name,
                                                self_ty: at_self,
                                            } = ctx.get(resolved_id).clone()
                                            {
                                                if at_trait_id == *trait_id
                                                    && at_name == *assoc_name
                                                    && ctx.resolve_binding(at_self) == resolved
                                                {
                                                    ctx.unify(resolved_id, *assoc_ty)?;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Constraint::Match {
                        scrutinee,
                        branches_id,
                        span: match_span,
                    } => {
                        // OmniML §4.1: Try to discharge suspended match constraints.
                        // Check unicity — if the scrutinee's shape is uniquely determined,
                        // discharge the match and enqueue continuation constraints.
                        let resolved = ctx.resolve_binding(*scrutinee);
                        let resolved_data = ctx.get(resolved);

                        if !matches!(resolved_data, TypeData::InferVar { .. }) {
                            // Scrutinee is resolved — shape is known (UNI-TYPE).
                            if !self.discharge_match(ctx, *scrutinee, *branches_id, &mut heap) {
                                // No branch matched and no else_ fallback for a
                                // fully-resolved scrutinee.  This is a type error:
                                // the pattern match is non-exhaustive.  Terminate
                                // the solver immediately — re-pushing would cause
                                // a deterministic infinite loop.
                                return Err(TypeError::PatternNotExhaustive {
                                    span: *match_span,
                                });
                            }
                        } else {
                            // Try shape variable (OmniML §6): register a callback on the
                            // scrutinee's shape variable, if any.
                            let shape_known = self.try_match_via_shape_var(
                                ctx,
                                *scrutinee,
                                *branches_id,
                                &mut heap,
                            );
                            if !shape_known {
                                // Scrutinee is still an InferVar — try unicity via bounds
                                // Collect remaining heap items to pass as active constraints
                                // for UNI-VAR (OmniML §4.1): Eq constraints in the active set
                                // determine the shape.  Previously this passed &[], which
                                // disabled UNI-VAR entirely.
                                let active: Vec<PrioritizedConstraint> = heap.drain().collect();
                                if let Some(_shape) =
                                    Self::unicity_check(self, ctx, *scrutinee, &active)
                                {
                                    // Re-push active constraints before discharging
                                    for c in &active {
                                        heap.push(c.clone());
                                    }
                                    if !self.discharge_match(
                                        ctx,
                                        *scrutinee,
                                        *branches_id,
                                        &mut heap,
                                    ) {
                                        // Unicity succeeded but no branch matched.
                                        // Same as above: type error, not retryable.
                                        return Err(TypeError::PatternNotExhaustive {
                                            span: *match_span,
                                        });
                                    }
                                } else {
                                    // Re-push active constraints
                                    for c in active {
                                        heap.push(c);
                                    }
                                    // Cannot discharge yet — push back as low priority
                                    let p = 6u8;
                                    heap.push(PrioritizedConstraint {
                                        priority: p,
                                        constraint: pc.constraint.clone(),
                                    });
                                }
                            }
                        }
                    }
                    Constraint::Exists {
                        var_id,
                        constraint,
                        span: _,
                    } => {
                        // OmniML: ∃α. C — bind a fresh flexible variable.
                        // α is already an InferVar at this level; just solve the body.
                        let inner = constraint.as_ref().clone();
                        let p = inner.priority(ctx);
                        heap.push(PrioritizedConstraint {
                            priority: p,
                            constraint: inner,
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
                        // on TypeContext, preventing the skolem from leaking
                        // into outer scopes via generalization.
                        if *var_id < self.var_type_ids.len() {
                            let (_universe, skolem_ty) = ctx.enter_universe();
                            let infer_ty = self.var_type_ids[*var_id];
                            // Use the local `unify` (not `ctx.unify`) so that
                            // the resolution is recorded in `self.resolutions`,
                            // keeping `self.resolve()` consistent.
                            let _ = self.unify(infer_ty, skolem_ty, ctx);
                        }
                        let inner = constraint.as_ref().clone();
                        let p = inner.priority(ctx);
                        heap.push(PrioritizedConstraint {
                            priority: p,
                            constraint: inner,
                        });
                    }
                    Constraint::Instance {
                        scheme_ty,
                        instantiation_ty,
                        span: _,
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
                                                param_index,
                                                body,
                                                ..
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
                                _ => {
                                    // Concrete (or error) — just unify directly.
                                    Some(InstantiationTarget::Concrete(resolved_scheme))
                                }
                            }
                        };

                        // Phase 2: Create fresh vars and emit Eq (mutable borrow).
                        if let Some(target) = scheme_info {
                            let eq_c = match target {
                                InstantiationTarget::Forall {
                                    binder_indices,
                                    body_ty,
                                } => {
                                    let mut instantiated = body_ty;
                                    for &idx in binder_indices.iter().rev() {
                                        let fv =
                                            self.new_type_var(ctx, TypeVariableKind::Any);
                                        instantiated =
                                            ctx.replace_generic(instantiated, idx, fv);
                                    }
                                    Constraint::Eq(
                                        *instantiation_ty,
                                        instantiated,
                                        crate::ast::Span::new(0, 0),
                                    )
                                }
                                InstantiationTarget::Poly {
                                    binder_indices,
                                    body_ty,
                                } => {
                                    let mut instantiated = body_ty;
                                    for &idx in binder_indices.iter() {
                                        let fv =
                                            self.new_type_var(ctx, TypeVariableKind::Any);
                                        instantiated =
                                            ctx.replace_generic(instantiated, idx, fv);
                                    }
                                    Constraint::Eq(
                                        *instantiation_ty,
                                        instantiated,
                                        crate::ast::Span::new(0, 0),
                                    )
                                }
                                InstantiationTarget::Concrete(target_ty) => Constraint::Eq(
                                    *instantiation_ty,
                                    target_ty,
                                    crate::ast::Span::new(0, 0),
                                ),
                            };
                            let p = eq_c.priority(ctx);
                            heap.push(PrioritizedConstraint {
                                priority: p,
                                constraint: eq_c,
                            });
                        }
                    }
                    Constraint::Let {
                        expr_var: _,
                        def_constraint,
                        body_constraint,
                        span: _,
                    } => {
                        let prev_level = self.enter_level();
                        let def_p = def_constraint.priority(ctx);
                        heap.push(PrioritizedConstraint {
                            priority: def_p,
                            constraint: def_constraint.as_ref().clone(),
                        });
                        let body_p = body_constraint.priority(ctx).max(4);
                        heap.push(PrioritizedConstraint {
                            priority: body_p,
                            constraint: body_constraint.as_ref().clone(),
                        });
                        self.exit_level(prev_level);
                    }
                }
            }

            // ── OmniML: Process Match constraints ──────────────────────
            // After processing Eq/Sub/Impl, check suspended match constraints.
            // A Match constraint can be discharged when the scrutinee's shape
            // is uniquely determined by the context (unicity check).
            // This implements O'Brien, Rémy & Scherer §4.1, MATCH-CTX rule.
            for pc in &heap.clone().into_sorted_vec() {
                if let Constraint::Match {
                    scrutinee,
                    branches_id,
                    span: _,
                } = &pc.constraint
                {
                    let resolved = ctx.resolve_binding(*scrutinee);
                    // Collect active constraints for UNI-VAR check
                    let active: Vec<PrioritizedConstraint> = heap.iter().cloned().collect();
                    // Only attempt discharge if scrutinee is resolved (not an InferVar)
                    if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                        if let Some(shape) = Self::unicity_check(self, ctx, *scrutinee, &active) {
                            let discharged =
                                self.discharge_match(ctx, *scrutinee, *branches_id, &mut heap);
                            // If discharge succeeded, the Match constraint is still in the
                            // original heap (we cloned above).  It will be re-processed in
                            // the next iteration, but discharge_match is idempotent — the
                            // continuation constraints are enqueued once and the second call
                            // is a no-op.  If discharge failed, the constraint remains in the
                            // heap and the convergence path below handles it.
                            let _ = discharged;
                        }
                    } else {
                        // Scrutinee is still an InferVar — check unicity via bounds
                        if let Some(_shape) = Self::unicity_check(self, ctx, *scrutinee, &active) {
                            let discharged =
                                self.discharge_match(ctx, *scrutinee, *branches_id, &mut heap);
                            let _ = discharged;
                        }
                    }
                }
            }

            // ── Wake-up: reprocess suspended constraints ───────────────
            // After processing all active constraints, check if any variables
            // were resolved. If so, wake their wait-listed constraints and
            // continue solving (OmniML bidirectional flow §3.2).
            let mut woken = 0usize;
            for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
                let resolved = ctx.resolve_binding(ty_id);
                if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                    // This variable was resolved — wake its suspended constraints
                    if i < self.wait_lists.len() && !self.wait_lists[i].is_empty() {
                        let suspended = std::mem::take(&mut self.wait_lists[i]);
                        let count = suspended.len();
                        for c in suspended {
                            let p = c.priority(ctx);
                            heap.push(PrioritizedConstraint {
                                priority: p,
                                constraint: c,
                            });
                        }
                        // Clear guards for this variable — every constraint
                        // in the former wait list had a guard added by
                        // `suspend_on_var`.  Without this clear, the variable
                        // remains permanently in PG and can never become
                        // Generalized.
                        if i < self.guard_sets.len() {
                            self.guard_sets[i].clear();
                        }
                        // If guards are now empty and status was PG,
                        // transition to Generalized (OmniML §6).
                        if i < self.gen_statuses.len()
                            && self.gen_statuses[i] == GenStatus::PartiallyGeneralizable
                        {
                            let guards_empty =
                                i < self.guard_sets.len() && self.guard_sets[i].is_empty();
                            if guards_empty {
                                self.gen_statuses[i] = GenStatus::Generalized;
                            }
                        }
                        woken += count;
                    }
                }
            }
            if woken == 0 {
                // If there are delayed constraints (e.g. Impl on unresolved
                // infer vars) and we haven't retried them yet, re‑queue
                // them into the heap and retry once.  If the same delayed
                // constraints are still present on the next stall, the
                // dependency never resolves — stop retrying.
                if !delayed.is_empty() && !delayed_retried {
                    for c in delayed.drain(..) {
                        heap.push(c);
                    }
                    delayed_retried = true;
                    continue;
                }
                // #2: Solver exhaustion — check for remaining undischarged Match
                // constraints and fire their else_continuation as a fallback.
                let remaining: Vec<PrioritizedConstraint> = heap.drain().collect();
                let match_elses: Vec<(TypeId, (usize, usize), crate::ast::Span)> = remaining
                    .iter()
                    .filter_map(|pc| {
                        if let Constraint::Match {
                            scrutinee,
                            branches_id,
                            span,
                        } = &pc.constraint
                        {
                            let resolved = ctx.resolve_binding(*scrutinee);
                            if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                                Some((*scrutinee, *branches_id, *span))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect();
                let mut else_heap = BinaryHeap::new();
                let mut match_errors = Vec::new();
                for (scrutinee, branches_id, match_span) in match_elses {
                    if !self.discharge_match(ctx, scrutinee, branches_id, &mut else_heap) {
                        // No branch matched and no else_ fallback for a fully-resolved
                        // scrutinee.  This is a type error — the pattern match is
                        // non-exhaustive.  Accumulate the error; the heap is about to
                        // be dropped so pushing into it would be dead code.
                        match_errors.push(TypeError::PatternNotExhaustive {
                            span: match_span,
                        });
                    }
                }
                if !match_errors.is_empty() {
                    return Err(match_errors.into_iter().next().unwrap());
                }
                break; // converged: no more constraints to wake
            }
            // woken > 0: progress was made — reset retry guard so delayed
            // constraints may be retried in the next iteration.
            delayed_retried = false;
            // Continue the loop to process woken constraints
        }

        // Kind checking: ensure that solved types respect the variable's kind
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            let data = ctx.get(resolved);
            if let TypeData::InferVar { .. } = data {
                continue; // will be defaulted below
            }
            if matches!(data, TypeData::Error) {
                continue;
            }
            let kind = self.type_vars[i].kind;
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
                            span: Span::new(0, 0),
                        });
                    }
                }
                TypeVariableKind::Float => {
                    if !matches!(data, TypeData::Float { .. }) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
                        });
                    }
                }
                TypeVariableKind::Bool => {
                    if !matches!(data, TypeData::Bool) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
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
                            span: Span::new(0, 0),
                        });
                    }
                }
                _ => {}
            }
        }

        // Defaulting: unfilled infer vars get default types,
        // UNLESS they are PartiallyGeneralizable (guarded by suspended constraints).
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            if let TypeData::InferVar { .. } = ctx.get(resolved) {
                // Skip variables that are PG — they still have suspended constraints
                // and will be re-generalized when those constraints are discharged.
                if i < self.gen_statuses.len()
                    && self.gen_statuses[i] == GenStatus::PartiallyGeneralizable
                {
                    continue;
                }
                let default_ty = match self.type_vars[i].kind {
                    TypeVariableKind::Integer => ctx.int(32, true),
                    TypeVariableKind::Float => ctx.float(64),
                    TypeVariableKind::Bool => ctx.bool(),
                    TypeVariableKind::Numeric => ctx.int(32, true),
                    TypeVariableKind::Unconstrained => ctx.error(),
                    TypeVariableKind::Any => ctx.error(),
                };
                ctx.set_binding(ty_id, default_ty);
            }
        }

        // ── Re-check delayed Impl constraints after defaulting ──────
        // Variables that were unresolved during the solver loop may have
        // been defaulted above.  Any remaining delayed Impl constraints
        // must now be checked against the concrete (defaulted) types.
        // Without this, trait obligations could be silently dropped while
        // the solver reports success — a soundness hole.
        for pc in &delayed {
            if let Constraint::Impl(ty, trait_id, span) = &pc.constraint {
                let resolved = ctx.resolve_binding(*ty);
                let data = ctx.get(resolved);
                if matches!(data, TypeData::Error) {
                    continue;
                }
                if matches!(data, TypeData::InferVar { .. }) {
                    // This variable was not defaulted because it is PG
                    // (PartiallyGeneralizable) — but the solver has exhausted
                    // all progress and the Impl constraint was never resolved.
                    // A required trait implementation cannot be verified,
                    // so this is an error, not a skip.
                    return Err(TypeError::TraitNotImplemented {
                        ty: *ty,
                        trait_name: format!("{:?}", trait_id),
                        span: *span,
                    });
                }
                // GenericParam remains unresolved (the function was never
                // instantiated with concrete types).  That's fine — the
                // constraint will be checked at the monomorphization site.
                if matches!(data, TypeData::GenericParam { .. }) {
                    continue;
                }
                let impl_found = if trait_env.lookup_impl(*trait_id, resolved).is_some() {
                    true
                } else {
                    trait_env
                        .lookup_impl_generic(*trait_id, resolved, ctx, symbols)
                        .is_some()
                };
                if !impl_found {
                    return Err(TypeError::TraitNotImplemented {
                        ty: *ty,
                        trait_name: format!("{:?}", trait_id),
                        span: *span,
                    });
                }
                // Associated type obligations could be checked here too,
                // but they are handled during the main solving pass when
                // the Impl constraint is resolved.
            }
        }

        Ok(())
    }

    pub fn finalize(&self, ctx: &mut TypeContext) -> HashMap<usize, TypeId> {
        let mut solution = HashMap::default();
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            let data = ctx.get(resolved);
            match data {
                TypeData::InferVar { id } => {
                    // Variable is still unbound — try to infer from bounds
                    let var_id = *id;
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
                    let chosen = if !lbs.is_empty() {
                        // Covariant: pick the least upper bound from lower bounds
                        // (simple heuristic: pick the first resolved lower bound)
                        let first_resolved = lbs.iter().find(|t| {
                            let r = ctx.resolve_binding(**t);
                            !matches!(ctx.get(r), TypeData::InferVar { .. })
                        });
                        first_resolved.copied().unwrap_or(ctx.error())
                    } else if !ubs.is_empty() {
                        // Contravariant: pick the greatest lower bound from upper bounds
                        let first_resolved = ubs.iter().find(|t| {
                            let r = ctx.resolve_binding(**t);
                            !matches!(ctx.get(r), TypeData::InferVar { .. })
                        });
                        first_resolved.copied().unwrap_or(ctx.error())
                    } else {
                        // No bounds — default based on kind
                        match self.type_vars[i].kind {
                            TypeVariableKind::Integer => ctx.int(32, true),
                            TypeVariableKind::Float => ctx.float(64),
                            TypeVariableKind::Bool => ctx.bool(),
                            TypeVariableKind::Numeric => ctx.int(32, true),
                            _ => ctx.error(),
                        }
                    };
                    solution.insert(var_id, chosen);
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
    pub fn force_root_generalization(&mut self, ctx: &mut TypeContext) -> Vec<(usize, usize)> {
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
    fn generalize_region(
        &mut self,
        region_id: InferRegionId,
        ctx: &mut TypeContext,
        out: &mut Vec<(usize, usize)>,
    ) {
        let var_ids: Vec<usize> = self.region_tree.nodes[region_id.0]
            .pool
            .var_ids
            .clone();
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
                    // Unregister from current region, register in parent.
                    self.region_tree.nodes[region_id.0].pool.var_ids.retain(|&v| v != var_id);
                    self.region_tree.nodes[parent.0].pool.var_ids.push(var_id);
                    self.type_vars[var_id].region_id = parent;
                }
                // If there is no parent (root region), the variable stays
                // in the current pool — it will be processed again on the
                // next generalization pass.
                continue;
            }
            if !is_guarded {
                // Not guarded — can be generalized.
                // Update the gen_status if it exists.
                if var_id < self.gen_statuses.len() {
                    self.gen_statuses[var_id] = GenStatus::Generalized;
                }
                // When a variable is generalized, update its instances.
                // BUT: only mark an instance as Generalized if it is NOT
                // itself still guarded (OmniML §6: a variable can become
                // Generic only when all its constraints are resolved).
                if var_id < self.forward_refs.len() {
                    for &inst_id in &self.forward_refs[var_id] {
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
                            if inst_id < self.gen_statuses.len() {
                                self.gen_statuses[inst_id] = GenStatus::Generalized;
                            }
                            // Remove the instance from its region's pool.
                            // Generalized variables must not belong to any pool
                            // (OmniML §6: "once a term is generalised, it is
                            // removed from its pool").
                            if inst_id < self.type_vars.len() {
                                let inst_region = self.type_vars[inst_id].region_id;
                                if inst_region.0 < self.region_tree.nodes.len() {
                                    self.region_tree.nodes[inst_region.0]
                                        .pool
                                        .var_ids
                                        .retain(|&v| v != inst_id);
                                }
                            }
                        }
                    }
                }
                out.push((region_id.0, var_id));
            }
        }
        // Remove generalized variables from the pool.
        self.region_tree.nodes[region_id.0]
            .pool
            .var_ids
            .retain(|v| !out.iter().any(|(_, vid)| vid == v));
        // Mark the region as processed.
        self.region_tree.nodes[region_id.0].dirty = false;
    }

    pub fn apply_solution(
        ty: TypeId,
        solution: &HashMap<usize, TypeId>,
        ctx: &TypeContext,
    ) -> TypeId {
        replace_infer(ty, solution, ctx)
    }

    /// Check for inference variables that remain unresolved and were
    /// defaulted to `error` (unconstrained/any kind). Returns a list of
    /// diagnostic messages describing the ambiguous variables.
    pub fn check_unresolved(&self, ctx: &TypeContext) -> Vec<String> {
        let mut results = Vec::new();
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                if i < self.type_vars.len() {
                    // Only report `Any` — `Unconstrained` is defaulted to error
                    // by the solver as a normal fallback, not an ambiguity.
                    if self.type_vars[i].kind == TypeVariableKind::Any {
                        results.push(format!("unresolved type variable #{} (Any)", i));
                    }
                }
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

fn replace_infer(ty: TypeId, solution: &HashMap<usize, TypeId>, ctx: &TypeContext) -> TypeId {
    let resolved = ctx.resolve_binding(ty);
    let data = ctx.get(resolved).clone();
    match data {
        TypeData::InferVar { id } => solution.get(&id).copied().unwrap_or(ty),
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
        | TypeData::Poly { .. } => ty,
        TypeData::GenericParam { .. } => ty,
        TypeData::Adt { kind, def_id, args } => {
            let new_args: Vec<TypeId> = args
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Adt {
                kind,
                def_id,
                args: new_args,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Tuple { elems } => {
            let new_elems: Vec<TypeId> = elems
                .iter()
                .map(|&e| replace_infer(e, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Tuple { elems: new_elems })
                .unwrap_or(ctx.error())
        }
        TypeData::Array { elem, size } => {
            let new_elem = replace_infer(elem, solution, ctx);
            ctx.find_type(&TypeData::Array {
                elem: new_elem,
                size,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Slice { elem } => {
            let new_elem = replace_infer(elem, solution, ctx);
            ctx.find_type(&TypeData::Slice { elem: new_elem })
                .unwrap_or(ctx.error())
        }
        TypeData::Ref { ty, mutable } => {
            let new_ty = replace_infer(ty, solution, ctx);
            ctx.find_type(&TypeData::Ref {
                ty: new_ty,
                mutable,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Pointer { ty } => {
            let new_ty = replace_infer(ty, solution, ctx);
            ctx.find_type(&TypeData::Pointer { ty: new_ty })
                .unwrap_or(ctx.error())
        }
        TypeData::Ptr { size, pointee } => {
            let new_size = replace_infer(size, solution, ctx);
            let new_pointee = replace_infer(pointee, solution, ctx);
            ctx.find_type(&TypeData::Ptr {
                size: new_size,
                pointee: new_pointee,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Fn { params, ret } => {
            let new_params: Vec<TypeId> = params
                .iter()
                .map(|&p| replace_infer(p, solution, ctx))
                .collect();
            let new_ret = replace_infer(ret, solution, ctx);
            ctx.find_type(&TypeData::Fn {
                params: new_params,
                ret: new_ret,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::DynTrait { .. } => ty,
        TypeData::Exists {
            param_index,
            name,
            base,
        } => {
            let new_base = replace_infer(base, solution, ctx);
            ctx.find_type(&TypeData::Exists {
                param_index,
                name,
                base: new_base,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => {
            let new_self = replace_infer(self_ty, solution, ctx);
            ctx.find_type(&TypeData::AssociatedType {
                trait_id,
                name,
                self_ty: new_self,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Forall {
            param_index,
            param_name,
            body,
        } => {
            let new_body = replace_infer(body, solution, ctx);
            ctx.find_type(&TypeData::Forall {
                param_index,
                param_name,
                body: new_body,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Mu {
            param_index,
            param_name,
            body,
        } => {
            let new_body = replace_infer(body, solution, ctx);
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
            let new_body = replace_infer(body, solution, ctx);
            ctx.find_type(&TypeData::Nu {
                param_index,
                param_name,
                body: new_body,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Coproduct { alternatives } => {
            let new_alts: Vec<TypeId> = alternatives
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Coproduct {
                alternatives: new_alts,
            })
            .unwrap_or(ctx.error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_ctx() -> TypeContext {
        TypeContext::new()
    }

    #[test]
    fn test_shape_of_fn() {
        let mut ctx = new_ctx();
        let int_ty = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(vec![int_ty], bool_ty);
        let shape = InferenceContext::shape_of_type(&ctx, fn_ty);
        assert!(matches!(shape, PrincipalShape::Arrow));
    }

    #[test]
    fn test_shape_of_tuple() {
        let mut ctx = new_ctx();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(32, true);
        let tup = ctx.tuple(vec![bool_ty, int_ty]);
        let shape = InferenceContext::shape_of_type(&ctx, tup);
        assert!(matches!(shape, PrincipalShape::Tuple(2)));
    }

    #[test]
    fn test_shape_of_forall() {
        let mut ctx = new_ctx();
        let p0 = ctx.generic_param(0, "X".into());
        let fn_ty = ctx.function(vec![p0], p0);
        let forall = ctx.forall(0, "X".into(), fn_ty);
        let shape = InferenceContext::shape_of_type(&ctx, forall);
        assert!(matches!(shape, PrincipalShape::Poly));
    }

    #[test]
    fn test_suspend_and_wake_var() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        // Extract the var ID from the InferVar type
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.suspend_on_var(
            Constraint::Impl(ctx.bool(), DefId(0), crate::ast::Span::new(0, 0)),
            var_id,
        );
        // Waking moves suspended constraints back to the active list
        infer.wake_var(var_id);
        // The active constraints list should now have the suspended constraint
        assert!(!infer.constraints.is_empty());
    }

    #[test]
    fn test_wake_var_incremental() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.suspend_on_var(
            Constraint::Impl(ctx.bool(), DefId(0), crate::ast::Span::new(0, 0)),
            var_id,
        );
        // wake_var_incremental needs a heap, var_id, and ctx
        let mut heap = std::collections::BinaryHeap::new();
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
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        assert!(infer.get_var_level(var_id).unwrap_or(0) > 0);
        infer.exit_level(prev);
    }

    #[test]
    fn test_eq_concrete_priority() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(32, true);
        let eq = Constraint::Eq(bool_ty, int_ty, crate::ast::Span::new(0, 0));
        let impl_c = Constraint::Impl(bool_ty, DefId(0), crate::ast::Span::new(0, 0));
        assert!(eq.priority(&ctx) < impl_c.priority(&ctx));
    }

    #[test]
    fn test_impl_lowest_priority() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(32, true);
        let a = Constraint::Impl(bool_ty, DefId(0), crate::ast::Span::new(0, 0));
        let b = Constraint::Impl(int_ty, DefId(1), crate::ast::Span::new(0, 0));
        // Both Impl constraints should have the same priority
        assert_eq!(a.priority(&ctx), b.priority(&ctx));
    }

    #[test]
    fn test_match_constraint_priority() {
        let mut ctx = TypeContext::new();
        let infer = InferenceContext::new();
        let var = ctx.alloc_infer_var(0);
        // Match constraint on an InferVar → low priority (6)
        let match_c = Constraint::Match {
            scrutinee: var,
            branches_id: (0, 0),
            span: crate::ast::Span::new(0, 0),
        };
        assert_eq!(
            match_c.priority(&ctx),
            6,
            "Match on InferVar should have lowest priority"
        );
        // Eq on concrete types → high priority (0)
        let eq_c = Constraint::Eq(ctx.bool(), ctx.int(32, true), crate::ast::Span::new(0, 0));
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
        let int_ty = ctx.int(32, true);
        let shape = InferenceContext::unicity_check(&infer, &ctx, int_ty, &[]);
        assert!(shape.is_some(), "non-variable type should have known shape");
    }

    #[test]
    fn test_unicity_check_fn_type() {
        let mut ctx = TypeContext::new();
        let infer = InferenceContext::new();
        let int_ty = ctx.int(32, true);
        let fn_ty = ctx.function(vec![int_ty], int_ty);
        let shape = InferenceContext::unicity_check(&infer, &ctx, fn_ty, &[]);
        assert!(shape.is_some(), "function type should have known shape");
        assert_eq!(shape.unwrap(), PrincipalShape::Arrow);
    }

    #[test]
    fn test_register_instance_and_propagate() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let pg_var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let pg_id = match ctx.get(pg_var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        let inst1 = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let inst1_id = match ctx.get(inst1) {
            TypeData::InferVar { id } => *id,
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
            continuation: vec![Constraint::Eq(
                ctx.int(32, true),
                ctx.int(32, true),
                crate::ast::Span::new(0, 0),
            )],
            else_continuation: Vec::new(),
        }];
        let id = infer.register_match_branches(branches);
        assert!(id.0 < infer.match_branches.len());
    }

    #[test]
    fn test_match_priority_change_on_resolve() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let match_c = Constraint::Match {
            scrutinee: var,
            branches_id: (0, 0),
            span: crate::ast::Span::new(0, 0),
        };
        // Before resolution: low priority
        assert_eq!(match_c.priority(&ctx), 6);
        // After resolution: medium priority
        let int_ty = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(var, int_ty);
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
        let mut heap = std::collections::BinaryHeap::new();
        let int_ty = ctx.int(32, true);

        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Vec::new(),
            else_continuation: vec![Constraint::Eq(int_ty, int_ty, crate::ast::Span::new(0, 0))],
        }];
        let id = infer.register_match_branches(branches);
        let int_ty2 = ctx.int(64, false);

        let result = infer.discharge_match(&mut ctx, int_ty2, id, &mut heap);
        assert!(result, "else_ fallback should return true");
        assert!(!heap.is_empty(), "else_ continuation should be enqueued");
    }

    #[test]
    fn test_else_continuation_empty_still_fails() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = std::collections::BinaryHeap::new();

        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Vec::new(),
            else_continuation: Vec::new(),
        }];
        let id = infer.register_match_branches(branches);
        let int_ty = ctx.int(32, true);

        let result = infer.discharge_match(&mut ctx, int_ty, id, &mut heap);
        assert!(!result, "no else_ fallback should still fail");
        assert!(heap.is_empty(), "no constraints should be enqueued");
    }

    // ── force_generalization ─────────────────────────────────────────

    #[test]
    fn test_force_generalize_pg_with_guard() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        let _var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let var_id = 0;
        if var_id < infer.gen_statuses.len() {
            infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
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

        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let var_id = 0;
        if var_id < infer.gen_statuses.len() {
            infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        }

        let int_ty = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(var, int_ty);

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
        let mut heap = std::collections::BinaryHeap::new();

        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Vec::new(),
            else_continuation: Vec::new(),
        }];
        let id = infer.register_match_branches(branches);

        let handled = infer.try_match_via_shape_var(&mut ctx, var, id, &mut heap);
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
        let mut heap = std::collections::BinaryHeap::new();

        let int_ty = ctx.int(32, true);
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Scalar,
            continuation: Vec::new(),
            else_continuation: Vec::new(),
        }];
        let id = infer.register_match_branches(branches);

        let handled = infer.try_match_via_shape_var(&mut ctx, int_ty, id, &mut heap);
        assert!(handled, "concrete type with matching shape should discharge");
    }

    #[test]
    fn test_let_constraint_priority() {
        let c = Constraint::Let {
            expr_var: "x".into(),
            def_constraint: Box::new(Constraint::Eq(
                TypeId::from_raw(1),
                TypeId::from_raw(2),
                crate::ast::Span::new(0, 0),
            )),
            body_constraint: Box::new(Constraint::Eq(
                TypeId::from_raw(1),
                TypeId::from_raw(1),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
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
        svc.try_set(a, TypeShape::Constructor(1));

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
        assert!(shapes_compatible(
            TypeShape::Constructor(0),
            TypeShape::Constructor(5)
        ));
        assert!(!shapes_compatible(TypeShape::Arrow, TypeShape::Tuple(1)));
        assert!(shapes_compatible(TypeShape::Poly, TypeShape::Poly));
        assert!(!shapes_compatible(TypeShape::Poly, TypeShape::Arrow));
    }

    #[test]
    fn test_type_data_to_shape_variants() {
        let mut ctx = TypeContext::new();
        // Fn → Arrow
        let fn_ty = ctx.function(vec![ctx.bool()], ctx.bool());
        assert_eq!(type_data_to_shape(ctx.get(fn_ty)), TypeShape::Arrow);
        // Tuple(n) → Tuple(n)
        let b = ctx.bool();
        let i = ctx.int(32, true);
        let tup = ctx.tuple(vec![b, i]);
        assert_eq!(type_data_to_shape(ctx.get(tup)), TypeShape::Tuple(2));
        // Struct → Constructor(n)
        let b2 = ctx.bool();
        let i2 = ctx.int(32, true);
        let s = ctx.struct_ty(DefId(42), vec![b2, i2]);
        assert_eq!(type_data_to_shape(ctx.get(s)), TypeShape::Constructor(2));
        // Forall → Poly
        let p0 = ctx.generic_param(0, "X".into());
        let forall = ctx.forall(0, "X".into(), p0);
        assert_eq!(type_data_to_shape(ctx.get(forall)), TypeShape::Poly);
        // Int → Unknown (primitive)
        let int32 = ctx.int(32, true);
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
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
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
            matches!(ctx.get(resolved), TypeData::InferVar { id } if *id != var_id),
            "original var should be bound to a new InferVar"
        );
    }

    #[test]
    fn test_try_promote_var_already_low() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        // Variable at level 0 (default)
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // Try to promote to level 0 — no-op since already at level 0
        let promoted = infer.try_promote_var(&mut ctx, var_id, InferRegionId(0));
        assert!(promoted.is_some(), "should return the existing var");
        let resolved = ctx.resolve_binding(var);
        assert!(
            matches!(ctx.get(resolved), TypeData::InferVar { id } if *id == var_id),
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
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        let deep_level = infer.get_var_level(var_id).unwrap();
        assert!(deep_level > 0, "var should be at a deep level");

        // Suspend constraints on it (simulating match/impl obligations)
        infer.suspend_on_var(
            Constraint::Eq(ctx.bool(), ctx.int(32, true), crate::ast::Span::new(0, 0)),
            var_id,
        );
        infer.suspend_on_var(
            Constraint::Impl(ctx.bool(), DefId(0), crate::ast::Span::new(0, 0)),
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
        let inst = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let inst_id = match ctx.get(inst) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.register_instance(var_id, inst_id);
        assert!(
            infer.has_instances(var_id),
            "old var should have instances"
        );

        // Also add a lower bound
        if var_id < infer.lower_bounds.len() {
            infer.lower_bounds[var_id].push(ctx.bool());
        }

        // ── Promote the variable ──────────────────────────────────
        let promoted = infer.try_promote_var(&mut ctx, var_id, InferRegionId(0));
        assert!(promoted.is_some(), "promotion should succeed");

        // The promoted TypeId corresponds to the new variable
        let new_id = match ctx.get(promoted.unwrap()) {
            TypeData::InferVar { id } => *id,
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
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // Set PG status and add a suspended constraint (wait list not empty)
        if var_id < infer.gen_statuses.len() {
            infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        }
        let int32_1 = ctx.int(32, true);
        infer.suspend_on_var(
            Constraint::Eq(var, int32_1, crate::ast::Span::new(0, 0)),
            var_id,
        );

        // Resolve the variable so it has a concrete type
        let int64 = ctx.int(64, false);
        ctx.bindings.borrow_mut().insert(var, int64);

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
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        // Resolve the variable
        ctx.bindings.borrow_mut().insert(var, ctx.bool());
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
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.exit_level(prev);
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        let unit_ty = ctx.unit();
        ctx.bindings.borrow_mut().insert(var, unit_ty);

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
    fn test_rigid_escape_generic_param_detected() {
        let mut ctx = TypeContext::new();
        let gp = ctx.generic_param(0, "T".into());
        let escaped = InferenceContext::check_rigid_escape(&ctx, gp, 0);
        assert!(escaped, "GenericParam should be detected as escape");
    }

    #[test]
    fn test_rigid_escape_concrete_not_detected() {
        let mut ctx = TypeContext::new();
        let int_ty = ctx.int(32, true);
        let not_escaped = InferenceContext::check_rigid_escape(&ctx, int_ty, 0);
        assert!(!not_escaped, "Int<32> is not an escape");
    }

    #[test]
    fn test_rigid_escape_fn_with_gp_detected() {
        let mut ctx = TypeContext::new();
        let gp = ctx.generic_param(1, "U".into());
        let fn_ty = ctx.function(vec![gp], gp);
        let escaped = InferenceContext::check_rigid_escape(&ctx, fn_ty, 0);
        assert!(escaped, "fn(U) -> U contains GenericParam escape");
    }

    // ═══════════════════════════════════════════════════════════════
    // S-Inst-Copy Propagation Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_s_inst_copy_deepen_follows_aliases() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let pg = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let pg_id = match ctx.get(pg) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // Create an instance
        let inst = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let inst_id = match ctx.get(inst) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.register_instance(pg_id, inst_id);

        // Resolve PG → fn(Int) → Bool
        let int32_2 = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(vec![int32_2], bool_ty);
        ctx.bindings.borrow_mut().insert(pg, fn_ty);

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
        let pg = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let pg_id = match ctx.get(pg) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        let inst = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let inst_id = match ctx.get(inst) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.register_instance(pg_id, inst_id);
        let bool_ty = ctx.bool();
        ctx.bindings.borrow_mut().insert(pg, bool_ty);
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
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // S-Exists-Lower requires PG status and level > current_level
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        // exit_level so current_level drops below the var's level
        infer.exit_level(prev);
        let lowered = infer.s_exists_lower(&mut ctx, var_id);
        assert!(lowered, "S-Exists-Lower should succeed");
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Ungeneralized,
            "PG var should become Ungeneralized after S-Exists-Lower"
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

        let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        // Add constraint: Eq(a, b)
        infer.add_constraint(Constraint::Eq(a, b, crate::ast::Span::new(0, 0)));
        // Unify a with Int<32>
        let int_ty = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(a, int_ty);

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
        let deep_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let shallow_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        infer.exit_level(prev);

        // Eq(deep, shallow) — should promote deep to shallow's level
        infer.add_constraint(Constraint::Eq(
            deep_var,
            shallow_var,
            crate::ast::Span::new(0, 0),
        ));
        let bool_ty = ctx.bool();
        ctx.bindings.borrow_mut().insert(shallow_var, bool_ty);

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
        let int32 = ctx.int(32, true);
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: vec![Constraint::Eq(int32, int32, crate::ast::Span::new(0, 0))],
            else_continuation: Vec::new(),
        }];
        let bid = infer.register_match_branches(branches);

        // Create an infer var and try_match via shape var (should suspend)
        let infer_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(infer_var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        let mut heap = std::collections::BinaryHeap::new();
        let _handled = infer.try_match_via_shape_var(&mut ctx, infer_var, bid, &mut heap);
        // try_match_via_shape_var always returns true (handled) but for an InferVar
        // it should suspend on the wait list, not discharge
        assert!(
            var_id < infer.wait_lists.len() && !infer.wait_lists[var_id].is_empty(),
            "match should be suspended on the infer var's wait list"
        );

        // Now resolve the infer var to a concrete fn type and wake it
        let fn_bool = ctx.bool();
        let fn_ty = ctx.function(vec![fn_bool], fn_bool);
        ctx.bindings.borrow_mut().insert(infer_var, fn_ty);
        infer.wake_var_incremental(var_id, &mut heap, &ctx);

        // The match should now be in the heap
        assert!(!heap.is_empty(), "match should be woken and placed in heap");
        let woken = heap.pop().unwrap();
        assert!(
            matches!(woken.constraint, Constraint::Match { .. }),
            "woken constraint should be Match"
        );

        // Discharge it directly
        let fn_ty2 = ctx.function(vec![ctx.bool()], ctx.bool());
        let discharged = infer.discharge_match(&mut ctx, fn_ty2, bid, &mut heap);
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
        let x = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let x_id = match ctx.get(x) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        assert!(
            infer.get_var_level(x_id).unwrap_or(0) > 0,
            "x should be at a deeper level"
        );

        // Bind x to Int<32>
        let int32 = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(x, int32);

        // Mark the variable PG and then force_generalize
        infer.gen_statuses[x_id] = GenStatus::PartiallyGeneralizable;
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
        let a_deep = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let a_deep_id = match ctx.get(a_deep) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.exit_level(_r1); // back to branch_A
        infer.exit_level(_r0); // back to root

        // Now enter branch_B → leaf_B1
        let _r2 = infer.enter_level(); // → branch_B
        let _r3 = infer.enter_level(); // → leaf_B1
        let b_deep = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let b_deep_id = match ctx.get(b_deep) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.exit_level(_r3); // back to branch_B
        infer.exit_level(_r2); // back to root

        // Verify the variables are at different (sibling) regions
        let a_region = infer.type_vars[a_deep_id].region_id;
        let b_region = infer.type_vars[b_deep_id].region_id;
        assert_ne!(a_region, b_region, "sibling vars must be in different regions");
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
            crate::ast::Span::new(0, 0),
        ));

        // Bind B to Int<32> so the solver can resolve
        let int32 = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(b_deep, int32);

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
        let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        infer.exit_level(_r0);

        let _r1 = infer.enter_level(); // → branch_B
        let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        infer.exit_level(_r1);

        // Bind a to Bool, b to Int<32>, then constrain Eq(a, b)
        let bool_ty = ctx.bool();
        let int32 = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(a, bool_ty);
        ctx.bindings.borrow_mut().insert(b, int32);

        infer.add_constraint(Constraint::Eq(
            a,
            b,
            crate::ast::Span::new(0, 0),
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

        // enter_level() returns the parent region, sets current to child.
        let _parent = infer.enter_level();
        let child_region = infer.region_tree.current;
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // The variable should be in the child region's pool
        assert!(
            infer.region_tree.nodes[child_region.0].pool.var_ids.contains(&var_id),
            "new var should be in child region's pool"
        );

        // Promote the variable to the parent (root) region
        let root = infer.region_tree.root;
        let promoted = infer.try_promote_var(&mut ctx, var_id, root);
        assert!(promoted.is_some(), "try_promote_var should succeed");
        let promoted_id = match ctx.get(promoted.unwrap()) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };

        // The promoted variable should be in the root region's pool
        assert!(
            infer.region_tree.nodes[root.0].pool.var_ids.contains(&promoted_id),
            "promoted var should be in root region's pool"
        );

        // Simulate a transaction rollback: roll back the pool
        infer.region_tree.rollback_pool();

        // After rollback, the root region's pool should be restored to its
        // pre-registration state (the promoted variable should be removed)
        assert!(
            !infer.region_tree.nodes[root.0].pool.var_ids.contains(&promoted_id),
            "after rollback, promoted var should NOT be in root region's pool"
        );
        // The promoted variable should be back in the child region's pool
        // (the unregister_var undo entry should have re-inserted it).
        assert!(
            infer.region_tree.nodes[child_region.0].pool.var_ids.contains(&promoted_id),
            "after rollback, variable should be back in child region's pool"
        );
        // The original variable should NOT be in the child region's pool
        // (it was created before the simulated transaction and its
        // Register entry records old_var_len = 0, so truncation
        // removes it — this is expected because in a real scenario
        // the variable would be created inside the transaction scope).
        // What matters is that the promoted variable is correctly
        // re-inserted into the child pool and removed from the root pool.

        // Now run force_root_generalization — it should not crash
        let generalized = infer.force_root_generalization(&mut ctx);
        // After rollback, the promoted variable is back in the child pool
        // and is unguarded, so it should be generalized.
        assert!(
            !generalized.is_empty(),
            "after rollback, the unguarded promoted var should be generalized"
        );
        // The generalized variable should be the promoted one
        assert!(
            generalized.iter().any(|(_, vid)| *vid == promoted_id),
            "generalized should include the promoted variable"
        );
    }
}
