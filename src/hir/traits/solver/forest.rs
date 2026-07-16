use crate::hir::traits::solver::obligation::{Obligation, Predicate, SolveError};
use crate::hir::types::{DefId, TypeContext, TypeId};
use rustc_hash::FxHashSet as HashSet;
use std::collections::VecDeque;

/// Maximum number of nodes before the forest is compacted.
/// Compacted nodes (resolved/errored) are removed to prevent unbounded
/// memory growth in long-running compilation sessions.
pub const MAX_NODES: usize = 4096;

/// A tree of pending trait obligations.
///
/// Simpler than rustc's `ObligationForest` because:
/// - No `OutlivesPredicate` / region outlives obligations
/// - No `ProcessResult` / `ObligationProcessor` trait (we use direct methods)
/// - Coinductive cycles (auto traits) are detected by path hash-set
/// - Lifetime parameters are treated as named generic indices, not inference vars
#[derive(Clone, Debug)]
pub struct ObligationForest {
    nodes: Vec<ObligationNode>,
    /// Queue of pending node indices to process.
    pending: VecDeque<usize>,
    /// Path hash-set for cycle detection: (node_idx, trait_id, resolved_self_ty, resolved_args_hash).
    /// Tracks the current evaluation path to detect cycles.
    /// Uses the node index for deterministic removal (resolved keys can change
    /// when inference variables are unified during evaluation).
    active_path: HashSet<(usize, DefId, TypeId, u64)>,
}

#[derive(Clone, Debug)]
pub struct ObligationNode {
    pub obligation: Obligation,
    pub state: ObligationState,
    /// Parent node index (None = root).
    pub parent: Option<usize>,
    /// Children that have been registered from selection.
    pub children: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum ObligationState {
    Pending,
    Evaluating,
    Resolved,
    Error(SolveError),
    /// Cycle detected — coinductive traits (auto traits, Sized) are
    /// treated as success; non-coinductive cycles are errors.
    CycleDetected,
    /// The obligation could not be resolved yet because the self_ty is
    /// still an inference variable.  Will be retried after the type is
    /// resolved by the old solver.
    Deferred,
}

impl ObligationForest {
    pub fn new() -> Self {
        ObligationForest {
            nodes: Vec::new(),
            pending: VecDeque::new(),
            active_path: HashSet::default(),
        }
    }

    /// Register a new root obligation.
    /// Returns the node index.
    pub fn register(&mut self, obligation: Obligation) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ObligationNode {
            obligation,
            state: ObligationState::Pending,
            parent: None,
            children: Vec::new(),
        });
        self.pending.push_back(idx);
        idx
    }

    /// Register a child obligation for a given parent.
    /// Returns the child node index.
    pub fn register_child(
        &mut self,
        obligation: Obligation,
        parent_idx: usize,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ObligationNode {
            obligation,
            state: ObligationState::Pending,
            parent: Some(parent_idx),
            children: Vec::new(),
        });
        self.nodes[parent_idx].children.push(idx);
        self.pending.push_back(idx);
        idx
    }

    /// Get the next pending obligation to process.
    /// Returns `None` if no pending obligations remain.
    pub fn next_pending(&mut self) -> Option<usize> {
        while let Some(idx) = self.pending.pop_front() {
            if matches!(
                self.nodes[idx].state,
                ObligationState::Pending | ObligationState::Deferred
            ) {
                return Some(idx);
            }
        }
        None
    }

    /// Mark a node as resolved.
    pub fn mark_resolved(&mut self, idx: usize) {
        self.nodes[idx].state = ObligationState::Resolved;
    }

    /// Mark a node as errored.
    pub fn mark_error(&mut self, idx: usize, error: SolveError) {
        self.nodes[idx].state = ObligationState::Error(error);
    }

    /// Mark a node as deferred — cannot be resolved yet because the
    /// self_ty is still an inference variable.  The node will be retried
    /// after the type is resolved by the old solver.
    pub fn mark_deferred(&mut self, idx: usize) {
        self.nodes[idx].state = ObligationState::Deferred;
        self.pending.push_back(idx);
    }

    /// Count the number of deferred nodes in the forest.
    pub fn deferred_count(&self) -> usize {
        self.nodes.iter().filter(|n| matches!(n.state, ObligationState::Deferred)).count()
    }

    /// Mark a node as evaluating (entering cycle detection).
    ///
    /// Uses *resolved* TypeIds for cycle detection so that two obligations
    /// whose inference variables have been unified are recognized as the same,
    /// preventing infinite recursion from repeated impl application.
    ///
    /// Stores the resolved key alongside the node index so that
    /// `leave_evaluating` can remove the entry by index, avoiding the problem
    /// of a key changing under unification.
    pub fn mark_evaluating(&mut self, idx: usize, ctx: &TypeContext) -> bool {
        let node = &self.nodes[idx];

        // Compute the resolved cycle key (following bindings through inference vars).
        let Some(resolved_key) = self.resolved_key_for_node(node, ctx) else {
            // Other predicates don't participate in cycle detection
            return true;
        };

        // Check for cycles using resolved keys: two obligations form a cycle
        // when they have the same trait_id, resolved self_ty, and resolved args.
        // This catches the case where an inference variable was unified during
        // evaluation of a parent, making two syntactically different obligations
        // semantically identical.
        let is_cycle = self.active_path.iter().any(|(_, t, s, a)| {
            (*t, *s, *a) == resolved_key
        });

        if is_cycle {
            // Cycle detected
            let is_coinductive = matches!(
                &node.obligation.predicate,
                Predicate::AutoTrait { .. } | Predicate::Sized { .. }
            );
            if is_coinductive {
                // Coinductive cycles are ok (e.g., Send: Send)
                self.nodes[idx].state = ObligationState::CycleDetected;
                false
            } else {
                // Non-coinductive cycle is an error
                self.nodes[idx].state = ObligationState::Error(
                    SolveError::CycleDetected {
                        predicate: node.obligation.predicate.clone(),
                    }
                );
                false
            }
        } else {
            let (trait_id, self_ty, args_hash) = resolved_key;
            self.active_path.insert((idx, trait_id, self_ty, args_hash));
            self.nodes[idx].state = ObligationState::Evaluating;
            true
        }
    }

    /// Remove a node from the active path (after evaluation completes).
    /// Uses the node index for deterministic removal, avoiding the problem
    /// of a key changing when inference variables are unified during evaluation.
    ///
    /// Note: this is O(n) in the active path size, but the active path is
    /// bounded by the obligation nesting depth (typically < 10).
    pub fn leave_evaluating(&mut self, idx: usize) {
        self.active_path.retain(|(stored_idx, _, _, _)| *stored_idx != idx);
    }

    /// Check if there are still pending obligations.
    pub fn has_pending(&self) -> bool {
        self.pending.iter().any(|&idx| {
            matches!(self.nodes[idx].state, ObligationState::Pending)
        })
    }

    /// Get the number of obligations (including resolved ones).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of pending obligations.
    pub fn pending_count(&self) -> usize {
        self.pending.iter().filter(|&&idx| {
            matches!(self.nodes[idx].state, ObligationState::Pending)
        }).count()
    }

    /// Get a reference to a node by index.
    pub fn get_node(&self, idx: usize) -> &ObligationNode {
        &self.nodes[idx]
    }

    /// Get a mutable reference to a node by index.
    pub fn get_node_mut(&mut self, idx: usize) -> &mut ObligationNode {
        &mut self.nodes[idx]
    }

    /// Get the obligation at a given index.
    pub fn obligation_at(&self, idx: usize) -> &Obligation {
        &self.nodes[idx].obligation
    }

    /// Get the state of a node by index.
    pub fn state_at(&self, idx: usize) -> &ObligationState {
        &self.nodes[idx].state
    }

    /// Collect all errors from the forest.
    pub fn collect_errors(&self) -> Vec<&SolveError> {
        self.nodes.iter().filter_map(|n| {
            match &n.state {
                ObligationState::Error(e) => Some(e),
                ObligationState::CycleDetected => {
                    // Non-coinductive cycles become errors
                    if !matches!(&n.obligation.predicate,
                        Predicate::AutoTrait { .. } | Predicate::Sized { .. }
                    ) {
                        // This shouldn't happen — cycles are detected in mark_evaluating
                        None
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }).collect()
    }

    /// Compact the forest by removing resolved and errored nodes.
    ///
    /// Called automatically when the node count exceeds `MAX_NODES`.
    /// Keeps only `Pending`, `Evaluating`, and `CycleDetected` nodes.
    /// Rebuilds the `pending` queue and updates parent/child indices.
    /// Also removes `active_path` entries for removed nodes.
    /// This prevents unbounded memory growth in long-running compilation.
    pub fn compact(&mut self) {
        // Pass 1: build the old-to-new index mapping for all surviving nodes.
        let mut old_to_new: Vec<Option<usize>> = vec![None; self.nodes.len()];
        let mut new_idx = 0;
        for (old_idx, node) in self.nodes.iter().enumerate() {
            let keep = match node.state {
                ObligationState::Pending
                | ObligationState::Evaluating
                | ObligationState::CycleDetected
                | ObligationState::Deferred => true,
                ObligationState::Resolved | ObligationState::Error(_) => false,
            };
            if keep {
                old_to_new[old_idx] = Some(new_idx);
                new_idx += 1;
            }
        }

        // Remove stale active_path entries and remap surviving indices.
        let mut new_active_path: HashSet<(usize, DefId, TypeId, u64)> = HashSet::default();
        for (stored_idx, trait_id, self_ty, args_hash) in self.active_path.drain() {
            if let Some(new_idx) = old_to_new.get(stored_idx).and_then(|&o| o) {
                new_active_path.insert((new_idx, trait_id, self_ty, args_hash));
            }
        }
        self.active_path = new_active_path;

        // Pass 2: construct the new node list with the complete mapping.
        let mut new_nodes: Vec<ObligationNode> = Vec::with_capacity(new_idx);
        let mut new_pending: VecDeque<usize> = VecDeque::new();
        for (old_idx, node) in self.nodes.iter().enumerate() {
            if let Some(new_idx) = old_to_new[old_idx] {
                let mut new_node = node.clone();
                new_node.parent = node.parent.and_then(|p| old_to_new[p]);
                new_node.children = node.children.iter().filter_map(|c| old_to_new[*c]).collect();
                new_nodes.push(new_node);
                if matches!(node.state, ObligationState::Pending) {
                    new_pending.push_back(new_idx);
                }
            }
        }

        self.nodes = new_nodes;
        self.pending = new_pending;
    }

    /// Compute the resolved active_path key for a node, resolving inference
    /// variables through the TypeContext so that semantically equivalent
    /// obligations (after unification) are detected as cycles.
    fn resolved_key_for_node(
        &self,
        node: &ObligationNode,
        ctx: &TypeContext,
    ) -> Option<(DefId, TypeId, u64)> {
        match &node.obligation.predicate {
            Predicate::Trait { trait_id, self_ty, args } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                let resolved_args_hash = resolved_args_hash(ctx, args);
                Some((*trait_id, resolved_self, resolved_args_hash))
            }
            Predicate::AutoTrait { trait_id, self_ty } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                Some((*trait_id, resolved_self, 0))
            }
            Predicate::Sized { ty } => {
                let resolved_ty = ctx.resolve_binding(*ty);
                Some((DefId(usize::MAX), resolved_ty, 0))
            }
            _ => None,
        }
    }
}

/// Compute a hash for a slice of TypeIds for cycle detection.
fn args_hash(args: &[TypeId]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    args.len().hash(&mut hasher);
    for arg in args {
        arg.hash(&mut hasher);
    }
    hasher.finish()
}

/// Compute a hash for a slice of TypeIds, resolving each through the
/// TypeContext first so that semantically equivalent args (after unification)
/// produce the same hash.
fn resolved_args_hash(ctx: &TypeContext, args: &[TypeId]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    args.len().hash(&mut hasher);
    for arg in args {
        let resolved = ctx.resolve_binding(*arg);
        resolved.hash(&mut hasher);
    }
    hasher.finish()
}