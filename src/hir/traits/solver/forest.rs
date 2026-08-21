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
    /// Inference variables that are blocking this obligation from being
    /// resolved.  Populated when the node is marked as `Deferred`.
    /// Used by the fulfillment context to selectively re-evaluate nodes
    /// when inference variables are resolved.
    pub stalled_on: Vec<TypeId>,
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

/// The outcome of [`ObligationForest::mark_evaluating`].
///
/// An enum (not a bool) so the caller's control flow is forced to
/// distinguish the two cases at compile time: a node whose evaluation was
/// ENTERED must be paired with [`leave_evaluating`], while a node that hit
/// a cycle must NOT — the active-path key belongs to the ancestor that
/// inserted it, and a spurious `leave_evaluating` would remove the
/// ancestor's key and corrupt its cycle detection for the rest of its
/// evaluation.
///
/// [`leave_evaluating`]: ObligationForest::leave_evaluating
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnterEvaluation {
    /// Evaluation was entered.  The node's key is on the active path (or
    /// the predicate does not participate in cycle detection, in which
    /// case no key was inserted and `leave_evaluating` is a harmless
    /// no-op).  The caller MUST call `leave_evaluating` when done.
    Entered,
    /// A cycle was detected.  `mark_evaluating` set the node's state to
    /// `CycleDetected` (coinductive) or `Error` (inductive).  The caller
    /// must NOT call `leave_evaluating`.
    CycleDetected,
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
            stalled_on: Vec::new(),
        });
        self.pending.push_back(idx);
        idx
    }

    /// Register a child obligation for a given parent.
    /// Returns the child node index.
    pub fn register_child(&mut self, obligation: Obligation, parent_idx: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ObligationNode {
            obligation,
            state: ObligationState::Pending,
            parent: Some(parent_idx),
            children: Vec::new(),
            stalled_on: Vec::new(),
        });
        self.nodes[parent_idx].children.push(idx);
        self.pending.push_back(idx);
        idx
    }

    /// Get the next pending obligation to process.
    /// Returns `None` if no pending obligations remain.
    ///
    /// Only returns nodes in the `Pending` state.  `Deferred` nodes are
    /// not returned — they must first be recycled back to `Pending` by
    /// [`recycle_ready_deferred`] when their `stalled_on` variables are
    /// resolved.
    pub fn next_pending(&mut self) -> Option<usize> {
        while let Some(idx) = self.pending.pop_front() {
            if matches!(self.nodes[idx].state, ObligationState::Pending) {
                return Some(idx);
            }
            // Skip resolved/error/cycle-detected/deferred nodes.
        }
        None
    }

    /// Peek at the next pending node WITHOUT popping it — for error
    /// reporting paths that must not consume the queue (a popped-but-
    /// unprocessed node would be silently dropped).
    pub fn peek_pending(&self) -> Option<usize> {
        self.pending
            .iter()
            .copied()
            .find(|idx| matches!(self.nodes[*idx].state, ObligationState::Pending))
    }

    /// Move deferred nodes whose `stalled_on` variables have been resolved
    /// (are no longer inference variables) back to the `Pending` state and
    /// push them onto the pending queue for re-evaluation.
    ///
    /// Returns `true` if any node was recycled.
    ///
    /// This is the bridge between the selective re-evaluation tracked by
    /// `stalled_on` and the fulfillment loop: the loop calls this method
    /// before checking for progress, so that ready deferred nodes are
    /// picked up by `next_pending`.
    pub fn recycle_ready_deferred<'input>(&mut self, ctx: &TypeContext<'input>) -> bool {
        let mut any_ready = false;
        for node in &mut self.nodes {
            if matches!(node.state, ObligationState::Deferred)
                && node.stalled_on.iter().any(|&ty| !ctx.is_infer_var(ty))
            {
                node.state = ObligationState::Pending;
                any_ready = true;
            }
        }
        if any_ready {
            // Re-queue all nodes that are now Pending (including those
            // just recycled and any that were already Pending).
            for (idx, node) in self.nodes.iter().enumerate() {
                if matches!(node.state, ObligationState::Pending) {
                    self.pending.push_back(idx);
                }
            }
        }
        any_ready
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
    /// `stalled_on` records which inference variables are blocking resolution.
    pub fn mark_deferred(&mut self, idx: usize, stalled_on: Vec<TypeId>) {
        self.nodes[idx].state = ObligationState::Deferred;
        self.nodes[idx].stalled_on = stalled_on;
        self.pending.push_back(idx);
    }

    /// Count the number of deferred nodes in the forest.
    pub fn deferred_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n.state, ObligationState::Deferred))
            .count()
    }

    /// The first deferred obligation (in registration order), for reporting
    /// the obligation that could not be resolved after the final pass.
    pub fn first_deferred(&self) -> Option<&Obligation> {
        self.nodes
            .iter()
            .find(|n| matches!(n.state, ObligationState::Deferred))
            .map(|n| &n.obligation)
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
    ///
    /// Returns [`EnterEvaluation::Entered`] when the caller owns evaluation
    /// and MUST call `leave_evaluating` afterwards.  Returns
    /// [`EnterEvaluation::CycleDetected`] when the node's key is already on
    /// the active path (inserted by an ancestor) — the node's state has been
    /// set to `CycleDetected` (coinductive) or `Error` (inductive), and the
    /// caller must NOT call `leave_evaluating` (that would remove the
    /// ancestor's key).
    ///
    /// `coinductive_trait` reports whether the node's trait (for
    /// `Predicate::Trait` goals) is a user-declared coinductive trait
    /// (`@coinductive`).  The caller computes it because the forest has no
    /// access to the symbol table.
    pub fn mark_evaluating<'input>(
        &mut self,
        idx: usize,
        ctx: &TypeContext<'input>,
        coinductive_trait: bool,
    ) -> EnterEvaluation {
        let node = &self.nodes[idx];

        // Compute the resolved cycle key (following bindings through inference vars).
        let Some(resolved_key) = self.resolved_key_for_node(node, ctx) else {
            // Other predicates don't participate in cycle detection
            return EnterEvaluation::Entered;
        };

        // Check for cycles using resolved keys: two obligations form a cycle
        // when they have the same trait_id, resolved self_ty, and resolved args.
        // This catches the case where an inference variable was unified during
        // evaluation of a parent, making two syntactically different obligations
        // semantically identical.
        let is_cycle = self
            .active_path
            .iter()
            .any(|(_, t, s, a)| (*t, *s, *a) == resolved_key);

        if is_cycle {
            // Cycle detected
            let is_coinductive = matches!(
                &node.obligation.predicate,
                Predicate::AutoTrait { .. } | Predicate::Sized { .. }
            ) || coinductive_trait;
            if is_coinductive {
                // Coinductive cycles are ok (e.g., Send: Send)
                self.nodes[idx].state = ObligationState::CycleDetected;
                EnterEvaluation::CycleDetected
            } else {
                // Non-coinductive cycle is an error
                self.nodes[idx].state = ObligationState::Error(SolveError::CycleDetected {
                    predicate: node.obligation.predicate.clone(),
                });
                EnterEvaluation::CycleDetected
            }
        } else {
            let (trait_id, self_ty, args_hash) = resolved_key;
            self.active_path.insert((idx, trait_id, self_ty, args_hash));
            self.nodes[idx].state = ObligationState::Evaluating;
            EnterEvaluation::Entered
        }
    }

    /// Remove a node from the active path (after evaluation completes).
    /// Uses the node index for deterministic removal, avoiding the problem
    /// of a key changing when inference variables are unified during evaluation.
    ///
    /// Note: this is O(n) in the active path size, but the active path is
    /// bounded by the obligation nesting depth (typically < 10).
    pub fn leave_evaluating(&mut self, idx: usize) {
        self.active_path
            .retain(|(stored_idx, _, _, _)| *stored_idx != idx);
    }

    /// Check if there are still pending obligations.
    pub fn has_pending(&self) -> bool {
        self.pending
            .iter()
            .any(|&idx| matches!(self.nodes[idx].state, ObligationState::Pending))
    }

    /// Check if any deferred node has a resolved `stalled_on` variable.
    ///
    /// When a deferred node's blocking inference variable gets bound to a
    /// concrete type, the node is ready for re-evaluation.  The fulfillment
    /// loop uses this to avoid stalling while there is still progress to
    /// be made by re-evaluating unblocked deferred nodes.
    pub fn has_ready_deferred<'input>(&self, ctx: &TypeContext<'input>) -> bool {
        self.nodes.iter().any(|n| {
            matches!(n.state, ObligationState::Deferred)
                && n.stalled_on.iter().any(|&ty| !ctx.is_infer_var(ty))
        })
    }

    /// Get the number of obligations (including resolved ones).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of pending obligations.
    pub fn pending_count(&self) -> usize {
        self.pending
            .iter()
            .filter(|&&idx| matches!(self.nodes[idx].state, ObligationState::Pending))
            .count()
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
        self.nodes
            .iter()
            .filter_map(|n| {
                match &n.state {
                    ObligationState::Error(e) => Some(e),
                    ObligationState::CycleDetected => {
                        // Non-coinductive cycles become errors
                        if !matches!(
                            &n.obligation.predicate,
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
            })
            .collect()
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
                new_node.children = node
                    .children
                    .iter()
                    .filter_map(|c| old_to_new[*c])
                    .collect();
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
    /// variables through the TypeContext<'input> so that semantically equivalent
    /// obligations (after unification) are detected as cycles.
    fn resolved_key_for_node<'input>(
        &self,
        node: &ObligationNode,
        ctx: &TypeContext<'input>,
    ) -> Option<(DefId, TypeId, u64)> {
        match &node.obligation.predicate {
            Predicate::Trait {
                trait_id,
                self_ty,
                args,
            } => {
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
/// TypeContext<'input> first so that semantically equivalent args (after unification)
/// produce the same hash.
fn resolved_args_hash<'input>(ctx: &TypeContext<'input>, args: &[TypeId]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    args.len().hash(&mut hasher);
    for arg in args {
        let resolved = ctx.resolve_binding(*arg);
        resolved.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::traits::solver::obligation::{ObligationCause, ObligationCauseCode};

    fn sized_obligation(ty: TypeId) -> Obligation {
        Obligation {
            cause: ObligationCause {
                span: crate::ast::DUMMY_SPAN,
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Sized { ty },
            recursion_depth: 0,
        }
    }

    fn trait_obligation(trait_id: crate::hir::types::DefId, self_ty: TypeId) -> Obligation {
        Obligation {
            cause: ObligationCause {
                span: crate::ast::DUMMY_SPAN,
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Trait {
                trait_id,
                self_ty,
                args: vec![],
            },
            recursion_depth: 0,
        }
    }

    /// A `Predicate::Trait` cycle is classified by the caller-supplied
    /// `coinductive_trait` flag: `true` (a user trait declared `@coinductive`)
    /// leaves the node in `CycleDetected` state, `false` records an inductive
    /// `CycleDetected` error.
    #[test]
    fn test_mark_evaluating_user_coinductive_trait() {
        let mut ctx = TypeContext::new();
        let mut forest = ObligationForest::new();
        let int_ty = ctx.int(32, true);

        let a = forest.register(trait_obligation(crate::hir::types::DefId(42), int_ty));
        assert_eq!(
            forest.mark_evaluating(a, &ctx, true),
            EnterEvaluation::Entered,
            "first entry of a key must be Entered"
        );

        let b = forest.register(trait_obligation(crate::hir::types::DefId(42), int_ty));
        assert_eq!(
            forest.mark_evaluating(b, &ctx, true),
            EnterEvaluation::CycleDetected,
            "a nested same-key trait obligation must be CycleDetected"
        );
        assert!(
            matches!(forest.state_at(b), ObligationState::CycleDetected),
            "an @coinductive trait cycle must leave the node in CycleDetected state"
        );

        let c = forest.register(trait_obligation(crate::hir::types::DefId(42), int_ty));
        assert_eq!(
            forest.mark_evaluating(c, &ctx, false),
            EnterEvaluation::CycleDetected,
            "a nested same-key trait obligation must be CycleDetected"
        );
        assert!(
            matches!(
                forest.state_at(c),
                ObligationState::Error(SolveError::CycleDetected { .. })
            ),
            "an inductive trait cycle must record a CycleDetected error"
        );

        forest.leave_evaluating(a);
    }

    /// Pins the `EnterEvaluation` contract: only an `Entered` node owns an
    /// active-path key and may call `leave_evaluating`; a `CycleDetected`
    /// node must NOT (the key belongs to the ancestor).  A spurious
    /// `leave_evaluating` after a cycle would corrupt the ancestor's cycle
    /// detection for the rest of its evaluation.
    #[test]
    fn test_mark_evaluating_enter_leave_contract() {
        let mut ctx = TypeContext::new();
        let mut forest = ObligationForest::new();
        let int_ty = ctx.int(32, true);

        // First entry of the key: Entered — owns the active-path key.
        let a = forest.register(sized_obligation(int_ty));
        assert_eq!(
            forest.mark_evaluating(a, &ctx, false),
            EnterEvaluation::Entered,
            "first entry of a key must be Entered"
        );

        // Nested same-key obligations: CycleDetected — the caller must NOT
        // call leave_evaluating, and the node state records the cycle.
        let b = forest.register(sized_obligation(int_ty));
        assert_eq!(
            forest.mark_evaluating(b, &ctx, false),
            EnterEvaluation::CycleDetected,
            "a nested same-key obligation must be CycleDetected"
        );
        assert!(
            matches!(forest.state_at(b), ObligationState::CycleDetected),
            "coinductive cycle must leave the node in CycleDetected state"
        );

        // The ancestor's key must still be owned by the ancestor: a third
        // same-key obligation is STILL a cycle (the key was not removed by
        // the intermediate node's non-leave).
        let c = forest.register(sized_obligation(int_ty));
        assert_eq!(
            forest.mark_evaluating(c, &ctx, false),
            EnterEvaluation::CycleDetected,
            "the ancestor's key must survive the intermediate node (no spurious leave)"
        );

        // Once the ancestor leaves, the key is free again.
        forest.leave_evaluating(a);
        let d = forest.register(sized_obligation(int_ty));
        assert_eq!(
            forest.mark_evaluating(d, &ctx, false),
            EnterEvaluation::Entered,
            "the key must be re-entered after the ancestor leaves"
        );
        forest.leave_evaluating(d);
    }

    /// A non-cycle-participant predicate (e.g. `Eq`) is `Entered` without
    /// inserting a key, so `leave_evaluating` afterwards is a harmless
    /// no-op — the pair must not disturb a real ancestor's key.
    #[test]
    fn test_mark_evaluating_non_participant_leave_is_noop() {
        let mut ctx = TypeContext::new();
        let mut forest = ObligationForest::new();
        let int_ty = ctx.int(32, true);

        let a = forest.register(sized_obligation(int_ty));
        assert_eq!(
            forest.mark_evaluating(a, &ctx, false),
            EnterEvaluation::Entered
        );

        // Eq predicates do not participate in cycle detection.
        let eq = Obligation {
            cause: ObligationCause {
                span: crate::ast::DUMMY_SPAN,
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Eq {
                a: int_ty,
                b: int_ty,
            },
            recursion_depth: 0,
        };
        let e = forest.register(eq);
        assert_eq!(
            forest.mark_evaluating(e, &ctx, false),
            EnterEvaluation::Entered,
            "non-participant predicates enter without a cycle check"
        );
        // leave_evaluating on the non-participant must NOT remove the
        // ancestor's key.
        forest.leave_evaluating(e);

        let c = forest.register(sized_obligation(int_ty));
        assert_eq!(
            forest.mark_evaluating(c, &ctx, false),
            EnterEvaluation::CycleDetected,
            "the ancestor's key must survive a non-participant leave"
        );
        forest.leave_evaluating(a);
    }
}
