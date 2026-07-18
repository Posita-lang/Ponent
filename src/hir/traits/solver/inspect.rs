//! # Proof Tree Inspection (Diagnostics & Debugging)
//!
//! Analogous to `rustc_next_trait_solver::solve::inspect`.
//! Provides a proof tree that records how each goal was evaluated,
//! which candidates were tried, and what the final result was.
//!
//! The proof tree is normally **not built** (it is a no‑op) to avoid
//! any performance overhead during normal compilation.

use crate::hir::traits::solver::obligation::{ImplSource, Obligation, SolveError};
use crate::hir::types::DefId;

// ── Proof tree types ───────────────────────────────────────────────

/// The kind of goal being evaluated, used for cycle handling and
/// diagnostic classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalKind {
    Trait,
    AutoTrait,
    Sized,
    ProjectionEq,
    ProjectionNormalize,
    CopyLike,
}

/// The result of a single candidate evaluation, recorded in the proof tree.
#[derive(Debug, Clone)]
pub struct CandidateResult {
    pub candidate_idx: usize,
    pub success: bool,
    pub error: Option<SolveError>,
}

/// A single node in the proof tree, representing one goal evaluation.
#[derive(Debug, Clone)]
pub struct GoalNode {
    pub obligation: Obligation,
    pub kind: GoalKind,
    pub trait_id: Option<DefId>,
    pub candidates: Vec<CandidateResult>,
    /// Indices into the builder's flat `nodes` vector.
    pub child_indices: Vec<usize>,
    pub result: Result<ImplSource, SolveError>,
}

// ── Proof tree builder ─────────────────────────────────────────────

/// Builder for constructing a proof tree during goal evaluation.
///
/// Uses a flat `Vec<GoalNode>` internally so that every node — both
/// root and nested — can be addressed by a single stable index.
/// The `roots` field stores the indices of top‑level (root) nodes.
/// Nested nodes reference their children via `child_indices`.
///
/// When `is_noop()` returns `true`, all methods are no‑ops.
pub struct ProofTreeBuilder {
    active: bool,
    /// Flat storage of ALL nodes (roots + nested).
    nodes: Vec<GoalNode>,
    /// Indices of root‑level nodes (for `roots()` accessor).
    roots: Vec<usize>,
    /// Stack of node indices forming the current evaluation path.
    stack: Vec<usize>,
}

impl ProofTreeBuilder {
    pub fn new(active: bool) -> Self {
        ProofTreeBuilder {
            active,
            nodes: Vec::new(),
            roots: Vec::new(),
            stack: Vec::new(),
        }
    }

    pub fn is_noop(&self) -> bool {
        !self.active
    }

    /// Start evaluating a new goal.  Returns the flat index of the new node.
    pub fn push_goal(&mut self, obligation: Obligation, kind: GoalKind) -> Option<usize> {
        if !self.active {
            return None;
        }
        let trait_id = match &obligation.predicate {
            crate::hir::traits::solver::Predicate::Trait { trait_id, .. }
            | crate::hir::traits::solver::Predicate::AutoTrait { trait_id, .. } => Some(*trait_id),
            _ => None,
        };
        let ob_for_error = obligation.clone();
        let idx = self.nodes.len();
        let node = GoalNode {
            obligation,
            kind,
            trait_id,
            candidates: Vec::new(),
            child_indices: Vec::new(),
            result: Err(SolveError::Overflow {
                obligation: Box::new(ob_for_error),
                depth: 0,
            }),
        };
        self.nodes.push(node);

        // If there is a parent on the stack, link this node as a child.
        if let Some(&parent) = self.stack.last() {
            if let Some(p) = self.nodes.get_mut(parent) {
                p.child_indices.push(idx);
            }
        } else {
            // No parent → this is a root node.
            self.roots.push(idx);
        }

        self.stack.push(idx);
        Some(idx)
    }

    /// Record a candidate evaluation result for the current (top‑of‑stack) goal.
    pub fn push_candidate(&mut self, candidate_idx: usize, result: Result<(), SolveError>) {
        if !self.active {
            return;
        }
        let (success, error) = match result {
            Ok(()) => (true, None),
            Err(e) => (false, Some(e)),
        };
        if let Some(&idx) = self.stack.last() {
            if let Some(node) = self.nodes.get_mut(idx) {
                node.candidates.push(CandidateResult {
                    candidate_idx,
                    success,
                    error,
                });
            }
        }
    }

    /// Set the final result of the current goal and pop it from the stack.
    pub fn finish_goal(&mut self, result: Result<ImplSource, SolveError>) {
        if !self.active {
            return;
        }
        if let Some(idx) = self.stack.pop() {
            if let Some(node) = self.nodes.get_mut(idx) {
                node.result = result;
            }
        }
    }

    /// Reconstruct the tree of `GoalNode`s from the flat storage.
    ///
    /// Each root node is expanded recursively by pulling its children
    /// out of the flat `nodes` vector into the nested `GoalNode` structure.
    pub fn roots(&self) -> Vec<GoalNode> {
        self.roots
            .iter()
            .filter_map(|&idx| self.build_subtree(idx))
            .collect()
    }

    /// Recursively build a `GoalNode` subtree from a flat index.
    fn build_subtree(&self, idx: usize) -> Option<GoalNode> {
        let node = self.nodes.get(idx)?;
        let nested: Vec<GoalNode> = node
            .child_indices
            .iter()
            .filter_map(|&child| self.build_subtree(child))
            .collect();
        Some(GoalNode {
            obligation: node.obligation.clone(),
            kind: node.kind,
            trait_id: node.trait_id,
            candidates: node.candidates.clone(),
            child_indices: Vec::new(), // flattened away
            result: node.result.clone(),
        })
    }
}

impl Default for ProofTreeBuilder {
    fn default() -> Self {
        Self::new(false)
    }
}
