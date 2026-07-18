//! # DepGraph — Dependency Tracking for Incremental Re-evaluation
//!
//! Analogous to `rustc_middle::dep_graph`.
//! Records which query executions read which other query executions,
//! forming a dependency graph.  When an input changes, the graph
//! enables incremental re-evaluation.
//!
//! ## Thread safety
//!
//! - `edges`, `rev_edges`, `dirty` are protected by `RwLock` so that
//!   multiple threads can read the graph concurrently.
//! - `task_stack` is thread-local (`thread_local!`), so each thread
//!   has its own execution context.  This is the same pattern used
//!   by Rust's `DepGraph` for implicit task deps.

use std::collections::VecDeque;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};

use rustc_hash::FxHashSet as HashSet;

// ── DepNodeIndex ─────────────────────────────────────────────────

/// A unique ID for each query execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DepNodeIndex(pub u32);

impl DepNodeIndex {
    pub const ZERO: DepNodeIndex = DepNodeIndex(0);
}

// ── TaskDeps ─────────────────────────────────────────────────────

/// Records which `DepNodeIndex` values are read during a single
/// query's execution.
#[derive(Debug)]
pub struct TaskDeps {
    pub current_node: DepNodeIndex,
    reads: HashSet<DepNodeIndex>,
}

impl TaskDeps {
    pub fn new(current_node: DepNodeIndex) -> Self {
        TaskDeps {
            current_node,
            reads: HashSet::default(),
        }
    }

    pub fn read(&mut self, other: DepNodeIndex) {
        if other != self.current_node {
            self.reads.insert(other);
        }
    }

    pub fn into_reads(self) -> Vec<DepNodeIndex> {
        let mut reads: Vec<_> = self.reads.into_iter().collect();
        reads.sort();
        reads
    }

    pub fn current_node(&self) -> DepNodeIndex {
        self.current_node
    }
}

// ── Thread-local task stack ──────────────────────────────────────

thread_local! {
    /// Per-thread stack of active `TaskDeps`.  Each query thread has its
    /// own execution context, so the task stack is thread-local rather
    /// than shared.  This is the same pattern as Rust's implicit task
    /// deps (`TaskDepsRef` in `dep_graph/mod.rs`).
    static TASK_STACK: std::cell::RefCell<Vec<TaskDeps>> = const { std::cell::RefCell::new(Vec::new()) };
}

// ── DepGraph ─────────────────────────────────────────────────────

/// The dependency graph: records which query execution read which
/// other query executions, and tracks which nodes are dirty.
///
/// Maintains both forward edges (node -> reads) and reverse edges
/// (read -> readers) so that `invalidate()` can find all transitive
/// readers in O(1) per dirty node rather than O(N) per scan.
///
/// Thread-safe: `edges`, `rev_edges`, `dirty` are protected by `RwLock`;
/// `task_stack` is thread-local.
pub struct DepGraph {
    /// The next available `DepNodeIndex` (atomic for thread-safe allocation).
    next_index: AtomicU32,
    /// Forward edges: for each node, the set of nodes it **read**.
    /// Protected by RwLock for concurrent read access.
    edges: RwLock<Vec<HashSet<DepNodeIndex>>>,
    /// Reverse edges: for each node, the set of nodes that **read it**.
    /// Protected by RwLock for concurrent read access.
    rev_edges: RwLock<Vec<HashSet<DepNodeIndex>>>,
    /// Which nodes are dirty (need re-computation).
    /// Protected by RwLock for concurrent read access.
    dirty: RwLock<HashSet<DepNodeIndex>>,
}

impl DepGraph {
    pub fn new() -> Self {
        DepGraph {
            next_index: AtomicU32::new(1),
            edges: RwLock::new(Vec::new()),
            rev_edges: RwLock::new(Vec::new()),
            dirty: RwLock::new(HashSet::default()),
        }
    }

    /// Allocate a new `DepNodeIndex` for a query execution.
    /// Uses `AtomicU32::fetch_add` for thread-safe allocation.
    pub fn allocate_node_index(&self) -> DepNodeIndex {
        let idx = DepNodeIndex(self.next_index.fetch_add(1, Ordering::Relaxed));
        let idx_u = idx.0 as usize;
        let mut edges = self.edges.write().unwrap();
        let mut rev = self.rev_edges.write().unwrap();
        if idx_u >= edges.len() {
            edges.resize_with(idx_u + 1, HashSet::default);
            rev.resize_with(idx_u + 1, HashSet::default);
        }
        idx
    }

    /// Begin recording reads for a query execution.
    /// Push a new `TaskDeps` onto the current thread's task stack.
    pub fn start_task(&self, node: DepNodeIndex) {
        TASK_STACK.with(|s| s.borrow_mut().push(TaskDeps::new(node)));
    }

    /// Record that the current task (top of current thread's stack) read
    /// another query.  If no task is active, this is a no-op.
    pub fn read(&self, other: DepNodeIndex) {
        TASK_STACK.with(|s| {
            if let Some(task) = s.borrow_mut().last_mut() {
                task.read(other);
            }
        });
    }

    /// Get the `DepNodeIndex` of the current task (top of current thread's stack).
    pub fn current_node(&self) -> Option<DepNodeIndex> {
        TASK_STACK.with(|s| s.borrow().last().map(|t| t.current_node))
    }

    /// Finish recording reads for the current task and commit edges.
    pub fn finish_task(&self) {
        let task = TASK_STACK.with(|s| s.borrow_mut().pop());
        if let Some(task) = task {
            let node = task.current_node;
            let reads = task.into_reads();
            let idx = node.0 as usize;
            let mut edges = self.edges.write().unwrap();
            let mut rev = self.rev_edges.write().unwrap();
            if idx >= edges.len() {
                edges.resize_with(idx + 1, HashSet::default);
                rev.resize_with(idx + 1, HashSet::default);
            }
            // Forward edges: node -> reads
            edges[idx].extend(reads.iter().copied());
            // Reverse edges: for each read, record that node reads it
            for &r in &reads {
                let r_idx = r.0 as usize;
                if r_idx < rev.len() {
                    rev[r_idx].insert(node);
                }
            }
        }
    }

    /// Mark a node as dirty and propagate to all transitive readers.
    /// Uses reverse edges for O(1) per dirty node instead of O(N) scan.
    pub fn invalidate(&self, node: DepNodeIndex) {
        let mut dirty = self.dirty.write().unwrap();
        let rev = self.rev_edges.read().unwrap();
        let mut queue = VecDeque::new();
        queue.push_back(node);
        while let Some(current) = queue.pop_front() {
            if dirty.insert(current) {
                if let Some(rev_entry) = rev.get(current.0 as usize) {
                    for &reader in rev_entry {
                        queue.push_back(reader);
                    }
                }
            }
        }
    }

    /// Check whether a node is green (up-to-date, not dirty).
    pub fn is_green(&self, node: DepNodeIndex) -> bool {
        !self.dirty.read().unwrap().contains(&node)
    }

    /// Mark a node as re-computed (green again).
    pub fn mark_green(&self, node: DepNodeIndex) {
        self.dirty.write().unwrap().remove(&node);
    }

    /// Reset the entire graph (for fresh compilation).
    /// All fields use interior mutability, so `&self` suffices.
    pub fn reset(&self) {
        self.next_index.store(1, Ordering::Relaxed);
        self.edges.write().unwrap().clear();
        self.rev_edges.write().unwrap().clear();
        self.dirty.write().unwrap().clear();
        TASK_STACK.with(|s| s.borrow_mut().clear());
    }
}

impl Default for DepGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_chain(dg: &DepGraph) -> (DepNodeIndex, DepNodeIndex, DepNodeIndex) {
        let a = dg.allocate_node_index();
        let b = dg.allocate_node_index();
        let c = dg.allocate_node_index();
        dg.start_task(a);
        dg.read(b);
        dg.finish_task();
        dg.start_task(b);
        dg.read(c);
        dg.finish_task();
        (a, b, c)
    }

    #[test]
    fn test_rev_edges_enables_o1_invalidation() {
        let dg = DepGraph::new();
        let (a, b, c) = setup_chain(&dg);
        dg.invalidate(c);
        assert!(!dg.is_green(c));
        assert!(!dg.is_green(b));
        assert!(!dg.is_green(a));
    }

    #[test]
    fn test_rev_edges_independent_nodes() {
        let dg = DepGraph::new();
        let a = dg.allocate_node_index();
        let b = dg.allocate_node_index();
        dg.start_task(a);
        dg.read(b);
        dg.finish_task();
        dg.invalidate(a);
        assert!(!dg.is_green(a));
        assert!(dg.is_green(b));
    }

    #[test]
    fn test_rev_edges_fan_out() {
        let dg = DepGraph::new();
        let root = dg.allocate_node_index();
        let x = dg.allocate_node_index();
        let y = dg.allocate_node_index();
        let z = dg.allocate_node_index();
        for &reader in &[x, y, z] {
            dg.start_task(reader);
            dg.read(root);
            dg.finish_task();
        }
        dg.invalidate(root);
        assert!(!dg.is_green(x));
        assert!(!dg.is_green(y));
        assert!(!dg.is_green(z));
    }

    #[test]
    fn test_allocate_node_resizes_rev_edges() {
        let dg = DepGraph::new();
        let a = dg.allocate_node_index();
        let b = dg.allocate_node_index();
        assert!(a.0 < b.0);
        assert!(dg.rev_edges.read().unwrap().len() > b.0 as usize);
    }
}
