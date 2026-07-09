use crate::hir::guard_set::GuardSet;
use rustc_hash::FxHashMap as HashMap;
use std::collections::HashSet;

/// Unique identifier for a region node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub usize);

/// A pool of types belonging to a region (OmniML generalization.ml Pool).
/// Each region tracks the types and rigid variables that belong to it.
#[derive(Debug, Clone)]
pub struct Pool {
    /// Inference variable indices that belong to this region.
    pub var_ids: Vec<usize>,
    /// Rigid (skolem) variable indices that belong to this region.
    pub rigid_var_ids: Vec<usize>,
}

impl Pool {
    pub fn new() -> Self {
        Pool {
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
        !self.var_ids.is_empty()
    }
}

/// A node in the region tree.
#[derive(Debug, Clone)]
pub struct RegionNode {
    pub id: RegionId,
    pub level: usize,
    pub parent: Option<RegionId>,
    pub children: Vec<RegionId>,
    pub pool: Pool,
    /// Dirty state (OmniML Tree.With_dirty): set when a variable in
    /// this region is modified via unification. The dirty tree tracks
    /// which regions need generalization processing.
    pub dirty: bool,
    /// Sub-dirty set: children that are dirty (only meaningful when
    /// `dirty` is true, matching Tree.With_dirty.Dirty.children).
    pub dirty_children: Vec<RegionId>,
}

/// Region tree replacing the linear level system (OmniML §6, tree.ml).
///
/// The paper explicitly states: "levels no longer uniquely determine
/// a variable's region" — a tree is required when PG variables cause
/// regions to stay alive while other branches create new sibling regions
/// at the same level.
///
/// This implementation follows the OmniML reference:
/// - `Tree` (tree.ml:39): root node
/// - `Node` (tree.ml:22-37): id, level, parent, value (Pool)
/// - `With_dirty` (tree.ml:77-187): mark_dirty, drain_dirty
#[derive(Debug, Clone)]
pub struct RegionTree {
    pub nodes: Vec<RegionNode>,
    pub root: RegionId,
    pub current: RegionId,
}

impl RegionTree {
    pub fn new() -> Self {
        let root = RegionNode {
            id: RegionId(0),
            level: 0,
            parent: None,
            children: Vec::new(),
            pool: Pool::new(),
            dirty: false,
            dirty_children: Vec::new(),
        };
        RegionTree {
            nodes: vec![root],
            root: RegionId(0),
            current: RegionId(0),
        }
    }

    /// Enter a new child region. Returns the new RegionId.
    pub fn enter_region(&mut self) -> RegionId {
        let new_id = RegionId(self.nodes.len());
        let current_level = self.nodes[self.current.0].level;
        self.nodes.push(RegionNode {
            id: new_id,
            level: current_level + 1,
            parent: Some(self.current),
            children: Vec::new(),
            pool: Pool::new(),
            dirty: false,
            dirty_children: Vec::new(),
        });
        self.nodes[self.current.0].children.push(new_id);
        let old = self.current;
        self.current = new_id;
        old
    }

    /// Exit the current region, returning to the parent.
    pub fn exit_region(&mut self) {
        if let Some(parent) = self.nodes[self.current.0].parent {
            self.current = parent;
        }
    }

    /// Get the level of a region.
    pub fn get_level(&self, region_id: RegionId) -> usize {
        self.nodes[region_id.0].level
    }

    /// Compute the nearest common ancestor (LCA) of two regions
    /// by walking up the tree, comparing by level (OmniML tree.ml:52-66).
    pub fn nearest_common_ancestor(&self, a: RegionId, b: RegionId) -> RegionId {
        if a == b {
            return a;
        }
        let a_node = &self.nodes[a.0];
        let b_node = &self.nodes[b.0];
        if a_node.level < b_node.level {
            // b is deeper: move b up
            let b_parent = self.nodes[b.0]
                .parent
                .expect("all non-root nodes have parents");
            self.nearest_common_ancestor(a, b_parent)
        } else if a_node.level > b_node.level {
            // a is deeper: move a up
            let a_parent = self.nodes[a.0]
                .parent
                .expect("all non-root nodes have parents");
            self.nearest_common_ancestor(a_parent, b)
        } else {
            // Same level: move both up
            debug_assert!(a_node.level > 0, "LCA of root with non-root is impossible");
            let a_parent = self.nodes[a.0]
                .parent
                .expect("all non-root nodes have parents");
            let b_parent = self.nodes[b.0]
                .parent
                .expect("all non-root nodes have parents");
            self.nearest_common_ancestor(a_parent, b_parent)
        }
    }

    /// Check if `ancestor` is an ancestor of `node` (tree.ml:68-70).
    pub fn is_ancestor(&self, ancestor: RegionId, node: RegionId) -> bool {
        self.nearest_common_ancestor(ancestor, node) == ancestor
    }

    /// Mark a region and its ancestors as dirty (tree.ml:144-159).
    /// When a variable in `region` is modified, the region is marked dirty,
    /// and the dirty state propagates upward to the closest dirty ancestor.
    pub fn mark_dirty(&mut self, region_id: RegionId) {
        let node = &mut self.nodes[region_id.0];
        if node.dirty {
            return; // already dirty
        }
        node.dirty = true;

        // Find the closest dirty ancestor and register as its dirty child.
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
    }

    /// Mark the current region as dirty.
    pub fn mark_current_dirty(&mut self) {
        self.mark_dirty(self.current);
    }

    /// Drain dirty regions, calling `before` and `after` callbacks
    /// around each region's processing (tree.ml:161-182).
    /// `f` is called for each dirty region, processing it for generalization.
    pub fn drain_dirty<F>(&mut self, region_id: RegionId, f: &mut F)
    where
        F: FnMut(RegionId, &mut Self),
    {
        self.drain_dirty_node(region_id, f);
    }

    fn drain_dirty_node<F>(&mut self, node_id: RegionId, f: &mut F)
    where
        F: FnMut(RegionId, &mut Self),
    {
        let is_dirty = self.nodes[node_id.0].dirty;
        if !is_dirty {
            return;
        }

        // Process dirty children first (depth-first)
        let children: Vec<RegionId> = self.nodes[node_id.0].dirty_children.clone();
        for child_id in children {
            self.drain_dirty_node(child_id, f);
        }

        // Process this region
        f(node_id, self);

        // After processing, clear dirty state if no more dirty children remain
        if self.nodes[node_id.0].dirty_children.is_empty() {
            self.nodes[node_id.0].dirty = false;
            // Remove self from parent's dirty_children
            if let Some(parent_id) = self.nodes[node_id.0].parent {
                self.nodes[parent_id.0]
                    .dirty_children
                    .retain(|&c| c != node_id);
            }
        }
    }

    /// Drain dirty roots (tree.ml:184-187).
    pub fn drain_dirty_roots<F>(&mut self, f: &mut F)
    where
        F: FnMut(RegionId, &mut Self),
    {
        let root = self.root;
        self.drain_dirty(root, f);
    }

    /// Register an inference variable in the current region's pool.
    pub fn register_var(&mut self, var_id: usize) {
        self.nodes[self.current.0].pool.register_var(var_id);
    }

    /// Register a rigid (skolem) variable in the current region's pool.
    pub fn register_rigid_var(&mut self, var_id: usize) {
        self.nodes[self.current.0].pool.register_rigid_var(var_id);
    }

    /// Collect all dirty regions' IDs (for backwards compatibility).
    pub fn collect_dirty_ids(&self) -> Vec<RegionId> {
        let mut ids = Vec::new();
        for node in &self.nodes {
            if node.dirty {
                ids.push(node.id);
            }
        }
        ids
    }

    /// Collect IDs of all alive regions (regions with non-empty pools).
    pub fn collect_alive_ids(&self) -> Vec<RegionId> {
        let mut ids = Vec::new();
        for node in &self.nodes {
            if node.pool.is_alive() {
                ids.push(node.id);
            }
        }
        ids
    }
}
