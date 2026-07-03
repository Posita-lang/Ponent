use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionId(pub usize);

#[derive(Debug, Clone)]
pub struct Region {
    id: usize,
    parent: Option<RegionId>,
    children: Vec<RegionId>,
    frames: Vec<CtxFrame>,
    /// Dirty flag (OmniML `With_dirty`): set when a variable in this region
    /// is modified. Generalization only needs to process dirty regions.
    dirty: bool,
}

/// Tree of scopes replacing the linear loop_stack.
/// Enables partial generalization (PG/PI): variables can be generalized
/// per-region, supporting nonlinear let-polymorphism scope management.
#[derive(Debug, Clone)]
pub struct RegionTree {
    pub regions: Vec<Region>,
    pub root: RegionId,
    pub current: RegionId,
}

impl RegionTree {
    pub fn new() -> Self {
        let root = Region {
            id: 0,
            parent: None,
            children: Vec::new(),
            frames: Vec::new(),
            dirty: false,
        };
        let root_id = RegionId(0);
        RegionTree {
            regions: vec![root],
            root: root_id,
            current: root_id,
        }
    }

    pub fn current_frames(&self) -> &[CtxFrame] {
        &self.regions[self.current.0].frames
    }

    pub fn push_frame(&mut self, frame: CtxFrame) {
        self.regions[self.current.0].frames.push(frame);
    }

    pub fn pop_frame(&mut self) -> Option<CtxFrame> {
        self.regions[self.current.0].frames.pop()
    }

    pub fn enter_region(&mut self) -> RegionId {
        let new_id = RegionId(self.regions.len());
        self.regions.push(Region {
            id: new_id.0,
            parent: Some(self.current),
            children: Vec::new(),
            frames: Vec::new(),
            dirty: false,
        });
        self.regions[self.current.0].children.push(new_id);
        let old = self.current;
        self.current = new_id;
        old
    }

    pub fn exit_region(&mut self) {
        if let Some(parent) = self.regions[self.current.0].parent {
            self.current = parent;
        }
    }

    /// Iterate all frames from current region up to root.
    /// Returns frames in reverse order (innermost first), which is the
    /// same behavior as the old `loop_stack.iter().rev()`.
    pub fn iter_frames_rev(&self) -> RegionFrameIter<'_> {
        RegionFrameIter { tree: self, current: Some(self.current), frame_idx: None }
    }

    /// Mark the current region as dirty (a unification variable was bound).
    pub fn mark_dirty(&mut self) {
        self.regions[self.current.0].dirty = true;
    }

    /// Check whether the current region is dirty.
    pub fn is_dirty(&self) -> bool {
        self.regions[self.current.0].dirty
    }

    /// Collect levels of all dirty regions for generalization.
    /// Returns levels sorted descending (innermost first).
    pub fn collect_dirty_levels(&self) -> Vec<usize> {
        let mut levels: Vec<usize> = self.regions.iter()
            .filter(|r| r.dirty)
            .map(|r| r.id)
            .collect();
        levels.sort_by(|a, b| b.cmp(a)); // innermost first
        levels
    }
}

/// Iterator over frames: innermost region's frames first, then parent's, etc.
pub struct RegionFrameIter<'a> {
    tree: &'a RegionTree,
    current: Option<RegionId>,
    frame_idx: Option<usize>,
}

impl<'a> Iterator for RegionFrameIter<'a> {
    type Item = &'a CtxFrame;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let region_id = self.current?;
            let region = &self.tree.regions[region_id.0];
            let idx = self.frame_idx.get_or_insert(region.frames.len());
            if *idx > 0 {
                *idx -= 1;
                return Some(&region.frames[*idx]);
            }
            // Current region's frames exhausted — move to parent
            self.current = region.parent;
            self.frame_idx = None;
        }
    }
}
