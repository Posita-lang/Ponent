use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionId(usize);

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
    regions: Vec<Region>,
    root: RegionId,
    current: RegionId,
}

impl RegionTree {
    fn new() -> Self {
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

    fn current_frames(&self) -> &[CtxFrame] {
        &self.regions[self.current.0].frames
    }

    fn push_frame(&mut self, frame: CtxFrame) {
        self.regions[self.current.0].frames.push(frame);
    }

    fn pop_frame(&mut self) -> Option<CtxFrame> {
        self.regions[self.current.0].frames.pop()
    }

    fn enter_region(&mut self) -> RegionId {
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

    fn exit_region(&mut self) {
        if let Some(parent) = self.regions[self.current.0].parent {
            self.current = parent;
        }
    }

    /// Iterate all frames from current region up to root.
    /// Returns frames in reverse order (innermost first), which is the
    /// same behavior as the old `loop_stack.iter().rev()`.
    fn iter_frames_rev(&self) -> RegionFrameIter {
        RegionFrameIter { tree: self, current: Some(self.current), frame_idx: None }
    }

    /// Mark the current region as dirty (a unification variable was bound).
    fn mark_dirty(&mut self) {
        self.regions[self.current.0].dirty = true;
    }

    /// Check whether the current region is dirty.
    fn is_dirty(&self) -> bool {
        self.regions[self.current.0].dirty
    }

    /// Clear the dirty flag on the current region.
    fn clean_region(&mut self) {
        self.regions[self.current.0].dirty = false;
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

/// A SCAP-style guarantee describing the state transition from function entry
/// to function exit.  Following Feng & Shao (2006) §4, a guarantee `g` is a
/// relation `State → State → Prop`.  In Posita we track this at the type level
/// as an ordered pair of the ensures-condition (postcondition) and the frame
