use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionId(pub usize);

#[derive(Debug, Clone)]
pub struct Region {
    parent: Option<RegionId>,
    frames: Vec<CtxFrame>,
}

/// Tree of scopes replacing the linear loop_stack.
/// Tracks nested context frames (function, loop, closure, etc.) for
/// break/continue/label resolution.  Dirty tracking for generalization
/// is handled by the inference context's own InferRegionTree.
#[derive(Debug, Clone)]
pub struct RegionTree {
    pub regions: Vec<Region>,
    pub root: RegionId,
    pub current: RegionId,
}

impl RegionTree {
    pub fn new() -> Self {
        let root = Region {
            parent: None,
            frames: Vec::new(),
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

    /// Iterate all frames from current region up to root.
    /// Returns frames in reverse order (innermost first), which is the
    /// same behavior as the old `loop_stack.iter().rev()`.
    pub fn iter_frames_rev(&self) -> RegionFrameIter<'_> {
        RegionFrameIter {
            tree: self,
            current: Some(self.current),
            frame_idx: None,
        }
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
