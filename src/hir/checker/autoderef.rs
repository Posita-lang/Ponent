use crate::hir::types::TypeId;

/// Default maximum number of dereference steps in the autoderef chain.
pub const DEFAULT_MAX_DEREF_DEPTH: usize = 20;

/// An iterator that walks the autoderef chain of a type.
/// Each call to `next()` attempts to dereference the current type once
/// using built-in deref rules. Stops after `max_depth` steps.
pub struct AutoderefIter<'a> {
    checker: &'a crate::hir::checker::TypeChecker<'a>,
    current: Option<TypeId>,
    depth: usize,
    max_depth: usize,
}

impl<'a> AutoderefIter<'a> {
    /// Create a new autoderef iterator with a custom max depth.
    pub fn with_max_depth(
        checker: &'a crate::hir::checker::TypeChecker<'a>,
        ty: TypeId,
        max_depth: usize,
    ) -> Self {
        AutoderefIter {
            checker,
            current: Some(ty),
            depth: 0,
            max_depth,
        }
    }
}

impl<'a> Iterator for AutoderefIter<'a> {
    type Item = TypeId;

    fn next(&mut self) -> Option<TypeId> {
        let ty = self.current?;
        if self.depth >= self.max_depth {
            self.current = None;
            return Some(ty);
        }
        self.depth += 1;
        self.current = self.checker.builtin_deref_ty(ty);
        Some(ty)
    }
}
