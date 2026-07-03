use crate::hir::types::TypeId;

/// An iterator that walks the autoderef chain of a type.
/// Each call to `next()` attempts to dereference the current type once
/// using built-in deref rules. Stops after `max_depth` steps.
pub struct AutoderefIter<'a> {
    pub checker: &'a crate::hir::checker::TypeChecker<'a>,
    pub current: Option<TypeId>,
    pub depth: usize,
    pub max_depth: usize,
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
