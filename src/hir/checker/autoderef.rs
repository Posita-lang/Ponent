use crate::hir::types::TypeId;

/// Default maximum number of dereference steps in the autoderef chain.
pub const DEFAULT_MAX_DEREF_DEPTH: usize = 20;

/// An iterator that walks the autoderef chain of a type.
/// Each call to `next()` attempts to dereference the current type once
/// using built-in deref rules. Stops after `max_depth` steps.
pub struct AutoderefIter<'a, 'input> {
    checker: &'a crate::hir::checker::TypeChecker<'a, 'input>,
    current: Option<TypeId>,
    depth: usize,
    max_depth: usize,
    /// The visited types — a CYCLE GUARD: a `Deref` impl whose Target is
    /// (transitively) its own type (`type Target = Self`) would otherwise
    /// loop the full `max_depth` — the chain stops at the first repeat.
    seen: std::collections::HashSet<TypeId>,
}

impl<'a, 'input> AutoderefIter<'a, 'input> {
    /// Create a new autoderef iterator with a custom max depth.
    pub fn with_max_depth(
        checker: &'a crate::hir::checker::TypeChecker<'a, 'input>,
        ty: TypeId,
        max_depth: usize,
    ) -> Self {
        let mut seen = std::collections::HashSet::new();
        seen.insert(ty);
        AutoderefIter {
            checker,
            current: Some(ty),
            depth: 0,
            max_depth,
            seen,
        }
    }
}

impl<'a, 'input> Iterator for AutoderefIter<'a, 'input> {
    type Item = TypeId;

    fn next(&mut self) -> Option<TypeId> {
        let ty = self.current?;
        if self.depth >= self.max_depth {
            self.current = None;
            return Some(ty);
        }
        self.depth += 1;
        let next = self.checker.builtin_deref_ty(ty);
        // The cycle guard: stop the walk when a type repeats (a
        // self-referential `Deref` Target or a deref cycle between two
        // wrapper types).
        self.current = match next {
            Some(n) if self.seen.insert(n) => Some(n),
            _ => None,
        };
        Some(ty)
    }
}
