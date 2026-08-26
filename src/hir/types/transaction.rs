use super::*;

impl<'input> TypeContext<'input> {
    /// Begin a new transaction: push an empty undo log onto the stack.
    /// All subsequent binding changes (via `set_binding`) will be recorded
    /// for potential rollback, without cloning the entire binding table.
    pub fn begin_transaction(&self) {
        self.transaction_stack.borrow_mut().push(Vec::new());
        self.opaque_hidden_undo.borrow_mut().push(Vec::new());
    }

    /// Return the current transaction nesting depth.
    /// 0 means no transaction is active.
    pub fn transaction_depth(&self) -> usize {
        self.transaction_stack.borrow().len()
    }

    /// Commit the current transaction: discard the undo logs.
    pub fn commit_transaction(&self) {
        // Pop the current (innermost) transaction's binding undo log.
        let committed = self.transaction_stack.borrow_mut().pop();
        // Merge its entries into the parent transaction's log so that if
        // the parent later rolls back, it also undoes changes that were
        // committed by the inner transaction.
        if let Some(committed_log) = committed
            && let Some(parent_log) = self.transaction_stack.borrow_mut().last_mut()
        {
            parent_log.extend(committed_log);
        }
        // Merge opaque_hidden undo log into parent as well.
        let committed_opaque = self.opaque_hidden_undo.borrow_mut().pop();
        if let Some(committed_opaque_log) = committed_opaque
            && let Some(parent_log) = self.opaque_hidden_undo.borrow_mut().last_mut()
        {
            parent_log.extend(committed_opaque_log);
        }
        // κ cache may be invalidated by binding changes across transaction boundaries.
        self.kappa_cache.borrow_mut().clear();
        self.variance_cache.borrow_mut().clear();
    }

    /// Rollback the current transaction: reverse-apply every binding change
    /// and opaque_hidden change recorded in this transaction's undo logs.
    /// Roll back to a previously captured transaction depth (pops exactly
    /// the frames opened since `depth`).  Prefer this over balancing
    /// `begin_transaction`/`rollback_transaction` calls by counting.
    pub fn rollback_to(&self, depth: usize) {
        while self.transaction_depth() > depth {
            self.rollback_transaction();
        }
    }

    /// Also clears the unification cache so subsequent attempts re-evaluate.
    /// Note: the types arena (self.types) is NOT truncated — TypeId values
    /// may be held externally, and the arena is logically append-only.
    pub fn rollback_transaction(&self) {
        if let Some(log) = self.transaction_stack.borrow_mut().pop() {
            let mut bindings = self.bindings.borrow_mut();
            for (key, old) in log.into_iter().rev() {
                match old {
                    Some(v) => bindings.insert(key, v),
                    None => bindings.remove(&key),
                };
            }
        }
        // Rollback opaque_hidden changes.
        if let Some(log) = self.opaque_hidden_undo.borrow_mut().pop() {
            let mut opaque = self.opaque_hidden.borrow_mut();
            for (key, old) in log.into_iter().rev() {
                match old {
                    Some(v) => opaque.insert(key, v),
                    None => opaque.remove(&key),
                };
            }
        }
        self.unify_seen.borrow_mut().clear();
        self.kappa_cache.borrow_mut().clear();
        self.variance_cache.borrow_mut().clear();
    }
}
