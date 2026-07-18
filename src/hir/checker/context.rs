use super::*;

/// A scoped guard that owns the inference-scope lifecycle.
///
/// Responsibilities:
/// - Pops the infer stack on `commit` (before fallible work).
/// - Holds the saved `RegionTree` snapshot so `Drop` can restore it
///   if a panic occurs during solving.
/// - Restores saved fields (`current_function`, …) on drop.
pub(crate) struct ScopeGuard<'a, 'tcx> {
    pub(crate) checker: &'tcx mut TypeChecker<'a>,
    pub(crate) old_function: Option<DefId>,
    pub(crate) old_return: Option<TypeId>,
    pub(crate) old_trusted: bool,
    pub(crate) should_restore: bool,
    /// True after `exit_inference_scope` has been called (infer stack already popped).
    inference_popped: bool,
    /// Saved region tree snapshot — used on panic to discard frames pushed
    /// inside this scope.  Taken from `infer_stack` during `commit` and
    /// stored here before any fallible work runs.
    saved_tree: Option<region::RegionTree>,
}

impl<'a, 'tcx> ScopeGuard<'a, 'tcx> {
    pub(crate) fn new(checker: &'tcx mut TypeChecker<'a>) -> Self {
        let old_function = checker.current_function;
        let old_return = checker.current_return_type;
        let old_trusted = checker.current_function_trusted;
        ScopeGuard {
            checker,
            old_function,
            old_return,
            old_trusted,
            should_restore: true,
            inference_popped: false,
            saved_tree: None,
        }
    }

    /// Commit the inference scope.
    ///
    /// # Panic safety
    /// 1. Pop the infer stack + save the region-tree snapshot *before*
    ///    any fallible work.  `inference_popped` is set immediately after
    ///    the pop so `Drop` never double-pops.
    /// 2. Store the snapshot in `self.saved_tree` so `Drop` can restore
    ///    the region tree if solving panics.
    /// 3. On success discard the snapshot; on error restore it.
    pub(crate) fn commit(mut self) -> Result<(), DiagnosticCollector> {
        if !self.should_restore {
            return Ok(());
        }
        // ── Pop the infer stack ────────────────────────────────
        // SAFETY: enter_inference_scope pushes a pair; we are that pair.
        let (prev, saved_tree) = self.checker.infer_stack.pop().expect(
            "commit: infer_stack is empty — \
             enter_inference_scope was never called",
        );
        let mut current = mem::replace(&mut self.checker.infer, prev);
        self.inference_popped = true;
        // Store the tree snapshot so Drop can restore it on panic.
        self.saved_tree = Some(saved_tree);

        // ── Solve the popped context ───────────────────────────
        // This is the same logic that `exit_inference_scope` used to
        // contain, but without the stack-pop (which is now above).
        let result = self.checker.solve_current_ctx(&mut current);
        // Handle the result *inside* the guard so that saved_tree
        // and should_restore are still accessible.
        match &result {
            Ok(()) => {
                // Success — keep the bindings.
                self.checker.ctx.commit_transaction();
                // Discard the snapshot (region tree is consistent).
                self.saved_tree = None;
            }
            Err(_) => {
                // Inference failed — undo everything.
                self.checker.ctx.rollback_transaction();
                // Restore the region tree to its state at scope entry.
                if let Some(tree) = self.saved_tree.take() {
                    self.checker.region_tree = tree;
                }
            }
        }
        // Restore saved fields regardless of success/failure.
        self.checker.current_function = self.old_function;
        self.checker.current_return_type = self.old_return;
        self.checker.current_function_trusted = self.old_trusted;
        self.should_restore = false;
        result
    }

    pub(crate) fn defuse(mut self) {
        self.should_restore = false;
    }
}

impl<'a, 'tcx> Drop for ScopeGuard<'a, 'tcx> {
    fn drop(&mut self) {
        if self.should_restore {
            // Always roll back the transaction.
            self.checker.ctx.rollback_transaction();
            // Restore the inference context and region tree.
            if self.inference_popped {
                // commit() popped the stack but panicked during solving.
                // Restore the region tree from the saved snapshot.
                if let Some(tree) = self.saved_tree.take() {
                    self.checker.region_tree = tree;
                }
            } else {
                // commit() was never called — normal abort path.
                self.checker.abort_inference_scope();
            }
            self.checker.current_function = self.old_function;
            self.checker.current_return_type = self.old_return;
            self.checker.current_function_trusted = self.old_trusted;
        }
    }
}

impl<'a> TypeChecker<'a> {
    /// Save the current inference context and push a fresh one.
    pub(crate) fn enter_inference_scope(&mut self) {
        self.ctx.begin_transaction();
        let old = mem::replace(&mut self.infer, InferenceContext::new());
        self.infer_stack.push((old, self.region_tree.clone()));
    }

    /// Solve and finalise `ctx` (the inference context that was popped
    /// from the stack by the caller).  Does **not** touch the infer stack
    /// or the region tree — those are the caller's responsibility.
    ///
    /// Returns `Ok(())` on success.  On error the caller must roll back
    /// the transaction and restore the region tree.
    pub(crate) fn solve_current_ctx(
        &mut self,
        current: &mut InferenceContext,
    ) -> Result<(), DiagnosticCollector> {
        // ── Dirty region propagation ─────────────────────────────
        current.region_tree.mark_current_dirty();

        // ── Solve ───────────────────────────────────────────────────
        if let Err(err) = current.solve(self.ctx, self.trait_env, self.symbols) {
            let diag = Diagnostic::error(format!("type inference error: {:?}", err))
                .with_span(Span::new(0, 0));
            self.diagnostics.push(diag);
            return Err(mem::take(&mut self.diagnostics));
        }
        let _solution = current.finalize(self.ctx);

        // ── Check for unresolved constraints ─────────────────────────
        let unresolved = current.check_unresolved(self.ctx);
        let has_errors = !unresolved.is_empty();
        if has_errors {
            for msg in &unresolved {
                self.diagnostics.push(
                    Diagnostic::error(msg)
                        .with_code_str("E030")
                        .with_span(Span::new(0, 0)), // no span available from check_unresolved
                );
            }
            return Err(mem::take(&mut self.diagnostics));
        }

        Ok(())
    }

    /// Abort the current inference scope without solving constraints.
    /// Pops the inference stack and restores the previous inference context
    /// **and** region tree (via the saved snapshot), discarding any work
    /// done in the aborted scope.
    pub(crate) fn abort_inference_scope(&mut self) {
        let (prev, saved_tree) = self.infer_stack.pop().expect(
            "abort_inference_scope: infer_stack is empty — \
             enter_inference_scope was never called or was called twice",
        );
        self.infer = prev;
        self.region_tree = saved_tree;
    }

    /// Push a new context frame (e.g., entering a function body, loop, closure).
    pub(crate) fn push_ctx(&mut self, kind: CtxKind, span: Span, label: Option<String>) {
        self.region_tree.push_frame(CtxFrame { kind, span, label });
    }

    /// Pop the innermost context frame.
    pub(crate) fn pop_ctx(&mut self) {
        self.region_tree.pop_frame();
    }

    /// Push a new scope frame for local variable bindings.
    /// Returns a guard that pops the frame on drop — safe even under `?`.
    pub(crate) fn enter_var_scope(&self) -> VarScopeGuard {
        VarScopeGuard::new(self.local_variable_types.rc_clone())
    }

    /// Find the innermost break target.
    pub(crate) fn find_break_target<'b>(
        &self,
        label: Option<&'b str>,
    ) -> Option<(Span, Option<&'b str>)> {
        for frame in self.region_tree.iter_frames_rev() {
            match &frame.kind {
                CtxKind::Loop | CtxKind::While | CtxKind::For => {
                    if label.is_none() {
                        return Some((frame.span, None));
                    }
                    continue;
                }
                CtxKind::LabeledBlock => {
                    if let Some(lbl) = label {
                        if frame.label.as_deref() == Some(lbl) {
                            return Some((frame.span, Some(lbl)));
                        }
                    }
                }
                CtxKind::Closure | CtxKind::AsyncBlock => {
                    return None;
                }
                _ => {}
            }
        }
        None
    }

    /// Find the innermost continue target (only Loop, While, For).
    pub(crate) fn find_continue_target(&self, label: Option<&str>) -> Option<(Span, &str)> {
        for frame in self.region_tree.iter_frames_rev() {
            match &frame.kind {
                CtxKind::Loop | CtxKind::While | CtxKind::For => {
                    if label.is_some() {
                        continue;
                    }
                    let kind_str = match frame.kind {
                        CtxKind::Loop => "loop",
                        CtxKind::While => "while",
                        CtxKind::For => "for",
                        _ => unreachable!(),
                    };
                    return Some((frame.span, kind_str));
                }
                CtxKind::LabeledBlock => {
                    continue;
                }
                CtxKind::Closure | CtxKind::AsyncBlock => {
                    return None;
                }
                _ => {}
            }
        }
        None
    }
}
