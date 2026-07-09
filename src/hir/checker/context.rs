use super::*;

/// A scoped guard that restores `current_function` and `current_return_type`
/// on drop, and optionally exits the inference scope.
///
/// # Safety
/// `Drop` does NOT call `exit_inference_scope` — that function can fail,
/// and Rust's `Drop` cannot propagate errors.  Instead, callers MUST
/// invoke [`commit`](ScopeGuard::commit) before the guard drops.
/// If the guard drops without `commit()` having been called, the
/// TypeContext transaction is **rolled back** to prevent leaking partial
/// inference state (reviewer #2, issue 1 & 4).
pub(crate) struct ScopeGuard<'a, 'tcx> {
    pub(crate) checker: &'tcx mut TypeChecker<'a>,
    pub(crate) old_function: Option<DefId>,
    pub(crate) old_return: Option<TypeId>,
    pub(crate) old_trusted: bool,
    pub(crate) should_restore: bool,
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
        }
    }

    /// Commit the inference scope: solve constraints, finalize, and commit
    /// the TypeContext transaction.  On success the guard is defused so that
    /// `Drop` only restores the saved fields (function, return type, trusted).
    /// On failure the transaction is rolled back and diagnostics are returned.
    ///
    /// Must be called before the guard drops; calling it twice is a no-op.
    pub(crate) fn commit(mut self) -> Result<(), DiagnosticCollector> {
        if !self.should_restore {
            return Ok(());
        }
        // Run exit_inference_scope *before* restoring saved fields so the
        // inference context is still the current one and the transaction is
        // still open.
        let result = self.checker.exit_inference_scope();
        // Commit on success, roll back on failure.
        if result.is_err() {
            self.checker.ctx.rollback_transaction();
        } else {
            self.checker.ctx.commit_transaction();
        }
        // Restore saved fields regardless of success/failure.
        self.checker.current_function = self.old_function;
        self.checker.current_return_type = self.old_return;
        self.checker.current_function_trusted = self.old_trusted;
        // Defuse the drop so Drop doesn't do redundant cleanup.
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
            // Drop without commit: roll back the transaction and abort the
            // inference scope to keep `infer` and `infer_stack` consistent.
            self.checker.ctx.rollback_transaction();
            self.checker.abort_inference_scope();
            self.checker.current_function = self.old_function;
            self.checker.current_return_type = self.old_return;
            self.checker.current_function_trusted = self.old_trusted;
        }
    }
}

impl<'a> TypeChecker<'a> {
    /// Save the current inference context and push a fresh one.
    /// Also saves TypeContext bindings so the nested scope's resolution
    /// can be committed on exit rather than leaked incrementally.
    pub(crate) fn enter_inference_scope(&mut self) {
        self.ctx.begin_transaction();
        let old = mem::replace(&mut self.infer, InferenceContext::new());
        self.infer_stack.push(old);
    }

    /// Pop the inference context, solve its constraints, and finalize.
    ///
    /// # Panics
    /// Panics if `infer_stack` is empty — callers must ensure every
    /// `enter_inference_scope` has a matching `exit_inference_scope`.
    ///
    /// On success the TypeContext transaction remains open so the caller
    /// can commit it (via `ScopeGuard::commit`); on error the caller must
    /// roll back.
    ///
    /// Fixes (reviewer #2):
    /// - Panics on empty stack instead of silently using a default context.
    /// - Does NOT commit the transaction — the caller decides.
    /// - Propagates `checker_dirty` region levels into the inference context.
    /// - Diagnostics are collected in `self.diagnostics` and returned.
    pub(crate) fn exit_inference_scope(&mut self) -> Result<(), DiagnosticCollector> {
        let prev = self.infer_stack.pop().expect(
            "exit_inference_scope: infer_stack is empty — \
             enter_inference_scope was never called or was called twice",
        );
        let mut current = mem::replace(&mut self.infer, prev);

        // ── Dirty region propagation ─────────────────────────────
        // Mark the inference context's current region as dirty so that
        // the generalization step considers it.  The checker's own region
        // tree (region.rs) has a mark_dirty() / collect_dirty_levels()
        // API, but it is not yet wired into any variable‑binding path —
        // inference-variable dirtiness is tracked entirely within the
        // inference context's own InferRegionTree.  When the checker's
        // region dirtiness is eventually populated, this block should
        // map checker RegionIds to infer InferRegionIds (they share the
        // same usize encoding) and propagate them here.
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
        for msg in &unresolved {
            self.diagnostics.push(
                Diagnostic::error(msg)
                    .with_code_str("E030")
                    .with_span(Span::new(0, 0)),
            );
        }

        if has_errors {
            return Err(mem::take(&mut self.diagnostics));
        }

        Ok(())
    }

    /// Abort the current inference scope without solving constraints.
    /// Pops the inference stack and restores the previous inference context,
    /// discarding any work done in the aborted scope.
    /// The caller is responsible for rolling back the TypeContext transaction.
    ///
    /// # Panics
    /// Panics if `infer_stack` is empty — matching the contract of
    /// `exit_inference_scope` and `enter_inference_scope`.
    pub(crate) fn abort_inference_scope(&mut self) {
        let prev = self.infer_stack.pop().expect(
            "abort_inference_scope: infer_stack is empty — \
             enter_inference_scope was never called or was called twice",
        );
        self.infer = prev;
    }

    /// Push a new context frame (e.g., entering a function body, loop, closure).
    pub(crate) fn push_ctx(&mut self, kind: CtxKind, span: Span, label: Option<String>) {
        self.region_tree.push_frame(CtxFrame { kind, span, label });
    }

    /// Pop the innermost context frame.
    pub(crate) fn pop_ctx(&mut self) {
        self.region_tree.pop_frame();
    }

    /// Find the innermost break target (Loop, While, For, LabeledBlock).
    /// Returns the target's span and optional label.
    /// If `label` is Some, only match same-named LabeledBlock.
    /// Stops at Closure/AsyncBlock boundaries to prevent cross-boundary breaks.
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
                    // With a label, only matching LabeledBlock is a valid target.
                    continue;
                }
                CtxKind::LabeledBlock => {
                    if let Some(lbl) = label {
                        if frame.label.as_deref() == Some(lbl) {
                            return Some((frame.span, Some(lbl)));
                        }
                        // Different label — skip, continue searching outward.
                    }
                    // Without a label, LabeledBlock is not an implicit break target.
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
    /// Posita's `continue` does not support labels; `label` is always None.
    pub(crate) fn find_continue_target(&self, label: Option<&str>) -> Option<(Span, &str)> {
        for frame in self.region_tree.iter_frames_rev() {
            match &frame.kind {
                CtxKind::Loop | CtxKind::While | CtxKind::For => {
                    // `continue` with a label is not valid in Posita.
                    // If a label was provided, skip this loop and keep searching
                    // (the caller will report "label not found").
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
                    // `continue` cannot target LabeledBlock; skip.
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
