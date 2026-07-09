use super::*;

/// A scoped guard that restores `current_function` and `current_return_type`
/// on drop, and also exits the inference scope.
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

    pub(crate) fn defuse(mut self) {
        self.should_restore = false;
    }
}

impl<'a, 'tcx> Drop for ScopeGuard<'a, 'tcx> {
    fn drop(&mut self) {
        if self.should_restore {
            self.checker.current_function = self.old_function;
            self.checker.current_return_type = self.old_return;
            self.checker.current_function_trusted = self.old_trusted;
            self.checker.exit_inference_scope().ok();
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
    /// On exit, rolls back TypeContext bindings so the nested scope's
    /// infer var resolutions don't leak into the enclosing scope.
    pub(crate) fn exit_inference_scope(&mut self) -> Result<(), DiagnosticCollector> {
        let mut current = mem::replace(&mut self.infer, self.infer_stack.pop().unwrap_or_default());
        // Collect dirty region ids from the checker's RegionTree into the
        // inference context for generation-based generalization.
        // The checker's RegionTree (from region.rs) tracks scope-level dirty
        // markings; the inference context's InferRegionTree (from infer.rs)
        // tracks type variable regions. We bridge them here by wiring the
        // dirty levels.
        let checker_dirty = self.region_tree.collect_dirty_levels();
        // Mark corresponding regions in the inference context as dirty
        // (currently InferenceContext uses its own InferRegionTree, so
        // we propagate the checker's dirty state to it).
        current.region_tree.mark_current_dirty();
        if let Err(err) = current.solve(self.ctx, self.trait_env, self.symbols) {
            let diag = Diagnostic::error(format!("type inference error: {:?}", err))
                .with_span(Span::new(0, 0));
            self.diagnostics.push(diag);
            self.ctx.rollback_transaction();
            return Err(mem::take(&mut self.diagnostics));
        }
        let _solution = current.finalize(self.ctx);

        // ── Check for unresolved constraints ─────────────────────────
        let unresolved = current.check_unresolved(self.ctx);
        for msg in &unresolved {
            self.diagnostics.push(
                Diagnostic::error(msg)
                    .with_code_str("E030")
                    .with_span(Span::new(0, 0)),
            );
        }
        let has_errors = !unresolved.is_empty();

        // Commit TypeContext bindings — the nested scope's InferVar
        // resolutions produced by finalize() are now stable.
        self.ctx.commit_transaction();

        if has_errors {
            return Err(mem::take(&mut self.diagnostics));
        }

        Ok(())
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
