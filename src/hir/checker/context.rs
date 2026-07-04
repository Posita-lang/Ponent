use super::*;

/// A scoped guard that restores `current_function` and `current_return_type`
/// on drop, and also exits the inference scope.
pub(crate) struct ScopeGuard<'a, 'tcx> {
    pub(crate) checker: &'tcx mut TypeChecker<'a>,
    pub(crate) old_function: Option<DefId>,
    pub(crate) old_return: Option<TypeId>,
    pub(crate) should_restore: bool,
}

impl<'a, 'tcx> ScopeGuard<'a, 'tcx> {
    pub(crate) fn new(checker: &'tcx mut TypeChecker<'a>) -> Self {
        let old_function = checker.current_function;
        let old_return = checker.current_return_type;
        ScopeGuard {
            checker,
            old_function,
            old_return,
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
            self.checker.exit_inference_scope().ok();
        }
    }
}

impl<'a> TypeChecker<'a> {
    /// Save the current inference context and push a fresh one.
    pub(crate) fn enter_inference_scope(&mut self) {
        let old = mem::replace(&mut self.infer, InferenceContext::new());
        self.infer_stack.push(old);
    }

    /// Pop the inference context, solve its constraints, and finalize.
    pub(crate) fn exit_inference_scope(&mut self) -> Result<(), DiagnosticCollector> {
        let mut current = mem::replace(&mut self.infer, self.infer_stack.pop().unwrap_or_default());
        // Wire RegionTree dirty levels into inference context for generation-based generalization.
        current.region_dirty_levels = self.region_tree.collect_dirty_levels();
        if let Err(err) = current.solve(self.ctx, self.trait_env, self.symbols) {
            let diag = Diagnostic::error(format!("type inference error: {:?}", err))
                .with_span(Span::new(0, 0));
            self.diagnostics.push(diag);
            return Err(mem::take(&mut self.diagnostics));
        }
        let _solution = current.finalize(self.ctx);

        // ── Check for unresolved constraints ─────────────────────────
        let unresolved = current.check_unresolved(self.ctx);
        for msg in &unresolved {
            self.diagnostics.push(
                Diagnostic::error(msg)
                    .with_code("E030")
                    .with_span(Span::new(0, 0)),
            );
        }
        if !unresolved.is_empty() {
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
    pub(crate) fn find_continue_target<'b>(
        &self,
        label: Option<&'b str>,
    ) -> Option<(Span, &'b str)> {
        for frame in self.region_tree.iter_frames_rev() {
            match &frame.kind {
                CtxKind::Loop | CtxKind::While | CtxKind::For => {
                    if let Some(lbl) = label {
                        continue;
                    }
                    let kind_str = match frame.kind {
                        CtxKind::Loop => "loop",
                        CtxKind::While => "while",
                        CtxKind::For => "for",
                        _ => panic!(
                            "find_continue_target: unexpected context kind {:?}",
                            frame.kind
                        ),
                    };
                    return Some((frame.span, kind_str));
                }
                CtxKind::LabeledBlock => {
                    if let Some(lbl) = label {
                        if frame.label.as_deref() == Some(lbl) {
                            continue;
                        }
                        continue;
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
}
