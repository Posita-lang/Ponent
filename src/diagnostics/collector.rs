use super::{Diagnostic, EmissionGuarantee};

/// Global diagnostic context, analogous to rustc's `DiagCtxt`.
///
/// Collects diagnostics and provides utilities for error aggregation.
/// Inspired by Lean 4's `MessageLog`, diagnostics are split into two lists:
///
/// - **`unreported`** — fresh diagnostics that have not yet been flushed to
///   the output.  New diagnostics via [`push`](Self::push) always land here.
/// - **`reported`**  — diagnostics that have already been emitted or
///   checkpointed (e.g. saved to a snapshot for incremental compilation).
///
/// This separation enables incremental workflows: after a re-check, only
/// unreported diagnostics need to be displayed, avoiding the re-emission of
/// already-known errors.
///
/// `'src` is the lifetime of the source buffer the contained diagnostics may
/// borrow (see [`Diagnostic`]).
#[derive(Debug, Clone)]
pub struct DiagCtxt<'src> {
    /// Diagnostics already emitted or checkpointed.
    reported: Vec<Diagnostic<'src>>,
    /// Fresh diagnostics not yet emitted.
    unreported: Vec<Diagnostic<'src>>,
    /// If false, warning-level diagnostics are suppressed.
    pub can_emit_warnings: bool,
}

impl<'src> DiagCtxt<'src> {
    pub fn new() -> Self {
        DiagCtxt {
            reported: Vec::new(),
            unreported: Vec::new(),
            can_emit_warnings: true,
        }
    }

    /// Push a diagnostic into `unreported` and return an `EmissionGuarantee`
    /// proving the diagnostic was recorded (inspired by rustc's `ErrorGuaranteed`).
    pub fn push(&mut self, diag: Diagnostic<'src>) -> EmissionGuarantee {
        if diag.is_warning() && !self.can_emit_warnings {
            return EmissionGuarantee::suppressed();
        }
        self.unreported.push(diag);
        EmissionGuarantee::emitted()
    }

    /// Create and push an error diagnostic in one step.
    pub fn error(&mut self, msg: impl Into<String>) -> EmissionGuarantee {
        self.push(Diagnostic::error(msg))
    }

    /// Create and push a warning diagnostic in one step.
    pub fn warn(&mut self, msg: impl Into<String>) -> EmissionGuarantee {
        self.push(Diagnostic::warning(msg))
    }

    /// Create and push an error with a formatted message.
    pub fn error_fmt(&mut self, msg: impl Into<String>, code: &str) -> EmissionGuarantee {
        let diag = Diagnostic::error(msg).with_code_str(code);
        self.push(diag)
    }

    /// Extend `unreported` with a batch of diagnostics.
    pub fn extend(&mut self, diags: Vec<Diagnostic<'src>>) {
        self.unreported.extend(diags);
    }

    /// Attach `source` to every diagnostic (reported and unreported) that does
    /// not already carry one.  Called at a producer's boundary so downstream
    /// emitters can render snippets without cloning the source buffer.
    pub fn attach_source(&mut self, source: &'src str) {
        for diag in self.reported.iter_mut().chain(self.unreported.iter_mut()) {
            if diag.source().is_none() {
                diag.set_source(Some(source));
            }
        }
    }

    /// Total diagnostics (reported + unreported).
    pub fn len(&self) -> usize {
        self.reported.len() + self.unreported.len()
    }

    /// True when both lists are empty.
    pub fn is_empty(&self) -> bool {
        self.reported.is_empty() && self.unreported.is_empty()
    }

    /// Iterate over all diagnostics (reported first, then unreported).
    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic<'src>> {
        self.reported.iter().chain(self.unreported.iter())
    }

    /// Consume `self` and return all diagnostics (reported, then unreported).
    pub fn into_inner(self) -> Vec<Diagnostic<'src>> {
        let mut all = self.reported;
        all.extend(self.unreported);
        all
    }

    /// Return true if any diagnostic (reported or unreported) is an error.
    pub fn has_errors(&self) -> bool {
        self.reported.iter().any(|d| d.is_error()) || self.unreported.iter().any(|d| d.is_error())
    }

    /// Clear all diagnostics.
    pub fn unreported_len(&self) -> usize {
        self.unreported.len()
    }

    /// Remove the fresh diagnostics pushed since `len` — used to tolerate
    /// expected failures during recursive type resolution (an ADT payload
    /// referencing the ADT itself, which is not yet registered).
    pub fn truncate_unreported(&mut self, len: usize) {
        self.unreported.truncate(len);
    }

    pub fn clear(&mut self) {
        self.reported.clear();
        self.unreported.clear();
    }

    /// Emit all unreported diagnostics via the given emitter.
    pub fn emit(&self, emitter: &mut dyn super::emitter::DiagnosticEmitter) {
        emitter.emit_all(&self.unreported);
    }

    /// Drain all diagnostics and return them as a Result.
    /// Returns Ok(()) if no errors, Err(self) if any errors present.
    #[must_use]
    pub fn check(self) -> Result<(), Self> {
        if self.has_errors() { Err(self) } else { Ok(()) }
    }

    // ── Accessors ───────────────────────────────────────────────────

    /// All error-level diagnostics (reported + unreported).
    pub fn errors(&self) -> Vec<&Diagnostic<'src>> {
        self.iter().filter(|d| d.is_error()).collect()
    }

    /// All warning-level diagnostics (reported + unreported).
    pub fn warnings(&self) -> Vec<&Diagnostic<'src>> {
        self.iter().filter(|d| d.is_warning()).collect()
    }

    /// Iterate over all error-level diagnostics without allocating.
    pub fn for_each_error<F: FnMut(&Diagnostic<'src>)>(&self, mut f: F) {
        self.iter().filter(|d| d.is_error()).for_each(|d| f(d))
    }

    /// Iterate over all warning-level diagnostics without allocating.
    pub fn for_each_warning<F: FnMut(&Diagnostic<'src>)>(&self, mut f: F) {
        self.iter().filter(|d| d.is_warning()).for_each(|d| f(d))
    }

    /// Iterate over all diagnostics, yielding them with 1-based index.
    pub fn enumerate(&self) -> impl Iterator<Item = (usize, &Diagnostic<'src>)> {
        self.iter().enumerate().map(|(i, d)| (i + 1, d))
    }

    /// Total count of error diagnostics.
    pub fn error_count(&self) -> usize {
        self.iter().filter(|d| d.is_error()).count()
    }

    /// Total count of warning diagnostics.
    pub fn warning_count(&self) -> usize {
        self.iter().filter(|d| d.is_warning()).count()
    }

    /// Get a mutable reference to the last **unreported** diagnostic, if any.
    pub fn last_mut(&mut self) -> Option<&mut Diagnostic<'src>> {
        self.unreported.last_mut()
    }

    // ── reported / unreported management (inspired by Lean 4's MessageLog) ──

    /// Access the reported diagnostics.
    pub fn reported(&self) -> &[Diagnostic<'src>] {
        &self.reported
    }

    /// Access the unreported diagnostics.
    pub fn unreported(&self) -> &[Diagnostic<'src>] {
        &self.unreported
    }

    /// True when there are unreported diagnostics.
    pub fn has_unreported(&self) -> bool {
        !self.unreported.is_empty()
    }

    /// True when there are unreported errors.
    pub fn has_unreported_errors(&self) -> bool {
        self.unreported.iter().any(|d| d.is_error())
    }

    /// Move all unreported diagnostics into the reported list.
    ///
    /// After calling this, `unreported()` is empty and `reported()` contains
    /// everything.  Useful after a checkpoint / snapshot in incremental
    /// compilation.
    pub fn mark_all_reported(&mut self) {
        let mut pending = std::mem::take(&mut self.unreported);
        self.reported.append(&mut pending);
    }

    /// Drain the unreported list and return it.
    pub fn drain_unreported(&mut self) -> Vec<Diagnostic<'src>> {
        std::mem::take(&mut self.unreported)
    }

    /// Drain the reported list and return it.
    pub fn drain_reported(&mut self) -> Vec<Diagnostic<'src>> {
        std::mem::take(&mut self.reported)
    }

    // ── Severity downgrades (inspired by Lean 4's MessageLog) ───────

    /// Downgrade all **unreported** error-level diagnostics to warnings.
    ///
    /// This is the inverse of `--warn-as-error`: it can be used to forgive
    /// certain categories of errors (e.g. in a REPL or after a config change).
    pub fn errors_to_warnings(&mut self) {
        for diag in &mut self.unreported {
            if diag.is_error() {
                diag.level = super::DiagnosticLevel::Warning;
            }
        }
    }

    /// Downgrade all **unreported** error-level diagnostics to information.
    pub fn errors_to_infos(&mut self) {
        for diag in &mut self.unreported {
            if diag.is_error() {
                diag.level = super::DiagnosticLevel::Info;
            }
        }
    }

    /// Downgrade all **unreported** warning-level diagnostics to information.
    pub fn warnings_to_infos(&mut self) {
        for diag in &mut self.unreported {
            if diag.is_warning() {
                diag.level = super::DiagnosticLevel::Info;
            }
        }
    }
}

impl Default for DiagCtxt<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'src> From<Vec<Diagnostic<'src>>> for DiagCtxt<'src> {
    fn from(diags: Vec<Diagnostic<'src>>) -> Self {
        DiagCtxt {
            reported: Vec::new(),
            unreported: diags,
            can_emit_warnings: true,
        }
    }
}

impl<'src> IntoIterator for DiagCtxt<'src> {
    type Item = Diagnostic<'src>;
    type IntoIter = std::vec::IntoIter<Diagnostic<'src>>;

    fn into_iter(self) -> Self::IntoIter {
        let mut all = self.reported;
        all.extend(self.unreported);
        all.into_iter()
    }
}
