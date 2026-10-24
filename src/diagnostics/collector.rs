use super::{Diagnostic, EmissionGuarantee};
use crate::diagnostics::emitter::Emitter;
use crate::diagnostics::render::Renderer;
use crate::diagnostics::sink::MakeSink;
use std::borrow::Cow;
use std::io;

/// Global diagnostic context, analogous to rustc's `DiagCtxt`.
///
/// Collects diagnostics and provides utilities for error aggregation.
/// Inspired by Lean 4's `MessageLog`, diagnostics are kept in a single
/// `Vec` partitioned by a `reported_len` index:
///
/// ```text
/// diags: [ reported (0..reported_len) | unreported (reported_len..) ]
/// ```
///
/// - **`unreported`** — fresh diagnostics that have not yet been flushed to
///   the output.  New diagnostics via [`push`](Self::push) always land here
///   (appended to the tail).
/// - **`reported`**  — diagnostics that have already been emitted or
///   checkpointed (e.g. saved to a snapshot for incremental compilation).
///
/// This layout makes [`mark_all_reported`](Self::mark_all_reported) a single
/// index update, and [`into_inner`](Self::into_inner) / [`into_iter`](Self::into_iter)
/// zero-cost (no `extend` move), which matters on the per-function-body-check
/// hot path.  [`drain_unreported`](Self::drain_unreported) is `split_off`
/// (O(n) + one alloc) — it has no callers today, so the trade-off favours the
/// hot `into_inner`.
///
/// `'src` is the lifetime of the source buffer the contained diagnostics may
/// borrow (see [`Diagnostic`]).
#[derive(Debug, Clone)]
pub struct DiagCtxt<'src> {
    /// `[reported (0..reported_len) | unreported (reported_len..)]`.
    diags: Vec<Diagnostic<'src>>,
    reported_len: usize,
    /// If false, warning-level diagnostics are suppressed.
    pub can_emit_warnings: bool,
}

impl<'src> DiagCtxt<'src> {
    pub fn new() -> Self {
        DiagCtxt {
            diags: Vec::new(),
            reported_len: 0,
            can_emit_warnings: true,
        }
    }

    /// Push a diagnostic into `unreported` and return an `EmissionGuarantee`
    /// proving the diagnostic was recorded (inspired by rustc's `ErrorGuaranteed`).
    pub fn push(&mut self, diag: Diagnostic<'src>) -> EmissionGuarantee {
        if diag.is_warning() && !self.can_emit_warnings {
            return EmissionGuarantee::suppressed();
        }
        self.diags.push(diag);
        EmissionGuarantee::emitted()
    }

    /// Create and push an error diagnostic in one step.
    pub fn error(&mut self, msg: impl Into<Cow<'src, str>>) -> EmissionGuarantee {
        self.push(Diagnostic::error(msg))
    }

    /// Create and push a warning diagnostic in one step.
    pub fn warn(&mut self, msg: impl Into<Cow<'src, str>>) -> EmissionGuarantee {
        self.push(Diagnostic::warning(msg))
    }

    /// Create and push an error with a formatted message.
    pub fn error_fmt(&mut self, msg: impl Into<Cow<'src, str>>, code: &str) -> EmissionGuarantee {
        let diag = Diagnostic::error(msg).with_code_str(code);
        self.push(diag)
    }

    /// Extend `unreported` with a batch of diagnostics.
    ///
    /// Every diagnostic goes through [`push`](Self::push), so the
    /// `can_emit_warnings` filter applies to the batch exactly as it does to
    /// single pushes.
    pub fn extend<I: IntoIterator<Item = Diagnostic<'src>>>(&mut self, diags: I) {
        for diag in diags {
            self.push(diag);
        }
    }

    /// Attach `source` to every diagnostic (reported and unreported) that does
    /// not already carry one.  Called at a producer's boundary so downstream
    /// emitters can render snippets without cloning the source buffer.
    pub fn attach_source(&mut self, source: &'src str) {
        for diag in &mut self.diags {
            if diag.source().is_none() {
                diag.set_source(Some(source));
            }
        }
    }

    /// Total diagnostics (reported + unreported).
    pub fn len(&self) -> usize {
        self.diags.len()
    }

    /// True when both lists are empty.
    pub fn is_empty(&self) -> bool {
        self.diags.is_empty()
    }

    /// Iterate over all diagnostics (reported first, then unreported).
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Diagnostic<'src>> {
        self.diags.iter()
    }

    /// Iterate mutably over all diagnostics (reported first, then
    /// unreported) — the feed for [`Emitter::emit_all_iter`], which
    /// humanizes each diagnostic at its render boundary.
    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut Diagnostic<'src>> {
        self.diags.iter_mut()
    }

    /// Humanize every unreported diagnostic in place.
    ///
    /// Inspection paths (`errors()` / `warnings()` / `for_each_*`) read
    /// derived fields that stay empty on a `*_kind`-built diagnostic
    /// until `humanize()` runs.  The emitter does this itself at its
    /// render boundary; call this only when reading those fields
    /// *without* going through an emitter.  Never called by `push` —
    /// that would pay the humanization cost for diagnostics that may
    /// never be emitted.
    pub fn humanize_pending(&mut self) {
        for diag in &mut self.diags[self.reported_len..] {
            diag.humanize();
        }
    }

    /// Consume `self` and return all diagnostics (reported, then unreported).
    pub fn into_inner(self) -> Vec<Diagnostic<'src>> {
        self.diags
    }

    /// Return true if any diagnostic (reported or unreported) is an error.
    pub fn has_errors(&self) -> bool {
        self.diags.iter().any(|d| d.is_error())
    }

    /// Number of fresh (unreported) diagnostics.
    pub fn unreported_len(&self) -> usize {
        self.diags.len() - self.reported_len
    }

    /// Remove the fresh diagnostics pushed since `len` — used to tolerate
    /// expected failures during recursive type resolution (an ADT payload
    /// referencing the ADT itself, which is not yet registered).
    pub fn truncate_unreported(&mut self, len: usize) {
        self.diags.truncate(self.reported_len + len);
    }

    pub fn clear(&mut self) {
        self.diags.clear();
        self.reported_len = 0;
    }

    /// Emit all unreported diagnostics via the given emitter.
    ///
    /// The reported/unreported partition advances only after the emitter
    /// has fully succeeded, so a failed emission leaves the diagnostics
    /// unreported for a later retry.  Note the retry re-renders from the
    /// start of the batch: any output the failed attempt already flushed
    /// (a partial write to a fallible sink) WILL be emitted again —
    /// callers whose sinks can partially flush should tolerate that, not
    /// assume exactly-once output.
    pub fn emit<R: Renderer, M: MakeSink>(&mut self, emitter: &mut Emitter<R, M>) -> io::Result<()> {
        emitter.emit_all(&mut self.diags[self.reported_len..])?;
        self.reported_len = self.diags.len();
        Ok(())
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
        if self.diags.len() > self.reported_len {
            self.diags.last_mut()
        } else {
            None
        }
    }

    // ── reported / unreported management (inspired by Lean 4's MessageLog) ──

    /// Access the reported diagnostics.
    pub fn reported(&self) -> &[Diagnostic<'src>] {
        &self.diags[..self.reported_len]
    }

    /// Access the unreported diagnostics.
    pub fn unreported(&self) -> &[Diagnostic<'src>] {
        &self.diags[self.reported_len..]
    }

    /// True when there are unreported diagnostics.
    pub fn has_unreported(&self) -> bool {
        self.diags.len() > self.reported_len
    }

    /// True when there are unreported errors.
    pub fn has_unreported_errors(&self) -> bool {
        self.diags[self.reported_len..].iter().any(|d| d.is_error())
    }

    /// Move all unreported diagnostics into the reported list.
    ///
    /// After calling this, `unreported()` is empty and `reported()` contains
    /// everything.  Useful after a checkpoint / snapshot in incremental
    /// compilation.
    pub fn mark_all_reported(&mut self) {
        self.reported_len = self.diags.len();
    }

    /// Drain the unreported list and return it.
    pub fn drain_unreported(&mut self) -> Vec<Diagnostic<'src>> {
        self.diags.split_off(self.reported_len)
    }

    /// Drain the reported list and return it.
    pub fn drain_reported(&mut self) -> Vec<Diagnostic<'src>> {
        let mut all = std::mem::take(&mut self.diags);
        let unreported = all.split_off(self.reported_len);
        self.diags = unreported;
        self.reported_len = 0;
        all
    }

    // ── Severity downgrades (inspired by Lean 4's MessageLog) ───────

    /// Downgrade all **unreported** error-level diagnostics to warnings.
    ///
    /// This is the inverse of `--warn-as-error`: it can be used to forgive
    /// certain categories of errors (e.g. in a REPL or after a config change).
    pub fn errors_to_warnings(&mut self) {
        for diag in &mut self.diags[self.reported_len..] {
            if diag.is_error() {
                diag.set_level(super::DiagnosticLevel::Warning);
            }
        }
    }

    /// Downgrade all **unreported** error-level diagnostics to information.
    pub fn errors_to_infos(&mut self) {
        for diag in &mut self.diags[self.reported_len..] {
            if diag.is_error() {
                diag.set_level(super::DiagnosticLevel::Info);
            }
        }
    }

    /// Downgrade all **unreported** warning-level diagnostics to information.
    pub fn warnings_to_infos(&mut self) {
        for diag in &mut self.diags[self.reported_len..] {
            if diag.is_warning() {
                diag.set_level(super::DiagnosticLevel::Info);
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
            diags,
            reported_len: 0,
            can_emit_warnings: true,
        }
    }
}

impl<'src> IntoIterator for DiagCtxt<'src> {
    type Item = Diagnostic<'src>;
    type IntoIter = std::vec::IntoIter<Diagnostic<'src>>;

    fn into_iter(self) -> Self::IntoIter {
        self.diags.into_iter()
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    #[test]
    fn errors_to_warnings_updates_is_error_and_is_warning() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("e"));
        c.errors_to_warnings();
        let d = c.unreported().first().unwrap();
        assert!(!d.is_error());
        assert!(d.is_warning());
    }

    #[test]
    fn mark_all_reported_moves_partition() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.push(Diagnostic::warning("b"));
        assert_eq!(c.reported().len(), 0);
        assert_eq!(c.unreported().len(), 2);
        c.mark_all_reported();
        assert_eq!(c.reported().len(), 2);
        assert!(c.unreported().is_empty());
        assert!(!c.has_unreported());
        // New pushes land after the partition.
        c.push(Diagnostic::error("c"));
        assert_eq!(c.reported().len(), 2);
        assert_eq!(c.unreported().len(), 1);
    }

    #[test]
    fn into_inner_preserves_order() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.mark_all_reported();
        c.push(Diagnostic::warning("b"));
        let all = c.into_inner();
        assert_eq!(all.len(), 2);
        assert!(all[0].is_error());
        assert!(all[1].is_warning());
    }

    #[test]
    fn drain_unreported_leaves_reported() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.mark_all_reported();
        c.push(Diagnostic::warning("b"));
        let drained = c.drain_unreported();
        assert_eq!(drained.len(), 1);
        assert!(drained[0].is_warning());
        assert_eq!(c.reported().len(), 1);
        assert!(c.unreported().is_empty());
    }

    #[test]
    fn truncate_unreported_keeps_reported() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.mark_all_reported();
        c.push(Diagnostic::error("b"));
        c.push(Diagnostic::error("c"));
        let before = c.unreported_len();
        assert_eq!(before, 2);
        c.truncate_unreported(1);
        assert_eq!(c.unreported_len(), 1);
        assert_eq!(c.reported().len(), 1);
    }

    #[test]
    fn extend_applies_warning_filter() {
        let mut c = DiagCtxt::new();
        c.can_emit_warnings = false;
        c.extend(vec![Diagnostic::error("e"), Diagnostic::warning("w")]);
        assert_eq!(c.len(), 1);
        assert_eq!(c.unreported_len(), 1);
        assert!(c.unreported()[0].is_error());
        // Any IntoIterator works, and each item still goes through `push`.
        c.extend(std::iter::once(Diagnostic::warning("w2")));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn emit_advances_partition_and_does_not_duplicate() {
        use crate::diagnostics::emitter::Emitter;
        use crate::diagnostics::renderers::JsonRenderer;
        use crate::diagnostics::sink::DiagnosticSink;
        use crate::diagnostics::DiagOpts;
        use std::cell::RefCell;
        use std::rc::Rc;

        struct SharedSink(Rc<RefCell<String>>);
        impl DiagnosticSink for SharedSink {
            fn write_str(&mut self, s: &str) -> io::Result<()> {
                self.0.borrow_mut().push_str(s);
                Ok(())
            }
        }

        let out = Rc::new(RefCell::new(String::new()));
        let captured = Rc::clone(&out);
        let mut emitter = Emitter::new(
            JsonRenderer::new(false),
            move || SharedSink(Rc::clone(&captured)),
            DiagOpts::default(),
        );
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("boom"));
        c.emit(&mut emitter).unwrap();
        assert!(!c.has_unreported());
        // Nothing new was pushed, so a second emit must not re-render it.
        c.emit(&mut emitter).unwrap();
        let rendered = out.borrow();
        assert_eq!(rendered.matches("boom").count(), 1, "out: {rendered}");
    }
}
