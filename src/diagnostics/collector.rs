use super::{Diagnostic, EmissionGuarantee};
use crate::diagnostics::emitter::Emitter;
use crate::diagnostics::render::Renderer;
use crate::diagnostics::sink::MakeSink;
use crate::smallvec::SmallVec;
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
    ///
    /// Inline capacity 2 carries the overwhelmingly common "zero or one
    /// diagnostic per unit" batch on the stack, so constructing a context no
    /// longer heap-allocates up front. `Diagnostic` is large, so the inline
    /// store is kept at 2 rather than widened.
    diags: SmallVec<[Diagnostic<'src>; 2]>,
    reported_len: usize,
    /// Lowest index that may still lack `source`; `attach_source` starts here.
    first_unspanned: usize,
    /// If false, warning-level diagnostics are suppressed.
    /// Push-time gate: when `false`, warning-level diagnostics pushed
    /// *afterwards* are suppressed at `push` (returning a suppressed
    /// [`EmissionGuarantee`]).  This is a "build started, warnings disabled"
    /// compile-time flag: diagnostics already pushed before the flag was
    /// cleared remain in the context and are emitted by `emit`/`emit_all_iter`
    /// regardless.  Setting the flag back to `true` re-enables subsequent
    /// pushes but does not retroactively restore anything.
    can_emit_warnings: bool,
    /// Push-time severity tally over *all* entries (reported + unreported).
    /// `has_errors`, `error_count`, `warning_count`, `summary` and `check`
    /// used to scan every `Diagnostic` per call (`check` runs once per
    /// module), so the tally makes them O(1).  Mutators that add, remove or
    /// change severity must update it; `summary` cross-checks it in debug.
    errors: usize,
    warnings: usize,
    /// Errors in the unreported suffix; kept in lockstep with `errors` so
    /// `has_unreported_errors` is O(1) on the per-module path.
    unreported_errors: usize,
}

/// The inline `SmallVec<[Diagnostic; 2]>` is what keeps a zero/one-diagnostic
/// context on the stack; fail the build if `DiagCtxt` (or `Diagnostic`) grows
/// enough to make that a silent per-context stack cost.
const _: () = assert!(std::mem::size_of::<DiagCtxt<'_>>() <= 1024);

/// Mutable view restricted to the emitter's humanize boundary.  `Deref`
/// exposes read access; obtaining `&mut Diagnostic` requires the explicit
/// `into_diagnostic`, so a future caller cannot change a level through the
/// iterator by accident.
pub(crate) struct HumanizeOnly<'a, 'src>(&'a mut Diagnostic<'src>);

impl<'a, 'src> HumanizeOnly<'a, 'src> {
    pub(crate) fn humanize(&mut self) {
        self.0.humanize();
    }

    /// Explicit escape for the emitter's render path.
    pub(crate) fn into_diagnostic(self) -> &'a mut Diagnostic<'src> {
        self.0
    }
}

impl<'a, 'src> std::ops::Deref for HumanizeOnly<'a, 'src> {
    type Target = Diagnostic<'src>;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'src> DiagCtxt<'src> {
    /// Enable/disable warning emission for **subsequent** pushes.
    ///
    /// Push-time gate: diagnostics already pushed are unaffected, and
    /// re-enabling does not restore suppressed ones. Encapsulated behind a
    /// setter so a future invariant (e.g. forbidding mid-emit toggles) can be
    /// added without a breaking field change.
    pub fn set_can_emit_warnings(&mut self, enabled: bool) {
        self.can_emit_warnings = enabled;
    }
    pub fn new() -> Self {
        DiagCtxt {
            diags: SmallVec::new(),
            reported_len: 0,
            first_unspanned: 0,
            can_emit_warnings: true,
            errors: 0,
            warnings: 0,
            unreported_errors: 0,
        }
    }

    /// Push a diagnostic into `unreported` and return an `EmissionGuarantee`
    /// proving the diagnostic was recorded (inspired by rustc's `ErrorGuaranteed`).
    pub fn push(&mut self, diag: Diagnostic<'src>) -> EmissionGuarantee {
        match diag.level() {
            super::DiagnosticLevel::Fatal | super::DiagnosticLevel::Error => {
                self.errors += 1;
                self.unreported_errors += 1;
            }
            super::DiagnosticLevel::Warning => {
                if !self.can_emit_warnings {
                    return EmissionGuarantee::suppressed();
                }
                self.warnings += 1;
            }
            _ => {}
        }
        let index = self.diags.len();
        debug_assert!(self.first_unspanned <= index);
        let has_source = diag.source().is_some();
        self.diags.push(diag);
        // Invariant: `first_unspanned <= index` before the push, so a new
        // unspanned diag never moves the frontier left.  A spanned diag only
        // advances it when it was exactly the frontier element.
        if has_source && index == self.first_unspanned {
            self.first_unspanned += 1;
        }
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
        let iter = diags.into_iter();
        // `Vec::extend` would reserve from `size_hint`; the manual `push`
        // loop below must do it explicitly or a batch of N reallocates
        // O(log N) times.  A warning-filtered batch may reserve slightly
        // more than it keeps; that is cheaper than growing repeatedly.
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            self.diags.reserve(lower);
        }
        for diag in iter {
            self.push(diag);
        }
    }

    /// Attach `source` to every diagnostic (reported and unreported) that does
    /// not already carry one.  Called at a producer's boundary so downstream
    /// emitters can render snippets without cloning the source buffer.
    ///
    /// The reported region is included on purpose: a caller may re-inspect
    /// already-emitted diagnostics (e.g. to render them elsewhere), and source
    /// attachment is idempotent.  It is not re-rendered in place.
    pub fn attach_source(&mut self, source: &'src str) {
        debug_assert!(self.first_unspanned <= self.diags.len());
        let start = self.first_unspanned.min(self.diags.len());
        for diag in &mut self.diags[start..] {
            if diag.source().is_none() {
                diag.set_source(Some(source));
            }
        }
        self.first_unspanned = self.diags.len();
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
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Diagnostic<'src>> + ExactSizeIterator {
        self.diags.iter()
    }

    /// Iterate mutably over all diagnostics (reported first, then
    /// unreported) — the feed for [`Emitter::emit_all_iter`], which
    /// humanizes each diagnostic at its render boundary.
    pub(crate) fn iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = HumanizeOnly<'_, 'src>> + ExactSizeIterator {
        self.diags.iter_mut().map(HumanizeOnly)
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
    #[must_use]
    pub fn into_inner(self) -> Vec<Diagnostic<'src>> {
        self.diags.into_vec()
    }

    /// Return true if any diagnostic (reported or unreported) is an error.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }

    /// Number of fresh (unreported) diagnostics.
    #[must_use]
    pub fn unreported_len(&self) -> usize {
        debug_assert!(
            self.reported_len <= self.diags.len(),
            "reported_len exceeds diagnostics length"
        );
        self.diags.len().saturating_sub(self.reported_len)
    }

    /// Truncate the unreported list to at most `keep_unreported` items, preserving
    /// the reported prefix.  `keep_unreported` is a *count*, not an absolute
    /// diagnostic index; it is normally obtained from [`Self::unreported_len`].
    pub fn truncate_unreported(&mut self, keep_unreported: usize) {
        debug_assert!(
            keep_unreported <= self.unreported_len(),
            "truncate_unreported: stale count {keep_unreported} > {}",
            self.unreported_len()
        );
        let keep = self.reported_len.saturating_add(keep_unreported);
        if keep < self.diags.len() {
            for d in &self.diags[keep..] {
                if d.is_error() {
                    self.errors = self.errors.saturating_sub(1);
                    self.unreported_errors = self.unreported_errors.saturating_sub(1);
                } else if d.is_warning() {
                    self.warnings = self.warnings.saturating_sub(1);
                }
            }
            self.diags.truncate(keep);
        }
        self.first_unspanned = self.first_unspanned.min(self.diags.len());
    }

    pub fn clear(&mut self) {
        self.diags.clear();
        self.reported_len = 0;
        self.first_unspanned = 0;
        self.errors = 0;
        self.warnings = 0;
        self.unreported_errors = 0;
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
    pub fn emit<R: Renderer, M: MakeSink>(
        &mut self,
        emitter: &mut Emitter<R, M>,
    ) -> io::Result<()> {
        if self.reported_len == self.diags.len() {
            return Ok(());
        }
        emitter.emit_all(&mut self.diags[self.reported_len..])?;
        self.reported_len = self.diags.len();
        self.unreported_errors = 0;
        Ok(())
    }

    /// Drain all diagnostics and return them as a Result.
    /// Returns Ok(()) if no errors, Err(self) if any errors present.
    pub fn check(self) -> Result<(), Self> {
        if self.has_errors() { Err(self) } else { Ok(()) }
    }

    // ── Accessors ───────────────────────────────────────────────────

    /// Iterate over all error-level diagnostics (reported + unreported).
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic<'src>> + '_ {
        self.iter().filter(|d| d.is_error())
    }

    /// Iterate over all warning-level diagnostics (reported + unreported).
    pub fn warnings(&self) -> impl Iterator<Item = &Diagnostic<'src>> + '_ {
        self.iter().filter(|d| d.is_warning())
    }

    /// Iterate over all error-level diagnostics without allocating.
    pub fn for_each_error<F: FnMut(&Diagnostic<'src>)>(&self, mut f: F) {
        self.iter().filter(|d| d.is_error()).for_each(f)
    }

    /// Iterate over all warning-level diagnostics without allocating.
    pub fn for_each_warning<F: FnMut(&Diagnostic<'src>)>(&self, mut f: F) {
        self.iter().filter(|d| d.is_warning()).for_each(f)
    }

    /// Iterate over all diagnostics, yielding them with 1-based index.
    pub fn enumerate(&self) -> impl Iterator<Item = (usize, &Diagnostic<'src>)> {
        self.iter().enumerate().map(|(i, d)| (i + 1, d))
    }

    /// Total count of error diagnostics.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors
    }

    /// Total count of warning diagnostics.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.warnings
    }

    /// One-pass `(error_count, warning_count)`.
    ///
    /// `error_count` + `warning_count` are each a full scan; callers that
    /// need both (summary renderers, tests) can share this single pass
    /// instead of paying two.  `is_error` and `is_warning` are disjoint
    /// (Fatal/Error vs Warning), so the `else if` is exact.  `check` uses the
    /// O(1) `has_errors` tally; `summary` is also O(1) in release and only
    /// pays the debug cross-check below.
    ///
    /// In debug builds this additionally performs an O(n) tally/invariant
    /// cross-check; release builds are the O(1) accessor implied above.
    pub fn summary(&self) -> (usize, usize) {
        // Debug-only invariant check: every mutator that adds/removes/changes
        // a diagnostic must keep the tally exact.  Release pays nothing.
        #[cfg(debug_assertions)]
        {
            let mut e = 0usize;
            let mut w = 0usize;
            for d in &self.diags {
                if d.is_error() {
                    e += 1;
                } else if d.is_warning() {
                    w += 1;
                }
            }
            debug_assert_eq!(
                (self.errors, self.warnings),
                (e, w),
                "DiagCtxt severity tally drifted from the diagnostics"
            );
        }
        (self.errors, self.warnings)
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
        self.unreported_errors > 0
    }

    /// Move all unreported diagnostics into the reported list.
    ///
    /// After calling this, `unreported()` is empty and `reported()` contains
    /// everything.  Useful after a checkpoint / snapshot in incremental
    /// compilation.
    pub fn mark_all_reported(&mut self) {
        self.reported_len = self.diags.len();
        self.unreported_errors = 0;
    }

    /// Recompute every derived tally/frontier from `diags` and `reported_len`.
    /// `iter_mut` is intended to be humanize-only; this runs after the
    /// emitter's pass in release as well, so a caller that changed a level
    /// through the iterator cannot leave `has_errors`/`check()` disagreeing
    /// with the actual diagnostics.
    pub(crate) fn recount_derived_state(&mut self) {
        let mut errors = 0usize;
        let mut warnings = 0usize;
        let mut unreported_errors = 0usize;
        for (i, d) in self.diags.iter().enumerate() {
            if d.is_error() {
                errors += 1;
                if i >= self.reported_len {
                    unreported_errors += 1;
                }
            } else if d.is_warning() {
                warnings += 1;
            }
        }
        self.errors = errors;
        self.warnings = warnings;
        self.unreported_errors = unreported_errors;
        self.first_unspanned = self
            .diags
            .iter()
            .position(|d| d.source().is_none())
            .unwrap_or(self.diags.len());
    }

    /// Drain the unreported list and return it.
    ///
    /// **Retained deliberately** even though no in-tree caller uses it today:
    /// it is part of the public collector API and an external batch/checkpoint
    /// consumer may. The `mem::take` fast path below is pinned by
    /// `drain_unreported_leaves_reported`, so it is not unowned maintenance
    /// debt.
    pub fn drain_unreported(&mut self) -> Vec<Diagnostic<'src>> {
        // `SmallVec` here has no `split_off`, so drain through `into_vec`:
        // the take still avoids a per-element copy, and the split is a Vec
        // move. This path runs at most once per batch.
        let mut all = std::mem::take(&mut self.diags).into_vec();
        if self.reported_len == 0 {
            self.errors = 0;
            self.warnings = 0;
            self.unreported_errors = 0;
            self.first_unspanned = 0;
            return all;
        }
        let unreported = all.split_off(self.reported_len);
        for d in &unreported {
            if d.is_error() {
                self.errors = self.errors.saturating_sub(1);
            } else if d.is_warning() {
                self.warnings = self.warnings.saturating_sub(1);
            }
        }
        self.unreported_errors = 0;
        // Put the still-reported prefix back: `split_off` left it in `all`.
        self.diags = SmallVec::from_vec(all);
        self.first_unspanned = self.first_unspanned.min(self.diags.len());
        unreported
    }

    /// Drain the reported list and return it.
    pub fn drain_reported(&mut self) -> Vec<Diagnostic<'src>> {
        if self.reported_len == 0 {
            return Vec::new();
        }
        let reported_len = self.reported_len;
        let mut all = std::mem::take(&mut self.diags).into_vec();
        if reported_len == all.len() {
            self.reported_len = 0;
            self.errors = 0;
            self.warnings = 0;
            self.unreported_errors = 0;
            self.first_unspanned = 0;
            return all;
        }
        let unreported = all.split_off(reported_len);
        for d in &all {
            if d.is_error() {
                self.errors = self.errors.saturating_sub(1);
            } else if d.is_warning() {
                self.warnings = self.warnings.saturating_sub(1);
            }
        }
        self.diags = SmallVec::from_vec(unreported);
        self.reported_len = 0;
        self.first_unspanned = self.first_unspanned.saturating_sub(reported_len);
        all
    }

    // ── Severity downgrades (inspired by Lean 4's MessageLog) ───────

    /// Downgrade all **unreported** error-level diagnostics to warnings.
    ///
    /// This is the inverse of `--warn-as-error`: it can be used to forgive
    /// certain categories of errors (e.g. in a REPL or after a config change).
    pub fn errors_to_warnings(&mut self) {
        let mut moved = 0usize;
        for diag in &mut self.diags[self.reported_len..] {
            if diag.is_error() {
                diag.set_level(super::DiagnosticLevel::Warning);
                moved += 1;
            }
        }
        self.errors = self.errors.saturating_sub(moved);
        self.warnings += moved;
        self.unreported_errors = self.unreported_errors.saturating_sub(moved);
    }

    /// Downgrade all **unreported** error-level diagnostics to information.
    pub fn errors_to_infos(&mut self) {
        let mut moved = 0usize;
        for diag in &mut self.diags[self.reported_len..] {
            if diag.is_error() {
                diag.set_level(super::DiagnosticLevel::Info);
                moved += 1;
            }
        }
        self.errors = self.errors.saturating_sub(moved);
        self.unreported_errors = self.unreported_errors.saturating_sub(moved);
    }

    /// Downgrade all **unreported** warning-level diagnostics to information.
    pub fn warnings_to_infos(&mut self) {
        let mut moved = 0usize;
        for diag in &mut self.diags[self.reported_len..] {
            if diag.is_warning() {
                diag.set_level(super::DiagnosticLevel::Info);
                moved += 1;
            }
        }
        self.warnings = self.warnings.saturating_sub(moved);
    }
}

impl Default for DiagCtxt<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'src> From<Vec<Diagnostic<'src>>> for DiagCtxt<'src> {
    fn from(diags: Vec<Diagnostic<'src>>) -> Self {
        let mut errors = 0usize;
        let mut warnings = 0usize;
        let mut first_unspanned = None;
        for (i, d) in diags.iter().enumerate() {
            if d.is_error() {
                errors += 1;
            } else if d.is_warning() {
                warnings += 1;
            }
            if first_unspanned.is_none() && d.source().is_none() {
                first_unspanned = Some(i);
            }
        }
        let first_unspanned = first_unspanned.unwrap_or(diags.len());
        DiagCtxt {
            diags: SmallVec::from_vec(diags),
            reported_len: 0,
            first_unspanned,
            can_emit_warnings: true,
            errors,
            warnings,
            unreported_errors: errors,
        }
    }
}

impl<'src> IntoIterator for DiagCtxt<'src> {
    type Item = Diagnostic<'src>;
    type IntoIter = crate::smallvec::IntoIter<[Diagnostic<'src>; 2]>;

    fn into_iter(self) -> Self::IntoIter {
        self.diags.into_iter()
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    #[test]
    fn attach_source_advances_frontier_and_is_idempotent() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.attach_source("first");
        assert!(c.iter().all(|d| d.source() == Some("first")));
        // Idempotent: a second call does not change the recorded source.
        c.attach_source("second");
        assert!(c.iter().all(|d| d.source() == Some("first")));
        // A later unspanned push must be patched, while the earlier one stays.
        c.push(Diagnostic::error("b"));
        c.attach_source("third");
        assert_eq!(c.iter().nth(0).unwrap().source(), Some("first"));
        assert_eq!(c.iter().nth(1).unwrap().source(), Some("third"));
    }

    #[test]
    fn first_unspanned_does_not_advance_past_an_earlier_unspanned() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.push(Diagnostic::error("b").with_source("src"));
        c.attach_source("src2");
        assert_eq!(c.iter().nth(0).unwrap().source(), Some("src2"));
        assert_eq!(c.iter().nth(1).unwrap().source(), Some("src"));
    }

    #[test]
    fn errors_to_infos_updates_both_tallies() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("e"));
        c.errors_to_infos();
        assert_eq!(c.error_count(), 0);
        assert_eq!(c.warning_count(), 0);
        assert!(!c.has_unreported_errors());
        assert!(!c.unreported()[0].is_error());
    }

    #[test]
    fn warnings_to_infos_updates_warning_tally() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::warning("w"));
        c.warnings_to_infos();
        assert_eq!(c.warning_count(), 0);
        assert!(!c.unreported()[0].is_warning());
    }

    #[test]
    fn drain_reported_preserves_unreported_error_tally() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.mark_all_reported();
        c.push(Diagnostic::error("b"));
        let drained = c.drain_reported();
        assert_eq!(drained.len(), 1);
        assert!(c.has_unreported_errors());
        assert_eq!(c.error_count(), 1);
    }

    #[test]
    fn from_vec_initializes_tallies() {
        let ctx: DiagCtxt<'_> = vec![
            Diagnostic::error("e1"),
            Diagnostic::warning("w1"),
            Diagnostic::note("n1"),
            Diagnostic::error("e2"),
        ]
        .into();
        assert_eq!(ctx.error_count(), 2);
        assert_eq!(ctx.warning_count(), 1);
        assert!(ctx.has_unreported_errors());
        assert_eq!(ctx.unreported_len(), 4);
    }

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
    fn errors_to_warnings_clears_unreported_error_count() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("e"));
        c.errors_to_warnings();
        assert!(!c.has_unreported_errors());
    }

    #[test]
    fn errors_to_infos_clears_unreported_error_count() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("e"));
        c.errors_to_infos();
        assert!(!c.has_unreported_errors());
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
    fn drain_reported_edge_splits() {
        // reported_len == 0: nothing to drain, the buffer stays intact.
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.push(Diagnostic::warning("b"));
        let drained = c.drain_reported();
        assert!(drained.is_empty());
        assert_eq!(c.len(), 2, "unreported diags must remain untouched");

        // reported_len == len: everything drains, the buffer is emptied.
        c.mark_all_reported();
        let drained = c.drain_reported();
        assert_eq!(drained.len(), 2);
        assert_eq!(c.len(), 0);

        // Mixed split: one reported, one unreported.
        c.push(Diagnostic::error("c"));
        c.mark_all_reported();
        c.push(Diagnostic::warning("d"));
        let drained = c.drain_reported();
        assert_eq!(drained.len(), 1);
        assert!(drained[0].is_error());
        assert_eq!(c.len(), 1);
        assert!(c.unreported()[0].is_warning());
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
        c.set_can_emit_warnings(false);
        c.extend(vec![Diagnostic::error("e"), Diagnostic::warning("w")]);
        assert_eq!(c.len(), 1);
        assert_eq!(c.unreported_len(), 1);
        assert!(c.unreported()[0].is_error());
        // Any IntoIterator works, and each item still goes through `push`.
        c.extend(std::iter::once(Diagnostic::warning("w2")));
        assert_eq!(c.len(), 1);
    }

    /// `can_emit_warnings` is a *push-time* gate: diagnostics pushed before
    /// the flag was cleared stay in the context and are emitted; only later
    /// pushes are suppressed.  Pinning this so the doc contract cannot drift
    /// into an emit-time filter silently.
    #[test]
    fn warning_gate_is_push_time_not_emit_time() {
        use crate::diagnostics::DiagOpts;
        use crate::diagnostics::renderers::JsonRenderer;
        use crate::diagnostics::sink::DiagnosticSink;
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
        c.push(Diagnostic::warning("early"));
        c.set_can_emit_warnings(false);
        c.push(Diagnostic::warning("late"));
        // The late push was suppressed...
        assert_eq!(c.len(), 1);
        assert_eq!(c.unreported_len(), 1);
        // ...but the early warning is still emitted.
        c.emit(&mut emitter).unwrap();
        assert!(!c.has_unreported());
        assert!(
            out.borrow().contains("early"),
            "pre-existing warning must still be emitted: {out:?}"
        );
        assert!(
            !out.borrow().contains("late"),
            "suppressed push must not be emitted: {out:?}"
        );
    }

    #[test]
    fn unreported_error_count_tracks_push_truncate_and_report() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("first"));
        assert!(c.has_unreported_errors());
        c.truncate_unreported(0);
        assert!(!c.has_unreported_errors());
        drop(c);
    }

    #[test]
    fn emit_advances_partition_and_does_not_duplicate() {
        use crate::diagnostics::DiagOpts;
        use crate::diagnostics::emitter::Emitter;
        use crate::diagnostics::renderers::JsonRenderer;
        use crate::diagnostics::sink::DiagnosticSink;
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
