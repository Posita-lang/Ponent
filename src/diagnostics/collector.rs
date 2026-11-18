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
/// index update.  [`into_inner`](Self::into_inner) moves the `SmallVec` out
/// without an `extend` pass, but note that `SmallVec::into_vec` still
/// allocates and copies when the diagnostics are inline (the zero/one-diag
/// case) — the claim is "no per-element move through an intermediate
/// `extend`", not "no allocation".  [`into_iter`](Self::into_iter) is the
/// genuinely zero-cost consuming path (it reuses the inline buffer).
/// [`drain_unreported`](Self::drain_unreported) is one in-place `SmallVec`
/// drain plus one allocation for the returned `Vec`.
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
    /// Warnings in the unreported suffix — the symmetric counter that lets
    /// `warnings_to_infos` and `for_each_warning` gate and skip the reported
    /// prefix in O(1) instead of scanning the suffix to find out.
    unreported_warnings: usize,
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

    pub fn can_emit_warnings(&self) -> bool {
        self.can_emit_warnings
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
            unreported_warnings: 0,
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
                self.unreported_warnings += 1;
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

    /// Create and push an error with an attached error code.
    pub fn error_with_code(
        &mut self,
        msg: impl Into<Cow<'src, str>>,
        code: &str,
    ) -> EmissionGuarantee {
        let diag = Diagnostic::error(msg).with_code_str(code);
        self.push(diag)
    }

    #[deprecated(note = "use error_with_code")]
    pub fn error_fmt(&mut self, msg: impl Into<Cow<'src, str>>, code: &str) -> EmissionGuarantee {
        self.error_with_code(msg, code)
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
        let (lower, upper) = iter.size_hint();
        // Reserve from the UPPER bound: an exact-size producer (`Vec`, slice,
        // most in-tree iterators) reports `Some(exact)`, and using only the
        // lower bound reintroduced the O(log N) growth this reserve exists to
        // avoid on filtered batches.
        let hint = upper.unwrap_or(lower);
        if hint > 0 {
            self.diags.reserve(hint);
        }
        for diag in iter {
            self.push(diag);
        }
    }

    /// Attach `source` to every diagnostic (reported and unreported) that does
    /// not already carry one.  Called at a producer's boundary so downstream
    /// emitters can render snippets without cloning the source buffer.
    ///
    /// The reported prefix is touched only when some reported diagnostic still
    /// lacks source (`first_unspanned <= reported_len`); diags before the
    /// frontier already carry source by invariant and are not revisited.
    /// Re-attaching a *different* source to already-spanned diagnostics is not
    /// supported by this API — it is idempotent by design.
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

    /// Iterate mutably over ALL diagnostics (reported first, then
    /// unreported).  The name states the region explicitly: `emit` covers only
    /// the unreported suffix, so a caller that pairs this iterator with an
    /// emitter must intend a full re-render (the `emit_all_iter` contract),
    /// not a checkpoint emit.  `HumanizeOnly` still restricts mutation to
    /// humanization.
    pub(crate) fn iter_all_mut(
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
                    self.unreported_warnings = self.unreported_warnings.saturating_sub(1);
                }
            }
            self.diags.truncate(keep);
        }
        self.first_unspanned = self.first_unspanned.min(self.diags.len());
    }

    /// Reset diagnostic data but **preserve** `can_emit_warnings`, which is
    /// configuration rather than data.
    pub fn clear(&mut self) {
        self.diags.clear();
        self.reported_len = 0;
        self.first_unspanned = 0;
        self.errors = 0;
        self.warnings = 0;
        self.unreported_errors = 0;
        self.unreported_warnings = 0;
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
        debug_assert!(
            self.reported_len <= self.diags.len(),
            "reported_len desynchronized from the diagnostic vector"
        );
        if self.reported_len == self.diags.len() {
            return Ok(());
        }
        emitter.emit_all(&mut self.diags[self.reported_len..])?;
        self.reported_len = self.diags.len();
        self.unreported_errors = 0;
        self.unreported_warnings = 0;
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
        if self.errors == 0 {
            return;
        }
        if self.errors == self.unreported_errors {
            // Every error lives in the unreported suffix — skip the reported
            // prefix entirely (the tally proves it holds none).
            self.diags[self.reported_len..]
                .iter()
                .filter(|d| d.is_error())
                .for_each(f);
            return;
        }
        self.iter().filter(|d| d.is_error()).for_each(f)
    }

    /// Iterate over all warning-level diagnostics without allocating.
    pub fn for_each_warning<F: FnMut(&Diagnostic<'src>)>(&self, mut f: F) {
        if self.warnings == 0 {
            return;
        }
        if self.warnings == self.unreported_warnings {
            self.diags[self.reported_len..]
                .iter()
                .filter(|d| d.is_warning())
                .for_each(f);
            return;
        }
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
        let mut unreported_warnings = 0usize;
        let mut first_unspanned = None;
        for (i, d) in self.diags.iter().enumerate() {
            if d.is_error() {
                errors += 1;
                if i >= self.reported_len {
                    unreported_errors += 1;
                }
            } else if d.is_warning() {
                warnings += 1;
                if i >= self.reported_len {
                    unreported_warnings += 1;
                }
            }
            if first_unspanned.is_none() && d.source().is_none() {
                first_unspanned = Some(i);
            }
        }
        self.errors = errors;
        self.warnings = warnings;
        self.unreported_errors = unreported_errors;
        self.unreported_warnings = unreported_warnings;
        self.first_unspanned = first_unspanned.unwrap_or(self.diags.len());
    }

    /// Drain the unreported list and return it.
    ///
    /// **Retained deliberately** even though no in-tree caller uses it today:
    /// it is part of the public collector API and an external batch/checkpoint
    /// consumer may. The `mem::take` fast path below is pinned by
    /// `drain_unreported_leaves_reported`, so it is not unowned maintenance
    /// debt.
    pub fn drain_unreported(&mut self) -> Vec<Diagnostic<'src>> {
        if self.reported_len == 0 {
            self.errors = 0;
            self.warnings = 0;
            self.unreported_errors = 0;
            self.unreported_warnings = 0;
            self.first_unspanned = 0;
            return std::mem::take(&mut self.diags).into_vec();
        }
        // One allocation, no `into_vec` + `split_off` pair: `SmallVec::drain`
        // removes the suffix in place and the collect moves exactly those
        // elements out.  The retained prefix never leaves the inline buffer.
        let drained: Vec<Diagnostic<'src>> = self.diags.drain(self.reported_len..).collect();
        for d in &drained {
            if d.is_error() {
                self.errors = self.errors.saturating_sub(1);
            } else if d.is_warning() {
                self.warnings = self.warnings.saturating_sub(1);
            }
        }
        self.unreported_errors = 0;
        self.unreported_warnings = 0;
        self.first_unspanned = self.first_unspanned.min(self.diags.len());
        drained
    }

    /// Drain the reported list and return it.
    pub fn drain_reported(&mut self) -> Vec<Diagnostic<'src>> {
        let reported_len = self.reported_len;
        if reported_len == 0 {
            return Vec::new();
        }
        // Symmetric to `drain_unreported`: one in-place drain, one allocation.
        let drained: Vec<Diagnostic<'src>> = self.diags.drain(..reported_len).collect();
        for d in &drained {
            if d.is_error() {
                self.errors = self.errors.saturating_sub(1);
            } else if d.is_warning() {
                self.warnings = self.warnings.saturating_sub(1);
            }
        }
        // The drained region was reported by construction, so the
        // unreported counters are untouched.
        self.reported_len = 0;
        self.first_unspanned = self.first_unspanned.saturating_sub(reported_len);
        drained
    }

    // ── Severity downgrades (inspired by Lean 4's MessageLog) ───────

    /// Downgrade all unreported **Error** diagnostics to warnings.
    ///
    /// `Fatal` is deliberately excluded: fatalness can carry abort/exit
    /// semantics upstream, and "forgive errors" must not silently forgive a
    /// non-recoverable diagnostic.
    ///
    /// Respects the push-time `can_emit_warnings` gate: when warnings are
    /// disabled, the conversion is a no-op rather than injecting warnings
    /// that `emit` would then print.
    pub fn errors_to_warnings(&mut self) {
        if self.unreported_errors == 0 || !self.can_emit_warnings {
            return;
        }
        let target = self.unreported_errors;
        let mut moved = 0usize;
        for diag in &mut self.diags[self.reported_len..] {
            if moved == target {
                // Every Error in the suffix has been converted; the rest of
                // the scan can only find Info/Note entries.
                break;
            }
            if diag.level() == super::DiagnosticLevel::Error {
                diag.set_level(super::DiagnosticLevel::Warning);
                moved += 1;
            }
        }
        self.errors = self.errors.saturating_sub(moved);
        self.warnings = self.warnings.saturating_add(moved);
        self.unreported_errors = self.unreported_errors.saturating_sub(moved);
        // The converted entries are now unreported warnings.
        self.unreported_warnings = self.unreported_warnings.saturating_add(moved);
    }

    /// Downgrade all unreported **Error** diagnostics to information.
    /// `Fatal` is excluded for the same reason as [`errors_to_warnings`].
    pub fn errors_to_infos(&mut self) {
        if self.unreported_errors == 0 {
            return;
        }
        let target = self.unreported_errors;
        let mut moved = 0usize;
        for diag in &mut self.diags[self.reported_len..] {
            if moved == target {
                break;
            }
            if diag.level() == super::DiagnosticLevel::Error {
                diag.set_level(super::DiagnosticLevel::Info);
                moved += 1;
            }
        }
        self.errors = self.errors.saturating_sub(moved);
        self.unreported_errors = self.unreported_errors.saturating_sub(moved);
    }

    /// Downgrade all unreported warning-level diagnostics to information.
    pub fn warnings_to_infos(&mut self) {
        // Gate on the UNREPORTED tally, matching the sibling downgrades: with
        // every warning already in the reported prefix the loop below is dead
        // work (the old gate used the total and scanned the suffix anyway).
        if self.unreported_warnings == 0 {
            return;
        }
        let target = self.unreported_warnings;
        let mut moved = 0usize;
        for diag in &mut self.diags[self.reported_len..] {
            if moved == target {
                break;
            }
            if diag.is_warning() {
                diag.set_level(super::DiagnosticLevel::Info);
                moved += 1;
            }
        }
        self.warnings = self.warnings.saturating_sub(moved);
        self.unreported_warnings = self.unreported_warnings.saturating_sub(moved);
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
            unreported_warnings: warnings,
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
        // `DiagnosticSink` is sealed and requires `write_fmt`; the blanket
        // `impl<W: Write>` covers this test sink as soon as it implements
        // `Write`, which also exercises the production sink path.
        impl io::Write for SharedSink {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0
                    .borrow_mut()
                    .push_str(std::str::from_utf8(buf).expect("diagnostic output is UTF-8"));
                Ok(buf.len())
            }

            fn flush(&mut self) -> io::Result<()> {
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
        // `DiagnosticSink` is sealed and requires `write_fmt`; the blanket
        // `impl<W: Write>` covers this test sink as soon as it implements
        // `Write`, which also exercises the production sink path.
        impl io::Write for SharedSink {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0
                    .borrow_mut()
                    .push_str(std::str::from_utf8(buf).expect("diagnostic output is UTF-8"));
                Ok(buf.len())
            }

            fn flush(&mut self) -> io::Result<()> {
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

    #[test]
    fn fatal_is_not_downgraded() {
        let mut c = DiagCtxt::new();
        let mut d = Diagnostic::error("fatal-boom");
        d.set_level(crate::diagnostics::DiagnosticLevel::Fatal);
        c.push(d);
        c.errors_to_warnings();
        assert_eq!(
            c.iter().next().unwrap().level(),
            crate::diagnostics::DiagnosticLevel::Fatal
        );
        assert_eq!(c.warning_count(), 0);
        c.errors_to_infos();
        assert_eq!(
            c.iter().next().unwrap().level(),
            crate::diagnostics::DiagnosticLevel::Fatal
        );
    }

    #[test]
    fn errors_to_warnings_respects_disabled_gate() {
        let mut c = DiagCtxt::new();
        c.set_can_emit_warnings(false);
        c.push(Diagnostic::error("e"));
        c.errors_to_warnings();
        assert_eq!(c.error_count(), 1);
        assert_eq!(c.warning_count(), 0);
        assert_eq!(
            c.iter().next().unwrap().level(),
            crate::diagnostics::DiagnosticLevel::Error
        );
    }

    #[test]
    fn zero_tallies_skip_filter_scans() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::note("n"));
        let mut called = false;
        c.for_each_error(|_| called = true);
        c.for_each_warning(|_| called = true);
        c.errors_to_warnings();
        c.errors_to_infos();
        assert!(!called);
        assert_eq!(c.error_count(), 0);
        assert_eq!(c.warning_count(), 0);
    }
}
