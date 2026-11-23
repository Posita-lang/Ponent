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
    /// Severity tallies of the **reported** prefix (`0..reported_len`).
    ///
    /// The emitter only ever receives `&mut diags[reported_len..]`, so the
    /// prefix cannot change behind a recount's back; keeping its tallies lets
    /// [`Self::recount_derived_state`] scan only the unreported suffix instead
    /// of the whole vector after every emit (O(n) → O(suffix), O(n²) → O(n)
    /// over a context's lifetime).  Invariant:
    /// `errors == reported_errors + unreported_errors` (same for warnings).
    reported_errors: usize,
    reported_warnings: usize,
    /// Warnings dropped by the [`Self::set_can_emit_warnings`] gate.
    ///
    /// Monotone and *not* derivable from `diags` (a suppressed diagnostic is
    /// never stored), so no recount can or may recompute it.  Surfaced so a
    /// caller can print "N warnings suppressed" instead of losing them
    /// silently.
    suppressed_warnings: usize,
    /// Internal invariant violations reported (never panicked) at the emission
    /// boundary — currently the "diagnostic reached emission without a source"
    /// audit.  Counted so tests can assert exactly-once without parsing stderr;
    /// like `suppressed_warnings` it is monotone and never reset by `clear`.
    internal_bugs: usize,
    /// "Some diagnostic in the *reported prefix* may lack a source."
    ///
    /// Two flags rather than one: the audit scans only the unreported suffix, so
    /// a single flag could not be cleared on a clean suffix (a nonempty prefix
    /// was unaudited) and `reported_len > 0` froze it at `true` for the rest of
    /// the context's life — permanently disabling the O(1) gate on exactly the
    /// LSP re-publish loop it exists for.  Each region now carries the exact flag
    /// its own tally describes: `mark_all_reported` transfers suffix→prefix,
    /// prefix removal clears the prefix flag, `attach_source` (a full-vector
    /// patch) clears both.  Staleness remains sound in one direction only: a
    /// stale `true` costs a scan, a stale `false` would skip a real audit.
    may_have_unsourced_prefix: bool,
    /// Same, for the unreported suffix.  Armed by a sourceless `push`, cleared by
    /// a clean suffix audit (which covers exactly that region) and by
    /// `attach_source`.
    may_have_unsourced_suffix: bool,
}

/// The inline `SmallVec<[Diagnostic; 2]>` is what keeps a zero/one-diagnostic
/// context on the stack; fail the build if `DiagCtxt` (or `Diagnostic`) grows
/// enough to make that a silent per-context stack cost.
const _: () = assert!(std::mem::size_of::<DiagCtxt<'_>>() <= 1024);

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

    #[must_use]
    pub fn can_emit_warnings(&self) -> bool {
        self.can_emit_warnings
    }
    pub fn new() -> Self {
        DiagCtxt {
            diags: SmallVec::new(),
            reported_len: 0,
            may_have_unsourced_prefix: false,
            may_have_unsourced_suffix: false,
            can_emit_warnings: true,
            errors: 0,
            warnings: 0,
            unreported_errors: 0,
            unreported_warnings: 0,
            reported_errors: 0,
            reported_warnings: 0,
            suppressed_warnings: 0,
            internal_bugs: 0,
        }
    }

    /// Push a diagnostic into `unreported` and return an `EmissionGuarantee`
    /// proving the diagnostic was recorded (inspired by rustc's `ErrorGuaranteed`).
    pub fn push(&mut self, diag: Diagnostic<'src>) -> EmissionGuarantee {
        // Gate first, and only on `Warning`: a suppressed diagnostic must never
        // reach the counters.  The gate stays ahead of the counting so a
        // suppressed warning cannot be counted twice.
        if diag.is_warning() && !self.can_emit_warnings {
            // Count before dropping: the suppressed diagnostic never enters
            // `diags`, so this is the only record of it.
            self.suppressed_warnings += 1;
            return EmissionGuarantee::suppressed();
        }
        // Counting goes through the *same predicates* as `scan_derived` (the
        // debug oracle), `recount_derived_state`, the truncation scans and
        // `for_each_*`.  The old shape matched `level()` instead, so a
        // level↔predicate divergence biased this one site against the oracle:
        // `summary()` went O(1)-wrong and `check()` could return `Ok` for a
        // context holding an Error, all silently in release.  Now any such
        // divergence biases every site identically and the debug cross-check
        // catches it.  (`is_error()` must keep covering `Fatal`; that bridge is
        // pinned by the assert below and by the tallies' documented contract.)
        if diag.is_error() {
            debug_assert!(
                matches!(
                    diag.level(),
                    super::DiagnosticLevel::Fatal | super::DiagnosticLevel::Error
                ),
                "is_error() must not cover a non-error level"
            );
            self.errors += 1;
            self.unreported_errors += 1;
        } else if diag.is_warning() {
            self.warnings += 1;
            self.unreported_warnings += 1;
        }
        // Conservative marker for `audit_source_attachment`: set when a
        // diagnostic arrives without a source, cleared by `attach_source` (which
        // patches every one).  Staleness only costs a scan, never a missed audit.
        if diag.source().is_none() {
            self.may_have_unsourced_suffix = true;
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
        // Reserve the hint only when the iterator knows its size exactly.
        // `upper` alone is the *pre-filter* bound for adapters such as
        // `filter`/`filter_map`, so `extend(big.into_iter().filter(..))` would
        // reserve the whole input to keep a handful.  `Vec::extend` uses the
        // lower bound for the same reason; `lower == upper` recovers the exact
        // case (`Vec`, slice, most in-tree iterators) that motivated the
        // reserve.
        //
        // With warnings disabled every `Warning` in the batch is dropped at
        // `push`, so even an exact hint no longer bounds what is kept: reserve
        // nothing rather than capacity the push-time gate empties.
        if self.can_emit_warnings {
            let hint = match (lower, upper) {
                (l, Some(u)) if l == u => u,
                (l, _) => l,
            };
            if hint > 0 {
                self.diags.reserve(hint);
            }
            for diag in iter {
                self.push(diag);
            }
            return;
        }
        // Warnings are dropped at `push`, so the pre-filter hint is not a bound
        // on what is kept — but reserving *nothing* makes an all-error batch pay
        // O(log N) reallocations, and that is exactly the batch a
        // warnings-disabled build produces.  An exact-size source lets us count
        // what `push` will keep in one pass; the intermediate `Vec` is a single
        // exact allocation, cheaper than the growth chain it replaces.  An
        // inexact iterator keeps the old reserve-nothing behaviour.
        match (lower, upper) {
            (l, Some(u)) if l == u => {
                // Reserve the optimistic `u`, let the `push` gate drop warnings,
                // then give the slack back.  The old two-pass shape collected
                // into a temporary `Vec` (u·|Diagnostic| written), counted over it
                // (u·|D| read), then moved the survivors (2·kept·|D|): strictly
                // more traffic at every `kept ≤ u`, and an extra allocation for
                // an all-warning batch, which is the exact batch a
                // warnings-disabled build produces.
                self.diags.reserve(u);
                for diag in iter {
                    self.push(diag);
                }
                // Give the optimistic slack back only when it dominates: a
                // repeatedly mixed batch would otherwise shrink and regrow every
                // call, resurrecting the reallocation chain the reserve exists to
                // avoid.  An all-warning batch (slack == u) still reclaims it.
                if self.diags.len() * 2 < u {
                    self.diags.shrink_to_fit();
                }
            }
            _ => {
                for diag in iter {
                    self.push(diag);
                }
            }
        }
    }

    /// Extend from an owned batch, pre-filtering warnings in place.
    ///
    /// The `Vec` case of [`Self::extend`], made explicit: `retain` compacts the
    /// input without a temporary buffer, so the surviving count is exact and the
    /// reserve matches it.  An all-warning batch (what a warnings-disabled build
    /// receives) costs one scan and no allocation for the survivors, and the
    /// warnings it drops are credited to `suppressed_warnings` exactly as the
    /// `push` gate would credit them.
    pub fn extend_from_vec(&mut self, mut diags: Vec<Diagnostic<'src>>) {
        if self.can_emit_warnings {
            let kept = diags.len();
            if kept > 0 {
                self.diags.reserve(kept);
            }
            for diag in diags {
                self.push(diag);
            }
            return;
        }
        // `retain` drops the warnings before `push` can see them, so the
        // suppression tally has to be credited here.  Without this the same batch
        // through `extend` (one `push` per item) reported N suppressed warnings
        // and through this path reported zero — the "N warnings suppressed"
        // surface under-reported by exactly the batch size.
        let before = diags.len();
        diags.retain(|d| !d.is_warning());
        self.suppressed_warnings += before - diags.len();
        let kept = diags.len();
        if kept > 0 {
            self.diags.reserve(kept);
        }
        for diag in diags {
            self.push(diag);
        }
    }

    /// Attach `source` to every diagnostic (reported and unreported) that does
    /// not already carry one.  Called at a producer's boundary so downstream
    /// emitters can render snippets without cloning the source buffer.
    ///
    /// # Provenance lock (write-once)
    ///
    /// The `is_none` guard is not an idempotence optimisation: it is the
    /// provenance lock.  `source` identifies *which file segment* a diagnostic
    /// came from, and a wrong source is structurally undetectable at emission
    /// time (a wrong snippet looks exactly like a right one), so the first
    /// attach must win and later ones must never overwrite it.
    ///
    /// A full scan rather than a remembered frontier: the frontier was a
    /// micro-optimisation that needed a cross-function invariant (below which
    /// index is still unspanned) maintained at every push/truncate/drain/clear
    /// site, and every violation of it silently *dropped* sources instead of
    /// patching them.  With the guard in place a rescan is idempotent, so the
    /// only remaining failure mode — "still without source at emission" — is
    /// covered by the audit in [`Self::emit`].  Measured call count: 4
    /// production sites (two in the checker, two in the resolver), each once
    /// per file.
    pub fn attach_source(&mut self, source: &'src str) {
        if !(self.may_have_unsourced_prefix || self.may_have_unsourced_suffix) {
            // Both `false` means every stored diagnostic is known-sourced: set by
            // two clean audits, a full-vector patch, or a sourced-only batch, and
            // only ever re-armed by a sourceless `push`.  So the common steady
            // state (producer attaches after each push batch) skips the scan.
            return;
        }
        for diag in &mut self.diags {
            if diag.source().is_none() {
                diag.set_source(source);
            }
        }
        // Post-condition: *every* diagnostic in the vector now carries a source
        // (the loop is unconditional over the vector, and an already-sourced one
        // is left alone), so both region flags are clear.  A new sourceless
        // `push` re-arms the suffix flag.
        self.may_have_unsourced_prefix = false;
        self.may_have_unsourced_suffix = false;
    }

    /// Total diagnostics (reported + unreported).
    pub fn len(&self) -> usize {
        self.diags.len()
    }

    /// True when the underlying vector is empty — equivalently, when
    /// [`Self::reported`] and [`Self::unreported`] are both empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.diags.is_empty()
    }

    /// Iterate over all diagnostics (reported first, then unreported).
    #[must_use]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Diagnostic<'src>> + ExactSizeIterator {
        self.diags.iter()
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
        // *All* diagnostics, not just the suffix: `mark_all_reported` promotes
        // diagnostics to the reported prefix without an emitter ever seeing
        // them, so a `*_kind`-built one there would stay un-humanized forever —
        // a debug panic (or an empty message in release) on the very inspection
        // paths this method exists for.  `humanize` is latched and idempotent,
        // so already-humanized diagnostics cost one branch.
        for diag in &mut self.diags {
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
        // Clamp a stale count in release.  Without this, `reported_len + keep`
        // lands past the end, nothing is truncated, and the caller believes the
        // rollback happened.
        let keep_unreported = keep_unreported.min(self.unreported_len());
        let keep = self.reported_len.saturating_add(keep_unreported);
        if keep_unreported == 0 && keep < self.diags.len() {
            // Dropping the whole unreported suffix is the common checkpoint
            // discard, and the tallies already hold exactly its counts, so the
            // per-diagnostic subtraction is dead work.
            self.errors = self.errors.saturating_sub(self.unreported_errors);
            self.warnings = self.warnings.saturating_sub(self.unreported_warnings);
            self.unreported_errors = 0;
            self.unreported_warnings = 0;
            self.diags.truncate(keep);
        } else if keep < self.diags.len() {
            // Scan whichever side is smaller.  "Roll back one diagnostic" keeps
            // almost the whole suffix, so scanning the truncated tail would cost
            // O(suffix) for one subtraction; scanning the kept remainder and
            // deriving `truncated = totals - kept` is O(1) there.  Both branches
            // use the same predicates, so the partition is consistent.
            let truncated = self.diags.len() - keep;
            if truncated <= keep_unreported {
                for d in &self.diags[keep..] {
                    if d.is_error() {
                        self.errors = self.errors.saturating_sub(1);
                        self.unreported_errors = self.unreported_errors.saturating_sub(1);
                    } else if d.is_warning() {
                        self.warnings = self.warnings.saturating_sub(1);
                        self.unreported_warnings = self.unreported_warnings.saturating_sub(1);
                    }
                }
            } else {
                let mut kept_errors = 0usize;
                let mut kept_warnings = 0usize;
                for d in &self.diags[self.reported_len..keep] {
                    if d.is_error() {
                        kept_errors += 1;
                    } else if d.is_warning() {
                        kept_warnings += 1;
                    }
                }
                self.errors = self
                    .errors
                    .saturating_sub(self.unreported_errors.saturating_sub(kept_errors));
                self.warnings = self
                    .warnings
                    .saturating_sub(self.unreported_warnings.saturating_sub(kept_warnings));
                self.unreported_errors = kept_errors;
                self.unreported_warnings = kept_warnings;
            }
            self.diags.truncate(keep);
        }
        self.refresh_unsourced_flag();
    }

    /// Reset diagnostic data but **preserve** `can_emit_warnings`, which is
    /// configuration rather than data.
    /// Clear the conservative unsourced marker when the vector is empty.
    ///
    /// "Empty ⇒ no diagnostic can be missing a source" is the same argument
    /// `clear` documents, but only `clear` applied it: draining or truncating to
    /// empty left the flag set, so the next `emit` paid an O(suffix) audit and
    /// `attach_source` lost its O(1) fast path.  Called at the end of every
    /// mutator; the `is_empty` test encodes exactly the difference between "the
    /// region that could be unsourced is gone" (clear) and "a region that could
    /// be unsourced remains" (do not clear).
    #[inline]
    fn refresh_unsourced_flag(&mut self) {
        if self.diags.is_empty() {
            self.may_have_unsourced_prefix = false;
            self.may_have_unsourced_suffix = false;
        }
    }

    pub fn clear(&mut self) {
        self.diags.clear();
        self.reported_len = 0;
        self.errors = 0;
        self.warnings = 0;
        self.unreported_errors = 0;
        self.unreported_warnings = 0;
        self.reported_errors = 0;
        self.reported_warnings = 0;
        // Derived state: the vector is empty, so "some diagnostic may lack a
        // source" is provably false for both regions.
        self.may_have_unsourced_prefix = false;
        self.may_have_unsourced_suffix = false;
        // `suppressed_warnings` is deliberately NOT reset: it is a
        // session-lifetime observation, not derived state.
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
        self.audit_source_attachment();
        // §3.1: the mutation channel is closed by the *type*, not by an
        // observation.  `DiagCtxt` humanizes at the boundary (the emitter's batch
        // paths take `&Diagnostic`, since every `Renderer::render_into` already
        // does), so nothing downstream can change a severity — the pre/post
        // `scan_derived` pair that existed only to detect such a change, and the
        // `HumanizeOnly` wrapper that tried to restrict it, are both gone.  In
        // release the guarantee used to be "trust"; now it is the borrow checker.
        self.humanize_pending();
        emitter.emit_all(&self.diags[self.reported_len..])?;
        // Repair path (not a detector): an external checkpoint consumer may
        // have called `recount_derived_state` between emits, and this keeps the
        // tallies consistent with the vectors it owns.  Sampled rather than
        // unconditional: a context living across N modules with per-module
        // emits would otherwise pay O(N²) full scans in debug builds (the
        // release path's whole point is O(n) over the batch).  Sizes 1, 2, 4, …
        // cost ≈ 2n total across the context's life, and A1's snapshot assert
        // above stays unconditional.
        #[cfg(debug_assertions)]
        if self.diags.len().is_power_of_two() {
            self.recount_derived_state();
        }
        // Fold the suffix tallies into the prefix ones: the emitter only ever
        // saw `diags[reported_len..]` (and a debug-only scan above proves it did
        // not change their severity), so the prefix is unchanged and this keeps
        // `errors == reported + unreported`.
        // One copy of the partition fold: `mark_all_reported` documents the
        // "both tallies must be cleared" lesson, and an inline duplicate here is
        // exactly how that lesson gets lost on the next fix.
        self.mark_all_reported();
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
        if self.unreported_errors == 0 {
            // Mirror of the branch above: nothing is unreported, so every error
            // is in the reported prefix.  Post-emit inspection walks O(prefix)
            // instead of O(total).
            self.diags[..self.reported_len]
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
        if self.unreported_warnings == 0 {
            // Mirror of `for_each_error`: post-emit inspection walks O(prefix),
            // not O(total) over an empty suffix.
            self.diags[..self.reported_len]
                .iter()
                .filter(|d| d.is_warning())
                .for_each(f);
            return;
        }
        self.iter().filter(|d| d.is_warning()).for_each(f)
    }

    /// Iterate over all diagnostics, yielding them with 1-based index.
    #[must_use]
    pub fn enumerate(&self) -> impl Iterator<Item = (usize, &Diagnostic<'src>)> {
        self.iter().enumerate().map(|(i, d)| (i + 1, d))
    }

    /// Total count of error diagnostics.
    ///
    /// Counts **`Fatal` as well as `Error`** (see [`Self::push`]) — it is the
    /// "does this build fail" tally, matching [`Self::check`].  `Fatal` is
    /// excluded from the `errors_to_*` downgrades, so this number and
    /// [`Self::warning_count`] do not sum to the total after a downgrade.
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
            // Reuse the same derivation `recount_derived_state` and
            // `From<Vec>` use: a bespoke loop here could drift from them, which
            // is exactly what `scan_derived`'s doc cites as its reason to exist.
            let (e, w, _, _, _) = scan_derived(&self.diags, self.reported_len);
            debug_assert_eq!(
                (self.errors, self.warnings),
                (e, w),
                "DiagCtxt severity tally drifted from the diagnostics"
            );
        }
        (self.errors, self.warnings)
    }

    /// Attach failing-check context (span, and source when absent) to the most
    /// recent diagnostic.
    ///
    /// Purpose-built instead of handing out `&mut Diagnostic` (the old
    /// `last_mut`): the only in-tree caller —
    /// [`adorn_with_span`](crate::diagnostics::adorn_with_span) — needs exactly
    /// these two fields, and a raw handle lets a caller rewrite a *level* behind
    /// the O(1) severity tallies.  `emit` would then fold the stale
    /// `unreported_errors` into the reported prefix, the drift becomes
    /// permanent, and `check()` can return `Ok` for a context holding an
    /// `Error`.  Neither field touches the tallies, so this mutation is safe by
    /// construction.
    ///
    /// Returns `false` when the context is empty (the caller already guards on
    /// "the closure grew the context", so this is belt-and-braces).
    pub(crate) fn augment_last_diagnostic(
        &mut self,
        span: crate::ast::Span,
        source: Option<&'src str>,
    ) -> bool {
        // Target the *unreported suffix*, not "the last element": after an
        // `emit` the vector's tail is a reported diagnostic, and the whole O(1)
        // tally scheme treats the reported prefix as immutable behind
        // `HumanizeOnly`.  `last_mut()` spans the whole vector, so a closure
        // that pushed nothing would silently adorn a previously emitted
        // diagnostic — the wrong one — instead of returning `false`.
        let Some(diag) = self
            .diags
            .get_mut(self.reported_len..)
            .and_then(|suffix| suffix.last_mut())
        else {
            return false;
        };
        if diag.spans.first().is_none() {
            diag.spans = super::MultiSpan::new(span);
            diag.mark_spans_user_set();
            diag.invalidate_stable_id();
        }
        if let Some(source) = source
            // Accessor, not the field: every other site reads `source()`.
            && diag.source().is_none()
        {
            diag.set_source(source);
        }
        true
    }

    // ── reported / unreported management (inspired by Lean 4's MessageLog) ──

    /// Access the reported diagnostics.
    #[must_use]
    pub fn reported(&self) -> &[Diagnostic<'src>] {
        &self.diags[..self.reported_len]
    }

    /// Access the unreported diagnostics.
    #[must_use]
    pub fn unreported(&self) -> &[Diagnostic<'src>] {
        &self.diags[self.reported_len..]
    }

    /// True when there are unreported diagnostics.
    #[must_use]
    pub fn has_unreported(&self) -> bool {
        self.diags.len() > self.reported_len
    }

    /// True when there are unreported errors.
    #[must_use]
    pub fn has_unreported_errors(&self) -> bool {
        self.unreported_errors > 0
    }

    /// Move all unreported diagnostics into the reported list.
    ///
    /// After calling this, `unreported()` is empty and `reported()` contains
    /// everything.  Useful after a checkpoint / snapshot in incremental
    /// compilation.
    pub fn mark_all_reported(&mut self) {
        // The suffix's unsourced set becomes the prefix's: the same diagnostics,
        // unaudited, on the other side of the partition.
        self.may_have_unsourced_prefix |= self.may_have_unsourced_suffix;
        self.may_have_unsourced_suffix = false;
        self.reported_errors += self.unreported_errors;
        self.reported_warnings += self.unreported_warnings;
        self.reported_len = self.diags.len();
        self.unreported_errors = 0;
        // BOTH unreported tallies must be cleared.  `for_each_warning`'s fast
        // path is `warnings == unreported_warnings` → scan only the unreported
        // suffix; leaving the warning count stale makes that path scan a now
        // empty suffix and silently drop the just-reported warnings.
        self.unreported_warnings = 0;
    }

    /// Recompute every derived tally from `diags` and `reported_len`.
    ///
    /// The emitter only ever receives `&mut diags[reported_len..]` through
    /// a raw slice, so the type system does *not* prevent it: the debug-only
    /// before/after scan in [`Self::emit`] is what detects it.  The one escape
    /// is `into_diagnostic`.  This is therefore called
    /// * from [`Self::emit`] in **debug** builds (a cross-check that fails next
    ///   to the offending render path), and
    /// * by an external batch/checkpoint consumer that rewrote diagnostics on
    ///   purpose — the reason it is `pub(crate)` rather than debug-only.
    ///
    /// Release `emit` relies on the `HumanizeOnly` boundary instead of paying
    /// this scan on every emission.
    pub(crate) fn recount_derived_state(&mut self) {
        let reported_len = self.reported_len;
        let (unreported_errors, unreported_warnings) = {
            let suffix = &self.diags[reported_len..];
            let mut errors = 0usize;
            let mut warnings = 0usize;
            for d in suffix.iter() {
                if d.is_error() {
                    errors += 1;
                } else if d.is_warning() {
                    warnings += 1;
                }
            }
            (errors, warnings)
        };
        self.unreported_errors = unreported_errors;
        self.unreported_warnings = unreported_warnings;
        // Totals = immutable prefix tallies + the freshly scanned suffix.  The
        // emitter only receives `&mut diags[reported_len..]` (through
        // `HumanizeOnly`), so the prefix cannot have changed; rescanning it was
        // the O(n) work that made a long-lived context's emit loop quadratic.
        self.errors = self.reported_errors + unreported_errors;
        self.warnings = self.reported_warnings + unreported_warnings;
        // The full scan is now a debug-only cross-check: if a future mutator
        // touches the prefix, this fails loudly instead of leaving the O(1)
        // tallies silently wrong.
        #[cfg(debug_assertions)]
        {
            let (errors, warnings, ue, uw, _) = scan_derived(&self.diags, self.reported_len);
            debug_assert_eq!(
                (
                    self.errors,
                    self.warnings,
                    self.unreported_errors,
                    self.unreported_warnings
                ),
                (errors, warnings, ue, uw),
                "suffix-only recount diverged from the full scan"
            );
        }
    }

    /// Drain the unreported list and return it.
    ///
    /// **Retained deliberately** even though no in-tree caller uses it today:
    /// it is part of the public collector API and an external batch/checkpoint
    /// consumer may. The `mem::take` fast path below is pinned by
    /// `drain_unreported_leaves_reported`, so it is not unowned maintenance
    /// debt.
    #[must_use = "dropping the drained diagnostics destroys them"]
    pub fn drain_unreported(&mut self) -> Vec<Diagnostic<'src>> {
        if self.reported_len == 0 {
            self.errors = 0;
            self.warnings = 0;
            self.unreported_errors = 0;
            self.unreported_warnings = 0;
            self.refresh_unsourced_flag();
            return std::mem::take(&mut self.diags).into_vec();
        }
        // One allocation, no `into_vec` + `split_off` pair: `SmallVec::drain`
        // removes the suffix in place and the collect moves exactly those
        // elements out.  The retained prefix never leaves the inline buffer.
        let drained: Vec<Diagnostic<'src>> = self.diags.drain(self.reported_len..).collect();
        // O(1): the suffix tallies already hold exactly this region's counts
        // (`push`/`truncate`/`recount` keep them in lockstep), which is the same
        // identity `discard_unreported` already relies on.  Rescanning the
        // drained slice just re-derived a number the collector had.
        self.errors = self.errors.saturating_sub(self.unreported_errors);
        self.warnings = self.warnings.saturating_sub(self.unreported_warnings);
        #[cfg(debug_assertions)]
        {
            let mut errors = 0usize;
            let mut warnings = 0usize;
            for d in &drained {
                if d.is_error() {
                    errors += 1;
                } else if d.is_warning() {
                    warnings += 1;
                }
            }
            debug_assert_eq!(
                (errors, warnings),
                (self.unreported_errors, self.unreported_warnings),
                "drain_unreported tallies disagreed with the drained slice"
            );
        }
        self.unreported_errors = 0;
        self.unreported_warnings = 0;
        self.refresh_unsourced_flag();
        drained
    }

    /// Drain the reported list and return it.
    #[must_use = "dropping the drained diagnostics destroys them"]
    pub fn drain_reported(&mut self) -> Vec<Diagnostic<'src>> {
        let reported_len = self.reported_len;
        if reported_len == 0 {
            return Vec::new();
        }
        // Symmetric to `drain_unreported`: one in-place drain, one allocation,
        // and the reported-prefix tallies supply the subtraction in O(1).
        let drained: Vec<Diagnostic<'src>> = self.diags.drain(..reported_len).collect();
        self.errors = self.errors.saturating_sub(self.reported_errors);
        self.warnings = self.warnings.saturating_sub(self.reported_warnings);
        #[cfg(debug_assertions)]
        {
            let mut errors = 0usize;
            let mut warnings = 0usize;
            for d in &drained {
                if d.is_error() {
                    errors += 1;
                } else if d.is_warning() {
                    warnings += 1;
                }
            }
            debug_assert_eq!(
                (errors, warnings),
                (self.reported_errors, self.reported_warnings),
                "drain_reported tallies disagreed with the drained slice"
            );
        }
        // The drained region was reported by construction, so the
        // unreported counters are untouched.
        self.reported_len = 0;
        self.reported_errors = 0;
        self.reported_warnings = 0;
        self.refresh_unsourced_flag();
        drained
    }

    /// Drop the reported prefix without allocating.
    ///
    /// [`Self::drain_reported`] has to allocate a `Vec` for the caller; this is
    /// the "just get rid of it" form, which only fixes the tallies and the
    /// partition.
    pub fn discard_reported(&mut self) {
        let reported_len = self.reported_len;
        if reported_len == 0 {
            return;
        }
        self.diags.drain(..reported_len);
        self.reported_len = 0;
        self.reported_errors = 0;
        self.reported_warnings = 0;
        self.errors = self.unreported_errors;
        self.warnings = self.unreported_warnings;
        self.refresh_unsourced_flag();
    }

    /// Drop the unreported suffix without allocating.
    ///
    /// Symmetric to [`Self::discard_reported`]; the O(1) tally path is the same
    /// one `truncate_unreported(0)` takes.
    pub fn discard_unreported(&mut self) {
        // One tally algebra, not two: this is exactly `truncate_unreported(0)`
        // (the early-return and the no-op cases coincide), and the two
        // hand-maintained copies would have to be updated in lockstep for every
        // future tally field.
        self.truncate_unreported(0);
    }

    /// Warnings dropped by the `can_emit_warnings` gate since construction.
    ///
    /// Monotone: suppressed diagnostics are never stored, so this cannot be
    /// derived from [`Self::iter`] (and [`Self::clear`] deliberately keeps it).
    pub fn suppressed_warning_count(&self) -> usize {
        self.suppressed_warnings
    }

    /// Internal invariant violations reported at the emission boundary.
    ///
    /// Monotone, like [`Self::suppressed_warning_count`]: a bug report is an
    /// observation about this context's lifetime, not derived state.
    pub fn internal_bug_count(&self) -> usize {
        self.internal_bugs
    }

    /// Report an internal invariant violation: loud, counted, never panics.
    ///
    /// Diagnostic output must stay alive even when an upstream invariant
    /// broke — the user's compile is worth more than our assertion.  `stderr`
    /// makes the gap visible in release builds; the counter makes it
    /// assertable in tests without parsing output.
    fn report_internal_bug(&mut self, msg: &str) {
        self.internal_bugs += 1;
        // `eprintln!` panics when stderr is closed or broken (daemonized
        // processes, `2>&-`, closed CI pipes), which would take down the very
        // emission path this function exists to keep alive.  The counter stays
        // the assertable record; the write is best-effort.
        use std::io::Write as _;
        let _ = writeln!(io::stderr(), "ponent: internal bug: {msg}");
    }

    /// Terminal source audit, run by [`Self::emit`] before the handoff.
    ///
    /// This is the only component that can observe a breach occurring *after*
    /// the last `attach_source` (no later scan exists), and it also covers the
    /// one failure class the write-once attach leaves open: a diagnostic that
    /// never got a source at all.  `pub(crate)` so tests can drive it without
    /// constructing an emitter.  Scans only the batch about to be emitted, so a
    /// long-lived context stays O(suffix) per emit rather than O(n).
    pub(crate) fn audit_source_attachment(&mut self) {
        if !self.may_have_unsourced_suffix {
            // Every push since the last clean scan carried a source and no later
            // step can remove one, so the O(suffix) scan below is dead work.
            return;
        }
        let start = self.reported_len;
        let suffix = &self.diags[start..];
        let Some(first) = suffix.iter().position(|d| d.source().is_none()) else {
            // Exact now: this scan covered precisely the suffix, so it clears
            // precisely the suffix flag.  The prefix keeps its own flag (armed by
            // `mark_all_reported`), so a clean suffix with a nonempty prefix no
            // longer freezes the gate permanently *and* still cannot manufacture
            // the stale `false` the one-way soundness rule forbids.
            self.may_have_unsourced_suffix = false;
            return;
        };
        // Bug state only, so the second pass is acceptable — and reporting just
        // the first offender hid the blast radius exactly when it mattered
        // (500 unsourced diagnostics used to read as one).
        let total = suffix.iter().filter(|d| d.source().is_none()).count();
        let msg = if total == 1 {
            format!("diag #{} reached emission without a source", start + first)
        } else {
            format!(
                "{total} diagnostics reached emission without a source (first: #{})",
                start + first
            )
        };
        self.report_internal_bug(&msg);
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
    ///
    /// **The gate check is deliberately asymmetric with
    /// [`Self::warnings_to_infos`]**: this direction *creates* warnings, so
    /// with the gate off it would manufacture output the gate exists to
    /// suppress.  The other direction only relabels to `Info`, which is not
    /// gated, and cannot create printable warnings from nothing.
    ///
    /// Neither direction can silence warnings pushed *before* the gate was
    /// disabled — the gate is push-time only.  Use
    /// [`Self::truncate_unreported`] at a checkpoint boundary for that.
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
    ///
    /// Unlike [`Self::errors_to_warnings`] this does **not** consult the
    /// push-time `can_emit_warnings` gate: `Info` is not gated output, so the
    /// conversion cannot manufacture anything `emit` would have suppressed.
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

/// One pass over `diags` deriving every cached tally.
///
/// Shared by `From<Vec>` and [`DiagCtxt::recount_derived_state`] so the initial
/// construction and the post-emit recount cannot drift apart: `reported_len`
/// splits "total" from "unreported" counts, and `From<Vec>` passes `0` because
/// a freshly built context has nothing reported yet.
fn scan_derived(
    diags: &[Diagnostic<'_>],
    reported_len: usize,
) -> (usize, usize, usize, usize, bool) {
    let mut errors = 0usize;
    let mut warnings = 0usize;
    let mut unreported_errors = 0usize;
    let mut unreported_warnings = 0usize;
    let mut has_unsourced = false;
    for (i, d) in diags.iter().enumerate() {
        has_unsourced |= d.source().is_none();
        if d.is_error() {
            errors += 1;
            if i >= reported_len {
                unreported_errors += 1;
            }
        } else if d.is_warning() {
            warnings += 1;
            if i >= reported_len {
                unreported_warnings += 1;
            }
        }
    }
    (
        errors,
        warnings,
        unreported_errors,
        unreported_warnings,
        has_unsourced,
    )
}

impl Default for DiagCtxt<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'src> From<Vec<Diagnostic<'src>>> for DiagCtxt<'src> {
    fn from(diags: Vec<Diagnostic<'src>>) -> Self {
        // One shared scan with `recount_derived_state`: every diagnostic starts
        // unreported (`reported_len == 0`), so the totals and the unreported
        // tallies coincide.  Deriving them twice is how the initial
        // construction and the post-emit recount drift apart.
        let (errors, warnings, unreported_errors, unreported_warnings, has_unsourced) =
            scan_derived(&diags, 0);
        DiagCtxt {
            diags: SmallVec::from_vec(diags),
            reported_len: 0,
            can_emit_warnings: true,
            errors,
            warnings,
            unreported_errors,
            unreported_warnings,
            reported_errors: 0,
            reported_warnings: 0,
            suppressed_warnings: 0,
            internal_bugs: 0,
            // The constructor's own scan already answered this for the whole
            // (unreported) vector; assuming `true` made every sourced batch pay
            // one wasted audit at first emit and gave up the O(1) path.
            may_have_unsourced_prefix: false,
            may_have_unsourced_suffix: has_unsourced,
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
    fn attach_source_is_idempotent_and_patches_later_unspanned_pushes() {
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

    /// Provenance lock: the first attach wins for an already-sourced
    /// diagnostic, even when a differently-sourced neighbour appears between
    /// the two attaches.  Without the `is_none` guard the second attach would
    /// overwrite the first segment's source, and a wrong snippet is
    /// indistinguishable from a right one at emission time.
    #[test]
    fn attach_source_is_first_wins_for_already_sourced_diagnostics() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.push(Diagnostic::error("b").with_source("src"));
        c.attach_source("src2");
        assert_eq!(c.iter().nth(0).unwrap().source(), Some("src2"));
        assert_eq!(c.iter().nth(1).unwrap().source(), Some("src"));
    }

    /// The terminal audit reports once per emission batch that contains an
    /// unsourced diagnostic, never panics, and never drops the diagnostic:
    /// a compile with an incomplete snippet is still worth emitting.
    #[test]
    fn audit_source_attachment_reports_once_without_panicking() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("no source"));
        c.audit_source_attachment();
        assert_eq!(c.internal_bug_count(), 1);
        // A second emission is an independent observation: it reports again
        // rather than staying silent, but still does not panic or drop.
        c.audit_source_attachment();
        assert_eq!(c.internal_bug_count(), 2);
        assert_eq!(c.len(), 1);
        // Attaching a source clears the condition.
        c.attach_source("src");
        c.audit_source_attachment();
        assert_eq!(c.internal_bug_count(), 2);
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

    /// `mark_all_reported` must clear BOTH unreported tallies: with only the
    /// error count cleared, `for_each_warning`'s `warnings ==
    /// unreported_warnings` fast path scans the empty unreported suffix and
    /// silently drops the reported warnings.
    #[test]
    fn mark_all_reported_clears_the_unreported_warning_tally() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::warning("w"));
        c.mark_all_reported();
        let mut seen = 0;
        c.for_each_warning(|_| seen += 1);
        assert_eq!(seen, 1, "the reported warning must still be enumerated");
        assert_eq!(c.warning_count(), 1);

        // A later downgrade gates on the same tally: it must not resurrect the
        // already-reported warning, which would double-count it as unreported.
        c.warnings_to_infos();
        assert_eq!(c.warning_count(), 1);
        assert!(c.diags[0].is_warning(), "reported warning was converted");

        // A push after the mark starts from a clean base.
        c.push(Diagnostic::warning("w2"));
        c.warnings_to_infos();
        assert!(c.diags[0].is_warning(), "reported warning was converted");
        assert!(!c.diags[1].is_warning());
        assert_eq!(c.warning_count(), 1);
    }

    /// Dropping the whole unreported suffix takes the O(1) tally path; the
    /// tallies must come out exactly as the per-diagnostic subtraction would
    /// leave them.
    #[test]
    fn truncate_unreported_to_zero_clears_unreported_tallies() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("reported-error"));
        c.push(Diagnostic::warning("reported-warning"));
        c.mark_all_reported();
        c.push(Diagnostic::error("dropped-error"));
        c.push(Diagnostic::warning("dropped-warning"));
        c.push(Diagnostic::note("dropped-note"));
        assert_eq!(c.unreported_len(), 3);

        c.truncate_unreported(0);
        assert_eq!(c.unreported_len(), 0);
        assert_eq!(c.reported().len(), 2);
        assert_eq!(c.error_count(), 1);
        assert_eq!(c.warning_count(), 1);
        assert!(!c.has_unreported_errors());

        let mut warnings = 0;
        c.for_each_warning(|_| warnings += 1);
        assert_eq!(warnings, 1, "the reported warning must survive");
    }

    /// A stale (too large) count is clamped in release instead of silently
    /// truncating nothing.
    #[test]
    fn truncate_unreported_clamps_a_stale_count() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("a"));
        c.push(Diagnostic::error("b"));
        // `usize::MAX` reproduces the release-mode wrap the saturating add and
        // the clamp exist to prevent.
        let stale = if cfg!(debug_assertions) {
            2
        } else {
            usize::MAX
        };
        c.truncate_unreported(stale);
        assert_eq!(c.unreported_len(), 2);
        assert_eq!(c.error_count(), 2);
    }

    /// The `saturating_add` guard is only reachable where the debug assert is
    /// compiled out, and only with a non-empty reported prefix: with
    /// `reported_len > 0`, `reported_len + usize::MAX` wraps (dropping every
    /// diagnostic) unless the add saturates and the `min` clamps.  The test
    /// above exercises `reported_len == 0`, where no wrap can happen.
    #[cfg(not(debug_assertions))]
    #[test]
    fn truncate_unreported_saturates_with_a_reported_prefix() {
        let mut c = DiagCtxt::new();
        c.push(Diagnostic::error("reported"));
        c.mark_all_reported();
        c.push(Diagnostic::warning("kept"));
        c.truncate_unreported(usize::MAX);
        assert_eq!(c.reported().len(), 1, "the reported prefix must survive");
        assert_eq!(c.unreported_len(), 1, "nothing to truncate");
        assert_eq!(c.error_count(), 1);
        assert_eq!(c.warning_count(), 1);
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
    /// 1.1's pin: the same batch through `extend` and through `extend_from_vec`
    /// must report the same number of suppressed warnings.  The `retain` path
    /// dropped them before `push` could count them, so this parity was the whole
    /// bug.
    #[test]
    fn suppressed_count_parity_between_extend_and_extend_from_vec() {
        let batch = || {
            (0..8)
                .map(|i| {
                    if i % 2 == 0 {
                        Diagnostic::warning("w")
                    } else {
                        Diagnostic::error("e")
                    }
                })
                .collect::<Vec<_>>()
        };

        let mut via_extend = DiagCtxt::new();
        via_extend.set_can_emit_warnings(false);
        via_extend.extend(batch());

        let mut via_vec = DiagCtxt::new();
        via_vec.set_can_emit_warnings(false);
        via_vec.extend_from_vec(batch());

        assert_eq!(via_extend.suppressed_warning_count(), 4);
        assert_eq!(
            via_vec.suppressed_warning_count(),
            via_extend.suppressed_warning_count(),
            "the pre-filtered path must credit the same suppressions"
        );
        assert_eq!(via_vec.error_count(), via_extend.error_count());
        assert_eq!(via_vec.warning_count(), 0);
    }

    /// 1.4's pin: the two region flags are exact, so a clean suffix audit clears
    /// only the suffix flag and the O(1) `attach_source` gate survives a
    /// long-lived context with a reported prefix.
    #[test]
    fn unsourced_flags_are_per_region() {
        let mut ctx = DiagCtxt::new();
        ctx.push(Diagnostic::error("no source"));
        ctx.mark_all_reported();
        assert!(ctx.may_have_unsourced_prefix);
        assert!(!ctx.may_have_unsourced_suffix);

        // A clean suffix audit must not clear the prefix flag (the prefix was
        // never audited), but it does clear the suffix flag exactly.
        ctx.audit_source_attachment();
        assert!(ctx.may_have_unsourced_prefix);
        assert!(!ctx.may_have_unsourced_suffix);

        // A full-vector patch clears both, and the fast path is armed again.
        ctx.attach_source("src/x.pn");
        assert!(!ctx.may_have_unsourced_prefix);
        assert!(!ctx.may_have_unsourced_suffix);
        for d in ctx.iter() {
            assert!(d.source().is_some());
        }
    }

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
                    .push_str(simdutf8::basic::from_utf8(buf).expect("diagnostic output is UTF-8"));
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
                    .push_str(simdutf8::basic::from_utf8(buf).expect("diagnostic output is UTF-8"));
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
