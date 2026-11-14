//! Apply machine-applicable suggestions to source text (`--fix`).
//!
//! [`SuggestionBody::Replace`] already carries everything a rewrite needs:
//! the target [`Span`] and the replacement text.  This module walks the
//! emitted diagnostics, collects the applicable edits, and resolves overlaps
//! (first-wins, matching rustc's "conflicting suggestions" behaviour).
//!
//! Application is a single **forward** pass into a fresh buffer: each edit's
//! span offsets still refer to the original source, so a later edit never sees
//! an earlier replacement and no reverse-order dance is needed.  The
//! `start >= last` guard in [`FixSet::apply_filtered`] therefore holds by
//! construction for same-file, in-order edits and degrades to first-wins for a
//! caller that mixes files into one buffer.
//!
//! The write-back path itself (reading the source, writing the fixed file,
//! re-running the check) belongs in the CLI; this is the library layer that
//! makes the write-back trivial.
//!
//! # Example
//!
//! ```ignore
//! use ponent::diagnostics::{FixHints, FixSet};
//!
//! let set = FixSet::collect(&diags, FixHints::Auto);
//! if !set.is_empty() {
//!     let fixed = set.apply(source);          // Cow: borrows when empty
//!     std::fs::write(path, fixed.as_bytes())?;
//! }
//! ```

use crate::ast::{FileId, Span};
use crate::diagnostics::{Applicability, Diagnostic, SuggestionStyle};
use std::borrow::Cow;

/// Which suggestion applicability levels a fix pass accepts.
///
/// Mirrors rustc's `--fix` / `--fix-hints` semantics:
///
/// - [`Auto`](Self::Auto): only [`Applicability::MachineApplicable`] —
///   the safe `--fix` default.
/// - [`Review`](Self::Review): also [`Applicability::MaybeIncorrect`].
/// - [`All`](Self::All): every applicability, including
///   [`Applicability::HasPlaceholders`] and
///   [`Applicability::Unspecified`] (still never applies two overlapping
///   edits — see [`FixSet::collect`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FixHints {
    /// `--fix` (safe edits only).
    #[default]
    Auto,
    /// `--fix-hints=review` (machine-applicable plus review-needed).
    Review,
    /// `--fix-hints=all`.
    All,
}

impl FixHints {
    fn accepts(self, app: Applicability) -> bool {
        match self {
            FixHints::Auto => matches!(app, Applicability::MachineApplicable),
            FixHints::Review => matches!(
                app,
                Applicability::MachineApplicable | Applicability::MaybeIncorrect
            ),
            FixHints::All => true,
        }
    }
}

/// A single edit collected from a diagnostic's suggestions.
///
/// `replacement` keeps the suggestion's own `Cow`: a compiler-authored fix
/// is `Cow::Borrowed` (often `&'static str`), so collecting it clones a
/// pointer instead of copying the bytes onto the heap.  The lifetime is the
/// diagnostics' `'src`, so the borrow checker guarantees the set cannot
/// outlive the `Diagnostic` that owns the text.
#[derive(Debug, Clone)]
pub struct FixEdit<'src> {
    pub span: Span,
    pub replacement: Cow<'src, str>,
    pub applicability: Applicability,
}

/// Non-overlapping edits for one source buffer.
///
/// Sorted by `(file, start, end)`.  Overlapping candidates are discarded
/// after the first (lowest-offset wins) so applying the set produces a
/// consistent result — a conservative union would double-write overlapping
/// regions.  This matches rustc's behaviour on conflicting suggestions.
#[derive(Debug, Default, Clone)]
pub struct FixSet<'src> {
    edits: Vec<FixEdit<'src>>,
}

impl<'src> FixSet<'src> {
    /// Collect applicable fixes from `diags`.
    ///
    /// Only [`SuggestionBody::Replace`] suggestions are collected;
    /// text-only suggestions carry no span to rewrite.  Suggestions whose
    /// applicability is not accepted by `hints` are skipped.
    ///
    /// **Visibility policy:** suggestions styled
    /// [`SuggestionStyle::HideCodeAlways`] or
    /// [`SuggestionStyle::CompletelyHidden`] are skipped even when
    /// machine-applicable.  `--fix` applies only what the renderer shows the
    /// user; a hidden suggestion is invisible by construction, so silently
    /// rewriting source with it would make the tool change files in ways the
    /// output never mentioned.  (A future `--fix-hidden` flag could opt in
    /// by widening this filter.)
    ///
    /// **Conflict resolution is by position, never by applicability.**  When
    /// two edits overlap, the one that sorts first wins (a wide
    /// `MaybeIncorrect` edit can therefore shadow a narrower
    /// `MachineApplicable` one) — matching rustc; *not* "the safest edit
    /// wins".
    pub fn collect(diags: &[Diagnostic<'src>], hints: FixHints) -> Self {
        let mut candidates: Vec<FixEdit<'src>> = Vec::new();
        for diag in diags {
            for sug in diag.suggestions() {
                if matches!(
                    sug.style,
                    SuggestionStyle::HideCodeAlways | SuggestionStyle::CompletelyHidden
                ) {
                    continue;
                }
                if !hints.accepts(sug.applicability) {
                    continue;
                }
                let (Some(span), Some(replacement)) = (&sug.span, &sug.replacement) else {
                    continue;
                };
                // Reject reversed spans at the gate so `len()`, the overlap
                // cursor below and `apply_filtered` all agree on one edit
                // universe (zero-width spans stay legal: they are insertions).
                if span.start_as_usize() > span.end_as_usize() {
                    continue;
                }
                candidates.push(FixEdit {
                    span: *span,
                    // A pointer copy for a borrowed replacement; the common
                    // compiler-authored fix never touches the heap.
                    replacement: replacement.clone(),
                    applicability: sug.applicability,
                });
            }
        }
        if !candidates.is_sorted_by_key(|e| {
            (e.span.file, e.span.start_as_usize(), e.span.end_as_usize())
        }) {
            candidates.sort_by_key(|e| {
                (e.span.file, e.span.start_as_usize(), e.span.end_as_usize())
            });
        }
        // Drop overlapping candidates: first-wins.  Zero-length edits count
        // as overlapping too: two insertions at one point (or an insertion at
        // the edge of a replacement) would both run and compose into an
        // order-dependent result, so the later one is dropped.  Edits that
        // merely *touch* at a boundary (a replacement ending where a
        // non-zero-length edit starts) are disjoint and both kept.
        //
        // Compact in place with a write cursor instead of allocating a second
        // `Vec`: `FixEdit` carries a `Cow`, so a move is cheap, and a batch
        // with many overlapping suggestions would otherwise pay a full second
        // allocation + copy.
        let mut last: Option<(FileId, usize, usize)> = None;
        let mut write = 0usize;
        for read in 0..candidates.len() {
            let (file, start, end) = {
                let edit = &candidates[read];
                (
                    edit.span.file,
                    edit.span.start_as_usize(),
                    edit.span.end_as_usize(),
                )
            };
            if let Some((prev_file, prev_start, prev_end)) = last
                && prev_file == file
                && (start < prev_end
                    || (start == prev_end && (start == end || prev_start == prev_end)))
            {
                continue;
            }
            last = Some((file, start, end));
            if write != read {
                candidates.swap(write, read);
            }
            write += 1;
        }
        candidates.truncate(write);
        FixSet { edits: candidates }
    }

    /// Whether there is nothing to apply.
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Number of edits in the set.
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Iterate over the retained edits in source order.
    pub fn iter(&self) -> impl Iterator<Item = &FixEdit<'src>> + '_ {
        self.edits.iter()
    }

    /// Apply the edits to `source`, returning the rewritten text.
    ///
    /// The source must be the one the spans reference (the emitter contract:
    /// every diagnostic for one file shares that file's source buffer).
    /// Edits are applied in ascending order into a fresh buffer, so every
    /// edit's offsets refer to the original source.  Edits whose span is
    /// malformed (reversed,
    /// past the end, or not on a char boundary) are skipped rather than
    /// panicking — the diagnostic core guarantees well-formed spans, but a
    /// `--fix` run must never crash on a producer bug.
    ///
    /// An empty set borrows `source` unchanged, so a file with no applicable
    /// edits pays no copy to discover there is nothing to do.
    ///
    /// Prefer [`Self::apply_for_file`] whenever the batch may span multiple
    /// files: `apply` interprets *every* retained edit as an offset into this
    /// one buffer, so a foreign-file edit would corrupt a byte range it was
    /// never meant to touch (the `start < last` guard below catches only the
    /// accidental overlaps, not the wrong-file ones that happen to land in
    /// order).
    pub fn apply<'s>(&self, source: &'s str) -> Cow<'s, str> {
        // Anchor on the first edit's file and apply only that file's edits.
        // A multi-file set must never splice another file's byte offsets into
        // this buffer; previously only a debug build caught the mistake, and
        // release builds silently corrupted it.  Dropping the foreign edits is
        // the safe degradation; `apply_for_file` is the explicit per-file API.
        let anchor = self.edits.first().map(|e| e.span.file);
        debug_assert!(
            anchor.is_none_or(|f| self.edits.iter().all(|e| e.span.file == f)),
            "FixSet::apply called on a multi-file set; use apply_for_file",
        );
        self.apply_filtered(source, |e| anchor.is_none_or(|f| e.span.file == f))
    }

    /// Apply only the edits that target `file` to `source`, returning the
    /// rewritten text.
    ///
    /// [`Self::collect`] gathers suggestions from a whole batch, and a batch
    /// can carry diagnostics from several files; applying such a set to one
    /// file's buffer with [`Self::apply`] would splice another file's edits
    /// into it by raw byte offset.  This entry point filters by
    /// `FileId` first, so `--fix` write-back per file is correct by
    /// construction.
    ///
    /// A file with no matching edits borrows `source` unchanged — the
    /// filter runs in one pass and no per-file `FixSet` (nor a clone of its
    /// edits) is materialized.
    pub fn apply_for_file<'s>(&self, source: &'s str, file: FileId) -> Cow<'s, str> {
        self.apply_filtered(source, |e| e.span.file == file)
    }

    /// The single-file forward pass shared by [`Self::apply`] and
    /// [`Self::apply_for_file`], running only the edits for which `pred`
    /// holds.  No allocation happens unless at least one edit actually
    /// applies.
    fn apply_filtered<'s>(
        &self,
        source: &'s str,
        pred: impl Fn(&FixEdit<'src>) -> bool,
    ) -> Cow<'s, str> {
        self.apply_filtered_count(source, pred).0
    }

    /// Same one-pass apply, additionally returning the number of edits that
    /// actually spliced.  Malformed spans are skipped, so this can be smaller
    /// than `filter(pred).count()`.
    fn apply_filtered_count<'s>(
        &self,
        source: &'s str,
        pred: impl Fn(&FixEdit<'src>) -> bool,
    ) -> (Cow<'s, str>, usize) {
        let mut applied = 0usize;
        // One pass: allocate only when the first applicable edit is seen, and
        // return the borrowed source when none applies.  The old shape walked
        // `edits` three times (any-check, size sum, write); the write pass is
        // already the authoritative "did anything apply" test.
        let mut out: Option<String> = None;
        let mut last = 0usize;
        for edit in self.edits.iter() {
            if !pred(edit) {
                continue;
            }
            // `collect` sorts edits by `(file, start, end)` and drops
            // overlapping candidates within a file, so a single forward pass
            // builds the result in O(n + Σ|replacement|).
            //
            // Malformed spans (reversed, past the end, or not on a char
            // boundary) are skipped rather than panicking — the diagnostic
            // core guarantees well-formed spans, but a `--fix` run must never
            // crash on a producer bug.  For non-overlapping same-file edits
            // the checks against `source` are equivalent to checks against the
            // mutated buffer: higher-offset edits only ever modified the
            // region above a lower edit's end.
            let start = edit.span.start_as_usize();
            let end = edit.span.end_as_usize();
            if start > end || end > source.len() {
                continue;
            }
            if !source.is_char_boundary(start) || !source.is_char_boundary(end) {
                continue;
            }
            // `collect` already guarantees non-overlapping, in-order spans
            // within a file, so `start >= last` holds there.  The guard keeps
            // the pass sound if a caller mixes files into one set applied to a
            // single buffer by offset; first-wins, matching `collect`.
            if start < last {
                continue;
            }
            let buf = out.get_or_insert_with(|| String::with_capacity(source.len()));
            buf.push_str(&source[last..start]);
            buf.push_str(&edit.replacement);
            last = end;
            applied += 1;
        }
        let rewritten = match out {
            Some(mut buf) => {
                buf.push_str(&source[last..]);
                Cow::Owned(buf)
            }
            None => Cow::Borrowed(source),
        };
        (rewritten, applied)
    }
}

/// Convenience: collect fixes from a batch and apply them to `source`,
/// returning `(rewritten_source, edit_count)`.
///
/// `source` is assumed to belong to the file of the set's first edit; only
/// that file's edits are applied and counted, so a batch that spans files can
/// never splice a foreign offset into this buffer.  Prefer
/// [`apply_fixes_for_file`] when the `FileId` is known.
pub fn apply_fixes<'src, 's>(
    diags: &[Diagnostic<'src>],
    source: &'s str,
    hints: FixHints,
) -> (Cow<'s, str>, usize) {
    let set = FixSet::collect(diags, hints);
    let anchor = set.edits.first().map(|e| e.span.file);
    set.apply_filtered_count(source, |e| anchor.is_none_or(|f| e.span.file == f))
}

/// Like [`apply_fixes`] but for an explicit file: apply and count only the
/// edits targeting `file`.  This is the correct entry point for a driver that
/// writes fixes back per file.
pub fn apply_fixes_for_file<'src, 's>(
    diags: &[Diagnostic<'src>],
    source: &'s str,
    file: FileId,
    hints: FixHints,
) -> (Cow<'s, str>, usize) {
    let set = FixSet::collect(diags, hints);
    set.apply_filtered_count(source, |e| e.span.file == file)
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::diagnostics::Diagnostic;
    use crate::diagnostics::ErrCode;

    fn diag_with_fix(
        span: Span,
        replacement: &'static str,
        app: Applicability,
    ) -> Diagnostic<'static> {
        Diagnostic::error("placeholder")
            .with_code_str(ErrCode::E030)
            .with_span(span)
            .with_span_suggestion(span, replacement, "apply fix", app)
    }

    #[test]
    fn auto_collects_machine_applicable_only() {
        let src = "dem main(){}";
        let d1 = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "def",
            Applicability::MachineApplicable,
        );
        let d2 = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "foo",
            Applicability::MaybeIncorrect,
        );
        let set = FixSet::collect(&[d1, d2], FixHints::Auto);
        assert_eq!(set.len(), 1);
        assert_eq!(set.iter().next().unwrap().replacement, "def");
        assert_eq!(set.apply(src), "def main(){}");
    }

    #[test]
    fn review_accepts_maybe_incorrect() {
        let d1 = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "def",
            Applicability::MachineApplicable,
        );
        let d2 = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "foo",
            Applicability::MaybeIncorrect,
        );
        // Both accepted; d1 (MachineApplicable) wins the overlap.
        let set = FixSet::collect(&[d1, d2], FixHints::Review);
        assert_eq!(set.len(), 1);
        assert_eq!(set.iter().next().unwrap().replacement, "def");
    }

    #[test]
    fn overlapping_edits_are_dropped_first_wins() {
        let src = "abcdef";
        let d1 = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "XXX",
            Applicability::MachineApplicable,
        );
        let d2 = diag_with_fix(
            Span::new(FileId::DUMMY, 2, 5),
            "YYY",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[d1, d2], FixHints::Auto);
        assert_eq!(set.len(), 1);
        assert_eq!(set.iter().next().unwrap().replacement, "XXX");
        assert_eq!(set.apply(src), "XXXdef");
    }

    #[test]
    fn malformed_span_is_rejected_at_collect() {
        let src = "abcdef";
        // Simulate a producer that got the ends backwards (Span::new would
        // reject this, so build the struct literally).
        let sug = crate::diagnostics::Suggestion {
            message: "bad".into(),
            applicability: Applicability::MachineApplicable,
            style: crate::diagnostics::SuggestionStyle::ShowAlways,
            span: Some(Span {
                file: FileId::DUMMY,
                start: 4,
                end: 2,
            }),
            replacement: Some("YYY".into()),
        };
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_suggestions(vec![sug]);
        let set = FixSet::collect(&[diag], FixHints::Auto);
        assert_eq!(set.len(), 0);
        assert_eq!(set.apply(src), src);
    }

    /// Zero-length edits are insertions and stay collectible, but two
    /// insertions at one point (or an insertion abutting a replacement) are
    /// a conflict: first-wins by the `(file, start, end)` sort order.
    #[test]
    fn zero_length_edits_conflict_first_wins() {
        let src = "abcdef";
        let ins1 = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 3),
            "1",
            Applicability::MachineApplicable,
        );
        let ins2 = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 3),
            "2",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[ins1, ins2], FixHints::Auto);
        assert_eq!(set.len(), 1);
        assert_eq!(set.apply(src), "abc1def");

        // An insertion at the start edge of a replacement with the same
        // start sorts first (end 0 < 3) and wins; the replacement is dropped.
        let ins = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 0),
            "!",
            Applicability::MachineApplicable,
        );
        let repl = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "XXX",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[repl, ins], FixHints::Auto);
        assert_eq!(set.len(), 1);
        assert_eq!(set.apply(src), "!abcdef");

        // A zero-length edit at the *end* boundary of a previous replacement
        // is a conflict (inside-vs-outside is ambiguous), but a non-zero
        // edit starting exactly at the previous end is disjoint and kept.
        let repl = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "XXX",
            Applicability::MachineApplicable,
        );
        let ins_at_end = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 3),
            "!",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[repl, ins_at_end], FixHints::Auto);
        assert_eq!(set.len(), 1);
        assert_eq!(set.apply(src), "XXXdef");

        let repl = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "XXX",
            Applicability::MachineApplicable,
        );
        let after = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 5),
            "ZZ",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[repl, after], FixHints::Auto);
        assert_eq!(set.len(), 2);
        assert_eq!(set.apply(src), "XXXZZf");
    }

    #[test]
    fn disjoint_edits_apply_in_reverse_source_order() {
        let src = "aaa bbb ccc";
        let d1 = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "XXX",
            Applicability::MachineApplicable,
        );
        let d2 = diag_with_fix(
            Span::new(FileId::DUMMY, 8, 11),
            "ZZZ",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[d1, d2], FixHints::Auto);
        assert_eq!(set.len(), 2);
        assert_eq!(set.apply(src), "XXX bbb ZZZ");
    }

    /// Regression guard for the lifetime threading: a `Cow::Borrowed`
    /// replacement must stay borrowed.  Reintroducing `to_string()` in
    /// `collect` would silently bring back one heap copy per collected fix.
    #[test]
    fn collected_replacement_stays_borrowed() {
        let diag = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "def",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[diag], FixHints::Auto);
        assert!(
            matches!(set.iter().next().unwrap().replacement, Cow::Borrowed(_)),
            "borrowed suggestion was copied into an owned String"
        );
    }

    #[test]
    fn empty_set_returns_source_unchanged() {
        let src = "abc";
        let set = FixSet::collect(&[], FixHints::All);
        assert!(set.is_empty());
        // An empty set must borrow the source rather than copy it.
        assert!(
            matches!(set.apply(src), Cow::Borrowed(_)),
            "empty set copied the source"
        );
        assert_eq!(set.apply(src), "abc");
    }

    #[test]
    fn text_only_suggestions_are_skipped() {
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_suggestion("try something else");
        let set = FixSet::collect(&[diag], FixHints::All);
        assert!(
            set.is_empty(),
            "text-only suggestion must not produce a fix"
        );
    }

    #[test]
    fn malformed_span_is_skipped() {
        let src = "abc";
        // Simulate a producer that got the ends backwards.
        let sug = crate::diagnostics::Suggestion {
            message: "bad".into(),
            applicability: Applicability::MachineApplicable,
            style: crate::diagnostics::SuggestionStyle::ShowAlways,
            span: Some(Span {
                file: FileId::DUMMY,
                start: 5,
                end: 2,
            }),
            replacement: Some("x".into()),
        };
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_suggestions(vec![sug]);
        let set = FixSet::collect(&[diag], FixHints::All);
        // `collect` rejects reversed spans at the gate, so the edit never
        // enters the set (and `apply` consequently never sees one).
        assert_eq!(set.len(), 0);
        assert_eq!(set.apply(src), "abc");
    }

    #[test]
    fn non_char_boundary_span_is_skipped() {
        let src = "é"; // 2-byte char
        let sug = crate::diagnostics::Suggestion {
            message: "bad".into(),
            applicability: Applicability::MachineApplicable,
            style: crate::diagnostics::SuggestionStyle::ShowAlways,
            span: Some(Span {
                // end=1 sits inside `é`
                file: FileId::DUMMY,
                start: 0,
                end: 1,
            }),
            replacement: Some("x".into()),
        };
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 2))
            .with_suggestions(vec![sug]);
        let set = FixSet::collect(&[diag], FixHints::All);
        // Must not panic.
        let _ = set.apply(src);
    }

    #[test]
    fn apply_fixes_returns_count_and_new_source() {
        let src = "dem main(){}";
        let diag = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "def",
            Applicability::MachineApplicable,
        );
        let (fixed, n) = apply_fixes(&[diag], src, FixHints::Auto);
        assert_eq!(n, 1);
        assert_eq!(fixed, "def main(){}");
    }

    /// A machine-applicable but *hidden* suggestion must not be auto-applied
    /// by `--fix`: the renderer would never show it, so the rewrite would be
    /// invisible to the user (the `collect` visibility policy).
    #[test]
    fn hidden_machine_applicable_suggestions_are_not_collected() {
        let mk = |style| crate::diagnostics::Suggestion {
            message: "fix".into(),
            applicability: Applicability::MachineApplicable,
            style,
            span: Some(Span::new(FileId::DUMMY, 0, 3)),
            replacement: Some("XXX".into()),
        };
        let hidden = Diagnostic::error("boom").with_suggestions(vec![
            mk(SuggestionStyle::HideCodeAlways),
            mk(SuggestionStyle::CompletelyHidden),
        ]);
        assert!(
            FixSet::collect(&[hidden], FixHints::Auto).is_empty(),
            "hidden suggestions must not be collected"
        );
        // The visible control is still collected (same applicability).
        let visible = Diagnostic::error("boom")
            .with_suggestions(vec![mk(SuggestionStyle::ShowAlways)]);
        let set = FixSet::collect(&[visible], FixHints::Auto);
        assert_eq!(set.len(), 1);
        assert_eq!(set.apply("dem x"), "XXX x");
    }

    /// `apply_for_file` filters per file; a file with no edits borrows the
    /// source, and per-file buffers never see foreign-file offsets.
    #[test]
    fn apply_for_file_filters_by_file_and_borrows_when_untouched() {
        let f1 = FileId::from_raw(FileId::FIRST_REAL);
        let f2 = FileId::from_raw(FileId::FIRST_REAL + 1);
        let mk = |file, start, end, rep: &'static str| crate::diagnostics::Suggestion {
            message: "fix".into(),
            applicability: Applicability::MachineApplicable,
            style: SuggestionStyle::ShowAlways,
            span: Some(Span::new(file, start, end)),
            replacement: Some(rep.into()),
        };
        let d1 = Diagnostic::error("a").with_suggestions(vec![mk(f1, 0, 3, "XXX")]);
        let d2 = Diagnostic::error("b").with_suggestions(vec![mk(f2, 0, 3, "YYY")]);
        let set = FixSet::collect(&[d1, d2], FixHints::Auto);
        assert_eq!(set.len(), 2);
        // f1's buffer gets only f1's edit (the old clone-and-filter
        // behaviour, minus the allocation).
        assert_eq!(set.apply_for_file("abcdef", f1), "XXXdef");
        assert_eq!(set.apply_for_file("abcdef", f2), "YYYdef");
        // f3 has no edits: borrow, no copy.
        let out = set.apply_for_file("abcdef", FileId::DUMMY);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, "abcdef");
    }
}
