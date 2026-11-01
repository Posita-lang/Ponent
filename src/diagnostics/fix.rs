//! Apply machine-applicable suggestions to source text (`--fix`).
//!
//! [`SuggestionBody::Replace`] already carries everything a rewrite needs:
//! the target [`Span`] and the replacement text.  This module walks the
//! emitted diagnostics, collects the applicable edits, resolves overlaps
//! (first-wins, matching rustc's "conflicting suggestions" behaviour), and
//! applies them in reverse span order so earlier offsets stay valid.
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
use crate::diagnostics::{Applicability, Diagnostic};
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
    pub fn collect(diags: &[Diagnostic<'src>], hints: FixHints) -> Self {
        let mut candidates: Vec<FixEdit<'src>> = Vec::new();
        for diag in diags {
            for sug in diag.suggestions() {
                if !hints.accepts(sug.applicability) {
                    continue;
                }
                let (Some(span), Some(replacement)) = (&sug.span, &sug.replacement) else {
                    continue;
                };
                candidates.push(FixEdit {
                    span: *span,
                    // A pointer copy for a borrowed replacement; the common
                    // compiler-authored fix never touches the heap.
                    replacement: replacement.clone(),
                    applicability: sug.applicability,
                });
            }
        }
        candidates.sort_by_key(|e| (e.span.file, e.span.start_as_usize(), e.span.end_as_usize()));
        // Drop overlapping candidates: first-wins.
        let mut edits: Vec<FixEdit<'src>> = Vec::with_capacity(candidates.len());
        let mut last: Option<(FileId, usize)> = None;
        for edit in candidates {
            if let Some((file, end)) = last
                && file == edit.span.file
                && edit.span.start_as_usize() < end
            {
                continue;
            }
            last = Some((edit.span.file, edit.span.end_as_usize()));
            edits.push(edit);
        }
        FixSet { edits }
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
    pub fn apply<'s>(&self, source: &'s str) -> Cow<'s, str> {
        if self.edits.is_empty() {
            return Cow::Borrowed(source);
        }
        // `collect` sorts edits by `(file, start, end)` and drops overlapping
        // candidates within a file, so a single forward pass builds the result
        // in O(n + Σ|replacement|).  The reverse-`replace_range` loop this
        // replaces shifted the tail on every edit (`Σ (n − endᵢ) = O(k·n)`),
        // which is wasteful for files with many fixes.
        //
        // Malformed spans (reversed, past the end, or not on a char boundary)
        // are skipped rather than panicking — the diagnostic core guarantees
        // well-formed spans, but a `--fix` run must never crash on a producer
        // bug.  For non-overlapping same-file edits the checks against `source`
        // are equivalent to the old checks against the mutated buffer: higher-
        // offset edits only ever modified the region above a lower edit's end.
        let extra: usize = self.edits.iter().map(|e| e.replacement.len()).sum();
        let mut out = String::with_capacity(source.len() + extra);
        let mut last = 0usize;
        for edit in &self.edits {
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
            out.push_str(&source[last..start]);
            out.push_str(&edit.replacement);
            last = end;
        }
        out.push_str(&source[last..]);
        Cow::Owned(out)
    }
}

/// Convenience: collect fixes from a batch and apply them to `source`,
/// returning `(rewritten_source, edit_count)`.  Equivalent to
/// `FixSet::collect(...).apply(...)` plus a length; provided so the CLI
/// write-back is a one-liner.
pub fn apply_fixes<'src, 's>(
    diags: &[Diagnostic<'src>],
    source: &'s str,
    hints: FixHints,
) -> (Cow<'s, str>, usize) {
    let set = FixSet::collect(diags, hints);
    let n = set.len();
    (set.apply(source), n)
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
        assert_eq!(set.len(), 1);
        // `apply` must not panic on the malformed span — skips it.
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
}
