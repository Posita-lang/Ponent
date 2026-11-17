//! Apply machine-applicable suggestions to source text (`--fix`).
//!
//! [`SuggestionBody::Replace`] already carries everything a rewrite needs:
//! the target [`Span`] and the replacement text.  This module walks the
//! emitted diagnostics, collects the applicable edits, and resolves overlaps
//! by weighted interval scheduling (maximum applicability weight; equal
//! weights break ties by earliest end, then earliest start).
//!
//! Application is a single **forward** pass into a fresh buffer: each edit's
//! span offsets still refer to the original source, so a later edit never sees
//! an earlier replacement and no reverse-order dance is needed.  The
//! `start >= last` guard in [`FixSet::apply_filtered`] therefore holds by
//! construction for same-file, in-order edits; it is a cross-file safety guard
//! for a caller that mixes files into one buffer.
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

/// Static objective for overlap resolution.
///
/// 4 : 2 : 1 encodes "one MachineApplicable edit is worth two
/// MaybeIncorrect edits and four placeholder/unspecified edits" — safety
/// dominates count, but not absolutely: at `Review`, two narrow
/// MaybeIncorrect fixes beat one wide MachineApplicable fix only when the
/// latter is at most 2× wider.  This is a policy constant, not derived from
/// a theorem; change it here if `--fix` grows an objective flag.
fn applicability_weight(app: Applicability) -> u64 {
    match app {
        Applicability::MachineApplicable => 4,
        Applicability::MaybeIncorrect => 2,
        Applicability::HasPlaceholders | Applicability::Unspecified => 1,
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
/// Sorted by `(file, start, end)`.  Overlapping candidates are resolved by
/// maximum total applicability weight (weighted interval scheduling); equal
/// weights retain the edit that ends first, then the one that starts first.
#[derive(Debug, Default, Clone)]
pub struct FixSet<'src> {
    edits: Vec<FixEdit<'src>>,
}

#[derive(Default)]
struct ResolveScratch {
    p: Vec<usize>,
    w: Vec<u64>,
    take: Vec<bool>,
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
    /// **Conflict resolution maximises applicability weight.**  A set of
    /// pairwise-disjoint edits is chosen by weighted interval scheduling;
    /// equal-weight ties retain the edit that ends first, then the one that
    /// starts first (the deterministic `(end, start)` order).
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
        // Stable sort is load-bearing: zero-length dedup and the DP
        // `include > exclude` tiebreak preserve diagnostic input order for
        // equal keys.  Do not swap for `sort_unstable_by_key`.
        if !candidates
            .is_sorted_by_key(|e| (e.span.file, e.span.start_as_usize(), e.span.end_as_usize()))
        {
            candidates
                .sort_by_key(|e| (e.span.file, e.span.start_as_usize(), e.span.end_as_usize()));
        }
        // A single valid edit cannot overlap anything; skip the dedup/group/DP
        // machinery entirely (the common one-fix batch).
        if candidates.len() <= 1 {
            return FixSet { edits: candidates };
        }
        // Drop overlapping candidates by maximum total applicability weight.
        FixSet {
            edits: Self::resolve_overlaps(candidates),
        }
    }

    /// Maximum-weight independent set on the per-file interval graph.
    ///
    /// `candidates` must be sorted by `(file, start, end)` — `collect` sorts
    /// before calling; the zero-length dedup relies on equal
    /// `(file, start, start)` being adjacent.
    ///
    /// Intervals are half-open `[start, end)`.  Zero-length edits are points:
    /// duplicate insertions at one point collapse to the highest-weight one,
    /// but an insertion at a replacement's boundary is disjoint and survives.
    fn resolve_overlaps(candidates: Vec<FixEdit<'src>>) -> Vec<FixEdit<'src>> {
        // Candidates arrive sorted by `(file, start, end)`, so duplicate
        // zero-length runs are adjacent.  A peekable consuming iterator moves
        // each chosen edit out exactly once — no `Vec<Option<..>>` shadow
        // allocation and no per-element `take().expect()`.
        let key = |e: &FixEdit<'src>| (e.span.file, e.span.start_as_usize(), e.span.end_as_usize());
        let mut iter = candidates.into_iter().peekable();
        let mut deduped = Vec::with_capacity(iter.len());
        while let Some(edit) = iter.next() {
            let (file, start, end) = key(&edit);
            if start != end {
                deduped.push(edit);
                continue;
            }
            // Zero-length run: keep the heaviest applicability.  Strict `>`
            // preserves diagnostic input order on equal weight; the stable
            // sort makes that deterministic for a fixed batch.
            let mut best = edit;
            let mut best_weight = applicability_weight(best.applicability);
            while let Some(next) = iter.peek() {
                let (f, s, e) = key(next);
                if f != file || s != start || e != start {
                    break;
                }
                let next = iter.next().expect("peeked candidate");
                let weight = applicability_weight(next.applicability);
                if weight > best_weight {
                    best = next;
                    best_weight = weight;
                }
            }
            deduped.push(best);
        }

        let mut result = Vec::new();
        let mut scratch = ResolveScratch::default();
        let mut group: Vec<FixEdit<'src>> = Vec::new();
        let mut last_file: Option<FileId> = None;
        for edit in deduped {
            if last_file == Some(edit.span.file) {
                group.push(edit);
                continue;
            }
            if !group.is_empty() {
                let start = result.len();
                Self::resolve_file(std::mem::take(&mut group), &mut result, &mut scratch);
                debug_assert!(
                    result[start..]
                        .is_sorted_by_key(|e| { (e.span.start_as_usize(), e.span.end_as_usize()) }),
                    "resolve_file must emit non-overlapping edits in source order"
                );
            }
            last_file = Some(edit.span.file);
            group.push(edit);
        }
        if !group.is_empty() {
            let start = result.len();
            Self::resolve_file(group, &mut result, &mut scratch);
            debug_assert!(
                result[start..]
                    .is_sorted_by_key(|e| { (e.span.start_as_usize(), e.span.end_as_usize()) }),
                "resolve_file must emit non-overlapping edits in source order"
            );
        }
        result
    }

    fn resolve_file(
        mut group: Vec<FixEdit<'src>>,
        out: &mut Vec<FixEdit<'src>>,
        scratch: &mut ResolveScratch,
    ) {
        group.sort_by_key(|e| (e.span.end_as_usize(), e.span.start_as_usize()));
        let n = group.len();
        if n == 0 {
            return;
        }
        // Equal weights make weighted interval scheduling collapse to the
        // classic end-ordered greedy: the strict `include > exclude` tie rule
        // keeps the end-earliest edit at each step.  Zero-length edits still
        // compose at boundaries because `start >= last_end` is inclusive.
        let first_weight = applicability_weight(group[0].applicability);
        if group
            .iter()
            .all(|e| applicability_weight(e.applicability) == first_weight)
        {
            let mut last_end = 0usize;
            for edit in group {
                if edit.span.start_as_usize() >= last_end {
                    last_end = edit.span.end_as_usize();
                    out.push(edit);
                }
            }
            return;
        }
        // p[i] = number of earlier edits compatible with i.
        // `Option` slots let the backtrack move the chosen edit out instead
        // of cloning its `Cow` payload a second time.
        let mut group: Vec<Option<FixEdit<'src>>> = group.into_iter().map(Some).collect();
        scratch.p.clear();
        scratch.p.resize(n, 0);
        let p = &mut scratch.p;
        for (i, edit) in group.iter().enumerate() {
            let edit = edit.as_ref().expect("group slot filled");
            p[i] = group[..i].partition_point(|earlier| {
                earlier
                    .as_ref()
                    .expect("group slot filled")
                    .span
                    .end_as_usize()
                    <= edit.span.start_as_usize()
            });
        }
        // W[i] = best weight over the first i edits.  On equal weight we
        // exclude the current edit; because `group` is end-sorted, this
        // retains the edit that ends first, then starts first.
        scratch.w.clear();
        scratch.w.resize(n + 1, 0);
        scratch.take.clear();
        scratch.take.resize(n + 1, false);
        let w = &mut scratch.w;
        let take = &mut scratch.take;
        for i in 1..=n {
            let weight = applicability_weight(
                group[i - 1]
                    .as_ref()
                    .expect("group slot filled")
                    .applicability,
            );
            let include = weight + w[p[i - 1]];
            let exclude = w[i - 1];
            if include > exclude {
                w[i] = include;
                take[i] = true;
            } else {
                w[i] = exclude;
            }
        }
        let mut selected = Vec::new();
        let mut i = n;
        while i > 0 {
            if take[i] {
                selected.push(group[i - 1].take().expect("group slot filled"));
                i = p[i - 1];
            } else {
                i -= 1;
            }
        }
        selected.reverse();
        out.extend(selected);
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
    #[must_use = "apply returns the rewritten source; write it out or the fix is lost"]
    pub fn apply<'s>(&self, source: &'s str) -> Cow<'s, str> {
        self.apply_counted(source).0
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
    #[must_use = "apply_for_file returns the rewritten source; write it out or the fix is lost"]
    pub fn apply_for_file<'s>(&self, source: &'s str, file: FileId) -> Cow<'s, str> {
        self.apply_for_file_counted(source, file).0
    }

    /// [`Self::apply`] plus the number of edits that actually spliced.
    #[must_use = "apply_counted returns the rewritten source and count"]
    pub fn apply_counted<'s>(&self, source: &'s str) -> (Cow<'s, str>, usize) {
        let anchor = self.edits.first().map(|e| e.span.file);
        debug_assert!(
            anchor.is_none_or(|f| self.edits.iter().all(|e| e.span.file == f)),
            "FixSet::apply_counted called on a multi-file set; use apply_for_file_counted",
        );
        match anchor {
            Some(file) => self.apply_for_file_counted(source, file),
            None => (Cow::Borrowed(source), 0),
        }
    }

    /// [`Self::apply_for_file`] plus the number of edits that actually spliced.
    #[must_use = "apply_for_file_counted returns the rewritten source and count"]
    pub fn apply_for_file_counted<'s>(
        &self,
        source: &'s str,
        file: FileId,
    ) -> (Cow<'s, str>, usize) {
        let lo = self.edits.partition_point(|e| e.span.file < file);
        let hi = self.edits.partition_point(|e| e.span.file <= file);
        self.apply_slice_count(source, &self.edits[lo..hi])
    }

    /// The single-file forward pass over an already-selected contiguous range.
    /// No allocation happens unless at least one edit actually applies.
    fn apply_slice_count<'s>(
        &self,
        source: &'s str,
        edits: &[FixEdit<'src>],
    ) -> (Cow<'s, str>, usize) {
        let mut applied = 0usize;
        // One pass: allocate only when the first applicable edit is seen, and
        // return the borrowed source when none applies.  The old shape walked
        // `edits` three times (any-check, size sum, write); the write pass is
        // already the authoritative "did anything apply" test.
        let mut out: Option<String> = None;
        let mut last = 0usize;
        for edit in edits {
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
            // within a file, so `start >= last` holds there.  This guard is a
            // cross-file safety net for a caller that mixes files into one
            // buffer by offset; it is not the conflict-resolution policy.
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
    set.apply_counted(source)
}

/// Single-file convenience: collects once, then applies one file.
///
/// A driver writing several files should instead collect **once** and call
/// [`FixSet::apply_for_file_counted`] per file — this function re-collects on
/// every call, so using it in a per-file loop is O(files × diagnostics).
pub fn apply_fixes_for_file<'src, 's>(
    diags: &[Diagnostic<'src>],
    source: &'s str,
    file: FileId,
    hints: FixHints,
) -> (Cow<'s, str>, usize) {
    let set = FixSet::collect(diags, hints);
    set.apply_for_file_counted(source, file)
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
    fn identical_span_prefers_higher_applicability_weight() {
        let span = Span::new(FileId::DUMMY, 0, 3);
        let a = diag_with_fix(span, "AAA", Applicability::MaybeIncorrect);
        let b = diag_with_fix(span, "BBB", Applicability::MachineApplicable);
        let set = FixSet::collect(&[a, b], FixHints::All);
        assert_eq!(set.len(), 1);
        assert_eq!(
            set.iter().next().unwrap().replacement,
            "BBB",
            "higher-weight edit must win"
        );
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
    fn overlapping_edits_resolve_by_weight_then_end_order() {
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
    fn equal_weight_overlap_prefers_earlier_end() {
        let src = "abcdef";
        let wide = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 5),
            "X",
            Applicability::MachineApplicable,
        );
        let narrow = diag_with_fix(
            Span::new(FileId::DUMMY, 2, 3),
            "Y",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[wide, narrow], FixHints::Auto);
        assert_eq!(set.len(), 1);
        // Ends at 3 wins over ends at 5, despite starting later.
        assert_eq!(set.apply(src), "abYdef");
    }

    #[test]
    fn weighted_resolution_prefers_two_machine_edits_over_one_wide_maybe() {
        let src = "abcdef";
        let a = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 2),
            "XX",
            Applicability::MachineApplicable,
        );
        let b = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 5),
            "YY",
            Applicability::MachineApplicable,
        );
        let wide = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 5),
            "ZZ",
            Applicability::MaybeIncorrect,
        );
        let set = FixSet::collect(&[wide, a, b], FixHints::Review);
        assert_eq!(set.len(), 2);
        assert_eq!(set.apply(src), "XXcYYf");
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

    /// Zero-length edits are insertions; duplicates at one point collapse to the
    /// highest-weight edit, while an insertion at a replacement boundary is a
    /// disjoint half-open interval and composes with the replacement.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "multi-file")]
    fn apply_rejects_multi_file_set() {
        let d1 = diag_with_fix(
            Span::new(FileId::from_raw(1), 0, 1),
            "a",
            Applicability::MachineApplicable,
        );
        let d2 = diag_with_fix(
            Span::new(FileId::from_raw(2), 0, 1),
            "b",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[d1, d2], FixHints::Auto);
        let _ = set.apply("x");
    }

    #[test]
    fn insertion_inside_replacement_wins_as_shorter_interval() {
        let src = "abcdef";
        let repl = diag_with_fix(
            Span::new(FileId::DUMMY, 2, 5),
            "XXX",
            Applicability::MachineApplicable,
        );
        let ins = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 3),
            "!",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[repl, ins], FixHints::Auto);
        assert_eq!(set.len(), 1, "interior insertion is a conflict");
        assert_eq!(set.apply(src), "abc!def");
    }

    #[test]
    fn zero_length_edits_dedup_at_same_point_and_compose_at_boundaries() {
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

        // A zero-length insertion at a replacement boundary is a disjoint
        // half-open interval; both edits survive and compose in offset order.
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
        assert_eq!(set.len(), 2);
        assert_eq!(set.apply(src), "!XXXdef");

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
        assert_eq!(set.len(), 2);
        assert_eq!(set.apply(src), "XXX!def");

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
    fn disjoint_edits_apply_regardless_of_collection_order() {
        let src = "aaa bbb ccc";
        let make = || {
            (
                diag_with_fix(
                    Span::new(FileId::DUMMY, 0, 3),
                    "XXX",
                    Applicability::MachineApplicable,
                ),
                diag_with_fix(
                    Span::new(FileId::DUMMY, 8, 11),
                    "ZZZ",
                    Applicability::MachineApplicable,
                ),
            )
        };
        for reverse in [false, true] {
            let (d1, d2) = make();
            let set = if reverse {
                FixSet::collect(&[d2, d1], FixHints::Auto)
            } else {
                FixSet::collect(&[d1, d2], FixHints::Auto)
            };
            assert_eq!(set.len(), 2);
            assert_eq!(set.apply(src), "XXX bbb ZZZ");
        }
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
        assert_eq!(set.len(), 1, "collect has no source and cannot filter");
        let (out, applied) = set.apply_counted(src);
        assert_eq!(applied, 0);
        assert_eq!(out, src);
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
        let visible =
            Diagnostic::error("boom").with_suggestions(vec![mk(SuggestionStyle::ShowAlways)]);
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
