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
//! `start >= last` guard in `apply_slice_count` (the body shared by every
//! public `apply` entry point) therefore holds by construction for same-file,
//! in-order edits; it is a cross-file safety guard for a caller that mixes files
//! into one buffer.
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
            // Enumerated rather than `true`: a future `Applicability` variant
            // must be a *conscious* `--fix-hints=all` decision here, matching the
            // explicit `SuggestionStyle` arms in `collect`.
            FixHints::All => matches!(
                app,
                Applicability::MachineApplicable
                    | Applicability::MaybeIncorrect
                    | Applicability::HasPlaceholders
                    | Applicability::Unspecified
            ),
        }
    }
}

/// Static objective for overlap resolution.
///
/// 4 : 2 : 1 encodes "one MachineApplicable edit is worth two
/// MaybeIncorrect edits and four placeholder/unspecified edits" — safety
/// dominates count, but not absolutely: the weights are **per edit**, so two
/// MaybeIncorrect edits (4) *tie* one MachineApplicable edit (4), and the
/// geometry decides (the two `tie_at_double_width_*` tests pin both directions).
/// Width plays no part in the weight; geometry enters only through interval
/// compatibility — a wide span matters because it overlaps more candidates, not
/// because the DP charges by width.
///
/// **At exactly 2× the totals tie, and geometry decides.**  The DP refuses to
/// replace an equal total, so it keeps whichever set it reached first in the
/// `(end, start)` consideration order — two configurations with identical
/// weights and identical totals can resolve to opposite sets (pinned by
/// `tie_at_double_width_keeps_the_wider_edit` and
/// `tie_at_double_width_keeps_the_narrower_pair`).  This is a property of the
/// documented rule, not of this table.
///
/// The constant is policy, not a theorem; change it here if `--fix` grows an
/// objective flag.
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

/// Pinned like `ErrCode`'s `assert_send_sync`: a future field that breaks the
/// auto traits (an `Rc` payload, a `RefCell`) must fail *here* rather than at
/// some distant caller that wanted to hand a `FixSet` to a worker.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FixSet<'static>>();
};

#[derive(Default)]
struct ResolveScratch {
    // `u32`, not `usize`/`u64`: weights are ≤ 4 per edit (`w[i] ≤ 4i`) and
    // `p[i] < n`, so both saturate far beyond any batch a compiler can emit.  The
    // DP's working set is these three arrays, so 9 bytes/element instead of 17 is
    // a ~1.9× cache-density win on exactly the loop that touches them randomly.
    p: Vec<u32>,
    w: Vec<u32>,
    take: Vec<bool>,
    /// Backtrack result (indices, ascending) — reused across groups.
    idx: Vec<u32>,
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
    /// same-span equal-weight ties retain the edit that ends first, then the one
    /// that starts first (the deterministic `(end, start)` order).  Equal
    /// *total-weight* ties between different geometries are resolved by that
    /// consideration order rather than by a rule on widths — see
    /// [`applicability_weight`] for the two pinned directions.
    pub fn collect(diags: &[Diagnostic<'src>], hints: FixHints) -> Self {
        // Exact reservation: one pre-pass sums the suggestion counts, so the
        // vector does not pay its 4→8→16 growth chain on a suggestion-heavy
        // batch.  (`diags.len()` was neither an upper nor even a lower bound —
        // a diagnostic carries *all* of its suggestions — so calling it an
        // upper bound was simply wrong.)
        let mut candidates: Vec<FixEdit<'src>> =
            Vec::with_capacity(diags.iter().map(|d| d.suggestions().len()).sum());
        for diag in diags {
            for sug in diag.suggestions() {
                match sug.style {
                    // Invisible to the user: applying one would rewrite the
                    // source in a way the output never mentioned.
                    SuggestionStyle::HideCodeAlways | SuggestionStyle::CompletelyHidden => continue,
                    // Every other style is user-visible (the message and/or the
                    // code), so it participates.  Listing them explicitly keeps
                    // a newly added style a compile error *here* instead of a
                    // silent `--fix` policy change.
                    SuggestionStyle::ShowAlways
                    | SuggestionStyle::ShowCode
                    | SuggestionStyle::HideCodeInline => {}
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
    /// Intervals are half-open `[start, end)` **except zero-length edits, which
    /// are closed points**: a point conflicts with any interval *containing* its
    /// offset (so an interior insertion competes with the enclosing replacement
    /// and, being the shorter interval, wins the tie — pinned by
    /// `insertion_inside_replacement_wins_as_shorter_interval`), while a point at
    /// an interval's boundary composes with it.  Duplicate insertions at one
    /// point collapse to the highest weight.
    ///
    /// The distinction is load-bearing: under strict half-open semantics `[3,3)`
    /// would be the empty set, disjoint from everything, which is *not* what the
    /// resolution (or the pinned test) implements.
    fn resolve_overlaps(mut candidates: Vec<FixEdit<'src>>) -> Vec<FixEdit<'src>> {
        // Candidates arrive sorted by `(file, start, end)`, so duplicate
        // zero-length runs are adjacent.
        //
        // Compaction happens **in place** (write cursor + `swap`): the earlier
        // shape built a second full-size `Vec` for `deduped`, which in the common
        // case (no duplicate zero-length runs) was a pure copy of `candidates`.
        // Order is preserved — `write` only ever trails `read`, and every element
        // it swaps in is the next surviving candidate in input order.
        let key = |e: &FixEdit<'src>| (e.span.file, e.span.start_as_usize(), e.span.end_as_usize());
        let mut write = 0usize;
        let mut read = 0usize;
        while read < candidates.len() {
            let (file, start, end) = key(&candidates[read]);
            if start != end {
                candidates.swap(write, read);
                write += 1;
                read += 1;
                continue;
            }
            // Zero-length run: keep the heaviest applicability.  Strict `>`
            // preserves diagnostic input order on equal weight; the stable
            // sort makes that deterministic for a fixed batch.  The run is
            // skipped wholesale (`read = r`), so swapping the loser into `best`
            // cannot resurrect an element that was already examined.
            let mut best = read;
            let mut best_weight = applicability_weight(candidates[read].applicability);
            let mut r = read + 1;
            while r < candidates.len() {
                let (f, s, e) = key(&candidates[r]);
                if f != file || s != start || e != start {
                    break;
                }
                let weight = applicability_weight(candidates[r].applicability);
                if weight > best_weight {
                    best = r;
                    best_weight = weight;
                }
                r += 1;
            }
            candidates.swap(write, best);
            write += 1;
            read = r;
        }
        candidates.truncate(write);
        let deduped = candidates;

        let mut result: Vec<FixEdit<'src>> = Vec::with_capacity(deduped.len());
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
                group = Self::resolve_file(std::mem::take(&mut group), &mut result, &mut scratch);
                Self::debug_assert_disjoint(&result[start..]);
            }
            last_file = Some(edit.span.file);
            group.push(edit);
        }
        if !group.is_empty() {
            let start = result.len();
            // The shell (and its capacity) is dropped with the function here.
            Self::resolve_file(group, &mut result, &mut scratch);
            Self::debug_assert_disjoint(&result[start..]);
        }
        result
    }

    /// Pin the type's headline invariant — "non-overlapping edits for one source
    /// buffer" — rather than only the ordering.  `is_sorted_by_key` accepted a
    /// dropped dedup or a broken tie rule as long as the order still held, which
    /// is exactly what the `start >= last_end` machinery can regress into
    /// silently.
    fn debug_assert_disjoint(edits: &[FixEdit<'_>]) {
        for pair in edits.windows(2) {
            debug_assert!(
                pair[0].span.end_as_usize() <= pair[1].span.start_as_usize(),
                "resolve_file must emit pairwise-disjoint edits in source order, got {:?} then {:?}",
                pair[0].span,
                pair[1].span
            );
        }
    }

    /// Resolve one file's candidate group into `out`, **returning the group's
    /// allocation** for reuse.
    ///
    /// `mem::take` at the call site leaves a capacity-0 shell behind, so a batch
    /// spanning k files re-ran the 4→8→16 growth chain k times.  Draining instead
    /// of consuming (`drain(..)`) lets the shell — now with its original
    /// capacity — be assigned back, so the buffer is amortized across files the
    /// same way `ResolveScratch` already amortizes the DP arrays.
    fn resolve_file(
        mut group: Vec<FixEdit<'src>>,
        out: &mut Vec<FixEdit<'src>>,
        scratch: &mut ResolveScratch,
    ) -> Vec<FixEdit<'src>> {
        group.sort_by_key(|e| (e.span.end_as_usize(), e.span.start_as_usize()));
        let n = group.len();
        if n == 0 {
            return group;
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
            for edit in group.drain(..) {
                if edit.span.start_as_usize() >= last_end {
                    last_end = edit.span.end_as_usize();
                    out.push(edit);
                }
            }
            return group;
        }
        // `p[i]`/`w[i]`/`take[i]` are `u32`/`u32`/`bool`; the bound is structural
        // (`n` is a diagnostic batch size), and the DP's three arrays are its
        // dominant working set, so the narrower width is a cache win.
        debug_assert!(n < u32::MAX as usize, "fix batch larger than u32::MAX");
        // p[i] = number of earlier edits compatible with i.
        //
        // The backtrack collects *indices* rather than moving edits out of an
        // `Option` shadow: the shadow was a second full-size vector (roughly
        // doubling the DP's per-element footprint) plus a per-element
        // `take().expect()`, which is exactly the shape the dedup pass above
        // brags about avoiding.
        scratch.p.clear();
        scratch.p.resize(n, 0);
        let p = &mut scratch.p;
        for (i, edit) in group.iter().enumerate() {
            p[i] = group[..i].partition_point(|earlier| {
                earlier.span.end_as_usize() <= edit.span.start_as_usize()
            }) as u32;
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
            // The DP arrays are `u32` (see `ResolveScratch`); a weight is ≤ 4 by
            // construction, so narrowing here is exact.
            let weight = applicability_weight(group[i - 1].applicability) as u32;
            let include = weight + w[p[i - 1] as usize];
            let exclude = w[i - 1];
            if include > exclude {
                w[i] = include;
                take[i] = true;
            } else {
                w[i] = exclude;
            }
        }
        // Backtrack to indices (ascending), then merge once over the owned group.
        // The earlier shape built a `Vec<Option<..>>` shadow, collected a second
        // `Vec` of chosen edits and reversed it — two allocations and a doubled
        // element footprint for the same answer.
        let mut i = n;
        while i > 0 {
            if take[i] {
                scratch.idx.push((i - 1) as u32);
                i = p[i - 1] as usize;
            } else {
                i -= 1;
            }
        }
        scratch.idx.reverse();
        let start = out.len();
        let mut next = 0usize;
        for (idx, edit) in group.drain(..).enumerate() {
            if next < scratch.idx.len() && scratch.idx[next] as usize == idx {
                out.push(edit);
                next += 1;
            }
        }
        debug_assert_eq!(next, scratch.idx.len(), "backtrack index not consumed");
        // Free internal oracle: the DP already knows the optimum in `w[n]`, so the
        // emitted set's weight is an O(1) check per invocation.  The brute-force
        // test only reaches n ≤ 6; a backtrack regression would silently lose
        // weight on the large inputs it cannot reach.
        #[cfg(debug_assertions)]
        {
            let selected_weight: u32 = out[start..]
                .iter()
                .map(|e| applicability_weight(e.applicability) as u32)
                .sum();
            debug_assert_eq!(
                selected_weight, w[n],
                "backtrack dropped weight: w[n]={} but the selected set sums to {}",
                w[n], selected_weight
            );
        }
        group
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
    /// **Multi-file sets.** A batch can carry edits for several files.  In debug
    /// builds `apply` panics on such a set (`apply_rejects_multi_file_set`); in
    /// release builds it routes through [`Self::apply_single_file`], which keeps
    /// only the edits of the *first* edit's file, so a foreign file's byte
    /// offsets can never reach this buffer.  Neither behaviour is a silent
    /// mis-splice — the old doc warned about one that this code cannot produce.
    ///
    /// Prefer [`Self::apply_for_file_counted`] / [`Self::apply_for_file`] when the
    /// batch may span files: they make the file choice explicit instead of
    /// leaving it to the sorted order.  The `start < last` guard inside
    /// `apply_slice_count` remains a genuine *same-file* overlap backstop for a
    /// corrupted set; the cross-file defence is the anchor filter above it.
    #[must_use = "apply returns the rewritten source; write it out or the fix is lost"]
    pub fn apply<'s>(&self, source: &'s str) -> Cow<'s, str> {
        self.apply_single_file(source).0
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
    ///
    /// The name carries the precondition: this applies the edits of **one**
    /// file to **one** buffer.  `apply_counted` read as a drop-in variant of
    /// [`Self::apply`], so a multi-file caller silently got the first file's
    /// edits spliced into an unrelated buffer in release builds (the guard below
    /// is a `debug_assert!`).  Callers holding a batch must use
    /// [`Self::apply_for_file_counted`].
    #[must_use = "apply_single_file returns the rewritten source and count"]
    pub fn apply_single_file<'s>(&self, source: &'s str) -> (Cow<'s, str>, usize) {
        let anchor = self.edits.first().map(|e| e.span.file);
        debug_assert!(
            anchor.is_none_or(|f| self.edits.iter().all(|e| e.span.file == f)),
            "FixSet::apply_single_file called on a multi-file set; use apply_for_file_counted",
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
        // Two passes: an O(n) excess pre-pass sizes the single allocation, and
        // the write pass is the authoritative "did anything apply" test (so the
        // borrowed source is returned when nothing applies).  The old shape
        // walked `edits` three times (any-check, size sum, write).
        // Pre-sum how much the replacements outgrow the text they replace, so the
        // single allocation is the right size: `source.len()` alone is
        // undersized for rewrite-heavy batches and reallocs mid-pass.
        let excess: usize = edits
            .iter()
            .map(|e| {
                e.replacement.len().saturating_sub(
                    e.span
                        .end_as_usize()
                        .saturating_sub(e.span.start_as_usize()),
                )
            })
            .sum();
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
            let buf = out.get_or_insert_with(|| {
                // The buffer holds `source.len() - Σ|replaced| + Σ|replacement|`;
                // a rewrite-heavy batch outgrows `source.len()` and reallocs
                // mid-pass, copying everything already written.
                String::with_capacity(source.len().saturating_add(excess))
            });
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
    set.apply_single_file(source)
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

    /// Equal *total* weight (4 vs 2+2) resolved by geometry: the narrow pair
    /// covers `[0,5)` completely, so both sets fit — and the wider edit wins
    /// because the DP meets it before the pair completes.
    #[test]
    fn tie_at_double_width_keeps_the_wider_edit() {
        let wide = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 5),
            "WIDE",
            Applicability::MachineApplicable, // 4
        );
        let left = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "L",
            Applicability::MaybeIncorrect,
        );
        let right = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 5),
            "R",
            Applicability::MaybeIncorrect,
        );
        let set = FixSet::collect(&[wide, left, right], FixHints::All);
        let got: Vec<&str> = set.iter().map(|e| e.replacement.as_ref()).collect();
        assert_eq!(
            got,
            ["WIDE"],
            "the end-earliest consideration order keeps it"
        );
    }

    /// The same weights and the same 4-vs-4 total, a different geometry: the pair
    /// completes *before* the wide edit is considered, so the pair survives.
    /// Together with the test above this pins that exact-2× ties are decided by
    /// `(end, start)` order, not by a width rule.
    #[test]
    fn tie_at_double_width_keeps_the_narrower_pair() {
        let wide = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 5),
            "WIDE",
            Applicability::MachineApplicable, // 4
        );
        let left = diag_with_fix(
            Span::new(FileId::DUMMY, 0, 3),
            "L",
            Applicability::MaybeIncorrect,
        );
        let right = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 4),
            "R",
            Applicability::MaybeIncorrect,
        );
        let set = FixSet::collect(&[wide, left, right], FixHints::All);
        let got: Vec<&str> = set.iter().map(|e| e.replacement.as_ref()).collect();
        assert_eq!(got, ["L", "R"], "the pair reaches the total first");
    }

    /// The module's whole value proposition is a **maximum**-weight independent
    /// set, and the all-equal-weight greedy fast path is the fork most likely to
    /// lose it silently (a tie-rule tweak, a `p[i]` regression under the
    /// closed-point model).  Nothing pinned it before: the existing tests check
    /// tie-breaking and per-geometry parity, not optimality.
    ///
    /// Brute force over every subset, with the conflict relation the resolution
    /// actually implements: half-open intervals, zero-length edits as closed
    /// points that conflict only with an interval's *interior* (a boundary one
    /// composes), and two insertions at one point mutually exclusive.
    #[test]
    fn collect_is_a_maximum_weight_independent_set() {
        let mut seed: u64 = 0x5eed_f1ab_cdef_0011;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        for _ in 0..300 {
            let n = 2 + (next() % 5) as usize; // 2..=6
            let mut spans: Vec<(u32, u32)> = Vec::with_capacity(n);
            let mut diags: Vec<Diagnostic<'static>> = Vec::with_capacity(n);
            for _ in 0..n {
                let start = (next() % 10) as u32;
                let len = (next() % 4) as u32; // 0 => a zero-length insertion
                let app = match next() % 3 {
                    0 => Applicability::MachineApplicable,
                    1 => Applicability::MaybeIncorrect,
                    _ => Applicability::HasPlaceholders,
                };
                spans.push((start, start + len));
                diags.push(diag_with_fix(
                    Span::new(FileId::DUMMY, start, start + len),
                    "R",
                    app,
                ));
            }
            let weights: Vec<u64> = diags
                .iter()
                .map(|d| {
                    let app = d.suggestions()[0].applicability;
                    applicability_weight(app)
                })
                .collect();

            let conflicts = |a: (u32, u32), b: (u32, u32)| -> bool {
                let (sa, ea) = a;
                let (sb, eb) = b;
                match (sa == ea, sb == eb) {
                    // Two insertions: only the same point collides (they dedup).
                    (true, true) => sa == sb,
                    // A point only conflicts with an interval's interior.
                    (true, false) => sb < sa && sa < eb,
                    (false, true) => sa < sb && sb < ea,
                    (false, false) => sa < eb && sb < ea,
                }
            };

            let set = FixSet::collect(&diags, FixHints::All);
            let got: u64 = set
                .iter()
                .map(|e| applicability_weight(e.applicability))
                .sum();
            let mut want = 0u64;
            for mask in 0u32..(1 << n) {
                let chosen: Vec<usize> = (0..n).filter(|i| mask >> i & 1 == 1).collect();
                let disjoint = chosen.iter().enumerate().all(|(k, &i)| {
                    chosen[k + 1..]
                        .iter()
                        .all(|&j| !conflicts(spans[i], spans[j]))
                });
                if disjoint {
                    want = want.max(chosen.iter().map(|&i| weights[i]).sum());
                }
            }
            assert_eq!(got, want, "not maximum-weight for spans {spans:?}");
        }
    }

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

    /// An insertion at `len == source.len()` is inside the source by the
    /// `end > source.len()` check (equal, not greater) and its span is a char
    /// boundary, so it must apply.  Previously only reachable through the random
    /// brute-force test, which does not name the case.
    #[test]
    fn insertion_at_eof_applies() {
        let src = "abc";
        let eof = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 3),
            "!",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[eof], FixHints::Auto);
        assert_eq!(set.len(), 1, "an EOF insertion is a valid edit");
        assert_eq!(set.apply(src), "abc!");
    }

    /// `--fix-hints=all` is the only policy that accepts `Unspecified`, and the
    /// explicit variant list in `FixHints::accepts` is what keeps a future
    /// applicability tier a conscious decision.  Pin both directions.
    #[test]
    fn all_hints_collect_unspecified_but_auto_does_not() {
        let span = Span::new(FileId::DUMMY, 0, 1);
        let unspecified = diag_with_fix(span, "X", Applicability::Unspecified);

        let all = FixSet::collect(std::slice::from_ref(&unspecified), FixHints::All);
        assert_eq!(all.len(), 1, "All must accept Unspecified");

        for hints in [FixHints::Auto, FixHints::Review] {
            let set = FixSet::collect(std::slice::from_ref(&unspecified), hints);
            assert!(set.is_empty(), "{hints:?} must reject Unspecified");
        }
    }

    /// Unequal weights force the age-weighted interval-scheduling DP (equal
    /// weights collapse to the greedy path), so this pins the DP fork's
    /// closed-point semantics in both directions — a point edit composes with a
    /// boundary-touching interval, and conflicts when strictly inside one.
    #[test]
    fn dp_path_resolves_point_vs_containing_interval_by_weight() {
        let src = "abcdef";

        // Interior point (weight 4) against a MaybeIncorrect interval (weight 2):
        // they conflict, and the heavier point wins.
        let heavy_point = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 3),
            "!",
            Applicability::MachineApplicable,
        );
        let light_interval = diag_with_fix(
            Span::new(FileId::DUMMY, 1, 5),
            "YYY",
            Applicability::MaybeIncorrect,
        );
        let set = FixSet::collect(&[light_interval, heavy_point], FixHints::Review);
        assert_eq!(set.len(), 1, "interior point and interval conflict");
        assert_eq!(set.apply(src), "abc!def");

        // The same geometry inverted: a light point cannot displace a heavy
        // interval.
        let light_point = diag_with_fix(
            Span::new(FileId::DUMMY, 3, 3),
            "!",
            Applicability::Unspecified,
        );
        let heavy_interval = diag_with_fix(
            Span::new(FileId::DUMMY, 1, 5),
            "YYY",
            Applicability::MachineApplicable,
        );
        let set = FixSet::collect(&[light_point, heavy_interval], FixHints::All);
        assert_eq!(set.len(), 1, "the interval is retained");
        assert_eq!(set.apply(src), "aYYYf");
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
        let (out, applied) = set.apply_single_file(src);
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
