//! Knuth-Plass optimal line breaking with Unicode UAX #14 break opportunities.
//!
//! Replaces the greedy `textwrap::wrap` used by `GlyphRenderer::wrap_msg` and
//! `explain_error_code` with a total-cost minimiser:
//!
//! - **Break opportunities** come from `unicode_linebreak` (UAX #14), then are
//!   filtered so code tokens (`Int<32>`, `Vec<T>`, `(x, y)`, `foo-bar`) are
//!   never split across lines. CJK↔CJK and whitespace boundaries survive.
//!   Printable-ASCII paragraphs skip the UAX #14 state machine entirely
//!   ([`ascii_break_opportunities`]); `ascii_fast_path_breaks_match_uax` pins
//!   that shortcut to `unicode_linebreak` over every printable-ASCII flanking
//!   pair.
//! - **Break selection** is a DP over words minimising total badness:
//!   the cubed slack of every line (`((measure - line_width) / measure)^3`),
//!   plus a TeX-style demerit for the fitness jump between *adjacent* lines —
//!   a nearly-full line next to a nearly-empty one is charged even though
//!   each line looks cheap on its own.  The last line's own slack is free, so
//!   the paragraph is not stretched just to fill it, but the jump into it is.
//!
//! Hard newlines (`\n`) in the input are preserved as paragraph boundaries.
//! Every output segment borrows the input (`Cow::Borrowed`) — no allocation
//! unless an over-wide word has to be split at a grapheme boundary.
//!
//! # Fast paths, hottest first
//!
//! 1. Text that already fits on one line (and is printable ASCII) is borrowed
//!    verbatim: no segmentation, no DP.
//! 2. A paragraph that fits and has no edge whitespace is borrowed verbatim —
//!    the DP would emit exactly that one line, and any split's first line is
//!    strictly short of the measure so it costs more than the one-liner's 0.
//! 3. Printable-ASCII paragraphs compute break opportunities with an ASCII
//!    subset of UAX #14 instead of running the `unicode_linebreak` state
//!    machine (and never touch `unicode_segmentation`).
//! 4. Widths of printable-ASCII runs are byte lengths, decided two SWAR masks
//!    at a time; ASCII trailing whitespace never touches the Unicode tables.
//! 5. The DP carries a monotone window start (no binary search), one packed
//!    `u32` predecessor state instead of two arrays, and a const adjacency
//!    table instead of recomputing the fitness jump per candidate.
//!
//! Both width fast paths are gated on printable ASCII: tab expansion is
//! column-dependent and therefore *not* additive across the word/glue pieces
//! the DP sums, so a tab forces the general path.
//!
//! # Known approximations
//!
//! * **Tab width is measured from column 0 within each measured string.**
//!   [`measure_width`] threads the running column into the per-character
//!   width, so a tab is charged `tab_width - (col % tab_width)` at its actual
//!   column *inside the measured string* — but each call starts that column
//!   at 0.  The DP sums per-word widths, so it cannot know the line column a
//!   word lands at: tab-bearing text is over/under-measured by up to
//!   `tab_width - 1` cells at word boundaries.  Correct handling would need
//!   per-break-column arithmetic through the DP; diagnostic messages almost
//!   never contain tabs, so forcing the general path is the deliberate
//!   trade-off.
//! * **A single grapheme wider than the measure wins.**  [`Word`] docs claim
//!   `width ≤ max_width` "by construction" — that holds only when a
//!   grapheme cluster is at most one cell.  [`push_split_oversized`] packs
//!   graphemes up to `max_width` and never splits inside a cluster, so a
//!   cluster wider than the measure itself (e.g. a 2-cell emoji at
//!   `max_width = 1`) produces a piece whose width *exceeds* `max_width`.
//!   No inner assertion may claim otherwise; the DP simply emits the
//!   over-wide line, which is the best any cluster-respecting breaker can
//!   do.
//!
//! [`GlyphRenderer`]: crate::diagnostics::GlyphRenderer
//! [`explain_error_code`]: crate::diagnostics::explain_error_code

use super::ascii::is_printable_ascii;
use super::glyph::{WIDTH_CACHE, WidthCache, grapheme_width_cached};
use crate::smallvec::{SmallVec, smallvec};
use std::borrow::Cow;
use std::ops::Range;
use unicode_linebreak::{BreakOpportunity, linebreaks};
use unicode_segmentation::UnicodeSegmentation;

/// Demerit for an over-wide line.
///
/// Every soft term is normalised so that one completely empty line costs
/// `1.0`; the overflow term is deliberately left in *absolute* columns,
/// because running past the measure is a layout error, not a taste question.
/// `1000 · over³` puts even a one-column overflow two orders of magnitude
/// above any legal line (≤ 1.0) plus its fitness jumps (≤ 0.125), so an
/// over-full line is only chosen when no legal break exists.
const OVERFLOW_PENALTY: f64 = 1000.0;

/// Demerit per fitness step *beyond the first* between adjacent lines.
///
/// Same units as [`intrinsic_badness`] (cubes of the empty/measure ratio, so
/// one completely empty line costs `1.0`).  `1/16` is calibrated so that a
/// two-class jump — a nearly-full line followed by a nearly-empty one — costs
/// `1/8`, exactly what a line leaving half the measure empty costs.  The
/// first step is free (TeX's `adj_demerits` rule): a progressive rag is fine,
/// only a jarring one is charged.
const FITNESS_STEP_DEMERIT: f64 = 1.0 / 16.0;

/// Number of TeX-style fitness classes used to compare adjacent lines.
const N_FITNESS: usize = 4;

/// [`fitness_jump`] tabulated as `JUMP[cur][prev]`, so the DP's inner loop
/// reads a contiguous row of `prev` values for a fixed `cur` class — one
/// indexed load instead of an `abs_diff` plus a branch.  Rows and columns
/// are TeX fitness classes (0 Tight … 3 VeryLoose); a one-step difference is
/// free, every further step costs [`FITNESS_STEP_DEMERIT`].
/// `jump_table_matches_fitness_jump` pins the table to [`fitness_jump`].
const JUMP: [[f64; N_FITNESS]; N_FITNESS] = [
    // prev →                0                                1                             2
    /* cur=0 */
    [0.0, 0.0, FITNESS_STEP_DEMERIT, 2.0 * FITNESS_STEP_DEMERIT],
    /* cur=1 */ [0.0, 0.0, 0.0, FITNESS_STEP_DEMERIT],
    /* cur=2 */ [FITNESS_STEP_DEMERIT, 0.0, 0.0, 0.0],
    /* cur=3 */ [2.0 * FITNESS_STEP_DEMERIT, FITNESS_STEP_DEMERIT, 0.0, 0.0],
];

/// Optimal-wrap `text` to at most `max_width` display columns per line.
///
/// Hard newlines in `text` are kept: each `\n`-separated paragraph is wrapped
/// independently. A trailing `\n` does not produce an extra empty line, so
/// `wrap_optimal("a\n", w)` == `["a"]`, matching `textwrap::wrap`.
///
/// Returns a [`SmallVec`] because most messages wrap to 1–2 lines, so the
/// common case stays inline instead of allocating a `Vec` per label/help
/// header.  Callers only iterate the result, so the type is transparent.
pub(crate) fn wrap_optimal<'a>(
    text: &'a str,
    max_width: usize,
    tab_width: usize,
) -> SmallVec<[Cow<'a, str>; 4]> {
    // Defend the division contract at the boundary: every other renderer
    // clamps (`GlyphRenderer::with_opts`, `HtmlRenderer::with_opts`), but
    // `explain_error_code_with_depth` is `pub` and takes the width directly,
    // and `grapheme_width_cached` computes `tab_width - (col % tab_width)`.
    let tab_width = tab_width.max(1);
    // Every paragraph is a subslice of `text`, so bounding the whole text
    // bounds every `Word` offset (`push_words` indexes into one paragraph).
    // A 4 GiB diagnostic message is not a real input; a debug build catches
    // it long before the `u32` casts below would silently wrap.
    // `Word` offsets are `u32`.  A >4 GiB input is not a real diagnostic, but
    // silently truncating its ranges in release would be corruption, not
    // degradation: emit its hard-newline segments verbatim instead.  Keep the
    // debug assertion as the loud signal for fuzzed/测试 inputs.
    if text.len() > u32::MAX as usize {
        return text.split_terminator('\n').map(Cow::Borrowed).collect();
    }
    debug_assert!(
        text.len() <= u32::MAX as usize,
        "wrap_optimal: text exceeds u32 byte offsets ({} bytes)",
        text.len()
    );
    if text.is_empty() {
        return smallvec![Cow::Borrowed("")];
    }
    if max_width == 0 {
        // The contract "at most `max_width` display columns per line" is
        // unsatisfiable at 0 (any non-empty line violates it), so 0 means
        // "no wrapping within paragraphs": each hard newline survives as a
        // segment, paragraphs are emitted verbatim.  `DiagOpts` clamps
        // `--width 0` up before this is reached; the guard stays as the
        // direct-caller defence.  `split_terminator`, like every other path
        // below: a trailing `\n` must not produce a phantom empty segment
        // ("a\n" → ["a"], "\n" → [""], "\n\n" → ["", ""]).
        return text.split_terminator('\n').map(Cow::Borrowed).collect();
    }

    // Single-line fast path: text with no hard newline (one optional trailing
    // `\n`) whose core fits `max_width` is emitted verbatim by the DP below —
    // one line, zero badness, and no split can tie it (a split's first line is
    // strictly short of the measure, so it costs > 0).
    //
    // Edge *ASCII spaces* are trimmed: LB18 (`SP ÷`) makes every space run a
    // break opportunity, so the DP drops a leading space-only chunk (no
    // predecessor to fold into) and the trailing space of the last word
    // (`line_width` omits it).  Non-ASCII whitespace (NBSP, U+3000…) at the
    // edges is rejected by [`edges_have_whitespace`]: leading NBSP is LB12
    // glue the DP keeps inside the first word while trailing NBSP is stripped
    // by [`trimmed_len`]'s multi-byte fallback, and a leading NBSP run folded
    // with ASCII spaces is dropped as a whitespace-only chunk — in every case
    // a fast path emitting `trim_matches(' ')` would diverge, so we fall
    // through to the DP.
    //
    // Tabs are excluded: their width is column-dependent, and — decisive —
    // a tab is a UAX #14 break *opportunity*, so the DP can split
    // tab-bearing text into separate lines even when the whole text
    // measures within the measure (pinned by `fast_paths_match_dp` with
    // `"a\tb"` at width 5 → ["a", "b"]).  A verbatim fast path would
    // diverge, so any tab falls through to the DP.  `measure_width` is
    // SWAR for printable ASCII and a cached grapheme walk otherwise — both
    // far cheaper than UAX #14 + DP — so this gate covers short non-English
    // (CJK, kana, …) messages, not just ASCII.
    let single = text.strip_suffix('\n').unwrap_or(text);
    // One fused scan decides both fast-path gates: `memchr2` reports the
    // first `\n` or `\t` in a single pass, so the "no interior newline"
    // probe and the tab probe no longer cost two scans over the text.
    // The pair is derived exactly, so the gates cannot disagree: no layout
    // byte at all → no tab, no interior newline; a tab first → `has_tab` and
    // the fast path is refused outright (a tab is a UAX #14 break
    // opportunity, so the DP may split); a newline first → an interior
    // newline exists.
    //
    // `has_tab` is therefore implied by `no_interior_newline` (the `match`
    // derives both from one probe: `no_interior_newline` is true only in the
    // `None` arm, where `has_tab` is false).  The invariant lives in this
    // `match`, so the gate below does not repeat the `!has_tab` term — a
    // future refactor that splits the pair must update this `match`, which is
    // where the guard belongs.
    let first_layout = memchr::memchr2(b'\n', b'\t', single.as_bytes());
    let (has_tab, no_interior_newline) = match first_layout {
        None => (false, true),
        Some(p) if single.as_bytes()[p] == b'\t' => (true, false),
        Some(_) => (false, false),
    };
    if no_interior_newline && !has_tab {
        let trimmed = single.trim_matches(' ');
        // A tab-first text reports `no_interior_newline == false`, so tab
        // expansion (and its UAX #14 break opportunity) can never reach this
        // verbatim fast path.
        let fits = if trimmed.len() <= max_width {
            true
        } else if is_printable_ascii(trimmed.as_bytes()) {
            false
        } else {
            measure_width(trimmed, tab_width) <= max_width
        };
        if !trimmed.is_empty() && !edges_have_whitespace(trimmed) && fits {
            return smallvec![Cow::Borrowed(trimmed)];
        }
    }

    // `split_terminator` drops the phantom empty final paragraph a trailing
    // '\n' would otherwise leave — exactly what the old count-then-take
    // produced ("a\n" → ["a"], "a\n\nb" → ["a", "", "b"], "\n" → [""],
    // "\n\n" → ["", ""]) — without the separate `memchr_iter(...).count()`
    // scan pass.  `no_interior_newline` above remains in scope: the fast
    // path still uses it.
    // The badness/class tables and the DP scratch are fetched once for the
    // whole message: one TLS access + one `Rc` clone and at most five
    // allocations per *message*, instead of one set per paragraph.
    let tables = wrap_tables(max_width);
    // State indices are `u32`; every word is at least one byte, so rejecting
    // a paragraph above `u32::MAX / N_FITNESS` bytes keeps `n * N_FITNESS`
    // representable.  The fallback is the same always-valid line preserving
    // layout the DP uses on a backtrack failure.
    if text.len() >= u32::MAX as usize / N_FITNESS {
        return text.split_terminator('\n').map(Cow::Borrowed).collect();
    }
    // The DP scratch lives in a per-thread `thread_local!` (mirroring
    // `WRAP_TABLES`): a diagnostic wraps a *dozen* labels/help/child
    // messages, and each used to pay `WrapScratch::new()` — up to five
    // heap-touching `SmallVec` growths — per wrapping message.  Reusing one
    // scratch across the whole batch converts that to one-time growth per
    // thread.  (A `RenderCtx`-threaded variant would need `wrap_msg`'s
    // callers to pass ctx down; `chain.rs` wraps without a ctx, so the
    // thread-local is the honest reuse point.)  The borrow is held only
    // within this call; `wrap_optimal` never nests into itself, and the
    // returned `Cow`s borrow `text`, not the scratch.
    // [4] recursion guard: if a future caller nests `wrap_optimal` (or calls
    // it from `wrap_dp`'s scoring), the TLS borrow would panic.  Fall back to
    // a local scratch in that impossible-by-design case; the normal path stays
    // allocation-free.
    // #7: batch memo.  Identical static help/label strings recur across a batch;
    // hash them once and reuse the DP's segment ranges.  Hits are exact: the
    // stored content is compared byte-for-byte before ranges are applied, so a
    // hash collision can only miss, never mis-render.  Bounded, and a hit still
    // borrows the *current* text (no allocation).
    //
    // Size gate: the memo is for short recurring labels, not long prose.  A
    // long message pays the full-text hash on every miss and, on a miss, an
    // `Arc<str>` copy of the whole input; skipping it keeps the memo bounded
    // by `MEMO_MAX_TEXT * 256` bytes (128 KiB) instead of unbounded-by-bytes.
    let memo_key =
        (text.len() <= MEMO_MAX_TEXT).then(|| (wrap_text_hash(text), max_width, tab_width));
    if let Some(key) = memo_key
        && let Some(hit) = wrap_memo_get(key, text)
    {
        return hit;
    }
    let result = if WRAP_IN_USE.get() {
        let mut local = WrapScratch::new();
        wrap_optimal_with(text, max_width, tab_width, tables.as_deref(), &mut local)
    } else {
        WRAP_IN_USE.set(true);
        let _reset = WrapInUseGuard;
        WRAP_SCRATCH.with_borrow_mut(|scratch| {
            wrap_optimal_with(text, max_width, tab_width, tables.as_deref(), scratch)
        })
    };
    if let Some(key) = memo_key {
        wrap_memo_put(key, text, &result);
    }
    result
}

/// Memo admission bound: short recurring labels only.  Keeps worst-case
/// retained text at `MEMO_MAX_TEXT * 256` bytes even on a diverse batch.
const MEMO_MAX_TEXT: usize = 512;

/// Hash used only as a memo bucket; exact content comparison disambiguates.
fn wrap_text_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Process-wide memo used while `render_parallel` is active.
static SHARED_WRAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// RAII scope that switches wrap caches to shared mode for a parallel batch.
pub(crate) struct SharedWrapGuard;

impl Drop for SharedWrapGuard {
    fn drop(&mut self) {
        SHARED_WRAP.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(crate) fn enter_shared_wrap() -> SharedWrapGuard {
    SHARED_WRAP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    SharedWrapGuard
}

fn wrap_shared() -> bool {
    SHARED_WRAP.load(std::sync::atomic::Ordering::Relaxed) > 0
}

type WrapMemoMap = rustc_hash::FxHashMap<
    (u64, usize, usize),
    Vec<(std::sync::Arc<str>, SmallVec<[Range<usize>; 4]>)>,
>;

thread_local! {
    static WRAP_MEMO: std::cell::RefCell<WrapMemoMap> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

static SHARED_MEMO: std::sync::OnceLock<std::sync::RwLock<WrapMemoMap>> =
    std::sync::OnceLock::new();

fn memo_lookup<'a>(
    memo: &WrapMemoMap,
    key: (u64, usize, usize),
    text: &'a str,
) -> Option<SmallVec<[Cow<'a, str>; 4]>> {
    let entries = memo.get(&key)?;
    let (_, ranges) = entries.iter().find(|(content, _)| &**content == text)?;
    Some(
        ranges
            .iter()
            .map(|r| Cow::Borrowed(&text[r.clone()]))
            .collect(),
    )
}

fn wrap_memo_get<'a>(
    key: (u64, usize, usize),
    text: &'a str,
) -> Option<SmallVec<[Cow<'a, str>; 4]>> {
    if wrap_shared() {
        let memo = SHARED_MEMO
            .get_or_init(|| std::sync::RwLock::new(rustc_hash::FxHashMap::default()))
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        memo_lookup(&memo, key, text)
    } else {
        WRAP_MEMO.with_borrow(|memo| memo_lookup(memo, key, text))
    }
}

fn memo_insert(
    memo: &mut WrapMemoMap,
    key: (u64, usize, usize),
    text: &str,
    ranges: SmallVec<[Range<usize>; 4]>,
) {
    if memo.len() >= 256
        && let Some(victim) = memo.keys().next().copied()
    {
        memo.remove(&victim);
    }
    let entries = memo.entry(key).or_default();
    if !entries.iter().any(|(content, _)| &**content == text) {
        entries.push((std::sync::Arc::from(text), ranges));
    }
}

fn wrap_memo_put(key: (u64, usize, usize), text: &str, result: &[Cow<'_, str>]) {
    let ranges: SmallVec<[Range<usize>; 4]> = result
        .iter()
        .map(|seg| {
            let s = seg.as_ref();
            // Whitespace-only paragraphs push the `""` literal, which lives in
            // `.rodata`, not inside `text`; its pointer must not be turned into
            // an offset into the input.  `&text[0..0]` is the same empty slice.
            let off = if s.is_empty() {
                0
            } else {
                (s.as_ptr() as usize).saturating_sub(text.as_ptr() as usize)
            };
            debug_assert!(
                off + s.len() <= text.len(),
                "wrap memo requires segments borrowed from the input text"
            );
            off..off + s.len()
        })
        .collect();
    if wrap_shared() {
        let mut memo = SHARED_MEMO
            .get_or_init(|| std::sync::RwLock::new(rustc_hash::FxHashMap::default()))
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        memo_insert(&mut memo, key, text, ranges);
    } else {
        WRAP_MEMO.with_borrow_mut(|memo| memo_insert(memo, key, text, ranges));
    }
}

/// Shared body of [`wrap_optimal`] over a caller-supplied scratch buffer.
/// Split out so the recursion fallback can pass a local scratch instead of the
/// thread-local one.
fn wrap_optimal_with<'a>(
    text: &'a str,
    max_width: usize,
    tab_width: usize,
    tables: Option<&WrapTables>,
    scratch: &mut WrapScratch,
) -> SmallVec<[Cow<'a, str>; 4]> {
    let mut out: SmallVec<[Cow<'a, str>; 4]> = SmallVec::new();
    for para in text.split_terminator('\n') {
        if para.is_empty() {
            out.push(Cow::Borrowed(""));
            continue;
        }
        // Whole paragraph's core fits `max_width` ⇒ the DP emits it as one
        // line (zero badness; any split's first line is strictly short of the
        // measure, so it costs > 0).  See the top-of-function fast path for
        // the trimming / tab / non-ASCII-whitespace rationale.
        let trimmed = para.trim_matches(' ');
        // One printable-ASCII scan serves both the `fits` gate and
        // `collect_words`'s ASCII dispatch: printable ASCII excludes the tab
        // (0x09 < 0x20), so the old separate tab `memchr` is subsumed and the
        // most frequent wrap path pays one SWAR pass instead of up to two.
        // Derived from `trimmed` but applied to `para`: valid because
        // `trim_matches(' ')` only strips bytes in the printable set (space is
        // 0x20), so the two buffers have the same printable-ASCII verdict.  If
        // the trim is ever widened (e.g. to `char::is_whitespace`), re-derive
        // from `para` here or a leading tab would pass `Some(true)` for a
        // non-printable buffer.
        let ascii = is_printable_ascii(trimmed.as_bytes());
        // #11: the derivation from `trimmed` is only sound while the trim is
        // exactly `' '`; if it is ever widened to Unicode whitespace, this
        // fires in debug instead of silently letting a leading tab through.
        debug_assert_eq!(
            is_printable_ascii(para.as_bytes()),
            ascii,
            "trim_matches(' ') must preserve the printable-ASCII verdict"
        );
        // `fits_and_clean`: a printable-ASCII paragraph fits exactly when its
        // byte length fits (bytes == cells, no tab possible).  Otherwise the
        // tab probe still matters (a tab is one byte but up to `tab_width`
        // cells), and non-ASCII cells can exceed the byte count, so measure.
        let fits_and_clean = if ascii {
            trimmed.len() <= max_width
        } else {
            measure_width_unless_tab(trimmed, tab_width).is_some_and(|w| w <= max_width)
        };
        if !trimmed.is_empty() && !edges_have_whitespace(trimmed) && fits_and_clean {
            out.push(Cow::Borrowed(trimmed));
            continue;
        }
        let mut words = std::mem::take(&mut scratch.words);
        let mut breaks = std::mem::take(&mut scratch.breaks);
        collect_words_into(
            para,
            max_width,
            tab_width,
            Some(ascii),
            &mut words,
            &mut breaks,
        );
        scratch.breaks = breaks;
        if words.is_empty() {
            // A non-empty paragraph with no words is whitespace-only (spaces,
            // tabs, NBSP runs…).  It is layout, not noise: keep it as an empty
            // segment so a blank line between hard newlines survives and
            // callers indexing the last segment (`chain.rs`) never see an
            // empty result for non-empty input.
            scratch.words = words;
            // A non-empty paragraph with no words is whitespace-only.
            // ASCII spaces/tabs are layout noise; non-ASCII whitespace such
            // as NBSP/U+3000 is content and must survive verbatim.
            out.push(Cow::Borrowed(if para.is_ascii() { "" } else { para }));
            continue;
        }
        for range in wrap_dp(&words, max_width, tables, scratch) {
            out.push(Cow::Borrowed(&para[range]));
        }
        scratch.words = words;
    }
    out
}

// ── Last-line reserve ──────────────────────────────────────────

/// [`wrap_optimal`] with an indivisible suffix that must fit on the **last**
/// line, so that line's body budget is `max_width - last_reserve` while every
/// other line keeps the full measure.
///
/// This is what `chain.rs` renders: a call-chain entry is a wrapped message
/// plus a locator (`main.pn:12:3`, or a kind label) that rides the last line
/// only.  Wrapping twice — once at the full measure, then re-wrapping the
/// overflowing tail at the narrowed one — is the two-phase algorithm Φ of
/// `research/locator-wrap/`.  Each phase is optimal for its own budget, yet the
/// composition's line count is unboundedly bad: with
/// `max_width = 2M + 1`, `last_reserve = 2M` and `M` unit words, Φ emits `M`
/// lines where 2 suffice (the first `M-1` words on line 1, the last word plus
/// the suffix on line 2) — a ratio of `M/2 → ∞`.
///
/// The minimum line count is `L(j* - 1) + 1`, where `L` is the greedy block
/// count at the full measure (exchange-optimal, and nondecreasing in the word
/// count) and
///
/// ```text
/// j* = min { j : width(w_j … w_n) <= max_width - last_reserve }
/// ```
///
/// (`solution.md` §2.3): every legal layout's last block is a *suffix*, so its
/// start is `>= j*`; a layout whose last block starts at `j` needs
/// `>= L(j - 1) >= L(j* - 1)` lines above it.  So the suffix is the *longest*
/// feasible one and the prefix is wrapped at the full measure — non-final lines
/// never pay for the suffix, and the old "re-wrap the whole tail at the narrow
/// measure" waste disappears.
///
/// The prefix is laid out by [`wrap_dp`], not by greedy: the split point is
/// count-minimal, while the remaining freedom (which `L`-block prefix layout)
/// is precisely the secondary objective the badness/fitness model resolves —
/// and it is a real degree of freedom, not a formality; `solution.md` §3(a)
/// exhibits instances where the greedy prefix is strictly worse on squared
/// slack than an equally short alternative.
///
/// **Infeasible instances** (the last word alone exceeds
/// `max_width - last_reserve`) take policy P1 of `solution.md` §4: wrap at the
/// full measure and let the locator overflow.  It is the minimum-line-count
/// policy and the one the renderer's row-count contract already assumed.  P2
/// (locator on its own line) doubles the row count for deep chains, which is
/// the row-budget hazard `chain.rs` documents; P3 (truncate the locator) would
/// break its click target and oscillate with the line/column digits.
///
/// The reserve is Monge-compatible (`solution.md` §3c): a budget *decrease* on
/// a single final row only shifts the query point (`C_n → C_n + s`), leaving
/// the quadrangle inequality at `2·δ·(ε − Δ) > 0`.  A budget *increase*, or
/// several distinct final budgets, would not preserve it — so the prefix DP
/// keeps its totals-monotone argument unchanged.
///
/// Not memoized: the batch memo stores byte ranges keyed by
/// `(hash, width, tab_width)`, and a reserved layout differs from an
/// unreserved one under the same key.  The dominant repeated strings (kind
/// labels) are handled by `chain.rs`'s single-line fast path before reaching
/// here, so the lost reuse is the rare multi-line repeat.
pub(crate) fn wrap_optimal_reserved<'a>(
    text: &'a str,
    max_width: usize,
    tab_width: usize,
    last_reserve: usize,
) -> SmallVec<[Cow<'a, str>; 4]> {
    let tab_width = tab_width.max(1);
    // Identity when nothing is reserved; also routes every degenerate-width and
    // over-long-input guard to the single implementation that owns it.
    if last_reserve == 0 || max_width == 0 || text.len() > u32::MAX as usize {
        return wrap_optimal(text, max_width, tab_width);
    }
    // A trailing `\n` is not a paragraph (`wrap_optimal("a\n") == ["a"]`), so
    // the reserve belongs to the last *real* paragraph.
    let core = text.strip_suffix('\n').unwrap_or(text);
    // Paragraph line counts are additive (`solution.md` §1.4): every paragraph
    // before the last is unconditional, so only the final one sees the reserve.
    let (head, last_para) = match core.rsplit_once('\n') {
        Some((head, last)) => (Some(head), last),
        None => (None, core),
    };
    let mut out: SmallVec<[Cow<'a, str>; 4]> = SmallVec::new();
    if let Some(head) = head {
        out.extend(wrap_optimal(head, max_width, tab_width));
    }
    let tables = wrap_tables(max_width);
    // Same reentrancy discipline as `wrap_optimal`: the thread-local scratch is
    // never borrowed twice, and the returned `Cow`s borrow `text`, not it.
    let ranges = if WRAP_IN_USE.get() {
        let mut local = WrapScratch::new();
        reserved_ranges(
            last_para,
            max_width,
            tab_width,
            last_reserve,
            tables.as_deref(),
            &mut local,
        )
    } else {
        WRAP_IN_USE.set(true);
        let _reset = WrapInUseGuard;
        WRAP_SCRATCH.with_borrow_mut(|scratch| {
            reserved_ranges(
                last_para,
                max_width,
                tab_width,
                last_reserve,
                tables.as_deref(),
                scratch,
            )
        })
    };
    out.extend(ranges.into_iter().map(|r| Cow::Borrowed(&last_para[r])));
    out
}

/// Byte ranges of one paragraph's lines when its last line must leave
/// `last_reserve` columns free.  Always returns at least one range.
fn reserved_ranges(
    para: &str,
    max_width: usize,
    tab_width: usize,
    last_reserve: usize,
    tables: Option<&WrapTables>,
    scratch: &mut WrapScratch,
) -> SmallVec<[Range<usize>; 4]> {
    // No floor: `max_width - last_reserve == 0` is genuinely infeasible
    // (`solution.md` §2.1 requires `|w_n| <= A - s`), so it must take P1 below.
    // Flooring at 1 would instead spend a whole extra row to carry one body
    // column, which P1 strictly dominates on row count.
    let budget_last = max_width.saturating_sub(last_reserve);
    if para.is_empty() {
        return smallvec![0..0];
    }
    // Same admission rule as `wrap_optimal_with`: edge ASCII whitespace and any
    // tab decline the verbatim path (the DP trims spaces and treats a tab as a
    // break opportunity, so a verbatim fast path here would diverge).
    let trimmed = para.trim_matches(' ');
    let ascii = is_printable_ascii(trimmed.as_bytes());
    let fits_last = if ascii {
        trimmed.len() <= budget_last
    } else {
        measure_width_unless_tab(trimmed, tab_width).is_some_and(|w| w <= budget_last)
    };
    if !trimmed.is_empty() && !edges_have_whitespace(trimmed) && fits_last {
        let start = para.len() - para.trim_start_matches(' ').len();
        return smallvec![start..start + trimmed.len()];
    }
    let mut words = std::mem::take(&mut scratch.words);
    let mut breaks = std::mem::take(&mut scratch.breaks);
    collect_words_into(
        para,
        max_width,
        tab_width,
        Some(ascii),
        &mut words,
        &mut breaks,
    );
    scratch.breaks = breaks;
    if words.is_empty() {
        scratch.words = words;
        // Whitespace-only paragraph: layout noise for ASCII, content otherwise
        // (mirrors `wrap_optimal_with`).
        return if para.is_ascii() {
            smallvec![0..0]
        } else {
            smallvec![0..para.len()]
        };
    }
    // `j*` of `solution.md` §2.3, accumulated with the DP's own width model
    // (`Word::width` + `Word::trailing_space`) so the split cannot disagree with
    // the widths `wrap_dp` scores.  Scanning backwards and overwriting keeps the
    // *smallest* feasible index — the longest suffix.
    let last = words.len() - 1;
    let mut acc = words[last].width as usize;
    let mut split = (acc <= budget_last).then_some(last);
    for idx in (0..last).rev() {
        acc += words[idx].width as usize + words[idx].trailing_space as usize;
        if acc <= budget_last {
            split = Some(idx);
        }
    }
    let ranges = match split {
        // P1: no legal layout — the last word alone overruns the last line's
        // budget.  Wrap at the full measure; the locator overflows this one row.
        None => wrap_dp(&words, max_width, tables, scratch),
        Some(j) => {
            let mut ranges = if j == 0 {
                SmallVec::new()
            } else {
                wrap_dp(&words[..j], max_width, tables, scratch)
            };
            // The suffix line: everything from word `j` to the last word, with
            // the caller's reserve left free.  `Word::end` already excludes the
            // final word's trailing whitespace, matching the DP's line ranges.
            ranges.push(words[j].start as usize..words[last].end as usize);
            ranges
        }
    };
    scratch.words = words;
    ranges
}

// ── Word collection ────────────────────────────────────────────

/// A single atomic word, plus the run of whitespace that follows it.
///
/// `trailing_space` is the glue Knuth-Plass stretches/shrinks: it is *included*
/// between words on the same line and *dropped* at a line break.
///
/// Fields are `u32` rather than `usize`: a `Word` lives inside one
/// `\n`-separated paragraph of a diagnostic message, and a paragraph that
/// large (4 GiB of text) would OOM long before offset overflow.  The payoff
/// is layout: the four `u32` fields make
/// `Word` 16 B, halving `SmallVec<[Word; 16]>`'s inline footprint
/// (512 B → 256 B) so the common ≤16-word paragraph stays in one or two
/// cache lines and `wrap_dp`'s prefix-sum loop walks twice as many words
/// per line fetch.
///
/// `width ≤ max_width` holds for every word produced by the ASCII splitter
/// and by the ordinary path of [`push_split_oversized`] — but NOT
/// unconditionally: a single grapheme cluster wider than the measure (a
/// 2-cell emoji at `max_width = 1`) is emitted as one piece and exceeds the
/// measure, because clusters are never split.  Nothing may debug-assert
/// `width ≤ max_width` on `Word` in general.
struct Word {
    /// Start byte offset inside the paragraph (trailing whitespace excluded).
    start: u32,
    /// End byte offset inside the paragraph (trailing whitespace excluded).
    end: u32,
    /// Display width of `start..end` in terminal cells.
    width: u32,
    /// Display width of the whitespace following `end`.
    trailing_space: u32,
}

impl Word {
    /// Byte range inside the paragraph (trailing whitespace excluded).
    #[inline]
    fn range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

/// Split `text` into words, splitting any single word wider than `max_width`
/// at grapheme boundaries.
///
/// Dispatches once on printable-ASCII: the ASCII path then measures every
/// chunk by byte count (one cell per byte, and `ascii_break_opportunities`
/// has already excluded tabs, so no SWAR scan or grapheme walk is repeated
/// per word), and the UAX #14 path keeps the general measurement.
///
/// Returns a [`SmallVec`] because a paragraph almost always has ≤16 words; the
/// `Word` struct is 16 B so `[Word; 16]` fits inline (256 B) and the common
/// case skips the heap.
/// * `known_ascii` — the caller's printable-ASCII verdict on `text`, when it
///   already computed one (`Some(true)` skips the re-scan entirely;
///   `Some(false)` declines straight to the UAX path; `None` re-derives it).
fn collect_words(
    text: &str,
    max_width: usize,
    tab_width: usize,
    known_ascii: Option<bool>,
) -> SmallVec<[Word; 16]> {
    let mut words: SmallVec<[Word; 16]> = SmallVec::new();
    let mut breaks: SmallVec<[usize; 16]> = SmallVec::new();
    collect_words_into(
        text,
        max_width,
        tab_width,
        known_ascii,
        &mut words,
        &mut breaks,
    );
    words
}

/// Fill `words` (clearing it first) with [`collect_words`]'s result.
///
/// The buffer is owned by [`WrapScratch`] so a multi-paragraph message pays
/// one `SmallVec` growth per message instead of one per paragraph.
fn collect_words_into(
    text: &str,
    max_width: usize,
    tab_width: usize,
    known_ascii: Option<bool>,
    words: &mut SmallVec<[Word; 16]>,
    breaks: &mut SmallVec<[usize; 16]>,
) {
    words.clear();
    if ascii_break_opportunities_into(text, known_ascii, breaks) {
        push_words_ascii(text, breaks, max_width, words);
    } else {
        uax_break_opportunities_into(text, breaks);
        WIDTH_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            push_words(text, breaks, max_width, tab_width, words, &mut cache);
        });
    }
}

/// The reference implementation: UAX #14 opportunities filtered by
/// [`should_break_at`] (the code-token filter).
fn uax_break_opportunities(text: &str) -> SmallVec<[usize; 16]> {
    let mut breaks: SmallVec<[usize; 16]> = SmallVec::new();
    uax_break_opportunities_into(text, &mut breaks);
    breaks
}

fn uax_break_opportunities_into(text: &str, breaks: &mut SmallVec<[usize; 16]>) {
    // `linebreaks` yields the byte offset of the first character after the
    // break; `should_break_at` filters out positions that would split a code
    // token.
    breaks.clear();
    breaks.push(0);
    for (pos, op) in linebreaks(text) {
        if matches!(op, BreakOpportunity::Allowed | BreakOpportunity::Mandatory)
            && pos > *breaks.last().unwrap()
            && should_break_at(text, pos)
        {
            breaks.push(pos);
        }
    }
    if *breaks.last().unwrap() < text.len() {
        breaks.push(text.len());
    }
}

/// Break opportunities for a paragraph of printable ASCII, without running the
/// UAX #14 state machine.
///
/// For printable ASCII the filter in [`should_break_at`] accepts exactly the
/// positions with a space on one side, and UAX #14 leaves only the *ends* of
/// space runs standing:
///
/// * LB7 (`× SP`) forbids every break before or inside a space run;
/// * breaks adjacent to anything else are rejected by `should_break_at`;
/// * a mandatory break cannot occur (the paragraph is already `\n`-split).
///
/// What is left is `SP ÷` (LB18), minus the three earlier rules that forbid a
/// break after spaces — LB13, LB14, LB15 — which [`ascii_break_after_space`]
/// evaluates from ASCII bytes alone.  `ascii_fast_path_breaks_match_uax`
/// checks this against `unicode_linebreak`.
///
/// `known_ascii` is the caller's printable-ASCII verdict on `text`, when it
/// has already computed one: `Some(true)` skips the re-scan (debug-asserted),
/// `Some(false)` declines straight to the UAX path, `None` re-derives it.
fn ascii_break_opportunities(
    text: &str,
    known_ascii: Option<bool>,
) -> Option<SmallVec<[usize; 16]>> {
    let mut breaks: SmallVec<[usize; 16]> = SmallVec::new();
    ascii_break_opportunities_into(text, known_ascii, &mut breaks).then_some(breaks)
}

fn ascii_break_opportunities_into(
    text: &str,
    known_ascii: Option<bool>,
    breaks: &mut SmallVec<[usize; 16]>,
) -> bool {
    let b = text.as_bytes();
    match known_ascii {
        Some(true) => debug_assert!(is_printable_ascii(b)),
        Some(false) => return false,
        None => {
            if !is_printable_ascii(b) {
                return false;
            }
        }
    }
    breaks.clear();
    breaks.push(0);
    let mut i = 0usize;
    while let Some(off) = memchr::memchr(b' ', &b[i..]) {
        let run_start = i + off;
        let mut end = run_start + 1;
        while end < b.len() && b[end] == b' ' {
            end += 1;
        }
        // `end` is the end of a maximal space run — the only candidate position
        // printable ASCII can offer.
        if end < b.len() {
            let prev = if run_start == 0 {
                None
            } else {
                Some(b[run_start - 1])
            };
            if ascii_break_after_space(prev, b[end]) {
                breaks.push(end);
            }
        }
        i = end;
    }
    if *breaks.last().unwrap() < b.len() {
        breaks.push(b.len());
    }
    true
}

/// May a line break at the end of a space run?  `prev` is the byte before the
/// run (`None` at the start of the paragraph), `next` the byte after it.
///
/// Total over printable ASCII: every path is decidable from ASCII alone
/// (`ascii_fast_path_breaks_match_uax` proves the fast path is complete for
/// printable ASCII), so there is no fallback arm — the UAX #14 path is taken
/// *before* this is called, when the paragraph is not printable ASCII.
#[inline]
fn ascii_break_after_space(prev: Option<u8>, next: u8) -> bool {
    // LB13: `× CL | CP | EX | IS | SY` — never break before a closer, an
    // exclamation/interrogation, an infix separator or a symbol, *even after
    // spaces*: `}` `)` `]` `!` `?` `,` `.` `:` `;` `/`.
    if matches!(
        next,
        b'}' | b')' | b']' | b'!' | b'?' | b',' | b'.' | b':' | b';' | b'/'
    ) {
        return false;
    }
    // LB14: `OP SP* ×` — never break after an opening bracket plus spaces.
    if matches!(prev, Some(b'(') | Some(b'[') | Some(b'{')) {
        return false;
    }
    // LB15: `QU SP* × OP` — never break after a quote plus spaces into an
    // opener.  `"` and `'` are `QU`; the grave accent (U+0060) is `AL` in
    // the Unicode version `unicode-linebreak` 0.1.5 targets (and has been
    // `AL` since Unicode 14), so it does not trigger LB15 and a break after
    // its spaces is allowed.  (LB16 `CP SP* × NS` and LB17 `B2 SP* × B2`
    // have no ASCII members on either side.)
    if matches!(next, b'(' | b'[' | b'{') {
        !matches!(prev, Some(b'"') | Some(b'\''))
    } else {
        true
    }
}

/// Does this UAX #14 break opportunity survive the code-token filter?
///
/// We keep only whitespace-adjacent and CJK↔CJK breaks. That is enough for
/// prose (wraps at spaces) and CJK (wraps between ideographs), and it never
/// splits `Int<32>`, `Vec<T>`, `(x, y)`, `foo-bar`, `a.b`, or `` `code` ``.
fn should_break_at(text: &str, pos: usize) -> bool {
    if pos == 0 || pos >= text.len() || !text.is_char_boundary(pos) {
        return false;
    }
    let b = text.as_bytes();
    // ASCII on both sides: no CJK pair is possible and `is_whitespace` for
    // ASCII is exactly 0x09..=0x0D | 0x20 — skip the Unicode tables.
    if b[pos - 1] < 0x80 && b[pos] < 0x80 {
        return matches!(b[pos - 1], 0x09..=0x0D | 0x20) || matches!(b[pos], 0x09..=0x0D | 0x20);
    }
    let prev = text[..pos].chars().next_back().unwrap();
    let next = text[pos..].chars().next().unwrap();

    if prev.is_whitespace() || next.is_whitespace() {
        return true;
    }
    is_cjk(prev) && is_cjk(next)
}

/// CJK ideographs, kana, Hangul, and CJK punctuation / fullwidth forms.
/// The ranges UAX #14 treats as class `ID` (plus a few `CJ` variants) — these
/// are the characters that may legitimately break between themselves.
fn is_cjk(c: char) -> bool {
    let c = c as u32;
    if c < 0x2E80 {
        return false;
    }
    matches!(c,
        0x2E80..=0x2EFF  // CJK Radicals Supplement
        | 0x2F00..=0x2FDF  // Kangxi Radicals
        | 0x3000..=0x303F  // CJK Symbols and Punctuation
        | 0x3040..=0x309F  // Hiragana
        | 0x30A0..=0x30FF  // Katakana
        | 0x3100..=0x312F  // Bopomofo
        | 0x31C0..=0x31EF  // CJK Strokes
        | 0x3200..=0x32FF  // Enclosed CJK Letters and Months
        | 0x3300..=0x33FF  // CJK Compatibility
        | 0x3400..=0x4DBF  // CJK Extension A
        | 0x4E00..=0x9FFF  // CJK Unified Ideographs
        | 0xA000..=0xA48F  // Yi Syllables
        | 0xAC00..=0xD7AF  // Hangul Syllables
        | 0xF900..=0xFAFF  // CJK Compatibility Ideographs
        | 0xFF00..=0xFFEF  // Halfwidth and Fullwidth Forms
        | 0x20000..=0x2A6DF // CJK Extension B
        | 0x2A700..=0x2EBEF // CJK Extension C–F
        | 0x2F800..=0x2FA1F // CJK Compatibility Ideographs Supplement
        | 0x30000..=0x3134F // CJK Extension G
    )
}

/// Build the words between UAX #14 `breaks` — the general path, used whenever
/// [`ascii_break_opportunities`] declines (tabs, non-ASCII).  The chunk→word
/// rules (trailing-whitespace folding, oversized splitting) live here and are
/// mirrored, in byte-count form, by [`push_words_ascii`].
fn push_words(
    text: &str,
    breaks: &[usize],
    max_width: usize,
    tab_width: usize,
    words: &mut SmallVec<[Word; 16]>,
    cache: &mut WidthCache,
) {
    for w in breaks.windows(2) {
        let chunk_start = w[0];
        let chunk_end = w[1];
        if chunk_start >= chunk_end {
            continue;
        }

        // Trim trailing whitespace off the chunk; what remains is the word.
        let chunk = &text[chunk_start..chunk_end];
        let content_end = chunk_start + trimmed_len(chunk);

        if content_end == chunk_start {
            // Whitespace-only chunk: fold it into the previous word's glue.
            if let Some(prev) = words.last_mut() {
                // A realistic whitespace run is a handful of cells; the
                // `u32` field only wraps on a multi-gigabyte run.
                prev.trailing_space += measure_width_cached(chunk, tab_width, cache) as u32;
            }
            continue;
        }

        let content = &text[chunk_start..content_end];
        let w_content = measure_width_cached(content, tab_width, cache);
        let space = measure_width_cached(&text[content_end..chunk_end], tab_width, cache);

        if w_content > max_width {
            // Unavoidably wider than the target: split at grapheme boundaries
            // so combining marks / emoji stay intact.
            push_split_oversized(
                words,
                content,
                chunk_start,
                max_width,
                tab_width,
                space,
                cache,
            );
        } else {
            words.push(Word {
                start: chunk_start as u32,
                end: content_end as u32,
                width: w_content as u32,
                trailing_space: space as u32,
            });
        }
    }
}

/// ASCII twin of [`push_words`]: for printable ASCII (no tabs — those decline
/// [`ascii_break_opportunities`]) every byte is exactly one cell, so each
/// `measure_width` call collapses to a byte count and the whole pass is
/// `O(N)` integer arithmetic instead of `O(Σ ceil(len/8))` SWAR iterations
/// plus a grapheme walk per word.
fn push_words_ascii(
    text: &str,
    breaks: &[usize],
    max_width: usize,
    words: &mut SmallVec<[Word; 16]>,
) {
    debug_assert!(is_printable_ascii(text.as_bytes()));
    for w in breaks.windows(2) {
        let chunk_start = w[0];
        let chunk_end = w[1];
        if chunk_start >= chunk_end {
            continue;
        }

        let content_len = trimmed_len(&text[chunk_start..chunk_end]);

        if content_len == 0 {
            // Whitespace-only chunk: fold it into the previous word's glue.
            // Every byte here is a printable ASCII cell.
            if let Some(prev) = words.last_mut() {
                prev.trailing_space += (chunk_end - chunk_start) as u32;
            }
            continue;
        }

        let space = (chunk_end - chunk_start) - content_len;
        if content_len > max_width {
            push_split_oversized_ascii(words, chunk_start, content_len, max_width, space);
        } else {
            words.push(Word {
                start: chunk_start as u32,
                end: (chunk_start + content_len) as u32,
                width: content_len as u32,
                trailing_space: space as u32,
            });
        }
    }
}

/// Append the pieces of a printable-ASCII over-wide word (starting at byte
/// offset `offset`, byte length `len`) straight into `words`.  A grapheme
/// boundary is every byte, so the grapheme walk of [`push_split_oversized`]
/// reduces to fixed-width byte slicing.
fn push_split_oversized_ascii(
    words: &mut SmallVec<[Word; 16]>,
    offset: usize,
    len: usize,
    max_width: usize,
    trailing: usize,
) {
    let mut off = offset;
    let mut rest = len;
    // `max_width.max(1)` keeps the slice step positive (the grapheme walk in
    // `push_split_oversized` makes progress per piece the same way).
    let step = max_width.max(1);
    while rest > max_width {
        words.push(Word {
            start: off as u32,
            end: (off + step) as u32,
            width: step as u32,
            trailing_space: 0,
        });
        off += step;
        rest -= step;
    }
    if rest > 0 {
        words.push(Word {
            start: off as u32,
            end: (off + rest) as u32,
            width: rest as u32,
            trailing_space: trailing as u32,
        });
    } else if let Some(last) = words.last_mut() {
        // `max_width == 0` makes every byte its own piece; attach the glue to
        // the final piece instead of dropping it.
        last.trailing_space = trailing as u32;
    }
}

/// Append the grapheme-boundary pieces of an over-wide `text` (at byte offset
/// `offset` inside the paragraph, so ranges stay paragraph-relative) straight
/// into `words` — no intermediate `Vec`.  `trailing` is the glue that follows
/// the word and belongs to its last piece.
///
/// Pieces are packed up to `max_width` cells, but a single grapheme cluster
/// wider than the measure is emitted as one over-wide piece — clusters are
/// never split, so `Word::width ≤ max_width` cannot be guaranteed here (see
/// the module docs' "Known approximations").
fn push_split_oversized(
    words: &mut SmallVec<[Word; 16]>,
    text: &str,
    offset: usize,
    max_width: usize,
    tab_width: usize,
    trailing: usize,
    cache: &mut WidthCache,
) {
    let mut sub_start = 0usize;
    let mut sub_width = 0usize;
    for (i, g) in text.grapheme_indices(true) {
        let gw = grapheme_width_cached(g, sub_width, tab_width, cache);
        if sub_width > 0 && sub_width + gw > max_width {
            words.push(Word {
                start: (offset + sub_start) as u32,
                end: (offset + i) as u32,
                width: sub_width as u32,
                trailing_space: 0,
            });
            sub_start = i;
            sub_width = 0;
        }
        sub_width += gw;
    }
    if sub_start < text.len() {
        words.push(Word {
            start: (offset + sub_start) as u32,
            end: (offset + text.len()) as u32,
            width: sub_width as u32,
            trailing_space: trailing as u32,
        });
    }
}

// ── Display width ──────────────────────────────────────────────

/// Display width of `s` in terminal cells.
#[inline]
fn measure_width(s: &str, tab_width: usize) -> usize {
    WIDTH_CACHE.with(|cell| measure_width_cached(s, tab_width, &mut cell.borrow_mut()))
}

fn measure_width_cached(s: &str, tab_width: usize, cache: &mut WidthCache) -> usize {
    // Fast path: printable ASCII is one cell per byte, found without ever
    // building a grapheme iterator.  Mirrors the fast paths in
    // `line_display_width` / `visible_width`.
    if is_printable_ascii(s.as_bytes()) {
        return s.len();
    }
    let mut w = 0;
    for g in s.graphemes(true) {
        w += grapheme_width_cached(g, w, tab_width, cache);
    }
    w
}

/// `measure_width`, but returns `None` as soon as a tab is seen.
///
/// The fast-path admission test needs both facts; computing them separately
/// walked the grapheme stream twice (one `memchr` for the tab, then
/// `measure_width`).  Printable ASCII keeps the byte-length shortcut.
fn measure_width_unless_tab(s: &str, tab_width: usize) -> Option<usize> {
    if is_printable_ascii(s.as_bytes()) {
        return Some(s.len());
    }
    let mut width = 0;
    let mut saw_tab = false;
    WIDTH_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        for g in s.graphemes(true) {
            if g == "\t" {
                saw_tab = true;
                break;
            }
            width += grapheme_width_cached(g, width, tab_width, &mut cache);
        }
    });
    (!saw_tab).then_some(width)
}

/// `s.trim_end_matches(char::is_whitespace).len()` without touching the
/// Unicode tables when the tail is ASCII — glue is almost always plain spaces.
#[inline]
fn trimmed_len(s: &str) -> usize {
    let b = s.as_bytes();
    let mut k = b.len();
    while k > 0 {
        match b[k - 1] {
            // ASCII whitespace.
            0x09..=0x0D | 0x20 => k -= 1,
            // Non-whitespace ASCII: the scan already has the answer.  (A
            // multi-byte whitespace character always ends in a byte ≥ 0x80, so
            // this branch cannot cut one in half.)
            0x00..=0x7F => return k,
            // Something multi-byte behind the run: decode the trailing character once.
            // If it is not whitespace (the common CJK word ending), the answer
            // is immediate; only a whitespace char needs the full predicate.
            _ => {
                if s.chars().next_back().is_some_and(|ch| !ch.is_whitespace()) {
                    return k;
                }
                return s.trim_end_matches(char::is_whitespace).len();
            }
        }
    }
    0
}

/// After `trim_matches(' ')`, any remaining edge whitespace is non-ASCII
/// (NBSP, U+3000…).  The DP keeps leading non-ASCII whitespace as LB12 glue
/// inside the first word but trims trailing non-ASCII whitespace via
/// [`trimmed_len`]'s multi-byte fallback, and a leading non-ASCII-whitespace
/// run folded with ASCII spaces is dropped as a whitespace-only chunk — in
/// every case a fast path emitting `trim_matches(' ')` would diverge, so the
/// fast path rejects such edges and falls through to the DP.  Both char reads
/// are O(1) (boundary scan from each end of a `&str`).
#[inline]
fn edges_have_whitespace(s: &str) -> bool {
    s.chars().next().map_or(false, |c| c.is_whitespace())
        || s.chars().next_back().map_or(false, |c| c.is_whitespace())
}

// ── Dynamic program ────────────────────────────────────────────

/// Choose the line breaks for `words` (one paragraph) minimising total badness.
///
/// [`badness`] looks at the line above, so one scalar per prefix is no longer
/// enough: `dp` is indexed by `(word index, fitness class of the last line)`.
/// A transition adds the new line's intrinsic badness plus the fitness jump
/// out of the source state's class ([`JUMP`]); four classes keep the state
/// space at `4(n+1)` and the inner loop at 4.
///
/// `dp[j * N_FITNESS + c]` is the minimum cost of laying out `words[0..j]`
/// with its last line in class `c`; `prev` records the predecessor *state*
/// (`i * N_FITNESS + pc`) as a single `u32`, half the traffic of a
/// `(usize, u8)` pair and all the reconstruction walk needs.  O(n·k) time
/// (k = words that fit one line) after a min-plus precompute over the
/// `N_FITNESS` classes folds the inner `pc` scan into an O(1) lookup,
/// O(n) space.
///
/// Returns a [`SmallVec`] because a paragraph wraps to a handful of lines,
/// so the common case stays inline.
/// Tabulated badness/class lookup tables for the DP, keyed by `max_width`.
///
/// The tables depend only on `max_width`, which is fixed per render batch
/// (`DiagOpts`), but `wrap_dp` runs once per paragraph per message.  So they
/// live in a thread-local keyed by `max_width`: a batch of renders builds
/// them once, and only a change of `max_width` rebuilds.  Hoisted out of
/// `wrap_dp` so a multi-paragraph message does one TLS access + one `Rc`
/// clone per *message* instead of one per *paragraph*; the `Rc` is handed
/// down to every `wrap_dp` call.  Each table is ~7 KB (`[u8;801]` +
/// `[f64;801]`), so the 8-slot TLS worst case is ~56 KB.
const MAX_TABLE_WIDTH: usize = 800;
struct WrapTables {
    max_width: usize,
    class: [u8; MAX_TABLE_WIDTH + 1],
    badness: [f64; MAX_TABLE_WIDTH + 1],
}
type WrapTableSlot = SmallVec<[std::sync::Arc<WrapTables>; 8]>;

thread_local! {
    /// A small per-thread cache of badness/class tables, keyed by `max_width`.
    /// Eight slots (≈56 KB TLS worst case): the *actual* key churn is
    /// intra-render — one diagnostic wraps the header, label connector, note,
    /// help and diff block at 5–6 distinct widths — so a two-slot cache
    /// evicted and rebuilt the ~11 KB table several times per diagnostic.
    /// Eviction is LRU-by-last-use.
    static WRAP_TABLES: std::cell::RefCell<WrapTableSlot> =
        std::cell::RefCell::new(SmallVec::new());
    /// Reentrancy flag for the scratch borrow (see `wrap_optimal`).
    static WRAP_IN_USE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Per-thread `WrapScratch` reused across every `wrap_optimal` call on
    /// this thread (see the call site): one-time growth per thread instead
    /// of up to five `SmallVec` growths per wrapping message.
    static WRAP_SCRATCH: std::cell::RefCell<WrapScratch> =
        std::cell::RefCell::new(WrapScratch::new());
}

/// RAII clear for [`WRAP_IN_USE`]: the thread-local scratch must be released
/// even if a wrap body panics, or every later wrap on this thread would take
/// the (correct but allocating) local-scratch fallback forever.
struct WrapInUseGuard;

impl Drop for WrapInUseGuard {
    fn drop(&mut self) {
        WRAP_IN_USE.set(false);
    }
}

/// Shared table cache used while a parallel render batch is active.
static SHARED_TABLES: std::sync::OnceLock<std::sync::RwLock<WrapTableSlot>> =
    std::sync::OnceLock::new();

fn table_from_slot(slot: &mut WrapTableSlot, max_width: usize) -> std::sync::Arc<WrapTables> {
    // LRU-by-last-use: move a hit to the back so the widths that recur
    // within one render are the ones that survive eviction.
    if let Some(pos) = slot.iter().position(|t| t.max_width == max_width) {
        let t = slot.remove(pos);
        slot.push(std::sync::Arc::clone(&t));
        return t;
    }
    let inv_measure = 1.0 / (max_width.max(1) as f64);
    let mut t = WrapTables {
        max_width,
        class: [0; MAX_TABLE_WIDTH + 1],
        badness: [0.0; MAX_TABLE_WIDTH + 1],
    };
    for lw in 0..=max_width {
        t.class[lw] = fitness_class(lw, max_width);
        t.badness[lw] = intrinsic_badness(lw, max_width, inv_measure, false);
    }
    let arc = std::sync::Arc::new(t);
    if slot.len() >= 8 {
        slot.remove(0);
    }
    slot.push(std::sync::Arc::clone(&arc));
    arc
}

/// The badness/class tables for `max_width`, or `None` when the width
/// exceeds the tabulation bound (the DP then computes inline).
fn wrap_tables(max_width: usize) -> Option<std::sync::Arc<WrapTables>> {
    if max_width > MAX_TABLE_WIDTH {
        return None;
    }
    if wrap_shared() {
        let mut slot = SHARED_TABLES
            .get_or_init(|| std::sync::RwLock::new(SmallVec::new()))
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Some(table_from_slot(&mut slot, max_width))
    } else {
        Some(WRAP_TABLES.with(|cell| table_from_slot(&mut cell.borrow_mut(), max_width)))
    }
}

/// Scratch buffers shared by every `wrap_dp` call of one `wrap_optimal`:
/// each is cleared and re-sized per paragraph while its capacity — and its
/// heap allocation, once a paragraph outgrows the inline store — is kept
/// across paragraphs.  Turns 5 allocations per paragraph into at most 5 per
/// message.
struct WrapScratch {
    pws: SmallVec<[usize; 8]>,
    dp: SmallVec<[f64; 32]>,
    prev: SmallVec<[u32; 32]>,
    /// One bit per fitness class; lets the min-plus precompute skip the 4×4
    /// scan when only one class at `j` is reachable (the common case).
    reachable_mask: SmallVec<[u8; 32]>,
    best_cost: SmallVec<[[f64; N_FITNESS]; 8]>,
    best_src: SmallVec<[[u32; N_FITNESS]; 8]>,
    /// Word buffer for one paragraph, reused across paragraphs.
    words: SmallVec<[Word; 16]>,
    /// Break-opportunity buffer for one paragraph, reused across paragraphs.
    /// `ascii_break_opportunities` / `uax_break_opportunities` used to
    /// allocate a fresh `SmallVec` per paragraph; a long message with many
    /// paragraphs paid that allocator churn on every one.
    breaks: SmallVec<[usize; 16]>,
}

impl WrapScratch {
    fn new() -> Self {
        // Pre-reserve the common short-label size.  A first paragraph would
        // otherwise grow each buffer from zero inline capacity several times
        // in an LSP re-emit loop; 32 entries covers the dominant `n <= 32`
        // diagnostic and costs only a few KiB per thread.
        WrapScratch {
            pws: SmallVec::with_capacity(32),
            dp: SmallVec::with_capacity(32),
            prev: SmallVec::with_capacity(32),
            reachable_mask: SmallVec::with_capacity(32),
            best_cost: SmallVec::with_capacity(32),
            best_src: SmallVec::with_capacity(32),
            words: SmallVec::new(),
            breaks: SmallVec::new(),
        }
    }
}

fn wrap_dp(
    words: &[Word],
    max_width: usize,
    tables: Option<&WrapTables>,
    scratch: &mut WrapScratch,
) -> SmallVec<[Range<usize>; 4]> {
    let n = words.len();
    if n == 0 {
        return SmallVec::new();
    }
    if n == 1 {
        // One word, one line: exactly what the loop below computes, minus the
        // bookkeeping.
        return smallvec![words[0].range()];
    }
    // State indices are `u32` (`j * N_FITNESS + class`).  Use checked
    // arithmetic here as well as the public guard: on 32-bit `(n+1)*4` can
    // wrap even when `n` alone looks representable.
    let per_word = || {
        words
            .iter()
            .map(|w| w.start as usize..w.end as usize)
            .collect()
    };
    let Some(n_states) = n.checked_add(1).and_then(|v| v.checked_mul(N_FITNESS)) else {
        return per_word();
    };
    if n_states - 1 > u32::MAX as usize {
        return per_word();
    }
    debug_assert!(
        n <= u32::MAX as usize / N_FITNESS,
        "wrap_dp state index would overflow u32"
    );

    // One fused prefix array: `pws[m] = Σ_{k<m} (width_k + trailing_space_k)`.
    // The line covering `words[i..j)` then measures `base_j - pws[i]` where
    // `base_j = pws[j] - trailing_space_{j-1}` (the last word's trailing space
    // is dropped at the break) — one load and one subtract per candidate,
    // vs. four loads and three ops over the two separate arrays.
    let WrapScratch {
        pws,
        dp,
        prev,
        reachable_mask,
        best_cost,
        best_src,
        words: _,
        breaks: _,
    } = scratch;
    pws.clear();
    pws.resize(n + 1, 0);
    for (i, w) in words.iter().enumerate() {
        let Some(next) = pws[i]
            .checked_add(w.width as usize)
            .and_then(|v| v.checked_add(w.trailing_space as usize))
        else {
            return per_word();
        };
        pws[i + 1] = next;
    }

    // Hoisted once for the whole DP: `intrinsic_badness` would otherwise
    // recompute the `max` + `as f64` cast on every candidate line.
    let measure = max_width.max(1) as f64;
    let inv_measure = 1.0 / measure;

    // Precomputed lookup tables for the inner loop: `class_tab[lw]` =
    // `fitness_class(lw, max_width)` and `badness_tab[lw]` =
    // `intrinsic_badness(lw, max_width, inv_measure, false)` for every legal
    // line width `lw <= max_width`.  The inner loop's div-and-multiply badness
    // computation (~30 cycles) and 3-compare class computation collapse to
    // two indexed loads (~4 cycles) — a 3–4× reduction of the hottest loop.
    // Last-line badness for `lw <= max_width` is always 0, so the table stores
    // only the non-last-line value; the `is_last_line` branch is uniform
    // across every candidate for a fixed `j` and thus perfectly predicted.
    // Over-wide lines (`lw > max_width`) fall back to inline computation —
    // identical to the previous behavior.
    //
    // The tables are hoisted one level further: `wrap_optimal` fetches them
    // once per message and passes `&WrapTables` down, so a multi-paragraph
    // message pays one TLS access instead of one per paragraph.
    let use_table = tables.is_some();
    let (class_tab, badness_tab): (&[u8], &[f64]) = match tables {
        Some(t) => (&t.class[..], &t.badness[..]),
        None => (&[], &[]),
    };

    // All five buffers come from the shared `WrapScratch`: `clear` +
    // `resize` keeps their capacity (and any heap allocation) across the
    // paragraphs of one `wrap_optimal` call.  `dp`/`prev` are fully
    // overwritten via resize-with-init below, so stale contents are harmless.
    dp.clear();
    dp.resize(n_states, f64::INFINITY);
    prev.clear();
    prev.resize(n_states, u32::MAX);
    reachable_mask.clear();
    reachable_mask.resize(n + 1, 0);
    // `j == 0` is the empty prefix: nothing sits above the first line, so all
    // entry classes are free.
    for c in 0..N_FITNESS {
        dp[c] = 0.0;
    }
    reachable_mask[0] = (1u8 << N_FITNESS) - 1;

    // Min-plus precomputation over fitness classes.  For each prefix end `i`
    // and each class `class` of the *next* line, `best_cost[i][class]` is
    // `min_pc dp[i][pc] + JUMP[pc][class]` and `best_src[i][class]` records
    // the argmin `pi = i·N_FITNESS + pc`.  This decouples the `pc` scan from
    // the `(i, j)` transition: the inner loop drops from O(N_FITNESS) to O(1)
    // per `(i, j)` — a 4× reduction of the DP's hottest loop.
    //
    // `best_prev[i]` is computed once, right after `dp[i]` is finalised (at
    // the end of outer iteration `j == i`), so the total precompute cost is
    // O(n·N_FITNESS²) = O(16n) and the transition cost is O(n·k) — vs.
    // O(4n·k) before, a 2–4× speedup for the typical k ≈ 4–16 words per line.
    //
    // For `i == 0` the first line has no predecessor, so the jump term is
    // zero regardless of class: `best_cost[0][*] = 0.0`, `best_src[0][*] = 0`.
    // This unifies the old `i == 0` special case into the same lookup.
    best_cost.clear();
    best_cost.resize(n + 1, [f64::INFINITY; N_FITNESS]);
    best_src.clear();
    best_src.resize(n + 1, [0; N_FITNESS]);
    for class in 0..N_FITNESS {
        best_cost[0][class] = 0.0;
    }

    // Any over-wide line (width > `max_width`) is never chosen when a legal
    // break exists: its overflow penalty is cubic in the excess and scaled
    // by `OVERFLOW_PENALTY` (≥ 1000 for a one-column overflow), which exceeds
    // the cost of any legal line (intrinsic ≤ 1, fitness jump ≤ 1/8), so it
    // always loses.  Keeping `j - 1` as a start covers the rare case where no
    // legal break exists (a grapheme wider than `max_width`).
    //
    // `line_width(i, j)` is non-decreasing in `j` for fixed `i`, so the set of
    // starts whose line still fits only shrinks: `i_lo` walks forward at most
    // `n` times in total — no binary search — and the inner loop runs over
    // only the few words that fit roughly two lines.
    let mut i_lo = 0usize;
    for j in 1..=n {
        let is_last_line = j == n;
        // Hoisted per `j`: every candidate line ending at `j` measures
        // `base_j - pws[i]`, and `base_j` is independent of `i`.
        let base_j = pws[j] - words[j - 1].trailing_space as usize;
        let mut lw_lo = base_j - pws[i_lo];
        while i_lo + 1 < j && lw_lo > max_width {
            i_lo += 1;
            lw_lo = base_j - pws[i_lo];
        }
        // First candidate reuses `lw_lo` from the window advance above,
        // avoiding a redundant `line_width` call per `j`.
        {
            let (class, base) = if use_table && lw_lo <= max_width {
                let c = class_tab[lw_lo] as usize;
                let b = if is_last_line {
                    0.0
                } else {
                    badness_tab[lw_lo]
                };
                (c, b)
            } else {
                (
                    fitness_class(lw_lo, max_width) as usize,
                    intrinsic_badness(lw_lo, max_width, inv_measure, is_last_line),
                )
            };
            let s = j * N_FITNESS + class;
            let cost = best_cost[i_lo][class] + base;
            if cost < dp[s] {
                dp[s] = cost;
                prev[s] = best_src[i_lo][class];
                reachable_mask[j] |= 1u8 << class;
            }
        }
        for i in (i_lo + 1)..j {
            // Width of the line covering words[i..j): `pws` is monotone and
            // `trailing_space_{j-1} ≤ pws[j] - pws[i]` for `i < j`, so this
            // is the same quantity the two-array form computed.  The `i_lo`
            // window advance above guarantees `lw <= max_width` for every
            // `i > i_lo` (`pws` is monotone, so larger `i` yields smaller
            // `lw`; the first candidate — which *can* be over-wide — is
            // handled separately before this loop).
            let lw = base_j - pws[i];
            debug_assert!(
                lw <= max_width,
                "i_lo window must guarantee legal-width lines"
            );
            let (class, base) = if use_table {
                let c = class_tab[lw] as usize;
                let b = if is_last_line { 0.0 } else { badness_tab[lw] };
                (c, b)
            } else {
                (
                    fitness_class(lw, max_width) as usize,
                    intrinsic_badness(lw, max_width, inv_measure, is_last_line),
                )
            };
            let s = j * N_FITNESS + class;
            // `best_cost[i][class]` already folds the `min_pc` over the
            // predecessor's fitness class; `base` is independent of `pc`.
            // `∞ + base < ∞` is false, so unreachable states are skipped
            // without an explicit check and the loop stays branch-free.
            let cost = best_cost[i][class] + base;
            if cost < dp[s] {
                dp[s] = cost;
                prev[s] = best_src[i][class];
                reachable_mask[j] |= 1u8 << class;
            }
        }
        // `dp[j]` is now final — no future iteration writes it.  Precompute
        // the min-plus convolution for outer iterations `j' > j` that will
        // use `j` as a predecessor.  Skip `j == n`: its `best_prev` is never
        // read (there is no `j' > n`).
        let j0 = j * N_FITNESS;
        // Skip the min-plus precompute when no state at `j` is reachable —
        // the inner loop would do 4×4 `INFINITY + JUMP` additions and
        // comparisons only to leave `best_cost[j]` at `INFINITY` (its
        // initial value).  Checking all four fitness classes at `j` is
        // required: different classes can have different reachability.
        // Reads the one-byte reachability mask instead of four `f64`s.
        let mask = reachable_mask[j];
        if j < n && mask != 0 {
            if mask.count_ones() == 1 {
                // The common case: exactly one class at this prefix is
                // reachable.  The min-plus scan then has a single candidate,
                // so copy the JUMP row directly.
                let pc = mask.trailing_zeros() as usize;
                let pi = j0 + pc;
                for class in 0..N_FITNESS {
                    best_cost[j][class] = dp[pi] + JUMP[class][pc];
                    best_src[j][class] = pi as u32;
                }
            } else {
                for class in 0..N_FITNESS {
                    let mut bc = f64::INFINITY;
                    let mut bp = 0u32;
                    for pc in 0..N_FITNESS {
                        let pi = j0 + pc;
                        let cost = dp[pi] + JUMP[class][pc];
                        if cost < bc {
                            bc = cost;
                            bp = pi as u32;
                        }
                    }
                    best_cost[j][class] = bc;
                    best_src[j][class] = bp;
                }
            }
        }
        //
        // The argument for pruning only bounds the *jump* term:
        // `JUMP[class][pc]` is zero for `|class - pc| ≤ 1` and
        // ≥ FITNESS_STEP_DEMERIT otherwise. But the minimised quantity
        // is `dp[j][pc] + JUMP[class][pc]`, and `dp[j][pc]` is not
        // comparable across classes. Witness with `class = 0` (Tight)
        // and FITNESS_STEP_DEMERIT = 1/16:
        //
        //     pc   class      dp[j][pc]   JUMP[0][pc]    total
        //     0    Tight      0.20        0.0           0.20
        //     1    Normal     0.18        0.0           0.18
        //     2    Loose      0.05        0.0625        0.1125  ← min
        //     3    VeryLoose  0.03        0.125         0.155
        //
        // `pc = 2` is two classes away from `class = 0` and pays a
        // non-zero jump, yet wins because `dp[j][2]` is far cheaper
        // than `dp[j][0]` or `dp[j][1]`. An adjacent-only band discards
        // the true optimum.
        //
        // Reachability of the witness: `dp[j][0]` (last line Tight)
        // can carry a large accumulated cost if reaching Tight required
        // jumping out of a Loose/VeryLoose predecessor earlier, while
        // `dp[j][2]` (last line Loose) can be cheaper if reached
        // smoothly. Cumulative cost is unbounded in `n`, so the spread
        // `dp[j][0] - dp[j][2]` can exceed `2·FITNESS_STEP_DEMERIT =
        // 0.125` — no invariant bounds the cross-class gap. The only
        // universal lower bound is `dp[j][class] ≥ intrinsic(last line
        // of that class)`, and it goes the wrong way (Tight has the
        // smallest intrinsic cost), so it cannot rescue pruning.
        //
        // The `adjacent_lines_resist_a_jagged_edge` test pins this
        // tradeoff: the DP picks a slightly worse-intrinsic layout
        // because the smoother jump sequence wins, and that decision
        // depends on evaluating cross-class predecessors.
        //
        // Cost of the 4×4: ~16 adds + 12 compares per `j` (≈28 ops),
        // dominated by the O(n·k) `(i, j)` transition for realistic `n`.
        // (Monge/SMAWK note, since the structure is real and the
        // pruning temptation above sometimes mutates into a SMAWK
        // suggestion: for a fixed class `c` the `(i, j)` transition
        // matrix is `M[i][j] = best_cost[i][c] + g(base_j - pws[i])`
        // with `g` convex. Per-row offsets *preserve* Monge —
        // `A[i] + A[i+1]` cancels exactly in the Monge inequality — so
        // `best_cost[i][c]` being an arbitrary function of `i` does not
        // break the structure, and `g`'s convexity (both branches, with
        // `g'(M-) = g'(M+) = 0`) makes `C` Monge. SMAWK therefore
        // applies in principle, but the matrix is revealed online —
        // row `i` is only evaluable once `dp[i]` is finalised — so the
        // applicable variant is Larmore-Schieber at O(n·α(n)), not
        // classic offline SMAWK. The blocker is the constant factor:
        // Larmore-Schieber's per-comparison cost is ~30-50 ops versus
        // the current monotone-`i_lo` window's ~10-15, and for the
        // `n ≈ 5-50`, `k ≈ 5-20` of diagnostic wrapping the asymptotic
        // win is eaten by constants. The cheap middle ground — a
        // monotone argmin pointer `max(i_lo, i_star_{j-1})` — is valid
        // (column argmin of a Monge matrix is monotone in `j`, and the
        // proof uses only the Monge condition, so the per-row offset
        // does not break it) but gains nothing in practice: on a
        // non-last line the optimum is the most-filled legal line,
        // `i*(j) ≈ i_lo(j)`, so the pointer sits at `i_lo` and skips
        // nothing. Larmore-Schieber does win decisively once
        // `n ≳ 500` and `k ≳ 15` — roughly a 3-5× speedup — which
        // the current `perf_wrap_optimal` long-prose case (~130
        // words) does not yet reach. Realistic triggers:
        // `explain_error_code` wrapping a multi-KB explanation body,
        // a diagnostic whose message is a long prose paragraph of
        // short 2-3-char words (large `n` and large `k` at once), or
        // a future caller that wraps a whole file's contents.)
        //
        // Decision (recorded so this does not get "optimised" back):
        // keep the naive monotone window.  Crossover is
        // `n* ≈ 2·c_LS·α(n)/c_scan` ≈ 10-15 for the mid constants
        // (c_scan 10-15, c_LS 30-50, α≈1-2), and the pointer pins to
        // `i_lo` in this workload, so LS would pay its bookkeeping and
        // prune nothing.  Revisit only with a measured `n ≳ 10²-10³`
        // unclosed-window workload; the block-matrix LS variant is
        // additionally invalid for a general jump matrix `J` (the
        // lexicographic block order is Monge only for an additively
        // separable `J`), so a future attempt must keep the 4 classes
        // as 4 independent per-class recurrences.
    }

    // Cheapest terminal state, then walk the `prev` links back to the
    // paragraph start and reverse into source order.
    let mut c = 0usize;
    for cand in 1..N_FITNESS {
        if dp[n * N_FITNESS + cand] < dp[n * N_FITNESS + c] {
            c = cand;
        }
    }
    let mut ranges: SmallVec<[Range<usize>; 4]> = SmallVec::new();
    let mut s = n * N_FITNESS + c;
    loop {
        let pj = prev[s];
        debug_assert!(
            pj != u32::MAX,
            "wrap_dp backtracked into an unreachable terminal state"
        );
        // Release-mode guard: a renderer bug must never turn into a panic
        // (or an OOB index via `u32::MAX / N_FITNESS`) over its own output.
        if pj == u32::MAX {
            // Fallback: one word per line, the always-valid layout.
            ranges.clear();
            for w in words {
                ranges.push(w.start as usize..w.end as usize);
            }
            return ranges;
        }
        let j = s / N_FITNESS;
        let i = (pj as usize) / N_FITNESS;
        ranges.push(words[i].start as usize..words[j - 1].end as usize);
        if i == 0 {
            break;
        }
        s = pj as usize;
    }
    ranges.reverse();
    ranges
}

/// Cost of one line given the fitness class of the line directly above it.
///
/// Two components, both in intrinsic-badness units (one completely empty
/// line costs `1.0`):
///
/// * **Intrinsic** — Knuth-Plass' cubed slack, `(empty / measure)^3`, on
///   every line except the last (free, so a paragraph is never stretched to
///   fill it), and a cubic overflow term in absolute columns otherwise.
/// * **Adjacent** — the demerit for the fitness jump from the previous line
///   ([`fitness_jump`]).  This is what makes badness a function of *two*
///   lines: the same width can be cheap or expensive depending on how full
///   the line above it was.  The jump into the last line is charged even
///   though the last line's own slack is free — a one-word widow under a
///   full line is exactly the jarring edge this catches.
///
/// `prev_class` is `None` for a paragraph's first line, which has nothing
/// above it to be jarring against.
///
/// This is the reference scorer used by the tests; [`wrap_dp`] inlines the two
/// halves (with [`JUMP`] instead of `fitness_jump`) per candidate line.
#[inline]
#[allow(dead_code)] // tests only
fn badness(prev_class: Option<u8>, line_width: usize, max_width: usize, is_last_line: bool) -> f64 {
    let inv_measure = 1.0 / (max_width.max(1) as f64);
    intrinsic_badness(line_width, max_width, inv_measure, is_last_line)
        + fitness_jump(prev_class, fitness_class(line_width, max_width))
}

/// Single-line (pre-adjacency) badness: cubed slack, zero on the last line,
/// cubic overflow with a large multiplier otherwise.
///
/// `inv_measure` is `1.0 / (max_width.max(1) as f64)`, hoisted by the caller:
/// the slack ratio becomes a multiply, so both the table build and the
/// over-`MAX_TABLE_WIDTH` fallback are divide-free.
#[inline]
fn intrinsic_badness(
    line_width: usize,
    max_width: usize,
    inv_measure: f64,
    is_last_line: bool,
) -> f64 {
    if line_width <= max_width {
        if is_last_line {
            0.0
        } else {
            let slack = (max_width - line_width) as f64 * inv_measure;
            slack * slack * slack
        }
    } else {
        let over = (line_width - max_width) as f64;
        OVERFLOW_PENALTY * over * over * over
    }
}

/// TeX's four fitness classes: how full a line is, as the fraction of the
/// measure it leaves empty.
///
/// | class | empty / measure | name      |
/// |-------|-----------------|-----------|
/// | 0     | < 1/4           | Tight     |
/// | 1     | < 1/2           | Normal    |
/// | 2     | < 3/4           | Loose     |
/// | 3     | ≥ 3/4           | VeryLoose |
///
/// An over-full line leaves nothing empty, so it lands in class 0 — the same
/// "too tight" class TeX gives it.
///
/// Integer comparisons instead of `f64` division: `4*empty < measure` ⇔
/// `empty/measure < 0.25` when `measure > 0`, and so on for the 1/2 and 3/4
/// boundaries.  The comparisons run in `u128`: `max_width` is untrusted
/// (it comes from `--width` / `COLUMNS`), and for `max_width > usize::MAX / 4`
/// the naive `4 * empty` (or `3 * measure`) would wrap and silently produce
/// the wrong fitness class on the one path (`wrap_optimal`'s inline DP) that
/// still runs above `MAX_TABLE_WIDTH`.
#[inline]
fn fitness_class(line_width: usize, max_width: usize) -> u8 {
    let measure = max_width.max(1) as u128;
    let empty = max_width.saturating_sub(line_width) as u128;
    if 4 * empty < measure {
        0
    } else if 2 * empty < measure {
        1
    } else if 4 * empty < 3 * measure {
        2
    } else {
        3
    }
}

/// Demerit for the fitness jump between adjacent lines. One class step is
/// free (a progressive rag reads as intentional); each step beyond it costs
/// [`FITNESS_STEP_DEMERIT`].
#[inline]
fn fitness_jump(prev_class: Option<u8>, class: u8) -> f64 {
    match prev_class {
        None => 0.0,
        Some(prev) => {
            let steps = prev.abs_diff(class) as f64;
            (steps - 1.0).max(0.0) * FITNESS_STEP_DEMERIT
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// Concatenate output lines with a single space, mirroring how the
    /// diagnostic renderer prints them — lets a test compare against the
    /// original prose without worrying about where the breaks landed.
    fn joined(out: &[Cow<'_, str>]) -> String {
        out.join(" ")
    }

    /// Re-borrow every segment into one `String`, for asserting content
    /// survival when `join` would insert spurious spaces (CJK / oversized
    /// splits break at non-space boundaries).  `concat` pre-sums the byte
    /// lengths and allocates once — the same `Borrow<str>` blanket impl
    /// `joined` relies on.
    fn concatenated(out: &[Cow<'_, str>]) -> String {
        out.concat()
    }

    /// The `SmallVec<[Word; 16]>` inline footprint depends on `Word` being
    /// 16 bytes: one more `usize`-shaped field silently returns it to 24 or
    /// 32 bytes, doubling the inline array back to 384–512 B and
    /// reintroducing the cache pressure the `u32` packing removes.
    #[test]
    fn word_struct_is_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Word>(), 16);
    }

    #[test]
    fn empty_input_yields_one_empty_segment() {
        assert_eq!(
            wrap_optimal("", 10, 4).as_slice(),
            [Cow::Borrowed("")].as_slice()
        );
    }

    // ── Last-line reserve (`research/locator-wrap/`) ───────────

    /// Greedy block count at `budget` (`L` of solution.md §2.2).
    fn greedy_lines(widths: &[usize], budget: usize) -> usize {
        let mut lines = 1;
        let mut cur = 0usize;
        for (t, &w) in widths.iter().enumerate() {
            if t == 0 {
                cur = w;
            } else if cur + 1 + w <= budget {
                cur += 1 + w;
            } else {
                lines += 1;
                cur = w;
            }
        }
        lines
    }

    /// `OPT = L(j* - 1) + 1` of solution.md §2.3, or `None` when infeasible.
    fn minimal_lines(widths: &[usize], max_width: usize, reserve: usize) -> Option<usize> {
        // No floor: a zero last-line budget is infeasible (§2.1).
        let budget_last = max_width.saturating_sub(reserve);
        if widths.iter().any(|&w| w > max_width) {
            return None;
        }
        let mut acc = *widths.last()?;
        let mut split = (acc <= budget_last).then_some(widths.len() - 1);
        for idx in (0..widths.len() - 1).rev() {
            acc += widths[idx] + 1;
            if acc <= budget_last {
                split = Some(idx);
            }
        }
        let j = split?;
        Some(if j == 0 {
            1
        } else {
            greedy_lines(&widths[..j], max_width) + 1
        })
    }

    /// The two-phase composition Φ that `wrap_optimal_reserved` replaces
    /// (wrap at the full measure, then re-wrap the overflowing tail at the
    /// narrowed measure).  Kept here as the regression baseline.
    fn two_phase_lines(text: &str, max_width: usize, tab_width: usize, reserve: usize) -> usize {
        let segs = wrap_optimal(text, max_width, tab_width);
        let narrow = max_width.saturating_sub(reserve).max(1);
        let mut lines = segs.len();
        // Count only: re-wrapping into a second buffer would need an owned tail
        // (`Cow`s from `wrap_optimal` borrow their input), which is exactly the
        // borrow dance the single-pass rewrite removes.
        if reserve < max_width
            && let Some(last) = segs.last()
            && measure_width(last.as_ref(), tab_width) + reserve > max_width
        {
            lines = lines - 1 + wrap_optimal(last.as_ref(), narrow, tab_width).len();
        }
        lines
    }

    /// Fact 2: the composition emits `M` rows where two suffice.
    #[test]
    fn reserved_last_line_beats_the_two_phase_composition() {
        // A = 2M+1 = 9, reserve = 2M = 8 (M = 4) => budget_last = 1.
        let out = wrap_optimal_reserved("a b c d", 9, 4, 8);
        assert_eq!(
            out.as_slice(),
            [Cow::Borrowed("a b c"), Cow::Borrowed("d")].as_slice()
        );
        // Φ's first phase emits one row; the second re-wraps it into M rows.
        assert_eq!(
            wrap_optimal("a b c d", 9, 4).as_slice(),
            [Cow::Borrowed("a b c d")].as_slice()
        );
        assert_eq!(two_phase_lines("a b c d", 9, 4, 8), 4);
        assert_eq!(out.len(), 2);
    }

    /// Exhaustive sweep: the reserve changes only the last row, the layout
    /// never beats the minimal-line model, never exceeds the composition it
    /// replaces, and the infeasible branch is exactly P1.
    #[test]
    fn reserved_last_line_invariants() {
        for n in 1..=5usize {
            for code in 0..3usize.pow(n as u32) {
                let mut msg = String::new();
                let mut widths = Vec::with_capacity(n);
                for t in 0..n {
                    if t > 0 {
                        msg.push(' ');
                    }
                    let w = (code / 3usize.pow(t as u32)) % 3 + 1;
                    widths.push(w);
                    msg.push_str(&"a".repeat(w));
                }
                for max_width in 3..=14usize {
                    for reserve in 0..=max_width {
                        let out = wrap_optimal_reserved(&msg, max_width, 4, reserve);
                        // A break consumes the space it breaks at, while an
                        // unbroken run keeps it, so compare word sequences.
                        let laid_out: Vec<&str> =
                            out.iter().flat_map(|s| s.split_whitespace()).collect();
                        let expected: Vec<&str> = msg.split_whitespace().collect();
                        assert_eq!(
                            laid_out, expected,
                            "content lost: {msg:?} A={max_width} s={reserve}"
                        );
                        let last = out.len() - 1;
                        match minimal_lines(&widths, max_width, reserve) {
                            Some(opt) => {
                                for (i, seg) in out.iter().enumerate() {
                                    let w = measure_width(seg.as_ref(), 4);
                                    if i == last {
                                        assert!(
                                            w + reserve <= max_width,
                                            "last row + reserve overflows: {msg:?} A={max_width} s={reserve} out={out:?}"
                                        );
                                    } else {
                                        assert!(
                                            w <= max_width,
                                            "non-last row overflows: {msg:?} A={max_width} out={out:?}"
                                        );
                                    }
                                }
                                assert!(
                                    out.len() >= opt,
                                    "beats the minimal-line model: {msg:?} A={max_width} s={reserve} out={out:?}"
                                );
                            }
                            None => assert_eq!(
                                out.as_slice(),
                                wrap_optimal(&msg, max_width, 4).as_slice(),
                                "infeasible must take P1 (full measure): {msg:?} A={max_width} s={reserve}"
                            ),
                        }
                        assert!(
                            out.len() <= two_phase_lines(&msg, max_width, 4, reserve),
                            "worse than the composition it replaces: {msg:?} A={max_width} s={reserve} out={out:?}"
                        );
                    }
                }
            }
        }
    }

    /// The reserve belongs to the last *real* paragraph only; paragraph line
    /// counts are additive (solution.md §1.4).
    #[test]
    fn reserved_paragraph_applies_only_to_the_last_paragraph() {
        for text in ["a b c\nd e f", "a b c\nd e f\n"] {
            let out = wrap_optimal_reserved(text, 8, 4, 6);
            assert_eq!(
                out.as_slice(),
                [
                    Cow::Borrowed("a b c"),
                    Cow::Borrowed("d e"),
                    Cow::Borrowed("f")
                ]
                .as_slice(),
                "text: {text:?}"
            );
        }
        // A zero reserve is exactly `wrap_optimal`.
        for text in ["", "a", "a b c d", "a\nb c", "a  b\n\nc"] {
            for width in [1usize, 4, 9, 40] {
                assert_eq!(
                    wrap_optimal_reserved(text, width, 4, 0).as_slice(),
                    wrap_optimal(text, width, 4).as_slice(),
                    "text: {text:?} width: {width}"
                );
            }
        }
    }

    #[test]
    fn whitespace_only_paragraph_memo_round_trips() {
        // `" \n"` produces a `Cow::Borrowed("")` whose pointer lives in
        // `.rodata`; the memo used to store a bogus offset and panic on a hit.
        let first = wrap_optimal(" \n", 80, 4);
        let second = wrap_optimal(" \n", 80, 4);
        assert!(first.iter().all(|segment| segment.is_empty()));
        assert_eq!(first, second);
    }

    #[test]
    fn non_ascii_whitespace_only_paragraph_survives() {
        let nbsp = "\u{00A0}";
        assert_eq!(
            wrap_optimal(nbsp, 80, 4).as_slice(),
            [Cow::Borrowed(nbsp)].as_slice()
        );
    }

    #[test]
    fn single_short_word_stays_on_one_line() {
        assert_eq!(
            wrap_optimal("hello", 80, 4).as_slice(),
            [Cow::Borrowed("hello")].as_slice()
        );
    }

    #[test]
    fn every_line_fits_and_text_survives() {
        let src = "the quick brown fox jumps over the lazy dog";
        let out = wrap_optimal(src, 20, 4);
        for line in &out {
            assert!(line.len() <= 20, "over-wide line: {line:?}");
        }
        assert_eq!(joined(&out), src);
    }

    #[test]
    fn code_tokens_are_never_split() {
        // Without the code-token filter, UAX #14 would happily break
        // `Int<32>` at `<` or `>`. The filter keeps it whole.
        let out = wrap_optimal("expected Int<32>, found Bool", 12, 4);
        assert!(
            out.iter().any(|l| l.contains("Int<32>")),
            "Int<32> was split: {out:?}"
        );
        // Likewise for a `Vec<T>`-shaped generic and a parenthesised tuple.
        let out = wrap_optimal("value of type Vec<Point> here", 12, 4);
        assert!(out.iter().any(|l| l.contains("Vec<Point>")), "{out:?}");
        let out = wrap_optimal("tuple (u32, bool) here", 10, 4);
        assert!(out.iter().any(|l| l.contains("(u32,")), "{out:?}");
    }

    #[test]
    fn cjk_breaks_between_ideographs() {
        // Ten 2-cell CJK chars, budget 6 cells → three per line.
        let src = "这是一段很长的中文文本";
        let out = wrap_optimal(src, 6, 4);
        let cells: Vec<usize> = out
            .iter()
            .map(|l| l.chars().map(|_| 2).sum::<usize>())
            .collect();
        for (i, w) in cells.iter().enumerate() {
            assert!(*w <= 6, "line {i} too wide: {w} cells ({:?})", out[i]);
        }
        assert_eq!(concatenated(&out), src);
    }

    #[test]
    fn hard_newlines_are_preserved() {
        assert_eq!(
            wrap_optimal("line one\nline two", 80, 4).as_slice(),
            ["line one", "line two"].as_slice()
        );
    }

    #[test]
    fn trailing_newline_does_not_create_empty_line() {
        assert_eq!(
            wrap_optimal("hello\n", 80, 4).as_slice(),
            ["hello"].as_slice()
        );
    }

    #[test]
    fn interior_blank_line_is_preserved() {
        assert_eq!(
            wrap_optimal("a\n\nb", 80, 4).as_slice(),
            ["a", "", "b"].as_slice()
        );
    }

    /// Every fast path must be output-identical to the DP path it
    /// short-circuits.  Reference: run `collect_words` + `wrap_dp` by hand on
    /// the same input (both are visible to this module) and compare.  The
    /// corpus covers the fast paths' boundary conditions — exact fit
    /// (`width == max_width`), one under, one over, internal runs of spaces,
    /// CJK, tabs (whose width is column-dependent and therefore not additive
    /// across pieces), a trailing `\n`, and the excluded shapes
    /// (leading/trailing whitespace, interior newline) to confirm agreement
    /// there too.
    ///
    /// NOTE: the reference here calls `collect_words`, which itself goes
    /// through `ascii_break_opportunities`, so this pins the short-circuit
    /// logic only — it does *not* prove the ASCII break opportunities agree
    /// with UAX #14.  That is `ascii_fast_path_breaks_match_uax`'s job.
    #[test]
    fn fast_paths_match_dp() {
        let corpus = [
            "hello",
            "a b c",
            "a  b",             // interior space run survives verbatim
            "hello\n",          // trailing newline dropped
            "中文字",           // CJK, width 6
            "Int<32> + Vec<T>", // code tokens, no split opportunity taken
            "  leading",
            "trailing  ",
            "a\nb",
            "the quick brown fox",
            "a\nb c\n\nd",
            "a\tb", // tab: width is column-dependent, must not take a fast path
            "",
            " ",
            "\t",
            "x\u{200B}y", // zero-width char: width 2
        ];
        let tab_width = 4;
        for text in corpus {
            for max_width in [0, 1, 2, 5, 6, 10, 11, 12, 80] {
                let out = wrap_optimal(text, max_width, tab_width);
                let got: Vec<&str> = out.iter().map(|c| &**c).collect();
                // Hand-run reference (the path the fast paths replace), with
                // the same paragraph splitting as `wrap_optimal`.
                let paras: Vec<&str> = text.split('\n').collect();
                let n_paras = if text.ends_with('\n') && paras.len() > 1 {
                    paras.len() - 1
                } else {
                    paras.len()
                };
                let mut want: Vec<&str> = Vec::new();
                if text.is_empty() {
                    want = vec![""]; // `wrap_optimal`'s early return
                } else if max_width == 0 {
                    // Same `split_terminator` semantics as `wrap_optimal`'s
                    // width-0 branch (trailing `\n` never adds a segment).
                    want = text.split_terminator('\n').collect();
                } else {
                    for &para in &paras[..n_paras] {
                        if para.is_empty() {
                            want.push("");
                            continue;
                        }
                        let words = collect_words(para, max_width, tab_width, None);
                        if words.is_empty() {
                            // Whitespace-only paragraph: `wrap_optimal`
                            // preserves it as an empty segment (layout, not
                            // noise); the DP has no words to lay out.
                            want.push("");
                            continue;
                        }
                        let mut scratch = WrapScratch::new();
                        for r in wrap_dp(
                            &words,
                            max_width,
                            wrap_tables(max_width).as_deref(),
                            &mut scratch,
                        ) {
                            want.push(&para[r]);
                        }
                    }
                }
                assert_eq!(got, want, "text={text:?} max_width={max_width}");
            }
        }
    }

    /// Width 0 must use the same `split_terminator` paragraph semantics as
    /// every other width: a trailing `\n` never creates a phantom empty
    /// segment (regression: the width-0 branch used `split('\n')`, so
    /// `"a\n"` yielded `["a", ""]` at width 0 but `["a"]` at width 80).
    #[test]
    fn zero_width_matches_split_terminator_semantics() {
        fn w(t: &str) -> Vec<&str> {
            // The width-0 branch only ever returns `Cow::Borrowed` segments
            // (verbatim paragraphs), so borrowing the input is exact.
            wrap_optimal(t, 0, 4)
                .into_iter()
                .map(|c| match c {
                    Cow::Borrowed(s) => s,
                    Cow::Owned(_) => panic!("width-0 path must borrow"),
                })
                .collect()
        }
        assert_eq!(w("a\n"), ["a"]);
        assert_eq!(w("\n"), [""]);
        assert_eq!(w("\n\n"), ["", ""]);
        assert_eq!(w("a\nb"), ["a", "b"]);
        assert_eq!(w(""), [""]);
    }

    /// Greedy fails when a run of short words lets it nearly fill an early
    /// line, leaving the *remaining* short words to be stranded next to a
    /// heavy trailing word.
    ///
    /// Input: eight 1-cell words + one 9-cell word, budget 10.
    ///
    ///   Greedy:  "x x x x x"  "x x x"      "LLLLLLLLL"
    ///            width 9      width 5      width 9
    ///            slack 1      slack 5      last line (free)
    ///            intrinsic (1/10)³ + (5/10)³ = 0.126
    ///            fitness Tight → Loose → Tight: two 2-class jumps,
    ///            i.e. 2 × 1/16 = 0.125
    ///            total 0.251
    ///
    ///   DP:      "x x x x"    "x x x x"    "LLLLLLLLL"
    ///            width 7      width 7      width 9
    ///            slack 3      slack 3      last line (free)
    ///            intrinsic (3/10)³ + (3/10)³ = 0.054
    ///            fitness Normal → Normal → Tight: one 1-class step, free
    ///            total 0.054
    ///
    /// Greedy is *not* locally bad — every line it produces fits.  It is
    /// globally bad: by squeezing a 5th short word onto line 1, it forces a
    /// 3-word line 2 whose slack (5) dwarfs line 1's slack (1).  DP trades
    /// line 1's near-perfect slack for two balanced lines, and wins 0.054 <
    /// 0.251 — both on intrinsic badness and on the adjacency term, since
    /// greedy's Tight→Loose→Tight zig-zag costs two fitness jumps that DP's
    /// flat Normal→Normal run avoids entirely.
    #[test]
    fn dp_beats_greedy_when_short_words_strand_a_heavy_tail() {
        let src = "x x x x x x x x LLLLLLLLL";
        let out = wrap_optimal(src, 10, 4);

        // DP spreads the short words evenly instead of packing the first line.
        assert_eq!(
            out.as_slice(),
            ["x x x x", "x x x x", "LLLLLLLLL"].as_slice()
        );

        // Sanity: the DP output really is more balanced than greedy's would be.
        let dp_widths: Vec<usize> = out.iter().map(|l| l.len()).collect();
        let greedy_widths = [9usize, 5, 9];
        let dp_max_slack = dp_widths[..dp_widths.len() - 1]
            .iter()
            .map(|&w| 10 - w)
            .max()
            .unwrap();
        let greedy_max_slack = greedy_widths[..greedy_widths.len() - 1]
            .iter()
            .map(|&w| 10 - w)
            .max()
            .unwrap();
        assert!(
            dp_max_slack < greedy_max_slack,
            "DP should shrink the worst slack: dp={dp_widths:?} greedy={greedy_widths:?}"
        );
    }

    #[test]
    fn oversized_word_is_split_at_grapheme_boundaries() {
        // 20-char ASCII word, budget 6: must be split into ≤6-wide chunks.
        let out = wrap_optimal("aaaaaaaaaaaaaaaaaaaa", 6, 4);
        for line in &out {
            assert!(line.len() <= 6, "over-wide chunk: {line:?}");
        }
        assert_eq!(concatenated(&out), "aaaaaaaaaaaaaaaaaaaa");
    }

    /// One class of difference between adjacent lines is free; only the step
    /// beyond it is charged.  At a measure of 10: Tight (0 empty),
    /// Normal (3 empty), Loose (5 empty), VeryLoose (9 empty).
    #[test]
    fn one_class_step_between_lines_is_free() {
        let m = 10;
        let tight = fitness_class(m, m);
        let normal = fitness_class(7, m);
        let loose = fitness_class(5, m);
        let very_loose = fitness_class(1, m);
        assert_eq!((tight, normal, loose, very_loose), (0, 1, 2, 3));

        assert_eq!(fitness_jump(Some(tight), normal), 0.0);
        assert_eq!(fitness_jump(Some(normal), tight), 0.0);
        assert_eq!(fitness_jump(Some(loose), normal), 0.0);
        assert_eq!(fitness_jump(Some(tight), loose), FITNESS_STEP_DEMERIT);
        assert_eq!(
            fitness_jump(Some(tight), very_loose),
            2.0 * FITNESS_STEP_DEMERIT
        );
        // A paragraph's first line has nothing above it.
        assert_eq!(fitness_jump(None, very_loose), 0.0);
    }

    /// The tabulated form the DP's inner loop reads must not drift away from
    /// the formula it stands for.
    #[test]
    fn jump_table_matches_fitness_jump() {
        for prev in 0..N_FITNESS {
            for cur in 0..N_FITNESS {
                assert_eq!(
                    JUMP[cur][prev],
                    fitness_jump(Some(prev as u8), cur as u8),
                    "JUMP[{cur}][{prev}]"
                );
            }
        }
    }

    /// The ASCII break fast path must be indistinguishable from
    /// `unicode_linebreak` + the code-token filter.  Every printable ASCII
    /// char is put on both deciding sides of a space run (immediately before
    /// and immediately after it), in three shapes, so LB13/LB14/LB15 are all
    /// exercised.  With U+0060 resolved as `AL` (see
    /// [`ascii_break_after_space`]), every printable-ASCII paragraph is
    /// decided by the fast path — none declines — so this also asserts
    /// `accepted == total`.
    #[test]
    fn ascii_fast_path_breaks_match_uax() {
        let ascii: Vec<u8> = (0x20..=0x7E).collect();
        let mut shapes: Vec<String> = Vec::with_capacity(3 * ascii.len() * ascii.len());
        for &a in &ascii {
            for &b in &ascii {
                shapes.push(format!("{} {}", a as char, b as char));
                shapes.push(format!("x{}  {}y", a as char, b as char));
                shapes.push(format!("{}x {} {}", a as char, b as char, b as char));
            }
        }
        shapes.extend(
            [
                "a b c",
                "one  two   three",
                "(a b)",
                "a . b",
                "a ( b",
                "x ` (y",
                "\" (z",
                "hi  ",
                "  hi",
                "(( x (",
            ]
            .iter()
            .map(|s| s.to_string()),
        );

        let total = shapes.len();
        let mut accepted = 0usize;
        for s in &shapes {
            let want = uax_break_opportunities(s);
            match ascii_break_opportunities(s, None) {
                Some(got) => {
                    accepted += 1;
                    assert_eq!(got, want, "break mismatch for {s:?}");
                }
                None => panic!("printable ASCII must not decline: {s:?}"),
            }
        }
        assert_eq!(
            accepted, total,
            "every printable-ASCII shape should be decided by the fast path"
        );

        // The grave accent is `AL`, so `` ` `` + spaces + opener does not
        // decline (LB15 does not fire).
        assert!(ascii_break_opportunities("x ` (y", None).is_some());
        assert!(ascii_break_opportunities("x ` [y", None).is_some());

        // Anything non-ASCII or with a tab must decline rather than guess.
        assert!(ascii_break_opportunities("a\tb", None).is_none());
        assert!(ascii_break_opportunities("a\u{3042}b", None).is_none());
        assert!(ascii_break_opportunities("a\u{200B}b", None).is_none());
    }

    /// `trimmed_len` must agree with the Unicode predicate it stands in for,
    /// including the multi-byte whitespace it deliberately hands back.
    #[test]
    fn trimmed_len_matches_trim_end_matches() {
        for s in [
            "a",
            "a ",
            "a  ",
            " a ",
            "",
            " ",
            "\t",
            "a\t",
            "a\u{A0}",
            "a\u{A0} ",
            "a\u{3000}",
        ] {
            assert_eq!(
                trimmed_len(s),
                s.trim_end_matches(char::is_whitespace).len(),
                "{s:?}"
            );
        }
    }

    /// The single-line badness cannot see this.  `"aaaaaaaa b c"` at a
    /// measure of 10 has a strictly better *intrinsic* layout when line 1 is
    /// packed (`"aaaaaaaa b"`, nothing empty) and the tail word is left alone
    /// (`"c"`) — but that puts a Tight line next to a VeryLoose one, three
    /// classes apart.  The jump costs more than the two columns it saves.
    ///
    ///   packed:  "aaaaaaaa b"   "c"
    ///            intrinsic 0.000, jump Tight→VeryLoose = 2 × 1/16 = 0.125
    ///            total 0.125
    ///
    ///   smooth:  "aaaaaaaa"     "b c"
    ///            intrinsic (2/10)³ = 0.008, jump Tight→Loose = 1 × 1/16 = 0.0625
    ///            total 0.071
    #[test]
    fn adjacent_lines_resist_a_jagged_edge() {
        let src = "aaaaaaaa b c";
        assert_eq!(
            wrap_optimal(src, 10, 4).as_slice(),
            ["aaaaaaaa", "b c"].as_slice()
        );

        // The packed layout really is intrinsically better — the flip comes
        // from the adjacency term alone.
        let packed = ["aaaaaaaa b".to_string(), "c".to_string()];
        let smooth = ["aaaaaaaa".to_string(), "b c".to_string()];
        assert!(intrinsic_total(&packed, 10, 4) < intrinsic_total(&smooth, 10, 4));
        assert!(total_badness(&smooth, 10, 4) < total_badness(&packed, 10, 4));
    }

    /// The overflow term is in absolute columns while every soft term is
    /// normalised to ≤ ~1.13 per line, so even the most jarring legal layout
    /// costs less than one over-wide column.
    #[test]
    fn overflow_still_beats_any_raggedness() {
        let m = 10;
        let worst_legal = badness(None, 1, m, false)
            + badness(Some(fitness_class(1, m)), 1, m, false)
            + badness(Some(fitness_class(1, m)), 1, m, true);
        assert!(badness(None, m + 1, m, false) > worst_legal);
    }

    /// Differential property test: DP badness never exceeds greedy badness on
    /// random inputs.  Words are kept strictly narrower than `max_width` so
    /// `split_oversized` never fires — the comparison then stays on the
    /// legal-break path where both algorithms are defined identically.
    #[test]
    fn dp_never_worse_than_greedy_on_random_inputs() {
        // Deterministic LCG so failures reproduce without a seed log.
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };
        for _ in 0..2000 {
            let max_width = 6 + (next() % 12) as usize; // 6..=17
            let n = 3 + (next() % 12) as usize; // 3..=14 words
            // Word widths are 1..=max_width-1 so no single word is over-wide:
            // `split_oversized` stays dormant and `join(" ")` recovers `src`.
            let mut words: Vec<String> = Vec::with_capacity(n);
            for _ in 0..n {
                let w = 1 + (next() % (max_width as u64 - 1)) as usize;
                words.push("a".repeat(w));
            }
            let src = words.join(" ");
            let dp: Vec<String> = wrap_optimal(&src, max_width, 4)
                .into_iter()
                .map(|c| c.into_owned())
                .collect();

            // Reference greedy: pack as many words as fit.
            let mut greedy: Vec<String> = Vec::new();
            let mut line = String::new();
            for word in src.split(' ') {
                let candidate = if line.is_empty() {
                    word.len()
                } else {
                    line.len() + 1 + word.len()
                };
                if candidate <= max_width && !line.is_empty() {
                    line.push(' ');
                    line.push_str(word);
                } else if line.is_empty() {
                    line.push_str(word);
                } else {
                    greedy.push(std::mem::take(&mut line));
                    line.push_str(word);
                }
            }
            if !line.is_empty() {
                greedy.push(line);
            }

            assert!(
                total_badness(&dp, max_width, 4) <= total_badness(&greedy, max_width, 4) + 1e-9,
                "DP worse than greedy on {src:?} @ {max_width}:\n  dp={dp:?}\n  greedy={greedy:?}",
            );
            assert_eq!(dp.join(" "), src, "text mangled for {src:?}");
        }
    }

    /// Total cost of `lines`, exactly as `wrap_dp` scores a candidate: each
    /// line's intrinsic badness plus the fitness jump from its predecessor.
    /// Delegating to `badness` / `fitness_class` keeps the reference from
    /// drifting away from the implementation the way a re-derived formula
    /// would.  Width is measured with [`measure_width`] so CJK / emoji count
    /// by display cells, not by `char`s.
    fn total_badness(lines: &[String], max_width: usize, tab_width: usize) -> f64 {
        let mut prev: Option<u8> = None;
        let mut total = 0.0;
        for (idx, line) in lines.iter().enumerate() {
            let width = measure_width(line, tab_width);
            total += badness(prev, width, max_width, idx + 1 == lines.len());
            prev = Some(fitness_class(width, max_width));
        }
        total
    }

    /// Sum of each line's intrinsic badness only — what the pre-adjacency DP
    /// minimised, kept so tests can show the flip comes from adjacency alone.
    fn intrinsic_total(lines: &[String], max_width: usize, tab_width: usize) -> f64 {
        let inv_measure = 1.0 / (max_width.max(1) as f64);
        lines
            .iter()
            .enumerate()
            .map(|(idx, l)| {
                intrinsic_badness(
                    measure_width(l, tab_width),
                    max_width,
                    inv_measure,
                    idx + 1 == lines.len(),
                )
            })
            .sum()
    }
}
