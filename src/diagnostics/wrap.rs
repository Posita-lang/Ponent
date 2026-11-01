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
//! [`GlyphRenderer`]: crate::diagnostics::GlyphRenderer
//! [`explain_error_code`]: crate::diagnostics::explain_error_code

use super::ascii::is_printable_ascii;
use super::glyph::grapheme_width;
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
    if text.is_empty() {
        return smallvec![Cow::Borrowed("")];
    }
    if max_width == 0 {
        return text.split('\n').map(Cow::Borrowed).collect();
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
    // edges is rejected by [`edges_have_non_ascii_ws`]: leading NBSP is LB12
    // glue the DP keeps inside the first word while trailing NBSP is stripped
    // by [`trimmed_len`]'s multi-byte fallback, and a leading NBSP run folded
    // with ASCII spaces is dropped as a whitespace-only chunk — in every case
    // a fast path emitting `trim_matches(' ')` would diverge, so we fall
    // through to the DP.
    //
    // Tabs are excluded because their width is column-dependent and not
    // additive across the word/glue pieces the DP sums.  `measure_width` is
    // SWAR for printable ASCII and a cached grapheme walk otherwise — both
    // far cheaper than UAX #14 + DP — so this gate covers short non-English
    // (CJK, kana, …) messages, not just ASCII.
    let single = text.strip_suffix('\n').unwrap_or(text);
    let no_interior_newline = !single.contains('\n');
    if no_interior_newline {
        let trimmed = single.trim_matches(' ');
        if !trimmed.is_empty()
            && memchr::memchr(b'\t', trimmed.as_bytes()).is_none()
            && !edges_have_non_ascii_ws(trimmed)
            && measure_width(trimmed, tab_width) <= max_width
        {
            return smallvec![Cow::Borrowed(trimmed)];
        }
    }

    // Paragraph count, without materialising the `split` items: a trailing
    // '\n' leaves a phantom empty final paragraph, dropped so "a\n" → ["a"]
    // and not ["a", ""].  When the fast-path probe above found no interior
    // newline, there is exactly one paragraph and the `memchr_iter` count
    // pass is redundant — reuse the probe result instead.
    let n_paras = if no_interior_newline {
        1
    } else {
        let n_newlines = memchr::memchr_iter(b'\n', text.as_bytes()).count();
        if text.ends_with('\n') {
            n_newlines
        } else {
            n_newlines + 1
        }
    };

    let mut out: SmallVec<[Cow<'a, str>; 4]> = SmallVec::new();
    for para in text.split('\n').take(n_paras) {
        if para.is_empty() {
            out.push(Cow::Borrowed(""));
            continue;
        }
        // Whole paragraph's core fits `max_width` ⇒ the DP emits it as one
        // line (zero badness; any split's first line is strictly short of the
        // measure, so it costs > 0).  See the top-of-function fast path for
        // the trimming / tab / non-ASCII-whitespace rationale.
        let trimmed = para.trim_matches(' ');
        if !trimmed.is_empty()
            && memchr::memchr(b'\t', trimmed.as_bytes()).is_none()
            && !edges_have_non_ascii_ws(trimmed)
            && measure_width(trimmed, tab_width) <= max_width
        {
            out.push(Cow::Borrowed(trimmed));
            continue;
        }
        let words = collect_words(para, max_width, tab_width);
        if words.is_empty() {
            continue;
        }
        for range in wrap_dp(&words, max_width) {
            out.push(Cow::Borrowed(&para[range]));
        }
    }
    out
}

// ── Word collection ────────────────────────────────────────────

/// A single atomic word, plus the run of whitespace that follows it.
///
/// `trailing_space` is the glue Knuth-Plass stretches/shrinks: it is *included*
/// between words on the same line and *dropped* at a line break.
struct Word {
    /// Byte range inside the paragraph (trailing whitespace excluded).
    range: Range<usize>,
    /// Display width of `range` in terminal cells.
    width: usize,
    /// Display width of the whitespace following `range`.
    trailing_space: usize,
}

/// Split `text` into words at the break opportunities of [`break_opportunities`],
/// splitting any single word wider than `max_width` at grapheme boundaries.
///
/// Returns a [`SmallVec`] because a paragraph almost always has ≤16 words; the
/// `Word` struct is 32 B so `[Word; 16]` fits inline (512 B) and the common
/// case skips the heap.
fn collect_words(text: &str, max_width: usize, tab_width: usize) -> SmallVec<[Word; 16]> {
    let mut words: SmallVec<[Word; 16]> = SmallVec::new();
    push_words(
        text,
        &break_opportunities(text),
        max_width,
        tab_width,
        &mut words,
    );
    words
}

/// Byte offsets at which `text` may be broken, always `[0, …, text.len()]`.
fn break_opportunities(text: &str) -> SmallVec<[usize; 16]> {
    if let Some(breaks) = ascii_break_opportunities(text) {
        return breaks;
    }
    uax_break_opportunities(text)
}

/// The reference implementation: UAX #14 opportunities filtered by
/// [`should_break_at`] (the code-token filter).
fn uax_break_opportunities(text: &str) -> SmallVec<[usize; 16]> {
    // `linebreaks` yields the byte offset of the first character after the
    // break; `should_break_at` filters out positions that would split a code
    // token.
    let mut breaks: SmallVec<[usize; 16]> = smallvec![0];
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
    breaks
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
fn ascii_break_opportunities(text: &str) -> Option<SmallVec<[usize; 16]>> {
    let b = text.as_bytes();
    if !is_printable_ascii(b) {
        return None;
    }
    let mut breaks: SmallVec<[usize; 16]> = smallvec![0];
    let mut i = 0usize;
    while i < b.len() {
        if b[i] != b' ' {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < b.len() && b[i] == b' ' {
            i += 1;
        }
        // `i` is the end of a maximal space run — the only candidate position
        // printable ASCII can offer.
        if i < b.len() {
            let prev = if run_start == 0 {
                None
            } else {
                Some(b[run_start - 1])
            };
            match ascii_break_after_space(prev, b[i]) {
                Some(true) => breaks.push(i),
                Some(false) => {}
                // Undecidable from ASCII alone: hand the paragraph back to
                // UAX #14 instead of guessing.
                None => return None,
            }
        }
    }
    if *breaks.last().unwrap() < b.len() {
        breaks.push(b.len());
    }
    Some(breaks)
}

/// May a line break at the end of a space run?  `prev` is the byte before the
/// run (`None` at the start of the paragraph), `next` the byte after it.
///
/// Returns `None` when the answer depends on an ASCII line-break class we do
/// not pin down; the caller then defers to UAX #14.
#[inline]
fn ascii_break_after_space(prev: Option<u8>, next: u8) -> Option<bool> {
    // LB13: `× CL | CP | EX | IS | SY` — never break before a closer, an
    // exclamation/interrogation, an infix separator or a symbol, *even after
    // spaces*: `}` `)` `]` `!` `?` `,` `.` `:` `;` `/`.
    if matches!(
        next,
        b'}' | b')' | b']' | b'!' | b'?' | b',' | b'.' | b':' | b';' | b'/'
    ) {
        return Some(false);
    }
    // LB14: `OP SP* ×` — never break after an opening bracket plus spaces.
    if matches!(prev, Some(b'(') | Some(b'[') | Some(b'{')) {
        return Some(false);
    }
    // LB15: `QU SP* × OP` — never break after a quote plus spaces into an
    // opener.  `"` and `'` are `QU`; the grave accent (U+0060) is `AL` in
    // the Unicode version `unicode-linebreak` 0.1.5 targets (and has been
    // `AL` since Unicode 14), so it does not trigger LB15 and a break after
    // its spaces is allowed.  (LB16 `CP SP* × NS` and LB17 `B2 SP* × B2`
    // have no ASCII members on either side.)
    if matches!(next, b'(' | b'[' | b'{') {
        return match prev {
            Some(b'"') | Some(b'\'') => Some(false),
            _ => Some(true),
        };
    }
    Some(true)
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

/// Build the words between `breaks` — shared by the ASCII and UAX paths so the
/// chunk→word rules (trailing-whitespace folding, oversized splitting) exist
/// once.
fn push_words(
    text: &str,
    breaks: &[usize],
    max_width: usize,
    tab_width: usize,
    words: &mut SmallVec<[Word; 16]>,
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
                prev.trailing_space += measure_width(chunk, tab_width);
            }
            continue;
        }

        let content = &text[chunk_start..content_end];
        let w_content = measure_width(content, tab_width);
        let space = measure_width(&text[content_end..chunk_end], tab_width);

        if w_content > max_width {
            // Unavoidably wider than the target: split at grapheme boundaries
            // so combining marks / emoji stay intact.
            push_split_oversized(words, content, chunk_start, max_width, tab_width, space);
        } else {
            words.push(Word {
                range: chunk_start..content_end,
                width: w_content,
                trailing_space: space,
            });
        }
    }
}

/// Append the grapheme-boundary pieces of an over-wide `text` (at byte offset
/// `offset` inside the paragraph, so ranges stay paragraph-relative) straight
/// into `words` — no intermediate `Vec`.  `trailing` is the glue that follows
/// the word and belongs to its last piece.
fn push_split_oversized(
    words: &mut SmallVec<[Word; 16]>,
    text: &str,
    offset: usize,
    max_width: usize,
    tab_width: usize,
    trailing: usize,
) {
    let mut sub_start = 0usize;
    let mut sub_width = 0usize;
    for (i, g) in text.grapheme_indices(true) {
        let gw = grapheme_width(g, sub_width, tab_width);
        if sub_width > 0 && sub_width + gw > max_width {
            words.push(Word {
                range: offset + sub_start..offset + i,
                width: sub_width,
                trailing_space: 0,
            });
            sub_start = i;
            sub_width = 0;
        }
        sub_width += gw;
    }
    if sub_start < text.len() {
        words.push(Word {
            range: offset + sub_start..offset + text.len(),
            width: sub_width,
            trailing_space: trailing,
        });
    }
}

// ── Display width ──────────────────────────────────────────────

/// Display width of `s` in terminal cells.
#[inline]
fn measure_width(s: &str, tab_width: usize) -> usize {
    // Fast path: printable ASCII is one cell per byte, found without ever
    // building a grapheme iterator.  Mirrors the fast paths in
    // `line_display_width` / `visible_width`.
    if is_printable_ascii(s.as_bytes()) {
        return s.len();
    }
    let mut w = 0;
    for g in s.graphemes(true) {
        w += grapheme_width(g, w, tab_width);
    }
    w
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
            // Something multi-byte behind the run: let the exact predicate
            // decide.
            _ => return s.trim_end_matches(char::is_whitespace).len(),
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
fn edges_have_non_ascii_ws(s: &str) -> bool {
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
fn wrap_dp(words: &[Word], max_width: usize) -> SmallVec<[Range<usize>; 4]> {
    let n = words.len();
    if n == 0 {
        return SmallVec::new();
    }
    if n == 1 {
        // One word, one line: exactly what the loop below computes, minus the
        // bookkeeping.
        return smallvec![words[0].range.clone()];
    }

    // Prefix sums over widths and inter-word spaces, so line width for any
    // [i, j) slice is O(1).
    let mut pw: SmallVec<[usize; 32]> = smallvec![0usize; n + 1];
    let mut ps: SmallVec<[usize; 32]> = smallvec![0usize; n + 1];
    for (i, w) in words.iter().enumerate() {
        pw[i + 1] = pw[i] + w.width;
        ps[i + 1] = ps[i] + w.trailing_space;
    }

    // Hoisted once for the whole DP: `intrinsic_badness` would otherwise
    // recompute the `max` + `as f64` cast on every candidate line.
    let measure = max_width.max(1) as f64;

    let n_states = (n + 1) * N_FITNESS;
    let mut dp: SmallVec<[f64; 128]> = smallvec![f64::INFINITY; n_states];
    let mut prev: SmallVec<[u32; 128]> = smallvec![u32::MAX; n_states];
    // `j == 0` is the empty prefix: nothing sits above the first line, so all
    // entry classes are free.
    for c in 0..N_FITNESS {
        dp[c] = 0.0;
    }

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
    let mut best_cost: SmallVec<[[f64; N_FITNESS]; 32]> =
        smallvec![[f64::INFINITY; N_FITNESS]; n + 1];
    let mut best_src: SmallVec<[[u32; N_FITNESS]; 32]> = smallvec![[0; N_FITNESS]; n + 1];
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
        let mut lw_lo = line_width(&pw, &ps, i_lo, j);
        while i_lo + 1 < j && lw_lo > max_width {
            i_lo += 1;
            lw_lo = line_width(&pw, &ps, i_lo, j);
        }
        // First candidate reuses `lw_lo` from the window advance above,
        // avoiding a redundant `line_width` call per `j`.
        {
            let class = fitness_class(lw_lo, max_width) as usize;
            let base = intrinsic_badness(lw_lo, max_width, measure, is_last_line);
            let s = j * N_FITNESS + class;
            let cost = best_cost[i_lo][class] + base;
            if cost < dp[s] {
                dp[s] = cost;
                prev[s] = best_src[i_lo][class];
            }
        }
        for i in (i_lo + 1)..j {
            // Width of the line covering words[i..j): word widths, plus the
            // inter-word spaces, minus the *trailing* space of the last word
            // (dropped at the break).
            let lw = line_width(&pw, &ps, i, j);
            let class = fitness_class(lw, max_width) as usize;
            let base = intrinsic_badness(lw, max_width, measure, is_last_line);
            let s = j * N_FITNESS + class;
            // `best_cost[i][class]` already folds the `min_pc` over the
            // predecessor's fitness class; `base` is independent of `pc`.
            // `∞ + base < ∞` is false, so unreachable states are skipped
            // without an explicit check and the loop stays branch-free.
            let cost = best_cost[i][class] + base;
            if cost < dp[s] {
                dp[s] = cost;
                prev[s] = best_src[i][class];
            }
        }
        // `dp[j]` is now final — no future iteration writes it.  Precompute
        // the min-plus convolution for outer iterations `j' > j` that will
        // use `j` as a predecessor.  Skip `j == n`: its `best_prev` is never
        // read (there is no `j' > n`).
        if j < n {
            let j0 = j * N_FITNESS;
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
        let j = s / N_FITNESS;
        let i = (prev[s] as usize) / N_FITNESS;
        ranges.push(words[i].range.start..words[j - 1].range.end);
        if i == 0 {
            break;
        }
        s = prev[s] as usize;
    }
    ranges.reverse();
    ranges
}

/// Width of the line covering `words[i..j)`: the words' widths plus the glue
/// between them — every `trailing_space` except the last word's, which is
/// dropped at the break.
#[inline]
fn line_width(pw: &[usize], ps: &[usize], i: usize, j: usize) -> usize {
    (pw[j] - pw[i]) + (ps[j - 1] - ps[i])
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
    let measure = max_width.max(1) as f64;
    intrinsic_badness(line_width, max_width, measure, is_last_line)
        + fitness_jump(prev_class, fitness_class(line_width, max_width))
}

/// Single-line (pre-adjacency) badness: cubed slack, zero on the last line,
/// cubic overflow with a large multiplier otherwise.
///
/// `measure` is `max_width.max(1) as f64`, hoisted by the caller so the DP's
/// inner loop does not recompute the `max` + cast per candidate.  Passing the
/// reciprocal would let the slack ratio become a multiply, but the cubed slack
/// still needs a divide-free form only if the whole expression is tabled; the
/// hoist alone removes the redundant `max`/cast from the hottest path.
#[inline]
fn intrinsic_badness(line_width: usize, max_width: usize, measure: f64, is_last_line: bool) -> f64 {
    if line_width <= max_width {
        if is_last_line {
            0.0
        } else {
            let slack = (max_width - line_width) as f64 / measure;
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
/// boundaries.  `empty` and `measure` are at most a few thousand, so
/// `4 * empty` cannot overflow `usize`.  The `max(1)` guard keeps `measure`
/// nonzero and preserves the original `max_width == 0` behaviour.
#[inline]
fn fitness_class(line_width: usize, max_width: usize) -> u8 {
    let measure = max_width.max(1);
    let empty = max_width.saturating_sub(line_width);
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

    #[test]
    fn empty_input_yields_one_empty_segment() {
        assert_eq!(
            wrap_optimal("", 10, 4).as_slice(),
            [Cow::Borrowed("")].as_slice()
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
                    want = text.split('\n').collect();
                } else {
                    for &para in &paras[..n_paras] {
                        if para.is_empty() {
                            want.push("");
                            continue;
                        }
                        let words = collect_words(para, max_width, tab_width);
                        for r in wrap_dp(&words, max_width) {
                            want.push(&para[r]);
                        }
                    }
                }
                assert_eq!(got, want, "text={text:?} max_width={max_width}");
            }
        }
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
            match ascii_break_opportunities(s) {
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
        assert!(ascii_break_opportunities("x ` (y").is_some());
        assert!(ascii_break_opportunities("x ` [y").is_some());

        // Anything non-ASCII or with a tab must decline rather than guess.
        assert!(ascii_break_opportunities("a\tb").is_none());
        assert!(ascii_break_opportunities("a\u{3042}b").is_none());
        assert!(ascii_break_opportunities("a\u{200B}b").is_none());
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
        let measure = max_width.max(1) as f64;
        lines
            .iter()
            .enumerate()
            .map(|(idx, l)| {
                intrinsic_badness(
                    measure_width(l, tab_width),
                    max_width,
                    measure,
                    idx + 1 == lines.len(),
                )
            })
            .sum()
    }
}
