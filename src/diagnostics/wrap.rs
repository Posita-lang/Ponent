//! Knuth-Plass optimal line breaking with Unicode UAX #14 break opportunities.
//!
//! Replaces the greedy `textwrap::wrap` used by `GlyphRenderer::wrap_msg` and
//! `explain_error_code` with a total-cost minimiser:
//!
//! - **Break opportunities** come from `unicode_linebreak` (UAX #14), then are
//!   filtered so code tokens (`Int<32>`, `Vec<T>`, `(x, y)`, `foo-bar`) are
//!   never split across lines. CJK↔CJK and whitespace boundaries survive.
//! - **Break selection** is a DP over words minimising the sum of cubed slack
//!   (`(max_width - line_width)^3`). The last line is free, so the paragraph
//!   is not stretched just to fill it.
//!
//! Hard newlines (`\n`) in the input are preserved as paragraph boundaries.
//! Every output segment borrows the input (`Cow::Borrowed`) — no allocation
//! unless an over-wide word has to be split at a grapheme boundary.
//!
//! [`GlyphRenderer`]: crate::diagnostics::GlyphRenderer
//! [`explain_error_code`]: crate::diagnostics::explain_error_code

use super::glyph::grapheme_width;
use std::borrow::Cow;
use std::ops::Range;
use unicode_linebreak::{BreakOpportunity, linebreaks};
use unicode_segmentation::UnicodeSegmentation;

/// Overflow penalty multiplier. Large enough that a mildly over-wide line
/// beats an almost-empty one, small enough that overflowing is never preferred
/// when a legal break exists.
const OVERFLOW_PENALTY: f64 = 1000.0;

/// Optimal-wrap `text` to at most `max_width` display columns per line.
///
/// Hard newlines in `text` are kept: each `\n`-separated paragraph is wrapped
/// independently. A trailing `\n` does not produce an extra empty line, so
/// `wrap_optimal("a\n", w)` == `["a"]`, matching `textwrap::wrap`.
pub(crate) fn wrap_optimal<'a>(text: &'a str, max_width: usize, tab_width: usize) -> Vec<Cow<'a, str>> {
    if text.is_empty() {
        return vec![Cow::Borrowed("")];
    }
    if max_width == 0 {
        return text.split('\n').map(Cow::Borrowed).collect();
    }

    // Single-line fast path: text with no hard newline (one optional trailing
    // `\n`), no leading/trailing whitespace, and a display width within
    // `max_width` is emitted verbatim by the DP below — one line, zero
    // badness, and the DP's strict `<` keeps that line over any zero-cost tie
    // (`wrap_dp` evaluates the single-line candidate first).  Leading and
    // trailing whitespace are excluded because the DP drops them (the
    // whitespace-only first/last chunk folds into glue / the dropped trailing
    // space), so a verbatim borrow would not match.  Skips the UAX #14
    // segmentation and the O(n²) DP — the common shape for label messages.
    let single = text.strip_suffix('\n').unwrap_or(text);
    if !single.is_empty()
        && !single.contains('\n')
        && !single.starts_with(char::is_whitespace)
        && !single.ends_with(char::is_whitespace)
        && measure_width(single, tab_width) <= max_width
    {
        return vec![Cow::Borrowed(single)];
    }

    let paras: Vec<&'a str> = text.split('\n').collect();
    // A trailing '\n' leaves a phantom empty final paragraph; drop it so
    // "a\n" → ["a"] and not ["a", ""].
    let n_paras = if text.ends_with('\n') && paras.len() > 1 {
        paras.len() - 1
    } else {
        paras.len()
    };

    let mut out: Vec<Cow<'a, str>> = Vec::new();
    for &para in &paras[..n_paras] {
        if para.is_empty() {
            out.push(Cow::Borrowed(""));
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

/// Split `text` into words at UAX #14 break opportunities that pass the
/// code-token filter, splitting any single word wider than `max_width` at
/// grapheme boundaries.
fn collect_words(text: &str, max_width: usize, tab_width: usize) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();

    // Collect the breakpoints we accept. `linebreaks` yields the byte offset
    // of the first character after the break; `should_break_at` filters out
    // positions that would split a code token.
    let mut breaks: Vec<usize> = vec![0];
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

    for w in breaks.windows(2) {
        let chunk_start = w[0];
        let chunk_end = w[1];
        if chunk_start >= chunk_end {
            continue;
        }

        // Trim trailing whitespace off the chunk; what remains is the word.
        let chunk = &text[chunk_start..chunk_end];
        let content_len = chunk.trim_end_matches(char::is_whitespace).len();
        let content_end = chunk_start + content_len;

        if content_end == chunk_start {
            // Whitespace-only chunk: fold it into the previous word's glue.
            if let Some(prev) = words.last_mut() {
                prev.trailing_space += measure_width(&text[chunk_start..chunk_end], tab_width);
            }
            continue;
        }

        let content = &text[chunk_start..content_end];
        let w_content = measure_width(content, tab_width);
        let space = measure_width(&text[content_end..chunk_end], tab_width);

        if w_content > max_width {
            // Unavoidably wider than the target: split at grapheme boundaries
            // so combining marks / emoji stay intact.
            let sub = split_oversized(content, chunk_start, max_width, tab_width);
            let n = sub.len();
            for (i, mut sw) in sub.into_iter().enumerate() {
                if i == n - 1 {
                    sw.trailing_space = space;
                }
                words.push(sw);
            }
        } else {
            words.push(Word {
                range: chunk_start..content_end,
                width: w_content,
                trailing_space: space,
            });
        }
    }

    words
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
    let prev = text[..pos].chars().next_back().unwrap();
    let next = text[pos..].chars().next().unwrap();

    if prev.is_whitespace() || next.is_whitespace() {
        return true;
    }
    if is_cjk(prev) && is_cjk(next) {
        return true;
    }
    false
}

/// CJK ideographs, kana, Hangul, and CJK punctuation / fullwidth forms.
/// The ranges UAX #14 treats as class `ID` (plus a few `CJ` variants) — these
/// are the characters that may legitimately break between themselves.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
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

/// Split a word whose width exceeds `max_width` at grapheme boundaries.
/// `offset` is the word's byte offset inside the paragraph, so the returned
/// ranges are paragraph-relative.
fn split_oversized(text: &str, offset: usize, max_width: usize, tab_width: usize) -> Vec<Word> {
    let mut out = Vec::new();
    let mut sub_start = 0usize;
    let mut sub_width = 0usize;
    for (i, g) in text.grapheme_indices(true) {
        let gw = grapheme_width(g, sub_width, tab_width);
        if sub_width > 0 && sub_width + gw > max_width {
            out.push(Word {
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
        out.push(Word {
            range: offset + sub_start..offset + text.len(),
            width: sub_width,
            trailing_space: 0,
        });
    }
    out
}

// ── Display width ──────────────────────────────────────────────

fn measure_width(s: &str, tab_width: usize) -> usize {
    // Fast path: every byte printable ASCII (0x20..=0x7E). No tabs, no
    // control chars, no multi-byte clusters — each byte contributes exactly
    // one cell (`grapheme_width`'s single-byte arm), so the byte length is
    // the display width and the grapheme iterator is never constructed.
    // Mirrors the fast paths in `line_display_width` / `visible_width`.
    if s.bytes().all(|b| (0x20..0x7F).contains(&b)) {
        return s.len();
    }
    let mut w = 0;
    for g in s.graphemes(true) {
        w += grapheme_width(g, w, tab_width);
    }
    w
}

// ── Dynamic program ────────────────────────────────────────────

/// Choose the line breaks for `words` (one paragraph) minimising total badness.
///
/// `dp[j]` is the minimum cost to lay out `words[0..j]`; `prev[j]` records
/// the index where the last line of that layout starts, enabling backward
/// reconstruction of the break sequence. O(n·k) time (k = words that fit ~two
/// lines), O(n) space.
fn wrap_dp(words: &[Word], max_width: usize) -> Vec<Range<usize>> {
    let n = words.len();
    if n == 0 {
        return Vec::new();
    }

    // Prefix sums over widths and inter-word spaces, so line width for any
    // [i, j) slice is O(1).
    let mut pw = vec![0usize; n + 1];
    let mut ps = vec![0usize; n + 1];
    for (i, w) in words.iter().enumerate() {
        pw[i + 1] = pw[i] + w.width;
        ps[i + 1] = ps[i] + w.trailing_space;
    }

    let mut dp = vec![f64::INFINITY; n + 1];
    let mut prev = vec![0usize; n + 1];
    dp[0] = 0.0;

    // A line more than twice `max_width` over-wide is never chosen: its
    // overflow penalty is cubic in the excess and scaled by
    // `OVERFLOW_PENALTY`, so it always loses to any legal break.  Skipping
    // those starts bounds the inner loop.
    let width_bound = max_width.saturating_mul(2);

    for j in 1..=n {
        let is_last_line = j == n;
        // `line_width(i)` is non-increasing in `i` (both prefix-sum terms
        // shrink as `i` grows), so the over-wide starts form a prefix.  Find
        // the first start within the bound by binary search, then DP over the
        // rest — usually only the few words that fit roughly two lines,
        // giving O(k) per `j` rather than O(j).
        //
        // Start `j - 1` is always kept even if a single word alone exceeds
        // the bound, so `dp[j]` is set at least once and is finite for every
        // `j` — the old `dp[i].is_infinite()` guard was unreachable.
        let mut lo = 0usize;
        let mut hi = j;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let lw = (pw[j] - pw[mid]) + (ps[j - 1] - ps[mid]);
            if lw <= width_bound {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let i_lo = if lo == j { j - 1 } else { lo };
        for i in i_lo..j {
            // Width of the line covering words[i..j]: word widths, plus the
            // inter-word spaces, minus the *trailing* space of the last word
            // (dropped at the break).
            let line_width = (pw[j] - pw[i]) + (ps[j - 1] - ps[i]);
            let badness = badness(line_width, max_width, is_last_line);
            let cost = dp[i] + badness;
            if cost < dp[j] {
                dp[j] = cost;
                prev[j] = i;
            }
        }
    }

    // Reconstruct breaks backward, then reverse into source order.
    let mut ranges = Vec::new();
    let mut j = n;
    while j > 0 {
        let i = prev[j];
        ranges.push(words[i].range.start..words[j - 1].range.end);
        j = i;
    }
    ranges.reverse();
    ranges
}

/// Line badness. Cubed slack on regular lines (Knuth-Plass' tetrahedral cost);
/// zero on the last line so a paragraph is never stretched to fill it; cubic
/// overflow with a large multiplier otherwise, so an over-full line is only
/// chosen when no legal break exists.
#[inline]
fn badness(line_width: usize, max_width: usize, is_last_line: bool) -> f64 {
    if line_width <= max_width {
        if is_last_line {
            0.0
        } else {
            let slack = (max_width - line_width) as f64;
            slack * slack * slack
        }
    } else {
        let over = (line_width - max_width) as f64;
        OVERFLOW_PENALTY * over * over * over
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
        assert_eq!(wrap_optimal("", 10, 4), vec![Cow::Borrowed("")]);
    }

    #[test]
    fn single_short_word_stays_on_one_line() {
        assert_eq!(wrap_optimal("hello", 80, 4), vec![Cow::Borrowed("hello")]);
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
        assert_eq!(wrap_optimal("line one\nline two", 80, 4), vec!["line one", "line two"]);
    }

    #[test]
    fn trailing_newline_does_not_create_empty_line() {
        assert_eq!(wrap_optimal("hello\n", 80, 4), vec!["hello"]);
    }

    #[test]
    fn interior_blank_line_is_preserved() {
        assert_eq!(wrap_optimal("a\n\nb", 80, 4), vec!["a", "", "b"]);
    }

    /// The single-line fast path must be output-identical to the DP path it
    /// short-circuits.  Reference: run `collect_words` + `wrap_dp` by hand on
    /// the same input (both are visible to this module) and compare.  The
    /// corpus covers the fast path's boundary conditions — exact fit
    /// (`width == max_width`), one under, one over, internal runs of spaces,
    /// CJK, a trailing `\n`, and the excluded shapes (leading/trailing
    /// whitespace, interior newline) to confirm agreement there too.
    #[test]
    fn single_line_fast_path_matches_dp() {
        let corpus = [
            "hello",
            "a b c",
            "a  b",                  // interior space run survives verbatim
            "hello\n",               // trailing newline dropped
            "中文字",                // CJK, width 6
            "Int<32> + Vec<T>",      // code tokens, no split opportunity taken
            "  leading",
            "trailing  ",
            "a\nb",
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
                // Hand-run reference (the path the fast path replaces), with
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
    ///            cost 1³ + 5³ = 126
    ///
    ///   DP:      "x x x x"    "x x x x"    "LLLLLLLLL"
    ///            width 7      width 7      width 9
    ///            slack 3      slack 3      last line (free)
    ///            cost 3³ + 3³ = 54
    ///
    /// Greedy is *not* locally bad — every line it produces fits.  It is
    /// globally bad: by squeezing a 5th short word onto line 1, it forces a
    /// 3-word line 2 whose slack (5) dwarfs line 1's slack (1).  DP trades
    /// line 1's near-perfect slack for two balanced lines, and wins 54 < 126.
    #[test]
    fn dp_beats_greedy_when_short_words_strand_a_heavy_tail() {
        let src = "x x x x x x x x LLLLLLLLL";
        let out = wrap_optimal(src, 10, 4);

        // DP spreads the short words evenly instead of packing the first line.
        assert_eq!(out, vec!["x x x x", "x x x x", "LLLLLLLLL"]);

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
                total_badness(&dp, max_width) <= total_badness(&greedy, max_width) + 1e-9,
                "DP worse than greedy on {src:?} @ {max_width}:\n  dp={dp:?}\n  greedy={greedy:?}",
            );
            assert_eq!(dp.join(" "), src, "text mangled for {src:?}");
        }
    }

    /// Sum of cubed slack over non-final lines, matching `badness` (including
    /// the overflow branch, so an over-wide greedy line is penalised rather
    /// than silently treated as zero slack).
    fn total_badness(lines: &[String], max_width: usize) -> f64 {
        lines
            .iter()
            .rev()
            .skip(1)
            .map(|l| {
                let w = l.chars().count();
                if w <= max_width {
                    let slack = (max_width - w) as f64;
                    slack * slack * slack
                } else {
                    let over = (w - max_width) as f64;
                    OVERFLOW_PENALTY * over * over * over
                }
            })
            .sum()
    }
}
