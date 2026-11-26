//! Short single-line renderer — one greppable line per diagnostic.
//!
//! Mirrors the `gcc` / `rustc --error-format=short` shape that editor
//! error-list parsers, `:make` integrations, and CI log summarisers
//! consume:
//!
//! ```text
//! src/main.pn:12:5: error[E030]: type mismatch
//! ```
//!
//! Every component already exists in the diagnostic core (`DiagMessage`,
//! `LineIndex::linecol_display`, `ErrCode::code()`); this renderer is the
//! small adapter that serialises them onto one line.  No box drawing, no
//! source context, no suggestions, no summary — those belong to the
//! human renderer.

use super::fmt;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::render::{RenderCtx, Renderer};
use std::borrow::Cow;

/// One-line `file:line:col: level[code]: message` renderer.
///
/// A diagnostic with no resolvable location falls back to
/// `file:start..end: …` (byte range) or `level[code]: …` (no source at all),
/// so the shape stays greppable even for malformed input.
///
/// Colour, when enabled, is applied only to the `level[code]` tag — the
/// location and message stay uncoloured so a pipeline that strips ANSI
/// still sees a clean `file:line:col: message` line.
pub struct ShortRenderer {
    use_color: bool,
}

impl ShortRenderer {
    /// Plain text (no ANSI colour).  The `--error-format=short` default.
    #[must_use]
    pub fn new() -> Self {
        ShortRenderer { use_color: false }
    }

    /// Coloured `level[code]` tag (interactive terminals only).
    #[must_use]
    pub fn colored() -> Self {
        ShortRenderer { use_color: true }
    }
}

impl Default for ShortRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Byte offset of the first break character the flattening rule recognises.
///
/// **Test-only.**  The oracle used `memchr3` on the original byte set, which
/// cannot express U+2028/U+2029 (multi-byte), so it silently stopped being an
/// oracle for the whole rule the moment those were added.
#[cfg(test)]
fn first_break_byte(s: &str) -> Option<usize> {
    s.char_indices()
        .find(|&(_, c)| matches!(c, '\n' | '\r' | '\t' | '\u{2028}' | '\u{2029}'))
        .map(|(i, _)| i)
}

/// Reference implementation of the flattening rule — **test-only**.
///
/// Production goes through [`push_sanitized_flattened`], which fuses sanitizing
/// and flattening into one character pass; this two-stage form exists so the
/// differential test has an independent oracle.  It used to be the production
/// path, and its offset contract (a break offset into the *original* bytes) is a
/// panic-on-misuse footgun if anyone revives that architecture, hence the
/// assertion below.
#[cfg(test)]
#[inline]
fn push_flattened(out: &mut String, msg: &str, first: Option<usize>) {
    debug_assert!(
        first.is_none_or(|o| o < msg.len() && msg.is_char_boundary(o)),
        "push_flattened offset {first:?} is not a char boundary of a {} byte message",
        msg.len()
    );
    // Fast path: no break byte — straight copy.  The caller already scanned
    // for the first break while sanitizing, so this needs no second memchr3.
    let Some(_first) = first else {
        out.push_str(msg);
        return;
    };

    // Empty segments are exactly the interior/edges of a break run, so
    // skipping them *is* the run-collapsing rule — no second buffer needed.
    // `first` is used only by the assert above: the split below deliberately
    // walks from byte 0.  (An earlier comment claimed this seeded the loop with
    // the fast path's offset; it never did, and recreating the trim rule at a
    // shifted segment boundary is exactly the divergence this oracle exists to
    // catch.)
    // Segment edges abut a break, so a space there is part of the run: trim the
    // trailing spaces of every segment that a break follows and the leading
    // spaces of every segment that a break precedes, drop what is left empty,
    // and join with exactly one space.  This is the rule the fused path above
    // implements in one pass; the differential test pins them together.
    let mut segs = msg
        .split(['\n', '\r', '\t', '\u{2028}', '\u{2029}'])
        .peekable();
    let mut first_seg = true;
    let mut index = 0usize;
    while let Some(raw) = segs.next() {
        // `peek` is the last-segment test, so no collection is needed.
        let is_last = segs.peek().is_none();
        let mut seg = raw;
        if index > 0 {
            seg = seg.trim_start_matches(' ');
        }
        if !is_last {
            seg = seg.trim_end_matches(' ');
        }
        if !seg.is_empty() {
            if !first_seg {
                out.push(' ');
            }
            out.push_str(seg);
            first_seg = false;
        }
        index += 1;
    }
}

/// Sanitize + flatten in one pass over the original message.  The old
/// `sanitize_for_terminal_flatten_with_break` returned a break offset into the
/// *original* bytes; a U+FFFD replacement before that offset shifted the
/// sanitized string and made `push_flattened` start mid-character.  This
/// derives both decisions from the same character stream.
/// Could these bytes require a sanitize/flatten change at all?
///
/// Exact for everything the two rules care about:
///
/// * `< 0x20` — the C0 controls, which includes the three breaks (`\n`, `\r`, `\t`);
/// * `0x7F` — DEL;
/// * `0xC2` — the lead byte of every C1 control (`U+0080..=U+009F`), the only
///   controls outside C0/DEL;
/// * `0xE2` — the lead byte of U+200E/U+200F, U+2028/U+2029 (the Unicode line
///   separators) and the whole bidi override/isolate block, which is every
///   non-ASCII break or direction character this renderer touches.
///
/// False positives (an en dash, NBSP, any other U+2xxx) merely take the slow path,
/// so the predicate is *conservative-complete*: no byte that could change is
/// skipped.  That is what makes "no such byte ⇒ output is byte-identical to the
/// input" a proof rather than a hope, and it is the same argument `ascii.rs` proves
/// for its `has_e2` gate.
#[inline]
fn needs_flatten_or_sanitize(s: &[u8]) -> bool {
    // SWAR form of `b < 0x20 || b == 0x7F || b == 0xC2 || b == 0xE2`: this predicate
    // gates the dominant per-message fast path (message, file name, code — three
    // scans per diagnostic), and the scalar `iter().any()` runs four comparisons per
    // byte.
    //
    // Every mask is per-lane exact, using the "OR the high bit in, then subtract"
    // trick so no borrow crosses lanes:
    //   * `below20` is masked with `!x` so only high-bit-clear lanes can be flagged —
    //     without that, bytes 0x80..0x9F (UTF-8 continuation) would look like
    //     controls and send every non-ASCII message down the slow path;
    //   * the three equality masks are exact for any byte value.
    // A false *positive* would only be slower, but a false negative would leak a
    // break/control/bidi character into the "one physical line" output, so the
    // differential test below asserts equality with the scalar predicate over all
    // 256 bytes and a random corpus.
    const HI: u64 = 0x8080_8080_8080_8080;
    const ONES: u64 = 0x0101_0101_0101_0101;
    const LOW7: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    #[inline]
    fn rep(b: u8) -> u64 {
        u64::from(b) * ONES
    }
    /// High bit set in exactly the lanes equal to `k` **given every lane < 0x80**.
    ///
    /// Exact for 7-bit lanes: `(d | 0x80) - 1` keeps the high bit iff `d != 0`, and
    /// OR-ing `0x80` in first is what stops borrows from crossing lanes.
    #[inline]
    fn eq7(x: u64, k: u8) -> u64 {
        let d = (x & LOW7) ^ rep(k & 0x7F);
        !((d | HI).wrapping_sub(ONES)) & HI
    }
    /// High bit set in exactly the lanes below `0x20` (high-bit-clear lanes only).
    #[inline]
    fn below_20(x: u64) -> u64 {
        !((x | HI).wrapping_sub(rep(0x20))) & !x & HI
    }
    let mut i = 0usize;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().expect("8 bytes"));
        let high = x & HI;
        // 0x7F and the two UTF-8 lead bytes are matched on the low 7 bits *plus* the
        // high bit, which is exact: 0x7F has it clear while 0xC2/0xE2 have it set, and
        // no other byte shares those low-7 patterns with that high bit.  (Matching them
        // on all 8 bits was the bug: `x ^ 0xC2` can produce a lane of 0x80, whose
        // `(d|0x80)-1` of 0x7F looked exactly like a zero lane.)
        let del = eq7(x, 0x7F) & !high;
        let c2 = eq7(x, 0x42) & high; // 0xC2 & 0x7F == 0x42
        let e2 = eq7(x, 0x62) & high; // 0xE2 & 0x7F == 0x62
        if (below_20(x) | del | c2 | e2) != 0 {
            return true;
        }
        i += 8;
    }
    s[i..]
        .iter()
        .any(|&b| b < 0x20 || b == 0x7F || b == 0xC2 || b == 0xE2)
}

fn push_sanitized_flattened(out: &mut String, msg: &str) {
    // Sanitize and flatten in one pass, byte-identical to `push_flattened`'s
    // split/trim/join rule: segment edges abut a break, so a space there belongs
    // to the run — the *trailing* spaces of a segment a break follows are
    // retracted (`out.truncate`, the mark is the start of the current space run),
    // the *leading* ones after a break are absorbed, and segments join with
    // exactly one space.
    //
    // Suppressing the joiner alone is not enough: spaces pushed before the break
    // are already in the buffer.  `"a  \tb"` used to emit `"a  b"` where the
    // reference emits `"a b"`, and `" \ta"` emitted `" a"` vs. `"a"`.
    // Clean fast path.  Without a break, control or bidi byte, `pending_break` can
    // never be set, so no space run is truncated, no joiner is emitted and no char is
    // replaced: the output is byte-identical to the input.  The dominant case *is* a
    // clean ASCII sentence, and it now costs one scan plus one `push_str` instead of
    // a per-char state machine.
    if !needs_flatten_or_sanitize(msg.as_bytes()) {
        out.push_str(msg);
        return;
    }
    let mut emitted = false; // a non-space char survives the truncation
    let mut pending_break = false;
    let mut space_run: Option<usize> = None; // where the trailing space run began
    // Bulk path.  A run of *plain* chars — not a break, not a space, not a
    // control, not a bidi override — is copied with one `push_str` instead of one
    // `push` per char.  It cannot interact with the space/break state machine
    // (a run contains neither), so the rules above stay in force; only the store
    // changes.  Messages are the hot component of this format, and the dominant
    // case is a long ASCII sentence.
    let mut run_start: Option<usize> = None;
    let mut flush_run = |out: &mut String,
                         msg: &str,
                         upto: usize,
                         run_start: &mut Option<usize>,
                         pending_break: &mut bool,
                         space_run: &mut Option<usize>,
                         emitted: &mut bool| {
        let Some(start) = run_start.take() else {
            return;
        };
        if *pending_break {
            if *emitted {
                out.push(' ');
            }
            *pending_break = false;
        }
        out.push_str(&msg[start..upto]);
        *space_run = None;
        *emitted = true;
    };
    // `peekable`, not `for`, because a space run is consumed here: a long
    // interior run (`"a    b"`, tables, ASCII art) used to pay one `push` —
    // with its capacity check — per space.  The run is all spaces, so it cannot
    // touch the break state machine, and the differential flatten tests
    // (`merged_render_matches_three_walk_pipeline`) pin the bytes.
    let mut chars = msg.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if !matches!(c, '\n' | '\r' | '\t' | '\u{2028}' | '\u{2029}' | ' ')
            && !c.is_control()
            && !crate::diagnostics::is_bidi_override(c)
        {
            if run_start.is_none() {
                run_start = Some(i);
            }
            // Extend the run by *bytes*, not by decoding one char per iteration:
            // finding the run's end was the last per-byte cost left on the
            // dominant path (a long ASCII sentence).  The scan is a tight,
            // autovectorized `position` over the printable-ASCII set, and every
            // byte it admits is single-byte, so the char iterator can be advanced
            // by index.  A non-ASCII starter is left to the per-char path (its
            // continuation bytes would make byte-index advancement land
            // mid-character).
            if c.is_ascii() {
                let bytes = msg.as_bytes();
                let mut end = i + 1;
                // `first_byte_outside_range` instead of a `position` closure: the
                // short-circuiting `find` shape does not vectorize, while this walks 8
                // bytes per iteration with per-lane-exact SWAR masks (differentially
                // pinned against this same predicate in `ascii.rs`).  This is the
                // dominant per-message byte cost — a long ASCII sentence is one run.
                let rest =
                    crate::diagnostics::ascii::first_byte_outside_range(&bytes[end..], 0x21, 0x7E);
                end += if rest == usize::MAX {
                    bytes.len() - end
                } else {
                    rest
                };
                while let Some(&(j, _)) = chars.peek() {
                    if j < end {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            continue;
        }
        flush_run(
            out,
            msg,
            i,
            &mut run_start,
            &mut pending_break,
            &mut space_run,
            &mut emitted,
        );
        if crate::diagnostics::is_bidi_override(c) {
            continue;
        }
        // U+2028 LINE SEPARATOR / U+2029 PARAGRAPH SEPARATOR are Zl/Zp, so
        // `char::is_control` (Cc only) misses them — they used to reach the
        // output raw and split the "exactly one physical line" contract in
        // editors and JS consumers.  U+0085 (NEL) *is* Cc, so it becomes U+FFFD
        // below; treating the two separators as breaks is the consistent choice
        // for a format whose contract is one row.
        if matches!(c, '\n' | '\r' | '\t' | '\u{2028}' | '\u{2029}') {
            pending_break = true;
            if let Some(mark) = space_run.take() {
                out.truncate(mark);
            }
            continue;
        }
        let c = if c.is_control() { '\u{FFFD}' } else { c };
        if pending_break {
            // Leading spaces of the segment a break precedes are absorbed; the
            // loop stays "pending" so a whole run collapses.
            if c == ' ' {
                continue;
            }
            if emitted {
                out.push(' ');
            }
            pending_break = false;
        }
        if c == ' ' {
            if space_run.is_none() {
                // Char-boundary by construction: recorded right after a
                // non-space char or at the start of the buffer.
                space_run = Some(out.len());
            }
            // Consume the whole run and store it with one `push_str`.  `c` is
            // one byte, so the run starts at `i` and each following space is one
            // byte too.
            let mut end = i + 1;
            while let Some(&(j, ' ')) = chars.peek() {
                chars.next();
                end = j + 1;
            }
            out.push_str(&msg[i..end]);
            continue;
        }
        space_run = None;
        emitted = true;
        out.push(c);
    }
    flush_run(
        out,
        msg,
        msg.len(),
        &mut run_start,
        &mut pending_break,
        &mut space_run,
        &mut emitted,
    );
}

/// Flatten one untrusted single-line component (`file` name, unknown code),
/// borrowing when it is already clean.
///
/// The message path flattens inline (one pass straight into `out`); these two
/// components need the same rule but are also *measured* for the reservation,
/// so they materialize only when a break/control/bidi byte is present.  Clean
/// input — the dominant case — stays borrowed and allocation-free, and the
/// break-collapsing rules live in exactly one function
/// (`push_sanitized_flattened`) instead of being implied by the sanitizer's
/// break semantics.
fn flatten_component(s: &str) -> Cow<'_, str> {
    // The same byte predicate the message fast path uses, instead of a char-decode
    // pass: `needs_flatten_or_sanitize` is a superset of the old conditions (C0/DEL,
    // C1 via `0xC2`, U+2028/U+2029 and the bidi block via `0xE2`), so borrowing on
    // "no such byte" stays sound and the scan becomes SWAR-vectorizable.
    if !needs_flatten_or_sanitize(s.as_bytes()) {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    push_sanitized_flattened(&mut out, s);
    Cow::Owned(out)
}

impl Renderer for ShortRenderer {
    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
        // The contract `json.rs` states explicitly instead of relying on an
        // undocumented convention: the caller hands in an empty buffer.  A non-empty
        // one would concatenate two "greppable one-liners" into one, and the emitters'
        // panic-containment path (placeholder → clear → refill) is exactly the
        // machinery whose future regression this guards.
        debug_assert!(
            out.is_empty(),
            "short renderer expects an empty buffer, got {} bytes",
            out.len()
        );
        let file_raw = diag.file_name();
        // File names are untrusted (`set_file_name` / `with_file_name` are
        // public) and so are unknown codes, so both go through the same
        // sanitize+flatten pass as the message: sanitizing alone would leave a
        // `\n`/`\r`/`\t` in a component the one-line contract covers, and a
        // `\r` in a file name would make a terminal redraw the row.
        let file_owned = flatten_component(file_raw);
        let file = file_owned.as_ref();
        // The machine vocabulary follows rustc: fatal errors are spelled
        // `error` so CI summarisers gating on the `error` token do not
        // silently ignore them.  The human/JSON renderers keep the `fatal`
        // distinction.
        let level = if matches!(diag.level(), crate::diagnostics::DiagnosticLevel::Fatal) {
            "error"
        } else {
            diag.level().as_str()
        };
        // Borrow the code string directly; the `[E030]` framing and the
        // coloured tag are written straight into the output below, so no
        // intermediate `CompactString` is built.
        let code_display: Option<Cow<'_, str>> = diag.code().and_then(|c| {
            let s = c.code();
            if s.is_empty() {
                None
            } else if c.is_known() {
                // Registry codes are validated ^[EW]\d{3}$ printable ASCII.
                Some(Cow::Borrowed(s))
            } else {
                Some(flatten_component(s))
            }
        });

        // Resolve the location *before* allocating: the 1-based line/col
        // digit widths feed the capacity computation, and every accessor
        // (`spans()`, `source()`, `message()`) is called once.
        // `primary_ref()`, like `json.rs` and the emitter's batch path: `spans()`
        // is the full collection in insertion order, so a secondary registered first
        // would print *its* coordinates as THE location while the machine JSON
        // printed the primary's.  The `Cow` is bound first: `.first()` on a
        // temporary does not outlive the statement.
        let primaries = diag.spans.primary_ref();
        let span = primaries.first();
        // Hoisted: the renderer's contract is one accessor call each, and the
        // comment claimed that while calling `source()` three times.
        let src = diag.source();
        let linecol: Option<(usize, usize)> = match (span, src) {
            (Some(span), Some(source)) => {
                let (line, col) = cx
                    .line_index(source)
                    .linecol_display(span.start_as_usize().min(source.len()));
                Some((line + 1, col + 1))
            }
            _ => None,
        };

        // One fused pass handles bidi removal, control replacement and break
        // collapsing.  `raw_msg.len()` is only a lower bound: U+FFFD growth is
        // rare, and `String` grows amortized rather than paying a pre-scan.
        let raw_msg = diag.message();

        // ---- Size the reservation to a lower bound -----------------------
        //   location + tag + ": " + message + "\n"
        // Every term saturates: this feeds `reserve`, so for an absurd diagnostic the
        // right answer is "huge", not a debug-build overflow panic (`json.rs::estimate_cap`
        // documents the same rule for the same reason).
        let mut len = raw_msg
            .len()
            .saturating_add(2 /* ": " */)
            .saturating_add(1 /* "\n" */);
        match (linecol, span) {
            (Some((line, col)), _) => {
                if !file.is_empty() {
                    len = len.saturating_add(file.len()).saturating_add(1); // "file:"
                }
                len = len
                    .saturating_add(fmt::dec_digits(line as u64))
                    .saturating_add(1)
                    .saturating_add(fmt::dec_digits(col as u64))
                    .saturating_add(2); // "line:col: "
            }
            // Byte-range fallback: two `push_uint` calls (start, end) need at
            // most 20 digits each plus the `..`, i.e. 44 bytes — reserved here so
            // the two writes cannot grow the buffer.  (The old comment claimed a
            // bound for `Span`'s `Display`, which this branch never calls.)
            (None, Some(_)) => {
                if !file.is_empty() {
                    len = len.saturating_add(file.len()).saturating_add(1);
                }
                len = len.saturating_add(48); // "start..end: " upper bound
            }
            (None, None) => {
                if src.is_some() && !file.is_empty() {
                    len = len.saturating_add(file.len()).saturating_add(2); // "file: "
                }
            }
        }
        if self.use_color {
            len = len.saturating_add(4 + level.len() + 4); // "\x1b[1m" level "\x1b[0m"
            if let Some(code) = code_display.as_deref() {
                len = len.saturating_add(5 + code.len() + 5); // "\x1b[2m[" code "]\x1b[0m"
            }
        } else {
            len = len.saturating_add(level.len());
            if let Some(code) = code_display.as_deref() {
                len = len.saturating_add(code.len()).saturating_add(2); // "[code]"
            }
        }
        out.reserve(len);

        // ---- Location ----------------------------------------------------
        match (linecol, span) {
            (Some((line, col)), _) => {
                if !file.is_empty() {
                    out.push_str(file);
                    out.push(':');
                }
                fmt::push_uint(out, line as u64);
                out.push(':');
                fmt::push_uint(out, col as u64);
                out.push_str(": ");
            }
            // No source: a byte range is the best we can print.
            (None, Some(span)) => {
                if !file.is_empty() {
                    out.push_str(file);
                    out.push(':');
                }
                // `push_uint(.. as u64)`, not `push_u32(.. as u32)`: the
                // `usize → u32` cast silently truncates if an offset ever
                // exceeds 4 GiB, and this renderer's whole purpose is printing
                // the range faithfully.
                fmt::push_uint(out, span.start_as_usize() as u64);
                out.push_str("..");
                fmt::push_uint(out, span.end_as_usize() as u64);
                out.push_str(": ");
            }
            // No span at all: the file name is printed only when a source is
            // attached, i.e. when it *places* the message.  This is a decision,
            // not an oversight: a bare `src/lib.pn:` with neither coordinates nor
            // an excerpt is noise in the grep-style output this renderer exists
            // for (and it would print the `<input>` default for every
            // source-less diagnostic).  Pinned by
            // `no_source_no_span_omits_the_file_name`.
            (None, None) => {
                if src.is_some() && !file.is_empty() {
                    out.push_str(file);
                    out.push_str(": ");
                }
            }
        }

        // ---- level[code] tag ---------------------------------------------
        // When colour is on but the diagnostic has no code, skip the
        // `\x1b[2m…\x1b[0m` pair entirely — an empty `\x1b[2m\x1b[0m` is a
        // no-op for the terminal and just noise for anything that counts
        // escapes.
        if self.use_color {
            out.push_str("\x1b[1m");
            out.push_str(level);
            out.push_str("\x1b[0m");
            if let Some(code) = code_display.as_deref() {
                out.push_str("\x1b[2m[");
                out.push_str(code);
                out.push_str("]\x1b[0m");
            }
        } else {
            out.push_str(level);
            if let Some(code) = code_display.as_deref() {
                out.push('[');
                out.push_str(code);
                out.push(']');
            }
        }
        out.push_str(": ");

        // ---- Message, sanitized and flattened in a single pass ----------
        // A multi-line message would break the one-line format, and a stray
        // `\r` would make a terminal redraw the line.
        push_sanitized_flattened(out, raw_msg);
        out.push('\n');
    }

    // Machine formats do not carry a summary trailer — a parser reading
    // `:make` output would treat it as a bogus diagnostic.
    fn render_summary(&self, _errors: usize, _warnings: usize) -> String {
        String::new()
    }

    /// Every short render already ends in `'\n'`, so the batch separator is
    /// the empty string — otherwise a batch of short diagnostics would be
    /// double-spaced and the "one greppable line each" contract would break.
    fn render_separator(&self) -> &str {
        ""
    }

    /// `render_summary_full` defaults to `render_summary`, and this format
    /// overrides that to `""` — pin it, because a batch rendered with
    /// `--summary-full` must not gain a prose trailer in a machine format.
    fn render_summary_full(&self, _errors: usize, _warnings: usize, _files: usize) -> String {
        String::new()
    }

    /// The default marker has no trailing newline and no `level[code]:` prefix,
    /// so in this format it concatenated onto the next row
    /// (`...panickedsrc/main.pn:2:2: error…`) and carried no token a CI
    /// summariser can grep.  Every other row ends in `\n`, so this one does too.
    fn render_internal_error(&self, out: &mut String) {
        out.push_str("error[internal]: diagnostic rendering panicked\n");
    }

    fn render_suppressed_note(&self, _count: usize) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{FileId, Span};
    use crate::diagnostics::DiagOpts;
    use crate::diagnostics::ErrCode;
    use crate::diagnostics::render::RenderCtx;

    fn ctx(opts: &DiagOpts) -> RenderCtx<'_> {
        RenderCtx::new(opts, None)
    }

    /// The reservation is `raw_msg.len()`, a **lower bound**: U+FFFD replacement
    /// and the joiner spaces can grow the output by a byte or two, which `String`
    /// absorbs by reallocating.  What this test actually pins is the other half —
    /// that `push_flattened` never grows its input — so keep that separate from
    /// the "reservation is exact" claim the old comment made.
    /// Pin that over a random corpus in both sanitizer modes — a future
    /// "improvement" to `push_flattened` that grows output would otherwise
    /// make the reserve under-book.
    #[test]
    fn fused_sanitize_flatten_matches_reference_bytes() {
        for raw in [
            "",
            "plain",
            "a\nb",
            "a\r\n\tb",
            "\ta\t",
            "a\t\tb",
            "a\x01\nb",
            "a\u{202E}b",
            "a\u{0085}b",
            "e\u{0301}\u{FFFD}中",
            "x\x7f\ny",
            // Space runs on the *before* side of a break.  These were the
            // divergence class: suppressing only the post-break joiner left the
            // pre-break spaces in the buffer ("a  \tb" -> "a  b" vs. "a b").
            "a  \tb",
            "a \t",
            " \ta",
            "  \tb",
            "a  \t  b",
            "a\t ",
            "a \t ",
            "  \t  ",
            "\t a",
            "a\t\t  b",
        ] {
            let sanitized = crate::diagnostics::sanitize_for_terminal_flatten(raw);
            let break_at = first_break_byte(&sanitized);
            let mut want = String::new();
            push_flattened(&mut want, &sanitized, break_at);
            let mut got = String::new();
            push_sanitized_flattened(&mut got, raw);
            assert_eq!(got, want, "fused mismatch for {raw:?}");
        }
    }

    #[test]
    fn flattening_never_grows() {
        // xorshift64*, deterministic across runs and platforms.
        let mut x: u64 = 0x243F_6A88_85A3_08D3;
        let mut next = move || {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for _ in 0..20_000 {
            let len = (next() % 24) as usize;
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                // Bias toward the dirty ranges: C0, DEL, C1 lead/continuation,
                // tab/newline/CR, plain ASCII, non-ASCII lead.
                let c = match next() % 8 {
                    0 => (next() % 0x20) as u8,
                    1 => 0x7F,
                    2 => 0xC2,
                    3 => 0x80 + (next() % 0x20) as u8,
                    4 => [b'\t', b'\n', b'\r'][(next() % 3) as usize],
                    5..=6 => 0x20 + (next() % 0x5F) as u8,
                    _ => 0xC0 + (next() % 0x20) as u8,
                };
                bytes.push(c);
            }
            let Ok(s) = simdutf8::basic::from_utf8(&bytes) else {
                continue;
            };
            for keep_breaks in [true, false] {
                let sanitized = if keep_breaks {
                    crate::diagnostics::sanitize_for_terminal_flatten(s)
                } else {
                    crate::diagnostics::sanitize_for_terminal(s)
                };
                let mut want = String::new();
                push_flattened(&mut want, &sanitized, first_break_byte(&sanitized));
                assert!(
                    want.len() <= sanitized.len(),
                    "push_flattened grew {s:?} (keep_breaks={keep_breaks}): \
                     {} > {}",
                    want.len(),
                    sanitized.len()
                );
                // Equality against the oracle, not just non-growth: the size
                // check alone could not see the two implementations disagree.
                //
                // Only under `keep_breaks`: the fused path *is* the flattening
                // sanitizer's one-pass form, while `sanitize_for_terminal` (the
                // non-flatten one) maps a break to U+FFFD, so under
                // `keep_breaks == false` the two are fed legitimately different
                // inputs and only the size bound is meaningful.
                // The bound is the *flatten-sanitized* width: a 1-byte control
                // becomes a 3-byte U+FFFD, so comparing against the raw input
                // length would flag legitimate expansion.
                let flatten_sanitized = crate::diagnostics::sanitize_for_terminal_flatten(s);
                let mut got = String::new();
                push_sanitized_flattened(&mut got, s);
                assert!(
                    got.len() <= flatten_sanitized.len(),
                    "fused flattener grew {s:?} (keep_breaks={keep_breaks}): \
                     {} > {}",
                    got.len(),
                    flatten_sanitized.len()
                );
                if keep_breaks {
                    assert_eq!(
                        got, want,
                        "fused flatten diverged from the reference for {s:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn fatal_is_spelled_error_for_ci_token_compatibility() {
        let src = "x\n";
        let diag = Diagnostic::fatal("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_source(src);
        let opts = DiagOpts::default();
        let out = ShortRenderer::new().render(&diag, &ctx(&opts));
        assert!(out.contains(": error[E030]: "), "{out:?}");
        assert!(!out.contains("fatal"), "{out:?}");
    }

    #[test]
    fn source_without_span_and_empty_file_has_bare_prefix() {
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_source("fn main() {}\n")
            .with_file_name("");
        let out = ShortRenderer::new().render(&diag, &ctx(&DiagOpts::default()));
        assert_eq!(out, "error[E030]: boom\n");
    }

    #[test]
    fn renders_file_line_col_level_code_message() {
        let src = "let x = 1;\nlet y = 2;\n";
        let diag = Diagnostic::error("type mismatch")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 12, 13))
            .with_source(src)
            .with_file_name("src/main.pn");
        let opts = DiagOpts::default();
        let out = ShortRenderer::new().render(&diag, &ctx(&opts));
        assert_eq!(out, "src/main.pn:2:2: error[E030]: type mismatch\n");
    }

    #[test]
    fn no_source_falls_back_to_byte_range() {
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 3));
        let opts = DiagOpts::default();
        let out = ShortRenderer::new().render(&diag, &ctx(&opts));
        // Exact shape: editor error-list parsers consume this, and a regression
        // printing `0-3` / `[0,3)` or dropping the trailing `": "` would pass
        // every `contains` assertion while breaking them.
        assert_eq!(out, "<input>:0..3: error[E030]: boom\n");
    }

    /// Every untrusted component of the one-line format must be flattened, not
    /// merely sanitized: a break byte in a file name or an unknown code would
    /// otherwise split the row (and a `\r` would make a terminal redraw it).
    #[test]
    fn file_name_and_unknown_code_are_flattened() {
        let diag = Diagnostic::error("boom")
            .with_code_str("bad\tcode")
            .with_file_name("a\nb\r c")
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_source("x\n");
        let opts = DiagOpts::default();
        let out = ShortRenderer::new().render(&diag, &ctx(&opts));
        // A maximal break run collapses to one space, and a literal space that
        // abuts a break belongs to that run — so `a\nb\r c` flattens to `a b c`
        // (the space before `c` is absorbed).  Pinned exactly in both the fused
        // and the reference path.
        assert_eq!(out, "a b c:1:1: error[bad code]: boom\n");
        assert_eq!(
            out.lines().count(),
            1,
            "component break split the row: {out:?}"
        );
    }

    #[test]
    /// Pins: a machine format must never grow a prose trailer, and a
    /// contained render panic must still produce a greppable, newline-terminated
    /// row (the default marker was prose with neither).
    fn machine_format_rows_and_summary_stay_machine_shaped() {
        use super::Renderer as _;
        let renderer = super::ShortRenderer::default();
        let diag = Diagnostic::error("boom");
        let opts = crate::diagnostics::DiagOpts::default();
        let cx = crate::diagnostics::render::RenderCtx::new(&opts, None);
        let mut out = String::new();
        renderer.render_into(&diag, &cx, &mut out);
        assert!(
            out.ends_with('\n'),
            "every short row ends in a newline: {out:?}"
        );
        assert_eq!(renderer.render_summary_full(1, 0, 3), "");
        let mut panic_row = String::new();
        renderer.render_internal_error(&mut panic_row);
        assert!(panic_row.ends_with('\n'), "{panic_row:?}");
        assert!(
            panic_row.starts_with("error[internal]: "),
            "a CI summariser must be able to grep the row: {panic_row:?}"
        );
    }

    #[test]
    fn separator_is_empty() {
        // A non-empty separator would double-space the output and break the
        // one-line contract the summary/suppressed hooks were given.
        assert_eq!(ShortRenderer::new().render_separator(), "");
    }

    /// The shape whose omission used to look accidental (the old pin only used
    /// the default `<input>` name): a real file name is still omitted when there
    /// is neither a span nor a source, because a name alone cannot place the
    /// message.
    /// The SWAR gate must equal the scalar predicate for every byte and every length
    /// across the 8-byte boundary.
    ///
    /// A false negative would let a control/break/bidi character reach the "one
    /// physical line" output unchanged, so this is an equivalence check, not a
    /// spot-check: every single byte 0..=255 alone, and every byte inside an otherwise
    /// clean 8-, 9- and 16-byte row.
    #[test]
    fn swar_gate_matches_the_scalar_predicate() {
        fn scalar(s: &[u8]) -> bool {
            s.iter()
                .any(|&b| b < 0x20 || b == 0x7F || b == 0xC2 || b == 0xE2)
        }
        for b in 0u8..=255 {
            for len in [1usize, 8, 9, 16] {
                for pos in 0..len {
                    let mut row = vec![b'a'; len];
                    row[pos] = b;
                    assert_eq!(
                        needs_flatten_or_sanitize(&row),
                        scalar(&row),
                        "byte {b:#04x} at {pos} of a {len}-byte row"
                    );
                }
            }
        }
        // Clean rows of every length past the chunk size stay clean (no false
        // positive from the `!x` masking of the low-control mask).
        for len in 0..=24 {
            let row = vec![b'x'; len];
            assert!(!needs_flatten_or_sanitize(&row), "len {len}");
            let cjk = "中文".repeat(len / 2 + 1);
            let bytes = cjk.as_bytes();
            assert!(
                !needs_flatten_or_sanitize(bytes),
                "CJK text must not trip the gate (len {} bytes)",
                bytes.len()
            );
        }
    }

    /// Clean input must reach the identity fast path, and dirty input must not.
    ///
    /// The predicate is conservative-complete (`< 0x20`, DEL, `0xC2`, `0xE2`), so
    /// "no such byte ⇒ output == input" is a proof rather than a hope: nothing else
    /// can be a break, a control or a bidi character. Both directions are pinned — a
    /// predicate that flagged too much would merely be slow, while one that flagged
    /// too little is a correctness bug.
    #[test]
    fn clean_messages_take_the_identity_path_and_dirty_ones_do_not() {
        for clean in [
            "",
            "plain ascii sentence",
            "  leading and trailing spaces  ",
            "a   b     c",
            "中文 注释 with ascii",
            "тип: тип",
            "café résumé",
        ] {
            assert!(
                !needs_flatten_or_sanitize(clean.as_bytes()),
                "predicate flagged a clean string: {clean:?}"
            );
            let mut out = String::new();
            push_sanitized_flattened(&mut out, clean);
            assert_eq!(out, clean, "clean input must be byte-identical");
        }
        for dirty in [
            "a\nb",
            "a\rb",
            "a\tb",
            "a\u{85}b",
            "a\u{2028}b",
            "a\u{202E}b",
            "del\u{7f}",
            "nul\u{0}",
        ] {
            assert!(
                needs_flatten_or_sanitize(dirty.as_bytes()),
                "predicate missed a dirty string: {dirty:?}"
            );
        }
        // Predicate *false positives* still produce identity output: U+2014 (en dash)
        // and U+00A0 (NBSP) trip the `0xE2`/`0xC2` gates, so they take the state
        // machine — which changes nothing, because neither is a break, a control or a
        // bidi character.  That is what "conservative" buys: a wider scan, never a
        // wrong byte.
        for (flagged, text) in [
            (true, "en dash \u{2014} here"),
            (true, "nbsp\u{a0}here"),
            (false, "plain café"),
        ] {
            assert_eq!(
                needs_flatten_or_sanitize(text.as_bytes()),
                flagged,
                "{text:?}"
            );
            let mut out = String::new();
            push_sanitized_flattened(&mut out, text);
            assert_eq!(
                out, text,
                "false positives must still be identity: {text:?}"
            );
        }
        // `flatten_component` borrows exactly when the predicate says clean.
        for clean in ["plain", "  spaced  ", "中文"] {
            assert!(matches!(flatten_component(clean), Cow::Borrowed(_)));
        }
        for dirty in ["a\nb", "a\u{202E}b"] {
            assert!(matches!(flatten_component(dirty), Cow::Owned(_)));
        }
    }

    /// A span starting inside a multi-byte codepoint must floor to the char
    /// boundary, not emit a column that does not exist.
    ///
    /// `json.rs` pins the same invariant for its coordinates
    /// (`span_start_inside_a_multibyte_char_lands_on_the_boundary`); `short.rs`
    /// consumes the *same* `LineIndex` but had no equivalent test, so the floor
    /// behaviour was claimed in one renderer and merely inherited in the other.
    #[test]
    fn span_start_inside_a_multibyte_char_floors_to_the_boundary() {
        use super::Renderer as _;
        let opts = crate::diagnostics::DiagOpts::default();
        let cx = crate::diagnostics::render::RenderCtx::new(&opts, None);
        // (source, byte offset, expected column) — 1-based columns, char-based.
        let cases = [
            ("éa\n", 1usize, 1usize), // inside `é` (2 bytes) -> floors to byte 0
            ("雪a\n", 2, 1),          // inside `雪` (3 bytes) -> floors to byte 0
            ("雪a\n", 1, 1),          // also inside `雪`
            ("a雪b\n", 2, 2),         // inside `雪` -> floors to the `雪` boundary
        ];
        for (src, start, want_col) in cases {
            let span = Span::new(FileId::DUMMY, start as u32, start as u32 + 1);
            let diag = Diagnostic::error("boom")
                .with_span(span)
                .with_source(src)
                .with_file_name("x.pn");
            let mut out = String::new();
            super::ShortRenderer::default().render_into(&diag, &cx, &mut out);
            assert!(
                out.contains(&format!(":1:{want_col}:")),
                "start {start} of {src:?} must report column {want_col}: {out:?}"
            );
        }
    }

    /// The location must come from the **primary** span, not from insertion order.
    ///
    /// `with_span` + `with_additional_span` leaves `spans()` in insertion order
    /// (here: the later-positioned span first) while `primary_ref()` sorts, so
    /// reading `spans().first()` printed the additional span's coordinates as THE
    /// location while `json.rs` printed the primary's — two renderers disagreeing
    /// about where the same diagnostic is.
    #[test]
    fn location_comes_from_the_primary_span_not_insertion_order() {
        use super::Renderer as _;
        let opts = crate::diagnostics::DiagOpts::default();
        let cx = crate::diagnostics::render::RenderCtx::new(&opts, None);
        let src = "aaa\nbbb\n";
        let later = Span::new(FileId::DUMMY, 4, 7); // line 2, pushed first
        let earlier = Span::new(FileId::DUMMY, 0, 3); // line 1
        let diag = Diagnostic::error("boom")
            .with_span(later)
            .with_additional_span(earlier)
            .with_source(src)
            .with_file_name("x.pn");

        // Both accessors agree, and both are sorted: `MultiSpan` holds only primary
        // spans and every read goes through the dedup'd sorted view, so the
        // "secondary registered first" hazard cannot arise in
        // this model.  The alignment on `primary_ref()` is therefore consistency, not
        // repair — and this pins the equivalence so a future accessor split cannot
        // reintroduce the divergence silently.
        assert_eq!(diag.spans.primary_ref().first(), Some(&earlier));
        assert_eq!(
            diag.spans().first(),
            Some(earlier),
            "spans() must be the same sorted view as primary_ref()"
        );

        let mut out = String::new();
        super::ShortRenderer::default().render_into(&diag, &cx, &mut out);
        assert!(
            out.contains("1:1"),
            "the primary span's location is expected: {out:?}"
        );
        assert!(
            !out.contains("2:1"),
            "the additional span leaked into the location: {out:?}"
        );
    }

    #[test]
    fn no_source_no_span_omits_the_file_name() {
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_file_name("src/lib.pn");
        let opts = DiagOpts::default();
        let out = ShortRenderer::new().render(&diag, &ctx(&opts));
        assert_eq!(out, "error[E030]: boom\n");
    }

    /// A span-less, source-less diagnostic renders the bare shape — no location
    /// prefix at all (distinct from the file-omission pin above, which keeps a
    /// source attached).
    #[test]
    fn no_span_has_no_location_prefix() {
        let diag = Diagnostic::error("boom").with_code_str(ErrCode::E030);
        let opts = DiagOpts::default();
        let out = ShortRenderer::new().render(&diag, &ctx(&opts));
        assert_eq!(out, "error[E030]: boom\n");
    }

    #[test]
    fn source_without_span_keeps_file_prefix() {
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_file_name("input.ps")
            .with_source("fn main() {}\n");
        let opts = DiagOpts::default();
        let out = ShortRenderer::new().render(&diag, &ctx(&opts));
        assert_eq!(out, "input.ps: error[E030]: boom\n");
    }

    #[test]
    fn colored_wraps_only_the_tag() {
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 3));
        let opts = DiagOpts::default();
        let out = ShortRenderer::colored().render(&diag, &ctx(&opts));
        assert!(out.contains("\x1b[1merror\x1b[0m"), "{out:?}");
        assert!(out.contains("\x1b[2m[E030]\x1b[0m"), "{out:?}");
        // Location and message stay plain.
        assert!(!out.contains("\x1b[1m<input>"), "{out:?}");
    }

    #[test]
    fn no_summary_or_suppressed_note() {
        let mut r = ShortRenderer::new();
        assert_eq!(r.render_summary(3, 1), "");
        assert_eq!(r.render_suppressed_note(2), "");
    }

    #[test]
    fn multiline_message_is_flattened() {
        let src = "x\n";
        let diag = Diagnostic::error("line one\nline two")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_source(src);
        let opts = DiagOpts::default();
        let out = ShortRenderer::new().render(&diag, &ctx(&opts));
        assert_eq!(out.matches('\n').count(), 1, "must be one line: {out:?}");
        assert!(out.contains("line one line two"), "{out}");
    }

    // ── Tab normalization: the one-line format is a flat field stream ──

    /// A tab in a message must never reach the one-line output verbatim:
    /// tab-splitting consumers (editor error lists) would treat it as a
    /// field separator.  `push_flattened` consumes `\t` as a break run,
    /// collapsing it to a single space exactly like `\n`/`\r`.
    #[test]
    fn tabs_are_flattened_to_spaces() {
        let src = "x\n";
        for (raw, expect) in [
            ("a\tb", "a b"),
            ("a\t\tb", "a b"),
            ("a\t b", "a b"),
            ("\ta", "a"),
            ("a\t", "a"),
            ("a\n\tb", "a b"),
        ] {
            let diag = Diagnostic::error(raw)
                .with_code_str(ErrCode::E030)
                .with_span(Span::new(FileId::DUMMY, 0, 1))
                .with_source(src);
            let opts = DiagOpts::default();
            let out = ShortRenderer::new().render(&diag, &ctx(&opts));
            assert_eq!(out.matches('\t').count(), 0, "tab survived: {out:?}");
            assert!(out.contains(expect), "{raw:?} → {out:?}, want …{expect:?}");
        }
    }

    // ── CR / CRLF / consecutive newline folding ─────────────────────

    #[test]
    fn crlf_bare_cr_and_break_runs_are_flattened() {
        let src = "x\n";
        for raw in [
            "line one\r\nline two",
            "line one\rline two",
            "line one\n\r\nline two",
            "line one\n\nline two",
        ] {
            let diag = Diagnostic::error(raw)
                .with_code_str(ErrCode::E030)
                .with_span(Span::new(FileId::DUMMY, 0, 1))
                .with_source(src);
            let opts = DiagOpts::default();
            let out = ShortRenderer::new().render(&diag, &ctx(&opts));
            assert_eq!(out.matches('\n').count(), 1, "must be one line: {out:?}");
            assert!(!out.contains('\r'), "no CR may survive: {out:?}");
            assert!(out.contains("line one line two"), "{out:?}");
        }
    }

    #[test]
    fn edge_breaks_do_not_leave_stray_spaces() {
        let src = "x\n";
        let diag = Diagnostic::error("\nboom\r\n")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_source(src);
        let opts = DiagOpts::default();
        let out = ShortRenderer::new().render(&diag, &ctx(&opts));
        assert_eq!(out, "<input>:1:1: error[E030]: boom\n");
    }

    // ── Colored output without a code emits no empty escape ────────

    #[test]
    fn colored_without_code_emits_no_empty_dim_pair() {
        let diag = Diagnostic::error("boom").with_span(Span::new(FileId::DUMMY, 0, 3));
        let opts = DiagOpts::default();
        let out = ShortRenderer::colored().render(&diag, &ctx(&opts));
        // The dim pair only appears when there is a code to wrap.
        assert!(!out.contains("\x1b[2m\x1b[0m"), "empty dim pair: {out:?}");
        assert!(out.contains("\x1b[1merror\x1b[0m"), "{out:?}");
    }
}
