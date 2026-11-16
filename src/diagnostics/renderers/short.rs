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
    pub fn new() -> Self {
        ShortRenderer { use_color: false }
    }

    /// Coloured `level[code]` tag (interactive terminals only).
    pub fn colored() -> Self {
        ShortRenderer { use_color: true }
    }
}

impl Default for ShortRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Append `msg` with every layout byte removed — one pass, zero allocation.
///
/// A maximal run of `\n`/`\r`/`\t` (so also `\r\n`) collapses to a single
/// space; runs at the very start or end of the message are dropped outright,
/// so a message that arrives with a trailing newline does not leave a
/// dangling space before the output's own `\n`.  A tab is treated as a break
/// run too: the one-line short format is a flat, field-delimited text stream,
/// and a literal tab reaching the output could be misparsed as a field
/// separator by tab-splitting consumers (editor error lists).  This is what
/// keeps the one-line contract: a bare `\r` would otherwise make a terminal
/// redraw the line, and some line-oriented parsers would truncate it.
#[inline]
fn push_flattened(out: &mut String, msg: &str, first: Option<usize>) {
    let bytes = msg.as_bytes();

    // Fast path: no break byte — straight copy.  The caller already scanned
    // for the first break while sanitizing, so this needs no second memchr3.
    let Some(first) = first else {
        out.push_str(msg);
        return;
    };

    // Empty segments are exactly the interior/edges of a break run, so
    // skipping them *is* the run-collapsing rule — no second buffer needed.
    // Seed the loop with the offset the fast path already found instead of
    // re-scanning the prefix.
    let mut first_seg = true;
    let mut start = 0;
    let mut offset = Some(first);
    while start < bytes.len() {
        let end = match offset {
            Some(o) => start + o,
            None => bytes.len(),
        };
        let seg = &msg[start..end];
        if !seg.is_empty() {
            if !first_seg {
                out.push(' ');
            }
            out.push_str(seg);
            first_seg = false;
        }
        if offset.is_none() {
            break;
        }
        start = end + 1;
        offset = memchr::memchr3(b'\n', b'\r', b'\t', &bytes[start..]);
    }
}

/// Sanitize + flatten in one pass over the original message.  The old
/// `sanitize_for_terminal_flatten_with_break` returned a break offset into the
/// *original* bytes; a U+FFFD replacement before that offset shifted the
/// sanitized string and made `push_flattened` start mid-character.  This
/// derives both decisions from the same character stream.
fn push_sanitized_flattened(out: &mut String, msg: &str) {
    let mut wrote = false;
    let mut pending_break = false;
    for c in msg.chars() {
        if crate::diagnostics::is_bidi_override(c) {
            continue;
        }
        if matches!(c, '\n' | '\r' | '\t') {
            pending_break = true;
            continue;
        }
        let c = if c.is_control() { '\u{FFFD}' } else { c };
        if pending_break {
            if wrote {
                out.push(' ');
            }
            pending_break = false;
        }
        out.push(c);
        wrote = true;
    }
}

impl Renderer for ShortRenderer {
    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
        let file_raw = diag.file_name();
        // File names are untrusted (`set_file_name` / `with_file_name` are
        // public): route through the terminal sanitizer like the code below,
        // closing the only renderer without this guard.
        let file_owned = super::super::sanitize_for_terminal(file_raw);
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
                Some(super::super::sanitize_for_terminal(s))
            }
        });

        // Resolve the location *before* allocating: the 1-based line/col
        // digit widths feed the capacity computation, and every accessor
        // (`spans()`, `source()`, `message()`) is called once.
        let span = diag.spans().first();
        let linecol: Option<(usize, usize)> = match (span, diag.source()) {
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
        let mut len = raw_msg.len() + 2 /* ": " */ + 1 /* "\n" */;
        match (linecol, span) {
            (Some((line, col)), _) => {
                if !file.is_empty() {
                    len += file.len() + 1; // "file:"
                }
                len += fmt::dec_digits(line as u64) + 1 + fmt::dec_digits(col as u64) + 2; // "line:col: "
            }
            // Byte-range fallback: reserve a generous bound for `Span`'s
            // `Display` rather than formatting it twice.  Not on the hot
            // path; over-reserving a few dozen bytes here is fine.
            (None, Some(_)) => {
                if !file.is_empty() {
                    len += file.len() + 1;
                }
                len += 48; // "start..end: " upper bound
            }
            (None, None) => {
                if diag.source().is_some() && !file.is_empty() {
                    len += file.len() + 2; // "file: "
                }
            }
        }
        if self.use_color {
            len += 4 + level.len() + 4; // "\x1b[1m" level "\x1b[0m"
            if let Some(code) = code_display.as_deref() {
                len += 5 + code.len() + 5; // "\x1b[2m[" code "]\x1b[0m"
            }
        } else {
            len += level.len();
            if let Some(code) = code_display.as_deref() {
                len += code.len() + 2; // "[code]"
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
                fmt::push_uint(out, span.start_as_usize() as u64);
                out.push_str("..");
                fmt::push_uint(out, span.end_as_usize() as u64);
                out.push_str(": ");
            }
            (None, None) => {
                if diag.source().is_some() && !file.is_empty() {
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

    fn render(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String {
        let mut out = String::new();
        self.render_into(diag, cx, &mut out);
        out
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

    fn render_suppressed_note(&self, _count: usize) -> String {
        String::new()
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::{FileId, Span};
    use crate::diagnostics::DiagOpts;
    use crate::diagnostics::ErrCode;
    use crate::diagnostics::render::RenderCtx;

    fn ctx(opts: &DiagOpts) -> RenderCtx<'_> {
        RenderCtx::new(opts, None)
    }

    /// The reservation now uses the sanitized lengths directly, so the only
    /// property it relies on is that `push_flattened` never grows its input.
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
        ] {
            let sanitized = crate::diagnostics::sanitize_for_terminal_flatten(raw);
            let break_at = memchr::memchr3(b'\n', b'\r', b'\t', sanitized.as_bytes());
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
            let Ok(s) = std::str::from_utf8(&bytes) else {
                continue;
            };
            for keep_breaks in [true, false] {
                let sanitized = if keep_breaks {
                    crate::diagnostics::sanitize_for_terminal_flatten(s)
                } else {
                    crate::diagnostics::sanitize_for_terminal(s)
                };
                let mut out = String::new();
                push_flattened(
                    &mut out,
                    &sanitized,
                    memchr::memchr3(b'\n', b'\r', b'\t', sanitized.as_bytes()),
                );
                assert!(
                    out.len() <= sanitized.len(),
                    "push_flattened grew {s:?} (keep_breaks={keep_breaks}): \
                     {} > {}",
                    out.len(),
                    sanitized.len()
                );
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
        assert!(out.contains("<input>"), "{out}");
        assert!(out.contains("error[E030]"), "{out}");
        assert!(out.contains("boom"), "{out}");
    }

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
            ("a\t b", "a  b"),
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
