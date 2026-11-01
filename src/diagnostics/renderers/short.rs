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

/// Append `msg` with every line break removed — one pass, zero allocation.
///
/// A maximal run of `\n`/`\r` (so also `\r\n`) collapses to a single space;
/// runs at the very start or end of the message are dropped outright, so a
/// message that arrives with a trailing newline does not leave a dangling
/// space before the output's own `\n`.  This is what keeps the one-line
/// contract: a bare `\r` would otherwise make a terminal redraw the line,
/// and some line-oriented parsers would truncate it.
#[inline]
fn push_flattened(out: &mut String, msg: &str) {
    let bytes = msg.as_bytes();

    // Fast path: no break byte — straight copy.
    if memchr::memchr2(b'\n', b'\r', bytes).is_none() {
        out.push_str(msg);
        return;
    }

    // Empty segments are exactly the interior/edges of a break run, so
    // skipping them *is* the run-collapsing rule — no second buffer needed.
    let mut first = true;
    let mut start = 0;
    while start < bytes.len() {
        let offset = memchr::memchr2(b'\n', b'\r', &bytes[start..]);
        let end = match offset {
            Some(o) => start + o,
            None => bytes.len(),
        };
        let seg = &msg[start..end];
        if !seg.is_empty() {
            if !first {
                out.push(' ');
            }
            out.push_str(seg);
            first = false;
        }
        if offset.is_none() {
            break;
        }
        start = end + 1;
    }
}

impl Renderer for ShortRenderer {
    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
        let file = diag.file_name();
        let level = diag.level().as_str();
        // Borrow the code string directly; the `[E030]` framing and the
        // coloured tag are written straight into the output below, so no
        // intermediate `CompactString` is built.
        let code: Option<&str> = diag.code().map(|c| c.code());

        // Resolve the location *before* allocating: the 1-based line/col
        // digit widths feed the capacity computation, and every accessor
        // (`spans()`, `source()`, `message()`) is called once.
        let span = diag.spans().first();
        let linecol: Option<(usize, usize)> = match (span, diag.source()) {
            (Some(span), Some(source)) => {
                let (line, col) = cx.line_index(source).linecol_display(span.start_as_usize());
                Some((line + 1, col + 1))
            }
            _ => None,
        };

        // ---- Size the reservation to the byte ----------------------------
        //   location + tag + ": " + message + "\n"
        // Flattening only ever shrinks the message (break runs collapse to
        // one space), so `raw_msg.len()` is a safe upper bound and the
        // caller's buffer never reallocates.
        let raw_msg = diag.message();
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
            (None, None) => {}
        }
        if self.use_color {
            len += 4 + level.len() + 4; // "\x1b[1m" level "\x1b[0m"
            if let Some(code) = code {
                len += 5 + code.len() + 5; // "\x1b[2m[" code "]\x1b[0m"
            }
        } else {
            len += level.len();
            if let Some(code) = code {
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
                use std::fmt::Write as _;
                let _ = write!(out, "{span}");
                out.push_str(": ");
            }
            (None, None) => {}
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
            if let Some(code) = code {
                out.push_str("\x1b[2m[");
                out.push_str(code);
                out.push_str("]\x1b[0m");
            }
        } else {
            out.push_str(level);
            if let Some(code) = code {
                out.push('[');
                out.push_str(code);
                out.push(']');
            }
        }
        out.push_str(": ");

        // ---- Message, flattened in a single pass -------------------------
        // A multi-line message would break the one-line format, and a stray
        // `\r` would make a terminal redraw the line.  Flatten both, with no
        // `Cow`, no `replace`, and no scratch `String`.
        push_flattened(out, raw_msg);
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
