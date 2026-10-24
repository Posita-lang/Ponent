//! JSON renderer — streams a JSON array instead of buffering items.

use crate::ast::FileId;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::LineIndex;
use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::sink::DiagnosticSink;
use compact_str::{CompactString, format_compact};
use std::borrow::Cow;
use std::fmt::Write;
use std::io;

/// JSON renderer — outputs diagnostics as a JSON array.
/// Format inspired by Austral's `render_error_to_json`.
/// Builds JSON manually without external dependencies.
pub struct JsonRenderer {
    pretty: bool,
    /// Whether the next item is the first (controls the separator).
    first: bool,
}

impl JsonRenderer {
    pub fn new(pretty: bool) -> Self {
        JsonRenderer {
            pretty,
            first: true,
        }
    }

    pub fn new_pretty() -> Self {
        JsonRenderer::new(true)
    }

    pub fn new_compact() -> Self {
        JsonRenderer::new(false)
    }
}

impl Default for JsonRenderer {
    fn default() -> Self {
        Self::new_compact()
    }
}

impl Renderer for JsonRenderer {
    fn begin(&mut self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        self.first = true;
        out.write_str(if self.pretty { "[\n" } else { "[" })
    }

    fn render(&mut self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String {
        let sep = if self.first {
            ""
        } else if self.pretty {
            ",\n"
        } else {
            ","
        };
        self.first = false;
        format!("{sep}{}", self.diagnostic_to_json(diag, cx, 1))
    }

    fn render_summary(&mut self, _errors: usize, _warnings: usize) -> String {
        String::new()
    }

    fn render_suppressed_note(&mut self, _count: usize) -> String {
        String::new()
    }

    fn end(&mut self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        out.write_str(if self.pretty { "\n]\n" } else { "]\n" })
    }
}

impl JsonRenderer {
    /// Escape a string for JSON embedding, returning it wrapped in double
    /// quotes. Clean strings skip the per-character loop via a fast-path
    /// byte scan.
    ///
    /// The result is a [`CompactString`] rather than a `Cow` (it always needs
    /// the surrounding quotes, so it can never borrow) — short values like
    /// `"error"` / `"E030"` then stay inline instead of hitting the heap.
    ///
    /// Both paths build the `CompactString` directly so a value that fits
    /// inline (≤24 bytes) never touches the heap at all: the fast path via
    /// `format_compact!`, the escape path by writing straight into the
    /// buffer rather than allocating a `String` and converting it.
    fn escape_json(s: &str) -> CompactString {
        // Fast-path for the common case of well-formed ASCII/UTF-8 text that
        // contains no characters requiring escaping.
        //
        // Control characters (including `\u{0000}`) start with a byte < 0x20,
        // so we check the first byte upfront to skip the full `.bytes().any()`
        // scan and go straight to the slow path — avoiding two passes over the
        // input.
        let needs_escape = if let Some(b) = s.bytes().next() {
            b < 0x20
                || s.bytes()
                    .skip(1)
                    .any(|b| b == b'"' || b == b'\\' || b == b'\n' || b == b'\r' || b == b'\t')
        } else {
            false
        };
        if !needs_escape {
            return format_compact!("\"{s}\"");
        }
        let mut out = CompactString::with_capacity(s.len() + 2);
        out.push('"');
        for ch in s.chars() {
            push_json_escaped(&mut out, ch);
        }
        out.push('"');
        out
    }

    /// Indentation for `level` nesting steps.
    ///
    /// Diagnostic JSON nests at most a couple of levels deep, so the common
    /// levels borrow a `'static` slice instead of allocating; only deeper
    /// nesting (which this renderer does not currently produce) falls back
    /// to an owned `"  ".repeat(level)`.
    fn indent(level: usize) -> Cow<'static, str> {
        const INDENTS: &[&str] = &["", "  ", "    ", "      ", "        ", "          "];
        match INDENTS.get(level) {
            Some(s) => Cow::Borrowed(s),
            None => Cow::Owned("  ".repeat(level)),
        }
    }

    /// Serialize a `Debug` value as a *quoted* JSON string.
    /// `{:?}` alone renders enum variants as bare identifiers (`Primary`),
    /// which is not valid JSON.
    ///
    /// The `Debug` rendering is streamed through [`JsonDebugString`] and
    /// escaped as it is produced, so this allocates once (the output) instead
    /// of once for the `{:?}` buffer and again for the escaped string.
    fn debug_quoted<T: std::fmt::Debug>(v: &T) -> String {
        let mut out = String::with_capacity(64);
        out.push('"');
        {
            let mut w = JsonDebugString {
                out: &mut out,
                started: false,
                saw_open_quote: false,
                held: None,
                count: 0,
            };
            let _ = write!(w, "{v:?}");
            if let Some(h) = w.held {
                // Drop the closing quote only when a matching opening quote
                // was stripped and there was content between them.
                if !(h == '"' && w.saw_open_quote && w.count >= 2) {
                    push_json_escaped(w.out, h);
                }
            } else if w.saw_open_quote && w.count == 1 {
                // A lone `"`: `strip_prefix(..).and_then(strip_suffix(..))`
                // leaves it unstripped, so emit it escaped.
                push_json_escaped(w.out, '"');
            }
        }
        out.push('"');
        out
    }

    /// Join already-serialized items into a JSON array body, respecting the
    /// pretty/compact setting.  Generic over the item type so both `String`
    /// and `CompactString` field values can be passed.
    fn array<S: AsRef<str>>(&self, items: &[S], indent_lvl: usize) -> String {
        if items.is_empty() {
            return "[]".to_string();
        }
        if self.pretty {
            let ind = Self::indent(indent_lvl);
            let ind1 = Self::indent(indent_lvl + 1);
            // Build the final string directly in one pass — `format!` would
            // copy the whole body a second time.  No per-field temporary, no
            // intermediate `Vec`.
            let mut body = String::with_capacity(
                items.iter().map(|it| it.as_ref().len()).sum::<usize>()
                    + items.len() * (ind1.len() + 3)
                    + ind.len()
                    + 3,
            );
            body.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    body.push(',');
                }
                body.push('\n');
                body.push_str(&ind1);
                body.push_str(item.as_ref());
            }
            body.push('\n');
            body.push_str(&ind);
            body.push(']');
            body
        } else {
            // Build the compact body directly as well — no temporary joined
            // `String`, and it works for any `AsRef<str>` item type.
            let mut body = String::with_capacity(
                items.iter().map(|it| it.as_ref().len() + 1).sum::<usize>() + 2,
            );
            body.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    body.push(',');
                }
                body.push_str(item.as_ref());
            }
            body.push(']');
            body
        }
    }

    fn diagnostic_to_json(
        &self,
        diag: &Diagnostic<'_>,
        cx: &RenderCtx<'_>,
        indent_lvl: usize,
    ) -> String {
        // Collect the fields first, then join with commas — this makes a
        // trailing comma structurally impossible.  At most twelve fields can
        // be pushed (level, message, code, title, category, file, span,
        // labels, relatedErrors, help, suggestions, callChain), so reserve
        // the full count up front — no reallocation at any field count.
        let mut fields: Vec<String> = Vec::with_capacity(12);

        // level
        fields.push(format!(
            "\"level\": {}",
            Self::escape_json(diag.level.as_str())
        ));

        // message
        fields.push(format!("\"message\": {}", Self::escape_json(&diag.message)));

        // code
        if let Some(ref code) = diag.code {
            fields.push(format!("\"code\": {}", Self::escape_json(code.code())));
            if let Some(entry) = code.entry() {
                fields.push(format!("\"title\": {}", Self::escape_json(entry.title)));
                fields.push(format!(
                    "\"category\": {}",
                    Self::escape_json(entry.category.as_str())
                ));
            } else {
                fields.push("\"title\": null".to_string());
                fields.push("\"category\": null".to_string());
            }
        }

        // file (stable machine-readable id; the human header shows it too)
        fields.push(format!("\"file\": {}", Self::escape_json(diag.file_name())));

        // span (+ 1-based line/column when the source is attached)
        if let Some(span) = diag.spans.first() {
            if let Some(source) = diag.source() {
                let pos = cx.line_index(source).to_linecol(span.start_as_usize());
                fields.push(format!(
                    "\"span\": {{\"start\": {}, \"end\": {}, \"line\": {}, \"column\": {}}}",
                    span.start,
                    span.end,
                    pos.line + 1,
                    pos.col + 1
                ));
            } else {
                fields.push(format!(
                    "\"span\": {{\"start\": {}, \"end\": {}}}",
                    span.start, span.end
                ));
            }
        }

        // labels
        if !diag.labels.is_empty() {
            let items: Vec<String> = diag
                .labels
                .iter()
                .map(|lbl| {
                    format!(
                        "{{\"start\": {}, \"end\": {}, \"message\": {}, \"kind\": {}}}",
                        lbl.span.start,
                        lbl.span.end,
                        Self::escape_json(&lbl.message),
                        Self::debug_quoted(&lbl.kind),
                    )
                })
                .collect();
            fields.push(format!("\"labels\": {}", self.array(&items, indent_lvl)));
        }

        // related errors (code, message, span) — one entry per `RelatedError`.
        if !diag.related_errors.is_empty() {
            let items: Vec<String> = diag
                .related_errors
                .iter()
                .map(|rel| {
                    let code_part = rel
                        .code
                        .as_ref()
                        .map(|c| format!("\"code\": {}", Self::escape_json(c.code())))
                        .unwrap_or_else(|| "\"code\": null".to_string());
                    let span_part = rel.span.as_ref().map(|sp| {
                        format!("\"span\": {{\"start\": {}, \"end\": {}}}", sp.start, sp.end)
                    }).unwrap_or_else(|| "\"span\": null".to_string());
                    format!(
                        "{{{}, \"message\": {}, {}}}",
                        code_part,
                        Self::escape_json(&rel.message),
                        span_part,
                    )
                })
                .collect();
            fields.push(format!(
                "\"relatedErrors\": {}",
                self.array(&items, indent_lvl)
            ));
        }

        // help
        if let Some(ref help) = diag.help {
            fields.push(format!("\"help\": {}", Self::escape_json(help)));
        }

        // suggestions
        if !diag.suggestions.is_empty() {
            let items: Vec<CompactString> = diag
                .suggestions
                .iter()
                .map(|sug| {
                    let msg = Self::escape_json(&sug.message);
                    CompactString::from(format!(
                        "{{\"message\": {msg}, \"applicability\": {app}, \"style\": {style}}}",
                        msg = msg,
                        app = Self::escape_json(crate::diagnostics::applicability_name(sug.applicability)),
                        style = Self::escape_json(crate::diagnostics::suggestion_style_name(sug.style)),
                    ))
                })
                .collect();
            fields.push(format!(
                "\"suggestions\": {}",
                self.array(&items, indent_lvl)
            ));
        }

        // call chain — each entry carries its kind, the user message, and
        // (when a source is available and the entry has a span) its
        // file:line:col location, matching the terminal renderer.
        if let Some(ref chain) = diag.call_chain
            && !chain.is_empty()
        {
            let source = diag.source.as_ref();
            let file_name = diag.file_name().to_string();
            let idx = source.map(|s| LineIndex::new(s));
            let items: Vec<String> = chain
                .entries()
                .iter()
                .map(|entry| {
                    let mut parts: Vec<String> = Vec::with_capacity(3);
                    parts.push(format!(
                        "\"label\": {}",
                        Self::escape_json(&entry.message)
                    ));
                    parts.push(format!(
                        "\"kind\": {}",
                        Self::debug_quoted(&entry.kind)
                    ));
                    if let (Some(_src), Some(span)) = (source, entry.span) {
                        let pos = idx
                            .as_ref()
                            .expect("idx built whenever source present")
                            .to_linecol(span.start_as_usize());
                        let location = if file_name.is_empty() {
                            format!("{}:{}", pos.line + 1, pos.col + 1)
                        } else {
                            format!("{}:{}:{}", file_name, pos.line + 1, pos.col + 1)
                        };
                        parts.push(format!(
                            "\"location\": {}",
                            Self::escape_json(&location)
                        ));
                    }
                    format!("{{{}}}", parts.join(", "))
                })
                .collect();
            fields.push(format!("\"callChain\": {}", self.array(&items, indent_lvl)));
        }

        // Assemble the object — commas only between fields.
        if self.pretty {
            let ind = Self::indent(indent_lvl);
            // One pass into a single buffer: no per-field temporary, no
            // intermediate `Vec`, and no second copy of the whole body the
            // way `format!` would make one.
            let mut body = String::with_capacity(
                fields.iter().map(|f| f.len()).sum::<usize>() + fields.len() * (ind.len() + 3) + 3,
            );
            body.push('{');
            for (i, f) in fields.iter().enumerate() {
                if i > 0 {
                    body.push(',');
                }
                body.push('\n');
                body.push_str(&ind);
                body.push_str(f);
            }
            body.push('\n');
            body.push_str(&ind);
            body.push('}');
            body
        } else {
            // Build the object directly: `fields.join(",")` would allocate a
            // second `String` for the joined body only to copy it again.
            let mut body =
                String::with_capacity(fields.iter().map(|f| f.len() + 1).sum::<usize>() + 2);
            body.push('{');
            for (i, f) in fields.iter().enumerate() {
                if i > 0 {
                    body.push(',');
                }
                body.push_str(f);
            }
            body.push('}');
            body
        }
    }
}

/// Append one character to `out`, JSON-escaped. Shared by [`JsonRenderer::escape_json`]
/// and the streaming [`JsonDebugString`] adapter so both escape identically.
///
/// Generic over the output so `escape_json` can write straight into a
/// `CompactString` (keeping short results inline) while `JsonDebugString`
/// keeps using a `String`.
fn push_json_escaped<W: std::fmt::Write>(out: &mut W, c: char) {
    let _ = match c {
        '"' => out.write_str("\\\""),
        '\\' => out.write_str("\\\\"),
        '\n' => out.write_str("\\n"),
        '\r' => out.write_str("\\r"),
        '\t' => out.write_str("\\t"),
        c if c.is_control() => write!(out, "\\u{:04x}", c as u32),
        c => out.write_char(c),
    };
}

/// Streams a `Debug` rendering into a JSON string, escaping as it is produced
/// and stripping the single pair of surrounding quotes that `{:?}` adds around
/// string-like values — matching
/// `strip_prefix('"').and_then(|t| t.strip_suffix('"'))` exactly.
///
/// The most recent character is held back so a trailing closing quote can be
/// dropped once the whole rendering is known; `count` distinguishes a genuine
/// quote pair from a lone `"`.
struct JsonDebugString<'a> {
    out: &'a mut String,
    started: bool,
    saw_open_quote: bool,
    held: Option<char>,
    count: usize,
}

impl std::fmt::Write for JsonDebugString<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        for c in s.chars() {
            self.count += 1;
            if !self.started {
                self.started = true;
                if c == '"' {
                    self.saw_open_quote = true;
                    continue;
                }
                self.held = Some(c);
                continue;
            }
            if let Some(h) = self.held.take() {
                push_json_escaped(self.out, h);
            }
            self.held = Some(c);
        }
        Ok(())
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::diagnostics::Diagnostic;
    use crate::diagnostics::chain::{CallChain, ChainEntry, ChainEntryKind};

    fn json_of(diag: &Diagnostic<'_>, pretty: bool) -> String {
        let opts = crate::diagnostics::DiagOpts::default();
        let cx = crate::diagnostics::render::RenderCtx::new(&opts, None);
        JsonRenderer::new(pretty).diagnostic_to_json(diag, &cx, 0)
    }

    /// `escape_json` must keep short values inline (no heap) on **both**
    /// paths — the clean fast path and the escaping slow path.  This pins the
    /// `CompactString` inline guarantee that the whole migration relies on.
    #[test]
    fn escape_json_inlines_short_values() {
        // Fast path: no characters need escaping.
        let clean = JsonRenderer::escape_json("error");
        assert!(!clean.is_heap_allocated(), "clean short value on heap");

        // Slow path: contains a character that must be escaped, but the
        // escaped result is still ≤ 24 bytes.
        let escaped = JsonRenderer::escape_json("say \"hi\"");
        assert!(
            !escaped.is_heap_allocated(),
            "escaped short value on heap: {escaped:?}"
        );
        let v: serde_json::Value = serde_json::from_str(&escaped).expect("valid JSON");
        assert_eq!(v, serde_json::Value::String("say \"hi\"".to_string()));

        // A value too long to inline still round-trips (and is heap-backed).
        let long = JsonRenderer::escape_json(&"a\n".repeat(40));
        assert!(long.is_heap_allocated(), "long value should be heap-backed");
        serde_json::from_str::<serde_json::Value>(&long).expect("valid JSON");
    }

    #[test]
    fn test_json_is_valid_with_no_trailing_comma() {
        let diag = Diagnostic::error("cannot find value `x` here")
            .with_code_str("E012")
            .with_span(Span::new(FileId::DUMMY, 12, 13))
            .with_label(Span::new(FileId::DUMMY, 12, 13), "not found");
        let json = json_of(&diag, false);
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["labels"][0]["kind"], "Primary");
        assert!(!json.contains(",}"), "trailing comma: {json}");
    }

    #[test]
    fn test_json_pretty_vs_compact() {
        let diag = Diagnostic::error("x").with_span(Span::new(FileId::DUMMY, 0, 1));
        let compact = json_of(&diag, false);
        let pretty = json_of(&diag, true);
        assert!(!compact.contains('\n'), "compact had newlines: {compact}");
        assert!(pretty.contains('\n'), "pretty had no newlines: {pretty}");
        serde_json::from_str::<serde_json::Value>(&compact).expect("compact valid");
        serde_json::from_str::<serde_json::Value>(&pretty).expect("pretty valid");
    }

    #[test]
    fn debug_quoted_handles_embedded_quotes() {
        // Enum variant (no surrounding quotes) → quoted verbatim.
        assert_eq!(
            JsonRenderer::debug_quoted(&ChainEntryKind::Call),
            "\"Call\""
        );
        // String containing a quote: `Debug` is `"\""`; stripping exactly one
        // pair must preserve the embedded quote, and the result must parse.
        let q = JsonRenderer::debug_quoted(&String::from("\""));
        let v: serde_json::Value = serde_json::from_str(&q).expect("valid JSON");
        assert_eq!(v, serde_json::Value::String("\\\"".to_string()));
    }

    #[test]
    fn test_json_call_chain_kind_is_quoted() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry {
            kind: ChainEntryKind::Call,
            span: Some(Span::new(FileId::DUMMY, 0, 1)),
            message: "main".into(),
        });
        let diag = Diagnostic::error("x").with_call_chain(chain);
        let json = json_of(&diag, false);
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["callChain"][0]["kind"], "Call");
    }

    #[test]
    fn streams_valid_array_of_multiple_items() {
        use crate::diagnostics::DiagOpts;
        use crate::diagnostics::emitter::Emitter;
        use crate::diagnostics::sink::{StringSink, string};

        let mut diags = vec![
            Diagnostic::error("one").with_code_str("E030"),
            Diagnostic::warning("two"),
        ];
        for pretty in [false, true] {
            let mut emitter =
                Emitter::new(JsonRenderer::new(pretty), string(), DiagOpts::default());
            let mut sink = StringSink::new();
            emitter.emit_all_to(&mut diags, &mut sink).unwrap();
            let out = sink.into_string();
            let v: serde_json::Value = serde_json::from_str(&out)
                .unwrap_or_else(|e| panic!("invalid JSON ({pretty}): {e}\n{out}"));
            assert_eq!(v.as_array().unwrap().len(), 2, "out: {out}");
        }
    }
}
