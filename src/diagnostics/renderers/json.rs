//! JSON renderer — streams a JSON array instead of buffering items.

use crate::diagnostics::Diagnostic;
use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::sink::DiagnosticSink;
use std::borrow::Cow;
use std::fmt::Write;
use std::io;

/// JSON renderer — outputs diagnostics as a JSON array.
/// Format inspired by Austral's `render_error_to_json`.
/// Builds JSON manually without external dependencies.
pub struct JsonRenderer {
    pretty: bool,
}

impl JsonRenderer {
    pub fn new(pretty: bool) -> Self {
        JsonRenderer { pretty }
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
    fn begin(&self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        out.write_str(if self.pretty { "[\n" } else { "[" })
    }

    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
        self.write_diagnostic(diag, cx, 1, out);
    }

    fn render(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String {
        let hint = diag.message.len()
            + 128
            + diag.code.as_ref().map_or(0, |c| c.code().len() + 64)
            + diag.file_name().len()
            + 16
            + diag.spans.first().map_or(0, |_| 64)
            + diag
                .labels
                .iter()
                .map(|l| l.message.len() + 80)
                .sum::<usize>()
            + diag
                .related_errors
                .iter()
                .map(|r| {
                    r.message.len()
                        + r.code.as_ref().map_or(0, |c| c.code().len())
                        + if r.span.is_some() { 40 } else { 0 }
                        + 32
                })
                .sum::<usize>()
            + diag
                .suggestions
                .iter()
                .map(|s| s.message.len() + 64)
                .sum::<usize>()
            + diag.help.as_ref().map_or(0, |h| h.len() + 16)
            + diag.call_chain.as_ref().map_or(0, |c| {
                c.entries()
                    .iter()
                    .map(|e| e.message.len() + 50)
                    .sum::<usize>()
            });
        let mut out = String::with_capacity(hint);
        self.render_into(diag, cx, &mut out);
        out
    }

    fn render_separator(&self) -> &str {
        if self.pretty { ",\n" } else { "," }
    }

    fn render_summary(&self, _errors: usize, _warnings: usize) -> String {
        String::new()
    }

    fn render_suppressed_note(&self, _count: usize) -> String {
        String::new()
    }

    fn end(&self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        out.write_str(if self.pretty { "\n]\n" } else { "]\n" })
    }
}

impl JsonRenderer {
    /// Append `s` to `out` as a JSON string literal (quotes included).
    ///
    /// The destination buffer is the only allocation involved: the common
    /// clean case is a single `push_str`, and the escaping path writes
    /// straight into `out` rather than building a temporary. That matters
    /// because a diagnostic's whole JSON is now assembled in one buffer, so
    /// values that exceed the 24-byte `CompactString` inline limit touch the
    /// heap exactly once instead of twice.
    fn push_json_str(out: &mut impl Write, s: &str) {
        // A single SWAR scan decides whether escaping is needed: the only
        // characters JSON must escape are the control bytes (`U+0000`–
        // `U+001F`, i.e. any byte `< 0x20`), `"` and `\`. UTF-8 continuation
        // bytes are all `>= 0x80`, so no multi-byte character can be
        // mistaken for one of them — and the scan catches `\n`, `\r`, `\t`
        // at any position, not only byte 0.  See [`needs_json_escape`] for
        // the chunk-wise SWAR identities.
        let needs_escape = needs_json_escape(s);
        let _ = out.write_char('"');
        if needs_escape {
            let bytes = s.as_bytes();
            let mut start = 0;
            let mut i = 0;
            while i < bytes.len() {
                let b = bytes[i];
                if b < 0x20 || b == b'"' || b == b'\\' {
                    if i > start {
                        let _ = out.write_str(&s[start..i]);
                    }
                    push_json_escaped(out, b as char);
                    start = i + 1;
                }
                i += 1;
            }
            if start < bytes.len() {
                let _ = out.write_str(&s[start..]);
            }
        } else {
            let _ = out.write_str(s);
        }
        let _ = out.write_char('"');
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

    /// Stream one diagnostic object into `out`.
    ///
    /// Fields are appended straight into the single output buffer in
    /// declaration order. The separator is written *before* each field, so a
    /// trailing comma is structurally impossible — no intermediate field
    /// list is needed to guarantee it.
    fn write_diagnostic(
        &self,
        diag: &Diagnostic<'_>,
        cx: &RenderCtx<'_>,
        indent_lvl: usize,
        out: &mut String,
    ) {
        let ind = Self::indent(indent_lvl);
        let ind1 = Self::indent(indent_lvl + 1);
        let mut obj = JsonContainer::object(out, &ind, self.pretty);

        // Stable identifier: computed from `(code, primary span, message)`
        // via FNV-1a so tools can dedupe/suppress across runs. Emitted
        // first so tools that read only the head of the object get it
        // without scanning the whole diagnostic.
        obj.key("id");
        obj.u64_hex(diag.stable_id());

        obj.key("level");
        obj.string(diag.level.as_str());

        obj.key("message");
        obj.string(&diag.message);

        if let Some(code) = &diag.code {
            obj.key("code");
            obj.string(code.code());
            if let Some(entry) = code.entry() {
                obj.key("title");
                obj.string(entry.title);
                obj.key("category");
                obj.string(entry.category.as_str());
            } else {
                obj.key("title");
                obj.raw("null");
                obj.key("category");
                obj.raw("null");
            }
        }

        // file (stable machine-readable id; the human header shows it too)
        obj.key("file");
        obj.string(diag.file_name());

        // span (+ 1-based line/column when the source is attached)
        if let Some(span) = diag.spans.first() {
            obj.key("span");
            {
                let mut sp = JsonContainer::compact_object(&mut *obj.out);
                sp.key("start");
                sp.u64(span.start as u64);
                sp.key("end");
                sp.u64(span.end as u64);
                if let Some(source) = diag.source() {
                    let (line, col) = cx.line_index(source).linecol_display(span.start_as_usize());
                    sp.key("line");
                    sp.u64((line + 1) as u64);
                    sp.key("column");
                    sp.u64((col + 1) as u64);
                }
                sp.close();
            }
        }

        // labels
        if !diag.labels.is_empty() {
            obj.key("labels");
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for lbl in &diag.labels {
                arr.element();
                let mut it = JsonContainer::compact_object(&mut *arr.out);
                it.key("start");
                it.u64(lbl.span.start as u64);
                it.key("end");
                it.u64(lbl.span.end as u64);
                it.key("file");
                it.u64(lbl.span.file.as_u32() as u64);
                it.key("message");
                it.string(&lbl.message);
                it.key("kind");
                it.string(crate::diagnostics::annotation_kind_name(lbl.kind));
                it.close();
            }
            arr.close();
        }

        // related errors (code, message, span) — one entry per `RelatedError`.
        if !diag.related_errors.is_empty() {
            obj.key("relatedErrors");
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for rel in &diag.related_errors {
                arr.element();
                let mut it = JsonContainer::compact_object(&mut *arr.out);
                it.key("code");
                match &rel.code {
                    Some(c) => it.string(c.code()),
                    None => it.raw("null"),
                }
                it.key("message");
                it.string(&rel.message);
                it.key("span");
                match rel.span.as_ref() {
                    Some(sp) => {
                        let mut o = JsonContainer::compact_object(&mut *it.out);
                        o.key("start");
                        o.u64(sp.start as u64);
                        o.key("end");
                        o.u64(sp.end as u64);
                        o.close();
                    }
                    None => it.raw("null"),
                }
                it.close();
            }
            arr.close();
        }

        // help
        if let Some(help) = &diag.help {
            obj.key("help");
            obj.string(help);
        }

        // suggestions
        if !diag.suggestions.is_empty() {
            obj.key("suggestions");
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for sug in &diag.suggestions {
                arr.element();
                let mut it = JsonContainer::compact_object(&mut *arr.out);
                it.key("message");
                it.string(&sug.message);
                it.key("applicability");
                it.string(crate::diagnostics::applicability_name(sug.applicability));
                it.key("style");
                it.string(crate::diagnostics::suggestion_style_name(sug.style));
                it.close();
            }
            arr.close();
        }

        // call chain — each entry carries its kind, the user message, and
        // (when a source is available and the entry has a span) its
        // file:line:col location, matching the terminal renderer.
        if let Some(chain) = &diag.call_chain
            && !chain.is_empty()
        {
            obj.key("callChain");
            // Reuse the per-batch LRU line index (like the span path above and
            // every other renderer) instead of rebuilding a fresh `LineIndex`
            // per diagnostic — a batch of M diagnostics with a call chain over
            // one n-byte source would otherwise pay O(M·n) instead of O(n).
            let idx = diag.source.as_ref().map(|s| cx.line_index(s));
            let file_name = diag.file_name();
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for entry in chain.entries() {
                arr.element();
                // At most three parts (label, kind, location), streamed
                // straight into the output buffer.
                let mut it = JsonContainer::compact_object(&mut *arr.out);
                it.key("label");
                it.string(&entry.message);
                it.key("kind");
                it.string(crate::diagnostics::chain_entry_kind_name(entry.kind));
                if let (Some(idx), Some(span)) = (idx.as_ref(), entry.span) {
                    let (line, col) = idx.linecol_display(span.start_as_usize());
                    it.key("location");
                    it.location(file_name, (line + 1) as u64, (col + 1) as u64);
                }
                it.close();
            }
            arr.close();
        }

        obj.close();
    }
}

/// Streams one JSON container (object or array) into a single buffer.
///
/// Structural characters — separators, newlines, indentation, the closing
/// bracket — are written as elements are appended, so serializing a
/// diagnostic costs exactly one allocation (the output `String`) regardless
/// of how many fields, labels or suggestions it carries: nothing is
/// collected into a `Vec`, and no value is ever formatted twice.
struct JsonContainer<'a> {
    out: &'a mut String,
    /// Separator written before every element but the first.
    sep: &'static str,
    /// Indentation written before each element in pretty mode.
    elem_ind: &'a str,
    /// Indentation written before the closing bracket in pretty mode.
    close_ind: &'a str,
    pretty: bool,
    first: bool,
    close: char,
}

impl<'a> JsonContainer<'a> {
    /// `{` … `}` whose fields sit at `ind` — the diagnostic body.
    fn object(out: &'a mut String, ind: &'a str, pretty: bool) -> Self {
        out.push('{');
        Self {
            out,
            sep: ",",
            elem_ind: ind,
            close_ind: ind,
            pretty,
            first: true,
            close: '}',
        }
    }

    /// `[` … `]` whose elements sit at `elem_ind` and whose bracket closes at
    /// `close_ind`.
    fn array(out: &'a mut String, elem_ind: &'a str, close_ind: &'a str, pretty: bool) -> Self {
        out.push('[');
        Self {
            out,
            sep: ",",
            elem_ind,
            close_ind,
            pretty,
            first: true,
            close: ']',
        }
    }

    /// `{"k": v, "k": v}` — always compact: array elements render as one
    /// line each regardless of the pretty setting (matching the previous
    /// per-item output exactly).
    fn compact_object(out: &'a mut String) -> Self {
        out.push('{');
        Self {
            out,
            sep: ", ",
            elem_ind: "",
            close_ind: "",
            pretty: false,
            first: true,
            close: '}',
        }
    }

    /// Begin the next element: separator first, then (in pretty mode) the
    /// newline and indentation.
    fn element(&mut self) {
        if self.first {
            self.first = false;
        } else {
            self.out.push_str(self.sep);
        }
        if self.pretty {
            self.out.push('\n');
            self.out.push_str(self.elem_ind);
        }
    }

    /// Begin an object field: [`element`](Self::element) plus `"key": `.
    fn key(&mut self, key: &str) {
        self.element();
        self.out.push('"');
        self.out.push_str(key);
        self.out.push_str("\": ");
    }

    /// Value: a JSON string literal.
    fn string(&mut self, s: &str) {
        JsonRenderer::push_json_str(self.out, s);
    }

    /// Value: pre-rendered JSON (`null`, a small inline object, …).
    fn raw(&mut self, s: &str) {
        self.out.push_str(s);
    }

    /// Value: a bare `u64` (JSON number) — no quoting, no `core::fmt`.
    fn u64(&mut self, n: u64) {
        super::fmt::push_uint(self.out, n);
    }

    /// Value: a JSON string literal `"0x{:016x}"` — the stable diagnostic id.
    fn u64_hex(&mut self, n: u64) {
        self.out.push_str("\"0x");
        super::fmt::push_hex64(self.out, n);
        self.out.push('"');
    }

    /// Value: a JSON string `"file:line:col"` (or `"line:col"` when file is
    /// empty) — the file portion may need escaping, line/col are ASCII digits.
    fn location(&mut self, file: &str, line: u64, col: u64) {
        self.out.push('"');
        if !file.is_empty() {
            if needs_json_escape(file) {
                for c in file.chars() {
                    push_json_escaped(self.out, c);
                }
            } else {
                self.out.push_str(file);
            }
            self.out.push(':');
        }
        super::fmt::push_uint(self.out, line);
        self.out.push(':');
        super::fmt::push_uint(self.out, col);
        self.out.push('"');
    }

    /// Close the container, indenting the bracket in pretty mode.
    fn close(self) {
        if self.pretty && !self.first {
            self.out.push('\n');
            self.out.push_str(self.close_ind);
        }
        self.out.push(self.close);
    }
}

/// Does `s` contain a byte that JSON must escape — a control byte (`< 0x20`),
/// `"` or `\`?  One SWAR pass per 8-byte chunk instead of a scalar per-byte
/// scan, so the common clean-string check (the hot path of
/// [`JsonRenderer::push_json_str`]) costs a few ALU ops per chunk rather than
/// ~8 branch-taken per byte.
///
/// `b < 0x20` uses Mycroft's `hasless` trick `((x - SP) & ~x & HI)`; `b == '"'`
/// and `b == '\\'` use the `has_zero` trick `((q - LO) & ~q & HI)` on
/// `q = x ^ V_broadcast`.  Each is correct for the "any byte" predicate: a
/// cross-byte borrow can only produce a false positive at a byte equal to the
/// threshold, which requires a preceding byte to have already triggered a true
/// positive, so the overall `!= 0` answer stays sound.  UTF-8 continuation
/// bytes are all `≥ 0x80`, so no multi-byte character is mistaken for an
/// escape candidate.
#[inline]
fn needs_json_escape(s: &str) -> bool {
    const HI: u64 = 0x8080_8080_8080_8080;
    const LO: u64 = 0x0101_0101_0101_0101;
    const SP: u64 = 0x2020_2020_2020_2020;
    const QU: u64 = 0x2222_2222_2222_2222;
    const BS: u64 = 0x5C5C_5C5C_5C5C_5C5C;

    let b = s.as_bytes();
    let mut i = 0;
    while i + 8 <= b.len() {
        let x = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let ctrl = (x.wrapping_sub(SP) & !x & HI) != 0;
        let q = x ^ QU;
        let quote = (q.wrapping_sub(LO) & !q & HI) != 0;
        let r = x ^ BS;
        let backslash = (r.wrapping_sub(LO) & !r & HI) != 0;
        if ctrl || quote || backslash {
            return true;
        }
        i += 8;
    }
    b[i..].iter().any(|&c| c < 0x20 || c == b'"' || c == b'\\')
}

/// Append one character to `out`, JSON-escaped. Used by
/// [`JsonRenderer::push_json_str`] so escaping goes straight into the
/// caller's buffer without an intermediate copy.
///
/// Generic over the output so escaping can go straight into whatever buffer
/// the caller is building — `String` or `CompactString` alike.
fn push_json_escaped<W: std::fmt::Write>(out: &mut W, c: char) {
    let _ = match c {
        '"' => out.write_str("\\\""),
        '\\' => out.write_str("\\\\"),
        '\n' => out.write_str("\\n"),
        '\r' => out.write_str("\\r"),
        '\t' => out.write_str("\\t"),
        c if c.is_control() => {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let n = c as u32;
            let buf = [
                b'\\',
                b'u',
                HEX[((n >> 12) & 0xf) as usize],
                HEX[((n >> 8) & 0xf) as usize],
                HEX[((n >> 4) & 0xf) as usize],
                HEX[(n & 0xf) as usize],
            ];
            out.write_str(std::str::from_utf8(&buf).unwrap())
        }
        c => out.write_char(c),
    };
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::{FileId, Span};
    use crate::diagnostics::ErrCode;
    use crate::diagnostics::chain::{CallChain, ChainEntry, ChainEntryKind};
    use compact_str::CompactString;

    fn json_of(diag: &Diagnostic<'_>, pretty: bool) -> String {
        let opts = crate::diagnostics::DiagOpts::default();
        let cx = crate::diagnostics::render::RenderCtx::new(&opts, None);
        let mut out = String::new();
        JsonRenderer::new(pretty).write_diagnostic(diag, &cx, 0, &mut out);
        out
    }

    /// Escaping must keep short values inline (no heap) on **both** paths —
    /// the clean fast path and the escaping slow path. This pins the inline
    /// guarantee that the single-buffer assembly relies on: a diagnostic's
    /// JSON is now built in one `String`, and short literals never push it
    /// through an intermediate `CompactString` on the heap.
    #[test]
    fn push_json_str_inlines_short_values() {
        fn render(s: &str) -> CompactString {
            let mut out = CompactString::with_capacity(s.len() + 2);
            JsonRenderer::push_json_str(&mut out, s);
            out
        }

        // Fast path: no characters need escaping.
        let clean = render("error");
        assert!(!clean.is_heap_allocated(), "clean short value on heap");
        assert_eq!(clean.as_str(), "\"error\"");

        // Slow path: contains a character that must be escaped, but the
        // escaped result is still ≤ 24 bytes.
        let escaped = render("say \"hi\"");
        assert!(
            !escaped.is_heap_allocated(),
            "escaped short value on heap: {escaped:?}"
        );
        let v: serde_json::Value = serde_json::from_str(&escaped).expect("valid JSON");
        assert_eq!(v, serde_json::Value::String("say \"hi\"".to_string()));

        // A value too long to inline still round-trips (and is heap-backed).
        let long = render(&"a\n".repeat(40));
        assert!(long.is_heap_allocated(), "long value should be heap-backed");
        serde_json::from_str::<serde_json::Value>(&long).expect("valid JSON");
    }

    /// Control characters beyond the first byte must be escaped. The fast
    /// path's guard must detect a control char at *any* position, not only
    /// byte 0 — otherwise `\x07` (or any non-`\n`/`\r`/`\t` control) past the
    /// start is emitted raw and produces invalid JSON.
    #[test]
    fn push_json_str_escapes_non_first_control_char() {
        for ctrl in ['\u{0}', '\u{7}', '\u{1f}'] {
            let input = format!("a{ctrl}b");
            let mut escaped = String::new();
            JsonRenderer::push_json_str(&mut escaped, &input);
            let v: serde_json::Value = serde_json::from_str(&escaped)
                .unwrap_or_else(|e| panic!("invalid JSON for U+{:04X}: {e}", ctrl as u32));
            assert_eq!(v, serde_json::Value::String(input), "U+{:04X}", ctrl as u32);
            let needle = format!("\\u{:04x}", ctrl as u32);
            assert!(
                escaped.contains(&needle),
                "U+{:04X} not escaped as {needle}: {escaped}",
                ctrl as u32
            );
        }
    }

    #[test]
    fn json_round_trips_message_with_control_char() {
        let diag = Diagnostic::error("a\x07b").with_span(Span::new(FileId::DUMMY, 0, 1));
        let json = json_of(&diag, false);
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["message"], "a\u{7}b");
    }

    #[test]
    fn test_json_is_valid_with_no_trailing_comma() {
        let diag = Diagnostic::error("cannot find value `x` here")
            .with_code_str(ErrCode::E012)
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
    fn test_json_call_chain_kind_is_quoted() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 0, 1)),
            "main",
        ));
        let diag = Diagnostic::error("x").with_call_chain(chain);
        let json = json_of(&diag, false);
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["callChain"][0]["kind"], "Call");
    }

    #[test]
    fn stable_id_is_deterministic_and_covers_code_span_message() {
        let a = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 10, 15));
        let b = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 10, 15));
        let different_code = Diagnostic::error("boom")
            .with_code_str(ErrCode::E031)
            .with_span(Span::new(FileId::DUMMY, 10, 15));
        let different_message = Diagnostic::error("bang")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 10, 15));
        assert_eq!(a.stable_id(), b.stable_id(), "identical inputs must agree");
        assert_ne!(a.stable_id(), different_code.stable_id());
        assert_ne!(a.stable_id(), different_message.stable_id());
        // Present in JSON output.
        let json = json_of(&a, false);
        assert!(json.contains("\"id\": \"0x"), "missing id field: {json}");
    }

    #[test]
    fn streams_valid_array_of_multiple_items() {
        use crate::diagnostics::DiagOpts;
        use crate::diagnostics::emitter::Emitter;
        use crate::diagnostics::sink::{StringSink, string};

        let mut diags = vec![
            Diagnostic::error("one").with_code_str(ErrCode::E030),
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

    /// The SWAR `needs_json_escape` must agree with the scalar predicate it
    /// replaces for every byte value and at every 8-byte chunk boundary — the
    /// `hasless`/`has_zero` tricks carry cross-byte borrows that are only
    /// sound for the "any byte" predicate, so this pins them exhaustively.
    #[test]
    fn needs_json_escape_matches_scalar() {
        fn scalar(s: &str) -> bool {
            s.bytes().any(|b| b < 0x20 || b == b'"' || b == b'\\')
        }
        // Every ASCII single byte (0x00..=0x7F) as a 1-byte string — all are
        // valid UTF-8.  Multi-byte continuations can't be a single-byte `&str`.
        for byte in 0u8..0x80 {
            let buf = [byte];
            let s = std::str::from_utf8(&buf).unwrap();
            assert_eq!(needs_json_escape(s), scalar(s), "single byte {byte:#04x}");
        }
        // Escape candidate at each position 0..=16, padded clean on both sides
        // — hits every offset within and across 8-byte chunks.
        let clean = "aaaaaaaaaaaaaaaaa"; // 17 clean bytes
        for pos in 0..=16usize {
            for cand in [b'\n', b'\r', b'\t', b'"', b'\\', 0x00u8, 0x1f, 0x20, 0x7f] {
                let mut bytes = clean.as_bytes().to_vec();
                bytes[pos] = cand;
                let s = String::from_utf8(bytes).unwrap();
                assert_eq!(
                    needs_json_escape(&s),
                    scalar(&s),
                    "cand {cand:#04x} at pos {pos}"
                );
            }
        }
        // Lengths straddling the 8-byte chunk boundary with clean content.
        for len in 0..=17 {
            let s = &"abcdefghijklmnopqrstuvwxyz"[..len];
            assert_eq!(needs_json_escape(s), scalar(s), "clean len {len}");
        }
        // A long clean string (multiple full chunks) and one dirty at the end.
        let long_clean = &"a".repeat(200);
        assert_eq!(needs_json_escape(long_clean), scalar(long_clean));
        let long_dirty = format!("{}{}", "a".repeat(200), "\\");
        assert_eq!(needs_json_escape(&long_dirty), scalar(&long_dirty));
    }
}
