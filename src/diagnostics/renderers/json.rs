//! JSON renderer — streams a JSON array instead of buffering items.

use crate::diagnostics::Diagnostic;
use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::sink::DiagnosticSink;
use std::borrow::Cow;
use std::fmt::Write;
use std::io;

/// JSON renderer — outputs diagnostics as a JSON array.
/// Format inspired by Austral's `render_error_to_json`.
///
/// # Contract
///
/// The bytes are assembled by hand into one streaming buffer (no intermediate
/// document tree).  `memchr` is the **only** external dependency, used solely by
/// the long-string escape scanner; the short-string path is pure SWAR.
///
/// * **`code` is absent, not `null`, when a diagnostic has no code at all.**
///   `null` is reserved for a *present* code that the registry does not know
///   (`title`/`category` become `null`), so the two states stay distinguishable
///   without a sentinel.  Neither state requires a second consumer code path:
///   both "absent" and "not a string" are handled by the same optional lookup.
/// * **`line`/`column` are 1-based and derived from `LineIndex::linecol_display`,
///   i.e. `column` counts *display* columns** (tab expansion and wide-character
///   width are already applied), **not bytes, code points or UTF-16 units.**
///   A consumer that needs LSP/UTF-16 offsets must convert from the byte
///   `start` offset.  Every section resolves coordinates through
///   `anchor_linecol`, so the policy (and the EOF clamping) is uniform.
/// * **`additionalSpans`, `labels`, `suggestions[].span`, `children.span`,
///   `children.labels` and `callChain` all carry the same field names** for
///   offsets and coordinates; `file` is a resolved name and `fileId` a numeric
///   fallback for a file the diagnostic cannot name.
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
    fn begin(&self, out: &mut dyn DiagnosticSink) -> io::Result<()> {
        out.write_str(if self.pretty { "[\n" } else { "[" })
    }

    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
        // Batch output is a JSON array; indent the object one level under `[`
        // (fields one level deeper still), which is what every pretty-printer
        // and `jq` produce.
        self.write_diagnostic(diag, cx, 2, out);
    }

    fn render(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String {
        // One allocation: the same estimator the batch path uses.  The old form
        // sized from `message().len() + 128` and then let `write_diagnostic`
        // `reserve` a larger amount — a guaranteed realloc on any non-trivial
        // diagnostic.
        let mut out = String::with_capacity(Self::estimate_cap(diag));
        // Single-object output is not inside an array, so keep the brace at
        // column 0 (fields still one level in).
        self.write_diagnostic(diag, cx, 1, &mut out);
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

    fn end(&self, out: &mut dyn DiagnosticSink) -> io::Result<()> {
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
    ///
    /// U+2028/U+2029 are valid JSON string characters and are intentionally
    /// emitted raw; this renderer's output is not safe to embed verbatim in an
    /// HTML `<script>` context (escape them at that boundary if needed).
    fn push_json_str(out: &mut impl Write, s: &str) {
        // Escaping is decided and applied by [`push_json_escaped_str`]: a
        // single SWAR scan per 8-byte chunk finds each escape site, and the
        // runs between them are bulk-copied with `write_str`.  The clean case
        // (no control byte, `"`, or `\`) is one scan that finds nothing plus
        // one `write_str(s)`.
        let _ = out.write_str("\"");
        push_json_escaped_str(out, s);
        let _ = out.write_str("\"");
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

    /// Length-bounded capacity estimate for one diagnostic's JSON object.
    ///
    /// Sizing from collection *lengths* avoids a full traversal whose only
    /// purpose would be to feed `reserve`, while still getting the common case
    /// down to one allocation.  Every term is an average, so under-reserving
    /// only costs normal amortized growth.
    ///
    /// `children`, `help` and `additionalSpans` used to be missing — and they
    /// are exactly the sections that made label/child-heavy diagnostics
    /// reallocate: `children` entries carry a message, styled parts, a span and
    /// a full label array each.
    fn estimate_cap(diag: &Diagnostic<'_>) -> usize {
        192
            + diag.message().len()
            + diag.file_name().len() * 2
            + 64
            + diag.labels().len() * 128
            + diag.suggestions().len() * 96
            + diag.related_errors.len() * 96
            + diag.help().map_or(0, |h| h.len() + 16)
            + diag.children().len() * 160
            + diag.spans.primary_ref().len().saturating_sub(1) * 96
            + diag
                .call_chain
                .as_ref()
                .map_or(0, |c| c.entries().len() * 80)
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
        // `{` sits at the *outer* indentation, so `}` aligns with it; the
        // fields sit one level deeper.
        let ind0 = Self::indent(indent_lvl.saturating_sub(1));
        out.reserve(Self::estimate_cap(diag));
        let mut obj = JsonContainer::object(out, &ind0, &ind, self.pretty);

        // Resolve the anchor source's line index once per diagnostic: the
        // span, additional-span, related-error and call-chain paths all need
        // it, and each `cx.line_index` call is an LRU probe (or, past the
        // cache capacity, a full rescan).  `anchor_file` is the file those
        // line/column values are resolved against — a span from any *other*
        // file must not be mapped through this index.
        // The anchor is the diagnostic's **first primary span**; `MultiSpan`
        // holds only primary spans (secondaries live in `labels`), so `first()`
        // is a primary by construction, and the top-level `file` field is
        // derived from the same span below — the coordinates and the file name
        // can therefore never name different files.
        let anchor_file = diag.spans.first().map(|s| s.file);
        // Every index consumer (primary span, additional spans, related errors,
        // call chain) first requires an anchor file; labels/suggestions/children
        // never read line/col.  The old four extra disjuncts only forced an O(n)
        // line-index build for diagnostics that could not consume it.
        let needs_index = anchor_file.is_some();
        let idx = if needs_index {
            diag.source().map(|s| cx.line_index(s))
        } else {
            None
        };

        // Stable identifier: computed from `(code, primary span, message)`
        // via FNV-1a so tools can dedupe/suppress across runs. Emitted
        // first so tools that read only the head of the object get it
        // without scanning the whole diagnostic.
        obj.key("id");
        obj.u64_hex(diag.stable_id());

        obj.key("level");
        obj.string(diag.level.as_str());

        obj.key("message");
        // Accessors, not field access: `message()`/`labels()`/… carry
        // `debug_assert!(kind.is_none() || humanized)` — a caller that
        // renders a kind-built diagnostic without humanizing it must fail
        // loudly in debug builds (with `""` message in release), not
        // silently emit a JSON document missing its data.  `spans` has no
        // assert and keeps direct access.
        obj.string(diag.message());

        if let Some(code) = &diag.code {
            obj.key("code");
            obj.string(code.code());
            if let Some(entry) = code.entry() {
                obj.key("title");
                obj.string(entry.title);
                obj.key("category");
                obj.string(entry.category.key());
            } else {
                // Stable schema: `title`/`category` are always present for a
                // coded diagnostic; an unregistered code yields `null` rather
                // than key absence, so consumers do not need two code paths.
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
                if let Some((line, col)) =
                    anchor_linecol(idx.as_ref(), diag.source(), anchor_file, span)
                {
                    sp.key("line");
                    sp.u64(line);
                    sp.key("column");
                    sp.u64(col);
                }
                sp.close();
            }
        }

        // Additional primary spans beyond the first are currently dropped by
        // the singular `span` above; surface them so multi-location
        // diagnostics are not silently truncated in the machine format.
        // Borrowed view — one pointer, no per-diagnostic `Vec` clone (mirrors
        // the HTML renderer).
        let all_spans = diag.spans.primary_ref();
        if all_spans.len() > 1 {
            obj.key("additionalSpans");
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for sp in all_spans.iter().skip(1) {
                arr.element();
                let o = &mut *arr.out;
                o.push('{');
                // every additional span carries its owning file,
                // mirroring `labels`/`relatedErrors`; a bare {start,end}
                // cannot be placed by a machine consumer.
                match diag.file_name_for(sp.file) {
                    Some(name) => {
                        flat_key0(o, "file");
                        Self::push_json_str(o, name);
                    }
                    None => {
                        flat_key0(o, "fileId");
                        super::fmt::push_uint(o, sp.file.as_u32() as u64);
                    }
                }
                flat_key(o, "start");
                super::fmt::push_uint(o, sp.start as u64);
                flat_key(o, "end");
                super::fmt::push_uint(o, sp.end as u64);
                if let Some((line, col)) =
                    anchor_linecol(idx.as_ref(), diag.source(), anchor_file, *sp)
                {
                    flat_key(o, "line");
                    super::fmt::push_uint(o, line);
                    flat_key(o, "column");
                    super::fmt::push_uint(o, col);
                }
                o.push('}');
            }
            arr.close();
        }

        // labels
        if !diag.labels().is_empty() {
            obj.key("labels");
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for lbl in diag.labels() {
                arr.element();
                // Flat `{"k": v, …}` element: bytes pushed directly, no
                // per-element `JsonContainer` frame (see `flat_open`).
                let o = &mut *arr.out;
                o.push('{');
                flat_key0(o, "start");
                super::fmt::push_uint(o, lbl.span.start as u64);
                flat_key(o, "end");
                super::fmt::push_uint(o, lbl.span.end as u64);
                // Resolved coordinates, same policy as every other section
                // (`anchor_linecol`): a consumer must not have to reimplement
                // offset → line mapping per section, and it must not have to
                // handle "absent" and "clamped" as separate cases.
                if let Some((line, col)) =
                    anchor_linecol(idx.as_ref(), diag.source(), anchor_file, lbl.span)
                {
                    flat_key(o, "line");
                    super::fmt::push_uint(o, line);
                    flat_key(o, "column");
                    super::fmt::push_uint(o, col);
                }
                // Resolve the label's `FileId` to the same file-name string the
                // top-level `file` field uses; the numeric `FileId` is
                // process-local and must not leak into stable output.  Only an
                // unresolvable foreign file falls back to the id, under a
                // distinctly-named key.
                match diag.file_name_for(lbl.span.file) {
                    Some(name) => {
                        flat_key(o, "file");
                        Self::push_json_str(o, name);
                    }
                    None => {
                        flat_key(o, "fileId");
                        super::fmt::push_uint(o, lbl.span.file.as_u32() as u64);
                    }
                }
                flat_key(o, "message");
                Self::push_json_str(o, &lbl.message);
                flat_key(o, "kind");
                Self::push_json_str(o, crate::diagnostics::annotation_kind_name(lbl.kind));
                o.push('}');
            }
            arr.close();
        }

        // related errors (code, message, span) — one entry per `RelatedError`.
        if !diag.related_errors.is_empty() {
            obj.key("relatedErrors");
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for rel in &diag.related_errors {
                arr.element();
                let o = &mut *arr.out;
                o.push('{');
                // Whether the two sibling arrays are concerned, a related
                // error can point into a *different* file; name it (matching
                // `labels`/`callChain`) instead of emitting a bare span the
                // consumer cannot place.
                if let Some(sp) = rel.span.as_ref() {
                    match diag.file_name_for(sp.file) {
                        Some(name) => {
                            flat_key0(o, "file");
                            Self::push_json_str(o, name);
                        }
                        None => {
                            flat_key0(o, "fileId");
                            super::fmt::push_uint(o, sp.file.as_u32() as u64);
                        }
                    }
                    flat_key(o, "code");
                } else {
                    flat_key0(o, "code");
                }
                match &rel.code {
                    Some(c) => Self::push_json_str(o, c.code()),
                    None => o.push_str("null"),
                }
                flat_key(o, "message");
                Self::push_json_str(o, &rel.message);
                flat_key(o, "span");
                match rel.span.as_ref() {
                    Some(sp) => {
                        o.push('{');
                        flat_key0(o, "start");
                        super::fmt::push_uint(o, sp.start as u64);
                        flat_key(o, "end");
                        super::fmt::push_uint(o, sp.end as u64);
                        // Free when the anchor's source is attached *and*
                        // the span belongs to that same file (the common
                        // same-file case); a foreign span's offsets would
                        // otherwise be mapped through the wrong buffer.
                        if let Some((line, col)) =
                            anchor_linecol(idx.as_ref(), diag.source(), anchor_file, *sp)
                        {
                            flat_key(o, "line");
                            super::fmt::push_uint(o, line);
                            flat_key(o, "column");
                            super::fmt::push_uint(o, col);
                        }
                        o.push('}');
                    }
                    None => o.push_str("null"),
                }
                o.push('}');
            }
            arr.close();
        }

        // Child diagnostics carry user-visible notes/help (including styled
        // type/name highlights); machine consumers must see them too.
        if !diag.children().is_empty() {
            obj.key("children");
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for child in diag.children() {
                arr.element();
                let o = &mut *arr.out;
                o.push('{');
                flat_key0(o, "level");
                Self::push_json_str(o, child.level.as_str());
                flat_key(o, "message");
                let plain = child.plain_message();
                Self::push_json_str(o, plain.as_ref());
                if let Some(styled) = &child.styled_message {
                    flat_key(o, "styledMessage");
                    o.push('[');
                    for (i, part) in styled.0.iter().enumerate() {
                        if i > 0 {
                            o.push(',');
                        }
                        o.push('{');
                        flat_key0(o, "text");
                        Self::push_json_str(o, part.content.as_ref());
                        flat_key(o, "highlighted");
                        o.push_str(
                            if matches!(
                                part.style,
                                crate::diagnostics::StringPartStyle::Highlighted
                            ) {
                                "true"
                            } else {
                                "false"
                            },
                        );
                        o.push('}');
                    }
                    o.push(']');
                }
                if let Some(sp) = child.span {
                    flat_key(o, "span");
                    o.push('{');
                    flat_key0(o, "start");
                    super::fmt::push_uint(o, sp.start as u64);
                    flat_key(o, "end");
                    super::fmt::push_uint(o, sp.end as u64);
                    if let Some((line, col)) =
                        anchor_linecol(idx.as_ref(), diag.source(), anchor_file, sp)
                    {
                        flat_key(o, "line");
                        super::fmt::push_uint(o, line);
                        flat_key(o, "column");
                        super::fmt::push_uint(o, col);
                    }
                    match diag.file_name_for(sp.file) {
                        Some(name) => {
                            flat_key(o, "file");
                            Self::push_json_str(o, name);
                        }
                        None => {
                            flat_key(o, "fileId");
                            super::fmt::push_uint(o, sp.file.as_u32() as u64);
                        }
                    }
                    o.push('}');
                }
                if !child.labels.is_empty() {
                    flat_key(o, "labels");
                    o.push('[');
                    for (i, lbl) in child.labels.iter().enumerate() {
                        if i > 0 {
                            o.push(',');
                        }
                        o.push('{');
                        flat_key0(o, "start");
                        super::fmt::push_uint(o, lbl.span.start as u64);
                        flat_key(o, "end");
                        super::fmt::push_uint(o, lbl.span.end as u64);
                        if let Some((line, col)) =
                            anchor_linecol(idx.as_ref(), diag.source(), anchor_file, lbl.span)
                        {
                            flat_key(o, "line");
                            super::fmt::push_uint(o, line);
                            flat_key(o, "column");
                            super::fmt::push_uint(o, col);
                        }
                        match diag.file_name_for(lbl.span.file) {
                            Some(name) => {
                                flat_key(o, "file");
                                Self::push_json_str(o, name);
                            }
                            None => {
                                flat_key(o, "fileId");
                                super::fmt::push_uint(o, lbl.span.file.as_u32() as u64);
                            }
                        }
                        flat_key(o, "message");
                        Self::push_json_str(o, &lbl.message);
                        flat_key(o, "kind");
                        Self::push_json_str(o, crate::diagnostics::annotation_kind_name(lbl.kind));
                        o.push('}');
                    }
                    o.push(']');
                }
                o.push('}');
            }
            arr.close();
        }

        // help
        if let Some(help) = diag.help() {
            obj.key("help");
            obj.string(help);
        }

        // suggestions
        if !diag.suggestions().is_empty() {
            obj.key("suggestions");
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for sug in diag.suggestions() {
                arr.element();
                let o = &mut *arr.out;
                o.push('{');
                flat_key0(o, "message");
                Self::push_json_str(o, &sug.message);
                flat_key(o, "applicability");
                Self::push_json_str(o, crate::diagnostics::applicability_name(sug.applicability));
                flat_key(o, "style");
                Self::push_json_str(o, crate::diagnostics::suggestion_style_name(sug.style));
                // A machine consumer that wants to apply the fix needs the
                // edit itself, which the model already carries.  Emit the span
                // (using the same file/fileId convention as labels) and the
                // replacement text; both are optional by design.
                if let Some(sp) = sug.span {
                    flat_key(o, "span");
                    o.push('{');
                    flat_key0(o, "start");
                    super::fmt::push_uint(o, sp.start as u64);
                    flat_key(o, "end");
                    super::fmt::push_uint(o, sp.end as u64);
                    if let Some((line, col)) =
                        anchor_linecol(idx.as_ref(), diag.source(), anchor_file, sp)
                    {
                        flat_key(o, "line");
                        super::fmt::push_uint(o, line);
                        flat_key(o, "column");
                        super::fmt::push_uint(o, col);
                    }
                    match diag.file_name_for(sp.file) {
                        Some(name) => {
                            flat_key(o, "file");
                            Self::push_json_str(o, name);
                        }
                        None => {
                            flat_key(o, "fileId");
                            super::fmt::push_uint(o, sp.file.as_u32() as u64);
                        }
                    }
                    o.push('}');
                }
                if let Some(replacement) = &sug.replacement {
                    flat_key(o, "replacement");
                    Self::push_json_str(o, replacement);
                }
                o.push('}');
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
            // The hoisted anchor index (resolved once above) serves every
            // entry: a batch of M diagnostics with a call chain over one
            // n-byte source would otherwise pay O(M·n).
            let file_name = diag.file_name();
            let mut arr = JsonContainer::array(&mut *obj.out, &ind1, &ind, self.pretty);
            for entry in chain.entries() {
                arr.element();
                // At most three parts (label, kind, location), streamed
                // straight into the output buffer.
                let o = &mut *arr.out;
                o.push('{');
                flat_key0(o, "label");
                Self::push_json_str(o, &entry.message);
                flat_key(o, "kind");
                Self::push_json_str(o, crate::diagnostics::chain_entry_kind_name(entry.kind));
                // a chain entry pointing into a foreign file must carry
                // that file's identity; otherwise its location is silently
                // dropped and a JSON consumer cannot place it.
                if let Some(span) = entry.span {
                    match diag.file_name_for(span.file) {
                        Some(name) => {
                            flat_key(o, "file");
                            Self::push_json_str(o, name);
                        }
                        None => {
                            flat_key(o, "fileId");
                            super::fmt::push_uint(o, span.file.as_u32() as u64);
                        }
                    }
                }
                if let Some(span) = entry.span
                    && let Some((line, col)) =
                        anchor_linecol(idx.as_ref(), diag.source(), anchor_file, span)
                {
                    // Numeric fields a consumer can parse without splitting a
                    // composite string: file names may contain `:` (Windows
                    // drives, `ns::item`), which made `location` ambiguous.
                    flat_key(o, "line");
                    super::fmt::push_uint(o, line);
                    flat_key(o, "column");
                    super::fmt::push_uint(o, col);
                    flat_key(o, "location");
                    flat_location(o, file_name, line, col);
                }
                o.push('}');
            }
            arr.close();
        }

        obj.close();
    }
}

/// Inline flat-object emission for compact array elements.
///
/// The `labels` / `relatedErrors` / `suggestions` / `callChain` elements are
/// flat `{"k": v, …}` objects; pushing their bytes directly into the output
/// buffer (instead of constructing a [`JsonContainer::compact_object`] frame
/// per element) drops the per-element `sep`/`first`/`pretty` bookkeeping from
/// exactly the loops that scale with label/related/suggestion count.  The
/// byte shapes are pinned identical to `compact_object` (`"` + key +
/// `": "`, `", "` separators) — the JSON snapshots hold this.
fn flat_key(out: &mut String, key: &str) {
    out.push_str(", ");
    flat_key0(out, key);
}

/// First field of a flat object: no separator (the `{` was just pushed).
fn flat_key0(out: &mut String, key: &str) {
    out.push('"');
    out.push_str(key);
    out.push_str("\": ");
}

/// `push_json_str`-shaped helper for the inline flat-object path.
fn flat_location(out: &mut String, file: &str, line: u64, col: u64) {
    out.push('"');
    if !file.is_empty() {
        push_json_escaped_str(out, file);
        out.push(':');
    }
    super::fmt::push_uint(out, line);
    out.push(':');
    super::fmt::push_uint(out, col);
    out.push('"');
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
    ///
    /// The rule is **layout**, not style: a *multi-line* container (pretty mode
    /// writes a newline + indent before each element) uses `","` — a `", "`
    /// there would leave a trailing space at the end of the line before the
    /// newline.  A *single-line* container (`compact_object`, and the flat
    /// `{"k": v, …}` helpers) uses `", "` in both modes, since the space is the
    /// only thing separating adjacent fields.  So the two spellings coexist by
    /// design; this field documents which container is which.
    sep: &'static str,
    /// Indentation written before each element in pretty mode.
    elem_ind: &'a str,
    /// Indentation written before the closing bracket in pretty mode.
    close_ind: &'a str,
    pretty: bool,
    first: bool,
    close: char,
    /// Set by [`Self::close`]; the `Drop` tripwire fires if a container is
    /// dropped without closing, so an early `return` cannot silently emit
    /// invalid JSON (the "bracket always closed" invariant, mirroring the
    /// separator-before-field invariant that is structural).
    closed: bool,
}

impl<'a> JsonContainer<'a> {
    /// `{` … `}` whose fields sit at `field_ind` and whose `}` aligns with
    /// the `{` at `open_ind` — standard pretty-printer layout (`{` at column
    /// 0, fields at column 2, `}` at column 0), instead of the old shape
    /// where `}` sat at the fields' indentation, two columns right of `{`.
    /// Valid JSON either way; this matches every other pretty-printer.
    fn object(out: &'a mut String, open_ind: &'a str, field_ind: &'a str, pretty: bool) -> Self {
        // The brace sits at `open_ind` so a diagnostic inside an array is
        // indented (standard pretty layout); compact mode writes no indent.
        if pretty {
            out.push_str(open_ind);
        }
        out.push('{');
        Self {
            out,
            sep: ",",
            elem_ind: field_ind,
            close_ind: open_ind,
            pretty,
            first: true,
            close: '}',
            closed: false,
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
            closed: false,
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
            closed: false,
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

    /// Close the container, indenting the bracket in pretty mode.
    fn close(mut self) {
        if self.pretty && !self.first {
            self.out.push('\n');
            self.out.push_str(self.close_ind);
        }
        self.out.push(self.close);
        self.closed = true;
        // `Drop` runs at the end of this scope; `closed` is already true, so
        // the tripwire stays silent.
    }
}

impl Drop for JsonContainer<'_> {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        // A forgotten `close()` (an early `return`/`?` between `open` and
        // `close`) used to emit an unclosed bracket — invalid JSON on the wire.
        // Auto-close instead: the degradation becomes *valid* (empty) JSON, and
        // the debug assert still makes the bug loud in development.
        debug_assert!(
            false,
            "JsonContainer dropped without close() — auto-closing to keep the stream valid"
        );
        if self.pretty && !self.first {
            self.out.push('\n');
            self.out.push_str(self.close_ind);
        }
        self.out.push(self.close);
        self.closed = true;
    }
}

/// Resolve a span anchor to 1-based `(line, column)` in the anchor file.
///
/// This is the **single** policy for every JSON section that emits coordinates:
///
/// * `None` when no index was built, when the anchor file has no attached
///   source, or when the span belongs to a *foreign* file (the line index is the
///   anchor's, so mapping a foreign offset through it would emit coordinates in
///   the wrong file);
/// * offsets past EOF are **clamped**, never omitted.  An EOF-crossing span is
///   corrupt, but a clamped position is still placeable, and a machine consumer
///   must not need two code paths ("absent" versus "clamped") for one corruption
///   class.  The primary-span path used to omit, `additionalSpans`/
///   `relatedErrors` clamped, and `callChain.location` did neither — with a
///   comment claiming they agreed.
fn anchor_linecol(
    idx: Option<&crate::diagnostics::LineIndex<'_>>,
    source: Option<&str>,
    anchor: Option<crate::ast::FileId>,
    span: crate::ast::Span,
) -> Option<(u64, u64)> {
    let idx = idx?;
    let source = source?;
    if anchor != Some(span.file) {
        return None;
    }
    let (line, col) = idx.linecol_display(span.start_as_usize().min(source.len()));
    Some(((line + 1) as u64, (col + 1) as u64))
}

/// First byte JSON must escape — a control byte (`< 0x20`), `"` or `\` — or
/// `None` if none qualifies.
///
/// Adaptive dispatch, because the two regimes want different machinery:
///
/// * **≤ [`SHORT_FUSED_MAX`]** (diagnostic messages, labels, help text): the
///   fused SWAR scan.  One pass over ≤8 vector-sized chunks finds all three
///   classes at once; `memchr2`'s wider vectors cannot amortize their setup at
///   this length.
/// * **longer**: `memchr2` for the two structural bytes (wider than SWAR),
///   then the control scan bounded to the first structural hit, so a string
///   with an early quote never pays a full control pass.
///
/// Both alternatives are exact: the fused scan's combined mask flags the lowest
/// *true* candidate (a cross-byte borrow can only add a lane above a true one,
/// and `trailing_zeros` takes the lowest), and UTF-8 continuation bytes are all
/// `≥ 0x80`, so no multi-byte character is a candidate.
#[inline]
fn next_json_escape_byte(b: &[u8]) -> Option<usize> {
    if b.len() <= SHORT_FUSED_MAX {
        return next_escape_swar(b);
    }
    let structural = memchr::memchr2(b'"', b'\\', b);
    let limit = structural.unwrap_or(b.len());
    let ctrl = next_ctrl_byte(&b[..limit]);
    match (structural, ctrl) {
        (Some(s), Some(c)) => Some(s.min(c)),
        (s, c) => s.or(c),
    }
}

/// Length at which the fused SWAR scan stops paying for itself and `memchr2`'s
/// wider vectors take over.  A measurement decision (see the `benchmark` CI
/// job), not a correctness one: both scanners return the same index.
const SHORT_FUSED_MAX: usize = 64;

/// First escape candidate in `b` via one fused SWAR pass.
///
/// Three predicates per 8-byte chunk — `b < 0x20` (Mycroft `hasless`),
/// `b == '"'` and `b == '\\'` (both `has_zero` on an XOR broadcast) — OR'd into
/// one mask.  The tail pads with `0xFF` lanes: they cannot flag (their `hasless`
/// and `has_zero` masks are all zero, and the high-bit test clears them even
/// when a borrow propagates out of a real lane).
#[inline]
fn next_escape_swar(b: &[u8]) -> Option<usize> {
    const HI: u64 = 0x8080_8080_8080_8080;
    const LO: u64 = 0x0101_0101_0101_0101;
    const SP: u64 = 0x2020_2020_2020_2020;
    const QUOTE: u64 = 0x2222_2222_2222_2222;
    const BACKSLASH: u64 = 0x5C5C_5C5C_5C5C_5C5C;
    #[inline]
    fn has_zero(v: u64) -> u64 {
        v.wrapping_sub(LO) & !v & HI
    }
    #[inline]
    fn lane(mask: u64) -> usize {
        (mask.trailing_zeros() / 8) as usize
    }
    let mut i = 0;
    while i + 8 <= b.len() {
        let x = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let hits = (x.wrapping_sub(SP) & !x & HI)
            | has_zero(x ^ QUOTE)
            | has_zero(x ^ BACKSLASH);
        if hits != 0 {
            return Some(i + lane(hits));
        }
        i += 8;
    }
    let n = b.len() - i;
    if n == 0 {
        return None;
    }
    let mut buf = [0xFFu8; 8];
    buf[..n].copy_from_slice(&b[i..]);
    let x = u64::from_le_bytes(buf);
    let hits = (x.wrapping_sub(SP) & !x & HI) | has_zero(x ^ QUOTE) | has_zero(x ^ BACKSLASH);
    (hits != 0).then(|| i + lane(hits))
}

/// First byte `< 0x20` in `b` (JSON control range), or `None`.
///
/// Mycroft's `hasless` trick (`(x - 0x20) & ~x & 0x80`) skips clean 8-byte
/// chunks; the lowest flagged lane is always a true match (a cross-byte borrow
/// can only flag a byte that is itself `≥ 0x20` *after* a true `< 0x20` lane).
#[inline]
fn next_ctrl_byte(b: &[u8]) -> Option<usize> {
    const HI: u64 = 0x8080_8080_8080_8080;
    const SP: u64 = 0x2020_2020_2020_2020;
    let mut i = 0;
    while i + 8 <= b.len() {
        let x = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let ctrl = x.wrapping_sub(SP) & !x & HI;
        if ctrl != 0 {
            return Some(i + (ctrl.trailing_zeros() / 8) as usize);
        }
        i += 8;
    }
    // Tail (0..=7 remaining bytes — the dominant case for diagnostic messages,
    // labels and help text): pad into a stack buffer and run one SWAR pass
    // instead of a per-byte scalar loop.
    //
    // `0xFF` filler lanes cannot produce a false hit: `(0xFF - 0x20) & !0xFF`
    // is 0, so the high-bit test clears them even when a borrow from a real
    // lane propagates into them.  And a borrow cannot make a *real* lane flag
    // without a real `< 0x20` lane below it (the documented `hasless`
    // argument), so the lowest flagged lane is always a true match.
    let n = b.len() - i;
    if n == 0 {
        return None;
    }
    let mut buf = [0xFFu8; 8];
    buf[..n].copy_from_slice(&b[i..]);
    let x = u64::from_le_bytes(buf);
    let ctrl = x.wrapping_sub(SP) & !x & HI;
    (ctrl != 0).then(|| i + (ctrl.trailing_zeros() / 8) as usize)
}

/// Does `s` contain a byte that JSON must escape?  Test-only scalar-shaped
/// view of [`next_json_escape_byte`] so the positional and predicate SWAR
/// scans cannot drift apart.
#[cfg(test)]
#[inline]
fn needs_json_escape(s: &str) -> bool {
    next_json_escape_byte(s.as_bytes()).is_some()
}

/// Append `s` to `out` with JSON string escaping (no surrounding quotes).
/// Bulk-copies the runs between escape sites instead of a per-byte branch
/// loop; the clean case (no escapes) is a single `write_str`.
fn push_json_escaped_str<W: std::fmt::Write>(out: &mut W, s: &str) {
    let bytes = s.as_bytes();
    // Short strings (diagnostic messages, labels, help) take the fused scan:
    // one pass finds all three escape classes, and `memchr2`'s wider vectors
    // cannot amortize their setup below 64 bytes.  Each iteration resumes right
    // after the previous escape, so the total remains linear in `len`.
    if bytes.len() <= SHORT_FUSED_MAX {
        let mut start = 0usize;
        let mut i = 0usize;
        while let Some(rel) = next_escape_swar(&bytes[i..]) {
            let at = i + rel;
            if start < at {
                let _ = out.write_str(&s[start..at]);
            }
            push_json_escaped(out, bytes[at] as char);
            i = at + 1;
            start = i;
        }
        if start < bytes.len() {
            let _ = out.write_str(&s[start..]);
        }
        return;
    }
    // Stateful merged scan: the structural iterator (`"` / `\`) advances only
    // when its byte is actually escaped, and the control scan runs only up to
    // the next structural position.  Re-deriving the structural hit from
    // `bytes[start..]` on every iteration made control-heavy strings with no
    // quotes/backslashes O(n²) (one full `memchr2` miss per escaped control);
    // this makes the total scan linear.
    let mut structural = memchr::memchr2_iter(b'"', b'\\', bytes);
    let mut next_structural = structural.next();
    let mut start = 0usize;
    while start < bytes.len() {
        let limit = next_structural.unwrap_or(bytes.len());
        let ctrl = if start < limit {
            next_ctrl_byte(&bytes[start..limit]).map(|off| start + off)
        } else {
            None
        };
        let i = match (ctrl, next_structural) {
            (Some(c), _) => c,
            (None, Some(st)) => st,
            (None, None) => break,
        };
        if i > start {
            let _ = out.write_str(&s[start..i]);
        }
        // The escape candidates (`< 0x20`, `"`, `\`) are all ASCII, so
        // `bytes[i]` is a char boundary and `bytes[i] as char` is the
        // codepoint to escape.
        push_json_escaped(out, bytes[i] as char);
        start = i + 1;
        if Some(i) == next_structural {
            next_structural = structural.next();
        }
    }
    if start < bytes.len() {
        let _ = out.write_str(&s[start..]);
    }
}

/// Append one character to `out`, JSON-escaped. Used by
/// [`JsonRenderer::push_json_str`] so escaping goes straight into the
/// caller's buffer without an intermediate copy.
///
/// Generic over the output so escaping can go straight into whatever buffer
/// the caller is building — `String` or `CompactString` alike.
/// Write the escaped form of one byte/char that
/// [`next_json_escape_byte`] selected.
///
/// Contract: `c` comes from that scanner, i.e. it is one of `"`, `\`, or a
/// byte `< 0x20`.  A caller passing an arbitrary `char` would produce valid
/// but unexpected output (`0x7F` becomes `\u007f`, and U+2028/2029 are left
/// raw); the debug assert pins the contract rather than silently accepting
/// it.
fn push_json_escaped<W: std::fmt::Write>(out: &mut W, c: char) {
    debug_assert!(
        matches!(c, '"' | '\\' | '\n' | '\r' | '\t' | '\u{0}'..='\u{1f}'),
        "push_json_escaped called with a char the scanner would not select: {c:?}"
    );
    let _ = match c {
        '"' => out.write_str("\\\""),
        '\\' => out.write_str("\\\\"),
        '\n' => out.write_str("\\n"),
        '\r' => out.write_str("\\r"),
        '\t' => out.write_str("\\t"),
        c if c.is_control() => {
            // `0x7F`–`0x9F` are Unicode controls but not JSON-required
            // escapes (JSON only requires `< 0x20`). The scanner never
            // selects them, but this arm still emits a valid `\uXXXX`.
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
    fn push_json_str_escapes_backspace_and_formfeed() {
        for (raw, expect) in [("\x08", "\\u0008"), ("\x0C", "\\u000c")] {
            let input = format!("a{raw}b");
            let mut out = String::new();
            JsonRenderer::push_json_str(&mut out, &input);
            assert!(out.contains(expect), "{raw:?} → {out}");
            let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
            assert_eq!(v, serde_json::Value::String(input));
        }
    }

    #[test]
    fn push_json_str_resyncs_at_control_structural_boundaries() {
        for ctrl in ['\n', '\r', '\t', '\u{0}', '\u{1f}'] {
            for structural in ['"', '\\'] {
                for pad in 0..16 {
                    let input = format!("{}{ctrl}{structural}{}", "a".repeat(pad), "b".repeat(8));
                    let mut escaped = String::new();
                    JsonRenderer::push_json_str(&mut escaped, &input);
                    let v: serde_json::Value = serde_json::from_str(&escaped).unwrap_or_else(|e| {
                        panic!("invalid JSON ({ctrl:?} {structural:?} {pad}): {e}")
                    });
                    assert_eq!(v, serde_json::Value::String(input));

                    let input = format!("{}{structural}{ctrl}{}", "a".repeat(pad), "b".repeat(8));
                    let mut escaped = String::new();
                    JsonRenderer::push_json_str(&mut escaped, &input);
                    let v: serde_json::Value = serde_json::from_str(&escaped).unwrap_or_else(|e| {
                        panic!("invalid JSON ({structural:?} {ctrl:?} {pad}): {e}")
                    });
                    assert_eq!(v, serde_json::Value::String(input));
                }
            }
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

    /// A diagnostic whose message and file name carry every JSON escape
    /// candidate (`"`, `\`, control bytes) must still serialise to valid JSON
    /// that round-trips through `serde_json`.  Exercises the bulk-copy dirty
    /// path of `push_json_escaped_str` end-to-end through the renderer.
    #[test]
    fn json_dirty_message_round_trips() {
        let msg = "a\"b\\c\nd\re\tf\x07g\x1fh";
        let diag = crate::diagnostics::Diagnostic::error(msg)
            .with_span(crate::ast::Span::new(crate::ast::FileId::DUMMY, 0, 1))
            .with_source("x")
            .with_file_name("weird\"\\file\n.rs");
        for pretty in [false, true] {
            let out = json_of(&diag, pretty);
            let v: serde_json::Value = serde_json::from_str(&out)
                .unwrap_or_else(|e| panic!("invalid JSON ({pretty}): {e}\n{out}"));
            let obj = v.as_object().expect("diagnostic is an object");
            assert_eq!(
                obj.get("message").and_then(|m| m.as_str()),
                Some(msg),
                "message round-trip ({pretty})"
            );
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
        // `needs_json_escape` and `next_json_escape_byte` are two views of the
        // same SWAR predicate — pin both against the scalar reference and, when
        // a candidate exists, confirm the position actually names one.
        fn check(s: &str, ctx: &str) {
            let want = scalar(s);
            assert_eq!(needs_json_escape(s), want, "{ctx}: needs_json_escape");
            let pos = next_json_escape_byte(s.as_bytes());
            assert_eq!(pos.is_some(), want, "{ctx}: next_json_escape_byte");
            if let Some(i) = pos {
                let b = s.as_bytes()[i];
                assert!(
                    b < 0x20 || b == b'"' || b == b'\\',
                    "{ctx}: pos {i} = {b:#04x} is not an escape byte"
                );
            }
        }
        // Every ASCII single byte (0x00..=0x7F) as a 1-byte string — all are
        // valid UTF-8.  Multi-byte continuations can't be a single-byte `&str`.
        for byte in 0u8..0x80 {
            let buf = [byte];
            let s = std::str::from_utf8(&buf).unwrap();
            check(s, &format!("single byte {byte:#04x}"));
        }
        // Escape candidate at each position 0..=16, padded clean on both sides
        // — hits every offset within and across 8-byte chunks.
        let clean = "aaaaaaaaaaaaaaaaa"; // 17 clean bytes
        for pos in 0..=16usize {
            for cand in [b'\n', b'\r', b'\t', b'"', b'\\', 0x00u8, 0x1f, 0x20, 0x7f] {
                let mut bytes = clean.as_bytes().to_vec();
                bytes[pos] = cand;
                let s = String::from_utf8(bytes).unwrap();
                check(&s, &format!("cand {cand:#04x} at pos {pos}"));
            }
        }
        // Lengths straddling the 8-byte chunk boundary with clean content.
        for len in 0..=17 {
            let s = &"abcdefghijklmnopqrstuvwxyz"[..len];
            check(s, &format!("clean len {len}"));
        }
        // A long clean string (multiple full chunks) and one dirty at the end.
        let long_clean = &"a".repeat(200);
        check(long_clean, "long clean");
        let long_dirty = format!("{}{}", "a".repeat(200), "\\");
        check(&long_dirty, "long dirty");
    }

    #[test]
    fn child_diagnostics_are_serialized() {
        let diag = Diagnostic::error("parent").with_highlighted_note(
            crate::diagnostics::StyledString::new(vec![crate::diagnostics::StringPart {
                content: "child".into(),
                style: crate::diagnostics::StringPartStyle::Highlighted,
            }]),
        );
        let out = json_of(&diag, false);
        assert!(out.contains("\"children\""), "children missing: {out}");
        assert!(out.contains("\"styledMessage\""), "styled missing: {out}");
        assert!(
            out.contains("\"highlighted\": true"),
            "highlight missing: {out}"
        );
        assert!(out.contains("child"), "child text missing: {out}");
    }

    #[test]
    fn empty_pretty_batch_is_valid_empty_array() {
        let opts = crate::diagnostics::DiagOpts::default();
        let mut emitter = crate::diagnostics::emitter::Emitter::new(
            JsonRenderer::new(true),
            crate::diagnostics::sink::string(),
            opts,
        );
        let mut diags: Vec<Diagnostic<'static>> = Vec::new();
        let mut sink = crate::diagnostics::sink::StringSink::new();
        emitter.emit_all_to(&mut diags, &mut sink).unwrap();
        let out = sink.into_string();
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert_eq!(v, serde_json::Value::Array(Vec::new()));
    }

    /// An empty pretty batch closes as `[\n\n]\n`, **not** `[]`.
    ///
    /// Recorded decision, not an oversight: `begin` cannot know the element
    /// count (the `Renderer` trait's batch hooks take no count and are called
    /// unconditionally), and the leading `\n` has to be emitted before any
    /// element — the parallel path stitches per-worker chunks by index, so no
    /// single `render_into` may own the array prefix.  The alternatives were a
    /// batch pre-scan (costs a full pass to save one newline on an empty batch)
    /// or moving the prefix into the stitcher (breaks chunk independence).
    /// `[\n]\n` is valid JSON and parses to the same empty array.
    #[test]
    fn empty_pretty_batch_is_valid_and_pinned() {
        let mut sink = crate::diagnostics::sink::StringSink::new();
        let r = JsonRenderer::new(true);
        r.begin(&mut sink).unwrap();
        r.end(&mut sink).unwrap();
        let out = sink.into_string();
        assert_eq!(out, "[\n\n]\n");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&out).expect("valid JSON"),
            serde_json::Value::Array(Vec::new())
        );

        // Non-empty keeps the jq-style layout: `[\n` … `\n]\n`.
        let mut sink = crate::diagnostics::sink::StringSink::new();
        r.begin(&mut sink).unwrap();
        let opts = crate::diagnostics::DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);
        let diag = Diagnostic::error("boom").with_code_str(ErrCode::E030);
        let mut body = String::new();
        r.render_into(&diag, &cx, &mut body);
        sink.write_str(&body).unwrap();
        r.end(&mut sink).unwrap();
        let out = sink.into_string();
        assert!(out.starts_with("[\n"), "{out:?}");
        assert!(out.ends_with("\n]\n"), "{out:?}");
        serde_json::from_str::<serde_json::Value>(&out).expect("valid JSON");
    }
}
