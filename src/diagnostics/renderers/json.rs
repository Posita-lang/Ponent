//! JSON renderer — streams a JSON array instead of buffering items.

use crate::diagnostics::Diagnostic;
use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::sink::DiagnosticSink;
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
    #[must_use]
    pub fn new(pretty: bool) -> Self {
        JsonRenderer { pretty }
    }

    #[must_use]
    pub fn new_pretty() -> Self {
        JsonRenderer::new(true)
    }

    #[must_use]
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
        // Same buffer contract as `render_internal_error`: `out` must be empty.
        // Every emitter call site clears its per-diagnostic buffer first, and a
        // non-empty one would concatenate two objects into the batch array — the
        // exact corruption the placeholder path guards against, so the normal
        // path asserts it too instead of relying on an undocumented convention.
        debug_assert!(
            out.is_empty(),
            "render_into expects an empty buffer; got {} bytes",
            out.len()
        );
        // Batch output is a JSON array; indent the object one level under `[`
        // (fields one level deeper still), which is what every pretty-printer
        // and `jq` produce.  `BATCH_LEVEL` is shared with `render_internal_error`
        // so a change here cannot misalign the panic placeholder.
        self.write_diagnostic::<BATCH_LEVEL>(diag, cx, out);
    }

    /// The one deliberate `render` override: a standalone object is laid out at
    /// nesting level 1 (brace at column 0), whereas a batch element sits inside
    /// `[ … ]` at level 2.  That is a different layout, not a copy of the batch
    /// path, so it cannot be expressed by the default `render_into` delegate.
    fn render(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String {
        // `write_diagnostic` owns the sizing (`reserve(estimate_cap)`), so this
        // start empty: a `with_capacity(estimate_cap(..))` here evaluated the
        // same O(fields) estimator a second time and then made the inner
        // `reserve` a no-op.  `String::new()` + one `reserve` is still exactly
        // one allocation.
        let mut out = String::new();
        // Single-object output is not inside an array, so keep the brace at
        // column 0 (fields still one level in).
        self.write_diagnostic::<1>(diag, cx, &mut out);
        out
    }

    fn render_separator(&self) -> &str {
        if self.pretty { ",\n" } else { "," }
    }

    /// JSON is machine-parsed, so the flat-text placeholder must never appear in
    /// its output (see the override below, and `Renderer::is_structured`).
    fn is_structured(&self) -> bool {
        true
    }

    fn render_internal_error(&self, out: &mut String) {
        // A *structured* placeholder.  Splitting one item's render panic must
        // not break the array: the flat-text marker would make the whole batch
        // unparseable, turning "one diagnostic failed" into "the consumer sees
        // nothing".  Emitting a normal object keeps the stream valid and names
        // the failure where a machine consumer can see it.
        // Lay the placeholder out exactly like a normal batch element, which
        // `render_into` writes at `indent_lvl = 2`: the `{` sits at
        // `indent(1)` under `[` and its fields at `indent(2)`.  The old form
        // opened at column 0 with 2-space fields, visually breaking the array.
        //
        // **Buffer contract:** `out` must be empty.  The emitter's
        // `fill_internal_error` clears it before calling, precisely because a
        // panicking `render_into` leaves a syntactically-closed but semantically
        // garbage object behind (`JsonContainer::drop` auto-closes) and
        // appending to it would make the array unparseable — reintroducing the
        // failure this method exists to prevent.  The debug assertion makes a
        // future caller that forgets the clear fail loudly instead of producing
        // a corrupt stream.
        debug_assert!(
            out.is_empty(),
            "render_internal_error expects an empty buffer; got {} bytes",
            out.len()
        );
        // `BATCH_LEVEL` is the level `render_into` writes a batch element at, so
        // the placeholder lines up with its neighbours instead of relying on the
        // two sites agreeing by maintenance.
        let mut obj = JsonContainer::object(
            out,
            INDENTS[BATCH_LEVEL - 1],
            INDENTS[BATCH_LEVEL],
            self.pretty,
        );
        // Same field set and order as a normal element: a consumer that reads
        // `id` unconditionally must not meet an element without one, and the
        // placeholder is exactly the element it most wants to identify.
        obj.key("id");
        obj.u64_hex(INTERNAL_ERROR_ID);
        obj.key("level");
        obj.string("error");
        obj.key("message");
        obj.string("internal: diagnostic rendering panicked");
        obj.close();
    }

    fn render_summary(&self, _errors: usize, _warnings: usize) -> String {
        String::new()
    }

    /// Explicit rather than relying on the default's delegation: a trailer after
    /// the closing `]` makes the whole document unparseable, so the guarantee
    /// should not depend on the trait default staying a delegation.
    fn render_summary_full(&self, _errors: usize, _warnings: usize, _files: usize) -> String {
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
    /// `&mut String`, not `impl fmt::Write`: the whole document is assembled in
    /// one `String` (`JsonContainer.out`), and the generic bound advertised a
    /// fallible writer whose every error the old code discarded with `let _ =`.
    /// Narrowing removes the `Result` plumbing, the capacity check per call, and
    /// the one way this module could have produced silently truncated JSON.
    fn push_json_str(out: &mut String, s: &str) {
        // Escaping is decided and applied by [`push_json_escaped_str`]: a single
        // SWAR scan per 8-byte chunk finds each escape site, and the runs between
        // them are bulk-copied.  The clean case (no control byte, `"`, or `\\`) is
        // one scan that finds nothing plus one `push_str(s)`.
        // One reserve covers the quotes and the (common) unescaped body, so the
        // pushes below do not each re-check capacity.  An escaped body can still
        // grow past it; that is one amortized extension, not the norm.
        out.reserve(s.len() + 2);
        out.push('"');
        push_json_escaped_str(out, s);
        out.push('"');
    }

    /// [`push_json_str`](Self::push_json_str) for a value known clean by
    /// construction (a fixed vocabulary word such as a level, annotation kind,
    /// applicability or chain-entry kind).
    ///
    /// Those strings are ASCII with no `"`, `\\` or control byte, so the SWAR
    /// scan every general string pays can only find nothing.  The debug assert
    /// keeps the precondition honest for a future vocabulary entry; the reserve
    /// is exact (`"` + body + `"`), so this is two pushes and one capacity check
    /// per field instead of a whole scan.
    fn push_json_str_static(out: &mut String, s: &str) {
        debug_assert!(
            s.bytes()
                .all(|b| (0x20..0x7F).contains(&b) && b != b'"' && b != b'\\'),
            "push_json_str_static called with a value that needs escaping: {s:?}"
        );
        out.reserve(s.len() + 2);
        out.push('"');
        out.push_str(s);
        out.push('"');
    }

    /// Indentation for `level` nesting steps, saturated at the table's depth.
    ///
    /// `&'static str`, not `Cow<'static, str>`: the `Owned` arm only existed for
    /// levels past the table, which no caller reaches (production uses 0/1/2,
    /// the tests 0–2), and it made every field pay a discriminant check before
    /// its `push_str`.  Saturation keeps the type total without allocating.
    fn indent(level: usize) -> &'static str {
        INDENTS
            .get(level)
            .copied()
            .unwrap_or(INDENTS[INDENTS.len() - 1])
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
        // Every term saturates: the result feeds `reserve`, so the correct
        // semantic for an absurd diagnostic is "huge", not a debug-build overflow
        // panic (`+`/`*` wrap in release and are a real fuzzing hazard in debug).
        // The message is escaped, so its JSON form can exceed its text: a quote or
        // backslash doubles, a C0 control expands sixfold (`\\u00XX`).  Sizing
        // from the raw length guaranteed a realloc for exactly those "dirty"
        // messages, which is the case this single-allocation design exists for.
        // 2x covers the overwhelmingly common quoted/backslashed case; a
        // control-heavy message may still grow once, which is cheaper than a
        // full extra scan here (see `push_json_escaped_str`).
        let mut cap = 192usize.saturating_add(diag.message().len().saturating_mul(2));
        let fname = diag.file_name().len();
        cap = cap.saturating_add(fname.saturating_mul(2));
        cap = cap.saturating_add(64);
        // Every span-shaped object (`labels`, `additionalSpans`, `children`'
        // labels, `suggestions[].span`) carries the file name, so the per-object
        // constant was systematically short by exactly the path length — the
        // section that "made label-heavy diagnostics reallocate" kept
        // reallocating.  The name is added per object instead of once.
        // The 96-byte average was an *assumption*: a label message goes through the
        // same escaping as the main one, so a long one reallocated.  Folded per label
        // now, like the replacement below.
        let label_bytes: usize = diag
            .labels()
            .iter()
            .map(|l| {
                96usize
                    .saturating_add(fname)
                    .saturating_add(l.message.len().saturating_mul(2))
            })
            .fold(0usize, |a, b| a.saturating_add(b));
        cap = cap.saturating_add(label_bytes);
        // A registered code contributes its registry `title` (prose, escaped, and
        // uncounted): every diagnostic wearing a long-titled code reallocated once.
        cap = cap.saturating_add(
            diag.code()
                .and_then(|c| c.entry())
                .map_or(0, |e| e.title.len().saturating_mul(2).saturating_add(32)),
        );
        // A suggestion's `replacement` is its largest field and was not counted
        // at all: a code fix with a 200-byte replacement tripled the section.
        cap = cap.saturating_add(
            diag.suggestions()
                .iter()
                .map(|s| {
                    // `replacement` was called "the largest field" on faith; the
                    // message is emitted too, so both lengths are counted.
                    s.replacement
                        .as_ref()
                        .map_or(96, |r| 96usize.saturating_add(r.len().saturating_mul(2)))
                        .saturating_add(s.message.len().saturating_mul(2))
                })
                .fold(0usize, |a, b| a.saturating_add(b)),
        );
        // A related error carries `code`, a `message` and a full span object.  The
        // message length is folded in: a flat 160 per element was simultaneously too
        // large (most messages are short, so the estimate over-reserved) and too small
        // (an element quoting a whole line with a long file name blew past it).
        let related_extra: usize = diag
            .related_errors
            .iter()
            .map(|r| 96usize.saturating_add(r.message.len().saturating_mul(2)))
            .fold(0usize, |a, b| a.saturating_add(b));
        cap = cap.saturating_add(related_extra);
        cap = cap.saturating_add(diag.help().map_or(0, |h| h.len().saturating_add(16)));
        // Each child carries its own `message`, which the flat per-child constant covered
        // only for a short clause.
        let children_extra: usize = diag
            .children()
            .iter()
            .map(|c| {
                96usize.saturating_add(
                    c.message
                        .as_deref()
                        .map_or(0, |m| m.len())
                        .saturating_mul(2),
                )
            })
            .fold(0usize, |a, b| a.saturating_add(b));
        cap = cap.saturating_add(children_extra);
        // ...and each child carries its own `labels` array, which the per-child
        // constant cannot cover.
        // A child's `styledMessage` parts carry text the per-child constant cannot
        // cover; `StyledString` is a `Vec<StringPart>` whose `content` is the text.
        let child_styled: usize = diag
            .children()
            .iter()
            .filter_map(|c| c.styled_message.as_ref())
            .map(|styled| {
                styled
                    .0
                    .iter()
                    .map(|p| p.content.len().saturating_mul(2).saturating_add(24))
                    .fold(0usize, |a, b| a.saturating_add(b))
            })
            .fold(0usize, |a, b| a.saturating_add(b));
        cap = cap.saturating_add(child_styled);
        let child_labels: usize = diag.children().iter().map(|c| c.labels.len()).sum();
        cap = cap.saturating_add(child_labels.saturating_mul(96usize.saturating_add(fname)));
        cap = cap.saturating_add(
            diag.spans
                .primary_ref()
                .len()
                .saturating_sub(1)
                .saturating_mul(96usize.saturating_add(fname)),
        );
        cap.saturating_add(
            diag.call_chain
                .as_ref()
                .map_or(0, |c| c.entries().len().saturating_mul(80)),
        )
    }

    /// Stream one diagnostic object into `out`.
    ///
    /// Fields are appended straight into the single output buffer in
    /// declaration order. The separator is written *before* each field, so a
    /// trailing comma is structurally impossible — no intermediate field
    /// list is needed to guarantee it.
    /// Render one diagnostic object at nesting level `LVL`.
    ///
    /// `LVL` is a const generic rather than a parameter because every caller
    /// uses a literal (2 for the batch array, 1 for a single object, 0 for the
    /// test fixture): as a const, `INDENTS[LVL]` is folded to the literal, so the
    /// per-field `push_str` sees a constant slice instead of a bounds-checked
    /// load.  It is a private inherent method, so this introduces no lifetime or
    /// trait parameter — nothing outside this module can observe the change.
    fn write_diagnostic<const LVL: usize>(
        &self,
        diag: &Diagnostic<'_>,
        cx: &RenderCtx<'_>,
        out: &mut String,
    ) {
        // Evaluated per monomorphization, i.e. **before** the index below can panic
        // and in release too.  The previous `debug_assert!` sat *after*
        // `INDENTS[LVL]`, so it could never fire: `LVL ≥ 6` died on the direct index
        // with a message naming no caller, and `LVL = 5` silently saturated
        // `indent(LVL + 1)` into a visually wrong (but valid) layout.
        const {
            assert!(
                LVL < INDENTS.len(),
                "write_diagnostic's LVL exceeds the nesting table"
            );
        }
        let ind = INDENTS[LVL];
        let ind1 = Self::indent(LVL + 1);
        // `{` sits at the *outer* indentation, so `}` aligns with it; the
        // fields sit one level deeper.
        let ind0 = INDENTS[LVL.saturating_sub(1)];
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
        // Two accessors for "the first span" meet here: the top-level `span` field and
        // the anchor file read `spans.first()`, while `additionalSpans` reads
        // `primary_ref()` and skips one — so the schema depends on the two agreeing.
        // `short.rs` pins the equivalence for its own use; this is the JSON side of the
        // same contract, and it is what keeps "`span` == `primary_ref()[0]`" true.
        debug_assert_eq!(
            diag.spans.first().map(|s| s.file),
            diag.spans.primary_ref().first().map(|s| s.file),
            "spans.first() and primary_ref().first() disagree on the file — the JSON \
             section layout assumes one sorted view"
        );
        let anchor_file = diag.spans.first().map(|s| s.file);
        // `file_name_for` is a spans-first check plus a `foreign_sources` scan, and the
        // overwhelming majority of the labels/spans/children/suggestions/related errors
        // below belong to the anchor file — so the same answer was recomputed once per
        // object.  One resolution, reused for the anchor case.
        let anchor_name: Option<&str> = anchor_file.and_then(|f| diag.file_name_for(f));
        let name_for = |file: crate::ast::FileId| -> Option<&str> {
            if Some(file) == anchor_file {
                anchor_name
            } else {
                diag.file_name_for(file)
            }
        };
        // Every coordinate consumer — the primary span, additional spans, related
        // errors, the call chain, *and* labels/suggestions/children — resolves its
        // `line`/`column` through `anchor_linecol`, which returns `None` unless
        // the span's file is the anchor file.  So `anchor_file.is_some()` is the
        // complete gate: no index is needed (or usable) without an anchor.  The
        // old four extra disjuncts only forced an O(n) line-index build for
        // diagnostics that could not consume it.
        //
        // Deliberate policy this implies: a diagnostic with labels but no primary
        // span gets **no coordinates anywhere**, even with a source attached,
        // because coordinates are anchored to the first primary span's file.
        let needs_index = anchor_file.is_some();
        // One `source()` call per diagnostic: every `anchor_linecol` site below
        // (primary span, additional spans, labels, relatedErrors, children,
        // children labels, suggestions, call chain) used to re-enter the
        // accessor, whose `debug_assert!`s are the natural place for caching or
        // metrics later.  `human.rs`/`short.rs` pin the same one-call contract.
        let src = diag.source();
        let idx = if needs_index {
            src.map(|s| cx.line_index(s))
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
        obj.string_static(diag.level.as_str());

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
                if let Some((line, col)) = anchor_linecol(idx.as_ref(), src, anchor_file, span) {
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
                match name_for(sp.file) {
                    Some(name) => {
                        flat_key0(o, "file");
                        Self::push_json_str(o, name);
                    }
                    None => {
                        flat_key0(o, "fileId");
                        super::fmt::push_u32(o, sp.file.as_u32());
                    }
                }
                flat_key(o, "start");
                super::fmt::push_u32(o, sp.start);
                flat_key(o, "end");
                super::fmt::push_u32(o, sp.end);
                if let Some((line, col)) = anchor_linecol(idx.as_ref(), src, anchor_file, *sp) {
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
                // per-element `JsonContainer` frame (see `flat_key0`).
                let o = &mut *arr.out;
                o.push('{');
                flat_key0(o, "start");
                super::fmt::push_u32(o, lbl.span.start);
                flat_key(o, "end");
                super::fmt::push_u32(o, lbl.span.end);
                // Resolved coordinates, same policy as every other section
                // (`anchor_linecol`): a consumer must not have to reimplement
                // offset → line mapping per section, and it must not have to
                // handle "absent" and "clamped" as separate cases.
                if let Some((line, col)) = anchor_linecol(idx.as_ref(), src, anchor_file, lbl.span)
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
                match name_for(lbl.span.file) {
                    Some(name) => {
                        flat_key(o, "file");
                        Self::push_json_str(o, name);
                    }
                    None => {
                        flat_key(o, "fileId");
                        super::fmt::push_u32(o, lbl.span.file.as_u32());
                    }
                }
                flat_key(o, "message");
                Self::push_json_str(o, &lbl.message);
                flat_key(o, "kind");
                Self::push_json_str_static(o, crate::diagnostics::annotation_kind_name(lbl.kind));
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
                // `first` must be tracked *around* the label block.  The label is
                // the first field whenever present, and emitting it with
                // `flat_key` wrote a leading `, ` while leaving `first` stale, so
                // the following field omitted its comma:
                // `{, "label": …"file": …}` — invalid JSON for every related
                // error carrying a label.
                let mut first = true;
                // `RelatedError::label` is retained precisely for backends like
                // this one (the human renderer shows only message/code).
                //
                // Shape: `{"message": …}`, **not** a `labels[]` element — the
                // field carries no span, so there are no `start`/`end`/`line`/
                // `column`/`kind` to reuse, and the old comment's "a consumer
                // reuses one parser" was simply wrong.  The object (rather than a
                // bare string) keeps room for a future `code`/`kind` without a
                // wire break; `suggestions[].message` stays a bare string for the
                // same reason it always was (there is nothing to nest under it).
                // Absent optionals stay omitted.
                if let Some(lbl) = rel.label.as_ref() {
                    flat_key_dyn(o, &mut first, "label");
                    o.push('{');
                    flat_key0(o, "message");
                    Self::push_json_str(o, lbl);
                    o.push('}');
                }
                // Whether the two sibling arrays are concerned, a related
                // error can point into a *different* file; name it (matching
                // `labels`/`callChain`) instead of emitting a bare span the
                // consumer cannot place.
                // Absent optionals are *omitted*, never `null`.  That is the
                // convention every other nested optional field already uses
                // (`suggestions[].span`/`replacement`, `children[].span`,
                // `callChain[].span`) and what the header contract states, so a
                // consumer needs exactly one "missing" test per section.  An
                // empty code counts as absent, matching `short.rs`, which drops
                // it rather than rendering `error[]`.
                // No element-level `file`/`fileId`: the identity belongs **only**
                // inside the span object below, as it already does for `labels`,
                // `additionalSpans`, `suggestions[].span` and `children[].labels`.
                // The element-level copy existed only when a span existed (so it never
                // named a file the span did not), cost ~`file.len()+10` bytes per
                // related error, and was one more divergence for a consumer with one
                // generic `parse_span_object` to special-case.
                if let Some(c) = rel.code.as_ref().filter(|c| !c.code().is_empty()) {
                    flat_key_dyn(o, &mut first, "code");
                    Self::push_json_str(o, c.code());
                }
                flat_key_dyn(o, &mut first, "message");
                Self::push_json_str(o, &rel.message);
                match rel.span.as_ref() {
                    Some(sp) => {
                        flat_key(o, "span");
                        o.push('{');
                        flat_key0(o, "start");
                        super::fmt::push_u32(o, sp.start);
                        flat_key(o, "end");
                        super::fmt::push_u32(o, sp.end);
                        // Free when the anchor's source is attached *and*
                        // the span belongs to that same file (the common
                        // same-file case); a foreign span's offsets would
                        // otherwise be mapped through the wrong buffer.
                        if let Some((line, col)) =
                            anchor_linecol(idx.as_ref(), src, anchor_file, *sp)
                        {
                            flat_key(o, "line");
                            super::fmt::push_uint(o, line);
                            flat_key(o, "column");
                            super::fmt::push_uint(o, col);
                        }
                        // The file identity belongs *inside* the span object, as
                        // it already does for `labels`, `additionalSpans`,
                        // `suggestions[].span` and `children[].labels`.  It used
                        // to sit only at the element level, so a consumer with one
                        // generic `parse_span_object` had to special-case this one
                        // section — the redundancy costs ~20 bytes per related
                        // error and buys one parse path for every `*.span`.
                        match name_for(sp.file) {
                            Some(name) => {
                                flat_key(o, "file");
                                Self::push_json_str(o, name);
                            }
                            None => {
                                flat_key(o, "fileId");
                                super::fmt::push_u32(o, sp.file.as_u32());
                            }
                        }
                        o.push('}');
                    }
                    None => {}
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
                Self::push_json_str_static(o, child.level.as_str());
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
                    super::fmt::push_u32(o, sp.start);
                    flat_key(o, "end");
                    super::fmt::push_u32(o, sp.end);
                    if let Some((line, col)) = anchor_linecol(idx.as_ref(), src, anchor_file, sp) {
                        flat_key(o, "line");
                        super::fmt::push_uint(o, line);
                        flat_key(o, "column");
                        super::fmt::push_uint(o, col);
                    }
                    match name_for(sp.file) {
                        Some(name) => {
                            flat_key(o, "file");
                            Self::push_json_str(o, name);
                        }
                        None => {
                            flat_key(o, "fileId");
                            super::fmt::push_u32(o, sp.file.as_u32());
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
                        super::fmt::push_u32(o, lbl.span.start);
                        flat_key(o, "end");
                        super::fmt::push_u32(o, lbl.span.end);
                        if let Some((line, col)) =
                            anchor_linecol(idx.as_ref(), src, anchor_file, lbl.span)
                        {
                            flat_key(o, "line");
                            super::fmt::push_uint(o, line);
                            flat_key(o, "column");
                            super::fmt::push_uint(o, col);
                        }
                        match name_for(lbl.span.file) {
                            Some(name) => {
                                flat_key(o, "file");
                                Self::push_json_str(o, name);
                            }
                            None => {
                                flat_key(o, "fileId");
                                super::fmt::push_u32(o, lbl.span.file.as_u32());
                            }
                        }
                        flat_key(o, "message");
                        Self::push_json_str(o, &lbl.message);
                        flat_key(o, "kind");
                        Self::push_json_str_static(
                            o,
                            crate::diagnostics::annotation_kind_name(lbl.kind),
                        );
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
                Self::push_json_str_static(
                    o,
                    crate::diagnostics::applicability_name(sug.applicability),
                );
                flat_key(o, "style");
                Self::push_json_str_static(o, crate::diagnostics::suggestion_style_name(sug.style));
                // A machine consumer that wants to apply the fix needs the
                // edit itself, which the model already carries.  Emit the span
                // (using the same file/fileId convention as labels) and the
                // replacement text; both are optional by design.
                if let Some(sp) = sug.span {
                    flat_key(o, "span");
                    o.push('{');
                    flat_key0(o, "start");
                    super::fmt::push_u32(o, sp.start);
                    flat_key(o, "end");
                    super::fmt::push_u32(o, sp.end);
                    if let Some((line, col)) = anchor_linecol(idx.as_ref(), src, anchor_file, sp) {
                        flat_key(o, "line");
                        super::fmt::push_uint(o, line);
                        flat_key(o, "column");
                        super::fmt::push_uint(o, col);
                    }
                    match name_for(sp.file) {
                        Some(name) => {
                            flat_key(o, "file");
                            Self::push_json_str(o, name);
                        }
                        None => {
                            flat_key(o, "fileId");
                            super::fmt::push_u32(o, sp.file.as_u32());
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
                Self::push_json_str_static(
                    o,
                    crate::diagnostics::chain_entry_kind_name(entry.kind),
                );
                // a chain entry pointing into a foreign file must carry
                // that file's identity; otherwise its location is silently
                // dropped and a JSON consumer cannot place it.
                if let Some(span) = entry.span {
                    // Resolve the entry's owning file name *once* and use it for
                    // both the `file`/`fileId` field and `location`.  In the
                    // common case that keeps a single object from naming two
                    // different files; the one exception is "the entry is in the
                    // anchor file but the anchor has no resolvable display name",
                    // where `file` degrades to `fileId` while `location` uses the
                    // hoisted fallback name — the coordinates are still right,
                    // and `call_chain_location_falls_back_when_anchor_is_unresolvable`
                    // pins that branch rather than letting the claim stand as an
                    // absolute.
                    let entry_file = name_for(span.file);
                    match entry_file {
                        Some(name) => {
                            flat_key(o, "file");
                            Self::push_json_str(o, name);
                        }
                        None => {
                            flat_key(o, "fileId");
                            super::fmt::push_u32(o, span.file.as_u32());
                        }
                    }
                    if let Some((line, col)) = anchor_linecol(idx.as_ref(), src, anchor_file, span)
                    {
                        // Numeric fields a consumer can parse without splitting
                        // a composite string: file names may contain `:`
                        // (Windows drives, `ns::item`), which made `location`
                        // ambiguous.
                        flat_key(o, "line");
                        super::fmt::push_uint(o, line);
                        flat_key(o, "column");
                        super::fmt::push_uint(o, col);
                        flat_key(o, "location");
                        flat_location(o, entry_file.unwrap_or(file_name), line, col);
                    }
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
    // One reserve for `key` plus the punctuation, so the pushes below do not
    // each re-check capacity; this runs once per field of every related error,
    // label and suggestion.  `", \"" (3) + key + "\": " (3)` is `key.len() + 6`:
    // the old `+ 5` under-reserved by one byte, so the final `push_str` still
    // re-checked capacity — exactly the invariant this reserve exists to hold.
    out.reserve(key.len() + 6);
    // One `push_str` for `, "` instead of two: these run ~8x per label /
    // related error / suggestion, so the saved capacity check is not noise.
    out.push_str(", \"");
    out.push_str(key);
    out.push_str("\": ");
}

/// Like [`flat_key`] / [`flat_key0`], choosing the separator from a running
/// `first` flag.
///
/// The manual `if first { flat_key0 } else { flat_key }` pattern had to be
/// threaded through every optional field of the related-error object, and it
/// already produced invalid JSON once (`{, "label": …}`): the flag update and
/// the key emission were two edits at each site.  One call site per field makes
/// that impossible; `flat_key`/`flat_key0` remain for the fields whose position
/// is statically known.
fn flat_key_dyn(out: &mut String, first: &mut bool, key: &str) {
    if *first {
        *first = false;
        flat_key0(out, key);
    } else {
        flat_key(out, key);
    }
}

/// First field of a flat object: no separator (the `{` was just pushed).
fn flat_key0(out: &mut String, key: &str) {
    // One reserve for `key` plus the punctuation, so the pushes below do not
    // each re-check capacity; this runs once per field of every related error,
    // label and suggestion.
    // `"` (1) + key + `": "` (3) == key.len() + 4 for the
    // leading-quote form; the `, "` variant needs one more.
    out.reserve(key.len() + 4);
    out.push_str("\"");
    out.push_str(key);
    out.push_str("\": ");
}

/// `push_json_str`-shaped helper for the inline flat-object path.
fn flat_location(out: &mut String, file: &str, line: u64, col: u64) {
    // One reserve for `"file:line:col"`: the escaped file may expand, so this
    // covers the common (clean) case and the escapes below grow amortized.  The
    // other flat helpers all reserve; this one did not.
    out.reserve(file.len().saturating_mul(2).saturating_add(24));
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
    /// Buffer length right after the last [`Self::key`] was written: the position
    /// a value must appear at.  `Drop` compares against `out.len()`, so a key left
    /// without a value auto-closes as `"x": null` instead of the invalid `"x": }`.
    ///
    /// A position marker rather than a `pending: bool` so no value method has to
    /// cooperate: any value write appends bytes, so equality alone answers it and a
    /// missed method cannot desynchronise the flag.
    pending_value_at: Option<usize>,
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
            pending_value_at: None,
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
            pending_value_at: None,
        }
    }

    /// `{"k": v, "k": v}` — always compact: array elements render as one
    /// line each regardless of the pretty setting (matching the previous
    /// per-item output exactly).
    #[must_use]
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
            pending_value_at: None,
        }
    }

    /// Emit `null` for a key written without a value.
    ///
    /// Called from both `close()` and `Drop`: repairing only in `Drop` meant
    /// `close()` (which sets `closed = true` and so silences the tripwire) was the
    /// one exit path where a dangling key shipped as `"x": }`.
    fn repair_pending(&mut self) {
        if self
            .pending_value_at
            .is_some_and(|mark| self.out.len() == mark)
        {
            self.out.push_str("null");
            self.pending_value_at = None;
        }
    }

    /// Begin the next element: separator first, then (in pretty mode) the
    /// newline and indentation.
    fn element(&mut self) {
        // Repair first: `key("x")` followed by another `key(..)`/`element()` would
        // otherwise write the separator and make `out.len() > mark`, after which no
        // repair path can fire and the stream is invalid.  One well-predicted
        // branch on a path that already does several `push_str`s.
        if let Some(mark) = self.pending_value_at.take() {
            // Only a *dangling* key is repaired: the flat helpers write their
            // value straight into `out` without clearing the marker, so any bytes
            // after the mark mean the value did arrive.
            if self.out.len() == mark {
                self.out.push_str("null");
            }
        }
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
        // One reserve for the five pushes below (separator/newline/indent are
        // already written, so only this field's bytes count).  `key()` runs
        // ~15 times per diagnostic; without this each of the four pushes
        // re-checked capacity, contradicting the discipline every other writer
        // in this file follows.
        self.out.reserve(self.elem_ind.len() + key.len() + 5);
        self.out.push('"');
        self.out.push_str(key);
        self.out.push_str("\": ");
        self.pending_value_at = Some(self.out.len());
    }

    /// Value: a JSON string literal.
    fn string(&mut self, s: &str) {
        JsonRenderer::push_json_str(self.out, s);
    }

    /// Value: a JSON string literal from a fixed vocabulary word (see
    /// [`JsonRenderer::push_json_str_static`]) — no escape scan.
    fn string_static(&mut self, s: &str) {
        JsonRenderer::push_json_str_static(self.out, s);
    }

    /// Value: pre-rendered JSON (`null`, a small inline object, …).
    fn raw(&mut self, s: &str) {
        // A zero-byte payload would leave `"key": }` on the wire because the
        // position marker assumes every value write appends bytes (`string`
        // writes ≥ 2, `u64` ≥ 1, `u64_hex` ≥ 18).  Degrade to `null` — the same
        // repair the `Drop` tripwire applies — and keep it loud in debug.
        if s.is_empty() {
            debug_assert!(false, "raw(\"\") would emit `\"key\": }}`; emitting null");
            self.out.push_str("null");
        } else {
            self.out.push_str(s);
        }
        self.pending_value_at = None;
    }

    /// Value: a bare `u64` (JSON number) — no quoting, no `core::fmt`.
    fn u64(&mut self, n: u64) {
        super::fmt::push_uint(self.out, n);
    }

    /// Value: a JSON string literal `"0x{:016x}"` — the stable diagnostic id.
    fn u64_hex(&mut self, n: u64) {
        // `"0x` + 16 hex digits + `"` is exactly 18 bytes, and this was the only writer in
        // this file without a reserve — the other three push once or route through one.
        self.out.reserve(18);
        self.out.push_str("\"0x");
        super::fmt::push_hex64(self.out, n);
        self.out.push('"');
    }

    /// Close the container, indenting the bracket in pretty mode.
    fn close(mut self) {
        // Repair *before* the newline and the closing bracket: a dangling key
        // must get its value on the key's own line, and `"x": }` is invalid JSON
        // either way.  The doc above `repair_pending` claimed this call existed;
        // it did not, and `Drop`'s tripwire is silenced by `closed = true`, so
        // `key("x")` followed by `close()` had no repair path at all in release.
        self.repair_pending();
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
        // Not while unwinding: a render panic (the very thing
        // `render_internal_error` contains) unwinds through the live containers,
        // and an assert *panicking during unwinding* aborts the process,
        // destroying that containment.  A forgotten `close()` in normal control
        // flow still trips.
        if !std::thread::panicking() {
            debug_assert!(
                false,
                "JsonContainer dropped without close() — auto-closing to keep the stream valid"
            );
        }
        // A key whose value never arrived would auto-close as `"x": }`, which does
        // not parse; `null` keeps the degradation valid.  A missing *element*
        // cannot be repaired this way (the flat helpers bypass the bookkeeping),
        // so a container dropped right after `element()` can still leave a trailing
        // comma — the doc says so rather than overclaiming.
        self.repair_pending();
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
/// **Test-only.**  Production dispatch lives inside [`push_json_escaped_str`]
/// (fused scan ≤ [`SHORT_FUSED_MAX`], `memchr2` + bounded control scan above);
/// this entry point exists so the differential test can pin the *positional*
/// contract of the same predicate.  It used to be described here as the
/// production dispatcher, which was a doc-only architecture that the code did
/// not have.
#[cfg(test)]
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
/// Lane mask for the three JSON escape classes in one 8-byte word.
///
/// `hasless` for the control range (`< 0x20`; the lowest flagged lane is always
/// a true hit — a borrow can only flag a byte that is itself `>= 0x20` *after* a
/// real `< 0x20` lane) plus Mycroft `has_zero` for `"` and `\\`.
#[inline]
fn escape_mask(x: u64) -> u64 {
    lt_20(x) | eq_zero(x ^ QUOTE) | eq_zero(x ^ BACKSLASH)
}

/// Exact per-lane "byte `< 0x20`" mask — **no inter-lane borrow**.
///
/// The classic `hasless` trick (`(x - 0x2020…) & !x & HI`) is only
/// *existentially* exact: a real hit borrows into the lane above it, so a
/// `0x20` sitting right after a control byte is flagged too.  That was harmless
/// while every caller read only the **lowest** flagged lane (a borrow can only
/// flag a lane *above* a true hit).  Draining the whole chunk — which is what
/// makes `EscapeIter`/`CtrlIter` single-pass — turns the false positives into
/// real ones, and `push_json_escaped` then indexes `C0_ESCAPED[0x20]` out of
/// bounds: a lost diagnostic, not a cosmetic bug.
///
/// Fix: compute the comparison without subtraction.  `(b & 0x7F) + (0x80 - 0x20)`
/// has bit 7 set iff `(b & 0x7F) >= 0x20`, and the largest lane sum is
/// `0x7F + 0x60 = 0xDF < 0x100`, so no lane can carry into its neighbour.  The
/// `!x` term drops bytes `>= 0x80` (whose low seven bits may look small).
#[inline]
fn lt_20(x: u64) -> u64 {
    const LO7: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ADD20: u64 = 0x6060_6060_6060_6060;
    const HI: u64 = 0x8080_8080_8080_8080;
    !x & !((x & LO7).wrapping_add(ADD20)) & HI
}

/// Exact per-lane "byte `== 0`" mask — **no inter-lane borrow**.
///
/// Mycroft `has_zero` (`(v - LO) & !v & HI`) is likewise only existentially
/// exact: after a true zero lane, the lane above it is flagged whatever its
/// value (a `'#'` after a `'"'`, a `']'` after a `'\\'`).  `(v & 0x7F) + 0x7F`
/// has bit 7 set for every non-zero low seven bits, and tops out at
/// `0x7F + 0x7F = 0xFE`, so no lane carries.
#[inline]
fn eq_zero(v: u64) -> u64 {
    const LO7: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const HI: u64 = 0x8080_8080_8080_8080;
    !v & !((v & LO7).wrapping_add(LO7)) & HI
}

/// Lane mask for JSON control bytes (`< 0x20`) in one 8-byte word.
#[inline]
fn ctrl_mask(x: u64) -> u64 {
    lt_20(x)
}

/// `0x22` broadcast — `'"'`.
const QUOTE: u64 = 0x2222_2222_2222_2222;
/// `0x5C` broadcast — `'\\'`.
const BACKSLASH: u64 = 0x5C5C_5C5C_5C5C_5C5C;

/// One pass over `b`, yielding the position of every escape byte in ascending
/// order.
///
/// The previous shape re-entered the scanner at `hit + 1`, so an escape-dense
/// string (a message full of `\\n`, or a binary snippet) re-scanned the same
/// 8-byte chunk up to eight times: linear, but with an 8× constant on exactly the
/// inputs that are already slow.  Draining the whole chunk mask before advancing
/// makes each chunk cost one SWAR step.
///
/// Tail handling pads with `0xFF`, whose lanes cannot flag: `(0xFF - 0x20) & !0xFF`
/// is 0 and `has_zero(0xFF ^ c)` is 0 for every `c` the masks use.
struct EscapeIter<'a> {
    b: &'a [u8],
    /// Start of the next chunk to scan.
    pos: usize,
    /// Start of the chunk `mask` describes.
    chunk: usize,
    mask: u64,
}

impl<'a> EscapeIter<'a> {
    #[inline]
    fn new(b: &'a [u8]) -> Self {
        EscapeIter {
            b,
            pos: 0,
            chunk: 0,
            mask: 0,
        }
    }
}

impl Iterator for EscapeIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.mask != 0 {
                let lane = (self.mask.trailing_zeros() / 8) as usize;
                // Clear this lane; every lane keeps its flag in bit 7 of its byte.
                self.mask &= !(0x80u64 << (lane * 8));
                return Some(self.chunk + lane);
            }
            let remaining = self.b.len() - self.pos;
            if remaining >= 8 {
                let x = u64::from_le_bytes(
                    self.b[self.pos..self.pos + 8]
                        .try_into()
                        .expect("8-byte chunk"),
                );
                self.mask = escape_mask(x);
                self.chunk = self.pos;
                self.pos += 8;
                continue;
            }
            if remaining == 0 {
                return None;
            }
            let mut buf = [0xFFu8; 8];
            buf[..remaining].copy_from_slice(&self.b[self.pos..]);
            self.mask = escape_mask(u64::from_le_bytes(buf));
            self.chunk = self.pos;
            self.pos += remaining;
        }
    }
}

/// One pass over `b`, yielding the position of every JSON control byte
/// (`< 0x20`) in ascending order.
///
/// Same restructuring as [`EscapeIter`], for the long path's control scan.
struct CtrlIter<'a> {
    b: &'a [u8],
    pos: usize,
    chunk: usize,
    mask: u64,
}

impl<'a> CtrlIter<'a> {
    #[inline]
    fn new(b: &'a [u8]) -> Self {
        CtrlIter {
            b,
            pos: 0,
            chunk: 0,
            mask: 0,
        }
    }
}

impl Iterator for CtrlIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.mask != 0 {
                let lane = (self.mask.trailing_zeros() / 8) as usize;
                self.mask &= !(0x80u64 << (lane * 8));
                return Some(self.chunk + lane);
            }
            let remaining = self.b.len() - self.pos;
            if remaining >= 8 {
                let x = u64::from_le_bytes(
                    self.b[self.pos..self.pos + 8]
                        .try_into()
                        .expect("8-byte chunk"),
                );
                self.mask = ctrl_mask(x);
                self.chunk = self.pos;
                self.pos += 8;
                continue;
            }
            if remaining == 0 {
                return None;
            }
            let mut buf = [0xFFu8; 8];
            buf[..remaining].copy_from_slice(&self.b[self.pos..]);
            self.mask = ctrl_mask(u64::from_le_bytes(buf));
            self.chunk = self.pos;
            self.pos += remaining;
        }
    }
}

/// First escape byte in `b`, or `None`.  Thin wrapper over [`EscapeIter`] so the
/// positional contract and the production drain share one implementation.
#[inline]
fn next_escape_swar(b: &[u8]) -> Option<usize> {
    EscapeIter::new(b).next()
}

/// First byte `< 0x20` in `b` (JSON control range), or `None`.
///
/// [`CtrlIter`] with the per-lane exact [`lt_20`] mask — **not** Mycroft's `hasless`
/// (`(x - 0x20) & ~x & 0x80`), which this comment used to describe.  That form is
/// first-hit safe (a cross-byte borrow can only flag a lane above a true hit) but not
/// lane-exact, and these iterators drain *every* flagged lane per chunk, so an extra
/// flagged lane would become a wrong escape site.  `escape_masks_are_exact_per_lane`
/// pins the property the implementation relies on; the borrow argument would justify
/// the opposite implementation.
#[inline]
fn next_ctrl_byte(b: &[u8]) -> Option<usize> {
    CtrlIter::new(b).next()
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
/// Bytes [`push_json_escaped_str`] adds on top of `bytes.len()`.
///
/// Walks 8 bytes at a time with the same [`escape_mask`] the scanner uses, so the count
/// cannot drift from the bytes actually written *and* mirrors the writer's three weight
/// classes: structural (`"`, `\` → 2 bytes, +1), the named C0 escapes (`\n`, `\r`, `\t`
/// → 2 bytes, +1, the one class the first sketch of this helper got wrong by charging
/// the full `\u00XX` width), and every other C0 byte (`\u00XX` → 6 bytes, +5).
fn escaped_extra(bytes: &[u8]) -> usize {
    const NEWLINE: u64 = 0x0A0A_0A0A_0A0A_0A0A;
    const CR: u64 = 0x0D0D_0D0D_0D0D_0D0D;
    const TAB: u64 = 0x0909_0909_0909_0909;
    let mut extra = 0usize;
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        // `try_into` cannot fail: the window is exactly 8 bytes.
        let x = u64::from_le_bytes(
            bytes[i..i + 8]
                .try_into()
                .expect("eight-byte window for the escape mask"),
        );
        let mask = escape_mask(x);
        if mask != 0 {
            let structural =
                ((eq_zero(x ^ QUOTE) | eq_zero(x ^ BACKSLASH)) & mask).count_ones() as usize;
            let named = ((eq_zero(x ^ NEWLINE) | eq_zero(x ^ CR) | eq_zero(x ^ TAB)) & mask)
                .count_ones() as usize;
            let wide = mask.count_ones() as usize - structural - named;
            extra += structural + named + wide * 5;
        }
        i += 8;
    }
    for &b in &bytes[i..] {
        extra += match b {
            b'"' | b'\\' | b'\n' | b'\r' | b'\t' => 1,
            _ if b < 0x20 => 5,
            // Not flagged by the scanner, so it is copied verbatim.
            _ => 0,
        };
    }
    extra
}

fn push_json_escaped_str(out: &mut String, s: &str) {
    let bytes = s.as_bytes();
    // Short strings (diagnostic messages, labels, help) take the fused scan:
    // one pass finds all three escape classes, and `memchr2`'s wider vectors
    // cannot amortize their setup below 64 bytes.  Each iteration resumes right
    // after the previous escape, so the total remains linear in `len`.
    if bytes.len() <= SHORT_FUSED_MAX {
        // Drain every escape in the current 8-byte chunk before advancing: the
        // old shape restarted the scan at `at + 1`, re-scanning the same chunk
        // up to eight times on escape-dense input.
        // Exact capacity without buffering positions: the escape mask already has one
        // bit per hit lane, so `count_ones()` *is* the hit count once the structural
        // class (`"`/`\`, +1 each) is subtracted from the C0 class (`\u00XX`, +5 each).
        // The previous shape collected positions into `SmallVec<[usize; 8]>`, which
        // spilled to the heap at the **ninth** escape — an ordinary nine-line message —
        // defeating the single-allocation guarantee this reserve exists for, and it
        // buffered data the drain below re-derives for free.
        out.reserve(bytes.len().saturating_add(escaped_extra(bytes)));
        let mut start = 0usize;
        for at in EscapeIter::new(bytes) {
            if start < at {
                out.push_str(&s[start..at]);
            }
            push_json_escaped(out, bytes[at]);
            start = at + 1;
        }
        if start < bytes.len() {
            out.push_str(&s[start..]);
        }
        return;
    }
    // Stateful merged scan: the structural iterator (`"` / `\`) advances only
    // when its byte is actually escaped, and the control scan runs only up to
    // the next structural position.  Re-deriving the structural hit from
    // `bytes[start..]` on every iteration made control-heavy strings with no
    // quotes/backslashes O(n²) (one full `memchr2` miss per escaped control);
    // this makes the total scan linear.
    // Two single-pass streams, merged: `memchr2` (SIMD) for the quote and
    // backslash, and the SWAR control iterator.  The old shape restarted the
    // control scan at `start` after every escape, re-scanning already-consumed
    // chunks on control-dense input; a resumable iterator keeps its position, so
    // each byte is examined exactly once.
    let mut structural = memchr::memchr2_iter(b'"', b'\\', bytes);
    let mut ctrl = CtrlIter::new(bytes);
    let mut next_structural = structural.next();
    let mut next_ctrl = ctrl.next();
    let mut start = 0usize;
    loop {
        let i = match (next_ctrl, next_structural) {
            (Some(c), Some(st)) => c.min(st),
            (Some(c), None) => c,
            (None, Some(st)) => st,
            (None, None) => break,
        };
        if i > start {
            out.push_str(&s[start..i]);
        }
        // Every escape candidate is ASCII, so `bytes[i]` is a char boundary.
        push_json_escaped(out, bytes[i]);
        start = i + 1;
        if next_ctrl == Some(i) {
            next_ctrl = ctrl.next();
        }
        if next_structural == Some(i) {
            next_structural = structural.next();
        }
    }
    if start < bytes.len() {
        out.push_str(&s[start..]);
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
/// The 32 C0 controls as JSON escapes, indexed by code point.
///
/// A closed table replaces building a six-byte `\uXXXX` array and *validating*
/// it (`simdutf8::basic::from_utf8(..).unwrap()`) on every control escape — which
/// also made `simdutf8` a runtime dependency the module's header contract does
/// not mention.  The named escapes (`\"`, `\\`, `\n`, `\r`, `\t`) are matched
/// before this table is consulted, so their slots are unreachable.
/// Pretty-print indentation, two spaces per level.
///
/// At module scope (not inside `indent`) so `write_diagnostic::<LVL>` can index
/// it with a compile-time constant, which folds the bounds check and the load
/// away entirely; the runtime accessor is the fallback for callers that cannot
/// name a level.
/// One indentation step per nesting level.
///
/// An array, not a slice: the length is part of the type, and — the point of the
/// change — the `LVL` bound can then be asserted in a `const` block inside the
/// const-generic writer, which is evaluated **per monomorphization**.  An
/// out-of-range `write_diagnostic::<6>` becomes a build error naming the
/// instantiation instead of a bare index panic (or, at `LVL = 5`, a *silent*
/// saturation: `indent(LVL + 1)` saturates, so the layout was merely wrong).
const INDENTS: [&str; 6] = ["", "  ", "    ", "      ", "        ", "          "];

/// Nesting level of one batch element: `[` at 0, the object's `{` at this level,
/// its fields one deeper.  Named because two sites must agree — `render_into`
/// writes real elements here, and `render_internal_error` must lay its
/// placeholder out identically or the array visually breaks on a panic.
const BATCH_LEVEL: usize = 2;

/// `id` reported by the panic placeholder.
///
/// Every real element carries `"id": "0x<stable_id>"`; a consumer that reads
/// `id` unconditionally must not meet an element without one.  This is a fixed
/// sentinel (not a hash of anything the failing diagnostic might have produced),
/// so a panic is identifiable and does not move between runs.
///
/// `pub(crate)` because `Diagnostic::compute_stable_id` steers a colliding hash
/// away from it: the sentinel only identifies a panic if no real diagnostic can
/// wear the same id.
pub(crate) const INTERNAL_ERROR_ID: u64 = 0x1DEA_D1ED_1DEA_D1ED;

// Indent widths are not pinned
// by a test (pinning a guessed layout is how such a test rots), so a corrupted
// table must fail the build instead of shipping valid-but-misaligned output.  (An
// earlier version of this comment cited a `fmt.rs` machine-check on a `SPACES`
// const; no such const exists, so the reference is gone.)
const _: () = {
    let mut i = 0;
    let mut ok = true;
    while i < INDENTS.len() {
        let b = INDENTS[i].as_bytes();
        if b.len() != i * 2 {
            ok = false;
        }
        let mut j = 0;
        while j < b.len() {
            if b[j] != b' ' {
                ok = false;
            }
            j += 1;
        }
        i += 1;
    }
    assert!(ok, "INDENTS[i] must be exactly 2*i spaces (two per level)");
};

/// Machine-checked: every entry is exactly `\uXXXX` (6 ASCII bytes), the shape
/// the six-byte fast path and the `\u` decode both assume.  A stray edit to a
/// literal is a compile error instead of a wrong escape on the wire.
const _: () = {
    let mut i = 0;
    while i < 32 {
        // Content, not shape: entry *i* must encode byte *i* exactly.  Checking only
        // `len() == 6` and a leading backslash let a typo (`"\u000c"` pasted at index 8), a
        // reorder, or an editor auto-fix pass the assert and silently emit the wrong escape
        // on the wire — while the doc above claimed "a stray edit is a compile error".
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let entry = C0_ESCAPED[i].as_bytes();
        assert!(entry.len() == 6, "C0 escape entries are 6 bytes");
        assert!(
            entry[0] == b'\\' && entry[1] == b'u' && entry[2] == b'0' && entry[3] == b'0',
            "C0 entries are spelled \\u00XX"
        );
        assert!(
            entry[4] == HEX[i >> 4] && entry[5] == HEX[i & 0xF],
            "C0_ESCAPED[i] must be the exact escape for i"
        );
        i += 1;
    }
};
const C0_ESCAPED: [&str; 32] = [
    "\\u0000", "\\u0001", "\\u0002", "\\u0003", "\\u0004", "\\u0005", "\\u0006", "\\u0007",
    "\\u0008", "\\u0009", "\\u000a", "\\u000b", "\\u000c", "\\u000d", "\\u000e", "\\u000f",
    "\\u0010", "\\u0011", "\\u0012", "\\u0013", "\\u0014", "\\u0015", "\\u0016", "\\u0017",
    "\\u0018", "\\u0019", "\\u001a", "\\u001b", "\\u001c", "\\u001d", "\\u001e", "\\u001f",
];

fn push_json_escaped(out: &mut String, b: u8) {
    // The scanner (`next_escape_swar`, or the long path's `memchr2` + control
    // scan) only ever *selects bytes*: `"`, `\`, or a byte `< 0x20`.  Taking
    // `char` forced two arms the scanner can never reach — hex-building for C1
    // controls and a `write_char` fallback — into the hottest escape path, and
    // made the match non-total over the real domain.  The doc used to promise
    // `\u007f` output that was unreachable by construction.
    debug_assert!(
        b < 0x20 || b == b'"' || b == b'\\',
        "push_json_escaped called with a byte the scanner would not select: {b:#04x}"
    );
    out.push_str(match b {
        b'"' => "\\\"",
        b'\\' => "\\\\",
        b'\n' => "\\n",
        b'\r' => "\\r",
        b'\t' => "\\t",
        // `0x00..=0x1f` other than the named escapes above: one table load.
        b => C0_ESCAPED[b as usize],
    });
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::{FileId, Span};
    use crate::diagnostics::ErrCode;
    use crate::diagnostics::chain::{CallChain, ChainEntry, ChainEntryKind};
    use compact_str::CompactString;

    /// Absent optionals inside `relatedErrors` are **omitted**, matching every
    /// other nested optional field (`suggestions[].span/replacement`,
    /// `children[].span`, `callChain[].span`) and the header contract's "absent,
    /// not null".  An empty code counts as absent, like `short.rs`.
    ///
    /// The label is the first field of the related-error object whenever it is
    /// present, which is exactly where a `flat_key` (comma-prefixed) write and a
    /// stale `first` both produced invalid JSON.  Pin the whole shape through a
    /// real parser, not a substring check.
    #[test]
    fn related_errors_with_a_label_round_trip() {
        use crate::diagnostics::RelatedError;
        let span = Span::new(FileId::DUMMY, 3, 9);
        for (label, code) in [
            (Some("L"), None),
            (Some("L"), Some(crate::diagnostics::ErrCode::new("E030"))),
            (None, Some(crate::diagnostics::ErrCode::new("E030"))),
            // A present-but-empty label must behave like any other present
            // label: emitted as `""`, and the following field still gets its
            // comma.  (`Some("")` is reachable from a backend that sanitized the
            // label to nothing.)
            (Some(""), Some(crate::diagnostics::ErrCode::new("E030"))),
        ] {
            let diag = Diagnostic::error("outer").with_related(RelatedError {
                code,
                message: "inner".into(),
                span: Some(span),
                label: label.map(std::borrow::Cow::Borrowed),
            });
            for pretty in [false, true] {
                let out = json_of(&diag, pretty);
                let v: serde_json::Value =
                    serde_json::from_str(&out).expect("relatedErrors must stay valid JSON");
                let rel = &v["relatedErrors"][0];
                assert_eq!(rel["message"], "inner");
                assert_eq!(rel["span"]["start"], 3);
                match label {
                    Some(l) => assert_eq!(rel["label"]["message"], l),
                    None => assert!(rel.get("label").is_none(), "absent label must be omitted"),
                }
                // A leading comma (or a missing one after the label) is exactly
                // what the broken `first` tracking produced; the parser above
                // already rejects both, this only names the failure.
                assert!(
                    !out.contains("{,") && !out.contains("{ ,"),
                    "leading comma in relatedErrors: {out}"
                );
            }
        }
    }

    #[test]
    fn related_errors_omit_absent_code_and_span() {
        use crate::diagnostics::RelatedError;
        let diag = Diagnostic::error("outer").with_related(RelatedError {
            code: None,
            message: "plain".into(),
            span: None,
            label: None,
        });
        let out = json_of(&diag, false);
        let section = out
            .split_once("\"relatedErrors\"")
            .map(|(_, rest)| rest)
            .expect("relatedErrors section present");
        // `flat_key` writes `"key": ` (with a space) even in compact mode.
        assert!(section.contains("\"message\""), "{section}");
        assert!(section.contains("plain"), "{section}");
        assert!(
            !section.contains("\"code\""),
            "absent code must be omitted, not null: {section}"
        );
        assert!(
            !section.contains("\"span\""),
            "absent span must be omitted, not null: {section}"
        );
    }

    /// Pretty mode is one field per line; compact mode is exactly one line — the
    /// property a consumer of the machine format relies on.  (Neither mode's
    /// indent widths are asserted as a literal here: pinning a guessed layout is
    /// how such a test rots.)
    #[test]
    fn pretty_is_multiline_and_compact_is_single_line() {
        let span = Span::new(FileId::DUMMY, 12, 13);
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(span)
            .with_label(span, "not found");
        let pretty = json_of(&diag, true);
        assert!(pretty.starts_with('{'), "{pretty}");
        assert!(pretty.ends_with("\n}"), "{pretty}");
        assert!(pretty.lines().count() > 2, "{pretty}");
        let compact = json_of(&diag, false);
        assert!(
            !compact.trim_end_matches('\n').contains('\n'),
            "compact must be one line: {compact}"
        );
    }

    fn json_of(diag: &Diagnostic<'_>, pretty: bool) -> String {
        let opts = crate::diagnostics::DiagOpts::default();
        let cx = crate::diagnostics::render::RenderCtx::new(&opts, None);
        let mut out = String::new();
        JsonRenderer::new(pretty).write_diagnostic::<0>(diag, &cx, &mut out);
        out
    }

    /// Escaping must keep short values inline (no heap) on **both** paths —
    /// the clean fast path and the escaping slow path. This pins the inline
    /// guarantee that the single-buffer assembly relies on: a diagnostic's
    /// JSON is now built in one `String`, and short literals never push it
    /// through an intermediate `CompactString` on the heap.
    #[test]
    fn push_json_str_inlines_short_values() {
        // The writer is `&mut String` (the production buffer), so the inline
        // property is measured on the conversion: a short escaped result still
        // fits `CompactString`'s 24-byte inline limit, a long one does not.
        fn render(s: &str) -> CompactString {
            let mut out = String::with_capacity(s.len() + 2);
            JsonRenderer::push_json_str(&mut out, s);
            CompactString::from(out)
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

    /// A chain entry outside the anchor file can be *named* but not *located*:
    /// `anchor_linecol` only computes coordinates for anchor-file spans, so the
    /// entry must carry `file` and no `line`/`column`/`location` — never a
    /// half-filled location and never the anchor's name in place of the real one.
    #[test]
    fn call_chain_foreign_entry_names_file_and_omits_coordinates() {
        let foreign = FileId::from_raw(2);
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(foreign, 5, 9)),
            "from real.rs",
        ));
        let diag = Diagnostic::error("x")
            .with_source("a\nb\n")
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_foreign_source(foreign, "real.rs", "a\nb\n")
            .with_call_chain(chain);
        let v: serde_json::Value =
            serde_json::from_str(&json_of(&diag, false)).expect("valid JSON");
        let entry = &v["callChain"][0];
        assert_eq!(entry["file"], "real.rs", "{entry}");
        for absent in ["line", "column", "location", "fileId"] {
            assert!(
                entry.get(absent).is_none(),
                "{absent} must be absent: {entry}"
            );
        }
    }

    #[test]
    fn call_chain_file_and_location_agree_under_file_name_override() {
        // Anchor span in the diagnostic's own file, whose display name is
        // overridden; the chain entry points into a foreign file.
        let foreign = FileId::from_raw(2);
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(foreign, 0, 1)),
            "from real.rs",
        ));
        let diag = Diagnostic::error("x")
            .with_file_name("custom.pn")
            .with_source("a\nb\n")
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_foreign_source(foreign, "real.rs", "a\nb\n")
            .with_call_chain(chain);
        let v: serde_json::Value =
            serde_json::from_str(&json_of(&diag, false)).expect("valid JSON");
        let entry = &v["callChain"][0];
        assert_eq!(entry["file"], "real.rs", "{entry}");
        // Coordinates are only emitted for spans in the anchor file (the line
        // index is the anchor's), but if they were emitted they must describe
        // the same file as `file` — both now come from one resolution.
        if let Some(loc) = entry["location"].as_str() {
            assert!(
                loc.starts_with("real.rs:"),
                "file/location disagree: {entry}"
            );
        }

        // An entry in the anchor file takes the overridden name in both
        // places.
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 0, 1)),
            "top",
        ));
        let diag = Diagnostic::error("x")
            .with_file_name("custom.pn")
            .with_source("a\nb\n")
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_call_chain(chain);
        let v: serde_json::Value =
            serde_json::from_str(&json_of(&diag, false)).expect("valid JSON");
        let entry = &v["callChain"][0];
        assert_eq!(entry["file"], "custom.pn", "{entry}");
        assert_eq!(entry["location"], "custom.pn:1:1", "{entry}");
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
    /// A corrupt span may start *inside* a multi-byte codepoint — the same
    /// corruption class as an EOF-crossing offset.  `anchor_linecol` clamps to
    /// `source.len()`; the char-boundary floor lives one level down in
    /// `LineIndex::to_linecol`, so the whole class is total (`floor_char_boundary`)
    /// rather than slicing a `&str` mid-codepoint.  Pin it here so a future
    /// "optimisation" that indexes the source directly cannot regress it.
    #[test]
    fn span_start_inside_a_multibyte_char_lands_on_the_boundary() {
        // (source, span.start, expected 1-based column after the floor)
        for (src, inside, want_col) in [
            ("éa\n", 1u32, 1u64),
            ("雪a\n", 1, 1),
            ("雪a\n", 2, 1),
            // Inside 雪 (bytes 1..4) floors to byte 1 — the start of 雪 — which is
            // character 1, i.e. column 2.  The point is that it is a *boundary*,
            // not that the column is 1.
            ("a雪b\n", 2, 2),
        ] {
            let diag = Diagnostic::error("x").with_source(src).with_span(Span::new(
                FileId::DUMMY,
                inside,
                inside + 1,
            ));
            let v: serde_json::Value =
                serde_json::from_str(&json_of(&diag, false)).expect("valid JSON");
            // The floor puts every one of these on the first character of line 1.
            assert_eq!(v["span"]["line"], 1, "{src:?} start {inside}");
            assert_eq!(v["span"]["column"], want_col, "{src:?} start {inside}");
        }
    }

    /// `needs_index` gates every coordinate on the diagnostic having a first
    /// span.  A diagnostic with labels but *no* primary span must emit the labels
    /// without coordinates — never line/column against a source its spans were
    /// not resolved against.
    #[test]
    fn labels_without_a_primary_span_carry_no_coordinates() {
        let diag = Diagnostic::error("x")
            .with_source("a\nb\n")
            .with_label(Span::new(FileId::DUMMY, 2, 3), "here");
        let v: serde_json::Value =
            serde_json::from_str(&json_of(&diag, false)).expect("valid JSON");
        let label = &v["labels"][0];
        assert_eq!(label["message"], "here");
        for absent in ["line", "column", "location"] {
            assert!(
                label.get(absent).is_none(),
                "{absent} must be absent without a primary span: {label}"
            );
        }
    }

    /// Adjacency matrix for the SWAR escape masks.
    ///
    /// `hasless`/`has_zero` propagate borrows across byte lanes, so a whole-chunk
    /// mask flags *false positives* above a true hit: a space right after a
    /// control byte, `'#'` right after `'"'`, `']'` right after `'\'`.  The old
    /// per-chunk "lowest lane wins" reading was safe; the drain-all-lanes
    /// iterators are only safe with masks that are exact **per lane**, so any
    /// regression here turns ordinary indented text into a `C0_ESCAPED` index
    /// panic (and, through the panic handler, a lost diagnostic).
    #[test]
    fn escape_masks_are_exact_per_lane() {
        let needles = [b'"', b'\\', b'\n', b'\t', 0x00u8, 0x01, 0x1f];
        let neighbours = [0x20u8, b'#', b']', b'a', 0x7f, 0x80, 0xff, b'\n', b'\t'];
        for n in needles {
            for m in neighbours {
                for len in 1..=8usize {
                    let mut bytes = vec![0x41u8; len];
                    bytes[len - 1] = n;
                    let mut s = String::from_utf8(bytes).expect("ASCII");
                    if len < 8 {
                        s.push(m as char);
                    }
                    let mut out = String::new();
                    JsonRenderer::push_json_str(&mut out, &s);
                    assert_eq!(
                        out,
                        serde_json::to_string(&s).expect("serde escapes"),
                        "needle {n:#04x} neighbour {m:#04x} len {len}: {s:?}"
                    );
                }
            }
        }
    }

    /// End-to-end differential against `serde_json`'s escaper, driven through the
    /// **production** entry point rather than the test-only predicate view:
    /// `next_json_escape_byte` mirrors the dispatch policy, so a change to
    /// `SHORT_FUSED_MAX` or to the long path's stream merge could drift while that
    /// test kept passing.  Also pins lengths 63/64/65 (the short/long boundary)
    /// with an escape as the final byte, and a control-dense 4 KiB string, where
    /// the old restart-per-hit scanner did ~8× redundant work.
    /// Every C0 byte must escape to `\u00XX` with **its own index**, and
    /// [`escaped_extra`] must predict the added bytes exactly.
    ///
    /// The table is const-asserted entry by entry now, but the *writer* and the
    /// *reservation* are two code paths: a drift in either still yields a legal escape of
    /// the wrong byte, or an under-count that silently grows a second allocation.  The
    /// mixed corpus runs past the old `SmallVec<[usize; 8]>` inline buffer — the case that
    /// used to heap-allocate in the middle of the "one allocation" claim.
    #[test]
    fn every_c0_byte_escapes_to_its_own_index() {
        for b in 0u8..0x20 {
            let input = String::from_utf8(vec![b]).expect("C0 bytes are single-byte UTF-8");
            let mut out = String::new();
            push_json_escaped_str(&mut out, &input);
            // `serde_json` is *not* a spelling oracle here: it emits `\b`/`\f` where this
            // writer uses the (equally legal) `\u00XX`, so comparing bytes would pin one
            // arbitrary choice.  What matters is that the escape decodes back to the same
            // byte — checked by a round-trip through the reference parser — and that its
            // *shape* matches the writer's three weight classes, which is what
            // `escaped_extra` reserves for.
            let decoded: String = serde_json::from_str(&format!("\"{out}\"")).expect("valid JSON");
            assert_eq!(decoded, input, "byte {b:#04x} did not round-trip");
            let named = matches!(b, b'\n' | b'\r' | b'\t');
            assert_eq!(
                out.len(),
                if named { 2 } else { 6 },
                "byte {b:#04x} has the wrong escape width: {out:?}"
            );
            assert_eq!(
                escaped_extra(input.as_bytes()),
                out.len() - input.len(),
                "escaped_extra must match the bytes written for {b:#04x}"
            );
        }
        // Mixed corpus past the old `SmallVec<[usize; 8]>` inline buffer, so all three
        // weight classes (structural, named, wide C0) are present in one chunk stream.
        let mixed = "\"a\u{1}b\\c\u{2}\u{3}\u{4}\u{5}\t\n\r\u{6}\u{7}\"";
        let mut out = String::new();
        push_json_escaped_str(&mut out, mixed);
        assert_eq!(
            escaped_extra(mixed.as_bytes()),
            out.len() - mixed.len(),
            "escaped_extra must stay exact past the ninth escape"
        );

        // Mechanical one-allocation pin: `push_json_escaped_str` reserves
        // `len + escaped_extra`, so a buffer pre-sized to exactly that must never
        // reallocate.  This is the property the old collect-into-`SmallVec` shape had only
        // on strings with fewer than nine escapes.  (`push_json_str` adds the two quotes
        // on top with its own reserve; that outer layer is unchanged and is not what this
        // pins.)
        let mut pre = String::with_capacity(mixed.len() + escaped_extra(mixed.as_bytes()));
        let capacity = pre.capacity();
        push_json_escaped_str(&mut pre, mixed);
        assert_eq!(
            pre.capacity(),
            capacity,
            "an exact reserve must make the append allocation-free"
        );
    }

    #[test]
    fn push_json_str_matches_serde_json_across_the_dispatch_boundary() {
        fn check(s: &str, ctx: &str) {
            let mut out = String::new();
            JsonRenderer::push_json_str(&mut out, s);
            let want = serde_json::to_string(s).expect("serde_json escapes any &str");
            assert_eq!(out, want, "{ctx} (len {}): {s:?}", s.len());
        }

        // Boundary band: an escape as the last byte of a 63/64/65/66-byte value,
        // plus the same with a control and with a quote.
        for len in [0usize, 1, 7, 8, 9, 62, 63, 64, 65, 66, 127, 128, 129] {
            for esc in ['\n', '"', '\\', '\u{7}'] {
                let mut s = "a".repeat(len.saturating_sub(1));
                if len > 0 {
                    s.push(esc);
                }
                check(&s, "boundary");
            }
        }
        // Control-dense long value: every byte an escape, past the fused threshold.
        check(&"\n".repeat(4096), "dense-newline");
        // Structural-dense long value (exercises the memchr2 stream).
        check(&"\"".repeat(4096), "dense-quote");
        // Mixed, including a multi-byte codepoint adjacent to escapes, so the
        // byte-level scanner and the char-boundary slice agree.
        check(&"é\n雪\"\\\u{1}".repeat(200), "mixed");
    }

    #[test]
    fn needs_json_escape_matches_scalar() {
        fn scalar(s: &str) -> bool {
            s.bytes().any(|b| b < 0x20 || b == b'"' || b == b'\\')
        }
        // `needs_json_escape` and `next_json_escape_byte` are two views of the
        // same SWAR predicate — pin both against the scalar reference and, when
        // a candidate exists, confirm the position actually names one.
        // The first-hit position is what the bulk copy in
        // `push_json_escaped_str` slices on; asserting only that *some* escape
        // byte was found let a regression that returns a later hit pass while
        // silently copying an escape verbatim.
        fn scalar_first(s: &str) -> Option<usize> {
            s.bytes().position(|b| b < 0x20 || b == b'"' || b == b'\\')
        }
        fn check(s: &str, ctx: &str) {
            let want = scalar(s);
            assert_eq!(needs_json_escape(s), want, "{ctx}: needs_json_escape");
            let pos = next_json_escape_byte(s.as_bytes());
            assert_eq!(pos.is_some(), want, "{ctx}: next_json_escape_byte");
            assert_eq!(
                pos,
                scalar_first(s),
                "{ctx}: next_json_escape_byte must name the first escape, not any"
            );
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
            let s = simdutf8::basic::from_utf8(&buf).unwrap();
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

    /// A render panic must not break the batch.  `Renderer::render_internal_error`
    /// exists for exactly this: the flat-text marker the other formats use would
    /// splice prose into the array and make the *whole* batch unparseable, so
    /// JSON overrides it with a real object.
    #[test]
    fn internal_error_placeholder_stays_valid_json() {
        for pretty in [true, false] {
            let r = JsonRenderer::new(pretty);
            let mut body = String::new();
            r.render_internal_error(&mut body);
            let value: serde_json::Value =
                serde_json::from_str(&body).expect("the placeholder must parse as JSON");
            assert_eq!(
                value["level"], "error",
                "the placeholder must name its severity: {body:?}"
            );
            assert!(
                value["message"]
                    .as_str()
                    .is_some_and(|m| m.contains("panicked")),
                "the placeholder must name the failure: {body:?}"
            );
            // Schema uniformity: a consumer that reads `id` unconditionally must
            // not meet an element without one — and the placeholder is exactly
            // the element it most needs to identify.
            let id = value["id"].as_str().expect("placeholder must carry an id");
            assert!(
                id.starts_with("0x") && id.len() > 2,
                "placeholder id must be hex-shaped like a real one: {id:?}"
            );
            assert_eq!(value["id"], format!("0x{INTERNAL_ERROR_ID:016x}"));

            // …and inside a batch the array still parses end to end.
            let mut sink = crate::diagnostics::sink::StringSink::new();
            r.begin(&mut sink).unwrap();
            body.clear();
            r.render_internal_error(&mut body);
            sink.write_str(&body).unwrap();
            r.end(&mut sink).unwrap();
            let batch = sink.into_string();
            serde_json::from_str::<serde_json::Value>(&batch).unwrap_or_else(|e| {
                panic!("a panicking diagnostic must not corrupt the batch: {e}\n{batch:?}")
            });
        }
    }
}
