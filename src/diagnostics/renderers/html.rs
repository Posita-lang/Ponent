//! HTML renderer — streams an HTML document instead of buffering it.

use crate::ast::FileId;
use crate::ast::Span;
use crate::diagnostics::emitter::stylesheet::dark_css;

use crate::diagnostics::Diagnostic;
use crate::diagnostics::DiagnosticLevel;
use crate::diagnostics::LineIndex;
use crate::diagnostics::glyph::{
    LabelWindow, SectionScratch, byte_offset_to_display, compute_line_annotation_ranges,
    expand_tabs_from_col, render_line_merged_with,
};
use crate::diagnostics::label::Label;
use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::renderers::fmt;
use crate::diagnostics::sink::DiagnosticSink;
use crate::smallvec::{SmallVec, smallvec};
use litemap::LiteMap;
use std::borrow::Cow;
use std::fmt::Write as _;
use std::io;

struct EscapeEntry {
    value: std::sync::Arc<str>,
    referenced: bool,
}

/// HTML renderer — generates diagnostic output as HTML.
/// Inspired by Austral's Reporter.ml which produces interactive call trees.
pub struct HtmlRenderer {
    /// Explicit `with_context_lines` override.  When `None` the context
    /// count comes from the batch `RenderCtx`'s `DiagOpts` (emitter path) or
    /// from `fallback_context_lines` (single-shot path).
    context_lines_override: Option<usize>,
    /// `DiagOpts::context_lines` snapshot for the no-`RenderCtx` path.
    fallback_context_lines: usize,
    /// `DiagOpts::tab_width` snapshot for the no-`RenderCtx` path.
    fallback_tab_width: usize,
}

impl HtmlRenderer {
    pub fn new() -> Self {
        let opts = crate::diagnostics::DiagOpts::default();
        HtmlRenderer {
            context_lines_override: None,
            fallback_context_lines: opts.context_lines,
            fallback_tab_width: opts.tab_width.max(1),
        }
    }

    /// Build a renderer whose `context_lines` and `tab_width` come from
    /// `opts`, mirroring [`GlyphRenderer::with_opts`] — the CLI/config
    /// `--context-lines` / `--tab-width` flags reach HTML output through
    /// this constructor.  A batch `RenderCtx`'s `DiagOpts` still takes
    /// precedence (see [`Self::with_context_lines`] for an explicit
    /// override that beats both).
    pub fn with_opts(opts: &crate::diagnostics::DiagOpts) -> Self {
        HtmlRenderer {
            context_lines_override: None,
            fallback_context_lines: opts.context_lines,
            fallback_tab_width: opts.tab_width.max(1),
        }
    }

    /// Override the number of source lines of context shown around a span,
    /// mirroring [`GlyphRenderer::with_context_lines`](crate::diagnostics::GlyphRenderer::with_context_lines).
    pub fn with_context_lines(mut self, n: usize) -> Self {
        self.context_lines_override = Some(n);
        self
    }
}

impl Default for HtmlRenderer {
    fn default() -> Self {
        Self::new()
    }
}

fn localized_slug(slug: &str) -> String {
    let mut s = String::new();
    crate::diagnostics::i18n::render_into(slug, &[], &mut s);
    s
}

fn summary_inner(out: &mut String, errors: usize, warnings: usize, files: Option<usize>) {
    crate::diagnostics::i18n::render_into(
        "html-summary",
        &[
            ("errors", Cow::Owned(errors.to_string())),
            ("warnings", Cow::Owned(warnings.to_string())),
        ],
        out,
    );
    if let Some(files) = files {
        out.push(' ');
        crate::diagnostics::i18n::render_into(
            "html-summary-files",
            &[("files", Cow::Owned(files.to_string()))],
            out,
        );
    }
}

impl Renderer for HtmlRenderer {
    fn begin(&self, out: &mut dyn DiagnosticSink) -> io::Result<()> {
        let template = "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n\
             <title>__PONENT_TITLE__</title>\n\
             <style>\n";
        let dark = dark_css();
        let rest = "\n\
             body { font-family: 'DejaVu Sans Mono', monospace; background: var(--bg); color: var(--fg); padding: 20px; }\n\
             #theme-toggle { position: fixed; top: 12px; right: 12px; z-index: 100; }\n\
             #theme-toggle button { background: var(--toggle-bg); color: var(--fg); border: 1px solid var(--border); border-radius: 6px; padding: 6px 14px; cursor: pointer; font-family: inherit; font-size: 0.85em; }\n\
             #theme-toggle button:hover { opacity: 0.8; }\n\
             .diag { margin: 12px 0; padding: 12px; border-radius: 6px; }\n\
             .diag.error { border-left: 4px solid var(--error-border); background: var(--error-bg); }\n\
             .diag.warning { border-left: 4px solid var(--warning-border); background: var(--warning-bg); }\n\
             .diag.help { border-left: 4px solid var(--help-border); background: var(--help-bg); }\n\
             .diag.code { color: var(--accent); font-weight: bold; }\n\
             .diag.message { color: var(--fg); }\n\
             .diag .span { color: var(--muted); }\n\
             .diag .suggestion { color: var(--suggestion); }\n\
             .diag .suggestion-diff { margin: 4px 0 4px 20px; white-space: pre-wrap; font-size: 0.9em; }\n\
             .diag .suggestion-diff .diff-del { display: block; color: var(--error-border); }\n\
             .diag .suggestion-diff .diff-add { display: block; color: var(--help-border); }\n\
             .diag .help-text { color: var(--help-text); }\n\
             .diag .highlight { color: var(--accent); font-weight: bold; }\n\
             .chain { margin-left: 20px; color: var(--muted); font-size: 0.9em; }\n\
             .chain .entry { margin: 2px 0; }\n\
             .chain .entry .label { color: var(--accent); }\n\
             .related { margin: 4px 0 4px 20px; color: var(--muted); font-size: 0.9em; list-style: none; }\n\
             .related .related-entry { margin: 2px 0; }\n\
             .related .related-entry .code { color: var(--accent); font-weight: bold; }\n\
             .summary { margin-top: 16px; padding: 8px; border-top: 1px solid var(--border); color: var(--muted); }\n\
             details { margin: 4px 0; }\n\
             summary { cursor: pointer; color: var(--accent); }\n\
             .source-line { white-space: pre; overflow-x: auto; font-family: monospace; color: var(--source-fg); }\n\
             .source-line .line-num { color: var(--muted); user-select: none; }\n\
             .source-line .highlight { background: var(--highlight-bg); border-radius: 2px; }\n\
             .source-line .underline { color: var(--underline); }\n\
             .source-line .label { color: var(--accent); white-space: pre-wrap; }\n\
             .explain { background: var(--diag-bg); padding: 12px; border-radius: 6px; margin-top: 8px; }\n\
             .explain h4 { color: var(--accent); margin: 0 0 8px 0; }\n\
             .explain pre { white-space: pre-wrap; color: var(--explain-fg); }\n\
             .diag>summary { margin: 0 0 8px 0; }\n\
             .source-line .line-num { cursor: pointer; }\n\
             </style>\n\
             <script>\n\
             (function(){\n\
               // Runs synchronously in <head>: it only touches\n\
               // `document.documentElement`, which already exists, so the\n\
               // light class is set before the body paints (no theme flash).\n\
               var t = localStorage.getItem('ponent-theme');\n\
               if (t === 'light' || (!t && matchMedia('(prefers-color-scheme: light)').matches)) {\n\
                 document.documentElement.classList.add('light');\n\
               }\n\
             })();\n\
             </script>\n\
             </head>\n<body>\n\
             <div id='theme-toggle'><button id='theme-toggle-btn'>__PONENT_TOGGLE__</button></div>\n\
             <script>\n\
             (function(){\n\
               function toggle() {\n\
                 var light = document.documentElement.classList.toggle('light');\n\
                 localStorage.setItem('ponent-theme', light ? 'light' : 'dark');\n\
               }\n\
               document.getElementById('theme-toggle-btn').addEventListener('click', toggle);\n\
               document.addEventListener('click', function(e) {\n\
                 var n = e.target.closest('.line-num');\n\
                 if (n && n.dataset.location) {\n\
                   navigator.clipboard.writeText(n.dataset.location).catch(function(){});\n\
                 }\n\
               });\n\
             })();\n\
             </script>\n\
             <h1>__PONENT_H1__</h1>\n";
        let mut page = String::with_capacity(template.len() + dark.len() + rest.len());
        page.push_str(template);
        page.push_str(dark);
        page.push_str(rest);
        // `replacen(.., 1)` rather than `replace`: a translation that happened
        // to contain another marker would otherwise be substituted again by the
        // next pass, and every substitution is escaped because the page is HTML
        // and Fluent text is external input (`<` or `&` in a translation must not
        // become markup).  Three full-string scans remain; the page is built once
        // per batch, so a single-pass writer would be a larger rewrite for no
        // measurable gain.
        let localized = page
            .replacen(
                "__PONENT_TITLE__",
                &self.escape(&localized_slug("html-doc-title")),
                1,
            )
            .replacen("__PONENT_H1__", &self.escape(&localized_slug("html-h1")), 1)
            .replacen(
                "__PONENT_TOGGLE__",
                &self.escape(&localized_slug("html-toggle-theme")),
                1,
            );
        out.write_str(&localized)
    }

    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
        self.diagnostic_to_html_impl(diag, Some(cx), out);
    }

    fn render_summary(&self, errors: usize, warnings: usize) -> String {
        let mut s = String::from("<div class='summary' role='status'>");
        summary_inner(&mut s, errors, warnings, None);
        s.push_str("</div>\n");
        s
    }

    fn render_summary_full(&self, errors: usize, warnings: usize, files: usize) -> String {
        let mut s = String::from("<div class='summary' role='status'>");
        summary_inner(&mut s, errors, warnings, (files > 1).then_some(files));
        s.push_str("</div>\n");
        s
    }

    fn end(&self, out: &mut dyn DiagnosticSink) -> io::Result<()> {
        out.write_str("</body>\n</html>")
    }

    /// HTML's trailer is markup, not the terminal `= note:` line.
    ///
    /// Overrides **`_into`**, which is the single hook: the default
    /// `render_suppressed_note` already routes through it, so overriding the `String`-returning
    /// form (as this did) left the `_into` path — the one the emitter calls to reuse its buffer
    /// — falling back to the terminal text.  Two overridable hooks for one behaviour is exactly
    /// the drift this file's contract discipline forbids.
    fn render_suppressed_note_into(&self, count: usize, out: &mut String) {
        out.push_str("<div class='summary' role='status'>");
        crate::diagnostics::i18n::render_into(
            "suppressed-note",
            &[("count", Cow::Owned(count.to_string()))],
            out,
        );
        out.push_str("</div>\n");
    }
}

impl HtmlRenderer {
    /// Single-shot entry point (tests): builds its own line index.
    #[cfg(test)]
    fn diagnostic_to_html(&self, diag: &Diagnostic<'_>) -> String {
        let mut out = String::new();
        self.diagnostic_to_html_impl(diag, None, &mut out);
        out
    }

    /// Shared body.  `cx` supplies the per-batch line index when rendering
    /// through the emitter, so the file is not rescanned per diagnostic.
    fn diagnostic_to_html_impl(
        &self,
        diag: &Diagnostic<'_>,
        cx: Option<&RenderCtx<'_>>,
        out: &mut String,
    ) {
        // Output is context-window-sized, not file-sized; cap the hint so a
        // large source does not over-reserve.
        let hint = diag
            .source
            .as_ref()
            .map_or(512, |s| (s.len() / 4).clamp(512, 16 * 1024));
        out.reserve(hint);
        let html = out;
        let level_class = if matches!(diag.level, DiagnosticLevel::Fatal) {
            // [3]6: the stylesheet defines `.diag.error`, not `.fatal`;
            // fatal diagnostics must not lose their severity styling.
            "error"
        } else {
            diag.level.as_str()
        };
        // `with_code_str` explicitly supports untrusted input (unknown codes
        // are stored verbatim), so the code string must be escaped like any
        // other external text.
        // Escape the code string once and reuse for both the body code span
        // and the summary `[code]` — `escape` is a full scan, so two calls
        // per diagnostic is pure waste.
        let code_esc: Option<Cow<'_, str>> = diag.code.as_ref().map(|c| {
            if c.is_known() {
                Cow::Borrowed(c.code())
            } else {
                self.escape(c.code())
            }
        });

        html.push_str("<details class='diag ");
        html.push_str(level_class);
        html.push_str("' open role='alert'>\n");

        // Collapsible summary line — `error[E030]: boom` (rustc form); the
        // diagnostic starts expanded (`open`) so streaming output keeps
        // the same visible content, users just gain the collapse control.
        // `diag.message` is escaped once and reused for both the summary and
        // the body — `escape` is a full scan, so two calls per diagnostic is
        // pure waste.
        let level_label = diag.level.label();
        // Route through the accessors so a kind-built diagnostic that was
        // never humanized fails the same debug assert as the JSON renderer
        // instead of silently rendering an empty message.
        // Bound, not inlined: `display_message` may return an owned fallback, and the
        // escaped text outlives the temporary the inline form would drop at the end
        // of the statement (E0716).
        let message = diag.display_message();
        let msg_esc = self.escape(&message);
        html.push_str("<summary><span class='code'>");
        html.push_str(level_label);
        html.push_str("</span>: <span class='message'>");
        html.push_str(&msg_esc);
        html.push_str("</span></summary>\n");

        // Main message — `code_esc` carries its own `<span class='code'>` when
        // present; pushed in three pieces to avoid a per-diagnostic `format!`.
        html.push_str("<div>");
        if let Some(esc) = &code_esc {
            html.push_str("<span class='code'>");
            html.push_str(esc);
            html.push_str("</span> ");
        }
        html.push_str("<span class='message'>");
        html.push_str(&msg_esc);
        html.push_str("</span></div>\n");

        // Span location.  When the diagnostic carries a source buffer we can
        // resolve the primary span to `file:line:col` (1-based) instead of the
        // opaque byte range; build the line index up front so both the span
        // header and the source section below share it.
        let source_ctx = diag.source.as_ref();
        let mut idx_opt: Option<LineIndex<'_>> = None;
        if let Some(source) = source_ctx {
            // Only build it when something below actually reads line/column:
            // the span header, a label, a related error, a same-file chain entry,
            // or an in-place suggestion diff.  A source-attached diagnostic with
            // none of those (summary-level errors do occur) previously paid an
            // O(file) scan on an LRU miss plus a full `LineIndex::new` when
            // single-shot.
            //
            // This disjunction IS the pin, as in `human.rs`: narrowing it
            // silently drops coordinates rather than failing loudly, so keep it
            // in sync with the `idx_opt` reads in this function.
            let needs_index = diag.spans().first().is_some()
                || !diag.labels().is_empty()
                || !diag.related_errors().is_empty()
                || diag
                    .call_chain
                    .as_ref()
                    .is_some_and(|c| c.entries().iter().any(|e| e.span.is_some()))
                || diag
                    .suggestions()
                    .iter()
                    .any(|s| s.span.is_some() && s.replacement.is_some());
            if needs_index {
                idx_opt = Some(match cx {
                    Some(cx) => cx.line_index(source),
                    None => LineIndex::new(source),
                });
            }
        }

        if let Some(span) = diag.spans.first() {
            html.push_str("<div class='span'>at ");
            match (&idx_opt, diag.source.as_ref()) {
                (Some(idx), Some(source)) => {
                    // Stream the location: the old `format!` built one
                    // `String` and `escape` built a second, per diagnostic.
                    let file = diag.file_name();
                    if !file.is_empty() {
                        // [1]2: the visible `at file:line:col` must be
                        // sanitized like every other renderer; a raw
                        // control/bidi filename could forge terminal state in
                        // the HTML body even though the attribute is clean.
                        html.push_str(&self.escape_filename(file));
                        html.push(':');
                    }
                    let (line, col) = idx.linecol_display(span.start_as_usize().min(source.len()));
                    fmt::push_uint(html, (line + 1) as u64);
                    html.push(':');
                    fmt::push_uint(html, (col + 1) as u64);
                }
                _ => {
                    let opaque = span.to_string();
                    html.push_str(&self.escape(&opaque));
                }
            }
            html.push_str("</div>\n");
        }

        // Source context — render directly from Diagnostic struct fields
        if let Some(source) = source_ctx {
            let idx = idx_opt
                .as_ref()
                .expect("idx_opt is Some whenever a source is present");
            // Multi-file guard: the source buffer belongs to the first
            // primary span's file (emitter contract), so only labels from
            // that file can be located in it.  Labels from other files are
            // rendered below as byte-range notes instead of bogus
            // line/column positions.
            let primary_spans = diag.spans.primary_ref();
            let anchor_file = primary_spans.first().map(|sp| sp.file);
            // Multi-file guard (mirrors the glyph renderer): `source` is the
            // anchor file's buffer, so primary spans from other files have
            // meaningless offsets in it — they would be clamped into bogus
            // sections/columns.  Filter them out here; the foreign-source
            // loop below renders them against their own buffers.
            let anchor_primary_spans: SmallVec<[Span; 4]> = primary_spans
                .iter()
                .copied()
                .filter(|&sp| Some(sp.file) == anchor_file)
                .collect();
            // Borrowed view — one pointer per label, no clone.  Single pass:
            // anchor-file labels feed the source section, foreign labels are
            // grouped by file in the same walk (`LiteMap`, matching the
            // glyph renderer) instead of a second O(L) filter pass later.
            // A handful of `FileId` keys makes a sorted flat vector cheaper
            // than Fx's SipHash setup.
            let mut labels: SmallVec<[&Label<'_>; 4]> = SmallVec::new();
            let mut foreign_labels_by_file: LiteMap<FileId, SmallVec<[&Label<'_>; 4]>> =
                LiteMap::new();
            for lbl in diag.labels() {
                if anchor_file.is_none_or(|anchor| lbl.span.file == anchor) {
                    labels.push(lbl);
                } else {
                    foreign_labels_by_file
                        .entry(lbl.span.file)
                        .or_insert_with(SmallVec::new)
                        .push(lbl);
                }
            }
            let pre_start = html.len();
            html.push_str("<pre class='source-line'>");
            if self.render_source_html_into(
                source,
                &anchor_primary_spans,
                &labels,
                idx,
                diag.file_name(),
                cx,
                html,
            ) {
                html.push_str("</pre>\n");
            } else {
                html.truncate(pre_start);
                // No source section could be drawn (no primary span in this
                // buffer, or an empty source) — but the anchor-file labels
                // still own user-visible text.  Emit them as notes so no
                // label message silently disappears.
                for lbl in &labels {
                    if lbl.message.is_empty() {
                        continue;
                    }
                    html.push_str("<span class='label'>= note: ");
                    html.push_str(&self.escape(&lbl.message));
                    html.push_str(" at ");
                    html.push_str(&self.escape(&lbl.span.to_string()));
                    html.push_str("</span>\n");
                }
                // Additional primaries carry no message but *do* carry a
                // location; the source-less path already emits `at <span>` for
                // them, and dropping them here silently lost where they point.
                for sp in anchor_primary_spans.iter().skip(1) {
                    html.push_str("<span class='label'>= note: at ");
                    html.push_str(&self.escape(&sp.to_string()));
                    html.push_str("</span>\n");
                }
            }
            // Labels filtered out of the source section (from other files)
            // stay visible as range notes; their line/col would be
            // meaningless against this buffer.
            let mut foreign_rendered_files: SmallVec<[FileId; 4]> = SmallVec::new();
            // ── Foreign-file source sections ──
            // Cross-file diagnostics render one extra section per
            // registered foreign source that has primary spans or labels
            // in it.
            if let Some(foreign) = diag.foreign_sources.as_deref() {
                // ── Pre-group foreign-file data ───────────────────────
                // Partition spans/labels by file once so each foreign file's
                // section does not re-scan the full `primary_spans` /
                // `diag.labels` lists.  Turns the foreign loop from
                // O(F·(S+L)) into O(S+L+F).  `LiteMap` over a handful of
                // `FileId` keys: O(log n) binary search, no hash-table shell.
                let mut foreign_spans_by_file: LiteMap<FileId, SmallVec<[Span; 2]>> =
                    LiteMap::new();
                for &sp in primary_spans.iter() {
                    if Some(sp.file) != anchor_file {
                        foreign_spans_by_file
                            .entry(sp.file)
                            .or_insert_with(SmallVec::new)
                            .push(sp);
                    }
                }
                for (fid, fname, fsource) in foreign.iter() {
                    if Some(*fid) == anchor_file {
                        continue;
                    }
                    let empty_spans: SmallVec<[Span; 2]> = SmallVec::new();
                    let f_primaries: &[Span] = foreign_spans_by_file
                        .get(fid)
                        .unwrap_or(&empty_spans)
                        .as_slice();
                    let empty_labels: SmallVec<[&Label<'_>; 4]> = SmallVec::new();
                    let f_labels: &[&Label<'_>] = foreign_labels_by_file
                        .get(fid)
                        .unwrap_or(&empty_labels)
                        .as_slice();
                    if f_primaries.is_empty() && f_labels.is_empty() {
                        continue;
                    }
                    let f_idx = match cx {
                        Some(cx) => cx.line_index(fsource),
                        None => crate::diagnostics::LineIndex::new(fsource),
                    };
                    let pre_start = html.len();
                    html.push_str("<pre class='source-line'>");
                    if self.render_source_html_into(
                        fsource,
                        f_primaries,
                        f_labels,
                        &f_idx,
                        fname.as_ref(),
                        cx,
                        html,
                    ) {
                        html.push_str("</pre>\n");
                        // Only a file whose section actually drew suppresses
                        // the fallback note path below; a file whose render
                        // failed (e.g. no primary span and empty source)
                        // must fall through so its labels stay visible.
                        foreign_rendered_files.push(*fid);
                    } else {
                        html.truncate(pre_start);
                    }
                }
            }
            // Sorted once so the fallback loops below are `O(log F)` per label
            // with no linear `contains` over the foreign files (the glyph
            // renderer moved to a map for the same reason).
            foreign_rendered_files.sort_unstable();
            foreign_rendered_files.dedup();
            for lbl in diag
                .labels
                .iter()
                .filter(|l| anchor_file.is_some_and(|anchor| l.span.file != anchor))
            {
                if lbl.message.is_empty() {
                    continue;
                }
                if foreign_rendered_files.binary_search(&lbl.span.file).is_ok() {
                    continue;
                }
                html.push_str("<span class='label'>= note: ");
                html.push_str(&self.escape(&lbl.message));
                html.push_str(" at ");
                html.push_str(&self.escape(&lbl.span.to_string()));
                html.push_str("</span>\n");
            }
            // Primary spans whose foreign file was never registered (or whose
            // section could not draw) must not vanish: surface each as a
            // location-only note, mirroring the glyph renderer's fallback.
            for sp in primary_spans.iter() {
                if Some(sp.file) == anchor_file
                    || foreign_rendered_files.binary_search(&sp.file).is_ok()
                {
                    continue;
                }
                // Per-span file name: a foreign span must keep its own name,
                // never the anchor's.
                let loc = match diag.file_name_for(sp.file) {
                    Some(f) if !f.is_empty() && f != "<input>" => format!("{f} {sp}"),
                    _ => sp.to_string(),
                };
                html.push_str("<span class='label'>at ");
                html.push_str(&self.escape(&loc));
                html.push_str("</span>\n");
            }
        } else {
            // No source attached: labels and additional primary spans would
            // otherwise vanish (only the first span's location header above
            // survives).  Mirror the glyph renderer's no-source fallback.
            for sp in diag.spans.primary_ref().iter().skip(1) {
                let loc = match diag.file_name_for(sp.file) {
                    Some(f) if !f.is_empty() && f != "<input>" => format!("{f} {sp}"),
                    _ => sp.to_string(),
                };
                html.push_str("<div class='span'>at ");
                html.push_str(&self.escape(&loc));
                html.push_str("</div>\n");
            }
            for lbl in diag.labels() {
                if lbl.message.is_empty() {
                    continue;
                }
                let loc = match diag.file_name_for(lbl.span.file) {
                    Some(f) if !f.is_empty() && f != "<input>" => format!("{f} {}", lbl.span),
                    _ => lbl.span.to_string(),
                };
                html.push_str("<span class='label'>= note: ");
                html.push_str(&self.escape(&lbl.message));
                html.push_str(" at ");
                html.push_str(&self.escape(&loc));
                html.push_str("</span>\n");
            }
        }

        // Related errors (Zig-style numbered context, matching the human
        // renderer's `1.` / `2.` header numbering).
        if !diag.related_errors.is_empty() {
            html.push_str("<ul class='related'>\n");
            for rel in diag.related_errors() {
                html.push_str("<li class='related-entry'>");
                if let Some(c) = &rel.code {
                    html.push_str("<span class='code'>");
                    html.push_str(&self.escape(c.code()));
                    html.push_str("</span> ");
                }
                html.push_str("<span class='message'>");
                html.push_str(&self.escape(&rel.message));
                html.push_str("</span>");
                if let Some(sp) = &rel.span {
                    html.push_str(" <span class='span'>at ");
                    html.push_str(&self.escape(&sp.to_string()));
                    html.push_str("</span>");
                }
                html.push_str("</li>\n");
            }
            html.push_str("</ul>\n");
        }

        // Child diagnostics (notes/help) are part of the diagnostic model;
        // machine and HTML renderers must not silently drop them.
        for child in diag.children() {
            html.push_str("<div class='label'>");
            html.push_str(child.level.label());
            html.push_str(": ");
            if let Some(styled) = &child.styled_message
                && !styled.0.is_empty()
            {
                for part in &styled.0 {
                    match part.style {
                        crate::diagnostics::StringPartStyle::Highlighted => {
                            html.push_str("<span class='highlight'>");
                            html.push_str(&self.escape(part.content.as_ref()));
                            html.push_str("</span>");
                        }
                        crate::diagnostics::StringPartStyle::Normal => {
                            html.push_str(&self.escape(part.content.as_ref()));
                        }
                    }
                }
            } else {
                html.push_str(&self.escape(child.plain_message().as_ref()));
            }
            if let Some(span) = child.span {
                html.push_str(" at ");
                html.push_str(&self.escape(&span.to_string()));
            }
            html.push_str("</div>\n");
        }

        // Suggestions.  Mirror the human renderer's visibility policy: a
        // `HideCodeAlways` / `CompletelyHidden` suggestion is internal and must
        // not become user-facing HTML.  `Suggestion`'s `Display` also carries
        // the applicability tag (`(maybe incorrect)`, …), so a consumer sees
        // the same confidence signal as the terminal output.
        for suggestion in diag.suggestions() {
            if matches!(
                suggestion.style,
                crate::diagnostics::SuggestionStyle::HideCodeAlways
                    | crate::diagnostics::SuggestionStyle::CompletelyHidden
            ) {
                continue;
            }
            html.push_str("<div class='suggestion'>suggestion: ");
            html.push_str(&self.escape(&suggestion.to_string()));
            html.push_str("</div>\n");

            let (Some(idx), Some(span), Some(replacement)) = (
                idx_opt.as_ref(),
                suggestion.span,
                suggestion.replacement.as_deref(),
            ) else {
                continue;
            };
            let anchor_file = diag.spans().first().map(|s| s.file);
            if anchor_file != Some(span.file)
                || replacement.contains('\n')
                || replacement.contains('\r')
            {
                continue;
            }
            // Anchor source when there is one (this path renders in-place edits
            // against it); otherwise the span is used as-is and `to_linecol`'s
            // own bounds apply.
            let span = match diag.source() {
                Some(src) => Self::clamp_span_to_source(span, src.len()),
                None => span,
            };
            let start_pos = idx.to_linecol(span.start_as_usize());
            let end_pos = idx.to_linecol(span.end_as_usize());
            let (line, col) = (start_pos.line, start_pos.col);
            let (end_line, end_col) = (end_pos.line, end_pos.col);
            if line != end_line {
                continue;
            }
            let old_line = idx.line(line);
            if !old_line.is_char_boundary(col) || !old_line.is_char_boundary(end_col) {
                continue;
            }
            let new_line = format!(
                "{}{}{}",
                &old_line[..col],
                replacement,
                &old_line[end_col..]
            );
            let tab_width = cx.map_or(self.fallback_tab_width, |c| c.opts.tab_width.max(1));
            // HTML's diff block has no line-number/`│` gutter: the `- `/`+ `
            // payload starts at the block's left edge, so tab stops advance
            // from column 0 (the CSS margin is a pixel offset, not a cell).
            let diff_text_col = 0;
            let old_display = expand_tabs_from_col(old_line, tab_width, diff_text_col);
            let new_display = expand_tabs_from_col(&new_line, tab_width, diff_text_col);
            html.push_str("<div class='suggestion-diff'><span class='diff-del'>- ");
            html.push_str(&self.escape(old_display.as_ref()));
            html.push_str("</span>\n<span class='diff-add'>+ ");
            html.push_str(&self.escape(new_display.as_ref()));
            html.push_str("</span></div>\n");
        }

        // Help
        if let Some(help) = diag.help() {
            html.push_str("<div class='help-text'>help: ");
            html.push_str(&self.escape(help));
            html.push_str("</div>\n");
        }

        // Call chain (Zig-style) — entries carry kind + user message, plus
        // file:line:col when a source is available and the entry has a span
        // (mirrors the JSON and terminal renderers).
        if let Some(ref chain) = diag.call_chain
            && !chain.is_empty()
        {
            let file_name = diag.file_name();
            // Coordinate resolution is only valid against the anchor file's
            // line index; a foreign-file entry would otherwise print a
            // plausible but wrong `line:col` (the human and JSON renderers
            // already guard this).
            let anchor_file = diag.spans().first().map(|s| s.file);
            let idx = idx_opt.as_ref();
            html.push_str("<div class='chain'>referenced by:\n");
            for entry in chain.entries() {
                html.push_str("<div class='entry'><span class='label'>");
                html.push_str(&self.escape(&entry.message));
                html.push_str("</span>: ");
                html.push_str(&entry.kind.as_str());
                if let (Some(idx), Some(span)) = (idx, entry.span)
                    && anchor_file == Some(span.file)
                {
                    let limit = diag.source().map_or(span.start_as_usize(), |s| s.len());
                    let (line, col) = idx.linecol_display(span.start_as_usize().min(limit));
                    html.push_str(" at ");
                    if !file_name.is_empty() {
                        html.push_str(&self.escape_filename(file_name));
                        html.push(':');
                    }
                    fmt::push_uint(html, (line + 1) as u64);
                    html.push(':');
                    fmt::push_uint(html, (col + 1) as u64);
                } else if let Some(span) = entry.span
                    && let Some(name) = diag.file_name_for(span.file)
                {
                    // No index for a foreign file: name it without inventing a
                    // line/column rather than silently dropping the location.
                    html.push_str(" at ");
                    html.push_str(&self.escape_filename(name));
                }
                html.push_str("</div>\n");
            }
            html.push_str("</div>\n");
        }

        // Explain section (if error code present)
        if let Some(ref code) = diag.code {
            let explain = code.explain();
            if !explain.is_empty() {
                html.push_str("<details class='explain'>\n<summary>Explain ");
                // `code_esc` (escaped once at the top of this function) is
                // still in scope and borrows `diag.code` — reuse it instead
                // of paying a second full escape scan per diagnostic.
                if let Some(esc) = &code_esc {
                    html.push_str(esc);
                }
                html.push_str("</summary>\n<pre>");
                html.push_str(&self.escape(explain));
                html.push_str("</pre>\n</details>\n");
            }
        }

        html.push_str("</details>\n");
    }

    /// Render the source context as HTML, directly from the Diagnostic's
    /// source text, spans, and labels — without going through GlyphRenderer's
    /// terminal-formatted output.
    /// Clamp a span's byte offsets into `source_len`.
    ///
    /// The header of the rendered section has always done this, because a
    /// malformed or foreign span can carry an offset past the buffer; the
    /// `to_linecol`/`span_lines` callers below did not, so the same input that
    /// the header tolerated could panic (or silently slice a wrong window) on the
    /// underline path.  Every index query now goes through this.
    fn clamp_span_to_source(sp: Span, source_len: usize) -> Span {
        let len = u32::try_from(source_len).unwrap_or(u32::MAX);
        Span::new(sp.file, sp.start.min(len), sp.end.min(len))
    }

    fn render_source_html_into<'src>(
        &self,
        source: &str,
        primary_spans: &[Span],
        labels: &[&Label<'src>],
        idx: &LineIndex<'_>,
        file_name: &str,
        cx: Option<&RenderCtx<'_>>,
        out: &mut String,
    ) -> bool {
        let Some(&span) = primary_spans.first() else {
            return false;
        };
        // Empty source has no rows to draw. `text_line_count()` reports 1
        // for `""` (str::lines-style), so guard on the source itself rather
        // than materializing every line just to test for emptiness.
        if source.is_empty() {
            return false;
        }

        // The caller supplies the line index (built once per source, not
        // per diagnostic); every conversion below is O(log n).
        let starts = idx.starts();

        // Character column, not `to_linecol`'s byte column: the start
        // position is printed as `file:line:col` (the `data-location`
        // click-to-copy target), which editors count in characters.
        // Clamp: a malformed or foreign span can carry an offset past the
        // registered source, and `linecol_display` would index out of range
        // (the JSON renderer already guards this).
        let (start_line, start_col) = idx.linecol_display(span.start_as_usize().min(source.len()));

        // Option resolution: an explicit `with_context_lines` beats the
        // batch `DiagOpts`, which beats this renderer's constructor snapshot
        // (the single-shot `diagnostic_to_html` path has no `RenderCtx`).
        let context_lines = self.context_lines_override.unwrap_or(match cx {
            Some(c) => c.opts.context_lines,
            None => self.fallback_context_lines,
        });
        let tab_width = match cx {
            Some(c) => c.opts.tab_width.max(1),
            None => self.fallback_tab_width,
        };
        let text_last_line = idx.text_line_count().saturating_sub(1);

        // One `to_linecol` pair per primary span, computed once: the
        // last-line clamp loop and the section-merge loop below both need
        // exactly these (start_line, end_line) values, so resolving them
        // inline there paid `3·S` binary searches where `2·S` suffice.
        let primary_lines: SmallVec<[(usize, usize); 4]> = primary_spans
            .iter()
            .map(|&sp| {
                let sp = Self::clamp_span_to_source(sp, source.len());
                let s = idx.to_linecol(sp.start_as_usize());
                // End line comes from `span_lines`, not `to_linecol(end)`:
                // the exclusive-end byte can land on a line start, which
                // would over-count one line — the same distinction the glyph
                // renderer handles with `span_lines` for its window.
                let e = idx.span_lines(sp).1;
                (s.line, e)
            })
            .collect();

        // Additional primary spans are rendered as extra `^` annotations —
        // mirroring glyph.rs's `extras_labels`.  Without this, only the
        // first primary span's underline and window were drawn, silently
        // dropping the rest (a correctness bug: `with_additional_span` added
        // spans that the HTML renderer never displayed).
        let extra_labels: SmallVec<[Label<'src>; 2]> = primary_spans[1..]
            .iter()
            .map(|&sp| Label::new(sp, ""))
            .collect();
        let all_labels: SmallVec<[&Label<'src>; 4]> =
            labels.iter().copied().chain(extra_labels.iter()).collect();

        // Window covers every primary span, not just the first.  The anchor is
        // `primary_lines[0]` (insertion order) — spans added with
        // `with_additional_span` may be out of source order, so the anchor is
        // snapshotted before the section list is sorted below.
        let primary_first_lines = primary_lines[0];
        let mut last_line = std::cmp::min(
            primary_first_lines.1.saturating_add(context_lines),
            text_last_line,
        );
        for &(_, e_line) in &primary_lines {
            last_line = last_line.max(std::cmp::min(
                e_line.saturating_add(context_lines),
                text_last_line,
            ));
        }
        let line_num_width = (last_line + 1).checked_ilog10().unwrap_or(0) as usize + 1;

        // ── Disjoint context sections (gap folding) ──────────────
        // Each primary span contributes a `[start − ctx, end + ctx]` window;
        // overlapping or adjacent windows merge (same rule the glyph renderer
        // uses for its source sections).  The rows *between* disjoint
        // sections are not drawn: a fold marker (`... (N lines omitted)`)
        // stands in for them, mirroring the human renderer.  Without this,
        // one primary on line 10 and another on line 1000 made the HTML
        // renderer walk and emit every one of the ~990 gap rows — O(gap)
        // work and output where the glyph renderer pays O(sections).
        // Section construction must see ascending start lines: the merge only
        // extends `sections.last_mut()`, so an out-of-source-order span (e.g.
        // a `with_additional_span` at line 5 added after one at line 10)
        // would otherwise fold into a section that starts *after* it and drop
        // its rows entirely.
        let mut sorted_lines: SmallVec<[(usize, usize); 4]> = primary_lines.clone();
        sorted_lines.sort_by_key(|&(s, _)| s);
        let mut sections: SmallVec<[(usize, usize); 4]> = SmallVec::new();
        for &(s_line, e_line) in &sorted_lines {
            let s0 = s_line.saturating_sub(context_lines);
            let e0 = std::cmp::min(e_line.saturating_add(context_lines), text_last_line);
            // Defensive: an inverted window is only reachable from a malformed
            // span, and it must not advance the fold cursor (`prev_section_last`)
            // into a marker for rows it never rendered.
            if s0 > e0 {
                continue;
            }
            if let Some(last) = sections.last_mut()
                && s0 <= last.1.saturating_add(1)
            {
                last.1 = last.1.max(e0);
            } else {
                sections.push((s0, e0));
            }
        }

        // Per-label (start_line, end_line), converted once.
        // `SmallVec<[(usize, usize); 4]>`: labels-per-diagnostic rarely exceeds
        // 4, so the common case stays inline — one fewer heap allocation per
        // source section.
        let label_line_ranges: SmallVec<[(usize, usize); 4]> = all_labels
            .iter()
            .map(|l| idx.span_lines(Self::clamp_span_to_source(l.span, source.len())))
            .collect();
        // Start-line-sorted permutation, computed once and shared by every
        // section so `LabelWindow` merges into a pre-sorted order instead of
        // re-sorting the union per section (`K·L log L` → `L log L + K·L`),
        // mirroring the glyph renderer.
        let mut label_order: SmallVec<[usize; 4]> = (0..label_line_ranges.len()).collect();
        label_order.sort_by_key(|&i| label_line_ranges[i].0);
        // Per-label "message emitted inline" flags: the fallback below must
        // not re-emit a message that a source row already rendered, but a
        // line-range test is not enough — a label point on the phantom line
        // after a trailing newline lies inside the range yet never enters
        // `underlines`.
        let n_labels = all_labels.len();
        let mut label_bits: u64 = 0;
        let mut label_extra: SmallVec<[bool; 4]> = if n_labels > 64 {
            smallvec![false; n_labels - 64]
        } else {
            smallvec![]
        };
        // One definition of the two-tier bitset (inline u64 + spill Vec), so
        // the set and query sites cannot drift on the `- 64` offset.
        let mark_label = |bits: &mut u64, extra: &mut SmallVec<[bool; 4]>, i: usize| {
            if i < 64 {
                *bits |= 1u64 << i;
            } else if let Some(slot) = extra.get_mut(i - 64) {
                *slot = true;
            }
        };
        let label_rendered = |bits: u64, extra: &SmallVec<[bool; 4]>, i: usize| -> bool {
            if i < 64 {
                (bits >> i) & 1 != 0
            } else {
                extra.get(i - 64).copied().unwrap_or(false)
            }
        };

        // ── Per-batch `SectionScratch` ────────────────────────────
        // Threaded through the `RenderCtx` exactly as the glyph renderer's
        // `write_source_section` does: the emitter path grows the heap
        // buffers once per batch instead of reallocating per diagnostic.
        // The no-ctx path (`diagnostic_to_html`) falls back to a fresh
        // local.  `underlines_buf` / `meta_buf` cannot be threaded — they
        // hold `&'a str` borrows into `extra_labels` / `labels`, which are
        // function-local — but both are `SmallVec<…; 8>`, inline for ≤7
        // annotations (the common case), so they contribute no heap.
        let mut local_scratch;
        let mut guard = cx.map(|c| c.scratch.borrow_mut());
        let scratch: &mut SectionScratch = match guard.as_deref_mut() {
            Some(s) => s,
            None => {
                local_scratch = SectionScratch::new();
                &mut local_scratch
            }
        };
        // `LabelWindow` takes its scratch by value; `mem::take` moves the
        // container out (no alloc) and we hand it back at the end so the
        // next section reuses its capacity.
        let window_scratch = std::mem::take(&mut scratch.label_window);
        let mut label_window = LabelWindow::new(&label_line_ranges, &label_order, window_scratch);

        let html = out;
        // Underline lane allocator — the row loop reuses it across lines.
        let lanes = &mut scratch.lanes;
        let row_buf = &mut scratch.row_buf;
        // Scratch buffers — reused across the batch like every other
        // `SectionScratch` field.  The payload carries only the label index;
        // messages resolve from `all_labels` at use sites, so neither buffer
        // holds a borrow into the per-section label list.
        let mut underlines_buf = std::mem::take(&mut scratch.underlines_buf);
        let mut meta_buf = std::mem::take(&mut scratch.meta_buf);

        // The filename is constant across every source row, so escape it once
        // here and reuse the (borrowed, allocation-free) result per row instead
        // of `format!`-ing a fresh `file:line:col` string only to escape and
        // discard it on each iteration.
        let escaped_file = self.escape_filename(file_name);
        // The `file:` prefix of the `data-location` attribute is also constant
        // per row; building it once keeps the row loop to pointer copies.
        // Built from parts — a clean path (the overwhelmingly common shape)
        // keeps `escape` borrowed, so the whole prefix is one small `String`
        // with no `format!` machinery.
        let escaped_file_prefix = if file_name.is_empty() {
            String::new()
        } else {
            let mut p = String::with_capacity(escaped_file.len() + 1);
            p.push_str(&escaped_file);
            p.push(':');
            p
        };

        // Walk the disjoint sections; a counted fold marker stands in for
        // every gap (mirrors the glyph renderer's
        // `... (N lines omitted)` between its source sections).
        let mut prev_section_last: Option<usize> = None;
        if let Some(&(first, _)) = sections.first()
            && first > 0
        {
            html.push_str("<span class='line-num'>");
            fmt::push_spaces(html, line_num_width);
            html.push_str(" │ </span>");
            html.push_str("<span class='fold'>");
            fmt::push_fold_marker(html, first);
            html.push_str("</span>\n");
        }
        for &(sec_first, sec_last) in &sections {
            if let Some(pl) = prev_section_last {
                let gap = sec_first.saturating_sub(pl + 1);
                if gap > 0 {
                    html.push_str("<span class='line-num'>");
                    fmt::push_spaces(html, line_num_width);
                    html.push_str(" │ </span>");
                    html.push_str("<span class='fold'>");
                    fmt::push_fold_marker(html, gap);
                    html.push_str("</span>\n");
                }
            }
            prev_section_last = Some(sec_last);

            for line_idx in sec_first..=sec_last {
                // O(1) slice via the shared index instead of indexing a
                // whole-file `Vec<&str>` that was collected only for this.
                let line = idx.line(line_idx);
                // Fast skip: when the primary span's line range does not cover
                // this row *and* no label window covers it, the full computation
                // would produce empty ranges — so clear the buffers directly and
                // keep the output byte-identical.  `LabelWindow::advance` is
                // idempotent for the same row (its cursor and expiry heap only
                // move forward), so pre-advancing here does not disturb the
                // re-advance inside `compute_line_annotation_ranges` on the rows
                // that do need it.  The advance itself must happen every row to
                // keep later rows' active-label state correct.
                if line_idx < primary_first_lines.0 || line_idx > primary_first_lines.1 {
                    label_window.advance(line_idx);
                    if label_window.active().is_empty() {
                        scratch.ranges_buf.clear();
                        meta_buf.clear();
                    } else {
                        compute_line_annotation_ranges(
                            span,
                            &all_labels,
                            &mut label_window,
                            source,
                            line,
                            line_idx,
                            starts,
                            &mut scratch.ranges_buf,
                            &mut meta_buf,
                            false,
                        );
                    }
                } else {
                    compute_line_annotation_ranges(
                        span,
                        &all_labels,
                        &mut label_window,
                        source,
                        line,
                        line_idx,
                        starts,
                        &mut scratch.ranges_buf,
                        &mut meta_buf,
                        true,
                    );
                }
                // One fused grapheme walk resolves this row's byte ranges to
                // display columns, measures the line width, and emits the
                // HTML-escaped, highlight-wrapped source — the same core the glyph
                // renderer uses, with `push_html_escaped` as its emit callback.
                // The row is rendered straight into `html` (`append = true`
                // preserves the gutter pushed just above), so the old
                // `scratch.rendered` → `html` full-row copy is gone.
                //
                // ── Line number ─────────────────────────────────────
                // write! straight into the output buffer (reuses its
                // capacity) instead of two per-row format! allocations.
                // `data-location` carries the click-to-copy target: the
                // primary span's line gets the full
                // `file:line:col` (1-based col from the span start),
                // every other row gets `file:line` (or `line` when the
                // diagnostic's filename is empty).  The filename is
                // pre-escaped above the loop; line/column are ASCII, so
                // the attribute is assembled straight into `html`.
                html.push_str("<span class='line-num' data-location='");
                if line_idx == start_line {
                    if file_name.is_empty() {
                        fmt::push_uint(html, (line_idx + 1) as u64);
                        html.push(':');
                        fmt::push_uint(html, (start_col + 1) as u64);
                    } else {
                        html.push_str(&escaped_file_prefix);
                        fmt::push_uint(html, (line_idx + 1) as u64);
                        html.push(':');
                        fmt::push_uint(html, (start_col + 1) as u64);
                    }
                } else if file_name.is_empty() {
                    fmt::push_uint(html, (line_idx + 1) as u64);
                } else {
                    html.push_str(&escaped_file_prefix);
                    fmt::push_uint(html, (line_idx + 1) as u64);
                }
                html.push_str("' aria-label='Line ");
                fmt::push_uint(html, (line_idx + 1) as u64);
                html.push_str("'>");
                fmt::push_uint_padded(html, (line_idx + 1) as u64, line_num_width);
                html.push_str(" │ </span>");

                let display_width = render_line_merged_with(
                    crate::diagnostics::glyph::LineRenderArgs {
                        line,
                        ranges: &scratch.ranges_buf,
                        meta: &meta_buf,
                        out: &mut underlines_buf,
                        rendered: html,
                        highlight_on: "<span class='highlight'>",
                        highlight_off: "</span>",
                        display_buf: &mut scratch.display_buf,
                        order_buf: &mut scratch.order_buf,
                        done_buf: &mut scratch.done_buf,
                        active_buf: &mut scratch.active_buf,
                        tab_width,
                        content_start_col: line_num_width + 3,
                        append: true,
                    },
                    push_html_escaped,
                );
                let underlines = &underlines_buf;
                html.push('\n');

                // ── Underline annotations (one row per overlap depth) ───
                if !underlines.is_empty() {
                    let lane_count = lanes.assign(underlines);
                    for lane_idx in 0..lane_count {
                        html.push_str("<span class='line-num'>");
                        fmt::push_spaces(html, line_num_width);
                        html.push_str(" │ </span>");
                        // Plain (uncoloured) row for HTML — per-char colouring
                        // is the terminal renderer's job; here the CSS class
                        // `.underline` carries the visual weight.
                        let visible =
                            lanes.row(underlines, lane_idx, display_width, row_buf, |_ch| "", "");
                        if visible {
                            // `row` is called with `color_for = |_| ""` and
                            // `reset = ""`, so `row_buf` contains only padding
                            // spaces and the underline chars `^` / `~` / `-`
                            // (see `AnnotationKind::underline_char`) — none of
                            // which need HTML escaping.  Skipping `escape` avoids
                            // two dead SIMD scans per underline row.
                            debug_assert!(
                                row_buf
                                    .bytes()
                                    .all(|b| matches!(b, b' ' | b'^' | b'~' | b'-')),
                                "row_buf contains a char requiring escape: {row_buf:?}"
                            );
                            html.push_str("<span class='underline'>");
                            html.push_str(&row_buf);
                            html.push_str("</span>");
                        }
                        html.push('\n');
                    }

                    // ── Label messages ──────────────────────────────
                    for (col, _ulen, _ch, label_idx) in underlines.iter().copied() {
                        // The payload carries only the label index; the message
                        // resolves from the section's `all_labels` (the synthetic
                        // primary has `None` and an empty message).
                        let msg: &str = match label_idx {
                            Some(i) => all_labels[i].message.as_ref(),
                            None => "",
                        };
                        if msg.is_empty() {
                            continue;
                        }
                        // Suppress the message on every line except the label's
                        // final covered line — mirrors `GlyphRenderer`'s
                        // `is_multiline_nonfinal`. Keyed by label index so two
                        // labels sharing one message stay distinct.
                        let is_multiline_nonfinal =
                            label_idx.is_some_and(|i| label_line_ranges[i].1 != line_idx);
                        if is_multiline_nonfinal {
                            continue;
                        }
                        if let Some(i) = label_idx {
                            mark_label(&mut label_bits, &mut label_extra, i);
                        }
                        html.push_str("<span class='line-num'>");
                        fmt::push_spaces(html, line_num_width);
                        html.push_str(" │ </span>");
                        // `col` is a display column → pad with that many spaces.
                        // Write the connector directly into `html` — `escape`'s
                        // fast path (no allocation for space/`|`) made the
                        // intermediate `String` the only real cost here.
                        html.push_str("<span class='label'>");
                        fmt::push_spaces(html, col);
                        html.push('|');
                        html.push(' ');
                        self.push_escaped_cached(html, msg);
                        html.push_str("</span>\n");
                    }
                }

                // Multi-line label middle rows: connector only.  The shared
                // range policy suppresses middle-line underlines, so derive the
                // connector column from the label span, matching the glyph
                // renderer's policy exactly.
                for &i in label_window.active() {
                    let (start, end) = label_line_ranges[i];
                    if line_idx <= start || line_idx >= end || all_labels[i].message.is_empty() {
                        continue;
                    }
                    let span_start = all_labels[i].span.start_as_usize();
                    // Middle rows are not the label's start line; measure the
                    // connector from the start line so it lines up with the
                    // opening caret instead of collapsing to column 0.
                    let start_line = idx.line(start);
                    let start_line_start = starts.get(start).map_or(0, |&s| s as usize);
                    let start_off = span_start
                        .saturating_sub(start_line_start)
                        .min(start_line.len());
                    let col = byte_offset_to_display(start_line, start_off, tab_width);
                    html.push_str("<span class='line-num'>");
                    fmt::push_spaces(html, line_num_width);
                    html.push_str(" │ </span>");
                    html.push_str("<span class='label'>");
                    fmt::push_spaces(html, col);
                    html.push('|');
                    html.push_str("</span>\n");
                }
            }
        } // section loop

        // A label whose lines fall outside the rendered rows still owns
        // user-visible text (e.g. a span at EOF lands on the phantom line).
        // Emit it as a note, mirroring GlyphRenderer's fallback, so no
        // label message is silently dropped from the HTML view.  Keyed off
        // what was actually written, not the line range: a label can lie
        // inside the range yet still miss every drawn row (e.g. an EOF
        // point) and must fall back too.
        for (i, lbl) in all_labels.iter().enumerate() {
            let rendered = label_rendered(label_bits, &label_extra, i);
            if lbl.message.is_empty() || rendered {
                continue;
            }
            // `start`, matching the glyph fallback (`at start` is the
            // conventional diagnostic location); the old `end` here made the
            // HTML note point one span past where the terminal note points
            // for the same label.
            let (start_line, start_col) =
                idx.linecol_display(lbl.span.start_as_usize().min(source.len()));
            html.push_str("<span class='label'>= note: ");
            self.push_escaped_cached(html, &lbl.message);
            html.push_str(" at ");
            fmt::push_uint(html, (start_line + 1) as u64);
            html.push(':');
            fmt::push_uint(html, (start_col + 1) as u64);
            html.push_str("</span>\n");
        }

        // Restore the taken label-window scratch so the next section reuses
        // its capacity.
        scratch.label_window = label_window.into_scratch();
        scratch.underlines_buf = underlines_buf;
        scratch.meta_buf = meta_buf;
        true
    }

    /// Escape into `out` through a bounded process-wide memo.  Static
    /// label/help strings recur across every diagnostic in a batch; caching
    /// the sanitized+escaped form turns the second..Nth occurrence into a
    /// copy instead of a sanitize scan + escape.  Bounded so hostile dynamic
    /// messages cannot grow it without limit.
    fn push_escaped_cached(&self, out: &mut String, s: &str) {
        use std::cell::RefCell;
        // Thread-local mirrors `WRAP_MEMO`/`WIDTH_CACHE`: `render_parallel`
        // workers otherwise contend on one global `Mutex` for the same static
        // label strings.  Duplication is bounded and per-thread.
        thread_local! {
            static ESCAPE_CACHE: RefCell<(
                rustc_hash::FxHashMap<compact_str::CompactString, EscapeEntry>,
                std::collections::VecDeque<compact_str::CompactString>,
            )> = RefCell::new((
                rustc_hash::FxHashMap::default(),
                std::collections::VecDeque::new(),
            ));
        }
        // Hash lookup costs more than escaping a tiny string, so keep short
        // strings entirely out of the cache.
        if s.len() <= 64 {
            let escaped = self.escape(s);
            out.push_str(escaped.as_ref());
            return;
        }
        // The cache is a map plus a FIFO of the same keys; every path keeps them
        // in step, and the eviction loop below *relies* on that (it pops while
        // `len >= 1024`, breaking when the FIFO runs dry).  If they ever desync,
        // that loop stops truncating and the map grows without bound, so make the
        // invariant loud in debug and recover instead of unbounding in release.
        #[cfg(debug_assertions)]
        ESCAPE_CACHE.with(|cell| {
            let cache = cell.borrow();
            debug_assert_eq!(
                cache.0.len(),
                cache.1.len(),
                "escape cache map/FIFO desynchronised"
            );
        });
        let hit = ESCAPE_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            cache.0.get_mut(s).map(|entry| {
                entry.referenced = true;
                std::sync::Arc::clone(&entry.value)
            })
        });
        if let Some(value) = hit {
            out.push_str(&value);
            return;
        }
        let escaped = self.escape(s);
        if escaped.len() > 4096 {
            out.push_str(escaped.as_ref());
            return;
        }
        // Allocate the owned key only on a miss; hits above use `&str`.
        let key = compact_str::CompactString::from(s);
        let value: std::sync::Arc<str> = std::sync::Arc::from(escaped.as_ref());
        ESCAPE_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            while cache.0.len() >= 1024 {
                let Some(old) = cache.1.pop_front() else {
                    // Desync recovery (the invariant is asserted above in debug):
                    // without a FIFO entry this `break` would leave the map at or
                    // above the cap, and every subsequent insert would grow it
                    // without bound.  Dropping the map is always correct — it is
                    // a cache.
                    cache.0.clear();
                    break;
                };
                let evict = match cache.0.get_mut(&old) {
                    Some(entry) if entry.referenced => {
                        entry.referenced = false;
                        false
                    }
                    Some(_) => true,
                    None => false,
                };
                if evict {
                    cache.0.remove(&old);
                    break;
                }
                cache.1.push_back(old);
            }
            cache.0.insert(
                key.clone(),
                EscapeEntry {
                    value: std::sync::Arc::clone(&value),
                    referenced: false,
                },
            );
            cache.1.push_back(key);
        });
        out.push_str(&value);
    }

    /// Escape text for HTML element content *and* single/double-quoted
    /// attribute values.  `'` is escaped in addition to the classic four so
    /// this helper stays safe if it is ever used inside a `class='…'`
    /// attribute.
    fn escape<'a>(&self, s: &'a str) -> Cow<'a, str> {
        // Control bytes (C0/DEL/C1, excluding the layout-intended tab/newline
        // the multiline sanitizer keeps) are replaced with U+FFFD before
        // escaping: HTML output must not carry raw terminal control sequences,
        // and this matches the human/short/JSON renderers' hygiene.  The
        // sanitizer borrows when the input is already clean; when it owns, the
        // escaped result must be owned too (it cannot borrow the temporary).
        match crate::diagnostics::sanitize_for_terminal_multiline(s) {
            Cow::Borrowed(clean) => self.escape_clean(clean),
            Cow::Owned(clean) => match self.escape_clean(&clean) {
                Cow::Borrowed(_) => Cow::Owned(clean),
                Cow::Owned(escaped) => Cow::Owned(escaped),
            },
        }
    }

    /// Filenames are a single-line field: unlike messages/source text, they
    /// must not retain `\n`/`\t`, or a `data-location` attribute and the
    /// visible location header could be split across lines.
    fn escape_filename<'a>(&self, s: &'a str) -> Cow<'a, str> {
        // One pass covers the dominant `src/foo.rs` shape: all bytes are
        // printable ASCII.  Track HTML specials without a second scan; only
        // non-ASCII/control filenames (or actual escapes) take a second pass.
        let mut needs_escape = false;
        for &b in s.as_bytes() {
            if !(0x20..0x7F).contains(&b) {
                return match crate::diagnostics::sanitize_for_terminal(s) {
                    Cow::Borrowed(clean) => self.escape_clean(clean),
                    Cow::Owned(clean) => match self.escape_clean(&clean) {
                        Cow::Borrowed(_) => Cow::Owned(clean),
                        Cow::Owned(escaped) => Cow::Owned(escaped),
                    },
                };
            }
            needs_escape |= matches!(b, b'&' | b'<' | b'>' | b'"' | b'\'');
        }
        if needs_escape {
            self.escape_clean(s)
        } else {
            Cow::Borrowed(s)
        }
    }

    fn escape_clean<'a>(&self, s: &'a str) -> Cow<'a, str> {
        let bytes = s.as_bytes();
        // Merged iterator: a single O(len) pass over the haystack regardless
        // of which escape class dominates (see [`HtmlEscapeIter`]).
        let mut iter = HtmlEscapeIter::new(bytes);
        let Some(first) = iter.next() else {
            return Cow::Borrowed(s);
        };
        // `&` expands 5×, so the old `len + len/4` guarantee under-reserved
        // for escape-dense label text; 2× avoids the realloc for the common
        // `<`/`>`/`"`/`'` cases at a small cost on clean strings.
        let mut out = String::with_capacity(s.len().saturating_mul(2));
        let mut start = 0;
        let mut idx = first;
        loop {
            out.push_str(&s[start..idx]);
            match bytes[idx] {
                b'&' => out.push_str("&amp;"),
                b'<' => out.push_str("&lt;"),
                b'>' => out.push_str("&gt;"),
                b'"' => out.push_str("&quot;"),
                b'\'' => out.push_str("&#39;"),
                _ => unreachable!(),
            }
            start = idx + 1;
            let Some(next) = iter.next() else {
                break;
            };
            idx = next;
        }
        out.push_str(&s[start..]);
        Cow::Owned(out)
    }
}

/// Lazily yields the byte positions of the full HTML escape set
/// (`& < > " '`) in a haystack.
///
/// Two `memchr` iterators — one for the tag class (`& < >`), one for the
/// quote class (`" '`) — advance independently and a merge selects the
/// earlier match, so every byte of the haystack is loaded at most once no
/// matter which class dominates.  This is what keeps [`HtmlRenderer::escape`]
/// O(len) for *every* input shape: the per-call two-step probe
/// ([`next_html_escape_byte`]) degrades to O(len·E) when quotes are dense
/// and `<>&` are absent (e.g. a line of string literals), because each
/// probe's `memchr3` scans the whole remaining suffix while finding nothing.
pub(crate) struct HtmlEscapeIter<'a> {
    tag: memchr::Memchr3<'a>,
    quote: memchr::Memchr2<'a>,
    next_tag: Option<usize>,
    next_quote: Option<usize>,
}

impl<'a> HtmlEscapeIter<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        let mut tag = memchr::memchr3_iter(b'&', b'<', b'>', bytes);
        let mut quote = memchr::memchr2_iter(b'"', b'\'', bytes);
        let next_tag = tag.next();
        let next_quote = quote.next();
        Self {
            tag,
            quote,
            next_tag,
            next_quote,
        }
    }
}

impl Iterator for HtmlEscapeIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let idx = match (self.next_tag, self.next_quote) {
            (Some(a), Some(q)) => a.min(q),
            (Some(a), None) => a,
            (None, Some(q)) => q,
            (None, None) => return None,
        };
        if self.next_tag == Some(idx) {
            self.next_tag = self.tag.next();
        }
        if self.next_quote == Some(idx) {
            self.next_quote = self.quote.next();
        }
        Some(idx)
    }
}

/// Escape a chunk of source text for HTML element content, appending it to
/// `out`.  This is the emit callback `render_line_merged_with` drives, so it
/// runs once per same-highlight run (or per grapheme) rather than once per
/// byte.  The escape set matches the source-row loop it replaced (`&`, `<`,
/// `>`, `"`); the all-clean fast path avoids the char walk entirely.
fn push_html_escaped(out: &mut String, s: &str) {
    // Source text is untrusted too: raw C0/DEL/C1 must not reach HTML (the
    // message path already sanitizes).  Per-run sanitization keeps offsets
    // within the callback; the row/annotation columns were computed before
    // this point.
    //
    // KNOWN LIMITATION (deliberately not "fixed" by sanitizing the
    // whole line up front): the underline columns come from
    // `compute_line_annotation_ranges`, which measures the **raw** line, while
    // this callback replaces one control byte with the three-byte U+FFFD.  A
    // source line containing C0/DEL/C1 can therefore show its carets shifted
    // relative to the visible text.  Hoisting the sanitize before the row
    // renderer would fix the alignment but invalidate every byte offset the
    // ranges were built from (they index the raw line), so it is a change to the
    // shared row pipeline, not a local one.  `control_byte_in_a_source_row_shifts_the_caret_lane`
    // pins the current behaviour so a future fix is a visible diff.
    let sanitized = crate::diagnostics::sanitize_for_terminal_multiline(s);
    let s = sanitized.as_ref();
    let bytes = s.as_bytes();
    // Merged iterators for the source-content escape set (`& < > "`): the
    // tag and quote classes advance independently and each byte of the
    // haystack is loaded at most once, so the pass stays O(len) even when
    // quotes are dense and `<>&` are absent (the per-call
    // `next_html_source_escape_byte` probe degraded to O(len·E) there).
    let mut tag = memchr::memchr3_iter(b'&', b'<', b'>', bytes);
    let mut quote = memchr::memchr_iter(b'"', bytes);
    let mut next_tag = tag.next();
    let mut next_quote = quote.next();
    let mut start = 0;
    loop {
        let idx = match (next_tag, next_quote) {
            (Some(a), Some(q)) => a.min(q),
            (Some(a), None) => a,
            (None, Some(q)) => q,
            (None, None) => break,
        };
        out.push_str(&s[start..idx]);
        match bytes[idx] {
            b'&' => out.push_str("&amp;"),
            b'<' => out.push_str("&lt;"),
            b'>' => out.push_str("&gt;"),
            b'"' => out.push_str("&quot;"),
            _ => unreachable!(),
        }
        start = idx + 1;
        if next_tag == Some(idx) {
            next_tag = tag.next();
        }
        if next_quote == Some(idx) {
            next_quote = quote.next();
        }
    }
    out.push_str(&s[start..]);
}

/// Next byte in `hay` that is one of `&`, `<`, `>`, `"`, `'` (full HTML
/// escape set).  SIMD-accelerated via `memchr3` + `memchr2`.
///
/// The second scan is **bounded by the first hit**: once an `&<>` is found
/// at `a`, any `"`/`'` after `a` is irrelevant — the first escape byte is
/// at or before `a`.  Only `hay[..a]` is searched for quotes, so each call
/// scans at most `2·a` bytes while the outer escape loop consumes `a+1`;
/// total work across the loop is O(len), not O(E·len) (a lone quote near
/// the end of a line full of `<` used to force a full suffix rescan per
/// escape byte).
/// Test-only: the bounded-second-scan reference the production
/// fused-iterator path replaced.  No in-crate caller remains —
/// `HtmlEscapeIter` subsumed it — but the differential test keeps pinning
/// the discipline against a naive reference, so it stays compiled under
/// `cfg(test)` only.
#[cfg(test)]
#[inline]
fn next_html_escape_byte(hay: &[u8]) -> Option<usize> {
    match memchr::memchr3(b'&', b'<', b'>', hay) {
        Some(a) => memchr::memchr2(b'"', b'\'', &hay[..a]).or(Some(a)),
        None => memchr::memchr2(b'"', b'\'', hay),
    }
}

/// Test-only: next byte in `hay` that is one of `&`, `<`, `>`, `"`
/// (source-content escape set, no apostrophe).  Same bounded-second-scan
/// discipline as [`next_html_escape_byte`].
#[cfg(test)]
#[inline]
fn next_html_source_escape_byte(hay: &[u8]) -> Option<usize> {
    match memchr::memchr3(b'&', b'<', b'>', hay) {
        Some(a) => memchr::memchr(b'"', &hay[..a]).or(Some(a)),
        None => memchr::memchr(b'"', hay),
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::diagnostics::Diagnostic;
    use crate::diagnostics::ErrCode;

    #[test]
    fn long_message_escape_cache_is_consistent() {
        let r = HtmlRenderer::new();
        let s = "a&b<c>".repeat(100);
        let mut first = String::new();
        let mut second = String::new();
        r.push_escaped_cached(&mut first, &s);
        r.push_escaped_cached(&mut second, &s);
        assert_eq!(first, second);
        assert!(!first.contains("<c>"), "raw HTML survived: {first}");
    }

    #[test]
    fn escape_filename_is_single_pass_and_sanitizes_controls() {
        let r = HtmlRenderer::new();
        assert!(matches!(
            r.escape_filename("src/main.rs"),
            Cow::Borrowed("src/main.rs")
        ));
        assert_eq!(r.escape_filename("a<b&c").as_ref(), "a&lt;b&amp;c");
        let cleaned = r.escape_filename("a\nb\x01c");
        assert!(
            !cleaned.contains('\n') && !cleaned.contains('\x01'),
            "{cleaned:?}"
        );
    }

    #[test]
    fn source_control_bytes_are_sanitized() {
        let diag = Diagnostic::error("x")
            .with_span(Span::new(crate::ast::FileId::DUMMY, 0, 1))
            .with_source("a\x01b\n");
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(!html.contains('\x01'), "raw control byte in source: {html}");
    }

    /// Multi-file guard, HTML analog of the glyph renderer's
    /// `foreign_primary_spans_render_as_location_only_notes`: a primary span
    /// from another file has meaningless offsets in this diagnostic's buffer.
    /// It must not stretch the context window or draw bogus sections/columns;
    /// the anchor file's own content still renders exactly once.
    #[test]
    fn foreign_primary_spans_do_not_distort_the_anchor_source() {
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::from_raw(1), 0, 3))
            .with_additional_span(Span::new(FileId::from_raw(2), 500, 505))
            .with_source("def x\nset y\n");
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        // `def` is keyword-highlighted, so the raw string "def x" never
        // appears contiguously; assert on the highlighted form instead.
        assert_eq!(
            html.matches("<span class='highlight'>def</span> x").count(),
            1,
            "html: {html}"
        );
        assert_eq!(html.matches("set y").count(), 1, "html: {html}");
        // A foreign offset clamped through the anchor's line index blew the
        // context window up to the last source line; two adjacent lines leave
        // nothing to fold, so any fold marker means a bogus foreign section.
        assert!(!html.contains("lines omitted"), "html: {html}");
    }

    /// A diagnostic with no primary span must not drop its labels: they are
    /// emitted as `= note:` fallback lines instead of a source section.
    #[test]
    fn labels_survive_a_missing_primary_span() {
        let diag = Diagnostic::error("boom")
            .with_label(Span::new(FileId::DUMMY, 0, 3), "look here")
            .with_source("abc\ndef\n");
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(html.contains("look here"), "label dropped: {html}");
        assert!(html.contains("= note:"), "html: {html}");
    }

    /// `with_opts` must drive the context window exactly like an explicit
    /// `with_context_lines`, and a wider default context must render more
    /// rows — the CLI's `--context-lines` reaches HTML output.
    #[test]
    fn with_opts_reaches_context_lines() {
        use crate::diagnostics::DiagOpts;

        let src = "a\nb\nc\nd\ne\n";
        let diag = Diagnostic::error("x")
            .with_span(Span::new(FileId::DUMMY, 7, 8))
            .with_source(src);
        let mut opts = DiagOpts::default();
        opts.context_lines = 0;
        let via_opts = HtmlRenderer::with_opts(&opts).diagnostic_to_html(&diag);
        let via_override = HtmlRenderer::new()
            .with_context_lines(0)
            .diagnostic_to_html(&diag);
        assert_eq!(via_opts, via_override);
        let wide = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(
            wide.lines().count() > via_opts.lines().count(),
            "default context must render more rows than context_lines=0"
        );
    }

    /// `context_lines = 0` must show exactly the covered lines — the old
    /// `e_line + 1 + context_lines` window (with `e_line` taken from the
    /// exclusive-end byte) dragged one or two lines below the span even at
    /// zero context.  Mirrors glyph's `zero_context_shows_no_line_below_the_span`.
    /// Rows are counted via the per-row `data-location` line-number spans
    /// (fold markers use a `line-num` span *without* `data-location`, so they
    /// don't pollute the count).
    #[test]
    fn zero_context_shows_exactly_the_covered_lines() {
        let src = "a\nb\nc\nd\ne\n";
        // Span covering `c` (line 2) mid-line: end byte 5 is the newline.
        let diag = Diagnostic::error("x")
            .with_span(Span::new(FileId::DUMMY, 4, 5))
            .with_source(src);
        let html = HtmlRenderer::new()
            .with_context_lines(0)
            .diagnostic_to_html(&diag);
        assert_eq!(
            html.matches("<span class='line-num' data-location=")
                .count(),
            1,
            "ctx=0 must render exactly the span's line: {html}"
        );
        assert!(
            html.contains(">c<") || html.contains("c\n"),
            "row text: {html}"
        );

        // Span ending exactly on a line start (exclusive end at byte 6,
        // the start of line 3): the last *covered* byte is line 2's
        // newline, so ctx=0 must render exactly one row — the old
        // `to_linecol(end)` + `+1` window rendered two.
        let diag = Diagnostic::error("x")
            .with_span(Span::new(FileId::DUMMY, 4, 6))
            .with_source(src);
        let html = HtmlRenderer::new()
            .with_context_lines(0)
            .diagnostic_to_html(&diag);
        assert_eq!(
            html.matches("<span class='line-num' data-location=")
                .count(),
            1,
            "line-start-ending span must not drag line 3 in at ctx=0: {html}"
        );
    }

    /// `context_lines = 1` shows exactly one line below the covered lines —
    /// not two (the second compounding `+1` the old window had).
    #[test]
    fn context_lines_one_shows_exactly_one_line_below() {
        let src = "a\nb\nc\nd\ne\n";
        let diag = Diagnostic::error("x")
            .with_span(Span::new(FileId::DUMMY, 4, 5)) // `c` on line 2
            .with_source(src);
        let html = HtmlRenderer::new()
            .with_context_lines(1)
            .diagnostic_to_html(&diag);
        assert_eq!(
            html.matches("<span class='line-num' data-location=")
                .count(),
            3,
            "ctx=1 must render lines 1-3 exactly: {html}"
        );
    }

    /// `tab_width` from the batch `DiagOpts` reaches the row renderer (the
    /// old code hardcoded `DEFAULT_TAB_WIDTH`), so tab-expanded underlines
    /// differ between tab stop widths.
    #[test]
    fn tab_width_follows_ctx_opts() {
        use crate::diagnostics::DiagOpts;

        let src = "\tfoo\nbar\n";
        let diag = Diagnostic::error("x")
            .with_span(Span::new(FileId::DUMMY, 1, 4))
            .with_source(src);
        let render = |tab: usize| {
            let opts = DiagOpts {
                tab_width: tab,
                ..Default::default()
            };
            let cx = RenderCtx::new(&opts, None);
            HtmlRenderer::new().render(&diag, &cx)
        };
        assert_ne!(render(1), render(8), "tab_width must reach the HTML rows");
    }

    /// The bounded-second-scan discipline of [`next_html_escape_byte`] and
    /// [`next_html_source_escape_byte`]: with a lone quote near the end of a
    /// line full of `<`, the old unbounded `memchr2`/`memchr` rescanned the
    /// whole suffix per escape byte — O(E·len).  Pinned two ways: the result
    /// must equal a naive reference for every interleave, and a 10⁴-byte
    /// adversarial input must finish under a generous wall-clock bound that
    /// the quadratic behaviour would blow by orders of magnitude.
    #[test]
    fn bounded_escape_scan_matches_naive_reference() {
        fn naive(hay: &[u8]) -> Option<usize> {
            hay.iter()
                .position(|&b| matches!(b, b'&' | b'<' | b'>' | b'"' | b'\''))
        }
        fn naive_source(hay: &[u8]) -> Option<usize> {
            hay.iter()
                .position(|&b| matches!(b, b'&' | b'<' | b'>' | b'"'))
        }
        // Every short interleave of one trigger byte and one quote.
        for a in [b'&', b'<', b'>'] {
            for q in [b'"', b'\''] {
                for (i, hay) in [
                    vec![a, q],
                    vec![q, a],
                    vec![a, b'x', q],
                    vec![q, b'x', a],
                    vec![b'x', a, b'y', q, b'z'],
                ]
                .into_iter()
                .enumerate()
                {
                    assert_eq!(
                        next_html_escape_byte(&hay),
                        naive(&hay),
                        "case {i} ({}, {})",
                        a as char,
                        q as char
                    );
                }
                let hay = vec![a, b'x', b'"'];
                assert_eq!(next_html_source_escape_byte(&hay), naive_source(&hay));
            }
        }
        // None cases.
        assert_eq!(next_html_escape_byte(b"plain text"), None);
        assert_eq!(next_html_source_escape_byte(b"plain 'text'"), None);

        // Quadratic-regression guard: 10⁴ bytes, escapes densely at the front
        // and a single trailing quote.  The old code rescans the ~5 KB suffix
        // once per escape byte; the bounded scan touches each byte O(1)
        // times.  The bound is deliberately loose (50 ms) — the quadratic
        // shape costs seconds, so only a regression trips it.
        let adversarial: String = std::iter::repeat('<')
            .take(10_000)
            .chain(std::iter::once('"'))
            .collect();
        let start = std::time::Instant::now();
        let mut sink = String::new();
        push_html_escaped(&mut sink, &adversarial);
        assert_eq!(sink.matches("&lt;").count(), 10_000);
        assert!(sink.ends_with("&quot;"));
        assert!(
            start.elapsed().as_millis() < 50,
            "escape pass took {:?} — quadratic rescan regressed?",
            start.elapsed()
        );
    }

    #[test]
    fn test_code_not_double_wrapped() {
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 1));
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        // The code span appears in the collapsible summary line and in
        // the body div — it must not be NESTED inside another code span
        // (the original regression).
        assert!(
            !html.contains("<span class='code'><span class='code'>"),
            "code span nested: {html}"
        );
        assert!(html.contains("<span class='code'>E030</span>"), "{html}");
        // The summary carries the level label as a code span too.
        assert!(
            html.contains("<summary>"),
            "missing collapsible summary: {html}"
        );
    }

    #[test]
    fn test_span_location_present() {
        let diag = Diagnostic::error("boom").with_span(Span::new(FileId::DUMMY, 0, 1));
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(html.contains("at 0..1"), "html: {html}");
    }

    /// a zero-width label at a line end gets a one-cell
    /// marker inline (shared with the glyph renderer) instead of vanishing
    /// or falling back to a location note.
    #[test]
    fn zero_width_label_renders_one_cell_marker() {
        let src = "aaa\nbbb\nccc\n";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 8, 11))
            .with_label(Span::new(FileId::DUMMY, 11, 11), "here")
            .with_source(src);
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert_eq!(html.matches("here").count(), 1, "html: {html}");
        assert!(!html.contains("at 3:4"), "html: {html}");
        // Primary `^^^` plus the insertion-point caret share one lane.
        assert!(
            html.contains("<span class='underline'>^^^^</span>"),
            "html: {html}"
        );
    }

    /// Multi-file guard: a label from another file cannot be located in the
    /// rendered buffer; it surfaces as a range-located note instead of being
    /// dropped or mis-rendered.
    #[test]
    fn foreign_labels_render_as_range_notes() {
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_label(Span::new(FileId::from_raw(2), 500, 505), "elsewhere")
            .with_source("x");
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(
            html.contains("elsewhere at 500..505"),
            "foreign label note missing: {html}"
        );
        assert_eq!(
            html.matches("elsewhere").count(),
            1,
            "foreign label rendered twice: {html}"
        );
    }

    #[test]
    fn test_code_str_is_html_escaped() {
        let diag = Diagnostic::error("boom")
            .with_code_str("</span><img src=x onerror=alert(1)>")
            .with_span(Span::new(FileId::DUMMY, 0, 1));
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(!html.contains("<img"), "code injected raw: {html}");
        assert!(html.contains("&lt;/span&gt;"), "code not escaped: {html}");
        // The escaped code string is well-formed inside both the
        // collapsible summary and the body code spans (the original
        // regression: an unescaped `</span>` in the code terminated the
        // code span early).  The summary carries a level-label code span
        // plus the body carries the actual code span.
        assert_eq!(
            html.matches("class='code'").count(),
            2,
            "expected 2 code spans (summary level + body): {html}"
        );
    }

    #[test]
    fn test_explain_summary_code_is_escaped() {
        // `ErrCode::explain()` returns a non-empty fallback for unknown codes,
        // so the `<details><summary>Explain …</summary>` site is reachable.
        let diag = Diagnostic::error("x")
            .with_code_str("<script>alert(1)</script>")
            .with_span(Span::new(FileId::DUMMY, 0, 1));
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(!html.contains("<script>"), "injected: {html}");
        assert!(html.contains("&lt;script&gt;"), "not escaped: {html}");
    }

    #[test]
    fn html_respects_context_lines() {
        let src = "l0\nl1\nl2\nl3\nl4\nl5\n";
        let diag = Diagnostic::error("x")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 3, 4))
            .with_source(src);
        let narrow = HtmlRenderer::new()
            .with_context_lines(0)
            .diagnostic_to_html(&diag);
        let wide = HtmlRenderer::new()
            .with_context_lines(3)
            .diagnostic_to_html(&diag);
        assert!(
            wide.lines().count() > narrow.lines().count(),
            "narrow:\n{narrow}\nwide:\n{wide}"
        );
    }

    #[test]
    fn streams_full_document() {
        use crate::diagnostics::DiagOpts;
        use crate::diagnostics::emitter::Emitter;
        use crate::diagnostics::sink::{StringSink, string};

        let diag = Diagnostic::error("boom").with_code_str(ErrCode::E030);
        let mut emitter = Emitter::new(HtmlRenderer::new(), string(), DiagOpts::default());
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut [diag], &mut sink).unwrap();
        let out = sink.into_string();
        assert!(out.starts_with("<!DOCTYPE html>"), "out: {out}");
        assert!(out.trim_end().ends_with("</html>"), "out: {out}");
        assert!(
            out.contains("<details class='diag error' open role='alert'>"),
            "out: {out}"
        );
        assert!(out.contains("class='summary'"), "out: {out}");
    }

    #[test]
    fn max_errors_truncates_and_notes_suppression() {
        use crate::diagnostics::DiagOpts;
        use crate::diagnostics::emitter::Emitter;
        use crate::diagnostics::sink::{StringSink, string};

        let mut diags: Vec<_> = (0..5)
            .map(|i| Diagnostic::error(format!("e{i}")).with_code_str(ErrCode::E030))
            .collect();
        let opts = DiagOpts {
            max_errors: Some(2),
            ..Default::default()
        };
        let mut emitter = Emitter::new(HtmlRenderer::new(), string(), opts);
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut diags, &mut sink).unwrap();
        let out = sink.into_string();
        assert_eq!(
            out.matches("<details class='diag error' open role='alert'>")
                .count(),
            2,
            "{out}"
        );
        assert!(
            out.contains("<div class='summary' role='status'>3 additional errors suppressed</div>"),
            "{out}"
        );
        // Summary still reports the true total.
        assert!(out.contains("5 errors"), "{out}");
    }

    #[test]
    fn reverse_errors_reorders_without_changing_counts() {
        use crate::diagnostics::DiagOpts;
        use crate::diagnostics::emitter::Emitter;
        use crate::diagnostics::sink::{StringSink, string};

        let mut diags = vec![Diagnostic::error("first"), Diagnostic::error("second")];
        let opts = DiagOpts {
            reverse_errors: true,
            ..Default::default()
        };
        let mut emitter = Emitter::new(HtmlRenderer::new(), string(), opts);
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut diags, &mut sink).unwrap();
        let out = sink.into_string();
        let second = out.find("second").expect("second present");
        let first = out.find("first").expect("first present");
        assert!(second < first, "not reversed: {out}");
        assert!(
            out.contains("2 errors"),
            "summary lost the true total: {out}"
        );
    }

    #[test]
    fn json_suppressed_note_is_empty() {
        use crate::diagnostics::DiagOpts;
        use crate::diagnostics::emitter::Emitter;
        use crate::diagnostics::renderers::JsonRenderer;
        use crate::diagnostics::sink::{StringSink, string};

        let mut diags: Vec<_> = (0..3).map(|i| Diagnostic::error(format!("e{i}"))).collect();
        let opts = DiagOpts {
            max_errors: Some(1),
            ..Default::default()
        };
        let mut emitter = Emitter::new(JsonRenderer::new(false), string(), opts);
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut diags, &mut sink).unwrap();
        let out = sink.into_string();
        // The suppressed note must NOT leak into the JSON document.
        assert!(!out.contains("suppressed"), "note leaked into JSON: {out}");
        // The document must still be a valid single JSON array.
        assert!(
            out.starts_with('[') && out.trim_end().ends_with(']'),
            "{out}"
        );
        assert_eq!(out.matches("\"message\"").count(), 1, "{out}");
    }

    #[test]
    fn html_multiline_label_message_rendered_once() {
        let source = "aaa\nbbb\nccc\n";
        let diag = Diagnostic::error("multi")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_label(Span::new(FileId::DUMMY, 0, 11), "spans three lines")
            .with_source(source);
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert_eq!(
            html.matches("spans three lines").count(),
            1,
            "multi-line label repeated:\n{html}"
        );
    }

    #[test]
    fn html_multiline_connector_uses_label_start_column() {
        let source = "    aaa\n    bbb\n    ccc\n";
        let diag = Diagnostic::error("multi")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 4, 7))
            .with_label(Span::new(FileId::DUMMY, 4, 23), "spans three lines")
            .with_source(source);
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(
            html.contains("<span class='label'>    |</span>"),
            "middle connector must use the label start column:\n{html}"
        );
    }

    #[test]
    fn html_single_line_label_not_suppressed() {
        let source = "let x = 1;\n";
        let diag = Diagnostic::error("single")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_label(Span::new(FileId::DUMMY, 4, 5), "expected int, found unit")
            .with_source(source);
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(
            html.contains("expected int, found unit"),
            "single-line label dropped:\n{html}"
        );
    }

    #[test]
    fn html_eof_span_renders_without_panic() {
        let source = "let x = 1;\n";
        let src_len = source.len() as u32;
        for end in [src_len, src_len + 1] {
            let diag = Diagnostic::error("eof")
                .with_code_str(ErrCode::E030)
                .with_span(Span::new(FileId::DUMMY, src_len, end))
                .with_label(Span::new(FileId::DUMMY, src_len, end), "at eof")
                .with_source(source);
            let html = HtmlRenderer::new().diagnostic_to_html(&diag);
            assert!(html.contains("at eof"), "EOF label dropped: {html}");
        }
    }

    /// Pins the deliberately unfixed limitation: the caret lane is
    /// measured on the **raw** line, while the emitted text replaces a control
    /// byte with the wider U+FFFD.  A future fix (sanitizing the whole row
    /// before the ranges are computed) must show up as a change here, not as a
    /// silent shift.
    #[test]
    fn control_byte_in_a_source_row_shifts_the_caret_lane() {
        // One C0 byte before the labelled token: 1 raw byte becomes 3 bytes of
        // U+FFFD, so the raw-column caret no longer sits under the token.
        let source = "a\x07bc def\n";
        let start = source.find("bc").unwrap() as u32;
        let diag = Diagnostic::error("ctrl")
            .with_span(Span::new(FileId::DUMMY, start, start + 2))
            .with_label(Span::new(FileId::DUMMY, start, start + 2), "here")
            .with_source(source);
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(
            html.contains('\u{FFFD}'),
            "the control byte must not reach the page raw: {html}"
        );
        assert!(html.contains("here"), "the label must still render: {html}");
    }

    #[test]
    fn child_diagnostics_render_with_highlight() {
        let diag = Diagnostic::error("parent").with_highlighted_note(
            crate::diagnostics::StyledString::new(vec![crate::diagnostics::StringPart {
                content: "child".into(),
                style: crate::diagnostics::StringPartStyle::Highlighted,
            }]),
        );
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(html.contains("note: "), "child label missing: {html}");
        assert!(
            html.contains("<span class='highlight'>child</span>"),
            "child highlight missing: {html}"
        );
    }

    #[test]
    fn filename_newlines_are_sanitized() {
        let diag = Diagnostic::error("x")
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_source("x")
            .with_file_name("a\nb.rs");
        let html = HtmlRenderer::new().diagnostic_to_html(&diag);
        assert!(!html.contains("a\nb.rs"), "raw newline in filename: {html}");
    }
}
