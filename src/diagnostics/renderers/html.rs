//! HTML renderer — streams an HTML document instead of buffering it.

use crate::ast::FileId;
use crate::ast::Span;
use crate::diagnostics::emitter::stylesheet::dark_css_cached;

use crate::diagnostics::Diagnostic;
use crate::diagnostics::LineIndex;
use crate::diagnostics::glyph::{
    DEFAULT_TAB_WIDTH, LabelWindow, UnderlineLanes, compute_line_annotation_ranges,
    render_line_merged_with,
};
use crate::diagnostics::label::Label;
use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::renderers::fmt;
use crate::diagnostics::sink::DiagnosticSink;
use crate::smallvec::{SmallVec, smallvec};
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Write as _;
use std::io;

/// HTML renderer — generates diagnostic output as HTML.
/// Inspired by Austral's Reporter.ml which produces interactive call trees.
pub struct HtmlRenderer {
    /// Number of source lines of context to show around a span.
    context_lines: usize,
}

impl HtmlRenderer {
    pub fn new() -> Self {
        HtmlRenderer {
            context_lines: crate::diagnostics::DiagOpts::default().context_lines,
        }
    }

    /// Override the number of source lines of context shown around a span,
    /// mirroring [`GlyphRenderer::with_context_lines`](crate::diagnostics::GlyphRenderer::with_context_lines).
    pub fn with_context_lines(mut self, n: usize) -> Self {
        self.context_lines = n;
        self
    }
}

impl Default for HtmlRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for HtmlRenderer {
    fn begin(&self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        out.write_str(
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n\
             <title>Ponent Compiler Diagnostics</title>\n\
             <style>\n",
        )?;
        out.write_str(dark_css_cached())?;
        out.write_str(
            "\n\
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
             .diag .help-text { color: var(--help-text); }\n\
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
               var t = localStorage.getItem('ponent-theme');\n\
               if (t === 'light' || (!t && matchMedia('(prefers-color-scheme: light)').matches)) {\n\
                 document.body.classList.add('light');\n\
               }\n\
             })();\n\
             </script>\n\
             </head>\n<body>\n\
             <div id='theme-toggle'><button id='theme-toggle-btn'>Toggle theme</button></div>\n\
             <script>\n\
             (function(){\n\
               function toggle() {\n\
                 var light = document.body.classList.toggle('light');\n\
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
             <h1>Ponent Compiler Diagnostics</h1>\n",
        )
    }

    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
        self.diagnostic_to_html_impl(diag, Some(cx), out);
    }

    fn render(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String {
        let mut out = String::new();
        self.render_into(diag, cx, &mut out);
        out
    }

    fn render_summary(&self, errors: usize, warnings: usize) -> String {
        format!(
            "<div class='summary' role='status'>{} error{}, {} warning{}</div>\n",
            errors,
            if errors == 1 { "" } else { "s" },
            warnings,
            if warnings == 1 { "" } else { "s" },
        )
    }

    fn render_summary_full(&self, errors: usize, warnings: usize, files: usize) -> String {
        if files <= 1 {
            return self.render_summary(errors, warnings);
        }
        format!(
            "<div class='summary' role='status'>{} error{}, {} warning{} in {} file{}</div>\n",
            errors,
            if errors == 1 { "" } else { "s" },
            warnings,
            if warnings == 1 { "" } else { "s" },
            files,
            if files == 1 { "" } else { "s" },
        )
    }

    fn end(&self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        out.write_str("</body>\n</html>")
    }

    fn render_suppressed_note(&self, count: usize) -> String {
        format!(
            "<div class='summary' role='status'>{count} additional error{} suppressed</div>\n",
            if count == 1 { "" } else { "s" }
        )
    }
}

impl HtmlRenderer {
    /// Single-shot entry point (tests): builds its own line index.
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
        let level_class = diag.level.as_str();
        // `with_code_str` explicitly supports untrusted input (unknown codes
        // are stored verbatim), so the code string must be escaped like any
        // other external text.
        // Escape the code string once and reuse for both the body code span
        // and the summary `[code]` — `escape` is a full scan, so two calls
        // per diagnostic is pure waste.
        let code_esc: Option<Cow<'_, str>> = diag.code.as_ref().map(|c| self.escape(c.code()));
        let code_str = code_esc
            .as_ref()
            .map(|esc| format!("<span class='code'>{}</span> ", esc))
            .unwrap_or_default();

        html.push_str("<details class='diag ");
        html.push_str(level_class);
        html.push_str("' open role='alert'>\n");

        // Collapsible summary line — `error[E030]: boom` (rustc form); the
        // diagnostic starts expanded (`open`) so streaming output keeps
        // the same visible content, users just gain the collapse control.
        let level_label = diag.level.label();
        let summary_code = code_esc
            .as_ref()
            .map(|esc| format!(" [{}]", esc))
            .unwrap_or_default();
        html.push_str("<summary><span class='code'>");
        html.push_str(&level_label);
        html.push_str("</span>: <span class='message'>");
        html.push_str(&self.escape(&diag.message));
        html.push_str("</span></summary>\n");

        // Main message — `code_str` already carries its own `<span class='code'>`.
        html.push_str("<div>");
        html.push_str(&code_str);
        html.push_str("<span class='message'>");
        html.push_str(&self.escape(&diag.message));
        html.push_str("</span></div>\n");

        // Span location.  When the diagnostic carries a source buffer we can
        // resolve the primary span to `file:line:col` (1-based) instead of the
        // opaque byte range; build the line index up front so both the span
        // header and the source section below share it.
        let source_ctx = diag.source.as_ref();
        let mut idx_opt: Option<LineIndex<'_>> = None;
        if let Some(source) = source_ctx {
            idx_opt = Some(match cx {
                Some(cx) => cx.line_index(source),
                None => LineIndex::new(source),
            });
        }

        if let Some(span) = diag.spans.first() {
            let location = match (&idx_opt, diag.source.as_ref()) {
                (Some(idx), Some(_)) => {
                    let file = diag.file_name();
                    let (line, col) = idx.linecol_display(span.start_as_usize());
                    if file.is_empty() {
                        format!("{}:{}", line + 1, col + 1)
                    } else {
                        format!("{file}:{}:{}", line + 1, col + 1)
                    }
                }
                _ => span.to_string(),
            };
            html.push_str("<div class='span'>at ");
            html.push_str(&self.escape(&location));
            html.push_str("</div>\n");
        }

        // Source context — render directly from Diagnostic struct fields
        if let Some(source) = source_ctx {
            // Multi-file guard: the source buffer belongs to the first
            // primary span's file (emitter contract), so only labels from
            // that file can be located in it.  Labels from other files are
            // rendered below as byte-range notes instead of bogus
            // line/column positions.
            let idx = idx_opt
                .as_ref()
                .expect("idx_opt is Some whenever a source is present");
            // Multi-file guard: the source buffer belongs to the first
            // primary span's file (emitter contract), so only labels from
            // that file can be located in it.  Labels from other files are
            // rendered below as byte-range notes instead of bogus
            // line/column positions.
            let primary_spans = diag.spans.primary();
            let anchor_file = primary_spans.first().map(|sp| sp.file);
            // Borrowed view — one pointer per label, no clone.
            let labels: SmallVec<[&Label<'_>; 4]> = diag
                .labels
                .iter()
                .filter(|l| anchor_file.is_none_or(|anchor| l.span.file == anchor))
                .collect();
            let pre_start = html.len();
            html.push_str("<pre class='source-line'>");
            if self.render_source_html_into(
                source,
                &primary_spans,
                &labels,
                idx,
                diag.file_name(),
                html,
            ) {
                html.push_str("</pre>\n");
            } else {
                html.truncate(pre_start);
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
                // O(F·(S+L)) into O(S+L+F).
                let mut foreign_spans_by_file: SmallVec<[(FileId, SmallVec<[Span; 2]>); 2]> =
                    SmallVec::new();
                for &sp in &primary_spans {
                    if Some(sp.file) != anchor_file {
                        if let Some(slot) = foreign_spans_by_file
                            .iter_mut()
                            .find(|(f, _)| *f == sp.file)
                        {
                            slot.1.push(sp);
                        } else {
                            foreign_spans_by_file.push((sp.file, smallvec![sp]));
                        }
                    }
                }
                let mut foreign_labels_by_file: SmallVec<[(FileId, SmallVec<[&Label<'_>; 4]>); 2]> =
                    SmallVec::new();
                for lbl in &diag.labels {
                    if Some(lbl.span.file) != anchor_file {
                        if let Some(slot) = foreign_labels_by_file
                            .iter_mut()
                            .find(|(f, _)| *f == lbl.span.file)
                        {
                            slot.1.push(lbl);
                        } else {
                            foreign_labels_by_file.push((lbl.span.file, smallvec![lbl]));
                        }
                    }
                }
                for (fid, fname, fsource) in foreign.iter() {
                    if Some(*fid) == anchor_file {
                        continue;
                    }
                    let f_primaries: &[Span] = foreign_spans_by_file
                        .iter()
                        .find(|(f, _)| *f == *fid)
                        .map(|(_, v)| v.as_slice())
                        .unwrap_or(&[]);
                    let f_labels: &[&Label<'_>] = foreign_labels_by_file
                        .iter()
                        .find(|(f, _)| *f == *fid)
                        .map(|(_, v)| v.as_slice())
                        .unwrap_or(&[]);
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
                        html,
                    ) {
                        html.push_str("</pre>\n");
                    } else {
                        html.truncate(pre_start);
                    }
                    foreign_rendered_files.push(*fid);
                }
            }
            for lbl in diag
                .labels
                .iter()
                .filter(|l| anchor_file.is_some_and(|anchor| l.span.file != anchor))
            {
                if lbl.message.is_empty() {
                    continue;
                }
                if foreign_rendered_files.contains(&lbl.span.file) {
                    continue;
                }
                html.push_str("<span class='label'>= note: ");
                html.push_str(&self.escape(&lbl.message));
                html.push_str(" at ");
                html.push_str(&self.escape(&lbl.span.to_string()));
                html.push_str("</span>\n");
            }
        }

        // Related errors (Zig-style numbered context, matching the human
        // renderer's `1.` / `2.` header numbering).
        if !diag.related_errors.is_empty() {
            html.push_str("<ul class='related'>\n");
            for rel in &diag.related_errors {
                let code_part = rel
                    .code
                    .as_ref()
                    .map(|c| format!("<span class='code'>{}</span> ", self.escape(c.code())))
                    .unwrap_or_default();
                html.push_str("<li class='related-entry'>");
                html.push_str(&code_part);
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

        // Suggestions
        for suggestion in &diag.suggestions {
            html.push_str("<div class='suggestion'>suggestion: ");
            html.push_str(&self.escape(&suggestion.message));
            html.push_str("</div>\n");
        }

        // Help
        if let Some(help) = &diag.help {
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
            let idx = idx_opt.as_ref();
            html.push_str("<div class='chain'>referenced by:\n");
            for entry in chain.entries() {
                html.push_str("<div class='entry'><span class='label'>");
                html.push_str(&self.escape(&entry.message));
                html.push_str("</span>: ");
                html.push_str(&entry.kind.as_str());
                if let (Some(idx), Some(span)) = (idx, entry.span) {
                    let (line, col) = idx.linecol_display(span.start_as_usize());
                    html.push_str(" at ");
                    if !file_name.is_empty() {
                        html.push_str(&self.escape(file_name));
                        html.push(':');
                    }
                    fmt::push_uint(html, (line + 1) as u64);
                    html.push(':');
                    fmt::push_uint(html, (col + 1) as u64);
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
                html.push_str(&self.escape(code.code()));
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
    fn render_source_html_into<'src>(
        &self,
        source: &str,
        primary_spans: &[Span],
        labels: &[&Label<'src>],
        idx: &LineIndex<'_>,
        file_name: &str,
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

        // §8: the caller supplies the line index (built once per source, not
        // per diagnostic); every conversion below is O(log n).
        let starts = idx.starts();

        // Character column, not `to_linecol`'s byte column: the start
        // position is printed as `file:line:col` (the `data-location`
        // click-to-copy target), which editors count in characters.
        let (start_line, start_col) = idx.linecol_display(span.start_as_usize());
        let end_pos = idx.to_linecol(span.end_as_usize());

        let context_lines = self.context_lines;
        let first_line = start_line.saturating_sub(context_lines);
        let text_last_line = idx.text_line_count().saturating_sub(1);

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

        // Window covers every primary span, not just the first.
        let mut last_line = std::cmp::min(end_pos.line + 1 + context_lines, text_last_line);
        for &sp in primary_spans {
            let e = idx.to_linecol(sp.end_as_usize());
            last_line = last_line.max(std::cmp::min(e.line + 1 + context_lines, text_last_line));
        }
        let line_num_width = (last_line + 1).checked_ilog10().unwrap_or(0) as usize + 1;

        // §4: per-label (start_line, end_line), converted once.
        // `SmallVec<[(usize, usize); 4]>`: labels-per-diagnostic rarely exceeds
        // 4, so the common case stays inline — one fewer heap allocation per
        // source section.
        let label_line_ranges: SmallVec<[(usize, usize); 4]> =
            all_labels.iter().map(|l| idx.span_lines(l.span)).collect();
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

        let html = out;
        // Hoisted out of the row loop to avoid per-iteration allocation.
        let mut rendered = String::with_capacity(256);
        // Underline lane allocator — owns the interval-graph colouring scratch
        // so the row loop reuses it across lines.
        let mut lanes = UnderlineLanes::new();
        let mut row_buf = String::new();
        let spaces = " ".repeat(line_num_width + 1);
        // Scratch buffers — reused each iteration; `clear()` preserves capacity.
        let cap = all_labels.len() + 1;
        let mut underlines_buf: SmallVec<[(usize, usize, char, &str, Option<usize>); 8]> =
            SmallVec::with_capacity(cap);
        let mut ranges_buf: Vec<(usize, usize)> = Vec::with_capacity(cap);
        let mut meta_buf: SmallVec<[(char, &str, Option<usize>); 8]> = SmallVec::with_capacity(cap);
        let mut display_buf: Vec<(usize, usize)> = Vec::with_capacity(cap);
        let mut order_buf: Vec<usize> = Vec::with_capacity(cap);
        let mut done_buf: Vec<bool> = Vec::with_capacity(cap);
        let mut active_buf: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
        // Sliding window over `all_labels`, ordered by start line once here;
        // each row then only visits the labels that can cover it, not all of
        // them.  No pre-sorted prefix: the HTML renderer draws one source
        // section per diagnostic, so there is no K-section loop to amortise
        // over.
        let mut label_window = LabelWindow::new(&label_line_ranges, &[]);

        // The filename is constant across every source row, so escape it once
        // here and reuse the (borrowed, allocation-free) result per row instead
        // of `format!`-ing a fresh `file:line:col` string only to escape and
        // discard it on each iteration.
        let escaped_file = self.escape(file_name);

        for line_idx in first_line..=last_line {
            // O(1) slice via the shared index instead of indexing a
            // whole-file `Vec<&str>` that was collected only for this.
            let line = idx.line(line_idx);
            compute_line_annotation_ranges(
                span,
                &all_labels,
                &mut label_window,
                source,
                line,
                line_idx,
                starts,
                &mut ranges_buf,
                &mut meta_buf,
            );
            // One fused grapheme walk resolves this row's byte ranges to
            // display columns, measures the line width, and emits the
            // HTML-escaped, highlight-wrapped source — the same core the glyph
            // renderer uses, with `push_html_escaped` as its emit callback.
            let display_width = render_line_merged_with(
                line,
                &ranges_buf,
                &meta_buf,
                &mut underlines_buf,
                &mut rendered,
                "<span class='highlight'>",
                "</span>",
                &mut display_buf,
                &mut order_buf,
                &mut done_buf,
                &mut active_buf,
                DEFAULT_TAB_WIDTH,
                push_html_escaped,
            );
            let underlines = &underlines_buf;

            // ── Line number ─────────────────────────────────────
            // write! straight into the output buffer (reuses its
            // capacity) instead of two per-row format! allocations.
            // `data-location` carries the click-to-copy target
            // (item 22): the primary span's line gets the full
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
                    html.push_str(&escaped_file);
                    html.push(':');
                    fmt::push_uint(html, (line_idx + 1) as u64);
                    html.push(':');
                    fmt::push_uint(html, (start_col + 1) as u64);
                }
            } else if file_name.is_empty() {
                fmt::push_uint(html, (line_idx + 1) as u64);
            } else {
                html.push_str(&escaped_file);
                html.push(':');
                fmt::push_uint(html, (line_idx + 1) as u64);
            }
            html.push_str("' aria-label='Line ");
            fmt::push_uint(html, (line_idx + 1) as u64);
            html.push_str("'>");
            fmt::push_uint_padded(html, (line_idx + 1) as u64, line_num_width);
            html.push_str(" │ </span>");

            // ── Source line (rendered by the fused walk above) ──
            html.push_str(&rendered);
            html.push('\n');

            // ── Underline annotations (one row per overlap depth) ───
            if !underlines.is_empty() {
                let lane_count = lanes.assign(underlines);
                for lane_idx in 0..lane_count {
                    html.push_str("<span class='line-num'>");
                    html.push_str(&spaces);
                    html.push_str(" │ </span>");
                    // Plain (uncoloured) row for HTML — per-char colouring
                    // is the terminal renderer's job; here the CSS class
                    // `.underline` carries the visual weight.
                    let visible = lanes.row(
                        underlines,
                        lane_idx,
                        display_width,
                        &mut row_buf,
                        |_ch| "",
                        "",
                    );
                    if visible {
                        html.push_str("<span class='underline'>");
                        html.push_str(&self.escape(&row_buf));
                        html.push_str("</span>");
                    }
                    html.push('\n');
                }

                // ── Label messages ──────────────────────────────
                for (col, _ulen, _ch, msg, label_idx) in underlines {
                    if msg.is_empty() {
                        continue;
                    }
                    // §4: suppress the message on every line except the label's
                    // final covered line — mirrors `GlyphRenderer`'s
                    // `is_multiline_nonfinal`. Keyed by label index so two
                    // labels sharing one message stay distinct.
                    let is_multiline_nonfinal =
                        label_idx.is_some_and(|i| label_line_ranges[i].1 != line_idx);
                    if is_multiline_nonfinal {
                        continue;
                    }
                    if let Some(i) = label_idx {
                        if *i < 64 {
                            label_bits |= 1u64 << *i;
                        } else {
                            label_extra[*i - 64] = true;
                        }
                    }
                    html.push_str("<span class='line-num'>");
                    html.push_str(&spaces);
                    html.push_str(" │ </span>");
                    // `col` is a display column → pad with that many spaces.
                    // Write the connector directly into `html` — `escape`'s
                    // fast path (no allocation for space/`|`) made the
                    // intermediate `String` the only real cost here.
                    html.push_str("<span class='label'>");
                    fmt::push_spaces(html, *col);
                    html.push('|');
                    html.push(' ');
                    html.push_str(&self.escape(msg));
                    html.push_str("</span>\n");
                }
            }
        }

        // A label whose lines fall outside the rendered rows still owns
        // user-visible text (e.g. a span at EOF lands on the phantom line).
        // Emit it as a note, mirroring GlyphRenderer's fallback, so no
        // label message is silently dropped from the HTML view.  Keyed off
        // what was actually written, not the line range: a label can lie
        // inside the range yet still miss every drawn row (e.g. an EOF
        // point) and must fall back too.
        for (i, lbl) in all_labels.iter().enumerate() {
            let rendered = if i < 64 {
                (label_bits >> i) & 1 != 0
            } else {
                label_extra[i - 64]
            };
            if lbl.message.is_empty() || rendered {
                continue;
            }
            let (end_line, end_col) = idx.linecol_display(lbl.span.end_as_usize());
            html.push_str("<span class='label'>= note: ");
            html.push_str(&self.escape(&lbl.message));
            html.push_str(" at ");
            fmt::push_uint(html, (end_line + 1) as u64);
            html.push(':');
            fmt::push_uint(html, (end_col + 1) as u64);
            html.push_str("</span>\n");
        }

        true
    }

    /// Escape text for HTML element content *and* single/double-quoted
    /// attribute values.  `'` is escaped in addition to the classic four so
    /// this helper stays safe if it is ever used inside a `class='…'`
    /// attribute.
    fn escape<'a>(&self, s: &'a str) -> Cow<'a, str> {
        if next_html_escape_byte(s.as_bytes()).is_none() {
            return Cow::Borrowed(s);
        }
        let mut out = String::with_capacity(s.len() + s.len() / 4);
        let bytes = s.as_bytes();
        let mut start = 0;
        while let Some(offset) = next_html_escape_byte(&bytes[start..]) {
            let idx = start + offset;
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
        }
        out.push_str(&s[start..]);
        Cow::Owned(out)
    }
}

/// Escape a chunk of source text for HTML element content, appending it to
/// `out`.  This is the emit callback `render_line_merged_with` drives, so it
/// runs once per same-highlight run (or per grapheme) rather than once per
/// byte.  The escape set matches the source-row loop it replaced (`&`, `<`,
/// `>`, `"`); the all-clean fast path avoids the char walk entirely.
fn push_html_escaped(out: &mut String, s: &str) {
    let bytes = s.as_bytes();
    let mut start = 0;
    while let Some(offset) = next_html_source_escape_byte(&bytes[start..]) {
        let idx = start + offset;
        out.push_str(&s[start..idx]);
        match bytes[idx] {
            b'&' => out.push_str("&amp;"),
            b'<' => out.push_str("&lt;"),
            b'>' => out.push_str("&gt;"),
            b'"' => out.push_str("&quot;"),
            _ => unreachable!(),
        }
        start = idx + 1;
    }
    out.push_str(&s[start..]);
}

/// Next byte in `hay` that is one of `&`, `<`, `>`, `"`, `'` (full HTML
/// escape set).  SIMD-accelerated via `memchr3` + `memchr2`.
#[inline]
fn next_html_escape_byte(hay: &[u8]) -> Option<usize> {
    match (
        memchr::memchr3(b'&', b'<', b'>', hay),
        memchr::memchr2(b'"', b'\'', hay),
    ) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Next byte in `hay` that is one of `&`, `<`, `>`, `"` (source-content
/// escape set, no apostrophe).
#[inline]
fn next_html_source_escape_byte(hay: &[u8]) -> Option<usize> {
    match (
        memchr::memchr3(b'&', b'<', b'>', hay),
        memchr::memchr(b'"', hay),
    ) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::diagnostics::Diagnostic;
    use crate::diagnostics::ErrCode;

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
}
