//! HTML renderer — streams an HTML document instead of buffering it.

use crate::ast::FileId;
use crate::ast::Span;
use crate::diagnostics::emitter::stylesheet::Stylesheet;

use crate::diagnostics::Diagnostic;
use crate::diagnostics::LineIndex;
use crate::diagnostics::glyph::{
    DEFAULT_TAB_WIDTH, LabelWindow, UnderlineLanes, compute_line_annotation_ranges,
    render_line_merged_with,
};
use crate::diagnostics::label::Label;
use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::sink::DiagnosticSink;
use crate::smallvec::SmallVec;
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
    fn begin(&mut self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        out.write_str(
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n\
             <title>Ponent Compiler Diagnostics</title>\n\
             <style>\n",
        )?;
        out.write_str(&Stylesheet::dark().to_html_css())?;
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
             \"  var t = localStorage.getItem('ponent-theme');\n\
             \"  if (t === 'light' || (!t && matchMedia('(prefers-color-scheme: light)').matches)) {\n\
             \"    document.body.classList.add('light');\n\
             \"  }\n\
             \"})();\n\
             </script>\n\
             </head>\n<body>\n\
             <div id='theme-toggle'><button id='theme-toggle-btn'>Toggle theme</button></div>\n\
             <script>\n\
             (function(){\n\
             \"  function toggle() {\n\
             \"    var light = document.body.classList.toggle('light');\n\
             \"    localStorage.setItem('ponent-theme', light ? 'light' : 'dark');\n\
             \"  }\n\
             \"  document.getElementById('theme-toggle-btn').addEventListener('click', toggle);\n\
             \"  document.addEventListener('click', function(e) {\n\
             \"    var n = e.target.closest('.line-num');\n\
             \"    if (n && n.dataset.location) {\n\
             \"      navigator.clipboard.writeText(n.dataset.location).catch(function(){});\n\
             \"    }\n\
             \"  });\n\
             \"})();\n\
             </script>\n\
             <h1>Ponent Compiler Diagnostics</h1>\n",
        )
    }

    fn render(&mut self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String {
        self.diagnostic_to_html_impl(diag, Some(cx))
    }

    fn render_summary(&mut self, errors: usize, warnings: usize) -> String {
        format!(
            "<div class='summary' role='status'>{} error{}, {} warning{}</div>\n",
            errors,
            if errors == 1 { "" } else { "s" },
            warnings,
            if warnings == 1 { "" } else { "s" },
        )
    }

    fn render_summary_full(&mut self, errors: usize, warnings: usize, files: usize) -> String {
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

    fn end(&mut self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        out.write_str("</body>\n</html>")
    }

    fn render_suppressed_note(&mut self, count: usize) -> String {
        format!(
            "<div class='summary' role='status'>{count} additional error{} suppressed</div>\n",
            if count == 1 { "" } else { "s" }
        )
    }
}

impl HtmlRenderer {
    /// Single-shot entry point (tests): builds its own line index.
    fn diagnostic_to_html(&self, diag: &Diagnostic<'_>) -> String {
        self.diagnostic_to_html_impl(diag, None)
    }

    /// Shared body.  `cx` supplies the per-batch line index when rendering
    /// through the emitter, so the file is not rescanned per diagnostic.
    fn diagnostic_to_html_impl(&self, diag: &Diagnostic<'_>, cx: Option<&RenderCtx<'_>>) -> String {
        // Output is context-window-sized, not file-sized; cap the hint so a
        // large source does not over-reserve.
        let hint = diag
            .source
            .as_ref()
            .map_or(512, |s| (s.len() / 4).clamp(512, 16 * 1024));
        let mut html = String::with_capacity(hint);
        let level_class = diag.level.as_str();
        // `with_code_str` explicitly supports untrusted input (unknown codes
        // are stored verbatim), so the code string must be escaped like any
        // other external text.
        let code_str = diag
            .code
            .as_ref()
            .map(|c| format!("<span class='code'>{}</span> ", self.escape(c.code())))
            .unwrap_or_default();

        let _ = write!(html, "<details class='diag {}' open role='alert'>\n", level_class);

        // Collapsible summary line — `error[E030]: boom` (rustc form); the
        // diagnostic starts expanded (`open`) so streaming output keeps
        // the same visible content, users just gain the collapse control.
        let level_label = diag.level.label();
        let summary_code = diag
            .code
            .as_ref()
            .map(|c| format!(" [{}]", self.escape(c.code())))
            .unwrap_or_default();
        let _ = write!(
            html,
            "<summary><span class='code'>{}</span>: <span class='message'>{}</span></summary>\n",
            level_label,
            self.escape(&diag.message),
        );

        // Main message — `code_str` already carries its own `<span class='code'>`.
        let _ = write!(
            html,
            "<div>{}<span class='message'>{}</span></div>\n",
            code_str,
            self.escape(&diag.message),
        );

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
                    let pos = idx.to_linecol(span.start_as_usize());
                    if file.is_empty() {
                        format!("{}:{}", pos.line + 1, pos.col + 1)
                    } else {
                        format!("{file}:{}:{}", pos.line + 1, pos.col + 1)
                    }
                }
                _ => span.to_string(),
            };
            let _ = write!(
                html,
                "<div class='span'>at {}</div>\n",
                self.escape(&location)
            );
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
            let ctx =
                self.render_source_html(source, primary_spans, &labels, idx, diag.file_name());
            if !ctx.is_empty() {
                let _ = write!(html, "<pre class='source-line'>{}</pre>\n", ctx);
            }
            // Labels filtered out of the source section (from other files)
            // stay visible as range notes; their line/col would be
            // meaningless against this buffer.
            for lbl in diag
                .labels
                .iter()
                .filter(|l| anchor_file.is_some_and(|anchor| l.span.file != anchor))
            {
                if lbl.message.is_empty() {
                    continue;
                }
                let _ = write!(
                    html,
                    "<span class='label'>= note: {} at {}</span>\n",
                    self.escape(&lbl.message),
                    self.escape(&lbl.span.to_string()),
                );
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
                let _ = write!(
                    html,
                    "<li class='related-entry'>{0}<span class='message'>{1}</span>",
                    code_part,
                    self.escape(&rel.message),
                );
                if let Some(sp) = &rel.span {
                    let _ = write!(
                        html,
                        " <span class='span'>at {}</span>",
                        self.escape(&sp.to_string()),
                    );
                }
                html.push_str("</li>\n");
            }
            html.push_str("</ul>\n");
        }

        // Suggestions
        for suggestion in &diag.suggestions {
            let _ = write!(
                html,
                "<div class='suggestion'>suggestion: {}</div>\n",
                self.escape(&suggestion.message),
            );
        }

        // Help
        if let Some(help) = &diag.help {
            let _ = write!(
                html,
                "<div class='help-text'>help: {}</div>\n",
                self.escape(help),
            );
        }

        // Call chain (Zig-style) — entries carry kind + user message, plus
        // file:line:col when a source is available and the entry has a span
        // (mirrors the JSON and terminal renderers).
        if let Some(ref chain) = diag.call_chain
            && !chain.is_empty()
        {
            let source = diag.source.as_ref();
            let file_name = diag.file_name();
            let idx = source.map(|s| LineIndex::new(s));
            html.push_str("<div class='chain'>referenced by:\n");
            for entry in chain.entries() {
                let location = match (idx.as_ref(), entry.span) {
                    (Some(idx), Some(span)) => {
                        let pos = idx.to_linecol(span.start_as_usize());
                        if file_name.is_empty() {
                            format!("{}:{}", pos.line + 1, pos.col + 1)
                        } else {
                            format!("{}:{}:{}", file_name, pos.line + 1, pos.col + 1)
                        }
                    }
                    _ => String::new(),
                };
                if location.is_empty() {
                    let _ = write!(
                        html,
                        "<div class='entry'><span class='label'>{}</span>: {}</div>\n",
                        self.escape(&entry.message),
                        entry.kind.as_str(),
                    );
                } else {
                    let _ = write!(
                        html,
                        "<div class='entry'><span class='label'>{}</span>: {} at {}</div>\n",
                        self.escape(&entry.message),
                        entry.kind.as_str(),
                        self.escape(&location),
                    );
                }
            }
            html.push_str("</div>\n");
        }

        // Explain section (if error code present)
        if let Some(ref code) = diag.code {
            let explain = code.explain();
            if !explain.is_empty() {
                let _ = write!(
                    html,
                    "<details class='explain'>\n\
                     <summary>Explain {}</summary>\n\
                     <pre>{}</pre>\n\
                     </details>\n",
                    self.escape(code.code()),
                    self.escape(explain),
                );
            }
        }

        html.push_str("</details>\n");
        html
    }

    /// Render the source context as HTML, directly from the Diagnostic's
    /// source text, spans, and labels — without going through GlyphRenderer's
    /// terminal-formatted output.
    fn render_source_html<'a, 'src>(
        &self,
        source: &str,
        primary_spans: &[Span],
        labels: &[&'a Label<'src>],
        idx: &LineIndex<'_>,
        file_name: &str,
    ) -> String {
        let Some(&span) = primary_spans.first() else {
            return String::new();
        };
        // Empty source has no rows to draw. `text_line_count()` reports 1
        // for `""` (str::lines-style), so guard on the source itself rather
        // than materializing every line just to test for emptiness.
        if source.is_empty() {
            return String::new();
        }

        // §8: the caller supplies the line index (built once per source, not
        // per diagnostic); every conversion below is O(log n).
        let starts = idx.starts();

        let start_pos = idx.to_linecol(span.start_as_usize());
        let end_pos = idx.to_linecol(span.end_as_usize());

        let context_lines = self.context_lines;
        let first_line = start_pos.line.saturating_sub(context_lines);
        let last_line = std::cmp::min(
            end_pos.line + 1 + context_lines,
            idx.text_line_count().saturating_sub(1),
        );
        let line_num_width = (last_line + 1).checked_ilog10().unwrap_or(0) as usize + 1;

        // §4: per-label (start_line, end_line), converted once.
        let label_line_ranges: Vec<(usize, usize)> =
            labels.iter().map(|l| idx.span_lines(l.span)).collect();
        // Per-label "message emitted inline" flags: the fallback below must
        // not re-emit a message that a source row already rendered, but a
        // line-range test is not enough — a label point on the phantom line
        // after a trailing newline lies inside the range yet never enters
        // `underlines`.
        let mut label_rendered = vec![false; labels.len()];

        let mut html = String::with_capacity(256);
        // Hoisted out of the row loop to avoid per-iteration allocation.
        let mut rendered = String::with_capacity(256);
        // Underline lane allocator — owns the interval-graph colouring scratch
        // so the row loop reuses it across lines.
        let mut lanes = UnderlineLanes::new();
        let spaces = " ".repeat(line_num_width + 1);
        // Scratch buffers — reused each iteration; `clear()` preserves capacity.
        // The primary span is passed straight to
        // `compute_line_annotation_ranges` (no synthetic `Label`/combined
        // `Vec` materialized per call).
        let mut underlines_buf: Vec<(usize, usize, char, &'a str, Option<usize>)> =
            Vec::with_capacity(labels.len() + 1);
        let mut ranges_buf: Vec<(usize, usize)> = Vec::with_capacity(labels.len() + 1);
        let mut meta_buf: Vec<(char, &'a str, Option<usize>)> =
            Vec::with_capacity(labels.len() + 1);
        let mut display_buf: Vec<(usize, usize)> = Vec::with_capacity(labels.len() + 1);
        let mut order_buf: Vec<usize> = Vec::with_capacity(labels.len() + 1);
        let mut done_buf: Vec<bool> = Vec::with_capacity(labels.len() + 1);
        let mut active_buf: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
        // Sliding window over `labels`, ordered by start line once here; each
        // row then only visits the labels that can cover it, not all of them.
        let mut label_window = LabelWindow::new(&label_line_ranges);

        for line_idx in first_line..=last_line {
            // O(1) slice via the shared index instead of indexing a
            // whole-file `Vec<&str>` that was collected only for this.
            let line = idx.line(line_idx);
            compute_line_annotation_ranges(
                span,
                labels,
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
            // diagnostic has no filename, e.g. the `<input>`
            // pseudo-file).
            let location = if line_idx == start_pos.line {
                if file_name.is_empty() {
                    format!("{}:{}", line_idx + 1, start_pos.col + 1)
                } else {
                    format!("{file_name}:{}:{}", line_idx + 1, start_pos.col + 1)
                }
            } else if file_name.is_empty() {
                format!("{}", line_idx + 1)
            } else {
                format!("{file_name}:{}", line_idx + 1)
            };
            let _ = write!(
                html,
                "<span class='line-num' data-location='{}' aria-label='Line {}'>{:>width$} │ </span>",
                self.escape(&location),
                line_idx + 1,
                line_idx + 1,
                width = line_num_width
            );

            // ── Source line (rendered by the fused walk above) ──
            html.push_str(&rendered);
            html.push('\n');

            // ── Underline annotations (one row per overlap depth) ───
            if !underlines.is_empty() {
                let lane_count = lanes.assign(underlines);
                let mut row_buf = String::new();
                for lane_idx in 0..lane_count {
                    let _ = write!(html, "<span class='line-num'>{spaces} │ </span>");
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
                        let _ = write!(
                            html,
                            "<span class='underline'>{}</span>",
                            self.escape(&row_buf)
                        );
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
                        label_rendered[*i] = true;
                    }
                    let _ = write!(html, "<span class='line-num'>{spaces} │ </span>");
                    // `col` is a display column → pad with that many spaces.
                    // Write the connector directly into `html` — `escape`'s
                    // fast path (no allocation for space/`|`) made the
                    // intermediate `String` the only real cost here.
                    html.push_str("<span class='label'>");
                    for _ in 0..*col {
                        html.push(' ');
                    }
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
        for (i, lbl) in labels.iter().enumerate() {
            if lbl.message.is_empty() || label_rendered[i] {
                continue;
            }
            let end_pos = idx.to_linecol(lbl.span.end_as_usize());
            let _ = write!(
                html,
                "<span class='label'>= note: {} at {}:{}</span>\n",
                self.escape(&lbl.message),
                end_pos.line + 1,
                end_pos.col + 1,
            );
        }

        html
    }

    /// Escape text for HTML element content *and* single/double-quoted
    /// attribute values.  `'` is escaped in addition to the classic four so
    /// this helper stays safe if it is ever used inside a `class='…'`
    /// attribute.
    fn escape<'a>(&self, s: &'a str) -> Cow<'a, str> {
        if !s
            .bytes()
            .any(|b| matches!(b, b'&' | b'<' | b'>' | b'"' | b'\''))
        {
            return Cow::Borrowed(s);
        }
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '&' => out.push_str("&amp;"),
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                '"' => out.push_str("&quot;"),
                '\'' => out.push_str("&#39;"),
                _ => out.push(c),
            }
        }
        Cow::Owned(out)
    }
}

/// Escape a chunk of source text for HTML element content, appending it to
/// `out`.  This is the emit callback `render_line_merged_with` drives, so it
/// runs once per same-highlight run (or per grapheme) rather than once per
/// byte.  The escape set matches the source-row loop it replaced (`&`, `<`,
/// `>`, `"`); the all-clean fast path avoids the char walk entirely.
fn push_html_escaped(out: &mut String, s: &str) {
    if !s.bytes().any(|b| matches!(b, b'&' | b'<' | b'>' | b'"')) {
        out.push_str(s);
        return;
    }
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::diagnostics::Diagnostic;

    #[test]
    fn test_code_not_double_wrapped() {
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
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
        assert!(html.contains("<summary>"), "missing collapsible summary: {html}");
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
            .with_code_str("E030")
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
            .with_code_str("E030")
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
            .with_code_str("E030")
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

        let diag = Diagnostic::error("boom").with_code_str("E030");
        let mut emitter = Emitter::new(HtmlRenderer::new(), string(), DiagOpts::default());
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut [diag], &mut sink).unwrap();
        let out = sink.into_string();
        assert!(out.starts_with("<!DOCTYPE html>"), "out: {out}");
        assert!(out.trim_end().ends_with("</html>"), "out: {out}");
        assert!(out.contains("<details class='diag error' open role='alert'>"), "out: {out}");
        assert!(out.contains("class='summary'"), "out: {out}");
    }

    #[test]
    fn max_errors_truncates_and_notes_suppression() {
        use crate::diagnostics::DiagOpts;
        use crate::diagnostics::emitter::Emitter;
        use crate::diagnostics::sink::{StringSink, string};

        let mut diags: Vec<_> = (0..5)
            .map(|i| Diagnostic::error(format!("e{i}")).with_code_str("E030"))
            .collect();
        let opts = DiagOpts {
            max_errors: Some(2),
            ..Default::default()
        };
        let mut emitter = Emitter::new(HtmlRenderer::new(), string(), opts);
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut diags, &mut sink).unwrap();
        let out = sink.into_string();
        assert_eq!(out.matches("<details class='diag error' open role='alert'>").count(), 2, "{out}");
        assert!(out.contains("<div class='summary' role='status'>3 additional errors suppressed</div>"), "{out}");
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
            .with_code_str("E030")
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
            .with_code_str("E030")
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
                .with_code_str("E030")
                .with_span(Span::new(FileId::DUMMY, src_len, end))
                .with_label(Span::new(FileId::DUMMY, src_len, end), "at eof")
                .with_source(source);
            let html = HtmlRenderer::new().diagnostic_to_html(&diag);
            assert!(html.contains("at eof"), "EOF label dropped: {html}");
        }
    }
}
