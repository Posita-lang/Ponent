use crate::ast::Span;
use crate::diagnostics::glyph::{
    byte_to_linecol, compute_line_underlines,
};
use crate::diagnostics::label::{AnnotationKind, Label};
use crate::diagnostics::Diagnostic;
use std::fmt::Write;

/// HTML emitter — generates diagnostic output as HTML.
/// Inspired by Austral's Reporter.ml which produces interactive call trees.
pub struct HtmlEmitter;

impl HtmlEmitter {
    pub fn new() -> Self {
        HtmlEmitter
    }
}

impl Default for HtmlEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl super::DiagnosticEmitter for HtmlEmitter {
    fn emit(&mut self, diag: &Diagnostic) {
        let html = self.diagnostic_to_html(diag);
        eprintln!("{}", html);
    }

    fn emit_all(&mut self, diags: &[Diagnostic]) {
        let mut html = String::from(
            "<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n\
             <title>Ponent Compiler Diagnostics</title>\n\
             <style>\n\
             body { font-family: 'DejaVu Sans Mono', monospace; background: #1e1e2e; color: #cdd6f4; padding: 20px; }\n\
             .diag { margin: 12px 0; padding: 12px; border-radius: 6px; }\n\
             .diag.error { border-left: 4px solid #f38ba8; background: #2a1e1e; }\n\
             .diag.warning { border-left: 4px solid #f9e2af; background: #2a2a1e; }\n\
             .diag.help { border-left: 4px solid #94e2d5; background: #1e2a2a; }\n\
             .diag.code { color: #89b4fa; font-weight: bold; }\n\
             .diag.message { color: #cdd6f4; }\n\
             .diag .span { color: #6c7086; }\n\
             .diag .suggestion { color: #a6e3a1; }\n\
             .diag .help-text { color: #94e2d5; }\n\
             .chain { margin-left: 20px; color: #6c7086; font-size: 0.9em; }\n\
             .chain .entry { margin: 2px 0; }\n\
             .chain .entry .label { color: #89b4fa; }\n\
             .summary { margin-top: 16px; padding: 8px; border-top: 1px solid #45475a; color: #6c7086; }\n\
             details { margin: 4px 0; }\n\
             summary { cursor: pointer; color: #89b4fa; }\n\
             .source-line { white-space: pre; font-family: monospace; color: #a6adc8; }\n\
             .source-line .line-num { color: #6c7086; user-select: none; }\n\
             .source-line .highlight { background: #f38ba833; border-radius: 2px; }\n\
             .source-line .underline { color: #f38ba8; }\n\
             .explain { background: #181825; padding: 12px; border-radius: 6px; margin-top: 8px; }\n\
             .explain h4 { color: #89b4fa; margin: 0 0 8px 0; }\n\
             .explain pre { white-space: pre-wrap; color: #bac2de; }\n\
             </style>\n</head>\n<body>\n<h1>Ponent Compiler Diagnostics</h1>\n",
        );

        for diag in diags {
            let _ = write!(html, "{}", self.diagnostic_to_html(diag));
        }

        let error_count = diags.iter().filter(|d| d.is_error()).count();
        let warning_count = diags.iter().filter(|d| d.is_warning()).count();
        let _ = write!(
            html,
            "<div class='summary'>{} error{}, {} warning{}</div>\n",
            error_count,
            if error_count == 1 { "" } else { "s" },
            warning_count,
            if warning_count == 1 { "" } else { "s" },
        );

        html.push_str("</body>\n</html>");
        eprintln!("{}", html);
    }
}

impl HtmlEmitter {
    fn diagnostic_to_html(&self, diag: &Diagnostic) -> String {
        let mut html = String::new();
        let level_class = diag.level.as_str();
        let code_str = diag
            .code
            .as_ref()
            .map(|c| format!("<span class='code'>{}</span> ", c.code()))
            .unwrap_or_default();

        html.push_str(&format!("<div class='diag {}'>\n", level_class));

        // Main message
        html.push_str(&format!(
            "<div><span class='code'>{}</span><span class='message'>{}</span></div>\n",
            code_str,
            self.escape(&diag.message),
        ));

        // Span location
        if let Some(span) = diag.spans.first() {
            html.push_str(&format!("<div class='span'>at {}</div>\n", span));
        }

        // Source context — render directly from Diagnostic struct fields
        if let Some(source) = diag.source.as_ref() {
            let ctx = self.render_source_html(source, &diag.spans.primary, &diag.labels);
            if !ctx.is_empty() {
                html.push_str(&format!("<pre class='source-line'>{}</pre>\n", ctx));
            }
        }

        // Suggestions
        for suggestion in &diag.suggestions {
            html.push_str(&format!(
                "<div class='suggestion'>suggestion: {}</div>\n",
                self.escape(suggestion),
            ));
        }

        // Help
        if let Some(help) = &diag.help {
            html.push_str(&format!(
                "<div class='help-text'>help: {}</div>\n",
                self.escape(help),
            ));
        }

        // Call chain (Zig-style)
        if let Some(ref chain) = diag.call_chain {
            if !chain.is_empty() {
                html.push_str("<div class='chain'>referenced by:\n");
                for entry in chain.entries() {
                    html.push_str(&format!(
                        "<div class='entry'><span class='label'>{}</span>: {}</div>\n",
                        self.escape(&entry.message),
                        entry.kind.as_str(),
                    ));
                }
                html.push_str("</div>\n");
            }
        }

        // Explain section (if error code present)
        if let Some(ref code) = diag.code {
            let explain = code.explain();
            if !explain.is_empty() {
                html.push_str(&format!(
                    "<details class='explain'>\n\
                     <summary>Explain {}</summary>\n\
                     <pre>{}</pre>\n\
                     </details>\n",
                    code.code(),
                    self.escape(explain),
                ));
            }
        }

        html.push_str("</div>\n");
        html
    }

    /// Render the source context as HTML, directly from the Diagnostic's
    /// source text, spans, and labels — without going through GlyphRenderer's
    /// terminal-formatted output.
    fn render_source_html(&self, source: &str, primary_spans: &[Span], labels: &[Label]) -> String {
        let Some(&span) = primary_spans.first() else {
            return String::new();
        };
        let lines: Vec<&str> = source.lines().collect();
        if lines.is_empty() {
            return String::new();
        }

        let start_pos = byte_to_linecol(source, span.start);
        let end_pos = byte_to_linecol(source, span.end);

        let context_lines = 2;
        let first_line = start_pos.line.saturating_sub(context_lines);
        let last_line = std::cmp::min(
            end_pos.line + 1 + context_lines,
            lines.len().saturating_sub(1),
        );
        let line_num_width = format!("{}", last_line + 1).len();

        // Collect all labels: primary span first, then existing labels
        let mut all_labels = Vec::with_capacity(labels.len() + 1);
        all_labels.push(Label {
            span,
            message: String::new(),
            kind: AnnotationKind::Primary,
        });
        all_labels.extend(labels.iter().cloned());

        let mut html = String::new();

        for line_idx in first_line..=last_line {
            let line = lines[line_idx];
            let underlines =
                compute_line_underlines(&all_labels, source, line, line_idx);

            // ── Build primary columns set for background highlighting ──
            let mut primary_cols: Vec<bool> = vec![false; line.len()];
            for (col, ulen, underline_char, _msg) in &underlines {
                if *underline_char == '^' {
                    let end = std::cmp::min(col + ulen, line.len());
                    for i in *col..end {
                        if i < primary_cols.len() {
                            primary_cols[i] = true;
                        }
                    }
                }
            }

            // ── Render line number ──
            let line_num_str = format!("{:>width$}", line_idx + 1, width = line_num_width);
            html.push_str(&format!(
                "<span class='line-num'>{line_num_str} │ </span>"
            ));

            // ── Render source line with primary span highlighting ──
            let mut rendered = String::with_capacity(line.len() + 64);
            let mut i = 0;
            while i < line.len() {
                if primary_cols[i] {
                    rendered.push_str("<span class='highlight'>");
                    let start = i;
                    while i < line.len() && primary_cols[i] {
                        i += 1;
                    }
                    rendered.push_str(&self.escape(&line[start..i]));
                    rendered.push_str("</span>");
                } else {
                    let start = i;
                    while i < line.len() && !primary_cols[i] {
                        i += 1;
                    }
                    rendered.push_str(&self.escape(&line[start..i]));
                }
            }
            html.push_str(&rendered);
            html.push('\n');

            // ── Render underline annotations ──
            if !underlines.is_empty() {
                let spaces = " ".repeat(line_num_width + 1);
                html.push_str(&format!(
                    "<span class='line-num'>{spaces} │ </span>"
                ));

                // Merge overlapping underlines (primary `^` takes precedence)
                let mut combined: Vec<char> = vec![' '; line.len()];
                fn priority(c: char) -> u8 {
                    match c {
                        '^' => 0,
                        '~' => 1,
                        '-' => 2,
                        _ => 3,
                    }
                }
                for (col, ulen, underline_char, _msg) in &underlines {
                    let end = std::cmp::min(col + ulen, line.len());
                    for i in *col..end {
                        if priority(*underline_char) < priority(combined[i]) {
                            combined[i] = *underline_char;
                        }
                    }
                }

                let combined_str: String = combined.iter().collect();
                let trimmed = combined_str.trim().to_string();
                if !trimmed.is_empty() {
                    let first_col =
                        combined.iter().position(|c| *c != ' ').unwrap_or(0);
                    let padded = " ".repeat(first_col) + &trimmed;
                    html.push_str(&format!(
                        "<span class='underline'>{}</span>",
                        self.escape(&padded)
                    ));
                }
                html.push('\n');

                // ── Render label message lines ──
                for (col, _ulen, _ch, msg) in &underlines {
                    if msg.is_empty() {
                        continue;
                    }
                    let msg_spaces = " ".repeat(line_num_width + 1);
                    html.push_str(&format!(
                        "<span class='line-num'>{msg_spaces} │ </span>"
                    ));
                    let mut connector = String::new();
                    for _ in 0..*col {
                        connector.push(' ');
                    }
                    connector.push('|');
                    html.push_str(&format!(
                        "<span class='label'>{} {}</span>",
                        self.escape(&connector),
                        self.escape(msg),
                    ));
                    html.push('\n');
                }
            }
        }

        html
    }

    fn escape(&self, s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
}
