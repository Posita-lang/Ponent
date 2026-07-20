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

        // Source context
        if let (Some(span), Some(source)) = (diag.spans.first(), diag.source.as_ref()) {
            let ctx =
                crate::diagnostics::label::SourceContext::new(source.as_str(), span, "<input>", 2);
            let rendered = ctx.render(span, &diag.labels, false);
            if !rendered.is_empty() {
                let escaped = self.escape(
                    &rendered
                        .lines()
                        .filter(|l| !l.is_empty())
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
                html.push_str(&format!("<pre class='source-line'>{}</pre>\n", escaped));
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

    fn escape(&self, s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
}
