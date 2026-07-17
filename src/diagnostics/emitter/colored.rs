use crate::diagnostics::{Diagnostic, level::DiagnosticLevel};

/// Colored terminal emitter — uses ANSI escape codes for rich output.
/// Format inspired by rustc's diagnostic output.
pub struct ColoredEmitter;

impl ColoredEmitter {
    pub fn new() -> Self {
        ColoredEmitter
    }
}

impl Default for ColoredEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl super::DiagnosticEmitter for ColoredEmitter {
    fn emit(&mut self, diag: &Diagnostic) {
        let level_str = diag.level.label();
        let code_str = diag.code.as_ref().map(|c| c.code()).unwrap_or("?");
        let color = diag.level.ansi_color();
        let bold = diag.level.ansi_bold_color();
        let reset = DiagnosticLevel::ansi_reset();

        // Line 1: bold error code + colored level + message
        let span_str = diag
            .span
            .map(|s| format!(" at {}{}{}", color, s, reset))
            .unwrap_or_default();
        eprintln!(
            "{bold}[{code} {level}]{reset} {msg}{span}",
            bold = bold,
            code = code_str,
            level = level_str,
            reset = reset,
            msg = diag.message,
            span = span_str,
        );

        // Source context with ^-underline rendering (Rust-style)
        if let (Some(span), Some(source)) = (diag.span, diag.source.as_ref()) {
            let ctx =
                crate::diagnostics::label::SourceContext::new(source.as_str(), span, "<input>", 2);
            let rendered = ctx.render(span, &diag.labels, true);
            if !rendered.is_empty() {
                eprintln!("{}", rendered);
            }
        }

        // Labels that couldn't be rendered in source context
        for lbl in &diag.labels {
            eprintln!(
                "  {}--> {}: {}{}{}",
                lbl.underline_char(),
                lbl.span,
                "\x1b[36m",
                lbl.message,
                DiagnosticLevel::ansi_reset(),
            );
        }

        // Call chain (Zig-style referenced by)
        if let Some(ref chain) = diag.call_chain {
            let rendered = chain.render(true);
            if !rendered.is_empty() {
                eprintln!("{}", rendered);
            }
        }

        // help: (cyan)
        if let Some(help) = &diag.help {
            eprintln!(
                "{cyan}help: {msg}{reset}",
                cyan = DiagnosticLevel::ansi_color_for_level(DiagnosticLevel::Help),
                msg = help,
                reset = reset,
            );
        }

        // suggestion(s)
        let reset = DiagnosticLevel::ansi_reset();
        for suggestion in &diag.suggestions {
            eprintln!(
                "{green}suggestion: {msg}{reset}",
                green = "\x1b[92m",
                msg = suggestion,
                reset = reset,
            );
        }
    }
}

impl DiagnosticLevel {
    pub fn ansi_color_for_level(level: DiagnosticLevel) -> &'static str {
        level.ansi_color()
    }
}
