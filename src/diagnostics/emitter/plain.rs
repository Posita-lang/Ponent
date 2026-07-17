use crate::diagnostics::{Diagnostic, error_code::ErrorCode, level::DiagnosticLevel};

/// Plain text emitter — minimal output without ANSI colors.
/// Used for non-interactive terminals or CI environments.
pub struct PlainEmitter;

impl PlainEmitter {
    pub fn new() -> Self {
        PlainEmitter
    }
}

impl Default for PlainEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl super::DiagnosticEmitter for PlainEmitter {
    fn emit(&mut self, diag: &Diagnostic) {
        let level_str = diag.level.label();
        let code_str = diag.code.as_ref().map(|c| c.code()).unwrap_or("?");

        // Line 1: error code + level + message + location
        let span_str = diag.span.map(|s| format!(" at {}", s)).unwrap_or_default();
        eprintln!("[{} {}] {}{}", code_str, level_str, diag.message, span_str);

        // Line 2+: labeled sub-spans
        for lbl in &diag.labels {
            eprintln!(
                "  {}--> {}: {}",
                lbl.underline_char(),
                lbl.span,
                lbl.message
            );
        }

        // Source context rendering with ^-underline (if source available)
        if let (Some(span), Some(source)) = (diag.span, diag.source.as_ref()) {
            let ctx =
                crate::diagnostics::label::SourceContext::new(source.as_str(), span, "<input>", 2);
            let rendered = ctx.render(span, &diag.labels, false);
            if !rendered.is_empty() {
                eprintln!("{}", rendered);
            }
        }

        // Call chain (Zig-style)
        if let Some(ref chain) = diag.call_chain {
            let rendered = chain.render(false);
            if !rendered.is_empty() {
                eprintln!("{}", rendered);
            }
        }

        // help:
        if let Some(help) = &diag.help {
            eprintln!("help: {}", help);
        }

        // suggestion(s):
        for suggestion in &diag.suggestions {
            eprintln!("suggestion: {}", suggestion);
        }
    }
}
