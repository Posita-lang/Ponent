use crate::diagnostics::Diagnostic;

pub mod colored;
pub mod html;
pub mod json;
pub mod plain;
pub mod stylesheet;

/// Trait for rendering diagnostics to different output formats.
/// Each emitter handles a specific format (plain text, colored, JSON, HTML).
pub trait DiagnosticEmitter {
    /// Emit a single diagnostic.
    fn emit(&mut self, diag: &Diagnostic);

    /// Emit multiple diagnostics, optionally with a summary.
    fn emit_all(&mut self, diags: &[Diagnostic]) {
        let error_count = diags.iter().filter(|d| d.is_error()).count();
        let warning_count = diags.iter().filter(|d| d.is_warning()).count();

        for diag in diags {
            self.emit(diag);
        }

        if error_count > 0 || warning_count > 0 {
            self.emit_summary(error_count, warning_count);
        }
    }

    /// Emit a summary line (e.g. "aborting due to N previous errors").
    fn emit_summary(&mut self, error_count: usize, warning_count: usize) {
        match (error_count, warning_count) {
            (0, 0) => {}
            (1, 0) => eprintln!("error: aborting due to previous error"),
            (n, 0) => eprintln!("error: aborting due to {} previous errors", n),
            (0, 1) => eprintln!("warning: 1 warning emitted"),
            (0, n) => eprintln!("warning: {} warnings emitted", n),
            (e, w) => eprintln!(
                "error: aborting due to {} previous error{}; {} warning{} emitted",
                e,
                if e == 1 { "" } else { "s" },
                w,
                if w == 1 { "" } else { "s" },
            ),
        }
    }
}
