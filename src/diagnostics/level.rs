use std::fmt;

/// The severity or kind of a diagnostic message.
///
/// Variant order is significant: derived `Ord` uses declaration order,
/// so least-severe variants come first, most-severe last.
/// This ensures `level >= DiagnosticLevel::Error` correctly matches
/// both `Error` and `Fatal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticLevel {
    Info,
    Note,
    Help,
    Warning,
    Error,
    Fatal,
}

impl DiagnosticLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            DiagnosticLevel::Fatal => "fatal",
            DiagnosticLevel::Error => "error",
            DiagnosticLevel::Warning => "warning",
            DiagnosticLevel::Help => "help",
            DiagnosticLevel::Note => "note",
            DiagnosticLevel::Info => "info",
        }
    }

    pub fn prefix(&self) -> &'static str {
        match self {
            DiagnosticLevel::Fatal => "F",
            DiagnosticLevel::Error => "E",
            DiagnosticLevel::Warning => "W",
            DiagnosticLevel::Help => "H",
            DiagnosticLevel::Note => "N",
            DiagnosticLevel::Info => "I",
        }
    }

    pub fn ansi_color(&self) -> &'static str {
        match self {
            DiagnosticLevel::Fatal => "\x1b[41;37m", // red bg, white text
            DiagnosticLevel::Error => "\x1b[31m",
            DiagnosticLevel::Warning => "\x1b[33m",
            DiagnosticLevel::Help => "\x1b[36m",
            DiagnosticLevel::Note => "\x1b[34m",
            DiagnosticLevel::Info => "\x1b[32m",
        }
    }

    pub fn ansi_bold_color(&self) -> &'static str {
        match self {
            DiagnosticLevel::Fatal => "\x1b[31;1m",
            DiagnosticLevel::Error => "\x1b[31;1m",
            DiagnosticLevel::Warning => "\x1b[33;1m",
            _ => self.ansi_color(),
        }
    }

    pub fn ansi_reset() -> &'static str {
        "\x1b[0m"
    }

    pub fn label(&self) -> &'static str {
        match self {
            DiagnosticLevel::Fatal => "fatal",
            DiagnosticLevel::Error => "error",
            DiagnosticLevel::Warning => "warning",
            DiagnosticLevel::Help => "help",
            DiagnosticLevel::Note => "note",
            DiagnosticLevel::Info => "info",
        }
    }
}
