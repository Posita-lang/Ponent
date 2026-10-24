use std::borrow::Cow;
use std::fmt;

/// The severity or kind of a diagnostic message.
///
/// The derived `Ord` still follows declaration order, but **variant order
/// is no longer load-bearing for severity semantics**: use
/// [`DiagnosticLevel::severity`] (or [`DiagnosticLevel::is_error`]) to
/// reason about severity, so reordering the enum can never silently change
/// behaviour.
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
    /// Stable machine-readable key (JSON `"level"` field, HTML CSS class).
    /// Never localized — consumers depend on a fixed vocabulary.
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

    /// Numeric severity, higher = worse.  Explicit mapping so ordering never
    /// depends on declaration order.
    pub fn severity(&self) -> u8 {
        match self {
            DiagnosticLevel::Info => 0,
            DiagnosticLevel::Note => 1,
            DiagnosticLevel::Help => 2,
            DiagnosticLevel::Warning => 3,
            DiagnosticLevel::Error => 4,
            DiagnosticLevel::Fatal => 5,
        }
    }

    /// Whether this level counts as an error for exit codes and the
    /// `max_errors` gate.
    pub fn is_error(&self) -> bool {
        matches!(self, DiagnosticLevel::Error | DiagnosticLevel::Fatal)
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
            // Keep Fatal visually consistent with `ansi_color` (white on red),
            // just bold: bright-white text on a red background.
            DiagnosticLevel::Fatal => "\x1b[1;41;97m",
            DiagnosticLevel::Error => "\x1b[31;1m",
            DiagnosticLevel::Warning => "\x1b[33;1m",
            _ => self.ansi_color(),
        }
    }

    pub fn ansi_reset() -> &'static str {
        "\x1b[0m"
    }

    pub fn label(&self) -> Cow<'static, str> {
        match self {
            DiagnosticLevel::Fatal => crate::tr!("level-fatal"),
            DiagnosticLevel::Error => crate::tr!("level-error"),
            DiagnosticLevel::Warning => crate::tr!("level-warning"),
            DiagnosticLevel::Help => crate::tr!("level-help"),
            DiagnosticLevel::Note => crate::tr!("level-note"),
            DiagnosticLevel::Info => crate::tr!("level-info"),
        }
    }
}
