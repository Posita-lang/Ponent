use std::cmp::Ordering;
use std::fmt;

/// The severity or kind of a diagnostic message.
///
/// **Variant order is not load-bearing for severity semantics**: the `Ord`/
/// `PartialOrd` impls below delegate to [`DiagnosticLevel::severity`], so a
/// `sort_by_key`/`max()`/range over the enum is reorder-proof.  Prefer
/// [`DiagnosticLevel::severity`] or [`DiagnosticLevel::is_error`] for explicit
/// severity reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Info,
    Note,
    Help,
    Warning,
    Error,
    Fatal,
}

impl Ord for DiagnosticLevel {
    fn cmp(&self, other: &Self) -> Ordering {
        // Delegate to the explicit severity mapping so a future reorder of the
        // declaration cannot silently change ordering semantics (the doc
        // promise this replaces).
        self.severity().cmp(&other.severity())
    }
}

impl PartialOrd for DiagnosticLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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

    #[inline(always)]
    pub const fn ansi_reset() -> &'static [u8; 4] {
        b"\x1b[0m"
    }

    /// Localized human-readable label for this level.
    ///
    /// Returns `&'static str` (not `Cow`): the label always comes from the
    /// per-language cache, so the `Cow::Owned` branch would be dead and the
    /// `Cow` shell pure overhead on the per-diagnostic hot path
    /// (`prefix_cols` / `write_header` / `Display`).
    pub fn label(&self) -> &'static str {
        // Explicit variant → index mapping: `*self as usize` would silently
        // swap labels if the enum were ever reordered; a match arms the
        // reorder into a non-exhaustive compile error instead.
        let idx = match self {
            DiagnosticLevel::Info => 0,
            DiagnosticLevel::Note => 1,
            DiagnosticLevel::Help => 2,
            DiagnosticLevel::Warning => 3,
            DiagnosticLevel::Error => 4,
            DiagnosticLevel::Fatal => 5,
        };
        crate::diagnostics::i18n::cached_label_static! {
            n = 6,
            variant = idx,
            init = [
                crate::diagnostics::i18n::fluent::level_info().into_owned(),
                crate::diagnostics::i18n::fluent::level_note().into_owned(),
                crate::diagnostics::i18n::fluent::level_help().into_owned(),
                crate::diagnostics::i18n::fluent::level_warning().into_owned(),
                crate::diagnostics::i18n::fluent::level_error().into_owned(),
                crate::diagnostics::i18n::fluent::level_fatal().into_owned(),
            ],
        }
    }
}
