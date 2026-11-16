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
    /// Never localized.  Note: unlike `ErrorCategory::as_str` (human text),
    /// this is the machine form; the human form is [`label`](Self::label).
    pub const fn as_str(&self) -> &'static str {
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
    /// depends on declaration order.  Info < Note < Help is ascending
    /// actionability, and all three sit below Warning because none gates the
    /// build.
    pub const fn severity(&self) -> u8 {
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
    pub const fn is_error(&self) -> bool {
        matches!(self, DiagnosticLevel::Error | DiagnosticLevel::Fatal)
    }

    pub const fn prefix(&self) -> &'static str {
        match self {
            DiagnosticLevel::Fatal => "F",
            DiagnosticLevel::Error => "E",
            DiagnosticLevel::Warning => "W",
            DiagnosticLevel::Help => "H",
            DiagnosticLevel::Note => "N",
            DiagnosticLevel::Info => "I",
        }
    }

    pub const fn ansi_color(&self) -> &'static str {
        match self {
            DiagnosticLevel::Fatal => "\x1b[41;37m", // red bg, white text
            DiagnosticLevel::Error => "\x1b[31m",
            DiagnosticLevel::Warning => "\x1b[33m",
            DiagnosticLevel::Help => "\x1b[36m",
            DiagnosticLevel::Note => "\x1b[34m",
            DiagnosticLevel::Info => "\x1b[32m",
        }
    }

    /// Bold form where the level has one; Help/Note/Info deliberately fall
    /// through to the non-bold color (they are annotations, not headlines).
    pub const fn ansi_bold_color(&self) -> &'static str {
        match self {
            // Keep Fatal visually consistent with `ansi_color` (white on red),
            // just bold: bright-white text on a red background.
            DiagnosticLevel::Fatal => "\x1b[1;41;97m",
            DiagnosticLevel::Error => "\x1b[31;1m",
            DiagnosticLevel::Warning => "\x1b[33;1m",
            _ => self.ansi_color(),
        }
    }

    #[inline]
    pub const fn ansi_reset() -> &'static str {
        "\x1b[0m"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_forms_are_stable_and_consistent() {
        for level in [
            DiagnosticLevel::Info,
            DiagnosticLevel::Note,
            DiagnosticLevel::Help,
            DiagnosticLevel::Warning,
            DiagnosticLevel::Error,
            DiagnosticLevel::Fatal,
        ] {
            let s = level.as_str();
            assert!(s.bytes().all(|b| b.is_ascii_lowercase()), "{s}");
            assert_eq!(
                level.prefix(),
                &s[..1].to_ascii_uppercase(),
                "prefix disagrees with machine form: {s}"
            );
        }
        assert_eq!(DiagnosticLevel::Fatal.as_str(), "fatal");
        assert_eq!(DiagnosticLevel::Error.as_str(), "error");
    }
}
