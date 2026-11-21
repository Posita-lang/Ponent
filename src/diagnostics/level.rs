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

// `Ord`'s contract requires `cmp(a, b) == Equal` **iff** `a == b`.  That holds
// only while every severity value is distinct: a copy-pasted severity would make
// two distinct variants compare equal, which `sort` tolerates but
// `BTreeMap`/binary search silently corrupt.  `severity()` is a `const fn`, so
// the ladder is a build-time fact (the same discipline as `error_code.rs`'s
// shape/slot asserts).  Pinning the exact values also makes a severity
// re-weighting a conscious edit to this block instead of silent drift.
const _: () = {
    assert!(DiagnosticLevel::Info.severity() == 0);
    assert!(DiagnosticLevel::Note.severity() == 1);
    assert!(DiagnosticLevel::Help.severity() == 2);
    assert!(DiagnosticLevel::Warning.severity() == 3);
    assert!(DiagnosticLevel::Error.severity() == 4);
    assert!(DiagnosticLevel::Fatal.severity() == 5);
};

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
    ///
    /// Every variant is written out: a catch-all arm would silently extend the
    /// "annotations are not headlines" policy to a future level that never
    /// consented to it, and this was the file's only silent arm on variant
    /// addition (every other accessor is exhaustive by construction).
    pub const fn ansi_bold_color(&self) -> &'static str {
        match self {
            // Keep Fatal visually consistent with `ansi_color` (white on red),
            // just bold: bright-white text on a red background.
            DiagnosticLevel::Fatal => "\x1b[1;41;97m",
            DiagnosticLevel::Error => "\x1b[31;1m",
            DiagnosticLevel::Warning => "\x1b[33;1m",
            DiagnosticLevel::Help => self.ansi_color(),
            DiagnosticLevel::Note => self.ansi_color(),
            DiagnosticLevel::Info => self.ansi_color(),
        }
    }

    /// SGR reset, emitted at every colour close.
    ///
    /// A `const` rather than a `const fn` so it is usable in const context
    /// (static tables, `concat!`-adjacent helpers) and needs no call at all at
    /// the hot close sites.
    ///
    /// This is a **full** reset: it clears bold/fg/bg together.  Correct for the
    /// current flat header layout; if a renderer ever colours a region larger
    /// than one header and closes inner spans with this, the enclosing style
    /// dies silently (selective resets — `ESC[22m`/`39m`/`49m` — are the escape
    /// hatch if nesting ever arrives).
    pub const ANSI_RESET: &str = "\x1b[0m";

    /// Localized human-readable label for this level.
    ///
    /// Returns `&'static str` (not `Cow`): the label always comes from the
    /// per-language cache, so the `Cow::Owned` branch would be dead and the
    /// `Cow` shell pure overhead on the per-diagnostic hot path
    /// (`prefix_cols` / `write_header` / `Display`).
    /// Table slot for this level, in the order the label tables are indexed.
    ///
    /// Explicit arms rather than `*self as usize`: a reordering of the enum
    /// becomes a non-exhaustive-match compile error instead of silently
    /// swapping labels.  One mapping shared by [`Self::label`] and
    /// [`Self::label_upper`], so the two tables cannot be aligned differently.
    ///
    /// Deliberately **not** `severity()`: the slot names a table position, and
    /// coupling it to the severity ladder would silently swap labels if the
    /// ladder were re-weighted.
    #[inline]
    const fn cache_slot(self) -> usize {
        match self {
            DiagnosticLevel::Info => 0,
            DiagnosticLevel::Note => 1,
            DiagnosticLevel::Help => 2,
            DiagnosticLevel::Warning => 3,
            DiagnosticLevel::Error => 4,
            DiagnosticLevel::Fatal => 5,
        }
    }

    pub fn label(&self) -> &'static str {
        // Explicit variant → index mapping: `*self as usize` would silently
        // swap labels if the enum were ever reordered; a match arms the
        // reorder into a non-exhaustive compile error instead.
        let idx = self.cache_slot();
        crate::diagnostics::i18n::cached_label_static! {
            n = 6,
            variant = idx,
            slugs = [
                "level-info",
                "level-note",
                "level-help",
                "level-warning",
                "level-error",
                "level-fatal",
            ],
        }
    }

    /// Uppercased localized label, cached per language.
    ///
    /// `mod.rs`'s `impl Display for Diagnostic` used to call
    /// `label().to_uppercase()`, allocating a fresh `String` per render for a
    /// value fully determined by `(level, language)` — at most 6 × `N_LANGS`
    /// entries.  Unicode uppercase, not `to_ascii_uppercase`: labels are
    /// localized (`Ошибка`), so an ASCII-only fold would leave Cyrillic
    /// lowercase.
    pub fn label_upper(&self) -> &'static str {
        // Same variant → init-array anchoring as `label()`: the index names the
        // table position, not `severity()`'s semantic weight (they coincide
        // today by construction and must not be coupled — a severity
        // re-weighting would otherwise silently swap labels).
        let idx = self.cache_slot();
        crate::diagnostics::i18n::cached_label_static! {
            n = 6,
            variant = idx,
            slugs_upper = [
                "level-info",
                "level-note",
                "level-help",
                "level-warning",
                "level-error",
                "level-fatal",
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

    const ALL: [DiagnosticLevel; 6] = [
        DiagnosticLevel::Info,
        DiagnosticLevel::Note,
        DiagnosticLevel::Help,
        DiagnosticLevel::Warning,
        DiagnosticLevel::Error,
        DiagnosticLevel::Fatal,
    ];

    /// Machine forms and prefixes are pairwise distinct (a copy-paste would make
    /// two levels indistinguishable to `--level` filters and CI token parsing).
    #[test]
    fn machine_forms_are_pairwise_distinct() {
        for (i, a) in ALL.iter().enumerate() {
            for b in &ALL[i + 1..] {
                assert_ne!(a.as_str(), b.as_str(), "{a:?} vs {b:?}");
                assert_ne!(a.prefix(), b.prefix(), "{a:?} vs {b:?}");
            }
        }
    }

    /// The severity ladder is the Ord delegate (compile-time pinned above); the
    /// runtime pin documents intent next to the values.
    #[test]
    fn severity_ladder_and_error_set() {
        assert_eq!(
            ALL.map(|l| l.severity()),
            [0, 1, 2, 3, 4, 5],
            "severity ladder changed: Ord/Eq consistency and documented order"
        );
        // Exactly `{Error, Fatal}`: a new gate-worthy level silently dropping
        // out of exit-code / `max_errors` accounting is the failure this pins.
        for l in ALL {
            assert_eq!(
                l.is_error(),
                matches!(l, DiagnosticLevel::Error | DiagnosticLevel::Fatal),
                "{l:?}"
            );
        }
        // Ord follows severity, and Equal only for the same variant.
        let mut sorted = ALL;
        sorted.sort();
        assert_eq!(
            sorted, ALL,
            "Ord must follow the declaration/severity order"
        );
    }

    /// Golden English labels: distinctness alone cannot catch two swapped
    /// accessors in the `label()` init array.
    #[test]
    fn label_goldens_are_stable_english() {
        let _guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::En);
        for (level, want) in [
            (DiagnosticLevel::Info, "info"),
            (DiagnosticLevel::Note, "note"),
            (DiagnosticLevel::Help, "help"),
            (DiagnosticLevel::Warning, "warning"),
            (DiagnosticLevel::Error, "error"),
            (DiagnosticLevel::Fatal, "fatal"),
        ] {
            assert_eq!(level.label(), want, "{level:?}");
            assert!(level.label_upper().len() >= want.len(), "{level:?}");
            assert_eq!(
                level.label_upper(),
                want.to_uppercase(),
                "label_upper must be the Unicode uppercase of label, {level:?}"
            );
        }
        // Per-variant distinctness, so an accessor swap that keeps arity fails.
        for (i, a) in ALL.iter().enumerate() {
            assert!(!a.label().is_empty(), "{a:?}");
            for b in &ALL[i + 1..] {
                assert_ne!(a.label(), b.label(), "{a:?} vs {b:?}");
            }
        }
    }

    /// Non-ASCII uppercasing: the one way `label_upper` can be subtly wrong
    /// (an ASCII fold would leave Cyrillic lowercase).
    #[test]
    fn label_upper_uses_unicode_folding() {
        let _guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::Ru);
        let err = DiagnosticLevel::Error;
        assert_eq!(err.label_upper(), err.label().to_uppercase());
        assert_ne!(
            err.label(),
            err.label_upper(),
            "ru error label is lowercase"
        );
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::En);
    }

    /// SGR goldens: a typo (`32m` where `31m` belongs) ships as green errors and
    /// no other test in the tree would notice.
    #[test]
    fn ansi_goldens_are_pinned() {
        assert_eq!(DiagnosticLevel::ANSI_RESET, "\x1b[0m");
        for (level, color) in [
            (DiagnosticLevel::Fatal, "\x1b[41;37m"),
            (DiagnosticLevel::Error, "\x1b[31m"),
            (DiagnosticLevel::Warning, "\x1b[33m"),
            (DiagnosticLevel::Help, "\x1b[36m"),
            (DiagnosticLevel::Note, "\x1b[34m"),
            (DiagnosticLevel::Info, "\x1b[32m"),
        ] {
            assert_eq!(level.ansi_color(), color, "{level:?}");
        }
        assert_eq!(DiagnosticLevel::Fatal.ansi_bold_color(), "\x1b[1;41;97m");
        assert_eq!(DiagnosticLevel::Error.ansi_bold_color(), "\x1b[31;1m");
        assert_eq!(DiagnosticLevel::Warning.ansi_bold_color(), "\x1b[33;1m");
        // The documented fall-through: annotations are not headlines.
        for level in [
            DiagnosticLevel::Help,
            DiagnosticLevel::Note,
            DiagnosticLevel::Info,
        ] {
            assert_eq!(level.ansi_bold_color(), level.ansi_color(), "{level:?}");
        }
    }
}
