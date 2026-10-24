//! Terminal capability detection.
//!
//! Single home for the `NO_COLOR` / `CLICOLOR_FORCE` / `TERM=dumb` / `isatty`
//! dance plus width, Unicode, and OSC 8 hyperlink probes, so downstream
//! consumers don't each reimplement it (and get it subtly wrong).
//!
//! Probes read the process environment and, where relevant, the `stderr`
//! `isatty` state, so results are cheap but not stable across a process
//! lifetime.  Callers that need a fixed answer (or that want to honour a CLI
//! flag) should snapshot into [`DiagOpts`](crate::diagnostics::DiagOpts) once
//! via [`DiagOpts::detect`](crate::diagnostics::DiagOpts::detect).

use std::io::{self, IsTerminal};
use terminal_size::{self, Width};

use crate::diagnostics::DEFAULT_MAX_WIDTH;

/// Terminal colour depth, used to downgrade 256-colour sequences on
/// terminals that only support the 16-colour ANSI palette.
///
/// Detected by [`detect_color_depth`]; the renderer uses it to choose
/// between `\x1b[48;5;236m` (256-colour dark-gray background) and
/// `\x1b[7m` (reverse — universally available) for source-highlight
/// backgrounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorDepth {
    /// 16-colour ANSI palette only.
    Sixteen,
    /// 256-colour palette (xterm-256color and friends).
    TwoFiftySix,
    /// 24-bit true colour.
    TrueColor,
}

impl ColorDepth {
    /// Whether this depth can render 256-colour palette sequences.
    pub fn supports_256(self) -> bool {
        matches!(self, ColorDepth::TwoFiftySix | ColorDepth::TrueColor)
    }
}

/// Detect the terminal's colour depth from `COLORTERM` and `TERM`.
///
/// Precedence:
///   1. `COLORTERM` contains `truecolor` or `24bit` → [`ColorDepth::TrueColor`]
///   2. `TERM` ends with `256color` → [`ColorDepth::TwoFiftySix`]
///   3. Otherwise → [`ColorDepth::Sixteen`]
pub fn detect_color_depth() -> ColorDepth {
    let colorterm = std::env::var("COLORTERM").ok();
    let term = std::env::var("TERM").ok();
    color_depth_decision(colorterm.as_deref(), term.as_deref())
}

pub(crate) fn color_depth_decision(colorterm: Option<&str>, term: Option<&str>) -> ColorDepth {
    if let Some(ct) = colorterm {
        let ct = ct.to_ascii_lowercase();
        if ct.contains("truecolor") || ct.contains("24bit") {
            return ColorDepth::TrueColor;
        }
    }
    if let Some(t) = term {
        if t.ends_with("256color") {
            return ColorDepth::TwoFiftySix;
        }
    }
    ColorDepth::Sixteen
}

/// Whether ANSI colour should be emitted on stderr.
///
/// Precedence:
///   1. `NO_COLOR` set (any value, including empty) → off.
///   2. `CLICOLOR_FORCE` set to anything other than `"0"` → on.
///   3. `FORCE_COLOR` set to anything other than `"0"` → on.
///   4. `TERM=dumb` → off.
///   5. Otherwise: on iff `stderr` is a terminal.
pub fn detect_color() -> bool {
    let clicolor_force = std::env::var("CLICOLOR_FORCE").ok();
    let force_color = std::env::var("FORCE_COLOR").ok();
    let term = std::env::var("TERM").ok();
    color_decision(
        std::env::var_os("NO_COLOR").is_some(),
        clicolor_force.as_deref(),
        force_color.as_deref(),
        term.as_deref(),
        io::stderr().is_terminal(),
    )
}

/// Pure decision core of [`detect_color`], factored out so the policy can be
/// tested without mutating process-global environment variables.
pub(crate) fn color_decision(
    no_color: bool,
    clicolor_force: Option<&str>,
    force_color: Option<&str>,
    term: Option<&str>,
    stderr_is_tty: bool,
) -> bool {
    if no_color {
        return false;
    }
    // `CLICOLOR_FORCE=0` means "don't force", not "force off" — it falls
    // through to normal detection.  Any other value (including empty) forces.
    if let Some(v) = clicolor_force
        && v != "0"
    {
        return true;
    }
    // `FORCE_COLOR` follows the same convention as `CLICOLOR_FORCE`:
    // `0` = don't force, any other value (including empty) = force on.
    // Widely used in the Node.js ecosystem (`FORCE_COLOR=1/2/3`).
    if let Some(v) = force_color
        && v != "0"
    {
        return true;
    }
    if term == Some("dumb") {
        return false;
    }
    stderr_is_tty
}

/// Whether the terminal can render box-drawing (`┌─└─│`) and wide glyphs.
///
/// Follows the POSIX precedence `LC_ALL` > `LC_CTYPE` > `LANG`, treating a
/// variable set to the empty string as unset.  Any locale name containing
/// `utf-8`/`utf8` (case-insensitive) counts as Unicode-capable; with no locale
/// information at all the answer is `true`, matching modern defaults.
///
/// This is deliberately **independent** of [`detect_color`]: a CI log with
/// colour forced on but `LANG=C` must still get ASCII box-drawing, not
/// mojibake.
pub fn detect_unicode() -> bool {
    let locale = ["LC_ALL", "LC_CTYPE", "LANG"]
        .iter()
        .find_map(|v| std::env::var(v).ok().filter(|s| !s.is_empty()));
    unicode_decision(locale.as_deref())
}

pub(crate) fn unicode_decision(locale: Option<&str>) -> bool {
    match locale {
        Some(v) => {
            let v = v.to_ascii_lowercase();
            v.contains("utf-8") || v.contains("utf8")
        }
        None => true,
    }
}

/// Target output width in display columns.
///
/// `COLUMNS` wins when set and parseable to a positive integer; otherwise the
/// real terminal size is probed via [`terminal_size`] (which falls back to
/// [`DEFAULT_MAX_WIDTH`] when the probe returns `None`).  `DEFAULT_MAX_WIDTH`
/// is also used when `COLUMNS` is set but unparseable or non-positive.
pub fn detect_width() -> usize {
    let columns = std::env::var("COLUMNS").ok();
    let cols = columns.as_deref();
    width_decision(cols, terminal_size::terminal_size().map(|(Width(w), _)| w as usize))
}

pub(crate) fn width_decision(columns: Option<&str>, term_size: Option<usize>) -> usize {
    columns
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&w| w > 0)
        .or(term_size)
        .unwrap_or(DEFAULT_MAX_WIDTH)
}

/// Whether the terminal can render OSC 8 hyperlinks (`ESC ] 8 ; ; URL ST`).
///
/// Two explicit overrides are honoured before the terminal probe:
///   * `NO_HYPERLINK` set (any value, including empty) → off.
///   * `FORCE_HYPERLINK` set (any non-`"0"` value) → on.
/// These match the `NO_COLOR` / `CLICOLOR_FORCE` convention so callers can
/// wire a single CLI flag through to both colour and hyperlink emission.
///
/// When neither override is active, the check delegates to the
/// [`supports-hyperlinks`] crate, which maintains a curated table of
/// terminal identifiers across all known families and is updated as new
/// terminals ship.  **When in doubt the answer is `false`**: on terminals
/// that don't implement OSC 8 the result ranges from "silently ignored" to
/// "literal `]8;;…` text", and the latter is not acceptable.
pub fn supports_hyperlinks() -> bool {
    if std::env::var_os("NO_HYPERLINK").is_some() {
        return false;
    }
    if std::env::var_os("FORCE_HYPERLINK")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        return true;
    }
    supports_hyperlinks::supports_hyperlinks()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_color_beats_everything() {
        assert!(!color_decision(
            true,
            Some("1"),
            Some("1"),
            Some("xterm-256color"),
            true
        ));
        assert!(!color_decision(true, None, None, None, true));
    }

    #[test]
    fn clicolor_force_only_forces_on_a_nonzero_value() {
        assert!(color_decision(false, Some("1"), None, Some("dumb"), false));
        assert!(color_decision(false, Some(""), None, Some("dumb"), false));
        // `0` means "don't force": fall through to the terminal check.
        assert!(!color_decision(false, Some("0"), None, Some("xterm"), false));
        assert!(color_decision(false, Some("0"), None, Some("xterm"), true));
    }

    #[test]
    fn force_color_forces_on_after_clicolor_force() {
        // FORCE_COLOR works when CLICOLOR_FORCE is absent.
        assert!(color_decision(false, None, Some("1"), Some("dumb"), false));
        assert!(color_decision(false, None, Some(""), Some("dumb"), false));
        // FORCE_COLOR=0 falls through.
        assert!(!color_decision(false, None, Some("0"), Some("xterm"), false));
        assert!(color_decision(false, None, Some("0"), Some("xterm"), true));
        // CLICOLOR_FORCE takes precedence over FORCE_COLOR (both set, CLICOLOR wins).
        assert!(color_decision(false, Some("1"), Some("0"), Some("dumb"), false));
    }

    #[test]
    fn dumb_terminal_disables_color() {
        assert!(!color_decision(false, None, None, Some("dumb"), true));
    }

    #[test]
    fn locale_utf8_detection() {
        assert!(unicode_decision(Some("en_US.UTF-8")));
        assert!(unicode_decision(Some("C.utf8")));
        assert!(!unicode_decision(Some("C")));
        assert!(!unicode_decision(Some("POSIX")));
        // No locale information at all → assume UTF-8.
        assert!(unicode_decision(None));
    }

    #[test]
    fn width_honours_columns_and_rejects_junk() {
        assert_eq!(width_decision(Some("120"), None), 120);
        assert_eq!(width_decision(Some("0"), None), DEFAULT_MAX_WIDTH);
        assert_eq!(width_decision(Some("nope"), None), DEFAULT_MAX_WIDTH);
        assert_eq!(width_decision(None, None), DEFAULT_MAX_WIDTH);
        // terminal_size wins when COLUMNS is absent.
        assert_eq!(width_decision(None, Some(80)), 80);
    }

    #[test]
    fn color_depth_detection() {
        // COLORTERM=truecolor → TrueColor
        assert_eq!(
            color_depth_decision(Some("truecolor"), Some("xterm")),
            ColorDepth::TrueColor
        );
        // COLORTERM=24bit → TrueColor (case-insensitive)
        assert_eq!(
            color_depth_decision(Some("24BIT"), None),
            ColorDepth::TrueColor
        );
        // TERM ends with 256color → TwoFiftySix
        assert_eq!(
            color_depth_decision(None, Some("xterm-256color")),
            ColorDepth::TwoFiftySix
        );
        // Neither → Sixteen
        assert_eq!(
            color_depth_decision(None, Some("xterm")),
            ColorDepth::Sixteen
        );
        assert_eq!(color_depth_decision(None, None), ColorDepth::Sixteen);
        // COLORTERM takes precedence over TERM.
        assert_eq!(
            color_depth_decision(Some("truecolor"), Some("xterm")),
            ColorDepth::TrueColor
        );
    }
}
