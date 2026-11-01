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

use std::ffi::OsStr;
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
    if let Some(t) = term
        && t.ends_with("256color")
    {
        return ColorDepth::TwoFiftySix;
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
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    color_decision(
        false,
        std::env::var_os("CLICOLOR_FORCE").as_deref(),
        std::env::var_os("FORCE_COLOR").as_deref(),
        std::env::var_os("TERM").as_deref(),
        io::stderr().is_terminal(),
    )
}

/// Pure decision core of [`detect_color`], factored out so the policy can be
/// tested without mutating process-global environment variables.
pub(crate) fn color_decision(
    no_color: bool,
    clicolor_force: Option<&OsStr>,
    force_color: Option<&OsStr>,
    term: Option<&OsStr>,
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
    if term == Some(OsStr::new("dumb")) {
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
    width_decision(
        cols,
        terminal_size::terminal_size().map(|(Width(w), _)| w as usize),
    )
}

pub(crate) fn width_decision(columns: Option<&str>, term_size: Option<usize>) -> usize {
    columns
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&w| w > 0)
        .or(term_size.filter(|&w| w > 0))
        .unwrap_or(DEFAULT_MAX_WIDTH)
}

/// Whether the terminal can render OSC 8 hyperlinks (`ESC ] 8 ; ; URL ST`).
///
/// Two explicit overrides are honoured before the terminal probe:
///   * `NO_HYPERLINK` set (any value, including empty) → off.
///   * `FORCE_HYPERLINK` set (any non-`"0"` value) → on.
///
/// These match the `NO_COLOR` / `CLICOLOR_FORCE` convention so callers can
/// wire a single CLI flag through to both colour and hyperlink emission.
///
/// When neither override is active, the check delegates to the
/// [`supports-hyperlinks`] crate, which maintains a curated table of
/// terminal identifiers across all known families and is updated as new
/// terminals ship.  **When in doubt the answer is `false`**: on terminals
/// that don't implement OSC 8 the result ranges from "silently ignored" to
/// "literal `]8;;…` text", and the latter is not acceptable.
///
/// The result is memoized in a process-global [`OnceLock`]: the probe reads
/// `NO_HYPERLINK` / `FORCE_HYPERLINK` and the `supports-hyperlinks` crate's
/// own env table, none of which change during a single CLI run.  Without the
/// cache a 1 000-diagnostic batch paid several thousand env-mutex acquisitions
/// for an answer that is fixed at startup.
static HYPERLINK_CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn supports_hyperlinks() -> bool {
    *HYPERLINK_CACHE.get_or_init(|| {
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
    })
}

/// Percent-encode the bytes that are unsafe in a `file://` URL path.
///
/// Keeps unreserved characters plus `/`, `:`, `-`, `_`, `.`, `~`; encodes
/// everything else (including `\x1b`, `\x07`, space, `#`, `?`, `%`).  This
/// keeps a filename from terminating the enclosing OSC 8 hyperlink early or
/// injecting terminal control codes.
pub(crate) fn encode_file_url_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b':' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                use std::fmt::Write;
                let _ = write!(out, "%{:02X}", b);
            }
        }
    }
    out
}

/// Convert a file path to a `file://` URL, handling WSL path translation.
///
/// When running inside WSL (Windows Subsystem for Linux), a file path like
/// `/home/user/file.rs` is not understood by Windows applications.  Terminals
/// running on Windows (e.g., Windows Terminal) that receive a `file:///home/...`
/// hyperlink will try to open it as `C:\home\...` and fail.
///
/// We detect WSL via the `WSL_DISTRO_NAME` environment variable and translate
/// the URL to the `\\wsl.localhost\<distro>\...` format, which Windows can open.
///
/// For paths inside mounted Windows drives (`/mnt/c/...`), we translate them
/// to native Windows file URLs (`file:///C:/...`) so they open directly in
/// Windows without going through the WSL share.
pub fn path_to_file_url(path: &std::path::Path) -> String {
    let abs_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mut abs_path_str = abs_path.to_string_lossy().into_owned();

    // `canonicalize()` on Windows returns verbatim paths (`\\?\C:\...`).
    // Strip the prefix for the URL.
    if let Some(stripped) = abs_path_str.strip_prefix(r"\\?\UNC\") {
        abs_path_str = format!("//{}", stripped);
    } else if let Some(stripped) = abs_path_str.strip_prefix(r"\\?\") {
        abs_path_str = stripped.to_string();
    } else if let Some(stripped) = abs_path_str.strip_prefix(r"\\.\") {
        abs_path_str = stripped.to_string();
    }

    // Windows paths use `\`, but URLs require `/`.
    let path_for_url = if cfg!(windows) || abs_path_str.contains('\\') {
        abs_path_str.replace('\\', "/")
    } else {
        abs_path_str
    };

    let distro = std::env::var("WSL_DISTRO_NAME").ok();
    format_file_url(&path_for_url, distro.as_deref())
}

fn format_file_url(path_for_url: &str, distro: Option<&str>) -> String {
    if let Some(distro) = distro
        && !distro.is_empty()
    {
        if let Some(stripped) = path_for_url.strip_prefix("/mnt/") {
            if stripped.len() >= 2
                && stripped.as_bytes()[1] == b'/'
                && stripped.as_bytes()[0].is_ascii_alphabetic()
            {
                let drive = stripped.as_bytes()[0].to_ascii_uppercase() as char;
                let rest = &stripped[1..];
                let translated = format!("/{drive}:{rest}");
                return format!("file://{}", encode_file_url_path(&translated));
            }
        }
        let encoded_distro = encode_file_url_path(distro);
        let encoded_path = encode_file_url_path(path_for_url);
        return format!("file://wsl.localhost/{}{}", encoded_distro, encoded_path);
    }

    // Windows drive letter: `C:/foo` → `file:///C:/foo` (three slashes).
    if path_for_url.len() >= 2 && path_for_url.as_bytes()[1] == b':' {
        return format!("file:///{}", encode_file_url_path(path_for_url));
    }

    format!("file://{}", encode_file_url_path(path_for_url))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_color_beats_everything() {
        assert!(!color_decision(
            true,
            Some(OsStr::new("1")),
            Some(OsStr::new("1")),
            Some(OsStr::new("xterm-256color")),
            true
        ));
        assert!(!color_decision(true, None, None, None, true));
    }

    #[test]
    fn clicolor_force_only_forces_on_a_nonzero_value() {
        assert!(color_decision(
            false,
            Some(OsStr::new("1")),
            None,
            Some(OsStr::new("dumb")),
            false
        ));
        assert!(color_decision(
            false,
            Some(OsStr::new("")),
            None,
            Some(OsStr::new("dumb")),
            false
        ));
        // `0` means "don't force": fall through to the terminal check.
        assert!(!color_decision(
            false,
            Some(OsStr::new("0")),
            None,
            Some(OsStr::new("xterm")),
            false
        ));
        assert!(color_decision(
            false,
            Some(OsStr::new("0")),
            None,
            Some(OsStr::new("xterm")),
            true
        ));
    }

    #[test]
    fn force_color_forces_on_after_clicolor_force() {
        // FORCE_COLOR works when CLICOLOR_FORCE is absent.
        assert!(color_decision(
            false,
            None,
            Some(OsStr::new("1")),
            Some(OsStr::new("dumb")),
            false
        ));
        assert!(color_decision(
            false,
            None,
            Some(OsStr::new("")),
            Some(OsStr::new("dumb")),
            false
        ));
        // FORCE_COLOR=0 falls through.
        assert!(!color_decision(
            false,
            None,
            Some(OsStr::new("0")),
            Some(OsStr::new("xterm")),
            false
        ));
        assert!(color_decision(
            false,
            None,
            Some(OsStr::new("0")),
            Some(OsStr::new("xterm")),
            true
        ));
        // CLICOLOR_FORCE takes precedence over FORCE_COLOR (both set, CLICOLOR wins).
        assert!(color_decision(
            false,
            Some(OsStr::new("1")),
            Some(OsStr::new("0")),
            Some(OsStr::new("dumb")),
            false
        ));
    }

    #[test]
    fn dumb_terminal_disables_color() {
        assert!(!color_decision(
            false,
            None,
            None,
            Some(OsStr::new("dumb")),
            true
        ));
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
        assert_eq!(width_decision(None, Some(0)), DEFAULT_MAX_WIDTH);
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

    #[test]
    fn format_file_url_wsl_and_windows() {
        assert_eq!(
            format_file_url("/home/user/file.rs", Some("Ubuntu")),
            "file://wsl.localhost/Ubuntu/home/user/file.rs"
        );
        assert_eq!(
            format_file_url("/mnt/c/Users/user/file.rs", Some("Ubuntu")),
            "file:///C:/Users/user/file.rs"
        );
        assert_eq!(format_file_url("/mnt/d/", Some("Ubuntu")), "file:///D:/");
        assert_eq!(
            format_file_url("/home/user/file.rs", None),
            "file:///home/user/file.rs"
        );
        assert_eq!(
            format_file_url("C:/Users/user/file.rs", None),
            "file:///C:/Users/user/file.rs"
        );
        assert_eq!(
            format_file_url("//server/share/file.rs", None),
            "file:////server/share/file.rs"
        );
        assert_eq!(
            format_file_url("/home/user/file.rs", Some("Ubuntu 20.04")),
            "file://wsl.localhost/Ubuntu%2020.04/home/user/file.rs"
        );
    }
}
