//! Terminal capability detection.
//!
//! Single home for the `NO_COLOR` / `CLICOLOR` / `CLICOLOR_FORCE` /
//! `FORCE_COLOR` / `TERM=dumb` / `isatty` dance plus width, Unicode, and
//! OSC 8 hyperlink probes, so downstream consumers don't each reimplement it
//! (and get it subtly wrong).
//!
//! Probes read the process environment and, where relevant, the `stderr`
//! `isatty` state, so colour/width/Unicode results are cheap but not stable
//! across a process lifetime.  Callers that need a fixed answer (or that want
//! to honour a CLI flag) should snapshot into
//! [`DiagOpts`](crate::diagnostics::DiagOpts) once via
//! [`DiagOpts::detect`](crate::diagnostics::DiagOpts::detect).
//!
//! The OSC 8 hyperlink probe is process-stable with first-call-wins semantics
//! (a `OnceLock`), because the underlying environment table cannot change
//! during a single CLI run and the probe is on the per-diagnostic path.

use std::borrow::Cow;
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

/// The colour decision and the depth it should be rendered at, computed from
/// one consistent environment snapshot so `detect_color` and
/// `detect_color_depth` cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorCaps {
    /// Whether ANSI colour should be emitted on `stderr`.
    pub color: bool,
    /// The palette depth to render at.
    pub depth: ColorDepth,
}

/// All terminal capabilities from one environment snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermCaps {
    pub color: bool,
    pub depth: ColorDepth,
    pub unicode: bool,
    pub width: usize,
}

impl TermCaps {
    pub fn detect() -> Self {
        let stderr_is_tty = io::stderr().is_terminal();
        let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
        let colorterm_env = std::env::var_os("COLORTERM");
        let term_env = std::env::var_os("TERM");
        let term_program_env = std::env::var_os("TERM_PROGRAM");
        let colorterm = colorterm_env.as_deref().and_then(OsStr::to_str);
        let term = term_env.as_deref().and_then(OsStr::to_str);
        let term_program = term_program_env.as_deref().and_then(OsStr::to_str);
        let clicolor = std::env::var_os("CLICOLOR");
        let clicolor_force = std::env::var_os("CLICOLOR_FORCE");
        let force_color = std::env::var_os("FORCE_COLOR");
        let wt_session = std::env::var_os("WT_SESSION").is_some_and(|v| !v.is_empty());
        let caps = color_caps_decision_with_terminal(
            no_color,
            clicolor.as_deref(),
            clicolor_force.as_deref(),
            force_color.as_deref(),
            colorterm,
            term,
            wt_session,
            term_program,
            stderr_is_tty,
        );
        let locale = ["LC_ALL", "LC_CTYPE", "LANG"]
            .iter()
            .find_map(|v| std::env::var_os(v).filter(|s| !s.is_empty()));
        let term_dumb = term == Some("dumb");
        let columns = std::env::var_os("COLUMNS");
        TermCaps {
            color: caps.color,
            depth: caps.depth,
            unicode: unicode_decision_os(locale.as_deref(), term_dumb),
            width: width_decision(
                columns.as_deref().and_then(OsStr::to_str),
                terminal_size::terminal_size_of(io::stderr())
                    .map(|(Width(w), _)| w as usize),
            ),
        }
    }
}

/// Snapshot the colour-related environment once, so `color` and `depth` can
/// never describe different environments.
pub fn detect_color_caps() -> ColorCaps {
    color_caps_from_env(io::stderr().is_terminal())
}

/// ASCII-only, allocation-free case-insensitive substring test.
fn contains_ignore_ascii_case(hay: &[u8], needle: &[u8]) -> bool {
    needle.is_empty()
        || (hay.len() >= needle.len() && hay.windows(needle.len()).any(|w| w.eq_ignore_ascii_case(needle)))
}

/// ASCII-only, allocation-free case-insensitive suffix test.
fn ends_with_ignore_ascii_case(hay: &[u8], needle: &[u8]) -> bool {
    hay.len() >= needle.len() && hay[hay.len() - needle.len()..].eq_ignore_ascii_case(needle)
}

/// Node/chalk-compatible `FORCE_COLOR` level: `1` → 16 colours,
/// `2` → 256, `3` → true colour.  Anything else carries no level.
fn force_color_level(v: &OsStr) -> Option<ColorDepth> {
    let s = v.to_str()?.trim();
    match s {
        "" | "true" => Some(ColorDepth::Sixteen),
        "false" | "0" => None,
        _ => s.parse::<u8>().ok().and_then(|n| match n {
            0 => None,
            1 => Some(ColorDepth::Sixteen),
            2 => Some(ColorDepth::TwoFiftySix),
            _ => Some(ColorDepth::TrueColor),
        }),
    }
}

/// `FORCE_COLOR=0`/`false` is a *hard off* in the Node/chalk convention (not
/// merely "don't force"): a CI script silencing colour must not get colour on
/// a TTY.
fn force_color_is_off(v: &OsStr) -> bool {
    matches!(v.to_str().map(str::trim), Some("0") | Some("false"))
}

/// Detect the terminal's colour depth from `FORCE_COLOR`, `COLORTERM` and
/// `TERM`.
///
/// Precedence:
///   1. `FORCE_COLOR=1|2|3` → 16/256/24-bit respectively (Node/chalk levels).
///   2. `COLORTERM` contains `truecolor` or `24bit` → [`ColorDepth::TrueColor`]
///   3. `TERM` ends with `truecolor` or `direct` → [`ColorDepth::TrueColor`]
///   4. `TERM` ends with `256color` → [`ColorDepth::TwoFiftySix`]
///   5. Otherwise → [`ColorDepth::Sixteen`]
pub fn detect_color_depth() -> ColorDepth {
    detect_color_caps().depth
}

/// [`color_depth_decision_with_terminal`] without the terminal-family
/// signals (kept for the pure decision core and its tests).
pub(crate) fn color_depth_decision(
    colorterm: Option<&str>,
    term: Option<&str>,
    force_color: Option<&OsStr>,
) -> ColorDepth {
    color_depth_decision_with_terminal(colorterm, term, force_color, false, None)
}

/// [`color_depth_decision`] plus the terminal-family signals that some
/// environments expose *instead of* `COLORTERM`.
pub(crate) fn color_depth_decision_with_terminal(
    colorterm: Option<&str>,
    term: Option<&str>,
    force_color: Option<&OsStr>,
    wt_session: bool,
    term_program: Option<&str>,
) -> ColorDepth {
    if let Some(depth) = force_color.and_then(force_color_level) {
        return depth;
    }
    // Windows Terminal and the iTerm/WezTerm/VS Code/Warp family are
    // truecolor even when `COLORTERM` is unset (older builds, some remotes).
    if wt_session {
        return ColorDepth::TrueColor;
    }
    if let Some(tp) = term_program {
        let tp = tp.as_bytes();
        if [b"iterm".as_slice(), b"wezterm".as_slice(), b"vscode".as_slice(), b"warp".as_slice()]
            .into_iter()
            .any(|n| contains_ignore_ascii_case(tp, n))
        {
            return ColorDepth::TrueColor;
        }
    }
    if let Some(ct) = colorterm {
        let ct = ct.as_bytes();
        if contains_ignore_ascii_case(ct, b"truecolor") || contains_ignore_ascii_case(ct, b"24bit") {
            return ColorDepth::TrueColor;
        }
    }
    if let Some(t) = term {
        let t = t.as_bytes();
        if ends_with_ignore_ascii_case(t, b"truecolor") || ends_with_ignore_ascii_case(t, b"direct") {
            return ColorDepth::TrueColor;
        }
        if ends_with_ignore_ascii_case(t, b"256color") {
            return ColorDepth::TwoFiftySix;
        }
    }
    ColorDepth::Sixteen
}

/// Whether ANSI colour should be emitted on stderr.
///
/// Precedence (matching `anstream`/`should-color`):
///   1. `NO_COLOR` set to a non-empty value → off (per no-color.org, the
///      empty string means "unset").
///   2. `CLICOLOR_FORCE` set to anything other than `"0"` → on.
///   3. `FORCE_COLOR=0|false` → off; any other value → on (Node semantics).
///   4. `CLICOLOR=0` → off.
///   5. `TERM=dumb` → off.
///   6. Otherwise: on iff `stderr` is a terminal.
pub fn detect_color() -> bool {
    detect_color_caps().color
}

/// Snapshot the colour-related environment once and fold it through the pure
/// decision cores.  Kept private so no caller can observe a half-updated
/// environment.
fn color_caps_from_env(stderr_is_tty: bool) -> ColorCaps {
    // no-color.org: "when present and not an empty string".  CI harnesses set
    // `NO_COLOR=""` to mean "inherit", so an empty value must NOT disable.
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
    let clicolor = std::env::var_os("CLICOLOR");
    let clicolor_force = std::env::var_os("CLICOLOR_FORCE");
    let force_color = std::env::var_os("FORCE_COLOR");
    let wt_session = std::env::var_os("WT_SESSION").is_some_and(|v| !v.is_empty());
    let colorterm_env = std::env::var_os("COLORTERM");
    let term_env = std::env::var_os("TERM");
    let term_program_env = std::env::var_os("TERM_PROGRAM");
    let colorterm = colorterm_env.as_deref().and_then(OsStr::to_str);
    let term = term_env.as_deref().and_then(OsStr::to_str);
    let term_program = term_program_env.as_deref().and_then(OsStr::to_str);
    color_caps_decision_with_terminal(
        no_color,
        clicolor.as_deref(),
        clicolor_force.as_deref(),
        force_color.as_deref(),
        colorterm,
        term,
        wt_session,
        term_program,
        stderr_is_tty,
    )
}

/// Pure colour/depth decision core, factored out so the policy can be tested
/// without mutating process-global environment variables.
pub(crate) fn color_caps_decision(
    no_color: bool,
    clicolor: Option<&OsStr>,
    clicolor_force: Option<&OsStr>,
    force_color: Option<&OsStr>,
    colorterm: Option<&str>,
    term: Option<&str>,
    stderr_is_tty: bool,
) -> ColorCaps {
    color_caps_decision_with_terminal(
        no_color,
        clicolor,
        clicolor_force,
        force_color,
        colorterm,
        term,
        false,
        None,
        stderr_is_tty,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn color_caps_decision_with_terminal(
    no_color: bool,
    clicolor: Option<&OsStr>,
    clicolor_force: Option<&OsStr>,
    force_color: Option<&OsStr>,
    colorterm: Option<&str>,
    term: Option<&str>,
    wt_session: bool,
    term_program: Option<&str>,
    stderr_is_tty: bool,
) -> ColorCaps {
    let color = color_decision(
        no_color,
        clicolor,
        clicolor_force,
        force_color,
        term.map(OsStr::new),
        stderr_is_tty,
    );
    ColorCaps {
        color,
        depth: color_depth_decision_with_terminal(
            colorterm,
            term,
            force_color,
            wt_session,
            term_program,
        ),
    }
}

/// Pure decision core of [`detect_color`].
pub(crate) fn color_decision(
    no_color: bool,
    clicolor: Option<&OsStr>,
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
    if let Some(v) = force_color {
        return !force_color_is_off(v);
    }
    if let Some(v) = clicolor
        && v == "0"
    {
        return false;
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
/// `TERM=dumb` is a *veto*: a terminal that cannot render control sequences
/// cannot render box-drawing either, even when the locale claims UTF-8.
///
/// This is deliberately **independent** of [`detect_color`]: a CI log with
/// colour forced on but `LANG=C` must still get ASCII box-drawing, not
/// mojibake.
pub fn detect_unicode() -> bool {
    let locale = ["LC_ALL", "LC_CTYPE", "LANG"]
        .iter()
        .find_map(|v| std::env::var_os(v).filter(|s| !s.is_empty()));
    let term_dumb = std::env::var_os("TERM").as_deref() == Some(OsStr::new("dumb"));
    unicode_decision_os(locale.as_deref(), term_dumb)
}

fn unicode_decision_os(locale: Option<&OsStr>, term_dumb: bool) -> bool {
    if term_dumb {
        return false;
    }
    match locale {
        Some(v) => {
            let v = v.as_encoded_bytes();
            contains_ignore_ascii_case(v, b"utf-8") || contains_ignore_ascii_case(v, b"utf8")
        }
        None => true,
    }
}

pub(crate) fn unicode_decision(locale: Option<&str>, term_dumb: bool) -> bool {
    match locale {
        Some(v) if !v.is_empty() => unicode_decision_os(Some(OsStr::new(v)), term_dumb),
        _ => unicode_decision_os(None, term_dumb),
    }
}

/// Target output width in display columns.
///
/// `COLUMNS` wins when set and parseable to a positive integer, capped at
/// [`u16::MAX`] (no real terminal — and no `terminal_size::Width` — can exceed
/// it, and an unbounded value poisons downstream `width - used` arithmetic);
/// otherwise the *stderr* terminal size is probed (diagnostics are written
/// there), falling back to [`DEFAULT_MAX_WIDTH`] when the probe returns
/// `None`.
pub fn detect_width() -> usize {
    let columns = std::env::var("COLUMNS").ok();
    width_decision(
        columns.as_deref(),
        // Diagnostics are rendered to stderr, so size *that* stream: probing
        // stdout would report the wrong width under `prog > out.log`.
        terminal_size::terminal_size_of(io::stderr()).map(|(Width(w), _)| w as usize),
    )
}

pub(crate) fn width_decision(columns: Option<&str>, term_size: Option<usize>) -> usize {
    columns
        .map(str::trim)
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&w| w > 0)
        .map(|w| w.min(u16::MAX as usize))
        .or(term_size.filter(|&w| w > 0))
        .unwrap_or(DEFAULT_MAX_WIDTH)
}

/// Whether the terminal can render OSC 8 hyperlinks (`ESC ] 8 ; ; URL ST`).
///
/// `NO_HYPERLINK` set to a non-empty value forces off (empty means unset,
/// mirroring `NO_COLOR`).  Everything else — `FORCE_HYPERLINK` semantics,
/// including `FORCE_HYPERLINK=0` forcing off and forcing links through a pipe,
/// plus the curated terminal table and the `stderr` TTY gate — is delegated to
/// the [`supports-hyperlinks`] crate via its `on(Stream)` entry point.  The
/// answer is memoized because the environment cannot change during a run and
/// this probe sits on the per-diagnostic path.
static HYPERLINK_CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn supports_hyperlinks() -> bool {
    *HYPERLINK_CACHE.get_or_init(|| {
        let no_hyperlink = std::env::var_os("NO_HYPERLINK").is_some_and(|v| !v.is_empty());
        if no_hyperlink {
            false
        } else {
            hyperlink_decision(false, supports_hyperlinks::on(supports_hyperlinks::Stream::Stderr))
        }
    })
}

pub(crate) fn hyperlink_decision(no_hyperlink: bool, probe_on_stderr: bool) -> bool {
    !no_hyperlink && probe_on_stderr
}

/// Whether `b` may appear literally in a `file://` URL path (RFC 3986
/// unreserved characters plus the path delimiters this crate emits).
const URL_SAFE: [bool; 256] = {
    let mut table = [false; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = matches!(
            i as u8,
            b'A'..=b'Z'
                | b'a'..=b'z'
                | b'0'..=b'9'
                | b'/'
                | b':'
                | b'-'
                | b'_'
                | b'.'
                | b'~'
        );
        i += 1;
    }
    table
};

#[inline]
fn is_url_safe(b: u8) -> bool {
    URL_SAFE[b as usize]
}

const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

/// Append the percent-encoded form of `s` to `out`.
///
/// Operates on *bytes*, not `str`: a Unix filename is arbitrary bytes, and a
/// `to_string_lossy` round-trip would percent-encode U+FFFD instead of the
/// original bytes, producing a URL that points at a file that does not exist.
/// The output is ASCII by construction.
/// Length of a Windows verbatim prefix that may be stripped, or `None`.
///
/// Stripping is only valid when the remainder actually looks like a Windows
/// path (drive `X:` or a UNC share).  On POSIX a `\` is a legal filename byte,
/// so a file literally named `\\?\foo` must keep its bytes instead of being
/// rewritten into a different URL.
fn verbatim_prefix_len(bytes: &[u8]) -> Option<usize> {
    if !cfg!(windows) {
        return None;
    }
    const UNC: &[u8] = b"\\\\?\\UNC\\";
    if bytes.len() >= UNC.len() && bytes[..UNC.len()].eq_ignore_ascii_case(UNC) {
        let rest = &bytes[UNC.len()..];
        if rest.contains(&b'\\') {
            return Some(UNC.len());
        }
    }
    for prefix in [b"\\\\?\\".as_slice(), b"\\\\.\\".as_slice()] {
        if let Some(rest) = bytes.strip_prefix(prefix) {
            let drive = rest.len() >= 2 && rest[0].is_ascii_alphabetic() && rest[1] == b':';
            if drive {
                return Some(prefix.len());
            }
        }
    }
    None
}

fn scan_url_path(s: &[u8]) -> (usize, usize) {
    let mut unsafe_count = 0usize;
    let mut first_unsafe = s.len();
    for (i, &b) in s.iter().enumerate() {
        if !is_url_safe(b) {
            unsafe_count += 1;
            if first_unsafe == s.len() {
                first_unsafe = i;
            }
        }
    }
    (first_unsafe, unsafe_count)
}

fn encode_scanned_path(out: &mut String, s: &[u8], first_unsafe: usize, unsafe_count: usize) {
    out.reserve(s.len().saturating_add(unsafe_count.saturating_mul(2)));
    // A safe byte is ASCII by construction (URL_SAFE only contains ASCII),
    // so every safe run is valid UTF-8 and can be appended without a copy.
    let push_safe = |out: &mut String, bytes: &[u8]| {
        out.push_str(std::str::from_utf8(bytes).expect("URL-safe bytes are ASCII"));
    };
    push_safe(out, &s[..first_unsafe]);
    let mut i = first_unsafe;
    while i < s.len() {
        if is_url_safe(s[i]) {
            let start = i;
            while i < s.len() && is_url_safe(s[i]) {
                i += 1;
            }
            push_safe(out, &s[start..i]);
        } else {
            let b = s[i];
            out.push('%');
            out.push(HEX_UPPER[(b >> 4) as usize] as char);
            out.push(HEX_UPPER[(b & 0xF) as usize] as char);
            i += 1;
        }
    }
}

fn encode_file_url_path_into(out: &mut String, s: &[u8]) {
    let (first_unsafe, unsafe_count) = scan_url_path(s);
    if first_unsafe == s.len() {
        out.push_str(std::str::from_utf8(s).expect("URL-safe bytes are ASCII"));
    } else {
        encode_scanned_path(out, s, first_unsafe, unsafe_count);
    }
}

/// Percent-encode the bytes that are unsafe in a `file://` URL path.
///
/// Keeps unreserved characters plus `/`, `:`, `-`, `_`, `.`, `~`; encodes
/// everything else (including `\x1b`, `\x07`, space, `#`, `?`, `%`).  This
/// keeps a filename from terminating the enclosing OSC 8 hyperlink early or
/// injecting terminal control codes.
pub(crate) fn encode_file_url_path(s: &[u8]) -> Cow<'_, str> {
    let (first_unsafe, unsafe_count) = scan_url_path(s);
    if first_unsafe == s.len() {
        // As above: the safe set is ASCII, so this is valid UTF-8.
        return Cow::Borrowed(std::str::from_utf8(s).expect("URL-safe bytes are ASCII"));
    }
    let mut out = String::with_capacity(s.len().saturating_add(unsafe_count.saturating_mul(2)));
    encode_scanned_path(&mut out, s, first_unsafe, unsafe_count);
    Cow::Owned(out)
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
///
/// Only *relative* paths are canonicalised.  An absolute path is already a
/// valid URL path, and resolving symlinks there would both change the
/// user-visible file the hyperlink targets (breaking workspace-relative
/// tooling) and pay a multi-syscall chain per diagnostic — expensive on
/// NFS/WSL mounts.
///
/// A relative path that does not exist yet (a not-yet-written output file)
/// cannot be canonicalised, so it is joined onto the process CWD instead of
/// being emitted as `file://<relative>` — where the first component would be
/// parsed as a URL *host*.  The URL is inherently host-specific either way
/// (`canonicalize` embeds an absolute path), so this changes no contract.
pub fn path_to_file_url(path: &std::path::Path) -> String {
    if path.as_os_str().is_empty() {
        return "file://".to_string();
    }
    let abs_path = if path.is_absolute() {
        Cow::Borrowed(path)
    } else {
        match path.canonicalize() {
            Ok(p) => Cow::Owned(p),
            Err(_) => {
                let base = std::env::current_dir()
                    .unwrap_or_else(|_| std::path::PathBuf::from(std::path::MAIN_SEPARATOR_STR));
                Cow::Owned(base.join(path))
            }
        }
    };
    let raw = abs_path.as_os_str().as_encoded_bytes();
    if raw.is_empty() {
        return "file://".to_string();
    }

    // `canonicalize()` on Windows returns verbatim paths (`\\?\C:\...`).
    // Strip the prefix byte-wise; the RFC 8089 canonical UNC form is
    // `file://server/share/file` — the authority carries the host, so the
    // remaining `server/share/file` is appended after `file://` by
    // `format_file_url`.  Only strip when the remainder really is a Windows
    // path shape: on POSIX `\` is a filename byte, and a name like `\\?\foo`
    // must not be rewritten.
    let strip = verbatim_prefix_len(raw);
    if cfg!(windows)
        && strip.is_none()
        && (raw.starts_with(b"\\\\?\\") || raw.starts_with(b"\\\\.\\"))
    {
        // Volume-GUID / device paths have no safe `file://` representation.
        return "file://".to_string();
    }
    // Windows paths use `\`, but URLs require `/`.  Gate strictly on the
    // platform: on POSIX a `\` is a legal filename byte and must be
    // percent-encoded (`%5C`), not rewritten into a path separator (which
    // would point the hyperlink at a different, nonexistent file).
    let slash = cfg!(windows) && raw.contains(&b'\\');
    // The absolute-POSIX common case needs no copy at all: borrow the path
    // bytes straight into the encoder.
    let bytes: Cow<'_, [u8]> = if strip.is_some() || slash {
        let mut owned = raw.to_vec();
        if let Some(n) = strip {
            owned.drain(..n);
        }
        if slash {
            for b in &mut owned {
                if *b == b'\\' {
                    *b = b'/';
                }
            }
        }
        Cow::Owned(owned)
    } else {
        Cow::Borrowed(raw)
    };

    let distro = std::env::var("WSL_DISTRO_NAME")
        .ok()
        .filter(|s| !s.is_empty());
    format_file_url(&bytes, distro.as_deref())
}

fn format_file_url(path: &[u8], distro: Option<&str>) -> String {
    let mut out = String::with_capacity(path.len() + distro.map_or(0, str::len) + 32);
    if let Some(distro) = distro
        && !distro.is_empty()
    {
        if let Some(stripped) = path.strip_prefix(b"/mnt/") {
            // Bare drive root (`/mnt/c`, no trailing slash) translates too:
            // it becomes `/C:/`, the native-drive URL for the drive root.
            if stripped.len() == 1 && stripped[0].is_ascii_alphabetic() {
                out.push_str("file:///");
                out.push(stripped[0].to_ascii_uppercase() as char);
                out.push_str(":/");
                return out;
            }
            if stripped.len() >= 2 && stripped[1] == b'/' && stripped[0].is_ascii_alphabetic() {
                out.push_str("file:///");
                out.push(stripped[0].to_ascii_uppercase() as char);
                out.push(':');
                encode_file_url_path_into(&mut out, &stripped[1..]);
                return out;
            }
        }
        out.push_str("file://wsl.localhost/");
        encode_file_url_path_into(&mut out, distro.as_bytes());
        encode_file_url_path_into(&mut out, path);
        return out;
    }

    // Windows drive letter: `C:/foo` → `file:///C:/foo` (three slashes).
    // An *uppercase* drive letter is accepted everywhere (a test/host may pass
    // a Windows path on POSIX); a lowercase `c:/x` is treated as an ordinary
    // POSIX directory name unless we are actually on Windows or a WSL distro
    // is in play.
    let looks_like_drive = path.len() >= 2
        && path[1] == b':'
        && (cfg!(windows) || distro.is_some() || path[0].is_ascii_uppercase());
    if looks_like_drive {
        out.push_str("file:///");
        encode_file_url_path_into(&mut out, path);
        return out;
    }

    // POSIX `//server/share/file` is the legacy authority spelling; emit the
    // RFC 8089 canonical form `file://server/share/file` instead of the
    // four-slash form strict parsers treat differently.
    if let Some(rest) = path.strip_prefix(b"//") {
        let looks_like_drive = rest.len() >= 2
            && rest[1] == b':'
            && (cfg!(windows) || distro.is_some() || rest[0].is_ascii_uppercase());
        if looks_like_drive {
            out.push_str("file:///");
        } else {
            out.push_str("file://");
        }
        encode_file_url_path_into(&mut out, rest);
        return out;
    }

    out.push_str("file://");
    encode_file_url_path_into(&mut out, path);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color(
        no_color: bool,
        clicolor: Option<&str>,
        clicolor_force: Option<&str>,
        force_color: Option<&str>,
        term: Option<&str>,
        tty: bool,
    ) -> bool {
        color_decision(
            no_color,
            clicolor.map(OsStr::new),
            clicolor_force.map(OsStr::new),
            force_color.map(OsStr::new),
            term.map(OsStr::new),
            tty,
        )
    }

    #[test]
    fn no_color_beats_everything() {
        assert!(!color(true, None, Some("1"), Some("1"), Some("xterm-256color"), true));
        assert!(!color(true, None, None, None, None, true));
    }

    /// C5: a relative path that does not exist yet cannot be canonicalised,
    /// but must still become an absolute `file:///` URL — emitting
    /// `file://src/main.rs` would parse `src` as the URL *host*.
    #[test]
    fn relative_nonexistent_path_still_becomes_absolute_url() {
        let url = path_to_file_url(std::path::Path::new("definitely/missing/ponent_probe.rs"));
        // A WSL host rewrites the absolute path into the `wsl.localhost`
        // authority; both forms are absolute and neither leaks the first
        // relative component into the authority.
        assert!(
            url.starts_with("file:///") || url.starts_with("file://wsl.localhost/"),
            "not absolute: {url}"
        );
        assert!(
            !url.starts_with("file://definitely"),
            "first component leaked into the URL authority: {url}"
        );
        assert!(url.contains("definitely/missing/ponent_probe.rs"), "{url}");
    }

    #[test]
    fn clicolor_force_only_forces_on_a_nonzero_value() {
        assert!(color(false, None, Some("1"), None, Some("dumb"), false));
        assert!(color(false, None, Some(""), None, Some("dumb"), false));
        // `0` means "don't force": fall through to the terminal check.
        assert!(!color(false, None, Some("0"), None, Some("xterm"), false));
        assert!(color(false, None, Some("0"), None, Some("xterm"), true));
    }

    #[test]
    fn force_color_follows_node_semantics() {
        // Any level forces on.
        for level in ["1", "2", "3", ""] {
            assert!(color(false, None, None, Some(level), Some("dumb"), false));
        }
        // `0`/`false` is a hard off — even on a TTY.
        assert!(!color(false, None, None, Some("0"), Some("xterm"), true));
        assert!(!color(false, None, None, Some("false"), Some("xterm"), true));
        // CLICOLOR_FORCE takes precedence over FORCE_COLOR.
        assert!(color(false, None, Some("1"), Some("0"), Some("dumb"), false));
        // NO_COLOR still wins over FORCE_COLOR.
        assert!(!color(true, None, None, Some("3"), Some("xterm"), true));
    }

    #[test]
    fn clicolor_zero_disables() {
        assert!(!color(false, Some("0"), None, None, Some("xterm"), true));
        assert!(color(false, Some("1"), None, None, Some("xterm"), true));
        // CLICOLOR_FORCE overrides CLICOLOR=0.
        assert!(color(false, Some("0"), Some("1"), None, Some("xterm"), false));
    }

    #[test]
    fn dumb_terminal_disables_color() {
        assert!(!color(false, None, None, None, Some("dumb"), true));
    }

    #[test]
    fn empty_no_color_does_not_disable() {
        // The empty string means "unset" per no-color.org; `detect_color`
        // implements that via `var_os().is_some_and(|v| !v.is_empty())`.
        assert!(color(false, None, None, None, Some("xterm"), true));
    }

    #[test]
    fn locale_utf8_detection() {
        assert!(unicode_decision(Some("en_US.UTF-8"), false));
        assert!(unicode_decision(Some("C.utf8"), false));
        assert!(!unicode_decision(Some("C"), false));
        assert!(!unicode_decision(Some("POSIX"), false));
        // No locale information at all → assume UTF-8.
        assert!(unicode_decision(None, false));
        // `TERM=dumb` is a veto regardless of locale.
        assert!(!unicode_decision(Some("en_US.UTF-8"), true));
    }

    #[test]
    fn width_honours_columns_and_rejects_junk() {
        assert_eq!(width_decision(Some("120"), None), 120);
        assert_eq!(width_decision(Some(" 80 "), None), 80);
        assert_eq!(width_decision(Some("0"), None), DEFAULT_MAX_WIDTH);
        assert_eq!(width_decision(Some("nope"), None), DEFAULT_MAX_WIDTH);
        assert_eq!(width_decision(None, None), DEFAULT_MAX_WIDTH);
        // terminal_size wins when COLUMNS is absent.
        assert_eq!(width_decision(None, Some(80)), 80);
        assert_eq!(width_decision(None, Some(0)), DEFAULT_MAX_WIDTH);
    }

    #[test]
    fn width_is_capped_to_u16_max() {
        // An unbounded COLUMNS must not poison downstream arithmetic.
        assert_eq!(width_decision(Some("18446744073709551615"), None), u16::MAX as usize);
        assert_eq!(width_decision(Some("65536"), None), u16::MAX as usize);
    }

    #[test]
    fn terminal_family_signals_select_truecolor() {
        // WT_SESSION is the only truecolor signal on older Windows Terminal.
        assert_eq!(
            color_depth_decision_with_terminal(None, Some("xterm"), None, true, None),
            ColorDepth::TrueColor
        );
        for tp in ["iTerm.app", "WezTerm", "vscode", "WarpTerminal"] {
            assert_eq!(
                color_depth_decision_with_terminal(None, Some("xterm"), None, false, Some(tp)),
                ColorDepth::TrueColor,
                "{tp}"
            );
        }
        // An unrelated TERM_PROGRAM must not upgrade the depth.
        assert_eq!(
            color_depth_decision_with_terminal(None, Some("xterm"), None, false, Some("Apple_Terminal")),
            ColorDepth::Sixteen
        );
    }

    #[test]
    fn force_color_true_and_empty_pin_sixteen() {
        // Node maps `true`/empty to level 1; the depth decision must not
        // escape the force and resolve to a higher depth via COLORTERM.
        for v in ["true", ""] {
            assert_eq!(
                color_depth_decision(Some("truecolor"), Some("xterm"), Some(OsStr::new(v))),
                ColorDepth::Sixteen,
                "FORCE_COLOR={v:?}"
            );
        }
    }

    #[test]
    fn verbatim_prefix_only_stripped_for_windows_shapes() {
        let windows = if cfg!(windows) { Some(4) } else { None };
        assert_eq!(verbatim_prefix_len(b"\\\\?\\C:\\x"), windows);
        let unc = if cfg!(windows) { Some(8) } else { None };
        assert_eq!(verbatim_prefix_len(b"\\\\?\\UNC\\server\\share"), unc);
        assert_eq!(verbatim_prefix_len(b"\\\\?\\unc\\server\\share"), unc);
        // A POSIX filename that merely starts with the prefix bytes keeps them.
        assert_eq!(verbatim_prefix_len(b"\\\\?\\foo"), None);
        assert_eq!(verbatim_prefix_len(b"\\\\.\\bar"), None);
    }

    #[test]
    fn color_depth_detection() {
        // COLORTERM=truecolor → TrueColor
        assert_eq!(
            color_depth_decision(Some("truecolor"), Some("xterm"), None),
            ColorDepth::TrueColor
        );
        // COLORTERM=24bit → TrueColor (case-insensitive)
        assert_eq!(
            color_depth_decision(Some("24BIT"), None, None),
            ColorDepth::TrueColor
        );
        // TERM ends with 256color → TwoFiftySix (case-insensitive).
        assert_eq!(
            color_depth_decision(None, Some("xterm-256color"), None),
            ColorDepth::TwoFiftySix
        );
        assert_eq!(
            color_depth_decision(None, Some("XTERM-256COLOR"), None),
            ColorDepth::TwoFiftySix
        );
        // Modern 24-bit TERM conventions.
        for term in ["iterm2-direct", "xterm-direct", "st-direct", "foot-direct", "xterm-truecolor"] {
            assert_eq!(
                color_depth_decision(None, Some(term), None),
                ColorDepth::TrueColor,
                "{term}"
            );
        }
        // Neither → Sixteen
        assert_eq!(
            color_depth_decision(None, Some("xterm"), None),
            ColorDepth::Sixteen
        );
        assert_eq!(color_depth_decision(None, None, None), ColorDepth::Sixteen);
        // COLORTERM takes precedence over TERM.
        assert_eq!(
            color_depth_decision(Some("truecolor"), Some("xterm"), None),
            ColorDepth::TrueColor
        );
    }

    #[test]
    fn force_color_level_clamps_and_trims() {
        assert_eq!(force_color_level(OsStr::new("4")), Some(ColorDepth::TrueColor));
        assert_eq!(force_color_level(OsStr::new(" 2")), Some(ColorDepth::TwoFiftySix));
        assert_eq!(force_color_level(OsStr::new("03")), Some(ColorDepth::TrueColor));
        assert_eq!(force_color_level(OsStr::new("0")), None);
        assert!(force_color_is_off(OsStr::new(" false ")));
    }

    #[test]
    fn empty_locale_and_needle_are_unset() {
        assert!(unicode_decision(Some(""), false));
        assert!(contains_ignore_ascii_case(b"anything", b""));
    }

    #[test]
    fn force_color_level_sets_depth() {
        assert_eq!(
            color_depth_decision(None, Some("xterm"), Some(OsStr::new("1"))),
            ColorDepth::Sixteen
        );
        assert_eq!(
            color_depth_decision(None, Some("xterm"), Some(OsStr::new("2"))),
            ColorDepth::TwoFiftySix
        );
        assert_eq!(
            color_depth_decision(None, Some("xterm"), Some(OsStr::new("3"))),
            ColorDepth::TrueColor
        );
        // Level wins over COLORTERM.
        assert_eq!(
            color_depth_decision(Some("truecolor"), Some("xterm"), Some(OsStr::new("1"))),
            ColorDepth::Sixteen
        );
    }

    #[test]
    fn caps_are_coherent() {
        let caps = color_caps_decision(
            false,
            None,
            None,
            Some(OsStr::new("2")),
            None,
            Some("xterm-256color"),
            true,
        );
        assert!(caps.color);
        assert_eq!(caps.depth, ColorDepth::TwoFiftySix);

        let caps = color_caps_decision(
            false,
            None,
            None,
            Some(OsStr::new("0")),
            Some("truecolor"),
            Some("xterm"),
            true,
        );
        assert!(!caps.color);
    }

    #[test]
    fn hyperlink_overrides() {
        assert!(!hyperlink_decision(true, true));
        assert!(!hyperlink_decision(false, false));
        assert!(hyperlink_decision(false, true));
    }

    #[test]
    fn encoding_is_byte_exact() {
        // Safe paths borrow.
        assert!(matches!(
            encode_file_url_path(b"/home/u/src/main.rs"),
            Cow::Borrowed(_)
        ));
        assert_eq!(encode_file_url_path(b"a b").as_ref(), "a%20b");
        assert_eq!(encode_file_url_path(b"\x1b]8;;").as_ref(), "%1B%5D8%3B%3B");
        // Non-UTF-8 bytes are encoded as the original bytes, not U+FFFD.
        assert_eq!(encode_file_url_path(&[0xFF]).as_ref(), "%FF");
        assert_eq!(encode_file_url_path(b"").as_ref(), "");
    }

    #[test]
    fn non_utf8_path_keeps_original_bytes() {
        // A latin-1 byte in a file name must produce a URL pointing at the
        // real bytes, not `%EF%BF%BD` (U+FFFD).
        let url = format_file_url(b"/tmp/caf\xE9.rs", None);
        assert_eq!(url, "file:///tmp/caf%E9.rs");
        assert!(!url.contains("%EF%BF%BD"));
    }

    #[test]
    fn format_file_url_wsl_and_windows() {
        assert_eq!(
            format_file_url(b"/home/user/file.rs", Some("Ubuntu")),
            "file://wsl.localhost/Ubuntu/home/user/file.rs"
        );
        assert_eq!(
            format_file_url(b"/mnt/c/Users/user/file.rs", Some("Ubuntu")),
            "file:///C:/Users/user/file.rs"
        );
        assert_eq!(format_file_url(b"/mnt/d/", Some("Ubuntu")), "file:///D:/");
        assert_eq!(format_file_url(b"/mnt/c", Some("Ubuntu")), "file:///C:/");
        assert_eq!(
            format_file_url(b"/home/user/file.rs", None),
            "file:///home/user/file.rs"
        );
        assert_eq!(
            format_file_url(b"C:/Users/user/file.rs", None),
            "file:///C:/Users/user/file.rs"
        );
        // RFC 8089 canonical authority form (not the legacy four-slash form).
        assert_eq!(
            format_file_url(b"//server/share/file.rs", None),
            "file://server/share/file.rs"
        );
        assert_eq!(
            format_file_url(b"//C:/Users/user/file.rs", None),
            "file:///C:/Users/user/file.rs"
        );
        assert_eq!(
            format_file_url(b"/home/user/file.rs", Some("Ubuntu 20.04")),
            "file://wsl.localhost/Ubuntu%2020.04/home/user/file.rs"
        );
        // A POSIX directory named `c:/x` is not a Windows drive.
        assert_eq!(format_file_url(b"c:/x", None), "file://c:/x");
    }

    #[test]
    fn empty_path_is_not_garbage() {
        assert_eq!(path_to_file_url(std::path::Path::new("")), "file://");
    }
}