//! Terminal capability detection.
//!
//! Single home for the `NO_COLOR` / `CLICOLOR` / `CLICOLOR_FORCE` /
//! `FORCE_COLOR` / `TERM=dumb` / `isatty` dance plus width, Unicode, and
//! OSC 8 hyperlink probes, so downstream consumers don't each reimplement it
//! (and get it subtly wrong).
//!
//! Every public detector is **process-stable with first-call-wins semantics**:
//! the first call snapshots the environment (and, where relevant, the `stderr`
//! `isatty` state) into a `OnceLock`, and later calls are a single atomic load.
//! The real hazard is therefore *ordering*, not instability — whatever the
//! environment looks like when the first detector runs decides the answer for
//! the whole run, so a CLI flag must be applied through
//! [`DiagOpts`](crate::diagnostics::DiagOpts) rather than by re-probing.
//! [`TermCaps::detect`] is the escape hatch for callers that genuinely want a
//! fresh probe.

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
    pub const fn supports_256(self) -> bool {
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
        // `TERM` is read exactly once and threaded into the colour decision:
        // reading it again below let `color` and `unicode` observe different
        // snapshots (the module's "one canonical env-read path" contract).
        let term_env = std::env::var_os("TERM");
        let term = term_env.as_deref().and_then(OsStr::to_str);
        // One canonical colour env-read path: `color_caps_from_env` performs
        // exactly the reads this function used to duplicate inline, so a
        // future variable cannot be added to one and forgotten in the other.
        let caps = color_caps_from_env(stderr_is_tty, term);
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
                terminal_size::terminal_size_of(io::stderr()).map(|(Width(w), _)| w as usize),
            ),
        }
    }
}

/// Process-stable capability snapshot: the first call wins, later calls are a
/// single atomic load.
///
/// This is the contract [`supports_hyperlinks`] already had, applied to the
/// rest of the module.  The convenience detectors below are public and a
/// caller may reasonably use one per diagnostic; without the snapshot that
/// means an `isatty` syscall (plus ~11 env reads) per call, and a TIOCGWINSZ
/// per `detect_width`.  Tests exercise the pure decision cores
/// (`color`, `color_depth_decision`, `width_decision`, `unicode_decision`)
/// rather than the environment probes, so the snapshot is not observable
/// there.  [`TermCaps::detect`] remains available for a caller that genuinely
/// wants a fresh probe (e.g. after deliberately changing the environment).
static GLOBAL_CAPS: std::sync::OnceLock<TermCaps> = std::sync::OnceLock::new();

fn global_caps() -> &'static TermCaps {
    GLOBAL_CAPS.get_or_init(TermCaps::detect)
}

/// Snapshot the colour-related environment once, so `color` and `depth` can
/// never describe different environments.
pub fn detect_color_caps() -> ColorCaps {
    let caps = global_caps();
    ColorCaps {
        color: caps.color,
        depth: caps.depth,
    }
}

/// ASCII-only, allocation-free case-insensitive substring test.
fn contains_ignore_ascii_case(hay: &[u8], needle: &[u8]) -> bool {
    needle.is_empty()
        || (hay.len() >= needle.len()
            && hay
                .windows(needle.len())
                .any(|w| w.eq_ignore_ascii_case(needle)))
}

/// ASCII-only, allocation-free case-insensitive suffix test.
fn ends_with_ignore_ascii_case(hay: &[u8], needle: &[u8]) -> bool {
    hay.len() >= needle.len() && hay[hay.len() - needle.len()..].eq_ignore_ascii_case(needle)
}

/// Node/chalk-compatible `FORCE_COLOR` level: `1` → 16 colours,
/// `2` → 256, `3` **or any larger numeric value** → true colour.  Anything
/// non-numeric (other than `true`/`false`/empty) carries no level.
/// Matching is case-sensitive (`TRUE` is not `true`), mirroring Node.
fn force_color_level(v: &OsStr) -> Option<ColorDepth> {
    let s = v.to_str()?.trim();
    match s {
        // Deliberate deviation from `supports-color`, which pins `true`/`""` to
        // level 1: pinning 16 colours *downgrades* a truecolor terminal, so we
        // only force colour on (the `force_color_is_off == false` half, handled
        // by `color_decision`) and let COLORTERM/TERM pick the depth.
        // Pinned by `force_color_true_and_empty_detect_depth`.
        "" | "true" => None,
        "false" => None,
        _ => match s.parse::<u128>() {
            Ok(0) => None,
            Ok(1) => Some(ColorDepth::Sixteen),
            Ok(2) => Some(ColorDepth::TwoFiftySix),
            Ok(_) => Some(ColorDepth::TrueColor),
            // All-digit input beyond `u128::MAX` is still a number ≥ 3, so the
            // doc's "any larger numeric value → true colour" holds at *every*
            // magnitude.  Without this arm a 40-digit generated CI string parsed
            // as a failure and silently fell back to depth *detection* — the
            // same boundary `force_color_is_off` solves by widening to `i128`.
            Err(_) if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) => {
                Some(ColorDepth::TrueColor)
            }
            Err(_) => None,
        },
    }
}

/// The three rules, in the order the body applies them (this list IS the
/// contract — the function name reads as "off unless…", which is not it):
///   1. `"false"` → off.
///   2. Any numeric value `≤ 0` → off, including magnitudes beyond `i64`:
///      `supports-color` clamps `level < 0` to 0, so `FORCE_COLOR=-1` from a CI
///      script means "silence colour", not "force it on".
///   3. Anything else — non-numeric garbage (`"abc"`, `"FALSE"`) or a numeric
///      `> 0` — is not an explicit off request, so colour stays on (the caller
///      then detects the depth).
/// `force_color_noncanonical_values_force_on` and
/// `force_color_off_survives_negative_overflow` pin the two edges.
fn force_color_is_off(v: &OsStr) -> bool {
    let Some(s) = v.to_str().map(str::trim) else {
        return false;
    };
    if s == "false" {
        return true;
    }
    // Modern `supports-color` clamps `level < 0` to 0, i.e. OFF: a CI script
    // setting `FORCE_COLOR=-1` means "silence colour", not "force it on".
    // Parsed as `i128`, not `i64`: a negative value *below* `i64::MIN` (a
    // generated CI string, a `-99999999999999999999`) must still read as OFF.
    // With `i64` the parse failed and fell through to "force on" — exactly the
    // bug the clamping rule exists to prevent, one boundary further out.
    // Non-numeric values (`"abc"`, `"FALSE"`) still mean force-on — they are
    // not an explicit off request.
    s.parse::<i128>().is_ok_and(|n| n <= 0)
}

/// Detect the terminal's colour depth from `FORCE_COLOR`, `COLORTERM` and
/// `TERM`.
///
/// Precedence, as implemented by [`color_depth_decision_with_terminal`] (which
/// is the authoritative list — this summary must not drift from it):
///   1. `FORCE_COLOR=1|2|3` → 16/256/24-bit respectively (Node/chalk levels).
///   2. A terminal-family signal *before* `COLORTERM`: `WT_SESSION` (Windows
///      Terminal) → [`ColorDepth::TrueColor`], and an exact `TERM_PROGRAM` in
///      `{iTerm.app, WezTerm, vscode, WarpTerminal, Warp}` → [`ColorDepth::TrueColor`].
///   3. `COLORTERM` contains `truecolor` or `24bit` → [`ColorDepth::TrueColor`]
///   4. `TERM` ends with `truecolor` or `direct` → [`ColorDepth::TrueColor`]
///   5. `TERM` ends with `256color` → [`ColorDepth::TwoFiftySix`]
///   6. Otherwise → [`ColorDepth::Sixteen`]
pub fn detect_color_depth() -> ColorDepth {
    detect_color_caps().depth
}

/// The colour-related environment as one named value.
///
/// Replaces nine positional parameters.  `colorterm` and `term` are both
/// `Option<&str>`, so a swapped pair type-checks and can even *pass* (a
/// `COLORTERM` value shaped like a `TERM`); named fields make every call site
/// auditable and turn the literal into its own documentation.  `Default` is the
/// "nothing set" environment, which is what the decision-core tests want.
#[derive(Clone, Copy, Default)]
pub(crate) struct ColorEnv<'a> {
    /// `NO_COLOR` set to a non-empty value (no-color.org).
    pub no_color: bool,
    pub clicolor: Option<&'a OsStr>,
    pub clicolor_force: Option<&'a OsStr>,
    pub force_color: Option<&'a OsStr>,
    pub colorterm: Option<&'a str>,
    pub term: Option<&'a str>,
    /// `WT_SESSION` set to a non-empty value.
    pub wt_session: bool,
    pub term_program: Option<&'a str>,
    pub terminal_emulator: Option<&'a str>,
    /// `stderr` is a terminal.  Part of the same snapshot on purpose: pairing a
    /// stale TTY probe with a fresh environment is exactly the half-updated view
    /// [`color_caps_from_env`] exists to prevent.
    pub stderr_is_tty: bool,
}

/// The depth-only subset of [`ColorEnv`], for callers that do not care about
/// colour-on/off (and for the pure decision core's tests).
pub(crate) fn color_depth_decision(
    colorterm: Option<&str>,
    term: Option<&str>,
    force_color: Option<&OsStr>,
) -> ColorDepth {
    color_depth_decision_env(&ColorEnv {
        colorterm,
        term,
        force_color,
        ..ColorEnv::default()
    })
}

/// Pure depth decision core: [`detect_color_depth`] without the environment read.
pub(crate) fn color_depth_decision_env(env: &ColorEnv<'_>) -> ColorDepth {
    if let Some(depth) = env.force_color.and_then(force_color_level) {
        return depth;
    }
    // Windows Terminal and the iTerm/WezTerm/VS Code/Warp family are
    // truecolor even when `COLORTERM` is unset (older builds, some remotes).
    if env.wt_session {
        return ColorDepth::TrueColor;
    }
    if let Some(tp) = env.term_program {
        // Exact, case-insensitive values only: `contains` treated
        // "not-iTerm.app" or "vscode-fork" as truecolor terminals.
        if ["iTerm.app", "WezTerm", "vscode", "WarpTerminal", "Warp"]
            .iter()
            .any(|n| tp.eq_ignore_ascii_case(n))
        {
            return ColorDepth::TrueColor;
        }
    }
    // JetBrains terminals (IDEA, CLion, …) set no `COLORTERM` at all, so
    // without this an IDE user with a truecolor terminal gets 256-colour
    // output.  `supports-color` reads the same variable with `includes`.
    if let Some(te) = env.terminal_emulator
        && contains_ignore_ascii_case(te.as_bytes(), b"jetbrains")
    {
        return ColorDepth::TrueColor;
    }
    if let Some(ct) = env.colorterm {
        let ct = ct.as_bytes();
        if contains_ignore_ascii_case(ct, b"truecolor") || contains_ignore_ascii_case(ct, b"24bit")
        {
            return ColorDepth::TrueColor;
        }
    }
    if let Some(t) = env.term {
        let t = t.as_bytes();
        // Some kitty builds do not set `COLORTERM`; `supports-color` special-cases
        // the TERM value itself, and without it those terminals fall through to
        // `Sixteen`.
        if t.eq_ignore_ascii_case(b"xterm-kitty") {
            return ColorDepth::TrueColor;
        }
        if ends_with_ignore_ascii_case(t, b"truecolor") || ends_with_ignore_ascii_case(t, b"direct")
        {
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
///   2. `CLICOLOR_FORCE` set to a non-empty value other than `"0"` → on.
///   3. `FORCE_COLOR` → see [`force_color_is_off`]: `false` and any numeric
///      `≤ 0` are off, non-numeric garbage is on, everything else is on.
///   4. `CLICOLOR=0` → off.
///   5. `TERM=dumb` → off.
///   6. Otherwise: on iff `stderr` is a terminal.
pub fn detect_color() -> bool {
    detect_color_caps().color
}

/// Trimmed view of an env value, with empty mapped to `None`.
///
/// The one convention for the numeric-ish colour switches (`CLICOLOR`,
/// `CLICOLOR_FORCE`, `FORCE_COLOR`): padding is stripped before any comparison,
/// so `" 0"` and `"0"` agree.  Kept separate from the `NO_COLOR` /
/// `WT_SESSION` / `NO_HYPERLINK` "present and non-empty" rule, which is
/// spec-mandated for those (an empty value means "inherit", a padded one is
/// still *present*).
fn env_trimmed(v: &OsStr) -> Option<&str> {
    let s = v.to_str()?.trim();
    (!s.is_empty()).then_some(s)
}

/// Snapshot the colour-related environment once and fold it through the pure
/// decision cores.  Kept private so no caller can observe a half-updated
/// environment.
fn color_caps_from_env(stderr_is_tty: bool, term: Option<&str>) -> ColorCaps {
    // no-color.org: "when present and not an empty string".  CI harnesses set
    // `NO_COLOR=""` to mean "inherit", so an empty value must NOT disable.
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
    let clicolor = std::env::var_os("CLICOLOR");
    let clicolor_force = std::env::var_os("CLICOLOR_FORCE");
    let force_color = std::env::var_os("FORCE_COLOR");
    let wt_session = std::env::var_os("WT_SESSION").is_some_and(|v| !v.is_empty());
    let colorterm_env = std::env::var_os("COLORTERM");
    let term_program_env = std::env::var_os("TERM_PROGRAM");
    let terminal_emulator_env = std::env::var_os("TERMINAL_EMULATOR");
    let colorterm = colorterm_env.as_deref().and_then(OsStr::to_str);
    let term_program = term_program_env.as_deref().and_then(OsStr::to_str);
    let terminal_emulator = terminal_emulator_env.as_deref().and_then(OsStr::to_str);
    color_caps_decision(&ColorEnv {
        no_color,
        clicolor: clicolor.as_deref(),
        clicolor_force: clicolor_force.as_deref(),
        force_color: force_color.as_deref(),
        colorterm,
        term,
        wt_session,
        term_program,
        terminal_emulator,
        stderr_is_tty,
    })
}

/// Pure colour/depth decision core, factored out so the policy can be tested
/// without mutating process-global environment variables.  One entry point: the
/// terminal-family signals live in [`ColorEnv`] instead of in a second
/// positional signature.
pub(crate) fn color_caps_decision(env: &ColorEnv<'_>) -> ColorCaps {
    ColorCaps {
        color: color_decision_env(env),
        depth: color_depth_decision_env(env),
    }
}

/// Pure decision core of [`detect_color`] (the colour-on/off half of
/// [`ColorEnv`] only).
pub(crate) fn color_decision(
    no_color: bool,
    clicolor: Option<&OsStr>,
    clicolor_force: Option<&OsStr>,
    force_color: Option<&OsStr>,
    term: Option<&OsStr>,
    stderr_is_tty: bool,
) -> bool {
    color_decision_env(&ColorEnv {
        no_color,
        clicolor,
        clicolor_force,
        force_color,
        term: term.and_then(OsStr::to_str),
        stderr_is_tty,
        ..ColorEnv::default()
    })
}

/// The single implementation behind [`color_decision`].
pub(crate) fn color_decision_env(env: &ColorEnv<'_>) -> bool {
    if env.no_color {
        return false;
    }
    // `CLICOLOR_FORCE=0` means "don't force", not "force off" — it falls
    // through to normal detection.  An *empty* value is treated as unset, like
    // `NO_COLOR`/`WT_SESSION`/`NO_HYPERLINK` here and like anstream's
    // `anstyle-query`: templated CI matrices routinely materialise empty
    // strings, and force-on-empty pushed colour into piped logs.
    if let Some(v) = env.clicolor_force.and_then(env_trimmed)
        && v != "0"
    {
        return true;
    }
    if let Some(v) = env.force_color {
        return !force_color_is_off(v);
    }
    if let Some(v) = env.clicolor.and_then(env_trimmed)
        && v == "0"
    {
        return false;
    }
    if env.term == Some("dumb") {
        return false;
    }
    env.stderr_is_tty
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
    // Process-stable snapshot (see `GLOBAL_CAPS`): the locale/`TERM` reads and
    // the `isatty`/TIOCGWINSZ probes happen once per process instead of once
    // per optional-detection call.
    global_caps().unicode
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
    // Process-stable snapshot (see `GLOBAL_CAPS`): one TIOCGWINSZ per process
    // instead of one per call.  The `COLUMNS`/`terminal_size_of` derivation
    // lives in `TermCaps::detect`, which still reads `COLUMNS` via `var_os` +
    // `to_str` like the rest of the module.
    global_caps().width
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
            hyperlink_decision(
                false,
                supports_hyperlinks::on(supports_hyperlinks::Stream::Stderr),
            )
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

use crate::diagnostics::HEX_UPPER;

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

fn wsl_distro_name() -> Option<&'static str> {
    static WSL_DISTRO: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    WSL_DISTRO
        .get_or_init(|| {
            let name = std::env::var("WSL_DISTRO_NAME")
                .ok()
                .filter(|s| !s.is_empty())?;
            // No `/proc` to verify against on macOS/BSD, so a forwarded
            // `WSL_DISTRO_NAME` (SSH/CI) would otherwise translate paths into
            // `file://wsl.localhost/...` URLs that point nowhere on that host.
            if !cfg!(target_os = "linux") {
                return None;
            }
            #[cfg(target_os = "linux")]
            {
                // SSH/CI can forward WSL_DISTRO_NAME to a native Linux host.
                // Only translate when the kernel also identifies as WSL; if
                // /proc cannot be read, keep the env-only fallback.
                if let Ok(version) = std::fs::read_to_string("/proc/version") {
                    // Allocation-free and ASCII-case-insensitive, like the rest
                    // of this module: lowercasing copied the whole kernel version
                    // string just to run two substring tests.
                    if !contains_ignore_ascii_case(version.as_bytes(), b"microsoft")
                        && !contains_ignore_ascii_case(version.as_bytes(), b"wsl")
                    {
                        return None;
                    }
                }
            }
            Some(name)
        })
        .as_deref()
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

fn encode_url_component_into(out: &mut String, s: &[u8]) {
    for &b in s {
        if b == b'/' || !is_url_safe(b) {
            out.push('%');
            out.push(HEX_UPPER[(b >> 4) as usize] as char);
            out.push(HEX_UPPER[(b & 0xF) as usize] as char);
        } else {
            out.push(b as char);
        }
    }
}

/// [`encode_file_url_path_pre`] with the scan already done.
///
/// `format_file_url` scans the path anyway to size its buffer, so re-scanning
/// here would be a second O(n) pass on exactly the long, heavily-encoded paths
/// the reserve exists for.
fn encode_file_url_path_pre(out: &mut String, s: &[u8], first_unsafe: usize, unsafe_count: usize) {
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
/// No path is canonicalised.  Relative paths are resolved with
/// [`std::path::absolute`], which joins the process CWD lexically: no syscalls,
/// no symlink resolution, and it works for a file that does not exist yet
/// (`canonicalize` failed on those, which is why the old code paid the syscall
/// chain and then re-joined the CWD anyway).
///
/// Preserving symlinks is deliberate: this is a *diagnostics* system, not a
/// sysroot/extern lookup — the hyperlink must open the file the diagnostic
/// names, not whatever inode a resolution chain ends at.  The two differ on
/// macOS (`/tmp` vs `/private/tmp`), so canonicalising would rewrite the very
/// path the user just read.
pub fn path_to_file_url(path: &std::path::Path) -> std::sync::Arc<str> {
    if path.as_os_str().is_empty() {
        return std::sync::Arc::from("file://");
    }
    let abs_path = if path.is_absolute() {
        Cow::Borrowed(path)
    } else {
        match std::path::absolute(path) {
            Ok(p) => Cow::Owned(p),
            // No CWD means no honest absolute URL.  Emit the explicit sentinel
            // instead of silently linking `/<path>`.
            Err(_) => return std::sync::Arc::from("file://"),
        }
    };
    let raw = abs_path.as_os_str().as_encoded_bytes();
    if raw.is_empty() {
        return std::sync::Arc::from("file://");
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
        return std::sync::Arc::from("file://");
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

    let distro = wsl_distro_name();
    file_url_cached(&bytes, distro)
}

/// Map `path` (already absolute, `/`-separated, verbatim prefix stripped) to its
/// finished `file://` URL, memoised per thread for the life of the process.
///
/// Sound because the URL is a pure function of the path bytes, the
/// process-stable [`wsl_distro_name`] snapshot and compile-time cfg.  Relative
/// inputs are resolved to absolute *before* the lookup, so a changed CWD can
/// never alias an old entry.
///
/// Thread-local rather than a shared map: the emitter renders in parallel, and a
/// global mutex on the per-diagnostic URL path would serialise it for no gain.
/// Each worker warms its own table; the cap bounds a pathological run, and
/// clearing is cheaper than an LRU for this access pattern (diagnostics cluster
/// on a handful of files).
fn file_url_cached(path: &[u8], distro: Option<&str>) -> std::sync::Arc<str> {
    thread_local! {
        static URLS: std::cell::RefCell<rustc_hash::FxHashMap<Box<[u8]>, std::sync::Arc<str>>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    const CACHE_CAP: usize = 1024;
    URLS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(hit) = cache.get(path) {
            return std::sync::Arc::clone(hit);
        }
        let url: std::sync::Arc<str> = std::sync::Arc::from(format_file_url(path, distro));
        if cache.len() >= CACHE_CAP {
            cache.clear();
        }
        cache.insert(
            path.to_vec().into_boxed_slice(),
            std::sync::Arc::clone(&url),
        );
        url
    })
}

fn format_file_url(path: &[u8], distro: Option<&str>) -> String {
    // One scan, in the caller: percent-encoding expands an unsafe byte to three
    // bytes, so the old `path.len() + 32` under-reserved by up to 2× and forced a
    // realloc on exactly the long/heavily-encoded paths this sizing exists for.
    let (first_unsafe, unsafe_count) = scan_url_path(path);
    // Capacity must cover the *longest* prefix this function can emit, not just
    // the path: the WSL form starts with `file://wsl.localhost/` (21 bytes) and
    // percent-encodes the distro name too (`Ubuntu 20.04` → 3× for two bytes),
    // so the old `+ distro.len() + 16` under-reserved by `5 + 2·unsafe_in_distro`
    // on exactly that branch — a guaranteed realloc plus a full-buffer copy per
    // uncached WSL URL.  `distro.len() * 3` is the worst case for the encoded
    // distro; distros are short, so the over-reserve is bounded and the exact
    // arithmetic needs no second scan.
    let mut out = String::with_capacity(
        path.len()
            + unsafe_count.saturating_mul(2)
            + distro.map_or(0, |d| d.len().saturating_mul(3) + 21)
            + 16,
    );
    if let Some(distro) = distro
        && !distro.is_empty()
        // Only POSIX-shaped paths belong under `wsl.localhost`: routing a
        // Windows-native path (`C:/...`, or a verbatim path stripped to that)
        // through this branch produced `file://wsl.localhost/UbuntuC:/...` —
        // a malformed URL with no separator.
        && path.starts_with(b"/")
    {
        // `//server/share/file` is a Windows UNC path as seen from Linux.
        // Routing it through `wsl.localhost` makes the 9P server traverse a
        // CIFS mount — which it usually cannot — and is slower even when it
        // can.  Windows can open the native UNC form directly.
        if let Some(rest) = path.strip_prefix(b"//") {
            out.push_str("file://");
            let (rest_first, rest_unsafe) = scan_url_path(rest);
            encode_file_url_path_pre(&mut out, rest, rest_first, rest_unsafe);
            return out;
        }
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
                let tail = &stripped[1..];
                let (tail_first, tail_unsafe) = scan_url_path(tail);
                encode_file_url_path_pre(&mut out, tail, tail_first, tail_unsafe);
                return out;
            }
        }
        out.push_str("file://wsl.localhost/");
        encode_url_component_into(&mut out, distro.as_bytes());
        encode_file_url_path_pre(&mut out, path, first_unsafe, unsafe_count);
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
        encode_file_url_path_pre(&mut out, path, first_unsafe, unsafe_count);
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
        let (rest_first, rest_unsafe) = scan_url_path(rest);
        encode_file_url_path_pre(&mut out, rest, rest_first, rest_unsafe);
        return out;
    }

    out.push_str("file://");
    encode_file_url_path_pre(&mut out, path, first_unsafe, unsafe_count);
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
        assert!(!color(
            true,
            None,
            Some("1"),
            Some("1"),
            Some("xterm-256color"),
            true
        ));
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
    fn clicolor_force_only_forces_on_a_nonempty_nonzero_value() {
        assert!(color(false, None, Some("1"), None, Some("dumb"), false));
        // An *empty* value is unset, matching `NO_COLOR`/`WT_SESSION` here and
        // `anstyle-query`: a templated CI matrix that materialises "" must not
        // force colour into piped logs.
        assert!(!color(false, None, Some(""), None, Some("dumb"), false));
        assert!(!color(false, None, Some(""), None, Some("xterm"), false));
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
        assert!(!color(
            false,
            None,
            None,
            Some("false"),
            Some("xterm"),
            true
        ));
        // CLICOLOR_FORCE takes precedence over FORCE_COLOR.
        assert!(color(
            false,
            None,
            Some("1"),
            Some("0"),
            Some("dumb"),
            false
        ));
        // NO_COLOR still wins over FORCE_COLOR.
        assert!(!color(true, None, None, Some("3"), Some("xterm"), true));
    }

    #[test]
    fn clicolor_zero_disables() {
        assert!(!color(false, Some("0"), None, None, Some("xterm"), true));
        assert!(color(false, Some("1"), None, None, Some("xterm"), true));
        // CLICOLOR_FORCE overrides CLICOLOR=0.
        assert!(color(
            false,
            Some("0"),
            Some("1"),
            None,
            Some("xterm"),
            false
        ));
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
        assert_eq!(
            width_decision(Some("18446744073709551615"), None),
            u16::MAX as usize
        );
        assert_eq!(width_decision(Some("65536"), None), u16::MAX as usize);
    }

    #[test]
    fn terminal_family_signals_select_truecolor() {
        // WT_SESSION is the only truecolor signal on older Windows Terminal.
        assert_eq!(
            color_depth_decision_env(&ColorEnv {
                term: Some("xterm"),
                wt_session: true,
                ..ColorEnv::default()
            }),
            ColorDepth::TrueColor
        );
        for tp in ["iTerm.app", "WezTerm", "vscode", "WarpTerminal", "Warp"] {
            assert_eq!(
                color_depth_decision_env(&ColorEnv {
                    term: Some("xterm"),
                    term_program: Some(tp),
                    ..ColorEnv::default()
                }),
                ColorDepth::TrueColor,
                "{tp}"
            );
        }
        // JetBrains terminals set no `COLORTERM`; kitty may not either.
        assert_eq!(
            color_depth_decision_env(&ColorEnv {
                term: Some("xterm"),
                terminal_emulator: Some("JetBrains-JediTerm"),
                ..ColorEnv::default()
            }),
            ColorDepth::TrueColor
        );
        assert_eq!(
            color_depth_decision_env(&ColorEnv {
                term: Some("xterm-kitty"),
                ..ColorEnv::default()
            }),
            ColorDepth::TrueColor
        );
        // An unrelated TERM_PROGRAM must not upgrade the depth.
        assert_eq!(
            color_depth_decision_env(&ColorEnv {
                term: Some("xterm"),
                term_program: Some("Apple_Terminal"),
                ..ColorEnv::default()
            }),
            ColorDepth::Sixteen
        );
    }

    #[test]
    fn force_color_true_and_empty_detect_depth() {
        // Modern chalk contract: `true`/empty mean "force colour ON and let
        // the level be DETECTED", so COLORTERM=truecolor must yield true
        // colour instead of being pinned to the 16-colour palette.
        for v in ["true", ""] {
            assert_eq!(
                color_depth_decision(Some("truecolor"), Some("xterm"), Some(OsStr::new(v))),
                ColorDepth::TrueColor,
                "FORCE_COLOR={v:?}"
            );
            assert!(
                color(false, None, None, Some(v), Some("dumb"), false),
                "FORCE_COLOR={v:?} still forces colour on"
            );
        }
    }

    #[test]
    fn force_color_noncanonical_values_force_on() {
        for v in ["abc", "FALSE", "18446744073709551616"] {
            assert!(!force_color_is_off(OsStr::new(v)), "{v}");
            assert!(
                color(false, None, None, Some(v), Some("dumb"), false),
                "{v} should force color on"
            );
        }
        // Modern supports-color clamps negative levels to OFF; a CI script
        // silencing colour with -1 must not get colour on a TTY.
        assert!(force_color_is_off(OsStr::new("-1")));
        assert!(!color(false, None, None, Some("-1"), Some("xterm"), true));
        assert_eq!(
            force_color_level(OsStr::new("256")),
            Some(ColorDepth::TrueColor)
        );
    }

    /// A negative value outside `i64` must still read as OFF.  Parsed as `i64`
    /// the parse failed and fell through to "force on", defeating the very
    /// clamping rule the comment above it documents.
    #[test]
    fn force_color_off_survives_negative_overflow() {
        assert!(force_color_is_off(OsStr::new("-99999999999999999999")));
        assert!(!color(
            false,
            None,
            None,
            Some("-99999999999999999999"),
            Some("xterm"),
            true
        ));
    }

    /// Rust's integer `FromStr` accepts a leading `+`, so `FORCE_COLOR=+2` and
    /// `COLUMNS=+80` behave like their unsigned spellings.  Node's `parseInt`
    /// accepts them too, so the behaviour is correct — but it was incidental;
    /// pin it so a future "strict env parsing" change is a visible decision.
    #[test]
    fn leading_plus_signs_parse_like_node() {
        assert_eq!(
            force_color_level(OsStr::new("+2")),
            Some(ColorDepth::TwoFiftySix)
        );
        assert!(!force_color_is_off(OsStr::new("+1")));
        assert_eq!(width_decision(Some("+80"), None), 80);
    }

    #[test]
    fn force_color_numeric_spellings_are_consistent() {
        assert_eq!(force_color_level(OsStr::new("00")), None);
        assert!(force_color_is_off(OsStr::new("00")));
        assert_eq!(
            color_depth_decision(None, Some("xterm"), Some(OsStr::new("256"))),
            ColorDepth::TrueColor
        );
        assert!(color(false, None, None, Some("256"), Some("dumb"), false));
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
        for term in [
            "iterm2-direct",
            "xterm-direct",
            "st-direct",
            "foot-direct",
            "xterm-truecolor",
        ] {
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
        assert_eq!(
            force_color_level(OsStr::new("4")),
            Some(ColorDepth::TrueColor)
        );
        assert_eq!(
            force_color_level(OsStr::new(" 2")),
            Some(ColorDepth::TwoFiftySix)
        );
        assert_eq!(
            force_color_level(OsStr::new("03")),
            Some(ColorDepth::TrueColor)
        );
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
        let caps = color_caps_decision(&ColorEnv {
            force_color: Some(OsStr::new("2")),
            term: Some("xterm-256color"),
            stderr_is_tty: true,
            ..ColorEnv::default()
        });
        assert!(caps.color);
        assert_eq!(caps.depth, ColorDepth::TwoFiftySix);

        let caps = color_caps_decision(&ColorEnv {
            force_color: Some(OsStr::new("0")),
            colorterm: Some("truecolor"),
            term: Some("xterm"),
            stderr_is_tty: true,
            ..ColorEnv::default()
        });
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
        // A UNC share stays a native UNC URL under WSL too: the 9P server
        // usually cannot traverse a CIFS mount, so `wsl.localhost/<distro>//
        // server/share` would be a dead (and slower) link.
        assert_eq!(
            format_file_url(b"//server/share/file.rs", Some("Ubuntu")),
            "file://server/share/file.rs"
        );
        // A Windows-native path must NOT be routed through `wsl.localhost`
        // (which produced `file://wsl.localhost/UbuntuC:/...`): it falls
        // through to the drive-letter branch instead.
        assert_eq!(
            format_file_url(b"C:/Users/user/file.rs", Some("Ubuntu")),
            "file:///C:/Users/user/file.rs"
        );
        // A POSIX directory named `c:/x` is not a Windows drive; on Windows the
        // same bytes are a drive path and must gain the third slash.
        if cfg!(windows) {
            assert_eq!(format_file_url(b"c:/x", None), "file:///c:/x");
        } else {
            assert_eq!(format_file_url(b"c:/x", None), "file://c:/x");
        }
    }

    #[test]
    fn empty_path_is_not_garbage() {
        assert_eq!(&*path_to_file_url(std::path::Path::new("")), "file://");
    }
}
