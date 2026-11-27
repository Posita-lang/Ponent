//! Terminal capability detection.
//!
//! Single home for the `NO_COLOR` / `CLICOLOR` / `CLICOLOR_FORCE` /
//! `FORCE_COLOR` / `TERM=dumb` / `isatty` dance plus width, Unicode, and
//! OSC 8 hyperlink probes, so downstream consumers don't each reimplement it
//! (and get it subtly wrong).
//!
//! Every public detector is **process-stable with first-call-wins semantics**:
//! the first call snapshots its capability's environment (and, where relevant,
//! the `stderr` `isatty` state) into a per-capability `LazyLock`, and later
//! calls are a single atomic load.  The snapshots are per-capability on purpose
//! — a caller that only asks for colour must not pay `TIOCGWINSZ`.  The one
//! documented exception is [`TermCaps::detect`], which always re-probes.
//! The real hazard is therefore *ordering*, not instability — whatever the
//! environment looks like when the first detector runs decides the answer for
//! the whole run, so a CLI flag must be applied through
//! [`DiagOpts`](crate::diagnostics::DiagOpts) rather than by re-probing.
//! [`TermCaps::detect`] is the escape hatch for callers that genuinely want a
//! fresh probe.

use crate::diagnostics::{DEFAULT_MAX_WIDTH, HEX_UPPER};
use std::borrow::Cow;
use std::ffi::OsStr;
use std::io::{self, IsTerminal};
use terminal_size::{self, Width};

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
    #[must_use]
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
    /// OSC 8 hyperlink support.  Without it a caller holding `TermCaps` still had to call
    /// [`supports_hyperlinks`] — a separate process-stable cache — to learn the fifth
    /// capability, so the "one snapshot" contract stopped one field short.
    pub hyperlinks: bool,
}

impl TermCaps {
    /// The four capabilities in one env snapshot.
    ///
    /// **Deliberately ignores [`HYPERLINK_OVERRIDE`]**, unlike [`supports_hyperlinks`].
    /// This is the raw, hermeticity-free reading of the environment: it is what the
    /// re-executed child probe prints, and that probe exists precisely to observe what the
    /// env actually says.  An in-process *test* must use the accessors
    /// (`supports_hyperlinks` and friends), which honour the pin — `detect()` would make
    /// such a test non-hermetic on a link-capable host.
    pub fn detect() -> Self {
        let stderr_is_tty = io::stderr().is_terminal();
        // `TERM` is read exactly once and threaded into the colour decision:
        // reading it again below let `color` and `unicode` observe different
        // snapshots (the module's "one canonical env-read path" contract).
        let term_env = std::env::var_os("TERM");
        let term = term_env.as_deref();
        // One canonical colour env-read path: `color_caps_from_env` performs
        // exactly the reads this function used to duplicate inline, so a
        // future variable cannot be added to one and forgotten in the other.
        let caps = color_caps_from_env(stderr_is_tty, term);
        let locale = ["LC_ALL", "LC_CTYPE", "LANG"].iter().find_map(|v| {
            std::env::var_os(v).filter(|s| {
                // Non-empty **and** not whitespace-only: a blank locale is as unset as an
                // empty one, the same rule `env_trimmed` and the no-color.org presence
                // check use.  A non-UTF-8 locale stays "set" on purpose — that is what lets
                // `unicode_decision_os` answer "set but not UTF-8", which is the truth.
                !s.is_empty() && s.to_str().is_none_or(|t| !t.trim().is_empty())
            })
        });
        // Case-insensitive like every other term comparison in this module
        // (`COLORTERM`, TERM suffixes, JetBrains); `TERM=DUMB` on an embedded
        // CI terminal would otherwise enable colour *and* box drawing.
        let term_dumb = is_dumb_term(term);
        let columns = std::env::var_os("COLUMNS");
        TermCaps {
            color: caps.color,
            depth: caps.depth,
            unicode: unicode_decision_os(locale.as_deref(), term_dumb),
            width: width_decision_impl(columns.as_deref().and_then(OsStr::to_str), || {
                terminal_size::terminal_size_of(io::stderr()).map(|(Width(w), _)| w as usize)
            }),
            hyperlinks: detect_hyperlinks_fresh(),
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
/// Colour-only probe: `isatty` plus the colour env reads, nothing else.
fn detect_color_caps_fresh() -> ColorCaps {
    let stderr_is_tty = io::stderr().is_terminal();
    let term_env = std::env::var_os("TERM");
    let term = term_env.as_deref();
    color_caps_from_env(stderr_is_tty, term)
}

/// Unicode-only probe: the locale reads plus `TERM`, no `isatty`, no TIOCGWINSZ.
fn detect_unicode_fresh() -> bool {
    let term_env = std::env::var_os("TERM");
    let term = term_env.as_deref();
    let locale = ["LC_ALL", "LC_CTYPE", "LANG"].iter().find_map(|v| {
        std::env::var_os(v).filter(|s| {
            // Non-empty **and** not whitespace-only: a blank locale is as unset as an
            // empty one, the same rule `env_trimmed` and the no-color.org presence
            // check use.  A non-UTF-8 locale stays "set" on purpose — that is what lets
            // `unicode_decision_os` answer "set but not UTF-8", which is the truth.
            !s.is_empty() && s.to_str().is_none_or(|t| !t.trim().is_empty())
        })
    });
    unicode_decision_os(locale.as_deref(), is_dumb_term(term))
}

/// Width-only probe: `COLUMNS` first, TIOCGWINSZ only if it misses.
fn detect_width_fresh() -> usize {
    let columns = std::env::var_os("COLUMNS");
    width_decision_impl(columns.as_deref().and_then(OsStr::to_str), || {
        terminal_size::terminal_size_of(io::stderr()).map(|(Width(w), _)| w as usize)
    })
}

// Per-capability snapshots instead of one `TermCaps`: a CLI that only asks for
// colour (never width, never hyperlinks) used to pay `TIOCGWINSZ` + three locale
// reads + a full colour decision for an answer it discarded.  The consistency
// contract is per-capability — `color` and `depth` must agree, which is why they
// share one cache — while `unicode`, `width` and `hyperlinks` are independent
// readings nothing depends on agreeing.
//
// `LazyLock` is used for initialisation ergonomics, *not* because its deref is
// cheaper: after initialisation it is an acquire load plus a state check, the
// same steady-state cost as `OnceLock::get().unwrap()`.  What it removes is the
// unwrap (and the `Option` in the type), not a branch.
static COLOR_CAPS: std::sync::LazyLock<ColorCaps> =
    std::sync::LazyLock::new(detect_color_caps_fresh);
static UNICODE_CAPS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(detect_unicode_fresh);
static WIDTH_CAPS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(detect_width_fresh);

/// Snapshot the colour-related environment once, so `color` and `depth` can
/// never describe different environments.
#[must_use]
pub fn detect_color_caps() -> ColorCaps {
    *COLOR_CAPS
}

/// ASCII-only, allocation-free case-insensitive substring test.
fn contains_ignore_ascii_case(hay: &[u8], needle: &[u8]) -> bool {
    let Some(&first) = needle.first() else {
        // Empty needle: "contains" is true by definition (pinned by
        // `empty_locale_and_needle_are_unset`).
        return true;
    };
    // First-byte discriminator: the old `windows(len).any(eq_ignore_ascii_case)` re-ran
    // the whole case-folded compare at every position.  Comparing one byte first means
    // the full compare runs only at candidates.  The `hay.len() >= needle.len()` guard
    // is redundant with `windows()` semantics but is a real early-out.
    if hay.len() < needle.len() {
        return false;
    }
    let lower = first.to_ascii_lowercase();
    let upper = first.to_ascii_uppercase();
    let last_start = hay.len() - needle.len();
    (0..=last_start).any(|i| {
        (hay[i] == lower || hay[i] == upper)
            && hay[i..i + needle.len()].eq_ignore_ascii_case(needle)
    })
}

/// ASCII-only, allocation-free case-insensitive suffix test.
fn ends_with_ignore_ascii_case(hay: &[u8], needle: &[u8]) -> bool {
    hay.len() >= needle.len() && hay[hay.len() - needle.len()..].eq_ignore_ascii_case(needle)
}

/// `FORCE_COLOR` level: `1` → 16 colours, `2` → 256, `3` **or any larger
/// numeric value** → true colour.  Anything non-numeric (other than
/// `true`/`false`/empty) carries no level.
///
/// **Known divergence from current `supports-color`, kept deliberately:** upstream
/// gates on `^\d+$` over the raw value, so a padded (`" 2 "`), signed (`"-1"`,
/// `"+2"`) or garbage (`"abc"`) value is simply *ignored* (falls through to TTY
/// detection), whereas this module trims, treats a negative level as an explicit
/// off, and treats non-numeric text as force-on.  These rules are pinned by
/// `force_color_noncanonical_values_force_on`,
/// `force_color_whitespace_only_forces_on` and
/// `force_color_off_survives_negative_overflow`; the divergence is chosen because
/// a CI script writing `FORCE_COLOR=-1` means "silence colour" and a script
/// writing garbage means "force it on".  If strict parity is ever wanted, this is
/// the one place to change (and those three tests to delete).
/// Optional sign plus an all-digit body.  `None` for an empty body or any
/// non-digit byte.
///
/// The value is deliberately **not** materialised: both `FORCE_COLOR` callers
/// classify by sign and by zero-ness/size class only, so classifying the digit
/// string directly removes every `i128`/`u128` boundary at once.  The two
/// separate parsers previously had to stay in sync by hand, and each had an
/// overflow hole one magnitude further out than the other.
fn signed_digits(s: &str) -> Option<(bool, &str)> {
    let neg = s.starts_with('-');
    let body = s
        .strip_prefix('-')
        .or_else(|| s.strip_prefix('+'))
        .unwrap_or(s);
    (!body.is_empty() && body.bytes().all(|b| b.is_ascii_digit())).then_some((neg, body))
}

/// One parse of `FORCE_COLOR` into both facts the two decisions need.
///
/// `force_color_is_off` and `force_color_level` are called on the same `OsStr`
/// when the variable is set, each doing its own `to_str` + `trim`.  Beyond the
/// saved pass, a single parser is what makes it structurally impossible for the
/// two to disagree on whether a value is numeric.
fn classify_force_color(v: &OsStr) -> (bool, Option<ColorDepth>) {
    let Some(s) = v.to_str().map(str::trim) else {
        return (false, None);
    };
    match signed_digits(s) {
        Some((neg, digits)) => {
            // `supports-color` clamps `level < 0` to 0, i.e. OFF, and `-0`/`+0`
            // are zero, hence off.  Digit classification is exact at *every*
            // magnitude, so a 40-digit negative cannot fall off an integer
            // boundary and silently read as "force on".
            if neg || digits.bytes().all(|b| b == b'0') {
                return (true, None);
            }
            (false, force_color_depth_from_digits(digits))
        }
        // Non-numeric values (`"abc"`, `"FALSE"`) mean force-on — not an
        // explicit off request.  Case-sensitive `false` only, mirroring Node.
        None => (s == "false", None),
    }
}

/// `FORCE_COLOR`'s Node/chalk level for a strictly-positive decimal string.
fn force_color_depth_from_digits(digits: &str) -> Option<ColorDepth> {
    let significant = digits.trim_start_matches('0');
    match significant.len() {
        // Unreachable: `classify_force_color` routes every all-zero digit string
        // (`"0"`, `"00"`, `"-0"`, `"+0"`) to `(true, None)` — off — before this
        // function is called, so `significant` is never empty.  Kept *and labelled*
        // rather than deleted: inside the exact-magnitude parser this is exactly where
        // a future "simplification" would silently move the `-0`/`00` contract.
        0 => None,
        1 => match significant.as_bytes()[0] {
            b'1' => Some(ColorDepth::Sixteen),
            b'2' => Some(ColorDepth::TwoFiftySix),
            // `0` is handled above, so this is 3..=9.
            _ => Some(ColorDepth::TrueColor),
        },
        // Any magnitude past a single significant digit is ≥ 10, i.e. "true
        // colour" — including 40-digit CI-generated strings and a `+`-prefixed
        // value, both of which the old `u128` fallback treated inconsistently.
        _ => Some(ColorDepth::TrueColor),
    }
}

fn force_color_level(v: &OsStr) -> Option<ColorDepth> {
    // Deliberate deviation from `supports-color`, which pins `true`/`""` to
    // level 1: pinning 16 colours *downgrades* a truecolor terminal, so we only
    // force colour on (the `force_color_is_off == false` half) and let
    // COLORTERM/TERM pick the depth.  Pinned by
    // `force_color_true_and_empty_detect_depth`.
    let (off, level) = classify_force_color(v);
    if off { None } else { level }
}

/// The three rules, in the order the body applies them (this list IS the
/// contract — the function name reads as "off unless…", which is not it):
///   1. `"false"` → off.
///   2. Any numeric value `≤ 0` → off, including magnitudes beyond `i64`: a
///      `FORCE_COLOR=-1` from a CI script means "silence colour", not "force it
///      on".  (Current `supports-color` ignores a negative value instead — see
///      `classify_force_color`'s divergence note.  We treat it as an explicit
///      off so the silencing idiom keeps working at every magnitude.)
///   3. Anything else — non-numeric garbage (`"abc"`, `"FALSE"`) or a numeric
///      `> 0` — is not an explicit off request, so colour stays on (the caller
///      then detects the depth).
/// `force_color_noncanonical_values_force_on` and
/// `force_color_off_survives_negative_overflow` pin the two edges.
fn force_color_is_off(v: &OsStr) -> bool {
    classify_force_color(v).0
}

/// Detect the terminal's colour depth from `FORCE_COLOR`, `COLORTERM` and
/// `TERM`.
///
/// Precedence, as implemented by [`color_depth_decision_env`] (which is the
/// authoritative list — this summary must not drift from it):
///   1. `FORCE_COLOR=1|2|3` → 16/256/24-bit respectively (Node/chalk levels;
///      any larger numeric value is 24-bit).
///   2. A terminal-family signal *before* `COLORTERM`: `WT_SESSION` (Windows
///      Terminal) → [`ColorDepth::TrueColor`], and an exact `TERM_PROGRAM` in
///      `{iTerm.app, WezTerm, vscode, WarpTerminal, Warp}` → [`ColorDepth::TrueColor`].
///   3. `TERMINAL_EMULATOR=JetBrains-JediTerm` → [`ColorDepth::TrueColor`].
///      (No `TERM` clause: JetBrains terminals are identified by this one
///      variable, and a `TERM` heuristic would also fire on the shells they
///      spawn elsewhere.)
///   4. `COLORTERM` contains `truecolor` or `24bit` → [`ColorDepth::TrueColor`]
///   5. `TERM` ends with `truecolor`, or carries `direct` as a **hyphen-delimited
///      token** after trailing digits are stripped (`xterm-direct2` ✓,
///      `xterm-direct-2024` is conservatively [`ColorDepth::Sixteen`], `indirect` ✗),
///      or is exactly `xterm-kitty`, `xterm-ghostty` or `wezterm` →
///      [`ColorDepth::TrueColor`]
///   6. `TERM` ends with `256color` → [`ColorDepth::TwoFiftySix`]
///   7. Otherwise → [`ColorDepth::Sixteen`]
///
/// The authoritative chain for *forced* levels lives in `color_depth_parts`; the
/// `*_decision_env` wrappers only cover the `FORCE_COLOR` half, so a reader checking
/// precedence must start there rather than at the wrapper.
#[must_use]
pub fn detect_color_depth() -> ColorDepth {
    detect_color_caps().depth
}

/// The colour-related environment as one named value.
///
/// Replaces ten positional parameters.  `colorterm` and `term` are both
/// `Option<&str>`, so a swapped pair type-checks and can even *pass* (a
/// `COLORTERM` value shaped like a `TERM`); named fields make every call site
/// auditable and turn the literal into its own documentation.  `Default` is the
/// "nothing set" environment, which is what the decision-core tests want.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ColorEnv<'a> {
    /// `NO_COLOR` set to a non-empty value (no-color.org).
    pub no_color: bool,
    pub clicolor: Option<&'a OsStr>,
    pub clicolor_force: Option<&'a OsStr>,
    pub force_color: Option<&'a OsStr>,
    pub colorterm: Option<&'a OsStr>,
    pub term: Option<&'a OsStr>,
    /// `WT_SESSION` set to a non-empty value.
    pub wt_session: bool,
    pub term_program: Option<&'a OsStr>,
    pub terminal_emulator: Option<&'a OsStr>,
    /// `stderr` is a terminal.  Part of the same snapshot on purpose: pairing a
    /// stale TTY probe with a fresh environment is exactly the half-updated view
    /// [`color_caps_from_env`] exists to prevent.
    pub stderr_is_tty: bool,
}

/// The depth-only subset of [`ColorEnv`], for callers that do not care about
/// colour-on/off (and for the pure decision core's tests).
/// Positional shim for tests only: production code builds a [`ColorEnv`], whose
/// named fields make the `colorterm`/`term` swap impossible.  Gated so the
/// three-`Option` signature cannot leak back into the crate surface.
#[cfg(test)]
pub(crate) fn color_depth_decision(
    colorterm: Option<&OsStr>,
    term: Option<&OsStr>,
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
    color_depth_parts(env, env.force_color.and_then(force_color_level))
}

/// [`color_depth_decision_env`], with the forced depth already resolved by the caller.
fn color_depth_parts(env: &ColorEnv<'_>, force_depth: Option<ColorDepth>) -> ColorDepth {
    if let Some(depth) = force_depth {
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
    // output.  `supports-color` reads the same variable with `includes` (exact,
    // case-sensitive); the substring test here is a deliberate, more forgiving
    // divergence — do not "fix" it into exact matching silently.
    if let Some(te) = env.terminal_emulator
        && contains_ignore_ascii_case(te.as_encoded_bytes(), b"jetbrains")
    {
        return ColorDepth::TrueColor;
    }
    if let Some(ct) = env.colorterm {
        let ct = ct.as_encoded_bytes();
        if contains_ignore_ascii_case(ct, b"truecolor") || contains_ignore_ascii_case(ct, b"24bit")
        {
            return ColorDepth::TrueColor;
        }
    }
    if let Some(t) = env.term {
        let t = t.as_encoded_bytes();
        // Some kitty builds do not set `COLORTERM`; `supports-color` special-cases
        // the TERM value itself, and without it those terminals fall through to
        // `Sixteen`.  Ghostty (`xterm-ghostty`) and WezTerm (`wezterm`) are in the
        // same upstream exact-match set for the same reason: older builds and
        // some ssh hops leave `COLORTERM` unset, so a stock truecolor terminal was
        // getting 16-colour output.
        for name in [b"xterm-kitty".as_slice(), b"xterm-ghostty", b"wezterm"] {
            if t.eq_ignore_ascii_case(name) {
                return ColorDepth::TrueColor;
            }
        }
        // `direct` may carry a trailing digit run (`xterm-direct2`,
        // `xterm-direct3` are stock ncurses entries, and forks add more), so
        // digits are stripped before the suffix test.  The feature must also
        // start a token: bare `direct`, or `...-direct`.  A plain suffix test
        // would match `TERM=indirect`, which is why the separator is checked.
        if ends_with_ignore_ascii_case(t, b"truecolor") {
            return ColorDepth::TrueColor;
        }
        let mut end = t.len();
        while end > 0 && t[end - 1].is_ascii_digit() {
            end -= 1;
        }
        let direct = &t[..end];
        let is_direct_token = direct.len() >= 6
            && ends_with_ignore_ascii_case(direct, b"direct")
            && (direct.len() == 6 || direct[direct.len() - 7] == b'-');
        if is_direct_token {
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
///   2. `CLICOLOR_FORCE` set to a non-**blank** value other than `"0"` → on
///      (padding is trimmed first, so `" 0 "` does not force — see
///      [`env_trimmed`]).
///   3. `FORCE_COLOR` → see [`force_color_is_off`]: `false` and any numeric
///      `≤ 0` are off, non-numeric garbage is on, everything else is on.
///   4. `CLICOLOR=0` → off.
///   5. `TERM=dumb` → off.
///   6. Otherwise: on iff `stderr` is a terminal.
#[must_use]
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
fn color_caps_from_env(stderr_is_tty: bool, term: Option<&OsStr>) -> ColorCaps {
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
    // Carried as `&OsStr`, not `&str`: `to_str()` dropped a non-UTF-8 value
    // entirely, so `COLORTERM=truecol\xF8rd` (or an unpaired surrogate in
    // `TERM_PROGRAM` on Windows) lost the signal in a module whose whole premise
    // is byte-exactness.  Every consumer below compares bytes.
    let colorterm = colorterm_env.as_deref();
    let term_program = term_program_env.as_deref();
    let terminal_emulator = terminal_emulator_env.as_deref();
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
    // One `FORCE_COLOR` classification per snapshot: `color_decision_env` asks "is this an
    // explicit off?" and `color_depth_decision_env` asks "which depth did it pin?" — two
    // questions about the same value.  Calling each separately re-`trim`ed and re-split the
    // same `OsStr` on every capability snapshot (which is the per-diagnostic path).
    let force = env.force_color.map(classify_force_color);
    ColorCaps {
        color: color_decision_parts(env, force.map(|(off, _)| off)),
        depth: color_depth_parts(env, force.and_then(|(_, depth)| depth)),
    }
}

/// The single implementation behind [`color_decision`].
pub(crate) fn color_decision_env(env: &ColorEnv<'_>) -> bool {
    color_decision_parts(env, env.force_color.map(|v| classify_force_color(v).0))
}

/// [`color_decision_env`], with the `FORCE_COLOR` verdict already classified by the
/// caller (see [`color_caps_decision`]).
fn color_decision_parts(env: &ColorEnv<'_>, force_off: Option<bool>) -> bool {
    if env.no_color {
        return false;
    }
    // `CLICOLOR_FORCE=0` means "don't force", not "force off" — it falls
    // through to normal detection.  An *empty* value is treated as unset, like
    // `NO_COLOR`/`WT_SESSION`/`NO_HYPERLINK` here and like anstream's
    // `anstyle-query`: templated CI matrices routinely materialise empty
    // strings, and force-on-empty pushed colour into piped logs.
    //
    // **``FORCE_COLOR=""`` deliberately does NOT follow this rule** — see the
    // arm below: chalk compatibility makes an empty `FORCE_COLOR` mean
    // force-on.  The two conventions collide inside this one function, so
    // neither arm may be "fixed" into the other; pinned by
    // `force_color_empty_diverges_from_clicolor_force_empty`.
    //
    // Three different empty/padded conventions meet in this function; the table
    // is the contract (each row is pinned by a test):
    //
    // | variable         | unset | empty      | padded (`" "`) | `"0"`   | `"false"` |
    // |------------------|-------|------------|----------------|---------|-----------|
    // | `NO_COLOR`       | no-op | no-op      | **present → off** | n/a  | n/a       |
    // | `CLICOLOR_FORCE` | no-op | unset      | unset          | no force| force on  |
    // | `FORCE_COLOR`    | no-op | force on   | force on       | off     | off       |
    // | `CLICOLOR`       | no-op | unset      | unset          | off     | no-op     |
    //
    // The `CLICOLOR_FORCE` row is a **deliberate divergence** from
    // `anstyle-query`, which reports *any* non-empty value (including `"0"` and
    // `" "`) as forced; this code trims and special-cases `"0"` so a templated
    // matrix cannot silence or force colour by accident.  Do not attribute our
    // behaviour to upstream when reading these rows.
    if let Some(v) = env.clicolor_force.and_then(env_trimmed)
        && v != "0"
    {
        return true;
    }
    if let Some(off) = force_off {
        // Unlike `CLICOLOR_FORCE` above, `FORCE_COLOR=""` forces colour *on*
        // (chalk/`supports-color` compatibility), and so does non-numeric
        // garbage.  That is a deliberate divergence from the empty-means-unset
        // rule one arm up — do not unify them without making it a decision.
        // (`force_off` is `Some` exactly when `FORCE_COLOR` is set: the classifier
        // returns a verdict for every `OsStr`, including non-UTF-8.)
        return !off;
    }
    if let Some(v) = env.clicolor.and_then(env_trimmed)
        && v == "0"
    {
        return false;
    }
    if is_dumb_term(env.term) {
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
#[must_use]
pub fn detect_unicode() -> bool {
    // Process-stable snapshot (`UNICODE_CAPS`): the locale/`TERM` reads happen
    // once per process instead of once per optional-detection call.  No
    // `isatty`/TIOCGWINSZ probe is paid here.
    *UNICODE_CAPS
}

fn unicode_decision_os(locale: Option<&OsStr>, term_dumb: bool) -> bool {
    // An empty locale name means "unset", not "set to something that is not
    // UTF-8".  This rule used to live at two call sites (`unicode_decision` and
    // `TermCaps::detect`), so a new caller of this function got the
    // plausible-but-wrong answer instead of the documented one; own it here.
    let locale = locale.filter(|v| !v.is_empty());
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

/// Target output width in display columns.
///
/// `COLUMNS` wins when set and parseable to a positive integer, capped at
/// [`u16::MAX`] (no real terminal — and no `terminal_size::Width` — can exceed
/// it, and an unbounded value poisons downstream `width - used` arithmetic);
/// otherwise the *stderr* terminal size is probed (diagnostics are written
/// there), falling back to [`DEFAULT_MAX_WIDTH`] when the probe returns
/// `None`.
#[must_use]
pub fn detect_width() -> usize {
    // Process-stable snapshot (`WIDTH_CAPS`): at most one TIOCGWINSZ per process,
    // and none at all when `COLUMNS` resolves.  The derivation lives in
    // `detect_width_fresh`, which reads `COLUMNS` via `var_os` + `to_str` like
    // the rest of the module.
    *WIDTH_CAPS
}

/// The `COLUMNS` half of the width decision.
///
/// Separable so [`TermCaps::detect`] can skip the `TIOCGWINSZ` probe entirely
/// when `COLUMNS` resolves — the argument to `width_decision` used to be
/// evaluated eagerly, costing one syscall per process for a value it discarded.
fn width_from_columns(columns: Option<&str>) -> Option<usize> {
    // One digit classifier for both `COLUMNS` and `FORCE_COLOR`: `signed_digits` splits
    // the sign and rejects a non-digit body, so the only thing left here is the clamp
    // policy.  The previous copy hand-rolled the same scan with different edge rules, so
    // `+80` and an overflow were reasoned about twice — in two functions whose contracts
    // must not drift.
    let s = columns.map(str::trim)?;
    match signed_digits(s) {
        Some((neg, body)) if !neg => match body.parse::<usize>() {
            Ok(w) => (w > 0).then_some(w.min(u16::MAX as usize)),
            // Overflow is still a positive number: clamp instead of treating it as
            // "unset" (which silently fell back to the terminal size).  No all-zero guard
            // is needed: `signed_digits` guarantees a non-empty all-digit body, so the only
            // `parse` failure is `PosOverflow`, which implies a non-zero digit — the
            // `.any(|b| b != b'0')` that used to sit here was provably always true.
            Err(_) => Some(u16::MAX as usize),
        },
        _ => None,
    }
}

/// The single width-decision core: `COLUMNS` first, then the probe, then the
/// default.  `probe` is lazy so the one production caller does not pay a
/// `TIOCGWINSZ` syscall when `COLUMNS` already wins.
fn width_decision_impl(columns: Option<&str>, probe: impl FnOnce() -> Option<usize>) -> usize {
    width_from_columns(columns)
        .or_else(probe)
        .filter(|&w| w > 0)
        .unwrap_or(DEFAULT_MAX_WIDTH)
}

/// Test-facing form: the probe result is passed by value.  Was two hand-kept
/// copies of the decision (`or` vs `or_else`), which is exactly the drift this
/// consolidation removes; also `#[cfg(test)]` because no production caller uses
/// the eager shape (the un-gated version tripped `dead_code` in plain builds).
#[cfg(test)]
pub(crate) fn width_decision(columns: Option<&str>, term_size: Option<usize>) -> usize {
    width_decision_impl(columns, || term_size)
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
/// The uncached hyperlink probe: `NO_HYPERLINK` veto, `TERM=dumb` veto, then the crate
/// probe.  Memoized by [`HYPERLINK_CAPS`] and called directly by [`TermCaps::detect`], so
/// the decision chain exists once instead of being open-coded twice.
fn detect_hyperlinks_fresh() -> bool {
    let no_hyperlink = std::env::var_os("NO_HYPERLINK").is_some_and(|v| !v.is_empty());
    // `TERM=dumb` means "cannot render control sequences", and an OSC 8
    // hyperlink *is* one: the upstream probe does not veto `dumb`
    // universally, so without this a dumb terminal received the escapes as
    // literal garbage.  The other two capabilities already consult it.
    let term_dumb = is_dumb_term(std::env::var_os("TERM").as_deref());
    // A vetoed caller must not pay the crate probe (env reads + a curated terminal
    // table + a tty check) — the same contract the width path states for
    // `TIOCGWINSZ`.  `hyperlink_decision` stays the tested pure core; only the probe
    // becomes lazy, which is also why the ordering here mirrors
    // `width_decision_impl`'s `FnOnce`.
    if no_hyperlink || term_dumb {
        return hyperlink_decision(no_hyperlink, term_dumb, false);
    }
    hyperlink_decision(
        no_hyperlink,
        term_dumb,
        supports_hyperlinks::on(supports_hyperlinks::Stream::Stderr),
    )
}

static HYPERLINK_CAPS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(detect_hyperlinks_fresh);

/// Test-only pin for the hyperlink *capability*, thread-scoped.
///
/// Hyperlink emission is deliberately decoupled from colour (see `glyph.rs`:
/// "capability, not palette" — a `--color=never` run on a capable terminal keeps its
/// clickable links), and the capability itself is a memoized env/tty probe.  A rendering
/// test that asserts "plain output carries no escape sequences" therefore has to *state*
/// the capability it assumes: inheriting the host terminal made `test_plain_render_has_no_ansi`
/// and the plain snapshots pass in a bare CI process and fail in a hyperlink-capable
/// terminal (`TERM=xterm-256color FORCE_HYPERLINK=1` reproduces it).
///
/// The default is `Some(false)` — the whole test suite is hermetic against the terminal,
/// and a test that needs links present opts in with [`with_hyperlinks_override`] instead
/// of silently becoming vacuous when the host happens to advertise them.
#[cfg(test)]
thread_local! {
    static HYPERLINK_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(Some(false)) };
}

/// Pin [`supports_hyperlinks`] for the duration of `f`, restoring the previous pin.
#[cfg(test)]
pub(crate) fn with_hyperlinks_override<R>(value: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            HYPERLINK_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let prev = HYPERLINK_OVERRIDE.with(|c| c.replace(Some(value)));
    let _restore = Restore(prev);
    f()
}

#[must_use]
pub fn supports_hyperlinks() -> bool {
    #[cfg(test)]
    if let Some(pinned) = HYPERLINK_OVERRIDE.with(std::cell::Cell::get) {
        return pinned;
    }
    *HYPERLINK_CAPS
}

pub(crate) fn hyperlink_decision(
    no_hyperlink: bool,
    term_dumb: bool,
    probe_on_stderr: bool,
) -> bool {
    !no_hyperlink && !term_dumb && probe_on_stderr
}

/// The `URL_SAFE` predicate as a 128-bit mask — now the **test oracle**, not the
/// implementation.
///
/// `is_url_safe` is a branchless arithmetic form (a table load per byte cannot be
/// vectorized: baseline targets have no gather), so the mask exists to prove that
/// form byte-exact over the whole domain.  It is `cfg(test)`-gated on purpose: a
/// production build that drifted back to the load would otherwise keep compiling.
///
/// Only indices 0 and 1 are ever read (`is_url_safe` short-circuits on
/// `b < 0x80`), and the builder below intentionally loops over the **full byte
/// domain**: a future "safe" byte ≥ 0x80 would index `bits[2]` and fail
/// const-evaluation, i.e. a compile error instead of the silent loss of the
/// ASCII invariant that the two `from_utf8_unchecked` calls rely on.
#[cfg(test)]
const URL_SAFE_BITS: [u64; 2] = {
    let mut bits = [0u64; 2];
    let mut i = 0usize;
    // Full domain on purpose: see the doc above — a non-ASCII entry would index
    // past `bits` and turn into a const-eval error.
    while i < 256 {
        // RFC 3986 unreserved characters plus the path delimiters this crate
        // emits.  This `matches!` is the single source of truth for the set.
        if matches!(
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
        ) {
            bits[i / 64] |= 1 << (i % 64);
        }
        i += 1;
    }
    bits
};

#[inline]
const fn is_url_safe(b: u8) -> bool {
    // Arithmetic, not a table load: `URL_SAFE_BITS` needs one memory access per byte,
    // and a table-lookup predicate cannot be auto-vectorized (baseline targets have no
    // gather), so the scan ran ~1 byte/cycle no matter what `target_feature` said.
    // This form is byte-exact against the table for all 256 values (asserted by
    // `url_safe_mask_matches_the_set_predicate`), and `position`/`count` over it
    // vectorize.
    //
    //   0x2D..=0x3A  `-` `.` `/` `0`-`9` `:`
    //   A-Z / a-z    via the 0x20 fold; `@ [ \ ] ^ _ `` wrap or exceed the range
    //   `_` 0x5F, `~` 0x7E — the fold maps both out of the alpha range, so they need
    //   their own arms.
    (b.wrapping_sub(b'-') <= 13) | ((b | 0x20).wrapping_sub(b'a') <= 25) | (b == b'_') | (b == b'~')
}

// Both directions, for all 256 bytes, **at compile time and in every build**.
//
// `URL_SAFE_BITS` is `#[cfg(test)]`, so the oracle test only ever ran under
// `cargo test`: a release build's two `from_utf8_unchecked` calls rested on a
// `debug_assert!` plus a claim in a comment — a silent-UB tail exactly in the build users
// ship.  The reference here is spelled as the *set* (`matches!` over ranges), i.e. the
// independent half of the comparison, so the arithmetic predicate is proven against it
// wherever this module is compiled.
const _: () = {
    let mut b = 0u32;
    while b < 256 {
        let byte = b as u8;
        let want = matches!(
            byte,
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
        assert!(
            is_url_safe(byte) == want,
            "is_url_safe drifted from the safe set for this byte"
        );
        // The escaping loops index the output with `from_utf8_unchecked`, so a safe byte
        // >= 0x80 would corrupt the URL in a way no downstream check could catch.
        assert!(!want || byte < 0x80, "the safe set must be ASCII");
        b += 1;
    }
};

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
    // Only the verbatim `\\?\` form is stripped: it is an *encoding* of an
    // ordinary path.  `\\.\` is the NT **device** namespace (`\\.\C:` is the
    // volume device, not the filesystem root), so stripping it produced a
    // `file:///C:/x` link to a different object than the diagnostic names.
    // Leaving it unstripped routes it to the device-path sentinel below.
    const VERBATIM: &[u8] = b"\\\\?\\";
    if let Some(rest) = bytes.strip_prefix(VERBATIM) {
        let drive = rest.len() >= 2 && rest[0].is_ascii_alphabetic() && rest[1] == b':';
        if drive {
            return Some(VERBATIM.len());
        }
    }
    None
}

fn scan_url_path(s: &[u8]) -> (usize, usize) {
    // Split the two questions.  The dominant case is a clean path
    // (`first_unsafe == s.len()`), where only *existence* matters, so counting was
    // pure waste; and once a path is dirty its unsafe bytes are usually a small tail,
    // so the second pass is short.  Both passes run over the branchless predicate
    // above, i.e. vectorized.
    match s.iter().position(|&b| !is_url_safe(b)) {
        None => (s.len(), 0),
        Some(first) => (
            first,
            s[first..].iter().filter(|&&b| !is_url_safe(b)).count(),
        ),
    }
}

/// `TERM=dumb` is a veto: a terminal that cannot render control sequences gets
/// neither colour nor box drawing.  ASCII case-insensitive like every other
/// `TERM` comparison here (`TERM=DUMB` on an embedded CI console).
fn is_dumb_term(term: Option<&OsStr>) -> bool {
    // `TERM` is a structured terminfo name whose `dumb` is a family root: `dumb`,
    // `dumb-emacs-ansi` (Emacs shell buffers, which advertise no cursor addressing so
    // box-drawing and colour escapes arrive as garbage), `dumb-vt100`, … The old explicit
    // pair missed the rest of the family.  A **prefix** test is right here and wrong for
    // `TERM_PROGRAM` (exact match there): that variable is a free-form product string
    // where a substring match would veto honest terminals (`Apple_Terminal`).  Same
    // shape of check, different evidence.
    term.is_some_and(|t| {
        let t = t.as_encoded_bytes();
        t.eq_ignore_ascii_case(b"dumb") || (t.len() >= 5 && t[..5].eq_ignore_ascii_case(b"dumb-"))
    })
}

fn wsl_distro_name() -> Option<&'static str> {
    static WSL_DISTRO: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    WSL_DISTRO
        .get_or_init(|| {
            let is_linux = cfg!(target_os = "linux");
            // Gate first: on macOS/BSD there is no `/proc` to verify against, so a
            // forwarded `WSL_DISTRO_NAME` (SSH/CI) would translate paths into
            // `file://wsl.localhost/...` URLs that point nowhere on that host.
            let distro = if is_linux {
                std::env::var("WSL_DISTRO_NAME").ok()
            } else {
                None
            };
            // `None` = unreadable/masked.  Only read on Linux: the pure core fails
            // closed on it, which is the documented policy.
            #[cfg(target_os = "linux")]
            let version = std::fs::read_to_string("/proc/version").ok();
            #[cfg(not(target_os = "linux"))]
            let version: Option<String> = None;
            wsl_distro_decision(is_linux, distro.as_deref(), version.as_deref()).map(str::to_owned)
        })
        .as_deref()
}

/// The WSL decision, as a pure function of its three inputs.
///
/// Every other probe in this module extracts such a core (`color_decision_env`,
/// `width_decision_impl`, `hyperlink_decision`, `unicode_decision_os`) precisely so
/// the policy is testable; this one carried four decisions inline — the cfg gate, empty
/// rejection, **fail closed on an unreadable `/proc/version`**, and the
/// `microsoft`/`wsl` kernel-substring test — with no test objecting.
///
/// 1. Not Linux → `None` (no `/proc` to verify against).
/// 2. Missing/empty `WSL_DISTRO_NAME` → `None`.
/// 3. Unreadable `/proc/version` → `None`: a distro whose kernel cannot be identified
///    is not translated, because trusting the env var alone produced
///    `file://wsl.localhost/<distro>/...` links that point nowhere for the host that
///    read them.
/// 4. The kernel string must mention `microsoft` or `wsl` (ASCII case-insensitive):
///    SSH/CI can forward the env var to a native Linux host.
fn wsl_distro_decision<'a>(
    is_linux: bool,
    distro: Option<&'a str>,
    proc_version: Option<&str>,
) -> Option<&'a str> {
    if !is_linux {
        return None;
    }
    let name = distro.filter(|s| !s.is_empty())?;
    let version = proc_version?;
    if !contains_ignore_ascii_case(version.as_bytes(), b"microsoft")
        && !contains_ignore_ascii_case(version.as_bytes(), b"wsl")
    {
        return None;
    }
    Some(name)
}

/// Append `s` percent-encoded to a **byte** buffer.
///
/// Bytes, not `String`: every emitted byte is ASCII by construction (the safe
/// set is ASCII, and a percent escape is three ASCII bytes), so `String::push`
/// re-checked a UTF-8 invariant that cannot fail, once per byte and three times
/// per escaped byte.  The caller converts to `String` exactly once at the end.
fn encode_scanned_path(out: &mut Vec<u8>, s: &[u8], first_unsafe: usize, unsafe_count: usize) {
    out.reserve(s.len().saturating_add(unsafe_count.saturating_mul(2)));
    out.extend_from_slice(&s[..first_unsafe]);
    let mut i = first_unsafe;
    while i < s.len() {
        // Only the *first* iteration is guaranteed to see an unsafe byte: every
        // later iteration resumes just past an escaped byte, which is usually the
        // start of a safe run.  The branch is therefore not redundant — an earlier
        // "it never fires" reading of this loop broke the byte-exact encoders.
        if is_url_safe(s[i]) {
            // `position` over the branchless predicate is the shape LLVM reliably
            // vectorizes into a 16/32-byte find; the hand-rolled `while` (whose exit value
            // fed a `slice` expression) stayed scalar.  The long safe run — a filename with
            // a couple of spaces — is the dominant case, so this is the length that matters.
            // Byte-identical output: both forms stop at the same first unsafe byte.
            let end = s[i..]
                .iter()
                .position(|&b| !is_url_safe(b))
                .map_or(s.len(), |rel| i + rel);
            out.extend_from_slice(&s[i..end]);
            i = end;
        } else {
            let b = s[i];
            out.push(b'%');
            out.push(HEX_UPPER[(b >> 4) as usize]);
            out.push(HEX_UPPER[(b & 0xF) as usize]);
            i += 1;
        }
    }
}

/// Scan statistics for `s[k..]`, derived from the full-path scan.
///
/// Every stripped prefix (`//`, `/mnt/X/`) consists only of `URL_SAFE` bytes
/// (`/` and alphanumerics), so for a safe prefix of length `k`:
/// `first_unsafe(sub) == first_unsafe(full) - k` and
/// `unsafe_count(sub) == unsafe_count(full)`.  Re-scanning the sub-slice — which
/// three branches of `format_file_url` did — was the second O(n) pass the
/// encoder's own doc says must not exist.
/// Remove a URL-safe prefix of length `k` from a `(first_unsafe, unsafe_count)`
/// pair.
///
/// (`#[inline]` used to sit above this block, which made the compiler attach the
/// previous function's doc to this one.)
///
/// The `k` call sites pass the length of a hard-coded prefix (`"/mnt/" + one
/// ASCII letter`, `"file:///"`, …).  That the prefix is entirely URL-safe is what
/// earns the subtraction, and today it holds only because the prefix is a
/// literal: a future configurable mount point or a non-ASCII drive letter would
/// silently break it, and the debug assert would fire only in debug builds.
fn sub_stats(full: (usize, usize), k: usize) -> (usize, usize) {
    // `assert!`, not `debug_assert!`: a future configurable prefix (a non-ASCII
    // mount point, a drive letter outside the safe set) would make this
    // underflow, and the wrapped `first_unsafe` then panics inside a slice index
    // with no hint of the cause — or silently feeds garbage statistics to the
    // encoder. The check is one compare on a cache-miss path, so failing *here*
    // costs nothing and localizes the bug.
    assert!(
        full.0 >= k,
        "the stripped prefix must be URL-safe for this derivation"
    );
    (full.0 - k, full.1)
}

fn encode_url_component_into(out: &mut Vec<u8>, s: &[u8]) {
    for &b in s {
        if b == b'/' || !is_url_safe(b) {
            out.push(b'%');
            out.push(HEX_UPPER[(b >> 4) as usize]);
            out.push(HEX_UPPER[(b & 0xF) as usize]);
        } else {
            out.push(b);
        }
    }
}

/// [`encode_file_url_path_pre`] with the scan already done.
///
/// `format_file_url` scans the path anyway to size its buffer, so re-scanning
/// here would be a second O(n) pass on exactly the long, heavily-encoded paths
/// the reserve exists for.
fn encode_file_url_path_pre(out: &mut Vec<u8>, s: &[u8], first_unsafe: usize, unsafe_count: usize) {
    if first_unsafe == s.len() {
        out.extend_from_slice(s);
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
    // One implementation, not two.  The old `FUSED_MAX = 4096` split kept a
    // short-path arm that found the first unsafe byte with `position` and then
    // pushed the *remaining* bytes one at a time into a `len + 16` buffer, and a
    // long-path arm that counted, reserved exactly and bulk-copied the safe runs.
    // Measured against each other (the differential test that bridged them), the
    // fused arm is never ahead: for a clean path both are one O(n) pass; for a
    // dirty one the unified form does one scan plus exact reserve plus `memcpy`
    // runs, i.e. strictly fewer byte loops and zero reallocation.
    let (first_unsafe, unsafe_count) = scan_url_path(s);
    if first_unsafe == s.len() {
        // SAFETY: every byte passed `is_url_safe`, whose mask is const-evaluated
        // from a `matches!` over ASCII ranges only (`URL_SAFE_BITS`), so `s` is
        // pure ASCII and therefore valid UTF-8.
        debug_assert!(s.iter().all(u8::is_ascii));
        return Cow::Borrowed(unsafe { std::str::from_utf8_unchecked(s) });
    }
    let mut out: Vec<u8> =
        Vec::with_capacity(s.len().saturating_add(unsafe_count.saturating_mul(2)));
    encode_scanned_path(&mut out, s, first_unsafe, unsafe_count);
    // SAFETY: every byte is either from the ASCII-safe set or `%XX` (three ASCII
    // bytes), so the buffer is valid UTF-8 by construction.
    debug_assert!(out.iter().all(u8::is_ascii));
    Cow::Owned(unsafe { String::from_utf8_unchecked(out) })
}

/// The empty-input URL sentinel, allocated once.
///
/// Three call sites (no bytes, an unrepresentable path, a path stripped to nothing) built a
/// fresh `Arc<str>` for this constant on every call — rare paths, but pure waste.
static FILE_URL_EMPTY: std::sync::LazyLock<std::sync::Arc<str>> =
    std::sync::LazyLock::new(|| std::sync::Arc::from("file://"));

/// Convert a file path to a `file://` URL, handling WSL path translation.
///
/// Assembly note (recorded so it is not re-attempted): the `Vec<u8>` here is
/// *not* an extra allocation — `String::from_utf8_unchecked` reuses its buffer,
/// so the only remaining copy is the caller's `Arc<str>::from(String)`, which is
/// inherent to borrowing a `&'static`-ish `Arc` out of a runtime-built string.  A
/// stack buffer (`SmallVec`/`ArrayVec`) would just move the same single
/// allocation to `into_vec()`, so there is nothing to win.
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
/// [`std::path::absolute`], which joins the process CWD lexically: no symlink
/// resolution, no `stat` chain, and it works for a file that does not exist yet
/// (`canonicalize` failed on those, which is why the old code paid the syscall
/// chain and then re-joined the CWD anyway).  It does read the CWD
/// (`getcwd`/`GetFullPathNameW`) for a relative input — *one* syscall, not a
/// chain — and only on the path that produces a URL at all.
///
/// Preserving symlinks is deliberate: this is a *diagnostics* system, not a
/// sysroot/extern lookup — the hyperlink must open the file the diagnostic
/// names, not whatever inode a resolution chain ends at.  The two differ on
/// macOS (`/tmp` vs `/private/tmp`), so canonicalising would rewrite the very
/// path the user just read.
/// Known Win32 edge cases: `std::path::absolute` does not normalize a trailing dot
/// or space (`C:\x. `) and does not understand alternate data streams
/// (`C:\x:stream`), so the emitted URL may be reinterpreted or rejected by Win32
/// when opened.  The path is **not** rewritten — silently changing a diagnostic's
/// file name is worse than a link that needs the reader's attention — so this stays a
/// documented limitation rather than a bug to rediscover.
#[must_use]
pub fn path_to_file_url(path: &std::path::Path) -> std::sync::Arc<str> {
    if path.as_os_str().is_empty() {
        return std::sync::Arc::clone(&FILE_URL_EMPTY);
    }
    let abs_path = if path.is_absolute() {
        Cow::Borrowed(path)
    } else {
        match std::path::absolute(path) {
            Ok(p) => Cow::Owned(p),
            // No CWD means no honest absolute URL.  Emit the explicit sentinel
            // instead of silently linking `/<path>`.
            Err(_) => return std::sync::Arc::clone(&FILE_URL_EMPTY),
        }
    };
    // `abs_path` cannot be empty here: an empty input returned above and
    // `std::path::absolute` never produces an empty path.  (The old second check
    // was unreachable; `standard_library_has_no_empty_absolute` documents that.)
    let raw = abs_path.as_os_str().as_encoded_bytes();

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
        return std::sync::Arc::clone(&FILE_URL_EMPTY);
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

    // The cache reads the distro itself: taking it as a parameter let a future
    // caller pass a different value for the same path and receive a stale URL,
    // because the key is the path alone.
    file_url_cached(&bytes)
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
/// Each worker warms its own table; an entry cap and a byte budget bound a
/// pathological run.  Eviction demotes the hot generation into `cold` rather
/// than clearing, so a working set slightly larger than the cap keeps hitting
/// (clear-all thrashed to zero hits); a cold hit promotes back.
/// Two-generation cache: the hot generation takes inserts, the cold one
/// holds the previous generation and is consulted second.
///
/// `clear()` on reaching the cap gave a cyclically-accessed working set of
/// `CACHE_CAP + 1` distinct paths **zero** hits: each iteration cleared the
/// map just before inserting the next entry.  Demoting the hot map instead
/// keeps the previous generation reachable, and a cold hit promotes back to
/// hot (segmented FIFO — no per-entry recency bookkeeping, no LRU metadata).
///
/// **Bound, stated exactly:** `hot` is capped by `CACHE_CAP` entries and
/// `max_bytes` (keys + URLs), while `cold` holds at most one previous
/// generation.  The per-thread worst case is therefore about `2 ×
/// max_bytes + one generation of keys`, not `max_bytes`: the previous comment
/// advertised the hot bound only, which is the number a reader budgets against.
struct UrlCache {
    hot: rustc_hash::FxHashMap<Box<[u8]>, std::sync::Arc<str>>,
    cold: rustc_hash::FxHashMap<Box<[u8]>, std::sync::Arc<str>>,
    hot_bytes: usize,
    /// Entry cap and byte budget.  Instance fields rather than consts so the
    /// eviction paths (demote, cold hit, promote) are *reachable* from tests:
    /// with the production cap of 1024 no test could ever fill the hot map.
    cap: usize,
    max_bytes: usize,
}
impl UrlCache {
    fn new() -> Self {
        Self::with_limits(CACHE_CAP, CACHE_MAX_BYTES)
    }
    fn with_limits(cap: usize, max_bytes: usize) -> Self {
        Self {
            hot: rustc_hash::FxHashMap::default(),
            cold: rustc_hash::FxHashMap::default(),
            hot_bytes: 0,
            cap,
            max_bytes,
        }
    }
    fn account(&mut self, key: &[u8], url: &str) {
        self.hot_bytes = self
            .hot_bytes
            .saturating_add(key.len())
            .saturating_add(url.len());
    }
}
// Entry cap *and* byte budget: the cap bounds a pathological run, the budget
// bounds memory per worker thread (keys + percent-encoded URLs), which the
// old entry-only cap did not.
const CACHE_CAP: usize = 1024;
const CACHE_MAX_BYTES: usize = 1 << 20;

/// Run `f` with a thread-local cache sized by `cap`/`max_bytes`, then restore
/// the production limits.  `URLS` is thread-local and libtest gives each test
/// its own thread, so an override cannot leak into another test.
#[cfg(test)]
fn with_cache_limits<R>(cap: usize, max_bytes: usize, f: impl FnOnce() -> R) -> R {
    /// Restores the production limits on drop — including during an unwind.
    ///
    /// The old shape restored only on the normal path, so an assert failing inside `f` left
    /// a cap-8 cache on this thread.  Contained today because libtest gives each test its own
    /// thread, but the sibling `with_hyperlinks_override` already uses this RAII shape and a
    /// future in-thread composition must not be able to leak a cap.
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            URLS.with(|c| *c.borrow_mut() = UrlCache::new());
        }
    }
    URLS.with(|c| *c.borrow_mut() = UrlCache::with_limits(cap, max_bytes));
    let _restore = Restore;
    f()
}

/// `(hot entries, cold entries, hot bytes)` — the eviction state a test needs
/// to prove that demote and promote actually ran.
#[cfg(test)]
fn cache_state() -> (usize, usize, usize) {
    URLS.with(|c| {
        let c = c.borrow();
        (c.hot.len(), c.cold.len(), c.hot_bytes)
    })
}
thread_local! {
    static URLS: std::cell::RefCell<UrlCache> = std::cell::RefCell::new(UrlCache::new());
}
fn file_url_cached(path: &[u8]) -> std::sync::Arc<str> {
    URLS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(hit) = cache.hot.get(path) {
            return std::sync::Arc::clone(hit);
        }
        // Promote: the working set has revisited this path.  The byte budget is
        // enforced here as well as on insert — promoting a whole cold generation
        // back used to re-accumulate bytes with no ceiling, so the documented
        // per-thread bound only held for a cache that never demoted (an all-hit
        // steady state never demotes).  Skipping the promote is still correct: the
        // entry goes back into `cold`.
        //
        // `remove_entry` moves the key's allocation into `hot`; the old shape
        // cloned the `Arc`, re-`to_vec()`d the key and then removed the cold entry
        // — one allocation, one copy and two refcount bumps for a move.
        if let Some((key, url)) = cache.cold.remove_entry(path) {
            let entry_bytes = path.len() + url.len();
            if cache.hot.len() < cache.cap
                && cache.hot_bytes.saturating_add(entry_bytes) <= cache.max_bytes
            {
                cache.account(&key, &url);
                cache.hot.insert(key, std::sync::Arc::clone(&url));
                return url;
            }
            // Budget refused: put the entry back so it stays reachable in `cold`.
            // Removing first and re-inserting keeps the budget check *before* the
            // move, exactly as before.
            let hit = std::sync::Arc::clone(&url);
            cache.cold.insert(key, url);
            return hit;
        }
        let url: std::sync::Arc<str> =
            std::sync::Arc::from(format_file_url(path, wsl_distro_name()));
        // Same shape as the promote check: demote only if the *incoming* entry
        // would not fit.  The old `hot_bytes >= max_bytes` demoted a whole
        // generation to make room for an entry that had room, and left the
        // documented ceiling two bounds instead of one.
        let insert_bytes = path.len() + url.len();
        if cache.hot.len() >= cache.cap
            || cache.hot_bytes.saturating_add(insert_bytes) > cache.max_bytes
        {
            // Demote rather than clear: the previous generation stays reachable.
            let hot = std::mem::take(&mut cache.hot);
            cache.cold = hot;
            cache.hot_bytes = 0;
        }
        // Deliberately still admitted even when the entry alone exceeds
        // `max_bytes`: the documented per-thread ceiling is `2 * max_bytes + one
        // entry` precisely because an oversized path is cached once and then
        // demoted, and `url_cache_promote_respects_the_byte_budget` pins that the
        // promote path never brings it back.  Refusing it here would change that
        // contract (and the test) for a pathological input whose recompute cost is
        // the only thing saved.
        cache.account(path, &url);
        cache.hot.insert(
            path.to_vec().into_boxed_slice(),
            std::sync::Arc::clone(&url),
        );
        url
    })
}

/// Convert the byte buffer to `String`, asserting the encoding invariant.
///
/// Every byte this module emits is ASCII (`URL_SAFE_BITS` is ASCII and a percent
/// escape is three ASCII bytes), so this validation cannot fail — one pass at the
/// end instead of a UTF-8 length check per emitted byte.
#[inline]
fn finish(out: Vec<u8>) -> String {
    // SAFETY: same invariant as the encoder above — see `URL_SAFE_BITS`.  The
    // comment already said this cannot fail; now it is not paid for.
    debug_assert!(out.iter().all(u8::is_ascii));
    unsafe { String::from_utf8_unchecked(out) }
}

/// The prefix a `//`-rooted path must take: the RFC 8089 authority form
/// (`file://server/share`) or the local-path form with a third slash
/// (`file:///C:/x`).
///
/// `//C:/x` is *not* a UNC share — `C:` would be parsed as the URL **host**, so
/// the drive shape needs the third slash.  `//server/share` keeps the authority
/// form.  Both `//` arms ([`format_file_url`]'s WSL/UNC arm and its POSIX arm)
/// call this so they cannot disagree about identical bytes; they did: the WSL
/// arm emitted the two-slash form unconditionally, so `//C:/x` produced
/// `file://C:/x` under WSL and `file:///C:/x` without a distro.
///
/// The `distro.is_some()` clause is subsumed by the WSL arm's own `Some(..)` and
/// only affects paths that cannot reach it (`//` implies a leading `/`), but it
/// is kept so the two call sites pass the same predicate they always did.
#[inline]
fn slash_authority_prefix(rest: &[u8], distro: Option<&str>) -> &'static [u8] {
    let looks_like_drive = rest.len() >= 2
        && rest[1] == b':'
        && (cfg!(windows) || distro.is_some() || rest[0].is_ascii_uppercase());
    if looks_like_drive {
        b"file:///"
    } else {
        b"file://"
    }
}

#[must_use]
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
    let mut out: Vec<u8> = Vec::with_capacity(
        path.len()
            + unsafe_count.saturating_mul(2)
            + distro.map_or(0, |d| d.len().saturating_mul(3).saturating_add(21))
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
            // `//C:/x` is a drive path, not a share: without the third slash
            // `C:` is parsed as the URL *host*.  This arm used to emit the
            // two-slash form unconditionally while the POSIX arm below checked
            // for a drive, so the same bytes produced a different (and
            // malformed) URL depending only on whether a WSL distro was known.
            out.extend_from_slice(slash_authority_prefix(rest, Some(distro)));
            // `//` is URL-safe, so the sub-slice statistics are derived from the
            // full-path scan instead of a second O(n) pass.
            let (rest_first, rest_unsafe) = sub_stats((first_unsafe, unsafe_count), 2);
            encode_file_url_path_pre(&mut out, rest, rest_first, rest_unsafe);
            return finish(out);
        }
        if let Some(stripped) = path.strip_prefix(b"/mnt/") {
            // Bare drive root (`/mnt/c`, no trailing slash) translates too:
            // it becomes `/C:/`, the native-drive URL for the drive root.
            if stripped.len() == 1 && stripped[0].is_ascii_alphabetic() {
                out.extend_from_slice(b"file:///");
                out.push(stripped[0].to_ascii_uppercase());
                out.extend_from_slice(b":/");
                return finish(out);
            }
            if stripped.len() >= 2 && stripped[1] == b'/' && stripped[0].is_ascii_alphabetic() {
                out.extend_from_slice(b"file:///");
                out.push(stripped[0].to_ascii_uppercase());
                out.push(b':');
                let tail = &stripped[1..];
                // `tail` is `path[6..]` (`/mnt/` + the drive byte), both
                // URL-safe, so derive its statistics from the full scan.
                let (tail_first, tail_unsafe) = sub_stats((first_unsafe, unsafe_count), 6);
                encode_file_url_path_pre(&mut out, tail, tail_first, tail_unsafe);
                return finish(out);
            }
        }
        // The prefix is identical for every WSL URL of one process (the distro
        // comes from a process-stable probe), so encode it once instead of
        // re-encoding the distro per uncached path.  A different distro (tests)
        // falls back to the direct path rather than thrashing the slot.
        // The distro is a **path segment** here, not the URL authority: RFC 3986 §3.3
        // allows `:` in a segment (`pchar`), so the path-safe encoder is the correct one
        // and a host-safe set would rewrite bytes without fixing anything.  (`:`
        // unencoded is also what `wsl.localhost/U:buntu/...` needs to round-trip, since
        // the distro name is opaque to the client.)  The review of this code read the
        // position as "authority"; `a_colon_in_the_distro_stays_a_legal_path_segment`
        // pins the real shape so a future edit cannot silently switch encoders.
        static WSL_PREFIX: std::sync::OnceLock<(Box<str>, Box<[u8]>)> = std::sync::OnceLock::new();
        let cached = WSL_PREFIX.get_or_init(|| {
            let mut v: Vec<u8> = Vec::with_capacity(32);
            v.extend_from_slice(b"file://wsl.localhost/");
            encode_url_component_into(&mut v, distro.as_bytes());
            (Box::from(distro), v.into_boxed_slice())
        });
        if &*cached.0 == distro {
            out.extend_from_slice(&cached.1);
        } else {
            out.extend_from_slice(b"file://wsl.localhost/");
            encode_url_component_into(&mut out, distro.as_bytes());
        }
        encode_file_url_path_pre(&mut out, path, first_unsafe, unsafe_count);
        return finish(out);
    }

    // Windows drive letter: `C:/foo` → `file:///C:/foo` (three slashes).
    // An *uppercase* drive letter is accepted everywhere (a test/host may pass
    // a Windows path on POSIX); a lowercase `c:/x` is treated as an ordinary
    // POSIX directory name unless we are actually on Windows or a WSL distro
    // is in play.
    //
    // Consequently the final fallback below can only emit `file://c:/x` — whose
    // `c` parses as a URL *host* — for input the public `path_to_file_url` cannot
    // produce on POSIX (absolute POSIX paths start with `/`, so `path[1]` is
    // never `:`).  It is a defensive/test-only shape, pinned by
    // `format_file_url_wsl_and_windows`; do not "fix" it without deciding what a
    // POSIX file literally named `c:` should link to.
    let looks_like_drive = path.len() >= 2
        && path[1] == b':'
        && (cfg!(windows) || distro.is_some() || path[0].is_ascii_uppercase());
    if looks_like_drive {
        out.extend_from_slice(b"file:///");
        encode_file_url_path_pre(&mut out, path, first_unsafe, unsafe_count);
        if path.len() == 2 {
            // Bare drive root (`C:`, e.g. from `\\?\C:` after the verbatim
            // prefix is stripped): every other root form ends with `/`
            // (`file:///C:/`), so append the one the strip removed.
            out.push(b'/');
        }
        return finish(out);
    }

    // POSIX `//server/share/file` is the legacy authority spelling; emit the
    // RFC 8089 canonical form `file://server/share/file` instead of the
    // four-slash form strict parsers treat differently.
    if let Some(rest) = path.strip_prefix(b"//") {
        out.extend_from_slice(slash_authority_prefix(rest, distro));
        let (rest_first, rest_unsafe) = sub_stats((first_unsafe, unsafe_count), 2);
        encode_file_url_path_pre(&mut out, rest, rest_first, rest_unsafe);
        return finish(out);
    }

    out.extend_from_slice(b"file://");
    encode_file_url_path_pre(&mut out, path, first_unsafe, unsafe_count);
    finish(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a [`ColorEnv`] from named test inputs.
    ///
    /// A struct, not six positional arguments: `clicolor_force` and
    /// `force_color` are both `Option<&str>` and a swapped pair type-checks and
    /// can even pass, which is exactly what `ColorEnv` exists to prevent.
    fn env(
        no_color: bool,
        clicolor: Option<&'static str>,
        clicolor_force: Option<&'static str>,
        force_color: Option<&'static str>,
        term: Option<&'static str>,
        stderr_is_tty: bool,
    ) -> ColorEnv<'static> {
        ColorEnv {
            no_color,
            clicolor: clicolor.map(OsStr::new),
            clicolor_force: clicolor_force.map(OsStr::new),
            force_color: force_color.map(OsStr::new),
            term: term.map(OsStr::new),
            stderr_is_tty,
            ..ColorEnv::default()
        }
    }

    fn color(
        no_color: bool,
        clicolor: Option<&'static str>,
        clicolor_force: Option<&'static str>,
        force_color: Option<&'static str>,
        term: Option<&'static str>,
        tty: bool,
    ) -> bool {
        color_decision_env(&env(
            no_color,
            clicolor,
            clicolor_force,
            force_color,
            term,
            tty,
        ))
    }

    /// `unicode_decision_os` with an optional `&str` locale: the empty-means-unset
    /// rule is owned by the core now, so tests exercise the same path production
    /// does.
    fn unicode(locale: Option<&str>, term_dumb: bool) -> bool {
        unicode_decision_os(locale.map(OsStr::new), term_dumb)
    }

    /// Child mode for [`env_wiring_drives_term_caps`].
    ///
    /// A no-op unless the parent asks for it.  The production detectors are
    /// per-capability `LazyLock` snapshots, so the *only* way to probe a given
    /// environment is a fresh process — the test binary re-executes itself with
    /// a scrubbed env instead.  (`TermCaps::detect` itself always re-probes; it
    /// is the one non-memoized path, which is exactly why this test cannot just
    /// call it with a mutated environment.)
    #[test]
    fn caps_probe_child() {
        if std::env::var_os("PONENT_CAPS_PROBE").is_none() {
            return;
        }
        let caps = TermCaps::detect();
        let depth = match caps.depth {
            ColorDepth::Sixteen => "16",
            ColorDepth::TwoFiftySix => "256",
            ColorDepth::TrueColor => "truecolor",
        };
        println!(
            "caps color={} depth={depth} unicode={} width={} hyperlinks={}",
            caps.color,
            caps.unicode,
            caps.width,
            // `caps.hyperlinks`, **not** `supports_hyperlinks()`: the latter is
            // short-circuited by `HYPERLINK_OVERRIDE`, which is `Some(false)` on every
            // thread of a test build — including this re-executed child's test thread. So
            // the printed value was a constant `false` and the three `["hyperlinks"]`
            // assertions in `env_wiring_drives_term_caps` passed no matter what the env
            // wiring did (a deleted `NO_HYPERLINK` veto or a broken `TERM=dumb` link veto
            // would have gone unnoticed). `TermCaps::detect` is deliberately override-free,
            // which is what makes this probe honest.
            caps.hyperlinks,
        );
    }

    /// The four WSL policies, as a pure table.
    ///
    /// The probe above was the last decision core in this module with no test object:
    /// the cfg gate, empty-name rejection, **fail-closed on an unreadable
    /// `/proc/version`** and the kernel-substring test all lived inline.
    #[test]
    fn wsl_distro_decision_applies_all_four_policies() {
        let wsl_kernel = "Linux version 5.15.0-1-microsoft-standard-WSL2";
        // 1. Not Linux: never translate (there is no `/proc` to verify against).
        assert_eq!(
            wsl_distro_decision(false, Some("Ubuntu"), Some(wsl_kernel)),
            None
        );
        // 2. Missing or empty name.
        assert_eq!(wsl_distro_decision(true, None, Some(wsl_kernel)), None);
        assert_eq!(wsl_distro_decision(true, Some(""), Some(wsl_kernel)), None);
        // 3. Unreadable `/proc/version` fails **closed**.
        assert_eq!(wsl_distro_decision(true, Some("Ubuntu"), None), None);
        // 4. A native Linux kernel is not WSL, despite a forwarded env var.
        assert_eq!(
            wsl_distro_decision(
                true,
                Some("Ubuntu"),
                Some("Linux version 6.8.0-generic (gcc 13.2.0)")
            ),
            None
        );
        // Both accepted spellings, ASCII case-insensitively.
        for kernel in [
            wsl_kernel,
            "Linux version 4.4.0-19041-Microsoft",
            "Linux version 5.10.16.3-microsoft-standard-WSL2",
            "linux version 5.15 WSL",
        ] {
            assert_eq!(
                wsl_distro_decision(true, Some("Ubuntu-22.04"), Some(kernel)),
                Some("Ubuntu-22.04"),
                "{kernel:?}"
            );
        }
        // `wsl` must match as a substring of the kernel string, not as a prefix.
        assert_eq!(
            wsl_distro_decision(true, Some("Debian"), Some("Linux 5.15 gnu-wsl-build")),
            Some("Debian")
        );
    }

    /// The env→`TermCaps` *wiring* is the half the pure decision-core tests
    /// cannot reach: which variable feeds which field, the trim/filter rules and
    /// the locale precedence.  A swapped field (`LANG` ↔ `LC_CTYPE`) or a typo
    /// would pass every other test in this module.
    ///
    /// Spawns a child process, so it is meaningless under Miri (no process
    /// spawn) and in sandboxes that forbid it — skip rather than fail.  (This block
    /// and its `#[cfg_attr]` used to sit above `wsl_distro_decision_applies_all_four_policies`
    /// — a pure table test that must run everywhere — so Miri CI went red on the spawn it
    /// never performed while the WSL policies were skipped for no reason.)
    /// `WT_SESSION` follows the same "empty means unset" rule as `NO_COLOR`.
    ///
    /// A templated CI matrix materialises `WT_SESSION=""`, and treating that as "running
    /// under Windows Terminal" handed every such job a truecolor depth it could not
    /// render.  A whitespace-only value stays "set" (it is not empty, and no real WT
    /// session is whitespace) — the rule is emptiness, not trim.
    #[test]
    fn wt_session_is_gated_on_emptiness_not_on_trim() {
        let caps = |wt: Option<&'static str>| {
            let mut env = ColorEnv {
                stderr_is_tty: true,
                ..ColorEnv::default()
            };
            env.wt_session = wt.is_some_and(|v| !v.is_empty());
            color_caps_decision(&env).depth
        };
        assert_eq!(caps(Some("")), ColorDepth::Sixteen, "empty is unset");
        assert_eq!(caps(None), ColorDepth::Sixteen);
        assert_eq!(
            caps(Some(" ")),
            ColorDepth::TrueColor,
            "whitespace is still set"
        );
        assert_eq!(caps(Some("3f2a")), ColorDepth::TrueColor);
    }

    /// The §5 boundary values, each pinned to the documented rule rather than to
    /// whatever the tables happen to produce.
    #[test]
    fn documented_boundaries_hold_for_empty_and_punctuation_values() {
        // `CLICOLOR=""` is *unset* (templated CI matrices materialise empty strings),
        // so it neither forces nor vetoes: the tty answer stands.
        assert!(color(false, Some(""), None, None, Some("xterm"), true));
        assert!(!color(false, Some(""), None, None, Some("xterm"), false));
        // `FORCE_COLOR="-"` is non-numeric garbage, not a negative number: only a real
        // digit run may silence colour (pinned by `force_color_is_off`'s contract).
        assert!(color(false, None, None, Some("-"), None, false));
        assert!(!color(false, None, None, Some("-1"), None, false));
        // `COLUMNS="\u{a0}80"`: `str::trim` is Unicode-aware and `char::is_whitespace()`
        // covers U+00A0, so the NBSP *is* trimmed and the width is 80.  Pinned because
        // the previous expectation here (fall back to the probe) was wrong — the
        // documented rule is "trim, then parse", not "ASCII-trim only".
        assert_eq!(
            width_decision(Some("\u{a0}80"), Some(100)),
            80,
            "NBSP is Unicode whitespace and must be trimmed"
        );
        // A whitespace-only value is not a width at all.
        assert_eq!(width_decision(Some(" \u{a0} "), Some(100)), 100);
        // A `//server/share` path is a native UNC path: it must never be routed through
        // `wsl.localhost` (that makes 9P traverse a CIFS mount), even with a distro known.
        let unc = format_file_url(b"//server/share/x", Some("Ubuntu"));
        assert!(
            !unc.contains("wsl.localhost"),
            "a UNC path must keep its native form: {unc}"
        );
    }

    /// A distro name is a path segment, so `:` stays a legal `pchar`.
    ///
    /// `file://wsl.localhost/<distro>/...` puts the distro in the first path segment
    /// (RFC 3986 §3.3 allows `:` there), and a space still gets encoded.  See the note at
    /// A bare Windows drive root stays native even when a distro is known.
    ///
    /// Reachable on POSIX, where `C:` is just two bytes: the drive-letter branch must win
    /// over the WSL one, or the URL becomes `file://wsl.localhost/<distro>/C:` and 9P has
    /// to traverse a path that does not exist.  The bare-root trailing-slash append was
    /// unpinned.
    #[test]
    fn a_bare_drive_root_with_a_distro_stays_native() {
        assert_eq!(
            format_file_url(b"C:", Some("Ubuntu")),
            "file:///C:/",
            "the bare-root branch appends the trailing slash"
        );
        assert_eq!(
            format_file_url(b"C:/Users/x", Some("Ubuntu")),
            "file:///C:/Users/x",
            "a path below the drive keeps its prefix"
        );
    }

    /// the emission site for why the path-safe set — not the host-safe one — is right.
    #[test]
    fn a_colon_in_the_distro_stays_a_legal_path_segment() {
        assert_eq!(
            format_file_url(b"/home/u/x", Some("U:buntu")),
            "file://wsl.localhost/U:buntu/home/u/x"
        );
        assert_eq!(
            format_file_url(b"/home/u/x", Some("Ubuntu 20.04")),
            "file://wsl.localhost/Ubuntu%2020.04/home/u/x",
            "a space in the distro must still be encoded"
        );
    }

    /// `truecolor` is matched without digit stripping, while `direct` strips its digit
    /// run — the asymmetry is deliberate and must not be "unified".
    ///
    /// `xterm-direct2`/`xterm-direct3` are stock ncurses entries (the digit is part of
    /// the name), whereas a trailing digit after `truecolor` is not a stock spelling, so
    /// `xterm-truecolor2` is *not* a truecolor terminal.  Both directions are pinned:
    /// a well-meaning "make the two branches consistent" edit would change one of them.
    #[test]
    fn direct_strips_its_digit_run_but_truecolor_does_not() {
        let depth = |term: &'static str| {
            color_depth_decision_env(&env(false, None, None, None, Some(term), false))
        };
        assert_eq!(depth("xterm-direct"), ColorDepth::TrueColor);
        assert_eq!(depth("xterm-direct2"), ColorDepth::TrueColor);
        assert_eq!(depth("xterm-direct16"), ColorDepth::TrueColor);
        assert_eq!(
            depth("indirect"),
            ColorDepth::Sixteen,
            "`direct` needs a token boundary"
        );
        assert_eq!(depth("xterm-truecolor"), ColorDepth::TrueColor);
        assert_eq!(
            depth("xterm-truecolor2"),
            ColorDepth::Sixteen,
            "a digit after `truecolor` is not a stock spelling"
        );
    }

    /// The `dumb` **family** is vetoed; a name that merely contains `dumb` is not.
    ///
    /// The predicate is a prefix test because `TERM` is a structured terminfo name whose
    /// `dumb` is a family root (`dumb-emacs-ansi`, `dumb-vt100`, …), and the asymmetry is
    /// the part that matters: `dumbx` and `xterm-dumb` are honest terminal names that a
    /// substring test would wrongly veto.
    #[test]
    fn dumb_term_covers_the_whole_family() {
        for name in [
            "dumb",
            "DUMB",
            "dumb-emacs-ansi",
            "dumb-vt100",
            "dumb-emacs",
            // The family root with an empty suffix.  It is not a stock terminfo entry, so
            // either answer is defensible — but the boundary must be *decided and pinned*:
            // the previous `t.len() > 5` silently excluded exactly this spelling.
            "dumb-",
        ] {
            assert!(
                is_dumb_term(Some(std::ffi::OsStr::new(name))),
                "{name} belongs to the dumb family and must be vetoed"
            );
        }
        for name in ["xterm-256color", "dumbx", "xterm-dumb", "dumb_emacs", ""] {
            assert!(
                !is_dumb_term(Some(std::ffi::OsStr::new(name))),
                "{name} is not the dumb family and must not be vetoed"
            );
        }
        assert!(!is_dumb_term(None), "an unset TERM is not `dumb`");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn env_wiring_drives_term_caps() {
        use std::collections::BTreeMap;

        fn probe(vars: &[(&str, &str)]) -> BTreeMap<String, String> {
            let mut cmd =
                std::process::Command::new(std::env::current_exe().expect("test binary path"));
            cmd.arg("--exact")
                // `module_path!()` includes the crate name in a bin target
                // (`ponent::diagnostics::term::tests`), while libtest's filters are
                // written without it — strip it so the wiring test keeps working
                // across a module rename instead of matching nothing.
                .arg(format!(
                    "{}::caps_probe_child",
                    module_path!()
                        .strip_prefix(concat!(env!("CARGO_CRATE_NAME"), "::"))
                        .unwrap_or(module_path!())
                ))
                .arg("--nocapture")
                .arg("--test-threads=1")
                .env_clear()
                .env("PONENT_CAPS_PROBE", "1")
                // Keep the child runnable without inheriting the developer's real
                // terminal environment (that is the whole point).
                .env("HOME", std::env::temp_dir());
            // Windows: `env_clear` also drops `SystemRoot`/`windir`, which the loader
            // needs to find `kernel32`/the CRT (rust-lang/rust#114737) — the child then
            // dies before running a single test, and the failure surfaces as "probe child
            // failed" rather than as a wiring bug.  Forward the Win32 setup variables and
            // nothing else that could influence the probe.
            #[cfg(windows)]
            for key in [
                "SystemRoot",
                "windir",
                "SystemDrive",
                "TEMP",
                "TMP",
                "PATHEXT",
                "COMSPEC",
            ] {
                if let Some(value) = std::env::var_os(key) {
                    cmd.env(key, value);
                }
            }
            // `PATH` is deliberately **not** forwarded: the child is `current_exe`, an
            // absolute path, so nothing needs resolving — inheriting the developer's
            // `PATH` only widened what the probe could accidentally execute.
            for (k, v) in vars {
                cmd.env(k, v);
            }
            let out = cmd.output().expect("spawn caps probe child");
            assert!(
                out.status.success(),
                "probe child failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let text = String::from_utf8_lossy(&out.stdout);
            // libtest prints a test's captured output inline with its status
            // line, so locate the payload by its own prefix rather than by the
            // start of the line.
            let start = text
                .find("caps color=")
                .unwrap_or_else(|| panic!("no caps payload in {text:?}"));
            let line = &text[start..];
            line.split_whitespace()
                .skip(1)
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        }

        // stderr is a pipe in the child, so `color` is only ever forced on by env.
        let plain = probe(&[]);
        assert_eq!(plain["color"], "false", "no tty and nothing forced");
        assert_eq!(plain["depth"], "16");
        assert_eq!(plain["unicode"], "true", "no locale means assume UTF-8");
        assert_eq!(plain["width"], DEFAULT_MAX_WIDTH.to_string());
        assert_eq!(plain["hyperlinks"], "false", "no tty, nothing forced");

        // Colour: suppress beats force, force wins over no-tty, and the level
        // feeds `depth` independently of the on/off bit.
        assert_eq!(probe(&[("NO_COLOR", "1")])["color"], "false");
        assert_eq!(probe(&[("FORCE_COLOR", "0")])["color"], "false");
        let lvl1 = probe(&[("FORCE_COLOR", "1")]);
        assert_eq!(
            (lvl1["color"].as_str(), lvl1["depth"].as_str()),
            ("true", "16")
        );
        let lvl3 = probe(&[("FORCE_COLOR", "3")]);
        assert_eq!(
            (lvl3["color"].as_str(), lvl3["depth"].as_str()),
            ("true", "truecolor")
        );
        // `CLICOLOR_FORCE` also forces, but its empty value means unset.
        assert_eq!(probe(&[("CLICOLOR_FORCE", "1")])["color"], "true");
        assert_eq!(probe(&[("CLICOLOR_FORCE", "")])["color"], "false");
        // Empty values mean "unset" for the no-color.org switches, and a
        // non-empty one suppresses even an explicit force.  These are what a
        // templated CI matrix materializes, and the
        // `var_os().is_some_and(|v| !v.is_empty())` chain had no end-to-end probe
        // (the unit helper takes a `bool`, so it could not express them).
        assert_eq!(
            probe(&[("NO_COLOR", ""), ("FORCE_COLOR", "1")])["color"],
            "true",
            "NO_COLOR='' must not disable colour"
        );
        assert_eq!(
            probe(&[("NO_COLOR", "1"), ("FORCE_COLOR", "1")])["color"],
            "false",
            "a non-empty NO_COLOR suppresses even an explicit force"
        );
        assert_eq!(
            probe(&[("WT_SESSION", ""), ("TERM", "xterm")])["depth"],
            "16",
            "an empty WT_SESSION must not upgrade the depth"
        );
        // Without a tty only the "not forced on" direction is observable, which is
        // exactly the clause an empty value must not trip.
        assert_eq!(probe(&[("NO_HYPERLINK", "")])["hyperlinks"], "false");
        // `NO_COLOR` keeps the raw "present and non-empty" rule (no trimming):
        // a padded value is still present, so it suppresses even an explicit
        // force.  The trimmed numeric switches are the deliberate exception.
        assert_eq!(
            probe(&[("NO_COLOR", " "), ("FORCE_COLOR", "1")])["color"],
            "false",
            "a padded NO_COLOR is still present and must suppress"
        );
        // Arm order: `FORCE_COLOR` precedes `CLICOLOR`, so a CI matrix exporting both sees
        // the force.  Unpinned before, and a reordering "cleanup" would flip it silently.
        assert_eq!(
            probe(&[("CLICOLOR", "0"), ("FORCE_COLOR", "1")])["color"],
            "true",
            "FORCE_COLOR must precede CLICOLOR"
        );
        // A forced level precedes the terminal-family signal: `WT_SESSION` alone upgrades
        // the depth, but `FORCE_COLOR=1` (16) must win.  Level-vs-`COLORTERM` was pinned,
        // level-vs-`WT_SESSION` was not.
        assert_eq!(
            probe(&[("FORCE_COLOR", "1"), ("WT_SESSION", "abc")])["depth"],
            "16",
            "the forced level must precede the WT_SESSION family signal"
        );
        // `NO_COLOR` is a raw presence rule: `"0"` is present, so it suppresses.  A
        // "helpful" `!= "0"` special case (which naive implementations add) would break
        // every caller that exports it for other tools.
        assert_eq!(probe(&[("NO_COLOR", "0")])["color"], "false");
        // A whitespace-only locale is as unset as an empty one, so box drawing survives a
        // host with a blank `LC_ALL`.
        assert_eq!(
            probe(&[("LC_ALL", " ")])["unicode"],
            "true",
            "a blank locale must be treated as unset"
        );

        // A colour-capable terminal still needs a tty (or a force) to *emit*.
        let ct = probe(&[("COLORTERM", "truecolor")]);
        assert_eq!(ct["color"], "false");
        assert_eq!(ct["depth"], "truecolor");

        // `TERM=dumb` vetoes both colour and Unicode, case-insensitively.
        for dumb in ["dumb", "DUMB"] {
            let d = probe(&[("TERM", dumb), ("FORCE_COLOR", "0")]);
            assert_eq!(d["color"], "false", "TERM={dumb}");
            assert_eq!(d["unicode"], "false", "TERM={dumb}");
        }

        // Locale feeds `unicode` only.
        assert_eq!(probe(&[("LANG", "ru_RU.UTF-8")])["unicode"], "true");
        assert_eq!(probe(&[("LANG", "C")])["unicode"], "false");
        assert_eq!(probe(&[("LC_ALL", "en_US.utf8")])["unicode"], "true");
        // LC_ALL outranks LC_CTYPE outranks LANG.
        let prec = probe(&[("LC_ALL", "C"), ("LANG", "ru_RU.UTF-8")]);
        assert_eq!(prec["unicode"], "false", "LC_ALL wins over LANG");

        // The trim-vs-present asymmetry between the two force switches: a blank
        // `CLICOLOR_FORCE` is "unset" (it is trimmed), while `NO_COLOR`'s rule stays
        // raw presence.  Only the env→core wiring can see the difference, because the
        // unit helper takes an already-classified `bool`.
        assert_eq!(
            probe(&[("CLICOLOR_FORCE", " ")])["color"],
            "false",
            "a blank CLICOLOR_FORCE is unset"
        );
        assert_eq!(
            probe(&[("FORCE_COLOR", " 2 ")])["depth"],
            "256",
            "a padded numeric FORCE_COLOR parses after trimming"
        );
        // `TERM_PROGRAM` is matched **exactly**, not by substring: a fork whose name
        // merely contains the family string must not inherit its depth.
        assert_eq!(
            probe(&[("TERM_PROGRAM", "not-iTerm.app")])["depth"],
            "16",
            "TERM_PROGRAM must be an exact match"
        );
        assert_eq!(
            probe(&[("TERM_PROGRAM", "iTerm.app")])["depth"],
            "truecolor"
        );
        // Windows Terminal's session marker upgrades the depth.
        assert_eq!(probe(&[("WT_SESSION", "x")])["depth"], "truecolor");
        // `TERM=…direct2` is `direct` as a token; `indirect` is not.  Both directions
        // matter: the digit suffix must not defeat the rule, and the substring must not
        // win it.
        assert_eq!(probe(&[("TERM", "xterm-direct2")])["depth"], "truecolor");
        assert_eq!(probe(&[("TERM", "indirect")])["depth"], "16");
        // Snapshot coherence: `color` and `depth` come from one `COLOR_CAPS`, so
        // suppressing colour cannot also reset the detected depth.  (Regression shape:
        // two independent env reads disagreeing.)
        let suppressed = probe(&[("NO_COLOR", "1"), ("COLORTERM", "truecolor")]);
        assert_eq!(
            (suppressed["color"].as_str(), suppressed["depth"].as_str()),
            ("false", "truecolor"),
            "colour suppression must not erase the detected depth"
        );
        // `NO_HYPERLINK=1`: only the off-direction is observable in a pipe (the probe
        // is already `false` there), which is why the *empty* case above is the
        // meaningful half — recorded so nobody reads this as a false-positive test.
        assert_eq!(probe(&[("NO_HYPERLINK", "1")])["hyperlinks"], "false");

        // `COLUMNS` feeds `width`, including the overflow clamp.
        assert_eq!(probe(&[("COLUMNS", "120")])["width"], "120");
        assert_eq!(probe(&[("COLUMNS", &"9".repeat(25))])["width"], "65535");
        assert_eq!(
            probe(&[("COLUMNS", "0")])["width"],
            DEFAULT_MAX_WIDTH.to_string()
        );
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

    /// A relative path that does not exist yet still has to produce a usable
    /// link: `std::path::absolute` joins the CWD lexically instead of failing,
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

    /// The empty-string conventions collide inside `color_decision_env`:
    /// `CLICOLOR_FORCE=""` is unset, `FORCE_COLOR=""` is force-**on** (chalk
    /// compatibility).  Named so a future "cleanup" that unifies them fails
    /// here instead of silently changing which CI matrices emit ANSI.
    #[test]
    fn force_color_empty_diverges_from_clicolor_force_empty() {
        // Piped output, otherwise no reason to colour.
        assert!(
            color(false, None, None, Some(""), None, false),
            "FORCE_COLOR=\"\" means force-on (chalk compat)"
        );
        assert!(
            !color(false, None, Some(""), None, None, false),
            "CLICOLOR_FORCE=\"\" means unset"
        );
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
        assert!(unicode(Some("en_US.UTF-8"), false));
        assert!(unicode(Some("C.utf8"), false));
        assert!(!unicode(Some("C"), false));
        assert!(!unicode(Some("POSIX"), false));
        // No locale information at all → assume UTF-8.
        assert!(unicode(None, false));
        // `TERM=dumb` is a veto regardless of locale.
        assert!(!unicode(Some("en_US.UTF-8"), true));
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
                term: Some(OsStr::new("xterm")),
                wt_session: true,
                ..ColorEnv::default()
            }),
            ColorDepth::TrueColor
        );
        for tp in ["iTerm.app", "WezTerm", "vscode", "WarpTerminal", "Warp"] {
            assert_eq!(
                color_depth_decision_env(&ColorEnv {
                    term: Some(OsStr::new("xterm")),
                    term_program: Some(OsStr::new(tp)),
                    ..ColorEnv::default()
                }),
                ColorDepth::TrueColor,
                "{tp}"
            );
        }
        // JetBrains terminals set no `COLORTERM`; kitty may not either.
        assert_eq!(
            color_depth_decision_env(&ColorEnv {
                term: Some(OsStr::new("xterm")),
                terminal_emulator: Some(OsStr::new("JetBrains-JediTerm")),
                ..ColorEnv::default()
            }),
            ColorDepth::TrueColor
        );
        assert_eq!(
            color_depth_decision_env(&ColorEnv {
                term: Some(OsStr::new("xterm-kitty")),
                ..ColorEnv::default()
            }),
            ColorDepth::TrueColor
        );
        // An unrelated TERM_PROGRAM must not upgrade the depth.
        assert_eq!(
            color_depth_decision_env(&ColorEnv {
                term: Some(OsStr::new("xterm")),
                term_program: Some(OsStr::new("Apple_Terminal")),
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
                color_depth_decision(
                    Some(OsStr::new("truecolor")),
                    Some(OsStr::new("xterm")),
                    Some(OsStr::new(v)),
                ),
                ColorDepth::TrueColor,
                "FORCE_COLOR={v:?}"
            );
            assert!(
                color(false, None, None, Some(v), Some("dumb"), false),
                "FORCE_COLOR={v:?} still forces colour on"
            );
        }
    }

    /// `FORCE_COLOR="   "` trims to empty, which is the chalk force-on rule —
    /// the same answer as `FORCE_COLOR=""`.  Unpinned until now, so a templated
    /// CI matrix materialising a padded value could have been "fixed" into
    /// silence without any test objecting.
    #[test]
    fn force_color_whitespace_only_forces_on() {
        for v in ["", " ", "   ", "\t"] {
            assert!(!force_color_is_off(OsStr::new(v)), "{v:?}");
            assert_eq!(force_color_level(OsStr::new(v)), None, "{v:?}");
            assert!(
                color(false, None, None, Some(v), Some("dumb"), false),
                "{v:?} should force color on"
            );
        }
        // Padded numeric values keep their level too.
        assert_eq!(
            force_color_level(OsStr::new(" 2 ")),
            Some(ColorDepth::TwoFiftySix)
        );
        assert!(force_color_is_off(OsStr::new(" false ")));
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
    fn force_color_classifies_any_magnitude() {
        // 40 digits: longer than `i128`/`u128` can hold, in both directions and
        // with an explicit `+`.  The old `i128`/`u128` parsers each had a hole
        // one magnitude further out; digit classification is exact everywhere.
        let long_pos = "9".repeat(40);
        let long_neg = format!("-{}", "9".repeat(40));
        assert_eq!(
            force_color_level(OsStr::new(&long_pos)),
            Some(ColorDepth::TrueColor)
        );
        assert_eq!(
            force_color_level(OsStr::new(&format!("+{long_pos}"))),
            Some(ColorDepth::TrueColor)
        );
        assert!(force_color_is_off(OsStr::new(&long_neg)));
        assert!(!force_color_is_off(OsStr::new(&long_pos)));
        // `-0` is zero, hence off; `+0`/`00` too.
        for zero in ["-0", "+0", "0", "00", "000"] {
            assert!(force_color_is_off(OsStr::new(zero)), "{zero}");
            assert_eq!(force_color_level(OsStr::new(zero)), None, "{zero}");
        }
        // Leading zeros do not change the level class.
        assert_eq!(
            force_color_level(OsStr::new("03")),
            Some(ColorDepth::TrueColor)
        );
        assert_eq!(
            force_color_level(OsStr::new("0002")),
            Some(ColorDepth::TwoFiftySix)
        );
    }

    #[test]
    fn columns_overflow_clamps_instead_of_unsetting() {
        // 25 digits: past `usize`.  It is still a huge positive width, so the
        // documented `u16::MAX` ceiling applies — not a silent fallback.
        assert_eq!(
            width_decision(Some(&"9".repeat(25)), None),
            u16::MAX as usize
        );
        assert_eq!(width_decision(Some("+0"), Some(120)), 120);
        assert_eq!(width_decision(Some("-80"), Some(120)), 120);
        assert_eq!(width_decision(Some("0"), Some(120)), 120);
        // Leading zeros parse successfully to a *tiny* value, so the overflow
        // clamp gives no protection: this is spec-honest (`1` is a positive
        // integer) but every downstream `width - used` must saturate.  Pinned so
        // a future "reject implausible widths" change is a visible decision.
        assert_eq!(
            width_decision(Some(&format!("{}1", "0".repeat(24))), None),
            1
        );
    }

    /// `direct` may carry a digit suffix (`xterm-direct2`, `xterm-direct3` are
    /// stock ncurses entries).  Still a suffix test: `indirect` must not match.
    #[test]
    fn term_direct_with_digit_suffix_is_truecolor() {
        for term in [
            "xterm-direct",
            "xterm-direct2",
            "xterm-direct3",
            "XTERM-DIRECT16",
        ] {
            assert_eq!(
                color_depth_decision_env(&ColorEnv {
                    term: Some(OsStr::new(term)),
                    ..ColorEnv::default()
                }),
                ColorDepth::TrueColor,
                "{term}"
            );
        }
        // `xterm-direct-2024`: trailing digits strip to `xterm-direct-`, which
        // does not end in `direct`, so it falls through to Sixteen.  Conservative
        // and correct; pinned so a future "strip digits more aggressively" edit
        // cannot silently promote it.
        for term in [
            "indirect",
            "xterm-direct-foo",
            "xterm-directx",
            "xterm-direct-2024",
        ] {
            assert_eq!(
                color_depth_decision_env(&ColorEnv {
                    term: Some(OsStr::new(term)),
                    ..ColorEnv::default()
                }),
                ColorDepth::Sixteen,
                "{term}"
            );
        }
    }

    /// A non-UTF-8 `FORCE_COLOR` cannot be decoded, so it is not an explicit
    /// "off" (`false` is the only off spelling) — colour stays forced on, but no
    /// depth is pinned.  The rule lived in a match arm nothing exercised.
    #[cfg(unix)]
    #[test]
    fn non_utf8_force_color_forces_on_without_pinning_depth() {
        use std::os::unix::ffi::OsStrExt;
        let raw = OsStr::from_bytes(&[0xFF, 0xFE]);
        assert!(
            !force_color_is_off(raw),
            "undecodable is not an off request"
        );
        assert_eq!(force_color_level(raw), None, "and it pins no depth");
        assert!(color_decision_env(&ColorEnv {
            force_color: Some(raw),
            ..ColorEnv::default()
        }));
    }

    #[test]
    fn dumb_term_is_case_insensitive() {
        for t in ["dumb", "DUMB", "Dumb", "dUmB"] {
            assert!(is_dumb_term(Some(OsStr::new(t))), "{t}");
        }
        for t in ["xterm", "dumber", ""] {
            assert!(!is_dumb_term(Some(OsStr::new(t))), "{t}");
        }
        assert!(!is_dumb_term(None));
    }

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
            color_depth_decision(None, Some(OsStr::new("xterm")), Some(OsStr::new("256"))),
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
            color_depth_decision(
                Some(OsStr::new("truecolor")),
                Some(OsStr::new("xterm")),
                None
            ),
            ColorDepth::TrueColor
        );
        // COLORTERM=24bit → TrueColor (case-insensitive)
        assert_eq!(
            color_depth_decision(Some(OsStr::new("24BIT")), None, None),
            ColorDepth::TrueColor
        );
        // TERM ends with 256color → TwoFiftySix (case-insensitive).
        assert_eq!(
            color_depth_decision(None, Some(OsStr::new("xterm-256color")), None),
            ColorDepth::TwoFiftySix
        );
        assert_eq!(
            color_depth_decision(None, Some(OsStr::new("XTERM-256COLOR")), None),
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
                color_depth_decision(None, Some(OsStr::new(term)), None),
                ColorDepth::TrueColor,
                "{term}"
            );
        }
        // Neither → Sixteen
        assert_eq!(
            color_depth_decision(None, Some(OsStr::new("xterm")), None),
            ColorDepth::Sixteen
        );
        assert_eq!(color_depth_decision(None, None, None), ColorDepth::Sixteen);
        // COLORTERM takes precedence over TERM.
        assert_eq!(
            color_depth_decision(
                Some(OsStr::new("truecolor")),
                Some(OsStr::new("xterm")),
                None
            ),
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
        assert!(unicode(Some(""), false));
        assert!(contains_ignore_ascii_case(b"anything", b""));
    }

    #[test]
    fn force_color_level_sets_depth() {
        assert_eq!(
            color_depth_decision(None, Some(OsStr::new("xterm")), Some(OsStr::new("1"))),
            ColorDepth::Sixteen
        );
        assert_eq!(
            color_depth_decision(None, Some(OsStr::new("xterm")), Some(OsStr::new("2"))),
            ColorDepth::TwoFiftySix
        );
        assert_eq!(
            color_depth_decision(None, Some(OsStr::new("xterm")), Some(OsStr::new("3"))),
            ColorDepth::TrueColor
        );
        // Level wins over COLORTERM.
        assert_eq!(
            color_depth_decision(
                Some(OsStr::new("truecolor")),
                Some(OsStr::new("xterm")),
                Some(OsStr::new("1"))
            ),
            ColorDepth::Sixteen
        );
    }

    #[test]
    fn caps_are_coherent() {
        let caps = color_caps_decision(&ColorEnv {
            force_color: Some(OsStr::new("2")),
            term: Some(OsStr::new("xterm-256color")),
            stderr_is_tty: true,
            ..ColorEnv::default()
        });
        assert!(caps.color);
        assert_eq!(caps.depth, ColorDepth::TwoFiftySix);

        let caps = color_caps_decision(&ColorEnv {
            force_color: Some(OsStr::new("0")),
            colorterm: Some(OsStr::new("truecolor")),
            term: Some(OsStr::new("xterm")),
            stderr_is_tty: true,
            ..ColorEnv::default()
        });
        assert!(!caps.color);
    }

    #[test]
    fn hyperlink_overrides() {
        assert!(!hyperlink_decision(true, false, true));
        assert!(!hyperlink_decision(false, true, true));
        assert!(!hyperlink_decision(false, false, false));
        assert!(hyperlink_decision(false, false, true));
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

    /// The eviction paths with limits a test can actually reach.
    ///
    /// With the production cap (1024) no test could fill `hot`, so neither
    /// demote, cold-hit nor promote ever executed: this test used to pass while
    /// proving only plain memoisation.  An injected cap of 8 over 12 paths forces
    /// a demote after the first eight, and the second full cycle must be all
    /// hits returning the *same* allocation — that is promote-or-cold-hit, never
    /// a rebuild.
    #[test]
    fn url_cache_demotes_and_promotes_under_an_injected_cap() {
        let paths: Vec<String> = (0..12)
            .map(|i| format!("/tmp/ponent-evict-{i}.pon"))
            .collect();
        with_cache_limits(8, 1 << 20, || {
            let first: Vec<_> = paths
                .iter()
                .map(|p| file_url_cached(p.as_bytes()))
                .collect();
            // 12 paths past a cap of 8: the ninth insert demotes the first eight
            // into `cold`, and the last four stay hot.
            let (hot, cold, _) = cache_state();
            assert_eq!(hot, 4, "the four post-demote inserts stay hot");
            assert_eq!(cold, 8, "the demoted generation is kept, not cleared");
            for (path, want) in paths.iter().zip(&first) {
                let got = file_url_cached(path.as_bytes());
                assert!(
                    std::sync::Arc::ptr_eq(&got, want),
                    "{path} was rebuilt instead of served from the cache"
                );
            }
        });
    }

    /// A cold hit must not promote when the byte budget is already spent.
    #[test]
    fn url_cache_promote_respects_the_byte_budget() {
        // One entry well over the budget (a long generated path), with the small
        // entries summing below it so a single demote is all that happens.
        let long = format!("/tmp/{}", "x".repeat(1000));
        let small = ["/tmp/a.pon", "/tmp/b.pon", "/tmp/c.pon"];
        // Cap 4 so the three small entries do not trigger a *second* demote,
        // which would drop the demoted `long` generation before the cold hit.
        with_cache_limits(4, 400, || {
            let long_url = file_url_cached(long.as_bytes());
            for p in small {
                file_url_cached(p.as_bytes());
            }
            // `long` has been demoted by the two-entry cap; hitting it again would
            // promote it if the promotion ignored `max_bytes`.
            let (hot_before, cold_before, bytes_before) = cache_state();
            let again = file_url_cached(long.as_bytes());
            assert!(
                std::sync::Arc::ptr_eq(&long_url, &again),
                "cold hit must hit"
            );
            let (hot_after, cold_after, bytes_after) = cache_state();
            // The entry alone exceeds the whole budget, so promoting it back would
            // break the documented per-thread ceiling (`max_bytes + one entry`):
            // the hit is served from `cold` and nothing moves.
            assert_eq!(
                (hot_after, bytes_after),
                (hot_before, bytes_before),
                "a cold hit must not promote past the byte budget"
            );
            assert_eq!(cold_after, cold_before, "the entry stays reachable in cold");
        });
    }

    /// A working set slightly above the hot cap must never rebuild.
    ///
    /// Two defects used to hide this: the fn had no `#[test]`, and it ran under
    /// the production cap (1024) over 40 paths, so no demote ever happened and
    /// it proved plain memoisation only — the exact failure its own comment
    /// describes.  Injected at 32 so the second cycle exercises promote and
    /// cold-hit.
    #[test]
    fn url_cache_promotes_and_survives_a_cycling_working_set() {
        with_cache_limits(32, 1 << 20, || {
            // A path count slightly above the hot cap used to thrash to zero
            // hits (clear-all evicted the map just before each insert).  After
            // one full cycle every path is either hot or cold, so a second cycle
            // is all hits and every returned `Arc` is the same allocation.
            let paths: Vec<String> = (0..40)
                .map(|i| format!("/tmp/ponent-cache-{i}.pon"))
                .collect();
            let mut first: Vec<std::sync::Arc<str>> = Vec::new();
            for p in &paths {
                first.push(path_to_file_url(std::path::Path::new(p)));
            }
            let (hot, cold, _) = cache_state();
            assert!(
                hot + cold > 0 && hot <= 32,
                "the cycle must populate the cache without exceeding the cap: \
                 hot={hot} cold={cold}"
            );
            for (p, prev) in paths.iter().zip(&first) {
                let again = path_to_file_url(std::path::Path::new(p));
                assert!(
                    std::sync::Arc::ptr_eq(prev, &again),
                    "{p} should be hot- or cold-hit, not rebuilt"
                );
            }
        });
    }

    /// Independent restatement of the safe set, used as an oracle below.
    fn expected_safe(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b"/:-_.~".contains(&b)
    }

    /// The 256-bit mask must agree with the set predicate on every byte value.
    #[test]
    fn url_safe_mask_matches_the_set_predicate() {
        // Two independent oracles for the same 256-value domain: `expected_safe`
        // (the documented set) and the compile-time mask.  The arithmetic predicate
        // must agree with *both* — that is what makes the no-table rewrite safe.
        for b in 0u8..=255 {
            assert_eq!(is_url_safe(b), expected_safe(b), "byte {b:#04x}");
            assert_eq!(
                is_url_safe(b),
                b < 0x80 && URL_SAFE_BITS[(b >> 6) as usize] >> (b & 63) & 1 == 1,
                "arithmetic and mask disagree on {b:#04x}"
            );
        }
        // No byte ≥ 0x80 may be safe — the `from_utf8_unchecked` calls elsewhere rely on
        // the set being ASCII.  (The mask's *word* index is `b >> 6`, so word 1 legitimately
        // covers 0x40..=0x7F; the builder loops the full 0..256 domain so a non-ASCII
        // "safe" entry would index `bits[2]` and become a const-eval error.)
        for b in 0x80u8..=0xFF {
            assert!(!is_url_safe(b), "byte {b:#04x} must not be URL-safe");
        }
    }

    /// Percent-decode a URL path for the round-trip property below.
    fn percent_decode(s: &str) -> Vec<u8> {
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' {
                let hi = (bytes[i + 1] as char).to_digit(16).expect("hex") as u8;
                let lo = (bytes[i + 2] as char).to_digit(16).expect("hex") as u8;
                out.push(hi << 4 | lo);
                i += 3;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        out
    }

    /// Full byte domain: every value is either emitted literally (iff it is in
    /// the safe set) or as an uppercase `%XX`, the output stays pure ASCII with
    /// no control bytes, and decoding recovers the input exactly.
    #[test]
    fn encoder_covers_the_full_byte_domain() {
        for b in 0u8..=255 {
            let path = [b'/', b'x', b];
            let encoded = encode_file_url_path(&path).into_owned();
            // `/` is safe in a *path* (the component encoder escapes it, the
            // path encoder does not), so the oracle is the set predicate alone.
            if expected_safe(b) {
                assert_eq!(
                    encoded.as_bytes(),
                    &path[..],
                    "safe byte {b:#04x} must be literal"
                );
            } else {
                let escaped = format!("%{b:02X}");
                assert!(
                    encoded.ends_with(&escaped),
                    "unsafe byte {b:#04x} must end with {escaped}: {encoded:?}"
                );
            }
            assert!(
                encoded.bytes().all(|c| c.is_ascii_graphic()),
                "encoded output must be printable ASCII: {encoded:?}"
            );
            assert_eq!(
                percent_decode(&encoded),
                path.to_vec(),
                "round trip failed for byte {b:#04x}"
            );
        }
        // One path containing every byte value: still pure ASCII, still
        // decodable, and `%` itself is escaped exactly once.
        let mut all = vec![b'/'];
        all.extend(0u8..=255);
        let encoded = encode_file_url_path(&all).into_owned();
        assert!(encoded.bytes().all(|c| c.is_ascii_graphic() || c == b'%'));
        assert_eq!(percent_decode(&encoded), all);
    }

    #[test]
    fn degenerate_paths_have_canonical_forms() {
        assert_eq!(format_file_url(b"/", None), "file:///");
        assert_eq!(format_file_url(b"//", None), "file://");
        // A POSIX backslash is a filename byte: encoded, never a separator.
        assert_eq!(
            format_file_url(b"/tmp/a\\b.rs", None),
            "file:///tmp/a%5Cb.rs"
        );
    }

    #[test]
    fn only_real_mnt_drives_translate() {
        // `/mnt/cdrom` is not a drive: the second byte must be `/`.
        assert_eq!(
            format_file_url(b"/mnt/cdrom/x", Some("Ubuntu")),
            "file://wsl.localhost/Ubuntu/mnt/cdrom/x"
        );
        // Uppercase drive letters are preserved as uppercase.
        assert_eq!(format_file_url(b"/mnt/C/x", Some("Ubuntu")), "file:///C:/x");
        // Bare `/mnt` and `/mnt/` are ordinary WSL paths, not drives.
        assert_eq!(
            format_file_url(b"/mnt", Some("Ubuntu")),
            "file://wsl.localhost/Ubuntu/mnt"
        );
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
        // The same bytes must not change shape because a WSL distro is known:
        // the WSL branch's `//` arm used to emit `file://C:/...` here (two
        // slashes), which parses `C:` as the URL *host*.
        assert_eq!(
            format_file_url(b"//C:/Users/user/file.rs", Some("Ubuntu")),
            "file:///C:/Users/user/file.rs"
        );
        // Under WSL the host *is* Windows, so a lowercase drive letter is still
        // a drive: both arms must agree on that too (on POSIX the uppercase
        // heuristic in `slash_authority_prefix` is what decides).
        assert_eq!(format_file_url(b"//c:/x", Some("Ubuntu")), "file:///c:/x");
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

    /// The short-path encoder (`<= FUSED_MAX`) and the long-path encoder are two
    /// implementations of one mapping, and nothing compared them: the split was
    /// the only place in this module without a differential test.  Feed the same
    /// byte patterns at lengths straddling the boundary and assert identical
    /// output *and* identical `Cow` shape (`Borrowed` iff every byte is safe).
    #[test]
    fn path_encoder_agrees_with_its_reference_across_the_short_split() {
        fn long_reference(s: &[u8]) -> Cow<'_, str> {
            let (first_unsafe, unsafe_count) = scan_url_path(s);
            if first_unsafe == s.len() {
                return Cow::Borrowed(std::str::from_utf8(s).unwrap());
            }
            let mut out = Vec::new();
            encode_scanned_path(&mut out, s, first_unsafe, unsafe_count);
            Cow::Owned(String::from_utf8(out).unwrap())
        }

        // Clean, single-unsafe, multi-unsafe, multi-byte, and mixed patterns.
        let patterns: [&[u8]; 6] = [b"a", b"/", b"a b", b"a b~c/d%E", b"caf\xc3\xa9", b"~_.$+"];
        for pat in patterns {
            for len in [1usize, 17, 4095, 4096, 4097, 8192] {
                let s: Vec<u8> = pat.iter().copied().cycle().take(len).collect();
                let short = encode_file_url_path(&s);
                let long = long_reference(&s);
                assert_eq!(
                    short,
                    long,
                    "len {len} pattern {:?}",
                    String::from_utf8_lossy(pat)
                );
                assert_eq!(
                    matches!(short, Cow::Borrowed(_)),
                    s.iter().all(|&b| is_url_safe(b)),
                    "borrowed shape differs at len {len} pattern {:?}",
                    String::from_utf8_lossy(pat)
                );
            }
        }
    }

    #[test]
    fn empty_path_is_not_garbage() {
        assert_eq!(&*path_to_file_url(std::path::Path::new("")), "file://");
    }
}
