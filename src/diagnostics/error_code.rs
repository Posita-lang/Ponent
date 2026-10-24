use compact_str::CompactString;
use std::borrow::Cow;
use std::fmt;
use std::sync::OnceLock;

/// The port number of the local explain server, if running.
static EXPLAIN_PORT: OnceLock<u16> = OnceLock::new();

/// Direct-address index over [`CODE_TABLE`], built at compile time.
///
/// The registry is validated at build time against `^[EW]\d{3}$`, so every
/// code is exactly one prefix byte plus three digits: slot
/// `prefix * 1000 + number`, with `u16::MAX` marking an absent code.  A
/// lookup is then a shape check + one array read — no lazy `OnceLock` init,
/// no SipHash, no lock, and the table is shared read-only across threads.
const CODE_INDEX_SLOTS: usize = 2 * 1000;
static CODE_INDEX: [u16; CODE_INDEX_SLOTS] = build_code_index();
/// The index stores table positions as `u16`; fail the build, not a lookup,
/// if the registry ever outgrows that.
const _: () = assert!(CODE_TABLE.len() < u16::MAX as usize);

/// Slot for a code matching the registry's `^[EW]\d{3}$` shape, or `None`.
const fn code_slot(code: &str) -> Option<usize> {
    let b = code.as_bytes();
    if b.len() != 4 {
        return None;
    }
    let prefix = match b[0] {
        b'E' => 0,
        b'W' => 1,
        _ => return None,
    };
    let mut n = 0usize;
    let mut i = 1;
    while i < 4 {
        let d = b[i];
        if !d.is_ascii_digit() {
            return None;
        }
        n = n * 10 + (d - b'0') as usize;
        i += 1;
    }
    Some(prefix * 1000 + n)
}

const fn build_code_index() -> [u16; CODE_INDEX_SLOTS] {
    let mut idx = [u16::MAX; CODE_INDEX_SLOTS];
    let mut i = 0;
    while i < CODE_TABLE.len() {
        if let Some(slot) = code_slot(CODE_TABLE[i].code) {
            idx[slot] = i as u16;
        }
        i += 1;
    }
    idx
}
/// Set the port for the local explain server.  Called by the CLI before
/// displaying error code URLs.
///
/// Returns `Err(previous)` when a port was already set, so a second caller can
/// observe that its value was ignored instead of silently assuming it took
/// effect.  Callers that intentionally re-set the port (e.g. tests) should
/// discard the result explicitly with `let _ = …`.
pub fn set_explain_port(port: u16) -> Result<(), u16> {
    EXPLAIN_PORT
        .set(port)
        .map_err(|_| *EXPLAIN_PORT.get().unwrap_or(&port))
}

/// The URL for the `--explain` feature.  Returns a `localhost` URL when
/// the local explain server is running, otherwise the canonical doc URL.
///
/// The code is percent-encoded as a single path segment in both branches, so
/// an untrusted code (CLI argument, `Diagnostic::with_code_str`) cannot escape
/// the segment or smuggle control characters into the URL.
fn explain_url(code: &str) -> String {
    explain_url_with_port(code, EXPLAIN_PORT.get().copied())
}

/// Build the explanation URL for a code, given an explicit port.
///
/// This is the pure, testable core of `explain_url`; callers that need the
/// process-global port (the production path) pass `EXPLAIN_PORT.get().copied()`.
///
/// No memoization: the only production caller is the explain server, which
/// builds one URL per served page, so a process-global mutex + hash + clone
/// would cost more than the `format!` it caches (and the port is read on
/// every call, so a later [`set_explain_port`] can never return a stale URL).
pub(crate) fn explain_url_with_port(code: &str, port: Option<u16>) -> String {
    if let Some(port) = port {
        explain_server_url(port, code)
    } else {
        // We will register this domain and set up a proper documentation
        // website in the future.  For now, the local explain server is the
        // primary way to view error explanations in a browser.
        format!(
            "https://doc.posita-lang.org/error_codes/{}.html",
            encode_url_path_segment(code)
        )
    }
}

/// Build the local explain-server URL for a raw (possibly unknown) code.
///
/// Single source of truth for the `http://127.0.0.1:<port>/<code>` form, used
/// by [`ErrCode::url`] and by the CLI when opening the browser.  The code is
/// percent-encoded as one path segment so untrusted input cannot escape the
/// segment or smuggle control characters into the URL.
pub(crate) fn explain_server_url(port: u16, code: &str) -> String {
    format!("http://127.0.0.1:{port}/{}", encode_url_path_segment(code))
}

/// Percent-encode a single URL path segment per RFC 3986.  Unreserved
/// characters pass through; everything else — including `/`, `:`, `%`,
/// control characters, and non-ASCII bytes — becomes `%XX`.  Stricter than
/// [`glyph::encode_file_url_path`](crate::diagnostics::glyph) because a code
/// is *one* segment, and a raw `\x1b` here would terminate the enclosing
/// OSC 8 hyperlink early.
fn encode_url_path_segment(s: &str) -> Cow<'_, str> {
    use std::fmt::Write as _;
    // Fast path: unreserved ASCII chars pass through byte-for-byte.
    if s.bytes()
        .all(|b| matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~'))
    {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    Cow::Owned(out)
}

/// Categorization of compiler error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Parse,
    Resolution,
    Type,
    Contract,
    Trait,
    Inference,
    Internal,
    Generic,
}

impl ErrorCategory {
    /// Stable machine-readable key (JSON `"category"` field).  Never
    /// localized — consumers depend on a fixed vocabulary.
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCategory::Parse => "Parse Error",
            ErrorCategory::Resolution => "Resolution Error",
            ErrorCategory::Type => "Type Error",
            ErrorCategory::Contract => "Contract Error",
            ErrorCategory::Trait => "Trait Error",
            ErrorCategory::Inference => "Inference Error",
            ErrorCategory::Internal => "Internal Error",
            ErrorCategory::Generic => "Error",
        }
    }

    /// i18n slug for [`Self::as_str`], resolved through Fluent so the
    /// human-readable category follows the active
    /// [`Lang`](super::i18n::Lang).
    pub fn label(&self) -> std::borrow::Cow<'static, str> {
        match self {
            ErrorCategory::Parse => crate::tr!("category-parse"),
            ErrorCategory::Resolution => crate::tr!("category-resolution"),
            ErrorCategory::Type => crate::tr!("category-type"),
            ErrorCategory::Contract => crate::tr!("category-contract"),
            ErrorCategory::Trait => crate::tr!("category-trait"),
            ErrorCategory::Inference => crate::tr!("category-inference"),
            ErrorCategory::Internal => crate::tr!("category-internal"),
            ErrorCategory::Generic => crate::tr!("category-generic"),
        }
    }
}

/// A compiler error or warning code (e.g. "E030", "W113").
///
/// Stored as a [`CompactString`]: codes are short (`E030`), so a known or
/// unknown code stays inline with no heap allocation.  Metadata (title,
/// category, explanation) is resolved through the same lookup table, removing
/// the need for exhaustive match arms on enum variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrCode(CompactString);

/// Error returned when an unknown error code string is used.
#[derive(Debug, Clone)]
pub struct UnknownCode<'a>(pub Cow<'a, str>);

/// Fallback metadata for error codes not present in [`CODE_TABLE`].
pub(crate) static UNKNOWN_CODE_ENTRY: CodeEntry = CodeEntry {
    code: "",
    title: "unknown error",
    category: ErrorCategory::Generic,
    explain: "No detailed explanation is available for this error code yet.",
};

impl fmt::Display for UnknownCode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `self.0` is the untrusted code string that failed lookup.  It is
        // surfaced verbatim in the message (e.g. via the explain server's
        // `io::Error` → `eprintln!` path), so neutralize control characters
        // that could inject terminal escapes.
        write!(
            f,
            "unknown error code: `{}`",
            super::sanitize_for_terminal(&self.0)
        )
    }
}

impl std::error::Error for UnknownCode<'_> {}

impl ErrCode {
    /// Create a new `ErrCode`, accepting an unknown code as-is (it will
    /// fall back to "unknown error" for `title()` / `category()` /
    /// `explain()`).
    ///
    /// This never panics in **any** profile.  The `debug_assert!` this
    /// constructor used to carry made a typo'd literal panic debug builds
    /// while silently degrading release ones — a divergence that hid bugs
    /// from CI.  `literal_codes_are_all_in_code_table` in the test module
    /// is the guard that catches such typos instead.
    ///
    /// A short code (e.g. `E030`) stays inline with no heap allocation.
    ///
    /// Prefer [`Self::try_new`] when the input comes from an external source
    /// (user input, CLI arguments, etc.).
    pub fn new(code: impl AsRef<str>) -> Self {
        let code = code.as_ref();
        // Fast path: an already-canonical code (the overwhelmingly common
        // case) hits the table with no allocation.
        if let Some(e) = lookup(code) {
            return ErrCode(CompactString::from(e.code));
        }
        // Retry case-insensitively (`e030` → `E030`), matching
        // `suggest_code`'s case-insensitive comparison — a lowercase code
        // must resolve to its entry, not degrade to unknown-code metadata.
        // The retry is allocation-free for real codes (see
        // [`lookup_uppercased`]).
        if let Some(e) = lookup_uppercased(code) {
            return ErrCode(CompactString::from(e.code));
        }
        // Unknown: stored verbatim — never invent case for arbitrary
        // strings that are not error codes.
        ErrCode(CompactString::from(code))
    }

    /// Build an `ErrCode` from a code expected to be in the table, keeping a
    /// `debug_assert!` so a typo still panics debug builds; release behaves
    /// exactly like [`Self::new`].
    ///
    /// Production code must not depend on the panic: it only marks "this is
    /// a programming error" for developers.
    pub(crate) fn new_unchecked(code: impl AsRef<str>) -> Self {
        let code = code.as_ref();
        let entry = lookup(code);
        debug_assert!(
            entry.is_some(),
            "unknown error code: {code:?} — must be added to error_codes/codes.toml",
        );
        match entry {
            Some(e) => ErrCode(CompactString::from(e.code)),
            None => ErrCode(CompactString::from(code)),
        }
    }

    /// Validate that `code` exists in the lookup table, returning
    /// [`UnknownCode`] if it does not.  This check runs in **all** build
    /// profiles (unlike the `debug_assert!` in [`Self::new_unchecked`]).
    ///
    /// On success the canonical entry is copied inline; only the
    /// unknown-code error borrows the rejected input.
    pub fn try_new<'a>(code: &'a str) -> Result<Self, UnknownCode<'a>> {
        // Same case-insensitive resolution as [`Self::new`]: a known code
        // in any case is accepted and stored canonically; the error keeps
        // the input verbatim so the user sees what they typed.
        match lookup(code).or_else(|| lookup_uppercased(code)) {
            Some(e) => Ok(ErrCode(CompactString::from(e.code))),
            None => Err(UnknownCode(Cow::Borrowed(code))),
        }
    }

    /// Build an `ErrCode` from a raw string **without** validating it against
    /// [`CODE_TABLE`].  Unlike [`Self::new`], this never panics; `title()`,
    /// `category()`, and `explain()` fall back to their generic "unknown"
    /// values.
    ///
    /// This is the escape hatch for callers that may receive untrusted or
    /// dynamically-constructed code strings (e.g. `Diagnostic::with_code_str`).
    pub(crate) fn unknown(code: impl Into<CompactString>) -> Self {
        ErrCode(code.into())
    }

    /// The code string, e.g. "E030" or "W113".
    pub fn code(&self) -> &str {
        self.0.as_str()
    }

    /// Resolve this error code to its table entry, if present.
    ///
    /// Callers that need multiple fields (`title`, `category`, `explain`)
    /// should use this once and derive each from the returned entry, avoiding
    /// repeated hash lookups.
    pub fn entry(&self) -> Option<&'static CodeEntry> {
        lookup(&self.0)
    }

    /// Return the resolved entry or the static unknown-code fallback.
    ///
    /// Equivalent to a single `lookup` call followed by `.unwrap_or(...)`,
    /// so callers can gather `title` / `category` / `explain` in one lookup.
    pub(crate) fn entry_or_unknown(&self) -> &'static CodeEntry {
        lookup(&self.0).unwrap_or(&UNKNOWN_CODE_ENTRY)
    }

    /// Short title, e.g. "type mismatch" or "duplicate definition".
    pub fn title(&self) -> &'static str {
        self.entry_or_unknown().title
    }

    /// The error category, e.g. `ErrorCategory::Type`.
    pub fn category(&self) -> ErrorCategory {
        self.entry_or_unknown().category
    }

    /// Full explanation text, displayed by `ponent --explain E030`.
    pub fn explain(&self) -> &'static str {
        self.entry_or_unknown().explain
    }

    /// The diagnostic URL for the `--explain` feature.
    ///
    /// [`explain_url`] percent-encodes the code as a single path segment, so an
    /// unknown code holding arbitrary bytes (see [`Self::unknown`]) cannot
    /// escape the segment or the enclosing link.
    pub fn url(&self) -> String {
        explain_url(&self.0)
    }

    /// The diagnostic URL, formatted as an ANSI hyperlink if the terminal supports it.
    ///
    /// The visible text is sanitized (control characters become U+FFFD) so an
    /// untrusted code cannot terminate the OSC 8 link early and inject further
    /// terminal escapes.
    pub fn url_ansi(&self) -> String {
        format!(
            "\x1b]8;;{}\x1b\\{}\x1b]8;;\x1b\\",
            self.url(),
            super::sanitize_for_terminal(&self.0),
        )
    }
}

/// A single entry in the error code lookup table.
pub(crate) struct CodeEntry {
    pub(crate) code: &'static str,
    pub(crate) title: &'static str,
    pub(crate) category: ErrorCategory,
    explain: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/error_codes_gen.rs"));

/// Look up a code string in the table.
pub(crate) fn lookup(code: &str) -> Option<&'static CodeEntry> {
    let slot = code_slot(code)?;
    let idx = CODE_INDEX[slot];
    if idx == u16::MAX {
        return None;
    }
    CODE_TABLE.get(idx as usize)
}

/// [`lookup`] on the ASCII-uppercased form of `code`, without allocating for
/// the common short-code case (`e030` → `E030`).
///
/// Table codes are at most a few bytes, so a stack buffer covers every real
/// retry; only longer inputs fall back to a temporary `String`.
fn lookup_uppercased(code: &str) -> Option<&'static CodeEntry> {
    const STACK: usize = 8;
    let bytes = code.as_bytes();
    if bytes.len() > STACK {
        return lookup(&code.to_ascii_uppercase());
    }
    let mut buf = [0u8; STACK];
    for (dst, src) in buf.iter_mut().zip(bytes) {
        *dst = src.to_ascii_uppercase();
    }
    // ASCII case folding preserves UTF-8 structure.
    let upper = std::str::from_utf8(&buf[..bytes.len()]).ok()?;
    lookup(upper)
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    /// A second `set_explain_port` must be observable rather than silently
    /// discarded.  `EXPLAIN_PORT` is a process-global `OnceLock`, so this test
    /// deliberately owns it (no other test sets it).
    #[test]
    fn set_explain_port_reports_second_set() {
        assert!(set_explain_port(12345).is_ok());
        assert_eq!(set_explain_port(54321), Err(12345));
    }

    /// `unknown` is the untrusted-input path; an ESC in the code must reach
    /// neither the URI nor the visible link text.
    ///
    /// The only ESC bytes in the result are the four that frame the OSC 8
    /// sequence (`\x1b]8;;` … `\x1b\\` … `\x1b]8;;` … `\x1b\\`).  This holds
    /// regardless of whether `EXPLAIN_PORT` was set by another test.
    #[test]
    fn url_ansi_cannot_inject_osc() {
        let ec = ErrCode::unknown("E0\x1b]8;;evil\x07");
        let out = ec.url_ansi();
        assert_eq!(out.matches('\x1b').count(), 4, "extra ESC in {out:?}");
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert!(out.contains("%1B"), "ESC not encoded in URL: {out:?}");
        assert!(out.contains("%07"), "BEL not encoded in URL: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced in text: {out:?}"
        );
    }

    /// The code is one path segment: a newline or `/` must be percent-encoded
    /// so it cannot escape the segment.  Host/suffix vary with `EXPLAIN_PORT`,
    /// so assert on the encoded segment rather than the whole URL.
    #[test]
    fn url_percent_encodes_unknown_codes() {
        let ec = ErrCode::unknown("E030\n../admin");
        let u = ec.url();
        assert!(u.contains("E030%0A..%2Fadmin"), "got: {u}");
        assert!(!u.contains('\n'), "newline leaked: {u:?}");
        assert!(!u.contains("../"), "path traversal not encoded: {u}");
    }

    /// Regression guard: encoding must not alter canonical codes.
    #[test]
    fn known_codes_are_unchanged() {
        let ec = ErrCode::new("E030");
        assert!(ec.url().contains("/E030"), "got: {}", ec.url());
        assert!(!ec.url().contains('%'), "got: {}", ec.url());
        assert!(ec.url_ansi().contains("E030"));
    }

    /// Pin the single-segment RFC 3986 contract now that the CLI builds its
    /// browser URL through this encoder: unreserved bytes pass through,
    /// reserved (`/`, `:`, `%`), whitespace, control, and non-ASCII bytes do
    /// not.
    #[test]
    fn encode_url_path_segment_contract() {
        assert_eq!(encode_url_path_segment("E030"), "E030");
        assert_eq!(encode_url_path_segment("a/b:c%d e"), "a%2Fb%3Ac%25d%20e");
        assert_eq!(encode_url_path_segment("\x1b\x07"), "%1B%07");
        assert_eq!(encode_url_path_segment("é"), "%C3%A9");
    }

    /// The shared localhost-URL builder (used by the CLI's browser open and by
    /// `ErrCode::url`) encodes the code as a single segment.
    #[test]
    fn explain_server_url_encodes_code() {
        assert_eq!(
            explain_server_url(1234, "E030"),
            "http://127.0.0.1:1234/E030"
        );
        assert_eq!(
            explain_server_url(1234, "E030\n../admin"),
            "http://127.0.0.1:1234/E030%0A..%2Fadmin"
        );
    }

    /// The untrusted code is echoed in `UnknownCode`'s `Display` (surfaced by
    /// the explain server as an `io::Error` and printed to the terminal), so
    /// control characters must not survive.
    #[test]
    fn unknown_code_display_is_sanitized() {
        let err = ErrCode::try_new("E0\x1b]8;;evil\x07").unwrap_err();
        let msg = err.to_string();
        assert!(!msg.contains('\x1b'), "ESC leaked: {msg:?}");
        assert!(!msg.contains('\x07'), "BEL leaked: {msg:?}");
        assert!(
            msg.contains('\u{FFFD}'),
            "control char not replaced: {msg:?}"
        );
    }

    /// Collect every `.rs` file under `dir` (recursively).
    fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                rs_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    /// Return the argument text of the parenthesised call whose `(` is at
    /// `open_idx`, skipping nested parens, string literals, and comments.
    fn argument_span(source: &str, open_idx: usize) -> Option<&str> {
        let bytes = source.as_bytes();
        let mut depth = 0i32;
        let mut i = open_idx;
        while i < bytes.len() {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&source[open_idx + 1..i]);
                    }
                }
                b'"' => {
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        if bytes[i] == b'\\' {
                            i += 1;
                        }
                        i += 1;
                    }
                }
                b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                }
                b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                    i += 2;
                    while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                        i += 1;
                    }
                    i += 1;
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// All `"..."` literals inside `arg` that look like a code (`E123`/`W123`),
    /// ignoring those inside line/block comments.
    ///
    /// Escaped quotes (`\"`) inside a literal are skipped so the scanner
    /// cannot end the literal early and resume inside the string body.
    fn code_literals(arg: &str) -> Vec<&str> {
        let bytes = arg.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'"' => {
                    let start = i + 1;
                    let mut j = start;
                    while j < bytes.len() && bytes[j] != b'"' {
                        if bytes[j] == b'\\' && j + 1 < bytes.len() {
                            j += 1;
                        }
                        j += 1;
                    }
                    let lit = &arg[start..j];
                    let code_shaped = lit.len() == 4
                        && matches!(lit.as_bytes()[0], b'E' | b'W')
                        && lit.as_bytes()[1..].iter().all(u8::is_ascii_digit);
                    if code_shaped {
                        out.push(lit);
                    }
                    i = j + 1;
                }
                b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                }
                b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                    i += 2;
                    while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                        i += 1;
                    }
                    i += 1;
                }
                _ => i += 1,
            }
        }
        out
    }

    /// Blank out comments and string/char literals (including raw strings),
    /// preserving byte offsets, so the outer scan never matches a callee name
    /// that only appears in a comment or a literal.  The result has the same
    /// byte length as `src`; offsets index both.
    fn strip_non_code(src: &str) -> String {
        let bytes = src.as_bytes();
        let mut out = vec![b' '; bytes.len()];
        let mut i = 0;
        while i < bytes.len() {
            // Raw string: r"...", r#"..."#; byte raw string: br#"..."#.
            let raw_start = if bytes[i] == b'r' {
                Some(i)
            } else if bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'r' {
                Some(i + 1)
            } else {
                None
            };
            if let Some(r) = raw_start {
                let mut h = r + 1;
                while h < bytes.len() && bytes[h] == b'#' {
                    h += 1;
                }
                if h < bytes.len() && bytes[h] == b'"' {
                    let hashes = h - (r + 1);
                    let mut j = h + 1;
                    loop {
                        if j >= bytes.len() {
                            j = bytes.len();
                            break;
                        }
                        if bytes[j] == b'"' {
                            let mut k = j + 1;
                            let mut cnt = 0;
                            while cnt < hashes && k < bytes.len() && bytes[k] == b'#' {
                                k += 1;
                                cnt += 1;
                            }
                            if cnt == hashes {
                                j = k;
                                break;
                            }
                        }
                        j += 1;
                    }
                    i = j;
                    continue;
                }
            }

            // String literal: "..." or byte string b"...".
            let str_quote = if bytes[i] == b'"' {
                Some(i)
            } else if bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                Some(i + 1)
            } else {
                None
            };
            if let Some(q) = str_quote {
                let mut j = q + 1;
                while j < bytes.len() && bytes[j] != b'"' {
                    if bytes[j] == b'\\' && j + 1 < bytes.len() {
                        j += 1;
                    }
                    j += 1;
                }
                i = (j + 1).min(bytes.len());
                continue;
            }

            // Char literal: 'x', '\n', '\u{...}'; byte char: b'x'.
            // Lifetimes ('src) are not char literals and are copied as-is.
            let ch_quote = if bytes[i] == b'\'' {
                Some(i)
            } else if bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                Some(i + 1)
            } else {
                None
            };
            if let Some(q) = ch_quote {
                let mut j = q + 1;
                if j < bytes.len() && bytes[j] == b'\\' {
                    j += 2;
                    if j - 1 < bytes.len()
                        && bytes[j - 1] == b'u'
                        && j < bytes.len()
                        && bytes[j] == b'{'
                    {
                        while j < bytes.len() && bytes[j] != b'}' {
                            j += 1;
                        }
                        j += 1;
                    }
                } else if j < bytes.len() {
                    let c = src[j..].chars().next().unwrap();
                    j += c.len_utf8();
                }
                if j < bytes.len() && bytes[j] == b'\'' {
                    i = j + 1;
                    continue;
                }
                // Lifetime or malformed — copy the quote and move on.
                out[i] = bytes[i];
                i += 1;
                continue;
            }

            // Comments.
            if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
                continue;
            }

            // Ordinary byte.
            out[i] = bytes[i];
            i += 1;
        }
        String::from_utf8(out).expect("stripping preserved UTF-8")
    }

    /// Every literal code passed to `ErrCode::new` / `with_code_str` must be
    /// present in [`CODE_TABLE`].  A missing entry would `debug_assert!`-panic
    /// (`ErrCode::new`) or silently degrade to "unknown error"
    /// (`with_code_str`).  Scans the crate source, including the conditional
    /// forms like `with_code_str(if … { "E109" } else { "E110" })`.
    #[test]
    fn literal_codes_are_all_in_code_table() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rs_files(&root, &mut files);

        let mut offenders = Vec::new();
        for file in &files {
            let Ok(src) = fs::read_to_string(file) else {
                continue;
            };
            // Scan the code-only view so callee names mentioned in comments or
            // literals (e.g. doc examples) are not treated as real call sites;
            // argument text is still extracted from the original `src`.
            let stripped = strip_non_code(&src);
            for callee in ["with_code_str", "ErrCode::new"] {
                let mut from = 0;
                while let Some(pos) = stripped[from..].find(callee) {
                    let after = from + pos + callee.len();
                    let tail = &src[after..];
                    let ws = tail.len() - tail.trim_start().len();
                    if tail.trim_start().starts_with('(') {
                        let open = after + ws;
                        if let Some(arg) = argument_span(&src, open) {
                            for lit in code_literals(arg) {
                                if lookup(lit).is_none() {
                                    offenders.push(format!("{}: {lit}", file.display()));
                                }
                            }
                        }
                    }
                    from = after;
                }
            }
        }

        offenders.sort();
        offenders.dedup();
        assert!(
            offenders.is_empty(),
            "codes passed to `ErrCode::new`/`with_code_str` but missing from CODE_TABLE:\n{}",
            offenders.join("\n"),
        );
    }

    #[test]
    fn scanner_extracts_nested_and_conditional_literals() {
        let src = r#"
            Diagnostic::error(m).with_code_str("E030");
            let d = Diagnostic::error(m)
                .with_code_str(if x { "E109" } else if y { "E112" } else { "E110" });
            let e = ErrCode::new("E067");
            let f = ErrCode::new(some_var);
            let g = ErrCode::new(/* "E998" */ "E001"); // "E999"
        "#;
        let mut found = Vec::new();
        for callee in ["with_code_str", "ErrCode::new"] {
            let mut from = 0;
            while let Some(pos) = src[from..].find(callee) {
                let after = from + pos + callee.len();
                let tail = &src[after..];
                let ws = tail.len() - tail.trim_start().len();
                if tail.trim_start().starts_with('(') {
                    let open = after + ws;
                    if let Some(arg) = argument_span(src, open) {
                        found.extend(code_literals(arg));
                    }
                }
                from = after;
            }
        }
        found.sort_unstable();
        assert_eq!(
            found,
            vec!["E001", "E030", "E067", "E109", "E110", "E112"],
            "scanner missed or over-matched literals"
        );
    }

    #[test]
    fn code_literals_skip_escaped_quotes() {
        // The `\"` must not terminate the first literal: without escape
        // handling the scanner resumes inside the string body and
        // mis-delimits the following literal as empty.
        let arg = r#""E0\"30" + "E030""#;
        assert_eq!(code_literals(arg), vec!["E030"]);
    }

    #[test]
    fn strip_non_code_blanks_comments_and_literals() {
        let src = r##"
            // with_code_str("E901")
            /* ErrCode::new("E902") */
            let s = "with_code_str(\"E903\")";
            let r = r#"ErrCode::new("E904")"#;
            let c = '"';
            let lt: &'static str = "ok";
            let real = Diagnostic::error(m).with_code_str("E030");
        "##;
        let stripped = strip_non_code(src);
        assert_eq!(stripped.len(), src.len(), "offsets must be preserved");
        // Only the real call site survives; comments, normal strings, raw
        // strings, and char literals are blanked.
        assert_eq!(stripped.matches("with_code_str").count(), 1);
        assert_eq!(stripped.matches("ErrCode::new").count(), 0);
    }

    /// `ErrCode::new` never panics — not even in debug builds — for codes
    /// missing from the table; the unknown code is kept verbatim so
    /// `title()`/`category()` can fall back to "unknown error".
    #[test]
    fn new_accepts_unknown_code_without_panicking() {
        // Built at runtime so the code-literal scanner in
        // `literal_codes_are_all_in_code_table` does not flag it.
        let unknown = format!("E{}", 999);
        let code = ErrCode::new(unknown.clone());
        assert_eq!(code.code(), "E999");
        assert_eq!(code.category(), ErrorCategory::Generic);
        assert_eq!(code.title(), "unknown error");
        // The validating constructor still rejects the same input.
        assert!(ErrCode::try_new(&unknown).is_err());
    }

    /// `new_unchecked` round-trips a table code.  Also the guard keeping
    /// `new_unchecked` alive: production has no call sites yet, but the
    /// escape hatch must stay ready and behaviour-identical for dev builds.
    #[test]
    fn new_unchecked_round_trips_table_code() {
        assert_eq!(ErrCode::new_unchecked("E030").code(), "E030");
    }

    /// The compile-time direct-address index must agree with a linear scan
    /// of the table for every entry, and must reject codes outside the
    /// registry's `^[EW]\d{3}$` shape (and well-shaped codes that are absent)
    /// without ever landing on a wrong slot.
    #[test]
    fn direct_address_index_matches_table() {
        for e in CODE_TABLE {
            assert_eq!(
                lookup(e.code).map(|x| x.code),
                Some(e.code),
                "index missing {}",
                e.code
            );
        }
        for code in ["E000", "E999", "W000", "W999"] {
            assert!(lookup(code).is_none(), "{code} unexpectedly present");
        }
        for code in ["", "E", "E03", "E0300", "e030", "X030", "E0a0", "E030 "] {
            assert!(lookup(code).is_none(), "{code:?} unexpectedly present");
        }
    }
}
