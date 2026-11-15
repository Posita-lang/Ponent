use compact_str::{CompactString, format_compact};
use litemap::LiteMap;
use std::borrow::Cow;
use std::fmt;
use std::sync::OnceLock;

use crate::smallvec::SmallVec;

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
#[inline]
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
            // A duplicate code would otherwise silently shadow the first
            // entry and make it unreachable through `lookup`, while
            // `direct_address_index_matches_table` still passed (last wins).
            // Fail the build instead of corrupting the registry.
            assert!(idx[slot] == u16::MAX, "duplicate code in CODE_TABLE");
            idx[slot] = i as u16;
        }
        i += 1;
    }
    idx
}

/// Per-slot memoization map shared by [`ErrCode::url`] / [`ErrCode::url_ansi`]
/// (one `SlotCache` static per (port-set, doc-URL) × (plain, ANSI) quadrant).
///
/// Shape matches `category_siblings_of`'s `SIBLING_CACHE`: a lazily-built
/// `RwLock<FxHashMap<slot, Arc<str>>>` instead of an eagerly-allocated
/// `CODE_INDEX_SLOTS`-sized array of `OnceLock` shells — a process that only
/// renders a handful of distinct codes pays for those codes, not ~256 KB of
/// shells across the four caches.  The hit path is one map lookup + a
/// refcount bump (`Arc<str>`, not a deep string clone: the OSC 8 form is
/// 40–80+ heap bytes, and `write_header` calls `url_ansi()` per diagnostic,
/// so a 1 000-diagnostic batch sharing one code would otherwise pay 1 000
/// identical allocations); `build` never runs while a lock is held on
/// `cache` (it may re-enter another `SlotCache`), so there is no
/// lock-ordering hazard.
type SlotCache = OnceLock<std::sync::RwLock<rustc_hash::FxHashMap<u16, std::sync::Arc<str>>>>;

fn cached_slot_string(
    cache: &SlotCache,
    slot: u16,
    build: impl FnOnce() -> CompactString,
) -> std::sync::Arc<str> {
    // One `OnceLock` load, then the map read for hits.  The double-checked
    // insert guarantees exactly one stored value per slot; a losing racer has
    // already paid `build()` before discovering the winner.  That duplicate
    // work is intentional: building under the map write lock would serialize
    // every miss, and `build` is a cheap pure `format_compact!`.
    let m = cache.get_or_init(|| std::sync::RwLock::new(rustc_hash::FxHashMap::default()));
    // Poison recovery: the guarded value is only `Arc<str>` built by a pure
    // function, so a poisoned map is still valid.  Silently bypassing the
    // cache on poison would turn every later call into a rebuild.
    {
        let r = m.read().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = r.get(&slot) {
            return std::sync::Arc::clone(v);
        }
    }
    let built: std::sync::Arc<str> = build().into();
    {
        let mut w = m.write().unwrap_or_else(|e| e.into_inner());
        // Double-checked insert: a racing caller may have built and inserted
        // the same `(slot)` value first.  Its value is identical (both are
        // pure functions of the slot), so returning the winner's copy skips
        // a redundant build and keeps exactly one entry per slot.
        if let Some(v) = w.get(&slot) {
            return std::sync::Arc::clone(v);
        }
        w.insert(slot, std::sync::Arc::clone(&built));
    }
    built
}

/// Set the port for the local explain server.  Called by the CLI before
/// displaying error code URLs.
///
/// Returns `Err(previous)` when a port was already set, so a second caller can
/// observe that its value was ignored instead of silently assuming it took
/// effect.  Callers that intentionally re-set the port (e.g. tests) should
/// discard the result explicitly with `let _ = …`.
///
/// Port `0` is accepted as-is: a caller that intends to listen validates its
/// own bind (the CLI binds before calling this), and rejecting `0` here would
/// need a second error channel without preventing a bad URL on its own.
pub fn set_explain_port(port: u16) -> Result<(), u16> {
    EXPLAIN_PORT
        .set(port)
        .map_err(|_| *EXPLAIN_PORT.get().unwrap_or(&port))
}

/// Whether the explain server is running (its port has been set).
pub fn is_explain_server_running() -> bool {
    EXPLAIN_PORT.get().is_some()
}

/// The URL for the `--explain` feature.  Returns a `localhost` URL when
/// the local explain server is running, otherwise the canonical doc URL.
///
/// The code is percent-encoded as a single path segment in both branches, so
/// an untrusted code (CLI argument, `Diagnostic::with_code_str`) cannot escape
/// the segment or smuggle control characters into the URL.
///
/// Returns a [`CompactString`] — the URL is short enough to inline in the
/// common case, and `url()`/`url_ansi()` are called per diagnostic that
/// carries a code.
fn explain_url(code: &str) -> CompactString {
    explain_url_with_port(code, EXPLAIN_PORT.get().copied())
}

/// Build the explanation URL for a code, given an explicit port.
///
/// This is the pure, testable core of `explain_url`; callers that need the
/// process-global port (the production path) pass `EXPLAIN_PORT.get().copied()`.
///
/// This function itself does not memoize — the per-call cost is one
/// `format_compact!`.  The hot-path caller [`ErrCode::url_ansi`] (invoked per
/// diagnostic in `write_header` when hyperlinks are active) caches its full
/// OSC 8 result in a [`SlotCache`] (a lazily-built `FxHashMap` keyed by
/// [`code_slot`]), gated on the explain port being set, so a batch of *N*
/// diagnostics sharing a code builds the link once and clones the `Arc<str>`
/// thereafter.  The cold-path caller (the explain server, one URL per served
/// page) bypasses that cache.
pub(crate) fn explain_url_with_port(code: &str, port: Option<u16>) -> CompactString {
    if let Some(port) = port {
        explain_server_url(port, code)
    } else {
        // We will register this domain and set up a proper documentation
        // website in the future.  For now, the local explain server is the
        // primary way to view error explanations in a browser.
        const DOC_URL_PREFIX: &str = "https://doc.posita-lang.org/error_codes/";
        const DOC_URL_SUFFIX: &str = ".html";
        let encoded = encode_url_path_segment(code);
        let mut out = CompactString::with_capacity(
            DOC_URL_PREFIX.len() + encoded.len() + DOC_URL_SUFFIX.len(),
        );
        out.push_str(DOC_URL_PREFIX);
        out.push_str(encoded.as_ref());
        out.push_str(DOC_URL_SUFFIX);
        out
    }
}

/// Build the local explain-server URL for a raw (possibly unknown) code.
///
/// Single source of truth for the `http://127.0.0.1:<port>/<code>` form, used
/// by [`ErrCode::url`] and by the CLI when opening the browser.  The code is
/// percent-encoded as one path segment so untrusted input cannot escape the
/// segment or smuggle control characters into the URL.
pub(crate) fn explain_server_url(port: u16, code: &str) -> CompactString {
    format_compact!("http://127.0.0.1:{port}/{}", encode_url_path_segment(code))
}

/// Percent-encode a single URL path segment per RFC 3986.  Unreserved
/// characters pass through; everything else — including `/`, `:`, `%`,
/// control characters, and non-ASCII bytes — becomes `%XX`.  Stricter than
/// [`term::encode_file_url_path`](crate::diagnostics::term) because a code
/// is *one* segment, and a raw `\x1b` here would terminate the enclosing
/// OSC 8 hyperlink early.
fn encode_url_path_segment(s: &str) -> Cow<'_, str> {
    use crate::diagnostics::renderers::fmt::HEX_UPPER;
    /// RFC 3986 unreserved set: `ALPHA / DIGIT / "-" / "." / "_" / "~"`.
    const fn is_unreserved(b: u8) -> bool {
        matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~')
    }
    // Fast path: unreserved ASCII chars pass through byte-for-byte.
    if s.bytes().all(is_unreserved) {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len().saturating_mul(3));
    for &b in s.as_bytes() {
        if is_unreserved(b) {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX_UPPER[(b >> 4) as usize] as char);
            out.push(HEX_UPPER[(b & 0xF) as usize] as char);
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
    ///
    /// This is the same vocabulary as the CLI `--only-category` values
    /// (`parse`, `resolution`, …), not the human display string from
    /// [`as_str`](Self::as_str).  JSON consumers that used to see
    /// `"Parse Error"` must read `"parse"`; the display renderers keep
    /// using `as_str`.
    pub fn key(&self) -> &'static str {
        match self {
            ErrorCategory::Parse => "parse",
            ErrorCategory::Resolution => "resolution",
            ErrorCategory::Type => "type",
            ErrorCategory::Contract => "contract",
            ErrorCategory::Trait => "trait",
            ErrorCategory::Inference => "inference",
            ErrorCategory::Internal => "internal",
            ErrorCategory::Generic => "generic",
        }
    }

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
        // Explicit variant → index mapping: `*self as usize` would silently
        // swap labels if the enum were ever reordered; a match arms the
        // reorder into a non-exhaustive compile error instead.
        let idx = match self {
            ErrorCategory::Parse => 0,
            ErrorCategory::Resolution => 1,
            ErrorCategory::Type => 2,
            ErrorCategory::Contract => 3,
            ErrorCategory::Trait => 4,
            ErrorCategory::Inference => 5,
            ErrorCategory::Internal => 6,
            ErrorCategory::Generic => 7,
        };
        crate::diagnostics::i18n::cached_label! {
            n = 8,
            variant = idx,
            init = [
                crate::diagnostics::i18n::fluent::category_parse().into_owned(),
                crate::diagnostics::i18n::fluent::category_resolution().into_owned(),
                crate::diagnostics::i18n::fluent::category_type().into_owned(),
                crate::diagnostics::i18n::fluent::category_contract().into_owned(),
                crate::diagnostics::i18n::fluent::category_trait().into_owned(),
                crate::diagnostics::i18n::fluent::category_inference().into_owned(),
                crate::diagnostics::i18n::fluent::category_internal().into_owned(),
                crate::diagnostics::i18n::fluent::category_generic().into_owned(),
            ],
        }
    }
}

impl std::str::FromStr for ErrorCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "parse" | "parse error" => Ok(ErrorCategory::Parse),
            "resolution" | "resolution error" => Ok(ErrorCategory::Resolution),
            "type" | "type error" => Ok(ErrorCategory::Type),
            "contract" | "contract error" => Ok(ErrorCategory::Contract),
            "trait" | "trait error" => Ok(ErrorCategory::Trait),
            "inference" | "inference error" => Ok(ErrorCategory::Inference),
            "internal" | "internal error" => Ok(ErrorCategory::Internal),
            "generic" | "error" => Ok(ErrorCategory::Generic),
            _ => Err(format!(
                "unknown error category `{s}` (valid: parse, resolution, type, \
                 contract, trait, inference, internal, generic)"
            )),
        }
    }
}

/// A compiler error or warning code (e.g. "E030", "W113").
///
/// A code resolved against [`CODE_TABLE`] is stored as a pointer to its
/// static entry: construction copies no string, `Clone` is a pointer copy,
/// and `title()` / `category()` / `explain()` need no further lookup.
/// A code that is not in the table (CLI input, `Diagnostic::with_code_str`)
/// stays inline in a [`CompactString`], so short unknown codes still never
/// allocate.  Equality and hashing compare the code string, so the two
/// representations are interchangeable.
#[derive(Clone)]
pub struct ErrCode(ErrCodeRepr);

/// The two storage forms behind [`ErrCode`].
#[derive(Clone)]
enum ErrCodeRepr {
    /// Resolved table entry — a `'static` pointer, no copy.
    Known(&'static CodeEntry),
    /// Unresolved code, stored verbatim (inline for short codes).
    ///
    /// Boxed so `ErrCode` itself stays two words: every diagnostic carries
    /// one, and the unresolved case is rare, so the indirection costs an
    /// allocation only where the old inline `CompactString` bloated the
    /// enum for *all* codes.  `canonical` caches the case-insensitive
    /// registry resolution at construction so `PartialEq`/`Hash` never
    /// re-run `resolve` — a `HashSet<ErrCode>` in a large batch used to pay
    /// a slot shape check + `CODE_INDEX` read per comparison/hash.  `None`
    /// means the code is genuinely unregistered (both `new` and `unknown`
    /// resolve before storing).
    Other(Box<OtherCode>),
}

/// Boxed payload of [`ErrCodeRepr::Other`].
#[derive(Clone)]
struct OtherCode {
    raw: CompactString,
    canonical: Option<&'static CodeEntry>,
}

impl fmt::Debug for ErrCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ErrCode").field(&self.code()).finish()
    }
}

/// Renders the code string (e.g. `E030`), so `format!("code {code}")` sites
/// stay uniform and the explain server's error paths can use it directly.
impl fmt::Display for ErrCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// Compare the *canonical* code strings, not the storage form.
///
/// `code()` echoes the stored spelling verbatim for untrusted input, but
/// identity must follow the same case-insensitive resolution as
/// `new`/`try_new`/`entry`: `ErrCode::new("e030")` and
/// `ErrCode::unknown("e030")` are the same logical code and must be equal.
/// Hashing uses the identical canonical view, so the two stay consistent as
/// `HashMap`/`HashSet` keys.
impl PartialEq for ErrCode {
    fn eq(&self, other: &Self) -> bool {
        self.canonical() == other.canonical()
    }
}

impl Eq for ErrCode {}

impl std::hash::Hash for ErrCode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(self.canonical(), state);
    }
}

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
        // case) hits the table with no allocation and stores the entry
        // pointer directly.
        if let Some(e) = lookup(code) {
            return ErrCode(ErrCodeRepr::Known(e));
        }
        // Retry case-insensitively (`e030` → `E030`), matching
        // `suggest_code`'s case-insensitive comparison — a lowercase code
        // must resolve to its entry, not degrade to unknown-code metadata.
        // The retry is allocation-free for real codes (see
        // [`lookup_uppercased`]).
        if let Some(e) = lookup_uppercased(code) {
            return ErrCode(ErrCodeRepr::Known(e));
        }
        // Unknown: stored verbatim — never invent case for arbitrary
        // strings that are not error codes.  Both lookups above already
        // failed, so the cached canonical resolution is `None`.
        ErrCode(ErrCodeRepr::Other(Box::new(OtherCode {
            raw: CompactString::from(code),
            canonical: None,
        })))
    }

    /// Validate that `code` exists in the lookup table, returning
    /// [`UnknownCode`] if it does not.  This check runs in **all** build
    /// profiles.
    ///
    /// On success the canonical entry is copied inline; only the
    /// unknown-code error borrows the rejected input.
    pub fn try_new<'a>(code: &'a str) -> Result<Self, UnknownCode<'a>> {
        // Same case-insensitive resolution as [`Self::new`]: a known code
        // in any case is accepted and stored canonically; the error keeps
        // the input verbatim so the user sees what they typed.
        match resolve(code) {
            Some(e) => Ok(ErrCode(ErrCodeRepr::Known(e))),
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
        let raw = code.into();
        // Resolve once here instead of on every `eq`/`hash`: an untrusted
        // spelling such as `e030` must still compare/hash as `E030`, but the
        // table walk is a construction cost, not a per-comparison one.
        let canonical = resolve(&raw);
        ErrCode(ErrCodeRepr::Other(Box::new(OtherCode { raw, canonical })))
    }

    /// The code string, e.g. "E030" or "W113".
    ///
    /// For [`Unknown`](Self::unknown) codes this is the stored spelling
    /// verbatim (untrusted echo semantics); use [`canonical`](Self::canonical)
    /// for identity.
    pub fn code(&self) -> &str {
        match &self.0 {
            ErrCodeRepr::Known(e) => e.code,
            ErrCodeRepr::Other(o) => o.raw.as_str(),
        }
    }

    /// `true` when the code came from the validated registry.  Registry codes
    /// are `^[EW]\d{3}$` (printable ASCII), so callers may skip terminal
    /// sanitization; only the untrusted [`Other`](ErrCodeRepr::Other) spelling
    /// needs it.
    pub(crate) fn is_known(&self) -> bool {
        matches!(self.0, ErrCodeRepr::Known(_))
    }

    /// The canonical spelling used by `Eq`/`Hash`: table entries resolve to
    /// their registered code, so case spellings of one code compare equal.
    /// Unknown codes with no entry fall back to their verbatim string, which
    /// keeps distinct junk codes distinct (`unknown("junk1") !=
    /// unknown("junk2")`).
    pub(crate) fn canonical(&self) -> &str {
        match &self.0 {
            ErrCodeRepr::Known(e) => e.code,
            ErrCodeRepr::Other(o) => o
                .canonical
                .map(|e| e.code)
                .unwrap_or(o.raw.as_str()),
        }
    }

    /// Resolve this error code to its table entry, if present.
    ///
    /// Callers that need multiple fields (`title`, `category`, `explain`)
    /// should use this once and derive each from the returned entry, avoiding
    /// repeated lookups (a shape check + one array read on `CODE_INDEX`).
    pub fn entry(&self) -> Option<&'static CodeEntry> {
        match &self.0 {
            ErrCodeRepr::Known(e) => Some(*e),
            // Preserve the old behaviour for `unknown` codes that happen to
            // name a real entry: they still resolve their metadata, matching
            // the case-insensitive resolution in `new`/`try_new`.
            ErrCodeRepr::Other(o) => o.canonical,
        }
    }

    /// Return the resolved entry or the static unknown-code fallback.
    ///
    /// Equivalent to a single `lookup` call followed by `.unwrap_or(...)`,
    /// so callers can gather `title` / `category` / `explain` in one lookup.
    pub(crate) fn entry_or_unknown(&self) -> &'static CodeEntry {
        match &self.0 {
            ErrCodeRepr::Known(e) => e,
            ErrCodeRepr::Other(o) => o.canonical.unwrap_or(&UNKNOWN_CODE_ENTRY),
        }
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
    ///
    /// **Memoization:** once the explain port is fixed ([`set_explain_port`] is
    /// set-once), the result depends only on the code string.  Codes matching
    /// the registry's `^[EW]\d{3}$` shape are cached per [`code_slot`] in a
    /// lazily-populated map (the same lock-per-key shape
    /// `category_siblings_of`'s `SIBLING_CACHE` uses) — a process that only
    /// ever prints a handful of codes never allocates a
    /// `CODE_INDEX_SLOTS`-sized table of `OnceLock` shells eagerly.  A batch
    /// of *N* diagnostics sharing a code builds the URL once and clones
    /// thereafter.  Unknown codes (arbitrary bytes, no slot) bypass the
    /// cache and rebuild every call.
    ///
    /// Returns an `Arc<str>`: a cache hit is a refcount bump, not a heap
    /// allocation + copy of an immutable 40–80+ byte URL.
    pub fn url(&self) -> std::sync::Arc<str> {
        // Canonical spelling, not the verbatim echo: `unknown("e030")` resolves
        // to the E030 entry (title/category/explain all do), so its URL must be
        // the registered `/E030` — not a `/e030` the doc site 404s — and it
        // must hit the per-slot cache like any other spelling of that code.
        let code = self.canonical();
        let Some(slot) = code_slot(code) else {
            return explain_url(code).into();
        };
        if EXPLAIN_PORT.get().is_some() {
            // Two disjoint caches (port set / port unset) mirror `url_ansi`'s
            // split so a later `set_explain_port` can never observe a stale
            // doc-URL host.
            static URL_CACHE: SlotCache = SlotCache::new();
            cached_slot_string(&URL_CACHE, slot as u16, || explain_url(code))
        } else {
            // No explain server: the doc URL depends only on the code.
            // Disjoint from the port cache (this branch is taken only while
            // the port is unset, and the port is set-once).
            static DOC_URL_CACHE: SlotCache = SlotCache::new();
            cached_slot_string(&DOC_URL_CACHE, slot as u16, || {
                explain_url_with_port(code, None)
            })
        }
    }

    /// The diagnostic URL, formatted as an ANSI hyperlink if the terminal supports it.
    ///
    /// The visible text is sanitized (control characters become U+FFFD) so an
    /// untrusted code cannot terminate the OSC 8 link early and inject further
    /// terminal escapes.
    ///
    /// **Memoization:** once the explain port is fixed ([`set_explain_port`] is
    /// set-once), the result depends only on the code string.  Codes matching
    /// the registry's `^[EW]\d{3}$` shape are cached per [`code_slot`] in the
    /// same lazily-populated map shape [`url`](Self::url) uses (see the
    /// `SlotCache` note there), so a batch of *N* diagnostics sharing a code
    /// builds the OSC 8 hyperlink once and clones thereafter.  Unknown codes
    /// (arbitrary bytes, no slot) bypass the cache and rebuild every call.
    pub fn url_ansi(&self) -> std::sync::Arc<str> {
        let Some(slot) = code_slot(self.canonical()) else {
            return self.url_ansi_build().into();
        };
        if EXPLAIN_PORT.get().is_some() {
            static URL_ANSI_CACHE: SlotCache = SlotCache::new();
            cached_slot_string(&URL_ANSI_CACHE, slot as u16, || self.url_ansi_build())
        } else {
            // No explain server: memoize the OSC 8 form of the doc URL, the
            // same way the doc URL itself is memoized in `url`.  Disjoint
            // from the port cache, so a later `set_explain_port` cannot
            // return a stale doc host.
            static DOC_URL_ANSI_CACHE: SlotCache = SlotCache::new();
            cached_slot_string(&DOC_URL_ANSI_CACHE, slot as u16, || self.url_ansi_build())
        }
    }

    /// Uncached core of [`url_ansi`](Self::url_ansi): build the OSC 8 hyperlink
    /// from the (uncached) [`url`](Self::url) and the sanitized code.  Separated
    /// so the cache path can `set` without holding a borrow across the build.
    fn url_ansi_build(&self) -> CompactString {
        // E6: build the URL exactly once. The old `format_compact!(self.url())`
        // allocated an `Arc<str>` for unknown codes (no registry slot → no
        // cache) only to format it again; assemble straight into the result.
        let mut out = CompactString::default();
        out.push_str("\x1b]8;;");
        if code_slot(self.canonical()).is_some() {
            out.push_str(&self.url());
        } else {
            out.push_str(&explain_url(self.canonical()));
        }
        out.push_str("\x1b\\");
        out.push_str(&super::sanitize_for_terminal(self.code()));
        out.push_str("\x1b]8;;\x1b\\");
        out
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

/// The direct-address index (`CODE_INDEX_SLOTS`) and `suggest_top3_direct`'s
/// 4-deletion enumeration both assume every code is exactly 4 bytes.  Fail the
/// build — not a lookup / a silently wrong suggestion — if the registry grows
/// a 5-char code.
const _: () = {
    let mut i = 0;
    while i < CODE_TABLE.len() {
        let b = CODE_TABLE[i].code.as_bytes();
        // Assert the full `^[EW]\d{3}$` shape, not just the length:
        // `build_code_index` silently skips an entry whose `code_slot` is
        // `None`, so `"EX30"`/`"E03a"` would compile, miss `CODE_INDEX`, and
        // be unreachable through `lookup`.  Fail the build instead.
        assert!(b.len() == 4);
        assert!(b[0] == b'E' || b[0] == b'W');
        assert!(b[1].is_ascii_digit());
        assert!(b[2].is_ascii_digit());
        assert!(b[3].is_ascii_digit());
        i += 1;
    }
};

/// Number of direct-address slots in [`CODE_INDEX`]. Callers that want a
/// lock-free, per-code table (the explain server's page cache) size their
/// array with this.
pub(crate) const DIRECT_ADDRESS_SLOTS: usize = CODE_INDEX_SLOTS;

/// Direct-address slot (`0..DIRECT_ADDRESS_SLOTS`) for a registry code.
///
/// Returns `None` for a string that does not match the `^[EW]\d{3}$` shape;
/// a shape-valid but unregistered code still yields a slot, but its
/// [`lookup`] entry is empty, so callers should resolve through
/// [`ErrCode::try_new`] first.
#[inline]
pub(crate) fn direct_slot(code: &str) -> Option<usize> {
    code_slot(code)
}

/// Case-insensitive registry resolution: canonical [`lookup`] first, then
/// [`lookup_uppercased`]. One choke point for the pattern that was copied
/// verbatim into `new`/`try_new`/`canonical`/`entry`/`entry_or_unknown`/
/// `related_codes_of` (E5).
#[inline]
pub(crate) fn resolve(code: &str) -> Option<&'static CodeEntry> {
    lookup(code).or_else(|| lookup_uppercased(code))
}

/// Look up a code string in the table.
#[inline]
pub(crate) fn lookup(code: &str) -> Option<&'static CodeEntry> {
    let slot = code_slot(code)?;
    let idx = CODE_INDEX[slot];
    if idx == u16::MAX {
        return None;
    }
    CODE_TABLE.get(idx as usize)
}

/// [`lookup`] on the ASCII-uppercased form of `code`, without allocating.
///
/// Every table code is `^[EW]\d{3}$` — exactly four bytes, with only byte 0
/// able to differ by case (bytes 1..3 are digits, on which
/// `to_ascii_uppercase` is the identity).  The build-time `len == 4`
/// assertion in [`CODE_TABLE`]'s shape check enforces that premise, so this
/// needs no stack buffer, no heap fallback, and no `unsafe` UTF-8
/// round-trip.
#[inline]
pub(crate) fn lookup_uppercased(code: &str) -> Option<&'static CodeEntry> {
    let b = code.as_bytes();
    if b.len() != 4 {
        return None;
    }
    let buf = [b[0].to_ascii_uppercase(), b[1], b[2], b[3]];
    lookup(std::str::from_utf8(&buf).ok()?)
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// `CODE_TABLE` grouped by category in first-appearance order, cached once.
///
/// `CODE_TABLE` is in document (toml) order, which is **not** grouped by
/// category — codes are interleaved across categories.  Callers that print
/// one header per category (`--list-codes`, the explain-server index page)
/// must group first; iterating `CODE_TABLE` directly would repeat a category
/// header each time the category reappears.
pub(crate) fn code_entries_by_category() -> &'static [(ErrorCategory, Vec<&'static CodeEntry>)] {
    static GROUPED: std::sync::OnceLock<Box<[(ErrorCategory, Vec<&'static CodeEntry>)]>> =
        std::sync::OnceLock::new();
    GROUPED.get_or_init(|| {
        let mut order: Vec<ErrorCategory> = Vec::new();
        let mut buckets: rustc_hash::FxHashMap<ErrorCategory, Vec<&'static CodeEntry>> =
            rustc_hash::FxHashMap::default();
        for e in CODE_TABLE {
            let bucket = buckets.entry(e.category).or_default();
            if bucket.is_empty() {
                order.push(e.category);
            }
            bucket.push(e);
        }
        order
            .into_iter()
            .map(|cat| (cat, buckets.remove(&cat).unwrap_or_default()))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    })
}

pub fn related_codes_of(code: &str) -> &'static [&'static str] {
    // `RELATED_CODES` is generated by build.rs from
    // `error_codes/codes.toml`'s `related` fields, so the "see also"
    // graph cannot drift from the registry the way the old hand-written
    // `PAIRS` slice could.
    // Resolve case-insensitively, exactly like `new`/`entry`/
    // `category_siblings_of`: `related_codes_of("e030")` used to return `&[]`
    // while the sibling list worked, because `lookup` demands canonical
    // `^[EW]\d{3}$` input.
    let Some(entry) = resolve(code) else {
        return &[];
    };
    let slot = code_slot(entry.code)
        .expect("every CODE_TABLE entry has a direct-address slot") as usize;
    static RELATED_CACHE: std::sync::OnceLock<
        Box<[std::sync::OnceLock<&'static [&'static str]>]>,
    > = std::sync::OnceLock::new();
    let cells = RELATED_CACHE.get_or_init(|| {
        (0..DIRECT_ADDRESS_SLOTS)
            .map(|_| std::sync::OnceLock::new())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    cells[slot].get_or_init(|| {
        RELATED_CODES
            .iter()
            .find_map(|(src, rel)| (*src == entry.code).then_some(*rel))
            .unwrap_or(&[])
    })
}

pub fn category_siblings_of(code: &str, max: usize) -> SmallVec<[&'static str; 4]> {
    // Resolve through the same case-insensitive path as `ErrCode::new`/
    // `entry` (`lookup` alone demands canonical `^[EW]\d{3}$`, so a
    // lowercase `e030` used to return an empty sibling list).  From here on
    // `code` is the CANONICAL table code, so the cache key can't fragment
    // across case spellings of the same code.
    let Some(entry) = resolve(code) else {
        return SmallVec::new();
    };
    // `lookup_uppercased` only ever returns table entries, and every table
    // entry passed the `^[EW]\d{3}$` build assertion — so a slot exists.  The
    // old `else { … }` fallback here was unreachable dead code duplicating
    // `build_siblings`' filtering.
    let slot = code_slot(entry.code)
        .expect("every CODE_TABLE entry has a direct-address slot") as u16;
    
    // Codes grouped by category, built once. CODE_TABLE is static so the
    // grouping is invariant; caching it turns the per-call O(table) scan into
    // O(category) — a real win on the terminal renderer's per-diagnostic
    // "see also" path, which calls this for every diagnostic without explicit
    // related codes.
    static BY_CATEGORY: std::sync::OnceLock<
        rustc_hash::FxHashMap<ErrorCategory, Vec<&'static str>>,
    > = std::sync::OnceLock::new();
    let by_category = BY_CATEGORY.get_or_init(|| {
        let mut map: rustc_hash::FxHashMap<ErrorCategory, Vec<&'static str>> =
            rustc_hash::FxHashMap::default();
        for e in CODE_TABLE {
            map.entry(e.category).or_default().push(e.code);
        }
        map
    });
    // Production callers ask for 4; a per-slot `OnceLock` array makes the
    // hit path a single atomic load, with no `RwLock` or `LiteMap` search.
    // Other `max` values keep the general keyed cache below.
    if max == 4 {
        static SIBLING4: std::sync::OnceLock<
            Box<[std::sync::OnceLock<SmallVec<[&'static str; 4]>>]>,
        > = std::sync::OnceLock::new();
        let cells = SIBLING4.get_or_init(|| {
            (0..DIRECT_ADDRESS_SLOTS)
                .map(|_| std::sync::OnceLock::new())
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });
        return cells[slot as usize]
            .get_or_init(|| build_siblings(by_category, entry, max))
            .clone();
    }
    // The per-code *answer* is memoised per `(slot, max)`: `max` is a
    // call-site constant in the renderers today, but the function is `pub`
    // and nothing enforces a single `max` — keying the cache by the slot
    // alone made a caller asking for `max = 2` after `max = 4` silently get
    // the four-entry list.  The slots live in a lazily-populated flat map —
    // a `LiteMap`, a sorted flat vector with O(log n) lookup and no
    // hash-table shell — rather than a fixed `CODE_INDEX_SLOTS`-sized
    // array, so a process that touches a handful of codes pays only for
    // those codes; the per-slot `OnceLock` keeps the hit path lock-free.
    static SIBLING_CACHE: std::sync::OnceLock<
        std::sync::RwLock<LiteMap<(u16, usize), std::sync::OnceLock<SmallVec<[&'static str; 4]>>>>,
    > = std::sync::OnceLock::new();
    let cache = SIBLING_CACHE
        .get_or_init(|| std::sync::RwLock::new(LiteMap::new()));
    let slot_hit = || -> Option<SmallVec<[&'static str; 4]>> {
        let m = cache.read().ok()?;
        m.get(&(slot, max)).and_then(OnceLock::get).cloned()
    };
    if let Some(cached) = slot_hit() {
        return cached;
    }
    // Build *before* taking the write lock: `build_siblings` walks the
    // category list and allocates, and holding the global lock across it
    // would serialize unrelated codes' "see also" lookups under parallel
    // rendering.  A race loses only the duplicate build; the winner's
    // per-slot `OnceLock` wins and both callers return the same answer.
    //
    // `build_siblings` is a pure function of `(entry.category, max)`, so the
    // value we built is byte-identical to whatever won the race — return it
    // directly instead of re-acquiring the read lock and cloning the winner.
    let built = build_siblings(by_category, &entry, max);
    if let Ok(mut m) = cache.write() {
        let _ = m.entry((slot, max)).or_default().set(built.clone());
    }
    built
}

/// The per-category sibling list filtered to `max` entries, excluding the
/// code itself — the value `SIBLING_CACHE` memoises per slot.
fn build_siblings(
    by_category: &rustc_hash::FxHashMap<ErrorCategory, Vec<&'static str>>,
    entry: &CodeEntry,
    max: usize,
) -> SmallVec<[&'static str; 4]> {
    let siblings: &[&'static str] = by_category
        .get(&entry.category)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    siblings
        .iter()
        .filter(|c| **c != entry.code)
        .take(max)
        .copied()
        .collect()
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Lowercase input must resolve like `ErrCode::new`/`entry` do: the
    /// sibling list is computed from the canonical entry, not returned empty
    /// because `lookup` demands `^[EW]\d{3}$`.
    #[test]
    fn category_siblings_accepts_lowercase_code() {
        let upper = category_siblings_of("E030", 4);
        let lower = category_siblings_of("e030", 4);
        assert!(!upper.is_empty(), "E030 has category siblings");
        assert_eq!(lower, upper, "lowercase input returned a different (empty) list");
        // And the case spellings share one cache entry (canonical key).
        assert_eq!(category_siblings_of("E030", 2).len(), 2);
        assert_eq!(category_siblings_of("e030", 2).len(), 2);
    }

    /// E3: `EXPLAIN_PORT` is process-global, and Cargo runs tests in parallel
/// threads within one binary. Serialize every test that mutates it or reads a
/// URL derived from it, so a future explicit host assertion cannot flake on
/// test ordering.
static PORT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A second `set_explain_port` must be observable rather than silently
    /// discarded.  `EXPLAIN_PORT` is a process-global `OnceLock`, so this test
    /// deliberately owns it (no other test sets it).
    #[test]
    fn set_explain_port_reports_second_set() {
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ec = ErrCode::unknown("E030\n../admin");
        let u = ec.url();
        assert!(u.contains("E030%0A..%2Fadmin"), "got: {u}");
        assert!(!u.contains('\n'), "newline leaked: {u:?}");
        assert!(!u.contains("../"), "path traversal not encoded: {u}");
    }

    /// Regression guard: encoding must not alter canonical codes.
    #[test]
    fn known_codes_are_unchanged() {
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ec = ErrCode::new(ErrCode::E030);
        assert!(ec.url().contains("/E030"), "got: {}", ec.url());
        assert!(!ec.url().contains('%'), "got: {}", ec.url());
        assert!(ec.url_ansi().contains("E030"));
    }

    /// The sibling cache is keyed by `(slot, max)`: a caller asking for a
    /// smaller `max` after a larger one must get the truncated list, not the
    /// first caller's cached answer (the cache is process-global).
    #[test]
    fn category_siblings_respects_per_max_cache_key() {
        let wide = category_siblings_of("E030", 4);
        let narrow = category_siblings_of("E030", 2);
        assert_eq!(narrow.len(), wide.len().min(2), "narrow max must truncate");
        for (n, w) in narrow.iter().zip(wide.iter()) {
            assert_eq!(n, w, "narrow list must be a prefix of the wide list");
        }
        // Asking again for the wide `max` must not be poisoned by the narrow
        // call that preceded it.
        assert_eq!(category_siblings_of("E030", 4).len(), wide.len());
    }

    /// The table-entry and verbatim storage forms must be interchangeable:
    /// equality, hashing, and metadata all compare the code string, so an
    /// `unknown` code that happens to name a real entry still resolves.
    #[test]
    fn known_and_verbatim_forms_are_equivalent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let known = ErrCode::new("E030");
        let verbatim = ErrCode::unknown("E030");
        assert_eq!(known, verbatim);
        assert_eq!(known.code(), verbatim.code());
        assert_eq!(known.title(), verbatim.title());
        assert_eq!(known.category(), verbatim.category());

        let hash = |c: &ErrCode| {
            let mut h = DefaultHasher::new();
            c.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(&known), hash(&verbatim));

        // Case spellings resolve to the same canonical identity: an unknown
        // verbatim lowercase code must equal the canonical table form, or a
        // `HashSet<ErrCode>` would hold two entries for one logical code.
        let lower_verbatim = ErrCode::unknown("e030");
        let lower_known = ErrCode::new("e030");
        assert_eq!(known, lower_verbatim);
        assert_eq!(lower_known, lower_verbatim);
        assert_eq!(hash(&known), hash(&lower_verbatim));
        // Unknown junk stays distinct (no false collapse through the empty
        // `UNKNOWN_CODE_ENTRY` fallback).
        assert_ne!(ErrCode::unknown("junk1"), ErrCode::unknown("junk2"));
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
            // `file_type()` does not follow symlinks, so a cyclic symlink
            // under `src/` cannot make this walk recurse forever.
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_dir() {
                rs_files(&path, out);
            } else if ft.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    /// Return the argument text of the parenthesised call whose `(` is at
    /// `open_idx`, skipping nested parens, string literals, and comments.
    /// Skip a raw string starting at the `r` of `r#"…"#` (or `br#"…"#`),
/// returning the index just past the closing delimiter (or `len`).
    fn skip_raw_string(bytes: &[u8], r: usize) -> Option<usize> {
        let mut h = r + 1;
        while h < bytes.len() && bytes[h] == b'#' {
            h += 1;
        }
        if h >= bytes.len() || bytes[h] != b'"' {
            return None;
        }
        let hashes = h - (r + 1);
        let mut j = h + 1;
        while j < bytes.len() {
            if bytes[j] == b'"' {
                let mut k = j + 1;
                let mut cnt = 0;
                while cnt < hashes && k < bytes.len() && bytes[k] == b'#' {
                    k += 1;
                    cnt += 1;
                }
                if cnt == hashes {
                    return Some(k);
                }
            }
            j += 1;
        }
        Some(bytes.len())
    }

    /// Skip a char literal starting at the opening `'`, honouring `'\''` and
    /// `'\u{…}'`, returning the index just past the closing quote (or `len`).
    fn skip_char_literal(bytes: &[u8], i: usize) -> usize {
        let len = bytes.len();
        let mut j = i + 1;
        if j >= len {
            return len;
        }
        if bytes[j] == b'\\' {
            j += 1;
            if j >= len {
                return len;
            }
            if bytes[j] == b'u' && j + 1 < len && bytes[j + 1] == b'{' {
                j += 2;
                while j < len && bytes[j] != b'}' {
                    j += 1;
                }
                if j < len {
                    j += 1;
                }
            } else {
                j += 1;
            }
        } else {
            let ch = std::str::from_utf8(&bytes[j..])
                .ok()
                .and_then(|s| s.chars().next());
            j += ch.map_or(1, char::len_utf8);
        }
        if j < len && bytes[j] == b'\'' {
            (j + 1).min(len)
        } else {
            // A lifetime such as `'static` is not a char literal: skip only
            // the quote so the following identifier is scanned normally.
            i + 1
        }
    }

    fn argument_span(source: &str, open_idx: usize) -> Option<&str> {
        let bytes = source.as_bytes();
        let mut depth = 0i32;
        let mut i = open_idx;
        while i < bytes.len() {
            // Raw strings (r"…", r#"…"#, br#"…"#) may contain parens/quotes
            // that must not affect the scan (E2).
            if bytes[i] == b'r' || (bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'r')
            {
                let r = if bytes[i] == b'r' { i } else { i + 1 };
                if let Some(end) = skip_raw_string(bytes, r) {
                    i = end;
                    continue;
                }
            }
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&source[open_idx + 1..i]);
                    }
                }
                // Char literals: `'('` must not change paren depth, `'"'` must
                // not flip the scanner into string mode (E2).
                b'\'' => {
                    i = skip_char_literal(bytes, i);
                    continue;
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
            // Raw strings are not code literals and may contain quotes that
            // would desynchronise the scan (E2).
            if bytes[i] == b'r' || (bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'r')
            {
                let r = if bytes[i] == b'r' { i } else { i + 1 };
                if let Some(end) = skip_raw_string(bytes, r) {
                    i = end;
                    continue;
                }
            }
            match bytes[i] {
                // A char literal such as `'"'` would otherwise flip the
                // scanner into string mode (E2).
                b'\'' => {
                    i = skip_char_literal(bytes, i);
                }
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
    fn scanner_handles_char_and_raw_literals() {
        // E2: a `'('` char literal must not change paren depth, a `'"'` must
        // not flip the scanner into string mode, and a raw string must not be
        // scanned as a code literal.
        fn codes_after<'a>(src: &'a str, callee: &str) -> Vec<&'a str> {
            let after = match src.find(callee) {
                Some(p) => p + callee.len(),
                None => return vec![],
            };
            let tail = &src[after..];
            let ws = tail.len() - tail.trim_start().len();
            let open = after + ws;
            argument_span(src, open)
                .map(code_literals)
                .unwrap_or_default()
        }
        let paren = r#"let a = ErrCode::new(if c == '(' { "E030" } else { "E031" });"#;
        let mut got = codes_after(paren, "ErrCode::new");
        got.sort_unstable();
        assert_eq!(got, vec!["E030", "E031"], "char '(' broke the scan");
        // A code-shaped body inside a raw string is not a literal.
        assert!(
            codes_after(r##"let b = ErrCode::new(r#"x"E998""#);"##, "ErrCode::new").is_empty(),
            "raw string body scanned as a code literal"
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
    fn code_literals_skip_lifetimes() {
        assert_eq!(
            code_literals("ErrCode::new(foo::<&'static str>(\"E030\"))"),
            vec!["E030"]
        );
        assert_eq!(
            code_literals("if x == '&' { \"E031\" } else { \"E032\" }"),
            vec!["E031", "E032"]
        );
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
