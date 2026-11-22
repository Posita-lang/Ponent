use compact_str::{CompactString, format_compact};
use std::borrow::Cow;
use std::fmt;
use std::sync::OnceLock;

/// The port number of the local explain server, if running.
static EXPLAIN_PORT: OnceLock<u16> = OnceLock::new();

/// Test-only override of [`EXPLAIN_PORT`].
///
/// `Some(Some(p))` forces the port, `Some(None)` forces "no port set".  The
/// process global is set-once, so without this a test cannot order "render, then
/// set, then re-render" — the user-visible contract of [`set_explain_port`].
/// Thread-local, and libtest gives every test its own thread, so an override
/// cannot leak into another test.
#[cfg(test)]
thread_local! {
    static TEST_EXPLAIN_PORT: std::cell::RefCell<Option<Option<u16>>> =
        const { std::cell::RefCell::new(None) };
}

/// The effective explain port: the test override when present, else the
/// process-global set-once value.
fn explain_port() -> Option<u16> {
    #[cfg(test)]
    if let Some(port) = TEST_EXPLAIN_PORT.with(|c| *c.borrow()) {
        return port;
    }
    EXPLAIN_PORT.get().copied()
}

/// Run `f` with `port` as the effective explain port, restoring the previous
/// override even on panic.
#[cfg(test)]
fn with_explain_port<R>(port: Option<u16>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Option<u16>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_EXPLAIN_PORT.with(|c| *c.borrow_mut() = self.0);
        }
    }
    let previous = TEST_EXPLAIN_PORT.with(|c| *c.borrow());
    TEST_EXPLAIN_PORT.with(|c| *c.borrow_mut() = Some(port));
    let _restore = Restore(previous);
    f()
}

/// Direct-address index over [`CODE_TABLE`], built at compile time.
///
/// The registry is validated at build time against `^[EW]\d{3}$`, so every
/// code is exactly one prefix byte plus three digits: slot
/// `prefix * 1000 + number`, with `u16::MAX` marking an absent code.  A
/// lookup is then a shape check + one array read — no lazy `OnceLock` init,
/// no SipHash, no lock, and the table is shared read-only across threads.
/// Array size of the direct-address index: the smallest power of two that
/// covers the slot space (`^[EW]\d{3}$` → 2000 entries).  Lookups index with
/// `slot & (CODE_INDEX_SLOTS - 1)`, which is the identity for every slot
/// `code_slot` can produce and lets LLVM drop the bounds check on the
/// per-diagnostic construction path.  The compile-time assertions below pin the
/// relationship the mask relies on.
const CODE_INDEX_SLOTS: usize = 2048;
/// Array size of each per-slot cache table; same power-of-two/mask argument.
const SLOT_CACHE_SLOTS: usize = 2048;
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
    code_slot_parts(b[0], b[1], b[2], b[3])
}

/// Shared slot arithmetic for the canonical `^[EW]\d{3}$` shape.  Extracted so
/// [`lookup_uppercased`] can compute a slot without building a stack buffer and
/// round-tripping through `from_utf8` plus the full shape check.
#[inline]
const fn code_slot_parts(prefix: u8, d1: u8, d2: u8, d3: u8) -> Option<usize> {
    let p = match prefix {
        b'E' => 0usize,
        b'W' => 1,
        _ => return None,
    };
    if !(d1.is_ascii_digit() && d2.is_ascii_digit() && d3.is_ascii_digit()) {
        return None;
    }
    Some(p * 1000 + (d1 - b'0') as usize * 100 + (d2 - b'0') as usize * 10 + (d3 - b'0') as usize)
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

/// Per-slot memoization table shared by [`ErrCode::url`] / [`ErrCode::url_ansi`]
/// (one `SlotCache` static per (port-set, doc-URL) × (plain, ANSI) quadrant).
///
/// Shape: one [`std::sync::OnceLock`] per direct-address slot, indexed by the
/// slot itself — the slot space is the closed `^[EW]\d{3}$` range
/// (`DIRECT_ADDRESS_SLOTS = 2000`), so the cache needs neither a hash nor a
/// lock.  The hit path is one `Acquire` load of the slot state, a null check and
/// a fat-pointer copy of the `&'static str`; `build` runs exactly once per slot
/// even under a true race (`get_or_init` parks the loser instead of the old
/// map's "both build, one insert wins").
///
/// The `OnceLock` slots are inline in the static; a slot's `Box<str>` payload is
/// heap-allocated on first use and never freed.  Only the pages a process
/// actually renders are ever committed (~48 KB of BSS per table, ~98 KB across
/// the two — BSS, not resident).  Only the port-*set* quadrants are cached: the
/// port-unset ones were deleted (registry codes are build-time constants there,
/// everything else is an uncached build), so there is nothing to invalidate when
/// the port is set later.
///
/// Storing `Box<str>` and *borrowing* it out (rather than `Arc<str>` and
/// cloning) is what makes the hit path free: the earlier design paid an atomic
/// RMW per diagnostic hyperlink.  The "raw-pointer tier" alternative would have
/// matched that cost but needed a deliberate `Box::into_raw` leak reconstructed
/// through raw pointers; borrowing from a `&'static OnceLock` gets the same
/// result with no `unsafe`, no leak, and no refcount.
///
/// # Invariant
///
/// Slot-keyed caching is sound only because [`code_slot`] is **injective**
/// over its accepted domain: the shape is `^[EW]\d{3}$` with a CASE-SENSITIVE
/// prefix, so two distinct shape-valid strings can never share a slot
/// (`"e030"` yields `None`, not slot 30).  That is what makes caching an
/// arbitrary spelling (e.g. `unknown("E999")`, whose visible link text is
/// `"E999"`) under its slot safe.  Do NOT widen the accepted prefix set —
/// `lookup_uppercased` is the case-insensitive path and deliberately does not
/// feed this cache.  `slot_keyed_caches_require_an_injective_code_slot`
/// pins this.
type SlotCache = [std::sync::OnceLock<Box<str>>; SLOT_CACHE_SLOTS];

/// The all-empty cache `static` initialiser.
///
/// MSRV note: this module (including its test-only scanners) relies on
/// let-chains (Rust 1.88 / edition 2024), `Option::is_none_or` (1.82) and inline
/// `const {}` blocks (1.79); the crate pins `rust-version = "1.89"` in
/// `Cargo.toml`, which is the binding floor.  A toolchain downgrade would fail
/// here, far from the feature that forced the bump, so the dependency is
/// recorded next to the construct.
const fn new_slot_cache() -> SlotCache {
    [const { std::sync::OnceLock::new() }; SLOT_CACHE_SLOTS]
}

#[inline]
fn cached_slot_string(
    cache: &'static SlotCache,
    slot: u16,
    build: impl FnOnce() -> String,
) -> &'static str {
    // The whole hit path: an `Acquire` load of this slot's state, a null check,
    // and a fat-pointer copy.  `code_slot` guarantees `slot <
    // DIRECT_ADDRESS_SLOTS`, so the index is in range by construction.
    //
    // The value lives in the static for the rest of the process, so borrowing it
    // is sound with no `unsafe`, no `Box::leak`, and no refcount: the previous
    // `OnceLock<Arc<str>>` paid an atomic RMW (`Arc::clone`) per diagnostic
    // hyperlink, which is exactly the cost the old "raw-pointer tier" note
    // wanted to remove but rejected as needing a deliberate leak.  A `Box<str>`
    // in the same slot gets the same result by *borrowing* instead of cloning.
    debug_assert!((slot as usize) < DIRECT_ADDRESS_SLOTS);
    &**cache[(slot as usize) & (SLOT_CACHE_SLOTS - 1)]
        // `String` straight from the builder: the old `() -> CompactString`
        // closure forced an `into_string()` hop (a copy for any URL short enough
        // to live inline) before `into_boxed_str()`.  Cache misses only, but the
        // hop was pure overhead.
        .get_or_init(|| build().into_boxed_str())
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
///
/// # Call this before emitting diagnostics
///
/// The per-slot URL caches are split into "port set" and "port unset"
/// quadrants, which is what makes them sound across a port change: no entry is
/// ever consulted on the wrong side.  They cannot, however, *invalidate* an
/// entry that was already built.  So a diagnostic rendered before the port was
/// set keeps its documentation URL, and re-rendering it afterwards yields
/// `127.0.0.1:<port>` — the same logical diagnostic showing two different link
/// targets.  Setting the port first is therefore a contract, not a performance
/// hint: a caller that renders first and sets the port later has already put the
/// documentation host in front of the user.
///
/// `EXPLAIN_PORT` is a set-once process global, so the ordering can only be
/// pinned in a test through the `#[cfg(test)]` thread-local override
/// (`with_explain_port`), which is what
/// `url_flips_between_doc_and_localhost_across_the_override` does.
pub fn set_explain_port(port: u16) -> Result<(), u16> {
    EXPLAIN_PORT.set(port).map_err(|_| {
        // `set` fails exactly when a value already exists, so `get` is `Some`
        // here; `expect` states the invariant instead of papering over it with
        // `unwrap_or(&port)`.
        *EXPLAIN_PORT
            .get()
            .expect("OnceLock::set failed ⇒ a value is already present")
    })
}

/// Whether the explain server is running (its port has been set).
pub fn is_explain_server_running() -> bool {
    EXPLAIN_PORT.get().is_some()
}

/// Build the explanation URL for a code, given an explicit port.
///
/// This is the pure, testable core of [`ErrCode::url`]; callers that need the
/// process-global port (the production path) pass `EXPLAIN_PORT.get().copied()`.
/// The code is percent-encoded as a single path segment in both branches, so an
/// untrusted code (CLI argument, `Diagnostic::with_code_str`) cannot escape the
/// segment or smuggle control characters into the URL.
///
/// This function itself does not memoize — the per-call cost is one
/// `format_compact!`.  The hot-path caller [`ErrCode::url_ansi`] (invoked per
/// diagnostic in `write_header` when hyperlinks are active) caches its full
/// OSC 8 result in a [`SlotCache`] — a *fixed* `SLOT_CACHE_SLOTS`-entry array of
/// `OnceLock`s indexed by [`code_slot`], not a hash map — and there is one such
/// array per quadrant ([`ErrCode::url`] and [`ErrCode::url_ansi`] each keep a
/// port-set and a port-unset cache), so the doc-URL path is cached too, not only
/// the localhost one.  A batch of *N* diagnostics sharing a code therefore pays
/// the build once and a fat-pointer copy per hit (no `Arc` refcount, no clone).
/// The cold-path caller (the explain server, one URL per served page) bypasses
/// that cache.
/// Write the localhost explain URL into `out` (the single format site).
fn explain_server_url_into(port: u16, code: &str, out: &mut String) {
    use std::fmt::Write as _;
    let _ = write!(
        out,
        "http://127.0.0.1:{port}/{}",
        encode_url_path_segment(code)
    );
}

pub(crate) fn explain_url_with_port(code: &str, port: Option<u16>) -> String {
    if let Some(port) = port {
        // Straight into the result: the old path built a `CompactString` and
        // called `into_string()`, which the `SlotCache` doc above claims was
        // removed — the hop was still there for localhost URLs, and it copies
        // whenever the URL is short enough to be inline-resident.
        let mut out = String::with_capacity(24);
        explain_server_url_into(port, code, &mut out);
        out
    } else {
        // We will register this domain and set up a proper documentation
        // website in the future.  For now, the local explain server is the
        // primary way to view error explanations in a browser.
        const DOC_URL_PREFIX: &str = "https://doc.posita-lang.org/error_codes/";
        const DOC_URL_SUFFIX: &str = ".html";
        let encoded = encode_url_path_segment(code);
        let mut out =
            String::with_capacity(DOC_URL_PREFIX.len() + encoded.len() + DOC_URL_SUFFIX.len());
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
    // The CLI keeps its `CompactString` contract; the format itself lives in one
    // place so the two callers cannot drift.
    let mut out = String::with_capacity(24);
    explain_server_url_into(port, code, &mut out);
    CompactString::from(out)
}

/// Percent-encode a single URL path segment per RFC 3986.  Unreserved
/// characters pass through; everything else — including `/`, `:`, `%`,
/// control characters, and non-ASCII bytes — becomes `%XX`.  Stricter than
/// [`term::encode_file_url_path`](crate::diagnostics::term) because a code
/// is *one* segment, and a raw `\x1b` here would terminate the enclosing
/// OSC 8 hyperlink early.
fn encode_url_path_segment(s: &str) -> Cow<'_, str> {
    use crate::diagnostics::HEX_UPPER;
    /// RFC 3986 unreserved set: `ALPHA / DIGIT / "-" / "." / "_" / "~"`.
    const fn is_unreserved(b: u8) -> bool {
        matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~')
    }
    // One scan for both decisions: the first byte that needs escaping decides
    // "borrow or build", and the prefix before it is copied verbatim (the old
    // form ran the fast-path `all()` to completion and then re-scanned from
    // byte 0 on the slow path).
    let Some(first_bad) = s.bytes().position(|b| !is_unreserved(b)) else {
        return Cow::Borrowed(s);
    };
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len() + 2 * (bytes.len() - first_bad));
    out.push_str(&s[..first_bad]);
    for &b in &bytes[first_bad..] {
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
    pub const fn key(&self) -> &'static str {
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

    /// Dense index into fixed-size category tables (`0..COUNT`).
    ///
    /// Written as an explicit match, like [`key`](Self::key) and
    /// [`as_str`](Self::as_str), so reordering the enum cannot silently permute
    /// a table: a new variant is a compile error here, and
    /// `category_index_is_dense_and_unique` pins the range.
    pub(crate) const fn index(&self) -> usize {
        match self {
            ErrorCategory::Parse => 0,
            ErrorCategory::Resolution => 1,
            ErrorCategory::Type => 2,
            ErrorCategory::Contract => 3,
            ErrorCategory::Trait => 4,
            ErrorCategory::Inference => 5,
            ErrorCategory::Internal => 6,
            ErrorCategory::Generic => 7,
        }
    }

    /// Number of categories; the length of every `index()`-keyed table.
    pub(crate) const COUNT: usize = 8;

    /// The canonical CLI keys, in `index()` order.
    ///
    /// Single source for both `--only-category` validation and the "valid: …"
    /// text in `from_str`'s error, so a new category cannot be accepted by the
    /// parser while a hand-written message calls it invalid.
    /// Derived from [`key`](Self::key), in `index()` order — the previous
    /// hand-maintained parallel array pinned only the arity, so renaming one
    /// `key()` arm without touching this table silently made `from_str` accept a
    /// key the CLI rejected (and the "valid: …" text advertise the wrong one).
    /// Now a new variant forces a change here, and a mismatch cannot compile.
    pub(crate) const CANONICAL_KEYS: [&'static str; Self::COUNT] = [
        ErrorCategory::Parse.key(),
        ErrorCategory::Resolution.key(),
        ErrorCategory::Type.key(),
        ErrorCategory::Contract.key(),
        ErrorCategory::Trait.key(),
        ErrorCategory::Inference.key(),
        ErrorCategory::Internal.key(),
        ErrorCategory::Generic.key(),
    ];

    /// Human-readable display string (e.g. `"Parse Error"`).  **Not** the
    /// machine-readable key — JSON/CLI consumers must use [`key`](Self::key),
    /// and localized output should use [`label`](Self::label).
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
        // The variant → table-slot mapping lives in `index()` (`pub(crate) const
        // fn`, next to the enum): reusing it keeps the "reorder is a compile
        // error" property while removing a second match that had to be edited
        // in lockstep.  The init array below pins its own arity against `n = 8`.
        let idx = self.index();
        crate::diagnostics::i18n::cached_label! {
            n = 8,
            variant = idx,
            slugs = [
                "category-parse",
                "category-resolution",
                "category-type",
                "category-contract",
                "category-trait",
                "category-inference",
                "category-internal",
                "category-generic",
            ],
        }
    }
}

impl std::str::FromStr for ErrorCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        for (name, category) in [
            ("parse", ErrorCategory::Parse),
            ("parse error", ErrorCategory::Parse),
            ("resolution", ErrorCategory::Resolution),
            ("resolution error", ErrorCategory::Resolution),
            ("type", ErrorCategory::Type),
            ("type error", ErrorCategory::Type),
            ("contract", ErrorCategory::Contract),
            ("contract error", ErrorCategory::Contract),
            ("trait", ErrorCategory::Trait),
            ("trait error", ErrorCategory::Trait),
            ("inference", ErrorCategory::Inference),
            ("inference error", ErrorCategory::Inference),
            ("internal", ErrorCategory::Internal),
            ("internal error", ErrorCategory::Internal),
            ("generic", ErrorCategory::Generic),
            ("error", ErrorCategory::Generic),
        ] {
            if s.eq_ignore_ascii_case(name) {
                return Ok(category);
            }
        }
        Err(format!(
            "unknown error category `{}` (valid: {})",
            super::sanitize_for_terminal(s),
            // Derived from the table above rather than a hand-written list: a new
            // category updates one place, and the CLI can never reject a value
            // `from_str` accepts.
            Self::CANONICAL_KEYS.join(", ")
        ))
    }
}

// `ErrCode` is sent across threads by the diagnostics pipeline (`Arc<OtherCode>`,
// `&'static ErrCodeInner`, `OnceLock<Box<str>>` are all `Send + Sync`).  Pin it
// here so a future payload field that is not `Sync` fails at this definition
// instead of at a distant use site.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ErrCode>();
};

/// A compiler error or warning code (e.g. "E030", "W113").
///
/// A registry code is a `&'static` pointer to its [`ErrCodeInner`], so
/// `ErrCode` is **two** words — the additional variant discriminant is
/// unavoidable (pinned by `err_code_stays_two_words`) — and it is `Clone`, not
/// `Copy`: cloning a `Known` code is free, while cloning an `Other` code pays one
/// atomic RMW on its `Arc<OtherCode>`.  Construction of a registry code is one
/// `CODE_INDEX` read (no allocation, no atomic), and `title()` / `category()` /
/// `explain()` need no further lookup.  A code that is not in the table (CLI
/// input, `Diagnostic::with_code_str`) allocates one `Arc<OtherCode>` on first
/// use — it is reference-counted and freed with the last handle, *not* interned
/// or leaked (see `ErrCodeRepr::Other`).  Equality and hashing compare the **canonical**
/// spelling (what
/// `PartialEq`/`Hash` use, via [`Self::canonical`]), so case spellings of one
/// registry code are one key: `unknown("e030") == new("E030")` even though
/// `code()` echoes `"e030"` for the unresolved form.  `code()` is the stored
/// spelling, **not** the identity — keying a `HashSet<ErrCode>` on it would
/// split one code into two entries.
#[derive(Clone)]
pub struct ErrCode(ErrCodeRepr);

/// The two storage forms behind [`ErrCode`].
#[derive(Clone)]
enum ErrCodeRepr {
    /// Registry code: a pointer into the generated `const CODE_INNER` slice.
    Known(&'static ErrCodeInner),
    /// Anything not spelled exactly like a registry code.  Reference counted, so
    /// the payload is freed with the last handle: handing out `&'static` here
    /// would require leaking or lifetime erasure, neither of which this
    /// repository accepts as a design shortcut.
    Other(std::sync::Arc<OtherCode>),
}

impl ErrCodeRepr {
    /// `true` when the visible spelling *is* the canonical one, i.e. the
    /// precomputed link text (and doc URL) applies verbatim.
    fn is_canonical_spelling(&self) -> bool {
        match self {
            ErrCodeRepr::Known(_) => true,
            // `None` (unregistered but shape-valid, e.g. `E999`) keeps the
            // slot-cache behaviour; only a non-canonical spelling of a
            // registered code must bypass it, because the cached string embeds
            // the visible text.
            ErrCodeRepr::Other(o) => o.canonical.is_none_or(|e| e.code == o.raw.as_str()),
        }
    }
}

/// One registry payload, generated at build time.
///
/// `doc_url` / `doc_url_ansi` are pure functions of the code, so they are baked
/// in: the "explain port not set" half of the URL caches is *deleted* rather
/// than memoized, and `url()` for a registry code is a field read.  The type is
/// fully immutable, which is what lets the generated slice be a `const`.
pub(crate) struct ErrCodeInner {
    /// Verbatim spelling (`unknown("e030")` echoes `e030`).
    pub(crate) display: &'static str,
    /// Canonical spelling — the identity `Eq`/`Hash` use.
    pub(crate) canonical: &'static str,
    /// Resolved registry entry, when there is one.
    pub(crate) entry: Option<&'static CodeEntry>,
    /// `true` for a validated registry spelling: `display` is printable ASCII
    /// and needs no terminal sanitization.
    pub(crate) known: bool,
    /// `FNV-1a` of [`Self::canonical`], so `Hash` is one `write_u64`.
    pub(crate) hash: u64,
    /// Documentation URL (no explain server running).
    pub(crate) doc_url: &'static str,
    /// OSC 8 hyperlink wrapping that URL.
    pub(crate) doc_url_ansi: &'static str,
}

/// Payload of an [`ErrCodeRepr::Other`] code.
struct OtherCode {
    /// Verbatim spelling — untrusted, echoed by [`ErrCode::code`].
    raw: CompactString,
    /// Case-insensitive resolution, cached so `Eq`/`Hash` never re-run `resolve`.
    canonical: Option<&'static CodeEntry>,
    /// `FNV-1a` of the canonical spelling, so `Hash` is one `write_u64`.
    hash: u64,
    /// Memoized [`super::sanitize_for_terminal`] result.
    sanitized: std::sync::OnceLock<Box<str>>,
}

impl OtherCode {
    fn sanitized(&self) -> &str {
        self.sanitized
            .get_or_init(|| super::sanitize_for_terminal(&self.raw).into())
    }
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
        // Registry codes are validated printable ASCII; only untrusted
        // `unknown` spellings need the control-byte scan/allocation.
        match &self.0 {
            ErrCodeRepr::Known(e) => f.write_str(e.display),
            // Memoized in the payload: a re-rendered unknown code used to scan
            // *and* allocate on every `Display`.
            ErrCodeRepr::Other(o) => f.write_str(o.sanitized()),
        }
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
        let (a, b) = (self.canonical(), other.canonical());
        // Two representations of one registry code canonicalise to the *same*
        // `&'static str` (`e.code`), so a fat-pointer compare settles the common
        // `HashSet<ErrCode>` dedup case in one instruction; junk codes live in
        // distinct allocations and fall through to the byte compare.  Registry
        // duplicates are rejected at build time, so pointer equality can never
        // conflate two distinct codes.
        //
        // `std::ptr::eq` on a wide pointer compares the address *and* the
        // metadata, so a same-address/different-length pair is already unequal
        // and no explicit length check is needed.  What makes the fast path
        // sound is the shape of `canonical()`: it returns whole allocations (an
        // entire `'static` literal or an entire `Arc<OtherCode>` payload), never
        // a derived slice, so equal addresses can only mean equal codes.
        // Every equal code now shares one `'static` canonical allocation:
        // registry codes point at `CodeEntry::code`, and an unknown spelling is
        // its own interned allocation.  Pointer equality is therefore exact, so
        // the byte-compare fallback the `Arc` version needed is gone.
        std::ptr::eq(a, b) || a == b
    }
}

impl Eq for ErrCode {}

/// Cold constructor for an unregistered code: the one `Arc<OtherCode>`
/// allocation an unknown code is documented to cost.
///
/// Out of line so `new`/`unknown`/`try_new` callers keep their registry-hit path
/// I-cache tight — the same treatment `ascii.rs::x86_level_init` gets.  `raw`
/// keeps the caller's verbatim spelling; `canonical` is the resolved entry when
/// a differently-cased spelling hit the table.
/// FNV-1a over the canonical spelling.
///
/// The exact function is an implementation detail, but it must be *the same* for
/// every representation of one code: `Hash` for every `ErrCode` writes
/// `code_hash(self.canonical())`, so the interned payload can precompute it.
const fn code_hash(s: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let bytes = s.as_bytes();
    let mut h = OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(PRIME);
        i += 1;
    }
    h
}

/// FNV-1a over the canonical spelling.
///
/// The hash must be **identical across representations**: a registry code and a
/// spelling of it that resolved through `Other` compare equal (see
/// `PartialEq`/`Hash`), so they must hash equal too.  It is precomputed because
/// `Hash` for an `ErrCode` is one `write_u64`; the trade-off is that the value
/// is unkeyed, so a `HashMap<ErrCode, _>` does not get SipHash's per-map keying
/// (inputs are compiler source and diagnostic strings, not attacker-chosen
/// keys).
/// Wrap `url` in an OSC 8 hyperlink around `visible`.
fn ansi_wrap(url: &str, visible: &str) -> String {
    // "\x1b]8;;" + url + "\x1b\\" + visible + "\x1b]8;;\x1b\\"
    let mut out = String::with_capacity(5 + url.len() + 2 + visible.len() + 7);
    out.push_str("\x1b]8;;");
    out.push_str(url);
    out.push_str("\x1b\\");
    out.push_str(visible);
    out.push_str("\x1b]8;;\x1b\\");
    out
}

/// Payload for a registry entry.
///
/// `resolve`/`lookup` only ever return `CODE_TABLE` entries, so the index from
/// `CODE_INDEX` is valid by construction; the assertion pins the invariant that
/// `CODE_INNER` and `CODE_TABLE` are parallel (both are generated in document
/// order).
#[inline]
fn inner_for_entry(e: &'static CodeEntry) -> &'static ErrCodeInner {
    let idx = CODE_INDEX[(e.slot as usize) & (CODE_INDEX_SLOTS - 1)];
    debug_assert_eq!(
        CODE_TABLE[idx as usize].code, e.code,
        "CODE_INDEX and CODE_INNER disagree with the entry"
    );
    &CODE_INNER[idx as usize]
}

/// Cold constructor for an unresolved code.
///
/// Ownership stays explicit: the spelling lives in a reference-counted payload
/// that is freed with the last diagnostic holding it, and the memoized sanitize
/// plus the cached canonical resolution ride along.  A process-global `'static`
/// table is *not* used — reaching `&'static` from an owned payload requires
/// leaking (or lifetime erasure), and unbounded permanent growth is exactly the
/// ownership shortcut this repository forbids.
#[cold]
#[inline(never)]
fn other_repr(raw: CompactString, canonical: Option<&'static CodeEntry>) -> ErrCode {
    let canonical_str = canonical.map_or(raw.as_str(), |e| e.code);
    ErrCode(ErrCodeRepr::Other(std::sync::Arc::new(OtherCode {
        hash: code_hash(canonical_str),
        raw,
        canonical,
        sanitized: std::sync::OnceLock::new(),
    })))
}

impl std::hash::Hash for ErrCode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // One `write_u64` for every code.  An interned payload precomputes the
        // same value (`OtherCode::hash`); a registry code forks over four bytes.
        // `Hash` needs only "equal values hash equally", which holds because
        // both paths derive the value from `canonical()` alone.
        let h = match &self.0 {
            ErrCodeRepr::Known(e) => e.hash,
            ErrCodeRepr::Other(o) => o.hash,
        };
        state.write_u64(h);
    }
}

/// Error returned when an unknown error code string is used.
#[derive(Debug, Clone)]
pub struct UnknownCode<'a>(pub Cow<'a, str>);

/// Fallback metadata for error codes not present in [`CODE_TABLE`].
///
/// Its `slot` is [`u16::MAX`], a sentinel that must **never** be fed to a
/// slot-keyed cache: `cached_slot_string` masks with `SLOT_CACHE_SLOTS - 1`, so
/// `u16::MAX` would silently land on slot 2047 and poison it for a real code.
/// `ErrCode::slot` therefore reads `OtherCode::canonical` and never this entry;
/// keep it that way (the `debug_assert` in `cached_slot_string` tripwires it).
pub(crate) static UNKNOWN_CODE_ENTRY: CodeEntry = CodeEntry {
    code: "",
    slot: u16::MAX,
    title: "unknown error",
    category: ErrorCategory::Generic,
    related: &[],
    siblings: &[],
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
    /// A registry code is a `'static` pointer with no allocation; an unknown
    /// code costs exactly one `Arc<OtherCode>` allocation.  (The enum itself
    /// stays two words either way — the earlier inline `CompactString` design
    /// made EVERY code pay that size.)
    ///
    /// Prefer [`Self::try_new`] when the input comes from an external source
    /// (user input, CLI arguments, etc.).
    pub fn new(code: impl AsRef<str>) -> Self {
        let code = code.as_ref();
        // One choke point ([`resolve`]): exact lookup first, then the
        // allocation-free case-insensitive retry (`e030` → `E030`) that
        // `suggest_code` also relies on.  Inlining the pair here (as this
        // function used to) is exactly the drift E5 consolidated away — a
        // future change to the retry rule must not need two edits.
        if let Some(e) = resolve(code) {
            return ErrCode(ErrCodeRepr::Known(inner_for_entry(e)));
        }
        // Unknown: stored verbatim — never invent case for arbitrary
        // strings that are not error codes.  `resolve` already failed, so the
        // cached canonical resolution is `None`.
        other_repr(CompactString::from(code), None)
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
            Some(e) => Ok(ErrCode(ErrCodeRepr::Known(inner_for_entry(e)))),
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
        // One table read: `resolve` covers both the exact and the uppercased
        // spelling, and comparing the entry's canonical `code` against `raw`
        // decides `Known` vs `Other` without a second lookup.  Exact canonical
        // spelling is observationally identical to a table hit, so it takes
        // `Known` (no `Arc<OtherCode>` allocation on the per-coded-diagnostic
        // path); every other spelling takes `Other` to preserve the verbatim
        // echo in `code()` and the URL link text.
        match resolve(&raw) {
            Some(e) if e.code == raw.as_str() => ErrCode(ErrCodeRepr::Known(inner_for_entry(e))),
            entry => other_repr(raw, entry),
        }
    }

    /// The code string, e.g. "E030" or "W113".
    ///
    /// For [`Unknown`](Self::unknown) codes this is the stored spelling
    /// verbatim (untrusted echo semantics); use [`canonical`](Self::canonical)
    /// for identity.
    #[must_use]
    pub fn code(&self) -> &str {
        match &self.0 {
            ErrCodeRepr::Known(e) => e.display,
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
            ErrCodeRepr::Known(e) => e.canonical,
            // A junk spelling is its own canonical form, which keeps distinct
            // junk codes distinct (`unknown("junk1") != unknown("junk2")`).
            ErrCodeRepr::Other(o) => o.canonical.map_or(o.raw.as_str(), |e| e.code),
        }
    }

    /// Resolve this error code to its table entry, if present.
    ///
    /// Callers that need multiple fields (`title`, `category`, `explain`)
    /// should use this once and derive each from the returned entry, avoiding
    /// repeated lookups (a shape check + one array read on `CODE_INDEX`).
    pub fn entry(&self) -> Option<&'static CodeEntry> {
        match &self.0 {
            ErrCodeRepr::Known(e) => e.entry,
            // An `unknown` spelling that names a real entry still resolves its
            // metadata, matching the case-insensitive resolution in
            // `new`/`try_new`.
            ErrCodeRepr::Other(o) => o.canonical,
        }
    }

    /// Return the resolved entry or the static unknown-code fallback.
    ///
    /// Equivalent to a single `lookup` call followed by `.unwrap_or(...)`,
    /// so callers can gather `title` / `category` / `explain` in one lookup.
    pub(crate) fn entry_or_unknown(&self) -> &'static CodeEntry {
        self.entry().unwrap_or(&UNKNOWN_CODE_ENTRY)
    }

    /// Short title, e.g. "type mismatch" or "duplicate definition".
    #[must_use]
    pub fn title(&self) -> &'static str {
        self.entry_or_unknown().title
    }

    /// The error category, e.g. `ErrorCategory::Type`.
    #[must_use]
    pub fn category(&self) -> ErrorCategory {
        self.entry_or_unknown().category
    }

    /// Full explanation text, displayed by `ponent --explain E030`.
    pub fn explain(&self) -> &'static str {
        self.entry_or_unknown().explain
    }

    /// The direct-address slot for this code, if its spelling is cacheable.
    ///
    /// Known entries carry their slot; an `Other` spelling gets one only when
    /// `code_slot` accepts the shape (`^[EW]\d{3}$`) or when it canonicalizes to
    /// a registered entry.  `None` means "bypass the slot cache" — the URL path
    /// then rebuilds on every call, which is correct and rare.
    fn slot(&self) -> Option<u16> {
        self.entry()
            .map(|e| e.slot)
            .or_else(|| code_slot(self.code()).map(|s| s as u16))
    }

    /// The diagnostic URL for the `--explain` feature.
    ///
    /// [`explain_url_with_port`] percent-encodes the code as a single path
    /// segment, so an unknown code holding arbitrary bytes (see [`Self::unknown`])
    /// cannot escape the segment or the enclosing link.
    ///
    /// **Memoization:** once the explain port is fixed ([`set_explain_port`] is
    /// set-once), the result depends only on the code string.  Codes matching the
    /// registry's `^[EW]\d{3}$` shape are cached per [`code_slot`] in a
    /// [`SlotCache`] — a fixed `SLOT_CACHE_SLOTS`-entry array of
    /// `OnceLock<Box<str>>` *inside the static*, so a hit is one indexed
    /// `Acquire` load, a null check, and a fat-pointer copy; the table's BSS is
    /// only committed page by page as slots are actually used (there is no
    /// eagerly built hash map to "avoid").  The cached form is *borrowed*, not
    /// refcounted: no `Arc::clone`, no atomic RMW, no per-hit allocation.  A
    /// batch of *N* diagnostics sharing a code therefore pays the build once and
    /// a cheap copy per hit.
    ///
    /// Unknown codes outside the `^[EW]\d{3}$` shape bypass the cache; a
    /// shape-valid unregistered code (e.g. `E999`) still occupies its slot and is
    /// cached, so repeated emissions do not rebuild the URL.
    ///
    /// Two such arrays exist — `(url, url_ansi)` for the port-*set* quadrant.
    /// The port-unset quadrants are not caches: registry codes read a build-time
    /// constant and everything else builds uncached, which is why a later
    /// `set_explain_port` can never observe a stale doc-URL host.
    #[must_use]
    pub fn url(&self) -> std::borrow::Cow<'static, str> {
        let slot = self.slot();
        self.url_for_slot(slot)
    }

    fn url_for_slot(&self, slot: Option<u16>) -> std::borrow::Cow<'static, str> {
        self.url_for_slot_with_port(slot, explain_port())
    }

    fn url_for_slot_with_port(
        &self,
        slot: Option<u16>,
        port: Option<u16>,
    ) -> std::borrow::Cow<'static, str> {
        // Canonical spelling, not the verbatim echo: `unknown("e030")` resolves
        // to the E030 entry (title/category/explain all do), so its URL must be
        // the registered `/E030` — not a `/e030` the doc site 404s — and it
        // must hit the per-slot cache like any other spelling of that code.
        let Some(port) = port else {
            // No explain server: a registry code's documentation URL is a
            // build-time constant, and a spelling that resolves to one borrows
            // that same constant.
            //
            // A genuinely unregistered *shape-valid* spelling (`E999`) used to
            // rebuild its doc URL — percent-encode scan plus a `String` — on
            // every call, on the per-diagnostic hyperlink path.  The old note
            // justified deleting the slot cache here with "there is nothing to
            // invalidate when the port is set later"; that is true of the
            // localhost URL, but the *doc* URL is a pure function of the
            // canonical code and contains no port, so there is nothing to
            // invalidate at all.  `code_slot` is injective over its domain
            // (pinned by `code_slot_is_a_bijection_onto_the_direct_address_range`),
            // so the slot is a sound key.
            return match self.entry() {
                Some(e) => std::borrow::Cow::Borrowed(inner_for_entry(e).doc_url),
                None => match self.slot() {
                    None => std::borrow::Cow::Owned(explain_url_with_port(self.canonical(), None)),
                    Some(slot) => {
                        static DOC_URL_CACHE: SlotCache = new_slot_cache();
                        std::borrow::Cow::Borrowed(cached_slot_string(&DOC_URL_CACHE, slot, || {
                            explain_url_with_port(self.canonical(), None)
                        }))
                    }
                },
            };
        };
        let Some(slot) = slot else {
            return std::borrow::Cow::Owned(explain_url_with_port(self.canonical(), Some(port)));
        };
        // The localhost form embeds the port, which is only known at runtime and
        // fixed once (`set_explain_port`), so it keeps its slot cache.
        static URL_CACHE: SlotCache = new_slot_cache();
        std::borrow::Cow::Borrowed(cached_slot_string(&URL_CACHE, slot, || {
            explain_url_with_port(self.canonical(), Some(port))
        }))
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
    /// fixed-array [`SlotCache`] described on [`url`](Self::url) — again one
    /// cache per `(port set, port unset)` quadrant, so the doc-URL hyperlink is
    /// cached as well.  A batch of *N* diagnostics sharing a code builds the
    /// OSC 8 hyperlink once and clones thereafter.
    ///
    /// Two kinds of code bypass the cache and rebuild every call:
    /// * codes with no slot at all (arbitrary bytes outside `^[EW]\d{3}$`), and
    /// * a *non-canonical* spelling of a registered code (`unknown("e030")`,
    ///   see [`Self::ansi_cacheable`]) — its link text is the verbatim spelling,
    ///   so caching it under slot 30 would make a later canonical `E030` render
    ///   the wrong visible text.
    fn ansi_cacheable(&self) -> bool {
        // The cached string embeds the *visible* spelling, so a non-canonical
        // spelling (`unknown("e030")`) must not populate the canonical slot —
        // that would make a later `E030` render the wrong link text.
        self.0.is_canonical_spelling()
    }

    #[must_use]
    pub fn url_ansi(&self) -> std::borrow::Cow<'static, str> {
        let Some(port) = explain_port() else {
            // Same as `url`, with one extra condition: the wrapper embeds the
            // *visible* spelling, so only a canonical spelling may borrow the
            // build-time constant.
            return match self.entry() {
                Some(e) if self.0.is_canonical_spelling() => {
                    std::borrow::Cow::Borrowed(inner_for_entry(e).doc_url_ansi)
                }
                _ => match self.slot() {
                    // Shape-valid but unregistered (`E999`): the wrapper embeds
                    // only the canonical spelling and the port-free doc URL, so
                    // the slot is a sound key with nothing to invalidate.
                    Some(slot) if self.ansi_cacheable() => {
                        static DOC_URL_ANSI_CACHE: SlotCache = new_slot_cache();
                        std::borrow::Cow::Borrowed(cached_slot_string(
                            &DOC_URL_ANSI_CACHE,
                            slot,
                            || self.url_ansi_build(Some(slot), None),
                        ))
                    }
                    _ => std::borrow::Cow::Owned(self.url_ansi_build(None, None)),
                },
            };
        };
        let Some(slot) = self.slot() else {
            // No registry slot (a malformed spelling): build the wrapper, whose
            // sanitize is memoized in the payload.
            return std::borrow::Cow::Owned(self.url_ansi_build(None, Some(port)));
        };
        // The cached string embeds the *visible* spelling, so only a canonical
        // spelling may populate the slot.
        if !self.ansi_cacheable() {
            return std::borrow::Cow::Owned(self.url_ansi_build(Some(slot), Some(port)));
        }
        static URL_ANSI_CACHE: SlotCache = new_slot_cache();
        std::borrow::Cow::Borrowed(cached_slot_string(&URL_ANSI_CACHE, slot, || {
            self.url_ansi_build(Some(slot), Some(port))
        }))
    }

    /// Uncached core of [`url_ansi`](Self::url_ansi): build the OSC 8 hyperlink
    /// from the (uncached) [`url`](Self::url) and the sanitized code.  Separated
    /// so the cache path can `set` without holding a borrow across the build.
    fn url_ansi_build(&self, slot: Option<u16>, port: Option<u16>) -> String {
        // E6: build the URL exactly once. The old `format_compact!(self.url())`
        // allocated for unknown codes (no registry slot → no cache) only to
        // format it again; assemble straight into the result.  The capacity is
        // exact because this path runs per call for non-cacheable spellings, so
        // every growth realloc was pure overhead.
        let url: std::borrow::Cow<'static, str> = if slot.is_some() {
            self.url_for_slot_with_port(slot, port)
        } else {
            std::borrow::Cow::Owned(explain_url_with_port(self.canonical(), port))
        };
        let visible: std::borrow::Cow<'_, str> = if self.is_known() {
            std::borrow::Cow::Borrowed(self.code())
        } else {
            super::sanitize_for_terminal(self.code())
        };
        ansi_wrap(&url, &visible)
    }
}

/// A single entry in the error code lookup table.
pub(crate) struct CodeEntry {
    pub(crate) code: &'static str,
    pub(crate) slot: u16,
    pub(crate) title: &'static str,
    pub(crate) category: ErrorCategory,
    pub(crate) related: &'static [&'static str],
    pub(crate) siblings: &'static [&'static str],
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
        assert!(
            CODE_TABLE[i].slot as usize
                == code_slot(CODE_TABLE[i].code).expect("validated code shape")
        );
        i += 1;
    }
};

/// Number of representable direct-address slots (`^[EW]\d{3}$` → 2 × 1000).
/// Callers that want a lock-free, per-code table (the explain server's page
/// cache) size their array with this; the module's own tables are padded to the
/// next power of two so they can be indexed with a mask.
pub(crate) const DIRECT_ADDRESS_SLOTS: usize = 2 * 1000;

// The mask-based indexing above is sound only if every table is large enough
// and a power of two, and if `code_slot` cannot exceed the real slot space.
const _: () = assert!(CODE_INDEX_SLOTS.is_power_of_two());
const _: () = assert!(SLOT_CACHE_SLOTS.is_power_of_two());
const _: () = assert!(DIRECT_ADDRESS_SLOTS <= CODE_INDEX_SLOTS);
const _: () = assert!(DIRECT_ADDRESS_SLOTS <= SLOT_CACHE_SLOTS);
// Stronger form of the same invariant: pin the *arithmetic* that produces
// slots, not only the hand-maintained count.  `code_slot` is `const`, so the
// largest code it can encode (`W999`) must stay inside both tables; this is
// what the `& (… - 1)` masking relies on.
const _: () = assert!(code_slot("W999").unwrap() + 1 <= CODE_INDEX_SLOTS);
const _: () = assert!(code_slot("W999").unwrap() + 1 <= SLOT_CACHE_SLOTS);

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

/// Case-insensitive registry resolution.
///
/// **`resolve` ≡ [`lookup_uppercased`]** — the older
/// `lookup(code).or_else(|| lookup_uppercased(code))` did redundant work.  Proof:
/// both require a four-byte code (`code_slot`/`code_slot_parts` return `None`
/// otherwise), and `lookup_uppercased` computes the slot from
/// `(b[0].to_ascii_uppercase(), b[1..])`.  For a code `lookup` accepts, `b[0]` is
/// already `E` or `W` (the table's only prefixes), on which `to_ascii_uppercase`
/// is the identity, so both compute the same slot.  For any other `b[0]`,
/// `lookup` is `None` while `lookup_uppercased` may still resolve `e`/`w` — a
/// strict superset.  Hence `lookup ⊆ lookup_uppercased` and the exact-lookup
/// branch was dead weight on every construction path
/// (`new`/`try_new`/`unknown`/`related_codes_of`/`category_siblings_of`).
///
/// [`lookup`] itself is kept: `direct_address_index_matches_table` and the
/// registry tests use it as the exact-case reference.
#[inline]
pub(crate) fn resolve(code: &str) -> Option<&'static CodeEntry> {
    lookup_uppercased(code)
}

/// Look up a code string in the table.
#[inline]
pub(crate) fn lookup(code: &str) -> Option<&'static CodeEntry> {
    let slot = code_slot(code)?;
    let idx = CODE_INDEX[slot & (CODE_INDEX_SLOTS - 1)];
    if idx == u16::MAX {
        return None;
    }
    // Direct index, not `get`: `build_code_index` only ever stores indices of
    // real entries — every other value is the `u16::MAX` "absent" sentinel
    // handled above — so `get`'s bounds check could never fail, and its `None`
    // arm shipped a branch on the hottest lookup (`ErrCode::new`'s path).  A
    // corrupt registry fails the build through the table/index asserts.
    Some(&CODE_TABLE[idx as usize])
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
    // Compute the slot directly: `to_ascii_uppercase` only affects byte 0
    // (bytes 1..3 are digits by the table's shape assertion), so this avoids
    // the stack buffer, the `from_utf8` hop, and the full re-parse `lookup`
    // would do.  No `&str` is constructed, so no UTF-8 validation is needed.
    let slot = code_slot_parts(b[0].to_ascii_uppercase(), b[1], b[2], b[3])?;
    let idx = CODE_INDEX[slot & (CODE_INDEX_SLOTS - 1)];
    if idx == u16::MAX {
        return None;
    }
    // Direct index, mirroring `lookup`: `idx` comes from the table's own
    // build-time index, so `get`'s `None` arm could never fire — it was a
    // branch on the hottest construction path (`new`/`try_new`/`unknown` all
    // funnel through here).  The build-time shape assertions are the proof.
    Some(&CODE_TABLE[idx as usize])
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
        // Exactly eight categories and a dense `index()`, so a fixed array beats
        // hashing every entry's discriminant and then doing eight
        // `remove`-with-`expect` lookups.  One pass, no hashing, no panics.
        let mut buckets: [Vec<&'static CodeEntry>; ErrorCategory::COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut present = [false; ErrorCategory::COUNT];
        let mut order: Vec<ErrorCategory> = Vec::with_capacity(ErrorCategory::COUNT);
        for e in CODE_TABLE {
            let i = e.category.index();
            if !present[i] {
                present[i] = true;
                order.push(e.category);
            }
            buckets[i].push(e);
        }
        let mut out: Vec<(ErrorCategory, Vec<&'static CodeEntry>)> =
            Vec::with_capacity(order.len());
        for cat in order {
            out.push((cat, std::mem::take(&mut buckets[cat.index()])));
        }
        out.into_boxed_slice()
    })
}

pub fn related_codes_of(code: &str) -> &'static [&'static str] {
    resolve(code).map(|entry| entry.related).unwrap_or(&[])
}

pub fn category_siblings_of(code: &str, max: usize) -> &'static [&'static str] {
    let Some(entry) = resolve(code) else {
        return &[];
    };
    &entry.siblings[..max.min(entry.siblings.len())]
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    /// The user-visible contract of [`set_explain_port`]: a render made before
    /// the port is known keeps the documentation host, a render after it yields
    /// the localhost link, and the country does not stick.  Driven through the
    /// test-only override because the real global is set-once.
    #[test]
    fn url_flips_between_doc_and_localhost_across_the_override() {
        let code = ErrCode::try_new("E030").expect("E030 is registered");
        let doc = with_explain_port(None, || code.url().into_owned());
        assert!(doc.starts_with("https://"), "doc URL expected, got {doc}");
        let local = with_explain_port(Some(41234), || code.url().into_owned());
        // Only the host is asserted: the slot caches are keyed per slot within
        // the port-*set* quadrant, and this test binary's process-global port may
        // already be 0 from `set_explain_port_reports_second_set`, so which port
        // an already-built entry embeds is not this contract (the doc says an
        // entry built under one port is not invalidated by a later set).
        assert!(
            local.contains("127.0.0.1:"),
            "localhost URL expected, got {local}"
        );
        let back = with_explain_port(None, || code.url().into_owned());
        assert_eq!(back, doc, "the override must not be sticky");
        // The ANSI wrapper follows the same flip.
        let ansi_local = with_explain_port(Some(41234), || code.url_ansi().into_owned());
        assert!(ansi_local.contains("127.0.0.1:"), "{ansi_local}");
    }

    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    #[test]
    fn skip_char_literal_handles_hex_escapes() {
        assert_eq!(skip_char_literal(b"'\\x41'foo", 0), 6);
        assert_eq!(skip_char_literal(b"'\\x28'", 0), 6);
    }

    /// Lowercase input must resolve like `ErrCode::new`/`entry` do: the
    /// sibling list is computed from the canonical entry, not returned empty
    /// because `lookup` demands `^[EW]\d{3}$`.
    #[test]
    fn related_codes_accepts_lowercase_code() {
        assert_eq!(
            related_codes_of("e030"),
            related_codes_of("E030"),
            "lowercase input must resolve through the same path"
        );
    }

    #[test]
    fn category_siblings_accepts_lowercase_code() {
        let upper = category_siblings_of("E030", 4);
        let lower = category_siblings_of("e030", 4);
        assert!(!upper.is_empty(), "E030 has category siblings");
        assert_eq!(
            lower, upper,
            "lowercase input returned a different (empty) list"
        );
        // Both spellings resolve through the same path; `category_siblings_of`
        // has no cache, these assertions only pin resolution equality.
        assert_eq!(category_siblings_of("E030", 2).len(), 2);
        assert_eq!(category_siblings_of("e030", 2).len(), 2);
    }

    /// `ErrorCategory::index()` must stay a dense, unique `0..COUNT` mapping:
    /// `code_entries_by_category` sizes a fixed array with it, and
    /// `CODE_INDEX`'s build-time assertions rely on the same discipline.
    #[test]
    fn category_index_is_dense_and_unique() {
        let all = [
            ErrorCategory::Parse,
            ErrorCategory::Resolution,
            ErrorCategory::Type,
            ErrorCategory::Contract,
            ErrorCategory::Trait,
            ErrorCategory::Inference,
            ErrorCategory::Internal,
            ErrorCategory::Generic,
        ];
        let mut seen = [false; ErrorCategory::COUNT];
        for cat in all {
            let i = cat.index();
            assert!(i < ErrorCategory::COUNT, "{cat:?} index out of range");
            assert!(!seen[i], "{cat:?} duplicates index {i}");
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "index() is not dense");
        // Grouping keeps first-appearance order and loses nothing.
        let grouped = code_entries_by_category();
        let total: usize = grouped.iter().map(|(_, v)| v.len()).sum();
        assert_eq!(total, CODE_TABLE.len());
        assert!(grouped.iter().all(|(_, v)| !v.is_empty()));
    }

    /// E3: `EXPLAIN_PORT` is process-global, and Cargo runs tests in parallel
    /// threads within one binary. Serialize every test that mutates it or reads a
    /// URL derived from it, so a future explicit host assertion cannot flake on
    /// test ordering.
    static PORT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A second `set_explain_port` must be observable rather than silently
    /// discarded.  `EXPLAIN_PORT` is a process-global `OnceLock`, so this test
    /// deliberately owns it (no other test sets it) — which is also why the
    /// documented "port 0 is accepted as-is" and "running ⇔ port set" contracts
    /// are pinned here rather than in a second owner test.
    /// The quadrant split guarantees no *stale* entry is consulted across a
    /// port change, but it cannot invalidate one already built: the same
    /// diagnostic rendered before and after `set_explain_port` shows two
    /// different link targets.  That is why setting the port first is a
    /// documented contract (`set_explain_port`) rather than a hint; this pins
    /// the two forms so a future change to either cannot hide the gap.
    #[test]
    fn url_flips_between_doc_and_localhost_across_the_port() {
        let code = ErrCode::new("E030");
        let doc = code.url_for_slot_with_port(None, None);
        let local = code.url_for_slot_with_port(None, Some(41234));
        assert!(
            doc.starts_with("http"),
            "no-port form should be the documentation URL: {doc}"
        );
        assert!(
            local.contains("127.0.0.1:41234"),
            "port form should be the localhost explain URL: {local}"
        );
        assert_ne!(doc, local, "the two quadrants must differ");
    }

    #[test]
    fn set_explain_port_reports_second_set() {
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Port 0 is accepted verbatim: validation belongs to whoever binds.
        assert!(set_explain_port(0).is_ok());
        assert!(
            is_explain_server_running(),
            "a set port must report the server as running"
        );
        assert_eq!(set_explain_port(54321), Err(0));
        assert!(is_explain_server_running(), "the first port keeps winning");
    }

    /// `unknown` is the untrusted-input path; an ESC in the code must reach
    /// neither the URI nor the visible link text.
    ///
    /// The only ESC bytes in the result are the four that frame the OSC 8
    /// sequence (`\x1b]8;;` … `\x1b\\` … `\x1b]8;;` … `\x1b\\`).  This holds
    /// regardless of whether `EXPLAIN_PORT` was set by another test.
    #[test]
    fn url_ansi_keeps_visible_spelling_per_instance() {
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let lower = ErrCode::unknown("e030");
        let upper = ErrCode::new("E030");
        let _ = lower.url_ansi();
        let upper_out = upper.url_ansi();
        assert!(
            upper_out.ends_with("E030\x1b]8;;\x1b\\"),
            "canonical link text poisoned: {upper_out:?}"
        );
        let lower_out = lower.url_ansi();
        assert!(
            lower_out.ends_with("e030\x1b]8;;\x1b\\"),
            "verbatim link text lost: {lower_out:?}"
        );
        // The non-cacheable path still emits the *canonical* URI: only the
        // visible text echoes the caller's spelling.  Assert on the segment so
        // the check holds whether or not a port was set by another test.
        assert!(
            !lower_out.contains("/e030"),
            "canonical URI segment lost: {lower_out:?}"
        );
        assert!(
            lower_out.contains("/E030"),
            "URI must point at the registered code: {lower_out:?}"
        );
    }

    /// A degenerate unknown code must stay defined rather than panic: the doc
    /// URL is assembled from the (empty) encoded segment, so the current shape
    /// is `…/error_codes/.html`.  Pinned so a future "reject empty" change is a
    /// visible decision instead of an accident.
    #[test]
    fn unknown_empty_code_has_a_defined_doc_url() {
        assert_eq!(
            explain_url_with_port("", None),
            "https://doc.posita-lang.org/error_codes/.html"
        );
        // No panic on the display/claim paths either.
        let ec = ErrCode::unknown("");
        assert_eq!(format!("{ec}"), "");
        assert!(!ec.is_known());
    }

    #[test]
    fn display_known_code_is_verbatim() {
        assert_eq!(format!("{}", ErrCode::new("E030")), "E030");
        assert_eq!(format!("{}", ErrCode::new("e030")), "E030");
    }

    #[test]
    fn display_sanitizes_untrusted_code() {
        let ec = ErrCode::unknown("E0\x1b]8;;evil\x07");
        let out = format!("{ec}");
        assert!(!out.contains('\x1b'), "ESC leaked: {out:?}");
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert!(out.contains('\u{FFFD}'), "controls not replaced: {out:?}");
    }

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

    /// `category_siblings_of` is pure and truncates to `max`; repeated calls
    /// with different maxima must not share state.
    #[test]
    fn category_siblings_is_pure_and_truncates() {
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
        // `Display` follows `code()`: the unresolved form echoes its spelling
        // verbatim, the resolved form shows the canonical table spelling.  The
        // asymmetry is the `code()` contract, and it used to be pinned only
        // indirectly through the ANSI URL path.
        assert_eq!(format!("{}", ErrCode::new("E030")), "E030");
        assert_eq!(format!("{}", ErrCode::unknown("e030")), "e030");

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
        // RFC 3986 §2.3 unreserved set: ALPHA / DIGIT / "-" / "." / "_" / "~"
        // pass through untouched.  The boundary the function exists to enforce.
        let unreserved: String = ('a'..='z')
            .chain('A'..='Z')
            .chain('0'..='9')
            .chain(['-', '.', '_', '~'])
            .collect();
        assert_eq!(encode_url_path_segment(&unreserved), unreserved);
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
    /// Width of `ErrCode` is a contract, not a detail: every `Diagnostic`
    /// carries an `Option` of it, so this bounds what a code costs on the hot
    /// struct.
    ///
    /// It is **two** words: a registry code would fit in one (`&'static
    /// CODE_INNER`), but an unknown spelling cannot be `&'static` without
    /// leaking or erasure — both forbidden here — so it is a reference-counted
    /// payload and the discriminant is unavoidable.  The per-code cost is
    /// therefore 16 bytes; what the payload buys is `Hash` in one `write_u64`
    /// and an `Eq` pointer fast path for *both* forms.
    #[test]
    fn err_code_stays_two_words() {
        assert_eq!(std::mem::size_of::<super::ErrCode>(), 16);
        assert_eq!(std::mem::size_of::<Option<super::ErrCode>>(), 16);
    }

    /// The generated payloads and the runtime interner must agree on the hash
    /// function, or `HashSet<ErrCode>` would split one code into two buckets and
    /// violate `Eq`/`Hash` consistency.
    #[test]
    fn code_inner_agrees_with_the_runtime_payload() {
        assert_eq!(super::CODE_INNER.len(), super::CODE_TABLE.len());
        for (i, inner) in super::CODE_INNER.iter().enumerate() {
            let entry = &super::CODE_TABLE[i];
            assert!(inner.known, "{}", entry.code);
            assert_eq!(inner.display, entry.code);
            assert_eq!(inner.canonical, entry.code);
            assert!(std::ptr::eq(
                inner.entry.expect("registry payload resolves"),
                entry
            ));
            assert_eq!(
                inner.hash,
                super::code_hash(inner.canonical),
                "{}",
                entry.code
            );
            // The doc URL and its OSC 8 wrapper are pure functions of the code.
            assert!(
                inner.doc_url.ends_with(&format!("/{}.html", entry.code)),
                "{}",
                inner.doc_url
            );
            assert!(inner.doc_url_ansi.contains(inner.doc_url));
        }
    }

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
    /// Index of the `r` in a raw-string opener at `i` (`r"…"`, `r#"…"#`,
    /// `br"…"`), or `None` when `i` does not start one.
    ///
    /// The identifier-boundary check mirrors [`is_call_site`]: an identifier
    /// ending in `r`/`br` immediately followed by a literal (`foobar"x"`) must
    /// not be swallowed as a raw string, or the embedded literal would vanish
    /// from the scan.  In valid Rust that adjacency is a syntax error, so this
    /// is defence-in-depth for the scanner, not a correctness fix for compiling
    /// code.
    fn raw_string_opener(bytes: &[u8], i: usize) -> Option<usize> {
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                return None;
            }
        }
        match bytes[i] {
            b'r' => Some(i),
            b'b' if bytes.get(i + 1) == Some(&b'r') => Some(i + 1),
            _ => None,
        }
    }

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

    /// Width of the UTF-8 character whose lead byte is `b`.
    ///
    /// The scanners are best-effort on malformed input, so an impossible or
    /// truncated lead byte (continuation, `0xF8..`) advances by one.
    /// Index just past the `*/` that closes the block comment starting at `i`.
    ///
    /// All three scanners (`argument_span`, `code_literals`, `strip_non_code`)
    /// carried their own version of this skip; they agreed by maintenance only,
    /// and the failure mode (comment text read as code) is silent.
    fn skip_block_comment(bytes: &[u8], i: usize) -> usize {
        // Depth-tracked: Rust block comments nest, so an inner `*/` must not
        // resume code scanning inside the outer comment — the failure mode this
        // shared helper exists to make impossible in all three scanners.
        let mut depth = 1usize;
        let mut j = i + 2;
        while j < bytes.len() && depth > 0 {
            if j + 1 < bytes.len() && bytes[j] == b'/' && bytes[j + 1] == b'*' {
                depth += 1;
                j += 2;
            } else if j + 1 < bytes.len() && bytes[j] == b'*' && bytes[j + 1] == b'/' {
                depth -= 1;
                j += 2;
            } else {
                j += 1;
            }
        }
        j
    }

    fn utf8_len_from_lead(b: u8) -> usize {
        match b {
            0x00..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => 1,
        }
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
            } else if bytes[j] == b'x' {
                // `'\xNN'` consumes the introducer plus exactly two hex
                // digits; otherwise the closing quote is mistaken for the
                // start of a lifetime.
                j = (j + 3).min(len);
            } else {
                j += 1;
            }
        } else {
            // Width from the lead byte.  The input is `&str`-backed and therefore
            // valid UTF-8, so `from_utf8(&bytes[j..])` was re-validating the
            // entire remaining file to learn one char's width: on the
            // `literal_codes_are_all_in_code_table` guard (which scans every
            // `.rs`) that made the scanner O(k·n) in char-literal count.
            j = (j + utf8_len_from_lead(bytes[j])).min(len);
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
            if let Some(r) = raw_string_opener(bytes, i)
                && let Some(end) = skip_raw_string(bytes, r)
            {
                i = end;
                continue;
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
                    // `skip_block_comment` returns the index just past the outer
                    // `*/`; step back one so this scanner's trailing `i += 1`
                    // lands exactly there.
                    i = skip_block_comment(bytes, i).saturating_sub(1);
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
    /// `true` when `stripped[start..]` begins a *real* call to the callee.
    ///
    /// The preceding byte must not continue an identifier (`MyErrCode::new`,
    /// `xwith_code_str`), while a path separator stays valid
    /// (`diagnostics::ErrCode::new`).  Known limits, documented rather than
    /// half-handled: `ErrCode ::new` (internal whitespace), an aliased import
    /// (`use ErrCode as EC; EC::new(..)`) and `Self::new(..)` inside
    /// `impl ErrCode` are all missed — tolerating them would need a
    /// whitespace-aware matcher, an alias table and impl-context tracking, and a
    /// miss only weakens the guard, it cannot corrupt a code.
    /// `scanner_keeps_documented_misses_out` pins all three.
    fn is_call_site(stripped: &str, start: usize) -> bool {
        match start.checked_sub(1).map(|i| stripped.as_bytes()[i]) {
            Some(prev) => !(prev.is_ascii_alphanumeric() || prev == b'_'),
            None => true,
        }
    }

    #[test]
    fn scanner_requires_an_identifier_boundary() {
        assert!(is_call_site("ErrCode::new(", 0));
        // `diagnostics::ErrCode::new(` — the preceding `:` is a path separator.
        assert!(is_call_site("diagnostics::ErrCode::new(", 13));
        // A longer identifier ending in the callee is not a call site.
        assert!(!is_call_site("MyErrCode::new(", 2));
        assert!(!is_call_site("xwith_code_str(", 1));
        assert!(!is_call_site("_with_code_str(", 1));
    }

    fn code_literals(arg: &str) -> Vec<&str> {
        let bytes = arg.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            // Raw strings are not code literals and may contain quotes that
            // would desynchronise the scan (E2).
            if let Some(r) = raw_string_opener(bytes, i)
                && let Some(end) = skip_raw_string(bytes, r)
            {
                i = end;
                continue;
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
                    // This arm owns `i` (no trailing increment), so leaving it
                    // directly past the outer `*/` is correct.
                    i = skip_block_comment(bytes, i);
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
    /// A literal opener (`"`, `b"`, `'`, `b'`, `r`, `br`) only starts a literal
    /// when it does not continue an identifier.  `raw_string_opener` already
    /// applies this guard; the plain string/char openers below reuse it so the
    /// three detections cannot drift apart.
    fn opener_boundary_ok(bytes: &[u8], i: usize) -> bool {
        !(i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_'))
    }

    fn strip_non_code(src: &str) -> String {
        let bytes = src.as_bytes();
        let mut out = vec![b' '; bytes.len()];
        let mut i = 0;
        while i < bytes.len() {
            // Raw string: r"...", r#"..."#; byte raw string: br#"..."#.
            //
            // Via the shared helper: `strip_non_code` was the one scanner still
            // carrying its own copy, after `skip_block_comment`'s doc recorded
            // that this exact pattern ("they agreed by maintenance only, and the
            // failure mode is silent") is how the scanners drift.  `None` means
            // "not quote-terminated", which falls through to the other forms.
            if let Some(r) = raw_string_opener(bytes, i)
                && let Some(end) = skip_raw_string(bytes, r)
            {
                i = end;
                continue;
            }

            // String literal: "..." or byte string b"...".  The opener must not
            // continue an identifier (`foo"…"` is not a literal), matching
            // `raw_string_opener`'s guard above.
            let str_quote = if bytes[i] == b'"' {
                opener_boundary_ok(bytes, i).then_some(i)
            } else if bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                opener_boundary_ok(bytes, i).then_some(i + 1)
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
                opener_boundary_ok(bytes, i).then_some(i)
            } else if bytes[i] == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                opener_boundary_ok(bytes, i).then_some(i + 1)
            } else {
                None
            };
            if let Some(q) = ch_quote {
                let mut j = q + 1;
                if j < bytes.len() && bytes[j] == b'\\' {
                    j += 1;
                    if j < bytes.len() {
                        match bytes[j] {
                            b'x' => j = (j + 3).min(bytes.len()),
                            b'u' if j + 1 < bytes.len() && bytes[j + 1] == b'{' => {
                                j += 2;
                                while j < bytes.len() && bytes[j] != b'}' {
                                    j += 1;
                                }
                                j = (j + 1).min(bytes.len());
                            }
                            _ => j += 1,
                        }
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
                // Skip past the OUTER `*/` in one place (`skip_block_comment`),
                // so the three scanners cannot drift apart.
                i = skip_block_comment(bytes, i);
                continue;
            }

            // Ordinary byte.
            out[i] = bytes[i];
            i += 1;
        }
        String::from_utf8(out).expect("stripping preserved UTF-8")
    }

    /// Every literal code passed to `ErrCode::new` / `with_code_str` must be
    /// present in [`CODE_TABLE`].  A missing entry silently degrades to
    /// "unknown error" in both `ErrCode::new` and `with_code_str`; this test is
    /// the sole guard.  Scans the crate source, including conditional
    /// forms like `with_code_str(if … { "E109" } else { "E110" })`.
    ///
    /// Call sites are matched on an identifier boundary ([`is_call_site`]), so a
    /// longer name such as `MyErrCode::new` is not treated as this callee and
    /// cannot demand registration of its literals.
    /// The documented scanner misses (`Self::new`, `ErrCode ::new` with internal
    /// whitespace, an aliased import) must stay misses.  Each only weakens the
    /// guard, and pinning them as intended behaviour keeps a future "helpful"
    /// matcher change visible instead of silently over-requiring registration.
    #[test]
    fn scanner_keeps_documented_misses_out() {
        for snippet in [
            "Self::new(\"E999\")",
            "ErrCode ::new(\"E999\")",
            "use ErrCode as EC;\nEC::new(\"E999\")",
        ] {
            let stripped = strip_non_code(snippet);
            assert!(
                !stripped.contains("ErrCode::new"),
                "{snippet:?} must not look like a call site: {stripped:?}"
            );
        }
        // The forms that *are* matched keep working.
        assert!(is_call_site("ErrCode::new(", 0));
    }

    /// (`ErrCode ::new`) and aliased imports are known misses; both only weaken
    /// the guard.
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
                    let start = from + pos;
                    let after = start + callee.len();
                    // Word boundary: `MyErrCode::new` and `xwith_code_str` are
                    // different identifiers, and treating them as call sites
                    // would demand registration of literals in innocent code.
                    // A preceding `:` stays valid (`diagnostics::ErrCode::new`).
                    if !is_call_site(&stripped, start) {
                        from = after;
                        continue;
                    }
                    // `stripped` preserves byte offsets, so comments between
                    // the callee and `(` are blanked but the `(` position is
                    // still valid in `src`.
                    let tail = &stripped[after..];
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
        let stripped = strip_non_code(src);
        let mut found = Vec::new();
        for callee in ["with_code_str", "ErrCode::new"] {
            let mut from = 0;
            while let Some(pos) = stripped[from..].find(callee) {
                let after = from + pos + callee.len();
                let tail = &stripped[after..];
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
    fn nested_block_comments_do_not_escape_the_scanners() {
        // Rust block comments nest.  Before depth tracking every scanner
        // stopped at the INNER `*/`, so the remainder of the outer comment was
        // rescanned as code — a code-shaped literal inside it could be counted
        // as a call site.
        let stripped = strip_non_code("/* outer /* inner */ still comment */ x");
        assert!(!stripped.contains("still comment"), "{stripped:?}");
        assert!(stripped.contains('x'), "{stripped:?}");
        let lits = code_literals("/* outer /* \"E123\" */ still comment */ \"W456\"");
        assert_eq!(lits, vec!["W456"]);
        // Unterminated nesting must terminate, not loop or panic.
        let _ = strip_non_code("/* outer /* inner");
        let _ = code_literals("/* outer /* \"E123\"");
    }

    /// Measurement, not an assertion: the warm slot-cache hit path that
    /// `write_header` walks once per diagnostic when OSC 8 hyperlinks are on.
    ///
    /// In-crate rather than in `benches/`: `error_code.rs` needs the generated
    /// code table and the Fluent bundles at build time, so it does not survive
    /// a standalone `#[path]` include the way `ascii.rs` does.
    ///
    /// ```text
    /// cargo test --bin ponent slot_cache_hit_path_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement only"]
    fn slot_cache_hit_path_cost() {
        let code = ErrCode::new("E030");
        // Warm the slot so the timed loop measures the hit path only.
        let _ = code.url_ansi();
        let _ = code.url();
        const N: usize = 200_000;
        let ansi = {
            let t0 = std::time::Instant::now();
            let mut acc = 0usize;
            for _ in 0..N {
                acc = acc.wrapping_add(code.url_ansi().len());
            }
            let ns = t0.elapsed().as_secs_f64() * 1e9 / N as f64;
            println!("url_ansi warm hit: {ns:.1} ns/call (acc={acc})");
            ns
        };
        let plain = {
            let t0 = std::time::Instant::now();
            let mut acc = 0usize;
            for _ in 0..N {
                acc = acc.wrapping_add(code.url().len());
            }
            let ns = t0.elapsed().as_secs_f64() * 1e9 / N as f64;
            println!("url warm hit:      {ns:.1} ns/call (acc={acc})");
            ns
        };
        // Sanity floor, not a performance assertion: a hit is a direct-address
        // load plus a fat-pointer borrow copy (no refcount, no allocation), so
        // even an extremely slow host stays far below this.  A regression to a
        // per-call build (microseconds) trips it.
        assert!(
            ansi < 500.0 && plain < 500.0,
            "slot cache hit path is not constant-time (ansi {ansi} ns, plain {plain} ns)"
        );
    }

    #[test]
    fn slot_keyed_caches_require_an_injective_code_slot() {
        // Port-sensitive: this test compares two `url()` results, and
        // `set_explain_port_reports_second_set` flips the process port from
        // another test thread.  Without the lock the two calls can straddle the
        // flip (doc URL vs `127.0.0.1:<port>` URL), which is an intermittent
        // failure, not a stable result.
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // The four slot-keyed caches are sound ONLY because `code_slot` accepts
        // the canonical `^[EW]\d{3}$` shape with a case-sensitive prefix: two
        // distinct shape-valid strings can never share a slot.  If a future
        // change widened the prefix set, `unknown("e030")` would populate slot
        // 30 with visible text `e030` and the canonical `E030` instance would
        // then render the wrong link text.
        assert!(lookup("E030").is_some());
        assert!(
            lookup("e030").is_none(),
            "code_slot must stay case-sensitive for slot-keyed caching"
        );
        assert_eq!(resolve("e030").map(|e| e.code), Some("E030"));
        // An unregistered canonical spelling keys its own slot, and repeated
        // lookups must be stable (the cache stores per-slot, per-spelling).
        let a = ErrCode::unknown("E999").url();
        let b = ErrCode::unknown("E999").url();
        assert_eq!(a, b, "slot-cached URL must be stable");
        // A lowercase spelling is NOT slot-cacheable, so it keeps its own
        // verbatim text in the link, while a lowercase spelling of a registered
        // code resolves to the canonical URI.
        let lower = ErrCode::unknown("e999").url();
        assert!(lower.contains("e999"), "{lower}");
        let canonical = ErrCode::unknown("e030").url();
        assert!(canonical.contains("/E030"), "{canonical}");
    }

    /// The stripped text must preserve byte length: `literal_codes_are_all_in_code_table`
    /// maps match offsets from `strip_non_code`'s output back into the original
    /// source, so a compaction here would silently mis-locate every literal.
    #[test]
    fn strip_non_code_blanks_comments_and_literals() {
        let src = r##"
            // with_code_str("E901")
            /* ErrCode::new("E902") */
            let s = "with_code_str(\"E903\")";
            let r = r#"ErrCode::new("E904")"#;
            let c = '"';
            let h = '\x41';
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

    /// `resolve ≡ lookup_uppercased` is proven in the docs; this executes the
    /// identity over a corpus (every registered code, its lowercase spelling, a
    /// leading-space variant, and junk shapes).
    #[test]
    fn resolve_equals_lookup_uppercased_over_a_corpus() {
        let mut corpus: Vec<String> = Vec::new();
        for e in CODE_TABLE {
            corpus.push(e.code.to_string());
            corpus.push(e.code.to_ascii_lowercase());
            corpus.push(format!(" {}", e.code));
        }
        corpus.extend(
            [
                "", "E", "E03", "E0300", "E030 ", "X030", "E0a0", "eZ9", "W999", "w000", "E999",
            ]
            .map(String::from),
        );
        for s in corpus {
            assert_eq!(
                resolve(&s).map(|e| e.code),
                lookup_uppercased(&s).map(|e| e.code),
                "resolve/lookup_uppercased disagree on {s:?}"
            );
        }
    }

    /// Case-insensitive construction, end to end: every registered code resolves
    /// from its lowercase spelling, while the *verbatim* lookup stays
    /// case-sensitive by design (the cache key is the exact spelling).
    #[test]
    fn every_registered_code_resolves_from_its_lowercase_spelling() {
        for e in CODE_TABLE {
            let lower = e.code.to_ascii_lowercase();
            assert_eq!(
                resolve(&lower).map(|r| r.code),
                Some(e.code),
                "lowercase spelling {lower:?} must resolve"
            );
            assert!(
                lookup(&lower).is_none(),
                "verbatim lookup must stay case-sensitive for {lower:?}"
            );
        }
    }

    /// `code_slot` is a bijection from the well-shaped code space onto
    /// `0..DIRECT_ADDRESS_SLOTS`.  Cache soundness rests on this (an injective
    /// key is what makes a slot's cached URL correct), and until now only three
    /// sample points pinned it.
    #[test]
    fn code_slot_is_a_bijection_onto_the_direct_address_range() {
        assert_eq!(
            DIRECT_ADDRESS_SLOTS,
            2 * 1000,
            "the slot space is exactly E/W × 000..=999"
        );
        let mut seen = vec![false; DIRECT_ADDRESS_SLOTS];
        let mut count = 0usize;
        for prefix in ['E', 'W'] {
            for digits in 0..1000u32 {
                let code = format!("{prefix}{digits:03}");
                let slot = code_slot(&code).unwrap_or_else(|| panic!("{code} has no slot"));
                assert!(slot < DIRECT_ADDRESS_SLOTS, "{code} -> {slot} out of range");
                assert!(!seen[slot], "slot {slot} reused by {code}");
                seen[slot] = true;
                count += 1;
                // A lowercase spelling must not occupy a slot: the cache key is
                // the exact spelling, and case resolution happens in `resolve`.
                let lower = code.to_ascii_lowercase();
                assert!(
                    code_slot(&lower).is_none(),
                    "{lower:?} must not be slot-cacheable"
                );
            }
        }
        assert_eq!(count, DIRECT_ADDRESS_SLOTS, "slot space not saturated");
        assert!(seen.iter().all(|&s| s), "some slot was never produced");
    }
}
