//! Registry codes, untrusted spellings, and the two rendered URL forms.
//!
//! ## Rejected / measurement-gated designs
//!
//! Recorded with their blockers so they are not re-derived (the same ledger style
//! `render.rs` and the emitter use):
//!
//! * **SWAR `encode_url_path_segment`.** Rejected unless a profile asks. The
//!   function is memoized per shape-valid spelling (`OtherCode::encoded`) and capped
//!   at `MAX_CODE_CHARS` for junk (≤512 B), so it is sub-µs end-to-end; a multi-lane
//!   classifier would also need the per-lane `(b − 0x30) | (0x39 − b)` form, because a
//!   single `0x30303030`-subtract comparison is wrong across lanes (cross-lane
//!   borrows).
//! * **`CODE_INDEX` as `Option<NonZeroU32>` indices instead of pointers.** Would save
//!   8 KB of `.data.rel.ro` and ~2,000 relocations, but adds a `lea` plus an index
//!   read on *every* construction path — the opposite trade from the change that put
//!   the pointer there (documented on the item). Measure binary size and relocations
//!   before reopening.
//! * **`AtomicPtr<Box<str>>` slot caches for the four `OnceLock` tables.** 24 B → 8 B
//!   per slot (128 KB of *virtual* BSS), but re-implements `OnceLock`'s
//!   park-the-loser protocol to save pages that are lazily committed anyway, and the
//!   hit path is unchanged. Rejected: pay complexity only where the hit path changes.
//! * **Precomputing doc URLs for every shape-valid slot into the generated table.**
//!   ~100–120 KB of *resident* `.rodata` for ~1,800 slots that mostly never render;
//!   the lazy per-slot cache commits page by page and is strictly better.
//! * **Tagged-pointer `ErrCode` (8 B, `NonZero<usize>` with a bit-0 tag).** The one
//!   genuinely large structural win (`ErrCode` 16 → 8 B, `Option<ErrCode>` 16 → 8 B,
//!   and the `Option` niche contract stops mattering), but it is `unsafe` pointer
//!   tagging and raw-pointer reconstruction for a struct-layout win on a path measured
//!   at single-digit nanoseconds. Revisit only if `Diagnostic` size ever appears in a
//!   profile, and only if the crate relaxes its no-unsafe rule.
//! * **Single-pass scanner in `literal_codes_are_all_in_code_table`.** Test-only, ~3×
//!   on that test; listed for completeness, not worth the rewrite on its own.
//! * **One `parts()` helper fusing the three enum dispatches** (`url_ansi`'s
//!   `ansi_cacheable` + `slot`, `url_for_slot_with_port`'s `slot` + `inner` +
//!   `canonical`, `visible_text`'s match).  Would save ~2-4 predictable branches per
//!   rendered diagnostic, but it is a pure refactor of correct code on the hot path;
//!   the standing risk ("three call sites of one operation drift") is already handled by
//!   the docs at each site.  Not done here because it needs its own before/after
//!   measurement to justify touching the header path, not because it is hard.
//! * **Streaming `write_url`** (a writer core so an uncacheable spelling stops
//!   allocating a `String` per render).  Real win, moderate refactor: it also
//!   consolidates the OSC framing that `write_ansi_hyperlink` and `url_ansi_build`
//!   currently duplicate.  Precondition is measurement on a profile that shows
//!   uncacheable spellings rendering repeatedly — today only junk shapes and
//!   non-canonical `unknown("e030")` do.
//! * **One `Memo` struct instead of three `OnceLock<Option<Box<str>>>`** in `OtherCode`
//!   (~24-32 B per unknown code, plus two lock round-trips).  Deferred with the C1
//!   `sanitized()` fix in place: merging the memos changes the invariant that fix
//!   documents ("`None` ⇒ borrow `render_text()`"), so it wants one deliberate edit
//!   rather than a drive-by while fixing the cap bypass.

use compact_str::CompactString;
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

/// Whether the effective port came from the per-thread test override.
///
/// The port-set `SlotCache`s are sound only under the production axiom "one port
/// per process, ever" (`set_explain_port` is set-once).  A thread-local override
/// breaks that axiom *per thread*: a test that sets `Some(41234)` populates
/// `URL_CACHE[30]` with `127.0.0.1:41234`, and another thread whose port is `0`
/// then hits the same entry and gets the wrong host.  The tests used to hide this
/// by asserting only `contains("127.0.0.1:")`; the port-set caches are bypassed
/// whenever the override is present, which restores the axiom and lets those
/// assertions be exact.
#[cfg(test)]
fn port_is_overridden() -> bool {
    TEST_EXPLAIN_PORT.with(|c| c.borrow().is_some())
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
/// Direct-address index: slot -> payload pointer.
///
/// Holds `&'static ErrCodeInner` rather than a `u16` table position: the payload
/// *is* an `ErrCode` (`ErrCode(ErrCodeRepr::Known(&'static ErrCodeInner))`), so
/// storing the pointer removes a bounds-checked `CODE_TABLE[idx]` read and the
/// `u16::MAX` sentinel from the hottest construction path, and makes
/// `ErrCode::slot()` a field read.  `Option<&T>` occupies the niche, so the
/// table stays 8 bytes per slot.  The slots hold relocations, so the table is
/// **not** lazily-committed BSS: it lives in `.data.rel.ro` and is fully resident
/// (16 KB).  This is the distinction `SlotCache`'s doc exists to keep straight — a
/// resident-size audit that assumed BSS would be wrong by the whole table.
static CODE_INDEX: [Option<&'static ErrCodeInner>; CODE_INDEX_SLOTS] = build_code_index(CODE_INNER);
/// The registry must stay indexable by a `u16` *slot* (see `slot()`): fail the
/// build, not a lookup, if it ever outgrows that.  (This was documented as "the
/// index stores table positions as `u16`" — the index holds
/// `Option<&'static ErrCodeInner>` now, so the note described a shape that no longer
/// existed while the assert itself stayed load-bearing.)
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

const fn build_code_index(
    inners: &'static [ErrCodeInner],
) -> [Option<&'static ErrCodeInner>; CODE_INDEX_SLOTS] {
    let mut idx: [Option<&'static ErrCodeInner>; CODE_INDEX_SLOTS] = [None; CODE_INDEX_SLOTS];
    let mut i = 0;
    while i < inners.len() {
        if let Some(slot) = code_slot(inners[i].canonical) {
            // A duplicate code would otherwise silently shadow the first entry
            // and make it unreachable through `lookup`.  Fail the build instead
            // of corrupting the registry.
            assert!(idx[slot].is_none(), "duplicate code in CODE_TABLE");
            idx[slot] = Some(&inners[i]);
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
/// There are **four** such tables, not two:
///
/// | port | plain | ANSI |
/// |---|---|---|
/// | set (localhost) | `URL_CACHE` | `URL_ANSI_CACHE` |
/// | unset (doc URL) | `DOC_URL_CACHE` | `DOC_URL_ANSI_CACHE` |
///
/// The port-*set* pair embeds the fixed port, so it is sound under the
/// set-once `set_explain_port` rule (`set_explain_port` itself documents this).
/// The port-*unset* pair is sound unconditionally: a doc URL is a pure function
/// of the canonical spelling and contains no port, so there is nothing to
/// invalidate, ever — an earlier version of this note claimed those quadrants
/// were "deleted", which was true of an older design and wrong once
/// `DOC_URL_CACHE`/`DOC_URL_ANSI_CACHE` landed.
///
/// The `OnceLock` slots are inline in the static; a slot's `Box<str>` payload is
/// heap-allocated on first use and never freed.  Only the pages a process
/// actually renders are ever committed: the four `SlotCache`s are
/// `OnceLock`-initialised, so their ~192 KB is lazily-committed BSS.  (`CODE_INDEX`
/// is *not* lazy — it is a const-initialized pointer table and therefore lives in
/// `.rodata`/`.data.rel.ro`, fully resident; the earlier note claimed its laziness
/// by association, which is the distinction that matters when auditing resident
/// size.)  Registry codes never consult the *port-unset*
/// pair: the generator emits both URL forms into `ErrCodeInner` (`doc_url` /
/// `doc_url_ansi`), so `url()` for a registry code is a field read.  They do use
/// the port-*set* pair, which is the whole point of its two quadrants.
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

/// Language-independent documentation host for error-code pages.
///
/// Module scope so the *streaming* writer ([`ErrCode::write_ansi_hyperlink`]) can emit the URL
/// without assembling a `String`; both writers therefore share one source for the framing.
const DOC_URL_PREFIX: &str = "https://doc.posita-lang.org/error_codes/";
const DOC_URL_SUFFIX: &str = ".html";

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
    // A real `assert!`, not `debug_assert!`: the expression below masks with
    // `SLOT_CACHE_SLOTS - 1`, so an out-of-range slot (the `u16::MAX` sentinel, or a future
    // refactor's mistake) would land on slot 2047 and **silently poison a real code's URL** in
    // release — a wrong-output-with-no-crash, which this subsystem treats as its worst class.  The
    // same precedent as `SlotSet::insert`'s release assert (loud at the definition site).  One
    // always-predicted branch against a hit path benched with large headroom.
    assert!(
        (slot as usize) < DIRECT_ADDRESS_SLOTS,
        "slot {slot} exceeds the direct-address range"
    );
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
/// discard the result explicitly with `swallow!(…)` (from `swallow-it`).
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

/// Whether an explain server is running, as the *effective* port sees it.
///
/// Routed through [`explain_port`] rather than reading `EXPLAIN_PORT` directly: under
/// `cfg(test)` the per-thread override shadows the process global, and reading the
/// global made the two disagree ("port 41234 is effective" but "server not running").
/// Production behaviour is identical — `explain_port` falls through to the global.
pub fn is_explain_server_running() -> bool {
    explain_port().is_some()
}

/// Every `ErrorCategory`, in one place: the const block below pins `index()` as a dense
/// bijection onto `0..COUNT` at **compile** time.
///
/// The runtime test alone left a release build free to ship a duplicate or out-of-range
/// index (`label()`'s slug array is ordered by `index()`, so a duplicate silently
/// permutes localized labels).  This file's policy is "fail the build, not a lookup".
const ALL_CATEGORIES: [ErrorCategory; ErrorCategory::COUNT] = [
    ErrorCategory::Parse,
    ErrorCategory::Resolution,
    ErrorCategory::Type,
    ErrorCategory::Contract,
    ErrorCategory::Trait,
    ErrorCategory::Inference,
    ErrorCategory::Internal,
    ErrorCategory::Generic,
];
const _: () = {
    let mut seen = [false; ErrorCategory::COUNT];
    let mut i = 0;
    while i < ALL_CATEGORIES.len() {
        let idx = ALL_CATEGORIES[i].index();
        assert!(
            idx < ErrorCategory::COUNT,
            "an ErrorCategory index is out of range"
        );
        assert!(!seen[idx], "two ErrorCategory variants share an index");
        seen[idx] = true;
        i += 1;
    }
    let mut j = 0;
    while j < ErrorCategory::COUNT {
        assert!(seen[j], "ErrorCategory::index is not dense");
        j += 1;
    }
};

/// Write the localhost explain URL into `out` (the single format site).
/// Write `http://127.0.0.1:{port}/{encoded}`.
///
/// Takes the **already percent-encoded** segment: encoding inside meant both
/// callers sized their buffer with the *raw* `code.len()` while the helper wrote up
/// to 3× that for a non-ASCII spelling, so every such cache miss paid a realloc —
/// in a function whose own comment brags that "the digits are now counted instead
/// of guessed".
fn explain_server_url_into<W: std::fmt::Write>(
    port: u16,
    encoded: &str,
    out: &mut W,
) -> std::fmt::Result {
    use std::fmt::Write as _;
    // Returns the sink's verdict instead of swallowing it: `write_ansi_hyperlink` already
    // propagates, and the generic bound admits fallible sinks.  `String`/`CompactString`
    // cannot fail, so those call sites discard it explicitly.
    write!(out, "http://127.0.0.1:{port}/{encoded}")
}

/// `"http://127.0.0.1:"` — the fixed prefix every localhost explain URL starts
/// with.  `with_capacity(24)` used to under-estimate the real minimum (17 +
/// port digits + `/` + a ≥4-byte code = 27), so **every** cache miss paid one
/// realloc; the digits are now counted instead of guessed.
const EXPLAIN_URL_ORIGIN_LEN: usize = "http://127.0.0.1:".len();
/// Digits in the largest `u16` port (65535).
const MAX_PORT_DIGITS: usize = 5;

/// The public documentation URL from an **already percent-encoded** segment.
///
/// Split out so the unknown-code path can feed `OtherCode::encoded_segment`'s memo:
/// the segment is a pure function of the spelling, and re-running the per-byte encode
/// scan (plus an allocation) on every render was the last per-render allocation here.
fn doc_url_from_encoded(encoded: &str) -> String {
    let mut out =
        String::with_capacity(DOC_URL_PREFIX.len() + encoded.len() + DOC_URL_SUFFIX.len());
    out.push_str(DOC_URL_PREFIX);
    out.push_str(encoded);
    out.push_str(DOC_URL_SUFFIX);
    out
}

/// Build the explanation URL for a code, given an explicit port.
///
/// This is the pure, testable core of [`ErrCode::url`]; callers that need the process-global port
/// (the production path) pass `EXPLAIN_PORT.get().copied()`.  The code is percent-encoded as a
/// single path segment in both branches, so an untrusted code (CLI argument,
/// `Diagnostic::with_code_str`) cannot escape the segment or smuggle control characters into the
/// URL.
///
/// Itself does not memoize: the port-set branch assembles a `String` and the port-unset branch
/// builds the doc URL from the pre-encoded segment, and both go through the `Other` payload's
/// segment memo rather than re-encoding.  The hot-path caller [`ErrCode::url_ansi`] (invoked per
/// diagnostic in `write_header` when hyperlinks are active) caches its full OSC 8 result in a
/// [`SlotCache`] — a *fixed* `SLOT_CACHE_SLOTS`-entry array of `OnceLock`s indexed by
/// [`code_slot`], not a hash map — and there is one such array per quadrant ([`ErrCode::url`] and
/// [`ErrCode::url_ansi`] each keep a port-set and a port-unset cache), so the doc-URL path is
/// cached too, not only the localhost one.  A batch of *N* diagnostics sharing a code therefore
/// pays the build once and a fat-pointer copy per hit (no `Arc` refcount, no clone).  The cold-path
/// caller (the explain server, one URL per served page) bypasses that cache.
///
/// (This doc used to sit above `explain_server_url_into`, which takes an already-encoded segment
/// and builds nothing — rustdoc attributed a per-call cost to a function the cost does not apply
/// to, and this function had no doc at all.  The same failure the i18n module records: "rustdoc
/// attributed a bool-returning contract to a unit function".)
pub(crate) fn explain_url_with_port(code: &str, port: Option<u16>) -> String {
    if let Some(port) = port {
        // Straight into the result: the old path built a `CompactString` and
        // called `into_string()`, which the `SlotCache` doc above claims was
        // removed — the hop was still there for localhost URLs, and it copies
        // whenever the URL is short enough to be inline-resident.
        let encoded = encode_url_path_segment(code);
        let mut out =
            String::with_capacity(EXPLAIN_URL_ORIGIN_LEN + MAX_PORT_DIGITS + 1 + encoded.len());
        swallow_it::swallow!(explain_server_url_into(port, encoded.as_ref(), &mut out)); // `String` sink
        out
    } else {
        // We will register this domain and set up a proper documentation
        // website in the future.  For now, the local explain server is the
        // primary way to view error explanations in a browser.
        doc_url_from_encoded(&encode_url_path_segment(code))
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
    // place (`explain_server_url_into`, now generic over the sink) so the two
    // callers cannot drift.  Building the `CompactString` directly drops the old
    // `String` → `CompactString::from` copy of the whole URL.
    let encoded = encode_url_path_segment(code);
    let mut out =
        CompactString::with_capacity(EXPLAIN_URL_ORIGIN_LEN + MAX_PORT_DIGITS + 1 + encoded.len());
    swallow_it::swallow!(explain_server_url_into(port, encoded.as_ref(), &mut out)); // `CompactString` sink
    out
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
    /// Bit position for the `only_categories` mask.
    ///
    /// Filtering by category used to hash an 8-variant `Copy` enum and probe a set;
    /// the file already made the same move for code filters (`SlotSet`: "no hasher,
    /// no probe, one cache line"), so the mask is the consistent shape.
    pub(crate) const fn mask_bit(&self) -> u16 {
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
        1u16 << idx
    }

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
    pub fn label(&self) -> &'static str {
        // The variant → table-slot mapping lives in `index()` (`pub(crate) const
        // fn`, next to the enum): reusing it keeps the "reorder is a compile
        // error" property while removing a second match that had to be edited
        // in lockstep.  The init array below pins its own arity against `n = 8`.
        let idx = self.index();
        // `cached_label_static!`, not `cached_label!`: the table is a `'static`
        // `OnceLock`, so the `Cow::Owned` branch was unreachable — a 32-byte shell plus a
        // per-call discriminant around a static slice.  `level.rs`, `kind.rs` and
        // `chain.rs` all made this move; this was the last `Cow`-returning label site.
        crate::diagnostics::i18n::cached_label_static! {
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

    /// Parse a category from its canonical key or a documented alias.
    ///
    /// The accepted set is **16** spellings: each canonical key from [`Self::CANONICAL_KEYS`], that
    /// key plus `" error"`, and bare `"error"` for [`Self::Generic`] — all ASCII-case-insensitive.
    /// The `Err` text lists only the canonical keys (that table is the single source for the
    /// "valid: …" list), so this doc is the only place the alias rule is stated; keep the two in
    /// sync rather than widening the list.  `from_str_accepts_every_documented_spelling` pins all
    /// 16.
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
///
/// # Why `Option<ErrCode>` is 16 bytes, and what would break it
///
/// `ErrCode` is two words.  The niche that lets `Option<ErrCode>` stay two words comes
/// from the `Known` variant's `&'static ErrCodeInner` — a reference can never be null,
/// and rustc descends through this newtype and the tagged enum to find it, so no extra
/// discriminant is needed.  Two consequences worth stating:
///
/// * This is **compiler layout behaviour**, not a language guarantee.  A rustc change,
///   or a third variant without a niche, flips `Option<ErrCode>` to 32 bytes.
///   `err_code_stays_two_words` is the guard: if it fails, the fallback cost is one
///   extra word on every `Option<ErrCode>` (e.g. `Diagnostic::code`), nothing worse.
/// * `url()`'s `DOC_URL_CACHE` arm deliberately omits the `cacheable` guard that
///   `url_ansi` has.  That is *structurally* sound, not an oversight: the arm is
///   reached only when `inner()` is `None`, and a slot-ful code always has an inner, so
///   the spelling is canonical by construction (`other_repr` is the only constructor
///   that stores no slot).  A future caller of `other_repr` that passed a registry
///   payload with no slot would invalidate it — which is why the argument lives here
///   and not in a comment at the call site.
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
            //
            // The `e.code == o.raw` arm is **unreachable through the current
            // constructors**: `unknown` routes an exact-match spelling to
            // `Known`, and `new` produces `canonical == None`.  It is kept
            // deliberately — it states the intended rule ("an exact spelling is
            // canonical") rather than relying on a constructor-level proof that
            // a future `other_repr` caller could invalidate.
            // Field read now (see `OtherCode::canonical_spelling`); the rule
            // itself is unchanged and still stated at the constructor.
            ErrCodeRepr::Other(o) => o.canonical_spelling,
        }
    }
}

/// One registry payload, generated at build time.
///
/// `doc_url` / `doc_url_ansi` are pure functions of the code, so they are baked
/// in: `url()` for a registry code is a field read.  What *is* memoized is the
/// unknown-code path — `DOC_URL_CACHE` / `DOC_URL_ANSI_CACHE` cover shape-valid
/// spellings that have no registry entry (`E999`), whose URL depends only on the
/// URL-safe spelling.  (This note used to claim that half of the caches was
/// "deleted rather than memoized", contradicting those two live statics; the
/// `SlotCache` doc already records correcting the same stale claim for the
/// port-override half.)  The type is fully immutable, which is what lets the
/// generated slice be a `const`.
pub(crate) struct ErrCodeInner {
    /// Verbatim spelling (`unknown("e030")` echoes `e030`).
    pub(crate) display: &'static str,
    /// Direct-address slot of the canonical spelling.
    ///
    /// Emitted literally by the generator instead of being re-derived through
    /// `CODE_INDEX`: `ErrCode` is a single `&'static ErrCodeInner`, so a field
    /// here makes `slot()` for a registry code one load rather than an
    /// entry -> slot two-hop.
    pub(crate) slot: u16,
    /// Canonical spelling — the identity `Eq`/`Hash` use.
    pub(crate) canonical: &'static str,
    /// Resolved registry entry, when there is one.
    pub(crate) entry: Option<&'static CodeEntry>,
    /// `true` for a validated registry spelling: `display` is printable ASCII
    /// and needs no terminal sanitization.
    pub(crate) known: bool,
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
    canonical: Option<&'static ErrCodeInner>,

    /// `code_slot` of this spelling, computed once at construction.
    ///
    /// `slot()` is on every cache-eligible lookup, and for a shape-valid
    /// unregistered code (`unknown("E999")` — the one `Other` class that *does*
    /// hit the slot caches) it used to re-parse four bytes through `code_slot` on
    /// every call.  The module already made exactly this trade for `Known` codes
    /// ("a field read, not an entry→slot two-hop"); the `Other` payload was the
    /// one representation left behind.
    slot: Option<u16>,
    /// Whether this spelling is the canonical one (see `is_canonical_spelling`).
    /// Precomputed for the same reason as `slot`.
    canonical_spelling: bool,
    /// The three derived render surfaces, computed in one `OnceLock`.
    ///
    /// Merged from three independent locks: a `Display` → hyperlink → URL sequence on one junk
    /// spelling (`unknown("junk\u{1b}")`) took all three, and the payload is ~16-24 B smaller this
    /// way.  Initialization is **acyclic by construction** — `capped` is computed first and the
    /// other two are derived from it in the same closure, so there is no lock re-entered and no
    /// ordering to maintain across functions (the old argument was "the locks are independent, so
    /// the nesting cannot deadlock").
    memos: std::sync::OnceLock<OtherMemos>,
    // NOTE: a payload-level memo of the doc URL was tried and reverted.  The
    // public `url()`/`url_ansi()` contract is `Cow<'static, str>`, and a
    // `OnceLock<Box<str>>` inside an `Arc<OtherCode>` can only lend a `&self`
    // lifetime: returning it as `'static` needs `Box::leak` or a raw-pointer
    // lifetime reconstruction, both of which this codebase forbids.  Making the
    // memo work would require changing the public return type (every caller
    // stores it) or internment (process-global growth for untrusted input), so
    // junk spellings keep rebuilding their URL; they are per-render expensive
    // only for diagnostics carrying a *bad* code string.
}

/// Cap for a *rendered* code spelling.
///
/// The spelling is the last attacker-influenced surface in this module without a budget:
/// `ErrCode::unknown` accepts arbitrary length and `Diagnostic::with_code_str` takes
/// LSP/CLI text, so a 100 KB spelling used to flow into the encoder, the OSC wrapper and
/// an unbounded `Display` line.  Applied at the render surfaces only — see
/// [`OtherCode::capped`] for why storage must stay verbatim.  Two distinct over-cap
/// spellings therefore render identically, which is acceptable for junk input and is
/// stated here rather than left implicit.
const MAX_CODE_CHARS: usize = 128;

/// The three derived render surfaces of one [`OtherCode`] payload.
///
/// `None` in a slot means "identity transform: borrow the previous level", exactly as the three
/// former per-surface `OnceLock`s did — `capped` → `raw`, `sanitized`/`encoded` → the capped text.
struct OtherMemos {
    /// Capped spelling ([`MAX_CODE_CHARS`]); `None` = fits, borrow `raw`.
    capped: Option<Box<str>>,
    /// [`super::sanitize_for_terminal`] of the capped spelling; `None` = already clean.
    sanitized: Option<Box<str>>,
    /// Percent-encoded capped spelling; `None` = unchanged by the encoder.
    encoded: Option<Box<str>>,
}

impl OtherCode {
    /// The three derived surfaces, computed once.
    ///
    /// Build order is load-bearing and local: `capped` first, then the other two derived from it.
    /// Nothing here calls back into an accessor, so the single lock is never re-entered.
    fn memos(&self) -> &OtherMemos {
        self.memos.get_or_init(|| {
            // `char_indices().nth(MAX_CODE_CHARS)` is the single-pass cap: `None` proves the
            // spelling fits, and `Some((off, _))` cuts exactly `MAX_CODE_CHARS` chars on a char
            // boundary, so the reservation is exact and `into_boxed_str` has nothing to shrink.
            // The old shape counted every char first (a 100 KB junk spelling paid a full decode
            // walk although the cap trips at 129) and then `collect`ed with a size hint it shrank
            // away.  Same pattern as `kind.rs::capped_payload`.
            let capped = self.raw.char_indices().nth(MAX_CODE_CHARS).map(|(off, _)| {
                let mut out = String::with_capacity(off + '…'.len_utf8());
                out.push_str(&self.raw[..off]);
                out.push('…');
                out.into_boxed_str()
            });
            let rendered: &str = capped.as_deref().unwrap_or(&self.raw);
            let sanitized = match super::sanitize_for_terminal(rendered) {
                std::borrow::Cow::Borrowed(_) => None,
                std::borrow::Cow::Owned(s) => Some(s.into_boxed_str()),
            };
            let encoded = match encode_url_path_segment(rendered) {
                std::borrow::Cow::Borrowed(_) => None,
                std::borrow::Cow::Owned(owned) => Some(owned.into_boxed_str()),
            };
            OtherMemos {
                capped,
                sanitized,
                encoded,
            }
        })
    }

    /// The spelling as every render surface shows it (capped, see [`MAX_CODE_CHARS`]).
    fn render_text(&self) -> &str {
        match &self.memos().capped {
            Some(capped) => capped,
            None => &self.raw,
        }
    }

    /// The sanitized capped spelling (`None` in the memo means "nothing needed replacing", so
    /// borrow the rendered text — never `raw`: an over-cap spelling with no control bytes would
    /// otherwise bypass `MAX_CODE_CHARS` on every visible surface while the URL segment stayed
    /// capped).
    fn sanitized(&self) -> &str {
        match &self.memos().sanitized {
            Some(clean) => clean,
            None => self.render_text(),
        }
    }

    /// The percent-encoded capped spelling, memoized in the payload.
    fn encoded_segment(&self) -> &str {
        match &self.memos().encoded {
            Some(encoded) => encoded,
            // Identity transform: borrow `render_text()` instead of boxing a copy.
            None => self.render_text(),
        }
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
        // `std::ptr::eq` on a wide pointer compares the **data address only** and ignores the
        // metadata (the length), so a same-address/different-length pair would compare *equal*
        // — the opposite of what an earlier revision of this comment claimed.  No length check is
        // needed because of a property of `canonical()`, not of `ptr::eq`: it returns whole
        // allocations (an entire `'static` literal, or an entire `Arc<OtherCode>` payload), never
        // a derived slice, so equal addresses can only mean the same allocation and therefore the
        // same code.  `ptr_eq_fast_path_never_conflates_distinct_codes` pins that invariant over
        // the whole registry — the argument lives in prose otherwise, and a future `canonical()`
        // that returned a trimmed slice would let two distinct codes compare equal and collapse in
        // a `HashSet`, silently suppressing a diagnostic.
        // The `|| a == b` fallback is load-bearing, not vestigial: junk codes are
        // **not** interned, so two `unknown("junk1")` values are two distinct
        // `Arc<OtherCode>` allocations that must compare by bytes.  (An earlier
        // version of this comment claimed the fallback was gone and that each
        // unknown spelling had "its own interned allocation" — both false.)  For
        // registry codes the fast path *is* exact when the generated
        // `ErrCodeInner::canonical` and `CodeEntry::code` are the same literal;
        // `code_inner_agrees_with_the_runtime_payload` pins their *string*
        // equality, and identical-literals-dedup is a linker detail rather than a
        // language guarantee, which is the second reason the byte compare stays.
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
/// No precomputed hash exists, deliberately.
///
/// `Hash for ErrCode` derives from `canonical()` through the caller's hasher, so it
/// agrees with `PartialEq` *structurally* rather than by keeping two stored values in
/// agreement — the old precomputed FNV value was inconsistent with `Eq` and bypassed
/// the map's own SipHash keying, which is exactly how a `HashMap<ErrCode, _>` became a
/// flooding target (see the `Hash` impl).  A generation canary was tried on top of it
/// and then deleted: the `canonical == entry.code` comparison in
/// `code_inner_agrees_with_the_runtime_payload` already subsumes it, and it cost 8
/// bytes in every registry entry's `.rodata`.
/// Wrap `url` in an OSC 8 hyperlink around `visible`.
/// Payload for a registry entry.
///
/// `resolve`/`lookup` only ever return `CODE_TABLE` entries, so the index from
/// `CODE_INDEX` is valid by construction; the assertion pins the invariant that
/// `CODE_INNER` and `CODE_TABLE` are parallel (both are generated in document
/// order).
#[inline]

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
fn other_repr(raw: CompactString, canonical: Option<&'static ErrCodeInner>) -> ErrCode {
    // No stored hash: `Hash for ErrCode` derives from `canonical()` (see its impl),
    // which is the *only* rule `PartialEq` uses — so consistency is structural
    // instead of two precomputed values that must be kept in agreement.
    // Both derived facts are computed here, where `resolve` has already run, so
    // the per-call accessors become field reads.
    let slot = canonical
        .map(|inner| inner.slot)
        .or_else(|| code_slot(raw.as_str()).map(|s| s as u16));
    let canonical_spelling = canonical.is_none_or(|inner| inner.canonical == raw.as_str());
    ErrCode(ErrCodeRepr::Other(std::sync::Arc::new(OtherCode {
        raw,
        canonical,
        slot,
        canonical_spelling,
        memos: std::sync::OnceLock::new(),
    })))
}

impl std::hash::Hash for ErrCode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash **exactly what `PartialEq` compares** — `canonical()` — instead of a
        // precomputed `u64`.
        //
        // The precomputed form was a fixed, unkeyed FNV-1a value, and it had two
        // problems:
        //
        // * **Not consistent with equality, by construction.**  `Eq` compares
        //   `canonical()` strings, while the stored value came from whichever
        //   spelling happened to build the payload; nothing forced the two to agree
        //   once the stored value and the compared string diverged, and
        //   `hash_is_consistent_across_representations` catches exactly that
        //   disagreement (a pinned test, not a hypothesis).
        // * **Flooding.**  A fixed inner `u64` bypasses the outer hasher's per-map
        //   keying — the standard defence — because a collision in our value is
        //   already a collision after SipHash.  Hashing the bytes here puts the
        //   attacker-controlled spelling (`ErrCode::unknown`) back under the map's
        //   own random key.
        //
        // Cost: a registry code now hashes four bytes instead of one `write_u64`.
        // The dedup set this exists for is `HashSet<ErrCode>` over diagnostics, where
        // that is noise against the surrounding work, and correctness of `Eq`/`Hash`
        // is not negotiable for a saving of a few nanoseconds.
        self.canonical().hash(state);
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
        if let Some(inner) = resolve(code) {
            return ErrCode(ErrCodeRepr::Known(inner));
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
            Some(inner) => Ok(ErrCode(ErrCodeRepr::Known(inner))),
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
            Some(inner) if inner.canonical == raw.as_str() => ErrCode(ErrCodeRepr::Known(inner)),
            inner => other_repr(raw, inner),
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
            ErrCodeRepr::Other(o) => o.canonical.map_or(o.raw.as_str(), |inner| inner.canonical),
        }
    }

    /// Resolve this error code to its table entry, if present.
    ///
    /// Callers that need multiple fields (`title`, `category`, `explain`)
    /// should use this once and derive each from the returned entry, avoiding
    /// repeated lookups (a shape check + one array read on `CODE_INDEX`).
    #[must_use]
    pub fn entry(&self) -> Option<&'static CodeEntry> {
        match &self.0 {
            ErrCodeRepr::Known(e) => e.entry,
            // An `unknown` spelling that names a real entry still resolves its
            // metadata, matching the case-insensitive resolution in
            // `new`/`try_new`.
            ErrCodeRepr::Other(o) => o.canonical.and_then(|inner| inner.entry),
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
    #[must_use]
    pub fn explain(&self) -> &'static str {
        self.entry_or_unknown().explain
    }

    /// The generated payload, when this code resolves to one.
    ///
    /// `Known` holds it directly; an `Other` spelling that resolves
    /// (`unknown("e030")`) keeps it in its canonical slot.  One helper so the URL
    /// paths stay one load instead of re-deriving it through `CODE_INDEX`.
    fn inner(&self) -> Option<&'static ErrCodeInner> {
        match &self.0 {
            ErrCodeRepr::Known(inner) => Some(inner),
            ErrCodeRepr::Other(o) => o.canonical,
        }
    }

    /// The direct-address slot for this code, **when one can be derived**.
    ///
    /// `Known` carries its slot in the payload (a field read); an `Other` spelling
    /// gets one when it canonicalizes to a registered entry or when `code_slot`
    /// accepts the shape (`^[EW]\d{3}$`).  `None` means "bypass the slot cache" —
    /// the URL path then builds uncached, which is correct and rare.
    ///
    /// **A slot says nothing about ANSI-cacheability.**  The URL is a pure function of the
    /// *canonical* spelling, so `URL_CACHE` legitimately shares one slot across representations
    /// (`unknown("e030")` and `new("E030")`); the OSC 8 wrapper embeds the *visible* spelling, so
    /// [`Self::url_ansi`] gates separately on [`Self::ansi_cacheable`].  The old first line read
    /// "…, if its spelling is cacheable", which contradicted the paragraph below it and taught the
    /// exact regression that gate prevents: deleting `ansi_cacheable` would fill
    /// `URL_ANSI_CACHE[30]` with the visible text `e030` and then render `E030` with the wrong
    /// label.
    fn slot(&self) -> Option<u16> {
        match &self.0 {
            // A field read: the slot is emitted into the payload, so this is not
            // an entry -> slot two-hop any more.
            ErrCodeRepr::Known(inner) => Some(inner.slot),
            // Field reads now (see `OtherCode::slot`).
            ErrCodeRepr::Other(o) => o.slot,
        }
    }

    /// The diagnostic URL for the `--explain` feature.
    ///
    /// [`explain_url_with_port`] percent-encodes the code as a single path
    /// segment, so an unknown code holding arbitrary bytes (see [`Self::unknown`])
    /// cannot escape the segment or the enclosing link.
    ///
    /// **Memoization:** once the explain port is fixed ([`set_explain_port`] is
    /// set-once), the result depends only on the code string.  A code that
    /// resolves to a registry payload (`Known`, or an `Other` spelling that
    /// canonicalizes to one) is a **field read** of `ErrCodeInner::doc_url` when
    /// the port is unset.  Codes matching the registry's `^[EW]\d{3}$` shape but
    /// *unregistered* (`E999`) are cached per [`code_slot`] in a
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
    /// The port-unset quadrant has its own pair (`DOC_URL_CACHE` /
    /// `DOC_URL_ANSI_CACHE`), covering exactly the shape-valid unregistered codes
    /// (`E999`) whose doc URL is a pure function of the spelling; registry codes
    /// read the generator-emitted constant instead, and shapes outside
    /// `^[EW]\d{3}$` build on every call (they have no slot to cache under).
    /// A later `set_explain_port` can never observe a stale doc-URL host because
    /// port-agnostic URLs are the *only* thing in that pair.
    #[must_use]
    pub fn url(&self) -> std::borrow::Cow<'static, str> {
        self.url_for_slot()
    }

    fn url_for_slot(&self) -> std::borrow::Cow<'static, str> {
        self.url_for_slot_with_port(explain_port())
    }

    fn url_for_slot_with_port(&self, port: Option<u16>) -> std::borrow::Cow<'static, str> {
        // Derived once here.  The parameter this grew from was always
        // `self.slot()`, so passing it added an argument plus a `debug_assert` that
        // could never fire — and `url_ansi_build` then re-derived anyway.
        let slot = self.slot();
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
            if let Some(inner) = self.inner() {
                return std::borrow::Cow::Borrowed(inner.doc_url);
            }
            return match slot {
                None => {
                    // A slot-less spelling has no per-slot cache; its encoding is
                    // memoized in the payload instead, so this branch is one memo read
                    // plus the URL assembly.
                    //
                    // This arm is reachable only with `slot == None`, and a slot-less
                    // code is always an `Other` payload (`other_repr` is the only
                    // constructor that stores no slot, and every registry code carries a
                    // generator slot).  The `Known => None` case that used to sit here was
                    // therefore dead — and had it ever been reached it would have built
                    // the URL from the *uncapped* `canonical()`, the same
                    // `MAX_CODE_CHARS` bypass `sanitized()` just had.
                    let ErrCodeRepr::Other(other) = &self.0 else {
                        unreachable!("a slot-less code is always an `Other` payload")
                    };
                    std::borrow::Cow::Owned(doc_url_from_encoded(other.encoded_segment()))
                }
                Some(slot) => {
                    static DOC_URL_CACHE: SlotCache = new_slot_cache();
                    std::borrow::Cow::Borrowed(cached_slot_string(&DOC_URL_CACHE, slot, || {
                        explain_url_with_port(self.canonical(), None)
                    }))
                }
            };
        };
        let Some(slot) = slot else {
            // Slot-less spellings must use the payload's `encoded` memo too: this arm used
            // to call `explain_url_with_port(canonical(), Some(port))`, which re-ran the
            // percent-encoder — a scan plus an allocation — on *every* render, exactly what
            // the memo exists to prevent.  The port is part of the URL, so the URL itself
            // stays uncached; the per-render segment does not.
            let ErrCodeRepr::Other(other) = &self.0 else {
                unreachable!("a registry code always carries a generator slot")
            };
            let encoded = other.encoded_segment();
            let mut out =
                String::with_capacity(EXPLAIN_URL_ORIGIN_LEN + MAX_PORT_DIGITS + 1 + encoded.len());
            swallow_it::swallow!(explain_server_url_into(port, encoded, &mut out)); // `String` sink
            return std::borrow::Cow::Owned(out);
        };
        // The localhost form embeds the port, which is only known at runtime and
        // fixed once (`set_explain_port`), so it keeps its slot cache — except
        // under the per-thread test override, where "one port per process" no
        // longer holds and a cross-thread hit would serve another port's URL.
        #[cfg(test)]
        if port_is_overridden() {
            return std::borrow::Cow::Owned(explain_url_with_port(self.canonical(), Some(port)));
        }
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
        // Evaluated once.  `ansi_cacheable()` *is* `is_canonical_spelling()`, and
        // this function consulted it three times in six lines (plus once inside the
        // URL path) on the per-diagnostic hyperlink path.
        let cacheable = self.ansi_cacheable();
        let slot = self.slot();
        let Some(port) = explain_port() else {
            // Same as `url`, with one extra condition: the wrapper embeds the
            // *visible* spelling, so only a canonical spelling may borrow the
            // build-time constant.
            if cacheable && let Some(inner) = self.inner() {
                return std::borrow::Cow::Borrowed(inner.doc_url_ansi);
            }
            return match slot {
                // Shape-valid but unregistered (`E999`): the wrapper embeds
                // only the canonical spelling and the port-free doc URL, so
                // the slot is a sound key with nothing to invalidate.
                Some(slot) if cacheable => {
                    static DOC_URL_ANSI_CACHE: SlotCache = new_slot_cache();
                    std::borrow::Cow::Borrowed(cached_slot_string(
                        &DOC_URL_ANSI_CACHE,
                        slot,
                        || self.url_ansi_build(None),
                    ))
                }
                _ => std::borrow::Cow::Owned(self.url_ansi_build(None)),
            };
        };
        let Some(slot) = slot else {
            // No registry slot (a malformed spelling): build the wrapper, whose
            // sanitize is memoized in the payload.
            return std::borrow::Cow::Owned(self.url_ansi_build(Some(port)));
        };
        // The cached string embeds the *visible* spelling, so only a canonical
        // spelling may populate the slot.
        if !cacheable {
            return std::borrow::Cow::Owned(self.url_ansi_build(Some(port)));
        }
        // Same override exception as `url`: the wrapper embeds the port through
        // its URL, so a cross-thread hit under the test override would serve
        // another port's hyperlink.
        #[cfg(test)]
        if port_is_overridden() {
            return std::borrow::Cow::Owned(self.url_ansi_build(Some(port)));
        }
        static URL_ANSI_CACHE: SlotCache = new_slot_cache();
        std::borrow::Cow::Borrowed(cached_slot_string(&URL_ANSI_CACHE, slot, || {
            self.url_ansi_build(Some(port))
        }))
    }

    /// Uncached core of [`url_ansi`](Self::url_ansi): build the OSC 8 hyperlink
    /// from the (uncached) [`url`](Self::url) and the sanitized code.  Separated
    /// so the cache path can `set` without holding a borrow across the build.
    ///
    /// (A streaming-contract paragraph about writing "into `out` without an intermediate buffer"
    /// used to sit here, stacked above [`Self::visible_text`] — it describes
    /// [`Self::write_ansi_hyperlink`], where it now lives.  rustdoc attributed a write-into-sink
    /// contract to two `&str`-returning items, the failure mode the i18n module already
    /// documented.)
    ///
    /// The text shown inside a hyperlink (or after the level prefix).
    ///
    /// One definition: `url_ansi_build` and `write_ansi_hyperlink` each carried an
    /// identical `match &self.0` block, and this module already applied the "two call
    /// sites of one operation are exactly how they drift" rule to the sanitize memo.
    fn visible_text(&self) -> &str {
        match &self.0 {
            ErrCodeRepr::Known(inner) => inner.display,
            ErrCodeRepr::Other(o) => o.sanitized(),
        }
    }

    /// Write this code's OSC 8 hyperlink into `out` without an intermediate buffer.
    ///
    /// [`url_ansi`](Self::url_ansi) returns a `Cow<'static, str>` and therefore has to own a
    /// `String` for every *uncacheable* spelling (a non-canonical `unknown("e030")`, junk
    /// shapes).  The per-diagnostic terminal path only concatenates the escape into an existing
    /// buffer, so that owned wrapper was pure copying: this writes the same bytes straight
    /// through.  `url_ansi` stays as the compat form.
    pub fn write_ansi_hyperlink<W: std::fmt::Write>(&self, out: &mut W) -> std::fmt::Result {
        // The URL is `Cow`: borrowed for registry/cached spellings (no copy at all),
        // owned only when a slot-less spelling had to be assembled.
        // `explain_port()`, matching every other URL surface in this file (`url`,
        // `url_ansi`).  A literal `None` made this function emit the *documentation* host
        // even while the explain server was running — contradicting the "same bytes
        // straight through" contract in its own doc, and the flip contract `set_explain_port`
        // states (a caller that rendered first has already shown the user the wrong host).
        let port = explain_port();
        let visible = self.visible_text();
        out.write_str("\x1b]8;;")?;
        // Slot-less spellings stream their URL instead of assembling an owned `String` and copying
        // it: `url_for_slot_with_port` builds one for exactly those quadrants (junk shapes,
        // over-cap clean spellings, the empty code), which made this function's "without an
        // intermediate buffer" claim false for the one class that needed it.  Slot-ful classes were
        // already borrowed from the caches / the generated constant, so they take the `Cow` path.
        if self.slot().is_none() {
            let ErrCodeRepr::Other(other) = &self.0 else {
                unreachable!("a slot-less code is always an `Other` payload")
            };
            let encoded = other.encoded_segment();
            match port {
                Some(port) => write!(out, "http://127.0.0.1:{port}/{encoded}")?,
                None => {
                    out.write_str(DOC_URL_PREFIX)?;
                    out.write_str(encoded)?;
                    out.write_str(DOC_URL_SUFFIX)?;
                }
            }
        } else {
            out.write_str(self.url_for_slot_with_port(port).as_ref())?;
        }
        out.write_str("\x1b\\")?;
        out.write_str(visible)?;
        out.write_str("\x1b]8;;\x1b\\")
    }

    fn url_ansi_build(&self, port: Option<u16>) -> String {
        // Build the URL exactly once. The old `format_compact!(self.url())`
        // allocated for unknown codes (no registry slot → no cache) only to
        // format it again; assemble straight into the result.  The capacity is
        // exact because this path runs per call for non-cacheable spellings, so
        // every growth realloc was pure overhead.
        // The URL is a pure function of the *canonical* spelling, so it is
        // cacheable whenever a slot exists — independently of whether the
        // *wrapper* (whose visible text is the verbatim spelling) is.  Deriving
        // it from `self.slot()` rather than from the caller's `slot` parameter
        // stops `unknown("e030")` from rebuilding `/E030` per render just because
        // its link text is uncacheable.
        let url: std::borrow::Cow<'static, str> = self.url_for_slot_with_port(port);
        // The sanitize is already memoized in the payload (`OtherCode::sanitized`,
        // used by `Display`) — this path used to re-run the scan *and* allocate on
        // every call for a non-cacheable spelling.  Two call sites of one operation
        // are exactly how they drift.
        let visible = self.visible_text();
        // One buffer for the whole escape: `ansi_wrap` allocated a second `String`
        // on top of the URL, so an uncached render paid two allocations.  The URL
        // is a `Cow` — borrowed for the registry/`Borrowed` cases (no copy beyond
        // the escape prefix), owned only when it had to be built above.
        let mut out = String::with_capacity(5 + url.len() + 2 + visible.len() + 7);
        out.push_str("\x1b]8;;");
        out.push_str(&url);
        out.push_str("\x1b\\");
        out.push_str(visible);
        out.push_str("\x1b]8;;\x1b\\");
        out
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
        // `CODE_INDEX` is built from `CODE_INNER`, and `build_code_index`
        // *silently skips* an entry whose `canonical` fails `code_slot` — so a
        // generator that emitted a 5-char canonical compiled fine and left every
        // lookup falling back to "unknown error", caught only if the test suite ran
        // on that binary.  Pin the payload table too: fail the build, not a lookup.
        let canonical = CODE_INNER[i].canonical.as_bytes();
        assert!(canonical.len() == 4);
        assert!(canonical[0] == b'E' || canonical[0] == b'W');
        assert!(canonical[1].is_ascii_digit());
        assert!(canonical[2].is_ascii_digit());
        assert!(canonical[3].is_ascii_digit());
        assert!(
            CODE_INNER[i].slot as usize
                == code_slot(CODE_INNER[i].canonical).expect("validated canonical shape")
        );
        i += 1;
    }
};

/// Compile-time string equality (`str::eq` is not usable in a `const` block).
const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

// **The two generated tables must stay parallel**, and until now that was pinned
// only by `code_inner_agrees_with_the_runtime_payload` — a *runtime* test that
// release builds never run (`cargo install`, dist builds).  `CODE_INDEX` is built
// from `CODE_INNER` while `title()`/`category()`/`explain()` resolve through
// `entry → CODE_TABLE`, so a generator that emitted the two tables in different
// orders (same lengths, wrong pairing) would compile cleanly and serve wrong
// metadata for every registry code in every release binary.
//
// `std::ptr::eq` is not a `const fn`, so the identity check compares the addresses
// as `usize` — same question, const-legal.
const _: () = {
    assert!(
        CODE_TABLE.len() == CODE_INNER.len(),
        "the generated CODE_TABLE and CODE_INNER must have one entry each"
    );
    let mut i = 0;
    while i < CODE_TABLE.len() {
        let entry = &CODE_TABLE[i];
        assert!(
            str_eq(CODE_INNER[i].canonical, entry.code),
            "CODE_INNER/CODE_TABLE disagree on the canonical spelling"
        );
        assert!(
            str_eq(CODE_INNER[i].display, entry.code),
            "CODE_INNER/CODE_TABLE disagree on the display spelling"
        );
        assert!(CODE_INNER[i].known, "a registry entry must be marked known");
        // Pointer identity is not expressible in const eval ("pointers do not have
        // an integer value"), so the pairing is checked on the payload's own
        // identifying fields — which is what a reordering would break anyway: if the
        // generator emitted the two tables in different orders, entry `i` of one would
        // name a different code/slot than entry `i` of the other.
        match CODE_INNER[i].entry {
            Some(payload) => {
                assert!(
                    str_eq(payload.code, entry.code),
                    "CODE_INNER's payload names a different code than CODE_TABLE[i]"
                );
                assert!(
                    payload.slot == entry.slot,
                    "CODE_INNER's payload carries a different slot than CODE_TABLE[i]"
                );
            }
            None => assert!(false, "a CODE_INNER registry entry must carry its payload"),
        }
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
pub(crate) fn resolve(code: &str) -> Option<&'static ErrCodeInner> {
    lookup_uppercased_inner(code)
}

/// Look up a code string in the table.
#[inline]
pub(crate) fn lookup_inner(code: &str) -> Option<&'static ErrCodeInner> {
    let slot = code_slot(code)?;
    // Direct index, not `get`: `code_slot` is a closed-range parser, so the
    // mask is in bounds for every `Some` (pinned by
    // `direct_address_index_matches_table`).  A miss is the `None` niche — no
    // sentinel compare, no second array read.
    CODE_INDEX[slot & (CODE_INDEX_SLOTS - 1)]
}

/// Look up a code string's registry entry.
#[inline]
pub(crate) fn lookup(code: &str) -> Option<&'static CodeEntry> {
    lookup_inner(code).and_then(|inner| inner.entry)
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
pub(crate) fn lookup_uppercased_inner(code: &str) -> Option<&'static ErrCodeInner> {
    let b = code.as_bytes();
    if b.len() != 4 {
        return None;
    }
    // Compute the slot directly: `to_ascii_uppercase` only affects byte 0
    // (bytes 1..3 are digits by the table's shape assertion), so this avoids
    // the stack buffer, the `from_utf8` hop, and the full re-parse `lookup`
    // would do.  No `&str` is constructed, so no UTF-8 validation is needed.
    let slot = code_slot_parts(b[0].to_ascii_uppercase(), b[1], b[2], b[3])?;
    CODE_INDEX[slot & (CODE_INDEX_SLOTS - 1)]
}

/// [`lookup`] on the ASCII-uppercased form of `code`, without allocating.
#[inline]
pub(crate) fn lookup_uppercased(code: &str) -> Option<&'static CodeEntry> {
    lookup_uppercased_inner(code).and_then(|inner| inner.entry)
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

#[must_use]
pub fn related_codes_of(code: &str) -> &'static [&'static str] {
    resolve(code)
        .and_then(|inner| inner.entry)
        .map_or(&[], |entry| entry.related)
}

#[must_use]
pub fn category_siblings_of(code: &str, max: usize) -> &'static [&'static str] {
    let Some(entry) = resolve(code).and_then(|inner| inner.entry) else {
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
        // Deliberately no `PORT_LOCK`: this test only exercises the *override* paths
        // (`set_explain_port_override`), which are thread-local by construction, so it
        // cannot observe another test's process-global port.
        let code = ErrCode::try_new("E030").expect("E030 is registered");
        let doc = with_explain_port(None, || code.url().into_owned());
        assert!(doc.starts_with("https://"), "doc URL expected, got {doc}");
        let local = with_explain_port(Some(41234), || code.url().into_owned());
        // Exact, not just the host: the port-set caches are bypassed whenever the
        // override is present, so a cached entry built under another port (this
        // test binary's process-global one may already be `0`) can no longer leak
        // in.  The old assertion stopped at `127.0.0.1:` precisely because it
        // could — which is the hazard the bypass removes.
        assert!(
            local.contains("127.0.0.1:41234"),
            "localhost URL with the override port expected, got {local}"
        );
        let back = with_explain_port(None, || code.url().into_owned());
        assert_eq!(back, doc, "the override must not be sticky");
        // The ANSI wrapper follows the same flip.
        let ansi_local = with_explain_port(Some(41234), || code.url_ansi().into_owned());
        assert!(
            ansi_local.contains("127.0.0.1:41234"),
            "hyperlinked URL with the override port expected, got {ansi_local}"
        );
    }

    /// `Eq` implies equal hashes **across representations** — structurally, because
    /// `Hash` now hashes exactly what `PartialEq` compares.
    #[test]
    fn hash_is_consistent_across_representations() {
        use std::hash::{BuildHasher, RandomState};
        // **One** `RandomState`, not one per call: each instance carries its own
        // random key, so `RandomState::new().hash_one(a) == RandomState::new().hash_one(b)`
        // fails for equal values — that mistake is what this test first caught.
        let rs = RandomState::new();
        let h = |c: &ErrCode| rs.hash_one(c);
        let known = ErrCode::new("E030");
        let verbatim = ErrCode::unknown("e030");
        assert_eq!(known, verbatim, "the two spellings must compare equal");
        assert_eq!(
            h(&known),
            h(&verbatim),
            "equal codes must hash equally (Hash derives from `canonical()`, the \
             same rule `PartialEq` compares)"
        );
        // Distinct codes stay distinct.
        assert_ne!(h(&ErrCode::new("E030")), h(&ErrCode::new("E031")));
        // The hash is a function of `canonical()` alone: two codes that canonicalise
        // to the same string hash equally even when one is a junk spelling of the
        // other's casing.
        assert_eq!(super::resolve("e030").map(|e| e.canonical), Some("E030"));
        // A junk spelling is its own canonical form, so its hash is stable across
        // constructions (one function, no per-instance salt).
        assert_eq!(h(&ErrCode::unknown("zzz")), h(&ErrCode::unknown("zzz")));
    }

    /// The contract `DiagFilter::code_matches` depends on (documented there):
    /// for the `Other` variant, `entry()` must resolve whenever the *uppercased*
    /// spelling is registered.  Without it, rustc-style `-A e030` would silently
    /// stop suppressing `ErrCode::unknown("e030")`.
    #[test]
    fn unknown_code_resolving_to_a_registered_spelling_has_an_entry() {
        let known = ErrCode::try_new("E030").expect("E030 is registered");
        let verbatim = ErrCode::unknown("e030");
        let entry = verbatim.entry().expect(
            "an unresolvable-spelling code that uppercases to a registry code must resolve",
        );
        assert_eq!(
            Some(entry.slot),
            known.entry().map(|e| e.slot),
            "the resolved entry must be the registry entry"
        );
        // A genuinely unregistered spelling resolves to nothing — the filter's
        // verbatim set is the only thing that can match it.
        assert!(ErrCode::unknown("zzz").entry().is_none());
    }

    /// The override is per-thread, so two threads with different ports must never
    /// observe each other's cached entry.  Before the bypass this was the exact
    /// hazard the loose `contains("127.0.0.1:")` assertion papered over: both
    /// threads share the process-global `URL_CACHE` keyed by slot alone.
    #[test]
    fn override_ports_do_not_leak_across_threads() {
        let run = |port: u16| {
            std::thread::spawn(move || {
                with_explain_port(Some(port), || {
                    ErrCode::try_new("E030")
                        .expect("E030 is registered")
                        .url()
                        .into_owned()
                })
            })
            .join()
            .expect("thread")
        };
        let a = run(41111);
        let b = run(42222);
        assert!(a.contains("127.0.0.1:41111"), "{a}");
        assert!(b.contains("127.0.0.1:42222"), "{b}");
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
        // The *same* list the `const _` block above pins, so the compile-time check and
        // the runtime test cannot drift (the test used to build its own copy).
        let all = ALL_CATEGORIES;
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
        let doc = code.url_for_slot_with_port(None);
        // `with_explain_port` also *bypasses* the process-global port-set cache
        // (see `port_is_overridden`), which is what makes the exact-port assertion
        // sound: another test in this binary may already have cached slot
        // `E030`'s localhost URL for a different port.
        let local = with_explain_port(Some(41234), || code.url_for_slot_with_port(Some(41234)));
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

    /// `write_ansi_hyperlink` and `url_ansi` must agree byte for byte, for every code class.
    ///
    /// Two hand-written producers of one OSC 8 wrapper: `url_ansi` assembles a `String`, the
    /// streaming writer emits into a sink.  They share `url_for_slot_with_port` and `visible_text`,
    /// but the streaming path now has its own slot-less arms — precisely when a differential pin
    /// earns its keep (`kind.rs::humanizer_and_humanize_into_agree` is the house precedent).
    #[test]
    fn ansi_hyperlink_matches_url_ansi_byte_for_byte() {
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let corpus = [
            ErrCode::new("E030"),                                 // Known, cacheable
            ErrCode::unknown("e030"),                             // resolved, non-cacheable
            ErrCode::unknown("E999"),                             // shape-valid, unregistered
            ErrCode::unknown("junk\u{1b}[31m"),                   // slot-less, controls
            ErrCode::unknown(format!("junk{}", "x".repeat(200))), // over-cap clean
            ErrCode::unknown(""),                                 // empty
        ];
        for code in &corpus {
            for port in [None, Some(41234u16)] {
                let via_cow = with_explain_port(port, || code.url_ansi().into_owned());
                let mut buf = String::new();
                with_explain_port(port, || {
                    code.write_ansi_hyperlink(&mut buf)
                        .expect("String sink never fails")
                });
                assert_eq!(via_cow, buf, "class {:?}, port {port:?}", code.code());
            }
        }
    }

    /// The plain URL is a pure function of the **canonical** spelling, so a resolved spelling must
    /// produce the registry code's URL byte for byte.
    ///
    /// This is the property `url_for_slot_with_port`'s port-unset arm rests on when it omits the
    /// `cacheable` guard ("structurally sound, not an oversight"); the argument lived in prose only.
    #[test]
    fn doc_url_is_identical_across_representations() {
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let known = with_explain_port(None, || ErrCode::new("E030").url().into_owned());
        let verbatim = with_explain_port(None, || ErrCode::unknown("e030").url().into_owned());
        assert_eq!(known, verbatim);
        assert!(
            known.starts_with("https://doc.posita-lang.org/error_codes/E030.html"),
            "{known}"
        );
    }

    /// `ptr::eq` ignores fat-pointer metadata, so the `PartialEq` fast path is sound **only**
    /// because `canonical()` returns whole allocations.
    ///
    /// If two distinct canonical spellings ever shared a data address — a `canonical()` that
    /// returned a trimmed slice, an interning scheme, a shared arena — they would compare equal
    /// and collapse in a `HashSet<ErrCode>`, silently suppressing a diagnostic.  The argument was
    /// prose only; this is the mechanical form, over the whole registry.
    #[test]
    fn ptr_eq_fast_path_never_conflates_distinct_codes() {
        let canon: Vec<&'static str> = super::CODE_INNER.iter().map(|e| e.canonical).collect();
        // Magnitude only guards against a misbound/empty table; the pairwise scan below is the
        // real assertion (the registry is generated, so its size is not this test's business).
        assert!(
            !canon.is_empty(),
            "registry corpus is empty — `CODE_INNER` is misbound"
        );
        for i in 0..canon.len() {
            for j in (i + 1)..canon.len() {
                if std::ptr::eq(canon[i], canon[j]) {
                    assert_eq!(
                        canon[i], canon[j],
                        "the ptr::eq fast path conflated two distinct registry codes"
                    );
                }
            }
        }
        // The intended `ptr::eq == true` case: two spellings of one registry code.
        assert_eq!(ErrCode::unknown("E030"), ErrCode::new("E030"));
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
        // The *rendered* consequence of accepting 0 is a deliberate trade, so pin it: the URL a
        // hyperlink shows is `http://127.0.0.1:0/<code>`, i.e. a link to nothing.  Rejecting 0
        // here would need a second error channel without preventing a bad URL, so the choice is
        // "accept verbatim and let the binder validate" — but a future "helpful" change treating
        // 0 as the documentation quadrant would silently rewrite every link, and this assertion is
        // what would fail first.
        assert_eq!(
            ErrCode::new("E030")
                .url_for_slot_with_port(Some(0))
                .as_ref(),
            "http://127.0.0.1:0/E030",
            "port 0 must render verbatim, not fall back to the docs host"
        );
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
        swallow_it::swallow!(lower.url_ansi());
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
    /// Property sweep for the injection firewall: the percent-encoder and the OSC 8
    /// framing.
    ///
    /// The hand-picked cases below pin the contract; these pin the *class*.  A
    /// deterministic PRNG (no new dependency) drives arbitrary byte strings —
    /// multibyte UTF-8, controls, `%`, `/`, `:`, quotes — through the encoder and
    /// checks the three properties the security argument rests on.
    #[test]
    fn encoder_and_osc_framing_hold_over_arbitrary_spellings() {
        fn decode(segment: &str) -> Vec<u8> {
            let b = segment.as_bytes();
            let mut out = Vec::with_capacity(b.len());
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'%' {
                    let hi = (b[i + 1] as char).to_digit(16).expect("hex") as u8;
                    let lo = (b[i + 2] as char).to_digit(16).expect("hex") as u8;
                    out.push(hi << 4 | lo);
                    i += 3;
                } else {
                    out.push(b[i]);
                    i += 1;
                }
            }
            out
        }

        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // A small alphabet with every hostile class, plus random UTF-8 bodies.
        let alphabet: &[&str] = &[
            "E", "0", "9", "/", ":", "%", " ", "\t", "\n", "\r", "\u{1b}", "\u{7}", "\u{0}",
            "\u{85}", "\u{2028}", "\u{202E}", "a", "Z", "~", "-", ".", "é", "中", "\"", "'", "<",
            ">", "&", "?", "#", "[", "]",
        ];
        for _ in 0..2_000 {
            // Up to 200 *components*, so spellings well past `MAX_CODE_CHARS` (128) are
            // generated: the previous `% 8` corpus never left the under-cap path, which
            // is exactly how the clean-over-cap arm of `sanitized()` went unnoticed.
            let n = (next() % 200) as usize;
            let mut spelling = String::new();
            for _ in 0..n {
                spelling.push_str(alphabet[(next() % alphabet.len() as u64) as usize]);
            }

            // (a) The segment decodes back to the exact input bytes.
            let encoded = encode_url_path_segment(&spelling);
            assert_eq!(
                decode(encoded.as_ref()),
                spelling.as_bytes(),
                "round trip failed for {spelling:?}"
            );
            // ...and contains nothing but unreserved characters and `%XX`.
            assert!(
                encoded
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-._~%".contains(&b)),
                "segment kept a hostile byte: {encoded:?}"
            );

            // (b) `url_ansi` is exactly one well-formed OSC 8 hyperlink: four ESCs,
            // two `]8;;` openers, and no BEL/CSI that could terminate it early.
            let code = super::ErrCode::unknown(spelling.as_str());
            let ansi = code.url_ansi();
            assert_eq!(
                ansi.matches('\u{1b}').count(),
                4,
                "expected 4 ESCs in {ansi:?}"
            );
            assert_eq!(ansi.matches("\u{1b}]8;;").count(), 2, "{ansi:?}");
            assert_eq!(ansi.matches("\u{1b}\\").count(), 2, "{ansi:?}");
            assert!(!ansi.contains('\u{7}'), "BEL in {ansi:?}");
            assert!(
                !ansi.replace("\u{1b}]8;;", "").contains("\u{1b}["),
                "CSI inside the hyperlink: {ansi:?}"
            );

            // (c) The emitted URL has no raw `/`, whitespace or control character
            // outside a `%XX` escape (a raw `/` would end the path segment).
            let url = code.url();
            let mut chars = url.chars().peekable();
            while let Some(c) = chars.next() {
                if c == '%' {
                    assert!(
                        chars.next().is_some_and(|h| h.is_ascii_hexdigit())
                            && chars.next().is_some_and(|l| l.is_ascii_hexdigit()),
                        "truncated escape in {url:?}"
                    );
                    continue;
                }
                let scheme_or_sep = c == '/' || c == ':'; // the fixed prefix only
                let prefix = url.starts_with("https://doc.posita-lang.org/error_codes/")
                    || url.starts_with("http://127.0.0.1:");
                assert!(
                    scheme_or_sep || !(c.is_control() || c.is_whitespace()),
                    "raw control/whitespace in {url:?}"
                );
                assert!(prefix || c != '/', "raw path separator in {url:?}");
            }
        }
    }

    /// `MAX_CODE_CHARS` must hold on the **clean** over-cap path too.
    ///
    /// `sanitized()`'s `None` arm means "nothing needed replacing", and it used to
    /// borrow `raw` — so an over-cap spelling with no control bytes (the common shape
    /// for junk input) flowed uncapped into `Display`, `visible_text()` and the ANSI
    /// link text, while the URL *segment* stayed capped: a 256-char label over a
    /// 128-char link.  The fuzz corpus never caught it because it generated at most
    /// seven characters.
    #[test]
    fn render_surfaces_cap_over_cap_clean_spellings() {
        let raw = "junk".repeat(64); // 256 chars, zero control bytes
        assert!(raw.chars().count() > MAX_CODE_CHARS);
        let code = super::ErrCode::unknown(raw.as_str());

        let shown = format!("{code}");
        assert_eq!(
            shown.chars().count(),
            MAX_CODE_CHARS + 1,
            "Display must cap at MAX_CODE_CHARS plus the ellipsis"
        );
        assert!(shown.ends_with('…'), "{shown:?}");
        assert_eq!(code.visible_text(), shown);

        // The hyperlink's *label* goes through the same surface; the segment inside
        // the OSC opener is capped independently by `encoded_segment`.
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ansi = code.url_ansi();
        assert!(
            ansi.ends_with("…\x1b]8;;\x1b\\"),
            "the hyperlink label must be capped too: {ansi:?}"
        );
    }

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
        // There is **no** generated `AsRef<str> for
        // ErrCode`.  `ErrCode::E030` (the const the call sites pass to
        // `with_code_str`/`new`) is a `&'static str` — the registry spelling — so those
        // calls pass a string and need no impl.  Asserted here so the distinction stays
        // visible: if the generator ever switched the constant to an `ErrCode`, this
        // stops compiling and the site is forced to decide.
        let spelling: &str = super::ErrCode::E030;
        assert_eq!(spelling, "E030");
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
            // The generated `hash` canary was deleted: it was a second copy of the
            // canonical spelling's FNV-1a value, compared here against `code_hash`,
            // while the comparison above (`canonical == entry.code`) already subsumes
            // it — and it cost 8 bytes in every registry entry's `.rodata`.
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
            // Byte (`br#"…"#`) and C (`cr#"…"#`) raw-string prefixes: the `r` index is
            // what `skip_raw_string` expects, and `skip_raw_string` itself is unchanged.
            // `cr` used to be missed — the `'r'` alone fails the identifier-boundary
            // guard above (`prev` is `'c'`, alphanumeric) — so a code-shaped literal
            // inside a `cr#` body was extracted as a real call site.  The `c"…"` form has
            // always been handled where the scanner sees the quote.
            b'b' | b'c' if bytes.get(i + 1) == Some(&b'r') => Some(i + 1),
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
    /// Skip a `"…"` literal, returning the index just past the closing quote.
    ///
    /// `i` points at the opening quote.  Escapes are honoured so a `\"` inside does
    /// not terminate the literal early — the same hazard `code_literals`' inline scan
    /// already handles for the plain `"` arm, shared here for the `b"…"`/`c"…"` arm.
    fn skip_string_literal(bytes: &[u8], i: usize) -> usize {
        let len = bytes.len();
        let mut j = i + 1;
        while j < len {
            match bytes[j] {
                b'\\' => j += 2,
                b'"' => return j + 1,
                _ => j += 1,
            }
        }
        len
    }

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
                // One escape-aware skip for every scanner: `skip_string_literal` returns past the
                // closing quote (or `len`), which is exactly where the common `i += 1` left this
                // inline copy.  The file's own doctrine is stated twice above ("three scanners
                // agree by maintenance only, and the failure mode is silent") — the same reason
                // `skip_block_comment` is shared.
                b'"' => {
                    i = skip_string_literal(bytes, i);
                    continue;
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
                // `b"…"` / `c"…"` cannot be passed to `ErrCode::new` (it takes
                // `impl AsRef<str>`), so their contents are data, not code spellings.
                // Skipping them here keeps the extraction faithful to what can
                // actually compile — a future bytes-taking callee would otherwise
                // inherit the false positives.
                // `opener_boundary_ok`, like every other literal-opener arm: an identifier
                // ending in `b`/`c` immediately followed by a quote (`my_cb"E030"`) is not a
                // byte-string opener.  That adjacency is impossible in valid Rust, so no real
                // call site changes behaviour — but this was the one arm where the "three
                // scanners agree" rule held by maintenance alone, which the file documents twice
                // as the shape that drifts.
                b'b' | b'c'
                    if i + 1 < bytes.len()
                        && bytes[i + 1] == b'"'
                        && opener_boundary_ok(bytes, i) =>
                {
                    i = skip_string_literal(bytes, i + 1);
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
            } else if (bytes[i] == b'b' || bytes[i] == b'c')
                && i + 1 < bytes.len()
                && bytes[i + 1] == b'"'
            {
                // `c"…"` is a plain C string literal and `code_literals` already
                // skips it; this blanking pass did not, so a callee name inside one
                // (`c"see ErrCode::new(\"E999\")"`) survived as "code" and made
                // `literal_codes_are_all_in_code_table` demand registration of a
                // spelling that is pure data.  `cr#"…"#` is handled by
                // `raw_string_opener` above; `c'…'` needs nothing (the char arm
                // blanks the content and a lone `c` cannot form a callee).
                opener_boundary_ok(bytes, i).then_some(i + 1)
            } else {
                None
            };
            if let Some(q) = str_quote {
                // Shared escape-aware skip (see `argument_span`'s note); `skip_string_literal`
                // already clamps to `len`, the `.min` is kept as the statement of that contract.
                i = skip_string_literal(bytes, q).min(bytes.len());
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

    /// Every accepted spelling, so the alias rule cannot drift away from its doc.
    #[test]
    fn from_str_accepts_every_documented_spelling() {
        for (spelling, expected) in [
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
            assert_eq!(
                spelling.parse::<ErrorCategory>().expect(spelling),
                expected,
                "{spelling:?}"
            );
            // Case-insensitivity is part of the contract.
            assert_eq!(
                spelling
                    .to_ascii_uppercase()
                    .parse::<ErrorCategory>()
                    .expect(spelling),
                expected,
                "{spelling:?} uppercase"
            );
        }
        // The rejected side: a canonical key with a *non*-Generic prefix plus "error" is not an
        // alias ("generic error" is not in the table, "parse error" is).
        assert!("generic error".parse::<ErrorCategory>().is_err());
        assert!("nonsense".parse::<ErrorCategory>().is_err());
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
        // Former over-match, now fixed: a *byte* or C-string literal (`b"E030"`,
        // `c"E030"`) is not a `&str` argument — `impl AsRef<str>` rejects those types,
        // so such a call cannot compile — and the scanner used to extract it anyway
        // because the `b`/`c` opener was skipped byte-wise.  Pinned in both directions
        // so the fix stays a decision instead of drifting back.
        assert_eq!(code_literals(r#""E030""#), vec!["E030"]);
        assert!(code_literals(r#"b"E030""#).is_empty());
        assert!(code_literals(r#"c"E030""#).is_empty());
        // A byte literal must not swallow the *next* real literal either.
        assert_eq!(code_literals(r#"b"x" ; "E031""#), vec!["E031"]);
        // Composed spellings are a *documented* miss: `concat!("E0", "30")` yields `"E0"` and
        // `"30"`, neither code-shaped, so the guard cannot see the code.  Same class as the
        // `Self::new` misses above (it only weakens the guard, it cannot corrupt a code); pinned so
        // a future matcher that starts extracting composed forms is a visible decision.
        assert!(code_literals(r#"concat!("E0", "30")"#).is_empty());

        // Known limitation, recorded rather than fixed: a `format!("{}", "E999")`
        // *inside* a scanned argument's literals would demand registration of a code
        // used purely as data.  The scanner reads the argument span textually, so the
        // literal is extracted; there is no in-tree instance, and the alternative
        // (parsing the argument expression) is a much larger commitment.
        //
        // The forms that *are* matched keep working.
        assert!(is_call_site("ErrCode::new(", 0));
    }

    /// (`ErrCode ::new`) and aliased imports are known misses; both only weaken
    /// the guard.
    #[test]
    fn literal_codes_are_all_in_code_table() {
        // Every crate root, not just `src`: a `with_code_str("E999")` in an
        // integration test, an example or a bench is a call site too, and the scan
        // used to exclude all three.
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        for root in ["src", "tests", "examples", "benches"] {
            let dir = manifest.join(root);
            if dir.is_dir() {
                rs_files(&dir, &mut files);
            }
        }

        let mut offenders = Vec::new();
        for file in &files {
            let Ok(src) = fs::read_to_string(file) else {
                continue;
            };
            // Scan the code-only view so callee names mentioned in comments or
            // literals (e.g. doc examples) are not treated as real call sites;
            // argument text is still extracted from the original `src`.
            let stripped = strip_non_code(&src);
            // `resolve`, not `lookup`: `ErrCode::new("e030")` succeeds at runtime
            // (case-insensitive), so a lowercase call-site literal compiles and works
            // while a `lookup`-based guard failed it.  `try_new` is scanned too — a
            // typo'd literal there failed only when that branch executed.
            for callee in ["with_code_str", "ErrCode::new", "ErrCode::try_new"] {
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
                                if resolve(lit).is_none() {
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
        swallow_it::swallow!(strip_non_code("/* outer /* inner"));
        swallow_it::swallow!(code_literals("/* outer /* \"E123\""));
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
        // Serialized against the port-flipping test: `set_explain_port(0)` is process-global, and
        // a flip between this benchmark's warm-up and its timed loop would move the measurement to
        // a cold quadrant and trip the sanity floor spuriously — the same comparability argument
        // the `ascii_dispatch` baselines apply.
        let _port = PORT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let code = ErrCode::new("E030");
        // Warm the slot so the timed loop measures the hit path only.
        swallow_it::swallow!(code.url_ansi());
        swallow_it::swallow!(code.url());
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
        assert_eq!(resolve("e030").map(|inner| inner.canonical), Some("E030"));
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

    /// `c"…"` is a plain C-string literal; blanking it keeps a callee name inside it
    /// from looking like a call site.
    ///
    /// `code_literals` already skipped `c"…"` and `raw_string_opener` covers
    /// `cr#"…"#`, but this pass's opener test only accepted `"` and `b"` — so
    /// `c"ErrCode::new(\"E999\")"` survived as "code" and
    /// `literal_codes_are_all_in_code_table` demanded registration of a spelling that
    /// is pure data.  The three scanners disagreeing is exactly the silent drift
    /// their docs warn about.
    #[test]
    fn strip_non_code_blanks_c_string_literals() {
        let src = r#"let s = c"ErrCode::new(\"E999\")"; let real = ErrCode::new("E030");"#;
        let stripped = strip_non_code(src);
        assert_eq!(stripped.len(), src.len(), "offsets must be preserved");
        assert_eq!(
            stripped.matches("ErrCode::new").count(),
            1,
            "only the real call site may survive: {stripped:?}"
        );
    }

    /// Inner-doc comments (`//!`) are item-level only, and are the one comment
    /// form a reader might assume this scanner misses.  They are not missed:
    /// `//!` is `/`, `/`, `!`, so the `//` arm matches on the *first* two bytes
    /// — the assumption comes from reading the marker as `/`, `!`.  Pinned
    /// anyway, because
    /// "the guard silently scans prose as code" is exactly the failure a future
    /// rewrite of the comment arms would reintroduce.
    #[test]
    fn strip_non_code_blanks_inner_doc_comments() {
        let src = "//! ErrCode::new(\"E999\")\nlet x = 1;\n";
        let stripped = strip_non_code(src);
        assert_eq!(stripped.len(), src.len(), "offsets must be preserved");
        assert_eq!(
            stripped.matches("ErrCode::new").count(),
            0,
            "inner-doc comment scanned as code: {stripped:?}"
        );
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
                resolve(&s).map(|inner| inner.canonical),
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
                resolve(&lower).map(|inner| inner.canonical),
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
