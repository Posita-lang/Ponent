/// Internationalisation support for diagnostic messages.
///
/// Messages live in `locales/*.ftl` (rustc-style Fluent), embedded at
/// compile time by `static_loader!`.  [`render`] resolves a stable slug
/// key (`type-mismatch`) through the active language's bundle — with
/// `en.ftl` as fallback — and returns any other key verbatim.
///
/// Language selection (in priority order):
///   1. `PONENT_LANG` environment variable (e.g. `en`, `ru`, `zh`)
///   2. `LANG` environment variable (system locale, e.g. `en_US.UTF-8`)
///   3. English (default)
use std::borrow::Cow;
use std::fmt::Display;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use fluent_bundle::{FluentArgs, FluentBundle, FluentResource, FluentValue};
use fluent_templates::{LanguageIdentifier, langid};

use crate::smallvec::SmallVec;

/// Build-time-generated typed functions for every Fluent slug in
/// `locales/en.ftl`.  Each function takes exactly the `$placeholders`
/// the slug declares, so a wrong key name or missing argument is a
/// compile error.  Use `fluent::type_mismatch(x, y)` instead of
/// `tr!("type-mismatch", expected = x, found = y)`.
pub mod fluent {
    include!(concat!(env!("OUT_DIR"), "/fluent_gen.rs"));
}

/// Wrapper marking a `tr!` argument that must be rendered through [`Display`]
/// (numbers, error types, values whose lifetime cannot be threaded through)
/// into an owned `Cow`.
///
/// `tr!` arguments are lifted into a `Cow<'_, str>` via [`TrValue`].  A
/// `&str`, `Cow<'_, str>` or `String` converts without re-formatting; any
/// other `Display` value must be wrapped explicitly so the (small)
/// allocation is visible at the call site rather than hidden in the macro.
pub struct DisplayArg<T>(pub T);

/// Lifts a `tr!` argument into a `Cow<'b, str>` without an unnecessary
/// `format!` when the value is already string-shaped.
///
/// The trait is parameterised over the value's own lifetime `'a` rather than
/// being fixed to `'static`: a borrowed argument (`&'src str`,
/// `Cow<'src, str>`) then stays borrowed for the duration of the render
/// call, instead of being forced through `to_string()` by a `'static`
/// return type.  `'a: 'b` lets the caller pick any shorter output lifetime.
pub trait TrValue<'a> {
    /// Convert into a `Cow<'b, str>` that borrows for at most `'a`.
    fn into_cow<'b>(self) -> Cow<'b, str>
    where
        'a: 'b;
}

impl<'a> TrValue<'a> for &'a str {
    fn into_cow<'b>(self) -> Cow<'b, str>
    where
        'a: 'b,
    {
        Cow::Borrowed(self)
    }
}

impl<'a> TrValue<'a> for Cow<'a, str> {
    fn into_cow<'b>(self) -> Cow<'b, str>
    where
        'a: 'b,
    {
        match self {
            Cow::Borrowed(s) => Cow::Borrowed(s),
            Cow::Owned(s) => Cow::Owned(s),
        }
    }
}

impl<'a> TrValue<'a> for String {
    fn into_cow<'b>(self) -> Cow<'b, str>
    where
        'a: 'b,
    {
        Cow::Owned(self)
    }
}

impl<'a, T: Display> TrValue<'a> for DisplayArg<T> {
    fn into_cow<'b>(self) -> Cow<'b, str>
    where
        'a: 'b,
    {
        Cow::Owned(self.0.to_string())
    }
}

/// Lift `Cow<'static, str>` to `Cow<'a, str>` without allocation.
///
/// `std` does not provide `From<Cow<'static, str>> for Cow<'a, str>` when
/// `'a != 'static`, so `tr!(...)` results (which are `Cow<'static, str>`)
/// cannot be passed directly to APIs expecting `Cow<'src, str>`.  The old
/// workaround `.into_owned().into()` allocated a `String` even when the
/// `Cow` was `Borrowed`.  This trait bridges the gap by matching on the
/// variant:
/// - `Borrowed(&'static str)` coerces to `Borrowed(&'a str)` via `'static: 'a`
/// - `Owned(String)` is moved as-is
pub trait LiftCow<'a> {
    fn lift(self) -> Cow<'a, str>;
}

impl<'a> LiftCow<'a> for Cow<'static, str> {
    fn lift(self) -> Cow<'a, str> {
        match self {
            Cow::Borrowed(s) => Cow::Borrowed(s),
            Cow::Owned(s) => Cow::Owned(s),
        }
    }
}

// Compile-time locale embedding: `include_str!` every file, so no runtime
// file access is needed (the offline explain server relies on that).
// We build the `FluentBundle`s ourselves instead of going through
// `fluent_templates`' `Loader`: its `try_lookup_with_args` takes a
// `HashMap` of args and re-copies every key/value into a `FluentArgs`
// internally, and it formats before we can even probe whether the slug
// exists.  Holding the bundles directly lets us probe with `get_message`
// (O(1), no formatting), then build `FluentArgs` straight from `pairs`
// (Vec-backed, linear search — the right shape for ≤4 args).  Isolating
// marks must be off: our renderer and byte-exact tests expect plain
// substituted text.  Order is [en, ru, zh] — index 0 is the fallback.
type Bundle = fluent_bundle::concurrent::FluentBundle<&'static FluentResource>;

static BUNDLES: OnceLock<[Bundle; N_LANGS]> = OnceLock::new();

fn bundles() -> &'static [Bundle; N_LANGS] {
    BUNDLES.get_or_init(|| {
        const EN_FTL: &str = include_str!("../../locales/en.ftl");
        const RU_FTL: &str = include_str!("../../locales/ru.ftl");
        const ZH_FTL: &str = include_str!("../../locales/zh.ftl");
        [
            make_bundle(langid!("en"), EN_FTL),
            make_bundle(langid!("ru"), RU_FTL),
            make_bundle(langid!("zh"), ZH_FTL),
        ]
    })
}

fn make_bundle(lang: LanguageIdentifier, src: &'static str) -> Bundle {
    // Locales are validated by `validate_locales` at build time, so parse
    // errors should not occur; keep the (recoverable) resource if they do.
    let res: &'static FluentResource = Box::leak(Box::new(match FluentResource::try_new(src.to_owned()) {
        Ok(r) => r,
        Err((r, _)) => r,
    }));
    let mut bundle = fluent_bundle::concurrent::FluentBundle::new_concurrent(vec![lang]);
    bundle.set_use_isolating(false);
    bundle.add_resource(res).expect("conflicting Fluent message ids");
    bundle
}

fn bundle_for(lang: &LanguageIdentifier) -> &'static Bundle {
    let bundles = bundles();
    // Matches the `language` subtag only: a `zh-Hant` identifier resolves to
    // the `zh` (Simplified) bundle, and `en-GB` to `en`.  That is correct for
    // the `Lang` enum path (only `Zh` exists) but would silently serve the
    // wrong script for an arbitrary `format_slug(langid!("zh-Hant"), …)`; add
    // a `ZhHant` locale + a script-aware match before exposing that.
    bundles
        .iter()
        .find(|b| b.locales[0].language == lang.language)
        .unwrap_or(&bundles[0])
}

/// Resolve + format `slug` in `lang`'s bundle, falling back to the `en`
/// bundle — the same chain the loader used: unknown slug or a format
/// error (e.g. a missing referenced argument) moves on to `en`, and `None`
/// only when `en` fails too, so the caller returns `msg` verbatim.
///
/// Not test-only: the generated `fluent_gen.rs` calls `super::format_slug`
/// for the typed zero/one-arg accessors.
fn format_slug(
    lang: &LanguageIdentifier,
    slug: &str,
    args: Option<&FluentArgs>,
) -> Option<String> {
    let mut out = String::new();
    format_slug_into(lang, slug, args, &mut out).then_some(out)
}

/// [`format_slug`] writing the formatted pattern straight into `out` —
/// `format_pattern` yields a `Cow<str>` that is `Borrowed` for every
/// substitution-free pattern, so routing it through an intermediate
/// `String` copied one more `into_owned()` + `push_str()` per render.
/// Returns `false` when the slug is unknown even after the `en` fallback
/// (or a format error occurred); `out` is then left untouched.
fn format_slug_into(
    lang: &LanguageIdentifier,
    slug: &str,
    args: Option<&FluentArgs>,
    out: &mut String,
) -> bool {
    let bundles = bundles();
    let primary = bundle_for(lang);
    let en = &bundles[0];

    // Hoisted out of `try_bundle`: a fresh `Vec::new()` per attempt is free
    // on the success path (lazy allocation) but pays a fresh `Drop` per
    // bundle attempt on the fallback path; one buffer reused across the
    // chain removes both.
    let mut errors = Vec::new();
    let mut try_bundle = |b: &Bundle, out: &mut String| -> bool {
        errors.clear();
        let Some(pattern) = b.get_message(slug).and_then(|m| m.value()) else {
            return false;
        };
        let value = b.format_pattern(pattern, args, &mut errors);
        if errors.is_empty() {
            out.push_str(&value);
            true
        } else {
            false
        }
    };

    let chain = std::iter::once(primary)
        .chain((!std::ptr::eq(primary, en)).then_some(en));
    for b in chain {
        if try_bundle(b, out) {
            return true;
        }
    }
    false
}

/// `LANG` sentinel meaning "not yet detected".
const LANG_UNSET: u8 = 0;
const LANG_EN: u8 = 1;
const LANG_RU: u8 = 2;
const LANG_ZH: u8 = 3;

const _: () = {
    assert!(Lang::En as u8 != LANG_UNSET);
    assert!(Lang::Ru as u8 != LANG_UNSET);
    assert!(Lang::Zh as u8 != LANG_UNSET);
    assert!(matches!(Lang::from_u8(LANG_EN), Some(Lang::En)));
    assert!(matches!(Lang::from_u8(LANG_RU), Some(Lang::Ru)));
    assert!(matches!(Lang::from_u8(LANG_ZH), Some(Lang::Zh)));
};

/// Number of supported languages. Single source of truth for the bundle
/// table, the language-id cache and the `cached_label!` arrays (A5) — adding
/// a `Lang` variant previously meant hunting three separate literals.
pub(crate) const N_LANGS: usize = 3;

static LANG: AtomicU8 = AtomicU8::new(LANG_UNSET);

/// Guards tests that mutate the process-global [`LANG`]: parallel tests
/// share one atomic, so a test that sets a language must hold this lock
/// while it renders.
#[cfg(test)]
static TEST_LANG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`TEST_LANG_LOCK`].  Poison is irrelevant for `()`, so a
/// panicking test cannot wedge the others.
#[cfg(test)]
pub(crate) fn acquire_test_lang_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LANG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sets `Lang` for the lifetime of the guard and restores the previous value
/// on drop, including on panic unwind.
#[cfg(test)]
pub(crate) struct LangGuard(Lang);

#[cfg(test)]
impl LangGuard {
    pub(crate) fn new(lang: Lang) -> Self {
        let previous = Lang::current();
        Lang::set(lang);
        Self(previous)
    }
}

#[cfg(test)]
impl Drop for LangGuard {
    fn drop(&mut self) {
        Lang::set(self.0);
    }
}

/// Supported languages.
///
/// The discriminants are the values stored in [`LANG`]; they are deliberately
/// aligned with the `LANG_*` constants (and not the default 0/1/2) so that
/// `lang as u8` round-trips through [`Lang::from_u8`].  Previously `Ru as u8`
/// was `1`, which collided with the "English" load arm and silently reverted
/// Russian/Chinese to English on the second lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Lang {
    En = LANG_EN,
    Ru = LANG_RU,
    Zh = LANG_ZH,
}

impl Lang {
    pub fn current() -> Self {
        let cached = LANG.load(Ordering::Relaxed);
        if cached != LANG_UNSET {
            return Self::from_u8(cached).unwrap_or(Lang::En);
        }
        // First access: detect from environment.  `compare_exchange` keeps an
        // explicit `set()` from being overwritten by a concurrent detection:
        // the loser discards its detected value instead of storing it, so
        // `set()` wins regardless of interleaving.
        let lang = detect_from_env();
        match LANG.compare_exchange(LANG_UNSET, lang as u8, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => lang,
            Err(explicit) => Self::from_u8(explicit).unwrap_or(Lang::En),
        }
    }

    /// Zero-based index of this language, for generated caches.
    ///
    /// The generated `ZERO_ARG_TABLE` indexes by this, not by
    /// `as usize - 1`: `En` may take the default discriminant 0, and the old
    /// subtraction underflowed.
    #[inline]
    pub fn index(self) -> usize {
        match self {
            Lang::En => 0,
            Lang::Ru => 1,
            Lang::Zh => 2,
        }
    }

    pub fn set(lang: Lang) {
        LANG.store(lang as u8, Ordering::Relaxed);
    }

    /// Decode a raw `LANG` value.  Returns `None` for the unset sentinel and
    /// any value outside the known discriminants.
    #[inline]
    pub const fn from_u8(n: u8) -> Option<Self> {
        match n {
            LANG_EN => Some(Lang::En),
            LANG_RU => Some(Lang::Ru),
            LANG_ZH => Some(Lang::Zh),
            _ => None,
        }
    }
}

fn detect_from_env() -> Lang {
    let explicit = std::env::var("PONENT_LANG").ok();
    let fallback = std::env::var("LANG").ok();
    detect_from_pair(explicit.as_deref(), fallback.as_deref())
}

fn detect_from_pair(explicit: Option<&str>, fallback: Option<&str>) -> Lang {
    match (explicit, fallback) {
        (Some(p), _) => detect_from_str(p),
        (None, Some(l)) => detect_from_str(l),
        (None, None) => Lang::En,
    }
}

/// Pure BCP-47 primary-subtag classification, split out so it is testable
/// without mutating the process environment.
fn detect_from_str(env: &str) -> Lang {
    // Compare the BCP-47 *primary* subtag exactly: a prefix test would let
    // `zha` (Zhuang, ISO 639-3) match `zh`, and `ru*` collides similarly.
    let primary = env.split(['-', '_']).next().unwrap_or("");
    if primary.eq_ignore_ascii_case("ru") || primary.eq_ignore_ascii_case("rus") {
        Lang::Ru
    } else if primary.eq_ignore_ascii_case("zh") || primary.eq_ignore_ascii_case("zho") {
        Lang::Zh
    } else {
        Lang::En
    }
}

/// Render `msg` with `pairs` substituted.
///
/// A Fluent slug present in `locales/*.ftl` resolves to the localized
/// message; anything else (unknown slug, non-slug legacy key) is returned
/// verbatim — placeholders are NOT substituted on that fallback path, so
/// callers must use slugs.
///
/// The fallback path borrows `msg` directly (`Cow::Borrowed`), so unknown
/// slugs and legacy keys never allocate — the caller's output buffer is the
/// only copy. Only the Fluent-resolved branch returns `Cow::Owned`.
pub fn render<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
) -> Cow<'msg, str> {
    if is_ftl_slug(msg) {
        // Zero-arg slug: build-time cached, returns borrowed — no formatting
        // and no allocation after the first call per (slug, lang).
        if let Some(c) = fluent::zero_arg_slug(msg) {
            return c;
        }
        if let Some(s) = try_fluent(msg, pairs) {
            return Cow::Owned(s);
        }
    }
    Cow::Borrowed(msg)
}

/// [`render`] into an existing buffer.
pub fn render_into<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
    out: &mut String,
) {
    if is_ftl_slug(msg) {
        if let Some(c) = fluent::zero_arg_slug(msg) {
            out.push_str(&c);
            return;
        }
        if try_fluent_into(msg, pairs, out) {
            return;
        }
    }
    out.push_str(msg);
}

/// FTL text identifiers are `[A-Za-z][A-Za-z0-9_-]*`; every legacy English
/// key contains spaces, backticks or braces, so the two key spaces cannot
/// collide and dispatch is unambiguous.
fn is_ftl_slug(msg: &str) -> bool {
    !msg.is_empty()
        && msg.as_bytes().iter().enumerate().all(|(i, b)| {
            if i == 0 {
                b.is_ascii_alphabetic()
            } else {
                b.is_ascii_alphanumeric() || *b == b'-' || *b == b'_'
            }
        })
}

/// `LanguageIdentifier` for the `i`-th supported language (0 = en fallback).
pub(crate) fn lang_id_at(i: usize) -> &'static LanguageIdentifier {
    // Same loud-failure stance as `cached_label!`: a bad index is a programmer
    // error in the established `discriminant - 1` mapping, so flag it by name
    // in debug and let the slice bounds check panic in release rather than
    // silently serving a different locale.
    debug_assert!(i < N_LANGS, "language index {i} out of range 0..{N_LANGS}");
    static IDS: std::sync::OnceLock<[LanguageIdentifier; N_LANGS]> = std::sync::OnceLock::new();
    let ids = IDS.get_or_init(|| [langid!("en"), langid!("ru"), langid!("zh")]);
    &ids[i]
}

fn lang_id() -> &'static LanguageIdentifier {
    // `Lang` discriminants start at 1 (`LANG_EN`), so the table index is
    // `discriminant - 1`.  A future 0-discriminant variant underflows and must
    // fail loudly instead of aliasing another language.
    let lang_idx = Lang::current().index();
    lang_id_at(lang_idx)
}

/// `None` when the slug is unknown even after the loader's fallback chain
/// (en) — or when a referenced argument is missing (the loader then reports
/// an unformattable pattern) — so the caller returns `msg` verbatim.
/// Argument keys whose values are semantically counts: only these need to
/// become Fluent numbers so plural selectors (`summary-errors`) see real
/// CLDR categories.  Everything else is substituted as plain text —
/// `try_number` silently coerced numeric-looking strings like `"00123"`
/// or `"1.0"` into numbers, corrupting values that are names, not counts.
///
/// Generated (not hardcoded): the set is every `$var` used as a select /
/// plural selector in `en.ftl` plus the legacy count names, so a future
/// locale message introducing a differently-named plural key cannot
/// silently fall to the `other` CLDR branch — the build regenerates the
/// list from the source of truth on every compile.
use fluent::is_numeric_arg_key;

fn make_value<'s>(v: Cow<'s, str>, key: &str) -> FluentValue<'s> {
    if !is_numeric_arg_key(key) {
        return FluentValue::String(v);
    }
    match v {
        Cow::Borrowed(s) => FluentValue::try_number(s),
        Cow::Owned(s) => match s.parse::<fluent_bundle::types::FluentNumber>() {
            Ok(n) => n.into(),
            Err(_) => FluentValue::String(Cow::Owned(s)),
        },
    }
}

fn try_fluent<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
) -> Option<String> {
    let mut out = String::new();
    try_fluent_into(msg, pairs, &mut out).then_some(out)
}

/// [`try_fluent`] writing the formatted pattern straight into `out` —
/// one copy total instead of `into_owned()` + a second `push_str()`.
/// Returns `false` when no bundle produced a clean rendering; `out` is
/// then left untouched.
fn try_fluent_into<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
    out: &mut String,
) -> bool {
    let lang = lang_id();
    if pairs.is_empty() {
        // Zero-argument fast path: one lookup per bundle attempt, no
        // argument map.
        return format_slug_into(lang, msg, None, out);
    }
    // Single-pass resolve+format: walk the bundle chain once, and build the
    // argument map lazily — only when a bundle actually has the message.
    // The old `slug_exists` probe + `format_slug` paid two `get_message`
    // hash lookups per bundle for the same answer.
    let bundles = bundles();
    let primary = bundle_for(lang);
    let en = &bundles[0];
    let build_args = || {
        let mut args = FluentArgs::new();
        pairs.iter().for_each(|(k, v)| {
            // `v.clone()` — not `FluentValue::String(v)` by value. `pairs`
            // arrives as `&[...]` borrowed from `DiagMessage::render_into(&self)`,
            // so `v` is behind a shared reference and cannot be moved (E0507).
            // Taking `pairs` by value would force `render_into` to own its
            // message, breaking the `&self` API (a diagnostic may render more
            // than once) and rippling through every caller. Under `&self`,
            // `clone` is the cheapest legal option: `Cow::Borrowed` copies a
            // pointer (zero heap, the common `&str`-arg path); only `Cow::Owned`
            // deep-copies — unavoidable without ownership.
            let value = make_value(v.clone(), k);
            args.set(*k, value);
        });
        args
    };
    // P2: resolve each bundle in turn and *retry formatting* on the next one
    // when a bundle has the message but fails to format it (a broken
    // translation, a missing argument in a translation, or a malformed
    // pattern).  The old `find_map` stopped at the first bundle with the
    // message, so one bad localized translation fell all the way back to the
    // raw slug even when `en` would have formatted cleanly.  The argument map
    // is still built at most once.
    let mut args: Option<FluentArgs> = None;
    let mut errors = Vec::new();
    let chain = std::iter::once(primary).chain((!std::ptr::eq(primary, en)).then_some(en));
    for b in chain {
        let Some(pattern) = b.get_message(msg).and_then(|m| m.value()) else {
            continue;
        };
        let args = args.get_or_insert_with(|| build_args());
        errors.clear();
        let val = b.format_pattern(pattern, Some(args), &mut errors);
        if errors.is_empty() {
            out.push_str(&val);
            return true;
        }
    }
    // Cold path only: a slug that exists in `en` yet fails to format there is
    // a caller/translation bug (bad argument key), not an unknown slug.
    debug_assert!(
        bundles[0].get_message(msg).is_none(),
        "slug {msg:?} exists but formatting failed on every bundle — \
         missing or misspelled argument? errors: {errors:?}"
    );
    false
}

/// Consuming variant of [`render_into`]: moves `pairs` by value so
/// `FluentValue::String(v)` takes ownership of each `Cow` instead of
/// cloning — `Cow::Borrowed` was already cheap (pointer copy), but
/// `Cow::Owned` now skips its `String::clone` entirely.  Called only
/// from `DiagMessage::render_into_owned`, which in turn is called only
/// from `humanize` (the kind is `take`n, so the message is fresh-owned).
pub fn render_into_owned<'msg, 'src>(
    msg: &'msg str,
    pairs: SmallVec<[(&'static str, Cow<'src, str>); 4]>,
    out: &mut String,
) {
    if is_ftl_slug(msg) {
        if let Some(c) = fluent::zero_arg_slug(msg) {
            out.push_str(&c);
            return;
        }
        if try_fluent_owned_into(msg, pairs, out) {
            return;
        }
    }
    out.push_str(msg);
}

/// [`render_into_owned`] for a caller that already knows `msg` is an FTL
/// slug (`DiagMessage::Slug`), skipping the `is_ftl_slug` byte scan.
pub fn render_into_slug<'msg, 'src>(
    msg: &'msg str,
    pairs: SmallVec<[(&'static str, Cow<'src, str>); 4]>,
    out: &mut String,
) {
    if let Some(c) = fluent::zero_arg_slug(msg) {
        out.push_str(&c);
        return;
    }
    if try_fluent_owned_into(msg, pairs, out) {
        return;
    }
    out.push_str(msg);
}

fn try_fluent_owned_into<'msg, 'src>(
    msg: &'msg str,
    pairs: SmallVec<[(&'static str, Cow<'src, str>); 4]>,
    out: &mut String,
) -> bool {
    let lang = lang_id();
    if pairs.is_empty() {
        return format_slug_into(lang, msg, None, out);
    }
    // Single-pass resolve (see `try_fluent`): one `get_message` hash lookup
    // per bundle.  The argument map is built once, only after a bundle with
    // the message is found — `pairs` is consumed, so the build cannot sit
    // inside the loop.
    let bundles = bundles();
    let primary = bundle_for(lang);
    let en = &bundles[0];
    let chain = std::iter::once(primary).chain((!std::ptr::eq(primary, en)).then_some(en));
    // Retry formatting on the next bundle when this one has the message but
    // fails to format it (see `try_fluent_into`).  `pairs` is consumed to
    // build the map, so build it lazily once and reuse it for the fallback
    // attempt instead of trying to reconstruct it.
    let mut pending_pairs = Some(pairs);
    let mut built_args: Option<FluentArgs> = None;
    let mut errors = Vec::new();
    for b in chain {
        let Some(pattern) = b.get_message(msg).and_then(|m| m.value()) else {
            continue;
        };
        let args = built_args.get_or_insert_with(|| {
            let mut args = FluentArgs::new();
            pending_pairs
                .take()
                .expect("pairs are taken only when the map is first built")
                .into_iter()
                .for_each(|(k, v)| {
                    // Owned path: `FluentValue::try_number` borrows the input,
                    // but `v` is consumed here, so parse into an owned
                    // `FluentNumber` instead.  The policy is the same as the
                    // borrowed path (`try_number` is `parse::<FluentNumber>` with
                    // a `String` fallback).
                    let value = make_value(v, &k);
                    args.set(k, value);
                });
            args
        });
        errors.clear();
        let val = b.format_pattern(pattern, Some(args), &mut errors);
        if errors.is_empty() {
            out.push_str(&val);
            return true;
        }
    }
    // Same cold-path probe as `try_fluent_into`: a slug that exists in `en`
    // yet fails to format there is a caller/translation bug.
    debug_assert!(
        bundles[0].get_message(msg).is_none(),
        "slug {msg:?} exists but formatting failed — missing or \
         misspelled argument? errors: {errors:?}"
    );
    false
}

/// Direct-indexed per-`Lang` cache for a fixed table of zero-argument
/// Fluent labels.  Each invocation returns `Cow::Borrowed` pointing at the
/// cached `String`, so the per-diagnostic label lookups
/// (`DiagnosticLevel::label`, `ChainEntryKind::as_str`,
/// `ComptimeReason::as_str`, `ErrorCategory::label`) allocate once per
/// language instead of once per call — turning `prefix_cols`'s
/// allocate-just-to-measure-`.len()` into a borrowed slice length too.
macro_rules! cached_label {
    (
        n = $n:literal,
        variant = $variant:expr,
        init = [$($init:expr),+ $(,)?],
    ) => {{
        static CACHES: [::std::sync::OnceLock<[::std::string::String; $n]>; $crate::diagnostics::i18n::N_LANGS] =
            [const { ::std::sync::OnceLock::new() }; $crate::diagnostics::i18n::N_LANGS];
        let lang_idx = $crate::diagnostics::i18n::Lang::current().index();
        // `Lang` starts at discriminant 1 (`LANG_EN = 1`), so the index is
        // `discriminant - 1` and must stay in `0..3`.  A new variant with
        // discriminant 0 would wrap to `usize::MAX`; the debug assert names
        // that, and in release the bounds check below panics loudly rather
        // than silently serving the wrong language (the old `.min(2)` coerced
        // a bad index to Chinese — strictly worse than a panic).
        debug_assert!(
            lang_idx < $crate::diagnostics::i18n::N_LANGS,
            "Lang discriminant out of the cached range"
        );
        let table = CACHES[lang_idx].get_or_init(|| [$($init),+]);
        ::std::borrow::Cow::Borrowed(&table[$variant])
    }};
}
pub(crate) use cached_label;

/// Zero-wrapper variant of [`cached_label!`]: returns `&'static str` directly.
///
/// `cached_label!` always produces `Cow::Borrowed` (the `init` expressions are
/// `into_owned()` `String`s stored in a `OnceLock<[String; N]>`, so the borrow
/// is `'static`), making the `Owned` branch dead and the `Cow` a 32-byte shell
/// around a `&'static str`.  This macro skips the `Cow` entirely, returning
/// `table[variant].as_str()` with the `'static` lifetime of the cached table.
macro_rules! cached_label_static {
    (
        n = $n:literal,
        variant = $variant:expr,
        init = [$($init:expr),+ $(,)?],
    ) => {{
        static CACHES: [::std::sync::OnceLock<[::std::string::String; $n]>; $crate::diagnostics::i18n::N_LANGS] =
            [const { ::std::sync::OnceLock::new() }; $crate::diagnostics::i18n::N_LANGS];
        let lang_idx = $crate::diagnostics::i18n::Lang::current().index();
        // Same invariant as `cached_label!`: `LANG_EN = 1` anchors the
        // `discriminant - 1` index in `0..3`; a 0-discriminant variant would
        // wrap and must fail the debug assertion, not index garbage in
        // release (the bounds check panics; no `.min` coercion).
        debug_assert!(
            lang_idx < $crate::diagnostics::i18n::N_LANGS,
            "Lang discriminant out of the cached range"
        );
        let table = CACHES[lang_idx].get_or_init(|| [$($init),+]);
        table[$variant].as_str()
    }};
}
pub(crate) use cached_label_static;

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// Regression for the discriminant/sentinel collision: every language must
    /// encode to a value that decodes back to itself, and the unset sentinel
    /// must never decode to a real language.  Previously `Ru as u8 == 1`,
    /// which `from_u8` (and the old `current` match) mapped back to `En`.
    #[test]
    fn lang_encoding_round_trips() {
        for lang in [Lang::En, Lang::Ru, Lang::Zh] {
            assert_eq!(
                Lang::from_u8(lang as u8),
                Some(lang),
                "{lang:?} does not round-trip through its discriminant"
            );
        }
        assert_eq!(Lang::from_u8(LANG_UNSET), None);
    }

    /// A slug that is known but whose arguments are missing (or an unknown
/// slug) must report failure and leave `out` untouched — the caller then
/// falls back to the verbatim message instead of rendering a bare slug or
/// partial text.
#[test]
fn format_slug_into_degrades_without_touching_out() {
    let en = lang_id_at(0);
    let mut out = String::from("prefix");
    assert!(!format_slug_into(en, "no-such-slug", None, &mut out));
    assert_eq!(out, "prefix");
    // `type-mismatch` needs two args; formatting without them fails.
    assert!(!format_slug_into(en, "type-mismatch", None, &mut out));
    assert_eq!(out, "prefix");
}

#[test]
    fn detect_from_str_uses_primary_subtag_exactly() {
        // Documented inputs resolve identically.
        assert_eq!(detect_from_str("en"), Lang::En);
        assert_eq!(detect_from_str("ru_RU"), Lang::Ru);
        assert_eq!(detect_from_str("ru-RU"), Lang::Ru);
        assert_eq!(detect_from_str("zh_TW"), Lang::Zh);
        assert_eq!(detect_from_str("zh-Hant"), Lang::Zh);
        // `zha` is Zhuang, not Chinese: a prefix test would misclassify it.
        assert_eq!(detect_from_str("zha_CN"), Lang::En);
        assert_eq!(detect_from_str("rux"), Lang::En);
        assert_eq!(detect_from_str("rus_RU"), Lang::Ru);
        assert_eq!(detect_from_str("zho_TW"), Lang::Zh);
        // Unset / POSIX locale sentinels.
        assert_eq!(detect_from_str(""), Lang::En);
        assert_eq!(detect_from_str("C"), Lang::En);

        // PONENT_LANG wins even when empty (explicit En); absent falls to LANG.
        assert_eq!(detect_from_pair(Some("ru"), Some("zh_TW")), Lang::Ru);
        assert_eq!(detect_from_pair(None, Some("zho_CN")), Lang::Zh);
        assert_eq!(detect_from_pair(Some(""), Some("ru")), Lang::En);
        assert_eq!(detect_from_pair(None, None), Lang::En);
    }

    /// The zero-arg slug cache must be language-aware: fetching under `En`,
    /// switching to `Zh`, and fetching again must change the output.  If the
    /// generated table ever latched a single language, this fails.
    #[test]
    fn zero_arg_slug_is_language_aware() {
        let _lang_guard = acquire_test_lang_lock();
        Lang::set(Lang::En);
        let en = fluent::zero_arg_slug("main-not-found").expect("main-not-found exists");
        Lang::set(Lang::Zh);
        let zh = fluent::zero_arg_slug("main-not-found").expect("main-not-found exists");
        assert_ne!(en, zh, "zero-arg slug cache did not follow Lang::set");
        Lang::set(Lang::En);
    }

    /// `render` under a non-EN latch must produce localised, parameterised
    /// output — the end-to-end path the RU/ZH direct-bundle test bypasses.
    #[test]
    fn render_localises_under_explicit_lang() {
        let _lang_guard = acquire_test_lang_lock();
        Lang::set(Lang::Ru);
        assert_eq!(
            render("main-not-found", &[]),
            "функция `main` не найдена в крейте"
        );
        Lang::set(Lang::En);
    }

    /// `render_into` must leave `out` untouched when the slug is unknown.
    #[test]
    fn render_into_unknown_slug_leaves_buffer_untouched() {
        let _lang_guard = acquire_test_lang_lock();
        Lang::set(Lang::En);
        let mut out = String::from("prefix:");
        render_into("definitely-unknown-slug", &[], &mut out);
        assert_eq!(out, "prefix:definitely-unknown-slug");
    }

    /// The stored discriminants must stay aligned with the `LANG_*` sentinels.
    #[test]
    fn lang_discriminants_match_sentinels() {
        assert_eq!(Lang::En as u8, LANG_EN);
        assert_eq!(Lang::Ru as u8, LANG_RU);
        assert_eq!(Lang::Zh as u8, LANG_ZH);
        assert_ne!(LANG_UNSET, Lang::En as u8);
    }

    /// Slug messages substitute their arguments through Fluent — a value
    /// that itself contains a placeholder is emitted verbatim, never
    /// re-expanded by a second pass.
    #[test]
    fn render_slug_substitutes_arguments_once() {
        let _lang_guard = acquire_test_lang_lock();
        Lang::set(Lang::En);
        let pairs: &[(&'static str, Cow<'static, str>)] = &[
            ("expected", Cow::Borrowed("{found}")),
            ("found", Cow::Borrowed("Int<32>")),
        ];
        assert_eq!(
            render("type-mismatch", pairs),
            "type mismatch: expected `{found}`, found `Int<32>`"
        );
        let pairs: &[(&'static str, Cow<'static, str>)] = &[
            ("expected", Cow::Borrowed("Int<32>")),
            ("found", Cow::Borrowed("Bool")),
        ];
        assert_eq!(
            render("type-mismatch", pairs),
            "type mismatch: expected `Int<32>`, found `Bool`"
        );
    }

    /// Unknown keys (slug or legacy) return the raw message unchanged.
    #[test]
    fn render_unknown_key_falls_back_to_raw_message() {
        let _lang_guard = acquire_test_lang_lock();
        Lang::set(Lang::En);
        assert_eq!(
            render("no placeholders here", &[("x", Cow::Borrowed("ignored"))]),
            "no placeholders here"
        );
        assert_eq!(render("no-such-slug", &[]), "no-such-slug");
    }

    /// CLDR plural categories must drive the summary slugs: one vs other,
    /// plus the mixed/warning variants used by the renderers.
    #[test]
    fn render_summary_errors_plural() {
        let _lang_guard = acquire_test_lang_lock();
        Lang::set(Lang::En);
        assert_eq!(
            render("summary-errors", &[("count", Cow::Borrowed("1"))]),
            "aborting due to previous error"
        );
        assert_eq!(
            render("summary-errors", &[("count", Cow::Borrowed("3"))]),
            "aborting due to 3 previous errors"
        );
        assert_eq!(
            render("summary-mixed-errors", &[("count", Cow::Borrowed("1"))]),
            "aborting due to 1 previous error"
        );
        assert_eq!(
            render("summary-mixed-errors", &[("count", Cow::Borrowed("2"))]),
            "aborting due to 2 previous errors"
        );
        assert_eq!(
            render("summary-warnings", &[("count", Cow::Borrowed("1"))]),
            "1 warning emitted"
        );
        assert_eq!(
            render("summary-warnings", &[("count", Cow::Borrowed("4"))]),
            "4 warnings emitted"
        );
    }

    /// The RU/ZH bundles must parse and resolve — loader parsing is lazy,
    /// so neither compilation nor the EN tests would catch a syntax error
    /// in a translated file.  Uses the loader directly (not `render`) to
    /// avoid racing the global `Lang` against parallel tests.
    #[test]
    fn ru_and_zh_bundles_parse_and_render() {
        let no_args = FluentArgs::new();
        assert_eq!(
            format_slug(&langid!("ru"), "main-not-found", Some(&no_args)),
            Some("функция `main` не найдена в крейте".to_string())
        );
        assert_eq!(
            format_slug(&langid!("zh"), "main-not-found", Some(&no_args)),
            Some("crate 中没有找到 `main` 函数".to_string())
        );
        let mut count = FluentArgs::new();
        count.set("count", FluentValue::try_number("2"));
        assert_eq!(
            format_slug(&langid!("ru"), "summary-errors", Some(&count)),
            Some("прерывание из-за 2 предыдущих ошибок".to_string())
        );
        assert_eq!(
            format_slug(&langid!("zh"), "summary-errors", Some(&count)),
            Some("因之前的 2 个错误中止".to_string())
        );
    }

    /// Parameterised messages must substitute their arguments byte-exactly
    /// (regression guard: the Fluent port must reproduce the old table
    /// values, which are now dead and gone).
    #[test]
    fn render_parameterised_messages_substitute_arguments() {
        let _lang_guard = acquire_test_lang_lock();
        Lang::set(Lang::En);
        let cases: [(&str, &[(&str, &str)], &str); 6] = [
            (
                "missing-method",
                &[("method", "to_bits"), ("trait", "BitCast")],
                "impl missing method `to_bits` required by trait `BitCast`",
            ),
            (
                "field-not-found",
                &[("field", "len"), ("type", "Slice<T>")],
                "no field `len` found on type `Slice<T>`",
            ),
            (
                "let-requires-initializer",
                &[],
                "`let` requires an explicit initializer; it cannot rely on a type's default value",
            ),
            (
                "comptime-step-limit",
                &[],
                "comptime step limit exceeded (possible infinite loop)",
            ),
            (
                "comptime-type-error",
                &[("e", "boom")],
                "comptime type error: boom",
            ),
            (
                "name-not-found",
                &[("name", "foo")],
                "name not found: `foo`",
            ),
        ];
        for (slug, kvs, want) in cases {
            let pairs: Vec<(&'static str, Cow<'static, str>)> = kvs
                .iter()
                .map(|(k, v)| (*k, Cow::Owned((*v).to_string())))
                .collect();
            assert_eq!(render(slug, &pairs), want, "slug {slug:?} diverged");
        }
    }

    /// An explicit `Lang::set` must be stable: once set, `current()` returns
    /// the explicit language forever. The startup race (detect overwriting an
    /// explicit choice) only exists before the first touch; the compare-
    /// exchange in `current()` closes it, and this test pins the observable
    /// half of that contract deterministically (the real interleaving needs
    /// the shuttle feature, which is excluded from this cfg).
    #[test]
    fn explicit_lang_set_is_stable() {
        let _lang_guard = acquire_test_lang_lock();
        Lang::set(Lang::Zh);
        assert_eq!(Lang::current(), Lang::Zh);
        assert_eq!(Lang::current(), Lang::Zh);
        // The latch is process-global; other tests assert English output,
        // so hand the process back in the default language.
        Lang::set(Lang::En);
    }

    /// Non-numeric argument keys must stay verbatim strings. Regression
    /// guard: `FluentValue::try_number("00123")` parses as the number 123
    /// and would render "123", silently corrupting zero-padded and
    /// decimal-looking values (codes, names, types).
    #[test]
    fn numeric_like_values_for_non_numeric_keys_render_verbatim() {
        let _lang_guard = acquire_test_lang_lock();
        Lang::set(Lang::En);
        let pairs: Vec<(&'static str, Cow<'static, str>)> = vec![
            ("field", Cow::Owned("00123".to_string())),
            ("type", Cow::Owned("1.0".to_string())),
        ];
        assert_eq!(
            render("field-not-found", &pairs),
            "no field `00123` found on type `1.0`"
        );
    }

    /// A borrowed argument must stay borrowed: the lifetime-parametric
    /// `TrValue` exists so a `Cow::Borrowed` value is not copied into a
    /// fresh `String` before Fluent renders it.
    #[test]
    fn borrowed_arguments_stay_borrowed() {
        let owned = String::from("Int<32>");
        let borrowed: Cow<'_, str> = Cow::Borrowed(owned.as_str());
        let lifted = TrValue::into_cow(borrowed);
        assert!(
            matches!(lifted, Cow::Borrowed(_)),
            "borrowed argument was copied into an owned String"
        );
        assert_eq!(lifted, "Int<32>");
    }
}
