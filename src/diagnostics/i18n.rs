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
use std::collections::HashMap;
use std::fmt::Display;
use std::sync::atomic::{AtomicU8, Ordering};

use fluent_bundle::FluentValue;
use fluent_templates::{LanguageIdentifier, Loader, langid, static_loader};

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

// Compile-time locale embedding: the macro walks `locales/` at expansion
// and `include_str!`s every file, so no runtime file access is needed
// (the offline explain server relies on that).  Isolating marks must be
// off: our renderer and byte-exact tests expect plain substituted text.
static_loader! {
    static LOCALES = {
        locales: "./locales",
        fallback_language: "en",
        customise: |bundle| bundle.set_use_isolating(false),
    };
}

/// `LANG` sentinel meaning "not yet detected".
const LANG_UNSET: u8 = 0;
const LANG_EN: u8 = 1;
const LANG_RU: u8 = 2;
const LANG_ZH: u8 = 3;

static LANG: AtomicU8 = AtomicU8::new(LANG_UNSET);

/// Guards tests that mutate the process-global [`LANG`]: parallel tests
/// share one atomic, so a test that sets a language must hold this lock
/// while it renders.
#[cfg(all(test, not(feature = "shuttle")))]
static TEST_LANG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`TEST_LANG_LOCK`].  Poison is irrelevant for `()`, so a
/// panicking test cannot wedge the others.
#[cfg(all(test, not(feature = "shuttle")))]
pub(crate) fn acquire_test_lang_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LANG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
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

    pub fn set(lang: Lang) {
        LANG.store(lang as u8, Ordering::Relaxed);
    }

    /// Decode a raw `LANG` value.  Returns `None` for the unset sentinel and
    /// any value outside the known discriminants.
    #[inline]
    fn from_u8(n: u8) -> Option<Self> {
        match n {
            LANG_EN => Some(Lang::En),
            LANG_RU => Some(Lang::Ru),
            LANG_ZH => Some(Lang::Zh),
            _ => None,
        }
    }
}

fn detect_from_env() -> Lang {
    let env = std::env::var("PONENT_LANG")
        .or_else(|_| std::env::var("LANG"))
        .unwrap_or_default()
        .to_lowercase();
    if env.starts_with("ru") {
        Lang::Ru
    } else if env.starts_with("zh") {
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
    if is_ftl_slug(msg)
        && let Some(s) = try_fluent(msg, pairs)
    {
        return Cow::Owned(s);
    }
    Cow::Borrowed(msg)
}

/// [`render`] into an existing buffer.
///
/// The fallback (unknown slug / legacy key) is written straight through
/// instead of being wrapped in a `Cow` first; a Fluent hit still has to
/// allocate its `String` internally, so this saves the wrapper, not the
/// formatting allocation.
pub fn render_into<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
    out: &mut String,
) {
    if is_ftl_slug(msg)
        && let Some(s) = try_fluent(msg, pairs)
    {
        out.push_str(&s);
        return;
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

fn lang_id() -> &'static LanguageIdentifier {
    static IDS: std::sync::OnceLock<[LanguageIdentifier; 3]> = std::sync::OnceLock::new();
    let ids = IDS.get_or_init(|| [langid!("en"), langid!("ru"), langid!("zh")]);
    &ids[Lang::current() as usize - 1]
}

/// `None` when the slug is unknown even after the loader's fallback chain
/// (en) — or when a referenced argument is missing (the loader then reports
/// an unformattable pattern) — so the caller returns `msg` verbatim.
/// Argument keys whose values are semantically counts: only these need to
/// become Fluent numbers so plural selectors (`summary-errors`) see real
/// CLDR categories.  Everything else is substituted as plain text —
/// `try_number` silently coerced numeric-looking strings like `"00123"`
/// or `"1.0"` into numbers, corrupting values that are names, not counts.
const NUMERIC_ARG_KEYS: &[&str] = &["count", "n", "num"];

fn try_fluent<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
) -> Option<String> {
    // Zero-argument fast path: the common static-message lookup never
    // allocates an argument map.
    if pairs.is_empty() {
        return LOCALES.try_lookup(lang_id(), msg);
    }
    let mut args = HashMap::with_capacity(pairs.len());
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
        let value = if NUMERIC_ARG_KEYS.contains(k) {
            FluentValue::try_number(v.as_ref())
        } else {
            FluentValue::String(v.clone())
        };
        args.insert(Cow::Borrowed(*k), value);
    });
    LOCALES.try_lookup_with_args(lang_id(), msg, &args)
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
    if is_ftl_slug(msg)
        && let Some(s) = try_fluent_owned(msg, pairs)
    {
        out.push_str(&s);
        return;
    }
    out.push_str(msg);
}

fn try_fluent_owned<'msg, 'src>(
    msg: &'msg str,
    pairs: SmallVec<[(&'static str, Cow<'src, str>); 4]>,
) -> Option<String> {
    if pairs.is_empty() {
        return LOCALES.try_lookup(lang_id(), msg);
    }
    let mut args = HashMap::with_capacity(pairs.len());
    pairs.into_iter().for_each(|(k, v)| {
        let value = if NUMERIC_ARG_KEYS.contains(&k)
            && let Ok(n) = v.as_ref().parse::<fluent_bundle::types::FluentNumber>()
        {
            n.into()
        } else {
            FluentValue::String(v)
        };
        args.insert(Cow::Borrowed(k), value);
    });
    LOCALES.try_lookup_with_args(lang_id(), msg, &args)
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
        static CACHES: [::std::sync::OnceLock<[::std::string::String; $n]>; 3] = [
            ::std::sync::OnceLock::new(),
            ::std::sync::OnceLock::new(),
            ::std::sync::OnceLock::new(),
        ];
        let lang_idx = ($crate::diagnostics::i18n::Lang::current() as usize) - 1;
        let table = CACHES[lang_idx].get_or_init(|| [$($init),+]);
        ::std::borrow::Cow::Borrowed(&table[$variant])
    }};
}
pub(crate) use cached_label;

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
        use fluent_templates::Loader;
        let no_args: HashMap<Cow<'static, str>, FluentValue> = HashMap::new();
        assert_eq!(
            LOCALES.try_lookup_with_args(&langid!("ru"), "main-not-found", &no_args),
            Some("функция `main` не найдена в крейте".to_string())
        );
        assert_eq!(
            LOCALES.try_lookup_with_args(&langid!("zh"), "main-not-found", &no_args),
            Some("crate 中没有找到 `main` 函数".to_string())
        );
        let mut count: HashMap<Cow<'static, str>, FluentValue> = HashMap::new();
        count.insert(Cow::Borrowed("count"), FluentValue::try_number("2"));
        assert_eq!(
            LOCALES.try_lookup_with_args(&langid!("ru"), "summary-errors", &count),
            Some("прерывание из-за 2 предыдущих ошибок".to_string())
        );
        assert_eq!(
            LOCALES.try_lookup_with_args(&langid!("zh"), "summary-errors", &count),
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
        let cases: [(&str, &[(&str, &str)], &str); 3] = [
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
