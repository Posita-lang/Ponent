/// Internationalisation support for diagnostic messages.
///
/// Messages live in `locales/*.ftl` (rustc-style Fluent), embedded at
/// compile time by `static_loader!`.  [`render`] resolves a stable slug
/// key (`type-mismatch`) through the active language's bundle — with
/// `en.ftl` as fallback — and returns any other key verbatim.
///
/// Language selection (in priority order, POSIX precedence):
///   1. `PONENT_LANG` environment variable (e.g. `en`, `ru`, `zh`)
///   2. `LC_ALL`
///   3. `LC_MESSAGES`
///   4. `LANG` (system locale, e.g. `en_US.UTF-8`)
///   5. English (default)
///
/// A variable set to the empty string is treated as unset, and a language
/// without a bundle (`de_DE`) falls back to English rather than failing.
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

/// Join `first` and the rest with the locale's disjunction ("or").
///
/// The single source for list conjunctions: `"a` or `b"` baked English "or" into
/// every locale's output, and that bug was fixed twice independently before this
/// helper existed.  Callers own the item text (including any quoting).
pub(crate) fn join_with_or<'a>(
    first: Cow<'a, str>,
    rest: impl IntoIterator<Item = Cow<'a, str>>,
) -> Cow<'a, str> {
    // Two buffers, alternating: the typed accessor allocated a fresh `String`
    // per pair (N−1 allocations for an N-item list, every one discarded except
    // the last) and `render_into` keeps the *result* buffer's capacity for the
    // next round instead of dropping it.
    let mut joined = first;
    let mut scratch = String::new();
    for next in rest {
        scratch.clear();
        crate::diagnostics::i18n::render_into(
            "suggestion-or",
            &[
                ("first", Cow::Borrowed(joined.as_ref())),
                ("next", Cow::Borrowed(next.as_ref())),
            ],
            &mut scratch,
        );
        // `render_into` falls back to writing the *raw slug* when no bundle carries the
        // message, so a renamed/removed `suggestion-or` would silently replace the whole
        // accumulated list with the literal string "suggestion-or" — user-visible
        // corruption whose only current guard is a presence test elsewhere.
        #[cfg(debug_assertions)]
        debug_assert_ne!(
            scratch, "suggestion-or",
            "join_with_or: `suggestion-or` is missing from every bundle — the list would \
             render as the slug itself"
        );
        joined = match joined {
            Cow::Owned(mut previous) => {
                // Trade buffers: `previous` takes the rendered text, `scratch`
                // keeps the old allocation for the next round.
                std::mem::swap(&mut previous, &mut scratch);
                Cow::Owned(previous)
            }
            Cow::Borrowed(_) => Cow::Owned(std::mem::take(&mut scratch)),
        };
    }
    joined
}

impl<'a, 'src: 'a> LiftCow<'a> for Cow<'src, str> {
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

/// Note on the release stance of `make_bundle`: a locale whose Fluent source fails to parse
/// degrades **silently to English** in release builds (the parse error is a debug assertion
/// only).  That is deliberate — a compiler must not refuse to emit diagnostics because a
/// translation is broken — but it is a decision, so `validate_locales` (CI) asserts that
/// every locale parses and that its slug set equals `en`'s; drift is caught at build time
/// rather than as a mixed-language diagnostic at run time.

fn make_bundle(lang: LanguageIdentifier, src: &'static str) -> Bundle {
    // Locales are validated by `validate_locales` at build time, so a parse
    // error here means the validator and the runtime disagree (version drift,
    // a validator bug, a build.rs bypass).  Fluent is error-RECOVERING: the
    // broken message is dropped from the resource, `add_resource` still
    // succeeds, and every render of that slug silently falls back to English.
    // Keep the recovered errors and fail loudly in debug builds instead of
    // letting a language quietly lose a message.
    let (res, errors) = match FluentResource::try_new(src.to_owned()) {
        Ok(r) => (r, Vec::new()),
        Err((r, e)) => (r, e),
    };
    debug_assert!(
        errors.is_empty(),
        "locale {}: Fluent parse errors that `validate_locales` missed: {errors:?}",
        lang
    );
    let res: &'static FluentResource = Box::leak(Box::new(res));
    let mut bundle = fluent_bundle::concurrent::FluentBundle::new_concurrent(vec![lang.clone()]);
    bundle.set_use_isolating(false);
    if let Err(err) = bundle.add_resource(res) {
        // Conflicting message ids inside one `.ftl`.  Same class as a parse error
        // above, and this module's stated stance is that a compiler must not refuse to
        // emit diagnostics because a translation is broken: Fluent adds every
        // non-conflicting message and reports the clash, so the recovered bundle keeps
        // working and a slug it lacks falls back to English.  Panicking here aborted the
        // process at first `bundles()` init — in release, over a locale problem, which
        // is the worst possible trade for a diagnostic system.  `validate_locales`
        // rejects duplicates at build time, so this is the runtime-only last resort.
        debug_assert!(
            false,
            "locale {lang}: conflicting Fluent message ids that `validate_locales` missed: {err:?}"
        );
    }
    bundle
}

fn bundle_index_for(lang: &LanguageIdentifier) -> usize {
    // Matches the `language` subtag only: `zh-Hant` resolves to the `zh`
    // (Simplified) bundle, `en-GB` to `en`.  Adding a locale must update
    // `LANG_*`/`N_LANGS` and this table together.
    match lang.language.as_str() {
        "ru" => 1,
        "zh" => 2,
        _ => 0,
    }
}

/// Resolve + format `slug` in `lang`'s bundle, falling back to the `en`
/// bundle — the same chain the loader used: unknown slug or a format
/// error (e.g. a missing referenced argument) moves on to `en`, and `None`
/// only when `en` fails too, so the caller returns `msg` verbatim.
///
/// Not test-only: the generated `fluent_gen.rs` calls `super::format_slug`
/// for the typed zero/one-arg accessors.
fn format_slug(lang: &LanguageIdentifier, slug: &str, args: Option<&FluentArgs>) -> Option<String> {
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
    format_slug_in_into(
        &bundles[bundle_index_for(lang)],
        &bundles[0],
        slug,
        args,
        out,
    )
}

/// Walk the `primary -> en` chain, retrying on the next bundle when a bundle
/// has the message but its pattern fails to format (a broken translation, an
/// argument missing from the translation, or a malformed pattern).
///
/// One implementation for every render path.  `format_slug_in_into`,
/// `try_fluent_into`, `try_fluent_owned_into` and `render_into_count` each
/// inlined this walk with slightly different tripwire text, so a future change to
/// the retry policy needed four coordinated edits — the "they agreed by
/// maintenance only" shape this module already documents elsewhere.
fn walk_chain<F>(primary: &Bundle, en: &Bundle, out: &mut String, mut fmt: F) -> bool
where
    F: FnMut(&Bundle, &mut String) -> bool,
{
    for b in std::iter::once(primary).chain((!std::ptr::eq(primary, en)).then_some(en)) {
        if fmt(b, out) {
            return true;
        }
    }
    false
}

/// Tripwire for the *render* paths: a slug that exists in `en` yet formatted on
/// no bundle is a caller/translation bug (a missing or misspelled argument), not
/// an unknown slug.
///
/// Deliberately separate from [`walk_chain`]: the text accessors
/// (`format_slug_into`, used by generated `fluent::foo(...)`) degrade to "return
/// the caller's string" and must stay silent — `format_slug_into_degrades_without_touching_out`
/// pins that.  One message for every render path, so the wording cannot drift.
///
/// `#[cfg(debug_assertions)]` on the function *and* on every call site: the body
/// is a single `debug_assert!`, so a release build had a call to an empty,
/// `#[cold]` (non-inlined, cold-section) function on the per-render hot path.
/// LLVM usually removes it, but "usually" is not a contract, and the annotation
/// is actively wrong for the release case.
#[cfg(debug_assertions)]
#[cold]
fn tripwire_unformatted(msg: &str, en: &Bundle, formatted: bool, path: UnformattedPath) {
    debug_assert!(
        formatted || en.get_message(msg).is_none(),
        "slug {msg:?} exists but formatting failed on the {} path — \
         missing or misspelled argument?",
        path.describe()
    );
}

/// Which call shape observed an unformatted slug.
///
/// One predicate with a mode, not three predicates with three texts: the texts drifted
/// once already (the owned zero-argument branch checked *every* bundle while the borrowed
/// one checked `en`), and a per-shape `debug_assert!` message is exactly the kind of copy
/// this module keeps collapsing.
#[cfg(debug_assertions)]
#[derive(Clone, Copy)]
enum UnformattedPath {
    Borrowed,
    Owned,
    ZeroArg,
}

#[cfg(debug_assertions)]
impl UnformattedPath {
    fn describe(self) -> &'static str {
        match self {
            Self::Borrowed => "borrowed",
            Self::Owned => "owned",
            Self::ZeroArg => "zero-argument",
        }
    }
}

/// Zero-argument variant of [`tripwire_unformatted`], sharing its message so the
/// two cannot drift.
///
/// The owned zero-argument branch used `bundles.iter().all(|b| ...)` (every
/// bundle) while [`tripwire_unformatted`] checks only `en`.  Both are true today
/// because the locale-coverage test pins equal slug sets, but two predicates for
/// one invariant is exactly the drift shape this module keeps removing; `en` is
/// the reference locale by construction.
/// Zero-argument call shape: same predicate, `ZeroArg` mode (see
/// [`tripwire_unformatted`]).  Kept as a named wrapper so the call sites read as what they
/// are, not as a mode constant at each of them.
#[cfg(debug_assertions)]
#[cold]
fn tripwire_unformatted_zero_arg(msg: &str, en: &Bundle, formatted: bool) {
    tripwire_unformatted(msg, en, formatted, UnformattedPath::ZeroArg);
}

fn format_slug_in_into(
    primary: &Bundle,
    en: &Bundle,
    slug: &str,
    args: Option<&FluentArgs>,
    out: &mut String,
) -> bool {
    // One `errors` buffer reused across bundle attempts (a fresh `Vec` per
    // attempt paid a `Drop` on the fallback path).
    // The primary bundle's pattern comes from the same memo the message paths use:
    // `get_message` is a `std`-hasher lookup the answer of which cannot change, and a
    // single humanized `TypeMismatch` renders up to three labels through here
    // (`label_expected`, `label_declared_here`, `found`).  The memo policy used to be
    // split — memoized on the borrowed message path, live on the owned one and on
    // every label — which is the "they agreed by maintenance" drift shape.
    let lang_idx = Lang::current().index();
    let primary_is_current = std::ptr::eq(primary, &bundles()[lang_idx]);
    let memoized = if primary_is_current {
        cached_primary_pattern(lang_idx, slug)
    } else {
        None
    };
    // Simple slugs (literal text plus `{ $var }` string substitutions) skip Fluent
    // entirely — no argument map, no `format_pattern`, no `Cow`.  Only for the *current*
    // language's own pattern: the retry-to-`en` walk below is Fluent's job.
    if primary_is_current && write_simple_into(slug, lang_idx, args, out) {
        return true;
    }
    let mut errors = Vec::new();
    walk_chain(primary, en, out, |b, out| {
        errors.clear();
        let looked_up = if primary_is_current && std::ptr::eq(b, primary) {
            memoized
        } else {
            b.get_message(slug).and_then(|m| m.value())
        };
        let Some(pattern) = looked_up else {
            return false;
        };
        let value = b.format_pattern(pattern, args, &mut errors);
        if errors.is_empty() {
            out.push_str(&value);
            true
        } else {
            false
        }
    })
}

/// One segment of a "simple" slug: literal text, or a `{ $var }` substitution.
#[derive(Debug)]
enum SimplePart {
    Text(&'static str),
    Var(&'static str),
}

/// Per-(slug, language) classification of a pattern as a simple writer.
///
/// "Simple" means the pattern's elements are only `TextElement` and
/// `Placeholder { VariableReference }` with a **non-numeric** variable.  For such a
/// pattern, Fluent's output is exactly "literal text with each substitution written
/// verbatim", so it can be produced without building a `FluentArgs` value tree, without
/// `format_pattern`'s walk and without its `Cow` — the caller's buffer is the only copy.
///
/// Classified **per language**, not from `en.ftl`: a translation of the same slug may use
/// a plural selector (ru/zh routinely do), and the coverage test only pins the placeholder
/// *sets*.  Anything not classified (a function, a term, a message reference, a numeric
/// key) or any argument that is not a plain `FluentValue::String` returns `false` and the
/// Fluent path runs — the fast path is an observation about this pattern, never a guess.
static SIMPLE_SLUGS: [[OnceLock<Option<Vec<SimplePart>>>; N_LANGS]; fluent::N_SLUGS] =
    [const { [const { OnceLock::new() }; N_LANGS] }; fluent::N_SLUGS];

fn simple_parts(pattern: &'static Pattern) -> Option<Vec<SimplePart>> {
    use fluent_syntax::ast::{Expression, InlineExpression, PatternElement};
    let mut out = Vec::with_capacity(pattern.elements.len());
    for el in &pattern.elements {
        match el {
            PatternElement::TextElement { value } => out.push(SimplePart::Text(value)),
            PatternElement::Placeable { expression } => {
                // `{ $var }` only.  A `Select` (plural/select) or any non-variable inline
                // expression (function, term, message reference, literal) is Fluent's job.
                let Expression::Inline(InlineExpression::VariableReference { id }) = expression
                else {
                    return None;
                };
                // Numeric keys reach Fluent as numbers so CLDR plural/select categories
                // apply; writing the raw string would be a different message.
                if fluent::is_numeric_arg_key(id.name) {
                    return None;
                }
                out.push(SimplePart::Var(id.name));
            }
        }
    }
    Some(out)
}

/// [`write_simple_into`] over raw pairs — the **hot-path** twin.
///
/// `try_fluent_into`/`try_fluent_owned_into` render every `DiagMessage::Slug` (both
/// `render_into_slug_ref` and `render_into_slug` land there), and both used to build a
/// `FluentArgs`, run `format_pattern` and copy its intermediate `String` into `out`.
/// Only the cold generated accessors went through [`write_simple_into`] — the fast path
/// guarded the cold surface while the hot one paid full price, the "two implementations
/// agreed by maintenance only" shape inverted.
///
/// Declines (leaving `out` untouched) when the pattern is not classified simple, when a
/// pattern variable has no pair or more than one (Fluent's `set` replaces, so a duplicate
/// would render differently), or when the slug is outside the generated set.
fn write_simple_pairs_into(
    slug: &str,
    lang_idx: usize,
    pairs: &[(&'static str, Cow<'_, str>)],
    out: &mut String,
) -> bool {
    let Some(idx) = fluent::slug_index(slug) else {
        return false;
    };
    let parts = SIMPLE_SLUGS[idx][lang_idx.min(N_LANGS - 1)]
        .get_or_init(|| cached_primary_pattern(lang_idx, slug).and_then(simple_parts));
    let Some(parts) = parts.as_ref() else {
        return false;
    };
    // Validate *and* size in one pass, then write: a partial write followed by a fallback
    // would duplicate text, and an exact reserve keeps this path allocation-free.
    //
    // The validation pass also records each variable's value, so the write pass does
    // not rescan `pairs` per variable.  `parts`/`pairs` are tiny today, so this is not
    // a measured win — it removes the "linear scan inside a linear pass" shape the
    // module's hot paths deliberately avoid, and the recorded order is checked against
    // the write order so the two passes cannot drift.
    let mut reserve = 0usize;
    let mut values: smallvec::SmallVec<[(usize, &str); 4]> = smallvec::SmallVec::new();
    for (i, part) in parts.iter().enumerate() {
        match part {
            SimplePart::Text(text) => reserve += text.len(),
            SimplePart::Var(name) => {
                let mut hits = pairs.iter().filter(|(k, _)| k == name);
                match (hits.next(), hits.next()) {
                    (Some((_, value)), None) => {
                        reserve += value.len();
                        values.push((i, value.as_ref()));
                    }
                    _ => return false,
                }
            }
        }
    }
    out.reserve(reserve);
    let mut next_value = values.iter();
    for (i, part) in parts.iter().enumerate() {
        match part {
            SimplePart::Text(text) => out.push_str(text),
            SimplePart::Var(_) => {
                let Some((at, value)) = next_value.next() else {
                    return false;
                };
                debug_assert_eq!(
                    *at, i,
                    "the value index must follow the write order it was built in"
                );
                out.push_str(value);
            }
        }
    }
    true
}

/// Write `slug` through the simple-segment path.  Returns `false` (writing nothing) when
/// this slug, language or argument set is not eligible.
fn write_simple_into(
    slug: &str,
    lang_idx: usize,
    args: Option<&FluentArgs>,
    out: &mut String,
) -> bool {
    let Some(idx) = fluent::slug_index(slug) else {
        return false;
    };
    let parts = SIMPLE_SLUGS[idx][lang_idx.min(N_LANGS - 1)]
        .get_or_init(|| cached_primary_pattern(lang_idx, slug).and_then(simple_parts));
    let Some(parts) = parts.as_ref() else {
        return false;
    };
    let Some(args) = args else {
        return false;
    };
    // Validate before writing: a partial write followed by a fallback would duplicate
    // text.  Every variable must resolve to a plain string — a number, a missing
    // argument or a non-string value is Fluent's business.
    for part in parts {
        if let SimplePart::Var(name) = part
            && !matches!(args.get(*name), Some(FluentValue::String(_)))
        {
            return false;
        }
    }
    // Exact reserve, mirroring the hot-path twin `write_simple_pairs_into`: a fixed guess
    // per variable (8 bytes) under-reserved for every parameterized label — a type name
    // reaches `MAX_PAYLOAD_CHARS` (512) after `capped_payload` — so `label_expected(Int<32>)`
    // reallocated mid-write on the per-label render path.  Both writers now compute the
    // same number from the same pieces; the reserve was the last behavioural asymmetry
    // between them (`simple_writer_matches_fluent_byte_for_byte` pins the bytes, not this).
    out.reserve(
        parts
            .iter()
            .map(|p| match p {
                SimplePart::Text(t) => t.len(),
                SimplePart::Var(name) => match args.get(*name) {
                    Some(FluentValue::String(value)) => value.len(),
                    // Unreachable (validated above), but a wrong guess here only
                    // over-reserves; never write 0.
                    _ => 8,
                },
            })
            .sum::<usize>(),
    );
    for part in parts {
        match part {
            SimplePart::Text(text) => out.push_str(text),
            SimplePart::Var(name) => {
                if let Some(FluentValue::String(value)) = args.get(*name) {
                    out.push_str(value.as_ref());
                }
            }
        }
    }
    true
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
    // `Diagnostic::humanized_lang` stores this discriminant in a `u8`.  A `Lang`
    // domain wider than 255 would alias there and make the re-humanization latch
    // serve stale text silently; pin the width while it is still cheap.
    assert!(N_LANGS < u8::MAX as usize);
};

/// Number of supported languages. Single source of truth for the bundle
/// table, the language-id cache and the `cached_label!` arrays — adding
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
    detect_from_values(
        std::env::var("PONENT_LANG").ok(),
        std::env::var("LC_ALL").ok(),
        std::env::var("LC_MESSAGES").ok(),
        std::env::var("LANG").ok(),
    )
}

fn detect_from_values(
    explicit: Option<String>,
    lc_all: Option<String>,
    lc_messages: Option<String>,
    lang: Option<String>,
) -> Lang {
    let explicit = explicit.as_deref().filter(|s| !s.is_empty());
    let fallback = [lc_all, lc_messages, lang]
        .into_iter()
        .flatten()
        .find(|s| !s.is_empty());
    detect_from_pair(explicit, fallback.as_deref())
}

fn detect_from_pair(explicit: Option<&str>, fallback: Option<&str>) -> Lang {
    match explicit.filter(|s| !s.is_empty()) {
        Some(p) => detect_from_str(p),
        None => fallback.map(detect_from_str).unwrap_or(Lang::En),
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
/// The zero-argument prelude every entry point used to repeat.
///
/// `Some(cached)` means the slug has a build-time cached rendering for the current
/// language (borrowed; no allocation after the first call per `(slug, lang)`); `None` means
/// "format it".  A non-empty argument list on a zero-arg slug is a caller bug — the
/// generator classifies a slug as zero-arg from `en.ftl`, so the arguments would be silently
/// dropped, the worst kind of defect in a diagnostics engine.  The assert is `#[cfg]`-gated
/// so release builds keep zero generated-table lookups here (a bare `debug_assert!` would
/// leave the lookup in release and rely on DCE, and "usually" is not a contract).
fn zero_arg_cached<'m>(slug: &'m str, arg_count: usize) -> Option<Cow<'m, str>> {
    if arg_count == 0 {
        return fluent::zero_arg_slug(slug);
    }
    #[cfg(debug_assertions)]
    debug_assert!(
        fluent::zero_arg_slug(slug).is_none(),
        "slug {slug:?} is zero-arg but was passed {arg_count} argument(s)"
    );
    None
}

/// [`zero_arg_cached`] for the into-buffer entry points: pushes the cached rendering and
/// reports whether the caller is done.
fn push_zero_arg_cached(out: &mut String, slug: &str, arg_count: usize) -> bool {
    match zero_arg_cached(slug, arg_count) {
        Some(cached) => {
            out.push_str(&cached);
            true
        }
        None => false,
    }
}

pub fn render<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
) -> Cow<'msg, str> {
    if is_ftl_slug(msg) {
        if let Some(cached) = zero_arg_cached(msg, pairs.len()) {
            return cached;
        }
        if let Some(s) = try_fluent(msg, pairs) {
            return Cow::Owned(s);
        }
    }
    Cow::Borrowed(msg)
}

/// [`render`] for a key the caller **knows** is a slug.
///
/// Skips `is_ftl_slug`'s byte scan, which `render` paid on every call.  That scan
/// exists to keep legacy English messages (`"unexpected token"`) from being looked up
/// as slugs; a literal `tr!("…")` key is a slug by contract, so the question is
/// already answered at the call site.  Behaviour is otherwise identical — an unknown
/// slug still falls back to the borrowed key.
#[must_use]
pub fn render_slug<'msg, 'src>(
    slug: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
) -> Cow<'msg, str> {
    if let Some(cached) = zero_arg_cached(slug, pairs.len()) {
        return cached;
    }
    match try_fluent(slug, pairs) {
        Some(s) => Cow::Owned(s),
        None => Cow::Borrowed(slug),
    }
}

/// [`render`] into an existing buffer.
pub fn render_into<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
    out: &mut String,
) {
    if is_ftl_slug(msg) {
        if push_zero_arg_cached(out, msg, pairs.len()) {
            return;
        }
        if try_fluent_into(msg, pairs, out) {
            return;
        }
    }
    out.push_str(msg);
}

/// Render `msg` with one **native numeric** argument.
///
/// [`render_into`] takes `Cow<str>` pairs, so a count had to be formatted into a
/// string (`push_uint` into an `ArrayString`) for Fluent to re-parse as a number
/// — that round trip only works because Fluent accepts numeric strings, which is
/// a fragile dependency, and it costs a formatting pass per call.
///
/// Passing the number as a number gives CLDR's plural rules the value directly.
/// This is the only place a native value is needed today (fold markers), so it
/// stays a focused entry point rather than a second, parallel argument
/// machinery: duplicating the resolve/retry loop would be the drift risk the
/// `render_into`/`try_fluent_into` pair exists to avoid.
pub fn render_into_count(msg: &str, key: &'static str, n: u64, out: &mut String) {
    // Legacy keys have no placeable, so the count would be *silently* dropped.
    // They are dead today, but a future caller could pass one and lose the
    // number with no signal; the sibling entry points assert on the same shape.
    let is_slug = is_ftl_slug(msg);
    debug_assert!(
        is_slug,
        "render_into_count requires an FTL slug; legacy keys cannot substitute a count ({msg:?})"
    );
    if is_slug {
        if let Some(cached) = fluent::zero_arg_slug(msg) {
            // A zero-arg slug cannot consume a count.  Falling through (as this
            // used to) pushed the *raw slug name* into user-visible output in
            // both profiles; the sibling entry points either render the cached
            // text or assert, so do both.
            debug_assert!(
                false,
                "slug {msg:?} takes no arguments but was passed a numeric count"
            );
            out.push_str(&cached);
            return;
        }
        let lang_idx = Lang::current().index();
        let bundles = bundles();
        let primary = &bundles[lang_idx];
        let en = &bundles[0];
        // Built lazily, like the `tr!` paths: an unknown slug (the
        // fallback-to-raw path) used to allocate the single-entry map for
        // nothing, and this runs once per fold marker.
        let mut built_args: Option<FluentArgs> = None;
        let mut errors = Vec::new();
        // Same resolve-then-retry-on-format-failure policy as the `tr!` paths.
        let formatted = walk_chain(primary, en, out, |b, out| {
            let Some(pattern) = b.get_message(msg).and_then(|m| m.value()) else {
                return false;
            };
            let args = built_args.get_or_insert_with(|| {
                let mut args = FluentArgs::new();
                args.set(key, FluentValue::from(n));
                args
            });
            errors.clear();
            let val = b.format_pattern(pattern, Some(args), &mut errors);
            if errors.is_empty() {
                out.push_str(&val);
                true
            } else {
                false
            }
        });
        // Tripwire parity with the render paths: a slug that exists in `en` but
        // fails to format here (a missing or misspelled argument) used to fall
        // through to the raw slug with no assert at all.
        #[cfg(debug_assertions)]
        tripwire_unformatted(msg, en, formatted, UnformattedPath::Borrowed);
        if formatted {
            // `walk_chain` already pushed the rendered value.  Without this the
            // function fell through and appended the *raw slug* after the text
            // (caught by `fold_marker_shape_is_pinned`): the pre-refactor loop
            // returned from inside the success arm.
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
    // error against `Lang::index()` (a 0-based `match`), so flag it by name in
    // debug and let the slice bounds check panic in release rather than silently
    // serving a different locale.  There is no `discriminant - 1` arithmetic
    // anywhere in this module; `index()` is the single mapping.
    debug_assert!(i < N_LANGS, "language index {i} out of range 0..{N_LANGS}");
    static IDS: std::sync::OnceLock<[LanguageIdentifier; N_LANGS]> = std::sync::OnceLock::new();
    let ids = IDS.get_or_init(|| [langid!("en"), langid!("ru"), langid!("zh")]);
    &ids[i]
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
fn make_value<'s>(v: Cow<'s, str>, key: &str) -> FluentValue<'s> {
    use fluent::is_numeric_arg_key;
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

/// [`render_into`] for a value the caller already knows is a slug.
///
/// Skips the `is_ftl_slug` scan — the type system already tagged this as a
/// `DiagMessage::Slug`, so the scan re-derived a known fact on every render.  The
/// owned twin (`render_into_slug`) existed for exactly this reason; the borrowed
/// path had no equivalent, which is how the two paths drift.
pub fn render_into_slug_ref<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
    out: &mut String,
) {
    if push_zero_arg_cached(out, msg, pairs.len()) {
        return;
    }
    if try_fluent_into(msg, pairs, out) {
        return;
    }
    out.push_str(msg);
}

/// A Fluent message pattern from a `'static` bundle.
type Pattern = fluent_syntax::ast::Pattern<&'static str>;

/// Memoized per-`(language, slug)` pattern from the **primary** bundle.
///
/// `Bundle::get_message` hashes the slug with `std`'s hasher into the bundle's
/// entry map on *every* render.  The pattern is immutable (`bundles()` returns
/// `&'static [Bundle; N_LANGS]`), so the primary lookup is memoized here; `None`
/// records "this language's bundle has no such message", which is the same answer
/// every time.
///
/// Keys are owned `Box<str>` because the slug arrives as `&'msg str`, not
/// `&'static str` — one allocation per *distinct slug ever*, not per render.  The
/// `primary → en` fallback and the "primary has it but its pattern fails to
/// format" retry both stay in `walk_chain`, which re-looks-up only in those rare
/// cases.
static PRIMARY_PATTERNS: OnceLock<
    [std::sync::RwLock<rustc_hash::FxHashMap<Box<str>, Option<&'static Pattern>>>; N_LANGS],
> = OnceLock::new();

/// Per-slug slot for the primary bundle's pattern: one `OnceLock` load per
/// (slug, language), with no `RwLock` acquire and no hash probe.
///
/// The slug set is closed (it comes from `en.ftl`), so the generator can give every
/// public slug a slot.  `PRIMARY_PATTERNS` below is now the fallback for spellings the
/// generated index cannot know — dynamically-shaped strings from tests and legacy keys.
static PRIMARY_PATTERN_SLOTS: [[OnceLock<Option<&'static Pattern>>; N_LANGS]; fluent::N_SLUGS] =
    [const { [const { OnceLock::new() }; N_LANGS] }; fluent::N_SLUGS];

fn cached_primary_pattern(lang_idx: usize, msg: &str) -> Option<&'static Pattern> {
    let caches = PRIMARY_PATTERNS.get_or_init(|| {
        std::array::from_fn(|_| std::sync::RwLock::new(rustc_hash::FxHashMap::default()))
    });
    if let Some(idx) = fluent::slug_index(msg) {
        return *PRIMARY_PATTERN_SLOTS[idx][lang_idx.min(N_LANGS - 1)]
            .get_or_init(|| bundles()[lang_idx].get_message(msg).and_then(|m| m.value()));
    }
    let cache = &caches[lang_idx];
    if let Some(hit) = cache
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(msg)
    {
        return *hit;
    }
    let pattern = bundles()[lang_idx].get_message(msg).and_then(|m| m.value());
    cache
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(Box::from(msg), pattern);
    pattern
}

/// [`try_fluent`] writing the formatted pattern straight into `out` — one copy
/// total instead of `into_owned()` + a second `push_str()`.
///
/// Returns `false` when no bundle produced a clean rendering; `out` is then left
/// untouched.  (These four lines sat above `render_into_slug_ref`, which returns `()`
/// — rustdoc attributed a `bool`-returning contract to a unit function.)
fn try_fluent_into<'msg, 'src>(
    msg: &'msg str,
    pairs: &[(&'static str, Cow<'src, str>)],
    out: &mut String,
) -> bool {
    // `Lang::index()` names the bundle directly.  The old
    // `lang_id()` -> `bundle_index_for(langid)` round trip recovered the same
    // index through a string match (and eagerly initialized the langid table
    // the render path never needed), so the render choice could in principle
    // disagree with `Lang::index()`.
    let lang_idx = Lang::current().index();
    let bundles = bundles();
    let primary = &bundles[lang_idx];
    let en = &bundles[0];
    if pairs.is_empty() {
        // Zero-argument fast path: one lookup per bundle attempt, no argument
        // map.  The "known slug that fails to format" tripwire must run here
        // too: this branch returns before the shared one below, so with the
        // default language (primary == en) a known slug missing its required
        // arguments used to fall through to the raw slug in *every* profile
        // without an assert firing — the tripwire had a hole on the one path
        // most likely to hit it.
        let ok = format_slug_in_into(primary, en, msg, None, out);
        // The shared tripwire, not an inline `debug_assert!`: an inline one keeps its
        // condition — here a live `en.get_message` lookup — in the release IR and trusts
        // DCE, and its message had already drifted from
        // `tripwire_unformatted_zero_arg`'s.  Every other tripwire site in this file
        // was migrated for exactly those two reasons.
        #[cfg(debug_assertions)]
        tripwire_unformatted_zero_arg(msg, en, ok);
        return ok;
    }
    // Simple patterns (literal text plus `{ $var }` string substitutions) skip Fluent on
    // this path too: no `FluentArgs` build, no `format_pattern` walk, no intermediate
    // `String`.  Declines are the walk below's business, unchanged.
    if write_simple_pairs_into(msg, lang_idx, pairs, out) {
        return true;
    }
    // Single-pass resolve+format: walk the bundle chain once.  The argument
    // map is built lazily, only after a bundle actually has the message.
    let mut built_args: Option<FluentArgs> = None;
    // Resolve each bundle in turn and *retry formatting* on the next one
    // when a bundle has the message but fails to format it (a broken
    // translation, a missing argument in a translation, or a malformed
    // pattern).
    let mut errors = Vec::new();
    // The primary bundle's pattern comes from the memo (see
    // `cached_primary_pattern`): `get_message` is a `std`-hasher lookup that every
    // render paid, and the answer cannot change.  The `en` arm still looks up live
    // — it only runs when the primary bundle lacks the message or its pattern
    // fails to format.
    let memoized_primary = cached_primary_pattern(lang_idx, msg);
    // `walk_chain`, like every other render path.  This was the last hand-rolled copy
    // of the retry walk, kept for a lifetime reason that does not hold:
    // `format_slug_in_into` already runs the identical memo-for-primary /
    // live-for-`en` split inside a `walk_chain` closure, because the memoized
    // `&'static Pattern` coerces to the closure's shorter bundle lifetime.  One
    // implementation means a change to the retry policy is one edit.
    let formatted = walk_chain(primary, en, out, |b, out| {
        let looked_up = if std::ptr::eq(b, primary) {
            memoized_primary
        } else {
            b.get_message(msg).and_then(|m| m.value())
        };
        let Some(pattern) = looked_up else {
            return false;
        };
        let args = built_args.get_or_insert_with(|| {
            let mut args = FluentArgs::new();
            pairs.iter().for_each(|(k, v)| {
                // Borrow instead of `v.clone()`: an `Owned` pair (every `DisplayArg`
                // number/error) paid a full `String` copy per render, and the map only
                // lives inside this function, so the slice borrow outlives it.  `Owned`
                // values are only *read* here, which is why a borrow suffices.
                let value = make_value(Cow::Borrowed(v.as_ref()), k);
                // `FluentArgs::set` replaces: a caller assembling pairs from
                // two sources would otherwise silently lose the first value.
                let key: &str = *k;
                debug_assert!(args.get(key).is_none(), "duplicate tr! argument {key:?}");
                args.set(*k, value);
            });
            args
        });
        errors.clear();
        let val = b.format_pattern(pattern, Some(args), &mut errors);
        if errors.is_empty() {
            out.push_str(&val);
            true
        } else {
            false
        }
    });
    #[cfg(debug_assertions)]
    tripwire_unformatted(msg, en, formatted, UnformattedPath::Borrowed);
    formatted
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
        if push_zero_arg_cached(out, msg, pairs.len()) {
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
    if push_zero_arg_cached(out, msg, pairs.len()) {
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
    let lang_idx = Lang::current().index();
    let bundles = bundles();
    let primary = &bundles[lang_idx];
    let en = &bundles[0];
    if pairs.is_empty() {
        let ok = format_slug_in_into(primary, en, msg, None, out);
        // Mirrors `known_slug_without_required_arguments_trips_the_tripwire` on
        // the borrowed path: a slug that exists but produced no output means the
        // caller passed no arguments to a parameterized message.  Without this
        // the owned path pushed the raw slug name into user-visible output
        // silently, in *both* profiles.
        //
        // Shared with the borrowed path through `tripwire_unformatted_zero_arg`
        // rather than repeating the predicate here: the two used *different*
        // predicates (`bundles.iter().all(..)` vs `en.get_message(..)`), which is
        // precisely the drift shape this module's `walk_chain` refactor removed.
        #[cfg(debug_assertions)]
        tripwire_unformatted_zero_arg(msg, en, ok);
        return ok;
    }
    // Simple patterns (literal text plus `{ $var }` string substitutions) skip Fluent on
    // this path too: no `FluentArgs` build, no `format_pattern` walk, no intermediate
    // `String`.  Declines are the walk below's business, unchanged.
    if write_simple_pairs_into(msg, lang_idx, &pairs, out) {
        return true;
    }
    // Single-pass resolve (see `try_fluent`): one `get_message` hash lookup
    // per bundle.  The argument map is built once, only after a bundle with
    // the message is found — `pairs` is consumed, so the build cannot sit
    // inside the loop.
    // `pairs` is consumed to build the map, so build it lazily once and reuse it
    // for the fallback attempt instead of reconstructing it.
    let mut pending_pairs = Some(pairs);
    let mut built_args: Option<FluentArgs> = None;
    let mut errors = Vec::new();
    // The owned path is what `Diagnostic::humanize` runs, and it paid a live
    // `get_message` lookup per bundle while its borrowed twin read the memo.  One
    // caching policy, one place.
    let memoized_primary = cached_primary_pattern(lang_idx, msg);
    let formatted = walk_chain(primary, en, out, |b, out| {
        let looked_up = if std::ptr::eq(b, primary) {
            memoized_primary
        } else {
            b.get_message(msg).and_then(|m| m.value())
        };
        let Some(pattern) = looked_up else {
            return false;
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
                    // `FluentNumber` instead.  Same policy as the borrowed path.
                    let value = make_value(v, &k);
                    debug_assert!(args.get(k).is_none(), "duplicate tr! argument {k:?}");
                    args.set(k, value);
                });
            args
        });
        errors.clear();
        let val = b.format_pattern(pattern, Some(args), &mut errors);
        if errors.is_empty() {
            out.push_str(&val);
            true
        } else {
            false
        }
    });
    // Gated like every other call site: the function itself only exists under
    // `debug_assertions`, so an ungated call here made *release* builds fail to
    // compile (`cannot find function tripwire_unformatted`) while `cargo test`
    // (debug) stayed green.  This is the owned render path `humanize` uses.
    #[cfg(debug_assertions)]
    tripwire_unformatted(msg, en, formatted, UnformattedPath::Borrowed);
    formatted
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
        slugs = [$($slug:literal),+ $(,)?],
    ) => {{
        // `cached_label_static!` is the single source of truth: it owns the
        // `OnceLock<[String; N]>` table, the index derivation, and the
        // language-explicit fill.  `Cow::Borrowed` over its `'static` table entry
        // is sound, so this used to duplicate ~30 lines of invariant code that
        // then drifted.
        ::std::borrow::Cow::Borrowed($crate::diagnostics::i18n::cached_label_static!(
            n = $n,
            variant = $variant,
            slugs = [$($slug),+],
        ))
    }};
}
pub(crate) use cached_label;

/// Zero-wrapper variant of [`cached_label!`]: returns `&'static str` directly.
///
/// `cached_label!` always produces `Cow::Borrowed` (the table holds `String`s in
/// a `OnceLock<[String; N]>`, so the borrow is `'static`), making the `Owned`
/// branch dead and the `Cow` a 32-byte shell around a `&'static str`.  This
/// macro skips the `Cow` entirely, returning `table[variant].as_str()` with the
/// `'static` lifetime of the cached table.
///
/// `slugs` are Fluent slug names; `slugs_upper` additionally uppercases each
/// label (needed for localized severity tags, where an ASCII-only fold would
/// leave Cyrillic lowercase).  Both forms resolve through
/// [`fluent::zero_arg_slug_at`] *at the slot's own language index*, so the fill
/// cannot observe a different language than the slot it lands in.
macro_rules! cached_label_static {
    (
        n = $n:literal,
        variant = $variant:expr,
        slugs = [$($slug:literal),+ $(,)?],
    ) => {
        $crate::diagnostics::i18n::cached_label_static!(
            @build n = $n, variant = $variant, upper = false, slugs = [$($slug),+],
        )
    };
    (
        n = $n:literal,
        variant = $variant:expr,
        slugs_upper = [$($slug:literal),+ $(,)?],
    ) => {
        $crate::diagnostics::i18n::cached_label_static!(
            @build n = $n, variant = $variant, upper = true, slugs = [$($slug),+],
        )
    };
    (
        @build n = $n:literal,
        variant = $variant:expr,
        upper = $upper:literal,
        slugs = [$($slug:literal),+ $(,)?],
    ) => {{
        static CACHES: [::std::sync::OnceLock<[::std::string::String; $n]>; $crate::diagnostics::i18n::N_LANGS] =
            [const { ::std::sync::OnceLock::new() }; $crate::diagnostics::i18n::N_LANGS];
        let lang_idx = $crate::diagnostics::i18n::Lang::current().index();
        // `index()` is the single `Lang`→slot mapping (a dense `match`); the
        // assert is belt-and-braces against `N_LANGS` drift, and names what it
        // checks (an index, not a discriminant).
        debug_assert!(
            lang_idx < $crate::diagnostics::i18n::N_LANGS,
            "Lang index out of the cached range"
        );
        let table = CACHES[lang_idx].get_or_init(|| {
            // Language-*explicit*: the strings are read at `lang_idx`, the same
            // index that selected the slot, so a `Lang::set` racing this closure
            // cannot write one language's labels into another language's slot.
            // (The previous `init = [fluent::level_error(), …]` form re-resolved
            // the global inside the closure and relied on an assertion to notice
            // the divergence — a permanent, silent poisoning in release.)
            [$({
                let s = $crate::diagnostics::i18n::fluent::zero_arg_slug_at($slug, lang_idx)
                    // `walk_chain`'s primary → en discipline is a contract on every
                    // *format* path; this label path resolved at `lang_idx` only,
                    // with `unwrap_or($slug)` as the backstop — so a label missing
                    // from ru/zh (which may define fewer keys than en) could reach
                    // user-visible output as `level-error`.
                    .or_else(|| $crate::diagnostics::i18n::fluent::zero_arg_slug_at($slug, 0))
                    .unwrap_or($slug);
                // Loud per slot, free in release: each element knows its own slug,
                // so this covers every (slug, language) pair without a test that
                // would have to enumerate them.
                // A *non-empty* result equal to the slug name is the leak this
                // guards.  An empty one is legitimate: `type-ctx-unspecified`'s
                // label is deliberately "", and some tables pad unused variants
                // with an empty slug.
                debug_assert!(
                    s.is_empty() || s != $slug || $slug.is_empty(),
                    "label slot for language {lang_idx} leaked the slug name {slug} — \
                     neither the locale nor the en fallback provides it",
                    slug = $slug
                );
                if $upper {
                    // `to_ascii_uppercase` avoids the Unicode tables for the
                    // en/ru labels that are pure ASCII; the localized (Cyrillic,
                    // CJK) ones still need the full fold.
                    if s.is_ascii() {
                        s.to_ascii_uppercase()
                    } else {
                        s.to_uppercase()
                    }
                } else {
                    s.to_string()
                }
            }),+]
        });
        table[$variant].as_str()
    }};
}
pub(crate) use cached_label_static;

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// The classifier's safety argument is a claim about the *legacy corpus*:
    /// "every legacy English message contains spaces, backticks or braces, so the
    /// two key spaces cannot collide".  A single word (`"error"`, `"note"`) has
    /// none of those, so it would classify as a slug and — if a slug of that name
    /// ever existed — render translated text instead of the caller's string.
    ///
    /// Pin the corpus the way `literal_codes_are_all_in_code_table` pins the code
    /// space: scan the diagnostics sources for `Diagnostic::<level>("…")`
    /// literals and require that none is slug-shaped.
    #[test]
    fn no_legacy_diagnostic_message_is_slug_shaped() {
        fn rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    rs_files(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }

        // The whole crate, not `src/diagnostics`: the classifier's argument is about
        // *every* `Diagnostic::<level>("…")` literal, and those are written in the
        // parser, HIR and typeck — a slug-shaped message there (`Diagnostic::error(
        // "overflow")`) would be dispatched to Fluent and render a translation instead
        // of the caller's string.  Scanning only this directory excluded exactly the
        // files where such a literal would be written.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rs_files(&root, &mut files);
        assert!(
            !files.is_empty(),
            "source scan found no files under {root:?}"
        );

        let mut offenders = Vec::new();
        for file in &files {
            let Ok(src) = std::fs::read_to_string(file) else {
                continue;
            };
            // Production half only: test fixtures use bare words (`"boom"`,
            // `"first"`) that are slug-shaped by construction, and a test module
            // never reaches a user-visible render.  Cutting at the first
            // `#[cfg(test` occurrence keeps the scan honest about the corpus the
            // classifier's argument is actually about.
            let cut = ["#[cfg(test", "#[cfg(all(test"]
                .iter()
                .filter_map(|marker| src.find(marker))
                .min();
            let src = cut.map_or(src.as_str(), |cut| &src[..cut]);
            for ctor in [
                "Diagnostic::error(\"",
                "Diagnostic::warning(\"",
                "Diagnostic::note(\"",
                "Diagnostic::help(\"",
                // Every constructor that takes a user-visible message, not just
                // the original four: a slug-shaped literal passed to any of these
                // would be dispatched to Fluent and render plausible-but-wrong
                // text, which is exactly what this scan exists to prevent.  The
                // guard is only as strong as its ctor enumeration.
                "Diagnostic::fatal(\"",
                "Diagnostic::info(\"",
                "Diagnostic::help_diag(\"",
                "Diagnostic::new(DiagnosticLevel::",
            ] {
                let mut from = 0;
                while let Some(pos) = src[from..].find(ctor) {
                    let start = from + pos + ctor.len();
                    let Some(end) = src[start..].find('"') else {
                        break;
                    };
                    let lit = &src[start..start + end];
                    if is_ftl_slug(lit) {
                        offenders.push(format!("{}: {lit:?}", file.display()));
                    }
                    from = start + end;
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "slug-shaped legacy messages would be dispatched to Fluent: {offenders:#?}"
        );
    }

    /// Static half of the runtime retry logic in [`try_fluent_into`].
    ///
    /// A locale may translate fewer keys than `en` (missing key → English text,
    /// which is acceptable), but it must never *add* a key `en` lacks (nothing
    /// can resolve it) and every shared key must take the **same placeholder
    /// set** — a translation that drops `{ $gap }` formats fine and silently
    /// loses the number, which is exactly the bug the runtime tripwire can only
    /// catch after the fact.
    ///
    /// Parsed textually: the shipped `.ftl` files are the input, not the
    /// generated accessors (the accessors cannot tell you what the *translation*
    /// contains).
    #[test]
    fn locales_are_covered_by_english_keys_and_placeholders() {
        type Args = std::collections::BTreeSet<String>;
        type Keys = std::collections::BTreeMap<String, Args>;

        fn collect_placeholders(s: &str, set: &mut Args) {
            let b = s.as_bytes();
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'$' {
                    let start = i + 1;
                    let mut j = start;
                    while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                        j += 1;
                    }
                    if j > start {
                        set.insert(s[start..j].to_string());
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
        }

        /// `key = value` starts a message (column 0); indented lines — including
        /// `[one] …` select arms and `.attr = …` — continue it.
        fn parse(src: &str) -> Keys {
            let mut out: Keys = Keys::new();
            let mut key: Option<String> = None;
            for line in src.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                let indented = line.starts_with([' ', '\t']);
                if !indented && let Some((k, value)) = trimmed.split_once('=') {
                    let k = k.trim().to_string();
                    let entry = out.entry(k.clone()).or_default();
                    collect_placeholders(value, entry);
                    key = Some(k);
                } else if let Some(k) = &key
                    && let Some(entry) = out.get_mut(k)
                {
                    collect_placeholders(trimmed, entry);
                }
            }
            out
        }

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("locales");
        let read = |name: &str| {
            std::fs::read_to_string(root.join(name))
                .unwrap_or_else(|e| panic!("cannot read locales/{name}: {e}"))
        };
        let en = parse(&read("en.ftl"));
        assert!(!en.is_empty(), "en.ftl parsed to zero keys");

        for other in ["ru.ftl", "zh.ftl"] {
            let loc = parse(&read(other));
            assert!(!loc.is_empty(), "{other} parsed to zero keys");
            for k in loc.keys() {
                // A `-term` is a *private* Fluent definition (a term referenced
                // from messages), legitimately locale-local: requiring en to
                // define it would be a false positive on any locale that adds
                // one.
                if k.starts_with('-') {
                    continue;
                }
                assert!(
                    en.contains_key(k),
                    "{other} defines {k:?}, which en.ftl does not"
                );
            }
            for (k, args) in &loc {
                assert_eq!(
                    &en[k], args,
                    "{other}: placeholder set for {k:?} differs from en (en={:?}, {other}={args:?})",
                    en[k]
                );
            }
            // Reverse containment: every *public* `en` key must exist in every locale.
            // This used to be a hand-maintained slug manifest in a second test, and it had
            // silently drifted — ~11 `kind.rs` slugs (`field-not-found`,
            // `argument-type-mismatch`, `label-expected`, …) were unguarded, so the most
            // common diagnostics could render English inside a ru/zh session with no test
            // failure.  Deriving the check from `en.ftl` itself covers the whole render
            // surface for every module at zero maintenance cost.
            for k in en.keys() {
                if k.starts_with('-') {
                    continue; // terms are locale-local, as above
                }
                assert!(
                    loc.contains_key(k),
                    "{other} is missing {k:?}, which en.ftl defines — that slug renders \
                     English (or the raw slug) in this locale"
                );
            }
        }
    }

    /// A *known* slug missing its required arguments is a caller bug, and the
    /// zero-argument path used to return the raw slug in every profile with no
    /// tripwire.  The assert now fires there as well; `type-mismatch` in `en`
    /// requires `$expected`/`$found`.
    ///
    /// Holds the language lock even though it passes today regardless of locale:
    /// the module's rule is that anything which renders must take the lock while
    /// it reads `Lang::current()`, and a future locale that formats this slug
    /// would otherwise be a latent flake.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "exists but formatting failed")]
    fn known_slug_without_required_arguments_trips_the_tripwire() {
        let _lang_guard = acquire_test_lang_lock();
        let _restore = LangGuard::new(Lang::En);
        let mut out = String::new();
        render_into("type-mismatch", &[], &mut out);
    }

    /// The pair-based writer must *decline* rather than render a partial message, and it
    /// must decline on a duplicate key too.
    ///
    /// A missing variable is exactly the case where Fluent raises an error and
    /// `walk_chain` retries `en`; if the fast writer instead wrote the literal prefix or
    /// silently substituted nothing, that retry would never happen.  Duplicates matter
    /// for the same reason in reverse: `FluentArgs::set` replaces (last wins), so a
    /// first-match writer would produce different bytes for the same input.
    #[test]
    fn simple_pair_writer_declines_missing_and_duplicate_arguments() {
        let _lang = acquire_test_lang_lock();
        let _en = LangGuard::new(Lang::En);
        let lang_idx = Lang::En.index();
        let mut accepted = 0usize;
        for (i, slug) in fluent::SLUG_NAMES.iter().enumerate() {
            let vars = fluent::SLUG_ARGS[i];
            let pairs: SmallVec<[(&'static str, Cow<'_, str>); 4]> =
                vars.iter().map(|n| (*n, Cow::Borrowed("v"))).collect();
            let mut out = String::new();
            if !write_simple_pairs_into(slug, lang_idx, &pairs, &mut out) {
                continue; // not classified simple, or missing in `en`
            }
            accepted += 1;
            if vars.is_empty() {
                continue;
            }
            let mut missing = String::new();
            assert!(
                !write_simple_pairs_into(slug, lang_idx, &[], &mut missing),
                "{slug:?} accepted an empty argument list"
            );
            assert!(
                missing.is_empty(),
                "a declined write must leave the buffer untouched: {missing:?}"
            );
            let mut dup: SmallVec<[(&'static str, Cow<'_, str>); 4]> = pairs.clone();
            dup.push((vars[0], Cow::Borrowed("other")));
            let mut dup_out = String::new();
            assert!(
                !write_simple_pairs_into(slug, lang_idx, &dup, &mut dup_out),
                "{slug:?} accepted a duplicated argument key"
            );
            assert!(dup_out.is_empty(), "{dup_out:?}");
        }
        assert!(
            accepted > 20,
            "only {accepted} slugs accepted a full argument list — the fast path is              narrower than it should be"
        );
    }

    /// The simple-segment writer must reproduce Fluent **byte for byte** for every slug
    /// in every language, and must actually be taken (a fast path nobody enters is a
    /// latent divergence).
    ///
    /// The oracle is `Bundle::format_pattern` itself, not another render path — both
    /// render paths now go through the fast writer, so comparing them would be
    /// `x == x`.  The corpus is the generated slug table plus each slug's declared
    /// variables, so the sweep is exhaustive over the message set rather than a
    /// hand-picked sample.
    #[test]
    fn simple_writer_matches_fluent_byte_for_byte() {
        let _lang = acquire_test_lang_lock();
        let mut took_fast_path = 0usize;
        let mut compared = 0usize;
        for lang in [Lang::En, Lang::Ru, Lang::Zh] {
            let _guard = LangGuard::new(lang);
            let lang_idx = lang.index();
            let bundle = &bundles()[lang_idx];
            for (i, slug) in fluent::SLUG_NAMES.iter().enumerate() {
                let vars = fluent::SLUG_ARGS[i];
                // Distinctive values so a substitution bug cannot hide behind a
                // coincidental match, and non-numeric keys only — a numeric key is
                // exactly the case the writer must decline.
                let mut args = FluentArgs::new();
                for (n, name) in vars.iter().enumerate() {
                    let value = format!("v{n}-{slug}");
                    args.set(*name, FluentValue::String(std::borrow::Cow::Owned(value)));
                }
                let mut got = String::new();
                if !write_simple_into(slug, lang_idx, Some(&args), &mut got) {
                    continue; // declined: Fluent owns this pattern
                }
                let Some(pattern) = bundle.get_message(slug).and_then(|m| m.value()) else {
                    continue; // missing in this locale; the walk would fall back to `en`
                };
                let mut errors = Vec::new();
                let want = bundle.format_pattern(pattern, Some(&args), &mut errors);
                assert!(
                    errors.is_empty(),
                    "slug {slug:?} in {lang:?} had Fluent errors: {errors:?}"
                );
                assert_eq!(
                    got,
                    want.as_ref(),
                    "simple writer diverged from Fluent for {slug:?} in {lang:?}"
                );
                // The hot-path twin must agree with the same oracle.  Both call shapes
                // equalling `want` is also what pins them to *each other* — one
                // classification, two entrances, no drift.
                let pairs: SmallVec<[(&'static str, Cow<'_, str>); 4]> = vars
                    .iter()
                    .enumerate()
                    .map(|(n, name)| (*name, Cow::Owned(format!("v{n}-{slug}"))))
                    .collect();
                let mut got_pairs = String::new();
                assert!(
                    write_simple_pairs_into(slug, lang_idx, &pairs, &mut got_pairs),
                    "the pair-based writer declined {slug:?} in {lang:?} while the \
                     FluentArgs writer accepted it — the entrances disagree"
                );
                assert_eq!(
                    got_pairs,
                    want.as_ref(),
                    "pair-based writer diverged from Fluent for {slug:?} in {lang:?}"
                );
                took_fast_path += 1;
                compared += 1;
            }
        }
        assert_eq!(compared, took_fast_path);
        assert!(
            took_fast_path > 20,
            "only {took_fast_path} slug/language pairs reached the fast path — the \
             classification must accept the ordinary label slugs, not just one or two"
        );
    }

    #[test]
    fn broken_localized_translation_retries_english() {
        let bad = make_bundle(langid!("ru"), "retry-probe = { $missing }");
        let en = make_bundle(langid!("en"), "retry-probe = recovered");
        let mut out = String::new();
        assert!(format_slug_in_into(
            &bad,
            &en,
            "retry-probe",
            None,
            &mut out
        ));
        assert_eq!(out, "recovered");
    }

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

        // PONENT_LANG wins; empty means unset and falls through.
        assert_eq!(detect_from_pair(Some("ru"), Some("zh_TW")), Lang::Ru);
        assert_eq!(detect_from_pair(None, Some("zho_CN")), Lang::Zh);
        assert_eq!(detect_from_pair(Some(""), Some("ru")), Lang::Ru);
        assert_eq!(detect_from_pair(None, None), Lang::En);
        // POSIX precedence inside the fallback chain.
        assert_eq!(
            detect_from_values(
                None,
                Some("zh_TW".into()),
                Some("ru_RU".into()),
                Some("en".into())
            ),
            Lang::Zh
        );
        assert_eq!(
            detect_from_values(None, None, Some("ru_RU".into()), Some("en".into())),
            Lang::Ru
        );
        assert_eq!(
            detect_from_values(None, None, None, Some("ru_RU".into())),
            Lang::Ru
        );
        assert_eq!(
            detect_from_values(Some("en".into()), Some("ru".into()), None, None),
            Lang::En
        );
        assert_eq!(
            detect_from_values(Some(String::new()), None, None, Some("ru".into())),
            Lang::Ru
        );
    }

    /// The zero-arg slug cache must be language-aware: fetching under `En`,
    /// switching to `Zh`, and fetching again must change the output.  If the
    /// generated table ever latched a single language, this fails.
    #[test]
    fn bundle_registry_mappings_agree() {
        // `bundle_index_for`, `lang_id_at` and the `bundles()` array order must
        // agree: a stale index compiles fine and silently serves the wrong
        // locale, which is exactly the bug class `N_LANGS`'s doc records.
        for i in 0..N_LANGS {
            assert_eq!(bundle_index_for(lang_id_at(i)), i);
        }
        // ... and the array literal itself.  That loop pins the *string* mapping
        // but not `bundles()`' order: swapping the ru and zh entries there
        // compiled and passed everything, serving zh text to ru users.  Pin the
        // array so order drift fails independently of which slugs happen to differ.
        for (i, bundle) in bundles().iter().enumerate() {
            assert_eq!(
                bundle_index_for(bundle.locales[0].as_ref()),
                i,
                "bundles()[{i}] is not the bundle for that locale"
            );
        }
    }

    #[test]
    fn zero_arg_slug_is_language_aware() {
        let _lang_guard = acquire_test_lang_lock();
        // `LangGuard` instead of a trailing manual restore: an assertion failure
        // would otherwise leave the process-global latch at `Zh` after the lock
        // is released, and the next test would see the polluted value.
        let _restore = LangGuard::new(Lang::En);
        let en = fluent::zero_arg_slug("main-not-found").expect("main-not-found exists");
        Lang::set(Lang::Zh);
        let zh = fluent::zero_arg_slug("main-not-found").expect("main-not-found exists");
        assert_ne!(en, zh, "zero-arg slug cache did not follow Lang::set");
    }

    /// `render` under a non-EN latch must produce localised, parameterised
    /// output — the end-to-end path the RU/ZH direct-bundle test bypasses.
    #[test]
    fn render_localises_under_explicit_lang() {
        let _lang_guard = acquire_test_lang_lock();
        let _restore = LangGuard::new(Lang::Ru);
        assert_eq!(
            render("main-not-found", &[]),
            "функция `main` не найдена в крейте"
        );
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
        // The latch is process-global; other tests assert English output, so the
        // guard hands the process back in the default language even on unwind.
        let _restore = LangGuard::new(Lang::Zh);
        assert_eq!(Lang::current(), Lang::Zh);
        assert_eq!(Lang::current(), Lang::Zh);
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

    /// The borrowed and owned argument paths parse numbers with *different*
    /// parsers (`FluentValue::try_number` vs `str::parse::<FluentNumber>`), so
    /// the same count string could select a different CLDR plural category
    /// depending on which entry point the caller used — a silent, user-visible
    /// divergence ("1 error" vs "1 errors") in a diagnostics engine.  Nothing
    /// compared them; pin that they agree on every plausible spelling.
    #[test]
    fn borrowed_and_owned_argument_paths_agree_on_numbers() {
        let _lang_guard = acquire_test_lang_lock();
        let _restore = LangGuard::new(Lang::En);
        let corpus = [
            "0", "1", "3", "00123", "+3", " 3", "3 ", "1.0", "1e3", "-1", "0x10", "",
        ];
        for raw in corpus {
            let borrowed = render("summary-errors", &[("count", Cow::Borrowed(raw))]).into_owned();
            let mut owned_out = String::new();
            let pairs = SmallVec::from_elem(("count", Cow::Owned(raw.to_string())), 1);
            render_into_owned("summary-errors", pairs, &mut owned_out);
            assert_eq!(
                borrowed, owned_out,
                "borrowed/owned argument paths disagree for count {raw:?}"
            );
        }
    }

    /// `Diagnostic::humanized_lang` uses `0` as its "never humanized" sentinel, so
    /// no real language may use that discriminant.  Pinned here rather than in
    /// `mod.rs` because this is where the numbering lives.
    #[test]
    fn no_language_uses_the_zero_discriminant() {
        for lang in [Lang::En, Lang::Ru, Lang::Zh] {
            assert_ne!(
                lang as u8, 0,
                "{lang:?} must not be 0 — it is the `humanized_lang` sentinel"
            );
        }
    }

    #[test]
    fn zero_arg_slugs_never_degrade_to_their_slug_name() {
        let _lang_guard = acquire_test_lang_lock();
        for i in 0..N_LANGS {
            let text = fluent::zero_arg_slug_at("main-not-found", i)
                .expect("main-not-found is a zero-arg slug in every bundle");
            assert!(!text.is_empty(), "empty label for language index {i}");
            assert_ne!(
                text, "main-not-found",
                "language index {i} rendered the raw slug name"
            );
        }
        // A bundle that lacks the message entirely must resolve through `en`.
        let empty_ru = make_bundle(langid!("ru"), "# deliberately no messages");
        let en = make_bundle(langid!("en"), "fallback-probe = english text");
        let mut out = String::new();
        assert!(
            format_slug_in_into(&empty_ru, &en, "fallback-probe", None, &mut out),
            "a message missing from the primary bundle must fall back to en"
        );
        assert_eq!(out, "english text", "fallback did not use the English text");
    }
}
