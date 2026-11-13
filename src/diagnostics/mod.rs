//! Ponent diagnostic system.
//!
//! A comprehensive, multi-format diagnostic system modeled after:
//! - **rustc**: error codes (`E030`), `^`-underline annotations, `--explain`, suggestions
//! - **Zig**: `referenced by` call chains
//! - **Austral**: multi-format output (plain / JSON / HTML), error type classification
//! - **Vale**: per-pass error humanizers, structured error types

// ── i18n: translation macro ────────────────────────────────────
//
// Delegates to `i18n::render`: the message is a stable Fluent slug
// (`type-mismatch`) resolved from `locales/*.ftl` in the active
// language; unknown keys fall back verbatim to the key itself.
//
// Usage:
//   tr!("type-mismatch", expected = "Int<32>", found = "Bool")
#[macro_export]
macro_rules! tr {
    ($msg:literal $(, $($key:ident = $val:expr),+ $(,)?)?) => {{
        // SmallVec<[T; 4]> keeps the pair list on the stack for the common
        // case (≤4 args; all current call sites use 1–2).  The old Vec
        // allocated on the heap even for a single argument.
        //
        // Values are lifted into `Cow<'_, str>` via `TrValue`: a `&str`,
        // `Cow<'_, str>` or `String` argument converts without re-formatting
        // (borrowed values stay borrowed); any other `Display` value must be
        // wrapped in `DisplayArg(...)` so the allocation is explicit at the
        // call site.
        let pairs: $crate::smallvec::SmallVec<[(&'static str, std::borrow::Cow<'_, str>); 4]> = {
            let mut v: $crate::smallvec::SmallVec<[(&'static str, std::borrow::Cow<'_, str>); 4]> =
                $crate::smallvec::smallvec![];
            $(
                $(
                    v.push((
                        stringify!($key),
                        $crate::diagnostics::i18n::TrValue::into_cow($val),
                    ));
                )+
            )?
            v
        };
        $crate::diagnostics::i18n::render($msg, &pairs)
    }};
}

pub(crate) mod ascii;
pub mod chain;
pub mod collector;
pub mod emitter;
pub mod error_code;
pub mod explain_server;
pub mod fix;
pub mod glyph;
pub mod i18n;
pub mod kind;
pub mod label;
pub mod level;
pub mod render;
pub mod renderers;
pub mod sink;
pub mod term;
pub(crate) mod wrap;

#[cfg(all(test, not(feature = "shuttle")))]
mod snapshots;

pub use chain::CallChain;
pub use collector::DiagCtxt;
pub use emitter::Emitter;
pub use error_code::{ErrCode, ErrorCategory};
pub use fix::{FixEdit, FixHints, FixSet, apply_fixes, apply_fixes_for_file};
pub use glyph::GlyphRenderer;
pub use i18n::DisplayArg;
pub use kind::{ComptimeReason, DiagnosticKind, Humanizer, TypeCtx};
pub use label::{AnnotationKind, Label, Snippet, SourcePos};
pub use level::DiagnosticLevel;
pub use render::{RenderCtx, Renderer};
pub use renderers::{HtmlRenderer, HumanRenderer, JsonRenderer, ShortRenderer};
pub use sink::{DiagnosticSink, MakeSink, StringSink};

use crate::ast::FileId;
use crate::ast::Span;
use crate::smallvec::{SmallVec, smallvec};
use arrayvec::ArrayVec;
use compact_str::{CompactString, format_compact};
use covariant_vec::CovariantVec;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::fmt;
use std::io;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Diagnostic options (inspired by GHC's DiagOpts) ────────────────

/// Default target line width (display columns) for rendered diagnostics and
/// `--explain` text.  Matches rustc's default of 100.
pub const DEFAULT_MAX_WIDTH: usize = 100;

/// When to emit ANSI colour codes, mirroring `--color=auto/always/never`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorChoice {
    /// Probe the terminal (see [`term::detect_color`]); no codes when piped.
    #[default]
    Auto,
    /// Always emit colour codes, even when piped.
    Always,
    /// Never emit colour codes.
    Never,
}

impl ColorChoice {
    /// Parse a `--color` flag value; unknown values fall back to [`Auto`](Self::Auto).
    pub fn from_str(s: &str) -> Self {
        match s {
            "always" => Self::Always,
            "never" => Self::Never,
            _ => Self::Auto,
        }
    }

    /// Resolve to a concrete `use_color` bit via [`term::detect_color`].
    pub fn resolve(self) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::Auto => term::detect_color(),
        }
    }
}

/// Diagnostic output format, centralising the CLI `--json` flag mapping so
/// every subcommand picks the same renderer for the same flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Human,
    Json,
}

impl OutputFormat {
    pub fn from_json_flag(json: bool) -> Self {
        if json { Self::Json } else { Self::Human }
    }
}

/// Centralized configuration for the diagnostic system.
///
/// Controls filtering, formatting, and display behaviour across all emitter
/// backends (terminal, HTML, JSON, etc.).
#[derive(Debug, Clone)]
pub struct DiagOpts {
    /// Maximum number of errors to report before stopping (`None` = unlimited).
    pub max_errors: Option<usize>,
    /// Treat all warnings as errors.
    pub warn_is_error: bool,
    /// Number of source-context lines to show above and below the primary span.
    pub context_lines: usize,
    /// Whether to emit ANSI colour codes (only relevant for terminal emitters).
    ///
    /// Sharp edge: [`DiagOpts::default`] hard-codes `use_color: true` while
    /// [`ColorChoice::default`] (see `term.rs`) is `Auto`.  Any caller that
    /// builds `DiagOpts::default()` *without* following up with
    /// [`DiagOpts::detect`] (or an explicit `--color` flag override) forces
    /// colour on even when stdout is a pipe.  This is intentional — the
    /// snapshot tests pin `DiagOpts::default()` output as colored — but it
    /// means batch callers must use `detect()` (or set `use_color` themselves)
    /// to get environment-correct behaviour.
    pub use_color: bool,
    /// Whether to emit Unicode box-drawing and highlight glyphs (only
    /// relevant for terminal emitters); `false` selects the ASCII charset.
    pub use_unicode: bool,
    /// Reverse the order in which errors are reported (most recent first).
    pub reverse_errors: bool,
    /// Target line width for rendered diagnostics (display columns).
    /// Messages longer than this are word-wrapped; continuation lines
    /// repeat the box prefix.  Matches rustc's default of 100.
    pub max_width: usize,
    /// Compact layout mode: drops the sub-box spacing and close rows and
    /// simplifies the sub-box header (`│  ┌─ file:1:1` → `│  file:1:1`).
    /// Off by default; the full box-drawing layout is unchanged.
    pub compact: bool,
    /// Terminal colour depth — controls whether 256-colour palette sequences
    /// are emitted or downgraded to the 16-colour ANSI palette.  Probed from
    /// `COLORTERM` / `TERM` by [`DiagOpts::detect`]; the default
    /// ([`ColorDepth::TwoFiftySix`]) matches the historical hard-coded
    /// behaviour.
    pub color_depth: term::ColorDepth,
    /// Tab stop width for display-column calculations.  Tabs expand to the
    /// next `tab_width`-column stop.  Defaults to 4 (matches the historical
    /// hard-coded `TAB_WIDTH`); projects using 2/8-space indents can override.
    pub tab_width: usize,
    pub only_codes: Vec<String>,
    pub only_categories: Vec<ErrorCategory>,
    pub only_files: Vec<String>,
    /// Codes to *suppress*, rustc `-A`-style: a diagnostic whose code is in
    /// this list is filtered out (not rendered, not counted).  The name
    /// follows rustc's `-A`/`--allow` lint levels — an "allowed" diagnostic
    /// is permitted to fail silently.  For the opposite (show *only* these
    /// codes) use [`DiagOpts::only_codes`].
    pub allow_codes: Vec<String>,
    pub quiet: bool,
    pub no_wrap: bool,
}

impl DiagOpts {
    /// Build options with colour, width, and charset probed from the
    /// environment and terminal, suitable for a CLI's default before flag
    /// parsing.
    ///
    /// A `--color=always/never` flag should overwrite `use_color` afterwards;
    /// `max_width` likewise for `--width`.  [`term::supports_hyperlinks`]
    /// exists for the renderer-side follow-up that consumes it.
    pub fn detect() -> Self {
        DiagOpts {
            use_color: term::detect_color(),
            use_unicode: term::detect_unicode(),
            max_width: term::detect_width(),
            color_depth: term::detect_color_depth(),
            tab_width: 4,
            ..Default::default()
        }
    }
}

impl Default for DiagOpts {
    fn default() -> Self {
        DiagOpts {
            max_errors: None,
            warn_is_error: false,
            context_lines: 2,
            use_color: true,
            use_unicode: true,
            reverse_errors: false,
            max_width: DEFAULT_MAX_WIDTH,
            compact: false,
            color_depth: term::ColorDepth::TwoFiftySix,
            tab_width: 4,
            only_codes: Vec::new(),
            only_categories: Vec::new(),
            only_files: Vec::new(),
            allow_codes: Vec::new(),
            quiet: false,
            no_wrap: false,
        }
    }
}

/// Precomputed view of [`DiagOpts`]'s filter fields for O(1) per-diagnostic
/// matching.
///
/// The exact-match filters (`allow_codes`, `only_codes`, `only_categories`)
/// are hashed once at construction; `only_files` stays a `Vec` because it
/// matches by substring (`file.contains(pat)`), not equality.  Building this
/// once per batch turns filter matching from O(diagnostics × filter_size)
/// into O(filter_size + diagnostics).  The common empty-filter case builds
/// three empty hash sets and an empty vec, so a diagnostic pays no lookup
/// work at all.
#[derive(Debug, Clone, Default)]
pub(crate) struct DiagFilter {
    allow_codes: rustc_hash::FxHashSet<String>,
    only_codes: rustc_hash::FxHashSet<String>,
    only_categories: rustc_hash::FxHashSet<ErrorCategory>,
    only_files: Vec<String>,
    /// Precomputed "no filter is set": the common unfiltered batch turns
    /// `matches` into one predictable branch instead of four `is_empty()`
    /// checks per diagnostic.
    is_empty: bool,
}

impl DiagFilter {
    /// Build the fast-match view from a [`DiagOpts`].
    pub(crate) fn from_opts(opts: &DiagOpts) -> Self {
        // Codes are resolved case-insensitively everywhere else in the
        // codebase (`ErrCode::new`, `try_new`, `suggest_code`), and
        // `matches` compares against `ErrCode::code()` — which is the
        // canonical table spelling.  Collecting the raw CLI strings here made
        // `--allow e030` silently match nothing.  Canonicalize once at
        // construction; unknown codes stay verbatim so they still filter.
        let canon = |pat: &str| -> String {
            crate::diagnostics::error_code::lookup_uppercased(pat)
                .map(|e| e.code.to_string())
                .unwrap_or_else(|| pat.to_string())
        };
        let allow_codes: rustc_hash::FxHashSet<String> =
            opts.allow_codes.iter().map(|s| canon(s)).collect();
        let only_codes: rustc_hash::FxHashSet<String> =
            opts.only_codes.iter().map(|s| canon(s)).collect();
        let only_categories: rustc_hash::FxHashSet<ErrorCategory> =
            opts.only_categories.iter().copied().collect();
        let only_files = opts.only_files.clone();
        let is_empty =
            allow_codes.is_empty() && only_codes.is_empty() && only_categories.is_empty() && only_files.is_empty();
        DiagFilter {
            allow_codes,
            only_codes,
            only_categories,
            only_files,
            is_empty,
        }
    }

    /// Whether `diag` passes the filter — mirrors the original
    /// [`Diagnostic::matches_filter`] exactly, with hash lookups for the
    /// exact-match fields.
    pub(crate) fn matches(&self, diag: &Diagnostic<'_>) -> bool {
        if self.is_empty {
            return true;
        }
        let code = diag.code();
        // Suppress list (rustc `-A` semantics): a code in `allow_codes` is
        // *dropped*.  Pinned by `allow_codes_suppresses_matching_diagnostics`.
        if !self.allow_codes.is_empty()
            && let Some(c) = code
            && self.allow_codes.contains(c.canonical())
        {
            return false;
        }
        if !self.only_codes.is_empty() {
            match code {
                Some(c) if self.only_codes.contains(c.canonical()) => {}
                _ => return false,
            }
        }
        if !self.only_categories.is_empty() {
            match code {
                Some(c) if self.only_categories.contains(&c.category()) => {}
                _ => return false,
            }
        }
        if !self.only_files.is_empty() {
            let file = diag.file_name();
            if !self
                .only_files
                .iter()
                .any(|pat| file.contains(pat.as_str()))
            {
                return false;
            }
        }
        true
    }
}

// ── Emission guarantee (inspired by rustc's ErrorGuaranteed) ─────

/// A token proving that a diagnostic (error or warning) has been emitted
/// (or intentionally suppressed).  `EmissionGuarantee::emitted()` means the
/// diagnostic was *actually* recorded; `EmissionGuarantee::suppressed()`
/// means it was filtered out (e.g. a warning when `can_emit_warnings` is
/// false).  Callers can call [`was_emitted`](Self::was_emitted) to
/// distinguish the two cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmissionGuarantee {
    _private: (),
    emitted: bool,
}

impl EmissionGuarantee {
    /// The emitted diagnostic was recorded.
    pub(crate) fn emitted() -> Self {
        EmissionGuarantee {
            _private: (),
            emitted: true,
        }
    }

    /// The diagnostic was suppressed (e.g. warning when warnings are off).
    pub(crate) fn suppressed() -> Self {
        EmissionGuarantee {
            _private: (),
            emitted: false,
        }
    }

    /// Returns `true` if the diagnostic was actually recorded.
    pub fn was_emitted(&self) -> bool {
        self.emitted
    }
}

/// A `Result` that guarantees a diagnostic (error or warning) has been
/// emitted on the `Err` path.  Analogous to rustc's `Result<T, ErrorGuaranteed>`
/// pattern — the `Err` variant carries an `EmissionGuarantee` proving that
/// the diagnostic was recorded, rather than carrying the diagnostic itself.
pub type DiagResult<T> = Result<T, EmissionGuaranteed>;

/// Opaque wrapper around `EmissionGuarantee` for use in `DiagResult`.
/// Functions that return `DiagResult<T>` produce an `EmissionGuaranteed`
/// on the error path, proving a diagnostic was emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmissionGuaranteed(EmissionGuarantee);

impl From<EmissionGuarantee> for EmissionGuaranteed {
    fn from(g: EmissionGuarantee) -> Self {
        EmissionGuaranteed(g)
    }
}

/// A lazily-formatted diagnostic message, analogous to rustc's
/// `DiagMessage`: the message is DATA until a renderer commits to a
/// concrete string.
///
/// Two shapes:
/// - [`DiagMessage::Static`] — the full English text (or any pre-rendered
///   string), borrowed when possible (`Cow<'src, str>`), so fully static
///   messages allocate NOTHING.
/// - [`DiagMessage::Slug`] — a Fluent slug plus typed arguments, resolved
///   through the active language's bundle at materialization time.  Word
///   order lives in `locales/*.ftl`, not in `format!` templates.
///
/// Materialization is [`DiagMessage::render_into`] (buffer-reusing) /
/// [`DiagMessage::to_string`]; `Diagnostic::humanize` is the single
/// boundary where the collector commits.
#[derive(Debug, Clone)]
pub enum DiagMessage<'src> {
    Static(Cow<'src, str>),
    /// A Fluent slug (`locales/*.ftl` message id) with its arguments.
    /// `pairs` is ordered for deterministic rendering; keys must match
    /// the `$placeholders` in the .ftl entries.
    Slug {
        slug: &'static str,
        pairs: SmallVec<[(&'static str, Cow<'src, str>); 4]>,
    },
}

impl<'src> DiagMessage<'src> {
    /// Static text, borrowing a `'static` string when possible.
    pub fn static_str(s: &'static str) -> Self {
        DiagMessage::Static(Cow::Borrowed(s))
    }

    /// Fluent slug with arguments (see [`DiagMessage::Slug`]).
    pub fn slug(slug: &'static str, pairs: SmallVec<[(&'static str, Cow<'src, str>); 4]>) -> Self {
        DiagMessage::Slug { slug, pairs }
    }

    /// Materialize into `out` (reusing its capacity across calls — the
    /// collector's hot path), routing slugs through the i18n bundle.
    pub fn render_into(&self, out: &mut String) {
        match self {
            DiagMessage::Static(s) => out.push_str(s),
            DiagMessage::Slug { slug, pairs } => {
                crate::diagnostics::i18n::render_into(slug, pairs, out);
            }
        }
    }

    /// Consuming variant of [`render_into`](Self::render_into): moves
    /// `Slug`-pair arguments into the Fluent map instead of cloning them.
    /// Only callable when the caller owns the message (e.g. `humanize`,
    /// which gets a fresh `DiagMessage` from `humanize_into`).
    pub fn render_into_owned(self, out: &mut String) {
        match self {
            DiagMessage::Static(s) => out.push_str(&s),
            DiagMessage::Slug { slug, pairs } => {
                crate::diagnostics::i18n::render_into_owned(slug, pairs, out);
            }
        }
    }

    /// Materialize into a fresh `String` (cold paths, tests).
    pub fn to_rendered(&self) -> String {
        let mut s = String::new();
        self.render_into(&mut s);
        s
    }

    /// Borrow the static text, if this message is fully static.
    pub fn as_static(&self) -> Option<&str> {
        match self {
            DiagMessage::Static(s) => Some(s),
            DiagMessage::Slug { .. } => None,
        }
    }
}

impl Default for DiagMessage<'_> {
    fn default() -> Self {
        DiagMessage::Static(Cow::Borrowed(""))
    }
}

impl From<String> for DiagMessage<'_> {
    fn from(s: String) -> Self {
        DiagMessage::Static(Cow::Owned(s))
    }
}

impl From<&str> for DiagMessage<'_> {
    fn from(s: &str) -> Self {
        DiagMessage::Static(Cow::Owned(s.to_owned()))
    }
}

// NOTE: no `From<&'static str>` specialization — it conflicts with
// `From<&str>` (the impls overlap for every 'static input).  Zero-copy
// static text goes through the explicit `DiagMessage::static_str`
// constructor instead.

impl<'a> std::fmt::Display for DiagMessage<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagMessage::Static(s) => f.write_str(s),
            DiagMessage::Slug { slug, pairs } => {
                f.write_str(&crate::diagnostics::i18n::render(slug, pairs))
            }
        }
    }
}

/// A set of spans that a diagnostic points to, supporting multiple primary
/// locations.  Analogous to rustc's `MultiSpan`.  When there is a single
/// primary span, `primary` contains one element; for multiple (e.g. a
/// duplicate definition + the previous definition), both are primary.
#[derive(Debug, Clone, Default)]
pub struct MultiSpan {
    // Invariant: when `!dirty`, sorted by `(file, start, end)` ascending
    // (`Span`'s `Ord`), no duplicates. All mutation goes through `push`,
    // which preserves this.  When `dirty`, the vector is unsorted and may
    // contain duplicates; `primary()` sorts + dedups before returning.
    primary: SmallVec<[Span; 2]>,
    dirty: bool,
    /// Cached sorted+deduped view for the `dirty` state, populated on the
    /// first `primary()`/`primary_ref()` read after a dirtying `push` (a
    /// boxed slice, not a clone of the inline `SmallVec`: the cache must not
    /// grow `Diagnostic` past its size guard, and the dirty path is rare
    /// enough that one heap allocation is noise).
    /// Interior mutability is safe here: `MultiSpan` only ever crosses
    /// threads by *move* (`&mut Diagnostic` chunks in the emitter's scoped
    /// threads need `Send`, not `Sync`, and `OnceLock<Box<[..]>>` stays
    /// `Send` *and* `Sync` — the latter is required because `Diagnostic`
    /// values are stored in the query cache (`FnBodyCheck`), whose storage
    /// is `Sync`).
    /// Invalidated by every `push`/`clear`.
    sorted_cache: std::sync::OnceLock<Box<[Span]>>,
}

impl MultiSpan {
    pub fn new(span: Span) -> Self {
        MultiSpan {
            primary: smallvec![span],
            dirty: false,
            sorted_cache: std::sync::OnceLock::new(),
        }
    }

    /// Insert a primary span, appending in O(1).
    ///
    /// Duplicates of the *tail* are ignored immediately; a duplicate of an
    /// earlier element is appended and removed by the deferred sort+dedup on
    /// the next [`primary`](Self::primary) / [`first`](Self::first) read, so
    /// the observable span set stays idempotent (safe for repeated
    /// `humanize`) while the `SmallVec` may transiently hold a duplicate.
    ///
    /// Compilers emit spans in ascending source order, so the fast path
    /// compares against the tail and appends in O(1).  An out-of-order span
    /// is appended in O(1) and the vector is marked `dirty`; the sort +
    /// dedup is deferred to the next [`primary`](Self::primary) or
    /// [`first`](Self::first) read, so a fully reverse-sorted push stream
    /// costs `O(n log n)` total instead of `O(n²)` memmoves.
    pub fn push(&mut self, span: Span) {
        // The sorted view cache is dropped on every push: `OnceCell::take`
        // is a single `Option` swap, no allocation, and re-population only
        // happens on the next dirty read.
        self.sorted_cache.take();
        match self.primary.last() {
            Some(last) if *last == span => {}
            Some(last) if *last > span => {
                self.primary.push(span);
                self.dirty = true;
            }
            _ => self.primary.push(span),
        }
    }

    /// The earliest primary span, or `None` if empty.  When `dirty`, scans
    /// for the minimum in `O(n)` without sorting; otherwise `O(1)`.  If a
    /// previous `primary_ref()` already populated the sorted cache, read its
    /// first entry instead of rescanning.
    pub fn first(&self) -> Option<Span> {
        if self.dirty {
            if let Some(c) = self.sorted_cache.get() {
                return c.first().copied();
            }
            self.primary.iter().min().copied()
        } else {
            self.primary.first().copied()
        }
    }

    /// Clear all primary spans.
    pub fn clear(&mut self) {
        self.primary.clear();
        self.dirty = false;
        self.sorted_cache.take();
    }

    /// Return the spans in sorted order.  When `dirty`, sorts + dedups once
    /// and caches the result in `sorted_cache` — repeated reads between
    /// pushes skip the `O(n log n)` re-sort; otherwise clones the
    /// already-sorted vector (inline for `n ≤ 2`, so no heap allocation in
    /// the common case).
    pub fn primary(&self) -> SmallVec<[Span; 2]> {
        // Route through `primary_ref` so the sort/dedup/cache discipline has
        // exactly one implementation: a dirty span sorts once and both
        // accessors then read the cached view.
        self.primary_ref().iter().copied().collect()
    }

    /// The primary spans as a borrowed slice when possible, avoiding the
    /// [`SmallVec`] clone that [`primary`](Self::primary) performs even when
    /// the vector is already sorted (`!dirty`, the common case —
    /// [`push`](Self::push) keeps it in ascending order).  On a `dirty`
    /// read it either borrows the cached sorted view or owns a freshly
    /// sorted + deduped `Vec` (which is then cached, so subsequent dirty
    /// reads borrow too).  Callers that only iterate or read `len`/`first`
    /// pay no clone on the fast path.
    pub fn primary_ref(&self) -> Cow<'_, [Span]> {
        if !self.dirty {
            Cow::Borrowed(self.primary.as_ref())
        } else if let Some(cached) = self.sorted_cache.get() {
            // The cache is behind `&self`, so the borrowed view is exact.
            Cow::Borrowed(cached)
        } else {
            let mut out: SmallVec<[Span; 2]> = self.primary.clone();
            out.sort_unstable();
            out.dedup();
            debug_assert!(
                out.windows(2).all(|w| w[0] <= w[1]),
                "MultiSpan::primary must be sorted by (file, start)"
            );
            // Cache the sorted view so repeated reads between pushes skip
            // the sort; the boxed-slice allocation is one small heap copy
            // against an `O(n log n)` sort per read.  Return a borrow of the
            // cache rather than cloning it: the cache lives behind `&self`,
            // so the borrow is exact and the dirty path costs no extra copy.
            let _ = self.sorted_cache.set(out.into_vec().into_boxed_slice());
            // Two threads may both compute and both attempt `set`; the loser
            // then reads the winner's value.  `OnceLock::set` is release and
            // `get` is acquire, so a successful `set` *or* a concurrent
            // winner's `set` is visible here — the value cannot be observed as
            // `None`.  This relies on `OnceLock`'s synchronization: replacing
            // it with an unsynchronized cell would make the read racy.
            debug_assert!(
                self.sorted_cache.get().is_some(),
                "sorted cache set immediately before this read"
            );
            Cow::Borrowed(self.sorted_cache.get().expect("just set"))
        }
    }
}

/// How confident we are that a suggestion is correct.
/// Inspired by rustc's `Applicability`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applicability {
    /// The suggestion can be applied automatically without review.
    MachineApplicable,
    /// The suggestion might be correct, but needs human review.
    MaybeIncorrect,
    /// The suggestion has placeholders that need to be filled in.
    HasPlaceholders,
    /// The applicability is unspecified.
    Unspecified,
}

impl Default for Applicability {
    /// `Unspecified` is the safe fallback: an auto-applied suggestion the
    /// author never classified (API misuse of the default) must not be
    /// treated as machine-applicable without review.
    fn default() -> Self {
        Applicability::Unspecified
    }
}

/// How a suggestion should be displayed.
/// Inspired by rustc's `SuggestionStyle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SuggestionStyle {
    /// Show the suggestion inline with the diagnostic.
    #[default]
    ShowAlways,
    /// Show the suggestion code, but hide the message.
    ShowCode,
    /// Show the message, but hide the inline code.
    HideCodeInline,
    /// Hide the suggestion entirely from the user.
    HideCodeAlways,
    /// Completely hidden (only visible to tools, e.g. IDE).
    CompletelyHidden,
}

/// The JSON wire name of an [`Applicability`] variant.
pub fn applicability_name(app: Applicability) -> &'static str {
    match app {
        Applicability::MachineApplicable => "MachineApplicable",
        Applicability::MaybeIncorrect => "MaybeIncorrect",
        Applicability::HasPlaceholders => "HasPlaceholders",
        Applicability::Unspecified => "Unspecified",
    }
}

/// The JSON wire name of a [`SuggestionStyle`] variant.
pub fn suggestion_style_name(style: SuggestionStyle) -> &'static str {
    match style {
        SuggestionStyle::ShowAlways => "ShowAlways",
        SuggestionStyle::ShowCode => "ShowCode",
        SuggestionStyle::HideCodeInline => "HideCodeInline",
        SuggestionStyle::HideCodeAlways => "HideCodeAlways",
        SuggestionStyle::CompletelyHidden => "CompletelyHidden",
    }
}

/// The JSON wire name of an [`AnnotationKind`] variant.
pub fn annotation_kind_name(k: AnnotationKind) -> &'static str {
    match k {
        AnnotationKind::Primary => "Primary",
        AnnotationKind::Secondary => "Secondary",
        AnnotationKind::Note => "Note",
        AnnotationKind::Help => "Help",
    }
}

/// The JSON wire name of a [`chain::ChainEntryKind`] variant.
pub fn chain_entry_kind_name(k: chain::ChainEntryKind) -> &'static str {
    match k {
        chain::ChainEntryKind::Call => "Call",
        chain::ChainEntryKind::Inlined => "Inlined",
        chain::ChainEntryKind::Reference => "Reference",
        chain::ChainEntryKind::Defines => "Defines",
    }
}

/// A structured suggestion with confidence and style metadata.
#[derive(Debug, Clone)]
pub struct Suggestion<'src> {
    pub message: Cow<'src, str>,
    pub applicability: Applicability,
    pub style: SuggestionStyle,
    /// Source span the suggestion replaces.  When present together with
    /// `replacement`, renderers emit a two-line `-`/`+` diff instead of a
    /// one-line help/note.
    pub span: Option<Span>,
    /// Replacement text for `span`.
    pub replacement: Option<Cow<'src, str>>,
}

impl<'src> std::fmt::Display for Suggestion<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tag = match self.applicability {
            Applicability::MachineApplicable => "",
            Applicability::MaybeIncorrect => " (maybe incorrect)",
            Applicability::HasPlaceholders => " (has placeholders)",
            Applicability::Unspecified => "",
        };
        write!(f, "{}{}", self.message, tag)
    }
}

/// A segment of styled text, used in notes and help messages.
/// Inspired by rustc's `StringPart`.
#[derive(Debug, Clone)]
pub struct StringPart<'src> {
    pub content: Cow<'src, str>,
    pub style: StringPartStyle,
}

/// Whether a `StringPart` should be highlighted or rendered normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPartStyle {
    Normal,
    Highlighted,
}

/// A sequence of styled text segments, used for rich diagnostic messages.
/// When rendered, highlighted segments are shown in a distinct color
/// (e.g. bright blue for type names) while normal segments use the default color.
#[derive(Debug, Clone)]
pub struct StyledString<'src>(pub Vec<StringPart<'src>>);

impl<'src> StyledString<'src> {
    pub fn new(parts: Vec<StringPart<'src>>) -> Self {
        StyledString(parts)
    }

    pub fn plain(text: impl Into<Cow<'src, str>>) -> Self {
        StyledString(vec![StringPart {
            content: text.into(),
            style: StringPartStyle::Normal,
        }])
    }

    pub fn highlighted(text: impl Into<Cow<'src, str>>) -> Self {
        StyledString(vec![StringPart {
            content: text.into(),
            style: StringPartStyle::Highlighted,
        }])
    }

    /// Render the styled string as plain text (without ANSI codes).
    pub fn to_plain(&self) -> String {
        self.0.iter().map(|p| p.content.as_ref()).collect()
    }
}

/// A complete compiler diagnostic (error, warning, help, note, or info).
///
/// Architecture inspired by multiple compilers:
/// - **rustc**: Error codes + `^`-underline annotations + suggestions + sub-diagnostics
/// - **Zig**: `referenced by` call chains for traceability
/// - **Austral**: Error type categorization + multi-format output
/// - **Vale**: Structured error variants per compiler pass
///
/// # Lazy humanization
///
/// When built from a structured [`DiagnosticKind`] via
/// [`Diagnostic::error_kind`] / [`fatal_kind`] / [`warning_kind`],
/// `message` / `labels` / `spans` / `help` / `suggestions` stay
/// unfilled until [`humanize`](Self::humanize) runs.  The constructors
/// deliberately skip that step so:
///
/// - **Filtered-out diagnostics pay nothing** — those dropped by
///   `max_errors` or suppressed warnings
///   (`can_emit_warnings = false`) are never humanized.
/// - **i18n language resolves at render time** — Fluent runs against
///   the *current* `Lang`, so a long-lived process (LSP) can
///   `Lang::set` between emits without freezing messages in the
///   language that happened to be active at construction.
///
/// The [`Emitter`] calls `humanize()` at its render boundary; callers
/// that read these fields directly must call `humanize()` themselves.
/// `kind` is retained (never dropped) as the structured source of
/// truth so `humanize()` can re-run if the kind is replaced.
/// Restores a temporarily-taken `Diagnostic::kind` if `humanize_into` panics.
/// `disarm` on the success path returns the kind without restoring twice.
struct RestoreKindOnDrop<'a, 'src> {
    slot: &'a mut Option<std::sync::Arc<DiagnosticKind<'src>>>,
    restore: Option<std::sync::Arc<DiagnosticKind<'src>>>,
}

impl Drop for RestoreKindOnDrop<'_, '_> {
    fn drop(&mut self) {
        if let Some(kind) = self.restore.take() {
            *self.slot = Some(kind);
        }
    }
}

impl<'a, 'src> RestoreKindOnDrop<'a, 'src> {
    fn disarm(mut self) -> std::sync::Arc<DiagnosticKind<'src>> {
        self.restore.take().expect("guard armed")
    }
}

#[derive(Debug)]
pub struct Diagnostic<'src> {
    level: DiagnosticLevel,
    message: Cow<'src, str>,
    code: Option<ErrCode>,
    /// Structured error kind (Vale-style ADT).  When present, the humanizer
    /// can derive `message`, `labels`, and other fields automatically.
    kind: Option<std::sync::Arc<DiagnosticKind<'src>>>,
    /// Primary spans (multiple for multi-location diagnostics).
    spans: MultiSpan,
    help: Option<Cow<'src, str>>,
    /// Structured suggestions with applicability and style metadata.
    suggestions: CovariantVec<'src, Suggestion<'src>, 1>,
    /// Multi-span annotations for source-context rendering
    labels: CovariantVec<'src, Label<'src>, 1>,
    /// Zig-style call chain tracing how the error was reached
    call_chain: Option<CallChain<'src>>,
    /// Retained source text for `^`-underline rendering.  Borrowed from the
    /// driver's source buffer, so attaching it to every diagnostic in a
    /// compilation costs nothing.
    source: Option<&'src str>,
    /// Source file name displayed in the `┌─ <input>:1:1` location header.
    file_name: Cow<'src, str>,
    /// Child diagnostics (notes, help messages attached to this diagnostic).
    children: Vec<Subdiag<'src>>,
    /// Related errors displayed alongside this diagnostic in a single merged box.
    related_errors: Vec<RelatedError<'src>>,
    foreign_sources: Option<Box<Vec<(FileId, Cow<'src, str>, &'src str)>>>,
    /// Whether `spans` were set by a user-facing builder rather than left
    /// at the constructor default.  `humanize()` leaves them alone in that
    /// case instead of deriving fresh primary spans from the kind's labels.
    spans_user_set: bool,
    /// Whether `help` was set by a user-facing builder rather than derived
    /// from `kind`. `humanize()` leaves it alone in that case instead of
    /// silently overwriting it (C3).
    help_user_set: bool,
    /// Whether `suggestions` were set by a user-facing builder; `humanize()`
    /// must not clear/replace them (C3), matching the spans precedent.
    suggestions_user_set: bool,
    /// How many entries at the *front* of `labels` were derived from `kind`
    /// by the last [`humanize`](Self::humanize).  User labels appended after
    /// that block (`with_label`, `with_secondary_label`, `augment`) survive
    /// re-humanization: the derived block is truncated and re-derived in the
    /// new language, the user block is preserved.  Labels are kind-derived
    /// first, then user-added — the same order the builders establish, since
    /// user builders only ever append.
    kind_derived_labels: u8,
    /// Whether [`humanize`](Self::humanize) has already populated `message`
    /// and `labels` from `kind`. Guards against redundant re-rendering on a
    /// second call (the operation is idempotent, but not free).
    humanized: bool,
    /// The [`Lang`](crate::diagnostics::i18n::Lang) discriminant active at
    /// the last [`humanize`](Self::humanize) (`0` = never humanized).  When
    /// the process language changes, the next `humanize()` re-derives
    /// `message`/`labels`/`help` from the retained `kind` in the new
    /// language — the LSP re-emit contract documented on
    /// [`error_kind`](Self::error_kind).
    humanized_lang: u8,
    /// Cached result of [`stable_id`](Self::stable_id). `0` means "not yet
    /// computed" — FNV-1a over non-empty input is virtually never exactly 0,
    /// and even if it were, the only consequence is recomputing on every
    /// call, which still yields a consistent result. See
    /// [`Self::invalidate_stable_id`].
    stable_id_cache: AtomicU64,
}

impl<'src> Clone for Diagnostic<'src> {
    fn clone(&self) -> Self {
        Diagnostic {
            level: self.level,
            message: self.message.clone(),
            code: self.code.clone(),
            kind: self.kind.clone(),
            spans: self.spans.clone(),
            help: self.help.clone(),
            suggestions: self.suggestions.clone(),
            labels: self.labels.clone(),
            call_chain: self.call_chain.clone(),
            source: self.source,
            file_name: self.file_name.clone(),
            children: self.children.clone(),
            related_errors: self.related_errors.clone(),
            foreign_sources: self.foreign_sources.clone(),
            spans_user_set: self.spans_user_set,
            help_user_set: self.help_user_set,
            suggestions_user_set: self.suggestions_user_set,
            kind_derived_labels: self.kind_derived_labels,
            humanized: self.humanized,
            humanized_lang: self.humanized_lang,
            stable_id_cache: AtomicU64::new(self.stable_id_cache.load(Ordering::Relaxed)),
        }
    }
}

impl<'src> Diagnostic<'src> {
    // ── Read-only accessors ─────────────────────────────────────────
    pub fn level(&self) -> DiagnosticLevel {
        self.level
    }
    pub fn message(&self) -> &str {
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "read message() before humanize()"
        );
        &self.message
    }
    pub(crate) fn message_unchecked(&self) -> &str {
        &self.message
    }
    pub fn code(&self) -> Option<&ErrCode> {
        self.code.as_ref()
    }
    pub fn kind(&self) -> Option<&DiagnosticKind<'src>> {
        self.kind.as_deref()
    }
    pub fn spans(&self) -> &MultiSpan {
        &self.spans
    }
    pub fn help(&self) -> Option<&str> {
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "read help() before humanize()"
        );
        self.help.as_deref()
    }
    pub fn suggestions(&self) -> &[Suggestion<'src>] {
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "read suggestions() before humanize()"
        );
        &self.suggestions
    }
    pub fn labels(&self) -> &[Label<'src>] {
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "read labels() before humanize()"
        );
        &self.labels
    }
    pub fn call_chain(&self) -> Option<&CallChain<'src>> {
        self.call_chain.as_ref()
    }
    pub fn source(&self) -> Option<&str> {
        self.source
    }
    pub fn file_name(&self) -> &str {
        &self.file_name
    }
    pub fn children(&self) -> &[Subdiag<'src>] {
        &self.children
    }
    pub fn children_mut(&mut self) -> &mut Vec<Subdiag<'src>> {
        &mut self.children
    }
    pub fn related_errors(&self) -> &[RelatedError<'src>] {
        &self.related_errors
    }

    // ── Mutable accessors ───────────────────────────────────────────
    pub fn labels_mut(&mut self) -> &mut CovariantVec<'src, Label<'src>, 1> {
        &mut self.labels
    }
    pub fn related_errors_mut(&mut self) -> &mut Vec<RelatedError<'src>> {
        &mut self.related_errors
    }
    pub fn set_source(&mut self, source: Option<&'src str>) {
        self.source = source;
    }
    pub fn set_file_name(&mut self, file_name: impl AsRef<str>) {
        self.file_name = Cow::Owned(file_name.as_ref().to_owned());
        // C9: `stable_id` hashes the file name, so a rename must invalidate a
        // previously computed id (otherwise a stale id keyed to the old name
        // is returned, reintroducing cross-file collisions).
        self.invalidate_stable_id();
    }

    pub fn with_foreign_source(
        mut self,
        file: FileId,
        name: impl Into<Cow<'src, str>>,
        source: &'src str,
    ) -> Self {
        let sources = self.foreign_sources.get_or_insert_with(Box::default);
        if !sources.iter().any(|(f, _, _)| *f == file) {
            sources.push((file, name.into(), source));
        }
        self
    }

    pub fn source_for_file(&self, file: FileId) -> Option<&'src str> {
        if self.spans().first().is_some_and(|s| s.file == file) {
            return self.source;
        }
        self.foreign_sources
            .as_deref()
            .and_then(|sources| sources.iter().find(|(f, _, _)| *f == file))
            .map(|(_, _, s)| *s)
    }

    pub fn file_name_for(&self, file: FileId) -> Option<&str> {
        if self.spans().first().is_some_and(|s| s.file == file) {
            return Some(self.file_name());
        }
        self.foreign_sources
            .as_deref()
            .and_then(|sources| sources.iter().find(|(f, _, _)| *f == file))
            .map(|(_, n, _)| n.as_ref())
    }

    /// A deterministic identifier derived from `(code, first primary span,
        /// message, file_name)`, for tools to deduplicate / track diagnostics
        /// across runs.  `file_name` participates so identical code/spans/messages
        /// in different files do not collide.
        ///
        /// Computed once with FNV-1a and cached in an [`AtomicU64`]; when the
        /// inputs are unchanged, a second call is just a relaxed load. Every
        /// entry point that can mutate the fields above
        /// ([`with_code`](Self::with_code), [`with_code_str`](Self::with_code_str),
        /// [`with_span`](Self::with_span), [`with_additional_span`](Self::with_additional_span),
        /// [`augment`](Self::augment), [`humanize`](Self::humanize),
        /// [`set_file_name`](Self::set_file_name), and
        /// [`adorn_with_span`](Self::adorn_with_span)) clears the cache via
        /// [`invalidate_stable_id`](Self::invalidate_stable_id), so callers
        /// always see a value consistent with the current inputs.
        pub fn stable_id(&self) -> u64 {
        let cached = self.stable_id_cache.load(Ordering::Relaxed);
        if cached != 0 {
            return cached;
        }
        let id = self.compute_stable_id();
        self.stable_id_cache.store(id, Ordering::Relaxed);
        id
    }

    #[inline]
    fn compute_stable_id(&self) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = FNV_OFFSET;
        let mut feed = |b: u8| {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        };
        if let Some(c) = self.code() {
            // `canonical`, not `code`: the latter echoes the verbatim spelling
            // for `ErrCode::unknown("e030")`, while `PartialEq`/`Hash` treat
            // it as `E030`.  Hashing the verbatim form would let two `==`
            // diagnostics emit different ids, breaking the dedup contract this
            // accessor exists for.
            for b in c.canonical().as_bytes() {
                feed(*b);
            }
        }
        feed(0);
        if let Some(span) = self.spans().first() {
            for b in span.file.as_u32().to_le_bytes() {
                feed(b);
            }
            for b in span.start.to_le_bytes() {
                feed(b);
            }
            for b in span.end.to_le_bytes() {
                feed(b);
            }
        }
        feed(0);
        // `message_unchecked`, not `message()`: `stable_id` is a public
        // accessor with no documented humanize precondition, so it must not
        // debug-panic on a kind-built diagnostic read before `humanize()`.
        // The cache is invalidated by `humanize`, so the id recomputes once
        // the real message is populated.
        for b in self.message_unchecked().as_bytes() {
            feed(*b);
        }
        // The file NAME (not `FileId`, which is process-local) so two
        // diagnostics with identical code/span/message in different files do
        // not collide and silently suppress each other in a tool's dedupe
        // set.  `set_file_name`/`with_file_name` therefore invalidate the
        // cache: a caller that reads the id, renames, and reads again must see
        // a new id keyed to the new name (C9), not the stale one.
        feed(0);
        for b in self.file_name().as_bytes() {
            feed(*b);
        }
        h
    }

    /// Clears the [`stable_id`](Self::stable_id) cache.
    ///
    /// Every mutator that changes `code`, the first primary span, or
    /// `message` must call this. Entry points that only touch fields not
    /// covered by the hash (`source`, `file_name`, `level`, `labels`,
    /// `related_errors`, `children`, `suggestions`, `help`, `call_chain`,
    /// `kind`) do not need to — the cache is naturally immune to those
    /// changes.
    #[inline]
    fn invalidate_stable_id(&mut self) {
        self.stable_id_cache.store(0, Ordering::Relaxed);
    }

    /// Whether this diagnostic passes the filter described by `opts`.
    ///
    /// Builds a [`DiagFilter`] from `opts` on every call, which is O(filter_size)
    /// allocations.  For checking many diagnostics against the same options
    /// (e.g. inside a loop), build the filter once with
    /// [`DiagFilter::from_opts`] and call [`matches_filter_with`] instead.
    ///
    /// [`matches_filter_with`]: Diagnostic::matches_filter_with
    pub fn matches_filter(&self, opts: &DiagOpts) -> bool {
        DiagFilter::from_opts(opts).matches(self)
    }

    /// Whether this diagnostic passes a pre-built [`DiagFilter`].
    ///
    /// Loop-safe variant of [`matches_filter`]: the caller builds the filter
    /// once and reuses it across many diagnostics, so the per-diagnostic cost
    /// is O(1) hash lookups rather than O(filter_size) allocations.
    pub(crate) fn matches_filter_with(&self, filter: &DiagFilter) -> bool {
        filter.matches(self)
    }
}

/// A child diagnostic attached to a parent `Diagnostic`.
/// Inspired by rustc's `Subdiag` — used for notes, help, or secondary
/// labels that are logically part of the same diagnostic group.
#[derive(Debug, Clone)]
pub struct Subdiag<'src> {
    pub level: DiagnosticLevel,
    /// Plain-text form of the message.
    ///
    /// `None` when the message exists only as [`styled_message`]
    /// ([`styled_message`]: Self::styled_message) — the plain form is then
    /// derived on demand by [`plain_message`](Self::plain_message) instead of
    /// being materialized at construction time.
    pub message: Option<Cow<'src, str>>,
    /// Optional styled version of the message, with highlighted parts.
    /// When present, the glyph renderer uses this instead of the plain
    /// `message` to render rich text with color highlights.
    pub styled_message: Option<StyledString<'src>>,
    /// Optional source span for the child.  Retained for tooling and other
    /// backends; the human renderer renders only the message, so this is
    /// currently carried but not drawn.
    pub span: Option<Span>,
    /// Child labels.  Retained for tooling and other backends; the human
    /// renderer renders only the message, so these are currently carried
    /// but not drawn.
    pub labels: Vec<Label<'src>>,
}

/// A related error displayed alongside the primary diagnostic in a single
/// merged box.  Each related error has its own error code, message, and
/// optional source span — for example, a type mismatch [E030] aggregated
/// into a duplicate definition [E019] diagnostic.
#[derive(Debug, Clone)]
pub struct RelatedError<'src> {
    pub code: Option<ErrCode>,
    pub message: Cow<'src, str>,
    pub span: Option<Span>,
    /// Optional label for the related span.  Retained for tooling and other
    /// backends; the human renderer shows only `message`/`code`.
    pub label: Option<Cow<'src, str>>,
}

impl<'src> Subdiag<'src> {
    /// Create a new sub-diagnostic.
    pub fn new(level: DiagnosticLevel, message: impl Into<Cow<'src, str>>) -> Self {
        Subdiag {
            level,
            message: Some(message.into()),
            styled_message: None,
            span: None,
            labels: Vec::new(),
        }
    }

    /// Plain-text message, borrowing the stored form when present and
    /// deriving it from [`styled_message`](Self::styled_message) otherwise.
    pub fn plain_message(&self) -> Cow<'_, str> {
        match (&self.message, &self.styled_message) {
            (Some(message), _) => Cow::Borrowed(message.as_ref()),
            (None, Some(styled)) => Cow::Owned(styled.to_plain()),
            (None, None) => Cow::Borrowed(""),
        }
    }

    /// Create a note sub-diagnostic.
    pub fn note(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Note, message)
    }

    /// Create a help sub-diagnostic.
    pub fn help_sub(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Help, message)
    }

    /// Attach a source span to this sub-diagnostic.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a label to this sub-diagnostic.
    pub fn with_label(mut self, span: Span, label: impl Into<Cow<'src, str>>) -> Self {
        self.labels.push(Label::new(span, label));
        self
    }
}

impl<'src> Diagnostic<'src> {
    // ── Constructors ───────────────────────────────────────────────

    pub fn new(level: DiagnosticLevel, message: impl Into<Cow<'src, str>>) -> Self {
        Diagnostic {
            level,
            message: message.into(),
            code: None,
            kind: None,
            spans: MultiSpan::default(),
            help: None,
            suggestions: CovariantVec::new(),
            labels: CovariantVec::new(),
            call_chain: None,
            source: None,
            file_name: "<input>".into(),
            children: Vec::new(),
            related_errors: Vec::new(),
            foreign_sources: None,
            humanized: false,
            humanized_lang: 0,
            spans_user_set: false,
            help_user_set: false,
            suggestions_user_set: false,
            kind_derived_labels: 0,
            stable_id_cache: AtomicU64::new(0),
        }
    }

    pub fn error(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Error, message)
    }

    pub fn fatal(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Fatal, message)
    }

    pub fn warning(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Warning, message)
    }

    /// Create an error diagnostic from a structured [`DiagnosticKind`].
    ///
    /// # Lazy humanization
    ///
    /// `message` / `labels` / `spans` / `help` / `suggestions` stay
    /// unfilled until [`humanize`](Self::humanize) runs.  The constructor
    /// deliberately skips that step so:
    ///
    /// - **Filtered-out diagnostics pay nothing** — those dropped by
    ///   `max_errors` or suppressed warnings (`can_emit_warnings = false`)
    ///   are never humanized.
    /// - **i18n language resolves at render time** — Fluent runs against
    ///   the *current* `Lang`, so a long-lived process (LSP) can
    ///   `Lang::set` between emits without freezing messages in the
    ///   language that happened to be active at construction.
    ///
    /// The [`Emitter`] calls `humanize()` at its render boundary; callers
    /// that read these fields directly must call `humanize()` themselves.
    /// `kind` is retained (never dropped) as the structured source of
    /// truth so `humanize()` can re-run if the kind is replaced.
    pub fn error_kind(kind: DiagnosticKind<'src>) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Error, "");
        diag.kind = Some(std::sync::Arc::new(kind));
        diag
    }

    /// Create a fatal diagnostic from a structured [`DiagnosticKind`].
    ///
    /// See [`error_kind`](Self::error_kind) for the lazy-humanization
    /// contract: derived fields stay unfilled until `humanize()` runs.
    pub fn fatal_kind(kind: DiagnosticKind<'src>) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Fatal, "");
        diag.kind = Some(std::sync::Arc::new(kind));
        diag
    }

    /// Create a warning diagnostic from a structured [`DiagnosticKind`].
    ///
    /// See [`error_kind`](Self::error_kind) for the lazy-humanization
    /// contract: derived fields stay unfilled until `humanize()` runs.
    pub fn warning_kind(kind: DiagnosticKind<'src>) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Warning, "");
        diag.kind = Some(std::sync::Arc::new(kind));
        diag
    }

    pub fn help_diag(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Help, message)
    }

    pub fn note(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Note, message)
    }

    pub fn info(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Info, message)
    }

    // ── Builder methods ────────────────────────────────────────────

    /// Set the primary source span for this diagnostic.
    ///
    /// Spans set here are preserved across [`humanize`](Self::humanize);
    /// they are not overwritten by spans derived from a `kind`.
    ///
    /// This replaces the whole span list, dropping any spans previously added
    /// by [`with_additional_span`](Self::with_additional_span); call
    /// `with_span` first when combining the two.
    pub fn with_span(mut self, span: Span) -> Self {
        self.spans = MultiSpan::new(span);
        self.spans_user_set = true;
        self.invalidate_stable_id();
        self
    }

    /// Add a secondary primary span (for multi-location diagnostics).
    ///
    /// Like [`with_span`](Self::with_span), this survives `humanize`.
    pub fn with_additional_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self.spans_user_set = true;
        self.invalidate_stable_id();
        self
    }

    /// Set the error code (e.g. `ErrCode::TypeMismatch` → `E030`).
    pub fn with_code(mut self, code: ErrCode) -> Self {
        self.code = Some(code);
        self.invalidate_stable_id();
        self
    }

    /// Convenience: set error code by string (backward compat).
    /// Prefer `with_code(ErrCode::new(...))` for the new API.
    ///
    /// Unlike `ErrCode::new`, this never panics on an unknown code: the code
    /// is validated against the table and, if absent, stored as-is so
    /// `title()` / `category()` / `explain()` fall back to their generic
    /// "unknown" values.  Safe for untrusted or dynamically-built strings.
    pub fn with_code_str(mut self, code: impl AsRef<str>) -> Self {
        let code = code.as_ref();
        self.code = Some(match ErrCode::try_new(code) {
            Ok(ec) => ec,
            Err(e) => ErrCode::unknown(e.0.into_owned()),
        });
        self.invalidate_stable_id();
        self
    }

    /// Add a help message.
    pub fn with_help(mut self, help: impl Into<Cow<'src, str>>) -> Self {
        self.help = Some(help.into());
        // C3: user help survives `humanize`; kind-derived help fills only
        // when the caller has not supplied its own.
        self.help_user_set = true;
        self
    }

    /// Add a single suggestion with the default (`Unspecified`) applicability
    /// and `ShowAlways` style.
    ///
    /// Classify via [`with_suggestion_style`](Self::with_suggestion_style)
    /// when the applicability is known (e.g. compiler-derived fixes).
    pub fn with_suggestion(mut self, suggestion: impl Into<Cow<'src, str>>) -> Self {
        self.suggestions.push(Suggestion {
            message: suggestion.into(),
            applicability: Applicability::default(),
            style: SuggestionStyle::ShowAlways,
            span: None,
            replacement: None,
        });
        self.suggestions_user_set = true;
        self
    }

    /// Add a suggestion with explicit applicability and style.
    pub fn with_suggestion_style(
        mut self,
        suggestion: impl Into<Cow<'src, str>>,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> Self {
        self.suggestions.push(Suggestion {
            message: suggestion.into(),
            applicability,
            style,
            span: None,
            replacement: None,
        });
        self.suggestions_user_set = true;
        self
    }

    /// Add a **structural** suggestion: a source span plus its replacement
    /// text.  The glyph renderer emits this as a two-line diff inside the
    /// source section, prefixed with a dim `√` (U+221A SQUARE ROOT,
    /// MachineApplicable) or `?` (MaybeIncorrect) applicability glyph on
    /// each diff line.  Single-line spans only: multi-line spans fall back
    /// to the text-only suggestion path.
    pub fn with_span_suggestion(
        mut self,
        span: Span,
        replacement: impl Into<Cow<'src, str>>,
        message: impl Into<Cow<'src, str>>,
        applicability: Applicability,
    ) -> Self {
        self.suggestions.push(Suggestion {
            message: message.into(),
            applicability,
            style: SuggestionStyle::ShowAlways,
            span: Some(span),
            replacement: Some(replacement.into()),
        });
        self.suggestions_user_set = true;
        self
    }

    /// Replace suggestions list.
    pub fn with_suggestions(mut self, suggestions: Vec<Suggestion<'src>>) -> Self {
        self.suggestions = suggestions.into();
        self.suggestions_user_set = true;
        self
    }

    /// Add a labeled sub-span for multi-span highlighting.
    pub fn with_label(mut self, span: Span, label: impl Into<Cow<'src, str>>) -> Self {
        self.labels.push(Label::new(span, label));
        self
    }

    /// Add a secondary (tilde-underline) label.
    pub fn with_secondary_label(mut self, span: Span, label: impl Into<Cow<'src, str>>) -> Self {
        self.labels.push(Label::secondary(span, label));
        self
    }

    /// Attach a related error to be displayed alongside this diagnostic
    /// in a single merged box (e.g. a type mismatch aggregated into a
    /// duplicate definition diagnostic).
    pub fn with_related(mut self, err: RelatedError<'src>) -> Self {
        self.related_errors.push(err);
        self
    }

    /// Attach a call chain (Zig-style `referenced by`).
    pub fn with_call_chain(mut self, chain: CallChain<'src>) -> Self {
        self.call_chain = Some(chain);
        self
    }

    /// Attach source text for `^`-underline rendering.
    ///
    /// The source is borrowed from the caller (e.g. the driver's source
    /// buffer), so no copy is made.
    pub fn with_source(mut self, source: &'src str) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the source file name for the `┌─ <file>:1:1` location header.
    pub fn with_file_name(mut self, file_name: impl Into<Cow<'src, str>>) -> Self {
        self.file_name = file_name.into();
        self.invalidate_stable_id();
        self
    }

    /// Set the structured diagnostic kind (Vale-style ADT).
    ///
    /// When present, the humanizer can derive `message`, `labels`, and other
    /// fields from the kind's structured data, producing more precise error
    /// messages than a flat string.
    pub fn with_kind(mut self, kind: DiagnosticKind<'src>) -> Self {
        self.kind = Some(std::sync::Arc::new(kind));
        // A new kind must be materialized by the next `humanize()`, and the
        // cached `stable_id` (hashed over `(code, span, message)`) is stale
        // until then — invalidate it exactly like every other mutator.
        self.humanized = false;
        self.invalidate_stable_id();
        self
    }

    /// Add a highlighted note child (styled text with type/name highlights).
    ///
    /// The plain-text form is derived lazily by
    /// [`Subdiag::plain_message`], so a note that is never rendered (filtered
    /// out by `max_errors` / `only_codes`, say) never allocates it.
    pub fn with_highlighted_note(mut self, styled: StyledString<'src>) -> Self {
        self.children.push(Subdiag {
            level: DiagnosticLevel::Note,
            message: None,
            styled_message: Some(styled),
            span: None,
            labels: Vec::new(),
        });
        self
    }

    /// Add a highlighted help child (styled text with type/name highlights).
    pub fn with_highlighted_help(mut self, styled: StyledString<'src>) -> Self {
        self.children.push(Subdiag {
            level: DiagnosticLevel::Help,
            message: None,
            styled_message: Some(styled),
            span: None,
            labels: Vec::new(),
        });
        self
    }

    /// Run the humanizer on `self.kind` (if present) to populate `message`,
    /// `labels`, `help`, and `suggestions` from the structured data.
    ///
    /// Idempotent for a fixed language: a `None` kind latches `humanized`
    /// and repeated calls short-circuit, so the emitter rendering the same
    /// batch twice pays nothing.  `with_kind` resets the latch so a
    /// re-kind'ed diagnostic is re-derived.
    ///
    /// The latch is keyed to the current [`Lang`] as well: a diagnostic
    /// humanized in one language re-derives `message`/`labels`/`help` in
    /// the new language when the process language changes between emits —
    /// the LSP re-translation contract documented on
    /// [`error_kind`](Self::error_kind).  The `kind` is retained (never
    /// dropped) as the structured source of truth that makes this possible;
    /// the one-`clone`-per-humanize cost replaces the old take-and-drain,
    /// which froze the rendered language forever and left `kind()` empty.
    ///
    /// Until this runs, `message` / `labels` / `spans` / `help` /
    /// `suggestions` of a `*_kind`-built diagnostic hold their
    /// constructor defaults.  The [`Emitter`] calls this at its render
    /// boundary; direct readers of those fields must call it themselves.
    ///
    /// Spans are an exception: once a builder sets them (`with_span`,
    /// `with_additional_span`, `augment`, `adorn_with_span`), `humanize`
    /// preserves them instead of deriving primary spans from the kind's
    /// labels — so attaching the kind first and the span second is the
    /// supported builder ordering.  `help`/`suggestions` are always
    /// re-derived from the kind.
    pub fn humanize(&mut self) {
        let lang = crate::diagnostics::i18n::Lang::current() as u8;
        if self.humanized {
            // No kind: message/labels were set manually — there is nothing
            // to re-resolve in another language.  Same language: the latch
            // holds and the call is free.
            if self.kind.is_none() || self.humanized_lang == lang {
                return;
            }
        }
        self.invalidate_stable_id();
        let kind_box = std::mem::take(&mut self.kind);
        let Some(kind_arc) = kind_box else {
            // No kind: message/labels were set manually.  Latch so later
            // calls skip the `kind` lookup entirely.
            self.humanized = true;
            self.humanized_lang = lang;
            return;
        };
        // Keep `kind` alive across the derive, and put it back even if
        // `humanize_into` panics: a retried render must not see `kind = None`
        // and emit an empty message.
        let kind_for_call = std::sync::Arc::clone(&kind_arc);
        let restore = RestoreKindOnDrop {
            slot: &mut self.kind,
            restore: Some(kind_arc),
        };
        // `kind` is retained: humanize a *clone* so `kind()` keeps serving
        // the structured data and a later `Lang::set` can re-derive the
        // rendered fields.  `humanize_into` still consumes its argument, so
        // the shared fields move into one output within that pass — exactly
        // one deep clone of the kind per humanize.
        //
        // A borrowing `humanize_into_ref(&self)` cannot replace this: the
        // returned `DiagMessage<'src>` / `Label<'src>` must borrow for the
        // kind's own `'src`, but a `&self` borrow only proves the shorter
        // lifetime of the `&self`, and the values are stored back into `self`
        // (an immutable borrow of `self.kind` cannot overlap the assignments
        // to `self.message`/`self.labels`).  Keeping the clone is therefore
        // structural, not laziness — and for `Cow::Borrowed` fields (the
        // overwhelmingly common kind) a clone is a handful of pointer copies.
        // Only kinds with large *owned* side fields (`Comptime.traceback`, a
        // long `reason`) pay a deep copy, once per language switch.
        let (message, labels, help, suggestions) = (*kind_for_call).clone().humanize_into();
        self.kind = Some(restore.disarm());
        // Static messages move straight into `message`: a borrowed static
        // stays borrowed (zero allocation), an owned one is moved — only
        // the Fluent-slug path needs the render buffer.
        self.message = match message {
            DiagMessage::Static(cow) => cow,
            slug => {
                let mut buf = String::new();
                slug.render_into_owned(&mut buf);
                Cow::Owned(buf)
            }
        };
        // Populate `spans` from the primary labels so the source-context
        // renderer knows which lines to display — but only when the spans
        // are still the constructor default.  Spans attached by a builder
        // (`with_span`, `with_additional_span`, `augment`,
        // `adorn_with_span`) were set deliberately and are preserved.
        if !self.spans_user_set {
            self.spans.clear();
            for lbl in &labels {
                if matches!(lbl.kind, AnnotationKind::Primary) {
                    self.spans.push(lbl.span);
                }
            }
        }
        // Labels: replace exactly the previous kind-derived block and
        // re-append user labels after the fresh block — kind-derived first,
        // then user-added (`with_label` / `with_secondary_label` /
        // `augment` only ever append), the order the builders establish.
        // Truncating just the derived block makes re-humanization on a
        // language change idempotent: derived labels refresh, user labels
        // neither duplicate nor vanish.  (A >255-label kind would saturate
        // the counter; kinds derive a handful of labels, so the bound is
        // unreachable in practice.  `min` below also tolerates external
        // mutation through `labels_mut()` that shrank the vector under our
        // bookkeeping.)
        let prev_derived = (self.kind_derived_labels as usize).min(self.labels.len());
        // Reuse both allocations instead of building fresh containers: move
        // the user-added tail aside, overwrite the (discarded) derived block
        // in place, then re-append the user labels.  `drain` leaves the
        // original buffer's capacity intact for the fresh derived block.
        let mut user: CovariantVec<'src, Label<'src>, 1> = CovariantVec::new();
        for lbl in self.labels.drain(prev_derived..) {
            user.push(lbl);
        }
        self.labels.truncate(0);
        for lbl in labels {
            self.labels.push(lbl);
        }
        debug_assert!(
            self.labels.len() <= u8::MAX as usize,
            "kind derives >255 labels; widen kind_derived_labels"
        );
        self.kind_derived_labels = self.labels.len().min(u8::MAX as usize) as u8;
        for lbl in user.drain(..) {
            self.labels.push(lbl);
        }
        // help/suggestions are re-derived from `kind` on every humanize: a
        // diagnostic can be re-kind'ed, and guarded rebuilding would keep
        // k1's derived notes on a diagnostic that no longer is k1.  Caller-
        // supplied help/suggestions are preserved, matching the spans
        // precedent (C3) — overwriting them silently lost user data.
        if !self.help_user_set {
            self.help = help;
        }
        if !self.suggestions_user_set {
            // Reuse the suggestions allocation across re-humanization
            // (language switches in an LSP re-derive every diagnostic).
            self.suggestions.clear();
            self.suggestions
                .extend(suggestions.into_iter().map(|s| Suggestion {
                    message: s,
                    // `humanize_into` yields plain strings with no
                    // per-suggestion classification, so these are *not* known
                    // to be mechanically applicable.  `Unspecified` is the
                    // crate's documented safe default: an auto-apply pass must
                    // not treat them as machine-applicable without review.
                    applicability: Applicability::Unspecified,
                    style: SuggestionStyle::ShowAlways,
                    span: None,
                    replacement: None,
                }));
        }
        self.humanized = true;
        self.humanized_lang = lang;
    }

    // ── Query methods ──────────────────────────────────────────────

    pub fn is_error(&self) -> bool {
        self.level.is_error()
    }

    pub fn is_warning(&self) -> bool {
        matches!(self.level, DiagnosticLevel::Warning)
    }

    /// Change the severity of this diagnostic.
    ///
    /// This is the single place through which severity changes should flow, so
    /// that any state derived from `level` (now or in the future) is kept
    /// consistent.  Used by [`DiagCtxt`](crate::diagnostics::DiagCtxt)'s
    /// downgrade helpers.
    pub fn set_level(&mut self, level: DiagnosticLevel) {
        self.level = level;
    }

    /// Get the main location string for display.
    pub fn location_string(&self) -> String {
        self.spans
            .first()
            .map(|s| format!("at {}", s))
            .unwrap_or_default()
    }

    /// Merge context from another diagnostic into `self`.
    ///
    /// If `other` has a span, source, labels, or a call chain and `self`
    /// does not yet have them, they are copied over.  This is useful for
    /// progressively enriching a diagnostic as it passes through compiler
    /// passes (the "context augmentation" pattern inspired by Austral).
    pub fn augment(&mut self, other: &Diagnostic<'src>) {
        if self.spans.first().is_none()
            && let Some(span) = other.spans.first()
        {
            self.spans = MultiSpan::new(span);
            self.spans_user_set = true;
            self.invalidate_stable_id();
        }
        if self.source.is_none() {
            self.source = other.source;
        }
        if self.call_chain.is_none() {
            self.call_chain.clone_from(&other.call_chain);
        }
        // Merge labels that aren't already present.  The straightforward
        // scan is O(self × other) in the label counts; the keyed set is
        // O(self + other).  The threshold guards *both* sides: a small
        // `self` merged with a large `other` would otherwise scan the
        // growing `to_add` on every iteration and degrade to O(|other|²).
        // Small merges on both sides keep the allocation-free scan (the
        // common case is one or two labels).
        // `to_add` collects first: the key set borrows `self.labels`, so it
        // must be dropped before the pushes.
        if !other.labels.is_empty() {
            let mut to_add: SmallVec<[&Label<'src>; 4]> = SmallVec::new();
            if self.labels.len() <= 8 && other.labels.len() <= 8 {
                for lbl in &other.labels {
                    // `to_add` is checked too: a duplicate *within* `other`
                    // must not be added twice (the scan sees earlier pushes).
                    if !self
                        .labels
                        .iter()
                        .chain(to_add.iter().copied())
                        .any(|l| l.span == lbl.span && l.message == lbl.message)
                    {
                        to_add.push(lbl);
                    }
                }
            } else {
                fn key<'a, 'src>(l: &'a Label<'src>) -> (FileId, u32, u32, &'a str) {
                    (l.span.file, l.span.start, l.span.end, &l.message)
                }
                let mut seen: std::collections::HashSet<(FileId, u32, u32, &str)> =
                    self.labels.iter().map(key).collect();
                for lbl in &other.labels {
                    if seen.insert(key(lbl)) {
                        to_add.push(lbl);
                    }
                }
            }
            for lbl in to_add {
                self.labels.push(lbl.clone());
            }
        }
        if let Some(other_sources) = other.foreign_sources.as_deref() {
            for (file, name, source) in other_sources.iter() {
                let self_sources = self.foreign_sources.get_or_insert_with(Box::default);
                if !self_sources.iter().any(|(f, _, _)| *f == *file) {
                    self_sources.push((*file, name.clone(), *source));
                }
            }
        }
    }
}

/// Wrap a function that may produce a diagnostic, augmenting any errors
/// it emits with the given span and source context.
///
/// This is the Ponent equivalent of Austral's `adorn_error_with_span`:
/// instead of catching exceptions, the closure receives a `&mut DiagCtxt`
/// and any diagnostics it pushes are augmented with `span` and `source`.
///
/// # Convention
///
/// This function **augments the last diagnostic** pushed to `ctxt` when the
/// closure returns `Err`.  It relies on the invariant that the diagnostic
/// produced by the failing closure is the most recent one in the context.
/// If the closure pushes multiple diagnostics, only the last one is
/// augmented — earlier ones are left untouched.
///
/// This is an intentional design choice: the "root cause" diagnostic is
/// typically the last one pushed, and augmenting it with the outer call
/// site's span/source is the most useful behaviour.  Callers that push
/// multiple diagnostics before failing should order them so that the
/// one needing augmentation is last.
///
/// # Guarantee
///
/// `EmissionGuaranteed` can only be obtained by calling `DiagCtxt::push()`.
/// Note, however, that `push` may return a *suppressed* guarantee without
/// recording anything — a warning pushed while `can_emit_warnings` is false.
/// Therefore a growing diagnostic count is *not* guaranteed on the `Err`
/// path; augmentation is applied only when the closure actually grew the
/// context (diagnostic count strictly larger than before the call), so a
/// pre-existing diagnostic is never silently re-spanned by an unrelated
/// failure.
///
/// # Example
/// ```ignore
/// adorn_with_span(ctxt, span, source, |ctxt| {
///     check_something(ctxt)?;
///     Ok(())
/// });
/// ```
pub fn adorn_with_span<'src, T>(
    ctxt: &mut DiagCtxt<'src>,
    span: Span,
    source: Option<&'src str>,
    f: impl FnOnce(&mut DiagCtxt<'src>) -> Result<T, EmissionGuaranteed>,
) -> Result<T, EmissionGuaranteed> {
    let before = ctxt.len();
    let result = f(ctxt);
    // If the closure failed, augment the last diagnostic with context.
    if result.is_err() {
        // The closure may have returned `Err` from a *suppressed* warning
        // push (which records nothing) — or from a check that failed without
        // pushing at all.  In both cases `last_mut()` would return an
        // *earlier, unrelated* diagnostic and corrupt its span/source, so
        // augment only when the closure actually grew the context.
        if ctxt.len() > before
            && let Some(diag) = ctxt.last_mut()
        {
            if diag.spans.first().is_none() {
                diag.spans = MultiSpan::new(span);
                diag.spans_user_set = true;
                diag.invalidate_stable_id();
            }
            if diag.source.is_none() {
                diag.source = source;
            }
        }
    }
    result
}

// ── ErrCode is now a string-based code — use `ErrCode::E030` ──
// The old `from_str` method with enum variants was removed in favour of
// the lookup table in `error_code.rs`.  Any string is accepted as a code.

// ── Display ───────────────────────────────────────────────────────

impl fmt::Display for Diagnostic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `Display` reads `self.message` directly, while every message
        // accessor enforces the humanize precondition.  A kind-built
        // diagnostic rendered before `humanize()` would otherwise silently
        // print an empty message; make that contract violation loud in debug
        // builds instead.
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "Display on a kind-built Diagnostic before humanize(): message is empty"
        );
        let code_str = self.code.as_ref().map(|c| c.code()).unwrap_or("?");
        let level_str = self.level.label().to_uppercase();

        // Inline span-writing — avoids one extra allocation per diagnostic
        // vs calling location_string().
        let span_part = self.spans.first().map(|s| format!("at {s}"));

        if let Some(ref span_str) = span_part {
            writeln!(
                f,
                "[{code} {level}] {msg} {span}",
                code = code_str,
                level = level_str,
                msg = self.message,
                span = span_str,
            )?;
        } else {
            writeln!(
                f,
                "[{code} {level}] {msg}",
                code = code_str,
                level = level_str,
                msg = self.message,
            )?;
        }

        // Labels
        for lbl in &self.labels {
            writeln!(
                f,
                "  {}--> {}: {}",
                lbl.underline_char(),
                lbl.span,
                lbl.message
            )?;
        }

        // Call chain
        if let Some(ref chain) = self.call_chain
            && !chain.is_empty()
        {
            write!(f, "{}", chain.render(false, self.source, &self.file_name))?;
        }

        // Help
        if let Some(help) = &self.help {
            writeln!(f, "help: {}", help)?;
        }

        // Suggestions
        for suggestion in &self.suggestions {
            writeln!(f, "suggestion: {}", suggestion)?;
        }

        Ok(())
    }
}

/// Render the `--explain` output for a given error code string.
/// When `use_color` is true, output includes syntax highlighting and ANSI
/// colors; otherwise the body is plain text with no color codes.
pub fn explain_error_code(code_str: &str, use_color: bool) -> String {
    explain_error_code_with_depth(
        code_str,
        use_color,
        term::detect_color_depth(),
        DEFAULT_MAX_WIDTH,
        glyph::DEFAULT_TAB_WIDTH,
    )
}

/// Like [`explain_error_code`] but with an explicit colour depth, wrap width
/// and tab width, so the caller can share a probed [`term::ColorDepth`] and a
/// configured `--width`/`--tab-width` across multiple calls.  (Before P5 this
/// hardcoded [`DEFAULT_MAX_WIDTH`], so `--width` narrowed diagnostics but not
/// `--explain` output.)
pub fn explain_error_code_with_depth(
    code_str: &str,
    use_color: bool,
    color_depth: term::ColorDepth,
    max_width: usize,
    tab_width: usize,
) -> String {
    use std::fmt::Write;

    // Use `try_new` to validate the code exists in the lookup table.
    let code = match ErrCode::try_new(code_str) {
        Ok(code) => code,
        Err(_) => {
            // `code_str` is untrusted (CLI input) and is echoed into terminal
            // output, so sanitize the copy shown while using the raw string
            // for suggestion matching.
            let error_label = DiagnosticLevel::Error.label();
            let unknown_msg = crate::diagnostics::i18n::fluent::unknown_error_code(
                sanitize_for_terminal(code_str),
            );
            return if use_color {
                format!(
                    "\x1b[1;31m{error_label}\x1b[0m {unknown_msg}\n\
                     {}",
                    suggest_code(code_str, true),
                )
            } else {
                // No `[E000]` placeholder: `E000` is asserted absent from the
                // registry, so printing it would invite users to look up (or
                // report) a code that cannot exist.
                format!(
                    "{error_label} {unknown_msg}\n{}",
                    suggest_code(code_str, false),
                )
            };
        }
    };
    // The explanation body dominates the output; pre-size from it so the
    // buffer is not regrown repeatedly while it is written.
    let explain = code.explain();
    let mut out = String::with_capacity(explain.len() + 128);

    // ── Header ──
    if use_color {
        let _ = writeln!(
            out,
            "\x1b[1m{code}:\x1b[0m \x1b[36m{title}\x1b[0m",
            code = code.code(),
            title = code.title(),
        );
    } else {
        let _ = writeln!(out, "{}: {}", code.code(), code.title());
    }
    let _ = writeln!(out);

    // ── Body: parse explanation text, highlight code blocks, wrap text ──
    let mut in_code_block = false;
    // Dark background for code blocks: 256-colour dark gray, or reverse on
    // 16-colour terminals.
    let code_bg = if color_depth.supports_256() {
        "\x1b[48;5;236m"
    } else {
        "\x1b[7m"
    };
    const RESET: &str = "\x1b[0m";

    for line in explain.lines() {
        if line.starts_with("  ") && !line.trim().is_empty() {
            // Code/example line — apply syntax highlighting + background
            if !in_code_block {
                let _ = writeln!(out);
                in_code_block = true;
            }
            let highlighted = crate::diagnostics::glyph::highlight_code(line, use_color);
            if use_color {
                let _ = writeln!(out, "  {code_bg}{highlighted}{RESET}");
            } else {
                let _ = writeln!(out, "  {highlighted}");
            }
        } else if line.trim().is_empty() {
            if in_code_block {
                in_code_block = false;
            }
            let _ = writeln!(out);
        } else {
            if in_code_block {
                in_code_block = false;
                let _ = writeln!(out);
            }
            // Regular text — wrap at the shared diagnostic width so the
            // explain text and rendered diagnostics agree on one width.
            for seg in crate::diagnostics::wrap::wrap_optimal(line, max_width, tab_width) {
                out.push_str(seg.as_ref());
                out.push('\n');
            }
        }
    }

    let explicit = crate::diagnostics::error_code::related_codes_of(code.code());
    let related: SmallVec<[&'static str; 4]> = if explicit.is_empty() {
        crate::diagnostics::error_code::category_siblings_of(code.code(), 4)
    } else {
        SmallVec::from_slice(explicit)
    };
    if !related.is_empty() {
        let _ = writeln!(out);
        if use_color {
            out.push_str("\x1b[1mSee also:\x1b[0m\n");
        } else {
            out.push_str("See also:\n");
        }
        for rel in &related {
            out.push_str("  ");
            out.push_str(rel);
            if let Some(entry) = crate::diagnostics::error_code::lookup(rel) {
                out.push_str("  ");
                out.push_str(entry.title);
            }
            out.push('\n');
        }
    }

    out
}

/// Length of the longest common prefix of `a` and `b` (in chars).
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Damerau-Levenshtein (OSA) edit distance between `a` and `b`.
///
/// Three-row rolling DP: O(min(|a|, |b|)) auxiliary space, O(|a|·|b|) time.
/// Counts an adjacent transposition (`lne`↔`len`, `E003`↔`E030`) as a single
/// edit, where plain Levenshtein scores 2 — the radius-1 `suggest_code` scan
/// otherwise silently drops the most common keyboard typo.  Error codes are
/// four characters, so this is effectively constant-time.
///
/// Every buffer (the two char slices and the three rows) lives inline on the
/// stack for the realistic sizes — codes are ≤ 5 chars — so a call allocates
/// nothing; a longer needle spills to the heap via `SmallVec` rather than
/// being rejected, keeping the function total.
fn osa_distance(a: &str, b: &str) -> usize {
    const INLINE: usize = 8;
    let mut a: SmallVec<[char; INLINE]> = a.chars().collect();
    let mut b: SmallVec<[char; INLINE]> = b.chars().collect();
    // Keep the shorter string on the `a` side so the DP rows stay small.
    if a.len() > b.len() {
        std::mem::swap(&mut a, &mut b);
    }
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return m;
    }
    let mut prev_prev: SmallVec<[usize; INLINE]> = smallvec![0; n + 1];
    let mut prev: SmallVec<[usize; INLINE]> = (0..=n).collect();
    let mut curr: SmallVec<[usize; INLINE]> = smallvec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[j - 1] == b[i - 1] { 0 } else { 1 };
            let mut d = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            if i >= 2 && j >= 2 && a[j - 1] == b[i - 2] && a[j - 2] == b[i - 1] {
                d = d.min(prev_prev[j - 2] + 1);
            }
            curr[j] = d;
        }
        std::mem::swap(&mut prev_prev, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// OSA distance restricted to the ≤ 1 question: `Some(d)` with `d ∈ {0, 1}`
/// when the Damerau-Levenshtein (OSA) distance of `a` and `b` is at most 1,
/// `None` when it is ≥ 2.
///
/// The full DP answers a Boolean question with `O(|a|·|b|)` work; for
/// `max_dist == 1` — every realistic code typo, since `suggest_code`'s radius
/// is `max(1, needle_len / 3)` and codes are ≤ 5 chars — the answer is a
/// 4-case check: equality (0), same length with exactly one substitution
/// (1), same length with one adjacent transposition (1), or length differing
/// by one with the shorter a subsequence-with-one-hole of the longer (1).
/// Comparison is over `char`s, matching [`osa_distance`]'s semantics for
/// non-ASCII needles.
///
/// `None` from this function does *not* pin the distance to ≥ 2 for needles
/// longer than 8 chars by itself — callers with `max_dist > 1` must fall
/// back to [`osa_distance`] (which `suggest_code` does).
fn osa_dist_le_1(a: &str, b: &str) -> Option<usize> {
    // Every table code is ≤ 5 chars, so an 8-char inline buffer covers all
    // equal-or-shorter sides without a heap allocation.
    const INLINE: usize = 8;
    let a: SmallVec<[char; INLINE]> = a.chars().collect();
    let b: SmallVec<[char; INLINE]> = b.chars().collect();
    if a == b {
        return Some(0);
    }
    let (s, l) = if a.len() <= b.len() { (&a, &b) } else { (&b, &a) };
    match l.len() - s.len() {
        0 => {
            // Same length: either one substitution or one adjacent
            // transposition of the two differing positions.
            let mut d0 = 0usize;
            let mut d1 = 0usize;
            let mut n = 0usize;
            for i in 0..s.len() {
                if s[i] != l[i] {
                    if n >= 2 {
                        return None;
                    }
                    if n == 0 {
                        d0 = i;
                    } else {
                        d1 = i;
                    }
                    n += 1;
                }
            }
            match n {
                1 => Some(1), // substitution
                2
                    if d1 == d0 + 1
                        && s[d0] == l[d1]
                        && s[d1] == l[d0] =>
                {
                    Some(1) // transposition
                }
                _ => None,
            }
        }
        1 => {
            // One insertion/deletion: the shared prefix, then the tail of
            // the longer string must equal the tail of the shorter.
            let mut i = 0;
            while i < s.len() && s[i] == l[i] {
                i += 1;
            }
            if l[i + 1..] == s[i..] {
                Some(1)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Summarise the known error codes as contiguous ranges, e.g.
/// `E001–E004, E007, W113`.  Derived from `CODE_TABLE` so the fallback help
/// text cannot drift out of sync as codes are added.
///
/// `CODE_TABLE` is a compile-time constant, so the rendered summary is
/// computed once and cached; recomputing it copies, sorts, and formats the
/// whole table on every unknown-code explanation otherwise.
fn known_code_ranges() -> &'static str {
    static RANGES: OnceLock<String> = OnceLock::new();
    RANGES.get_or_init(|| {
        let codes: Vec<&str> = crate::diagnostics::error_code::CODE_TABLE
            .iter()
            .map(|e| e.code)
            .collect();
        format_code_ranges(&codes)
    })
}

/// Summarise code strings as contiguous numeric ranges (see
/// [`known_code_ranges`]).  Split out so the ordering can be tested with
/// synthetic codes the current table does not contain.
fn format_code_ranges(codes: &[&str]) -> String {
    let mut codes: Vec<&str> = codes.to_vec();
    // Sort by (prefix, numeric value, raw string): within a prefix,
    // lexicographic order only matches numeric order while codes are
    // equal-width — a future 4-digit `E1000` would sort before `E999`
    // lexicographically.  `sort_unstable_by_key` avoids the Vec allocation
    // of `sort_by_cached_key`; the key is `Copy` and cheap to compute, and
    // the slice is small (typically 1–5 codes).
    codes.sort_unstable_by_key(|c| {
        (
            c.as_bytes().first().copied().unwrap_or(b'?'),
            c.get(1..)
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(u32::MAX),
            *c,
        )
    });
    let numeric = |c: &str| c.get(1..).and_then(|s| s.parse::<u32>().ok());
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < codes.len() {
        let start = codes[i];
        let prefix = start.get(..1).unwrap_or("?");
        let mut end = start;
        while i + 1 < codes.len() {
            let next = codes[i + 1];
            if next.get(..1).unwrap_or("?") != prefix {
                break;
            }
            match (numeric(end), numeric(next)) {
                (Some(a), Some(b)) if b == a + 1 => {
                    end = next;
                    i += 1;
                }
                _ => break,
            }
        }
        if start == end {
            parts.push(start.to_string());
        } else {
            parts.push(format!("{start}–{end}"));
        }
        i += 1;
    }
    parts.join(", ")
}

/// Generate a "did you mean?" suggestion for an unknown error code.
///
/// Uses OSA (Damerau-Levenshtein) edit distance against every entry in `CODE_TABLE`
/// and returns up to three closest candidates within a small radius
/// (scaled by input length).  Comparison is case-insensitive so a
/// lowercase typo like `e030` still matches `E030`.
///
/// Ranking, in order of priority:
///   1. smaller edit distance
///   2. longer common prefix with the input (so `E03` → `E030`, not `E003`)
///   3. lexicographically smaller code (stable, deterministic output)
/// Insert `(dist, prefix, code)` into a fixed-size top-3 by
/// `(dist asc, prefix desc, code asc)` — the ranking `suggest_code` pins.
///
/// Tie-break at the boundary: when `top` is already full, an entry that ties
/// all three incumbents is **dropped** ("first encountered wins at the
/// boundary") — it is not re-sorted into the full `(dist asc, prefix desc,
/// code asc)` order by evicting the worst incumbent.  Only an entry
/// *strictly better* than some incumbent takes its slot (`position` finds the
/// first strictly-better position; `pos == 3` means no incumbent is worse, so
/// the entry is discarded).
#[inline]
/// Insert `(dist, prefix, code)` into the fixed-size top-3, keeping the
/// `(dist asc, prefix desc, code asc)` order.  Tie-break at the boundary:
/// when the top is full (`pos == 3`), an entry that ties all three existing
/// ones is *dropped* — "first encountered wins at the boundary", not a full
/// re-sort of equal entries.  Only strictly-better entries evict.
fn insert_top3<'a>(
    top: &mut ArrayVec<(usize, usize, &'a str), 3>,
    dist: usize,
    prefix: usize,
    code: &'a str,
) {
    let entry = (dist, prefix, code);
    let pos = top
        .iter()
        .position(|t| {
            entry.0 < t.0
                || (entry.0 == t.0 && (entry.1 > t.1 || (entry.1 == t.1 && entry.2 < t.2)))
        })
        .unwrap_or(top.len());
    if pos < 3 {
        if top.is_full() {
            top.pop();
        }
        top.insert(pos, entry);
    }
}

/// Whether `a` (4 digit bytes) with exactly one deletion equals `b`
/// (3 digit bytes) — i.e. `b` is a subsequence of `a` obtained by deleting
/// one character.  One forward mismatch scan + one tail compare: no DP.
fn digits_match_with_one_deletion(a: &[u8], b: &[u8]) -> bool {
    debug_assert_eq!(a.len(), 4);
    debug_assert_eq!(b.len(), 3);
    let mut i = 0;
    while i < 3 && a[i] == b[i] {
        i += 1;
    }
    // Deleting `a[i]` splices the tail over the mismatch; when the loop
    // exhausts (first three digits equal), the deletion is at the end.
    a[i + 1..] == b[i..]
}

fn suggest_code(input: &str, use_color: bool) -> CompactString {
    // Normalise case so `e030` matches the canonical `E030`.
    let mut needle = CompactString::from(input);
    needle.as_mut_str().make_ascii_uppercase();
    let needle_len = needle.chars().count();
    // Longer inputs tolerate a slightly wider search radius; short ones
    // stay tight so we don't suggest unrelated codes.
    let max_dist = std::cmp::max(1, needle_len / 3);

    // Top-3 by (dist asc, prefix desc, code asc) via fixed-size insertion
    // sort — avoids the `Vec` + `O(n log n)` sort for a 3-element result.
    let nb = needle.as_bytes();
    // A needle shaped like the registry (`[EW]\d{3,4}`) takes the
    // direct-address neighbourhood enumeration; everything else keeps the
    // general table scan (which also carries the bucketed branch as the
    // differential-test reference for `suggest_top3_direct`).
    let bucketed = matches!(needle_len, 4 | 5)
        && matches!(nb[0], b'E' | b'W')
        && nb[1..].iter().all(u8::is_ascii_digit);
    let top = if bucketed {
        suggest_top3_direct(&needle)
    } else {
        suggest_top3_scan(&needle, max_dist, needle_len)
    };

    let candidates: ArrayVec<&str, 3> = top.iter().map(|(_, _, code)| *code).collect();

    if candidates.is_empty() {
        let help_label = DiagnosticLevel::Help.label();
        if use_color {
            format_compact!(
                "\x1b[1;34m{help_label}\x1b[0m: {valid_codes} — \n       {hint}",
                valid_codes = crate::diagnostics::i18n::fluent::help_valid_codes(known_code_ranges()),
                hint = crate::diagnostics::i18n::fluent::run_explain_hint_example(),
            )
        } else {
            format_compact!(
                "{help_label}: {valid_codes} — \n       {hint}",
                valid_codes = crate::diagnostics::i18n::fluent::help_valid_codes(known_code_ranges()),
                hint = crate::diagnostics::i18n::fluent::run_explain_hint_example(),
            )
        }
    } else {
        let help_label = DiagnosticLevel::Help.label();
        let did_you_mean =
            crate::diagnostics::i18n::fluent::did_you_mean(candidates.join("` or `"));
        let hint = crate::diagnostics::i18n::fluent::run_explain_hint_detail();
        if use_color {
            format_compact!("\x1b[1;34m{help_label}\x1b[0m: {did_you_mean}\n       {hint}")
        } else {
            format_compact!("{help_label}: {did_you_mean}\n       {hint}")
        }
    }
}

/// The general candidate scan: OSA distance against the (prefiltered) table,
/// plus the first-char bucketing shortcut for `[EW]\d{3,4}` needles.  In
/// production this only serves needles that do NOT match the registry shape
/// (short/long/non-EW/non-digit input); the shape-needle bucketed branch is
/// retained as the reference implementation that
/// `suggest_code_direct_matches_scan` differentially tests against
/// [`suggest_top3_direct`].
fn suggest_top3_scan(
    needle: &str,
    max_dist: usize,
    // #14: the caller already counts the needle's chars; recomputing it per
    // scan was a second full pass on every `--explain` typo.
    needle_len: usize,
) -> ArrayVec<(usize, usize, &'static str), 3> {
    // Top-3 by (dist asc, prefix desc, code asc) via fixed-size insertion
    // sort — avoids the `Vec` + `O(n log n)` sort for a 3-element result.
    let mut top: ArrayVec<(usize, usize, &str), 3> = ArrayVec::new();
    let needle_len = needle.chars().count();
    let nb = needle.as_bytes();

    // ── First-char bucketing ─────────────────────────────────────
    // The registry is validated to `^[EW]\d{3}$`, so a 4-char needle that
    // itself matches that shape can only match entries in one of two
    // buckets:
    //
    //   * same-prefix (`E###` vs `E###`): up to `max_dist` edits anywhere;
    //     run `osa_distance` on the whole code.
    //
    //   * cross-prefix (`E###` vs `W###`): a single prefix substitution
    //     consumes the *entire* `max_dist == 1` budget, so the 3-digit
    //     remainder must be an **exact** match — no OSA pass needed.
    //
    // No other prefixes exist, so the cross-prefix remainder check plus
    // the same-prefix OSA pass covers every match, halving the OSA calls.
    // `needle_len == 4` forces `max_dist == 1` (4/3 == 1); so does
    // `needle_len == 5` — but a 5-char needle can only match a 4-char code
    // at distance 1 by a *pure insertion* (the length delta already spends
    // the whole budget), so its same-prefix branch is an exact
    // delete-one-digit test and its cross-prefix branch can never match
    // (substitution + insertion = distance 2 > max_dist).  Extending the
    // bucket to 5 keeps an `E0308`-shaped typo from walking the whole table
    // through the full rolling DP.  Non-shape needles (shorter, longer,
    // non-EW prefix, non-digit body) fall through to the unrestricted scan
    // below.
    let bucketed = matches!(needle_len, 4 | 5)
        && matches!(nb[0], b'E' | b'W')
        && nb[1..].iter().all(u8::is_ascii_digit);
    if bucketed {
        let same_prefix = nb[0];
        let rest = &needle[1..];
        for e in crate::diagnostics::error_code::CODE_TABLE.iter() {
            let eb = e.code.as_bytes();
            if eb[0] != same_prefix {
                // Cross-prefix: for a 4-char needle a prefix substitution
                // consumes the whole `max_dist == 1` budget, so the digits
                // must match exactly; for a 5-char needle substitution +
                // insertion is distance 2 — never within budget.
                if needle_len == 4 && eb[0] == (if same_prefix == b'E' { b'W' } else { b'E' }) {
                    if &e.code[1..] != rest {
                        continue;
                    }
                    let prefix = common_prefix_len(&needle, e.code);
                    insert_top3(&mut top, 1, prefix, e.code);
                }
                continue;
            }
            // The table is validated to `^[EW]\d{3}$`, so every code is
            // exactly 4 bytes — `debug_assert_eq!` pins it.
            debug_assert_eq!(eb.len(), 4, "CODE_TABLE codes must be 4 bytes");
            let dist = if needle_len == 4 {
                // `max_dist == 1` here, so ask the ≤ 1 question directly: a
                // ~4-compare check instead of the full rolling DP.
                match osa_dist_le_1(&needle, e.code) {
                    Some(d) => d,
                    None => continue, // distance ≥ 2 > max_dist
                }
            } else {
                // `max_dist == 1` and the needle is one digit longer, so the
                // only surviving edit is a single insertion into the code:
                // the code's digits must be the needle's digits minus one.
                if !digits_match_with_one_deletion(rest.as_bytes(), &e.code.as_bytes()[1..]) {
                    continue;
                }
                1
            };
            if dist > max_dist {
                continue;
            }
            let prefix = common_prefix_len(&needle, e.code);
            insert_top3(&mut top, dist, prefix, e.code);
        }
    } else {
        for e in crate::diagnostics::error_code::CODE_TABLE
            .iter()
            // Length prefilter: each edit changes length by at most one, so
            // |Δlen| is a lower bound on the distance — an entry whose code
            // length differs by more than the radius can never match, and its
            // DP is skipped.  The needle uses a char count (not byte length) so
            // the bound stays exact for non-ASCII typos; every table code is
            // validated to `^[EW]\d{3}$`, so `e.code.len()` is already its char
            // count (constant 4) and skips the per-code char walk.
            .filter(|e| needle_len.abs_diff(e.code.len()) <= max_dist)
        {
            // `max_dist == 1` covers every needle up to 5 chars — the common
            // typo shape — so the ≤ 1 oracle answers it without the DP;
            // wider radii fall back to the full DP, which remains the
            // reference the differential tests pin.
            let dist = if max_dist == 1 {
                match osa_dist_le_1(&needle, e.code) {
                    Some(d) => d,
                    None => continue,
                }
            } else {
                osa_distance(&needle, e.code)
            };
            if dist > max_dist {
                continue;
            }
            let prefix = common_prefix_len(&needle, e.code);
            insert_top3(&mut top, dist, prefix, e.code);
        }
    }

    top
}

/// Direct-address neighbourhood enumeration for `[EW]\d{3,4}`-shaped needles,
/// replacing the 452-entry table walk of [`suggest_top3_scan`].
///
/// Every registry code at OSA distance ≤ 1 from a shaped needle is reachable
/// by exactly one of:
///
/// * identity (distance 0);
/// * one substitution — the prefix (`E↔W`, which forces an exact digit
///   remainder) or any single digit lane (10 digits each): 1 + 3·10 slots;
/// * one transposition of an adjacent digit pair: 3 slots;
/// * for a 5-digit needle, one insertion *into the code* — the length delta
///   already spends the whole radius, so the code's digits must be the
///   needle's digits minus one (the 4 deletion candidates).
///
/// That is ≤ 41 candidate slots per query (deduped), each resolved `O(1)`
/// through [`lookup`]'s `code_slot` + `CODE_INDEX` read, versus a full
/// 452-entry table walk whose per-entry test is itself several compares.
/// [`insert_top3`]'s comparator is a strict total order on distinct
/// `(dist, prefix, code)` entries, so enumeration order cannot change the
/// top-3 — pinned differentially against [`suggest_top3_scan`].
fn suggest_top3_direct(needle: &str) -> ArrayVec<(usize, usize, &'static str), 3> {
    // Only `[EW]\d{3}` / `[EW]\d{4}` needles have an enumerable
    // direct-address neighbourhood.  Anything else (typos like "E03",
    // longer strings, noise) has no enumeration shape — fall back to the
    // table scan so the entry point is total (the differential test feeds
    // exactly such inputs).  Production callers route shaped needles here
    // and unshaped ones straight to the scan, so this branch is cold.
    let nb = needle.as_bytes();
    let shaped = matches!(nb.len(), 4 | 5)
        && matches!(nb[0], b'E' | b'W')
        && nb[1..].iter().all(u8::is_ascii_digit);
    if !shaped {
        let max_dist = std::cmp::max(1, nb.len() / 3);
        return suggest_top3_scan(needle, max_dist, needle.chars().count());
    }
    /// Resolve one candidate slot (if a known code) into the top-3.
    fn consider(
        top: &mut ArrayVec<(usize, usize, &'static str), 3>,
        needle: &str,
        cand: &[u8; 4],
        dist: usize,
    ) {
        let code = std::str::from_utf8(cand).expect("E/W + ASCII digits is UTF-8");
        if let Some(e) = crate::diagnostics::error_code::lookup(code) {
            // Dedup by code: the 5-char deletion branch can enumerate the
            // same 4-char code at two `skip` positions when the needle's
            // digits repeat, and `insert_top3` does not dedup.  The same
            // code from the same needle always carries the same claimed
            // distance, so skipping a duplicate never changes the ranking.
            if top.iter().any(|t| t.2 == e.code) {
                return;
            }
            insert_top3(top, dist, common_prefix_len(needle, e.code), e.code);
        }
    }

    let nb = needle.as_bytes();
    let prefix = nb[0];
    let digits = &nb[1..];
    let mut top: ArrayVec<(usize, usize, &'static str), 3> = ArrayVec::new();

    if digits.len() == 3 {
        // Identity (distance 0) — the table lookup simply confirms presence.
        let mut cand = [prefix, digits[0], digits[1], digits[2]];
        consider(&mut top, needle, &cand, 0);
        // Prefix substitution (distance 1, whole budget).
        cand[0] = if prefix == b'E' { b'W' } else { b'E' };
        consider(&mut top, needle, &cand, 1);
        // Digit substitutions, one lane at a time (distance 1 each).  The
        // candidate is rebuilt from the needle every iteration so lanes set
        // by previous substitutions never leak into the next one.
        for p in 0..3 {
            for d in 0..10u32 {
                let dv = b'0' + d as u8;
                if dv == digits[p] {
                    continue; // identity, already considered
                }
                cand = [prefix, digits[0], digits[1], digits[2]];
                cand[p + 1] = dv;
                consider(&mut top, needle, &cand, 1);
            }
        }
        // Adjacent-digit transpositions (OSA distance 1 each).  Transposing
        // two *equal* digits reproduces the needle verbatim — a distance-0
        // duplicate, not a distance-1 neighbour — so skip that candidate;
        // `insert_top3` would otherwise record it at the claimed distance 1.
        for p in 0..2 {
            cand = [prefix, digits[0], digits[1], digits[2]];
            cand.swap(p + 1, p + 2);
            if cand == [prefix, digits[0], digits[1], digits[2]] {
                continue;
            }
            consider(&mut top, needle, &cand, 1);
        }
    } else {
        debug_assert_eq!(digits.len(), 4, "shaped needle is [EW]\\d{{3,4}}");
        // 5-char needle: only a single insertion into a 4-char code stays
        // within the radius — the code's digits are the needle's minus one.
        for skip in 0..4 {
            let mut cand = [prefix, 0u8, 0, 0];
            let mut k = 1;
            for (i, &d) in digits.iter().enumerate() {
                if i != skip {
                    cand[k] = d;
                    k += 1;
                }
            }
            consider(&mut top, needle, &cand, 1);
        }
    }
    top
}

/// List all available error codes with their titles, formatted for terminal output.
pub fn list_error_codes(use_color: bool) -> String {
    use std::fmt::Write;
    let (bold, cyan, dim, reset) = if use_color {
        ("\x1b[1m", "\x1b[36m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{bold}{}{reset}\n",
        crate::diagnostics::i18n::fluent::available_error_codes()
    );
    for (cat, entries) in crate::diagnostics::error_code::code_entries_by_category() {
        let _ = writeln!(out, " {cyan}── {} ──{reset}", cat.label(),);
        for entry in entries {
            let _ = writeln!(out, "  {bold}{}{reset}  {}", entry.code, entry.title,);
        }
    }
    let _ = writeln!(
        out,
        "\n{dim}{}{reset}",
        crate::diagnostics::i18n::fluent::run_explain_detail()
    );
    out
}

/// Convenience function to emit diagnostics to stderr.
/// Chooses plain or colored rendering based on the `use_color` flag.
///
/// Diagnostics are humanized at the emitter's render boundary, so their
/// i18n language resolves against the current `Lang` at emit time.
pub fn emit_diagnostics(diags: &mut [Diagnostic<'_>], use_color: bool) {
    let opts = DiagOpts {
        use_color,
        ..Default::default()
    };

    let renderer = HumanRenderer::with_opts(use_color, Some(&opts));
    let mut emitter = Emitter::new(renderer, sink::stderr_buffered(), opts);

    let _ = emitter.emit_all(diags);
}

/// Convenience function to emit a DiagCtxt without cloning its diagnostics.
///
/// Takes `&mut` because the emitter humanizes each diagnostic at its
/// render boundary (see [`Diagnostic`]'s lazy-humanization contract).
pub fn emit_collector(collector: &mut DiagCtxt<'_>, use_color: bool) {
    let opts = DiagOpts {
        use_color,
        ..Default::default()
    };
    let renderer = HumanRenderer::with_opts(use_color, Some(&opts));
    let mut emitter = Emitter::new(renderer, sink::stderr_buffered(), opts);
    let _ = emitter.emit_all_iter(collector.iter_mut());
}

/// Emit an iterator of diagnostics through a fresh sink.
///
/// Unlike [`emit_collector`], this avoids cloning each diagnostic into a
/// `Vec` first — it iterates by mutable reference, which the emitter
/// needs to humanize at its render boundary.
pub fn emit_diagnostics_iter<'r, 'src, I>(diags: I, use_color: bool) -> io::Result<()>
where
    'src: 'r,
    I: IntoIterator<Item = &'r mut Diagnostic<'src>>,
    I::IntoIter: DoubleEndedIterator,
{
    let opts = DiagOpts {
        use_color,
        ..Default::default()
    };
    let renderer = HumanRenderer::with_opts(use_color, Some(&opts));
    let mut emitter = Emitter::new(renderer, sink::stderr_buffered(), opts);
    emitter.emit_all_iter(diags)
}

/// Largest char-boundary offset `<= offset` (clamped to `source.len()`).
///
/// Spans are byte offsets and may — through arithmetic or malformed input —
/// land inside a multi-byte character.  Slicing `&source[..offset]` at such
/// an offset panics, so callers that index by byte must floor first.
pub(crate) fn floor_char_boundary(source: &str, offset: usize) -> usize {
    let mut o = std::cmp::min(offset, source.len());
    while o > 0 && !source.is_char_boundary(o) {
        o -= 1;
    }
    o
}

/// `char_col = b − cumulative_extra` for non-ASCII chars starting before `b`.
/// `sparse` is a sorted `[(start_byte, cumulative_extra)]` table; one
/// `partition_point` + one array read gives the answer in `O(log K)`.
/// `None` means the line is ASCII, so char col equals byte col (`b`).
/// Kept as the reference implementation for differential testing; the
/// production path uses the hinted variant below.
#[cfg_attr(not(test), allow(dead_code))]
fn char_col_from_sparse(sparse: Option<&[(u32, u32)]>, b: usize) -> usize {
    match sparse {
        Some(s) => {
            let k = s.partition_point(|&(p, _)| (p as usize) < b);
            let extra = if k > 0 { s[k - 1].1 as usize } else { 0 };
            b - extra
        }
        None => b,
    }
}

/// Hinted variant of [`char_col_from_sparse`]: `hint` caches the last
/// `partition_point` result, so a *sequence* of increasing `b` on the same
/// line — spans sorted by offset, the common render shape — advances a scalar
/// `O(1)` per step instead of binary-searching from scratch each time.  A
/// shrink (`b` below the hint's line position) falls back to `partition_point`
/// once and re-anchors the hint, so scattered access stays correct at
/// `O(log K)`.
///
/// The hint is an `AtomicUsize` (relaxed load/store) so the sparse tables can
/// live in a map shared across parallel render workers: a racy store can only
/// cost one re-anchor, never a wrong answer.
fn char_col_from_sparse_hinted(
    s: &[(u32, u32)],
    hint: &std::sync::atomic::AtomicUsize,
    b: usize,
) -> usize {
    use std::sync::atomic::Ordering;
    let mut k = hint.load(Ordering::Relaxed).min(s.len());
    // Advance while the next entry's start is still below `b`.
    while k < s.len() && (s[k].0 as usize) < b {
        k += 1;
    }
    // Overshot (`b` shrank since the hint was set): re-anchor via binary search.
    if k > 0 && (s[k - 1].0 as usize) >= b {
        k = s.partition_point(|&(p, _)| (p as usize) < b);
    }
    hint.store(k, Ordering::Relaxed);
    b - if k > 0 { s[k - 1].1 as usize } else { 0 }
}

/// Convert a byte-offset span into a 0-based (line, col) pair.
/// Single source of truth — previously duplicated verbatim in
/// `glyph::byte_to_linecol` and `label::byte_to_linecol`.
pub(crate) fn byte_to_linecol(source: &str, byte_offset: usize) -> label::SourcePos {
    let clamped = floor_char_boundary(source, byte_offset);
    let prefix = &source[..clamped];
    let line = memchr::memchr_iter(b'\n', prefix.as_bytes()).count();
    let start_of_line = memchr::memrchr(b'\n', prefix.as_bytes())
        .map(|i| i + 1)
        .unwrap_or(0);
    label::SourcePos {
        line,
        col: clamped - start_of_line,
    }
}

/// Whether `name` looks like a real on-disk file path rather than a
/// synthetic placeholder such as `<synthetic>` or an empty string.
pub(crate) fn is_real_file_name(name: &str) -> bool {
    !name.is_empty() && !name.starts_with('<')
}

/// [`sanitize_for_terminal`] variant for one-line renderers that flatten
/// breaks themselves: identical SWAR gates and replacement, but `\n`, `\r`
/// and `\t` survive as layout bytes — the caller consumes them as break runs
/// (the plain variant would turn them into U+FFFD first and defeat the
/// flattening).  The short renderer's `push_flattened` collapses all three
/// to a single space, so a tab never reaches the one-line output verbatim.
pub(crate) fn sanitize_for_terminal_flatten(s: &str) -> Cow<'_, str> {
    let bytes = s.as_bytes();
    let (has_high, dirty, has_c1) = scan_terminal_control_flags_flatten(bytes);
    if !has_high && !dirty {
        return Cow::Borrowed(s);
    }
    // Non-ASCII with no C0/C1/DEL *and* no bidi override/isolate: borrow.
    // Bidi formatting characters (U+202A–U+202E, U+2066–U+2069) are neither
    // controls nor visible, so the byte-level scan cannot see them — they are
    // probed separately (see [`sanitize_for_terminal`] for the spoofing
    // rationale) and must reach the owned path for removal.
    if has_high && !dirty && !has_c1 && !has_bidi_override_bytes(bytes) {
        return Cow::Borrowed(s);
    }
    // Bidi overrides/isolates are *removed* (not U+FFFD'd), matching
    // [`sanitize_filename`] and the source-row renderer: they carry no
    // content, only rendering direction, so replacement would just inject
    // noise while `\n`/`\r`/`\t` survive for the flattening caller.
    if has_bidi_override_bytes(bytes) {
        return Cow::Owned(s.chars()
            .filter(|c| !is_bidi_override(*c))
            .map(|c| {
                if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                    '\u{FFFD}'
                } else {
                    c
                }
            })
            .collect());
    }
    Cow::Owned(s.chars()
        .map(|c| {
            if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                '\u{FFFD}'
            } else {
                c
            }
        })
        .collect())
}

/// Replace control characters with U+FFFD so external text cannot inject
/// terminal escapes when echoed to the terminal.
///
/// Single source of truth — previously duplicated in `glyph`
/// (`filename_display`) and `error_code`.  Applied at every boundary that
/// writes untrusted text into ANSI output (OSC 8 hyperlink text, SGR colour
/// runs), where a raw `\x1b` or `\x07` would terminate the sequence early and
/// let the remaining bytes act as terminal commands.
///
/// Two fast paths keep the common cases allocation-free:
///
/// 1. **Printable ASCII** — one SWAR scan (see `ascii::is_printable_ascii`);
///    no control character is possible, so the input is borrowed.
/// 2. **Non-ASCII with no controls** — the dominant i18n case (CJK,
///    Cyrillic, accented Latin, …).  A `chars().any(is_control)` scan
///    short-circuits on the first control; if none is present, the input
///    is borrowed.  Only strings that actually contain a control character
///    pay the allocate-and-copy path.  `is_control` covers C0
///    (`0x00..=0x1F`), C1 (`0x80..=0x9F`) and DEL (`0x7F`).
///
/// Bidi overrides/isolates (U+202A–U+202E, U+2066–U+2069) are neither
/// controls nor visible, so they escape both fast paths' scans; a gated
/// `memchr(0xE2)` precheck catches them and they are **removed** in the
/// owned path (same convention as [`sanitize_filename`]) so a path like
/// `evil.cod\u{202E}exe` cannot spoof its extension in terminal output.
pub(crate) fn sanitize_for_terminal(s: &str) -> Cow<'_, str> {
    // One fused SWAR pass answers both gates.  The old sequence
    // (`is_printable_ascii` then `has_control_byte_level`) loaded the same
    // clean bytes twice — the printable-ASCII lane already rejects every
    // byte < 0x20, so the control probe re-scanned what the first pass had
    // just cleared.
    let bytes = s.as_bytes();
    let (has_high, dirty, has_c1) = scan_terminal_control_flags(bytes);
    if !has_high && !dirty {
        // Every byte in 0x20..=0x7E: printable ASCII, control-free.
        return Cow::Borrowed(s);
    }
    // The dominant i18n case (pure Cyrillic / CJK / accented Latin, no
    // controls): non-ASCII, but nothing to replace — borrow.  C1 is only
    // possible where high bytes exist, so the `memchr(0xC2)` probe is gated.
    // Bidi overrides/isolates (U+202A–U+202E, U+2066–U+2069) are invisible
    // and not controls, so they escape the SWAR scan — but they reorder
    // surrounding display text (`evil.cod\u{202E}exe` spoofs its extension),
    // so they are probed with the same gated `memchr(0xE2)` precheck and
    // force the owned path.  Owned: they are *removed*, matching
    // [`sanitize_filename`] — direction-only characters have no content, and
    // U+FFFD would just inject noise.
    let bidi = has_high && has_bidi_override_bytes(bytes);
    if has_high && !dirty && !has_c1 && !bidi {
        return Cow::Borrowed(s);
    }
    if bidi {
        Cow::Owned(
            s.chars()
                .filter(|c| !is_bidi_override(*c))
                .map(|c| if c.is_control() { '\u{FFFD}' } else { c })
                .collect(),
        )
    } else {
        Cow::Owned(
            s.chars()
                .map(|c| if c.is_control() { '\u{FFFD}' } else { c })
                .collect(),
        )
    }
}

/// One SWAR pass classifying a byte slice for the terminal sanitizers.
///
/// Returns `(has_high, dirty)`:
///
/// * `has_high` — any byte ≥ 0x80 (non-ASCII UTF-8 byte, lead or
///   continuation);
/// * `dirty` — any byte that a sanitizer must replace: byte < 0x20 (C0) or
///   `0x7F` (DEL).
///
/// Three masks per 8-byte chunk (`hasless(0x20)`, `has_zero(0x7F)`, the
/// high-bit OR) instead of the two full scans the previous two-gate
/// structure ran over clean input.  C1 (`0xC2 0x80..=0x9F`) is *not* folded
/// in — it needs a positional two-byte check, and it is only possible when
/// `has_high` is set, so callers probe it separately via
/// [`has_c1_sequence`].
fn scan_terminal_control_flags(s: &[u8]) -> (bool, bool, bool) {
    const HI: u64 = 0x8080_8080_8080_8080;
    const SP: u64 = 0x2020_2020_2020_2020;
    const DEL: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES: u64 = 0x0101_0101_0101_0101;

    let mut has_high = false;
    let mut dirty = false;
    let mut has_c1 = false;
    let mut i = 0;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        has_high |= x & HI != 0;
        has_c1 |= c1_in_chunk(s, i, x);
        // C0: any byte < 0x20 (existentially exact).
        dirty |= x.wrapping_sub(SP) & !x & HI != 0;
        // DEL: any byte == 0x7F (exact — any flag implies a real lane).
        let q = x ^ DEL;
        dirty |= q.wrapping_sub(ONES) & !q & HI != 0;
        i += 8;
    }
    for j in i..s.len() {
        let b = s[j];
        has_high |= b >= 0x80;
        dirty |= b < 0x20 || b == 0x7F;
        has_c1 |= b == 0xC2 && s.get(j + 1).is_some_and(|b| (0x80..=0x9F).contains(b));
    }
    (has_high, dirty, has_c1)
}

/// Whether one 8-byte chunk contains a C1 sequence start (`0xC2` followed by
/// `0x80..=0x9F`).  The follow byte may live in the next chunk, so `s` and
/// `base` are used for the positional check rather than the chunk alone.
#[inline]
fn c1_in_chunk(s: &[u8], base: usize, x: u64) -> bool {
    const HI: u64 = 0x8080_8080_8080_8080;
    const ONES: u64 = 0x0101_0101_0101_0101;
    const C2: u64 = 0x0202_0202_0202_0202;
    let q = x ^ C2;
    let mut mask = q.wrapping_sub(ONES) & !q & HI;
    while mask != 0 {
        let lane = (mask.trailing_zeros() / 8) as usize;
        let k = base + lane;
        if s.get(k + 1).is_some_and(|b| (0x80..=0x9F).contains(b)) {
            return true;
        }
        mask &= mask - 1;
    }
    false
}

/// Whether `s` contains a C1 control sequence (`0xC2` followed by
/// `0x80..=0x9F`) — the UTF-8 encoding of Unicode category `Cc` above DEL.
/// Only worth calling when the slice is known to contain high bytes.
fn has_c1_sequence(s: &[u8]) -> bool {
    let mut j = 0;
    while let Some(off) = memchr::memchr(0xC2, &s[j..]) {
        let k = j + off;
        if k + 1 < s.len() && (0x80..=0x9F).contains(&s[k + 1]) {
            return true;
        }
        j = k + 1;
    }
    false
}

/// One fused SWAR pass classifying a byte slice for
/// [`sanitize_for_terminal_multiline`]: returns `(has_high, dirty)` where
/// `has_high` is any byte ≥ 0x80 and `dirty` is any byte the multiline
/// sanitizer must replace — a C0 byte other than `\t`/`\n`, or DEL.  The
/// `\t`/`\n` exclusion keeps the old dirty-chunk discipline: the `< 0x20`
/// mask is only existentially exact (it cannot be masked per-lane), so a
/// chunk that trips it gets its *flagged lanes* resolved by exact byte
/// checks (`trailing_zeros`/8 → offset) — never a full 8-byte rescan, so a
/// multiline message whose every chunk contains `\t`/`\n` stays near full
/// SWAR speed.  C1 is probed separately via [`has_c1_sequence`], only when
/// `has_high`.
pub(crate) fn scan_terminal_control_flags_multiline(s: &[u8]) -> (bool, bool, bool) {
    const HI: u64 = 0x8080_8080_8080_8080;
    const SP: u64 = 0x2020_2020_2020_2020;
    const DEL: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES: u64 = 0x0101_0101_0101_0101;
    const TAB: u64 = 0x0909_0909_0909_0909;
    const NL: u64 = 0x0A0A_0A0A_0A0A_0A0A;

    let mut has_high = false;
    let mut dirty = false;
    let mut has_c1 = false;
    let mut i = 0;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        has_high |= x & HI != 0;
        has_c1 |= c1_in_chunk(s, i, x);
        // DEL: exact — any flag implies a genuine 0x7F lane.
        let q = x ^ DEL;
        if q.wrapping_sub(ONES) & !q & HI != 0 {
            dirty = true;
        }
        // C0: exact per-lane `< 0x20` mask (the `| HI` before the subtraction
        // prevents an inter-lane borrow), with `\t`/`\n` masked back out.
        // Unlike the inexact `hasless` idiom this needs no flagged-lane
        // resolution, so tab/newline-dense text stays O(chunks), not O(bytes).
        // High-byte lanes are excluded explicitly: this scanner cannot
        // early-return on `x & HI` (unlike `ascii.rs`), and the raw form also
        // flags bytes in `0x80..=0x9F`.
        let lt20 = !((x | HI).wrapping_sub(SP)) & HI & !(x & HI);
        if lt20 != 0 {
            let tab = !(((x ^ TAB) | HI).wrapping_sub(ONES)) & HI;
            let nl = !(((x ^ NL) | HI).wrapping_sub(ONES)) & HI;
            if lt20 & !tab & !nl != 0 {
                dirty = true;
            }
        }
        i += 8;
    }
    for j in i..s.len() {
        let b = s[j];
        has_high |= b >= 0x80;
        dirty |= (b < 0x20 && b != b'\t' && b != b'\n') || b == 0x7F;
        has_c1 |= b == 0xC2 && s.get(j + 1).is_some_and(|b| (0x80..=0x9F).contains(b));
    }
    (has_high, dirty, has_c1)
}

/// [`scan_terminal_control_flags_multiline`] variant for
/// [`sanitize_for_terminal_flatten`]: `\t`, `\n` *and* `\r` are excluded from
/// the dirty set, because those are exactly the layout bytes the flatten path
/// keeps or consumes (`push_flattened`).  A multi-line message with no other
/// controls therefore stays `Cow::Borrowed`, and flagged-lane resolution (not
/// whole-chunk rescans) keeps tab/newline/CR-dense text at SWAR speed.
fn scan_terminal_control_flags_flatten(s: &[u8]) -> (bool, bool, bool) {
    const HI: u64 = 0x8080_8080_8080_8080;
    const SP: u64 = 0x2020_2020_2020_2020;
    const DEL: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES: u64 = 0x0101_0101_0101_0101;
    const TAB: u64 = 0x0909_0909_0909_0909;
    const NL: u64 = 0x0A0A_0A0A_0A0A_0A0A;
    const CR: u64 = 0x0D0D_0D0D_0D0D_0D0D;

    let mut has_high = false;
    let mut dirty = false;
    let mut has_c1 = false;
    let mut i = 0;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        has_high |= x & HI != 0;
        has_c1 |= c1_in_chunk(s, i, x);
        let q = x ^ DEL;
        if q.wrapping_sub(ONES) & !q & HI != 0 {
            dirty = true;
        }
        // Exact C0 mask, minus high-byte lanes (see the multiline scanner),
        // then minus the layout bytes `push_flattened` consumes.
        let lt20 = !((x | HI).wrapping_sub(SP)) & HI & !(x & HI);
        if lt20 != 0 {
            let tab = !(((x ^ TAB) | HI).wrapping_sub(ONES)) & HI;
            let nl = !(((x ^ NL) | HI).wrapping_sub(ONES)) & HI;
            let cr = !(((x ^ CR) | HI).wrapping_sub(ONES)) & HI;
            if lt20 & !tab & !nl & !cr != 0 {
                dirty = true;
            }
        }
        i += 8;
    }
    for j in i..s.len() {
        let b = s[j];
        has_high |= b >= 0x80;
        dirty |= (b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r') || b == 0x7F;
        has_c1 |= b == 0xC2 && s.get(j + 1).is_some_and(|b| (0x80..=0x9F).contains(b));
    }
    (has_high, dirty, has_c1)
}

/// Unicode bidirectional formatting characters: the U+202A–U+202E
/// overrides and the U+2066–U+2069 isolates.  Invisible, but they reorder
/// the surrounding display text — a path like `evil.cod\u{202E}exe` can
/// render with a spoofed extension.
pub(crate) fn is_bidi_override(c: char) -> bool {
    ('\u{202A}'..='\u{202E}').contains(&c) || ('\u{2066}'..='\u{2069}').contains(&c)
}

/// Byte-level precheck for [`is_bidi_override`]: every bidi override/isolate
/// is a three-byte UTF-8 sequence starting with `0xE2` (`E2 80 AA..AE` for the
/// overrides, `E2 81 A6..A9` for the isolates).  Returns `false` for any
/// string that cannot contain one, letting callers skip the char-by-char
/// scan entirely.
pub(crate) fn has_bidi_override_bytes(s: &[u8]) -> bool {
    let mut i = 0;
    while let Some(off) = memchr::memchr(0xE2, &s[i..]) {
        let j = i + off;
        if j + 2 < s.len() {
            let b1 = s[j + 1];
            let b2 = s[j + 2];
            let is_override = (b1 == 0x80 && (0xAA..=0xAE).contains(&b2))
                || (b1 == 0x81 && (0xA6..=0xA9).contains(&b2));
            if is_override {
                return true;
            }
        }
        i = j + 1;
    }
    false
}

/// Sanitize a filename for terminal display: [`sanitize_for_terminal`],
/// plus bidirectional formatting characters removed outright.  Unlike
/// control characters they carry no content — only rendering direction —
/// so replacement with U+FFFD would just inject noise into the path;
/// removal kills the spoofing.
pub(crate) fn sanitize_filename(s: &str) -> Cow<'_, str> {
    // `sanitize_for_terminal` already *removes* bidirectional
    // overrides/isolates whenever `has_bidi_override_bytes` is true (its
    // `bidi` branch), and control characters were replaced on the same pass.
    // The old second `has_bidi_override_bytes` filter here was therefore
    // unreachable and cost a full `memchr(0xE2)` scan per filename render.
    // The composed contract is pinned by
    // `sanitize_filename_removes_bidi_overrides` in the tests.
    sanitize_for_terminal(s)
}

/// Multi-line variant of [`sanitize_for_terminal`] for text whose `\n`/`\t`
/// are layout intent (user messages, help, notes) rather than raw terminal
/// bytes: keeps them verbatim and replaces every other control character —
/// CR and the C1 range included — with U+FFFD.  Same fast paths as
/// [`sanitize_for_terminal`]: pure printable ASCII (plus `\n`/`\t`) is
/// borrowed, and so is any non-ASCII string whose only control characters
/// are `\n`/`\t` — the common i18n case, which the naive `collect` would
/// otherwise allocate and copy on every render.
pub(crate) fn sanitize_for_terminal_multiline(s: &str) -> Cow<'_, str> {
    // One fused SWAR pass answers both gates (see
    // [`scan_terminal_control_flags_multiline`]): the printable-ASCII probe
    // already rejects every byte < 0x20, so the control probe re-scanned the
    // same clean bytes a second time.
    let bytes = s.as_bytes();
    let (has_high, dirty, has_c1) = scan_terminal_control_flags_multiline(bytes);
    if !has_high && !dirty {
        // Printable ASCII plus `\t`/`\n` layout controls: borrow verbatim.
        return Cow::Borrowed(s);
    }
    // Byte-level gate: `\t`/`\n` preserved, every other C0/DEL/C1 rejected.
    // Pure i18n text without controls exits here after a bounded byte scan,
    // skipping the per-char `is_control` iteration entirely.  C1 is only
    // possible where high bytes exist, so the probe is gated.  As in
    // [`sanitize_for_terminal`], invisible bidi overrides/isolates escape the
    // control scan and force the owned path, where they are removed (same
    // convention as [`sanitize_filename`]).
    let bidi = has_high && has_bidi_override_bytes(bytes);
    if has_high && !dirty && !has_c1 && !bidi {
        return Cow::Borrowed(s);
    }
    let needs_replace = |c: char| c.is_control() && c != '\n' && c != '\t';
    if bidi {
        Cow::Owned(
            s.chars()
                .filter(|c| !is_bidi_override(*c))
                .map(|c| if needs_replace(c) { '\u{FFFD}' } else { c })
                .collect(),
        )
    } else {
        Cow::Owned(
            s.chars()
                .map(|c| if needs_replace(c) { '\u{FFFD}' } else { c })
                .collect(),
        )
    }
}

/// Precomputed line-start table for O(log n) byte-offset → line lookup.
///
/// The render path converts hundreds of spans per diagnostic; rescanning
/// the file from the start on every conversion (as `byte_to_linecol`
/// does) makes rendering O(spans × file_size). Build once per render and
/// share it across all conversions instead. `col` is a byte column within
/// the line, exactly like `byte_to_linecol`; display-cell conversion
/// happens later in the glyph renderer.
#[derive(Clone, Debug)]
pub(crate) struct LineIndex<'a> {
    /// Byte offset of the first byte of every line; always starts with 0.
    /// A trailing `'\n'` yields a final phantom entry equal to `total`.
    ///
    /// `Arc`-shared so a cached index can be handed to a later render by an
    /// O(1) clone instead of rebuilding or copying the table (see
    /// [`RenderCtx::line_index`](crate::diagnostics::render::RenderCtx::line_index)).
    starts: Arc<[u32]>,
    source: &'a str,
    /// Whether `source` is pure ASCII.  When true, byte columns equal
    /// character columns, so [`LineIndex::linecol_display`] skips the
    /// O(byte_col) char-count walk — the only remaining per-span linear
    /// cost in the JSON/HTML/short renderers on the ASCII sources that
    /// dominate real compiler output.  `is_ascii()` is a SIMD stdlib scan
    /// run once per source per batch (in [`LineIndex::new`]); the cached
    /// path carries the flag through [`LineIndex::from_shared`] so a cache
    /// hit never rescans.
    ascii: bool,
    /// Lazily-populated sparse non-ASCII char-extra table per line, keyed by
    /// line index.  Each entry is
    /// `Option<(CharPrefixTable, AtomicUsize)>`:
    /// `None` marks an ASCII line (char col == byte col, no table needed),
    /// `Some` holds one `(start_byte, cumulative_extra)` pair per non-ASCII
    /// char, where `cumulative_extra` is the running sum of `len_utf8 − 1`.
    /// Then `char_col(b) = b − cumulative_extra` for all non-ASCII chars whose
    /// `start_byte < b`, found by one `partition_point` + one array read.
    ///
    /// Only built on the non-ASCII path of [`linecol_display`](Self::linecol_display):
    /// without it, M spans landing on one long non-ASCII line each re-walk the
    /// line prefix (`O(M · byte_col)`); the table is built once per line in
    /// `O(K)` writes (K = non-ASCII char count) and reused by every later span
    /// on that line.  Memory is `O(K)` (8 bytes per non-ASCII char) vs the
    /// old dense prefix's `O(line.len())` (4 bytes per byte) — a single CJK
    /// char in a 1 MB line uses 8 bytes instead of 4 MB.  The map is
    /// `Arc`-shared across every index built from the same cached source
    /// (threaded through [`from_shared`](Self::from_shared) and the
    /// `RenderCtx` LRU), so a batch of M diagnostics over one long non-ASCII
    /// line pays one build, not M.  Behind a `RwLock` so the parallel render
    /// path can share one map across workers; storing `None` for ASCII lines
    /// (instead of an
    /// empty table) avoids any allocation per ASCII line in a mixed
    /// CJK/ASCII source — the common case for real code with comments.  The
    /// per-line table itself is an inline `SmallVec<[(u32, u32); 8]>`, not an
    /// inner `Arc`: the map is shared by reference, so the inner refcount was
    /// always 1 and only bought a heap allocation per non-ASCII line.
    ///
    /// CAVEAT: the `AtomicUsize` hint is shared through the map.  A racy
    /// `store` can only make a later query re-anchor via `partition_point`
    /// (the hint is a heuristic, never the answer), so the shared map is
    /// `Send + Sync` and one `LineIndex`/`RenderCtx` may hand its
    /// `char_prefixes` to parallel workers — see
    /// [`RenderCtx::preload_line_index`](crate::diagnostics::render::RenderCtx::preload_line_index).
    char_prefixes: Arc<crate::diagnostics::render::CharPrefixMap>,
    /// Process-wide monotonic id, assigned in [`LineIndex::new`] and carried
    /// through [`from_shared`](Self::from_shared).  Paired with the
    /// `(index_id, source_id, len)` memo key in [`LAST_CHAR_COL`] it makes
    /// the thread-local memo's ABA hazard (a later batch's index and source
    /// recycled at the same heap addresses) impossible, not just
    /// "unreachable in practice": a fresh `new()` always takes a generation
    /// no prior index ever held.
    generation: u64,
}

/// Source of [`LineIndex`]'s generation ids.  One relaxed `fetch_add` per
/// index construction — `LineIndex::new` runs once per source per batch, so
/// this is never on the span-conversion hot path.
static LINE_INDEX_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Fused newline-start collection and ASCII predicate in one `u64`-chunk
/// pass, replacing the two `memchr_iter` scans plus `is_ascii` the old
/// `LineIndex::new` ran.  Each 8-byte lane is tested for any high bit
/// (non-ASCII) and any `0x0A` (newline) via the standard `has_zero` SWAR
/// primitive (`(x ^ NL).wrapping_sub(ONES) & !(x ^ NL) & HI`).  The `Vec`
/// starts at `with_capacity(16)` and grows amortised — the previous
/// exact-capacity count pass is dropped because the combined scan already
/// sees every newline, so the win is one fewer full pass over the source
/// (the `is_ascii` SIMD scan), not a capacity saving.  For a ≥ 1 MB source
/// this is measurable; for small files the difference is noise.
///
/// Newline starts are `u32`: a diagnostic source over 4 GiB is not a real
/// input, and the halved table halves `line_of`'s binary-search cache
/// footprint (the table is read on every span conversion).  The `debug_assert`
/// guards the cast; sources are ≤ 4 GiB in every supported configuration.
fn scan_line_starts_and_ascii(source: &str) -> (Vec<u32>, bool) {
    const HI: u64 = 0x8080_8080_8080_8080;
    const NL: u64 = 0x0A0A_0A0A_0A0A_0A0A;
    const ONES: u64 = 0x0101_0101_0101_0101;

    let bytes = source.as_bytes();
    debug_assert!(
        bytes.len() <= u32::MAX as usize,
        "scan_line_starts_and_ascii: source exceeds u32 offsets ({} bytes)",
        bytes.len()
    );
    // Pre-size for large sources so the 16→2× growth chain doesn't pay ~16
    // reallocations (and megabytes of copies) on a 50 MB / 1M-line file.  A
    // heuristic average line length (~32 bytes) avoids the extra full `memchr`
    // counting pass; `Vec` growth absorbs the residual.  The 1 MB gate keeps
    // typical small sources from over-reserving.
    const LARGE_SOURCE_THRESHOLD: usize = 1 << 20;
    let capacity = if bytes.len() >= LARGE_SOURCE_THRESHOLD {
        bytes.len() / 32 + 16
    } else {
        16
    };
    let mut starts = Vec::with_capacity(capacity);
    starts.push(0);
    let mut ascii = true;
    let mut i = 0;
    while i + 8 <= bytes.len() {
        let x = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        ascii &= (x & HI) == 0;
        let q = x ^ NL;
        // Exact newline mask.  `(q | HI)` is 0x80 at a newline and also at
        // `x == 0x8A` (q == 0x80), a valid UTF-8 continuation byte (`Ċ`, `Ŋ`,
        // `ꀀ`…).  `!q` has bit 7 set only for q < 0x80, so it keeps only the
        // true newline lane while preserving the no-borrow property (`| HI`
        // makes every lane ≥ 0x80 before the subtraction).
        let mut mask = !((q | HI).wrapping_sub(ONES)) & !q & HI;
        while mask != 0 {
            let off = i + (mask.trailing_zeros() as usize / 8);
            starts.push((off + 1) as u32);
            mask &= mask - 1;
        }
        i += 8;
    }
    while i < bytes.len() {
        let b = bytes[i];
        ascii &= b < 0x80;
        if b == b'\n' {
            starts.push((i + 1) as u32);
        }
        i += 1;
    }
    (starts, ascii)
}

/// Soft cap on `LineIndex::char_prefixes` growth: the map gains one entry per
/// queried non-ASCII line and never evicts, so a long-lived process rendering
/// a very large CJK-heavy source would creep toward ~10 MB per cached index.
/// Past the cap the map clears wholesale — a query then rebuilds one line's
/// sparse table, the same cost as the first query for that line, so the
/// regression is bounded and only reaches files that genuinely have thousands
/// of distinct non-ASCII lines.
const CHAR_PREFIXES_CAP: usize = 4 * 1024;

/// Per-thread memo of the last [`LineIndex::char_col`] lookup, so a run of
/// span conversions on the same line of the same index skips the
/// `char_prefixes` `RwLock` read entirely (the lock is uncontended in the
/// per-worker render shape, but it is still an atomic acquire per span on
/// non-ASCII sources).
///
/// The key is `(index identity, source identity, line)`: index identity is
/// the `char_prefixes` `Arc`'s allocation address, source identity is the
/// indexed `str`'s address and length.  **Invariant:** the indexed source
/// text is immutable for the whole render batch (each `RenderCtx` carries one
/// immutable source), so a cached per-line prefix table is a pure function of
/// the line and can never go stale within a batch.  Both address components
/// guard against cross-batch ABA: after one index is dropped, a later index
/// for *different* content could have its `Arc` (or its source buffer)
/// allocated at the recycled address, and a single-address key would then
/// serve a stale table for another source.  Requiring both addresses *and*
/// the length to match makes that collision unreachable in practice for a
/// long-lived process (two independent allocations must both be recycled at
/// the same addresses between batches); `char_col` recomputes from the
/// shared map whenever the memo misses.  Thread-local is per render worker,
/// so the parallel renderer needs no extra synchronization.
thread_local! {
    static LAST_CHAR_COL: std::cell::RefCell<
        Option<(
            usize,     // index_id: `char_prefixes` Arc address
            usize,     // source_id: source buffer address
            usize,     // source length
            u64,       // LineIndex generation (rules out ABA reuse)
            usize,     // line index
            Option<std::sync::Arc<[(u32, u32)]>>,
            usize,     // hinted partition_point position
        )>,
    > = const { std::cell::RefCell::new(None) };
}

fn cap_char_prefixes(
    cache: &mut FxHashMap<
        usize,
        Option<(std::sync::Arc<[(u32, u32)]>, std::sync::atomic::AtomicUsize)>,
    >,
) {
    if cache.len() >= CHAR_PREFIXES_CAP {
        cache.clear();
    }
}

impl<'a> LineIndex<'a> {
    pub(crate) fn new(source: &'a str) -> Self {
        let (starts, ascii) = scan_line_starts_and_ascii(source);
        Self {
            starts: Arc::from(starts),
            source,
            ascii,
            char_prefixes: Arc::new(std::sync::RwLock::new(FxHashMap::default())),
            generation: LINE_INDEX_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Build an index for `source` from a table already computed for that same
    /// source (shared by [`RenderCtx::line_index`](crate::diagnostics::render::RenderCtx::line_index)).
    /// The caller guarantees `starts` describes `source`'s newlines,
    /// `ascii` equals `source.is_ascii()`, and `char_prefixes` was created for
    /// the same source — passing a stale flag makes
    /// [`linecol_display`](Self::linecol_display) produce wrong columns on
    /// non-ASCII sources, so cache insertions always source the flag and the
    /// prefix map from [`ascii`](Self::ascii) and
    /// [`char_prefixes_arc`](Self::char_prefixes_arc) of a freshly-built index.
    pub(crate) fn from_shared(
        source: &'a str,
        starts: Arc<[u32]>,
        ascii: bool,
        char_prefixes: Arc<crate::diagnostics::render::CharPrefixMap>,
        generation: u64,
    ) -> Self {
        debug_assert_eq!(
            ascii,
            source.is_ascii(),
            "from_shared: ascii flag does not match source"
        );
        debug_assert_eq!(
            starts.first().copied(),
            Some(0),
            "from_shared: starts[0] must be 0"
        );
        Self {
            starts,
            source,
            ascii,
            char_prefixes,
            generation,
        }
    }

    /// The index's generation id — the [`LAST_CHAR_COL`] memo-key component
    /// that rules out cross-batch address-reuse collisions.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// The shared line-start table, for caching by an O(1) `Arc` clone.
    pub(crate) fn starts_arc(&self) -> &Arc<[u32]> {
        &self.starts
    }

    /// The shared char-prefix cache, for caching by an O(1) `Arc` clone so a
    /// batch of diagnostics over one non-ASCII source reuses the per-line
    /// prefixes instead of rebuilding them per render.  The cache is
    /// `Send + Sync` (the hint is an `AtomicUsize`, the map behind a
    /// `RwLock`), so the parallel render path shares one map across workers.
    pub(crate) fn char_prefixes_arc(
        &self,
    ) -> &Arc<crate::diagnostics::render::CharPrefixMap> {
        &self.char_prefixes
    }

    /// Whether the indexed source is pure ASCII (byte cols == char cols).
    pub(crate) fn ascii(&self) -> bool {
        self.ascii
    }

    /// 0-based line containing `offset` (clamped to the source length).
    pub(crate) fn line_of(&self, offset: usize) -> usize {
        match self
            .starts
            .binary_search(&(offset.min(self.source.len()) as u32))
        {
            Ok(line) => line,
            Err(0) => 0,
            Err(next) => next - 1,
        }
    }

    /// Same mapping as [`byte_to_linecol`], via binary search.
    pub(crate) fn to_linecol(&self, byte_offset: usize) -> label::SourcePos {
        let clamped = floor_char_boundary(self.source, byte_offset);
        let line = self.line_of(clamped);
        label::SourcePos {
            line,
            col: clamped - self.starts[line] as usize,
        }
    }

    /// `(line, char_col)` for `byte_offset`: the line index and the
    /// character (not byte) column within that line, for display purposes.
    pub(crate) fn linecol_display(&self, byte_offset: usize) -> (usize, usize) {
        let pos = self.to_linecol(byte_offset);
        // ASCII fast path: byte columns equal character columns, so skip the
        // O(byte_col) `chars().count()` walk.  This is the only per-span
        // linear cost left in the JSON/HTML/short renderers, and real
        // compiler output is overwhelmingly ASCII.
        if self.ascii {
            return (pos.line, pos.col);
        }
        // O(1) table access: `lines().nth(pos.line)` would rescan the whole
        // file on every span conversion (JSON, HTML, short, call chains).
        // `line()` strips the same `\n` / `\r\n` terminator that
        // `str::lines` does, so the column arithmetic is unchanged.  The
        // char-count walk is amortised across every span on this line (and
        // every render sharing this index) by the per-line prefix cache in
        // [`char_col`](Self::char_col).
        (pos.line, self.char_col(pos.line, pos.col))
    }

    /// Character column of `byte_col` within line `line_idx`, via a lazily
    /// built per-line cache.
    ///
    /// `byte_col` must be a char boundary within the line (callers arrive via
    /// `floor_char_boundary`).  On the first query for a line, an ASCII line
    /// caches an empty table (char count = byte count, so the answer is
    /// `byte_col`); a non-ASCII line builds a sparse
    /// `(start_byte, cumulative_extra)` table — one entry per non-ASCII char,
    /// where `cumulative_extra` is the running sum of `len_utf8 − 1`.
    /// Later queries on the same line — across every span and every render in
    /// the batch that shares this index — are `O(1)` for ASCII lines and
    /// `O(log K)` (K = non-ASCII char count) for non-ASCII lines, both via
    /// `partition_point`.  Caching ASCII lines too avoids an `O(line_len)`
    /// `is_ascii()` rescan per span on non-ASCII sources with ASCII lines.
    /// Memory is `O(K)` (8 bytes per non-ASCII char) vs the old dense
    /// prefix's `O(line.len())`.
    fn char_col(&self, line_idx: usize, byte_col: usize) -> usize {
        // `line` (and the clamped `b`) are hoisted above both the cache-hit
        // and cache-miss paths: both need them, `line` is a pure function of
        // immutable state, and the compiler cannot merge two calls across the
        // lock boundary.
        let line = self.line(line_idx);
        let b = byte_col.min(line.len());
        // Identity of this index's shared prefix map (see `LAST_CHAR_COL`).
        let index_id = std::sync::Arc::as_ptr(&self.char_prefixes) as usize;
        // Identity of the indexed buffer: the second half of the memo key,
        // guarding against cross-batch address reuse (see `LAST_CHAR_COL`).
        let source_id = self.source.as_ptr() as usize;
        let source_len = self.source.len();
        let generation = self.generation;
        // Thread-local memo first: repeated queries on the same line — the
        // dominant shape, spans arriving in ascending order per line — skip
        // the lock below entirely.
        use std::sync::atomic::Ordering as _AtomicOrdering;
        let memo_hit = LAST_CHAR_COL.with(|cell| {
            let mut slot = cell.borrow_mut();
            match slot.as_mut() {
                Some((id, sid, slen, g, l, table, hint))
                    if *id == index_id
                        && *sid == source_id
                        && *slen == source_len
                        && *g == generation
                        && *l == line_idx =>
                match table {
                    // ASCII line: char col == byte col.
                    None => Some(b),
                    Some(t) => {
                        // The shared map's hint cannot be updated through the
                        // memo (it is a snapshot), but the hint is a heuristic
                        // — carry the advanced position inside the memo so a
                        // same-line run stays O(1) per step.
                        let hint_cell = std::sync::atomic::AtomicUsize::new(*hint);
                        let answer = char_col_from_sparse_hinted(t, &hint_cell, b);
                        *hint = hint_cell.load(_AtomicOrdering::Relaxed);
                        Some(answer)
                    }
                },
                _ => None,
            }
        });
        if let Some(answer) = memo_hit {
            return answer;
        }
        // Cache check *first*: a cached entry — `None` for ASCII lines, `Some`
        // sparse for non-ASCII — avoids the O(line_len) `is_ascii()` scan on
        // every call.  Without this hoist, ASCII lines in a non-ASCII source
        // re-scan `is_ascii()` per span because they never populated the cache
        // (the per-source `ascii` flag only short-circuits all-ASCII sources).
        // The map is behind a `RwLock` (shared across parallel workers); the
        // read lock is uncontended for the per-thread render shape.
        {
            let cache = self.char_prefixes.read().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(entry) = cache.get(&line_idx) {
                let answer = match entry {
                    Some((table, hint)) => {
                        let a = char_col_from_sparse_hinted(table, hint, b);
                        LAST_CHAR_COL.with(|cell| {
                            *cell.borrow_mut() = Some((
                                index_id,
                                source_id,
                                source_len,
                                generation,
                                line_idx,
                                Some(table.clone()),
                                hint.load(_AtomicOrdering::Relaxed),
                            ));
                        });
                        a
                    }
                    None => {
                        LAST_CHAR_COL.with(|cell| {
                            *cell.borrow_mut() = Some((
                                index_id,
                                source_id,
                                source_len,
                                generation,
                                line_idx,
                                None,
                                0,
                            ));
                        });
                        b // ASCII line — char col == byte col
                    }
                };
                return answer;
            }
        }
        // Non-ASCII source (the all-ASCII case never reaches `char_col`:
        // `linecol_display` short-circuits on the per-source `ascii` flag).
        // Build the sparse table directly — the SWAR walk
        // discovers "this line is actually pure ASCII" for free when it finds
        // no non-ASCII byte, so a separate `line.is_ascii()` SIMD pre-pass
        // over the line would be a redundant scan on exactly the lines this
        // cache-miss path exists for.
        // Build the sparse table: one `(start_byte, cumulative_extra)` pair
        // per non-ASCII char.  `cumulative_extra` is the running sum of
        // `len_utf8 − 1` — the extra bytes beyond what a 1-byte char would
        // use.  Since `b` is always a char boundary (callers arrive via
        // `floor_char_boundary`), every non-ASCII char either starts before
        // `b` (its full extra counts) or at/after `b` (none of it counts),
        // so `char_col = b − cumulative_extra` for chars with `start < b`.
        // O(K) writes + O(K) memory (K = non-ASCII char count).  Starts
        // inline: the dominant shape (a handful of CJK chars in an otherwise
        // ASCII line) never touches the heap; a genuinely CJK-heavy line
        // spills once and amortises the growth.
        let mut sparse: SmallVec<[(u32, u32); 8]> = SmallVec::new();
        let mut extra = 0u32;
        // Jump the ASCII runs with one SWAR probe per non-ASCII char
        // (`next_non_ascii_byte`) instead of decoding every character: for
        // the dominant mixed shape (mostly-ASCII line, occasional CJK) the
        // build is O(#non_ascii · chunk) rather than O(line_len) char decodes.
        let bytes = line.as_bytes();
        let mut i = 0usize;
        while let Some(off) = ascii::next_non_ascii_byte(&bytes[i..]) {
            let start = i + off;
            // `i` only ever sits on a char boundary (0, or just past a
            // decoded char), and in valid UTF-8 the first high byte after a
            // boundary is a lead byte — so this decode never fails.
            debug_assert!(line.is_char_boundary(start));
            let ch = line[start..].chars().next().unwrap();
            extra += (ch.len_utf8() - 1) as u32;
            sparse.push((start as u32, extra));
            i = start + ch.len_utf8();
        }
        if sparse.is_empty() {
            // An ASCII line inside a non-ASCII source: cache `None` so future
            // calls hit the cache above (`char_col_from_sparse(None, b)`
            // returns `b`), exactly as the old `is_ascii()` pre-pass did.
            let mut cache = self
                .char_prefixes
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cap_char_prefixes(&mut cache);
            cache.insert(line_idx, None);
            drop(cache);
            LAST_CHAR_COL.with(|cell| {
                *cell.borrow_mut() =
                    Some((index_id, source_id, source_len, generation, line_idx, None, 0))
            });
            return b;
        }
        let hint = std::sync::atomic::AtomicUsize::new(0);
        let answer = char_col_from_sparse_hinted(&sparse, &hint, b);
        let hint_val = hint.load(_AtomicOrdering::Relaxed);
        // Share one heap allocation between the cache and the memo: the old
        // `sparse.clone()` paid an O(K) copy on every non-ASCII line
        // transition.
        let table: std::sync::Arc<[(u32, u32)]> = sparse.into_vec().into();
        let mut cache = self
            .char_prefixes
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cap_char_prefixes(&mut cache);
        cache.insert(line_idx, Some((std::sync::Arc::clone(&table), hint)));
        drop(cache);
        LAST_CHAR_COL.with(|cell| {
            *cell.borrow_mut() = Some((
                index_id,
                source_id,
                source_len,
                generation,
                line_idx,
                Some(table),
                hint_val,
            ))
        });
        answer
    }

    /// `(start_line, end_line)` for a span: the lines of the first and last
    /// byte the span covers. `Span::end` is exclusive, so an end sitting
    /// exactly on a line start belongs to the *previous* line — attributing
    /// it to the following line made renderers suppress a label's message on
    /// the line that drew its underline and then treat it as already shown,
    /// so it was never emitted anywhere. A zero-width span covers no byte
    /// and keeps the position's own line.
    pub(crate) fn span_lines(&self, span: Span) -> (usize, usize) {
        let start = span.start_as_usize();
        let end = span.end_as_usize();
        let start_line = self.line_of(floor_char_boundary(self.source, start));
        let end_line = if end > start {
            self.line_of(floor_char_boundary(self.source, end - 1))
        } else {
            self.line_of(floor_char_boundary(self.source, end))
        };
        (start_line, end_line)
    }

    pub(crate) fn starts(&self) -> &[u32] {
        &self.starts[..]
    }

    pub(crate) fn line_count(&self) -> usize {
        self.starts.len()
    }

    pub(crate) fn source(&self) -> &'a str {
        self.source
    }

    /// `str::lines()`-style line count: a trailing `\n` does not create an
    /// extra "ghost" line, and an empty source counts zero lines.
    pub(crate) fn text_line_count(&self) -> usize {
        if self.source.is_empty() {
            return 0;
        }
        let last = self.starts.last().copied().unwrap_or(0) as usize;
        if last == self.source.len() && self.source.ends_with('\n') {
            self.starts.len() - 1
        } else {
            self.starts.len()
        }
    }

    /// Return the content of line `line_idx` (0-based), with the trailing `\n` / `\r\n` removed.
    pub(crate) fn line(&self, line_idx: usize) -> &'a str {
        let start = self.starts[line_idx] as usize;
        let end = self
            .starts
            .get(line_idx + 1)
            .map_or(self.source.len(), |&e| e as usize);
        let raw = &self.source[start..end];
        match raw.as_bytes().last() {
            Some(b'\n') => {
                let raw = &raw[..raw.len() - 1];
                raw.strip_suffix('\r').unwrap_or(raw)
            }
            _ => raw,
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::Span;

    #[test]
    fn test_basic_error() {
        let diag = Diagnostic::error("something went wrong")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 5, 12))
            .with_suggestion("try using `as` to cast");
        assert!(diag.is_error());
        assert_eq!(diag.code.as_ref().unwrap().code(), "E030");
    }

    /// C1/C2: identity must follow the *canonical* spelling.  A verbatim
    /// `unknown("e030")` equals `new("e030")` under `PartialEq`, so it must
    /// also hash to the same `stable_id` and be matched by `--allow e030`.
    #[test]
    fn stable_id_and_filter_use_canonical_code() {
        let known = Diagnostic::error("same").with_code(ErrCode::new("e030"));
        let verbatim = Diagnostic::error("same").with_code(ErrCode::unknown("e030"));
        assert_eq!(known.code(), verbatim.code(), "unknown() must echo verbatim");
        assert_eq!(
            known.stable_id(),
            verbatim.stable_id(),
            "stable_id must hash the canonical code, not the verbatim spelling"
        );

        let opts = DiagOpts {
            allow_codes: vec!["e030".to_string()],
            ..DiagOpts::default()
        };
        assert!(!known.matches_filter(&opts));
        assert!(
            !verbatim.matches_filter(&opts),
            "--allow e030 must suppress unknown(\"e030\")"
        );
    }

    #[test]
    fn test_with_labels() {
        let diag = Diagnostic::error("no field `z` on type Point")
            .with_code_str(ErrCode::E010)
            .with_span(Span::new(FileId::DUMMY, 50, 53))
            .with_label(Span::new(FileId::DUMMY, 0, 30), "type defined here");
        assert_eq!(diag.labels.len(), 1);
        assert_eq!(diag.labels[0].message, "type defined here");
    }

    #[test]
    fn test_error_code_explain() {
        let explain = explain_error_code("E030", true);
        assert!(explain.contains("type mismatch"));
        assert!(explain.contains("E030"));
    }

    #[test]
    fn test_explain_unknown_code() {
        let explain = explain_error_code("E03", true);
        assert!(explain.contains("unknown error code"));
        assert!(explain.contains("E03"));
        assert!(explain.contains("E030")); // did you mean suggestion
        assert!(explain.contains("help"));

        // Far from every known code — falls back to generic help.
        let explain = explain_error_code("Z999", true);
        assert!(explain.contains("unknown error code"));
        assert!(explain.contains("Z999"));
        assert!(explain.contains("valid error codes include"));
    }

    /// The unknown code is echoed into the printed explanation, so control
    /// characters from untrusted CLI input must not reach the terminal.
    #[test]
    fn explain_unknown_code_is_sanitized() {
        let out = explain_error_code("E0\x1b]8;;evil\x07", true);
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced: {out:?}"
        );
    }

    /// Multi-line sanitizer: `\n`/`\t` are layout intent and must survive;
    /// every other control character (ESC, BEL, CR, NEL) becomes U+FFFD.
    #[test]
    fn multiline_sanitizer_keeps_layout_controls_neuters_the_rest() {
        let out = sanitize_for_terminal_multiline("ok\n\tindent\x1b]8;;x\x07mid\r\u{85}");
        assert!(out.contains('\n'), "newline lost: {out:?}");
        assert!(out.contains('\t'), "tab lost: {out:?}");
        assert!(!out.contains('\x1b'), "ESC leaked: {out:?}");
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert!(!out.contains('\r'), "CR leaked: {out:?}");
        assert!(!out.contains('\u{85}'), "NEL leaked: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced: {out:?}"
        );
        assert!(matches!(
            sanitize_for_terminal_multiline("plain\n\ttext"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    /// A genuine C0 control byte sitting in the lane *after* a `\t`/`\n` must
    /// not be swallowed by the SWAR tab/newline exclusion's borrow
    /// propagation: `has_zero`'s mask is not position-exact, so a backspace
    /// adjacent to a tab used to be masked away and the sanitizer returned
    /// the raw control byte borrowed.
    #[test]
    fn multiline_sanitizer_catches_control_adjacent_to_layout_control() {
        for s in [
            "\t\x08abcdef", // backspace right after tab
            "\n\x08abcdef", // backspace right after newline
            "ab\t\x08cd",   // mid-string, chunk-interior
            "\x08\tabcdef", // backspace right *before* tab (borrow source)
            "ab\x7F\tcd",   // DEL adjacent to tab
            "ab\t\x7Fcd",   // tab adjacent to DEL
        ] {
            let out = sanitize_for_terminal_multiline(s);
            assert!(
                matches!(out, std::borrow::Cow::Owned(_)),
                "raw control byte leaked as Borrowed for {s:?}"
            );
            assert!(out.contains('\u{FFFD}'), "control not replaced: {out:?}");
            assert!(!out.contains('\x08') && !out.contains('\x7F'), "{out:?}");
        }
        // Layout controls themselves survive, alone or adjacent to each other.
        for s in ["\t\n", "a\t\nb", "\t\ta\n\n"] {
            assert!(
                matches!(
                    sanitize_for_terminal_multiline(s),
                    std::borrow::Cow::Borrowed(_)
                ),
                "layout controls forced owned for {s:?}"
            );
        }
    }

    /// The non-ASCII no-op fast path must borrow (not allocate) for
    /// localized text with no control characters — the dominant i18n case.
    /// Pure-CJK / Cyrillic / accented text contains no controls, so the
    /// sanitizer must return `Cow::Borrowed`, and the owned path must still
    /// replace only the control characters.
    #[test]
    fn sanitizer_borrows_non_ascii_without_controls() {
        // Borrowed (no allocation) for the i18n hot case.
        for s in [
            "функция main не найдена",
            "crate 中没有找到 main 函数",
            "Datei nicht gefunden",
            "café crème",
            "안녕하세요",
        ] {
            assert!(
                matches!(sanitize_for_terminal(s), std::borrow::Cow::Borrowed(_)),
                "sanitize_for_terminal allocated for {s:?}"
            );
            assert!(
                matches!(
                    sanitize_for_terminal_multiline(s),
                    std::borrow::Cow::Borrowed(_)
                ),
                "sanitize_for_terminal_multiline allocated for {s:?}"
            );
        }
        // A control character in a non-ASCII string still triggers replacement.
        let out = sanitize_for_terminal("中\x1b文");
        assert!(matches!(out, std::borrow::Cow::Owned(_)));
        assert!(out.contains('\u{FFFD}'), "{out:?}");
        assert!(!out.contains('\x1b'), "{out:?}");
        // `\n`/`\t` survive multiline without forcing the owned path to
        // replace them, and other controls become U+FFFD.
        let out = sanitize_for_terminal_multiline("中\n文\tx\x07");
        assert!(matches!(out, std::borrow::Cow::Owned(_)));
        assert!(out.contains('\n') && out.contains('\t'), "{out:?}");
        assert!(!out.contains('\x07'), "{out:?}");
    }

    #[test]
    fn test_suggest_code_osa_distance() {
        // Single substituted digit — same-length typo the old heuristic missed.
        let s = suggest_code("E031", true);
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Missing trailing digit.
        let s = suggest_code("E03", true);
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Lowercase typo still matches the canonical code.
        let s = suggest_code("e030", true);
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Transposed digits: `E300`↔`E030` is OSA-distance 1, so the
        // radius-1 scan must now suggest `E030` (plain Levenshtein scored 2
        // and silently dropped it).  `E300` is not a known code, so this
        // exercises the real unknown-code suggestion path.
        let s = suggest_code("E300", true);
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Complete garbage — no suggestions, fall back to generic help.
        let s = suggest_code("????", true);
        assert!(s.contains("valid error codes include"), "got: {s}");
    }

    #[test]
    fn test_osa_distance() {
        assert_eq!(osa_distance("", ""), 0);
        assert_eq!(osa_distance("abc", "abc"), 0);
        assert_eq!(osa_distance("abc", "abd"), 1);
        assert_eq!(osa_distance("abc", "ab"), 1);
        assert_eq!(osa_distance("abc", "abcd"), 1);
        assert_eq!(osa_distance("abc", ""), 3);
        assert_eq!(osa_distance("", "abc"), 3);
        assert_eq!(osa_distance("kitten", "sitting"), 3);
        assert_eq!(osa_distance("E03", "E030"), 1);
        assert_eq!(osa_distance("E03", "E001"), 2);
        // Adjacent transpositions cost 1 under OSA, 2 under plain Levenshtein.
        // These are the keyboard-typo cases the radius-1 scan used to drop.
        assert_eq!(osa_distance("lne", "len"), 1);
        assert_eq!(osa_distance("E003", "E030"), 1);
    }

    #[test]
    fn test_call_chain() {
        let mut chain = CallChain::new();
        chain.push(chain::ChainEntry::new(
            chain::ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 80, 90)),
            "main",
        ));
        let diag = Diagnostic::error("type mismatch")
            .with_code_str(ErrCode::E030)
            .with_call_chain(chain);
        assert!(diag.call_chain.is_some());
        assert!(diag.to_string().contains("referenced by"));
    }

    #[test]
    fn test_from_string_code() {
        let diag = Diagnostic::error("test").with_code_str(ErrCode::E030);
        assert_eq!(diag.code.unwrap().code(), "E030");
    }

    #[test]
    fn test_diagnostic_collector() {
        let mut collector = DiagCtxt::new();
        collector.push(Diagnostic::error("first error").with_code_str(ErrCode::E030));
        collector.push(Diagnostic::warning("first warning"));
        assert_eq!(collector.len(), 2);
        assert_eq!(collector.summary(), (1, 1));
        assert!(collector.has_errors());
    }

    #[test]
    fn test_multiple_suggestions() {
        let diag = Diagnostic::error("test")
            .with_suggestion("first option")
            .with_suggestion("second option");
        assert_eq!(diag.suggestions.len(), 2);
        let text = diag.to_string();
        assert!(text.contains("first option"));
        assert!(text.contains("second option"));
    }

    #[test]
    fn test_help_text() {
        let diag = Diagnostic::error("test").with_help("try adding a type annotation");
        assert_eq!(diag.help.as_deref(), Some("try adding a type annotation"));
        let text = diag.to_string();
        assert!(text.contains("try adding a type annotation"));
    }

    /// A highlighted note stores only the styled form; the plain text is
    /// derived on demand, so a note that is never rendered never allocates it.
    #[test]
    fn highlighted_note_derives_plain_message_on_demand() {
        let styled = StyledString::new(vec![
            StringPart {
                content: "expected ".into(),
                style: StringPartStyle::Normal,
            },
            StringPart {
                content: "Int<32>".into(),
                style: StringPartStyle::Highlighted,
            },
        ]);
        let diag = Diagnostic::error("mismatch").with_highlighted_note(styled);
        let child = &diag.children()[0];
        assert!(child.message.is_none(), "plain text materialized eagerly");
        assert_eq!(child.plain_message(), "expected Int<32>");
    }

    #[test]
    fn test_error_category() {
        assert_eq!(ErrCode::new(ErrCode::E030).category(), ErrorCategory::Type);
        assert_eq!(ErrCode::new(ErrCode::E001).category(), ErrorCategory::Parse);
        assert_eq!(
            ErrCode::new(ErrCode::E020).category(),
            ErrorCategory::Contract
        );
        assert_eq!(ErrCode::new(ErrCode::E041).category(), ErrorCategory::Trait);
    }

    #[test]
    fn test_fatal_counts_as_error() {
        assert!(Diagnostic::fatal("boom").is_error());
        assert!(Diagnostic::error("boom").is_error());
        assert!(!Diagnostic::warning("boom").is_error());

        let mut ctxt = DiagCtxt::new();
        ctxt.push(Diagnostic::fatal("boom"));
        assert!(ctxt.has_errors());
        assert_eq!(ctxt.error_count(), 1);
        assert!(ctxt.check().is_err());
    }

    #[test]
    fn test_adorn_with_suppressed_warning_does_not_panic() {
        use crate::diagnostics::EmissionGuaranteed;
        let mut ctxt = DiagCtxt::new();
        ctxt.set_can_emit_warnings(false);
        let result: Result<(), EmissionGuaranteed> = adorn_with_span(
            &mut ctxt,
            Span::new(FileId::DUMMY, 0, 5),
            Some("hello"),
            |ctxt| {
                // `push` returns a *suppressed* guarantee here and records
                // nothing, so the old debug_assert would have fired.
                Err(ctxt.push(Diagnostic::warning("suppressed")).into())
            },
        );
        assert!(result.is_err());
        assert!(ctxt.is_empty());
    }

    #[test]
    fn adorn_with_span_survives_humanize() {
        // The `adorn_with_span` + `*_kind` interaction: the adornment
        // attaches the closure's call-site span to a kind-built
        // diagnostic, and the emitter's later `humanize()` must keep it
        // instead of replacing it with kind-derived primary spans.
        use crate::diagnostics::EmissionGuaranteed;
        let mut ctxt = DiagCtxt::new();
        let adorn_span = Span::new(FileId::DUMMY, 10, 15);
        let kind_span = Span::new(FileId::DUMMY, 0, 3);
        let result: Result<(), EmissionGuaranteed> =
            adorn_with_span(&mut ctxt, adorn_span, Some("src"), |ctxt| {
                Err(ctxt
                    .push(Diagnostic::error("placeholder").with_kind(
                        DiagnosticKind::TypeMismatch {
                            expected: Cow::Borrowed("Int<32>"),
                            found: Cow::Borrowed("Str"),
                            span: kind_span,
                            found_span: None,
                            reason: None,
                            context: None,
                        },
                    ))
                    .into())
            });
        assert!(result.is_err());
        for diag in ctxt.iter_mut() {
            diag.humanize();
            assert_eq!(diag.spans.primary().as_slice(), &[adorn_span][..]);
            // Kind derivation itself is unaffected.
            assert_eq!(diag.labels.len(), 1);
            assert_eq!(diag.labels[0].span, kind_span);
        }
    }

    #[test]
    fn test_known_code_ranges_cover_table() {
        let ranges = known_code_ranges();
        assert!(ranges.contains("E001"), "got: {ranges}");
        assert!(ranges.contains("W113"), "got: {ranges}");
        assert!(ranges.contains("E601"), "got: {ranges}");
    }

    #[test]
    fn code_ranges_sort_numerically_within_prefix() {
        // A 4-digit code would sort BEFORE E999 lexicographically; the
        // numeric-aware key keeps numeric order (E1000 after E999), and
        // E999–E1000 merge because they are numerically contiguous.
        let ranges = format_code_ranges(&["E999", "E1000", "E001", "E002", "W113", "W9"]);
        assert_eq!(ranges, "E001–E002, E999–E1000, W9, W113");
    }

    #[test]
    fn err_code_resolves_case_insensitively() {
        // A lowercase code resolves to its canonical table entry, matching
        // `suggest_code`'s case-insensitive comparison — not unknown-code
        // metadata.
        assert_eq!(ErrCode::new("e030").code(), "E030");
        assert_eq!(ErrCode::new("e030").category(), ErrorCategory::Type);
        assert_eq!(ErrCode::try_new("e030").unwrap().code(), "E030");
        // Truly unknown strings stay verbatim (case is never invented).
        assert_eq!(ErrCode::new("e9999").code(), "e9999");
        let err = ErrCode::try_new("zzz").unwrap_err();
        assert_eq!(err.0.as_ref(), "zzz");
    }

    #[test]
    fn test_e067_is_registered() {
        // Regression: `?with_code_str(ErrCode::E067)` in the PANN pattern-annotation
        // path would `debug_assert!`-panic if the code were absent.
        assert!(ErrCode::try_new("E067").is_ok());
    }

    #[test]
    fn test_with_code_str_unknown_does_not_panic() {
        // `with_code_str` is public and may receive untrusted input; it must
        // not hit `ErrCode::new`'s debug assertion.
        let diag = Diagnostic::error("boom").with_code_str("ZZZ999");
        let code = diag.code().expect("code set");
        assert_eq!(code.code(), "ZZZ999");
        assert_eq!(code.title(), "unknown error");
    }

    #[test]
    fn tr_does_not_double_substitute() {
        // A substituted value that itself contains a placeholder must be
        // emitted verbatim, not re-expanded by a later replacement pass.
        let s = crate::tr!("type-mismatch", expected = "{found}", found = "Int<32>");
        assert_eq!(s, "type mismatch: expected `{found}`, found `Int<32>`");
    }

    #[test]
    fn tr_handles_unknown_and_unterminated_placeholders() {
        assert_eq!(
            crate::tr!("no placeholders here", x = "ignored"),
            "no placeholders here"
        );
    }

    #[test]
    fn humanize_after_rekind_refreshes_derived_notes() {
        // A diagnostic can be re-kind'ed (with_kind(k1).humanize() then
        // with_kind(k2).humanize()); k1's derived help must not survive on a
        // diagnostic that is no longer k1.
        let mut diag = Diagnostic::error("placeholder").with_kind(DiagnosticKind::TypeMismatch {
            expected: Cow::Borrowed("Int<32>"),
            found: Cow::Borrowed("Str"),
            span: Span::new(FileId::DUMMY, 0, 3),
            found_span: None,
            reason: None,
            context: None,
        });
        diag.humanize();
        assert_eq!(
            diag.help.as_deref(),
            Some("try using `as` to cast, or change the expression type")
        );
        let mut diag = diag.with_kind(DiagnosticKind::NoSuchField {
            field_name: Cow::Borrowed("z"),
            type_name: Cow::Borrowed("Point"),
            span: Span::new(FileId::DUMMY, 0, 3),
        });
        diag.humanize();
        assert_eq!(diag.help, None);
        assert!(
            diag.message.contains("no field `z` found on type `Point`"),
            "message: {}",
            diag.message
        );
    }

    #[test]
    fn humanize_preserves_user_set_spans() {
        // `with_kind` first, then a builder span: the span is deliberate
        // user context and must survive `humanize()` instead of being
        // replaced by spans derived from the kind's labels.
        let kind_span = Span::new(FileId::DUMMY, 0, 3);
        let user_span = Span::new(FileId::DUMMY, 40, 45);
        let mut diag = Diagnostic::error("placeholder")
            .with_kind(DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("Int<32>"),
                found: Cow::Borrowed("Str"),
                span: kind_span,
                found_span: None,
                reason: None,
                context: None,
            })
            .with_span(user_span);
        diag.humanize();
        // User span kept; the kind-derived span was NOT merged in.
        assert_eq!(diag.spans.primary().as_slice(), &[user_span][..]);
        // The rest of the kind derivation is unchanged.
        assert_eq!(diag.labels.len(), 1);
        assert_eq!(diag.labels[0].span, kind_span);
        assert_eq!(
            diag.help.as_deref(),
            Some("try using `as` to cast, or change the expression type")
        );
    }

    #[test]
    fn humanize_preserves_user_set_multiple_spans() {
        // `with_span` + `with_additional_span` on a kind-built diagnostic:
        // every user-set primary survives re-derivation.
        let user1 = Span::new(FileId::DUMMY, 40, 45);
        let user2 = Span::new(FileId::DUMMY, 100, 103);
        let mut diag = Diagnostic::error_kind(DiagnosticKind::NoSuchField {
            field_name: Cow::Borrowed("z"),
            type_name: Cow::Borrowed("Point"),
            span: Span::new(FileId::DUMMY, 0, 3),
        })
        .with_span(user1)
        .with_additional_span(user2);
        diag.humanize();
        assert_eq!(diag.spans.primary().as_slice(), &[user1, user2][..]);
    }

    #[test]
    fn humanize_derives_spans_when_not_user_set() {
        // No builder span: humanize must still derive primary spans from
        // the kind's labels (the original behaviour, unchanged).
        let kind_span = Span::new(FileId::DUMMY, 0, 3);
        let mut diag = Diagnostic::error("placeholder").with_kind(DiagnosticKind::TypeMismatch {
            expected: Cow::Borrowed("Int<32>"),
            found: Cow::Borrowed("Str"),
            span: kind_span,
            found_span: None,
            reason: None,
            context: None,
        });
        diag.humanize();
        assert_eq!(diag.spans.primary().as_slice(), &[kind_span][..]);
    }

    /// `allow_codes` is a *suppress* list (rustc `-A` semantics): a code in
    /// the set is dropped, everything else passes.  Pins the direction of the
    /// filter so the branch can never be "fixed" into the opposite silently.
    #[test]
    fn allow_codes_suppresses_matching_diagnostics() {
        let mut opts = DiagOpts::default();
        opts.allow_codes = vec!["E030".to_string()];
        let suppressed = Diagnostic::error("x").with_code_str("E030");
        let kept = Diagnostic::error("y").with_code_str("E031");
        let no_code = Diagnostic::error("z");
        assert!(!suppressed.matches_filter(&opts), "E030 must be suppressed");
        assert!(kept.matches_filter(&opts), "E031 must pass");
        assert!(no_code.matches_filter(&opts), "code-less diagnostics pass");
        // Empty list: no-op.
        let unfiltered = DiagOpts::default();
        assert!(suppressed.matches_filter(&unfiltered));
    }

    /// Filter codes resolve case-insensitively, like every other code API:
    /// `--allow e030` must suppress the canonical `E030` diagnostic instead
    /// of silently matching nothing.
    #[test]
    fn filter_codes_resolve_case_insensitively() {
        let mut opts = DiagOpts::default();
        opts.allow_codes = vec!["e030".to_string()];
        let suppressed = Diagnostic::error("x").with_code_str("E030");
        assert!(
            !suppressed.matches_filter(&opts),
            "lowercase allow pattern must suppress E030"
        );

        let mut only = DiagOpts::default();
        only.only_codes = vec!["e031".to_string()];
        assert!(
            Diagnostic::error("x")
                .with_code_str("E031")
                .matches_filter(&only),
            "lowercase only-code must match E031"
        );
        assert!(
            !Diagnostic::error("x")
                .with_code_str("E030")
                .matches_filter(&only),
            "only-code still excludes other codes"
        );
    }

    /// Regression: the SWAR newline mask is a *candidate* set, not exact — a
    /// `\n` followed by `0x0B` (VT) makes the borrow out of the zero lane set
    /// the next lane's high bit, and the old code pushed a fake line start.
    /// Line numbers after that point were off by one.
    #[test]
    fn line_scan_does_not_invent_newlines_after_vt() {
        // Lay the poison pattern across every 8-byte lane alignment.
        let mut src = String::new();
        for pad in 0..8 {
            src.push_str(&"x".repeat(pad));
            src.push_str("a\n\u{0B}b\n");
        }
        let idx = LineIndex::new(&src);
        // A `0x0B` is valid UTF-8 but never a line break: the only newlines
        // in `src` are the two per repetition.
        let mut expected = vec![0u32];
        let mut off = 0usize;
        for pad in 0..8 {
            // "xxx…a\n" — the `\n` after `a` sits at `off + pad + 1`, so the
            // poisoned line (`\x0Bb`) starts at `off + pad + 2`.
            expected.push((off + pad + 2) as u32);
            // "\x0Bb\n" — the `\n` after `b` sits at `off + pad + 4`, so the
            // next repetition's line starts at `off + pad + 5`.  The `0x0B`
            // itself must NOT have produced a start.
            off += pad + 5;
            expected.push(off as u32);
        }
        // Exactly 2 starts per repetition + the initial 0: a `0x0B` never
        // contributes a line start.
        assert_eq!(expected.len(), 1 + 2 * 8);
        assert_eq!(idx.starts(), expected.as_slice(), "src = {src:?}");
        // And the mapping round-trips for every line.
        for i in 0..idx.line_count() {
            let pos = idx.to_linecol(idx.starts()[i] as usize);
            assert_eq!((pos.line, pos.col), (i, 0));
        }
    }

    #[test]
    fn line_scan_ignores_utf8_continuation_0x8a() {
        // `Ċ` is C4 8A and `ꀀ` is F0 8A 80 80: `q = x ^ 0x0A` is 0x80 for the
        // continuation byte, which the old `(q | HI)` mask mistook for `\n`.
        // Lay the pattern across every 8-byte alignment and compare the whole
        // table with a scalar newline scan.
        let mut src = String::new();
        for pad in 0..8 {
            src.push_str(&"x".repeat(pad));
            src.push('Ċ');
            src.push('\n');
            src.push('ꀀ');
            src.push_str("z\n");
        }
        let idx = LineIndex::new(&src);
        let expected: Vec<u32> = std::iter::once(0)
            .chain(
                src.bytes()
                    .enumerate()
                    .filter(|(_, byte)| *byte == b'\n')
                    .map(|(i, _)| (i + 1) as u32),
            )
            .collect();
        assert_eq!(idx.starts(), expected.as_slice(), "src = {src:?}");
        for i in 0..idx.line_count() {
            let pos = idx.to_linecol(idx.starts()[i] as usize);
            assert_eq!((pos.line, pos.col), (i, 0));
        }
    }

    /// Boundary matrix for `LineIndex`: empty input, single byte, with and
    /// without trailing newlines, multi-byte leading char, and CRLF. All
    /// conversions (including EOF and past-EOF offsets) must stay panic-free
    /// and self-consistent: every line start maps back to (line, 0), and
    /// `to_linecol` never produces an out-of-range line.
    #[test]
    fn line_index_boundaries_stay_consistent() {
        let sources = ["", "a", "a\n", "a\n\n", "中文\n", "\t\n", "abc\r\ndef\r\n"];
        for src in sources {
            let idx = LineIndex::new(src);
            let eof = idx.to_linecol(src.len());
            assert!(
                eof.line < idx.line_count(),
                "eof line {} out of range for {src:?}",
                eof.line
            );
            let past = idx.to_linecol(usize::MAX);
            assert!(past.line < idx.line_count());
            let eof_span = Span::new(FileId::DUMMY, src.len() as u32, src.len() as u32);
            let (first, last) = idx.span_lines(eof_span);
            assert!(first <= last && last < idx.line_count(), "span {src:?}");
            let over = Span::new(FileId::DUMMY, 0, (src.len() as u32).saturating_add(8));
            let (first, last) = idx.span_lines(over);
            assert!(first <= last && last < idx.line_count(), "over {src:?}");
            for i in 0..idx.line_count() {
                let pos = idx.to_linecol(idx.starts()[i] as usize);
                assert_eq!((pos.line, pos.col), (i, 0), "start of line {i} in {src:?}");
            }
        }
        // Known values: trailing newline does not create a phantom text line.
        let idx = LineIndex::new("a\n");
        assert_eq!(idx.text_line_count(), 1);
        assert_eq!(idx.line(0), "a");
        let idx = LineIndex::new("a\n\n");
        assert_eq!(idx.text_line_count(), 2);
        assert_eq!(idx.line(1), "");
        let idx = LineIndex::new("中文\n");
        assert_eq!(idx.line(0), "中文");
        let pos = idx.to_linecol("中文".len());
        assert_eq!((pos.line, pos.col), (0, "中文".len()));
    }

    /// `linecol_display` returns *character* columns on non-ASCII sources,
    /// backed by the lazily-built per-line char-count prefix cache.  This
    /// pins the cache's correctness (dense prefix build + lookup), the
    /// ASCII-line-in-a-non-ASCII-source fast path, reuse across repeated
    /// queries on the same line, and sharing across indexes built from the
    /// same cached source via [`LineIndex::from_shared`].
    #[test]
    fn linecol_display_char_columns_via_prefix_cache() {
        // Line 0: two 3-byte chars "中" "文"; line 1: ASCII "ab"; line 2:
        // a 4-byte char "𝄞" (U+1D11E) then ASCII.
        let src = "中文\nab\n𝄞x\n";
        let idx = LineIndex::new(src);
        assert!(!idx.ascii());
        // Byte layout: 中=0..3, 文=3..6, \n=6, a=7, b=8, \n=9, 𝄞=10..14
        // (4 bytes), x=14, \n=15.  Char boundaries only — `linecol_display`
        // floors non-boundary offsets, so test the boundaries directly.
        assert_eq!(idx.linecol_display(0), (0, 0));
        assert_eq!(idx.linecol_display(3), (0, 1));
        assert_eq!(idx.linecol_display(6), (0, 2));
        // Line 1 is ASCII inside a non-ASCII source: char col == byte col,
        // exercised through the per-line `is_ascii` fast path in `char_col`.
        assert_eq!(idx.linecol_display(7), (1, 0));
        assert_eq!(idx.linecol_display(8), (1, 1));
        assert_eq!(idx.linecol_display(9), (1, 2));
        // Line 2: "𝄞" is 4 bytes, "x" follows at byte 14.
        assert_eq!(idx.linecol_display(10), (2, 0));
        assert_eq!(idx.linecol_display(14), (2, 1));
        assert_eq!(idx.linecol_display(15), (2, 2));

        // Repeated queries on the same line hit the cached prefix (the build
        // runs once); the answers must be stable across calls.
        assert_eq!(idx.linecol_display(3), (0, 1));
        assert_eq!(idx.linecol_display(14), (2, 1));

        // A second index sharing the cache (the RenderCtx hit path) reuses
        // the prefixes the first index built: build on `idx`, then verify
        // `shared` agrees without rebuilding.  Correctness is observable via
        // the answers; sharing via `from_shared` is what makes a batch of M
        // diagnostics over one long non-ASCII line pay one build, not M.
        idx.linecol_display(3);
        let shared = LineIndex::from_shared(
            src,
            std::sync::Arc::clone(idx.starts_arc()),
            idx.ascii(),
            std::sync::Arc::clone(idx.char_prefixes_arc()),
            idx.generation(),
        );
        assert_eq!(shared.linecol_display(3), (0, 1));
        assert_eq!(shared.linecol_display(14), (2, 1));
    }

    /// The hinted lookup must agree with the simple reference implementation
    /// on every offset, including an ABA-style shrink that forces the hint to
    /// re-anchor.  This is the differential oracle the doc comment on
    /// `char_col_from_sparse` promises.
    #[test]
    fn char_col_hinted_matches_reference() {
        // (start_byte, cumulative_extra) for chars that are 2–4 bytes wide,
        // interleaved with ASCII gaps.
        let table: &[(u32, u32)] = &[
            (2, 1),   // 2-byte char at byte 2
            (5, 3),   // 2-byte char at 5 (+1), then 4-byte (+3)
            (12, 4),  // another 2-byte char
            (20, 6),
        ];
        let hint = std::sync::atomic::AtomicUsize::new(0);
        // Ascending, descending, and repeated offsets: every path (advance,
        // overshoot re-anchor, unchanged) must agree with the reference.
        for b in (0..30).chain((0..30).rev()).chain([0, 3, 3, 12, 12, 28]) {
            let want = char_col_from_sparse(Some(table), b);
            let got = char_col_from_sparse_hinted(table, &hint, b);
            assert_eq!(got, want, "b={b}");
        }
        // `None` (ASCII line) is the identity on both.
        for b in 0..30 {
            assert_eq!(char_col_from_sparse(None, b), b);
        }
    }

    /// A non-empty span's end is exclusive: an end sitting exactly on a line
    /// start belongs to the previous line (the line holding the last covered
    /// byte), so renderers never attribute a label to a line they cannot
    /// draw it on. Zero-width spans have no covered byte and keep the
    /// position's own line.
    #[test]
    fn span_lines_end_on_line_start_belongs_to_previous_line() {
        let idx = LineIndex::new("aaa\nbbb\n");
        // End exactly on the start of line 1: the last covered byte ('a'
        // at 2) is on line 0.
        assert_eq!(idx.span_lines(Span::new(FileId::DUMMY, 0, 4)), (0, 0));
        // Zero-width at the same boundary keeps the following line.
        assert_eq!(idx.span_lines(Span::new(FileId::DUMMY, 4, 4)), (1, 1));
        // Span covering a whole line plus its newline.
        assert_eq!(idx.span_lines(Span::new(FileId::DUMMY, 0, 8)), (0, 1));
        // End inside a later line: attribution unchanged.
        assert_eq!(idx.span_lines(Span::new(FileId::DUMMY, 1, 5)), (0, 1));
    }

    // ── Renderer timing harness (benchmark-first, no speed asserts) ──
    // Run: `cargo test perf_ -- --nocapture`. Plain `#[test]`s (not
    // criterion) because the crate is binary-only, so an external
    // `benches/` target cannot import the renderer. The `[perf]` lines
    // are the before/after comparison for the renderer optimizations.

    fn perf_report(name: &str, samples_ms: &mut [f64], out_len: usize) {
        samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples_ms[samples_ms.len() / 2];
        let mean = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
        println!(
            "[perf] {name}: median {median:.2} ms, mean {mean:.2} ms over {} iters (output {out_len} bytes)",
            samples_ms.len()
        );
    }

    /// (a) Very long single-line file: stresses O(n) offset rescans per
    /// conversion plus the per-label grapheme re-walk in
    /// `byte_range_to_display`.
    #[test]
    fn perf_long_single_line() {
        let renderer = GlyphRenderer::new(false);
        let source = "x".repeat(500_000);
        let mut diag = Diagnostic::error("long line boom")
            .with_span(Span::new(FileId::DUMMY, 499_000, 499_010))
            .with_source(&source)
            .with_file_name("<input>");
        for i in 0..20 {
            let s = 480_000 + i * 500;
            diag = diag
                .with_secondary_label(Span::new(FileId::DUMMY, s, s + 10), format!("label {i}"));
        }
        for _ in 0..2 {
            let _ = std::hint::black_box(renderer.render_diagnostic(&diag));
        }
        let mut samples = Vec::with_capacity(5);
        let mut len = 0;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let out = renderer.render_diagnostic(&diag);
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
            len = std::hint::black_box(out).len();
        }
        perf_report("long_single_line(500KB,20 labels)", &mut samples, len);
    }

    /// (b) Many labels + primary spans on a multi-line file: stresses
    /// per-span section recomputation, the per-row label scan, and the
    /// message-string multiline matching.
    #[test]
    fn perf_many_labels_spans() {
        let renderer = GlyphRenderer::new(false);
        let mut source = String::new();
        let mut line_off: Vec<u32> = Vec::with_capacity(5_000);
        for i in 0..5_000 {
            line_off.push(u32::try_from(source.len()).unwrap_or(u32::MAX));
            source.push_str(&format!("let v{i} = {i} + {i};\n"));
        }
        let mut diag = Diagnostic::error("many labels")
            .with_span(Span::new(FileId::DUMMY, line_off[100], line_off[100] + 5))
            .with_source(&source)
            .with_file_name("<input>");
        for k in 1..8 {
            let l = k * 600;
            diag =
                diag.with_additional_span(Span::new(FileId::DUMMY, line_off[l], line_off[l] + 5));
        }
        for i in 0..200 {
            let l = (i * 25) % 5_000;
            diag = diag.with_secondary_label(
                Span::new(FileId::DUMMY, line_off[l], line_off[l] + 4),
                format!("note {i}"),
            );
        }
        for _ in 0..1 {
            let _ = std::hint::black_box(renderer.render_diagnostic(&diag));
        }
        let mut samples = Vec::with_capacity(5);
        let mut len = 0;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let out = renderer.render_diagnostic(&diag);
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
            len = std::hint::black_box(out).len();
        }
        perf_report(
            "many_labels(5k lines,8 spans,200 labels)",
            &mut samples,
            len,
        );
    }

    /// Multi-diagnostic batch over one large file: without a shared index,
    /// every diagnostic rescans the whole file (`render_diagnostic`); with
    /// the per-batch [`RenderCtx`] cache the scan happens once.  A/B in one
    /// run, no asserts (benchmark-first harness).
    #[test]
    fn perf_many_diags_one_source() {
        let source = "let x = 1;\n".repeat(20_000); // ~220 KB
        let diags: Vec<Diagnostic<'_>> = (0..200)
            .map(|i| {
                let off = u32::try_from((i * 997) % (source.len() - 2)).unwrap_or(u32::MAX);
                Diagnostic::error("boom")
                    .with_span(Span::new(FileId::DUMMY, off, off.saturating_add(1)))
                    .with_source(&source)
                    .with_file_name("<input>")
            })
            .collect();
        let renderer = GlyphRenderer::new(false);

        // Uncached: a fresh index per diagnostic (the old behaviour).
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for d in &diags {
                let _ = std::hint::black_box(renderer.render_diagnostic(d));
            }
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        perf_report("many_diags_uncached(220KB,200 diags)", &mut samples, 0);

        // Cached: one index per batch, shared by every diagnostic.
        let opts = DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for d in &diags {
                let idx = cx.line_index(d.source().expect("source attached"));
                let _ = std::hint::black_box(renderer.render_diagnostic_with_index(d, Some(&idx)));
            }
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        perf_report("many_diags_cached(220KB,200 diags)", &mut samples, 0);
    }

    /// Micro-isolation of item 1: K indexed conversions near the end of
    /// a 220KB file. Index build is O(n) once; each lookup is O(log n).
    #[test]
    fn perf_byte_to_linecol() {
        let source = "let x = 1;\n".repeat(20_000);
        let offs: Vec<usize> = (0..200).map(|i| source.len() - 1 - i * 100).collect();
        let idx = LineIndex::new(&source);
        for _ in 0..2 {
            for &o in &offs {
                std::hint::black_box(idx.to_linecol(o));
            }
        }
        let iters = 200;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            for &o in &offs {
                std::hint::black_box(idx.to_linecol(o));
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("[perf] LineIndex.to_linecol x200 on 220KB: {ms:.2} ms/iter");
    }

    /// Micro-isolation of the wrap fast paths: (1) short ASCII messages hit
    /// the `len <= max_width` guard without a grapheme walk; (2) long
    /// printable-ASCII prose exercises `ascii_break_opportunities`'s `memchr`
    /// space scan.  Run: `cargo test perf_wrap_optimal -- --nocapture`.
    #[test]
    fn perf_wrap_optimal() {
        // (1) Short ASCII: the len guard makes this ~O(1) per call.
        let short_msgs: Vec<String> = (0..200)
            .map(|i| format!("error code E{i:03} in module mod_{i}"))
            .collect();
        for _ in 0..2 {
            for m in &short_msgs {
                std::hint::black_box(wrap::wrap_optimal(m, 100, 4));
            }
        }
        let iters = 1000;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            for m in &short_msgs {
                std::hint::black_box(wrap::wrap_optimal(m, 100, 4));
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("[perf] wrap_optimal short ASCII x200: {ms:.4} ms/iter");

        // (2) Long prose: ~80 words repeated, wraps to several lines,
        // exercises the ascii_break_opportunities memchr space scan.
        let long_prose = "the quick brown fox jumps over the lazy dog and then \
            runs through the forest looking for a place to sleep but finds only \
            rocks and trees and a small stream that flows down the mountain side \
            into a valley where the wind blows cold and the leaves rustle loudly \
            while a distant owl hoots twice before falling silent again forever"
            .repeat(4);
        for _ in 0..2 {
            std::hint::black_box(wrap::wrap_optimal(&long_prose, 60, 4));
        }
        let iters = 2000;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(wrap::wrap_optimal(&long_prose, 60, 4));
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!(
            "[perf] wrap_optimal long prose ({} bytes): {ms:.4} ms/iter",
            long_prose.len()
        );
    }

    #[test]
    fn multispan_push_dedup_and_sort() {
        let f1 = FileId::from_raw(1);
        let f2 = FileId::from_raw(2);
        let s_a = Span::new(f1, 10, 20);
        let s_b = Span::new(f1, 30, 40);
        let s_c = Span::new(f2, 0, 5);
        let s_d = Span::new(f1, 0, 4);

        let mut ms = MultiSpan::new(s_b);
        ms.push(s_a);
        ms.push(s_c);
        ms.push(s_d);
        ms.push(s_a);
        ms.push(s_b);
        ms.push(s_c);

        assert_eq!(ms.primary().as_slice(), &[s_d, s_a, s_b, s_c]);
    }

    /// `MultiSpan::push` keeps the sorted+dedup invariant for interleaved
    /// traffic: descending pushes (binary-insert path), a duplicate landing
    /// in the middle, and an insertion between two existing entries.
    #[test]
    fn multispan_push_reverse_middle_and_dup() {
        let f1 = FileId::from_raw(1);
        let s_a = Span::new(f1, 10, 20);
        let s_mid = Span::new(f1, 20, 25);
        let s_b = Span::new(f1, 30, 40);
        let s_c = Span::new(f1, 50, 60);

        let mut ms = MultiSpan::new(s_c);
        ms.push(s_b); // descending → binary insert
        ms.push(s_a);
        ms.push(s_b); // middle duplicate
        ms.push(s_mid); // middle insertion
        ms.push(s_c); // duplicate of the tail-most element (pos 0 after sort)

        assert_eq!(ms.primary().as_slice(), &[s_a, s_mid, s_b, s_c]);
    }

    /// `Diagnostic::augment` merges labels without duplicating a
    /// `(span, message)` pair — on both the small allocation-free scan and
    /// the keyed-set path used when the target already carries many labels —
    /// and a label duplicated *inside* `other` is added only once.
    #[test]
    fn augment_dedups_labels_on_both_paths() {
        for n in [2usize, 20] {
            // n = 2 exercises the scan; n = 20 the keyed-set branch.
            let mut target = Diagnostic::error("target");
            let mut other = Diagnostic::error("other");
            for i in 0..n {
                let s = Span::new(FileId::DUMMY, (i * 4) as u32, (i * 4 + 2) as u32);
                target = target.with_secondary_label(s, format!("m{i}"));
                other = other.with_secondary_label(s, format!("m{i}"));
            }
            let dup = Span::new(FileId::DUMMY, 1000, 1002);
            other = other.with_secondary_label(dup, "dup");
            other = other.with_secondary_label(dup, "dup"); // duplicate within `other`
            let fresh = Span::new(FileId::DUMMY, 2000, 2002);
            other = other.with_secondary_label(fresh, "fresh");

            let before = target.labels.len();
            target.augment(&other);
            assert_eq!(
                target.labels.len(),
                before + 2,
                "n={n}: only `dup` (once) and `fresh` should be added"
            );
        }
    }

    /// `augment` with a small `self` and a large `other` must take the
    /// keyed-set path.  A threshold that only guards `self.labels.len()`
    /// would route this to the scan path, chaining the growing `to_add`
    /// on every iteration — O(|other|²) string comparisons.  This locks
    /// the both-sides threshold: the merge must stay linear and dedup
    /// correctly regardless of which side is large.
    #[test]
    fn augment_small_self_large_other_takes_keyed_path() {
        let mut target = Diagnostic::error("target");
        let s0 = Span::new(FileId::DUMMY, 0, 2);
        target = target.with_secondary_label(s0, "self0");

        let mut other = Diagnostic::error("other");
        let n = 500;
        for i in 0..n {
            let s = Span::new(FileId::DUMMY, (i * 4 + 100) as u32, (i * 4 + 102) as u32);
            other = other.with_secondary_label(s, format!("m{i}"));
        }

        target.augment(&other);
        // `self0` plus all `n` distinct labels from `other`.
        assert_eq!(target.labels.len(), 1 + n);
    }

    #[test]
    fn diagnostic_size_bounded() {
        let sz = std::mem::size_of::<Diagnostic<'static>>();
        let multi = std::mem::size_of::<MultiSpan>();
        let kind = std::mem::size_of::<DiagnosticKind<'static>>();
        println!(
            "[size] Diagnostic={sz}  MultiSpan={multi}  DiagnosticKind={kind}  \
             Box<DiagnosticKind>={}",
            std::mem::size_of::<Option<std::sync::Arc<DiagnosticKind<'static>>>>()
        );
        assert!(sz <= 416, "Diagnostic grew to {sz} bytes");
        // 416 = the historical 384-byte bound + 16 B for `MultiSpan`'s
        // sorted-view cache (`OnceLock<Box<[Span]>>`, added so a dirty
        // `MultiSpan` sorts once per dirty epoch instead of on every
        // `primary()` read) + 8 B of padding for the `kind_derived_labels`
        // bookkeeping field that makes `humanize` preserve user labels.
    }

    #[test]
    fn stable_id_cache_reflects_hashed_field_mutations() {
        let base = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 5));
        let base_id = base.stable_id();
        assert_eq!(base.stable_id(), base_id);

        let new_code = base.clone().with_code_str(ErrCode::E031);
        assert_ne!(new_code.stable_id(), base_id);

        let new_span = base.clone().with_span(Span::new(FileId::DUMMY, 20, 25));
        assert_ne!(new_span.stable_id(), base_id);

        let mut empty = Diagnostic::error("boom");
        let empty_id = empty.stable_id();
        empty.augment(&base);
        assert_ne!(empty.stable_id(), empty_id);

        let mut d = base.clone();
        d.set_level(DiagnosticLevel::Warning);
        d.set_source(Some("ignored"));
        // C9: `stable_id` hashes the file NAME, so a rename must invalidate
        // the cache and yield a new id (the old code returned the stale id,
        // reintroducing cross-file collisions).
        let before_rename = d.stable_id();
        d.set_file_name("ignored.pn");
        assert_ne!(d.stable_id(), before_rename);
        assert_ne!(d.stable_id(), base_id);

        // `with_kind` changes the eventual (humanized) message, so a cached
        // id computed before it must be invalidated, not returned stale.
        let kind_a = DiagnosticKind::NameNotFound {
            name: std::borrow::Cow::Borrowed("a"),
            span: Span::new(FileId::DUMMY, 0, 1),
            suggestions: Vec::new(),
        };
        let kind_b = DiagnosticKind::NameNotFound {
            name: std::borrow::Cow::Borrowed("b"),
            span: Span::new(FileId::DUMMY, 0, 1),
            suggestions: Vec::new(),
        };
        let mut d = base.clone().with_kind(kind_a);
        d.humanize();
        let id_a = d.stable_id();
        // `with_kind` invalidates the cache; the id only changes once the new
        // kind is materialized, which is what `humanize` does.
        d = d.with_kind(kind_b);
        d.humanize();
        assert_ne!(d.stable_id(), id_a);
    }

    /// The direct-address neighbourhood enumeration must produce the same
    /// top-3 as the full table scan for: every valid registry code (distance
    /// 0), single-edit typos of them, the corner codes E000/E999/W000/W999,
    /// cross-prefix typos, 5-digit insertions, and unrelated noise.
    #[test]
    fn suggest_code_direct_matches_scan() {
        let table_codes: Vec<&str> = crate::diagnostics::error_code::CODE_TABLE
            .iter()
            .map(|e| e.code)
            .collect();
        let mut inputs: Vec<String> = table_codes.iter().map(|c| c.to_string()).collect();
        // Lowercase spellings, corners, cross-prefix, insertions, noise.
        for c in table_codes.iter().step_by(17) {
            inputs.push(c.to_ascii_lowercase());
        }
        for c in ["E000", "E999", "W000", "W999", "E0000", "W1234", "E12345"] {
            inputs.push(c.to_string());
        }
        // Digit-corrupted typos of a sample of codes (all four lanes, plus
        // transpositions) and non-shaped noise.
        for c in table_codes.iter().step_by(5) {
            let b = c.as_bytes();
            for lane in 1..4 {
                for d in b'0'..=b'9' {
                    if d != b[lane] {
                        let mut t = b.to_vec();
                        t[lane] = d;
                        inputs.push(String::from_utf8(t).unwrap());
                    }
                }
            }
            let mut t = b.to_vec();
            t.swap(1, 2);
            inputs.push(String::from_utf8(t).unwrap());
        }
        inputs.push("????".into());
        inputs.push("E".into());
        inputs.push("ERR0123".into());
        for input in &inputs {
            // The scan reference needs the same uppercased needle.
            let mut needle = input.as_str().to_owned();
            needle.make_ascii_uppercase();
            let needle_len = needle.chars().count();
            let max_dist = std::cmp::max(1, needle_len / 3);
            let direct = suggest_top3_direct(&needle);
            let scan = suggest_top3_scan(&needle, max_dist, needle_len);
            assert_eq!(
                direct.as_slice(),
                scan.as_slice(),
                "input {input:?} (needle {needle:?})"
            );
        }
    }

    /// Plain (`use_color == false`) explain output for an unknown/typo'd code
    /// must contain no ANSI escapes — `suggest_code` used to hard-code its
    /// `\x1b[1;34m` help label into both color modes.
    #[test]
    fn plain_explain_of_unknown_code_has_no_ansi() {
        assert!(!suggest_code("E031", false).contains('\x1b'));
        assert!(suggest_code("E031", true).contains('\x1b'));
        for code in ["zzz", "E03", "????"] {
            let out = explain_error_code(code, false);
            assert!(!out.contains('\x1b'), "plain explain of {code}: {out:?}");
        }
    }

    /// `sanitize_filename` composes `sanitize_for_terminal`, which already
    /// removes bidi overrides/isolates.  Pin that composed contract: the
    /// (removed) second filter in `sanitize_filename` was unreachable, so
    /// this test is the guard if `sanitize_for_terminal` ever weakens.
    #[test]
    fn sanitize_filename_removes_bidi_overrides() {
        let spoof = "evil.code\u{202E}exe";
        assert_eq!(&*sanitize_filename(spoof), "evil.codeexe");
        // Controls are replaced by the same composed pass.
        assert_eq!(&*sanitize_filename("a\x07b"), "a\u{FFFD}b");
        // Clean ASCII stays borrowed (no extra scan cost).
        assert!(matches!(sanitize_filename("src/main.rs"), Cow::Borrowed(_)));
    }

    /// Bidi overrides/isolates must not survive any terminal sanitizer: they
    /// are invisible, so the control scans cannot see them (`evil.code\u{202E}exe`
    /// used to leak through with a spoofed extension).
    #[test]
    fn sanitizers_strip_bidi_overrides_and_isolates() {
        let spoof = "evil.code\u{202E}exe";
        for out in [
            sanitize_for_terminal(spoof).to_string(),
            sanitize_for_terminal_multiline(spoof).to_string(),
            sanitize_for_terminal_flatten(spoof).to_string(),
        ] {
            assert_eq!(out, "evil.codeexe", "spoof survived: {out:?}");
            assert!(!out.contains('\u{202E}'));
        }
        // Isolates (U+2066–U+2069) are removed too.
        let isolates = "a\u{2066}b\u{2067}c\u{2068}d\u{2069}e";
        let out = sanitize_for_terminal(isolates);
        assert_eq!(&*out, "abcde");
        // i18n text without bidi/controls still borrows.
        let clean = "тип: тип（中文）";
        assert!(matches!(
            sanitize_for_terminal(clean),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            sanitize_for_terminal_multiline(clean),
            Cow::Borrowed(_)
        ));
        // Layout controls survive their respective sanitizers alongside
        // bidi removal.
        let mixed = "a\u{202E}b\nc\td";
        assert_eq!(&*sanitize_for_terminal_multiline(mixed), "ab\nc\td");
        let mixed_cr = "a\u{202E}b\nc\rd";
        assert_eq!(&*sanitize_for_terminal_flatten(mixed_cr), "ab\nc\rd");
    }

    /// The flatten scanner excludes `\t`/`\n`/`\r` from the dirty set, so a
    /// multi-line message with no other controls stays `Cow::Borrowed` — and
    /// the SWAR flags must agree with the scalar predicate over random
    /// tab/newline/CR-heavy slices.
    #[test]
    fn flatten_scan_borrows_layout_only_controls() {
        for s in ["a\nb\rc\td", "\n\r\t", "line one\nline two\r\n\tthree"] {
            assert!(matches!(
                sanitize_for_terminal_flatten(s),
                Cow::Borrowed(_)
            ));
        }
        // Still replaced: DEL, VT, ESC, C1.
        for s in ["a\x7Fb", "a\x0Bb", "a\x1Bb", "a\u{0085}b"] {
            assert!(matches!(
                sanitize_for_terminal_flatten(s),
                Cow::Owned(_)
            ));
        }
        let reference = |s: &[u8]| {
            s.iter()
                .any(|&b| (b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r') || b == 0x7F)
        };
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_0001;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(7);
            seed
        };
        for _ in 0..10_000 {
            let len = (next() % 48) as usize;
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    let r = next() & 0xFF;
                    if r < 180 {
                        [b'\t', b'\n', b'\r'][(next() % 3) as usize]
                    } else {
                        r as u8
                    }
                })
                .collect();
            let (has_high, dirty, _has_c1) = scan_terminal_control_flags_flatten(&bytes);
            let _ = has_high;
            assert_eq!(dirty, reference(&bytes), "{bytes:?}");
        }
    }

    /// Alternating pushes and reads: the sorted-view cache must never serve a
    /// stale view, and every `primary()` result stays sorted + deduped.
    #[test]
    fn multispan_push_primary_interleave_stays_sorted() {
        let mut rng: u64 = 0xC0FF_EE00_0000_0001;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(11);
            rng
        };
        let mut ms = MultiSpan::default();
        let mut expect: Vec<Span> = Vec::new();
        for _ in 0..2_000 {
            let start = (next() % 64) as u32;
            let span = Span::new(FileId::DUMMY, start, start + 1 + (next() % 8) as u32);
            if next() & 1 == 0 {
                ms.push(span);
                if !expect.contains(&span) {
                    expect.push(span);
                }
            } else if !expect.is_empty() {
                expect.sort_unstable();
                expect.dedup();
                let mut got = ms.primary();
                got.dedup();
                assert_eq!(got.as_slice(), expect.as_slice());
                match ms.primary_ref() {
                    Cow::Borrowed(s) => assert_eq!(s, expect.as_slice()),
                    Cow::Owned(s) => assert_eq!(s, expect.as_slice()),
                }
            }
            if next() % 97 == 0 {
                ms.clear();
                expect.clear();
            }
        }
    }

    /// A kind-built diagnostic enriched with `with_label` keeps BOTH the
    /// kind-derived labels and the user label across humanize, and
    /// re-humanizing after a language change does not duplicate either.
    #[test]
    fn humanize_preserves_user_labels_alongside_kind_derived() {
        let kind_span = Span::new(FileId::DUMMY, 0, 3);
        let user_span = Span::new(FileId::DUMMY, 40, 45);
        let mut diag = Diagnostic::error("placeholder")
            .with_kind(DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("Int<32>"),
                found: Cow::Borrowed("Str"),
                span: kind_span,
                found_span: None,
                reason: None,
                context: None,
            })
            .with_label(user_span, "user context");
        diag.humanize();
        assert_eq!(diag.labels.len(), 2, "kind + user label");
        assert_eq!(diag.labels[0].span, kind_span);
        assert_eq!(diag.labels[1].span, user_span);
        // Re-humanize (latch cleared, same language): no duplication, order
        // stable — the derived block is rebuilt, the user label kept.
        diag.humanized = false;
        diag.humanize();
        assert_eq!(diag.labels.len(), 2);
        assert_eq!(diag.labels[0].span, kind_span);
        assert_eq!(diag.labels[1].span, user_span);
    }

    /// Regression: a closure that pushes a *suppressed* warning and returns
    /// `Err` must not corrupt an earlier, unrelated diagnostic via
    /// `last_mut()`.
    #[test]
    fn adorn_with_span_does_not_augment_unrelated_prior_diagnostic() {
        use crate::diagnostics::EmissionGuaranteed;
        let mut ctxt = DiagCtxt::new();
        ctxt.set_can_emit_warnings(false);
        // Pre-existing unreported diagnostic — `last_mut()` would return this
        // if the guard were missing.
        ctxt.push(Diagnostic::error("prior").with_span(Span::new(FileId::DUMMY, 100, 105)));
        let result: Result<(), EmissionGuaranteed> = adorn_with_span(
            &mut ctxt,
            Span::new(FileId::DUMMY, 0, 5),
            Some("hello"),
            |ctxt| {
                // Suppressed (records nothing), then fail without pushing.
                Err(ctxt.push(Diagnostic::warning("suppressed")).into())
            },
        );
        assert!(result.is_err());
        let prior = &ctxt.into_inner()[0];
        assert_eq!(
            prior.spans.primary().as_slice(),
            &[Span::new(FileId::DUMMY, 100, 105)][..],
            "the prior diagnostic's span must be untouched"
        );
        assert_eq!(prior.source, None);
    }
}
