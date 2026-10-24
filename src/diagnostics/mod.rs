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
        // Values are lifted into `Cow<'static, str>` via `TrValue`: a
        // `&'static str`, `Cow<'static, str>` or `String` argument converts
        // without re-formatting; any other `Display` value must be wrapped in
        // `DisplayArg(...)` so the allocation is explicit at the call site.
        let pairs: $crate::smallvec::SmallVec<[(&'static str, std::borrow::Cow<'static, str>); 4]> = {
            let mut v: $crate::smallvec::SmallVec<[(&'static str, std::borrow::Cow<'static, str>); 4]> =
                $crate::smallvec::smallvec![];
            $(
                $(
                    v.push((
                        stringify!($key),
                        $crate::diagnostics::i18n::TrValue::into_cow_static($val),
                    ));
                )+
            )?
            v
        };
        $crate::diagnostics::i18n::render($msg, &pairs)
    }};
}

pub mod chain;
pub mod collector;
pub mod emitter;
pub mod error_code;
pub mod explain_server;
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
pub use glyph::GlyphRenderer;
pub use i18n::DisplayArg;
pub use kind::{ComptimeReason, DiagnosticKind, Humanizer, TypeCtx};
pub use label::{AnnotationKind, Label, Snippet, SourcePos};
pub use level::DiagnosticLevel;
pub use render::{RenderCtx, Renderer};
pub use renderers::{HtmlRenderer, HumanRenderer, JsonRenderer};
pub use sink::{DiagnosticSink, MakeSink, StringSink};

use crate::ast::FileId;
use crate::ast::Span;
use arrayvec::ArrayVec;
use covariant_vec::CovariantVec;
use crate::smallvec::{SmallVec, smallvec};
use std::borrow::Cow;
use std::fmt;
use std::io;
use std::sync::Arc;
use std::sync::OnceLock;

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
        }
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
                out.push_str(&crate::diagnostics::i18n::render(slug, pairs))
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
    // Invariant: sorted by `(file, start, end)` ascending (`Span`'s `Ord`),
    // no duplicates. All mutation goes through `push`, which preserves this.
    primary: SmallVec<[Span; 2]>,
}

impl MultiSpan {
    pub fn new(span: Span) -> Self {
        MultiSpan {
            primary: smallvec![span],
        }
    }

    /// Insert a primary span, keeping the vector sorted by `(file, start)`.
    /// Duplicate spans are ignored (idempotent — safe for repeated `humanize`).
    ///
    /// Compilers emit spans in ascending source order, so the fast path
    /// compares against the tail and appends in O(1).  Only an
    /// out-of-order span pays the binary search + insert; a fully
    /// reverse-sorted push stream is the sole remaining O(n²) shape.
    pub fn push(&mut self, span: Span) {
        match self.primary.last() {
            Some(last) if *last == span => {}
            Some(last) if *last > span => {
                let pos = self.primary.partition_point(|s| s < &span);
                if self.primary.get(pos) != Some(&span) {
                    self.primary.insert(pos, span);
                }
            }
            _ => self.primary.push(span),
        }
    }

    pub fn first(&self) -> Option<Span> {
        self.primary.first().copied()
    }

    /// Clear all primary spans.
    pub fn clear(&mut self) {
        self.primary.clear();
    }

    /// Return the spans in sorted order.
    pub fn primary(&self) -> &[Span] {
        debug_assert!(
            self.primary.windows(2).all(|w| w[0] <= w[1]),
            "MultiSpan::primary must be sorted by (file, start)"
        );
        &self.primary
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionStyle {
    /// Show the suggestion inline with the diagnostic.
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

impl Default for SuggestionStyle {
    fn default() -> Self {
        SuggestionStyle::ShowAlways
    }
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
#[derive(Debug, Clone)]
pub struct Diagnostic<'src> {
    level: DiagnosticLevel,
    message: Cow<'src, str>,
    code: Option<ErrCode>,
    /// Structured error kind (Vale-style ADT).  When present, the humanizer
    /// can derive `message`, `labels`, and other fields automatically.
    kind: Option<Box<DiagnosticKind<'src>>>,
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
    /// Whether `spans` were set by a user-facing builder rather than left
    /// at the constructor default.  `humanize()` leaves them alone in that
    /// case instead of deriving fresh primary spans from the kind's labels.
    spans_user_set: bool,
    /// Whether [`humanize`](Self::humanize) has already populated `message`
    /// and `labels` from `kind`. Guards against redundant re-rendering on a
    /// second call (the operation is idempotent, but not free).
    humanized: bool,
}

impl<'src> Diagnostic<'src> {
    // ── Read-only accessors ─────────────────────────────────────────
    pub fn level(&self) -> DiagnosticLevel {
        self.level
    }
    pub fn message(&self) -> &str {
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
        self.help.as_deref()
    }
    pub fn suggestions(&self) -> &[Suggestion<'src>] {
        &self.suggestions
    }
    pub fn labels(&self) -> &[Label<'src>] {
        &self.labels
    }
    pub fn call_chain(&self) -> Option<&CallChain<'src>> {
        self.call_chain.as_ref()
    }
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
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
    }
}

/// A child diagnostic attached to a parent `Diagnostic`.
/// Inspired by rustc's `Subdiag` — used for notes, help, or secondary
/// labels that are logically part of the same diagnostic group.
#[derive(Debug, Clone)]
pub struct Subdiag<'src> {
    pub level: DiagnosticLevel,
    pub message: Cow<'src, str>,
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
            message: message.into(),
            styled_message: None,
            span: None,
            labels: Vec::new(),
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
            humanized: false,
            spans_user_set: false,
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
        diag.kind = Some(Box::new(kind));
        diag
    }

    /// Create a fatal diagnostic from a structured [`DiagnosticKind`].
    ///
    /// See [`error_kind`](Self::error_kind) for the lazy-humanization
    /// contract: derived fields stay unfilled until `humanize()` runs.
    pub fn fatal_kind(kind: DiagnosticKind<'src>) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Fatal, "");
        diag.kind = Some(Box::new(kind));
        diag
    }

    /// Create a warning diagnostic from a structured [`DiagnosticKind`].
    ///
    /// See [`error_kind`](Self::error_kind) for the lazy-humanization
    /// contract: derived fields stay unfilled until `humanize()` runs.
    pub fn warning_kind(kind: DiagnosticKind<'src>) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Warning, "");
        diag.kind = Some(Box::new(kind));
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
        self
    }

    /// Add a secondary primary span (for multi-location diagnostics).
    ///
    /// Like [`with_span`](Self::with_span), this survives `humanize`.
    pub fn with_additional_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self.spans_user_set = true;
        self
    }

    /// Set the error code (e.g. `ErrCode::TypeMismatch` → `E030`).
    pub fn with_code(mut self, code: ErrCode) -> Self {
        self.code = Some(code);
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
        self
    }

    /// Add a help message.
    pub fn with_help(mut self, help: impl Into<Cow<'src, str>>) -> Self {
        self.help = Some(help.into());
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
        self
    }

    /// Replace suggestions list.
    pub fn with_suggestions(mut self, suggestions: Vec<Suggestion<'src>>) -> Self {
        self.suggestions = suggestions.into();
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
        self
    }

    /// Set the structured diagnostic kind (Vale-style ADT).
    ///
    /// When present, the humanizer can derive `message`, `labels`, and other
    /// fields from the kind's structured data, producing more precise error
    /// messages than a flat string.
    pub fn with_kind(mut self, kind: DiagnosticKind<'src>) -> Self {
        self.kind = Some(Box::new(kind));
        // A new kind must be materialized by the next `humanize()`.
        self.humanized = false;
        self
    }

    /// Add a highlighted note child (styled text with type/name highlights).
    pub fn with_highlighted_note(mut self, styled: StyledString<'src>) -> Self {
        let msg = styled.to_plain();
        self.children.push(Subdiag {
            level: DiagnosticLevel::Note,
            message: msg.into(),
            styled_message: Some(styled),
            span: None,
            labels: Vec::new(),
        });
        self
    }

    /// Add a highlighted help child (styled text with type/name highlights).
    pub fn with_highlighted_help(mut self, styled: StyledString<'src>) -> Self {
        let msg = styled.to_plain();
        self.children.push(Subdiag {
            level: DiagnosticLevel::Help,
            message: msg.into(),
            styled_message: Some(styled),
            span: None,
            labels: Vec::new(),
        });
        self
    }

    /// Run the humanizer on `self.kind` (if present) to populate `message`,
    /// `labels`, `help`, and `suggestions` from the structured data.
    ///
    /// Idempotent: a `None` kind latches `humanized = true` and a second
    /// call short-circuits, so repeated calls (e.g. the emitter rendering
    /// the same batch twice) are free.  `with_kind` resets the latch so a
    /// re-kind'ed diagnostic is re-derived.
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
    ///
    /// The latch is keyed to `kind`, not language: once humanized, changing
    /// the process language does not re-resolve the already-rendered strings;
    /// [`with_kind`](Self::with_kind) resets the latch (see its docs).
    pub fn humanize(&mut self) {
        if self.humanized {
            return;
        }
        let Some(ref kind) = self.kind else {
            // No kind: message/labels were set manually.  Latch so later
            // calls skip the `kind` lookup entirely.
            self.humanized = true;
            return;
        };
        // Static messages move straight into `message`: a borrowed static
        // stays borrowed (zero allocation), an owned one is moved — only
        // the Fluent-slug path needs the render buffer.
        self.message = match kind.message() {
            DiagMessage::Static(cow) => cow,
            slug => {
                let mut buf = String::new();
                slug.render_into(&mut buf);
                Cow::Owned(buf)
            }
        };
        let labels = kind.labels();
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
        self.labels = labels.into();
        // help/suggestions are re-derived from `kind` on every humanize: a
        // diagnostic can be re-kind'ed (e.g. `with_kind(k1).humanize()` then
        // `with_kind(k2).humanize()`), and guarded rebuilding would keep
        // k1's derived notes on a diagnostic that no longer is k1.
        self.help = kind.help();
        self.suggestions = kind
            .suggestions()
            .into_iter()
            .map(|s| Suggestion {
                message: s.into(),
                applicability: Applicability::MachineApplicable,
                style: SuggestionStyle::ShowAlways,
                span: None,
                replacement: None,
            })
            .collect();
        self.humanized = true;
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
        }
        if self.source.is_none() {
            self.source = other.source;
        }
        if self.call_chain.is_none() {
            self.call_chain.clone_from(&other.call_chain);
        }
        // Merge labels that aren't already present.  The straightforward
        // scan is O(self × other) in the label counts; a large merge builds
        // a key set once instead, so a diagnostic carrying hundreds of
        // labels does not make this quadratic.  Small merges keep the
        // allocation-free scan (the common case is one or two labels).
        // `to_add` collects first: the key set borrows `self.labels`, so it
        // must be dropped before the pushes.
        if !other.labels.is_empty() {
            let mut to_add: SmallVec<[&Label<'src>; 4]> = SmallVec::new();
            if self.labels.len() <= 8 {
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
/// path; augmentation is applied only when a diagnostic is actually present.
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
    let result = f(ctxt);
    // If the closure failed, augment the last diagnostic with context.
    if result.is_err() {
        // The closure may have returned `Err` from a *suppressed* warning
        // push (which records nothing), so only augment when a diagnostic
        // is actually present.
        if let Some(diag) = ctxt.last_mut() {
            if diag.spans.first().is_none() {
                diag.spans = MultiSpan::new(span);
                diag.spans_user_set = true;
            }
            if diag.source.is_none() {
                diag.source = source;
            }
        }
    }
    result
}

// ── ErrCode is now a string-based code — use `ErrCode::new("E030")` ──
// The old `from_str` method with enum variants was removed in favour of
// the lookup table in `error_code.rs`.  Any string is accepted as a code.

// ── Display ───────────────────────────────────────────────────────

impl fmt::Display for Diagnostic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        glyph::DEFAULT_TAB_WIDTH,
    )
}

/// Like [`explain_error_code`] but with an explicit colour depth and tab
/// width, so the caller can share a probed [`term::ColorDepth`] and a
/// configured tab width across multiple calls.
pub fn explain_error_code_with_depth(
    code_str: &str,
    use_color: bool,
    color_depth: term::ColorDepth,
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
            let unknown_msg = crate::tr!(
                "unknown-error-code",
                code = DisplayArg(sanitize_for_terminal(code_str))
            );
            return if use_color {
                format!(
                    "\x1b[1;31m{error_label}\x1b[0m\x1b[2m[E000]\x1b[0m {unknown_msg}\n\
                     {}",
                    suggest_code(code_str),
                )
            } else {
                format!(
                    "{error_label}[E000] {unknown_msg}\n{}",
                    suggest_code(code_str),
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
            for seg in crate::diagnostics::wrap::wrap_optimal(line, DEFAULT_MAX_WIDTH, tab_width) {
                let _ = writeln!(out, "{seg}");
            }
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
    // lexicographically.
    codes.sort_unstable_by_key(|c| {
        (
            c.as_bytes().first().copied().unwrap_or(b'?'),
            c.get(1..).and_then(|s| s.parse::<u32>().ok()).unwrap_or(u32::MAX),
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
fn suggest_code(input: &str) -> String {
    // Normalise case so `e030` matches the canonical `E030`.
    let needle = input.to_ascii_uppercase();
    let needle_len = needle.chars().count();
    // Longer inputs tolerate a slightly wider search radius; short ones
    // stay tight so we don't suggest unrelated codes.
    let max_dist = std::cmp::max(1, needle_len / 3);

    let mut scored: Vec<(usize, usize, &str)> = crate::diagnostics::error_code::CODE_TABLE
        .iter()
        // Length prefilter: each edit changes length by at most one, so
        // |Δlen| is a lower bound on the distance — an entry whose code
        // length differs by more than the radius can never match, and its
        // DP is skipped.  The needle uses a char count (not byte length) so
        // the bound stays exact for non-ASCII typos; every table code is
        // validated to `^[EW]\d{3}$`, so `e.code.len()` is already its char
        // count (constant 4) and skips the per-code char walk.
        .filter(|e| needle_len.abs_diff(e.code.len()) <= max_dist)
        .map(|e| {
            let dist = osa_distance(&needle, e.code);
            let prefix = common_prefix_len(&needle, e.code);
            (dist, prefix, e.code)
        })
        .filter(|(dist, _, _)| *dist <= max_dist)
        .collect();

    // Closest first; longer shared prefix wins ties; alphabetical for stability.
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.2.cmp(b.2))
    });

    let candidates: ArrayVec<&str, 3> = scored.iter().take(3).map(|(_, _, code)| *code).collect();

    if candidates.is_empty() {
        format!(
            "\x1b[1;34m{help_label}\x1b[0m: {valid_codes} — \n       {hint}",
            help_label = DiagnosticLevel::Help.label(),
            valid_codes = crate::tr!(
                "help-valid-codes",
                codes = DisplayArg(known_code_ranges())
            ),
            hint = crate::tr!("run-explain-hint-example"),
        )
    } else {
        format!(
            "\x1b[1;34m{help_label}\x1b[0m: {did_you_mean}\n       {hint}",
            help_label = DiagnosticLevel::Help.label(),
            did_you_mean = crate::tr!(
                "did-you-mean",
                candidates = DisplayArg(candidates.join("` or `"))
            ),
            hint = crate::tr!("run-explain-hint-detail"),
        )
    }
}

/// List all available error codes with their titles, formatted for terminal output.
pub fn list_error_codes() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "\x1b[1m{}\x1b[0m\n", crate::tr!("available-error-codes"));
    let mut last_category = None;
    for entry in crate::diagnostics::error_code::CODE_TABLE {
        let cat = entry.category;
        if Some(cat) != last_category {
            let _ = writeln!(out, " \x1b[36m── {} ──\x1b[0m", cat.label(),);
            last_category = Some(cat);
        }
        let _ = writeln!(out, "  \x1b[1m{}\x1b[0m  {}", entry.code, entry.title,);
    }
    let _ = writeln!(
        out,
        "\n\x1b[2m{}\x1b[0m",
        crate::tr!("run-explain-detail")
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
    let mut emitter = Emitter::new(renderer, sink::stderr(), opts);

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
    let mut emitter = Emitter::new(renderer, sink::stderr(), opts);
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
    let mut emitter = Emitter::new(renderer, sink::stderr(), opts);
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

/// Replace control characters with U+FFFD so external text cannot inject
/// terminal escapes when echoed to the terminal.
///
/// Single source of truth — previously duplicated in `glyph`
/// (`filename_display`) and `error_code`.  Applied at every boundary that
/// writes untrusted text into ANSI output (OSC 8 hyperlink text, SGR colour
/// runs), where a raw `\x1b` or `\x07` would terminate the sequence early and
/// let the remaining bytes act as terminal commands.
pub(crate) fn sanitize_for_terminal(s: &str) -> Cow<'_, str> {
    // Fast path: ASCII text without control chars is already safe for terminal
    // use (no raw ESC/BEL can terminate SGR or OSC sequences early).  Must
    // check `is_ascii` first — a string containing only UTF-8 C1 sequences
    // (e.g. `\u{85}` encoded as `0xC2 0x85`) passes the bare byte check since
    // each raw byte is ≥ 0x20 and ≠ 0x7F, but those bytes are not printable.
    if s.is_ascii() && s.bytes().all(|b| b >= 0x20 && b != 0x7f) {
        return Cow::Borrowed(s);
    }
    Cow::Owned(
        s.chars()
            .map(|c| if c.is_control() { '\u{FFFD}' } else { c })
            .collect(),
    )
}

/// Unicode bidirectional formatting characters: the U+202A–U+202E
/// overrides and the U+2066–U+2069 isolates.  Invisible, but they reorder
/// the surrounding display text — a path like `evil.cod\u{202E}exe` can
/// render with a spoofed extension.
pub(crate) fn is_bidi_override(c: char) -> bool {
    ('\u{202A}'..='\u{202E}').contains(&c) || ('\u{2066}'..='\u{2069}').contains(&c)
}

/// Sanitize a filename for terminal display: [`sanitize_for_terminal`],
/// plus bidirectional formatting characters removed outright.  Unlike
/// control characters they carry no content — only rendering direction —
/// so replacement with U+FFFD would just inject noise into the path;
/// removal kills the spoofing.
pub(crate) fn sanitize_filename(s: &str) -> Cow<'_, str> {
    let base = sanitize_for_terminal(s);
    if !base.chars().any(is_bidi_override) {
        return base;
    }
    Cow::Owned(base.chars().filter(|c| !is_bidi_override(*c)).collect())
}

/// Multi-line variant of [`sanitize_for_terminal`] for text whose `\n`/`\t`
/// are layout intent (user messages, help, notes) rather than raw terminal
/// bytes: keeps them verbatim and replaces every other control character —
/// CR and the C1 range included — with U+FFFD.  Same ASCII fast path.
pub(crate) fn sanitize_for_terminal_multiline(s: &str) -> Cow<'_, str> {
    if s.is_ascii() {
        let safe = s
            .bytes()
            .all(|b| (b >= 0x20 && b != 0x7f) || b == b'\n' || b == b'\t');
        if safe {
            return Cow::Borrowed(s);
        }
    }
    Cow::Owned(
        s.chars()
            .map(|c| {
                if c.is_control() && c != '\n' && c != '\t' {
                    '\u{FFFD}'
                } else {
                    c
                }
            })
            .collect(),
    )
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
    starts: Arc<[usize]>,
    source: &'a str,
}

impl<'a> LineIndex<'a> {
    pub(crate) fn new(source: &'a str) -> Self {
        let mut starts = Vec::with_capacity(source.len() / 40 + 1);
        starts.push(0);
        for i in memchr::memchr_iter(b'\n', source.as_bytes()) {
            starts.push(i + 1);
        }
        Self {
            starts: Arc::from(starts),
            source,
        }
    }

    /// Build an index for `source` from a table already computed for that same
    /// source (shared by [`RenderCtx::line_index`](crate::diagnostics::render::RenderCtx::line_index)).
    /// The caller guarantees `starts` describes `source`'s newlines.
    pub(crate) fn from_shared(source: &'a str, starts: Arc<[usize]>) -> Self {
        Self { starts, source }
    }

    /// The shared line-start table, for caching by an O(1) `Arc` clone.
    pub(crate) fn starts_arc(&self) -> &Arc<[usize]> {
        &self.starts
    }

    /// 0-based line containing `offset` (clamped to the source length).
    pub(crate) fn line_of(&self, offset: usize) -> usize {
        match self.starts.binary_search(&offset.min(self.source.len())) {
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
            col: clamped - self.starts[line],
        }
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

    pub(crate) fn starts(&self) -> &[usize] {
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
        let last = self.starts.last().copied().unwrap_or(0);
        if last == self.source.len() && self.source.ends_with('\n') {
            self.starts.len() - 1
        } else {
            self.starts.len()
        }
    }

    /// Return the content of line `line_idx` (0-based), with the trailing `\n` / `\r\n` removed.
    pub(crate) fn line(&self, line_idx: usize) -> &'a str {
        let start = self.starts[line_idx];
        let end = self
            .starts
            .get(line_idx + 1)
            .copied()
            .unwrap_or(self.source.len());
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
            .with_code(ErrCode::new("E030"))
            .with_span(Span::new(FileId::DUMMY, 5, 12))
            .with_suggestion("try using `as` to cast");
        assert!(diag.is_error());
        assert_eq!(diag.code.as_ref().unwrap().code(), "E030");
    }

    #[test]
    fn test_with_labels() {
        let diag = Diagnostic::error("no field `z` on type Point")
            .with_code(ErrCode::new("E010"))
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

    #[test]
    fn test_suggest_code_osa_distance() {
        // Single substituted digit — same-length typo the old heuristic missed.
        let s = suggest_code("E031");
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Missing trailing digit.
        let s = suggest_code("E03");
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Lowercase typo still matches the canonical code.
        let s = suggest_code("e030");
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Transposed digits: `E300`↔`E030` is OSA-distance 1, so the
        // radius-1 scan must now suggest `E030` (plain Levenshtein scored 2
        // and silently dropped it).  `E300` is not a known code, so this
        // exercises the real unknown-code suggestion path.
        let s = suggest_code("E300");
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Complete garbage — no suggestions, fall back to generic help.
        let s = suggest_code("????");
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
        chain.push(chain::ChainEntry {
            kind: chain::ChainEntryKind::Call,
            span: Some(Span::new(FileId::DUMMY, 80, 90)),
            message: "main".into(),
        });
        let diag = Diagnostic::error("type mismatch")
            .with_code(ErrCode::new("E030"))
            .with_call_chain(chain);
        assert!(diag.call_chain.is_some());
        assert!(diag.to_string().contains("referenced by"));
    }

    #[test]
    fn test_from_string_code() {
        let diag = Diagnostic::error("test").with_code_str("E030");
        assert_eq!(diag.code.unwrap().code(), "E030");
    }

    #[test]
    fn test_diagnostic_collector() {
        let mut collector = DiagCtxt::new();
        collector.push(Diagnostic::error("first error").with_code(ErrCode::new("E030")));
        collector.push(Diagnostic::warning("first warning"));
        assert_eq!(collector.len(), 2);
        assert_eq!(collector.error_count(), 1);
        assert_eq!(collector.warning_count(), 1);
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

    #[test]
    fn test_error_category() {
        assert_eq!(ErrCode::new("E030").category(), ErrorCategory::Type);
        assert_eq!(ErrCode::new("E001").category(), ErrorCategory::Parse);
        assert_eq!(ErrCode::new("E020").category(), ErrorCategory::Contract);
        assert_eq!(ErrCode::new("E041").category(), ErrorCategory::Trait);
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
        ctxt.can_emit_warnings = false;
        let result: Result<(), EmissionGuaranteed> =
            adorn_with_span(&mut ctxt, Span::new(FileId::DUMMY, 0, 5), Some("hello"), |ctxt| {
                // `push` returns a *suppressed* guarantee here and records
                // nothing, so the old debug_assert would have fired.
                Err(ctxt.push(Diagnostic::warning("suppressed")).into())
            });
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
                    .push(
                        Diagnostic::error("placeholder").with_kind(DiagnosticKind::TypeMismatch {
                            expected: Cow::Borrowed("Int<32>"),
                            found: Cow::Borrowed("Str"),
                            span: kind_span,
                            found_span: None,
                            reason: None,
                            context: None,
                        }),
                    )
                    .into())
            });
        assert!(result.is_err());
        for diag in ctxt.iter_mut() {
            diag.humanize();
            assert_eq!(diag.spans.primary(), &[adorn_span][..]);
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
        // Regression: `with_code_str("E067")` in the PANN pattern-annotation
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
            diag.message.contains("no field `z` on type `Point`"),
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
        assert_eq!(diag.spans.primary(), &[user_span][..]);
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
        assert_eq!(diag.spans.primary(), &[user1, user2][..]);
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
        assert_eq!(diag.spans.primary(), &[kind_span][..]);
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
            let over = Span::new(
                FileId::DUMMY,
                0,
                (src.len() as u32).saturating_add(8),
            );
            let (first, last) = idx.span_lines(over);
            assert!(first <= last && last < idx.line_count(), "over {src:?}");
            for i in 0..idx.line_count() {
                let pos = idx.to_linecol(idx.starts()[i]);
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
            diag = diag.with_secondary_label(Span::new(FileId::DUMMY, s, s + 10), format!("label {i}"));
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
            diag = diag.with_additional_span(Span::new(FileId::DUMMY, line_off[l], line_off[l] + 5));
        }
        for i in 0..200 {
            let l = (i * 25) % 5_000;
            diag = diag
                .with_secondary_label(Span::new(FileId::DUMMY, line_off[l], line_off[l] + 4), format!("note {i}"));
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

        assert_eq!(ms.primary(), &[s_d, s_a, s_b, s_c]);
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

        assert_eq!(ms.primary(), &[s_a, s_mid, s_b, s_c]);
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

    #[test]
    fn diagnostic_size_bounded() {
        let sz = std::mem::size_of::<Diagnostic<'static>>();
        let multi = std::mem::size_of::<MultiSpan>();
        let kind = std::mem::size_of::<DiagnosticKind<'static>>();
        println!(
            "[size] Diagnostic={sz}  MultiSpan={multi}  DiagnosticKind={kind}  \
             Box<DiagnosticKind>={}",
            std::mem::size_of::<Option<Box<DiagnosticKind<'static>>>>()
        );
        assert!(sz < 384, "Diagnostic grew to {sz} bytes");
    }
}
