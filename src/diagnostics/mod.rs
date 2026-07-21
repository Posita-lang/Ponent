//! Ponent diagnostic system.
//!
//! A comprehensive, multi-format diagnostic system modeled after:
//! - **rustc**: error codes (`E030`), `^`-underline annotations, `--explain`, suggestions
//! - **Zig**: `referenced by` call chains
//! - **Austral**: multi-format output (plain / JSON / HTML), error type classification
//! - **Vale**: per-pass error humanizers, structured error types

// ── i18n: translation macro ────────────────────────────────────
//
// Looks up the format string in the current language's message table,
// then substitutes `{param}` placeholders with the given values.
// Placeholders are sorted by length (longest first) to avoid
// overlapping key name issues (e.g. `{name}` vs `{name_with_underscore}`).
//
// Usage:
//   tr!("type mismatch: expected `{expected}`, found `{found}",
//       expected = "Int<32>", found = "Bool")
#[macro_export]
macro_rules! tr {
    ($msg:literal $(, $($key:ident = $val:expr),+ $(,)?)?) => {{
        let template = $crate::diagnostics::i18n::lookup($msg);
        // If the lookup returned the fallback placeholder, use the original
        // format string instead — it's more informative than "(untitled)".
        // This happens when a message key hasn't been added to the i18n table.
        let template = if template == "(untitled)" { $msg } else { template };
        let mut result = template.to_string();
        // Collect placeholders, sort by length (longest first) to avoid
        // overlapping key name issues (e.g. `{name}` vs `{name_with_underscore}`).
        {
            let mut pairs: Vec<(String, String)> = Vec::new();
            $(
                $(
                    pairs.push((format!("{{{}}}", stringify!($key)), format!("{}", $val)));
                )+
            )?
            pairs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
            for (placeholder, val) in pairs {
                result = result.replace(&placeholder, &val);
            }
        }
        result
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

pub use chain::CallChain;
pub use collector::DiagCtxt;
pub use emitter::{
    DiagnosticEmitter, colored::ColoredEmitter, html::HtmlEmitter, json::JsonEmitter,
    plain::PlainEmitter,
};
pub use error_code::{ErrorCategory, ErrCode};
pub use glyph::GlyphRenderer;
pub use kind::{DiagnosticKind, Humanizer};
pub use label::{AnnotationKind, Label, Snippet, SourcePos};
pub use level::DiagnosticLevel;

use crate::ast::Span;
use std::fmt;

// ── Diagnostic options (inspired by GHC's DiagOpts) ────────────────

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
    /// Reverse the order in which errors are reported (most recent first).
    pub reverse_errors: bool,
}

impl Default for DiagOpts {
    fn default() -> Self {
        DiagOpts {
            max_errors: None,
            warn_is_error: false,
            context_lines: 2,
            use_color: true,
            reverse_errors: false,
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
    pub fn emitted() -> Self {
        EmissionGuarantee { _private: (), emitted: true }
    }

    /// The diagnostic was suppressed (e.g. warning when warnings are off).
    pub fn suppressed() -> Self {
        EmissionGuarantee { _private: (), emitted: false }
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

/// A lazily-formatted diagnostic message, analogous to rustc's `DiagMessage`.
/// Currently a simple wrapper around `String`; future versions may support
/// delayed formatting with arguments.
#[derive(Debug, Clone)]
pub struct DiagMessage {
    inner: String,
}

impl DiagMessage {
    pub fn new(msg: impl Into<String>) -> Self {
        DiagMessage { inner: msg.into() }
    }

    pub fn into_string(self) -> String {
        self.inner
    }

    pub fn as_str(&self) -> &str {
        &self.inner
    }
}

impl From<String> for DiagMessage {
    fn from(s: String) -> Self {
        DiagMessage::new(s)
    }
}

impl From<&str> for DiagMessage {
    fn from(s: &str) -> Self {
        DiagMessage::new(s)
    }
}

/// A set of spans that a diagnostic points to, supporting multiple primary
/// locations.  Analogous to rustc's `MultiSpan`.  When there is a single
/// primary span, `primary` contains one element; for multiple (e.g. a
/// duplicate definition + the previous definition), both are primary.
#[derive(Debug, Clone, Default)]
pub struct MultiSpan {
    pub primary: Vec<Span>,
}

impl MultiSpan {
    pub fn new(span: Span) -> Self {
        MultiSpan {
            primary: vec![span],
        }
    }

    pub fn push(&mut self, span: Span) {
        self.primary.push(span);
    }

    pub fn first(&self) -> Option<Span> {
        self.primary.first().copied()
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
    fn default() -> Self {
        Applicability::MachineApplicable
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

/// A structured suggestion with confidence and style metadata.
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub message: String,
    pub applicability: Applicability,
    pub style: SuggestionStyle,
}

impl std::fmt::Display for Suggestion {
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
pub struct StringPart {
    pub content: String,
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
pub struct StyledString(pub Vec<StringPart>);

impl StyledString {
    pub fn new(parts: Vec<StringPart>) -> Self {
        StyledString(parts)
    }

    pub fn plain(text: impl Into<String>) -> Self {
        StyledString(vec![StringPart {
            content: text.into(),
            style: StringPartStyle::Normal,
        }])
    }

    pub fn highlighted(text: impl Into<String>) -> Self {
        StyledString(vec![StringPart {
            content: text.into(),
            style: StringPartStyle::Highlighted,
        }])
    }

    /// Render the styled string as plain text (without ANSI codes).
    pub fn to_plain(&self) -> String {
        self.0.iter().map(|p| p.content.as_str()).collect()
    }
}

/// A complete compiler diagnostic (error, warning, help, note, or info).
///
/// Architecture inspired by multiple compilers:
/// - **rustc**: Error codes + `^`-underline annotations + suggestions + sub-diagnostics
/// - **Zig**: `referenced by` call chains for traceability
/// - **Austral**: Error type categorization + multi-format output
/// - **Vale**: Structured error variants per compiler pass
#[derive(Debug, Clone)]
pub struct Diagnostic {
    level: DiagnosticLevel,
    message: String,
    code: Option<ErrCode>,
    /// Structured error kind (Vale-style ADT).  When present, the humanizer
    /// can derive `message`, `labels`, and other fields automatically.
    kind: Option<DiagnosticKind>,
    /// Primary spans (multiple for multi-location diagnostics).
    spans: MultiSpan,
    help: Option<String>,
    /// Structured suggestions with applicability and style metadata.
    suggestions: Vec<Suggestion>,
    /// Multi-span annotations for source-context rendering
    labels: Vec<Label>,
    /// Zig-style call chain tracing how the error was reached
    call_chain: Option<CallChain>,
    /// Retained source text for `^`-underline rendering
    source: Option<String>,
    /// Source file name displayed in the `┌─ <input>:1:1` location header.
    file_name: String,
    /// Child diagnostics (notes, help messages attached to this diagnostic).
    children: Vec<Subdiag>,
    /// Related errors displayed alongside this diagnostic in a single merged box.
    related_errors: Vec<RelatedError>,
}

impl Diagnostic {
    // ── Read-only accessors ─────────────────────────────────────────
    pub fn level(&self) -> DiagnosticLevel { self.level }
    pub fn message(&self) -> &str { &self.message }
    pub fn code(&self) -> Option<&ErrCode> { self.code.as_ref() }
    pub fn kind(&self) -> Option<&DiagnosticKind> { self.kind.as_ref() }
    pub fn spans(&self) -> &MultiSpan { &self.spans }
    pub fn help(&self) -> Option<&str> { self.help.as_deref() }
    pub fn suggestions(&self) -> &[Suggestion] { &self.suggestions }
    pub fn labels(&self) -> &[Label] { &self.labels }
    pub fn call_chain(&self) -> Option<&CallChain> { self.call_chain.as_ref() }
    pub fn source(&self) -> Option<&str> { self.source.as_deref() }
    pub fn file_name(&self) -> &str { &self.file_name }
    pub fn children(&self) -> &[Subdiag] { &self.children }
    pub fn children_mut(&mut self) -> &mut Vec<Subdiag> { &mut self.children }
    pub fn related_errors(&self) -> &[RelatedError] { &self.related_errors }

    // ── Mutable accessors ───────────────────────────────────────────
    pub fn labels_mut(&mut self) -> &mut Vec<Label> { &mut self.labels }
    pub fn related_errors_mut(&mut self) -> &mut Vec<RelatedError> { &mut self.related_errors }
    pub fn set_source(&mut self, source: Option<String>) { self.source = source; }
    pub fn set_file_name(&mut self, file_name: String) { self.file_name = file_name; }
}

/// A child diagnostic attached to a parent `Diagnostic`.
/// Inspired by rustc's `Subdiag` — used for notes, help, or secondary
/// labels that are logically part of the same diagnostic group.
#[derive(Debug, Clone)]
pub struct Subdiag {
    pub level: DiagnosticLevel,
    pub message: String,
    /// Optional styled version of the message, with highlighted parts.
    /// When present, the glyph renderer uses this instead of the plain
    /// `message` to render rich text with color highlights.
    pub styled_message: Option<StyledString>,
    pub span: Option<Span>,
    pub labels: Vec<Label>,
}

/// A related error displayed alongside the primary diagnostic in a single
/// merged box.  Each related error has its own error code, message, and
/// optional source span — for example, a type mismatch [E030] aggregated
/// into a duplicate definition [E019] diagnostic.
#[derive(Debug, Clone)]
pub struct RelatedError {
    pub code: Option<ErrCode>,
    pub message: String,
    pub span: Option<Span>,
    pub label: Option<String>,
}

impl Subdiag {
    /// Create a new sub-diagnostic.
    pub fn new(level: DiagnosticLevel, message: impl Into<String>) -> Self {
        Subdiag {
            level,
            message: message.into(),
            styled_message: None,
            span: None,
            labels: Vec::new(),
        }
    }

    /// Create a note sub-diagnostic.
    pub fn note(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Note, message)
    }

    /// Create a help sub-diagnostic.
    pub fn help_sub(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Help, message)
    }

    /// Attach a source span to this sub-diagnostic.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a label to this sub-diagnostic.
    pub fn with_label(mut self, span: Span, label: impl Into<String>) -> Self {
        self.labels.push(Label::new(span, label));
        self
    }
}

impl Diagnostic {
    // ── Constructors ───────────────────────────────────────────────

    pub fn new(level: DiagnosticLevel, message: impl Into<String>) -> Self {
        Diagnostic {
            level,
            message: message.into(),
            code: None,
            kind: None,
            spans: MultiSpan::default(),
            help: None,
            suggestions: Vec::new(),
            labels: Vec::new(),
            call_chain: None,
            source: None,
            file_name: "<input>".into(),
            children: Vec::new(),
            related_errors: Vec::new(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Error, message)
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Warning, message)
    }

    /// Create an error diagnostic from a structured [`DiagnosticKind`].
    /// The humanizer runs immediately to populate `message`, `labels`,
    /// `help`, and `suggestions` from the kind's data.
    pub fn error_kind(kind: DiagnosticKind) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Error, "");
        diag.kind = Some(kind);
        diag.humanize();
        diag
    }

    /// Create a warning diagnostic from a structured [`DiagnosticKind`].
    pub fn warning_kind(kind: DiagnosticKind) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Warning, "");
        diag.kind = Some(kind);
        diag.humanize();
        diag
    }

    pub fn help_diag(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Help, message)
    }

    pub fn note(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Note, message)
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Info, message)
    }

    // ── Builder methods ────────────────────────────────────────────

    /// Set the primary source span for this diagnostic.
    pub fn with_span(mut self, span: Span) -> Self {
        self.spans = MultiSpan::new(span);
        self
    }

    /// Add a secondary primary span (for multi-location diagnostics).
    pub fn with_additional_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self
    }

    /// Set the error code (e.g. `ErrCode::TypeMismatch` → `E030`).
    pub fn with_code(mut self, code: ErrCode) -> Self {
        self.code = Some(code);
        self
    }

    /// Convenience: set error code by string (backward compat).
    /// Prefer `with_code(ErrCode::new(...))` for the new API.
    pub fn with_code_str(mut self, code: impl Into<String>) -> Self {
        self.code = Some(ErrCode::new(code));
        self
    }

    /// Add a help message.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Add a single suggestion with default applicability and style.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(Suggestion {
            message: suggestion.into(),
            applicability: Applicability::MachineApplicable,
            style: SuggestionStyle::ShowAlways,
        });
        self
    }

    /// Add a suggestion with explicit applicability and style.
    pub fn with_suggestion_style(
        mut self,
        suggestion: impl Into<String>,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> Self {
        self.suggestions.push(Suggestion {
            message: suggestion.into(),
            applicability,
            style,
        });
        self
    }

    /// Replace suggestions list.
    pub fn with_suggestions(mut self, suggestions: Vec<Suggestion>) -> Self {
        self.suggestions = suggestions;
        self
    }

    /// Add a labeled sub-span for multi-span highlighting.
    pub fn with_label(mut self, span: Span, label: impl Into<String>) -> Self {
        self.labels.push(Label::new(span, label));
        self
    }

    /// Add a secondary (tilde-underline) label.
    pub fn with_secondary_label(mut self, span: Span, label: impl Into<String>) -> Self {
        self.labels.push(Label::secondary(span, label));
        self
    }

    /// Attach a related error to be displayed alongside this diagnostic
    /// in a single merged box (e.g. a type mismatch aggregated into a
    /// duplicate definition diagnostic).
    pub fn with_related(mut self, err: RelatedError) -> Self {
        self.related_errors.push(err);
        self
    }

    /// Attach a call chain (Zig-style `referenced by`).
    pub fn with_call_chain(mut self, chain: CallChain) -> Self {
        self.call_chain = Some(chain);
        self
    }

    /// Attach source text for `^`-underline rendering.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set the source file name for the `┌─ <file>:1:1` location header.
    pub fn with_file_name(mut self, file_name: impl Into<String>) -> Self {
        self.file_name = file_name.into();
        self
    }

    /// Set the structured diagnostic kind (Vale-style ADT).
    ///
    /// When present, the humanizer can derive `message`, `labels`, and other
    /// fields from the kind's structured data, producing more precise error
    /// messages than a flat string.
    pub fn with_kind(mut self, kind: DiagnosticKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Add a highlighted note child (styled text with type/name highlights).
    pub fn with_highlighted_note(mut self, styled: StyledString) -> Self {
        let msg = styled.to_plain();
        self.children.push(Subdiag {
            level: DiagnosticLevel::Note,
            message: msg,
            styled_message: Some(styled),
            span: None,
            labels: Vec::new(),
        });
        self
    }

    /// Add a highlighted help child (styled text with type/name highlights).
    pub fn with_highlighted_help(mut self, styled: StyledString) -> Self {
        let msg = styled.to_plain();
        self.children.push(Subdiag {
            level: DiagnosticLevel::Help,
            message: msg,
            styled_message: Some(styled),
            span: None,
            labels: Vec::new(),
        });
        self
    }

    /// Run the humanizer on `self.kind` (if present) to populate `message`,
    /// `labels`, `help`, and `suggestions` from the structured data.
    ///
    /// This is a no-op if `kind` is `None` (the message/labels were already
    /// set manually).
    pub fn humanize(&mut self) {
        let Some(ref kind) = self.kind else { return };
        self.message = kind.message();
        let labels = kind.labels();
        // Populate `spans` from the primary labels so the source-context
        // renderer knows which lines to display.
        for lbl in &labels {
            if matches!(lbl.kind, AnnotationKind::Primary) {
                self.spans.push(lbl.span);
            }
        }
        self.labels = labels;
        if self.help.is_none() {
            self.help = kind.help();
        }
        if self.suggestions.is_empty() {
            self.suggestions = kind.suggestions().into_iter().map(|s| Suggestion {
                message: s,
                applicability: Applicability::MachineApplicable,
                style: SuggestionStyle::ShowAlways,
            }).collect();
        }
    }

    // ── Query methods ──────────────────────────────────────────────

    pub fn is_error(&self) -> bool {
        matches!(self.level, DiagnosticLevel::Error)
    }

    pub fn is_warning(&self) -> bool {
        matches!(self.level, DiagnosticLevel::Warning)
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
    pub fn augment(&mut self, other: &Diagnostic) {
        if self.spans.first().is_none() {
            if let Some(span) = other.spans.first() {
                self.spans = MultiSpan::new(span);
            }
        }
        if self.source.is_none() {
            self.source.clone_from(&other.source);
        }
        if self.call_chain.is_none() {
            self.call_chain.clone_from(&other.call_chain);
        }
        // Merge labels that aren't already present
        for lbl in &other.labels {
            if !self.labels.iter().any(|l| l.span == lbl.span && l.message == lbl.message) {
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
/// # Safety
///
/// `EmissionGuaranteed` can only be obtained by calling `DiagCtxt::push()`,
/// which records a diagnostic before returning the token.  Therefore, if
/// `f` returns `Err(EmissionGuaranteed)`, at least one diagnostic must have
/// been pushed, and `last_mut()` is guaranteed to return `Some`.
///
/// # Example
/// ```ignore
/// adorn_with_span(ctxt, span, source, |ctxt| {
///     check_something(ctxt)?;
///     Ok(())
/// });
/// ```
pub fn adorn_with_span<T>(
    ctxt: &mut DiagCtxt,
    span: Span,
    source: Option<&str>,
    f: impl FnOnce(&mut DiagCtxt) -> Result<T, EmissionGuaranteed>,
) -> Result<T, EmissionGuaranteed> {
    let diag_count_before = ctxt.len();
    let result = f(ctxt);
    // If the closure failed, augment the last diagnostic with context.
    if result.is_err() {
        // Assert that at least one diagnostic was pushed by the closure.
        debug_assert!(
            ctxt.len() > diag_count_before,
            "adorn_with_span: closure returned Err but no diagnostic was pushed — \
             the closure must push a diagnostic via `ctxt.push(...)` before returning Err",
        );
        if let Some(diag) = ctxt.last_mut() {
            if diag.spans.first().is_none() {
                diag.spans = MultiSpan::new(span);
            }
            if diag.source.is_none() {
                diag.source = source.map(|s| s.to_string());
            }
        }
    }
    result
}

// ── ErrCode is now a string-based code — use `ErrCode::new("E030")` ──
// The old `from_str` method with enum variants was removed in favour of
// the lookup table in `error_code.rs`.  Any string is accepted as a code.

// ── Display ───────────────────────────────────────────────────────

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code_str = self.code.as_ref().map(|c| c.code()).unwrap_or("?");
        let level_str = self.level.label().to_uppercase();
        let span_str = self.location_string();

        writeln!(
            f,
            "[{code} {level}] {msg} {span}",
            code = code_str,
            level = level_str,
            msg = self.message,
            span = span_str,
        )?;

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
        if let Some(ref chain) = self.call_chain {
            if !chain.is_empty() {
                write!(f, "{}", chain.render(false))?;
            }
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
/// Output is formatted with syntax highlighting, text wrapping, and ANSI colors.
pub fn explain_error_code(code_str: &str) -> String {
    use std::fmt::Write;

    // Use `try_new` to validate the code exists in the lookup table.
    let code = match ErrCode::try_new(code_str) {
        Ok(code) => code,
        Err(_) => {
            return format!(
                "\x1b[1;31merror\x1b[0m\x1b[2m[E000]\x1b[0m: unknown error code `{code_str}`\n\
                 {help}",
                help = suggest_code(code_str),
            );
        }
    };
    let mut out = String::new();

    // ── Header: bold error code + title ──
    let _ = writeln!(
        out,
        "\x1b[1m{code}:\x1b[0m \x1b[36m{title}\x1b[0m",
        code = code.code(),
        title = code.title(),
    );
    let _ = writeln!(out);

    // ── Body: parse explanation text, highlight code blocks, wrap text ──
    let explain = code.explain();
    let mut in_code_block = false;
    // Dark background for code blocks (256-color: dark gray)
    const CODE_BG: &str = "\x1b[48;5;236m";
    const RESET: &str = "\x1b[0m";

    for line in explain.lines() {
        if line.starts_with("  ") && !line.trim().is_empty() {
            // Code/example line — apply syntax highlighting + background
            if !in_code_block {
                let _ = writeln!(out);
                in_code_block = true;
            }
            let highlighted = crate::diagnostics::glyph::highlight_code(line, true);
            let _ = writeln!(out, "  {CODE_BG}{highlighted}{RESET}");
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
            // Regular text — wrap at 78 columns
            let wrapped = textwrap::fill(line, 78);
            let _ = write!(out, "{}", wrapped);
            let _ = writeln!(out);
        }
    }

    out
}

/// Generate a "did you mean?" suggestion for an unknown error code.
/// Uses a simple prefix/suffix matching heuristic against the CODE_TABLE.
fn suggest_code(input: &str) -> String {
    // Find candidates that share a common prefix or suffix with the input.
    let candidates: Vec<&str> = crate::diagnostics::error_code::CODE_TABLE
        .iter()
        .map(|e| e.code)
        .filter(|code| {
            let common = code.chars().zip(input.chars()).take_while(|(a, b)| a == b).count();
            common >= 3 && (code.len() != input.len())
        })
        .take(3)
        .collect();

    if candidates.is_empty() {
        "\x1b[1;34mhelp\x1b[0m: valid error codes include E001–E061, E101–E103, W113 — \n       run `ponent explain <CODE>` with a valid code (e.g. `ponent explain E030`)".to_string()
    } else {
        format!(
            "\x1b[1;34mhelp\x1b[0m: did you mean `{}`?\n       run `ponent explain <CODE>` with a valid error code",
            candidates.join("` or `"),
        )
    }
}

/// List all available error codes with their titles, formatted for terminal output.
pub fn list_error_codes() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "\x1b[1mAvailable error codes:\x1b[0m\n");
    let mut last_category = None;
    for entry in crate::diagnostics::error_code::CODE_TABLE {
        let cat = entry.category;
        if Some(cat) != last_category {
            let _ = writeln!(
                out,
                " \x1b[36m── {} ──\x1b[0m",
                cat.as_str(),
            );
            last_category = Some(cat);
        }
        let _ = writeln!(
            out,
            "  \x1b[1m{}\x1b[0m  {}",
            entry.code,
            entry.title,
        );
    }
    let _ = writeln!(
        out,
        "\n\x1b[2mRun `ponent explain <CODE>` for details on a specific error code.\x1b[0m"
    );
    out
}

/// Convenience function to emit diagnostics to stderr.
/// Chooses plain or colored emitter based on the `use_color` flag.
pub fn emit_diagnostics(diags: &[Diagnostic], use_color: bool) {
    if use_color {
        let mut emitter = ColoredEmitter::new();
        emitter.emit_all(diags);
    } else {
        let mut emitter = PlainEmitter::new();
        emitter.emit_all(diags);
    }
}

/// Convenience function to emit a DiagCtxt.
pub fn emit_collector(collector: &DiagCtxt, use_color: bool) {
    let diags: Vec<Diagnostic> = collector.iter().cloned().collect();
    emit_diagnostics(&diags, use_color);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;

    #[test]
    fn test_basic_error() {
        let diag = Diagnostic::error("something went wrong")
            .with_code(ErrCode::new("E030"))
            .with_span(Span::new(5, 12))
            .with_suggestion("try using `as` to cast");
        assert!(diag.is_error());
        assert_eq!(diag.code.as_ref().unwrap().code(), "E030");
    }

    #[test]
    fn test_with_labels() {
        let diag = Diagnostic::error("no field `z` on type Point")
            .with_code(ErrCode::new("E010"))
            .with_span(Span::new(50, 53))
            .with_label(Span::new(0, 30), "type defined here");
        assert_eq!(diag.labels.len(), 1);
        assert_eq!(diag.labels[0].message, "type defined here");
    }

    #[test]
    fn test_error_code_explain() {
        let explain = explain_error_code("E030");
        assert!(explain.contains("type mismatch"));
        assert!(explain.contains("E030"));
    }

    #[test]
    fn test_explain_unknown_code() {
        let explain = explain_error_code("E03");
        assert!(explain.contains("unknown error code"));
        assert!(explain.contains("E03"));
        assert!(explain.contains("E030")); // did you mean suggestion
        assert!(explain.contains("help"));

        let explain = explain_error_code("E999");
        assert!(explain.contains("unknown error code"));
        assert!(explain.contains("E999"));
        // No candidates should match "E999" — falls back to generic help
        assert!(explain.contains("valid error codes include"));
    }

    #[test]
    fn test_call_chain() {
        let mut chain = CallChain::new();
        chain.push(chain::ChainEntry {
            kind: chain::ChainEntryKind::Call,
            span: Span::new(80, 90),
            message: "main".to_string(),
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
        assert_eq!(
            ErrCode::new("E020").category(),
            ErrorCategory::Contract
        );
        assert_eq!(
            ErrCode::new("E041").category(),
            ErrorCategory::Trait
        );
    }
}
