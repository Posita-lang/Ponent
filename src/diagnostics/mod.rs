//! Ponent diagnostic system.
//!
//! A comprehensive, multi-format diagnostic system modeled after:
//! - **rustc**: error codes (`E030`), `^`-underline annotations, `--explain`, suggestions
//! - **Zig**: `referenced by` call chains
//! - **Austral**: multi-format output (plain / JSON / HTML), error type classification
//! - **Vale**: per-pass error humanizers, structured error types

pub mod chain;
pub mod collector;
pub mod emitter;
pub mod error_code;
pub mod glyph;
pub mod label;
pub mod level;

pub use chain::CallChain;
pub use collector::DiagCtxt;
pub use emitter::{
    DiagnosticEmitter, colored::ColoredEmitter, html::HtmlEmitter, json::JsonEmitter,
    plain::PlainEmitter,
};
pub use error_code::{ErrorCategory, ErrorCode, WarningCode};
pub use glyph::GlyphRenderer;
pub use label::{AnnotationKind, Label, SourceContext};
pub use level::DiagnosticLevel;

use crate::ast::Span;
use std::fmt;

// ── Emission guarantee (inspired by rustc's ErrorGuaranteed) ─────

/// A token proving that a diagnostic (error or warning) has been emitted.
///
/// Functions that can fail should return `DiagResult<T>` and the caller
/// pushes it into the collector.  `EmissionGuarantee` tracks whether the
/// diagnostic was *actually* pushed, preventing "silent swallowing" of errors.
///
/// See rustc's `ErrorGuaranteed` for the original pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmissionGuarantee {
    _private: (),
}

impl EmissionGuarantee {
    /// The emitted diagnostic was an error or warning.
    pub fn emitted() -> Self {
        EmissionGuarantee { _private: () }
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
        MultiSpan { primary: vec![span] }
    }

    pub fn push(&mut self, span: Span) {
        self.primary.push(span);
    }

    pub fn first(&self) -> Option<Span> {
        self.primary.first().copied()
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
    pub level: DiagnosticLevel,
    pub message: String,
    pub code: Option<ErrorCode>,
    /// Warning code (e.g. `WarningCode::Shadowing` → "W113").
    pub warning_code: Option<WarningCode>,
    /// Primary spans (multiple for multi-location diagnostics).
    pub spans: MultiSpan,
    pub help: Option<String>,
    pub suggestions: Vec<String>,
    /// Multi-span annotations for source-context rendering
    pub labels: Vec<Label>,
    /// Zig-style call chain tracing how the error was reached
    pub call_chain: Option<CallChain>,
    /// Retained source text for `^`-underline rendering
    pub source: Option<String>,
    /// Child diagnostics (notes, help messages attached to this diagnostic).
    /// Inspired by rustc's `Subdiag` — emitted immediately after the parent.
    pub children: Vec<Subdiag>,
    /// Related errors displayed alongside this diagnostic in a single merged box.
    /// Each related error has its own error code, message, and optional span.
    pub related_errors: Vec<RelatedError>,
}

/// A child diagnostic attached to a parent `Diagnostic`.
/// Inspired by rustc's `Subdiag` — used for notes, help, or secondary
/// labels that are logically part of the same diagnostic group.
#[derive(Debug, Clone)]
pub struct Subdiag {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Option<Span>,
    pub labels: Vec<Label>,
}

/// A related error displayed alongside the primary diagnostic in a single
/// merged box.  Each related error has its own error code, message, and
/// optional source span — for example, a type mismatch [E030] aggregated
/// into a duplicate definition [E019] diagnostic.
#[derive(Debug, Clone)]
pub struct RelatedError {
    pub code: Option<ErrorCode>,
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
            warning_code: None,
            spans: MultiSpan::default(),
            help: None,
            suggestions: Vec::new(),
            labels: Vec::new(),
            call_chain: None,
            source: None,
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

    /// Set the error code (e.g. `ErrorCode::TypeMismatch` → `E030`).
    pub fn with_code(mut self, code: ErrorCode) -> Self {
        self.code = Some(code);
        self
    }

    /// Convenience: set error code by string (backward compat).
    /// Prefer `with_code(ErrorCode::...)` for the new enum-based API.
    pub fn with_code_str(mut self, code: impl Into<String>) -> Self {
        let code_str = code.into();
        if let Some(ec) = ErrorCode::from_str(&code_str) {
            self.code = Some(ec);
        } else if let Some(wc) = WarningCode::from_str(&code_str) {
            self.warning_code = Some(wc);
        }
        self
    }

    /// Add a help message.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Add a single suggestion.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Replace suggestions list.
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
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

    // ── Query methods ──────────────────────────────────────────────

    pub fn is_error(&self) -> bool {
        matches!(self.level, DiagnosticLevel::Error)
    }

    pub fn is_warning(&self) -> bool {
        matches!(self.level, DiagnosticLevel::Warning)
    }

    /// Get the main location string for display.
    pub fn location_string(&self) -> String {
        self.spans.first().map(|s| format!("at {}", s)).unwrap_or_default()
    }
}

// ── Forward-compatible aliases for old-style string codes ─────────

impl ErrorCode {
    /// Attempt to parse a string like `"E030"` into an `ErrorCode`.
    pub fn from_str(code: &str) -> Option<Self> {
        match code {
            "E001" => Some(ErrorCode::ExpectedToken),
            "E002" => Some(ErrorCode::UnexpectedEOF),
            "E003" => Some(ErrorCode::UnexpectedToken),
            "E004" => Some(ErrorCode::ParseError),
            "E005" => Some(ErrorCode::ExpectedIdentifier),
            "E006" => Some(ErrorCode::RecursionLimitExceeded),
            "E010" => Some(ErrorCode::NoSuchField),
            "E011" => Some(ErrorCode::TypeNotFound),
            "E012" => Some(ErrorCode::NameNotFound),
            "E013" => Some(ErrorCode::UndefinedType),
            "E014" => Some(ErrorCode::GenericArgsOnNonGeneric),
            "E015" => Some(ErrorCode::CannotResolveImport),
            "E016" => Some(ErrorCode::NoDefaultValue),
            "E017" => Some(ErrorCode::ArraySizeNotConstant),
            "E018" => Some(ErrorCode::UnexpectedTopLevel),
            "E019" => Some(ErrorCode::DuplicateDefinition),
            "E020" => Some(ErrorCode::ContractNonBool),
            "E021" => Some(ErrorCode::EnsuresNonBool),
            "E022" => Some(ErrorCode::DecreasesNonInt),
            "E030" => Some(ErrorCode::TypeMismatch),
            "E031" => Some(ErrorCode::InvalidBinaryOp),
            "E032" => Some(ErrorCode::InvalidUnaryOp),
            "E033" => Some(ErrorCode::WrongNumberOfArgs),
            "E034" => Some(ErrorCode::ExpectedBool),
            "E035" => Some(ErrorCode::ExpectedInteger),
            "E036" => Some(ErrorCode::ExpectedResult),
            "E037" => Some(ErrorCode::ExpectedFuture),
            "E038" => Some(ErrorCode::ReturnOutsideFunction),
            "E039" => Some(ErrorCode::ReturnWithoutValue),
            "E040" => Some(ErrorCode::LeaveOutsideLoop),
            "E041" => Some(ErrorCode::ContinueOutsideLoop),
            "E042" => Some(ErrorCode::CannotLeaveClosure),
            "E043" => Some(ErrorCode::SetNoPattern),
            "E044" => Some(ErrorCode::LetNeedsInit),
            "E045" => Some(ErrorCode::InvalidLValue),
            "E046" => Some(ErrorCode::ContractBoolAtReturn),
            "E100" => Some(ErrorCode::TraitNotFound),
            "E101" => Some(ErrorCode::ImplMissingMethod),
            "E102" => Some(ErrorCode::ImplMissingAssocType),
            "E103" => Some(ErrorCode::ImplSignatureMismatch),
            "E104" => Some(ErrorCode::OrphanImpl),
            "E105" => Some(ErrorCode::InherentImplOnNonAdt),
            "E106" => Some(ErrorCode::TraitViolatesTermination),
            "E600" => Some(ErrorCode::SafeCastFromRef),
            "E601" => Some(ErrorCode::SafeCastNonPrimitive),
            "E602" => Some(ErrorCode::UnsafeCastRefToInt),
            "E603" => Some(ErrorCode::UnsafeCastIncompatible),
            "E604" => Some(ErrorCode::UnknownAttribute),
            _ => None,
        }
    }
}

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
pub fn explain_error_code(code_str: &str) -> String {
    match ErrorCode::from_str(code_str) {
        Some(code) => {
            format!("{}: {}\n\n{}", code.code(), code.title(), code.explain())
        }
        None => format!("Unknown error code: {}", code_str),
    }
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
            .with_code(ErrorCode::TypeMismatch)
            .with_span(Span::new(5, 12))
            .with_suggestion("try using `as` to cast");
        assert!(diag.is_error());
        assert_eq!(diag.code.as_ref().unwrap().code(), "E030");
    }

    #[test]
    fn test_with_labels() {
        let diag = Diagnostic::error("no field `z` on type Point")
            .with_code(ErrorCode::NoSuchField)
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
    fn test_call_chain() {
        let mut chain = CallChain::new();
        chain.push(chain::ChainEntry {
            kind: chain::ChainEntryKind::Call,
            span: Span::new(80, 90),
            message: "main".to_string(),
        });
        let diag = Diagnostic::error("type mismatch")
            .with_code(ErrorCode::TypeMismatch)
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
        collector.push(Diagnostic::error("first error").with_code(ErrorCode::TypeMismatch));
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
        assert_eq!(ErrorCode::TypeMismatch.category(), ErrorCategory::Type);
        assert_eq!(ErrorCode::ExpectedToken.category(), ErrorCategory::Parse);
        assert_eq!(
            ErrorCode::ContractNonBool.category(),
            ErrorCategory::Contract
        );
        assert_eq!(
            ErrorCode::ImplMissingMethod.category(),
            ErrorCategory::Trait
        );
    }
}
