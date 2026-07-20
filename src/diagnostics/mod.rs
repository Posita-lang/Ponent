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
pub use error_code::{ErrorCategory, ErrCode};
pub use glyph::GlyphRenderer;
pub use label::{AnnotationKind, Label, SourcePos};
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
    pub code: Option<ErrCode>,
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
        self.spans
            .first()
            .map(|s| format!("at {}", s))
            .unwrap_or_default()
    }
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

    // Check if the code exists in the lookup table before proceeding.
    let Some(code) = crate::diagnostics::error_code::lookup(code_str) else {
        return format!(
            "\x1b[1;31merror\x1b[0m\x1b[2m[E000]\x1b[0m: unknown error code `{code_str}`\n\
             {help}",
            help = suggest_code(code_str),
        );
    };

    let code = ErrCode::new(code_str);
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
