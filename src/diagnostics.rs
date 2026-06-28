use crate::ast::Span;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Help,
    Note,
    Info,
}

impl DiagnosticLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            DiagnosticLevel::Error => "error",
            DiagnosticLevel::Warning => "warning",
            DiagnosticLevel::Help => "help",
            DiagnosticLevel::Note => "note",
            DiagnosticLevel::Info => "info",
        }
    }

    pub fn prefix(&self) -> &'static str {
        match self {
            DiagnosticLevel::Error => "E",
            DiagnosticLevel::Warning => "W",
            DiagnosticLevel::Help => "H",
            DiagnosticLevel::Note => "N",
            DiagnosticLevel::Info => "I",
        }
    }

    pub fn ansi_color(&self) -> &'static str {
        match self {
            DiagnosticLevel::Error => "\x1b[31m",
            DiagnosticLevel::Warning => "\x1b[33m",
            DiagnosticLevel::Help => "\x1b[36m",
            DiagnosticLevel::Note => "\x1b[34m",
            DiagnosticLevel::Info => "\x1b[32m",
        }
    }

    pub fn ansi_reset() -> &'static str {
        "\x1b[0m"
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Option<Span>,
    pub code: Option<String>,
    pub help: Option<String>,
    pub suggestions: Vec<String>,
    /// Additional labeled sub-spans for multi-span error highlighting
    pub labels: Vec<(Span, String)>,
}

impl Diagnostic {
    pub fn new(level: DiagnosticLevel, message: impl Into<String>) -> Self {
        Diagnostic {
            level,
            message: message.into(),
            span: None,
            code: None,
            help: None,
            suggestions: Vec::new(),
            labels: Vec::new(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Error, message)
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Warning, message)
    }

    pub fn help(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Help, message)
    }

    pub fn note(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Note, message)
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Info, message)
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions = suggestions;
        self
    }

    /// Add a labeled sub-span to this diagnostic for multi-span highlighting.
    /// The span should point to a related location (e.g. the definition of a type
    /// that caused a type mismatch).
    pub fn with_label(mut self, span: Span, label: impl Into<String>) -> Self {
        self.labels.push((span, label.into()));
        self
    }

    pub fn is_error(&self) -> bool {
        matches!(self.level, DiagnosticLevel::Error)
    }

    pub fn is_warning(&self) -> bool {
        matches!(self.level, DiagnosticLevel::Warning)
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let level_str = self.level.as_str().to_uppercase();
        let code_str = self.code.as_deref().unwrap_or("?");
        let span_str = self.span.map(|s| format!(" at {}", s)).unwrap_or_default();
        writeln!(
            f,
            "{}[{} {}]{} {}{}",
            self.level.ansi_color(),
            code_str,
            level_str,
            DiagnosticLevel::ansi_reset(),
            self.message,
            span_str
        )?;
        if let Some(help) = &self.help {
            writeln!(
                f,
                "{}help: {}{}",
                DiagnosticLevel::ansi_color_for_level(DiagnosticLevel::Help),
                help,
                DiagnosticLevel::ansi_reset()
            )?;
        }
        for suggestion in &self.suggestions {
            writeln!(
                f,
                "{}suggestion: {}{}",
                DiagnosticLevel::ansi_color_for_level(DiagnosticLevel::Help),
                suggestion,
                DiagnosticLevel::ansi_reset()
            )?;
        }
        for (label_span, label_text) in &self.labels {
            writeln!(
                f,
                "{}  --> {}: {}{}",
                DiagnosticLevel::ansi_color_for_level(DiagnosticLevel::Help),
                label_span,
                label_text,
                DiagnosticLevel::ansi_reset()
            )?;
        }
        Ok(())
    }
}

impl DiagnosticLevel {
    fn ansi_color_for_level(level: DiagnosticLevel) -> &'static str {
        level.ansi_color()
    }
}

#[derive(Debug, Clone, Default)]
pub struct DiagnosticCollector {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticCollector {
    pub fn new() -> Self {
        DiagnosticCollector {
            diagnostics: Vec::new(),
        }
    }

    pub fn push(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    pub fn extend(&mut self, diags: Vec<Diagnostic>) {
        self.diagnostics.extend(diags);
    }

    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Diagnostic> {
        self.diagnostics.iter()
    }

    pub fn into_inner(self) -> Vec<Diagnostic> {
        self.diagnostics
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error())
    }

    pub fn clear(&mut self) {
        self.diagnostics.clear();
    }

    pub fn emit(&self, emitter: &mut dyn DiagnosticEmitter) {
        emitter.emit_all(&self.diagnostics);
    }
}

impl From<Vec<Diagnostic>> for DiagnosticCollector {
    fn from(diags: Vec<Diagnostic>) -> Self {
        DiagnosticCollector { diagnostics: diags }
    }
}

pub trait DiagnosticEmitter {
    fn emit(&mut self, diag: &Diagnostic);
    fn emit_all(&mut self, diags: &[Diagnostic]) {
        for diag in diags {
            self.emit(diag);
        }
    }
}

pub struct PlainEmitter;

impl PlainEmitter {
    pub fn new() -> Self {
        PlainEmitter
    }
}

impl Default for PlainEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagnosticEmitter for PlainEmitter {
    fn emit(&mut self, diag: &Diagnostic) {
        let level_str = diag.level.as_str().to_uppercase();
        let code_str = diag.code.as_deref().unwrap_or("?");
        let span_str = diag.span.map(|s| format!(" at {}", s)).unwrap_or_default();
        eprintln!("[{} {}] {}{}", code_str, level_str, diag.message, span_str);
        if let Some(help) = &diag.help {
            eprintln!("  help: {}", help);
        }
        for suggestion in &diag.suggestions {
            eprintln!("  suggestion: {}", suggestion);
        }
    }
}

pub struct ColoredEmitter;

impl ColoredEmitter {
    pub fn new() -> Self {
        ColoredEmitter
    }
}

impl Default for ColoredEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagnosticEmitter for ColoredEmitter {
    fn emit(&mut self, diag: &Diagnostic) {
        let level_str = diag.level.as_str().to_uppercase();
        let code_str = diag.code.as_deref().unwrap_or("?");
        let span_str = diag.span.map(|s| format!(" at {}", s)).unwrap_or_default();
        eprintln!(
            "{}{}[{} {}]{} {}{}",
            diag.level.ansi_color(),
            if diag.level == DiagnosticLevel::Error {
                "error: "
            } else {
                ""
            },
            code_str,
            level_str,
            DiagnosticLevel::ansi_reset(),
            diag.message,
            span_str
        );
        if let Some(help) = &diag.help {
            eprintln!(
                "{}help: {}{}",
                DiagnosticLevel::ansi_color_for_level(DiagnosticLevel::Help),
                help,
                DiagnosticLevel::ansi_reset()
            );
        }
        for suggestion in &diag.suggestions {
            eprintln!(
                "{}suggestion: {}{}",
                DiagnosticLevel::ansi_color_for_level(DiagnosticLevel::Help),
                suggestion,
                DiagnosticLevel::ansi_reset()
            );
        }
    }
}

pub fn emit_diagnostics(diags: &[Diagnostic], use_color: bool) {
    let mut emitter: Box<dyn DiagnosticEmitter> = if use_color {
        Box::new(ColoredEmitter::new())
    } else {
        Box::new(PlainEmitter::new())
    };
    emitter.emit_all(diags);
}

pub fn emit_collector(collector: &DiagnosticCollector, use_color: bool) {
    emit_diagnostics(&collector.diagnostics, use_color);
}
