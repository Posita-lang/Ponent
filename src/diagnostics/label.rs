use crate::ast::Span;
use std::fmt;

/// Kind of annotation underline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationKind {
    /// Solid underline `^^^` — the primary error location.
    Primary,
    /// Tilde underline `~~~` — a secondary related location.
    Secondary,
    /// Dashed underline `---` — a note-level location.
    Note,
}

/// A labeled annotation pointing to a span of source code.
#[derive(Debug, Clone)]
pub struct Label {
    pub span: Span,
    pub message: String,
    pub kind: AnnotationKind,
}

impl Label {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Primary,
        }
    }

    pub fn secondary(span: Span, message: impl Into<String>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Secondary,
        }
    }

    pub fn note(span: Span, message: impl Into<String>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Note,
        }
    }

    pub fn with_kind(mut self, kind: AnnotationKind) -> Self {
        self.kind = kind;
        self
    }

    /// The underline character for this annotation kind.
    pub fn underline_char(&self) -> char {
        match self.kind {
            AnnotationKind::Primary => '^',
            AnnotationKind::Secondary => '~',
            AnnotationKind::Note => '-',
        }
    }
}

// ── Byte-offset → line:column conversion ──────────────────────

/// A (line, column) position in source code.
/// `line` and `col` are 0-based internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePos {
    /// 0-based line index.
    pub line: usize,
    /// 0-based byte offset from the start of the line.
    pub col: usize,
}

/// Convert a byte-offset span into a 0-based (line, col) pair.
fn byte_to_linecol(source: &str, byte_offset: usize) -> SourcePos {
    let len = source.len();
    let clamped = std::cmp::min(byte_offset, len);
    let prefix = &source[..clamped];
    let line = prefix.matches('\n').count();
    let start_of_line = prefix.rfind('\n').map(|i| i + 1).unwrap_or(0);
    SourcePos {
        line,
        col: clamped - start_of_line,
    }
}

// ── miette-based source context rendering ─────────────────────

use miette::{
    Diagnostic, GraphicalReportHandler, GraphicalTheme, LabeledSpan, NamedSource, ReportHandler,
    SourceCode, SourceSpan,
};

/// Wrapper that implements `miette::Diagnostic` for our `Label` list.
struct MietteDiag<'a> {
    source: NamedSource<String>,
    labels: &'a [Label],
}

impl<'a> std::fmt::Debug for MietteDiag<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MietteDiag")
            .field("labels", &self.labels.len())
            .finish()
    }
}

impl<'a> std::error::Error for MietteDiag<'a> {}

impl<'a> std::fmt::Display for MietteDiag<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compile error")
    }
}

impl<'a> Diagnostic for MietteDiag<'a> {
    fn source_code(&self) -> Option<&dyn SourceCode> {
        Some(&self.source as &dyn SourceCode)
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        let labels: Vec<LabeledSpan> = self
            .labels
            .iter()
            .map(|lbl| {
                let len = lbl.span.end.saturating_sub(lbl.span.start);
                let span = SourceSpan::new(lbl.span.start.into(), len);
                let label = if lbl.message.is_empty() {
                    None
                } else {
                    Some(lbl.message.clone())
                };
                match lbl.kind {
                    AnnotationKind::Primary => {
                        LabeledSpan::new_primary_with_span(label, span)
                    }
                    _ => LabeledSpan::new_with_span(label, span),
                }
            })
            .collect();
        Some(Box::new(labels.into_iter()))
    }
}

// ── Source context extraction ─────────────────────────────────

/// Renders source code snippets with `^`-underline annotations.
///
/// Format (Rust-inspired):
///   --> file.ps:5:12
///    |
///  5 |     return x + 1;
///    |            ^ expected Bool, found Int<32>
///
/// Secondary labels use `~~~` and notes use `---`.
pub struct SourceContext {
    source: String,
    filename: String,
    context_lines: usize,
}

impl SourceContext {
    /// Build source context over the entire source.
    /// `context_lines` controls how many lines of surrounding context
    /// are shown around each annotated line.
    pub fn new(source: &str, _span: Span, filename: &str, context_lines: usize) -> Self {
        SourceContext {
            source: source.to_string(),
            filename: filename.to_string(),
            context_lines,
        }
    }

    /// Render the source context with annotations using `miette`.
    ///
    /// `primary_span` is the main error location, `labels` are additional
    /// annotations (secondary/note).  Both are rendered as underlines.
    pub fn render(
        &self,
        primary_span: Span,
        labels: &[Label],
        use_color: bool,
    ) -> String {
        // Merge primary_span into the labels list as a primary label
        // so miette can render the main error location underline.
        let mut all_labels: Vec<Label> = Vec::with_capacity(labels.len() + 1);
        all_labels.push(Label {
            span: primary_span,
            message: String::new(),
            kind: AnnotationKind::Primary,
        });
        all_labels.extend_from_slice(labels);

        let diag = MietteDiag {
            source: NamedSource::new(&self.filename, self.source.clone()),
            labels: &all_labels,
        };

        let handler = if use_color {
            GraphicalReportHandler::new()
                .with_context_lines(self.context_lines)
        } else {
            GraphicalReportHandler::new_themed(GraphicalTheme::ascii())
                .with_context_lines(self.context_lines)
        };

        let mut out = String::new();
        handler.render_report(&mut out, &diag).ok();
        out
    }
}

/// Convert byte offset to human-readable "line:col" string (1-based).
pub fn byte_offset_to_string(source: &str, byte_offset: usize) -> String {
    let pos = byte_to_linecol(source, byte_offset);
    format!("{}:{}", pos.line + 1, pos.col + 1)
}
