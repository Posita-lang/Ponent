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
    /// Help annotation — no underline, rendered as `| help: message`.
    Help,
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

    pub fn help(span: Span, message: impl Into<String>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Help,
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
            AnnotationKind::Help => ' ', // no underline, rendered as `| help: message`
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
pub fn byte_to_linecol(source: &str, byte_offset: usize) -> SourcePos {
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

/// Convert byte offset to human-readable "line:col" string (1-based).
pub fn byte_offset_to_string(source: &str, byte_offset: usize) -> String {
    let pos = byte_to_linecol(source, byte_offset);
    format!("{}:{}", pos.line + 1, pos.col + 1)
}

// ── Snippet: a contiguous slice of source code with annotations ──

/// A contiguous slice of source code with its annotations, ready for
/// rendering.  Inspired by the `Snippet` type in `annotate-snippets-rs`.
///
/// Each `Snippet` represents one continuous region of source text (e.g. a
/// function body, a type definition) together with the labels / underlines
/// that should be drawn over it.
#[derive(Debug, Clone)]
pub struct Snippet {
    /// Display name / file path of the source (e.g. `"src/main.pn"`).
    pub origin: String,
    /// 1‑based line number of the first line in `source`.
    pub line_start: usize,
    /// The raw source text (may span multiple lines).
    pub source: String,
    /// Annotations (labels) attached to this snippet.
    pub annotations: Vec<Label>,
    /// Whether to fold long runs of unannotated lines into `…`.
    pub fold: bool,
}

impl Snippet {
    pub fn new(origin: impl Into<String>, source: impl Into<String>) -> Self {
        Snippet {
            origin: origin.into(),
            line_start: 1,
            source: source.into(),
            annotations: Vec::new(),
            fold: false,
        }
    }

    pub fn with_line_start(mut self, line_start: usize) -> Self {
        self.line_start = line_start;
        self
    }

    pub fn with_annotation(mut self, label: Label) -> Self {
        self.annotations.push(label);
        self
    }

    pub fn with_annotations(mut self, labels: impl IntoIterator<Item = Label>) -> Self {
        self.annotations.extend(labels);
        self
    }

    pub fn with_fold(mut self, fold: bool) -> Self {
        self.fold = fold;
        self
    }
}
