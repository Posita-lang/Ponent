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
