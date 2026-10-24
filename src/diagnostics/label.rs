use crate::ast::{FileId, Span};
use crate::sourcemap::SourceMap;
use std::borrow::Cow;

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
///
/// The message is [`Cow<'a, str>`]: static text (`"duplicate definition"`)
/// borrows a `&'static str` with no allocation, while computed text (e.g.
/// `format!("expected {ty}")`) owns a `String`.  When `'a` is tied to a
/// source buffer's lifetime the borrow checker guarantees a source-slice
/// label cannot outlive the buffer it was cut from.
#[derive(Debug, Clone)]
pub struct Label<'a> {
    pub span: Span,
    pub message: Cow<'a, str>,
    pub kind: AnnotationKind,
}

impl<'a> Label<'a> {
    pub fn new(span: Span, message: impl Into<Cow<'a, str>>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Primary,
        }
    }

    pub fn secondary(span: Span, message: impl Into<Cow<'a, str>>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Secondary,
        }
    }

    pub fn note(span: Span, message: impl Into<Cow<'a, str>>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Note,
        }
    }

    pub fn help(span: Span, message: impl Into<Cow<'a, str>>) -> Self {
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
        self.kind.underline_char()
    }
}

impl AnnotationKind {
    /// The underline character for this annotation kind. Exposed on the kind
    /// itself so callers that synthesize an annotation (e.g. the renderer's
    /// primary span) do not need to build a throwaway [`Label`].
    pub fn underline_char(self) -> char {
        match self {
            AnnotationKind::Primary => '^',
            AnnotationKind::Secondary => '~',
            AnnotationKind::Note => '-',
            AnnotationKind::Help => ' ', // no underline, rendered as `| help: message`
        }
    }
}

// ── Byte-offset → line:column conversion ──────────────────────

/// A (line, column) position in source code.
///
/// Both fields are 0-based.  `col` is a **byte** offset from the start of the
/// line, *not* a display column: on lines containing tabs or wide characters
/// (CJK/emoji) the visual column is smaller.  Emitters that need display
/// columns must use the grapheme/width helpers instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePos {
    /// 0-based line index.
    pub line: usize,
    /// 0-based byte offset from the start of the line.
    pub col: usize,
}

/// Convert byte offset to human-readable "line:col" string (1-based).
///
/// Delegates to the [`SourceMap`]'s precomputed line table for `file`
/// rather than rescanning text, so offsets must be into that file.
pub fn byte_offset_to_string(map: &SourceMap, file: FileId, byte_offset: u32) -> String {
    let pos = map.file(file).line_col(byte_offset);
    format!("{}:{}", pos.line + 1, pos.col + 1)
}

// ── Snippet: a contiguous slice of source code with annotations ──

/// A contiguous slice of source code with its annotations, ready for
/// rendering.  Inspired by the `Snippet` type in `annotate-snippets-rs`.
///
/// Each `Snippet` represents one continuous region of source text (e.g. a
/// function body, a type definition) together with the labels / underlines
/// that should be drawn over it.
///
/// The source text is borrowed (`&'a str`) rather than owned: this forces the
/// caller to prove that the underlying source buffer outlives every snippet
/// constructed from it, preventing silent stale reads when incremental
/// compilation rewrites the buffer.
#[derive(Debug, Clone)]
pub struct Snippet<'a> {
    /// Display name / file path of the source (e.g. `"src/main.pn"`).
    pub origin: Cow<'a, str>,
    /// 1‑based line number of the first line in `source`.
    pub line_start: usize,
    /// The raw source text (may span multiple lines).
    pub source: &'a str,
    /// Annotations (labels) attached to this snippet.
    pub annotations: Vec<Label<'a>>,
    /// Whether to fold long runs of unannotated lines into `…`.
    pub fold: bool,
}

impl<'a> Snippet<'a> {
    pub fn new(origin: impl Into<Cow<'a, str>>, source: &'a str) -> Self {
        Snippet {
            origin: origin.into(),
            line_start: 1,
            source,
            annotations: Vec::new(),
            fold: false,
        }
    }

    pub fn with_line_start(mut self, line_start: usize) -> Self {
        assert!(
            line_start >= 1,
            "Snippet::with_line_start is 1-based; got {line_start}"
        );
        self.line_start = line_start;
        self
    }

    pub fn with_annotation(mut self, label: Label<'a>) -> Self {
        self.annotations.push(label);
        self
    }

    pub fn with_annotations(mut self, labels: impl IntoIterator<Item = Label<'a>>) -> Self {
        self.annotations.extend(labels);
        self
    }

    pub fn with_fold(mut self, fold: bool) -> Self {
        self.fold = fold;
        self
    }
}
