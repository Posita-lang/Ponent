use crate::ast::Span;
use smallvec::SmallVec;
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
    /// Identity used when merging label sets (`Diagnostic::augment`).
    ///
    /// Deliberately **excludes** `kind`: two labels on the same span with the same
    /// text are the same annotation even if one is primary and one secondary, and
    /// that is the rule the merge always applied.  Spelled as a named method
    /// rather than a derived `PartialEq` (which would include `kind`) so "label
    /// identity" has exactly one definition instead of being re-derived in the
    /// scan and the keyed-set paths.
    #[must_use]
    pub fn dedup_key(&self) -> (crate::ast::FileId, u32, u32, &str) {
        (
            self.span.file,
            self.span.start,
            self.span.end,
            &self.message,
        )
    }

    #[must_use]
    pub fn new(span: Span, message: impl Into<Cow<'a, str>>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Primary,
        }
    }

    #[must_use]
    pub fn secondary(span: Span, message: impl Into<Cow<'a, str>>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Secondary,
        }
    }

    #[must_use]
    pub fn note(span: Span, message: impl Into<Cow<'a, str>>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Note,
        }
    }

    /// An **inline help annotation** at a source position — rendered as
    /// `| help: message` beside the underlined source, with no underline of its
    /// own.
    ///
    /// Not to be confused with the diagnostic-level help text
    /// (`Diagnostic::help`, the humanizer's `help()`): that is one suggestion
    /// string for the whole diagnostic, while this is a positioned annotation
    /// that belongs in `Diagnostic::labels`.  A help *text* string placed here
    /// renders as a plausible-but-wrong source annotation.
    #[must_use]
    pub fn help(span: Span, message: impl Into<Cow<'a, str>>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Help,
        }
    }

    #[must_use]
    pub fn with_kind(mut self, kind: AnnotationKind) -> Self {
        self.kind = kind;
        self
    }

    /// The underline character for this annotation kind.
    ///
    /// `None` means the kind has no underline at all (see
    /// [`AnnotationKind::underline_char`]) — a renderer must then take the
    /// no-underline path, instead of emitting a run of spaces that silently
    /// corrupts the frame.
    pub fn underline_char(&self) -> Option<char> {
        self.kind.underline_char()
    }
}

impl AnnotationKind {
    /// The underline character for this annotation kind, or `None` when the kind
    /// is rendered without one.
    ///
    /// Exposed on the kind itself so callers that synthesize an annotation
    /// (e.g. the renderer's primary span) do not need to build a throwaway
    /// [`Label`].  Returning `Option` rather than `' '` for
    /// [`AnnotationKind::Help`] makes "special-case this" a compile-time
    /// obligation for every renderer: the old `' '` placeholder let a uniform
    /// iteration emit invisible spaces.
    pub fn underline_char(self) -> Option<char> {
        match self {
            AnnotationKind::Primary => Some('^'),
            AnnotationKind::Secondary => Some('~'),
            AnnotationKind::Note => Some('-'),
            // No underline: rendered as `| help: message`.
            AnnotationKind::Help => None,
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
///
/// Note: [`Snippet::line_start`](crate::diagnostics::label::Snippet::line_start)
/// is **1-based**, unlike `line` here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePos {
    /// 0-based line index.
    pub line: usize,
    /// 0-based byte offset from the start of the line.
    pub col: usize,
}

/// `line:col`, both **1-based** — the form every renderer prints.
///
/// The `+1` and the byte-column caveat above are the contract, so they live in
/// one place instead of being re-derived per renderer.  `renderers::short`
/// deliberately keeps its integer path (it fuses the digit widths into its
/// capacity computation); everything else should format through this.
impl std::fmt::Display for SourcePos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line + 1, self.col + 1)
    }
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
    /// **1-based** line number of the first line in `source` — deliberately the
    /// opposite convention from [`SourcePos::line`], which is 0-based.  The two
    /// exist for different consumers (snippets print human line numbers;
    /// positions are index arithmetic), so the conversion happens once at the
    /// boundary rather than inside either type.
    pub line_start: usize,
    /// The raw source text (may span multiple lines).
    pub source: &'a str,
    /// Annotations (labels) attached to this snippet.
    ///
    /// `SmallVec<[Label; 2]>`, not `Vec`: an annotated snippet carries one or two
    /// labels in every realistic shape, so the inline case skips the allocation
    /// entirely.  This is the same reasoning `Diagnostic::labels` uses with
    /// `CovariantVec<_, 1>` — the inline capacity differs only because
    /// `Diagnostic` has a 384-byte size guard while `Snippet` is built and
    /// consumed within one render.
    pub annotations: SmallVec<[Label<'a>; 2]>,
    /// Whether to fold long runs of unannotated lines into `…`.
    pub fold: bool,
}

impl<'a> Snippet<'a> {
    #[must_use]
    pub fn new(origin: impl Into<Cow<'a, str>>, source: &'a str) -> Self {
        Snippet {
            origin: origin.into(),
            line_start: 1,
            source,
            annotations: SmallVec::new(),
            fold: false,
        }
    }

    #[must_use]
    pub fn with_line_start(mut self, line_start: usize) -> Self {
        debug_assert!(
            line_start >= 1,
            "Snippet::with_line_start is 1-based; got {line_start}"
        );
        self.line_start = line_start.max(1);
        self
    }

    #[must_use]
    pub fn with_annotation(mut self, label: Label<'a>) -> Self {
        self.annotations.push(label);
        self
    }

    #[must_use]
    pub fn with_annotations(mut self, labels: impl IntoIterator<Item = Label<'a>>) -> Self {
        self.annotations.extend(labels);
        self
    }

    #[must_use]
    pub fn with_fold(mut self, fold: bool) -> Self {
        self.fold = fold;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::FileId;

    fn span(a: u32, b: u32) -> Span {
        Span::new(FileId::DUMMY, a, b)
    }

    /// `Display` is the single place the `+1` lives, so it is the single place a
    /// renderer's 0-based→1-based drift can be caught.
    #[test]
    fn source_pos_display_is_one_based() {
        assert_eq!(
            SourcePos { line: 0, col: 0 }.to_string(),
            "1:1",
            "file start"
        );
        assert_eq!(SourcePos { line: 4, col: 7 }.to_string(), "5:8");
        // Start of the second line (not the newline of the first): pins the
        // line-index convention against a producer that counts newlines-as-lines.
        assert_eq!(SourcePos { line: 1, col: 0 }.to_string(), "2:1");
    }

    /// Byte columns, not display columns: on a tab-indented line the two differ,
    /// and a renderer switching to display columns must fail here.
    #[test]
    fn source_pos_col_is_a_byte_offset() {
        let line = "\tx";
        let byte_col = line.find('x').unwrap();
        let pos = SourcePos {
            line: 0,
            col: byte_col,
        };
        // The *display* column of `x` is 5 (the tab advances to the next 4-column stop),
        // so "1:2" is the guard: a `Display` that printed display columns would emit
        // "1:5".  (An `assert_ne!(pos.col + 1, 5)` used to sit here — it tested the
        // literal this test constructs, the `x == x` tautology `stylesheet.rs` records as
        // a past bug — and deleting it left the body with no assertion at all, which is
        // why this pins the actual `Display` output.)
        assert_eq!(pos.to_string(), "1:2");
    }

    /// The glyph table is consumed by every renderer; swapping `~`/`-` must be a
    /// loud failure.
    #[test]
    fn underline_chars_are_pinned() {
        assert_eq!(AnnotationKind::Primary.underline_char(), Some('^'));
        assert_eq!(AnnotationKind::Secondary.underline_char(), Some('~'));
        assert_eq!(AnnotationKind::Note.underline_char(), Some('-'));
        // `None`, not `' '`: a placeholder glyph let a uniform iteration emit a
        // run of invisible spaces.  The type now forces the no-underline path.
        assert_eq!(AnnotationKind::Help.underline_char(), None);
        let label = Label::secondary(span(0, 1), "s");
        assert_eq!(label.underline_char(), Some('~'));
    }

    #[test]
    fn constructors_set_the_right_kind_and_keep_borrows() {
        // A borrowed message must stay borrowed: an `Into<Cow>` impl that
        // silently called `to_string()` would allocate on every static label.
        let borrowed: Cow<'static, str> = Cow::Borrowed("static text");
        let label = Label::new(span(0, 1), borrowed.clone());
        assert!(matches!(label.message, Cow::Borrowed(_)), "{label:?}");
        assert_eq!(label.kind, AnnotationKind::Primary);
        assert_eq!(
            Label::secondary(span(0, 1), borrowed.clone()).kind,
            AnnotationKind::Secondary
        );
        assert_eq!(
            Label::note(span(0, 1), borrowed.clone()).kind,
            AnnotationKind::Note
        );
        assert_eq!(Label::help(span(0, 1), borrowed).kind, AnnotationKind::Help);
    }

    /// Debug-only: a 0-based line start is a producer bug and must be loud.
    /// `#[should_panic]` rather than `catch_unwind` so it still means something
    /// under `panic = "abort"`, and cannot pass by accepting either outcome.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "line")]
    fn snippet_line_start_zero_trips_the_assert() {
        swallow_it::swallow!(Snippet::new("f.pn", "fn main() {}\n").with_line_start(0));
    }

    #[test]
    fn snippet_line_start_is_clamped_and_one_based() {
        let src = "fn main() {}\n";
        // Release-only: the clamp is deliberately silent there (see the
        // `debug_assert` in `with_line_start`), so pin it.
        #[cfg(not(debug_assertions))]
        assert_eq!(
            Snippet::new("f.pn", src).with_line_start(0).line_start,
            1,
            "release clamps a 0-based line start to 1"
        );
        assert_eq!(
            Snippet::new("f.pn", src).line_start,
            1,
            "default is 1-based"
        );
        assert_eq!(Snippet::new("f.pn", src).with_line_start(5).line_start, 5);
        assert_eq!(
            Snippet::new("f.pn", src)
                .with_line_start(1)
                .with_annotations([Label::new(span(0, 2), "a"), Label::note(span(3, 4), "b")])
                .annotations
                .len(),
            2
        );
    }
}
