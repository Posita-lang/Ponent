//! The doc/trivia side channel (two-channel lexing).
//!
//! The parser's grammar stream intentionally keeps trivia skipped and
//! routes `///`/`//!` doc comments into item `doc` fields for
//! compatibility.  This module is the *second* channel: the same
//! upfront lex also records every doc comment (anchored to the next
//! semantic token) and every skipped trivia run (with its span), so
//! that docs/formatter/IDE features can consume them WITHOUT any
//! semantic query ever depending on source text or positions.
//!
//! Everything here is position-bearing by design and therefore NEVER
//! takes part in `StableHash` fingerprints — see the `StableHash`
//! contract in `hir::query`.

use crate::ast::{FileId, Span};

/// A semantic-token index in the parser's grammar stream — the same
/// numbering the doc anchors use.  Distinct from [`ByteOffset`]: mixing a
/// token index into a byte position (or vice versa) is a type error, not a
/// silent misindexing.  Never hashed (position channel).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenIndex(usize);

impl TokenIndex {
    pub const ZERO: TokenIndex = TokenIndex(0);

    #[inline]
    pub const fn new(raw: usize) -> Self {
        TokenIndex(raw)
    }

    /// The underlying stream position (for `tokens[idx]` indexing and
    /// arithmetic at the parser's single access point).
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

/// A byte offset into one source file — [`Span`]'s raw currency.  Distinct
/// from [`TokenIndex`] for the same reason that one is distinct from the
/// other: the position sidetable maps `TokenIndex → (ByteOffset, ByteOffset)`
/// and the types must not be interchangeable.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteOffset(u32);

impl ByteOffset {
    #[inline]
    pub const fn new(raw: u32) -> Self {
        ByteOffset(raw)
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Doc-channel + trivia-channel contents of one source file, captured
/// during the parser's upfront lex.
#[derive(Debug, Clone, PartialEq)]
pub struct DocMap<'src> {
    /// Doc comments (`///`, `//!`) in source order, anchored to the
    /// index of the next semantic token in the parser's stream.
    pub docs: Vec<DocEntry<'src>>,
    /// Skipped trivia runs (whitespace / comments), in source order.
    pub trivia: Vec<TriviaEntry>,
    /// POSITION-CHANNEL SIDETABLE (#3): byte span of every semantic
    /// token, indexed by the token index used for doc anchors.  This
    /// is the ONLY table mapping token index → byte offset; it lives
    /// on the position side of the two-channel design and, like
    /// `trivia`, NEVER takes part in `StableHash` — fingerprinted
    /// values must not embed positions (stable-hash audit).
    ///
    /// Offsets are `(start, end)` in the file identified by `file`,
    /// mirroring [`Span`]'s `u32` byte-offset representation.
    pub token_offsets: Vec<(ByteOffset, ByteOffset)>,
    /// The file this doc map was captured from; stamped into every
    /// span built from the position sidetable so `token_span` can
    /// reconstruct a `Span` without carrying the id per offset.
    pub file: FileId,
}

impl<'src> Default for DocMap<'src> {
    fn default() -> Self {
        Self {
            docs: Vec::new(),
            trivia: Vec::new(),
            token_offsets: Vec::new(),
            file: FileId::DUMMY,
        }
    }
}

/// One doc comment captured into the doc channel.
#[derive(Debug, Clone, PartialEq)]
pub struct DocEntry<'src> {
    /// Trimmed comment body (`/// @spec foo` → `@spec foo`).
    pub text: &'src str,
    /// Index of the next semantic token in the parser's token stream
    /// at the time this comment was seen — the item it documents.
    pub anchor: TokenIndex,
    /// `//!` module-level doc (as opposed to `///`).
    pub is_module: bool,
    /// Byte span of the raw comment (diagnostics/IDE only; never
    /// hashed).
    pub span: Span,
}

/// One skipped trivia run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriviaEntry {
    pub span: Span,
    pub kind: TriviaKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriviaKind {
    Whitespace,
    Comment,
}

impl<'src> DocMap<'src> {
    /// POSITION CHANNEL: byte span of semantic token `index`
    /// (the same index used for doc anchors).  `None` if out of range.
    pub fn token_span(&self, index: TokenIndex) -> Option<Span> {
        self.token_offsets
            .get(index.as_usize())
            .map(|&(s, e)| Span::new(self.file, s.as_u32(), e.as_u32()))
    }

    /// Doc lines anchored strictly inside `start..end` (an item's
    /// span).  Text only — spans stay behind.
    pub fn doc_lines_between(&self, start: u32, end: u32) -> impl Iterator<Item = &str> + '_ {
        self.docs
            .iter()
            .filter(move |d| d.span.start >= start && d.span.end <= end)
            .map(|d| d.text)
    }

    /// Doc lines whose ANCHOR token lies inside `start..end` (an item's
    /// span).  A `///` comment precedes the item it documents, so the
    /// comment's own span is NOT inside the item span — the anchor
    /// (the next semantic token, i.e. the item's `fn`/attribute token)
    /// is.  Module docs (`//!`) never anchor to an item and are
    /// excluded.  This is the lookup the `DocTokens` query uses.
    pub fn doc_lines_anchored_in(&self, start: u32, end: u32) -> impl Iterator<Item = &str> + '_ {
        self.docs
            .iter()
            .filter(|d| !d.is_module)
            .filter(move |d| {
                self.token_span(d.anchor)
                    .is_some_and(|s| s.start >= start && s.end <= end)
            })
            .map(|d| d.text)
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// #3: every semantic token gets a byte span in the position
    /// sidetable, in stream order; doc comments and trivia produce no
    /// entries.  Anchors stay consistent: a doc entry's `anchor` index
    /// resolves to the byte span of the token that follows it.
    #[test]
    fn test_token_offsets_sidetable() {
        // Top-level `let` is rejected (E003), so the two `let` statements
        // live inside a top-level `comptime` block.
        let source = "/// doc\ncomptime {\nlet x = 1;\n  // trivia\nlet y = 2;\n}";
        let bump = bumpalo::Bump::new();
        let arena = crate::arena::SyncArena::new(&bump);
        let program = crate::parser::Parser::new(source, FileId::DUMMY, &arena)
            .parse_program()
            .expect("parse must succeed");
        let dm = &program.doc_map;

        // 13 semantic tokens: comptime { let x = 1 ; let y = 2 ; }
        assert_eq!(dm.token_offsets.len(), 13);
        // Trivia: every inter-token gap is recorded (whitespace runs
        // included); the source has at least the `// trivia` comment.
        assert!(
            dm.trivia.iter().any(|t| t.kind == TriviaKind::Comment),
            "the `// trivia` comment must appear as a Comment trivia run"
        );

        // Anchor 0 (the `/// doc`'s anchor) is the first token of the item
        // the doc attaches to — `comptime`.
        let item_head = dm
            .token_span(TokenIndex::new(0))
            .expect("anchor 0 must resolve");
        assert_eq!(
            &source[item_head.start as usize..item_head.end as usize],
            "comptime"
        );

        // The first `let` is token 2.
        let first_let = dm
            .token_span(TokenIndex::new(2))
            .expect("anchor 2 must resolve");
        assert_eq!(
            &source[first_let.start as usize..first_let.end as usize],
            "let"
        );

        // The second `let` sits after the trivia run (index 7).
        let second_let = dm
            .token_span(TokenIndex::new(7))
            .expect("anchor 7 must resolve");
        assert_eq!(
            &source[second_let.start as usize..second_let.end as usize],
            "let"
        );

        // Out of range → None.
        assert!(dm.token_span(TokenIndex::new(13)).is_none());
        assert!(dm.token_span(TokenIndex::new(usize::MAX)).is_none());
    }
}
