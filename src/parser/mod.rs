//! Posita parser: core token pipeline, checkpoint, and module dispatch.
//!
//! This file defines the `Parser` struct and the low-level token handling
//! machinery. All grammar rules are implemented in sibling modules.

use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::symbol::Symbol;
use bitflags::bitflags;
use logos::Logos;
use std::mem;

// ── Track‑A (Engineering Canon) ──────────────────────────────────────────────
// Keyword-as-identifier policy (see `Token::as_ident_symbol` in lexer.rs):
//
// Posita keywords are *not* reserved in path position.  Any keyword token may
// appear after `::` — e.g. `T::default()`, `T::move()`, `Module::type` — because
// method / variant / associated-type names are always user-chosen identifiers
// that happen to overlap with keyword strings.
//
// When adding a new keyword:
//   1. Add it to the lexer (`Token` enum in lexer.rs).
//   2. Add it to `Token::as_ident_symbol()`.
//   3. This module will *automatically* accept it after `::` in paths, patterns,
//      and expressions — no parser changes needed.
//   4. Add a parser test that exercises the keyword in a qualified path.
// ──────────────────────────────────────────────────────────────────────────────

// -----------------------------------------------------------------------------
// Sub‑modules (grammar domains)
// -----------------------------------------------------------------------------
mod attr;
mod const_arg;
mod expr;
mod generics;
mod item;
mod pattern;
mod recovery;
mod stmt;
mod ty;
mod type_def;

#[cfg(test)]
mod tests;

// -----------------------------------------------------------------------------
// Public exports
// -----------------------------------------------------------------------------
pub(crate) use recovery::edit_distance_limited_into;
pub use recovery::{KeywordContext, did_you_mean_keyword};

// -----------------------------------------------------------------------------
// Core types
// -----------------------------------------------------------------------------

/// Whether parser recovery is allowed at a given call site (rustc-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recovery {
    Allowed,
    Forbidden,
}

/// A token that the parser expects, used for error recovery.
#[derive(Debug, Clone)]
pub struct ExpectedToken {
    /// The expected token kind.
    pub tok: Token,
    /// A human-readable description of what was expected.
    pub desc: &'static str,
}

impl ExpectedToken {
    pub fn new(tok: Token, desc: &'static str) -> Self {
        ExpectedToken { tok, desc }
    }
}

/// Result of parser recovery (rustc-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recovered {
    No,
    Yes(ErrorGuaranteed),
}

/// A token representing that an error has been reported (rustc-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorGuaranteed(u32);

impl ErrorGuaranteed {
    pub fn new_unchecked(id: u32) -> Self {
        ErrorGuaranteed(id)
    }
}

bitflags! {
    /// Parse‑time context restrictions that influence how tokens are interpreted.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct ParseRestrictions: u8 {
        const NO_STRUCT_LITERAL = 1 << 0;
        const ALLOW_TYPE_PARAMS = 1 << 1;
        const STMT_EXPR         = 1 << 2;
        const VALUE_BLOCK       = 1 << 3;
        /// When set, comparison operators (>, >=, <, <=, ==, !=) are
        /// treated as expression terminators rather than infix operators.
        /// Used inside generic argument parsing so that a const expression
        /// like `Val >> 2` does not consume the closing `>`.
        const NO_COMPARISON     = 1 << 4;
    }
}

/// A fully-buffered token: the token itself plus its source span.
/// All tokens are lexed upfront into a `Vec<SpannedToken>`, avoiding
/// the lifetime coupling of `logos::Lexer` and enabling arbitrary
/// lookahead without cloning the lexer state. This mirrors the
/// `TokenCursor` / `TokenStream` architecture used by rustc.
#[derive(Debug, Clone)]
struct SpannedToken {
    token: Token,
    span: Span,
}

/// A snapshot of the parser's token-stream state for backtracking.
/// Created by [`Parser::checkpoint`] and restored by [`Parser::restore`].
/// Inspired by rustc's `Parser::checkpoint` / `Parser::rewind`.
struct Checkpoint {
    cursor: usize,
    peeked: Option<Result<Token, ()>>,
    pending: Vec<Token>,
    progress: u64,
    sync_stall: Option<u64>,
}

// -----------------------------------------------------------------------------
// Parser struct
// -----------------------------------------------------------------------------

pub struct Parser<'input> {
    arena: &'input bumpalo::Bump,
    tokens: Vec<SpannedToken>,
    cursor: usize,
    peeked: Option<Result<Token, ()>>,
    pending: Vec<Token>,
    pub diagnostics: Vec<Diagnostic>,
    recursion_depth: usize,
    max_recursion_depth: usize,
    restrictions: ParseRestrictions,
    /// Once set after a top-level parse error, suppresses subsequent
    /// "unexpected token at top level" diagnostics until one item
    /// parses successfully.  Prevents the cascade of 20+ errors from
    /// a single typo like `defw`.
    cascade_suppressed: bool,
    /// Whether parser recovery is allowed (rustc-style).
    recovery: Recovery,
    /// Span of the last unexpected token, used to detect infinite recovery loops.
    last_unexpected_token_span: Option<Span>,
    /// Stack of expected token types for better error messages.
    expected_token_types: Vec<ExpectedToken>,
    /// Monotonic count of tokens consumed by `advance` — detects
    /// zero-progress error-recovery loops (the stall detector).
    progress: u64,
    /// Progress value when `synchronize` last stopped before a protected
    /// statement-start keyword; if reached again at the same progress,
    /// recovery force-consumes a token (no net progress was made).
    sync_stall: Option<u64>,
}

impl<'input> Parser<'input> {
    // -------------------------------------------------------------------------
    // Construction and token pipeline
    // -------------------------------------------------------------------------

    pub fn new(source: &str, arena: &'input bumpalo::Bump) -> Self {
        let mut tokens = Vec::new();
        let mut lexer = Token::lexer(source);
        let mut diagnostics = Vec::new();
        loop {
            let (token, span_range) = match lexer.next() {
                Some(Ok(Token::WhitespaceOrComment)) => continue,
                Some(Ok(token)) => (token, lexer.span()),
                Some(Err(())) => {
                    let bad_span = lexer.span();
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "unexpected character '{}'",
                            &source[bad_span.start..bad_span.end.min(source.len())]
                        ))
                        .with_span(Span::new(bad_span.start, bad_span.end)),
                    );
                    continue;
                }
                None => break,
            };
            tokens.push(SpannedToken {
                token,
                span: Span::new(span_range.start, span_range.end),
            });
        }
        Parser {
            arena,
            tokens,
            cursor: 0,
            peeked: None,
            pending: Vec::new(),
            diagnostics,
            recursion_depth: 0,
            max_recursion_depth: 256,
            restrictions: ParseRestrictions::STMT_EXPR,
            cascade_suppressed: false,
            recovery: Recovery::Allowed,
            last_unexpected_token_span: None,
            expected_token_types: Vec::new(),
            progress: 0,
            sync_stall: None,
        }
    }

    /// Allocate an AST node in the parser's arena, returning a SHARED
    /// `&'input` reference (bumpalo::Bump::alloc returns `&mut T`; the
    /// AST fields expect shared references).
    fn alloc_shared<T>(arena: &'input bumpalo::Bump, val: T) -> &'input T {
        arena.alloc(val)
    }

    fn next_token(&mut self) -> Result<Token, ()> {
        // Check the pending stack first (e.g. Shr-split Gt).
        if let Some(tok) = self.pending.pop() {
            return Ok(tok);
        }
        if self.cursor < self.tokens.len() {
            let st = &self.tokens[self.cursor];
            self.cursor += 1;
            Ok(st.token.clone())
        } else {
            Err(())
        }
    }

    fn peek(&mut self) -> &Result<Token, ()> {
        if self.peeked.is_none() {
            self.peeked = Some(self.next_token());
        }
        self.peeked.as_ref().expect("peek called before next_token")
    }

    fn advance(&mut self) -> Result<Token, ()> {
        let tok = match self.peeked.take() {
            Some(tok) => tok,
            None => self.next_token(),
        };
        if tok.is_ok() {
            self.progress += 1;
        }
        tok
    }

    /// Return the token AFTER the one `peek()` would return
    /// (lookahead-2), without consuming anything.
    ///
    /// `peek()` consumes pending tokens FIRST (`next_token` pops the
    /// pending stack), so when `pending` is non-empty the "next" token
    /// is the NEW stack top after the pop — or the token-stream position
    /// when the stack empties (the pop does not advance `cursor`).
    /// `peeked` caches the token `peek()` already consumed; once it is
    /// populated the pending stack is necessarily empty (a prior
    /// `next_token` drained it), so the two branches never overlap.
    fn peek_next(&mut self) -> Option<Token> {
        // Invariant: `pending` holds at most ONE token at any time — every
        // `pending.push` (`>>` split in `expect_gt`, `<<` split in the
        // nested-projection path) is immediately followed by consuming it
        // via `expect_gt`.  Therefore a populated `peeked` (which would
        // have popped the stack) and a non-empty `pending` cannot
        // co-occur.  The assert guards this invariant: if a future
        // extension ever pushes >=2 tokens, the pending branch below
        // (which assumes `peeked` is empty) must be revisited.
        debug_assert!(
            !(self.peeked.is_some() && !self.pending.is_empty()),
            "peek_next: peeked cache and pending stack cannot both be non-empty"
        );
        if let Some(tok) = self.pending.last() {
            if self.pending.len() > 1 {
                return Some(self.pending[self.pending.len() - 2].clone());
            }
            return self.tokens.get(self.cursor).map(|st| st.token.clone());
        }
        self.peek();
        self.tokens.get(self.cursor).map(|st| st.token.clone())
    }

    fn span(&self) -> Span {
        if self.cursor > 0 && self.cursor - 1 < self.tokens.len() {
            self.tokens[self.cursor - 1].span
        } else if self.cursor < self.tokens.len() {
            self.tokens[self.cursor].span
        } else if !self.tokens.is_empty() {
            self.tokens[self.tokens.len() - 1].span
        } else {
            Span::new(0, 0)
        }
    }

    // -------------------------------------------------------------------------
    // Expect / consume helpers
    // -------------------------------------------------------------------------

    pub fn expect(&mut self, expected: Token) -> Result<Token, Diagnostic> {
        if matches!(self.peek(), Ok(tok) if tok == &expected) {
            self.advance().ok();
            Ok(expected)
        } else {
            let found = match self.peek() {
                Ok(tok) => format!("{:?}", tok),
                Err(()) => "end of file".to_string(),
            };
            Err(
                Diagnostic::error(format!("expected {:?}, found {}", expected, found))
                    .with_code_str("E001")
                    .with_help(format!(
                        "expected `{:?}` but saw `{}` — check for missing or extra tokens",
                        expected, found
                    ))
                    .with_suggestion(format!(
                        "try adding `{:?}` before the `{}`",
                        expected, found
                    ))
                    .with_span(self.span()),
            )
        }
    }

    /// Consume `>` potentially by splitting `>>` into two `>` tokens.
    /// This mirrors rustc's `break_and_eat` approach: in generic contexts,
    /// `>>` is ambiguous — it could be a right-shift or two closing angle brackets.
    /// `expect_gt()` greedily treats it as the latter, pushing the second `>`
    /// onto the pending stack for the outer generic level.
    fn expect_gt(&mut self) -> Result<(), Diagnostic> {
        match self.peek() {
            Ok(Token::Gt) => {
                self.advance().ok();
                Ok(())
            }
            Ok(Token::Shr) => {
                self.advance().ok();
                self.pending.push(Token::Gt);
                Ok(())
            }
            _ => Err(Diagnostic::error("expected '>'")
                .with_code_str("E004")
                .with_span(self.span())),
        }
    }

    fn expect_one_of(&mut self, edible: &[Token], inedible: &[Token]) -> Result<Recovered, ()> {
        for exp in edible {
            if matches!(self.peek(), Ok(tok) if tok == exp) {
                self.advance().ok();
                return Ok(Recovered::No);
            }
        }
        for exp in inedible {
            if matches!(self.peek(), Ok(tok) if tok == exp) {
                return Ok(Recovered::No);
            }
        }
        let expected = edible.first().or_else(|| inedible.first()).cloned();
        match self.unexpected_err(&expected) {
            Some(recovered) => Ok(recovered),
            None => Err(()),
        }
    }

    // -------------------------------------------------------------------------
    // Checkpoint‑based backtracking
    // -------------------------------------------------------------------------

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            cursor: self.cursor,
            peeked: self.peeked.clone(),
            pending: self.pending.clone(),
            progress: self.progress,
            sync_stall: self.sync_stall,
        }
    }

    fn restore(&mut self, cp: &Checkpoint) {
        self.cursor = cp.cursor;
        self.peeked = cp.peeked.clone();
        self.pending = cp.pending.clone();
        self.progress = cp.progress;
        self.sync_stall = cp.sync_stall;
    }

    /// Try parsing with `f`. If `f` succeeds, keep the result and the
    /// advanced parser state. If `f` fails, restore the parser state
    /// to before `f` was called and return the error.
    fn try_parse<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, Diagnostic>,
    ) -> Result<T, Diagnostic> {
        let cp = self.checkpoint();
        match f(self) {
            Ok(val) => Ok(val),
            Err(e) => {
                self.restore(&cp);
                Err(e)
            }
        }
    }

    /// Try parsing with `f`. If `f` succeeds, keep the result. If `f`
    /// fails, restore the parser state and call `fallback` instead.
    ///
    /// This is the primary pattern for ambiguous constructs: try one
    /// interpretation, and if it doesn't work, try the other.
    fn try_parse_or<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, Diagnostic>,
        fallback: impl FnOnce(&mut Self) -> Result<T, Diagnostic>,
    ) -> Result<T, Diagnostic> {
        let cp = self.checkpoint();
        match f(self) {
            Ok(val) => Ok(val),
            Err(_) => {
                self.restore(&cp);
                fallback(self)
            }
        }
    }

    // -------------------------------------------------------------------------
    // Restriction management
    // -------------------------------------------------------------------------

    fn with_restrictions<T>(
        &mut self,
        extra: ParseRestrictions,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let old = self.restrictions;
        self.restrictions |= extra;
        let result = f(self);
        self.restrictions = old;
        result
    }

    fn without_restrictions<T>(
        &mut self,
        remove: ParseRestrictions,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let old = self.restrictions;
        self.restrictions -= remove;
        let result = f(self);
        self.restrictions = old;
        result
    }

    // -------------------------------------------------------------------------
    // Recovery mode
    // -------------------------------------------------------------------------

    fn set_recovery(&mut self, recovery: Recovery) -> Recovery {
        let old = self.recovery;
        self.recovery = recovery;
        old
    }

    fn with_recovery<F, T>(&mut self, recovery: Recovery, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        let old = self.set_recovery(recovery);
        let result = f(self);
        self.recovery = old;
        result
    }

    fn push_expected(&mut self, tok: Token, desc: &'static str) {
        self.expected_token_types
            .push(ExpectedToken::new(tok, desc));
    }

    fn pop_expected(&mut self) {
        self.expected_token_types.pop();
    }

    // -------------------------------------------------------------------------
    // Keyword‑to‑ident conversion (path positions)
    // -------------------------------------------------------------------------

    fn keyword_to_ident(&self, tok: &Token) -> Option<Symbol> {
        tok.as_ident_symbol()
    }

    // -------------------------------------------------------------------------
    // Top‑level entry point
    // -------------------------------------------------------------------------

    /// Parse the full program source into an AST.
    ///
    /// # Errors
    ///
    /// Returns `Err(Vec<Diagnostic>)` with one or more parse errors if the
    /// source contains syntax errors.  The parser attempts to recover and
    /// continue after each error, so multiple diagnostics may be returned.
    #[must_use]
    pub fn parse_program(&mut self) -> Result<Program<'input>, Vec<Diagnostic>> {
        let start = self.span().start;
        let mut items = Vec::new();
        loop {
            match self.peek() {
                Err(()) => break,
                _ => match self.parse_item() {
                    Ok(item) => {
                        if !matches!(item, Stmt::Error(_)) {
                            self.cascade_suppressed = false;
                        }
                        items.push(item);
                    }
                    Err(diag) => {
                        self.diagnostics.push(diag);
                        // Top-level recovery: skip to the next genuine
                        // top-level keyword (def/type/trait/impl/import/…)
                        // WITHOUT swallowing it — those are legal item
                        // starts, so the outer loop parses them fresh.
                        // (`synchronize` would consume them; it is only
                        // correct for the block-level statement loop.)
                        self.skip_to_next_top_level();
                    }
                },
            }
        }
        let end = self.span().end;
        let span = Span::new(start, end);
        if self.diagnostics.is_empty() {
            Ok(Program { items, span })
        } else {
            Err(mem::take(&mut self.diagnostics))
        }
    }
}
