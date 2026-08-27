//! Error recovery, token synchronization, and `did_you_mean` suggestions.

use super::*;
use crate::lexer::Token;

impl<'input> Parser<'input> {
    /// Generate a diagnostic for an unexpected token, optionally with recovery.
    pub(super) fn unexpected_err(&mut self, expected: &Option<Token>) -> Option<Recovered> {
        let span = self.span();
        let found = self.peek().clone();
        let found_str = match &found {
            Ok(tok) => format!("{:?}", tok),
            Err(()) => "end of file".to_string(),
        };
        let expected_str = if let Some(tok) = expected {
            format!("{:?}", tok)
        } else if !self.expected_token_types.is_empty() {
            self.expected_token_types.last().expect("expected_token_types should be non-empty when formatting error — push_expected was not called before expect_one_of").desc.to_string()
        } else {
            "something".to_string()
        };

        let diag = Diagnostic::error(format!("expected {}, found {}", expected_str, found_str))
            .with_span(span);
        self.diagnostics.push(diag);

        if self.recovery == Recovery::Forbidden {
            return None;
        }

        if self.last_unexpected_token_span == Some(span) {
            return None;
        }
        self.last_unexpected_token_span = Some(span);
        Some(Recovered::Yes(ErrorGuaranteed::new_unchecked(0)))
    }

    /// Synchronize the token stream after a parse error.
    pub(super) fn synchronize(&mut self) {
        // NOTE: every call site invokes this AFTER a parse error has been
        // reported (`Err(diag) => { push(diag); synchronize(); }`), so the
        // "consume only after an error" discipline is guaranteed by the
        // callers — no extra error-state tracking is needed here.
        //
        // Consume semicolon if present (it's a statement TERMINATOR — not
        // a statement start — so consuming it cannot skip the next valid
        // statement's first token; the next statement starts fresh).
        if matches!(self.peek(), Ok(Token::Semicolon)) {
            self.advance().ok();
            self.sync_stall = None;
            return;
        }
        // Skip tokens silently until we hit a sync token or EOF.
        // The original error has already been reported by `expect` or `expect_one_of`;
        // synchronize is only responsible for advancing the token stream to a safe
        // point so that subsequent parsing can continue.  Emitting additional
        // diagnostics here would cascade noise over the real error.
        loop {
            match self.peek() {
                Err(()) => return,
                // A block's closing brace belongs to the enclosing block —
                // stop before it so `parse_block_inner`'s `RBrace => break`
                // guard ends the block cleanly (consuming it would let the
                // loop keep parsing tokens after the block's close).
                Ok(Token::RBrace) => return,
                // A statement-start keyword begins the NEXT statement —
                // stop before it instead of swallowing it, otherwise the
                // next statement's first token is lost and the follow-up
                // diagnostics are distorted (e.g. `set x = 1 + <err>; set y
                // = 5` must keep its `set y = 5` as a variable def).
                //
                // Stall detector: the protected stop TRUSTS the statement
                // loop to consume the token next.  If recovery previously
                // stopped before a protected token at the SAME progress
                // value, the statement loop made no net consumption — force
                // consume one token so forward progress is guaranteed even
                // when a statement arm cannot consume its keyword.
                Ok(tok) if Self::is_stmt_start_keyword(tok) => {
                    let current = self.progress;
                    if self.sync_stall == Some(current) {
                        self.advance().ok();
                        self.sync_stall = None;
                        return;
                    }
                    self.sync_stall = Some(current);
                    return;
                }
                // Sync tokens that cannot start a statement in this
                // position — consume them so error recovery makes progress
                // instead of spinning forever on the same token (the
                // stuck-test regression: `def main() { type R = Int<32>; }`
                // looped on `Token::Type`).
                Ok(tok) if Self::is_sync_token(tok) => {
                    self.advance().ok();
                    self.sync_stall = None;
                    return;
                }
                _ => {
                    self.advance().ok();
                    self.sync_stall = None;
                }
            }
        }
    }

    /// Skip tokens that are ONLY valid inside a function body (Return,
    /// While, For, If, etc.) and stop at genuine top-level keywords or EOF.
    /// Set/Let ARE valid at top level (global variable declarations), so
    /// they stop the skip (i.e. are NOT swallowed).
    pub(super) fn skip_to_next_top_level(&mut self) {
        loop {
            match self.peek() {
                Ok(Token::Def)
                | Ok(Token::Type)
                | Ok(Token::Trait)
                | Ok(Token::Impl)
                | Ok(Token::Constraint)
                | Ok(Token::Edition)
                | Ok(Token::Import)
                | Ok(Token::From)
                | Ok(Token::Extern)
                | Ok(Token::Comptime)
                | Ok(Token::Async)
                | Ok(Token::At)
                | Ok(Token::Set)
                | Ok(Token::Let)
                | Ok(Token::Layout) => return,
                Err(()) => return,
                _ => {
                    self.advance().ok();
                }
            }
        }
    }

    /// Constant-time sync-token check: the old `sync_tokens`
    /// array was linearly scanned per skipped token during error recovery —
    /// O(skipped × count).  A `matches!` compiles to a kind switch with no
    /// payload comparison.
    fn is_sync_token(tok: &Token) -> bool {
        matches!(
            tok,
            Token::Semicolon
                | Token::RBrace
                | Token::Def
                | Token::Set
                | Token::Let
                | Token::Type
                | Token::Import
                | Token::From
                | Token::Extern
                | Token::Edition
                | Token::At
                | Token::Comptime
                | Token::Generate
                | Token::Async
        )
    }

    /// Whether `tok` can begin a valid statement via `parse_stmt`
    /// (block context).  Error recovery stops BEFORE these tokens so the
    /// caller's statement loop parses the next statement fresh; sync
    /// tokens that cannot begin a statement are consumed instead, to
    /// guarantee forward progress (no infinite error-recovery loop).
    fn is_stmt_start_keyword(tok: &Token) -> bool {
        matches!(
            tok,
            Token::Set
                | Token::Let
                | Token::If
                | Token::While
                | Token::For
                | Token::Loop
                | Token::Leave
                | Token::Continue
                | Token::Return
                | Token::Def
                | Token::Type
                | Token::Trait
                | Token::Impl
                | Token::Constraint
                | Token::Comptime
                | Token::Generate
                | Token::ScopeCleanup
                | Token::Trigger
                | Token::Unsafe
                | Token::Ghost
                | Token::Isolate
                | Token::Match
                | Token::LBrace
                | Token::At
        )
    }
}

// -----------------------------------------------------------------------------
// `did_you_mean` suggestion engine
// -----------------------------------------------------------------------------

/// The syntactic context for keyword suggestions.
#[derive(Clone, Copy)]
pub enum KeywordContext {
    TopLevel,
    Expression,
    Statement,
    Type,
    Generic,
}

/// Compute Levenshtein distance between two strings.
/// The limited edit distance with reusable buffers: returns the
/// Levenshtein distance between `a` and `b`, or `limit + 1` if it
/// exceeds `limit` (the caller only cares whether it is within the
/// limit, so the computation can stop early).  The row buffers are
/// provided by the caller — the per-keyword allocation is avoided.
pub(crate) fn edit_distance_limited_into(
    prev: &mut Vec<usize>,
    curr: &mut Vec<usize>,
    a_buf: &mut Vec<char>,
    b_buf: &mut Vec<char>,
    a: &str,
    b: &str,
    limit: usize,
) -> usize {
    // Reuse the caller-provided buffers (clear + extend) instead of
    // allocating two `Vec<char>` on every call — `did_you_mean_keyword`
    // calls this once per candidate keyword.
    a_buf.clear();
    a_buf.extend(a.chars());
    b_buf.clear();
    b_buf.extend(b.chars());
    let a_len = a_buf.len();
    let b_len = b_buf.len();
    // Length-difference pruning: a distance can never be smaller than
    // the character-length difference.
    if a_len.abs_diff(b_len) > limit {
        return limit + 1;
    }
    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }
    prev.clear();
    prev.extend(0..=b_len);
    curr.resize(b_len + 1, 0);
    let overflow = limit + 1;
    for (i, &ca) in a_buf.iter().enumerate() {
        curr[0] = i + 1;
        let lo = (i + 1).saturating_sub(limit).max(1);
        let hi = ((i + 1) + limit).min(b_len);
        if lo > hi {
            continue;
        }
        for j in lo..=hi {
            let cb = b_buf[j - 1];
            let cost = if ca == cb { 0 } else { 1 };
            curr[j] = std::cmp::min(
                std::cmp::min(curr[j - 1] + 1, prev[j] + 1),
                prev[j - 1] + cost,
            );
        }
        if (lo..=hi).all(|j| curr[j] > limit) {
            return overflow;
        }
        for j in (1..lo).chain(hi + 1..=b_len) {
            curr[j] = overflow;
        }
        std::mem::swap(prev, curr);
    }
    let result = prev[b_len];
    if result > limit { limit + 1 } else { result }
}

/// Suggest a close-matching keyword when an unknown identifier is encountered.
/// Returns `Some("did you mean `def`?")` or similar.
///
/// `context` restricts the candidate keyword set to the current syntactic
/// position, so that e.g. `fn` at top level suggests `def` rather than `for`.
pub fn did_you_mean_keyword(input: &str, context: KeywordContext) -> Option<String> {
    // Reusable character buffers for the edit-distance computation — one
    // allocation per call instead of one per candidate keyword.
    let mut a_buf = Vec::new();
    let mut b_buf = Vec::new();
    let keywords: &[&str] = match context {
        KeywordContext::TopLevel => &[
            "def",
            "type",
            "trait",
            "import",
            "from",
            "edition",
            "constraint",
            "extern",
            "impl",
            "comptime",
            "async",
            "set",
            "let",
            "layout",
            "generate",
        ],
        KeywordContext::Expression => &[
            "true", "false", "if", "else", "match", "for", "while", "return", "leave", "continue",
            "move", "not", "and", "or", "sizeof", "alignof", "catch", "panic", "unsafe", "old",
            "exists", "forall", "poly", "unbox", "await", "task", "ieee",
        ],
        KeywordContext::Statement => &[
            "set",
            "let",
            "return",
            "leave",
            "continue",
            "if",
            "else",
            "while",
            "for",
            "unsafe",
            "ghost",
            "scope_cleanup",
            "trigger",
            "isolate",
        ],
        KeywordContext::Type => &[
            "Int", "UInt", "Float", "Bool", "Char", "Byte", "USize", "Never", "Rational", "dyn",
            "ref", "mut", "ieee",
        ],
        KeywordContext::Generic => &[
            "def",
            "type",
            "trait",
            "import",
            "from",
            "edition",
            "constraint",
            "extern",
            "impl",
            "comptime",
            "async",
            "set",
            "let",
            "if",
            "else",
            "while",
            "for",
            "return",
            "leave",
            "continue",
            "match",
            "ghost",
            "propagates",
            "overrides",
            "trigger",
            "scope_cleanup",
            "true",
            "false",
            "ieee",
        ],
    };
    // ASCII fast path: byte-based case folding avoids the allocation of
    // `to_lowercase()`; non-ASCII falls back to the Unicode fold.
    let input_lower = if input.is_ascii() {
        input.to_ascii_lowercase()
    } else {
        input.to_lowercase()
    };
    let first = input_lower.chars().next();
    let mut best = None;
    let max_kw_len = keywords.iter().map(|k| k.len()).max().unwrap_or(0);
    let mut prev = Vec::with_capacity(max_kw_len + 1);
    let mut curr = vec![0usize; max_kw_len + 1];

    for &kw in keywords {
        if first != kw.chars().next() {
            continue;
        }
        let d = edit_distance_limited_into(
            &mut prev,
            &mut curr,
            &mut a_buf,
            &mut b_buf,
            &input_lower,
            kw,
            2,
        );
        if d <= 2 {
            match best {
                None => best = Some((kw, d)),
                Some((_, db)) if d < db => best = Some((kw, d)),
                _ => {}
            }
        }
    }
    // Lenient pass: if no candidate found, allow wider distance for short inputs.
    // This catches cases like `fn` → `def` (distance 3, but the only sensible match).
    // For LONGER inputs the distance must be small RELATIVE to the length
    // (rustc's MaxEditDistance style) — a distance equal to the input length
    // (e.g. `xyzabc` → `leave`) is an unrelated input and must NOT be
    // suggested (the previous wide threshold produced false suggestions).
    if best.is_none() {
        for &kw in keywords {
            let threshold = input.len().max(3);
            let d = edit_distance_limited_into(
                &mut prev,
                &mut curr,
                &mut a_buf,
                &mut b_buf,
                &input_lower,
                kw,
                threshold,
            );
            if d <= threshold && (input.len() < 4 || d * 2 <= input.len()) {
                match best {
                    None => best = Some((kw, d)),
                    Some((_, db)) if d < db => best = Some((kw, d)),
                    _ => {}
                }
            }
        }
    }
    best.map(|(kw, _)| format!("did you mean `{}`?", kw))
}
