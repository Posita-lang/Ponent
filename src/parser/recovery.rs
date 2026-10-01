//! Error recovery, token synchronization, and `did_you_mean` suggestions.

use super::*;
use crate::lexer::Token;

impl<'src, 'input> Parser<'src, 'input> {
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
                Err(()) => return,
                Ok(tok) if Self::is_top_level_start(tok) => return,
                Ok(_) => {
                    self.advance().ok();
                }
            }
        }
    }

    /// Constant-time top-level-item-start check, mirroring `is_sync_token` /
    /// `is_stmt_start_keyword`.  `Set`/`Let` ARE valid at top level (global
    /// variable declarations), so they stop the skip (i.e. are NOT swallowed).
    fn is_top_level_start(tok: &Token) -> bool {
        matches!(
            tok,
            Token::Def
                | Token::Type
                | Token::Trait
                | Token::Impl
                | Token::Constraint
                | Token::Edition
                | Token::Import
                | Token::From
                | Token::Extern
                | Token::Comptime
                | Token::Async
                | Token::At
                | Token::Set
                | Token::Let
                | Token::Layout
        )
    }

    /// Constant-time sync-token check: the old `sync_tokens`
    /// array was linearly scanned per skipped token during error recovery —
    /// O(skipped × count).  A `matches!` compiles to a kind switch with no
    /// payload comparison.
    ///
    /// Only tokens that are genuinely safe recovery-consumption points AND
    /// cannot begin a statement appear here — variants that are also
    /// statement-start keywords (`Def`, `Set`, `Let`, `Type`, `Comptime`,
    /// `Generate`, `At`) are intercepted by `is_stmt_start_keyword` first in
    /// `synchronize`, so listing them here would be dead membership.
    fn is_sync_token(tok: &Token) -> bool {
        matches!(
            tok,
            Token::Semicolon
                | Token::RBrace
                | Token::Import
                | Token::From
                | Token::Extern
                | Token::Edition
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
    a: &[char],
    b: &[char],
    limit: usize,
) -> usize {
    let a_len = a.len();
    let b_len = b.len();
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
    let overflow = limit + 1;
    prev.clear();
    prev.extend(0..=b_len);
    curr.resize(b_len + 1, overflow);
    for (i, &ca) in a.iter().enumerate() {
        // Fill the whole row with the overflow sentinel up front so stale
        // values cannot leak between rows; the band below then overwrites
        // only the in-band cells.  `fill` lowers to a memset-like loop and
        // replaces the previous out-of-band `chain` refill loop.
        curr.fill(overflow);
        curr[0] = i + 1;
        let lo = (i + 1).saturating_sub(limit).max(1);
        let hi = ((i + 1) + limit).min(b_len);
        if lo > hi {
            // Unreachable after length-difference pruning, but a safe
            // early-out: the whole row is out of band, so the distance
            // must already exceed the limit.
            return overflow;
        }
        for j in lo..=hi {
            let cb = b[j - 1];
            let cost = if ca == cb { 0 } else { 1 };
            curr[j] = std::cmp::min(
                std::cmp::min(curr[j - 1] + 1, prev[j] + 1),
                prev[j - 1] + cost,
            );
        }
        if (lo..=hi).all(|j| curr[j] > limit) {
            return overflow;
        }
        std::mem::swap(prev, curr);
    }
    let result = prev[b_len];
    if result > limit { limit + 1 } else { result }
}

/// Reusable scratch buffers for the keyword-suggestion engine.
///
/// `did_you_mean_keyword` allocates four `Vec`s per call (input chars,
/// keyword chars, and the two edit-distance rows).  For one-shot compiler
/// runs that is negligible, but in dense-diagnostic environments (IDE/LSP
/// incremental re-parses, fuzzing malformed identifiers) suggestion
/// generation runs many times per parse; owning the buffers lets callers
/// reuse the same allocations across calls.
#[derive(Debug, Default)]
pub struct SuggestionBuffers {
    /// The lowercased input identifier, collected once per suggestion pass.
    input_chars: Vec<char>,
    /// The current candidate keyword, refilled once per candidate.
    keyword_chars: Vec<char>,
    /// Edit-distance DP row `i - 1`.
    prev: Vec<usize>,
    /// Edit-distance DP row `i`.
    curr: Vec<usize>,
}

/// Suggest a close-matching keyword when an unknown identifier is encountered.
/// Returns `Some("did you mean `def`?")` or similar.
///
/// `context` restricts the candidate keyword set to the current syntactic
/// position, so that e.g. `fn` at top level suggests `def` rather than `for`.
///
/// Convenience wrapper that allocates fresh [`SuggestionBuffers`] per call.
/// Callers that generate suggestions in bulk should prefer
/// [`did_you_mean_keyword_with_buffers`] to amortize the allocations.
pub fn did_you_mean_keyword(input: &str, context: KeywordContext) -> Option<String> {
    did_you_mean_keyword_with_buffers(&mut SuggestionBuffers::default(), input, context)
}

/// Buffer-reusing variant of [`did_you_mean_keyword`] — see [`SuggestionBuffers`].
pub fn did_you_mean_keyword_with_buffers(
    buffers: &mut SuggestionBuffers,
    input: &str,
    context: KeywordContext,
) -> Option<String> {
    let SuggestionBuffers {
        input_chars,
        keyword_chars,
        prev,
        curr,
    } = buffers;
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
    // Collect the input ONCE per suggestion pass — re-splitting it into a
    // `Vec<char>` per candidate keyword was O(k · input_len) of repeated work.
    input_chars.clear();
    input_chars.extend(input_lower.chars());
    let first = input_chars.first().copied();
    let mut best = None;

    for &kw in keywords {
        if first != kw.chars().next() {
            continue;
        }
        keyword_chars.clear();
        keyword_chars.extend(kw.chars());
        let d = edit_distance_limited_into(prev, curr, input_chars, keyword_chars, 2);
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
            keyword_chars.clear();
            keyword_chars.extend(kw.chars());
            let d = edit_distance_limited_into(prev, curr, input_chars, keyword_chars, threshold);
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
