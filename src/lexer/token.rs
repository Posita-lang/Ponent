//! Token definition and classification.

use crate::symbol::Symbol;
use bitflags::bitflags;
use memchr::memchr2;

// ── Whitespace run scan (SIMD) ─────────────────────────────────────────
//
// Lexical whitespace is the contiguous range 0x09..=0x0D (HT, LF, VT, FF,
// CR) plus 0x20 (space) — SYNTAX.md §Whitespace.  logos's DFA scans this
// byte-at-a-time; for long indentation runs (deeply nested source) a
// vectorized run-scan is materially faster.  The regex matches a single
// byte and hands control to `whitespace_run`, which SIMD-scans the
// remainder and bumps past the whole run in one callback.
//
// Correctness of the single-byte regex: no other token begins with a
// whitespace byte (every `#[token]`/`#[regex]` arm below starts with a
// non-whitespace byte), so shortening `[...]+` to one byte cannot change
// longest-match.  The callback only runs at the top level — logos has
// already decided the position is outside any string/comment — so the
// SIMD scan can never eat into a `"…"` or `//` body.  `FilterResult::Skip`
// preserves the gap-based trivia reconstruction in `parser::mod`
// byte-for-byte: whitespace bytes stay in the `prev_end..span_range.start`
// gap, never entering the token stream, so `TriviaKind` classification
// (`gap.contains("//") || gap.contains("/*")`) is unchanged.

/// Longest run of ASCII lexical whitespace at the start of `bytes`:
/// `\t \n \x0B \x0C \r` (0x09..=0x0D) plus space (0x20).
#[inline]
pub(super) fn scan_whitespace(bytes: &[u8]) -> usize {
    let len = bytes.len();
    if len < 16 {
        return scan_whitespace_scalar(bytes);
    }
    let p = bytes.as_ptr();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is checked at runtime; `p` is valid for `len` bytes.
            return unsafe { scan_whitespace_avx2(p, len) };
        }
        // SAFETY: SSE2 is baseline on x86_64.
        return unsafe { scan_whitespace_sse2(p, len) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        return unsafe { scan_whitespace_neon(p, len) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scan_whitespace_scalar(bytes)
    }
}

#[inline]
fn scan_whitespace_scalar(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .position(|&b| !matches!(b, 0x09..=0x0D | 0x20))
        .unwrap_or(bytes.len())
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn scan_whitespace_sse2(mut p: *const u8, mut len: usize) -> usize {
    use std::arch::x86_64::*;
    let start = p;
    let (eight, fourteen, space) = unsafe {
        // SAFETY: SSE2 is baseline on x86_64; set1 is always available.
        (_mm_set1_epi8(8), _mm_set1_epi8(14), _mm_set1_epi8(32))
    };
    while len >= 16 {
        // SAFETY: `len >= 16` guarantees a valid unaligned 16-byte load at `p`.
        let mask = unsafe {
            let v = _mm_loadu_si128(p as *const __m128i);
            // 0x09..=0x0D ⇔ v > 8 && 14 > v (signed).  Bytes ≥ 0x80 are
            // negative as i8, so they fail `v > 8` and cannot enter the
            // range — no UTF-8 lead/continuation byte is misclassified.
            let range = _mm_and_si128(_mm_cmpgt_epi8(v, eight), _mm_cmpgt_epi8(fourteen, v));
            let ws = _mm_or_si128(range, _mm_cmpeq_epi8(v, space));
            _mm_movemask_epi8(ws) as u32
        };
        let non_ws = !mask & 0xFFFF;
        if non_ws != 0 {
            return (p as usize - start as usize) + non_ws.trailing_zeros() as usize;
        }
        p = unsafe { p.add(16) };
        len -= 16;
    }
    let off = p as usize - start as usize;
    // SAFETY: `len < 16` remaining bytes at `p` are valid by the original slice.
    let tail = unsafe { std::slice::from_raw_parts(p, len) };
    off + scan_whitespace_scalar(tail)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn scan_whitespace_avx2(mut p: *const u8, mut len: usize) -> usize {
    use std::arch::x86_64::*;
    let start = p;
    let (eight, fourteen, space) = unsafe {
        // SAFETY: guarded by `is_x86_feature_detected!("avx2")` at the call site.
        (
            _mm256_set1_epi8(8),
            _mm256_set1_epi8(14),
            _mm256_set1_epi8(32),
        )
    };
    while len >= 32 {
        // SAFETY: `len >= 32` guarantees a valid unaligned 32-byte load at `p`.
        let mask = unsafe {
            let v = _mm256_loadu_si256(p as *const __m256i);
            let range =
                _mm256_and_si256(_mm256_cmpgt_epi8(v, eight), _mm256_cmpgt_epi8(fourteen, v));
            let ws = _mm256_or_si256(range, _mm256_cmpeq_epi8(v, space));
            _mm256_movemask_epi8(ws) as u32
        };
        let non_ws = !mask;
        if non_ws != 0 {
            return (p as usize - start as usize) + non_ws.trailing_zeros() as usize;
        }
        p = unsafe { p.add(32) };
        len -= 32;
    }
    let off = p as usize - start as usize;
    // SAFETY: SSE2 is baseline; `p` is valid for the remaining `len` bytes.
    off + unsafe { scan_whitespace_sse2(p, len) }
}

// aarch64 has no `pmovmskb`-style horizontal mask extract.  The idiomatic
// fast path is `vminvq_s8` (single instruction: minimum of 16 lanes) to
// test "all whitespace", and a scalar scan of the 16-byte window only on
// the cold path (first non-whitespace byte).  Simulating movemask with a
// `vshrn`+pack sequence is more instructions and slower for the
// all-whitespace hot path that dominates real source.  This is the
// architecture-correct choice, not a fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn scan_whitespace_neon(mut p: *const u8, mut len: usize) -> usize {
    use std::arch::aarch64::*;
    let start = p;
    let (eight, fourteen, space) = unsafe {
        // SAFETY: NEON is baseline on aarch64.
        (vdupq_n_s8(8), vdupq_n_s8(14), vdupq_n_s8(32))
    };
    while len >= 16 {
        // SAFETY: `len >= 16` guarantees a valid 16-byte load at `p`.
        let all_ws = unsafe {
            let v = vld1q_u8(p);
            let vs = vreinterpretq_s8_u8(v);
            let range = vandq_s8(vcgtq_s8(vs, eight), vcgtq_s8(fourteen, vs));
            let ws = vorrq_s8(range, vceqq_s8(vs, space));
            // `ws` is 0xFF per whitespace lane, 0x00 otherwise.  `vminvq_s8`
            // is -1 iff every lane is 0xFF.
            vminvq_s8(ws) == -1
        };
        if !all_ws {
            // SAFETY: 16 bytes at `p` are valid (`len >= 16`).
            let chunk = unsafe { std::slice::from_raw_parts(p, 16) };
            return (p as usize - start as usize) + scan_whitespace_scalar(chunk);
        }
        p = p.add(16);
        len -= 16;
    }
    let off = p as usize - start as usize;
    // SAFETY: `len < 16` remaining bytes at `p` are valid by the original slice.
    let tail = unsafe { std::slice::from_raw_parts(p, len) };
    off + scan_whitespace_scalar(tail)
}

/// Scan a (possibly nested) block comment starting at an already-matched
/// `/*`. `/*` increments depth, `*/` decrements it; the token ends when
/// depth returns to zero. An unterminated comment consumes the rest of
/// the input and yields a lexer error so diagnostics can report it.
///
/// Only `*` and `/` can change the depth.  A pure `memchr2` skip loop
/// measured 9.5× faster than the byte-at-a-time loop it replaced on
/// marker-sparse bodies (license headers) but 2.2× *slower* on
/// marker-dense ones (star banners, `* // * *` runs), where the call
/// overhead per hit dominates.  Hence the hybrid: scan a small scalar
/// window first, and only fall back to the vectorized skip when the
/// window is marker-free.  Semantics are identical to the loop this
/// replaced (see the `nested_block_comment` / `deeply_nested_block_comment`
/// / `unterminated_block_comment_is_error` / `block_comment_with_stars_and_slashes`
/// regression tests).
pub(super) fn scan_block_comment(rest: &[u8]) -> Result<usize, usize> {
    let mut rest = rest;
    let mut consumed = 0usize;
    let mut depth = 1usize;
    const WINDOW: usize = 8;
    loop {
        let window = rest.len().min(WINDOW);
        let mut i = 0;
        while i < window && !matches!(rest[i], b'*' | b'/') {
            i += 1;
        }
        if i == window {
            // No marker within the window: skip to the next one vectorized.
            match memchr2(b'*', b'/', &rest[window..]) {
                Some(rel) => i = window + rel,
                None => {
                    // Unterminated: consume the rest of the input and fail.
                    // Without consuming `rest.len()`, a dangling byte would
                    // be re-lexed as a spurious token after the error
                    // (e.g. `/* x` → error + `Ident(x)`).
                    return Err(consumed + rest.len());
                }
            }
        }
        match rest[i] {
            b'*' if rest.get(i + 1) == Some(&b'/') => {
                // `*/` — close one level.
                depth -= 1;
                consumed += i + 2;
                rest = &rest[i + 2..];
                if depth == 0 {
                    return Ok(consumed);
                }
            }
            b'/' if rest.get(i + 1) == Some(&b'*') => {
                // `/*` — open one level (nested).
                depth += 1;
                consumed += i + 2;
                rest = &rest[i + 2..];
            }
            _ => {
                // A lone `*` or `/` (or a marker with no room for its
                // pair): skip past it.
                consumed += i + 1;
                rest = &rest[i + 1..];
            }
        }
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TokenClass: u16 {
        /// Tokens that can begin an expression.
        const EXPR_START      = 1 << 0;
        /// Comparison operators (>, >=, <, <=, ==, !=).
        const COMPARISON      = 1 << 1;
        /// Binary operators (arithmetic, shifts, bitwise).
        const BINARY_OPERATOR = 1 << 2;
    }
}
/// The coarse token class used for "may this token appear here" checks.
#[inline]
pub fn token_class(tok: &Token) -> TokenClass {
    use Token::*;
    let mut c = TokenClass::empty();
    if matches!(
        tok,
        IntLiteral(_)
            | FloatLiteral(_)
            // Hex/binary literals start expressions exactly like decimal
            // ones (`x = 0xFF;`) — same class of omission as the missing
            // `Shr` below.
            | HexLiteral(_)
            | BinLiteral(_)
            | True
            | False
            | CharLiteral(_)
            | ByteCharLiteral(_)
            | StringLiteral(_)
            | ByteStringLiteral(_)
            | Ident(_)
            | LParen
            | LBracket
            | Minus
            | Plus
            | Bang
            | Tilde
    ) {
        c |= TokenClass::EXPR_START;
    }
    if matches!(tok, Gt | Ge | Lt | Le | EqEq | Neq) {
        c |= TokenClass::COMPARISON;
    }
    if matches!(
        tok,
        Plus | Minus | Star | Slash | Percent | Shl | Shr | Ampersand | Pipe | Caret
            // Overflow-suffixed arithmetic (`+%`, `+?`, `+!`, …) are binary
            // arithmetic operators too.  Compound assignments (`+=`, `<<=`, …)
            // are *not* binary operators and stay unclassified here.
            | PlusWrap | MinusWrap | StarWrap
            | PlusSaturate | MinusSaturate | StarSaturate
            | PlusTrap | MinusTrap | StarTrap
    ) {
        c |= TokenClass::BINARY_OPERATOR;
    }
    c
}
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Token<'src> {
    // `allow_greedy = true` is LOAD-BEARING, not optional: in logos 0.16+ a
    // `[^\n]*` tail requires it, and it keeps `//` from winning over the
    // longer `///` / `//!` doc-comment regexes below.  Do not "clean up".
    WhitespaceOrComment,
    DocComment(&'src str),
    Bom,
    ModuleDocComment(&'src str),
    Def,
    Set,
    Type,
    Const,
    /// `codomain` — reserved by SYNTAX.md for `ensures` clauses; may not
    /// be used as an ordinary variable or parameter name anywhere.
    /// Deliberately NOT in `as_ident_symbol`.
    Codomain,
    With,
    Default,
    Return,
    If,
    Else,
    For,
    In,
    While,
    Loop,
    Leave,
    Continue,
    Comptime,
    Generate,
    Import,
    As,
    True,
    False,
    Auto,
    And,
    Or,
    Not,
    Sizeof,
    Alignof,
    Catch,
    Panic,
    Unsafe,
    Let,
    Finally,
    Where,
    When,
    Requires,
    Ensures,
    Invariant,
    Constraint,
    Move,
    Dyn,
    By,
    Copy,
    Ref,
    /// `ro` — reserved by SYNTAX.md: meaningful only immediately after
    /// `&` (the read-only borrow operator).  Deliberately NOT in
    /// `as_ident_symbol`; any occurrence outside `&ro` is a syntax error.
    Ro,
    Mut,
    Wrap,
    Saturate,
    Trap,
    Ieee,
    SelfKw,
    NoDefault,
    Extern,
    Pub,
    Edition,
    Deprecated,
    Experimental,
    Endian,
    BitOrder,
    Align,
    Pad,
    Packed,
    Async,
    Await,
    Task,
    Channel,
    Linear,
    Consume,
    Pure,
    Io,
    Trusted,
    Ghost,
    ScopeCleanup,
    Trigger,
    Layout,
    Validate,
    MissingMatch,
    ApplyLemma,
    Exists,
    Forall,
    Implies,
    Trait,
    Impl,
    Decreases,
    Diverges,
    Terminates,
    Cfg,
    Isolate,
    Hint,
    MustUse,
    MustHandle,
    LinkProof,
    Exhaustive,
    NoAllocError,
    NoPanic,
    DebugInfo,
    IeeeContracts,
    Old,
    AuditLog,
    Interrupt,
    Match,
    Propagates,
    Overrides,
    Poly,
    Unbox,
    Ident(Symbol),
    IntSuffix(&'src str),
    UIntSuffix(&'src str),
    HexIntSuffix(&'src str),
    HexUIntSuffix(&'src str),
    BinIntSuffix(&'src str),
    BinUIntSuffix(&'src str),
    // ── Literal payloads: raw `&'src str` — the source text, verbatim. ──
    //
    // Parsing happens at CONSUMPTION time (the parser's `parse_literal`
    // site and `display.rs::to_user_string`), so a token that is skipped,
    // recovered over, or never consumed (e.g. `--emit=tokens`, LSP
    // hover) never pays for parsing or for interning a `LitError`.
    // Keeping the slice — rather than an eager `Result<IntLit, LitError>`
    // / `Result<f64, LitError>` — is what drops `Token` from 48 B to
    // 24 B and makes it `Copy`; the suffix variants above already
    // followed this convention.  The regexes are UNCHANGED from the
    // eager-payload design: the one-unit escape-recovery contract lives
    // entirely in the patterns, not in the (now removed) callbacks.
    FloatLiteral(&'src str),
    IntLiteral(&'src str),
    HexLiteral(&'src str),
    BinLiteral(&'src str),
    // Escape-recovery regex (SYNTAX.md §Escape Sequences).  The body is
    // EXACTLY ONE unit — one plain char, or one escape:
    //   `\x` + hex run   (valid `\x41` and malformed `\x41A` / `\x`)
    //   `\u` + opt `{` + hex run + opt `}`  (valid `\u{41}` and malformed
    //                                        `\u{41` / `\u{1234567}` / `\u{}`)
    //   `\` + any non-newline char          (unknown escapes `\q`, `\e`, …)
    // so ANY malformed escape still lexes as ONE CharLiteral token; the
    // error surfaces when the slice is parsed at consumption time —
    // uniform hard-error recovery, never token-splitting.
    // The one-unit bound is LOAD-BEARING, not style: `'` is shared with
    // attribute access (`x'len`), lifetimes, and labels.  A permissive
    // body (e.g. `(?:[^'\\]|\\.)+`) would match the span BETWEEN two
    // stray apostrophes (`'len y'first` → CharLiteral over `len y`),
    // breaking attribute access.  Multi-plain-char bodies (`'ab'`) are
    // lexically indistinguishable from attribute access and therefore
    // split — the parser reports them via its Apostrophe handling.
    CharLiteral(&'src str),
    /// `b'x'` — byte character literal (SYNTAX.md: type `Byte` = `UInt<8>`).
    /// The regex is permissive (`\\.`) so invalid escapes surface as
    /// parse errors at consumption time (string-literal-style recovery)
    /// rather than token-splitting.
    ByteCharLiteral(&'src str),
    ByteStringLiteral(&'src str),
    StringLiteral(&'src str),
    Apostrophe,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    PlusWrap,
    MinusWrap,
    StarWrap,
    PlusSaturate,
    MinusSaturate,
    /// `*?` — saturating multiply. STILL UNDER RESEARCH: the BII template
    /// domain is linear, multiplication is outside the subset, and
    /// lowering fails closed on it. `/?` has no token yet.
    StarSaturate,
    PlusTrap,
    MinusTrap,
    StarTrap,
    Ampersand,
    Pipe,
    Caret,
    Shl,
    ShlEq,
    Shr,
    ShrEq,
    Tilde,
    EqEq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    Assign,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    Bang,
    Question,
    Dot,
    DotDot,
    DotDotEq,
    ColonColon,
    Colon,
    Semicolon,
    Comma,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Arrow,
    At,
    FatArrow,
    Ellipsis,
}

// ── Size budget (compile-time) ─────────────────────────────────────────
//
// Pinned at ≤ 24 B.  Every payload-carrying variant now stores either a
// raw `&'src str` (16 B — all literals and doc comments) or an interned
// `Symbol` (4 B); literal parsing happens at consumption time
// (`literal.rs`), so no inline `i128`, no `Result<_, LitError>`, and no
// 16-byte alignment pressure remain.  The enum needs only 8-byte
// alignment — 16 B slice plus the tag in head padding — and every
// payload is `Copy`, so `Token` is `Copy`.  This halves the parser's
// upfront token buffer (spans live separately in `DocMap::token_offsets`)
// versus the previous 48 B budget.  `const_assert!`
// (formerly the `token_is_compact` test) makes the budget an always-on
// check in every build instead of a test-time one: if this trips, a
// payload gained width — re-justify here before raising the bound.  The
// lifetime is irrelevant to layout; `'static` is just the spelling a
// const item needs.
static_assertions::const_assert!(core::mem::size_of::<Token<'static>>() <= 24);
