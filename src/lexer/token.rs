//! Token definition and classification.

use crate::symbol::Symbol;
use bitflags::bitflags;
use logos::{Lexer, Logos};
use memchr::memchr2;

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
fn block_comment<'src>(lex: &mut Lexer<'src, Token<'src>>) -> logos::FilterResult<(), ()> {
    let mut rest = lex.remainder().as_bytes();
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
                    lex.bump(consumed + rest.len());
                    return logos::FilterResult::Error(());
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
                    lex.bump(consumed);
                    return logos::FilterResult::Emit(());
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

/// U+FEFF (BOM) handling — SYNTAX.md §Whitespace, BOM, and Unicode
/// Versioning: the BOM is an encoding signature allowed only as the
/// first bytes of a file, where the lexer strips it; anywhere else it
/// is an error.  Implemented at the token layer so every `Token::lexer`
/// entry point (parser, `--emit=tokens`, vfs) behaves identically:
/// file-start BOMs are skipped (never entering the token stream), and a
/// mid-file BOM yields exactly one lexer error (spans unchanged).
fn strip_bom<'src>(lex: &mut Lexer<'src, Token<'src>>) -> logos::FilterResult<(), ()> {
    if lex.span().start == 0 {
        // The token match has already consumed the BOM; Skip drops it
        // silently (it never enters the token stream).
        logos::FilterResult::Skip
    } else {
        // The match consumed the BOM; record exactly one lexer error for
        // its span and continue after it.  No `bump` here — bump is
        // relative to the end of the matched region, which logos already
        // advanced past (a second bump panics "Invalid Lexer bump").
        logos::FilterResult::Error(())
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
#[derive(Logos, Debug, PartialEq, Eq, Clone, Copy)]
pub enum Token<'src> {
    #[regex("[ \t\r\n\x0B\x0C]+", logos::skip)]
    // `allow_greedy = true` is LOAD-BEARING, not optional: in logos 0.16+ a
    // `[^\n]*` tail requires it, and it keeps `//` from winning over the
    // longer `///` / `//!` doc-comment regexes below.  Do not "clean up".
    #[regex("//[^\n]*", logos::skip, allow_greedy = true)]
    #[regex("/\\*", block_comment)]
    WhitespaceOrComment,
    #[regex("///[^\n]*", |lex| (lex.slice()[3..].trim()) as &'src str, allow_greedy = true)]
    DocComment(&'src str),
    #[token("\u{FEFF}", strip_bom)]
    Bom,
    #[regex("//![^\n]*", |lex| (lex.slice()[3..].trim()) as &'src str, allow_greedy = true)]
    ModuleDocComment(&'src str),
    #[token("def")]
    Def,
    #[token("set")]
    Set,
    #[token("type")]
    Type,
    #[token("const")]
    Const,
    /// `codomain` — reserved by SYNTAX.md for `ensures` clauses; may not
    /// be used as an ordinary variable or parameter name anywhere.
    /// Deliberately NOT in `as_ident_symbol`.
    #[token("codomain")]
    Codomain,
    #[token("with")]
    With,
    #[token("default")]
    Default,
    #[token("return")]
    Return,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("while")]
    While,
    #[token("loop")]
    Loop,
    #[token("leave")]
    Leave,
    #[token("continue")]
    Continue,
    #[token("comptime")]
    Comptime,
    #[token("generate")]
    Generate,
    #[token("import")]
    Import,
    #[token("as")]
    As,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("auto")]
    Auto,
    #[token("and")]
    And,
    #[token("or")]
    Or,
    #[token("not")]
    Not,
    #[token("sizeof")]
    Sizeof,
    #[token("alignof")]
    Alignof,
    #[token("catch")]
    Catch,
    #[token("panic")]
    Panic,
    #[token("unsafe")]
    Unsafe,
    #[token("let")]
    Let,
    #[token("finally")]
    Finally,
    #[token("where")]
    Where,
    #[token("when")]
    When,
    #[token("requires")]
    Requires,
    #[token("ensures")]
    Ensures,
    #[token("invariant")]
    Invariant,
    #[token("constraint")]
    Constraint,
    #[token("move")]
    Move,
    #[token("dyn")]
    Dyn,
    #[token("by")]
    By,
    #[token("copy")]
    Copy,
    #[token("ref")]
    Ref,
    /// `ro` — reserved by SYNTAX.md: meaningful only immediately after
    /// `&` (the read-only borrow operator).  Deliberately NOT in
    /// `as_ident_symbol`; any occurrence outside `&ro` is a syntax error.
    #[token("ro")]
    Ro,
    #[token("mut")]
    Mut,
    #[token("wrap")]
    Wrap,
    #[token("saturate")]
    Saturate,
    #[token("trap")]
    Trap,
    #[token("ieee")]
    Ieee,
    #[token("Self")]
    SelfKw,
    #[token("no_default")]
    NoDefault,
    #[token("extern")]
    Extern,
    #[token("pub")]
    Pub,
    #[token("edition")]
    Edition,
    #[token("deprecated")]
    Deprecated,
    #[token("experimental")]
    Experimental,
    #[token("endian")]
    Endian,
    #[token("bit_order")]
    BitOrder,
    #[token("align")]
    Align,
    #[token("pad")]
    Pad,
    #[token("packed")]
    Packed,
    #[token("async")]
    Async,
    #[token("await")]
    Await,
    #[token("task")]
    Task,
    #[token("channel")]
    Channel,
    #[token("linear")]
    Linear,
    #[token("consume")]
    Consume,
    #[token("pure")]
    Pure,
    #[token("io")]
    Io,
    #[token("trusted")]
    Trusted,
    #[token("ghost")]
    Ghost,
    #[token("scope_cleanup")]
    ScopeCleanup,
    #[token("trigger")]
    Trigger,
    #[token("layout")]
    Layout,
    #[token("validate")]
    Validate,
    #[token("missing_match")]
    MissingMatch,
    #[token("apply_lemma")]
    ApplyLemma,
    #[token("exists")]
    Exists,
    #[token("forall")]
    Forall,
    #[token("implies")]
    Implies,
    #[token("trait")]
    Trait,
    #[token("impl")]
    Impl,
    #[token("decreases")]
    Decreases,
    #[token("diverges")]
    Diverges,
    #[token("terminates")]
    Terminates,
    #[token("cfg")]
    Cfg,
    #[token("isolate")]
    Isolate,
    #[token("hint")]
    Hint,
    #[token("must_use")]
    MustUse,
    #[token("must_handle")]
    MustHandle,
    #[token("link_proof")]
    LinkProof,
    #[token("exhaustive")]
    Exhaustive,
    #[token("no_alloc_error")]
    NoAllocError,
    #[token("no_panic")]
    NoPanic,
    #[token("debug_info")]
    DebugInfo,
    #[token("ieee_contracts")]
    IeeeContracts,
    #[token("old")]
    Old,
    #[token("audit_log")]
    AuditLog,
    #[token("interrupt")]
    Interrupt,
    #[token("match")]
    Match,
    #[token("propagates")]
    Propagates,
    #[token("overrides")]
    Overrides,
    #[token("poly")]
    Poly,
    #[token("unbox")]
    Unbox,
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| Symbol::intern(lex.slice()))]
    Ident(Symbol),
    #[regex("[0-9][0-9_]*i[0-9]+")]
    IntSuffix(&'src str),
    #[regex("[0-9][0-9_]*u[0-9]+")]
    UIntSuffix(&'src str),
    #[regex("0x[0-9a-fA-F][0-9a-fA-F_]*i[0-9]+")]
    HexIntSuffix(&'src str),
    #[regex("0x[0-9a-fA-F][0-9a-fA-F_]*u[0-9]+")]
    HexUIntSuffix(&'src str),
    #[regex("0b[01][01_]*i[0-9]+")]
    BinIntSuffix(&'src str),
    #[regex("0b[01][01_]*u[0-9]+")]
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
    #[regex("[0-9][0-9_]*\\.[0-9][0-9_]*([eE][+-]?[0-9][0-9_]*)?")]
    #[regex("[0-9][0-9_]*[eE][+-]?[0-9][0-9_]*")]
    FloatLiteral(&'src str),
    #[regex("[0-9][0-9_]*")]
    IntLiteral(&'src str),
    #[regex("0x[0-9a-fA-F][0-9a-fA-F_]*")]
    HexLiteral(&'src str),
    #[regex("0b[01][01_]*")]
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
    #[regex("'(?:[^'\\\\]|\\\\(?:x[0-9a-fA-F]{0,}|u\\{?[0-9a-fA-F]{0,}\\}?|[^\\n]))'")]
    CharLiteral(&'src str),
    /// `b'x'` — byte character literal (SYNTAX.md: type `Byte` = `UInt<8>`).
    /// The regex is permissive (`\\.`) so invalid escapes surface as
    /// parse errors at consumption time (string-literal-style recovery)
    /// rather than token-splitting.
    #[regex("b'(\\\\.|[^'\\\\])*'")]
    ByteCharLiteral(&'src str),
    #[regex("b\"(\\\\.|[^\"\\\\])*\"")]
    ByteStringLiteral(&'src str),
    #[regex("\"(\\\\.|[^\"\\\\])*\"")]
    StringLiteral(&'src str),
    #[token("'")]
    Apostrophe,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("+%")]
    PlusWrap,
    #[token("-%")]
    MinusWrap,
    #[token("*%")]
    StarWrap,
    #[token("+?")]
    PlusSaturate,
    #[token("-?")]
    MinusSaturate,
    /// `*?` — saturating multiply. STILL UNDER RESEARCH: the BII template
    /// domain is linear, multiplication is outside the subset, and
    /// lowering fails closed on it. `/?` has no token yet.
    #[token("*?")]
    StarSaturate,
    #[token("+!")]
    PlusTrap,
    #[token("-!")]
    MinusTrap,
    #[token("*!")]
    StarTrap,
    #[token("&")]
    Ampersand,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
    #[token("<<")]
    Shl,
    #[token("<<=")]
    ShlEq,
    #[token(">>")]
    Shr,
    #[token(">>=")]
    ShrEq,
    #[token("~")]
    Tilde,
    #[token("==")]
    EqEq,
    #[token("!=")]
    Neq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("=")]
    Assign,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("!")]
    Bang,
    #[token("?")]
    Question,
    #[token(".")]
    Dot,
    #[token("..")]
    DotDot,
    #[token("..=")]
    DotDotEq,
    #[token("::")]
    ColonColon,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("->")]
    Arrow,
    #[token("@")]
    At,
    #[token("=>")]
    FatArrow,
    #[token("...")]
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
