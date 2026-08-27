//! Token definition and classification.

use crate::symbol::Symbol;
use bitflags::bitflags;
use logos::Logos;

use super::literal::{
    parse_byte_string_literal, parse_char_literal, parse_float_literal, parse_int_literal,
    parse_string_literal,
};

bitflags! {
    #[derive(Clone, Copy, Debug, Default)]
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
pub fn token_class(tok: &Token) -> TokenClass {
    use Token::*;
    let mut c = TokenClass::empty();
    if matches!(
        tok,
        IntLiteral(_)
            | FloatLiteral(_)
            | True
            | False
            | CharLiteral(_)
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
        Plus | Minus | Star | Slash | Percent | Shl | Ampersand | Pipe | Caret
    ) {
        c |= TokenClass::BINARY_OPERATOR;
    }
    c
}
#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    #[regex("[ \t\r\n\x0C]+", logos::skip)]
    #[regex("//[^\n]*", logos::skip, allow_greedy = true)]
    #[regex("/\\*[^\\*]*\\*+(?:[^/\\*][^\\*]*\\*+)*/", logos::skip)]
    WhitespaceOrComment,
    #[regex("///[^\n]*", |lex| lex.slice()[3..].trim().to_string(), allow_greedy = true)]
    DocComment(String),
    #[regex("//![^\n]*", |lex| lex.slice()[3..].trim().to_string(), allow_greedy = true)]
    ModuleDocComment(String),
    #[token("def")]
    Def,
    #[token("set")]
    Set,
    #[token("type")]
    Type,
    #[token("const")]
    Const,
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
    #[token("from")]
    From,
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
    #[token("on")]
    On,
    #[token("on_timeout")]
    OnTimeout,
    #[token("on_cancel")]
    OnCancel,
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
    #[token("round")]
    Round,
    #[token("trunc")]
    Trunc,
    #[token("ceil")]
    Ceil,
    #[token("floor")]
    Floor,
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
    #[regex("[0-9][0-9_]*i[0-9]+", |lex| lex.slice().to_string())]
    IntSuffix(String),
    #[regex("[0-9][0-9_]*u[0-9]+", |lex| lex.slice().to_string())]
    UIntSuffix(String),
    #[regex("0x[0-9a-fA-F][0-9a-fA-F_]*i[0-9]+", |lex| lex.slice().to_string())]
    HexIntSuffix(String),
    #[regex("0x[0-9a-fA-F][0-9a-fA-F_]*u[0-9]+", |lex| lex.slice().to_string())]
    HexUIntSuffix(String),
    #[regex("0b[01][01_]*i[0-9]+", |lex| lex.slice().to_string())]
    BinIntSuffix(String),
    #[regex("0b[01][01_]*u[0-9]+", |lex| lex.slice().to_string())]
    BinUIntSuffix(String),
    #[regex("[0-9][0-9_]*\\.[0-9][0-9_]*([eE][+-]?[0-9][0-9_]*)?", |lex| {
        parse_float_literal(lex.slice()).map_err(|e| e.to_string())
    })]
    #[regex("[0-9][0-9_]*[eE][+-]?[0-9][0-9_]*", |lex| {
        parse_float_literal(lex.slice()).map_err(|e| e.to_string())
    })]
    FloatLiteral(Result<f64, String>),
    #[regex("[0-9][0-9_]*", |lex| {
        parse_int_literal(lex.slice(), 10, "integer literal overflow")
    })]
    IntLiteral(Result<i128, String>),
    #[regex("0x[0-9a-fA-F][0-9a-fA-F_]*", |lex| {
        parse_int_literal(&lex.slice()[2..], 16, "hex literal overflow")
    })]
    HexLiteral(Result<i128, String>),
    #[regex("0b[01][01_]*", |lex| {
        parse_int_literal(&lex.slice()[2..], 2, "binary literal overflow")
    })]
    BinLiteral(Result<i128, String>),
    #[regex("'(?:[^'\\\\]|\\\\(?:[nrt\\\\\"'0]|x[0-9a-fA-F]{2}|u\\{[0-9a-fA-F]{1,6}\\}))'", |lex| parse_char_literal(lex.slice()))]
    CharLiteral(Result<u8, String>),
    #[regex("b\"(\\\\.|[^\"\\\\])*\"", |lex| parse_byte_string_literal(lex.slice()))]
    ByteStringLiteral(Result<Vec<u8>, String>),
    #[regex("\"(\\\\.|[^\"\\\\])*\"", |lex| parse_string_literal(lex.slice()))]
    StringLiteral(Result<String, String>),
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
