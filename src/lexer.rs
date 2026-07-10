use logos::Logos;

fn parse_char_literal(s: &str) -> Result<u8, String> {
    let inner = &s[1..s.len() - 1];
    let mut chars = inner.chars();
    match chars.next() {
        Some('\\') => match chars.next() {
            Some('n') => Ok(b'\n'),
            Some('r') => Ok(b'\r'),
            Some('t') => Ok(b'\t'),
            Some('\\') => Ok(b'\\'),
            Some('"') => Ok(b'"'),
            Some('\'') => Ok(b'\''),
            Some('0') => Ok(b'\0'),
            Some('x') => {
                let hex: String = chars.by_ref().take(2).collect();
                if hex.len() != 2 {
                    return Err("\\x must be followed by exactly 2 hex digits".to_string());
                }
                if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err("invalid hex digit in char literal".to_string());
                }
                if let Some(c) = chars.clone().next() {
                    if c.is_ascii_hexdigit() {
                        return Err("expected exactly 2 hex digits after \\x".to_string());
                    }
                }
                u8::from_str_radix(&hex, 16)
                    .map_err(|_| "invalid hex digit in char literal".to_string())
            }
            Some('u') => {
                if chars.next() != Some('{') {
                    return Err("expected '{' after \\u in char literal".to_string());
                }
                let mut scalar = String::new();
                let mut closed = false;
                for c in chars.by_ref().take(6) {
                    if c == '}' {
                        closed = true;
                        break;
                    }
                    if !c.is_ascii_hexdigit() {
                        return Err("invalid hex digit in \\u{...} in char literal".to_string());
                    }
                    scalar.push(c);
                }
                if !closed {
                    if let Some(c) = chars.next() {
                        if c != '}' {
                            return Err("too many hex digits in \\u{...} (max 6)".to_string());
                        }
                    } else {
                        return Err("unclosed \\u{...} in char literal".to_string());
                    }
                }
                let code = u32::from_str_radix(&scalar, 16)
                    .map_err(|_| "invalid hex in \\u{...} in char literal".to_string())?;
                if code > 0xFF || (0xD800..=0xDFFF).contains(&code) || code > 0x10FFFF {
                    return Err("unicode scalar in char literal must be 0x00..0xFF, not a surrogate, and valid Unicode".to_string());
                }
                Ok(code as u8)
            }
            _ => Err("unknown escape sequence in char literal".to_string()),
        },
        Some(c) => {
            if c.len_utf8() == 1 {
                Ok(c as u8)
            } else {
                Err(
                    "multi-byte characters not allowed in char literal (use ASCII or \\u)"
                        .to_string(),
                )
            }
        }
        None => Err("empty char literal".to_string()),
    }
}

fn parse_string_literal(s: &str) -> Result<String, String> {
    let inner = &s[1..s.len() - 1];
    let mut result = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let next = chars
                .next()
                .ok_or("unexpected end of string after backslash")?;
            match next {
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                '\\' => result.push('\\'),
                '"' => result.push('"'),
                '\'' => result.push('\''),
                '0' => result.push('\0'),
                'x' => {
                    let hex: String = chars.by_ref().take(2).collect();
                    if hex.len() != 2 {
                        return Err("\\x must be followed by exactly 2 hex digits".to_string());
                    }
                    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                        return Err("invalid hex digit in string literal".to_string());
                    }
                    if let Some(c) = chars.clone().next() {
                        if c.is_ascii_hexdigit() {
                            return Err("expected exactly 2 hex digits after \\x".to_string());
                        }
                    }
                    let byte = u8::from_str_radix(&hex, 16)
                        .map_err(|_| "invalid hex digit in string literal".to_string())?;
                    result.push(byte as char);
                }
                'u' => {
                    if chars.next() != Some('{') {
                        return Err("expected '{' after \\u in string literal".to_string());
                    }
                    let mut scalar = String::new();
                    let mut closed = false;
                    for c in chars.by_ref().take(6) {
                        if c == '}' {
                            closed = true;
                            break;
                        }
                        if !c.is_ascii_hexdigit() {
                            return Err(
                                "invalid hex digit in \\u{...} in string literal".to_string()
                            );
                        }
                        scalar.push(c);
                    }
                    if !closed {
                        if let Some(c) = chars.next() {
                            if c != '}' {
                                return Err("too many hex digits in \\u{...} (max 6)".to_string());
                            }
                        } else {
                            return Err("unclosed \\u{...} in string literal".to_string());
                        }
                    }
                    let code = u32::from_str_radix(&scalar, 16)
                        .map_err(|_| "invalid hex in \\u{...} in string literal".to_string())?;
                    let c = std::char::from_u32(code).ok_or_else(|| {
                        format!("invalid unicode scalar {:#x} in string literal", code)
                    })?;
                    result.push(c);
                }
                _ => {
                    return Err(format!(
                        "unknown escape sequence '\\{}' in string literal",
                        next
                    ));
                }
            }
        } else {
            result.push(c);
        }
    }
    Ok(result)
}

fn parse_byte_string_literal(s: &str) -> Result<Vec<u8>, String> {
    let inner = &s[2..s.len() - 1];
    let mut result = Vec::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let next = chars
                .next()
                .ok_or("unexpected end of byte string after backslash")?;
            match next {
                'n' => result.push(b'\n'),
                'r' => result.push(b'\r'),
                't' => result.push(b'\t'),
                '\\' => result.push(b'\\'),
                '"' => result.push(b'"'),
                '\'' => result.push(b'\''),
                '0' => result.push(b'\0'),
                'x' => {
                    let hex: String = chars.by_ref().take(2).collect();
                    if hex.len() != 2 {
                        return Err("\\x must be followed by exactly 2 hex digits".to_string());
                    }
                    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                        return Err("invalid hex digit in byte string literal".to_string());
                    }
                    if let Some(c) = chars.clone().next() {
                        if c.is_ascii_hexdigit() {
                            return Err("expected exactly 2 hex digits after \\x".to_string());
                        }
                    }
                    let byte = u8::from_str_radix(&hex, 16)
                        .map_err(|_| "invalid hex digit in byte string literal".to_string())?;
                    result.push(byte);
                }
                'u' => return Err("\\u{...} is not allowed in byte string literals".to_string()),
                _ => {
                    return Err(format!(
                        "unknown escape sequence '\\{}' in byte string literal",
                        next
                    ));
                }
            }
        } else {
            result.push(c as u8);
        }
    }
    Ok(result)
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

    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

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
        let s = lex.slice().replace('_', "");
        match s.parse::<f64>() {
            Ok(val) if val.is_finite() => Ok(val),
            Ok(_) => Err("float literal must be finite (no NaN or Inf)".to_string()),
            Err(_) => Err("invalid float literal".to_string()),
        }
    })]
    #[regex("[0-9][0-9_]*[eE][+-]?[0-9][0-9_]*", |lex| {
        let s = lex.slice().replace('_', "");
        match s.parse::<f64>() {
            Ok(val) if val.is_finite() => Ok(val),
            Ok(_) => Err("float literal must be finite (no NaN or Inf)".to_string()),
            Err(_) => Err("invalid float literal".to_string()),
        }
    })]
    FloatLiteral(Result<f64, String>),

    #[regex("[0-9][0-9_]*", |lex| {
        let s = lex.slice().replace('_', "");
        s.parse::<i128>().map(Ok).unwrap_or_else(|_| Err("integer literal overflow".to_string()))
    })]
    IntLiteral(Result<i128, String>),
    #[regex("0x[0-9a-fA-F][0-9a-fA-F_]*", |lex| {
        let s = lex.slice()[2..].replace('_', "");
        i128::from_str_radix(&s, 16).map(Ok).unwrap_or_else(|_| Err("hex literal overflow".to_string()))
    })]
    HexLiteral(Result<i128, String>),
    #[regex("0b[01][01_]*", |lex| {
        let s = lex.slice()[2..].replace('_', "");
        i128::from_str_radix(&s, 2).map(Ok).unwrap_or_else(|_| Err("binary literal overflow".to_string()))
    })]
    BinLiteral(Result<i128, String>),

    #[regex("'(?:[^'\\\\]|\\\\[^']*|\\\\')'", |lex| parse_char_literal(lex.slice()))]
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
    #[token(">>")]
    Shr,
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

#[cfg(test)]
mod tests {
    use super::*;
    use logos::Logos;

    fn check_tokens(source: &str, expected: Vec<Token>) {
        let mut lexer = Token::lexer(source);
        for exp in expected {
            loop {
                let tok = lexer
                    .next()
                    .expect("unexpected end of token stream")
                    .expect("lexer error");
                if tok == Token::WhitespaceOrComment {
                    continue;
                }
                assert_eq!(
                    tok,
                    exp,
                    "unexpected token at '{}'",
                    &source[lexer.span().start..lexer.span().end]
                );
                break;
            }
        }
        while let Some(t) = lexer.next() {
            match t {
                Ok(Token::WhitespaceOrComment) => {}
                Ok(_) => panic!("extra tokens after expected end"),
                Err(_) => panic!("extra error tokens after expected end"),
            }
        }
    }

    fn collect_all_tokens(source: &str) -> Vec<Result<Token, ()>> {
        Token::lexer(source).collect()
    }

    #[test]
    fn test_parse_char_literal_fn() {
        assert_eq!(parse_char_literal(r"'\n'").unwrap(), b'\n');
        assert_eq!(parse_char_literal(r"'\x41'").unwrap(), b'A');
        assert_eq!(parse_char_literal(r"'\u{7F}'").unwrap(), 0x7F);
        assert_eq!(parse_char_literal(r"'a'").unwrap(), b'a');
        assert_eq!(parse_char_literal(r"'\u{80}'").unwrap(), 0x80);
        assert_eq!(parse_char_literal(r"'\u{FF}'").unwrap(), 0xFF);
        assert!(parse_char_literal(r"'\u{100}'").is_err());
        assert!(parse_char_literal(r"'\u{D800}'").is_err());
        assert!(parse_char_literal(r"'\u{10FFFF}'").is_err());
    }

    #[test]
    fn test_parse_string_literal_fn() {
        assert_eq!(
            parse_string_literal(r#""hello\nworld""#).unwrap(),
            "hello\nworld"
        );
        assert_eq!(parse_string_literal(r#""\u{00E9}""#).unwrap(), "é");
        assert_eq!(parse_string_literal(r#""\x41\x42""#).unwrap(), "AB");
    }

    #[test]
    fn test_parse_byte_string_literal_fn() {
        assert_eq!(
            parse_byte_string_literal(r#"b"hello\nworld""#).unwrap(),
            b"hello\nworld".to_vec()
        );
        assert_eq!(
            parse_byte_string_literal(r#"b"\x00\xFF""#).unwrap(),
            vec![0x00, 0xFF]
        );
        assert!(parse_byte_string_literal(r#"b"\u{41}""#).is_err());
    }

    #[test]
    fn keywords_all() {
        check_tokens(
            "def set type with default return if else for in while loop leave continue comptime import from as true false auto",
            vec![
                Token::Def,
                Token::Set,
                Token::Type,
                Token::With,
                Token::Default,
                Token::Return,
                Token::If,
                Token::Else,
                Token::For,
                Token::In,
                Token::While,
                Token::Loop,
                Token::Leave,
                Token::Continue,
                Token::Comptime,
                Token::Import,
                Token::From,
                Token::As,
                Token::True,
                Token::False,
                Token::Auto,
            ],
        );
    }

    #[test]
    fn more_keywords() {
        check_tokens(
            "and or not sizeof alignof catch panic unsafe let finally where requires ensures invariant constraint move dyn by copy ref mut wrap saturate trap Self no_default extern pub edition deprecated experimental endian bit_order align pad packed async await task channel linear consume pure io trusted ghost scope_cleanup trigger validate missing_match apply_lemma exists forall on on_timeout on_cancel trait impl decreases terminates cfg isolate hint must_use must_handle link_proof exhaustive no_alloc_error no_panic debug_info old audit_log interrupt round trunc ceil floor propagates overrides",
            vec![
                Token::And,
                Token::Or,
                Token::Not,
                Token::Sizeof,
                Token::Alignof,
                Token::Catch,
                Token::Panic,
                Token::Unsafe,
                Token::Let,
                Token::Finally,
                Token::Where,
                Token::Requires,
                Token::Ensures,
                Token::Invariant,
                Token::Constraint,
                Token::Move,
                Token::Dyn,
                Token::By,
                Token::Copy,
                Token::Ref,
                Token::Mut,
                Token::Wrap,
                Token::Saturate,
                Token::Trap,
                Token::SelfKw,
                Token::NoDefault,
                Token::Extern,
                Token::Pub,
                Token::Edition,
                Token::Deprecated,
                Token::Experimental,
                Token::Endian,
                Token::BitOrder,
                Token::Align,
                Token::Pad,
                Token::Packed,
                Token::Async,
                Token::Await,
                Token::Task,
                Token::Channel,
                Token::Linear,
                Token::Consume,
                Token::Pure,
                Token::Io,
                Token::Trusted,
                Token::Ghost,
                Token::ScopeCleanup,
                Token::Trigger,
                Token::Validate,
                Token::MissingMatch,
                Token::ApplyLemma,
                Token::Exists,
                Token::Forall,
                Token::On,
                Token::OnTimeout,
                Token::OnCancel,
                Token::Trait,
                Token::Impl,
                Token::Decreases,
                Token::Terminates,
                Token::Cfg,
                Token::Isolate,
                Token::Hint,
                Token::MustUse,
                Token::MustHandle,
                Token::LinkProof,
                Token::Exhaustive,
                Token::NoAllocError,
                Token::NoPanic,
                Token::DebugInfo,
                Token::Old,
                Token::AuditLog,
                Token::Interrupt,
                Token::Round,
                Token::Trunc,
                Token::Ceil,
                Token::Floor,
                Token::Propagates,
                Token::Overrides,
            ],
        );
    }

    #[test]
    fn integer_literals() {
        check_tokens(
            "42 0xFF 0b1010 42i32 0xFFu8 0b1101u4",
            vec![
                Token::IntLiteral(Ok(42)),
                Token::HexLiteral(Ok(255)),
                Token::BinLiteral(Ok(10)),
                Token::IntSuffix("42i32".into()),
                Token::HexUIntSuffix("0xFFu8".into()),
                Token::BinUIntSuffix("0b1101u4".into()),
            ],
        );
    }

    #[test]
    fn integer_overflow_errors() {
        // These values overflow i128 and should produce errors.
        check_tokens(
            "999999999999999999999999999999999999999",
            vec![Token::IntLiteral(Err("integer literal overflow".into()))],
        );
        check_tokens(
            "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
            vec![Token::HexLiteral(Err("hex literal overflow".into()))],
        );
        check_tokens(
            "0b11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
            vec![Token::BinLiteral(Err("binary literal overflow".into()))],
        );
        check_tokens("-42", vec![Token::Minus, Token::IntLiteral(Ok(42))]);
    }

    #[test]
    fn float_literals() {
        check_tokens(
            "3.14 2.5e-3 1_000.5 1e10",
            vec![
                Token::FloatLiteral(Ok(3.14)),
                Token::FloatLiteral(Ok(0.0025)),
                Token::FloatLiteral(Ok(1000.5)),
                Token::FloatLiteral(Ok(1e10)),
            ],
        );
    }

    #[test]
    fn char_literals() {
        check_tokens(
            r"'\n' '\t' '\\' '\'' '\x41' 'a'",
            vec![
                Token::CharLiteral(Ok(b'\n')),
                Token::CharLiteral(Ok(b'\t')),
                Token::CharLiteral(Ok(b'\\')),
                Token::CharLiteral(Ok(b'\'')),
                Token::CharLiteral(Ok(b'A')),
                Token::CharLiteral(Ok(b'a')),
            ],
        );
    }

    #[test]
    fn string_literals() {
        let source = r#""hello" "\nworld\t" "\u{00E9}""#;
        let expected = vec![
            Token::StringLiteral(Ok("hello".into())),
            Token::StringLiteral(Ok("\nworld\t".into())),
            Token::StringLiteral(Ok("é".into())),
        ];
        check_tokens(source, expected);
    }

    #[test]
    fn byte_string_literals() {
        let source = r#"b"hello" b"\n\x00\xFF""#;
        let expected = vec![
            Token::ByteStringLiteral(Ok(b"hello".to_vec())),
            Token::ByteStringLiteral(Ok(vec![b'\n', 0x00, 0xFF])),
        ];
        check_tokens(source, expected);
    }

    #[test]
    fn operators_and_delimiters() {
        check_tokens(
            "+ - * / % +% -? *! & | ^ << >> ~ == != < > <= >= = += -= *= /= ! ? . .. ..= :: : ; , ( ) { } [ ] -> @ => ... '",
            vec![
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::Percent,
                Token::PlusWrap,
                Token::MinusSaturate,
                Token::StarTrap,
                Token::Ampersand,
                Token::Pipe,
                Token::Caret,
                Token::Shl,
                Token::Shr,
                Token::Tilde,
                Token::EqEq,
                Token::Neq,
                Token::Lt,
                Token::Gt,
                Token::Le,
                Token::Ge,
                Token::Assign,
                Token::PlusEq,
                Token::MinusEq,
                Token::StarEq,
                Token::SlashEq,
                Token::Bang,
                Token::Question,
                Token::Dot,
                Token::DotDot,
                Token::DotDotEq,
                Token::ColonColon,
                Token::Colon,
                Token::Semicolon,
                Token::Comma,
                Token::LParen,
                Token::RParen,
                Token::LBrace,
                Token::RBrace,
                Token::LBracket,
                Token::RBracket,
                Token::Arrow,
                Token::At,
                Token::FatArrow,
                Token::Ellipsis,
                Token::Apostrophe,
            ],
        );
    }

    #[test]
    fn comments_and_docs() {
        let source = "// line comment\n/// doc comment\n//! module doc\nx";
        let mut lex = Token::lexer(source);
        loop {
            let tok = lex.next().unwrap().unwrap();
            if tok == Token::WhitespaceOrComment {
                continue;
            }
            assert_eq!(tok, Token::DocComment("doc comment".into()));
            break;
        }
        loop {
            let tok = lex.next().unwrap().unwrap();
            if tok == Token::WhitespaceOrComment {
                continue;
            }
            assert_eq!(tok, Token::ModuleDocComment("module doc".into()));
            break;
        }
        loop {
            let tok = lex.next().unwrap().unwrap();
            if tok == Token::WhitespaceOrComment {
                continue;
            }
            assert_eq!(tok, Token::Ident("x".into()));
            break;
        }
        assert!(lex.next().is_none());
    }

    #[test]
    fn block_comment_skip() {
        let source = "a/* block comment */b";
        let mut lex = Token::lexer(source);
        let mut toks = Vec::new();
        while let Some(t) = lex.next() {
            match t {
                Ok(Token::WhitespaceOrComment) => {}
                Ok(other) => toks.push(other),
                Err(_) => panic!("lexer error"),
            }
        }
        assert_eq!(
            toks,
            vec![Token::Ident("a".into()), Token::Ident("b".into())]
        );
    }

    #[test]
    fn invalid_char_error() {
        let source = "` hello";
        let mut lex = Token::lexer(source);
        assert!(lex.next().unwrap().is_err());
        loop {
            let tok = lex.next().unwrap().unwrap();
            if tok == Token::WhitespaceOrComment {
                continue;
            }
            assert_eq!(tok, Token::Ident("hello".into()));
            break;
        }
    }

    #[test]
    fn empty_input() {
        let source = "";
        let mut lex = Token::lexer(source);
        assert!(lex.next().is_none());
    }

    #[test]
    fn ascii_identifier() {
        check_tokens(
            "hello world",
            vec![Token::Ident("hello".into()), Token::Ident("world".into())],
        );
    }

    #[test]
    fn comprehensive_small_example() {
        let source = r#"
        edition = "2026";
        type Age = exists n: UInt<8> invariant n >= 18;
        def main() -> Result<(), AppError> {
            set x: Int<32> = 42 + 15;
            // line comment
            /// doc comment
            let y = "hello\nworld";
            return Ok(());
        }
        "#;
        let expected = vec![
            Token::Edition,
            Token::Assign,
            Token::StringLiteral(Ok("2026".into())),
            Token::Semicolon,
            Token::Type,
            Token::Ident("Age".into()),
            Token::Assign,
            Token::Exists,
            Token::Ident("n".into()),
            Token::Colon,
            Token::Ident("UInt".into()),
            Token::Lt,
            Token::IntLiteral(Ok(8)),
            Token::Gt,
            Token::Invariant,
            Token::Ident("n".into()),
            Token::Ge,
            Token::IntLiteral(Ok(18)),
            Token::Semicolon,
            Token::Def,
            Token::Ident("main".into()),
            Token::LParen,
            Token::RParen,
            Token::Arrow,
            Token::Ident("Result".into()),
            Token::Lt,
            Token::LParen,
            Token::RParen,
            Token::Comma,
            Token::Ident("AppError".into()),
            Token::Gt,
            Token::LBrace,
            Token::Set,
            Token::Ident("x".into()),
            Token::Colon,
            Token::Ident("Int".into()),
            Token::Lt,
            Token::IntLiteral(Ok(32)),
            Token::Gt,
            Token::Assign,
            Token::IntLiteral(Ok(42)),
            Token::Plus,
            Token::IntLiteral(Ok(15)),
            Token::Semicolon,
            Token::DocComment("doc comment".into()),
            Token::Let,
            Token::Ident("y".into()),
            Token::Assign,
            Token::StringLiteral(Ok("hello\nworld".into())),
            Token::Semicolon,
            Token::Return,
            Token::Ident("Ok".into()),
            Token::LParen,
            Token::LParen,
            Token::RParen,
            Token::RParen,
            Token::Semicolon,
            Token::RBrace,
        ];
        check_tokens(source, expected);
    }

    #[test]
    fn test_unicode_escape_handling() {
        assert_eq!(parse_string_literal(r#""\u{41}""#).unwrap(), "A");
        assert_eq!(parse_string_literal(r#""\u{0041}BC""#).unwrap(), "ABC");
        assert_eq!(parse_string_literal(r#""\u{7F}""#).unwrap(), "\u{7F}");
        assert!(parse_string_literal(r#""\u{41""#).is_err());
        assert!(parse_string_literal(r#""\u{GG}""#).is_err());
        assert_eq!(
            parse_string_literal(r#""hello\u{26}world""#).unwrap(),
            "hello\u{26}world"
        );
        assert_eq!(parse_char_literal(r"'\u{41}'").unwrap(), b'A');
        assert_eq!(parse_char_literal(r"'\u{7F}'").unwrap(), 0x7F);
        assert!(parse_char_literal(r"'\u{80}'").unwrap() == 0x80);
        assert!(parse_char_literal(r"'\u{41").is_err());
        check_tokens(
            r#""\u{41}BC""#,
            vec![Token::StringLiteral(Ok("ABC".into()))],
        );
        check_tokens(
            r#""\u{41""#,
            vec![Token::StringLiteral(Err(
                "unclosed \\u{...} in string literal".into(),
            ))],
        );
    }

    #[test]
    fn numeric_literal_safety_does_not_panic_on_overflow() {
        let huge_int = "999999999999999999999999999999999999999";
        let tokens: Vec<_> = Token::lexer(huge_int)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(
            tokens,
            vec![Token::IntLiteral(Err("integer literal overflow".into()))]
        );

        let huge_hex = "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
        let tokens: Vec<_> = Token::lexer(huge_hex)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(
            tokens,
            vec![Token::HexLiteral(Err("hex literal overflow".into()))]
        );

        let huge_float = "1e9999";
        let tokens: Vec<_> = Token::lexer(huge_float)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(
            tokens,
            vec![Token::FloatLiteral(Err(
                "float literal must be finite (no NaN or Inf)".into()
            ))]
        );
    }

    #[test]
    fn nested_block_comment_handling() {
        let source = "a/* still comment */b";
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(
            tokens,
            vec![Token::Ident("a".into()), Token::Ident("b".into())]
        );
    }

    #[test]
    fn unclosed_string_emits_error() {
        let source = "\"unclosed";
        let tokens: Vec<_> = collect_all_tokens(source);
        assert!(tokens.iter().any(|r| r.is_err()));
    }

    #[test]
    fn null_byte_in_source_emits_error() {
        let source = "a\0b";
        let tokens: Vec<_> = collect_all_tokens(source);
        assert!(tokens.iter().any(|r| r.is_err()));
    }

    #[test]
    fn escape_x_short_hex() {
        assert!(parse_string_literal(r#""\x1""#).is_err());
        assert!(parse_string_literal(r#""\xAG""#).is_err());
        assert!(parse_string_literal(r#""\x41A""#).is_err());
    }

    #[test]
    fn escape_u_incomplete() {
        assert!(parse_string_literal(r#""\u""#).is_err());
        assert!(parse_string_literal(r#""\u{""#).is_err());
        assert!(parse_string_literal(r#""\u{41""#).is_err());
    }

    #[test]
    fn escape_u_surrogate_pair_yields_error() {
        assert!(parse_string_literal(r#""\u{D800}""#).is_err());
    }

    #[test]
    fn escape_u_max_code_point() {
        let s = parse_string_literal(r#""\u{10FFFF}""#).unwrap();
        assert_eq!(s, "\u{10FFFF}");
    }

    #[test]
    fn multiple_escapes_in_one_string() {
        let src = r#""\n\t\\\0\x41\u{263A}""#;
        let expected = "\n\t\\\0\x41\u{263A}";
        assert_eq!(parse_string_literal(src).unwrap(), expected);
    }

    #[test]
    fn byte_string_escapes() {
        assert_eq!(
            parse_byte_string_literal(r#"b"\x00\x7F\x80\xFF""#).unwrap(),
            vec![0x00, 0x7F, 0x80, 0xFF]
        );
        assert!(parse_byte_string_literal(r#"b"\x1G""#).is_err());
        assert!(parse_byte_string_literal(r#"b"\u{41}""#).is_err());
    }

    #[test]
    fn char_literal_unicode_rejects_non_ascii() {
        assert!(parse_char_literal(r"'\u{100}'").is_err());
        assert!(parse_char_literal(r"'\u{D800}'").is_err());
    }

    #[test]
    fn ident_with_keyword_prefix() {
        check_tokens(
            "defi letx type2",
            vec![
                Token::Ident("defi".into()),
                Token::Ident("letx".into()),
                Token::Ident("type2".into()),
            ],
        );
    }

    #[test]
    fn long_identifier() {
        let long = "a".repeat(10_000);
        let source = long.as_str();
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(tokens, vec![Token::Ident(long)]);
    }

    #[test]
    fn many_errors_accumulated() {
        let source = "` ~ # $ % ^ & * ( )";
        let error_count = collect_all_tokens(source)
            .iter()
            .filter(|r| r.is_err())
            .count();
        assert!(error_count > 0);
    }

    #[test]
    fn unterminated_string() {
        let source = r#""unclosed"#;
        let result = collect_all_tokens(source);
        assert!(result.iter().any(|r| r.is_err()));
    }

    #[test]
    fn doc_comment_trimming() {
        let source = "///   hello world   \n";
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(tokens, vec![Token::DocComment("hello world".into())]);
    }

    #[test]
    fn module_doc_comment_trimming() {
        let source = "//!   module docs   \n";
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(tokens, vec![Token::ModuleDocComment("module docs".into())]);
    }

    #[test]
    fn line_comment_stops_at_newline() {
        check_tokens("// comment\na", vec![Token::Ident("a".into())]);
    }

    #[test]
    fn mixed_whitespace_and_comments() {
        check_tokens(
            " \t  // skip\n  /* skip */ \n x",
            vec![Token::Ident("x".into())],
        );
    }

    #[test]
    fn numeric_literal_with_underscores() {
        check_tokens(
            "1_000 0xDead_Beef 0b1111_0000",
            vec![
                Token::IntLiteral(Ok(1000)),
                Token::HexLiteral(Ok(0xDEADBEEF)),
                Token::BinLiteral(Ok(0b11110000)),
            ],
        );
    }

    #[test]
    fn integer_suffixes_with_underscores() {
        check_tokens(
            "1_000i32 0xFF_u8 0b1010u8",
            vec![
                Token::IntSuffix("1_000i32".into()),
                Token::HexUIntSuffix("0xFF_u8".into()),
                Token::BinUIntSuffix("0b1010u8".into()),
            ],
        );
    }

    #[test]
    fn lookahead_for_fat_arrow_and_ellipsis() {
        check_tokens(
            "=> ... .. .= ..=.",
            vec![
                Token::FatArrow,
                Token::Ellipsis,
                Token::DotDot,
                Token::Dot,
                Token::Assign,
                Token::DotDotEq,
                Token::Dot,
            ],
        );
    }

    #[test]
    fn consecutive_operators() {
        check_tokens(
            "+% -? *!",
            vec![Token::PlusWrap, Token::MinusSaturate, Token::StarTrap],
        );
    }

    #[test]
    fn block_comment_with_stars_and_slashes() {
        let source = "/***/a/*/*/b";
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(
            tokens,
            vec![Token::Ident("a".into()), Token::Ident("b".into())]
        );
    }

    #[test]
    fn multiple_errors_and_recovery() {
        let source = "`hello` world ` again";
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("hello".into()),
                Token::Ident("world".into()),
                Token::Ident("again".into()),
            ]
        );
    }

    #[test]
    fn error_token_generation() {
        let source = "`";
        let mut lex = Token::lexer(source);
        assert!(lex.next().unwrap().is_err());
    }

    #[test]
    fn span_after_lexing() {
        let source = "def foo";
        let mut lex = Token::lexer(source);
        let tok = lex.next().unwrap().unwrap();
        assert_eq!(tok, Token::Def);
        let span = lex.span();
        assert_eq!(&source[span], "def");
    }

    #[test]
    fn all_overflow_operators() {
        check_tokens(
            "+% -% *% +? -? *? +! -! *!",
            vec![
                Token::PlusWrap,
                Token::MinusWrap,
                Token::StarWrap,
                Token::PlusSaturate,
                Token::MinusSaturate,
                Token::StarSaturate,
                Token::PlusTrap,
                Token::MinusTrap,
                Token::StarTrap,
            ],
        );
    }

    #[test]
    fn float_special_values_are_rejected() {
        check_tokens(
            "1e9999",
            vec![Token::FloatLiteral(Err(
                "float literal must be finite (no NaN or Inf)".into(),
            ))],
        );
        check_tokens(
            "-1e9999",
            vec![
                Token::Minus,
                Token::FloatLiteral(Err("float literal must be finite (no NaN or Inf)".into())),
            ],
        );
    }

    #[test]
    fn byte_string_literal_with_invalid_hex() {
        let source = r#"b"\xGG""#;
        check_tokens(
            source,
            vec![Token::ByteStringLiteral(Err(
                "invalid hex digit in byte string literal".into(),
            ))],
        );
    }

    #[test]
    fn doc_comment_empty() {
        let source = "///\n";
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(tokens, vec![Token::DocComment(String::new())]);
    }

    #[test]
    fn module_doc_comment_empty() {
        let source = "//!\n";
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(tokens, vec![Token::ModuleDocComment(String::new())]);
    }

    #[test]
    fn apostrophe_in_attribute_access() {
        let source = "x'len y'first";
        check_tokens(
            source,
            vec![
                Token::Ident("x".into()),
                Token::Apostrophe,
                Token::Ident("len".into()),
                Token::Ident("y".into()),
                Token::Apostrophe,
                Token::Ident("first".into()),
            ],
        );
    }

    #[test]
    fn apostrophe_not_confusing_with_char_literal() {
        let source = "'a' x'len";
        check_tokens(
            source,
            vec![
                Token::CharLiteral(Ok(b'a')),
                Token::Ident("x".into()),
                Token::Apostrophe,
                Token::Ident("len".into()),
            ],
        );
    }

    #[test]
    fn carriage_return_skipped() {
        let source = "a\r\nb";
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(
            tokens,
            vec![Token::Ident("a".into()), Token::Ident("b".into())]
        );
    }

    #[test]
    fn invalid_char_escape_reports_error() {
        let source = r"'\q'";
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(
            tokens,
            vec![Token::CharLiteral(Err(
                "unknown escape sequence in char literal".into()
            ))]
        );
    }

    #[test]
    fn invalid_string_escape_reports_error() {
        let source = r#""\p""#;
        let tokens: Vec<_> = Token::lexer(source)
            .filter_map(|r| r.ok())
            .filter(|t| *t != Token::WhitespaceOrComment)
            .collect();
        assert_eq!(
            tokens,
            vec![Token::StringLiteral(Err(
                "unknown escape sequence '\\p' in string literal".into()
            ))]
        );
    }

    #[test]
    fn integer_max_values() {
        check_tokens("9223372036854775807", vec![Token::IntLiteral(Ok(9223372036854775807))]);
        check_tokens("0x7FFFFFFFFFFFFFFF", vec![Token::HexLiteral(Ok(0x7FFFFFFFFFFFFFFF))]);
    }

    #[test]
    fn escaped_quote_in_string() {
        let src = r#""\"""#;
        check_tokens(src, vec![Token::StringLiteral(Ok("\"".into()))]);
    }

    #[test]
    fn strict_hex_escape_length() {
        assert!(parse_string_literal(r#""\x41A""#).is_err());
        assert!(parse_char_literal(r"'\x41A'").is_err());
    }

    #[test]
    fn test_compact_code_no_spaces() {
        check_tokens(
            "def main(){set x=1+2;if(x<=3){return x;}}",
            vec![
                Token::Def,
                Token::Ident("main".into()),
                Token::LParen,
                Token::RParen,
                Token::LBrace,
                Token::Set,
                Token::Ident("x".into()),
                Token::Assign,
                Token::IntLiteral(Ok(1)),
                Token::Plus,
                Token::IntLiteral(Ok(2)),
                Token::Semicolon,
                Token::If,
                Token::LParen,
                Token::Ident("x".into()),
                Token::Le,
                Token::IntLiteral(Ok(3)),
                Token::RParen,
                Token::LBrace,
                Token::Return,
                Token::Ident("x".into()),
                Token::Semicolon,
                Token::RBrace,
                Token::RBrace,
            ],
        );
    }
}
