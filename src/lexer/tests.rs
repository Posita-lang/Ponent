//! Lexer tests.

use super::literal::*;
use super::*;
use crate::ast::IntLit;
use crate::lexer::Token;
use crate::symbol::Symbol;
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

/// Like `check_tokens`, but the expected token payloads are provided as
/// *live* slices borrowed from `source`.  Use this wherever the expected
/// payload slice is a runtime value (a `String`, a `&str` field, …) that
/// outlives the lexing call.
fn check_tokens_borrowed(source: &str, expected: impl FnOnce(&str) -> Vec<Token<'_>>) {
    let expected = expected(source);
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

fn collect_all_tokens(source: &str) -> Vec<Result<Token<'_>, ()>> {
    Token::lexer(source).collect()
}

#[test]
fn test_parse_char_literal_fn() {
    assert_eq!(parse_char_literal(r"'\n'").unwrap(), '\n' as u32);
    assert_eq!(parse_char_literal(r"'\x41'").unwrap(), 'A' as u32);
    assert_eq!(parse_char_literal(r"'\u{7F}'").unwrap(), 0x7F);
    assert_eq!(parse_char_literal(r"'a'").unwrap(), 'a' as u32);
    assert_eq!(parse_char_literal(r"'\u{80}'").unwrap(), 0x80);
    assert_eq!(parse_char_literal(r"'\u{FF}'").unwrap(), 0xFF);
    // SYNTAX.md: Char is a Unicode scalar value — code points above 0xFF
    // are legal; only surrogates and > 0x10FFFF are rejected.
    assert_eq!(parse_char_literal(r"'\u{100}'").unwrap(), 0x100);
    assert_eq!(parse_char_literal(r"'\u{10FFFF}'").unwrap(), 0x10FFFF);
    assert_eq!(parse_char_literal(r"'é'").unwrap(), 0xE9);
    assert!(parse_char_literal(r"'\u{D800}'").is_err());
    assert!(parse_char_literal(r"'\u{110000}'").is_err());
    assert!(parse_char_literal(r"'\u{}'").is_err());
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
        "def set type with default return if else for in while loop leave continue comptime generate import as true false auto",
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
            Token::Generate,
            Token::Import,
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
        "and or not sizeof alignof catch panic unsafe let finally where requires ensures invariant constraint move dyn by copy ref ro mut wrap saturate trap Self no_default extern pub edition deprecated experimental endian bit_order align pad packed async await task channel linear consume pure io trusted ghost scope_cleanup trigger validate missing_match apply_lemma exists forall trait impl decreases terminates cfg isolate hint must_use must_handle link_proof exhaustive no_alloc_error no_panic debug_info ieee_contracts old codomain audit_log interrupt propagates overrides",
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
            Token::Ro,
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
            Token::IeeeContracts,
            Token::Old,
            Token::Codomain,
            Token::AuditLog,
            Token::Interrupt,
            Token::Propagates,
            Token::Overrides,
        ],
    );
}

#[test]
fn every_keyword_resolves_in_path_position() {
    // Invariant: every `#[token("…")]` keyword in token.rs must lex as its
    // dedicated token (never as Ident) and resolve to its surface name via
    // `as_ident_symbol` — "any keyword is valid after ::" (`T::default()`).
    // The list below is every keyword surface form in token.rs; regenerate
    // with:
    //   grep -oP '#\[token\("\K[^"]+' src/lexer/token.rs
    // If you add a keyword and this test fails, you forgot the
    // `as_ident_symbol` arm (see the checklist in keyword.rs).
    let keywords = [
        "Self",
        "align",
        "alignof",
        "and",
        "apply_lemma",
        "as",
        "async",
        "audit_log",
        "auto",
        "await",
        "bit_order",
        "by",
        "catch",
        "cfg",
        "channel",
        "comptime",
        "const",
        "constraint",
        "consume",
        "continue",
        "copy",
        "debug_info",
        "decreases",
        "def",
        "default",
        "deprecated",
        "diverges",
        "dyn",
        "edition",
        "else",
        "endian",
        "ensures",
        "exhaustive",
        "exists",
        "experimental",
        "extern",
        "false",
        "finally",
        "for",
        "forall",
        "generate",
        "ghost",
        "hint",
        "ieee",
        "ieee_contracts",
        "if",
        "impl",
        "implies",
        "import",
        "in",
        "interrupt",
        "invariant",
        "io",
        "isolate",
        "layout",
        "leave",
        "let",
        "linear",
        "link_proof",
        "loop",
        "match",
        "missing_match",
        "move",
        "must_handle",
        "must_use",
        "mut",
        "no_alloc_error",
        "no_default",
        "no_panic",
        "not",
        "old",
        "or",
        "overrides",
        "packed",
        "pad",
        "panic",
        "poly",
        "propagates",
        "pub",
        "pure",
        "ref",
        "requires",
        "return",
        "saturate",
        "scope_cleanup",
        "set",
        "sizeof",
        "task",
        "terminates",
        "trait",
        "trap",
        "trigger",
        "true",
        "trusted",
        "type",
        "unbox",
        "unsafe",
        "validate",
        "when",
        "where",
        "while",
        "with",
        "wrap",
    ];
    for kw in keywords {
        let toks: Vec<_> = Token::lexer(kw).filter_map(|r| r.ok()).collect();
        assert_eq!(
            toks.len(),
            1,
            "`{kw}` must lex as exactly one token, got {toks:?}"
        );
        assert!(
            !matches!(toks[0], Token::Ident(_)),
            "`{kw}` has a dedicated #[token] and must not lex as Ident"
        );
        assert_eq!(
            toks[0].as_ident_symbol(),
            Some(Symbol::intern(kw)),
            "keyword `{kw}` must resolve to its surface name in path position"
        );
    }
}

/// SYNTAX.md: `ro` and `codomain` are reserved words, not identifiers —
/// `ro` is only valid immediately after `&`, `codomain` only in `ensures`
/// clauses.  They deliberately have NO `as_ident_symbol` arm; this pins
/// both that they lex as their dedicated tokens and that they never
/// resolve in path position.
#[test]
fn reserved_words_are_never_identifiers() {
    for kw in ["ro", "codomain"] {
        let toks: Vec<_> = Token::lexer(kw).filter_map(|r| r.ok()).collect();
        assert_eq!(toks.len(), 1, "`{kw}` must lex as one token, got {toks:?}");
        assert!(
            toks[0].as_ident_symbol().is_none(),
            "reserved word `{kw}` must NOT resolve in path position"
        );
    }
    check_tokens("ro codomain", vec![Token::Ro, Token::Codomain]);
    // Longer identifiers sharing the `ro` prefix are unaffected
    // (logos picks the longest match; `Ro` only wins on exact `ro`).
    check_tokens("round", vec![Token::Ident(Symbol::intern("round"))]);
    check_tokens("rope", vec![Token::Ident(Symbol::intern("rope"))]);
}

/// SYNTAX.md's keyword list does not include `from`, `on`, `on_timeout`,
/// `on_cancel`, `round`, `trunc`, `ceil`, `floor` — the parser matches
/// them contextually inside specific productions (`ensures on Ok/Err`,
/// `as Int<32> round`), so they must lex as plain
/// identifiers and resolve in path position.  Pin against re-reserving.
#[test]
fn contextual_keywords_are_plain_identifiers() {
    for kw in [
        "from",
        "on",
        "on_timeout",
        "on_cancel",
        "round",
        "trunc",
        "ceil",
        "floor",
    ] {
        let toks: Vec<_> = Token::lexer(kw).filter_map(|r| r.ok()).collect();
        assert_eq!(
            toks,
            vec![Token::Ident(Symbol::intern(kw))],
            "`{kw}` must lex as a plain identifier"
        );
        assert_eq!(
            toks[0].as_ident_symbol(),
            Some(Symbol::intern(kw)),
            "contextual keyword `{kw}` must resolve in path position"
        );
    }
}

/// SYNTAX.md: byte characters `b'x'` (type `Byte` = `UInt<8>`); `\u{...}`
/// is NOT valid in them.  Before the `ByteCharLiteral` token existed,
/// `b'x'` mis-lexed as `Ident("b")` followed by a char literal.
#[test]
fn byte_char_literals() {
    check_tokens(
        r"b'a' b'\n' b'\x00' b'\\' b'\''",
        vec![
            Token::ByteCharLiteral(r"b'a'"),
            Token::ByteCharLiteral(r"b'\n'"),
            Token::ByteCharLiteral(r"b'\x00'"),
            Token::ByteCharLiteral(r"b'\\'"),
            Token::ByteCharLiteral(r"b'\''"),
        ],
    );
    assert_eq!(parse_byte_char_literal(r"b'A'").unwrap(), b'A');
    assert_eq!(parse_byte_char_literal(r"b'\x41'").unwrap(), 0x41);
    assert_eq!(parse_byte_char_literal(r"b'\n'").unwrap(), b'\n');
    // \u{...} is rejected in byte chars (SYNTAX.md).
    assert!(parse_byte_char_literal(r"b'\u{41}'").is_err());
    // Non-ASCII source characters must go through \xNN.
    assert!(parse_byte_char_literal(r"b'é'").is_err());
    // Values above 0xFF cannot be expressed.
    assert!(parse_byte_char_literal(r"b'\xFF'").is_ok());
}

// (The Token size budget that used to live here as the `token_is_compact`
// test is now an always-on compile-time check —
// `static_assertions::const_assert!` in `token.rs`, right after the enum —
// so it runs in every build, not only under `cargo test`.  Its rationale
// lives there too.)

#[test]
fn integer_literals() {
    check_tokens(
        "42 0xFF 0b1010 42i32 0xFFu8 0b1101u4",
        vec![
            Token::IntLiteral("42"),
            Token::HexLiteral("0xFF"),
            Token::BinLiteral("0b1010"),
            Token::IntSuffix("42i32"),
            Token::HexUIntSuffix("0xFFu8"),
            Token::BinUIntSuffix("0b1101u4"),
        ],
    );
}

#[test]
fn integer_overflow_errors() {
    // These values overflow i128.  The lexer now emits the raw slice and
    // defers parsing to consumption time; overflow must therefore still be
    // detectable through the same production parse function the parser
    // uses.
    let huge_int = "999999999999999999999999999999999999999";
    let huge_hex = "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
    let huge_bin = "0b".to_string() + &"1".repeat(129);
    check_tokens_borrowed(huge_int, |s| vec![Token::IntLiteral(s)]);
    check_tokens_borrowed(huge_hex, |s| vec![Token::HexLiteral(s)]);
    check_tokens_borrowed(&huge_bin, |s| vec![Token::BinLiteral(s)]);
    // The parse function must still report the overflow errors.
    assert_eq!(
        parse_int_literal(huge_int, 10, "integer literal overflow").unwrap_err().as_str(),
        "integer literal overflow"
    );
    assert_eq!(
        parse_int_literal(&huge_hex[2..], 16, "hex literal overflow")
            .unwrap_err()
            .as_str(),
        "hex literal overflow"
    );
    assert_eq!(
        parse_int_literal(&huge_bin[2..], 2, "binary literal overflow")
            .unwrap_err()
            .as_str(),
        "binary literal overflow"
    );
    check_tokens("-42", vec![Token::Minus, Token::IntLiteral("42")]);
}

/// Differential oracle for the length-gated `parse_int_literal` rewrite:
/// a per-digit checked `u128` accumulation (the ungated semantics the
/// length-gated fast path must reproduce) must agree with the new
/// implementation — value AND error message — on every generated input,
/// including the gate boundaries (38/39 decimal, 32/33 hex, 128/129
/// binary digits), underscore-crossing shapes, and invalid digits.
/// Values above `i128::MAX` are legal up to `u128::MAX` (the `UInt<128>`
/// range) and must come out as `IntLit::Large`; the oracle's small/large
/// split goes through the production `IntLit::normalize`, not the
/// scanner's own helper, so the two are genuinely independent.
#[test]
fn parse_int_literal_matches_checked_reference() {
    // Ungated per-digit checked u128 accumulation; the error ordering
    // (invalid digit before overflow) matches the slow path.
    fn checked_reference(s: &str, radix: u32, overflow_msg: &str) -> Result<IntLit, LitError> {
        let overflow_err = LitError::from(overflow_msg);
        let mut acc: u128 = 0;
        for c in s.chars() {
            if c == '_' {
                continue;
            }
            let d = c
                .to_digit(radix)
                .ok_or(LitError::from("invalid digit in literal"))? as u128;
            acc = acc
                .checked_mul(radix as u128)
                .and_then(|v| v.checked_add(d))
                .ok_or(overflow_err)?;
        }
        Ok(IntLit::normalize(num_bigint::BigInt::from(acc)))
    }
    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut cases: Vec<(String, u32)> = Vec::new();
    for radix in [10u32, 16, 2] {
        // Adversarial alphabets: valid digits, `_`, and one invalid char
        // per radix so the invalid-digit path is exercised too.
        let alphabet: &[u8] = match radix {
            10 => b"0123456789_af",
            16 => b"0123456789abcdefABCDEF_g",
            _ => b"01_2",
        };
        for _ in 0..4000 {
            let len = (xorshift(&mut state) % 60 + 1) as usize;
            let mut s = String::with_capacity(len);
            for _ in 0..len {
                let idx = (xorshift(&mut state) as usize) % alphabet.len();
                s.push(alphabet[idx] as char);
            }
            cases.push((s, radix));
        }
    }
    // Gate boundaries and underscore-crossing shapes, exact lengths.
    cases.push(("9".repeat(38), 10)); // largest fast-path decimal
    cases.push(("9".repeat(39), 10)); // first slow-path decimal
    cases.push(("0".repeat(39), 10)); // slow path must not spuriously overflow
    cases.push(("F".repeat(32), 16)); // u128::MAX
    cases.push(("F".repeat(33), 16));
    cases.push(("8".repeat(32), 16)); // 0x8000…0 = 2^127 > i128::MAX → Large
    cases.push(("7".repeat(32), 16)); // just under the i128 boundary
    cases.push(("1".repeat(128), 2)); // u128::MAX
    cases.push(("1".repeat(129), 2));
    cases.push(("1_0".repeat(19), 10)); // 38 bytes / 19 digits
    cases.push(("1_0".repeat(20), 10)); // 40 bytes / 20 digits
    cases.push(("9_".repeat(19), 10)); // 38 bytes / 19 digits

    for (s, radix) in &cases {
        let expected = checked_reference(s, *radix, "overflow under test");
        let got = parse_int_literal(s, *radix, "overflow under test");
        assert_eq!(expected, got, "radix {radix}, input {s:?}");
    }
}

/// Differential oracle for [`parse_suffixed_int_literal`]: the VALUE
/// half is checked against an ungated per-digit checked `u128`
/// accumulation (values up to `u128::MAX` are legal — the `UInt<128>`
/// range — and split into `IntLit::Small`/`Large` by the production
/// `IntLit::normalize`) on regex-shaped inputs plus the gate / boundary /
/// underscore cases; the WIDTH half is checked against a plain `u32`
/// parse of the `[0-9]+` run.
#[test]
fn parse_suffixed_int_literal_matches_reference() {
    fn reference(s: &str, radix: u32) -> Option<(IntLit, u32)> {
        let body = match radix {
            16 | 2 => s.get(2..)?,
            _ => s,
        };
        // The token regexes pin `[DIGIT][DIGIT_]*` before the marker; a
        // marker-less body (unreachable for real tokens) fails closed
        // like the old `split(..).next()` shape.
        let marker = body.find(|c: char| c == 'i' || c == 'u')?;
        let mut acc: u128 = 0;
        for c in body[..marker].chars() {
            if c == '_' {
                continue;
            }
            let d = c.to_digit(radix)? as u128;
            acc = acc.checked_mul(radix as u128)?.checked_add(d)?;
        }
        // WIDTH — a plain u32 parse of the `[0-9]+` run, failing closed
        // on empty/overflow.
        let width: u32 = body[marker + 1..].parse().ok()?;
        Some((IntLit::normalize(num_bigint::BigInt::from(acc)), width))
    }
    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    // Pinned values for the tokens the lexer suite already exercises.
    assert_eq!(
        parse_suffixed_int_literal("42i32", 10),
        Some((IntLit::Small(42), 32))
    );
    assert_eq!(
        parse_suffixed_int_literal("1_000u8", 10),
        Some((IntLit::Small(1000), 8))
    );
    assert_eq!(
        parse_suffixed_int_literal("0xFFu8", 16),
        Some((IntLit::Small(255), 8))
    );
    assert_eq!(
        parse_suffixed_int_literal("0b1010u8", 2),
        Some((IntLit::Small(10), 8))
    );
    // An out-of-range width is NOT a lexer failure: the scanner returns
    // it and the resolver rejects `Int<999>` / `Int<0>` with a proper
    // diagnostic (parity with the hand-written `42: Int<999>`).
    assert_eq!(
        parse_suffixed_int_literal("42i999", 10),
        Some((IntLit::Small(42), 999))
    );
    assert_eq!(
        parse_suffixed_int_literal("42i0", 10),
        Some((IntLit::Small(42), 0))
    );
    // The `UInt<128>` top half is representable (Large), matching the
    // unsuffixed scanner — `…u128` is sugar for `…: UInt<128>`.
    assert_eq!(
        parse_suffixed_int_literal("340282366920938463463374607431768211455u128", 10),
        Some((
            IntLit::Large(Box::new(num_bigint::BigInt::from(u128::MAX))),
            128
        ))
    );
    // A width that cannot be represented at all fails closed...
    assert_eq!(
        parse_suffixed_int_literal("42i99999999999999999999", 10),
        None
    );
    // ...as do malformed shapes the regexes cannot produce: no `i`/`u`
    // marker, and an empty width run.
    assert_eq!(parse_suffixed_int_literal("42", 10), None);
    assert_eq!(parse_suffixed_int_literal("42i", 10), None);

    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut cases: Vec<(String, u32)> = Vec::new();
    for radix in [10u32, 16, 2] {
        // Regex-shaped alphabets: valid digits + `_` separators only.  The
        // invalid-digit arm is unreachable for regex-matched input, so the
        // generator stays inside the shape the lexer can produce.
        let alphabet: &[u8] = match radix {
            10 => b"0123456789_",
            16 => b"0123456789abcdefABCDEF_",
            _ => b"01_",
        };
        for _ in 0..3000 {
            let mut s = String::new();
            match radix {
                16 => s.push_str("0x"),
                2 => s.push_str("0b"),
                _ => {}
            }
            // `[DIGIT][DIGIT_]*`: the first byte is always a real digit.
            loop {
                let c = alphabet[(xorshift(&mut state) as usize) % alphabet.len()];
                if c != b'_' {
                    s.push(c as char);
                    break;
                }
            }
            let rest = (xorshift(&mut state) % 45) as usize;
            for _ in 0..rest {
                s.push(alphabet[(xorshift(&mut state) as usize) % alphabet.len()] as char);
            }
            s.push(if xorshift(&mut state) % 2 == 0 { 'i' } else { 'u' });
            // `[0-9]+` width — never empty, never underscored.
            let wlen = (xorshift(&mut state) % 6 + 1) as usize;
            for _ in 0..wlen {
                s.push((b'0' + (xorshift(&mut state) % 10) as u8) as char);
            }
            cases.push((s, radix));
        }
    }
    cases.extend([
        ("42i32".into(), 10),
        ("42u32".into(), 10),
        ("1_000i32".into(), 10),
        ("1_000u8".into(), 10),
        ("42i999".into(), 10), // width out of range is not a lexer error
        ("0i0".into(), 10),
        ("1u1".into(), 10),
        ("1__0i8".into(), 10), // `[0-9_]*` allows a double separator
        ("0xFFi16".into(), 16),
        ("0xFFu8".into(), 16),
        ("0xFF_u8".into(), 16),
        ("0x0i1".into(), 16),
        ("0b1101u4".into(), 2),
        ("0b1010u8".into(), 2),
        ("0b0u1".into(), 2),
        ("1_".repeat(19) + "1i8", 10), // 39 bytes → slow path, underscores
        ("9".repeat(38) + "i8", 10),   // largest fast-path decimal
        ("1".repeat(39) + "i8", 10),   // first slow-path decimal, no overflow
        ("9".repeat(39) + "i8", 10),   // slow-path overflow
        ("0".repeat(100) + "1i8", 10), // slow path: leading zeros must not overflow
        (format!("0x{}i8", "7".repeat(32)), 16), // i128::MAX
        (format!("0x{}i8", "8".repeat(32)), 16), // 2^127 > i128::MAX
        (format!("0x{}i8", "F".repeat(32)), 16), // u128::MAX
        (format!("0x{}i8", "F".repeat(33)), 16), // slow-path overflow
        (format!("0b{}u8", "1".repeat(127)), 2), // i128::MAX
        (format!("0b{}u8", "1".repeat(128)), 2), // u128::MAX
        (format!("0b{}u8", "1".repeat(129)), 2), // slow-path overflow
    ]);

    for (s, radix) in &cases {
        let expected = reference(s, *radix);
        let got = parse_suffixed_int_literal(s, *radix);
        assert_eq!(expected, got, "radix {radix}, input {s:?}");
    }
}

#[test]
fn float_literals() {
    check_tokens(
        "3.14 2.5e-3 1_000.5 1e10",
        vec![
            Token::FloatLiteral("3.14"),
            Token::FloatLiteral("2.5e-3"),
            Token::FloatLiteral("1_000.5"),
            Token::FloatLiteral("1e10"),
        ],
    );
}

#[test]
fn char_literals() {
    check_tokens(
        r"'\n' '\t' '\\' '\'' '\x41' 'a'",
        vec![
            Token::CharLiteral(r"'\n'"),
            Token::CharLiteral(r"'\t'"),
            Token::CharLiteral(r"'\\'"),
            Token::CharLiteral("'\\''"),
            Token::CharLiteral(r"'\x41'"),
            Token::CharLiteral(r"'a'"),
        ],
    );
}

#[test]
fn string_literals() {
    let source = r#""hello" "\nworld\t" "\u{00E9}""#;
    let expected = vec![
        Token::StringLiteral(r#""hello""#),
        Token::StringLiteral(r#""\nworld\t""#),
        Token::StringLiteral(r#""\u{00E9}""#),
    ];
    check_tokens(source, expected);
}

#[test]
fn byte_string_literals() {
    let source = r#"b"hello" b"\n\x00\xFF""#;
    let expected = vec![
        Token::ByteStringLiteral(r#"b"hello""#),
        Token::ByteStringLiteral(r#"b"\n\x00\xFF""#),
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
        assert_eq!(tok, Token::DocComment("doc comment"));
        break;
    }
    loop {
        let tok = lex.next().unwrap().unwrap();
        if tok == Token::WhitespaceOrComment {
            continue;
        }
        assert_eq!(tok, Token::ModuleDocComment("module doc"));
        break;
    }
    loop {
        let tok = lex.next().unwrap().unwrap();
        if tok == Token::WhitespaceOrComment {
            continue;
        }
        assert_eq!(tok, Token::Ident(Symbol::intern("x")));
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
        vec![
            Token::Ident(Symbol::intern("a")),
            Token::Ident(Symbol::intern("b"))
        ]
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
        assert_eq!(tok, Token::Ident(Symbol::intern("hello")));
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
        vec![
            Token::Ident(Symbol::intern("hello")),
            Token::Ident(Symbol::intern("world")),
        ],
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
        Token::StringLiteral(r#""2026""#),
        Token::Semicolon,
        Token::Type,
        Token::Ident(Symbol::intern("Age")),
        Token::Assign,
        Token::Exists,
        Token::Ident(Symbol::intern("n")),
        Token::Colon,
        Token::Ident(Symbol::intern("UInt")),
        Token::Lt,
        Token::IntLiteral("8"),
        Token::Gt,
        Token::Invariant,
        Token::Ident(Symbol::intern("n")),
        Token::Ge,
        Token::IntLiteral("18"),
        Token::Semicolon,
        Token::Def,
        Token::Ident(Symbol::intern("main")),
        Token::LParen,
        Token::RParen,
        Token::Arrow,
        Token::Ident(Symbol::intern("Result")),
        Token::Lt,
        Token::LParen,
        Token::RParen,
        Token::Comma,
        Token::Ident(Symbol::intern("AppError")),
        Token::Gt,
        Token::LBrace,
        Token::Set,
        Token::Ident(Symbol::intern("x")),
        Token::Colon,
        Token::Ident(Symbol::intern("Int")),
        Token::Lt,
        Token::IntLiteral("32"),
        Token::Gt,
        Token::Assign,
        Token::IntLiteral("42"),
        Token::Plus,
        Token::IntLiteral("15"),
        Token::Semicolon,
        Token::DocComment("doc comment"),
        Token::Let,
        Token::Ident(Symbol::intern("y")),
        Token::Assign,
        Token::StringLiteral(r#""hello\nworld""#),
        Token::Semicolon,
        Token::Return,
        Token::Ident(Symbol::intern("Ok")),
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
    assert_eq!(parse_char_literal(r"'\u{41}'").unwrap(), 'A' as u32);
    assert_eq!(parse_char_literal(r"'\u{7F}'").unwrap(), 0x7F);
    assert_eq!(parse_char_literal(r"'\u{80}'").unwrap(), 0x80);
    assert!(parse_char_literal(r"'\u{41").is_err());
    check_tokens(
        r#""\u{41}BC""#,
        vec![Token::StringLiteral(r#""\u{41}BC""#)],
    );
    let src = r#""\u{41""#;
    check_tokens_borrowed(src, |s| vec![Token::StringLiteral(s)]);
    // The error now surfaces at consumption time through the parse fn.
    let err = parse_string_literal(src).unwrap_err();
    assert_eq!(err.as_str(), "unclosed \\u{...} in string literal");
}

#[test]
fn numeric_literal_safety_does_not_panic_on_overflow() {
    // The lexer emits raw slices for all numerics; overflow no longer
    // occurs at lex time.  The parser's consumption path calls the
    // production `parse_int_literal` / `parse_float_literal`, which is
    // where overflow is detected.
    let huge_int = "999999999999999999999999999999999999999";
    let tokens: Vec<_> = Token::lexer(huge_int)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(tokens, vec![Token::IntLiteral(huge_int)]);
    // Consumption must still surface the overflow error.
    let err = parse_int_literal(huge_int, 10, "integer literal overflow").unwrap_err();
    assert_eq!(err.as_str(), "integer literal overflow");

    let huge_hex = "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
    let tokens: Vec<_> = Token::lexer(huge_hex)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(tokens, vec![Token::HexLiteral(huge_hex)]);
    let err = parse_int_literal(&huge_hex[2..], 16, "hex literal overflow").unwrap_err();
    assert_eq!(err.as_str(), "hex literal overflow");

    let huge_float = "1e9999";
    let tokens: Vec<_> = Token::lexer(huge_float)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(tokens, vec![Token::FloatLiteral(huge_float)]);
    let err = parse_float_literal(huge_float).unwrap_err();
    assert_eq!(err.as_str(), "float literal overflow");
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
        vec![
            Token::Ident(Symbol::intern("a")),
            Token::Ident(Symbol::intern("b"))
        ]
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
fn char_literal_accepts_non_ascii_scalars() {
    // SYNTAX.md: Char is a Unicode scalar value, so non-ASCII code points
    // are valid; surrogates and values above 0x10FFFF are not.
    assert_eq!(parse_char_literal(r"'\u{100}'").unwrap(), 0x100);
    assert_eq!(parse_char_literal(r"'é'").unwrap(), 0xE9);
    assert!(parse_char_literal(r"'\u{D800}'").is_err());
    assert!(parse_char_literal(r"'\u{DFFF}'").is_err());
    assert!(parse_char_literal(r"'\u{110000}'").is_err());
}

#[test]
fn ident_with_keyword_prefix() {
    check_tokens(
        "defi letx type2",
        vec![
            Token::Ident(Symbol::intern("defi")),
            Token::Ident(Symbol::intern("letx")),
            Token::Ident(Symbol::intern("type2")),
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
    assert_eq!(tokens, vec![Token::Ident(Symbol::intern(&long))]);
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
    assert_eq!(tokens, vec![Token::DocComment("hello world")]);
}

#[test]
fn module_doc_comment_trimming() {
    let source = "//!   module docs   \n";
    let tokens: Vec<_> = Token::lexer(source)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(tokens, vec![Token::ModuleDocComment("module docs")]);
}

#[test]
fn line_comment_stops_at_newline() {
    check_tokens("// comment\na", vec![Token::Ident(Symbol::intern("a"))]);
}

#[test]
fn mixed_whitespace_and_comments() {
    check_tokens(
        " \t  // skip\n  /* skip */ \n x",
        vec![Token::Ident(Symbol::intern("x"))],
    );
}

#[test]
fn numeric_literal_with_underscores() {
    check_tokens(
        "1_000 0xDead_Beef 0b1111_0000",
        vec![
            Token::IntLiteral("1_000"),
            Token::HexLiteral("0xDead_Beef"),
            Token::BinLiteral("0b1111_0000"),
        ],
    );
}

#[test]
fn integer_suffixes_with_underscores() {
    check_tokens(
        "1_000i32 0xFF_u8 0b1010u8",
        vec![
            Token::IntSuffix("1_000i32"),
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
fn nested_block_comment() {
    let source = "a/* outer /* inner */ still comment */b";
    let tokens: Vec<_> = Token::lexer(source)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(
        tokens,
        vec![
            Token::Ident(Symbol::intern("a")),
            Token::Ident(Symbol::intern("b"))
        ]
    );
}

#[test]
fn deeply_nested_block_comment() {
    let source = "/* a /* b /* c */ d */ e */ x";
    let tokens: Vec<_> = Token::lexer(source)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(tokens, vec![Token::Ident(Symbol::intern("x"))]);
}

#[test]
fn unterminated_block_comment_is_error() {
    let source = "a/* unterminated";
    let mut lex = Token::lexer(source);
    assert_eq!(
        lex.next().unwrap().unwrap(),
        Token::Ident(Symbol::intern("a"))
    );
    assert!(lex.next().unwrap().is_err());
    // Regression: the unterminated-comment error must consume the rest of
    // the input — no spurious token may follow the error.
    assert!(lex.next().is_none());
}

#[test]
fn block_comment_with_stars_and_slashes() {
    let source = "/***/a/*/*/b";
    let tokens: Vec<_> = Token::lexer(source)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    // `/*/*/b` opens a nested comment that never closes: the whole
    // remainder is consumed by the error, so no `Ident(b)` may follow.
    assert_eq!(tokens, vec![Token::Ident(Symbol::intern("a"))]);
    let saw_error = Token::lexer(source).any(|r| r.is_err());
    assert!(saw_error, "unterminated trailing comment must error");
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
            Token::Ident(Symbol::intern("hello")),
            Token::Ident(Symbol::intern("world")),
            Token::Ident(Symbol::intern("again")),
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
        vec![Token::FloatLiteral("1e9999")],
    );
    let err = parse_float_literal("1e9999").unwrap_err();
    assert_eq!(err.as_str(), "float literal overflow");
    check_tokens(
        "-1e9999",
        vec![
            Token::Minus,
            Token::FloatLiteral("1e9999"),
        ],
    );
}

#[test]
fn byte_string_literal_with_invalid_hex() {
    let source = r#"b"\xGG""#;
    check_tokens_borrowed(source, |s| vec![Token::ByteStringLiteral(s)]);
    let err = parse_byte_string_literal(source).unwrap_err();
    assert_eq!(err.as_str(), "invalid hex digit in byte string literal");
}

#[test]
fn doc_comment_empty() {
    let source = "///\n";
    let tokens: Vec<_> = Token::lexer(source)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(tokens, vec![Token::DocComment("")]);
}

#[test]
fn module_doc_comment_empty() {
    let source = "//!\n";
    let tokens: Vec<_> = Token::lexer(source)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(tokens, vec![Token::ModuleDocComment("")]);
}

#[test]
fn apostrophe_in_attribute_access() {
    let source = "x'len y'first";
    check_tokens(
        source,
        vec![
            Token::Ident(Symbol::intern("x")),
            Token::Apostrophe,
            Token::Ident(Symbol::intern("len")),
            Token::Ident(Symbol::intern("y")),
            Token::Apostrophe,
            Token::Ident(Symbol::intern("first")),
        ],
    );
}

#[test]
fn apostrophe_not_confusing_with_char_literal() {
    let source = "'a' x'len";
    check_tokens_borrowed(
        source,
        |s| {
            let char_slice = s.split_whitespace().next().unwrap();
            vec![
                Token::CharLiteral(char_slice),
                Token::Ident(Symbol::intern("x")),
                Token::Apostrophe,
                Token::Ident(Symbol::intern("len")),
            ]
        },
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
        vec![
            Token::Ident(Symbol::intern("a")),
            Token::Ident(Symbol::intern("b"))
        ]
    );
}

#[test]
fn invalid_char_escape_reports_error() {
    // SYNTAX.md §Escape Sequences: unknown escapes are hard
    // errors.  The lexer emits the raw slice as ONE CharLiteral token
    // (never split into stray tokens); the escape error surfaces at
    // consumption time through the same production parse function.
    let source = r"'\q'";
    let tokens: Vec<_> = Token::lexer(source)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(tokens, vec![Token::CharLiteral(source)]);
    let err = parse_char_literal(source).unwrap_err();
    assert_eq!(
        err.as_str(),
        "unknown escape sequence '\\q' in char literal"
    );
}

#[test]
fn invalid_string_escape_reports_error() {
    let source = r#""\p""#;
    let tokens: Vec<_> = Token::lexer(source)
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(tokens, vec![Token::StringLiteral(source)]);
    let err = parse_string_literal(source).unwrap_err();
    assert_eq!(
        err.as_str(),
        "unknown escape sequence '\\p' in string literal"
    );
}

#[test]
fn integer_max_values() {
    check_tokens(
        "9223372036854775807",
        vec![Token::IntLiteral("9223372036854775807")],
    );
    check_tokens(
        "0x7FFFFFFFFFFFFFFF",
        vec![Token::HexLiteral("0x7FFFFFFFFFFFFFFF")],
    );
}

#[test]
fn escaped_quote_in_string() {
    let src = r#""\"""#;
    check_tokens(src, vec![Token::StringLiteral(src)]);
}

#[test]
fn strict_hex_escape_length() {
    assert!(parse_string_literal(r#""\x41A""#).is_err());
    assert!(parse_char_literal(r"'\x41A'").is_err());
}

#[test]
fn hex_escape_multibyte_cases_keep_old_messages() {
    // The allocation-free rewrite gates on the UTF-8 byte length of the
    // collected run (what `String::len()` measured): a lone 2-byte char
    // passes the gate and is rejected as a non-hex digit, while a 2-byte
    // char plus a following digit is longer than 2 bytes.
    let err = parse_string_literal(r#""\xé""#).unwrap_err();
    assert!(err.as_str().contains("invalid hex digit"), "{}", err);
    let err = parse_string_literal(r#""\xéA""#).unwrap_err();
    assert!(
        err.as_str()
            .contains("\\x must be followed by exactly 2 hex digits"),
        "{}",
        err
    );
}

// ── SYNTAX.md lexical decisions ────────────────────────────

#[test]
fn escape_x_ascii_only_in_text_contexts() {
    // E-lex-escape-range: `\x` is a byte everywhere; in '...' / "..."
    // it is ASCII-only (0x00–0x7F).  \x7F is the boundary — allowed.
    assert_eq!(parse_string_literal(r#""\x7F""#).unwrap(), "\u{7F}");
    // \x80+ is rejected with a repair hint naming the code point.
    let err = parse_string_literal(r#""\xE9""#).unwrap_err();
    let msg = err.as_str();
    assert!(msg.contains("out of range"), "{msg}");
    assert!(
        msg.contains("\\u{E9}"),
        "hint must name the code point: {msg}"
    );
    assert!(
        msg.contains("b\"...\""),
        "hint must name byte strings: {msg}"
    );
    // Same rule in char literals.
    assert_eq!(parse_char_literal(r"'\x7F'").unwrap(), 0x7F);
    let err = parse_char_literal(r"'\xE9'").unwrap_err();
    assert!(err.as_str().contains("out of range"), "{}", err);
    // \u{...} remains the door for non-ASCII scalars.
    assert_eq!(parse_string_literal(r#""\u{E9}""#).unwrap(), "é");
    assert_eq!(parse_char_literal(r"'\u{E9}'").unwrap(), 0xE9);
    // Byte contexts keep the full 0x00–0xFF range.
    assert_eq!(
        parse_byte_string_literal(r#"b"\x80\xFF""#).unwrap(),
        vec![0x80, 0xFF]
    );
    assert_eq!(parse_byte_char_literal(r"b'\xFF'").unwrap(), 0xFF);
}

#[test]
fn removed_escapes_get_xnn_hint() {
    // The five removed escapes are unknown-escape hard errors whose
    // message points at the `\xNN` replacement.
    for (src, esc, nn) in [
        (r#""\a""#, "\\a", "x07"),
        (r#""\b""#, "\\b", "x08"),
        (r#""\v""#, "\\v", "x0B"),
        (r#""\f""#, "\\f", "x0C"),
        (r#""\?""#, "\\?", "x3F"),
    ] {
        let err = parse_string_literal(src).unwrap_err();
        let msg = err.as_str();
        assert!(msg.contains("unknown escape sequence"), "{msg}");
        assert!(msg.contains(esc), "{msg}");
        assert!(msg.contains(nn), "hint must give the \\xNN form: {msg}");
    }
}

#[test]
fn char_literal_must_be_exactly_one_unit() {
    // One-unit rule: `'ab'` is a hard error, not a truncated char.
    let err = parse_char_literal(r"'ab'").unwrap_err();
    assert!(err.as_str().contains("exactly one character"), "{}", err);
}

#[test]
fn bom_stripped_only_at_file_start() {
    // File-start BOM is an encoding signature: skipped, never a token.
    let toks: Vec<_> = Token::lexer("\u{FEFF}x")
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(toks, vec![Token::Ident(Symbol::intern("x"))]);
    // Mid-file BOM is a lexical error (exactly one, then lexing stops).
    let results: Vec<_> = Token::lexer("a\u{FEFF}b").collect();
    let errs = results.iter().filter(|r| r.is_err()).count();
    assert_eq!(errs, 1, "mid-file BOM must be one error: {results:?}");
}

#[test]
fn vertical_tab_is_lexical_whitespace() {
    // \x0B (VT) is in the ASCII lexical-whitespace closed set.
    let toks: Vec<_> = Token::lexer("a\u{0B}b")
        .filter_map(|r| r.ok())
        .filter(|t| *t != Token::WhitespaceOrComment)
        .collect();
    assert_eq!(
        toks,
        vec![
            Token::Ident(Symbol::intern("a")),
            Token::Ident(Symbol::intern("b"))
        ]
    );
    // ...and the `\x0B` escape still works inside strings (value layer).
    assert_eq!(parse_string_literal(r#""\x0B""#).unwrap(), "\u{0B}");
}

#[test]
fn test_compact_code_no_spaces() {
    check_tokens(
        "def main(){set x=1+2;if(x<=3){return x;}}",
        vec![
            Token::Def,
            Token::Ident(Symbol::intern("main")),
            Token::LParen,
            Token::RParen,
            Token::LBrace,
            Token::Set,
            Token::Ident(Symbol::intern("x")),
            Token::Assign,
            Token::IntLiteral("1"),
            Token::Plus,
            Token::IntLiteral("2"),
            Token::Semicolon,
            Token::If,
            Token::LParen,
            Token::Ident(Symbol::intern("x")),
            Token::Le,
            Token::IntLiteral("3"),
            Token::RParen,
            Token::LBrace,
            Token::Return,
            Token::Ident(Symbol::intern("x")),
            Token::Semicolon,
            Token::RBrace,
            Token::RBrace,
        ],
    );
}

/// LEXER-LEVEL coverage for the `ieee` keyword token (the parser's
/// `Token::Ieee` arm is dead code — the keyword flows through
/// `as_ident_symbol`, so a lexer regression would silently turn
/// `with overflow = ieee` into a bare identifier error at the
/// parser; this test pins the tokenization itself).
#[test]
fn test_ieee_keyword_token() {
    check_tokens("ieee", vec![Token::Ieee]);
    // The keyword-in-path-position property: keywords are NOT
    // reserved after `::` (`Mod::ieee` is a valid path segment).
    // The LEXER still emits the keyword token `Ieee` — the
    // identifier recovery happens in the PARSER via
    // `as_ident_symbol` (which maps `Token::Ieee` back to the
    // symbol `ieee`).  This pins the lexer-level tokenization.
    check_tokens(
        "Mod::ieee",
        vec![
            Token::Ident(Symbol::intern("Mod")),
            Token::ColonColon,
            Token::Ieee,
        ],
    );
}

/// Table test for `token_class` — `Shr` was missing from
/// `BINARY_OPERATOR` precisely because this function had zero tests.
#[test]
fn token_class_table() {
    use TokenClass as TC;
    let table: &[(Token, TC)] = &[
        // EXPR_START ∩ BINARY_OPERATOR (prefix unary AND infix binary).
        (Token::Plus, TC::EXPR_START | TC::BINARY_OPERATOR),
        (Token::Minus, TC::EXPR_START | TC::BINARY_OPERATOR),
        // BINARY_OPERATOR only.
        (Token::Star, TC::BINARY_OPERATOR),
        (Token::Slash, TC::BINARY_OPERATOR),
        (Token::Percent, TC::BINARY_OPERATOR),
        (Token::Shl, TC::BINARY_OPERATOR),
        // Regression: `Shr` was missing from BINARY_OPERATOR.
        (Token::Shr, TC::BINARY_OPERATOR),
        (Token::Ampersand, TC::BINARY_OPERATOR),
        (Token::Pipe, TC::BINARY_OPERATOR),
        (Token::Caret, TC::BINARY_OPERATOR),
        // Overflow-suffixed arithmetic are binary operators too.
        (Token::PlusWrap, TC::BINARY_OPERATOR),
        (Token::MinusWrap, TC::BINARY_OPERATOR),
        (Token::StarWrap, TC::BINARY_OPERATOR),
        (Token::PlusSaturate, TC::BINARY_OPERATOR),
        (Token::MinusSaturate, TC::BINARY_OPERATOR),
        (Token::StarSaturate, TC::BINARY_OPERATOR),
        (Token::PlusTrap, TC::BINARY_OPERATOR),
        (Token::MinusTrap, TC::BINARY_OPERATOR),
        (Token::StarTrap, TC::BINARY_OPERATOR),
        // COMPARISON.
        (Token::Gt, TC::COMPARISON),
        (Token::Ge, TC::COMPARISON),
        (Token::Lt, TC::COMPARISON),
        (Token::Le, TC::COMPARISON),
        (Token::EqEq, TC::COMPARISON),
        (Token::Neq, TC::COMPARISON),
        // EXPR_START.
        (Token::Ident(Symbol::intern("x")), TC::EXPR_START),
        (Token::IntLiteral("1"), TC::EXPR_START),
        (Token::HexLiteral("0x1"), TC::EXPR_START),
        (Token::BinLiteral("0b1"), TC::EXPR_START),
        (Token::FloatLiteral("1.0"), TC::EXPR_START),
        (Token::True, TC::EXPR_START),
        (Token::False, TC::EXPR_START),
        (Token::CharLiteral("'a'"), TC::EXPR_START),
        (Token::StringLiteral(r#""s""#), TC::EXPR_START),
        (
            Token::ByteStringLiteral(r#"b"b""#),
            TC::EXPR_START,
        ),
        (Token::LParen, TC::EXPR_START),
        (Token::LBracket, TC::EXPR_START),
        (Token::Bang, TC::EXPR_START),
        (Token::Tilde, TC::EXPR_START),
        // Deliberately unclassified: compound assignments are NOT binary
        // operators (documented in token_class).
        (Token::PlusEq, TC::empty()),
        (Token::ShlEq, TC::empty()),
        (Token::ShrEq, TC::empty()),
        (Token::Assign, TC::empty()),
        (Token::Def, TC::empty()),
        (Token::Semicolon, TC::empty()),
        (Token::Dot, TC::empty()),
    ];
    for (tok, expected) in table {
        assert_eq!(token_class(tok), *expected, "token {tok:?}");
    }
}

/// Table test for `to_user_string` — including the error paths that were
/// found swallowed for Hex/Bin literals and the Unicode-scalar
/// char rendering.
/// Raw overflow slices for the `to_user_string_table` error rows — the
/// token carries the raw text, and the table's expected strings are what
/// `to_user_string` prints after running the parser on that slice.
fn overflow_dec() -> String {
    "9".repeat(39)
}

fn overflow_hex() -> String {
    "0x".to_string() + &"F".repeat(40)
}

fn overflow_bin() -> String {
    "0b".to_string() + &"1".repeat(129)
}

#[test]
fn to_user_string_table() {
    // The error rows use raw slices that *produce* the error at
    // consumption time (the lazy design): the token carries the raw
    // text, and `to_user_string` runs the parser on it.
    let table: &[(Token, &str)] = &[
        (Token::Ident(Symbol::intern("x")), "`x`"),
        (Token::IntLiteral("42"), "the number `42`"),
        (
            Token::IntLiteral(&overflow_dec()),
            "invalid integer literal: integer literal overflow",
        ),
        // Regression: hex/bin errors must surface, not be
        // swallowed into a bare "… literal".
        (Token::HexLiteral("0xFF"), "the number `255`"),
        (
            Token::HexLiteral(&overflow_hex()),
            "invalid hex literal: hex literal overflow",
        ),
        (
            Token::BinLiteral(&overflow_bin()),
            "invalid binary literal: binary literal overflow",
        ),
        (
            Token::FloatLiteral("1e9999"),
            "invalid float literal: float literal overflow",
        ),
        (Token::CharLiteral("'a'"), "the character `'a'`"),
        (Token::CharLiteral("'\u{e9}'"), "the character `'é'`"),
        (
            Token::CharLiteral(r"'\q'"),
            "invalid character literal: unknown escape sequence '\\q' in char literal",
        ),
        (
            Token::StringLiteral(r#""\u{41""#),
            "invalid string literal: unclosed \\u{...} in string literal",
        ),
        (
            Token::ByteStringLiteral(r#"b"\x00\xFF""#),
            "byte string `[0, 255]`",
        ),
        (Token::Shl, "`<<`"),
        (Token::Shr, "`>>`"),
        (Token::PlusWrap, "`+%`"),
        (Token::SelfKw, "`Self`"),
        (Token::WhitespaceOrComment, "whitespace or comment"),
        (
            Token::DocComment("hello"),
            "documentation comment `///hello`",
        ),
        (
            Token::ModuleDocComment("mod"),
            "module documentation comment `//!mod`",
        ),
    ];
    for (tok, expected) in table {
        assert_eq!(tok.to_user_string(), *expected, "token {tok:?}");
    }
}

/// Pin the *lax* numeric lexes the regexes accept.  These are
/// intentional recovery choices, not spec'd forms — SYNTAX.md does not
/// even document `_` separators yet.  If a future change tightens the
/// regexes, update these expectations deliberately rather than by accident.
#[test]
fn lax_numeric_lexes_are_pinned() {
    // Trailing underscores parse cleanly after `_`-stripping.
    check_tokens("1_.5", vec![Token::FloatLiteral("1_.5")]);
    check_tokens("1.5_", vec![Token::FloatLiteral("1.5_")]);
    check_tokens("1e1_", vec![Token::FloatLiteral("1e1_")]);
    // `0b12` silently splits: the binary regex stops at the first
    // non-`01` digit and `2` re-lexes as a decimal literal.
    check_tokens(
        "0b12",
        vec![Token::BinLiteral("0b1"), Token::IntLiteral("2")],
    );
    // Bare `0x` has no hex digit after the prefix, so it falls apart
    // into `0` + identifier `x`.
    check_tokens(
        "0x",
        vec![Token::IntLiteral("0"), Token::Ident(Symbol::intern("x"))],
    );
    // A leading underscore is not a hex digit either: `0x_1` → same split.
    check_tokens(
        "0x_1",
        vec![
            Token::IntLiteral("0"),
            Token::Ident(Symbol::intern("x_1")),
        ],
    );
    // The suffix width is NOT validated by the lexer — `i999` matches the
    // IntSuffix regex; the parser desugars it to `Int<999>` and the
    // checker rejects the width there.
    check_tokens("42i999", vec![Token::IntSuffix("42i999")]);
}

