//! Escape-sequence parsing for character, string, byte-string, integer,
//! and float literals.
//!
//! These functions are called at LITERAL-CONSUMPTION TIME (the parser's
//! literal site, `display.rs::to_user_string`, and the test oracles) —
//! not at lex time.  The lexer hands the token a raw `&'src str` slice;
//! a token that is skipped, recovered over, or never consumed never
//! pays for parsing or for interning an error.  The functions must
//! never panic on malformed input: every rejection path returns `Err`.
//!
//! Errors are carried by [`LitError`] — an interned [`Symbol`] handle, so
//! a static message costs 4 bytes, is stored exactly once per process,
//! and never allocates at the error site.

use std::borrow::Cow;
use std::fmt;

use crate::ast::IntLit;
use crate::symbol::Symbol;
use memchr::memchr;
use num_bigint::BigInt;

/// A literal-parse error message returned by this module's parse
/// functions (called at literal-consumption time, not lex time).
///
/// History: `String` (32 bytes per payload, allocated on every error
/// branch), then `Cow<'static, str>` — whose zero-allocation story was
/// defeated the moment the token boundary called `.to_string()`.  Now an
/// interned [`Symbol`]: 4 bytes, `Copy`, and `PartialEq` compares interner
/// indices (equal strings are guaranteed equal indices), so tests can
/// build expected tokens from plain `'static` literals, and dynamic
/// messages (e.g. `invalid unicode scalar 0x…`) are deduplicated per
/// process.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LitError(Symbol);

impl LitError {
    /// The message text (a `&'static str` into the interner arena).
    pub fn as_str(self) -> &'static str {
        self.0.as_str()
    }
}

impl From<&str> for LitError {
    fn from(s: &str) -> Self {
        LitError(Symbol::intern(s))
    }
}

impl From<String> for LitError {
    fn from(s: String) -> Self {
        LitError(Symbol::intern(&s))
    }
}

impl From<LitError> for String {
    fn from(e: LitError) -> String {
        e.0.as_str().to_owned()
    }
}

impl fmt::Display for LitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0.as_str())
    }
}

impl fmt::Debug for LitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LitError({:?})", self.0.as_str())
    }
}

/// Scan the two hex digits of a `\xNN` escape (the caller has already
/// consumed `\x`).  `ctx` names the literal kind for error messages
/// (e.g. `"string literal"`).
///
/// Shared by the char / string / byte-string parsers so the
/// "exactly 2 hex digits, no third" rule lives in exactly one place.
///
/// Allocation-free: the two digits land in stack locals and the value is
/// computed by multiply/add.  The previous `take(2).collect::<String>()` +
/// `u8::from_str_radix` cost one heap `String` per escape, which dominated
/// escape-dense literals and large byte-string constant pools.  The gate
/// still measures the UTF-8 byte length of the collected run — exactly what
/// `String::len()` did — so one lone 2-byte char reaches the digit check and
/// every diagnostic stays byte-identical.
#[inline]
fn scan_hex_escape(chars: &mut std::str::Chars<'_>, ctx: &str) -> Result<u8, LitError> {
    let mut buf: [Option<char>; 2] = [None; 2];
    let mut bytes = 0usize;
    for slot in &mut buf {
        match chars.next() {
            Some(c) => {
                bytes += c.len_utf8();
                *slot = Some(c);
            }
            None => break,
        }
    }
    // Byte-length gate first, digit validity second — the order the
    // collect-then-check version had.
    if bytes != 2 {
        return Err("\\x must be followed by exactly 2 hex digits".into());
    }
    let [Some(d0), Some(d1)] = buf else {
        // A lone 2-byte char passed the gate; it is not an ASCII hex
        // digit, which is the same rejection the old `all(...)` made.
        return Err(format!("invalid hex digit in {}", ctx).into());
    };
    if !d0.is_ascii_hexdigit() || !d1.is_ascii_hexdigit() {
        return Err(format!("invalid hex digit in {}", ctx).into());
    }
    if let Some(c) = chars.clone().next()
        && c.is_ascii_hexdigit()
    {
        return Err("expected exactly 2 hex digits after \\x".into());
    }
    // Both digits are verified hex: 15*16 + 15 = 0xFF by construction, so
    // the direct sum needs no fallible parse.
    let val = d0.to_digit(16).unwrap() * 16 + d1.to_digit(16).unwrap();
    Ok(u8::try_from(val).expect("two hex digits fit in u8"))
}

/// Scan a `\u{NNNNNN}` escape (the caller has already consumed `\u`) and
/// return the raw code point; each caller applies its own validity policy
/// (string: `char::from_u32`; char: scalar-value check; byte string:
/// rejected outright).
///
/// Reads hex digits until `}` — more than 6 digits, a non-hex character,
/// a missing `}`, or an empty `{}` are all errors.  This plain loop
/// replaces the earlier `take(6)` + `closed` + peek dance.
///
/// Allocation-free: digits accumulate into a `u32` on the stack instead of
/// a `String`.  The six-digit cap bounds the value at 0xFFFFFF, so
/// `val * 16` cannot overflow and the old `u32::from_str_radix` round-trip
/// (whose error arm was already unreachable, since every char had passed
/// `is_ascii_hexdigit`) is gone.
#[inline]
fn scan_unicode_escape(chars: &mut std::str::Chars<'_>, ctx: &str) -> Result<u32, LitError> {
    if chars.next() != Some('{') {
        return Err(format!("expected '{{' after \\u in {}", ctx).into());
    }
    let mut val: u32 = 0;
    let mut n = 0usize;
    loop {
        match chars.next() {
            Some('}') => break,
            Some(c) if c.is_ascii_hexdigit() => {
                if n >= 6 {
                    return Err("too many hex digits in \\u{...} (max 6)".into());
                }
                val = val * 16 + c.to_digit(16).unwrap();
                n += 1;
            }
            Some(_) => {
                return Err(format!("invalid hex digit in \\u{{...}} in {}", ctx).into());
            }
            None => return Err(format!("unclosed \\u{{...}} in {}", ctx).into()),
        }
    }
    if n == 0 {
        return Err(format!("empty \\u{{...}} in {}", ctx).into());
    }
    Ok(val)
}

/// Build the hard error for an escape form outside the shared grammar
/// (`E-lex-unknown-escape`).  The compiler never degrades an unknown
/// escape to a literal backslash.  The five lexically-removed escapes
/// (`\a \b \v \f \?`) get a repair hint naming their exact `\xNN` form.
#[cold]
fn unknown_escape_error(c: char, ctx: &str) -> LitError {
    const REMOVED: &[(char, &str)] = &[
        ('a', "x07"),
        ('b', "x08"),
        ('v', "x0B"),
        ('f', "x0C"),
        ('?', "x3F"),
    ];
    let hint = match REMOVED.iter().find(|(e, _)| *e == c) {
        Some((_, nn)) => format!(
            "; \\{} was removed — write the equivalent \\xNN form (\\x{})",
            c, nn
        ),
        None => String::new(),
    };
    format!("unknown escape sequence '\\{}' in {}{}", c, ctx, hint).into()
}

/// Parse a character literal.  Per SYNTAX.md, `Char` is a Unicode scalar
/// value (`UInt<32>`, `value <= 0x10FFFF`, never a surrogate), so the
/// result is the scalar as `u32` — `'\u{10FFFF}'` and `'é'` are valid.
#[inline]
pub(crate) fn parse_char_literal(s: &str) -> Result<u32, LitError> {
    let inner = &s[1..s.len() - 1];
    let mut chars = inner.chars();
    let value = match chars.next() {
        Some('\\') => {
            let esc = chars
                .next()
                .ok_or("unexpected end of char literal after backslash")?;
            match esc {
                'n' => u32::from('\n'),
                'r' => u32::from('\r'),
                't' => u32::from('\t'),
                '\\' => u32::from('\\'),
                '"' => u32::from('"'),
                '\'' => u32::from('\''),
                '0' => u32::from('\0'),
                'x' => {
                    let b = scan_hex_escape(&mut chars, "char literal")?;
                    // E-lex-escape-range: `\x` is a byte everywhere; in
                    // text contexts it is ASCII-only (0x00–0x7F).  Code
                    // points go through `\u{...}`.
                    if b > 0x7F {
                        return Err(format!(
                            "\\x{:02X} in a char literal is out of range (ASCII 0x00-0x7F only); for the code point write \\u{{{:X}}}, for raw bytes use a byte string b\"...\"",
                            b, b
                        )
                        .into());
                    }
                    u32::from(b)
                }
                'u' => {
                    let code = scan_unicode_escape(&mut chars, "char literal")?;
                    // SYNTAX.md: `Char` is a Unicode scalar value — a
                    // surrogate half or anything above 0x10FFFF is
                    // rejected.
                    if (0xD800..=0xDFFF).contains(&code) || code > 0x10FFFF {
                        return Err(
                            "char literal \\u{...} must be a Unicode scalar value (not a surrogate, at most 0x10FFFF)"
                                .into(),
                        );
                    }
                    code
                }
                other => return Err(unknown_escape_error(other, "char literal")),
            }
        }
        // SYNTAX.md: a character literal is a Unicode scalar value, so
        // multi-byte source characters are accepted ('é', '中', …).
        Some(c) => u32::from(c),
        None => return Err("empty char literal".into()),
    };
    // One-unit rule: the literal must resolve to exactly one scalar value.
    if chars.next().is_some() {
        return Err(
            "char literal must contain exactly one character (use a string \"...\" for sequences)"
                .into(),
        );
    }
    Ok(value)
}

/// Parse a byte character literal `b'x'` (SYNTAX.md: type `Byte`, an alias
/// for `UInt<8>`).  Per SYNTAX.md, `\u{...}` is NOT valid here — only the
/// simple escapes and `\xNN`; a non-ASCII source character must be written
/// as `\xNN`.
#[inline]
pub(crate) fn parse_byte_char_literal(s: &str) -> Result<u8, LitError> {
    let inner = &s[2..s.len() - 1];
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
            Some('x') => scan_hex_escape(&mut chars, "byte char literal"),
            Some('u') => Err("\\u{...} is not allowed in byte char literals (use \\xNN)".into()),
            Some(other) => Err(unknown_escape_error(other, "byte char literal")),
            None => Err("unexpected end of byte char literal after backslash".into()),
        },
        Some(c) if c.is_ascii() => Ok(c as u8),
        Some(_) => Err("non-ASCII character not allowed in byte char literal (use \\xNN)".into()),
        None => Err("empty byte char literal".into()),
    }
}

pub(crate) fn parse_string_literal(s: &str) -> Result<Cow<'_, str>, LitError> {
    let inner = &s[1..s.len() - 1];
    // Fast path: no escape sequences — zero-copy borrow of the source slice.
    if !inner.contains('\\') {
        return Ok(Cow::Borrowed(inner));
    }
    // Escape path — cold: only reached when the literal actually contains a
    // backslash, which is the rare case for typical source files.
    parse_string_literal_escaped(inner)
}

/// Decode a string-literal body that is known to contain at least one
/// backslash.  Bulk-copies clean spans up to each `\`, decodes one escape,
/// and repeats — long strings with sparse escapes approach memcpy instead
/// of one `push` per char.
#[cold]
fn parse_string_literal_escaped(inner: &str) -> Result<Cow<'_, str>, LitError> {
    let mut result = String::with_capacity(inner.len());
    let mut rest = inner;
    while let Some(at) = memchr(b'\\', rest.as_bytes()) {
        result.push_str(&rest[..at]);
        let mut chars = rest[at..].chars();
        chars.next(); // the backslash
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
                let byte = scan_hex_escape(&mut chars, "string literal")?;
                // E-lex-escape-range: `\x` is a byte everywhere; in
                // text contexts it is ASCII-only (0x00–0x7F).  Code
                // points go through `\u{...}`.
                if byte > 0x7F {
                    return Err(format!(
                        "\\x{:02X} in a string literal is out of range (ASCII 0x00-0x7F only); for the code point write \\u{{{:X}}}, for raw bytes use a byte string b\"...\"",
                        byte, byte
                    )
                    .into());
                }
                result.push(char::from(byte));
            }
            'u' => {
                let code = scan_unicode_escape(&mut chars, "string literal")?;
                let c = std::char::from_u32(code).ok_or_else(|| {
                    format!("invalid unicode scalar {:#x} in string literal", code)
                })?;
                result.push(c);
            }
            _ => return Err(unknown_escape_error(next, "string literal")),
        }
        rest = chars.as_str();
    }
    result.push_str(rest);
    Ok(Cow::Owned(result))
}

pub(crate) fn parse_byte_string_literal(s: &str) -> Result<Cow<'_, [u8]>, LitError> {
    let inner = &s[2..s.len() - 1];
    // Fast path: no escapes and all ASCII — zero-copy borrow of the source
    // slice.  `str::is_ascii` is word-at-a-time, unlike a `bytes().all`.
    if !inner.contains('\\') && inner.is_ascii() {
        return Ok(Cow::Borrowed(inner.as_bytes()));
    }
    // Escape sequences only shorten or keep the length — a safe capacity
    // bound that avoids reallocations during growth.
    let mut result = Vec::with_capacity(inner.len());
    // Escape path: bulk-copy the clean span up to the next `\` (validating
    // it is ASCII), decode one escape, repeat.
    let mut rest = inner;
    while let Some(at) = memchr(b'\\', rest.as_bytes()) {
        let clean = &rest[..at];
        if !clean.is_ascii() {
            // Note: `\u{...}` is NOT valid in byte strings (see
            // SYNTAX.md, Escape Sequences) — the hint must not suggest it.
            return Err(
                "non-ASCII character not allowed in byte string literal (use \\xNN)".into(),
            );
        }
        result.extend_from_slice(clean.as_bytes());
        let mut chars = rest[at..].chars();
        chars.next(); // the backslash
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
                let byte = scan_hex_escape(&mut chars, "byte string literal")?;
                result.push(byte);
            }
            'u' => return Err("\\u{...} is not allowed in byte string literals".into()),
            other => return Err(unknown_escape_error(other, "byte string literal")),
        }
        rest = chars.as_str();
    }
    if !rest.is_ascii() {
        return Err("non-ASCII character not allowed in byte string literal (use \\xNN)".into());
    }
    result.extend_from_slice(rest.as_bytes());
    Ok(Cow::Owned(result))
}

pub(crate) fn parse_int_literal(
    s: &str,
    radix: u32,
    overflow_msg: &str,
) -> Result<IntLit, LitError> {
    // Error construction is LAZY (`ok_or_else`): `LitError::from` interns,
    // and the invalid-digit / overflow arms are unreachable for
    // regex-matched input — an eager `ok_or(LitError::from(..))` pays a
    // hash lookup per *digit*, which measured ~10× the arithmetic cost
    // (the digit loops below are the only work left on the happy path).
    // The closures pin the error type explicitly, so `?`'s residual
    // conversion has nothing left to infer.
    //
    // Fast path — length-gated unchecked accumulation.  For a byte length
    // within the radix's provable-fit bound the value cannot wrap `u128`,
    // so no per-digit `checked_*` is needed and the only post-processing
    // is the small/large split:
    //   radix 16: ≤ 32 bytes → ≤ 32 digits → ≤ 16³²−1 = u128::MAX
    //   radix 2:  ≤ 128 bytes → ≤ 128 digits → ≤ u128::MAX
    //   radix 10: ≤ 38 bytes → ≤ 38 digits → ≤ 10³⁸−1 < u128::MAX
    // Underscores count toward the byte length, so the digit count can
    // only be smaller — the gate is conservative, never wrong.  (The
    // sign never reaches this function: `-42` lexes as `Minus` + literal,
    // so unsigned accumulation is exact.)
    //
    // The `u128` cap is the type system's widest integer (`UInt<128>`):
    // values above `i128::MAX` become [`IntLit::Large`] (BigInt), values
    // above `u128::MAX` are `overflow_msg` — no type can hold them.
    let fast_max_bytes = match radix {
        16 => 32,
        2 => 128,
        10 => 38,
        _ => 0,
    };
    if s.len() <= fast_max_bytes {
        let mut acc: u128 = 0;
        for b in s.bytes() {
            if b == b'_' {
                continue;
            }
            let d = u128::from(
                char::from(b)
                    .to_digit(radix)
                    .ok_or_else(|| LitError::from("invalid digit in literal"))?,
            );
            acc = acc * u128::from(radix) + d;
        }
        return Ok(int_lit_from_u128(acc));
    }
    // Slow path — cold: only reached for literals too long to fit in u128
    // without checked arithmetic (≥ 39 decimal digits, ≥ 33 hex, ≥ 129 bin).
    parse_int_literal_slow(s, radix, overflow_msg)
}

/// Checked-accumulation path for integer literals too long for the fast
/// unchecked loop.  Cold: reached only when the digit run exceeds the
/// provable-fit bound for `radix`.
#[cold]
fn parse_int_literal_slow(
    s: &str,
    radix: u32,
    overflow_msg: &str,
) -> Result<IntLit, LitError> {
    let mut acc: u128 = 0;
    for c in s.chars() {
        if c == '_' {
            continue;
        }
        let d = u128::from(
            c.to_digit(radix)
                .ok_or_else(|| LitError::from("invalid digit in literal"))?,
        );
        acc = acc
            .checked_mul(u128::from(radix))
            .and_then(|v| v.checked_add(d))
            .ok_or_else(|| LitError::from(overflow_msg))?;
    }
    Ok(int_lit_from_u128(acc))
}

/// Split an unsigned 128-bit accumulator into the AST's small/large
/// representation: `i128`-representable values stay inline (the common
/// case — zero allocation), anything up to `u128::MAX` — the widest
/// integer type, `UInt<128>` — becomes a `BigInt`.  Both integer
/// scanners funnel through here so the split cannot drift between the
/// suffixed and unsuffixed paths.
fn int_lit_from_u128(acc: u128) -> IntLit {
    match i128::try_from(acc) {
        Ok(v) => IntLit::Small(v),
        Err(_) => IntLit::Large(Box::new(BigInt::from(acc))),
    }
}

/// Parse a suffixed integer literal token — the raw slice carried by
/// `Token::IntSuffix` and the five sibling variants (`42i32`, `1_000u8`,
/// `0xFFi16`, `0b1010u4`) — into its `(value, bit width)` pair.  `radix`
/// is the radix of the token's prefix (`0x` → 16, `0b` → 2, none → 10),
/// chosen by the caller from the token variant.
///
/// Zero allocation on the small path; `IntLit::Large` (the only
/// allocating shape) appears only for values above `i128::MAX`, which
/// are legal up to `u128::MAX` (`UInt<128>`).  `None` means the token
/// cannot be desugared: no `i`/`u` marker (malformed — unreachable
/// for regex-matched tokens), the value exceeds `u128::MAX`, or the
/// width run overflows `u32`.  The width itself is *not* range-checked here:
/// `42i999` yields `(42, 999)` and the checker rejects the width with a
/// diagnostic spanning the width literal, exactly as it does for the
/// hand-written `42: Int<999>` (SYNTAX.md §Literals: "The suffix is
/// syntactic sugar and the type is fully checked").  Unlike
/// [`parse_int_literal`] a failure does not build a [`LitError`] — the
/// call site reports its own diagnostic, and `LitError::from` would
/// intern on a path that never reads the message.
///
/// The six suffix regexes in `token.rs` pin the shape
/// `[0x|0b] DIGIT [DIGIT_]* (i|u) [0-9]+`, so the digit run is
/// non-empty, the width run is non-empty and `_`-free, and neither `i`
/// nor `u` is a valid digit in any of the three radixes — the first
/// `i`/`u` byte always terminates the digit run, with no radix
/// awareness needed.
pub(crate) fn parse_suffixed_int_literal(s: &str, radix: u32) -> Option<(IntLit, u32)> {
    // Skip the `0x` / `0b` prefix (the hex/bin regexes guarantee it); a
    // short slice fails closed instead of panicking on `&s[2..]`.
    let body = match radix {
        16 | 2 => s.get(2..)?,
        _ => s,
    };
    // The first `i`/`u` byte terminates the digit run.
    let marker = body.bytes().position(|b| b == b'i' || b == b'u')?;
    let digits = &body[..marker];
    // The old path's `from_str_radix("")` was an error; keep that.
    if digits.is_empty() {
        return None;
    }
    // Width run — `[0-9]+` per the regexes.  Hand-rolled checked
    // accumulation, like the value loops below: `str::parse::<u32>`
    // does not inline into this function and measures ~19 ns/literal
    // (2.5× the whole value scan).  An over-u32 width
    // (`42i99999999999999999999`) fails closed; an empty run is
    // unreachable for regex-matched tokens and fails closed too.
    let width_run = &body[marker + 1..];
    if width_run.is_empty() {
        return None;
    }
    let mut width: u32 = 0;
    for b in width_run.bytes() {
        let d = char::from(b).to_digit(10)?;
        width = width.checked_mul(10)?.checked_add(d)?;
    }
    // Fast path — length-gated unchecked accumulation, with the same
    // bound table and rationale as [`parse_int_literal`] (underscores
    // count toward the byte length, so the gate is conservative, never
    // wrong):
    //   radix 16: ≤ 32 bytes → ≤ 16³²−1 = u128::MAX
    //   radix 2:  ≤ 128 bytes → ≤ u128::MAX
    //   radix 10: ≤ 38 bytes → ≤ 10³⁸−1 < u128::MAX
    let fast_max_bytes = match radix {
        16 => 32,
        2 => 128,
        10 => 38,
        _ => 0,
    };
    if digits.len() <= fast_max_bytes {
        let mut acc: u128 = 0;
        for b in digits.bytes() {
            if b == b'_' {
                continue;
            }
            let d = u128::from(char::from(b).to_digit(radix)?);
            acc = acc * u128::from(radix) + d;
        }
        return Some((int_lit_from_u128(acc), width));
    }
    // Slow path — cold: only reached for runs too long to prove fit in u128
    // without checked arithmetic.
    parse_suffixed_int_literal_slow(digits, radix, width)
}

/// Checked-accumulation path for a suffixed integer literal's digit run that
/// exceeds the fast path's provable-fit bound.  Cold: reached only when the
/// run is ≥ 39 decimal / 33 hex / 129 binary digits.
#[cold]
fn parse_suffixed_int_literal_slow(
    digits: &str,
    radix: u32,
    width: u32,
) -> Option<(IntLit, u32)> {
    let mut acc: u128 = 0;
    for c in digits.chars() {
        if c == '_' {
            continue;
        }
        let d = u128::from(c.to_digit(radix)?);
        acc = acc.checked_mul(u128::from(radix))?.checked_add(d)?;
    }
    Some((int_lit_from_u128(acc), width))
}

/// Parse a float literal.  The fast path parses the ORIGINAL slice
/// directly (zero allocation) when it contains no `_` separator — which
/// is the overwhelming majority of literals; only literals WITH `_` fall
/// back to a cleaned String.  Errors are [`LitError`] statics, so the
/// happy path performs no allocation.
pub(crate) fn parse_float_literal(s: &str) -> Result<f64, LitError> {
    let parse = |t: &str| -> Result<f64, LitError> {
        match t.parse::<f64>() {
            Ok(val) => match val.classify() {
                std::num::FpCategory::Zero
                | std::num::FpCategory::Normal
                | std::num::FpCategory::Subnormal => Ok(val),
                // Float default `trap`: report the SPECIFIC
                // anomaly — overflow (→ ±inf) vs NaN — for the compile error.
                std::num::FpCategory::Infinite => Err("float literal overflow".into()),
                // Defense in depth: the numeric regexes cannot produce NaN
                // (`nan` lexes as an Ident); kept so a future regex change
                // fails closed instead of silently accepting NaN.
                std::num::FpCategory::Nan => Err("float literal NaN".into()),
            },
            Err(_) => Err("invalid float literal".into()),
        }
    };
    if !s.contains('_') {
        Ok(parse(s)?)
    } else if s.len() <= 64 {
        // Stripping `_` never grows the literal, so a 64-byte stack buffer
        // covers every realistic underscore-bearing literal without the
        // heap `String`.  The regexes bound literals only by input length,
        // so longer ones still take the allocating path below.
        let mut buf = [0u8; 64];
        let mut n = 0usize;
        for b in s.bytes() {
            if b != b'_' {
                buf[n] = b;
                n += 1;
            }
        }
        // Float-literal bytes are ASCII by the regexes; `from_utf8` keeps
        // the never-panic contract if that ever changes.
        match std::str::from_utf8(&buf[..n]) {
            Ok(cleaned) => Ok(parse(cleaned)?),
            Err(_) => Err("invalid float literal".into()),
        }
    } else {
        let cleaned: String = s.chars().filter(|c| *c != '_').collect();
        Ok(parse(&cleaned)?)
    }
}
