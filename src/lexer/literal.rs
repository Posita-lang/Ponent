//! Escape-sequence parsing for character, string, byte-string, integer,
//! and float literals.
//!
//! These functions are called by the Logos lexer callbacks during
//! tokenization. They must never panic on malformed input.

pub(crate) fn parse_char_literal(s: &str) -> Result<u8, String> {
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
                if let Some(c) = chars.clone().next()
                    && c.is_ascii_hexdigit()
                {
                    return Err("expected exactly 2 hex digits after \\x".to_string());
                }
                u8::from_str_radix(&hex, 16)
                    .map_err(|_| "invalid hex digit in char literal".to_string())
            }
            Some('u') => {
                if chars.next() != Some('{') {
                    return Err("expected '{' after \\u in char literal".to_string());
                }
                let mut buf = [0u8; 6];
                let mut len = 0;
                let mut closed = false;
                for c in chars.by_ref().take(6) {
                    if c == '}' {
                        closed = true;
                        break;
                    }
                    if !c.is_ascii_hexdigit() {
                        return Err("invalid hex digit in \\u{...} in char literal".to_string());
                    }
                    buf[len] = c as u8;
                    len += 1;
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
                let code = u32::from_str_radix(std::str::from_utf8(&buf[..len]).unwrap(), 16)
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
/// Parse a float literal: the fast path parses the ORIGINAL slice
/// directly (zero allocation) when it contains no `_` separator — which
/// is the overwhelming majority of literals; only literals WITH `_` fall
/// back to a cleaned String (the integer-side zero-allocation
/// optimization previously was not applied to floats).
/// Parse a float literal.  The error messages are STATIC strings — the
/// `&'static str` error type avoids allocating on every error branch
/// (the callers map to `String` only if they must — the happy path
/// performs no allocation).
pub(crate) fn parse_float_literal(s: &str) -> Result<f64, &'static str> {
    let parse = |t: &str| match t.parse::<f64>() {
        Ok(val) => match val.classify() {
            std::num::FpCategory::Zero
            | std::num::FpCategory::Normal
            | std::num::FpCategory::Subnormal => Ok(val),
            // The committee ruling (float default `trap`): report the SPECIFIC
            // anomaly — overflow (→ ±inf) vs NaN — for the compile error.
            std::num::FpCategory::Infinite => Err("float literal overflow"),
            std::num::FpCategory::Nan => Err("float literal NaN"),
        },
        Err(_) => Err("invalid float literal"),
    };
    if !s.contains('_') {
        parse(s)
    } else {
        let cleaned: String = s.chars().filter(|c| *c != '_').collect();
        parse(&cleaned)
    }
}
/// Parse an integer literal with `_` separators in a single pass — zero
/// allocation (the previous `replace('_', "")` + `parse` allocated a new
/// String and scanned twice).  Overflow is reported with checked
/// arithmetic.
pub(crate) fn parse_int_literal(s: &str, radix: u32, overflow_msg: &str) -> Result<i128, String> {
    let mut acc: i128 = 0;
    for c in s.chars() {
        if c == '_' {
            continue;
        }
        let d = c
            .to_digit(radix)
            .ok_or_else(|| "invalid digit in literal".to_string())? as i128;
        acc = acc
            .checked_mul(radix as i128)
            .and_then(|v| v.checked_add(d))
            .ok_or_else(|| overflow_msg.to_string())?;
    }
    Ok(acc)
}
pub(crate) fn parse_string_literal(s: &str) -> Result<String, String> {
    let inner = &s[1..s.len() - 1];
    let mut result = String::with_capacity(inner.len());
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
                    if let Some(c) = chars.clone().next()
                        && c.is_ascii_hexdigit()
                    {
                        return Err("expected exactly 2 hex digits after \\x".to_string());
                    }
                    let byte = u8::from_str_radix(&hex, 16)
                        .map_err(|_| "invalid hex digit in string literal".to_string())?;
                    result.push(byte as char);
                }
                'u' => {
                    if chars.next() != Some('{') {
                        return Err("expected '{' after \\u in string literal".to_string());
                    }
                    let mut buf = [0u8; 6];
                    let mut len = 0;
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
                        buf[len] = c as u8;
                        len += 1;
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
                    let code =
                        u32::from_str_radix(std::str::from_utf8(&buf[..len]).unwrap(), 16)
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
pub(crate) fn parse_byte_string_literal(s: &str) -> Result<Vec<u8>, String> {
    let inner = &s[2..s.len() - 1];
    // Escape sequences only shorten or keep the length — a safe capacity
    // bound that avoids reallocations during growth.
    let mut result = Vec::with_capacity(inner.len());
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
                    if let Some(c) = chars.clone().next()
                        && c.is_ascii_hexdigit()
                    {
                        return Err("expected exactly 2 hex digits after \\x".to_string());
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
            if !c.is_ascii() {
                return Err(
                    "non-ASCII character not allowed in byte string literal (use \\x or \\u{...})"
                        .to_string(),
                );
            }
            result.push(c as u8);
        }
    }
    Ok(result)
}
