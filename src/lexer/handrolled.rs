//! Hand-rolled lexer — produces a byte-identical `(Token, Span)` stream
//! matching the prior `logos`-generated lexer.
//!
//! Architecture: a single `Lexer` struct holding `src: &'src [u8]` and a
//! byte cursor `pos`.  `next()` loops on `dispatch` until a token= an Emit
//! is produced (skipping whitespace/line-comments) or EOF.  Each matcher
//! returns a `Step` describing how to advance.
//!
//! Correctness is pinned by `differential_lexer_matches_logos` (tests.rs),
//! which lexes a broad corpus with both this lexer and the live `logos`
//! `Token::lexer` and asserts byte-identical token streams.

use super::token::{Token, scan_block_comment, scan_whitespace};
use crate::symbol::Symbol;
use std::ops::Range;

/// Hand-rolled lexer cursor.  `src` is the raw source bytes (valid UTF-8,
/// borrowed from the caller's `&'src str`); `pos` is the next unread byte
/// offset; `last_span` is the span of the most recently emitted token
/// (mirroring `logos::Lexer::span`).
pub struct Lexer<'src> {
    src: &'src [u8],
    pos: usize,
    last_span: Range<usize>,
}

impl<'src> Lexer<'src> {
    #[inline]
    pub fn new(src: &'src str) -> Self {
        Lexer {
            src: src.as_bytes(),
            pos: 0,
            last_span: 0..0,
        }
    }

    #[inline]
    pub fn span(&self) -> Range<usize> {
        self.last_span.clone()
    }

    #[inline]
    fn byte(&self, off: usize) -> Option<u8> {
        self.src.get(self.pos + off).copied()
    }

    /// Slice a `&'src str` over `[start..end)`.  The source is valid UTF-8
    /// and every matcher stops on a UTF-8 boundary, so this never panics.
    #[inline]
    fn slice(&self, start: usize, end: usize) -> &'src str {
        std::str::from_utf8(&self.src[start..end]).expect("matcher stopped on char boundary")
    }
}

/// Outcome of matching at the current position.
enum Step<'src> {
    /// Emit `token` covering `[start..end)`.
    Emit(Token<'src>, usize, usize),
    /// Skip `[start..end)` (whitespace / line comment) — never enters the
    /// token stream, mirroring `logos::skip`.
    Skip(usize),
    /// Lexer error covering `[start..end)`, mirroring `logos::FilterResult::Error`.
    Error(usize, usize),
}

impl<'src> Token<'src> {
    /// Entry point — mirrors the prior `logos::Logos::lexer` derive.
    #[inline]
    pub fn lexer(src: &'src str) -> Lexer<'src> {
        Lexer::new(src)
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = Result<Token<'src>, ()>;

    fn next(&mut self) -> Option<Result<Token<'src>, ()>> {
        loop {
            if self.pos >= self.src.len() {
                return None;
            }
            let start = self.pos;
            let b = self.src[start];
            match self.dispatch(start, b) {
                Step::Emit(tok, s, e) => {
                    self.last_span = s..e;
                    self.pos = e;
                    return Some(Ok(tok));
                }
                Step::Skip(e) => {
                    self.pos = e;
                    continue;
                }
                Step::Error(s, e) => {
                    self.last_span = s..e;
                    self.pos = e;
                    return Some(Err(()));
                }
            }
        }
    }
}

impl<'src> Lexer<'src> {
    fn dispatch(&self, start: usize, b: u8) -> Step<'src> {
        use Token::*;
        match b {
            // ── Whitespace (SIMD run-scan, skip) ──
            0x09..=0x0D | 0x20 => {
                let n = scan_whitespace(&self.src[start..]);
                Step::Skip(start + n)
            }
            // ── `/` : comment / doc / block / SlashEq / Slash ──
            b'/' => match (self.byte(1), self.byte(2)) {
                (Some(b'/'), Some(b'/')) => self.doc_comment(start, 3),
                (Some(b'/'), Some(b'!')) => self.module_doc_comment(start, 3),
                (Some(b'/'), _) => self.line_comment(start),
                (Some(b'*'), _) => self.block_comment(start),
                (Some(b'='), _) => Step::Emit(SlashEq, start, start + 2),
                _ => Step::Emit(Slash, start, start + 1),
            },
            // ── BOM U+FEFF (EF BB BF) ──
            0xEF => match (self.byte(1), self.byte(2)) {
                (Some(0xBB), Some(0xBF)) => {
                    if start == 0 {
                        Step::Skip(start + 3)
                    } else {
                        Step::Error(start, start + 3)
                    }
                }
                _ => Step::Error(start, start + 1),
            },
            // ── `b` : byte-char / byte-string / ident (must precede the
            //    ident-range arm — `b` is in `a..=z`) ──
            b'b' => match self.byte(1) {
                Some(b'\'') => self.byte_char(start),
                Some(b'"') => self.byte_string(start),
                _ => self.ident(start),
            },
            // ── Ident start (and keyword classification) ──
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.ident(start),
            // ── Numeric literal ──
            b'0'..=b'9' => self.numeric(start),
            // ── Char literal (one-unit) or Apostrophe fallback ──
            b'\'' => self.char_literal(start),
            // ── String literal ──
            b'"' => self.string_literal(start),
            // ── Punctuation / operators (longest-match lookahead) ──
            b'+' => match self.byte(1) {
                Some(b'=') => Step::Emit(PlusEq, start, start + 2),
                Some(b'%') => Step::Emit(PlusWrap, start, start + 2),
                Some(b'?') => Step::Emit(PlusSaturate, start, start + 2),
                Some(b'!') => Step::Emit(PlusTrap, start, start + 2),
                _ => Step::Emit(Plus, start, start + 1),
            },
            b'-' => match self.byte(1) {
                Some(b'=') => Step::Emit(MinusEq, start, start + 2),
                Some(b'%') => Step::Emit(MinusWrap, start, start + 2),
                Some(b'?') => Step::Emit(MinusSaturate, start, start + 2),
                Some(b'!') => Step::Emit(MinusTrap, start, start + 2),
                Some(b'>') => Step::Emit(Arrow, start, start + 2),
                _ => Step::Emit(Minus, start, start + 1),
            },
            b'*' => match self.byte(1) {
                Some(b'=') => Step::Emit(StarEq, start, start + 2),
                Some(b'%') => Step::Emit(StarWrap, start, start + 2),
                Some(b'?') => Step::Emit(StarSaturate, start, start + 2),
                Some(b'!') => Step::Emit(StarTrap, start, start + 2),
                _ => Step::Emit(Star, start, start + 1),
            },
            b'%' => Step::Emit(Percent, start, start + 1),
            b'&' => Step::Emit(Ampersand, start, start + 1),
            b'|' => Step::Emit(Pipe, start, start + 1),
            b'^' => Step::Emit(Caret, start, start + 1),
            b'~' => Step::Emit(Tilde, start, start + 1),
            b'<' => match (self.byte(1), self.byte(2)) {
                (Some(b'<'), Some(b'=')) => Step::Emit(ShlEq, start, start + 3),
                (Some(b'<'), _) => Step::Emit(Shl, start, start + 2),
                (Some(b'='), _) => Step::Emit(Le, start, start + 2),
                _ => Step::Emit(Lt, start, start + 1),
            },
            b'>' => match (self.byte(1), self.byte(2)) {
                (Some(b'>'), Some(b'=')) => Step::Emit(ShrEq, start, start + 3),
                (Some(b'>'), _) => Step::Emit(Shr, start, start + 2),
                (Some(b'='), _) => Step::Emit(Ge, start, start + 2),
                _ => Step::Emit(Gt, start, start + 1),
            },
            b'=' => match self.byte(1) {
                Some(b'=') => Step::Emit(EqEq, start, start + 2),
                Some(b'>') => Step::Emit(FatArrow, start, start + 2),
                _ => Step::Emit(Assign, start, start + 1),
            },
            b'!' => match self.byte(1) {
                Some(b'=') => Step::Emit(Neq, start, start + 2),
                _ => Step::Emit(Bang, start, start + 1),
            },
            b'?' => Step::Emit(Question, start, start + 1),
            b'.' => match (self.byte(1), self.byte(2)) {
                (Some(b'.'), Some(b'.')) => Step::Emit(Ellipsis, start, start + 3),
                (Some(b'.'), Some(b'=')) => Step::Emit(DotDotEq, start, start + 3),
                (Some(b'.'), _) => Step::Emit(DotDot, start, start + 2),
                _ => Step::Emit(Dot, start, start + 1),
            },
            b':' => match self.byte(1) {
                Some(b':') => Step::Emit(ColonColon, start, start + 2),
                _ => Step::Emit(Colon, start, start + 1),
            },
            b';' => Step::Emit(Semicolon, start, start + 1),
            b',' => Step::Emit(Comma, start, start + 1),
            b'(' => Step::Emit(LParen, start, start + 1),
            b')' => Step::Emit(RParen, start, start + 1),
            b'{' => Step::Emit(LBrace, start, start + 1),
            b'}' => Step::Emit(RBrace, start, start + 1),
            b'[' => Step::Emit(LBracket, start, start + 1),
            b']' => Step::Emit(RBracket, start, start + 1),
            b'@' => Step::Emit(At, start, start + 1),
            // ── Unrecognised byte (e.g. stray UTF-8 lead, `$`, backtick) ──
            _ => Step::Error(start, start + 1),
        }
    }

    // ── Comments ──────────────────────────────────────────────────────

    /// `//[^\n]*` — line comment, skipped (never enters the token stream).
    #[inline]
    fn line_comment(&self, start: usize) -> Step<'src> {
        let body = &self.src[start + 2..];
        let n = memchr_newline(body);
        Step::Skip(start + 2 + n)
    }

    /// `///[^\n]*` — doc comment, emits `DocComment(slice[3..].trim())`.
    #[inline]
    fn doc_comment(&self, start: usize, prefix: usize) -> Step<'src> {
        let body = &self.src[start + prefix..];
        let n = memchr_newline(body);
        let end = start + prefix + n;
        Step::Emit(
            Token::DocComment(self.slice(start + prefix, end).trim()),
            start,
            end,
        )
    }

    /// `//![^\n]*` — module doc comment, emits `ModuleDocComment(slice[3..].trim())`.
    #[inline]
    fn module_doc_comment(&self, start: usize, prefix: usize) -> Step<'src> {
        let body = &self.src[start + prefix..];
        let n = memchr_newline(body);
        let end = start + prefix + n;
        Step::Emit(
            Token::ModuleDocComment(self.slice(start + prefix, end).trim()),
            start,
            end,
        )
    }

    /// `/* … */` (nested) — emits `WhitespaceOrComment` on close, `Error` if
    /// unterminated.  Mirrors `logos::FilterResult::Emit` (NOT skip).
    #[inline]
    fn block_comment(&self, start: usize) -> Step<'src> {
        let body = &self.src[start + 2..];
        match scan_block_comment(body) {
            Ok(consumed) => Step::Emit(Token::WhitespaceOrComment, start, start + 2 + consumed),
            Err(consumed) => Step::Error(start, start + 2 + consumed),
        }
    }

    // ── Identifiers + keywords ───────────────────────────────────────

    /// `[a-zA-Z_][a-zA-Z0-9_]*` then keyword classification.  `ro` and
    /// `codomain` resolve to their dedicated variants (reserved words);
    /// they are absent from `as_ident_symbol` but present here.
    #[inline]
    fn ident(&self, start: usize) -> Step<'src> {
        let n = 1 + scan_ident_continuation(&self.src[start + 1..]);
        let end = start + n;
        Step::Emit(classify_ident(self.slice(start, end)), start, end)
    }

    // ── Char literal (one-unit) ──────────────────────────────────────

    /// `'(?:[^'\\]|\\(?:x[0-9a-fA-F]{0,}|u\{?[0-9a-fA-F]{0,}\}?|[^\n]))'`.
    /// On any mismatch the regex does not match — fall back to `Apostrophe`
    /// (consume only the leading `'`), mirroring logos longest-match.
    #[inline]
    fn char_literal(&self, start: usize) -> Step<'src> {
        match try_char_literal(self.src, start) {
            Some(end) => Step::Emit(Token::CharLiteral(self.slice(start, end)), start, end),
            None => Step::Emit(Token::Apostrophe, start, start + 1),
        }
    }

    /// `b'(\\.|[^'\\])*'`.  On mismatch, `b` is an ident start — the caller
    /// re-dispatches via `ident` (see `dispatch`'s `b'b'` arm).
    #[inline]
    fn byte_char(&self, start: usize) -> Step<'src> {
        match try_byte_char(self.src, start) {
            Some(end) => Step::Emit(Token::ByteCharLiteral(self.slice(start, end)), start, end),
            None => self.ident(start),
        }
    }

    /// `b"(\\.|[^"\\])*"`.  On mismatch, `b` is an ident start.
    #[inline]
    fn byte_string(&self, start: usize) -> Step<'src> {
        match try_quoted(self.src, start + 2, b'"') {
            Ok(end) => Step::Emit(Token::ByteStringLiteral(self.slice(start, end)), start, end),
            Err(_) => self.ident(start),
        }
    }

    /// `"(\\.|[^"\\])*"`.  Unterminated → `Error` (no closing quote).
    #[inline]
    fn string_literal(&self, start: usize) -> Step<'src> {
        match try_quoted(self.src, start + 1, b'"') {
            Ok(end) => Step::Emit(Token::StringLiteral(self.slice(start, end)), start, end),
            Err(consumed) => Step::Error(start, consumed),
        }
    }

    // ── Numeric literals ─────────────────────────────────────────────

    fn numeric(&self, start: usize) -> Step<'src> {
        let (tok, end) = scan_numeric(self.src, start);
        Step::Emit(tok, start, end)
    }
}

// ── Free functions (no `&self` needed) ────────────────────────────────

#[inline]
fn memchr_newline(bytes: &[u8]) -> usize {
    memchr::memchr(b'\n', bytes).unwrap_or(bytes.len())
}

#[inline]
fn is_ident(b: u8) -> bool {
    matches!(b, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
}

/// Length of the `[a-zA-Z0-9_]*` run at the start of `bytes` (SIMD).
#[inline]
fn scan_ident_continuation(bytes: &[u8]) -> usize {
    let len = bytes.len();
    if len < 16 {
        return scan_ident_scalar(bytes);
    }
    let p = bytes.as_ptr();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { scan_ident_avx2(p, len) };
        }
        return unsafe { scan_ident_sse2(p, len) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { scan_ident_neon(p, len) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scan_ident_scalar(bytes)
    }
}

#[inline]
fn scan_ident_scalar(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .position(|&b| !is_ident(b))
        .unwrap_or(bytes.len())
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn scan_ident_sse2(mut p: *const u8, mut len: usize) -> usize {
    use std::arch::x86_64::*;
    let start = p;
    let (lo_d, hi_d, lo_u, hi_u, lo_l, hi_l, us) = unsafe {
        (
            _mm_set1_epi8(0x2F),
            _mm_set1_epi8(0x3A),
            _mm_set1_epi8(0x40),
            _mm_set1_epi8(0x5B),
            _mm_set1_epi8(0x60),
            _mm_set1_epi8(0x7B),
            _mm_set1_epi8(0x5F),
        )
    };
    while len >= 16 {
        let mask = unsafe {
            let v = _mm_loadu_si128(p as *const __m128i);
            let d = _mm_and_si128(_mm_cmpgt_epi8(v, lo_d), _mm_cmpgt_epi8(hi_d, v));
            let u = _mm_and_si128(_mm_cmpgt_epi8(v, lo_u), _mm_cmpgt_epi8(hi_u, v));
            let l = _mm_and_si128(_mm_cmpgt_epi8(v, lo_l), _mm_cmpgt_epi8(hi_l, v));
            let s = _mm_cmpeq_epi8(v, us);
            _mm_movemask_epi8(_mm_or_si128(_mm_or_si128(_mm_or_si128(d, u), l), s)) as u32
        };
        let non = !mask & 0xFFFF;
        if non != 0 {
            return (p as usize - start as usize) + non.trailing_zeros() as usize;
        }
        p = unsafe { p.add(16) };
        len -= 16;
    }
    let off = p as usize - start as usize;
    off + scan_ident_scalar(unsafe { std::slice::from_raw_parts(p, len) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn scan_ident_avx2(mut p: *const u8, mut len: usize) -> usize {
    use std::arch::x86_64::*;
    let start = p;
    let (lo_d, hi_d, lo_u, hi_u, lo_l, hi_l, us) = unsafe {
        (
            _mm256_set1_epi8(0x2F),
            _mm256_set1_epi8(0x3A),
            _mm256_set1_epi8(0x40),
            _mm256_set1_epi8(0x5B),
            _mm256_set1_epi8(0x60),
            _mm256_set1_epi8(0x7B),
            _mm256_set1_epi8(0x5F),
        )
    };
    while len >= 32 {
        let mask = unsafe {
            let v = _mm256_loadu_si256(p as *const __m256i);
            let d = _mm256_and_si256(_mm256_cmpgt_epi8(v, lo_d), _mm256_cmpgt_epi8(hi_d, v));
            let u = _mm256_and_si256(_mm256_cmpgt_epi8(v, lo_u), _mm256_cmpgt_epi8(hi_u, v));
            let l = _mm256_and_si256(_mm256_cmpgt_epi8(v, lo_l), _mm256_cmpgt_epi8(hi_l, v));
            let s = _mm256_cmpeq_epi8(v, us);
            _mm256_movemask_epi8(_mm256_or_si256(
                _mm256_or_si256(_mm256_or_si256(d, u), l),
                s,
            )) as u32
        };
        let non = !mask;
        if non != 0 {
            return (p as usize - start as usize) + non.trailing_zeros() as usize;
        }
        p = unsafe { p.add(32) };
        len -= 32;
    }
    let off = p as usize - start as usize;
    off + unsafe { scan_ident_sse2(p, len) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn scan_ident_neon(mut p: *const u8, mut len: usize) -> usize {
    use std::arch::aarch64::*;
    let start = p;
    let (lo_d, hi_d, lo_u, hi_u, lo_l, hi_l, us) = unsafe {
        (
            vdupq_n_s8(0x2F),
            vdupq_n_s8(0x3A),
            vdupq_n_s8(0x40),
            vdupq_n_s8(0x5B),
            vdupq_n_s8(0x60),
            vdupq_n_s8(0x7B),
            vdupq_n_s8(0x5F),
        )
    };
    while len >= 16 {
        let all_ident = unsafe {
            let v = vld1q_u8(p);
            let vs = vreinterpretq_s8_u8(v);
            let d = vandq_s8(vcgtq_s8(vs, lo_d), vcgtq_s8(hi_d, vs));
            let u = vandq_s8(vcgtq_s8(vs, lo_u), vcgtq_s8(hi_u, vs));
            let l = vandq_s8(vcgtq_s8(vs, lo_l), vcgtq_s8(hi_l, vs));
            let s = vceqq_s8(vs, us);
            let id = vorrq_s8(vorrq_s8(vorrq_s8(d, u), l), s);
            vminvq_s8(id) == -1
        };
        if !all_ident {
            let chunk = unsafe { std::slice::from_raw_parts(p, 16) };
            return (p as usize - start as usize) + scan_ident_scalar(chunk);
        }
        p = p.add(16);
        len -= 16;
    }
    let off = p as usize - start as usize;
    off + scan_ident_scalar(unsafe { std::slice::from_raw_parts(p, len) })
}

#[inline]
fn is_hex(b: u8) -> bool {
    matches!(b, b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F')
}

#[inline]
fn is_bin(b: u8) -> bool {
    matches!(b, b'0' | b'1')
}

#[inline]
fn is_dec(b: u8) -> bool {
    b.is_ascii_digit()
}

/// Try to match the char-literal regex at `start` (`src[start] == b'\''`).
/// Returns `Some(end)` on success, `None` on mismatch (caller emits
/// `Apostrophe`).
fn try_char_literal(src: &[u8], start: usize) -> Option<usize> {
    let mut i = start + 1;
    let b = *src.get(i)?;
    if b == b'\\' {
        // escape: \x[hex]* | \u\{?[hex]*\}? | \[^\n]
        i += 1;
        let e = *src.get(i)?;
        match e {
            b'x' => {
                i += 1;
                while i < src.len() && is_hex(src[i]) {
                    i += 1;
                }
            }
            b'u' => {
                i += 1;
                if src.get(i) == Some(&b'{') {
                    i += 1;
                }
                while i < src.len() && is_hex(src[i]) {
                    i += 1;
                }
                if src.get(i) == Some(&b'}') {
                    i += 1;
                }
            }
            b'\n' => return None, // [^\n]
            _ => i += 1,          // unknown escape = one unit
        }
    } else if b == b'\'' || b == b'\\' {
        return None; // [^'\\]
    } else {
        i += 1; // plain byte (logos regex is byte-wise here)
    }
    if src.get(i) == Some(&b'\'') {
        Some(i + 1)
    } else {
        None
    }
}

/// Try `b'(\\.|[^'\\])*'` at `start` (`src[start..start+2] == b"b'"`).
fn try_byte_char(src: &[u8], start: usize) -> Option<usize> {
    let mut i = start + 2;
    loop {
        let b = *src.get(i)?;
        if b == b'\'' {
            return Some(i + 1);
        }
        if b == b'\\' {
            i += 1;
            let e = *src.get(i)?;
            if e == b'\n' {
                return None; // \.  the . does not cross a newline
            }
            i += 1;
        } else {
            i += 1; // [^'\\]
        }
    }
}

/// Try a `"`- or `b"`-delimited run `(\\.|[^delim\\])*delim` starting at
/// `body_start` (just past the opening quote).  Returns `Some(end)` (one
/// past the closing quote) or `None` (unterminated).
fn try_quoted(src: &[u8], body_start: usize, delim: u8) -> Result<usize, usize> {
    let mut i = body_start;
    loop {
        let b = match src.get(i) {
            Some(&b) => b,
            None => return Err(i),
        };
        if b == delim {
            return Ok(i + 1);
        }
        if b == b'\\' {
            i += 1;
            match src.get(i) {
                Some(&e) if e == b'\n' => return Err(i),
                Some(_) => i += 1,
                None => return Err(i),
            }
        } else {
            i += 1;
        }
    }
}

/// Scan a numeric literal at `start` (`src[start]` is `0..=9`).
/// Returns `(token, end)`.  Order mirrors logos longest-match: the longest
/// matching prefix wins, with prefix-specific regexes (hex/bin/suffix)
/// tried before plain int/float.
fn scan_numeric<'src>(src: &'src [u8], start: usize) -> (Token<'src>, usize) {
    if src[start] == b'0' {
        match src.get(start + 1).copied() {
            Some(b'x') | Some(b'X') => return scan_hex(src, start),
            Some(b'b') | Some(b'B') => return scan_bin(src, start),
            _ => {}
        }
    }
    scan_decimal(src, start)
}

/// `0x[0-9a-fA-F][0-9a-fA-F_]*` (optionally suffixed `i`/`u` + digits).
/// If the byte after `0x` is not a hex digit, fall back to `IntLiteral("0")`
/// and let the caller re-lex `x…` as an ident.
fn scan_hex<'src>(src: &'src [u8], start: usize) -> (Token<'src>, usize) {
    let after_prefix = start + 2;
    if src.get(after_prefix).map(|&b| is_hex(b)) != Some(true) {
        return (Token::IntLiteral("0"), start + 1);
    }
    let mut i = after_prefix + 1;
    while i < src.len() && (is_hex(src[i]) || src[i] == b'_') {
        i += 1;
    }
    let body_end = i;
    if let Some(end) = scan_suffix(src, i) {
        let s = std::str::from_utf8(&src[start..end]).unwrap();
        return (token_hex_suffix(s, src[body_end]), end);
    }
    let s = std::str::from_utf8(&src[start..body_end]).unwrap();
    (Token::HexLiteral(s), body_end)
}

/// `0b[01][01_]*` (optionally suffixed).
fn scan_bin<'src>(src: &'src [u8], start: usize) -> (Token<'src>, usize) {
    let after_prefix = start + 2;
    if src.get(after_prefix).map(|&b| is_bin(b)) != Some(true) {
        return (Token::IntLiteral("0"), start + 1);
    }
    let mut i = after_prefix + 1;
    while i < src.len() && (is_bin(src[i]) || src[i] == b'_') {
        i += 1;
    }
    let body_end = i;
    if let Some(end) = scan_suffix(src, i) {
        let s = std::str::from_utf8(&src[start..end]).unwrap();
        return (token_bin_suffix(s, src[body_end]), end);
    }
    let s = std::str::from_utf8(&src[start..body_end]).unwrap();
    (Token::BinLiteral(s), body_end)
}

/// Decimal int / float, optionally suffixed.
fn scan_decimal<'src>(src: &'src [u8], start: usize) -> (Token<'src>, usize) {
    let mut i = start + 1;
    while i < src.len() && (is_dec(src[i]) || src[i] == b'_') {
        i += 1;
    }
    // float: `[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9][0-9_]*)?`
    if src.get(i) == Some(&b'.') && src.get(i + 1).map(|&b| is_dec(b)) == Some(true) {
        i += 1;
        while i < src.len() && (is_dec(src[i]) || src[i] == b'_') {
            i += 1;
        }
        i = scan_exponent(src, i);
        let s = std::str::from_utf8(&src[start..i]).unwrap();
        return (Token::FloatLiteral(s), i);
    }
    // float: `[0-9][0-9_]*[eE][+-]?[0-9][0-9_]*`
    if matches!(src.get(i), Some(b'e') | Some(b'E')) {
        if let Some(end) = try_exponent(src, i) {
            let s = std::str::from_utf8(&src[start..end]).unwrap();
            return (Token::FloatLiteral(s), end);
        }
    }
    let body_end = i;
    // int suffix `[0-9][0-9_]*[iu][0-9]+`
    if let Some(end) = scan_suffix(src, i) {
        let s = std::str::from_utf8(&src[start..end]).unwrap();
        return (token_dec_suffix(s, src[body_end]), end);
    }
    let s = std::str::from_utf8(&src[start..body_end]).unwrap();
    (Token::IntLiteral(s), body_end)
}

/// If `src[i]` is `i`/`u` followed by `[0-9]+`, return the end offset.
#[inline]
fn scan_suffix(src: &[u8], i: usize) -> Option<usize> {
    if matches!(src.get(i), Some(b'i') | Some(b'u')) {
        let mut j = i + 1;
        if src.get(j).map(|&b| is_dec(b)) == Some(true) {
            while j < src.len() && is_dec(src[j]) {
                j += 1;
            }
            return Some(j);
        }
    }
    None
}

/// Given that `src[i]` is `e`/`E`, try to match `[eE][+-]?[0-9][0-9_]*` and
/// return the end offset, or `None`.
#[inline]
fn try_exponent(src: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    if matches!(src.get(j), Some(b'+') | Some(b'-')) {
        j += 1;
    }
    if src.get(j).map(|&b| is_dec(b)) == Some(true) {
        while j < src.len() && (is_dec(src[j]) || src[j] == b'_') {
            j += 1;
        }
        return Some(j);
    }
    None
}

/// Having matched an exponent sign marker at `i`, consume the rest of the
/// exponent (used by the `.`-float path where the exponent is optional).
#[inline]
fn scan_exponent(src: &[u8], i: usize) -> usize {
    if matches!(src.get(i), Some(b'e') | Some(b'E')) {
        try_exponent(src, i).unwrap_or(i)
    } else {
        i
    }
}

#[inline]
fn token_dec_suffix<'src>(s: &'src str, marker: u8) -> Token<'src> {
    if marker == b'i' {
        Token::IntSuffix(s)
    } else {
        Token::UIntSuffix(s)
    }
}

#[inline]
fn token_hex_suffix<'src>(s: &'src str, marker: u8) -> Token<'src> {
    if marker == b'i' {
        Token::HexIntSuffix(s)
    } else {
        Token::HexUIntSuffix(s)
    }
}

#[inline]
fn token_bin_suffix<'src>(s: &'src str, marker: u8) -> Token<'src> {
    if marker == b'i' {
        Token::BinIntSuffix(s)
    } else {
        Token::BinUIntSuffix(s)
    }
}

/// Keyword → `Token` classification.  `ro` and `codomain` map to their
/// reserved variants (present here, absent from `as_ident_symbol`).
/// Everything else falls through to `Ident(intern(slice))`.
fn classify_ident(s: &str) -> Token<'static> {
    use Token::*;
    match s {
        "def" => Def,
        "set" => Set,
        "type" => Type,
        "const" => Const,
        "codomain" => Codomain,
        "with" => With,
        "default" => Default,
        "return" => Return,
        "if" => If,
        "else" => Else,
        "for" => For,
        "in" => In,
        "while" => While,
        "loop" => Loop,
        "leave" => Leave,
        "continue" => Continue,
        "comptime" => Comptime,
        "generate" => Generate,
        "import" => Import,
        "as" => As,
        "true" => True,
        "false" => False,
        "auto" => Auto,
        "and" => And,
        "or" => Or,
        "not" => Not,
        "sizeof" => Sizeof,
        "alignof" => Alignof,
        "catch" => Catch,
        "panic" => Panic,
        "unsafe" => Unsafe,
        "let" => Let,
        "finally" => Finally,
        "where" => Where,
        "when" => When,
        "requires" => Requires,
        "ensures" => Ensures,
        "invariant" => Invariant,
        "constraint" => Constraint,
        "move" => Move,
        "dyn" => Dyn,
        "by" => By,
        "copy" => Copy,
        "ref" => Ref,
        "ro" => Ro,
        "mut" => Mut,
        "wrap" => Wrap,
        "saturate" => Saturate,
        "trap" => Trap,
        "ieee" => Ieee,
        "Self" => SelfKw,
        "no_default" => NoDefault,
        "extern" => Extern,
        "pub" => Pub,
        "edition" => Edition,
        "deprecated" => Deprecated,
        "experimental" => Experimental,
        "endian" => Endian,
        "bit_order" => BitOrder,
        "align" => Align,
        "pad" => Pad,
        "packed" => Packed,
        "async" => Async,
        "await" => Await,
        "task" => Task,
        "channel" => Channel,
        "linear" => Linear,
        "consume" => Consume,
        "pure" => Pure,
        "io" => Io,
        "trusted" => Trusted,
        "ghost" => Ghost,
        "scope_cleanup" => ScopeCleanup,
        "trigger" => Trigger,
        "layout" => Layout,
        "validate" => Validate,
        "missing_match" => MissingMatch,
        "apply_lemma" => ApplyLemma,
        "exists" => Exists,
        "forall" => Forall,
        "implies" => Implies,
        "trait" => Trait,
        "impl" => Impl,
        "decreases" => Decreases,
        "diverges" => Diverges,
        "terminates" => Terminates,
        "cfg" => Cfg,
        "isolate" => Isolate,
        "hint" => Hint,
        "must_use" => MustUse,
        "must_handle" => MustHandle,
        "link_proof" => LinkProof,
        "exhaustive" => Exhaustive,
        "no_alloc_error" => NoAllocError,
        "no_panic" => NoPanic,
        "debug_info" => DebugInfo,
        "ieee_contracts" => IeeeContracts,
        "old" => Old,
        "audit_log" => AuditLog,
        "interrupt" => Interrupt,
        "match" => Match,
        "propagates" => Propagates,
        "overrides" => Overrides,
        "poly" => Poly,
        "unbox" => Unbox,
        _ => Ident(Symbol::intern(s)),
    }
}
