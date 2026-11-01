//! Shared integer/space formatting primitives — no `core::fmt`, no allocation.
//!
//! `core::fmt` for integers drags in the width/precision/flags machinery and
//! an indirect call through `Formatter`; these are straight div-mod loops
//! into stack buffers plus one `push` per ASCII digit, shared by the JSON,
//! HTML, and short renderers.

/// Append a decimal `u64` without `core::fmt` and without allocating.
///
/// 20 digits covers every 64-bit value (`2^64 - 1` has 20 decimal digits).
#[inline]
pub fn push_uint(out: &mut String, mut n: u64) {
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    for &b in &buf[i..] {
        out.push(b as char);
    }
}

/// Append `n` as a zero-padded 16-digit lowercase hex string (no `0x` prefix).
///
/// Used for the stable diagnostic id (`"0x{:016x}"`); the fixed width makes
/// ids comparable as byte strings without parsing.
#[inline]
pub fn push_hex64(out: &mut String, n: u64) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 16];
    let mut n = n;
    for i in (0..16).rev() {
        buf[i] = HEX[(n & 0xf) as usize];
        n >>= 4;
    }
    for &b in &buf {
        out.push(b as char);
    }
}

/// Append `n` space characters without per-iteration overhead.
///
/// Large counts are pushed in 64-byte chunks; the common source-rendering
/// counts (line-number widths, label indentation) are a single sliced push.
#[inline]
pub fn push_spaces(out: &mut String, n: usize) {
    const SPACES: &str = "                                                                ";
    let mut remaining = n;
    while remaining >= SPACES.len() {
        out.push_str(SPACES);
        remaining -= SPACES.len();
    }
    if remaining > 0 {
        out.push_str(&SPACES[..remaining]);
    }
}

/// Append `n` right-justified to `width` columns (left-padded with spaces).
#[inline]
pub fn push_uint_padded(out: &mut String, n: u64, width: usize) {
    let digits = dec_digits(n);
    if width > digits {
        push_spaces(out, width - digits);
    }
    push_uint(out, n);
}

/// Decimal digit count of `n` (≥ 1).
#[inline]
pub const fn dec_digits(mut n: u64) -> usize {
    let mut d = 1;
    while n >= 10 {
        n /= 10;
        d += 1;
    }
    d
}
