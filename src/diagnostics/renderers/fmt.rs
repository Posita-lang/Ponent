//! Shared integer/space formatting primitives — straight div-mod loops instead
//! of `core::fmt`, and no allocation.
//!
//! "No `core::fmt`" is a property of the *algorithms* here, not of the sink
//! bounds: the writers take `std::fmt::Write`, so the guarantee holds because
//! `String::write_str`/`push_str` are `#[inline]` in practice.  It would not hold
//! for a `dyn Write`, and is unverified for `CompactString` — see the F20 note in
//! the review backlog about a `JsonSink` trait if that ever matters.
//!
//! `core::fmt` for integers drags in the width/precision/flags machinery and
//! an indirect call through `Formatter`; these are straight div-mod loops
//! into stack buffers plus one `push` per ASCII digit, shared by the JSON,
//! HTML, and short renderers.

/// Const-built table of the two ASCII digits for every `u8` value `0..100`,
/// laid out as `[d0(0), d1(0), d0(1), d1(1), …]`.  Lets `push_uint` emit two
/// digits per div-mod and one `push_str` instead of one `push` per digit.
const fn build_digit_pairs() -> [u8; 200] {
    let mut t = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        t[2 * i] = b'0' + (i / 10) as u8;
        t[2 * i + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    t
}
static DIGITS: [u8; 200] = build_digit_pairs();

/// Append a decimal `u64` without `core::fmt`'s number machinery and without
/// allocating.
///
/// Generic over the sink so the no-`core::fmt` contract actually holds at every
/// call site: `String` and `ArrayString` both implement `fmt::Write`, and
/// `String::write_str` delegates to `push_str` (std marks it `#[inline]`), so
/// monomorphization leaves existing call sites' codegen unchanged.
///
/// Two digits per iteration (halving the div-mod count for `n ≥ 100`) and a
/// single write of a stack buffer instead of one capacity-checked `push` per
/// digit.  20 digits covers every 64-bit value (`2^64 - 1` has 20).
///
/// Backed by `itoa` rather than a hand-rolled div-mod ladder: `itoa` is the
/// fastest known integer formatter (two-digit table + one reciprocal multiply
/// per pair), and — the part that matters here — it returns a `&str`, which
/// deletes the `from_utf8_unchecked` this function used to need.  The bytes are
/// pinned by `push_uint_matches_display` and `push_u32_matches_push_uint`, so a
/// formatter swap cannot change output silently.
#[inline]
pub fn push_uint<W: std::fmt::Write + ?Sized>(out: &mut W, n: u64) {
    let mut buf = itoa::Buffer::new();
    // The `write_str` result is discarded deliberately: these sinks are
    // infallible in practice (`String`, `ArrayString`), and an error would mean
    // the sink is broken, not that the number is unrepresentable.
    let _ = out.write_str(buf.format(n));
}

/// `push_uint` for the u32 values this module is almost always given (span
/// offsets, file ids, line/column numbers).
///
/// A separate entry point because callers hold `u32` and the `as u64` widening
/// used to be the only way in; `itoa` has a dedicated `u32` path, so the call
/// site and the formatter agree on the width.
#[inline]
pub fn push_u32<W: std::fmt::Write + ?Sized>(out: &mut W, n: u32) {
    let mut buf = itoa::Buffer::new();
    let _ = out.write_str(buf.format(n));
}

/// Const-built table mapping each `u8` to its two lowercase hex digits,
/// laid out as `[h0(0), h1(0), h0(1), h1(1), …]`.  Lets `push_hex64` emit a
/// byte's two hex chars per iteration and one `push_str` at the end.
const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";

const fn build_hex_pairs() -> [u8; 512] {
    let mut t = [0u8; 512];
    let mut i = 0;
    while i < 256 {
        t[2 * i] = HEX_LOWER[(i >> 4) as usize];
        t[2 * i + 1] = HEX_LOWER[(i & 0xF) as usize];
        i += 1;
    }
    t
}
static HEX_PAIRS: [u8; 512] = build_hex_pairs();

/// Append `n` as a zero-padded 16-digit lowercase hex string (no `0x` prefix).
///
/// Used for the stable diagnostic id (`"0x{:016x}"`); the fixed width makes
/// ids comparable as byte strings without parsing.  One `push_str` of a
/// 16-byte stack buffer instead of sixteen `push`es.
#[inline]
pub fn push_hex64<W: std::fmt::Write + ?Sized>(out: &mut W, mut n: u64) {
    let mut buf = [0u8; 16];
    let mut i = 16;
    while i > 0 {
        i -= 2;
        let b = (n & 0xFF) as usize;
        buf[i] = HEX_PAIRS[2 * b];
        buf[i + 1] = HEX_PAIRS[2 * b + 1];
        n >>= 8;
    }
    // SAFETY: `buf` holds only ASCII hex digit bytes from `HEX_PAIRS`
    // (const-built from `HEX_LOWER` = `b"0123456789abcdef"`), all in
    // `0x30..=0x39 | 0x61..=0x66`.
    let _ = out.write_str(unsafe { std::str::from_utf8_unchecked(&buf) });
}

/// Append `n` space characters without per-iteration overhead.
///
/// Large counts are pushed in 64-byte chunks; the common source-rendering
/// counts (line-number widths, label indentation) are a single sliced push.
#[inline]
pub fn push_spaces(out: &mut String, n: usize) {
    if n == 0 {
        return;
    }
    out.reserve(n);
    // One `memset` (libc's SIMD fill) instead of `n / 64` `push_str` calls, each
    // re-checking capacity and copying a 64-byte literal.  Indents are wide
    // enough to matter on a table-heavy render and this is the only fill.
    //
    // SAFETY: `reserve(n)` guarantees `n` spare bytes; `0x20` is ASCII, so
    // appending `n` space bytes keeps the buffer valid UTF-8; `n > 0` was
    // handled above, so the pointer is non-dangling and `add` stays in bounds.
    unsafe {
        let v = out.as_mut_vec();
        let len = v.len();
        std::ptr::write_bytes(v.as_mut_ptr().add(len), b' ', n);
        v.set_len(len + n);
    }
}

/// Compile-time check that a `&str` is all spaces.
const fn all_spaces(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b' ' {
            return false;
        }
        i += 1;
    }
    true
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

/// Append the fold marker text shared by the glyph and HTML renderers:
/// `...` when nothing was omitted, else the locale's own wrapper around the
/// pluralized count.  One source of truth so the two renderers cannot drift on
/// the wording, and the plural correctness is inherited from Fluent (including
/// Russian's `few`/`many` categories) rather than re-implemented with a binary
/// `== 1` split.
#[inline]
pub fn push_fold_marker(out: &mut String, gap: usize) {
    out.push_str("...");
    if gap > 0 {
        // The wrapper — the separating space *and* the brackets — is
        // locale-owned (English/Russian ` (…)`, CJK ` （…）`); hardcoding it
        // here handed translators half a sentence.  The count goes in as a
        // native number: no `ArrayString`, no string→number re-parse.
        crate::diagnostics::i18n::render_into_count("fold-marker-lines", "gap", gap as u64, out);
    }
}

/// Decimal digit count of `n` (≥ 1).
///
/// `u64::ilog10` is constant-time on modern rustc (LLVM emits the tables);
/// the loop it replaces is up to 20 divisions.  Still `const fn`.
#[inline]
pub const fn dec_digits(n: u64) -> usize {
    if n == 0 { 1 } else { n.ilog10() as usize + 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every off-by-one the pair/tail split could produce is caught by comparing
    /// against `to_string`, plus an exhaustive small-range sweep.
    #[test]
    fn push_uint_matches_display() {
        for n in [
            0u64,
            1,
            9,
            10,
            11,
            99,
            100,
            101,
            999,
            1000,
            u32::MAX as u64,
            1_000_000_000_000_000_000,
            u64::MAX,
        ] {
            let mut out = String::new();
            push_uint(&mut out, n);
            assert_eq!(out, n.to_string(), "push_uint({n})");
        }
        for n in 0..=10_000u64 {
            let mut out = String::new();
            push_uint(&mut out, n);
            assert_eq!(out, n.to_string(), "push_uint({n})");
        }
    }

    #[test]
    fn push_hex64_is_fixed_width_lowercase() {
        for (n, want) in [
            (0u64, "0000000000000000"),
            (1, "0000000000000001"),
            (0xdead_beef, "00000000deadbeef"),
            (u64::MAX, "ffffffffffffffff"),
        ] {
            let mut out = String::new();
            push_hex64(&mut out, n);
            assert_eq!(out, want, "push_hex64({n:#x})");
        }
        for n in [0u64, 7, 1 << 63, u64::MAX] {
            let mut out = String::new();
            push_hex64(&mut out, n);
            assert_eq!(out.len(), 16);
            assert!(
                out.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            );
            assert_eq!(out, format!("{n:016x}"));
        }
    }

    #[test]
    fn push_spaces_covers_chunk_and_tail_paths() {
        for n in [0usize, 1, 63, 64, 65, 127, 128, 1000] {
            let mut out = String::from("|");
            push_spaces(&mut out, n);
            assert_eq!(out.len(), n + 1, "n = {n}");
            assert!(out[1..].bytes().all(|b| b == b' '), "n = {n}");
        }
    }

    #[test]
    fn push_uint_padded_never_truncates() {
        for (n, width, want) in [
            (0u64, 0usize, "0"),
            (0, 3, "  0"),
            (5, 3, "  5"),
            (5, 1, "5"),
            (1234, 2, "1234"),
            (u64::MAX, 5, "18446744073709551615"),
        ] {
            let mut out = String::new();
            push_uint_padded(&mut out, n, width);
            assert_eq!(out, want, "push_uint_padded({n}, {width})");
        }
    }

    #[test]
    /// `push_u32` must be byte-identical to `push_uint` on every value the
    /// formatters actually pass it — otherwise the JSON `start`/`end`/`fileId`
    /// fields would silently diverge from the `line`/`column` ones.
    #[test]
    fn push_u32_matches_push_uint() {
        let mut x = 0x2545_F491u32;
        for extra in [0u32, 1, 9, 10, 99, 100, 999, 1_000, u32::MAX - 1, u32::MAX] {
            for _ in 0..500 {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                let mut a = String::new();
                let mut b = String::new();
                push_u32(&mut a, x);
                push_uint(&mut b, x as u64);
                assert_eq!(a, b, "push_u32({x})");
                let mut a = String::new();
                let mut b = String::new();
                push_u32(&mut a, extra);
                push_uint(&mut b, extra as u64);
                assert_eq!(a, b, "push_u32({extra})");
            }
        }
    }

    fn dec_digits_boundaries() {
        for (n, want) in [
            (0u64, 1usize),
            (9, 1),
            (10, 2),
            (99, 2),
            (100, 3),
            (u64::MAX, 20),
        ] {
            assert_eq!(dec_digits(n), want, "dec_digits({n})");
        }
        // Every `10^k` and `10^k - 1` for k = 1..=19: exactly where a
        // log-approximation table errs.  `to_string().len()` is the oracle, so
        // this also pins the table against the standard library rather than
        // against a second copy of the same arithmetic.
        let mut pow = 1u64;
        for _ in 0..20 {
            if pow > 1 {
                assert_eq!(
                    dec_digits(pow - 1),
                    pow.to_string().len() - 1,
                    "10^k-1 ({})",
                    pow - 1
                );
            }
            assert_eq!(dec_digits(pow), pow.to_string().len(), "10^k ({pow})");
            pow = pow.saturating_mul(10);
        }
        // A deterministic spread of random values (xorshift, so a failure is
        // reproducible without a seed file).
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..10_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            assert_eq!(dec_digits(x), x.to_string().len(), "dec_digits({x})");
        }
    }

    /// `gap == 0` is exactly `...`; `gap > 0` pins the ASCII wrapper around the
    /// Fluent body.  The wrapper is what a §5-style i18n change would move.
    #[test]
    fn fold_marker_shape_is_pinned() {
        let _guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::En);

        let mut out = String::new();
        push_fold_marker(&mut out, 0);
        assert_eq!(out, "...");

        let mut out = String::new();
        push_fold_marker(&mut out, 1);
        assert_eq!(out, "... (1 line omitted)");

        let mut out = String::new();
        push_fold_marker(&mut out, 5);
        assert_eq!(out, "... (5 lines omitted)");
    }
}
