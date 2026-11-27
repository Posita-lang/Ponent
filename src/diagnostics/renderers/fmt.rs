//! Shared integer/space formatting primitives — no `core::fmt`, no allocation.
//!
//! Decimal formatting is `itoa` (two-digit table + one reciprocal multiply per
//! pair, and it hands back a `&str`, so no `from_utf8_unchecked`); hex is a
//! 512-byte pair table; spaces are a slice push.  `core::fmt` for integers would
//! drag in the width/precision/flags machinery and an indirect call through
//! `Formatter` for every digit.
//!
//! The sinks are `&mut String`, deliberately: the earlier `impl fmt::Write` bound
//! advertised a fallible writer whose errors every call site discarded with
//! `let _ =`, which is the one way these helpers could have silently truncated a
//! number.  (`json.rs` narrowed its own string writers for the same reason; the
//! two now agree.)  The `no core::fmt` guarantee rests on the `String`-specific
//! fast paths of `write_str`/`push_str`, so it is unverified for any other
//! sink: a `dyn Write`-style or `CompactString`-backed sink would need its own
//! verification before this module could be reused there (a `JsonSink` trait is
//! one candidate shape; not pursued).  Shared by the JSON, HTML and short
//! renderers.

/// Append a decimal `u64` without `core::fmt`'s number machinery and without
/// allocating.
///
/// Backed by `itoa`: the fastest known integer formatter (two-digit table plus
/// one reciprocal multiply per pair), returning a `&str` so no
/// `from_utf8_unchecked` is needed.  Bytes are pinned by
/// `push_uint_matches_display`, so a formatter swap cannot change output
/// silently.
///
/// Takes `&mut String` rather than `impl fmt::Write`: every caller passes a
/// `String`, and the generic bound used to be paired with `let _ = write_str(..)`,
/// i.e. it advertised fallibility the code discarded.  A non-`String` sink that
/// ever needs this should call `.as_str()` and append itself.
#[inline]
pub fn push_uint(out: &mut String, n: u64) {
    let mut buf = itoa::Buffer::new();
    out.push_str(buf.format(n));
}

/// `push_uint` for the u32 values this module is almost always given (span
/// offsets, file ids, line/column numbers).
///
/// A separate entry point because callers hold `u32` and the `as u64` widening
/// used to be the only way in; `itoa` has a dedicated `u32` path, so the call
/// site and the formatter agree on the width.
#[inline]
pub fn push_u32(out: &mut String, n: u32) {
    let mut buf = itoa::Buffer::new();
    out.push_str(buf.format(n));
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

/// Write `n` as exactly 16 lowercase hex digits (no `0x` prefix).
///
/// Used for the stable diagnostic id (`"0x{:016x}"`); the fixed width makes ids comparable
/// as byte strings without parsing.
///
/// Writes the 16 digits **directly into `out`'s reserved tail** and extends the length with
/// `set_len` — no stack buffer, no second `push_str` copy.  This is the JSON writer's hottest
/// helper (one call per diagnostic's `"id"`), so that copy was pure overhead.
///
/// # Safety
///
/// The contract of the `unsafe` block below, stated where an auditor of it will look: `reserve(16)`
/// guarantees at least 16 spare bytes (`start + 16 <= capacity`); all 16 are written **before**
/// `set_len`, from `HEX_PAIRS`, which is const-built from `HEX_LOWER = b"0123456789abcdef"` so
/// every byte is ASCII hex and the result is valid UTF-8; `set_len` extends by exactly the bytes
/// written; and `&mut String` is exclusive, so no reader can observe the buffer mid-write.
///
/// (This doc used to describe a 16-byte stack buffer plus `push_str`, and located the safety
/// envelope *in that stack buffer*.  Both were one revision behind the code: the block touches
/// `out.as_mut_vec()`, not a stack buffer, so a reader auditing from the doc verified the wrong
/// invariant — the exact defect class this project treats a comment as.)
#[inline]
pub fn push_hex64(out: &mut String, mut n: u64) {
    out.reserve(16);
    // Writing straight into the reserved tail instead of building a 16-byte stack buffer
    // and copying it in: this is the JSON writer's hottest helper (one call per
    // diagnostic's `"id"`), and the copy plus the stack buffer construction are pure
    // overhead.  Until now the code carried a *second* reason to avoid it — that it would
    // need another `unsafe` block — but a second block with a narrow, locally checkable
    // contract is not the "unsafe for convenience" the project forbids, and the yield is
    // on the per-diagnostic path.
    //
    // SAFETY: `reserve(16)` guarantees 16 spare bytes.  All 16 are written before
    // `set_len`, from `HEX_PAIRS` — const-built from `HEX_LOWER = b"0123456789abcdef"`, so
    // every byte is ASCII hex (`0x30..=0x39 | 0x61..=0x66`) and the result is valid UTF-8.
    // `set_len` only ever extends by exactly the bytes written, and no reader can observe
    // the buffer during the block (`&mut String` is exclusive).
    unsafe {
        let v = out.as_mut_vec();
        let start = v.len();
        // `reserve` guarantees `start + 16 <= capacity`.
        let dst = v.as_mut_ptr().add(start);
        let mut i = 16;
        while i > 0 {
            i -= 2;
            let b = (n & 0xFF) as usize;
            std::ptr::write(dst.add(i), HEX_PAIRS[2 * b]);
            std::ptr::write(dst.add(i + 1), HEX_PAIRS[2 * b + 1]);
            n >>= 8;
        }
        v.set_len(start + 16);
    }
}

/// Append `n` space characters with one fill.
///
/// One `reserve` + one `write_bytes` (libc's SIMD memset), not the chunked
/// `push_str` loop the previous doc described and the code no longer contained:
/// every `push_str` of a 64-byte literal re-checks capacity and copies, so wide
/// indents on a table-heavy render paid `n / 64` capacity checks.
#[inline]
pub fn push_spaces(out: &mut String, n: usize) {
    if n == 0 {
        return;
    }
    // `String::reserve` **aborts** on capacity overflow (`capacity_overflow`), it does not panic,
    // so an absurd `n` from an unclamped `--width` would take the process down rather than produce
    // a bad layout.  `sink.rs` clamps every capacity entry point against `MAX_SINK_CAPACITY` for
    // this reason; the trust contract here is that `n` comes from clamped layout options
    // (`DiagOpts::max_width` is `caps.width.max(1)` on the detected path), so the bound is a
    // tripwire rather than a clamp — a caller that hands over a raw user width must clamp first.
    debug_assert!(
        n <= 1 << 20,
        "push_spaces called with {n} spaces — a width was not clamped by its producer"
    );
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
/// Branch-light estimate plus one correction, instead of `u64::ilog10`.  The previous
/// doc claimed "`ilog10` is constant-time on modern rustc (LLVM emits the tables)";
/// std's `ilog10` is a *division loop* — strength-reduced, but variable-trip — so the
/// claim was doing load-bearing work it does not support.  `(bits · 1233) >> 12` is the
/// standard log10-from-bits bound (within one), and the single comparison against the
/// power-of-ten table settles the off-by-one.  `dec_digits_boundaries` pins this
/// exhaustively against `to_string().len()`.
#[inline]
pub const fn dec_digits(n: u64) -> usize {
    const POW10: [u64; 20] = [
        1,
        10,
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
        10_000_000_000,
        100_000_000_000,
        1_000_000_000_000,
        10_000_000_000_000,
        100_000_000_000_000,
        1_000_000_000_000_000,
        10_000_000_000_000_000,
        100_000_000_000_000_000,
        1_000_000_000_000_000_000,
        10_000_000_000_000_000_000,
    ];
    if n == 0 {
        return 1;
    }
    let bits = 64 - n.leading_zeros() as usize;
    // `<= 19` for every `u64`, so `POW10[t]` and `POW10[t + 1]` stay in range.
    let t = ((bits * 1233) >> 12) as usize;
    let t = if n < POW10[t] { t - 1 } else { t };
    t + 1
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
    /// `push_hex64` against the formatter over a randomized corpus.
    fn push_hex64_matches_the_formatter_on_xorshift_corpus() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..2048 {
            // xorshift64: exercises every bit position without a rng dependency.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut out = String::new();
            push_hex64(&mut out, state);
            assert_eq!(out, format!("{state:016x}"));
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
    fn push_spaces_fills_zero_and_wide_counts() {
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

    /// `push_u32` must be byte-identical to `push_uint` on every value the
    /// formatters actually pass it — otherwise the JSON `start`/`end`/`fileId`
    /// fields would silently diverge from the `line`/`column` ones.
    #[test]
    fn push_u32_matches_push_uint() {
        let mut x = 0x2545_F491u32;
        for extra in [0u32, 1, 9, 10, 99, 100, 999, 1_000, u32::MAX - 1, u32::MAX] {
            // The boundary values are checked once, not 500 times: the random sweep
            // exists for `x`.
            let mut a = String::new();
            let mut b = String::new();
            push_u32(&mut a, extra);
            push_uint(&mut b, extra as u64);
            assert_eq!(a, b, "push_u32({extra})");
            for _ in 0..500 {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                let mut a = String::new();
                let mut b = String::new();
                push_u32(&mut a, x);
                push_uint(&mut b, x as u64);
                assert_eq!(a, b, "push_u32({x})");
            }
        }
    }

    #[test]
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
