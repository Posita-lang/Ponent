//! Branch-light ASCII predicates shared by the renderers.
//!
//! Every predicate here answers "can this byte slice be emitted without any
//! per-byte work elsewhere?" — i.e. one display cell per byte, no escape
//! parsing, no grapheme segmentation. The renderers use these as fast-path
//! gates; a `true` answer lets them skip the Unicode tables entirely.

/// `true` when every byte is printable ASCII (`0x20..=0x7E`).
///
/// SWAR over `u128` chunks: two masks per sixteen bytes instead of two
/// compares per byte.  The lane proof is width-independent — the exact-mask
/// forms have no inter-lane borrow, and the `hasless` form's top-lane
/// borrow-out implies that lane already flagged — so the scalar tail is only
/// the final ≤15 bytes.
///
/// # Chunk independence
///
/// `wrapping_sub` discards the borrow out of each chunk's top lane, so lane 0
/// of every chunk enters with `borrow_in = 0`.  This is why the mask-only form
/// is sound across chunk boundaries: a top-lane borrow-out implies the top
/// lane flagged, implies `return false` already happened, so a leak can never
/// produce a false *reject* of the next chunk.
#[inline]
pub(crate) fn is_printable_ascii(s: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if s.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was just checked at runtime; the kernel only
        // reads full chunks and defers its tail back to this function.
        return unsafe { x86_64::is_printable_ascii(s) };
    }
    const HI: u128 = 0x8080_8080_8080_8080_8080_8080_8080_8080;
    const SP: u128 = 0x2020_2020_2020_2020_2020_2020_2020_2020;
    const DEL: u128 = 0x7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F;
    const ONES: u128 = 0x0101_0101_0101_0101_0101_0101_0101_0101;

    let mut i = 0usize;
    while i + 16 <= s.len() {
        let x = u128::from_le_bytes(s[i..i + 16].try_into().unwrap());
        // A lane's high bit survives `x | (x - 0x20…)` iff the byte is
        // `< 0x20` (the subtraction borrows) or `>= 0x80` (its own bit).
        // The borrow may over-flag a *following* printable byte, but the
        // predicate is existential: the triggering non-printable byte always
        // flags itself, so the overall answer is unchanged. Do not "fix" the
        // mask into a per-byte test.
        if (x | x.wrapping_sub(SP)) & HI != 0 {
            return false;
        }
        // 0x7F is the one byte inside [0x20, 0x80) that is not printable.
        let v = x ^ DEL;
        if v.wrapping_sub(ONES) & !v & HI != 0 {
            return false;
        }
        i += 16;
    }
    s[i..].iter().all(|&b| matches!(b, 0x20..=0x7E))
}

/// `true` when every byte is printable ASCII **or** the two layout controls
/// `\t` (0x09) / `\n` (0x0A). This is the predicate
/// `sanitize_for_terminal_multiline` needs: it preserves tab and newline as
/// layout intent, replaces every other control (CR, C1, DEL) with U+FFFD.
///
/// SWAR with **exact** control masks: setting every byte's high bit before the
/// subtraction (`x | HI`) makes each minuend byte ≥ 0x80, so no inter-lane
/// borrow can occur and the word subtraction is exactly per-lane.  There are
/// therefore no borrow false positives to resolve, and no per-lane loop:
///
/// * `lt20 = !((x|HI) - SP) & HI` — high bit clear iff `b + 0x60 < 0x80` iff
///   `b < 0x20`, exactly.
/// * `tab  = !(((x ^ TAB)|HI) - ONES) & HI` — high bit clear iff `b ^ 0x09 ==
///   0` iff `b == 0x09`, exactly (the 0x08-neighbour trap of a `has_zero`
///   value mask cannot occur here because the minuend never borrows).
/// * `nl` likewise with 0x0A.
///
/// Precondition: every byte is ≤ 0x7F, guaranteed by the high-byte early
/// return below (without it, `b + 0x80 - 0x20` would overflow for large
/// bytes).
#[inline]
pub(crate) fn is_printable_ascii_multiline(s: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if s.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime-checked AVX2; tail is deferred to this function.
        return unsafe { x86_64::is_printable_ascii_multiline(s) };
    }
    const HI: u128 = 0x8080_8080_8080_8080_8080_8080_8080_8080;
    const SP: u128 = 0x2020_2020_2020_2020_2020_2020_2020_2020;
    const TAB: u128 = 0x0909_0909_0909_0909_0909_0909_0909_0909;
    const NL: u128 = 0x0A0A_0A0A_0A0A_0A0A_0A0A_0A0A_0A0A_0A0A;
    const DEL: u128 = 0x7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F;
    const ONES: u128 = 0x0101_0101_0101_0101_0101_0101_0101_0101;

    let mut i = 0usize;
    while i + 16 <= s.len() {
        let x = u128::from_le_bytes(s[i..i + 16].try_into().unwrap());
        // Any byte >= 0x80: not printable, and no per-byte exception exists.
        if x & HI != 0 {
            return false;
        }
        // DEL: `has_zero(x ^ 0x7F)` — any flag implies a genuine 0x7F lane.
        let q = x ^ DEL;
        if q.wrapping_sub(ONES) & !q & HI != 0 {
            return false;
        }
        // Exact `< 0x20` lanes (see the derivation above).
        let lt20 = !((x | HI).wrapping_sub(SP)) & HI;
        if lt20 != 0 {
            let tab = !(((x ^ TAB) | HI).wrapping_sub(ONES)) & HI;
            let nl = !(((x ^ NL) | HI).wrapping_sub(ONES)) & HI;
            // A `< 0x20` lane that is neither tab nor newline is an offender.
            if (lt20 & !tab & !nl) != 0 {
                return false;
            }
        }
        i += 16;
    }
    s[i..]
        .iter()
        .all(|&b| b == 0x09 || b == 0x0A || matches!(b, 0x20..=0x7E))
}

/// `true` when every byte is printable ASCII **or** the tab (`0x09`).
///
/// The gate for `render_line_merged_with`'s tab-aware byte-level fast path:
/// source code is routinely tab-indented, and the pure-printable gate
/// rejected tab (`0x09 < 0x20`), pushing every tab-indented row into the
/// grapheme walk.  SWAR shape identical to
/// [`is_printable_ascii_multiline`]'s exact-mask form: `tab` is an exact lane
/// mask, so there is no borrow resolution loop to run.
#[inline]
pub(crate) fn is_printable_ascii_or_tab(s: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if s.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime-checked AVX2; tail is deferred to this function.
        return unsafe { x86_64::is_printable_ascii_or_tab(s) };
    }
    const HI: u128 = 0x8080_8080_8080_8080_8080_8080_8080_8080;
    const SP: u128 = 0x2020_2020_2020_2020_2020_2020_2020_2020;
    const TAB: u128 = 0x0909_0909_0909_0909_0909_0909_0909_0909;
    const DEL: u128 = 0x7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F;
    const ONES: u128 = 0x0101_0101_0101_0101_0101_0101_0101_0101;

    let mut i = 0usize;
    while i + 16 <= s.len() {
        let x = u128::from_le_bytes(s[i..i + 16].try_into().unwrap());
        // A high byte is never printable and has no per-byte exception, so
        // reject before paying for the other masks.
        if x & HI != 0 {
            return false;
        }
        // DEL: `has_zero(x ^ 0x7F)` — any flag implies a genuine 0x7F lane.
        let v = x ^ DEL;
        if v.wrapping_sub(ONES) & !v & HI != 0 {
            return false;
        }
        // Exact `< 0x20` lanes (see `is_printable_ascii_multiline`).
        let lt20 = !((x | HI).wrapping_sub(SP)) & HI;
        if lt20 != 0 {
            let tab = !(((x ^ TAB) | HI).wrapping_sub(ONES)) & HI;
            if (lt20 & !tab) != 0 {
                return false;
            }
        }
        i += 16;
    }
    s[i..].iter().all(|&b| b == b'\t' || matches!(b, 0x20..=0x7E))
}

/// Tri-state classification of one row, so callers that need both the
/// printable and printable-or-tab predicates (the glyph fast path tested them
/// sequentially) pay one SWAR pass instead of two.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AsciiClass {
    /// Every byte is printable ASCII (`0x20..=0x7E`).
    Printable,
    /// Printable ASCII plus horizontal tabs; no other control bytes.
    PrintableOrTab,
    /// Anything else (high byte, DEL, other C0/C1).
    Other,
}

/// Classify `s` in one `u64`-chunk pass with the exact control masks.
///
/// Exactness relies on rejecting high bytes first (as
/// [`is_printable_ascii_multiline`] does): the `(x | HI) - SP` form is only
/// exact when every lane is ≤ `0x7F`.
#[inline]
pub(crate) fn classify_ascii(s: &[u8]) -> AsciiClass {
    #[cfg(target_arch = "x86_64")]
    if s.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime-checked AVX2; tail is deferred to this function.
        return unsafe { x86_64::classify_ascii(s) };
    }
    const HI: u128 = 0x8080_8080_8080_8080_8080_8080_8080_8080;
    const SP: u128 = 0x2020_2020_2020_2020_2020_2020_2020_2020;
    const TAB: u128 = 0x0909_0909_0909_0909_0909_0909_0909_0909;
    const DEL: u128 = 0x7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F;
    const ONES: u128 = 0x0101_0101_0101_0101_0101_0101_0101_0101;

    let mut saw_tab = false;
    let mut i = 0usize;
    while i + 16 <= s.len() {
        let x = u128::from_le_bytes(s[i..i + 16].try_into().unwrap());
        if x & HI != 0 {
            return AsciiClass::Other;
        }
        let v = x ^ DEL;
        if v.wrapping_sub(ONES) & !v & HI != 0 {
            return AsciiClass::Other;
        }
        let lt20 = !((x | HI).wrapping_sub(SP)) & HI;
        if lt20 != 0 {
            let tab = !(((x ^ TAB) | HI).wrapping_sub(ONES)) & HI;
            if (lt20 & !tab) != 0 {
                return AsciiClass::Other;
            }
            saw_tab = true;
        }
        i += 16;
    }
    for &b in &s[i..] {
        if b == b'\t' {
            saw_tab = true;
        } else if !matches!(b, 0x20..=0x7E) {
            return AsciiClass::Other;
        }
    }
    if saw_tab {
        AsciiClass::PrintableOrTab
    } else {
        AsciiClass::Printable
    }
}

/// Byte offset of the first byte ≥ 0x80 in `s`, or `None` if `s` is pure ASCII.
///
/// SWAR over `u128` chunks: every non-ASCII UTF-8 byte (lead or continuation)
/// has its high bit set, so one AND per 16-byte chunk decides whether the
/// chunk holds a candidate; a `trailing_zeros` on the flagged lanes locates it.
/// Used by `char_col`'s sparse-table build to jump the ASCII runs of a mixed
/// line without decoding every character — `O(chunk)` per non-ASCII char
/// instead of `O(line_len)` char decodes.
///
/// If `s` begins mid-UTF-8-sequence, the returned offset may be a continuation
/// byte; callers that resume mid-line must handle that (the lead-byte
/// assumption only holds for a scan started at a char boundary).
#[inline]
pub(crate) fn next_non_ascii_byte(s: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    if s.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime-checked AVX2; tail is deferred to this function.
        return unsafe { x86_64::next_non_ascii_byte(s) };
    }
    const HI: u128 = 0x8080_8080_8080_8080_8080_8080_8080_8080;

    let mut i = 0usize;
    while i + 16 <= s.len() {
        let x = u128::from_le_bytes(s[i..i + 16].try_into().unwrap());
        let hi = x & HI;
        if hi != 0 {
            // First flagged lane (little-endian ⇒ lowest byte index).
            return Some(i + (hi.trailing_zeros() / 8) as usize);
        }
        i += 16;
    }
    s[i..].iter().position(|&b| b >= 0x80).map(|off| i + off)
}

/// Runtime-dispatched AVX2 kernels.
///
/// The public predicates below keep their branch-light SWAR body as the
/// fallback; when the CPU reports AVX2 they hand full 32-byte chunks to these
/// kernels (one `movemask` decides 32 lanes at once instead of two `u128`
/// masks).  SSE2 is baseline on `x86_64`, but AVX2 is not, hence the runtime
/// check at every entry.  Non-x86 targets stay on SWAR, which LLVM already
/// auto-vectorizes on aarch64.
///
/// Every kernel processes only full chunks and defers its `< 32`-byte tail to
/// the public SWAR entry, so the chunk/lane proofs of the SWAR code remain
/// the single source of truth for the tail.  The tail deliberately avoids a
/// masked AVX2 load: the fallback is one cached feature branch plus at most
/// 31 bytes of SWAR, while masked tails add unsafe edge cases for no measured
/// gain on diagnostic-sized inputs.
#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn is_printable_ascii(s: &[u8]) -> bool {
        let sp = _mm256_set1_epi8(0x20);
        let del = _mm256_set1_epi8(0x7F);
        let mut i = 0usize;
        while i + 32 <= s.len() {
            // SAFETY: `i + 32 <= len`, and the caller checked AVX2 support.
            let v = unsafe { _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i) };
            let lt = unsafe { _mm256_cmpgt_epi8(sp, v) };
            let eq = unsafe { _mm256_cmpeq_epi8(v, del) };
            // High bytes (0x80..=0xFF) are negative as i8, so `sp > v` is true
            // for them; `(lt | eq)` rejects high bytes, <0x20, and DEL in one
            // movemask.  No separate high-bit test is needed.
            if unsafe { _mm256_movemask_epi8(_mm256_or_si256(lt, eq)) } != 0 {
                return false;
            }
            i += 32;
        }
        super::is_printable_ascii(&s[i..])
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn is_printable_ascii_multiline(s: &[u8]) -> bool {
        let sp = _mm256_set1_epi8(0x20);
        let del = _mm256_set1_epi8(0x7F);
        let tab_v = _mm256_set1_epi8(0x09);
        let nl_v = _mm256_set1_epi8(0x0A);
        let mut i = 0usize;
        while i + 32 <= s.len() {
            // SAFETY: `i + 32 <= len`, and the caller checked AVX2 support.
            let v = unsafe { _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i) };
            let eq = unsafe { _mm256_cmpeq_epi8(v, del) };
            if unsafe { _mm256_movemask_epi8(eq) } != 0 {
                return false;
            }
            // High bytes are negative i8, so `sp > v` flags them; tab/newline
            // exclusions cannot rescue a high byte.
            let lt = unsafe { _mm256_cmpgt_epi8(sp, v) };
            if unsafe { _mm256_movemask_epi8(lt) } != 0 {
                let tab = unsafe { _mm256_cmpeq_epi8(v, tab_v) };
                let nl = unsafe { _mm256_cmpeq_epi8(v, nl_v) };
                let off = unsafe {
                    _mm256_andnot_si256(nl, _mm256_andnot_si256(tab, lt))
                };
                if unsafe { _mm256_movemask_epi8(off) } != 0 {
                    return false;
                }
            }
            i += 32;
        }
        super::is_printable_ascii_multiline(&s[i..])
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn is_printable_ascii_or_tab(s: &[u8]) -> bool {
        let sp = _mm256_set1_epi8(0x20);
        let del = _mm256_set1_epi8(0x7F);
        let tab_v = _mm256_set1_epi8(0x09);
        let mut i = 0usize;
        while i + 32 <= s.len() {
            // SAFETY: `i + 32 <= len`, and the caller checked AVX2 support.
            let v = unsafe { _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i) };
            let eq = unsafe { _mm256_cmpeq_epi8(v, del) };
            if unsafe { _mm256_movemask_epi8(eq) } != 0 {
                return false;
            }
            // High bytes are negative i8 and therefore flagged by `lt`.
            let lt = unsafe { _mm256_cmpgt_epi8(sp, v) };
            if unsafe { _mm256_movemask_epi8(lt) } != 0 {
                let tab = unsafe { _mm256_cmpeq_epi8(v, tab_v) };
                let off = unsafe { _mm256_andnot_si256(tab, lt) };
                if unsafe { _mm256_movemask_epi8(off) } != 0 {
                    return false;
                }
            }
            i += 32;
        }
        super::is_printable_ascii_or_tab(&s[i..])
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn classify_ascii(s: &[u8]) -> super::AsciiClass {
        let sp = _mm256_set1_epi8(0x20);
        let del = _mm256_set1_epi8(0x7F);
        let tab_v = _mm256_set1_epi8(0x09);
        let mut saw_tab = false;
        let mut i = 0usize;
        while i + 32 <= s.len() {
            // SAFETY: `i + 32 <= len`, and the caller checked AVX2 support.
            let v = unsafe { _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i) };
            let eq = unsafe { _mm256_cmpeq_epi8(v, del) };
            if unsafe { _mm256_movemask_epi8(eq) } != 0 {
                return super::AsciiClass::Other;
            }
            // High bytes are negative i8, so `lt` flags them; tab is the only
            // excluded control and can never be a high byte.
            let lt = unsafe { _mm256_cmpgt_epi8(sp, v) };
            if unsafe { _mm256_movemask_epi8(lt) } != 0 {
                let tab = unsafe { _mm256_cmpeq_epi8(v, tab_v) };
                let off = unsafe { _mm256_andnot_si256(tab, lt) };
                if unsafe { _mm256_movemask_epi8(off) } != 0 {
                    return super::AsciiClass::Other;
                }
                saw_tab = true;
            }
            i += 32;
        }
        match super::classify_ascii(&s[i..]) {
            super::AsciiClass::Other => super::AsciiClass::Other,
            super::AsciiClass::PrintableOrTab => super::AsciiClass::PrintableOrTab,
            super::AsciiClass::Printable if saw_tab => super::AsciiClass::PrintableOrTab,
            super::AsciiClass::Printable => super::AsciiClass::Printable,
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn next_non_ascii_byte(s: &[u8]) -> Option<usize> {
        let mut i = 0usize;
        while i + 32 <= s.len() {
            // SAFETY: `i + 32 <= len`, and the caller checked AVX2 support.
            let v = unsafe { _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i) };
            let mask = unsafe { _mm256_movemask_epi8(v) };
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 32;
        }
        super::next_non_ascii_byte(&s[i..]).map(|off| i + off)
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    #[test]
    fn next_non_ascii_matches_scalar_reference() {
        let reference = |s: &[u8]| s.iter().position(|&b| b >= 0x80);
        // Empty, short tails, and lengths straddling the 8-byte chunk edge.
        for s in [
            &b""[..],
            b"a",
            b"1234567",
            b"12345678",
            b"123456789",
            b"ab\xC3\xA9cd",
            b"\xF0\x9F\x98\x80", // emoji lead first
            b"abc\xE4\xB8\xADdef\xE4\xB8\xAD",
        ] {
            assert_eq!(next_non_ascii_byte(s), reference(s), "input {s:?}");
        }
        // Every lane position of a high byte inside an 8-byte chunk.
        for lane in 0..8 {
            let mut v = [b'a'; 16];
            v[lane] = 0x80;
            assert_eq!(next_non_ascii_byte(&v), Some(lane));
        }
        // Every lane of a 16-byte (u128) chunk: the widened kernel must not
        // stop at the old 8-byte boundary.
        for lane in 0..16 {
            let mut v = [b'a'; 32];
            v[lane] = 0x80;
            assert_eq!(next_non_ascii_byte(&v), Some(lane), "lane {lane}");
        }
    }

    fn ref_single(s: &[u8]) -> bool {
        s.iter().all(|&b| (0x20..=0x7E).contains(&b))
    }
    fn ref_multi(s: &[u8]) -> bool {
        s.iter()
            .all(|&b| b == 0x09 || b == 0x0A || (0x20..=0x7E).contains(&b))
    }
    fn ref_tab(s: &[u8]) -> bool {
        s.iter().all(|&b| b == 0x09 || (0x20..=0x7E).contains(&b))
    }

    /// Deterministic LCG: covers every byte value and every length 0..=40,
    /// so both the SWAR body and the ≤7-byte tail are exercised.
    #[test]
    fn swar_matches_scalar_on_random_slices() {
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };
        for _ in 0..10_000 {
            let len = (next() % 41) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| (next() & 0xFF) as u8).collect();
            assert_eq!(is_printable_ascii(&bytes), ref_single(&bytes), "{bytes:?}");
            assert_eq!(
                is_printable_ascii_multiline(&bytes),
                ref_multi(&bytes),
                "{bytes:?}"
            );
            assert_eq!(
                is_printable_ascii_or_tab(&bytes),
                ref_tab(&bytes),
                "{bytes:?}"
            );
        }
    }

    /// The multiline predicate is existentially "no control other than
    /// `\t`/`\n`".  Random slices skewed towards `\t`/`\n` (the shape of real
    /// multiline text, where *every* 8-byte chunk trips the `< 0x20` SWAR
    /// mask) must agree with the scalar reference — this pins the
    /// flagged-lane fallback against both borrow-propagation false positives
    /// and tab/newline-dense chunks.
    #[test]
    fn multiline_matches_exists_reference_on_tab_newline_heavy_slices() {
        let reference = |s: &[u8]| {
            s.iter()
                .any(|&b| (b < 0x20 && b != b'\t' && b != b'\n') || b == 0x7F || b >= 0x80)
        };
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(3);
            seed
        };
        // Aligned and misaligned tab/newline-heavy inputs, plus tails.
        for s in [
            &b""[..], b"\t", b"\n", b"\t\n\t\n\t\n", b"\n\n\n\n\n\n\n\n", b"\t\n\r\t\n\r",
            b"a\tb\nc\rd", b"\x01\t\n\x0B", b"\t\t\t\t\t\t\t\t",
        ] {
            assert_eq!(is_printable_ascii_multiline(s), !reference(s), "{s:?}");
        }
        for _ in 0..10_000 {
            let len = (next() % 48) as usize;
            // ~2/3 of bytes drawn from `\t`/`\n` so most chunks trip the mask.
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    let r = next() & 0xFF;
                    if r < 170 {
                        [b'\t', b'\n'][(next() & 1) as usize]
                    } else {
                        r as u8
                    }
                })
                .collect();
            assert_eq!(
                is_printable_ascii_multiline(&bytes),
                !reference(&bytes),
                "{bytes:?}"
            );
        }
    }

    /// Every byte value at every position within an 8-byte window: pins the
    /// SWAR's borrow-propagation corner (a non-printable byte's borrow must
    /// not corrupt a neighbouring printable lane's result).
    #[test]
    fn swar_covers_every_byte_value_in_every_lane() {
        for b in 0u8..=0xFF {
            for lane in 0..8 {
                let mut v = [b'a'; 8];
                v[lane] = b;
                assert_eq!(
                    is_printable_ascii(&v),
                    ref_single(&v),
                    "b={b:#04x} lane={lane}"
                );
                assert_eq!(
                    is_printable_ascii_multiline(&v),
                    ref_multi(&v),
                    "b={b:#04x} lane={lane}"
                );
                assert_eq!(
                    is_printable_ascii_or_tab(&v),
                    ref_tab(&v),
                    "b={b:#04x} lane={lane}"
                );
            }
        }
    }

    /// The implication lattice the callers rely on: printable implies
    /// printable-or-tab implies printable-multiline.  A future edit that
    /// loosens one predicate inconsistently with its siblings is caught here.
    #[test]
    fn predicates_form_an_implication_lattice() {
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(7);
            seed
        };
        for _ in 0..10_000 {
            let len = (next() % 24) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| (next() & 0xFF) as u8).collect();
            let single = is_printable_ascii(&bytes);
            let tab = is_printable_ascii_or_tab(&bytes);
            let multi = is_printable_ascii_multiline(&bytes);
            assert!(!single || tab, "{bytes:?}");
            assert!(!tab || multi, "{bytes:?}");
        }
    }

    /// Two-control adjacency: every neighbouring pair of bytes that the exact
    /// masks must classify, in every lane.  `\t` then `\x08` is the
    /// backspace-adjacent-to-tab shape that breaks a naive `has_zero` value
    /// mask; `\x01` then space is the borrow-false-positive shape.
    #[test]
    fn exact_masks_classify_two_control_adjacency() {
        for lane in 0..7 {
            let mut v = [b'a'; 16];
            v[lane] = 0x09;
            v[lane + 1] = 0x08;
            assert_eq!(is_printable_ascii_or_tab(&v), ref_tab(&v), "{v:?}");
            assert_eq!(is_printable_ascii_multiline(&v), ref_multi(&v), "{v:?}");

            let mut w = [b'a'; 16];
            w[lane] = 0x01;
            w[lane + 1] = 0x20;
            assert_eq!(is_printable_ascii_or_tab(&w), ref_tab(&w), "{w:?}");
            assert_eq!(is_printable_ascii_multiline(&w), ref_multi(&w), "{w:?}");
        }
    }

    /// `next_non_ascii_byte` scanned from a non-char-boundary returns the
    /// continuation byte offset (its documented contract).
    #[test]
    fn next_non_ascii_from_mid_sequence_returns_continuation_offset() {
        let s = "\u{1F600}".as_bytes(); // 4-byte emoji: F0 9F 98 80
        assert_eq!(next_non_ascii_byte(&s[..]), Some(0));
        // Starting at the first continuation byte still finds a high byte
        // immediately (offset 0 of the slice), not a decoded char.
        assert_eq!(next_non_ascii_byte(&s[1..]), Some(0));
        assert_eq!(next_non_ascii_byte(&s[2..]), Some(0));
    }

    #[test]
    fn multiline_allows_tab_and_newline_only_among_controls() {
        assert!(is_printable_ascii_multiline(b"a\tb\nc"));
        assert!(!is_printable_ascii_multiline(b"a\rb")); // CR rejected
        assert!(!is_printable_ascii_multiline(b"a\x0Bb")); // VT
        assert!(!is_printable_ascii_multiline(b"a\x7Fb")); // DEL
        assert!(!is_printable_ascii_multiline("a中b".as_bytes())); // non-ASCII
    }

    /// The `\t`-then-space borrow false positive: a genuine control at lane
    /// `k` flags lane `k+1` when it is `0x20`, and the flagged-lane loop must
    /// not reject it. A whole 8-byte chunk is required to reach the SWAR body
    /// (`b"\t "` alone is a ≤7-byte tail and would hide the bug).
    #[test]
    fn tab_predicate_tolerates_borrow_false_positive() {
        assert!(is_printable_ascii_or_tab(b"\t abcdef"));
        assert!(is_printable_ascii_or_tab(b"\t\t\t  \t  x"));
        assert!(is_printable_ascii_or_tab(b"a\t b\t c\t "));
        assert!(is_printable_ascii_or_tab(b"\t "));
        // A genuine non-tab control followed by a space must still reject.
        assert!(!is_printable_ascii_or_tab(b"\t\x07 abcdef"));
        assert!(!is_printable_ascii_or_tab(b"\x01 abcdef"));
        assert!(!is_printable_ascii_or_tab(b"\t abcde\x7F"));
    }

    /// : the OR-form's borrow can carry out of lane 7 of one chunk into
    /// lane 0 of the next. Pin that an offender at lane 7 of an 8-byte chunk
    /// inside a longer buffer is still rejected for every possible next byte.
    #[test]
    fn or_form_lane7_borrow_out_covers_every_next_byte() {
        for b in 0u8..=255 {
            let mut buf = [b'a'; 16];
            // Offender in lane 7 of the first chunk.
            buf[7] = 0x07;
            buf[8] = b;
            assert!(
                !is_printable_ascii(&buf),
                "lane-7 offender not rejected with next byte {b:#04x}"
            );
            // And the same offender at lane 0 of the second chunk.
            let mut buf2 = [b'a'; 16];
            buf2[8] = 0x07;
            buf2[9] = b;
            assert!(!is_printable_ascii(&buf2));
        }
    }
}
