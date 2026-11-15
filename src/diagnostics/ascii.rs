//! Branch-light ASCII predicates shared by the renderers.
//!
//! Every predicate here answers "can this byte slice be emitted without any
//! per-byte work elsewhere?" — i.e. one display cell per byte, no escape
//! parsing, no grapheme segmentation. The renderers use these as fast-path
//! gates; a `true` answer lets them skip the Unicode tables entirely.

/// The control bytes a fast-path predicate admits.
///
/// The three public predicates differ only by this set; encoding it as a type
/// keeps their scalar tails and their documentation in lockstep.
#[derive(Clone, Copy)]
enum AllowedControls {
    None,
    Tab,
    TabAndNewline,
}

impl AllowedControls {
    #[inline]
    const fn allows(self, b: u8) -> bool {
        match self {
            Self::None => matches!(b, 0x20..=0x7E),
            Self::Tab => b == b'\t' || matches!(b, 0x20..=0x7E),
            Self::TabAndNewline => b == b'\t' || b == b'\n' || matches!(b, 0x20..=0x7E),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_level() -> u8 {
    static LEVEL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    let cached = LEVEL.load(std::sync::atomic::Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let level = if std::arch::is_x86_feature_detected!("avx512bw") {
        3
    } else if std::arch::is_x86_feature_detected!("avx2") {
        2
    } else {
        1
    };
    LEVEL.store(level, std::sync::atomic::Ordering::Relaxed);
    level
}

#[cfg(test)]
thread_local! {
    static SWAR_FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[inline]
fn swar_forced() -> bool {
    #[cfg(test)]
    {
        SWAR_FORCED.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// `true` when every byte is printable ASCII (`0x20..=0x7E`).
///
/// The widest available SIMD path handles full chunks; the `u128` SWAR body
/// below is the fallback for short tails.  The SWAR lane proof is
/// width-independent — the exact-mask forms have no inter-lane borrow, and the
/// `hasless` form's top-lane borrow-out implies that lane already flagged — so
/// the scalar tail is only the final bytes below the selected vector width.
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
    if !swar_forced() && s.len() >= 64 && x86_level() >= 3 {
        return unsafe { x86_64::is_printable_ascii_avx512(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 32 && x86_level() >= 2 {
        // SAFETY: AVX2 support was just checked at runtime; the kernel only
        // reads full chunks and defers its tail back to this function.
        return unsafe { x86_64::is_printable_ascii(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 16 {
        // SSE2 is baseline on x86_64; no runtime probe is needed.
        return unsafe { x86_64::is_printable_ascii_sse2(s) };
    }
    #[cfg(target_arch = "aarch64")]
    if !swar_forced() && s.len() >= 16 {
        return unsafe { neon::is_printable_ascii(s) };
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
        let v = x ^ DEL;
        if ((x | x.wrapping_sub(SP)) | (v.wrapping_sub(ONES))) & HI != 0 {
            return false;
        }
        i += 16;
    }
    const HI64: u64 = 0x8080_8080_8080_8080;
    const SP64: u64 = 0x2020_2020_2020_2020;
    const DEL64: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES64: u64 = 0x0101_0101_0101_0101;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        let v = x ^ DEL64;
        if ((x | x.wrapping_sub(SP64)) | (v.wrapping_sub(ONES64))) & HI64 != 0 {
            return false;
        }
        i += 8;
    }
    s[i..].iter().all(|&b| AllowedControls::None.allows(b))
}

/// `true` when every byte is printable ASCII **or** the two layout controls
/// `\t` (0x09) / `\n` (0x0A). This is the predicate
/// `sanitize_for_terminal_multiline` needs: it preserves tab and newline as
/// layout intent, replaces every other control (CR, C1, DEL) with U+FFFD.
///
/// The SIMD kernels above implement this same [`AllowedControls`] set.  The
/// SWAR body below is the fallback for the final short tail.  Its
/// `< 0x20`, tab and newline masks are **exact** (each minuend byte has its
/// high bit forced with `x | HI`, so the word subtraction cannot borrow
/// between lanes).  The high-byte/DEL rejection is deliberately *not* an
/// exact mask: it is existential — any high bit in `x` or a DEL lane in
/// `x ^ 0x7F` is itself the proof, and a borrow chain can only originate at
/// such a lane within the same chunk.
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
    if !swar_forced() && s.len() >= 64 && x86_level() >= 3 {
        return unsafe { x86_64::is_printable_ascii_multiline_avx512(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 32 && x86_level() >= 2 {
        // SAFETY: runtime-checked AVX2; tail is deferred to this function.
        return unsafe { x86_64::is_printable_ascii_multiline(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 16 {
        return unsafe { x86_64::is_printable_ascii_multiline_sse2(s) };
    }
    #[cfg(target_arch = "aarch64")]
    if !swar_forced() && s.len() >= 16 {
        return unsafe { neon::is_printable_ascii_multiline(s) };
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
        let q = x ^ DEL;
        if (x | (q.wrapping_sub(ONES))) & HI != 0 {
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
    const HI64: u64 = 0x8080_8080_8080_8080;
    const SP64: u64 = 0x2020_2020_2020_2020;
    const TAB64: u64 = 0x0909_0909_0909_0909;
    const NL64: u64 = 0x0A0A_0A0A_0A0A_0A0A;
    const DEL64: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES64: u64 = 0x0101_0101_0101_0101;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        let q = x ^ DEL64;
        if (x | (q.wrapping_sub(ONES64))) & HI64 != 0 {
            return false;
        }
        let lt20 = !((x | HI64).wrapping_sub(SP64)) & HI64;
        if lt20 != 0 {
            let tab = !(((x ^ TAB64) | HI64).wrapping_sub(ONES64)) & HI64;
            let nl = !(((x ^ NL64) | HI64).wrapping_sub(ONES64)) & HI64;
            if (lt20 & !tab & !nl) != 0 {
                return false;
            }
        }
        i += 8;
    }
    s[i..]
        .iter()
        .all(|&b| AllowedControls::TabAndNewline.allows(b))
}

/// `true` when every byte is printable ASCII **or** the tab (`0x09`).
///
/// The gate for `render_line_merged_with`'s tab-aware byte-level fast path:
/// source code is routinely tab-indented, and the pure-printable gate
/// rejected tab (`0x09 < 0x20`), pushing every tab-indented row into the
/// grapheme walk.  [`AllowedControls::Tab`] is the exact set; the SIMD kernels
/// and the SWAR fallback both implement it.
#[inline]
pub(crate) fn is_printable_ascii_or_tab(s: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 64 && x86_level() >= 3 {
        return unsafe { x86_64::is_printable_ascii_or_tab_avx512(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 32 && x86_level() >= 2 {
        // SAFETY: runtime-checked AVX2; tail is deferred to this function.
        return unsafe { x86_64::is_printable_ascii_or_tab(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 16 {
        return unsafe { x86_64::is_printable_ascii_or_tab_sse2(s) };
    }
    #[cfg(target_arch = "aarch64")]
    if !swar_forced() && s.len() >= 16 {
        return unsafe { neon::is_printable_ascii_or_tab(s) };
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
        let v = x ^ DEL;
        if (x | (v.wrapping_sub(ONES))) & HI != 0 {
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
    const HI64: u64 = 0x8080_8080_8080_8080;
    const SP64: u64 = 0x2020_2020_2020_2020;
    const TAB64: u64 = 0x0909_0909_0909_0909;
    const DEL64: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES64: u64 = 0x0101_0101_0101_0101;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        let v = x ^ DEL64;
        if (x | (v.wrapping_sub(ONES64))) & HI64 != 0 {
            return false;
        }
        let lt20 = !((x | HI64).wrapping_sub(SP64)) & HI64;
        if lt20 != 0 {
            let tab = !(((x ^ TAB64) | HI64).wrapping_sub(ONES64)) & HI64;
            if (lt20 & !tab) != 0 {
                return false;
            }
        }
        i += 8;
    }
    s[i..].iter().all(|&b| AllowedControls::Tab.allows(b))
}

#[inline]
fn compose_classify_tail(saw_tab: bool, tail: AsciiClass) -> AsciiClass {
    if tail == AsciiClass::Other {
        AsciiClass::Other
    } else if tail == AsciiClass::PrintableOrTab || saw_tab {
        AsciiClass::PrintableOrTab
    } else {
        AsciiClass::Printable
    }
}

/// Tri-state classification of one row, so callers that need both the
/// printable and printable-or-tab predicates (the glyph fast path tested them
/// sequentially) pay one SWAR pass instead of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AsciiClass {
    /// Every byte is printable ASCII (`0x20..=0x7E`).
    Printable,
    /// Printable ASCII plus horizontal tabs; no other control bytes.
    PrintableOrTab,
    /// Anything else (high byte, DEL, other C0/C1).
    Other,
}

/// Classify `s` in one pass at the widest available vector width, with the
/// SWAR fallback for the tail.
///
/// The SWAR fallback rejects high bytes first (as
/// [`is_printable_ascii_multiline`] does): the `(x | HI) - SP` form is only
/// exact when every lane is ≤ `0x7F`.
#[inline]
pub(crate) fn classify_ascii(s: &[u8]) -> AsciiClass {
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 64 && x86_level() >= 3 {
        return unsafe { x86_64::classify_ascii_avx512(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 32 && x86_level() >= 2 {
        // SAFETY: runtime-checked AVX2; tail is deferred to this function.
        return unsafe { x86_64::classify_ascii(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 16 {
        return unsafe { x86_64::classify_ascii_sse2(s) };
    }
    #[cfg(target_arch = "aarch64")]
    if !swar_forced() && s.len() >= 16 {
        return unsafe { neon::classify_ascii(s) };
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
        let v = x ^ DEL;
        if (x | (v.wrapping_sub(ONES))) & HI != 0 {
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
        } else if !AllowedControls::None.allows(b) {
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
/// SIMD kernels scan the widest available chunk; the SWAR fallback uses
/// `u128` chunks: every non-ASCII UTF-8 byte (lead or continuation) has its
/// high bit set, so one AND per 16-byte chunk decides whether the chunk holds
/// a candidate; a `trailing_zeros` on the flagged lanes locates it.
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
    if !swar_forced() && s.len() >= 64 && x86_level() >= 3 {
        return unsafe { x86_64::next_non_ascii_byte_avx512(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 32 && x86_level() >= 2 {
        // SAFETY: runtime-checked AVX2; tail is deferred to this function.
        return unsafe { x86_64::next_non_ascii_byte(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 16 {
        return unsafe { x86_64::next_non_ascii_byte_sse2(s) };
    }
    #[cfg(target_arch = "aarch64")]
    if !swar_forced() && s.len() >= 16 {
        return unsafe { neon::next_non_ascii_byte(s) };
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

/// Runtime-dispatched SIMD kernels.
///
/// `x86_64` uses AVX-512BW for 64-byte chunks, AVX2 for 32-byte chunks, and
/// SSE2 for 16-byte chunks; `aarch64` uses NEON for 16-byte chunks.  The
/// scalar SWAR body remains the tail for every kernel, so the lane proofs in
/// this module stay the single source of truth below the vector width.
///
/// Every kernel processes only full chunks.  The tail deliberately avoids a
/// masked load: the fallback is one cached feature branch plus at most
/// `width - 1` bytes of SWAR, while masked tails add unsafe edge cases for no
/// measured gain on diagnostic-sized inputs.
#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn is_printable_ascii(s: &[u8]) -> bool {
        let sp = _mm256_set1_epi8(0x20);
        let del = _mm256_set1_epi8(0x7F);
        let mut i = 0usize;
        while i + 64 <= s.len() {
            let a = unsafe { _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i) };
            let b = unsafe { _mm256_loadu_si256(s.as_ptr().add(i + 32) as *const __m256i) };
            let bad_a = unsafe {
                _mm256_or_si256(_mm256_cmpgt_epi8(sp, a), _mm256_cmpeq_epi8(a, del))
            };
            let bad_b = unsafe {
                _mm256_or_si256(_mm256_cmpgt_epi8(sp, b), _mm256_cmpeq_epi8(b, del))
            };
            if unsafe { _mm256_movemask_epi8(_mm256_or_si256(bad_a, bad_b)) } != 0 {
                return false;
            }
            i += 64;
        }
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
            let lt = unsafe { _mm256_cmpgt_epi8(sp, v) };
            let tab = unsafe { _mm256_cmpeq_epi8(v, tab_v) };
            let nl = unsafe { _mm256_cmpeq_epi8(v, nl_v) };
            let allowed = unsafe { _mm256_or_si256(tab, nl) };
            let off = unsafe { _mm256_andnot_si256(allowed, lt) };
            let eq = unsafe { _mm256_cmpeq_epi8(v, del) };
            let bad = unsafe { _mm256_or_si256(eq, off) };
            if unsafe { _mm256_movemask_epi8(bad) } != 0 {
                return false;
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
            let lt = unsafe { _mm256_cmpgt_epi8(sp, v) };
            let tab = unsafe { _mm256_cmpeq_epi8(v, tab_v) };
            let off = unsafe { _mm256_andnot_si256(tab, lt) };
            let eq = unsafe { _mm256_cmpeq_epi8(v, del) };
            let bad = unsafe { _mm256_or_si256(eq, off) };
            if unsafe { _mm256_movemask_epi8(bad) } != 0 {
                return false;
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
            let lt = unsafe { _mm256_cmpgt_epi8(sp, v) };
            let tab = unsafe { _mm256_cmpeq_epi8(v, tab_v) };
            let off = unsafe { _mm256_andnot_si256(tab, lt) };
            let eq = unsafe { _mm256_cmpeq_epi8(v, del) };
            let bad = unsafe { _mm256_or_si256(eq, off) };
            if unsafe { _mm256_movemask_epi8(bad) } != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { _mm256_movemask_epi8(tab) } != 0;
            i += 32;
        }
        super::compose_classify_tail(saw_tab, super::classify_ascii(&s[i..]))
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn next_non_ascii_byte(s: &[u8]) -> Option<usize> {
        let mut i = 0usize;
        while i + 128 <= s.len() {
            let m0 = unsafe {
                _mm256_movemask_epi8(_mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i))
            };
            let m1 = unsafe {
                _mm256_movemask_epi8(_mm256_loadu_si256(
                    s.as_ptr().add(i + 32) as *const __m256i
                ))
            };
            let m2 = unsafe {
                _mm256_movemask_epi8(_mm256_loadu_si256(
                    s.as_ptr().add(i + 64) as *const __m256i
                ))
            };
            let m3 = unsafe {
                _mm256_movemask_epi8(_mm256_loadu_si256(
                    s.as_ptr().add(i + 96) as *const __m256i
                ))
            };
            if (m0 | m1 | m2 | m3) != 0 {
                for (off, mask) in [(0, m0), (32, m1), (64, m2), (96, m3)] {
                    if mask != 0 {
                        return Some(i + off + mask.trailing_zeros() as usize);
                    }
                }
            }
            i += 128;
        }
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

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn is_printable_ascii_avx512(s: &[u8]) -> bool {
        let sp = _mm512_set1_epi8(0x20);
        let del = _mm512_set1_epi8(0x7F);
        let mut i = 0usize;
        while i + 256 <= s.len() {
            let a = unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) };
            let b = unsafe { _mm512_loadu_si512(s.as_ptr().add(i + 64).cast()) };
            let c = unsafe { _mm512_loadu_si512(s.as_ptr().add(i + 128).cast()) };
            let d = unsafe { _mm512_loadu_si512(s.as_ptr().add(i + 192).cast()) };
            let bad = (_mm512_cmpgt_epi8_mask(sp, a) | _mm512_cmpeq_epi8_mask(a, del))
                | (_mm512_cmpgt_epi8_mask(sp, b) | _mm512_cmpeq_epi8_mask(b, del))
                | (_mm512_cmpgt_epi8_mask(sp, c) | _mm512_cmpeq_epi8_mask(c, del))
                | (_mm512_cmpgt_epi8_mask(sp, d) | _mm512_cmpeq_epi8_mask(d, del));
            if bad != 0 {
                return false;
            }
            i += 256;
        }
        while i + 64 <= s.len() {
            let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) };
            let lt = unsafe { _mm512_cmpgt_epi8_mask(sp, v) };
            let eq = unsafe { _mm512_cmpeq_epi8_mask(v, del) };
            if lt | eq != 0 {
                return false;
            }
            i += 64;
        }
        super::is_printable_ascii(&s[i..])
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn is_printable_ascii_multiline_avx512(s: &[u8]) -> bool {
        let sp = _mm512_set1_epi8(0x20);
        let del = _mm512_set1_epi8(0x7F);
        let tab_v = _mm512_set1_epi8(0x09);
        let nl_v = _mm512_set1_epi8(0x0A);
        let mut i = 0usize;
        while i + 64 <= s.len() {
            let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) };
            let lt = unsafe { _mm512_cmpgt_epi8_mask(sp, v) };
            let tab = unsafe { _mm512_cmpeq_epi8_mask(v, tab_v) };
            let nl = unsafe { _mm512_cmpeq_epi8_mask(v, nl_v) };
            let off = lt & !(tab | nl);
            let eq = unsafe { _mm512_cmpeq_epi8_mask(v, del) };
            if eq | off != 0 {
                return false;
            }
            i += 64;
        }
        super::is_printable_ascii_multiline(&s[i..])
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn is_printable_ascii_or_tab_avx512(s: &[u8]) -> bool {
        let sp = _mm512_set1_epi8(0x20);
        let del = _mm512_set1_epi8(0x7F);
        let tab_v = _mm512_set1_epi8(0x09);
        let mut i = 0usize;
        while i + 64 <= s.len() {
            let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) };
            let lt = unsafe { _mm512_cmpgt_epi8_mask(sp, v) };
            let tab = unsafe { _mm512_cmpeq_epi8_mask(v, tab_v) };
            let off = lt & !tab;
            let eq = unsafe { _mm512_cmpeq_epi8_mask(v, del) };
            if eq | off != 0 {
                return false;
            }
            i += 64;
        }
        super::is_printable_ascii_or_tab(&s[i..])
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn classify_ascii_avx512(s: &[u8]) -> super::AsciiClass {
        let sp = _mm512_set1_epi8(0x20);
        let del = _mm512_set1_epi8(0x7F);
        let tab_v = _mm512_set1_epi8(0x09);
        let mut saw_tab = false;
        let mut i = 0usize;
        while i + 64 <= s.len() {
            let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) };
            let lt = unsafe { _mm512_cmpgt_epi8_mask(sp, v) };
            let tab = unsafe { _mm512_cmpeq_epi8_mask(v, tab_v) };
            let off = lt & !tab;
            let eq = unsafe { _mm512_cmpeq_epi8_mask(v, del) };
            if eq | off != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= tab != 0;
            i += 64;
        }
        super::compose_classify_tail(saw_tab, super::classify_ascii(&s[i..]))
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn next_non_ascii_byte_avx512(s: &[u8]) -> Option<usize> {
        let mut i = 0usize;
        while i + 256 <= s.len() {
            let m0 = unsafe { _mm512_movepi8_mask(_mm512_loadu_si512(s.as_ptr().add(i).cast())) };
            let m1 =
                unsafe { _mm512_movepi8_mask(_mm512_loadu_si512(s.as_ptr().add(i + 64).cast())) };
            let m2 =
                unsafe { _mm512_movepi8_mask(_mm512_loadu_si512(s.as_ptr().add(i + 128).cast())) };
            let m3 =
                unsafe { _mm512_movepi8_mask(_mm512_loadu_si512(s.as_ptr().add(i + 192).cast())) };
            if (m0 | m1 | m2 | m3) != 0 {
                for (off, mask) in [(0, m0), (64, m1), (128, m2), (192, m3)] {
                    if mask != 0 {
                        return Some(i + off + mask.trailing_zeros() as usize);
                    }
                }
            }
            i += 256;
        }
        while i + 64 <= s.len() {
            let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) };
            let mask = unsafe { _mm512_movepi8_mask(v) };
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 64;
        }
        super::next_non_ascii_byte(&s[i..]).map(|off| i + off)
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn is_printable_ascii_sse2(s: &[u8]) -> bool {
        let sp = _mm_set1_epi8(0x20);
        let del = _mm_set1_epi8(0x7F);
        let mut i = 0usize;
        while i + 32 <= s.len() {
            let a = unsafe { _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i) };
            let b = unsafe { _mm_loadu_si128(s.as_ptr().add(i + 16) as *const __m128i) };
            let bad_a =
                unsafe { _mm_or_si128(_mm_cmpgt_epi8(sp, a), _mm_cmpeq_epi8(a, del)) };
            let bad_b =
                unsafe { _mm_or_si128(_mm_cmpgt_epi8(sp, b), _mm_cmpeq_epi8(b, del)) };
            if unsafe { _mm_movemask_epi8(_mm_or_si128(bad_a, bad_b)) } != 0 {
                return false;
            }
            i += 32;
        }
        while i + 16 <= s.len() {
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i) };
            let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
            let eq = unsafe { _mm_cmpeq_epi8(v, del) };
            if unsafe { _mm_movemask_epi8(_mm_or_si128(lt, eq)) } != 0 {
                return false;
            }
            i += 16;
        }
        super::is_printable_ascii(&s[i..])
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn is_printable_ascii_multiline_sse2(s: &[u8]) -> bool {
        let sp = _mm_set1_epi8(0x20);
        let del = _mm_set1_epi8(0x7F);
        let tab_v = _mm_set1_epi8(0x09);
        let nl_v = _mm_set1_epi8(0x0A);
        let mut i = 0usize;
        while i + 16 <= s.len() {
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i) };
            let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
            let tab = unsafe { _mm_cmpeq_epi8(v, tab_v) };
            let nl = unsafe { _mm_cmpeq_epi8(v, nl_v) };
            let allowed = unsafe { _mm_or_si128(tab, nl) };
            let off = unsafe { _mm_andnot_si128(allowed, lt) };
            let eq = unsafe { _mm_cmpeq_epi8(v, del) };
            let bad = unsafe { _mm_or_si128(eq, off) };
            if unsafe { _mm_movemask_epi8(bad) } != 0 {
                return false;
            }
            i += 16;
        }
        super::is_printable_ascii_multiline(&s[i..])
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn is_printable_ascii_or_tab_sse2(s: &[u8]) -> bool {
        let sp = _mm_set1_epi8(0x20);
        let del = _mm_set1_epi8(0x7F);
        let tab_v = _mm_set1_epi8(0x09);
        let mut i = 0usize;
        while i + 16 <= s.len() {
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i) };
            let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
            let tab = unsafe { _mm_cmpeq_epi8(v, tab_v) };
            let off = unsafe { _mm_andnot_si128(tab, lt) };
            let eq = unsafe { _mm_cmpeq_epi8(v, del) };
            let bad = unsafe { _mm_or_si128(eq, off) };
            if unsafe { _mm_movemask_epi8(bad) } != 0 {
                return false;
            }
            i += 16;
        }
        super::is_printable_ascii_or_tab(&s[i..])
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn classify_ascii_sse2(s: &[u8]) -> super::AsciiClass {
        let sp = _mm_set1_epi8(0x20);
        let del = _mm_set1_epi8(0x7F);
        let tab_v = _mm_set1_epi8(0x09);
        let mut saw_tab = false;
        let mut i = 0usize;
        while i + 16 <= s.len() {
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i) };
            let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
            let tab = unsafe { _mm_cmpeq_epi8(v, tab_v) };
            let off = unsafe { _mm_andnot_si128(tab, lt) };
            let eq = unsafe { _mm_cmpeq_epi8(v, del) };
            let bad = unsafe { _mm_or_si128(eq, off) };
            if unsafe { _mm_movemask_epi8(bad) } != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { _mm_movemask_epi8(tab) } != 0;
            i += 16;
        }
        super::compose_classify_tail(saw_tab, super::classify_ascii(&s[i..]))
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn next_non_ascii_byte_sse2(s: &[u8]) -> Option<usize> {
        let mut i = 0usize;
        while i + 64 <= s.len() {
            let m0 = unsafe {
                _mm_movemask_epi8(_mm_loadu_si128(s.as_ptr().add(i) as *const __m128i))
            };
            let m1 = unsafe {
                _mm_movemask_epi8(_mm_loadu_si128(s.as_ptr().add(i + 16) as *const __m128i))
            };
            let m2 = unsafe {
                _mm_movemask_epi8(_mm_loadu_si128(s.as_ptr().add(i + 32) as *const __m128i))
            };
            let m3 = unsafe {
                _mm_movemask_epi8(_mm_loadu_si128(s.as_ptr().add(i + 48) as *const __m128i))
            };
            if (m0 | m1 | m2 | m3) != 0 {
                for (off, mask) in [(0, m0), (16, m1), (32, m2), (48, m3)] {
                    if mask != 0 {
                        return Some(i + off + mask.trailing_zeros() as usize);
                    }
                }
            }
            i += 64;
        }
        while i + 16 <= s.len() {
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i) };
            let mask = unsafe { _mm_movemask_epi8(v) };
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 16;
        }
        super::next_non_ascii_byte(&s[i..]).map(|off| i + off)
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn any_nonzero(v: uint8x16_t) -> bool {
        unsafe { vmaxvq_u8(v) != 0 }
    }

    #[inline]
    unsafe fn first_high_lane(v: uint8x16_t) -> Option<usize> {
        let high = unsafe { vcgeq_u8(v, vdupq_n_u8(0x80)) };
        let packed = unsafe { vshrn_n_u16(vreinterpretq_u16_u8(high), 4) };
        let bits = unsafe { vget_lane_u64(vreinterpret_u64_u8(packed), 0) };
        (bits != 0).then_some((bits.trailing_zeros() / 4) as usize)
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn is_printable_ascii(s: &[u8]) -> bool {
        let sp = unsafe { vdupq_n_u8(0x20) };
        let del = unsafe { vdupq_n_u8(0x7F) };
        let mut i = 0usize;
        while i + 32 <= s.len() {
            let a = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            let b = unsafe { vld1q_u8(s.as_ptr().add(i + 16)) };
            let bad = |v: uint8x16_t| unsafe {
                let lt = vcltq_u8(v, sp);
                let eq = vceqq_u8(v, del);
                let high = vcgeq_u8(v, vdupq_n_u8(0x80));
                vorrq_u8(vorrq_u8(lt, eq), high)
            };
            if unsafe { any_nonzero(vorrq_u8(bad(a), bad(b))) } {
                return false;
            }
            i += 32;
        }
        while i + 16 <= s.len() {
            let v = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            let lt = unsafe { vcltq_u8(v, sp) };
            let eq = unsafe { vceqq_u8(v, del) };
            let bad = unsafe { vorrq_u8(lt, eq) };
            let high = unsafe { vcgeq_u8(v, vdupq_n_u8(0x80)) };
            if unsafe { any_nonzero(vorrq_u8(bad, high)) } {
                return false;
            }
            i += 16;
        }
        super::is_printable_ascii(&s[i..])
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn is_printable_ascii_multiline(s: &[u8]) -> bool {
        let sp = unsafe { vdupq_n_u8(0x20) };
        let del = unsafe { vdupq_n_u8(0x7F) };
        let tab_v = unsafe { vdupq_n_u8(0x09) };
        let nl_v = unsafe { vdupq_n_u8(0x0A) };
        let mut i = 0usize;
        while i + 16 <= s.len() {
            let v = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            let lt = unsafe { vcltq_u8(v, sp) };
            let tab = unsafe { vceqq_u8(v, tab_v) };
            let nl = unsafe { vceqq_u8(v, nl_v) };
            let allowed = unsafe { vorrq_u8(tab, nl) };
            let off = unsafe { vbicq_u8(lt, allowed) };
            let eq = unsafe { vceqq_u8(v, del) };
            let high = unsafe { vcgeq_u8(v, vdupq_n_u8(0x80)) };
            if unsafe { any_nonzero(vorrq_u8(vorrq_u8(eq, off), high)) } {
                return false;
            }
            i += 16;
        }
        super::is_printable_ascii_multiline(&s[i..])
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn is_printable_ascii_or_tab(s: &[u8]) -> bool {
        let sp = unsafe { vdupq_n_u8(0x20) };
        let del = unsafe { vdupq_n_u8(0x7F) };
        let tab_v = unsafe { vdupq_n_u8(0x09) };
        let mut i = 0usize;
        while i + 16 <= s.len() {
            let v = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            let lt = unsafe { vcltq_u8(v, sp) };
            let tab = unsafe { vceqq_u8(v, tab_v) };
            let off = unsafe { vbicq_u8(lt, tab) };
            let eq = unsafe { vceqq_u8(v, del) };
            let high = unsafe { vcgeq_u8(v, vdupq_n_u8(0x80)) };
            if unsafe { any_nonzero(vorrq_u8(vorrq_u8(eq, off), high)) } {
                return false;
            }
            i += 16;
        }
        super::is_printable_ascii_or_tab(&s[i..])
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn classify_ascii(s: &[u8]) -> super::AsciiClass {
        let sp = unsafe { vdupq_n_u8(0x20) };
        let del = unsafe { vdupq_n_u8(0x7F) };
        let tab_v = unsafe { vdupq_n_u8(0x09) };
        let mut saw_tab = false;
        let mut i = 0usize;
        while i + 16 <= s.len() {
            let v = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            let lt = unsafe { vcltq_u8(v, sp) };
            let tab = unsafe { vceqq_u8(v, tab_v) };
            let off = unsafe { vbicq_u8(lt, tab) };
            let eq = unsafe { vceqq_u8(v, del) };
            let high = unsafe { vcgeq_u8(v, vdupq_n_u8(0x80)) };
            if unsafe { any_nonzero(vorrq_u8(vorrq_u8(eq, off), high)) } {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { any_nonzero(tab) };
            i += 16;
        }
        super::compose_classify_tail(saw_tab, super::classify_ascii(&s[i..]))
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn next_non_ascii_byte(s: &[u8]) -> Option<usize> {
        let mut i = 0usize;
        while i + 64 <= s.len() {
            let chunks = [
                unsafe { vld1q_u8(s.as_ptr().add(i)) },
                unsafe { vld1q_u8(s.as_ptr().add(i + 16)) },
                unsafe { vld1q_u8(s.as_ptr().add(i + 32)) },
                unsafe { vld1q_u8(s.as_ptr().add(i + 48)) },
            ];
            for (off, v) in chunks.into_iter().enumerate() {
                if let Some(lane) = unsafe { first_high_lane(v) } {
                    return Some(i + off * 16 + lane);
                }
            }
            i += 64;
        }
        while i + 16 <= s.len() {
            let v = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            if let Some(lane) = unsafe { first_high_lane(v) } {
                return Some(i + lane);
            }
            i += 16;
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
        // Cross every unrolled block boundary: a high byte only in the last
        // chunk must still be located exactly.
        for at in [64, 127, 128, 255, 256, 300] {
            let mut v = vec![b'a'; 320];
            v[at] = 0xC3;
            assert_eq!(next_non_ascii_byte(&v), Some(at), "offset {at}");
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

    #[test]
    fn forced_swar_matches_scalar_for_all_predicates() {
        SWAR_FORCED.with(|f| f.set(true));
        let ref_classify = |s: &[u8]| {
            let mut tab = false;
            for &b in s {
                if b == b'\t' {
                    tab = true;
                } else if !(0x20..=0x7E).contains(&b) {
                    return AsciiClass::Other;
                }
            }
            if tab {
                AsciiClass::PrintableOrTab
            } else {
                AsciiClass::Printable
            }
        };
        let mut seed: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(7);
            seed
        };
        let adversarial = [
            0x00, 0x01, 0x09, 0x0A, 0x0D, 0x1F, 0x20, 0x41, 0x7E, 0x7F, 0x80, 0xC2, 0xFF,
        ];
        for _ in 0..6000 {
            let len = (next() % 81) as usize;
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    if next() & 3 == 0 {
                        adversarial[(next() as usize) % adversarial.len()]
                    } else {
                        (next() & 0xFF) as u8
                    }
                })
                .collect();
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
            assert_eq!(classify_ascii(&bytes), ref_classify(&bytes), "{bytes:?}");
            assert_eq!(
                next_non_ascii_byte(&bytes),
                bytes.iter().position(|&b| b >= 0x80),
                "{bytes:?}"
            );
        }
        SWAR_FORCED.with(|f| f.set(false));
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
            &b""[..],
            b"\t",
            b"\n",
            b"\t\n\t\n\t\n",
            b"\n\n\n\n\n\n\n\n",
            b"\t\n\r\t\n\r",
            b"a\tb\nc\rd",
            b"\x01\t\n\x0B",
            b"\t\t\t\t\t\t\t\t",
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

    /// Long inputs cross the SSE2 (16B), AVX2 (32B), and AVX-512BW (64B)
    /// dispatch thresholds.  Every predicate is checked against its scalar
    /// reference at each width and at the ±1 tail boundaries.
    #[test]
    fn long_simd_paths_match_scalar_reference() {
        for len in [64, 65, 95, 96, 127, 128, 129, 191, 192, 255, 256, 257, 1024] {
            let printable = vec![b'~'; len];
            assert!(is_printable_ascii(&printable), "len {len}");
            assert!(is_printable_ascii_or_tab(&printable), "len {len}");
            assert!(is_printable_ascii_multiline(&printable), "len {len}");
            assert_eq!(
                classify_ascii(&printable),
                AsciiClass::Printable,
                "len {len}"
            );

            let mut tabbed = printable.clone();
            tabbed[len / 2] = b'\t';
            assert!(!is_printable_ascii(&tabbed), "len {len}");
            assert!(is_printable_ascii_or_tab(&tabbed), "len {len}");
            assert!(is_printable_ascii_multiline(&tabbed), "len {len}");
            assert_eq!(
                classify_ascii(&tabbed),
                AsciiClass::PrintableOrTab,
                "len {len}"
            );

            let mut multiline = tabbed.clone();
            multiline[len - 1] = b'\n';
            assert!(is_printable_ascii_multiline(&multiline), "len {len}");
            assert!(!is_printable_ascii_or_tab(&multiline), "len {len}");

            for offender in [b'\r', b'\x7F', 0x80, 0xFF] {
                let mut bad = printable.clone();
                bad[len / 3] = offender;
                assert!(!is_printable_ascii(&bad), "len {len}, byte {offender:#04x}");
                assert!(
                    !is_printable_ascii_or_tab(&bad),
                    "len {len}, byte {offender:#04x}"
                );
                assert!(
                    !is_printable_ascii_multiline(&bad),
                    "len {len}, byte {offender:#04x}"
                );
                assert_eq!(
                    classify_ascii(&bad),
                    AsciiClass::Other,
                    "len {len}, byte {offender:#04x}"
                );
            }
        }
    }

    /// `next_non_ascii_byte` must locate a high byte in every 64-bit AVX-512
    /// lane, including the second half of a 128-byte scan.
    #[test]
    fn next_non_ascii_covers_avx512_lanes() {
        for lane in 0..128 {
            let mut v = vec![b'a'; 256];
            v[lane] = 0x80;
            assert_eq!(next_non_ascii_byte(&v), Some(lane), "lane {lane}");
        }
    }
}
