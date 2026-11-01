//! Branch-light ASCII predicates shared by the renderers.
//!
//! Every predicate here answers "can this byte slice be emitted without any
//! per-byte work elsewhere?" — i.e. one display cell per byte, no escape
//! parsing, no grapheme segmentation. The renderers use these as fast-path
//! gates; a `true` answer lets them skip the Unicode tables entirely.

/// `true` when every byte is printable ASCII (`0x20..=0x7E`).
///
/// SWAR: two masks per eight bytes instead of two compares per byte.
/// The one-byte tail is a bounds-checked `matches!`, which LLVM vectorises
/// for the ≤7-byte remainder.
#[inline]
pub(crate) fn is_printable_ascii(s: &[u8]) -> bool {
    const HI: u64 = 0x8080_8080_8080_8080;
    const SP: u64 = 0x2020_2020_2020_2020;
    const DEL: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES: u64 = 0x0101_0101_0101_0101;

    let mut i = 0usize;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        // A lane's high bit survives `x | (x - 0x20…)` iff the byte is
        // `< 0x20` (the subtraction borrows) or `>= 0x80` (its own bit).
        if (x | x.wrapping_sub(SP)) & HI != 0 {
            return false;
        }
        // 0x7F is the one byte inside [0x20, 0x7F) that is not printable.
        let v = x ^ DEL;
        if v.wrapping_sub(ONES) & !v & HI != 0 {
            return false;
        }
        i += 8;
    }
    s[i..].iter().all(|&b| (0x20..=0x7E).contains(&b))
}

/// `true` when every byte is printable ASCII **or** the two layout controls
/// `\t` (0x09) / `\n` (0x0A). This is the predicate
/// `sanitize_for_terminal_multiline` needs: it preserves tab and newline as
/// layout intent, replaces every other control (CR, C1, DEL) with U+FFFD.
///
/// A 256-byte lookup table beats SWAR here: the allowed set has a hole
/// (`0x0B..=0x1F`) that no single mask pair expresses, and the per-byte load
/// is one cycle — the strings are messages, not megabytes.
#[inline]
pub(crate) fn is_printable_ascii_multiline(s: &[u8]) -> bool {
    static ALLOWED: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = 0usize;
        while i < 256 {
            let b = i as u8;
            t[i] = b == 0x09 || b == 0x0A || (b >= 0x20 && b <= 0x7E);
            i += 1;
        }
        t
    };
    s.iter().all(|&b| ALLOWED[b as usize])
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    fn ref_single(s: &[u8]) -> bool {
        s.iter().all(|&b| (0x20..=0x7E).contains(&b))
    }
    fn ref_multi(s: &[u8]) -> bool {
        s.iter()
            .all(|&b| b == 0x09 || b == 0x0A || (0x20..=0x7E).contains(&b))
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
            }
        }
    }

    #[test]
    fn multiline_allows_tab_and_newline_only_among_controls() {
        assert!(is_printable_ascii_multiline(b"a\tb\nc"));
        assert!(!is_printable_ascii_multiline(b"a\rb")); // CR rejected
        assert!(!is_printable_ascii_multiline(b"a\x0Bb")); // VT
        assert!(!is_printable_ascii_multiline(b"a\x7Fb")); // DEL
        assert!(!is_printable_ascii_multiline("a中b".as_bytes())); // non-ASCII
    }
}
