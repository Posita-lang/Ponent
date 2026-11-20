//! Branch-light ASCII predicates shared by the renderers.
//!
//! Every predicate here answers "can this byte slice be emitted without any
//! per-byte work elsewhere?" — i.e. one display cell per byte, no escape
//! parsing, no grapheme segmentation. The renderers use these as fast-path
//! gates; a `true` answer lets them skip the Unicode tables entirely.

/// The byte set every fast path admits as *plain printable*: `0x20..=0x7E`.
///
/// This used to be an `AllowedControls { None, Tab, TabAndNewline }` ladder that
/// kept three scalar tails in lockstep.  Once the predicates became views of one
/// `AsciiClass` classifier, only the `None` arm was still constructed and the
/// other two were dead code describing classes the classifier expresses
/// directly — so the ladder is gone and the invariant it carried lives here:
/// **high bytes are never printable**.  The SIMD kernels rely on it (their `lt`
/// masks catch signed-negative bytes) and the SWAR body relies on it
/// independently (`x | (v - ONES)`), so admitting `>= 0x80` would break both.
#[inline]
const fn is_printable_byte(b: u8) -> bool {
    matches!(b, 0x20..=0x7E)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_level_for(len: usize) -> u8 {
    // The `len < 32` short-circuit deliberately does NOT consult
    // `swar_forced`: that is why each SSE2 dispatch arm below keeps its own
    // `!swar_forced()` conjunct.  For `len < 32` the level is 0 without a
    // forced check, so without the arm-level conjunct a forced-SWAR run would
    // silently execute SSE2 for `len >= 16`, defeating the forced path — the
    // conjunct is load-bearing, not redundant.
    if len < 32 { 0 } else { x86_level() }
}

#[cfg(target_arch = "x86_64")]
static LEVEL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Cached ISA level: `3` = AVX-512BW, `2` = AVX2, `1` = SSE2 baseline.
///
/// Split so the hot path is one `Relaxed` load that inlines into the caller's
/// predicate body, while the detection (three `is_x86_feature_detected!` reads
/// plus a store) lives in a cold, never-inlined helper.  The manual cache is
/// still worth keeping over reading the macros every call: it collapses three
/// cached loads into one, and it is the module's own contract rather than an
/// implementation detail of the macros.
#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_level() -> u8 {
    // Order is load-bearing: `swar_forced()` FIRST, cache second.  A forced
    // thread must observe level 0 even when another thread already populated
    // `LEVEL`; reading the cache first would return the detected level and send
    // a forced-SWAR test straight into AVX2/AVX-512 (only the SSE2 arm carries
    // its own `!swar_forced()` conjunct, because `x86_level_for` short-circuits
    // `len < 32` without consulting it).  The TLS read this costs exists only
    // under `test`/`bench-hooks`; in production `swar_forced()` folds to
    // `false` and the cache load dominates either way.
    if swar_forced() {
        return 0;
    }
    let cached = LEVEL.load(std::sync::atomic::Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    x86_level_init()
}

/// One-time detection.  `swar_forced()` is checked by the caller and the forced
/// path returns before reaching here, so a test that forces SWAR can never
/// poison the cache for the rest of the process.
#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
fn x86_level_init() -> u8 {
    // The AVX2 arm below only tests `level >= 2`, and the AVX-512 kernel's
    // 32..63-byte tail defers *to that arm*.  So level 3 must imply AVX2, not
    // merely assume the folklore that every AVX-512BW part also has AVX2 —
    // probing `avx512bw` alone skips the `avx2` probe entirely, which would make
    // the SAFETY comments literally false on a hypothetical BW-without-AVX2
    // part (vpbroadcastb is AVX2; the tail would execute an illegal
    // instruction).  One extra cached feature read, paid once per process.
    let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
    let level = if std::arch::is_x86_feature_detected!("avx512bw") && has_avx2 {
        3
    } else if has_avx2 {
        2
    } else {
        1
    };
    LEVEL.store(level, std::sync::atomic::Ordering::Relaxed);
    level
}

/// AVX-512 processes 64 bytes per instruction, so the 1× 64-byte loop already
/// covers `[64, 256)`; entering at 64 spends the wider kernel instead of the
/// 32-byte AVX2 one with no correctness change (every kernel handles any
/// length through its own loop plus the deferred parent tail).
///
/// Downclocking tradeoff, stated rather than assumed: on parts with AVX-512
/// frequency licensing (Skylake-X, some Ice Lake server SKUs) the transition
/// can cost more than the 1 load it saves relative to the 2 AVX2 loads a
/// 64-byte row requires,
/// and the penalty lands on the surrounding non-AVX-512 code.  Modern parts
/// (Sapphire Rapids, Zen 4) do not downclock at all.  Raising the threshold to
/// 128 would make every 64–127-byte row take AVX2 instead, so it is kept at 64
/// until a measurement on the target mix says otherwise — this constant is the
/// knob, and no correctness argument depends on its value.
const AVX512_MIN_LEN: usize = 64;

// Structural, not a debug assert: the AVX-512 kernel's inner loop is one 64-byte
// chunk and its tail re-enters the dispatcher, so a threshold below 64 would
// re-dispatch at the same length and recurse until the stack dies.  In release
// that would be a production stack overflow; here it is a build error.  One
// assertion covers both dispatchers (`classify_ascii`, `next_non_ascii_byte`).
const _: () = assert!(AVX512_MIN_LEN >= 64);

#[cfg(any(test, feature = "bench-hooks"))]
thread_local! {
    static SWAR_FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Pin the SWAR tier on/off for the current thread.
///
/// The unit tests set `SWAR_FORCED` directly; `bench-hooks` exposes the same
/// switch so `benches/ascii_dispatch.rs` can compare the vector and scalar
/// paths inside one binary (wall-clock comparisons across binaries on this box
/// drift by more than the effect being measured).
#[cfg(any(test, feature = "bench-hooks"))]
pub(crate) fn set_swar_forced(on: bool) {
    SWAR_FORCED.with(|f| f.set(on));
}

#[inline]
fn swar_forced() -> bool {
    #[cfg(any(test, feature = "bench-hooks"))]
    {
        SWAR_FORCED.with(std::cell::Cell::get)
    }
    #[cfg(not(any(test, feature = "bench-hooks")))]
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
///
/// On x86_64/aarch64/wasm32 the `u128` SWAR body below is **unreachable in
/// production**: `len < 16` takes `classify_scalar` and `len >= 16` enters a
/// vector kernel, whose tail is resolved by the next-narrower kernel and finally
/// by `classify_scalar` — never by SWAR.  It is the main body only on targets
/// with no vector backend, and on vector targets it runs only under the
/// forced-SWAR test/bench tier.
#[inline]
#[must_use]
pub(crate) fn is_printable_ascii(s: &[u8]) -> bool {
    // Defined by the class: one pass, no separate kernel family.
    matches!(classify_ascii(s), AsciiClass::Printable)
}

/// `true` when every byte is printable ASCII **or** the two layout controls
/// `\t` (0x09) / `\n` (0x0A). This is the predicate
/// `sanitize_for_terminal_multiline` needs: it preserves tab and newline as
/// layout intent, replaces every other control (CR, C1, DEL) with U+FFFD.
///
/// The SIMD kernels above implement this same byte set.  On targets with a
/// vector backend the `u128` SWAR body below is reachable only under the
/// forced-SWAR test/bench tier — production tails are resolved by narrower
/// kernels and `classify_scalar`, never by SWAR.  Its
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
#[must_use]
pub(crate) fn is_printable_ascii_multiline(s: &[u8]) -> bool {
    // `!= Other` is the multiline set exactly: the fourth class is defined as
    // printable + tab + newline.
    !matches!(classify_ascii(s), AsciiClass::Other)
}

/// `true` when every byte is printable ASCII **or** the tab (`0x09`).
///
/// The gate for `render_line_merged_with`'s tab-aware byte-level fast path:
/// source code is routinely tab-indented, and the pure-printable gate
/// rejected tab (`0x09 < 0x20`), pushing every tab-indented row into the
/// grapheme walk.  The exact set is `0x20..=0x7E` plus `0x09`; the SIMD kernels
/// and the SWAR fallback both implement it.
#[inline]
#[must_use]
pub(crate) fn is_printable_ascii_or_tab(s: &[u8]) -> bool {
    // The two narrow classes are exactly the tab set.
    matches!(
        classify_ascii(s),
        AsciiClass::Printable | AsciiClass::PrintableOrTab
    )
}

/// Combine a kernel's accumulated tab/newline sightings with its tail's class.
///
/// The result is the **widest** accepted set: a newline anywhere outranks a tab,
/// which outranks plain printable, and any offender wins over all three.
///
/// This is exactly `max` under [`AsciiClass`]'s severity order
/// (`Printable < PrintableOrTab < PrintableOrTabNewline < Other`): the kernel's
/// own contribution is the widest class its sightings justify, and `Other` tops
/// the order, so an offending tail propagates without a special case (the old
/// hand-rolled version special-cased it).  Naming it as `max` is what makes the
/// `Ord` derive load-bearing, and it makes the empty-slice identity
/// (`class(∅) = Printable`) follow from the monoid rather than being a special
/// case.
#[inline]
fn compose_classify_tail(saw_tab: bool, saw_newline: bool, tail: AsciiClass) -> AsciiClass {
    let kernel = if saw_newline {
        AsciiClass::PrintableOrTabNewline
    } else if saw_tab {
        AsciiClass::PrintableOrTab
    } else {
        AsciiClass::Printable
    };
    kernel.max(tail)
}

/// Tri-state classification of one row, so callers that need both the
/// printable and printable-or-tab predicates (the glyph fast path tested them
/// sequentially) pay one SWAR pass instead of two.
///
/// The variant order is the severity order (`Printable` is the cleanest) and
/// is pinned by `predicates_form_an_implication_lattice`; `Ord`/`Hash` exist so
/// callers can sort or key on the class without a wrapper enum of their own.
///
/// `#[repr(u8)]` plus the `const` canary below make the order a compile-time
/// property: `compose_classify_tail` *is* `max`, so an IDE "sort variants"
/// refactor that permuted the severity order would otherwise fail only at
/// runtime (via the randomized max-equivalence test) instead of failing the
/// build.  The canonical discriminants also keep logs/debugging stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum AsciiClass {
    /// Every byte is printable ASCII (`0x20..=0x7E`).
    Printable,
    /// Printable ASCII plus horizontal tabs; no other control bytes.
    PrintableOrTab,
    /// Printable ASCII plus tabs and newlines — exactly the
    /// [`is_printable_ascii_multiline`] accepted set.
    PrintableOrTabNewline,
    /// Anything else (high byte, DEL, other C0/C1).
    Other,
}

// Compile-time canary for the severity order `compose_classify_tail` relies on
// (`kernel.max(tail)`).  `Ord` compares discriminants, so this is exactly the
// order the derive produces; permuting the variants now fails the build.
const _: () = {
    assert!((AsciiClass::Printable as u8) < (AsciiClass::PrintableOrTab as u8));
    assert!((AsciiClass::PrintableOrTab as u8) < (AsciiClass::PrintableOrTabNewline as u8));
    assert!((AsciiClass::PrintableOrTabNewline as u8) < (AsciiClass::Other as u8));
};

/// Scalar classification over the same byte set the vector and SWAR bodies
/// implement — printable plus tab plus newline: the widest accepted class.
#[inline]
fn classify_scalar(s: &[u8]) -> AsciiClass {
    let mut saw_tab = false;
    let mut saw_newline = false;
    for &b in s {
        // Common case first: a printable byte leaves in one fused range test,
        // where testing tab and newline first charged it two failed compares.
        // This runs per byte of every row shorter than 16 bytes (gutters,
        // separators, short diff context), so the order is worth stating.
        // The class is max-semantics, so check order cannot change the answer.
        if is_printable_byte(b) {
            continue;
        }
        if b == b'\t' {
            saw_tab = true;
        } else if b == b'\n' {
            saw_newline = true;
        } else {
            return AsciiClass::Other;
        }
    }
    if saw_newline {
        AsciiClass::PrintableOrTabNewline
    } else if saw_tab {
        AsciiClass::PrintableOrTab
    } else {
        AsciiClass::Printable
    }
}

/// Four-way classification of one row: the **widest** accepted set that
/// contains `s`.  The single primitive the three predicates are defined by:
///
/// - `is_printable_ascii(s)` ⟺ `class == Printable`
/// - `is_printable_ascii_or_tab(s)` ⟺ `class ∈ {Printable, PrintableOrTab}`
/// - `is_printable_ascii_multiline(s)` ⟺ `class != Other`
///
/// pinned by `classify_ascii_matches_predicates_on_random_bytes` and
/// `newline_is_the_fourth_class_and_survives_the_vector_widths`.
///
/// Formally `class(s)` is the `max`, over [`AsciiClass`]'s severity lattice, of
/// the per-byte contributions; `Printable` is the identity for concatenation, so
/// the empty slice's contribution is `Printable` — not a special case, and the
/// reason `is_printable_ascii(&[])` is `true`.
/// `classify_is_a_monoid_homomorphism` pins `class(a ++ b) = max(class(a), class(b))`.
///
/// One pass at the widest available vector width, SWAR for the tail.
///
/// # SWAR lane proof
///
/// The high-byte/DEL rejection is deliberately *not* an exact mask: it is
/// existential — any high bit in `x`, or a DEL lane in `x ^ 0x7F`, is itself
/// the proof, and a borrow chain can only originate at such a lane within the
/// same chunk.  The `< 0x20`, tab and newline masks **are** exact (each minuend
/// byte has its high bit forced with `x | HI`, so the word subtraction cannot
/// borrow between lanes):
///
/// * `lt20 = !((x|HI) - SP) & HI` — high bit clear iff `b + 0x60 < 0x80` iff
///   `b < 0x20`, exactly.
/// * `tab = !(((x ^ TAB)|HI) - ONES) & HI` — high bit clear iff `b ^ 0x09 == 0`
///   iff `b == 0x09`, exactly (the 0x08-neighbour trap of a `has_zero` value
///   mask cannot occur here because the minuend never borrows); `nl` likewise
///   with 0x0A.
///
/// High bytes are rejected first: the `(x | HI) - SP` form is only exact when
/// every lane is ≤ `0x7F` (otherwise `b + 0x80 - 0x20` overflows).
#[inline]
#[must_use]
pub(crate) fn classify_ascii(s: &[u8]) -> AsciiClass {
    // Tail protocol, stated exactly: the **SWAR body** resolves its `< 16`
    // remainder inline via `classify_scalar` and never re-enters the dispatcher.
    // A **vector kernel's** tail deliberately *does* re-enter it (see the AVX2
    // SAFETY note and the `AVX512_MIN_LEN` comment) — that is what lets every
    // tier handle any length with one loop plus a single downward delegation.
    // Recursion terminates because each tier's gate is strictly narrower than
    // its parent's.  (An earlier wording claimed kernels must not re-enter; it
    // contradicted the kernels and invited "fixing" them into decoding up to
    // 63-byte tails scalar-ly.)
    if s.len() < 16 {
        return classify_scalar(s);
    }
    #[cfg(target_arch = "x86_64")]
    let level = x86_level_for(s.len());
    #[cfg(target_arch = "x86_64")]
    if s.len() >= AVX512_MIN_LEN && level >= 3 {
        // SAFETY: `level >= 3` iff `is_x86_feature_detected!("avx512bw")`
        // succeeded inside `x86_level_init`; `avx512f` is implied by
        // `avx512bw` on every part implementing both, which is exactly what
        // the kernel's `#[target_feature(enable = "avx512f,avx512bw")]`
        // requires.
        return unsafe { x86_64::classify_ascii_avx512(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if s.len() >= 32 && level >= 2 {
        // SAFETY: `level >= 2` iff `is_x86_feature_detected!("avx2")`
        // succeeded; the kernel's 32-byte loads are AVX2-only.  The tail is
        // deferred back to this function, so a kernel never re-enters the
        // dispatcher for 16..31.
        return unsafe { x86_64::classify_ascii_avx2(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 16 {
        // SAFETY: SSE2 is baseline for every x86_64 target, so the 16-byte
        // intrinsics need no runtime detection.  `!swar_forced()` is the
        // test/bench tier switch; it lives on this arm because
        // `x86_level_for` returns 0 for `len < 32` without consulting it.
        return unsafe { x86_64::classify_ascii_sse2(s) };
    }
    #[cfg(target_arch = "aarch64")]
    if !swar_forced() && s.len() >= 16 {
        // SAFETY: NEON is baseline for every aarch64 target;
        // `!swar_forced()` is the test/bench tier switch.
        return unsafe { neon::classify_ascii(s) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if !swar_forced() && s.len() >= 16 {
        // SAFETY: `target_feature = "simd128"` is a compile-time gate; wasm
        // has no runtime feature detection, so the module (and this arm) only
        // exist when the binary is built for a SIMD-capable engine.  A
        // runtime-only gate would be UB on an engine without vectors.
        return unsafe { wasm::classify_ascii(s) };
    }
    const HI: u128 = 0x8080_8080_8080_8080_8080_8080_8080_8080;
    const SP: u128 = 0x2020_2020_2020_2020_2020_2020_2020_2020;
    const TAB: u128 = 0x0909_0909_0909_0909_0909_0909_0909_0909;
    const NL: u128 = 0x0A0A_0A0A_0A0A_0A0A_0A0A_0A0A_0A0A_0A0A;
    const DEL: u128 = 0x7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F_7F7F;
    const ONES: u128 = 0x0101_0101_0101_0101_0101_0101_0101_0101;

    let mut saw_tab = false;
    let mut saw_newline = false;
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
            let nl = !(((x ^ NL) | HI).wrapping_sub(ONES)) & HI;
            // A `< 0x20` lane that is neither tab nor newline is an offender.
            if (lt20 & !tab & !nl) != 0 {
                return AsciiClass::Other;
            }
            saw_tab |= tab != 0;
            saw_newline |= nl != 0;
        }
        i += 16;
    }
    // One implementation of the byte-set rule for the tail: `classify_scalar`
    // early-returns `Other` and is `#[inline]`, and `compose_classify_tail` is
    // `max`, so an offending tail propagates exactly as the hand-rolled loop
    // did.  A second copy of "printable ∪ {\t, \n}" here could drift (a future
    // edit admitting `\r` in one place only would be silent).
    compose_classify_tail(saw_tab, saw_newline, classify_scalar(&s[i..]))
}

/// Byte offset of the first byte ≥ 0x80 in `s`, or `usize::MAX` if `s` is
/// pure ASCII.
///
/// The sentinel form is deliberate: `Option<usize>` has no niche, so it costs
/// a 16-byte return slot (two GPRs / a stack slot under SysV) and a tag branch
/// at every call site.  This function is called inside `char_col`'s sparse
/// table build, and every kernel already produces a mask plus
/// `trailing_zeros`, so `usize::MAX` collapses the "not found" case to a
/// single GPR.  Use [`next_non_ascii`] off the hot path.
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
#[must_use]
pub(crate) fn next_non_ascii_byte(s: &[u8]) -> usize {
    // Same short-input reasoning as the predicate families: this is the *entry*
    // point (called per non-ASCII character from `char_col`'s sparse-table
    // build), so a short remaining run must not pay `x86_level_for` plus the
    // whole `cfg` chain to reach a `position()`.  `usize::MAX` is the sentinel
    // shared with every kernel tail.
    if s.len() < 16 {
        return s.iter().position(|&b| b >= 0x80).unwrap_or(usize::MAX);
    }
    #[cfg(target_arch = "x86_64")]
    let level = x86_level_for(s.len());
    #[cfg(target_arch = "x86_64")]
    if s.len() >= AVX512_MIN_LEN && level >= 3 {
        // SAFETY: `level >= 3` iff `is_x86_feature_detected!("avx512bw")`
        // succeeded inside `x86_level_init`; `avx512f` is implied on every
        // part implementing both.
        return unsafe { x86_64::next_non_ascii_byte_avx512(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if s.len() >= 32 && level >= 2 {
        // SAFETY: `level >= 2` iff `is_x86_feature_detected!("avx2")`
        // succeeded; the 32-byte loads are AVX2-only.  Tail deferred here.
        return unsafe { x86_64::next_non_ascii_byte_avx2(s) };
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 16 {
        // SAFETY: SSE2 is baseline on x86_64; `!swar_forced()` is the
        // test/bench tier switch, load-bearing because `x86_level_for`
        // returns 0 for `len < 32` without consulting it.
        return unsafe { x86_64::next_non_ascii_byte_sse2(s) };
    }
    #[cfg(target_arch = "aarch64")]
    if !swar_forced() && s.len() >= 16 {
        // SAFETY: NEON is baseline on aarch64; `!swar_forced()` is the
        // test/bench tier switch.
        return unsafe { neon::next_non_ascii_byte(s) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if !swar_forced() && s.len() >= 16 {
        // SAFETY: `target_feature = "simd128"` is a compile-time gate — wasm
        // has no runtime feature detection, so the module is only built for
        // an engine that guarantees vector support.
        return unsafe { wasm::next_non_ascii_byte(s) };
    }
    const HI: u128 = 0x8080_8080_8080_8080_8080_8080_8080_8080;

    let mut i = 0usize;
    while i + 16 <= s.len() {
        let x = u128::from_le_bytes(s[i..i + 16].try_into().unwrap());
        let hi = x & HI;
        if hi != 0 {
            // First flagged lane (little-endian ⇒ lowest byte index).
            return i + (hi.trailing_zeros() / 8) as usize;
        }
        i += 16;
    }
    const HI64: u64 = 0x8080_8080_8080_8080;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        let hi = x & HI64;
        if hi != 0 {
            return i + (hi.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    s[i..]
        .iter()
        .position(|&b| b >= 0x80)
        .map_or(usize::MAX, |off| i + off)
}

/// `Some(offset)` view of [`next_non_ascii_byte`] for callers that are not on
/// the byte-scanning hot path.
#[inline]
pub(crate) fn next_non_ascii(s: &[u8]) -> Option<usize> {
    let off = next_non_ascii_byte(s);
    if off == usize::MAX { None } else { Some(off) }
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
    pub(super) unsafe fn classify_ascii_avx2(s: &[u8]) -> super::AsciiClass {
        let sp = _mm256_set1_epi8(0x20);
        let del = _mm256_set1_epi8(0x7F);
        let tab_v = _mm256_set1_epi8(0x09);
        let nl_v = _mm256_set1_epi8(0x0A);
        let mut saw_tab = false;
        let mut saw_newline = false;
        let mut i = 0usize;
        while i + 64 <= s.len() {
            let a = unsafe { _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i) };
            let b = unsafe { _mm256_loadu_si256(s.as_ptr().add(i + 32) as *const __m256i) };
            let classify = |v: __m256i| unsafe {
                let lt = _mm256_cmpgt_epi8(sp, v);
                let tab = _mm256_cmpeq_epi8(v, tab_v);
                let nl = _mm256_cmpeq_epi8(v, nl_v);
                // Tab and newline are the two allowed layout controls; every
                // other `< 0x20` lane, and DEL, is an offender.
                let allowed_ctrl = _mm256_or_si256(tab, nl);
                let off = _mm256_andnot_si256(allowed_ctrl, lt);
                let eq = _mm256_cmpeq_epi8(v, del);
                (
                    _mm256_or_si256(eq, off),
                    _mm256_movemask_epi8(tab),
                    _mm256_movemask_epi8(nl),
                )
            };
            let (bad_a, tab_a, nl_a) = classify(a);
            let (bad_b, tab_b, nl_b) = classify(b);
            if unsafe { _mm256_movemask_epi8(_mm256_or_si256(bad_a, bad_b)) } != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= tab_a != 0 || tab_b != 0;
            saw_newline |= nl_a != 0 || nl_b != 0;
            i += 64;
        }
        while i + 32 <= s.len() {
            // SAFETY: `i + 32 <= len`, and the caller checked AVX2 support.
            let v = unsafe { _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i) };
            let lt = unsafe { _mm256_cmpgt_epi8(sp, v) };
            let tab = unsafe { _mm256_cmpeq_epi8(v, tab_v) };
            let nl = unsafe { _mm256_cmpeq_epi8(v, nl_v) };
            let off = unsafe { _mm256_andnot_si256(_mm256_or_si256(tab, nl), lt) };
            let eq = unsafe { _mm256_cmpeq_epi8(v, del) };
            let bad = unsafe { _mm256_or_si256(eq, off) };
            if unsafe { _mm256_movemask_epi8(bad) } != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { _mm256_movemask_epi8(tab) } != 0;
            saw_newline |= unsafe { _mm256_movemask_epi8(nl) } != 0;
            i += 32;
        }
        super::compose_classify_tail(saw_tab, saw_newline, super::classify_ascii(&s[i..]))
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn next_non_ascii_byte_avx2(s: &[u8]) -> usize {
        let mut i = 0usize;
        while i + 128 <= s.len() {
            let m0 = unsafe {
                _mm256_movemask_epi8(_mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i))
            };
            let m1 = unsafe {
                _mm256_movemask_epi8(_mm256_loadu_si256(s.as_ptr().add(i + 32) as *const __m256i))
            };
            let m2 = unsafe {
                _mm256_movemask_epi8(_mm256_loadu_si256(s.as_ptr().add(i + 64) as *const __m256i))
            };
            let m3 = unsafe {
                _mm256_movemask_epi8(_mm256_loadu_si256(s.as_ptr().add(i + 96) as *const __m256i))
            };
            if (m0 | m1 | m2 | m3) != 0 {
                for (off, mask) in [(0, m0), (32, m1), (64, m2), (96, m3)] {
                    if mask != 0 {
                        return i + off + mask.trailing_zeros() as usize;
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
                return i + mask.trailing_zeros() as usize;
            }
            i += 32;
        }
        let tail = super::next_non_ascii_byte(&s[i..]);
        if tail == usize::MAX {
            usize::MAX
        } else {
            i + tail
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn classify_ascii_avx512(s: &[u8]) -> super::AsciiClass {
        let sp = _mm512_set1_epi8(0x20);
        let del = _mm512_set1_epi8(0x7F);
        let tab_v = _mm512_set1_epi8(0x09);
        let nl_v = _mm512_set1_epi8(0x0A);
        let mut saw_tab = false;
        let mut saw_newline = false;
        let mut i = 0usize;
        while i + 256 <= s.len() {
            let class = |v: __m512i| unsafe {
                let lt = _mm512_cmpgt_epi8_mask(sp, v);
                let tab = _mm512_cmpeq_epi8_mask(v, tab_v);
                let nl = _mm512_cmpeq_epi8_mask(v, nl_v);
                let off = lt & !(tab | nl);
                let eq = _mm512_cmpeq_epi8_mask(v, del);
                (eq | off, tab, nl)
            };
            let (ba, ta, na) = class(unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) });
            let (bb, tb, nb) = class(unsafe { _mm512_loadu_si512(s.as_ptr().add(i + 64).cast()) });
            let (bc, tc, nc) = class(unsafe { _mm512_loadu_si512(s.as_ptr().add(i + 128).cast()) });
            let (bd, td, nd) = class(unsafe { _mm512_loadu_si512(s.as_ptr().add(i + 192).cast()) });
            if ba | bb | bc | bd != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= (ta | tb | tc | td) != 0;
            saw_newline |= (na | nb | nc | nd) != 0;
            i += 256;
        }
        while i + 64 <= s.len() {
            let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) };
            let lt = unsafe { _mm512_cmpgt_epi8_mask(sp, v) };
            let tab = unsafe { _mm512_cmpeq_epi8_mask(v, tab_v) };
            let nl = unsafe { _mm512_cmpeq_epi8_mask(v, nl_v) };
            let off = lt & !(tab | nl);
            let eq = unsafe { _mm512_cmpeq_epi8_mask(v, del) };
            if eq | off != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= tab != 0;
            saw_newline |= nl != 0;
            i += 64;
        }
        super::compose_classify_tail(saw_tab, saw_newline, super::classify_ascii(&s[i..]))
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn next_non_ascii_byte_avx512(s: &[u8]) -> usize {
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
                        return i + off + mask.trailing_zeros() as usize;
                    }
                }
            }
            i += 256;
        }
        while i + 64 <= s.len() {
            let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) };
            let mask = unsafe { _mm512_movepi8_mask(v) };
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += 64;
        }
        let tail = super::next_non_ascii_byte(&s[i..]);
        if tail == usize::MAX {
            usize::MAX
        } else {
            i + tail
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn classify_ascii_sse2(s: &[u8]) -> super::AsciiClass {
        let sp = _mm_set1_epi8(0x20);
        let del = _mm_set1_epi8(0x7F);
        let tab_v = _mm_set1_epi8(0x09);
        let nl_v = _mm_set1_epi8(0x0A);
        let mut saw_tab = false;
        let mut saw_newline = false;
        let mut i = 0usize;
        while i + 16 <= s.len() {
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i) };
            let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
            let tab = unsafe { _mm_cmpeq_epi8(v, tab_v) };
            let nl = unsafe { _mm_cmpeq_epi8(v, nl_v) };
            let off = unsafe { _mm_andnot_si128(_mm_or_si128(tab, nl), lt) };
            let eq = unsafe { _mm_cmpeq_epi8(v, del) };
            let bad = unsafe { _mm_or_si128(eq, off) };
            if unsafe { _mm_movemask_epi8(bad) } != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { _mm_movemask_epi8(tab) } != 0;
            saw_newline |= unsafe { _mm_movemask_epi8(nl) } != 0;
            i += 16;
        }
        super::compose_classify_tail(saw_tab, saw_newline, super::classify_ascii(&s[i..]))
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn next_non_ascii_byte_sse2(s: &[u8]) -> usize {
        let mut i = 0usize;
        while i + 64 <= s.len() {
            let m0 =
                unsafe { _mm_movemask_epi8(_mm_loadu_si128(s.as_ptr().add(i) as *const __m128i)) };
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
                        return i + off + mask.trailing_zeros() as usize;
                    }
                }
            }
            i += 64;
        }
        while i + 16 <= s.len() {
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i) };
            let mask = unsafe { _mm_movemask_epi8(v) };
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += 16;
        }
        let tail = super::next_non_ascii_byte(&s[i..]);
        if tail == usize::MAX {
            usize::MAX
        } else {
            i + tail
        }
    }
}

// `simd128` must be a *compile-time* feature: wasm has no runtime feature
// detection, so a `#[target_feature(enable = "simd128")]` function compiled
// into a module for a non-SIMD engine is either invalid or UB when invoked.
// Gating the module keeps the emitted wasm free of vector opcodes unless the
// build asked for them; the dispatcher arm carries the same `cfg`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;

    /// `a & !b` — the **opposite** argument order from x86's
    /// `_mm256_andnot_si256(a, b)`, which computes `!a & b`.  Both call sites
    /// are correct today, but the inverted convention between backends is a
    /// classic port-bug incubator, so it is documented (and pinned by
    /// `andnot_is_a_and_not_b`) instead of left to memory.
    #[inline]
    unsafe fn andnot(a: v128, b: v128) -> v128 {
        unsafe { v128_and(a, v128_not(b)) }
    }

    #[cfg(test)]
    mod tests {
        use std::arch::wasm32::*;

        use super::andnot;

        #[test]
        fn andnot_is_a_and_not_b() {
            let a = u8x16_splat(0xF0);
            let b = u8x16_splat(0x0F);
            assert_eq!(unsafe { u8x16_bitmask(andnot(a, b)) }, u16::MAX);
            assert_eq!(unsafe { u8x16_bitmask(andnot(b, a)) }, 0);
        }
    }

    #[inline]
    unsafe fn any(v: v128) -> bool {
        unsafe { u8x16_bitmask(v) != 0 }
    }

    #[target_feature(enable = "simd128")]
    pub(super) unsafe fn classify_ascii(s: &[u8]) -> super::AsciiClass {
        let sp = u8x16_splat(0x20);
        let del = u8x16_splat(0x7F);
        let tab = u8x16_splat(0x09);
        let nl = u8x16_splat(0x0A);
        let mut saw_tab = false;
        let mut saw_newline = false;
        let mut i = 0usize;
        while i + 32 <= s.len() {
            let a = unsafe { v128_load(s.as_ptr().add(i).cast()) };
            let b = unsafe { v128_load(s.as_ptr().add(i + 16).cast()) };
            let classify = |v: v128| unsafe {
                let tabm = u8x16_eq(v, tab);
                let nlm = u8x16_eq(v, nl);
                (
                    v128_or(
                        u8x16_eq(v, del),
                        andnot(i8x16_lt(v, sp), v128_or(tabm, nlm)),
                    ),
                    u8x16_bitmask(tabm),
                    u8x16_bitmask(nlm),
                )
            };
            let (ba, ta, na) = classify(a);
            let (bb, tb, nb) = classify(b);
            if unsafe { any(v128_or(ba, bb)) } {
                return super::AsciiClass::Other;
            }
            saw_tab |= ta != 0 || tb != 0;
            saw_newline |= na != 0 || nb != 0;
            i += 32;
        }
        while i + 16 <= s.len() {
            let v = unsafe { v128_load(s.as_ptr().add(i).cast()) };
            let tabm = unsafe { u8x16_eq(v, tab) };
            let nlm = unsafe { u8x16_eq(v, nl) };
            let bad = unsafe {
                v128_or(
                    u8x16_eq(v, del),
                    andnot(i8x16_lt(v, sp), v128_or(tabm, nlm)),
                )
            };
            if unsafe { any(bad) } {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { u8x16_bitmask(tabm) != 0 };
            saw_newline |= unsafe { u8x16_bitmask(nlm) != 0 };
            i += 16;
        }
        super::compose_classify_tail(saw_tab, saw_newline, super::classify_ascii(&s[i..]))
    }

    #[target_feature(enable = "simd128")]
    pub(super) unsafe fn next_non_ascii_byte(s: &[u8]) -> usize {
        let mut i = 0usize;
        while i + 64 <= s.len() {
            for off in [0usize, 16, 32, 48] {
                let v = unsafe { v128_load(s.as_ptr().add(i + off).cast()) };
                let bits = unsafe { u8x16_bitmask(v) };
                if bits != 0 {
                    return i + off + bits.trailing_zeros() as usize;
                }
            }
            i += 64;
        }
        while i + 16 <= s.len() {
            let v = unsafe { v128_load(s.as_ptr().add(i).cast()) };
            let bits = unsafe { u8x16_bitmask(v) };
            if bits != 0 {
                return i + bits.trailing_zeros() as usize;
            }
            i += 16;
        }
        let tail = super::next_non_ascii_byte(&s[i..]);
        if tail == usize::MAX {
            usize::MAX
        } else {
            i + tail
        }
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
        // The nibble-packing below assumes little-endian lane order: the lowest
        // *input* lane must land in the lowest nibble of the 64-bit view.  On a
        // big-endian aarch64 target (`aarch64_be-*`) that mapping flips and the
        // `trailing_zeros() / 4` lane index would be silently wrong, so fail the
        // build there instead of reporting bogus offsets.  Every other kernel in
        // this module is endianness-safe by construction (the u128 SWAR body
        // uses `from_le_bytes`; wasm is little-endian by spec).
        #[cfg(target_endian = "big")]
        compile_error!("ascii::neon lane extraction is unverified on big-endian aarch64");
        // `vcgeq_u8` yields 0xFF/0x00 per lane.  Reinterpreted as u16 lanes,
        // adjacent byte lanes form 0xFFFF / 0x00FF / 0xFF00 / 0x0000, and
        // `vshrn_n_u16(.., 4)` narrows each u16 to its top 4 bits per original
        // byte — so each output byte holds two nibbles, one per input byte.
        // Reading the 64-bit word little-endian puts the lowest input lane in
        // the lowest nibble, so `trailing_zeros() / 4` is the original lane
        // index.  Do not "simplify" the divisor to `/8`: that would be a byte
        // index into the packed word, not a lane index.
        let high = unsafe { vcgeq_u8(v, vdupq_n_u8(0x80)) };
        let packed = unsafe { vshrn_n_u16(vreinterpretq_u16_u8(high), 4) };
        let bits = unsafe { vget_lane_u64(vreinterpret_u64_u8(packed), 0) };
        (bits != 0).then_some((bits.trailing_zeros() / 4) as usize)
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn classify_ascii(s: &[u8]) -> super::AsciiClass {
        let sp = unsafe { vdupq_n_s8(0x20) };
        let del = unsafe { vdupq_n_u8(0x7F) };
        let tab_v = unsafe { vdupq_n_u8(0x09) };
        let nl_v = unsafe { vdupq_n_u8(0x0A) };
        let mut saw_tab = false;
        let mut saw_newline = false;
        let mut i = 0usize;
        while i + 32 <= s.len() {
            let a = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            let b = unsafe { vld1q_u8(s.as_ptr().add(i + 16)) };
            let classify = |v: uint8x16_t| unsafe {
                let lt = vcltq_s8(vreinterpretq_s8_u8(v), sp);
                let tab = vceqq_u8(v, tab_v);
                let nl = vceqq_u8(v, nl_v);
                let off = vbicq_u8(lt, vorrq_u8(tab, nl));
                (
                    vorrq_u8(vceqq_u8(v, del), off),
                    any_nonzero(tab),
                    any_nonzero(nl),
                )
            };
            let (bad_a, tab_a, nl_a) = classify(a);
            let (bad_b, tab_b, nl_b) = classify(b);
            if unsafe { any_nonzero(vorrq_u8(bad_a, bad_b)) } {
                return super::AsciiClass::Other;
            }
            saw_tab |= tab_a || tab_b;
            saw_newline |= nl_a || nl_b;
            i += 32;
        }
        while i + 16 <= s.len() {
            let v = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            let lt = unsafe { vcltq_s8(vreinterpretq_s8_u8(v), sp) };
            let tab = unsafe { vceqq_u8(v, tab_v) };
            let nl = unsafe { vceqq_u8(v, nl_v) };
            let off = unsafe { vbicq_u8(lt, vorrq_u8(tab, nl)) };
            let eq = unsafe { vceqq_u8(v, del) };
            if unsafe { any_nonzero(vorrq_u8(eq, off)) } {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { any_nonzero(tab) };
            saw_newline |= unsafe { any_nonzero(nl) };
            i += 16;
        }
        super::compose_classify_tail(saw_tab, saw_newline, super::classify_ascii(&s[i..]))
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn next_non_ascii_byte(s: &[u8]) -> usize {
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
                    return i + off * 16 + lane;
                }
            }
            i += 64;
        }
        while i + 16 <= s.len() {
            let v = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            if let Some(lane) = unsafe { first_high_lane(v) } {
                return i + lane;
            }
            i += 16;
        }
        let tail = super::next_non_ascii_byte(&s[i..]);
        if tail == usize::MAX {
            usize::MAX
        } else {
            i + tail
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    /// Test-side view of the sentinel API: the explicit import shadows the
    /// glob-imported `next_non_ascii_byte`, so every existing assertion keeps
    /// comparing against `Some(offset)`.
    use super::next_non_ascii as next_non_ascii_byte;
    use super::*;

    /// RAII guard for the test-only forced-SWAR switch.  A panic anywhere
    /// inside the guarded scope would otherwise leave the thread-local set,
    /// silently turning every later SWAR test on that thread into an SSE2 test
    /// (and hiding exactly the SWAR borrow-chain bugs those tests exist to
    /// catch).
    struct ForceSwarGuard;

    impl ForceSwarGuard {
        fn new() -> Self {
            SWAR_FORCED.with(|f| f.set(true));
            Self
        }
    }

    impl Drop for ForceSwarGuard {
        fn drop(&mut self) {
            SWAR_FORCED.with(|f| f.set(false));
        }
    }

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

    /// Cartesian over control/edge bytes at ADJACENT lanes.  The SWAR bodies
    /// reason about a carry out of lane *k* into lane *k+1* (see the DEL/~1
    /// borrow chain), and a single bad byte in a sea of `'a'` cannot exercise
    /// that interaction.  10² pairs × 15 lane offsets × 4 predicates is cheap
    /// and pins the borrow-propagation reasoning exhaustively.
    #[test]
    fn swar_adjacent_control_pairs_match_scalar() {
        const BYTES: [u8; 10] = [0x00, 0x01, 0x09, 0x0A, 0x1F, 0x20, 0x7E, 0x7F, 0x80, 0xFF];
        let _swar = ForceSwarGuard::new();
        for &a in &BYTES {
            for &b in &BYTES {
                for lane in 0..15 {
                    let mut v = [b'x'; 16];
                    v[lane] = a;
                    v[lane + 1] = b;
                    assert_eq!(
                        is_printable_ascii(&v),
                        ref_single(&v),
                        "single {a:#04x},{b:#04x} lane {lane}"
                    );
                    assert_eq!(
                        is_printable_ascii_or_tab(&v),
                        ref_tab(&v),
                        "tab {a:#04x},{b:#04x} lane {lane}"
                    );
                    assert_eq!(
                        is_printable_ascii_multiline(&v),
                        ref_multi(&v),
                        "multi {a:#04x},{b:#04x} lane {lane}"
                    );
                    assert_eq!(
                        classify_ascii(&v),
                        ref_classify(&v),
                        "class {a:#04x},{b:#04x} lane {lane}"
                    );
                }
            }
        }
    }
    fn ref_classify(s: &[u8]) -> AsciiClass {
        let mut tab = false;
        let mut newline = false;
        for &b in s {
            if b == b'\t' {
                tab = true;
            } else if b == b'\n' {
                newline = true;
            } else if !(0x20..=0x7E).contains(&b) {
                return AsciiClass::Other;
            }
        }
        if newline {
            AsciiClass::PrintableOrTabNewline
        } else if tab {
            AsciiClass::PrintableOrTab
        } else {
            AsciiClass::Printable
        }
    }

    /// Deterministic LCG: covers every byte value and every length 0..=40, so
    /// both the u128 SWAR body (len ≥ 16 under the forced tier) and the ≤15-byte
    /// scalar tail are exercised.  Forced, because on x86_64/aarch64 an unforced
    /// run would test SSE2/NEON and never reach the SWAR body this pins.
    #[test]
    fn swar_matches_scalar_on_random_slices() {
        let _swar = ForceSwarGuard::new();
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
    fn forced_swar_covers_all_u128_lanes_and_bytes() {
        let _swar = ForceSwarGuard::new();
        for lane in 0..16 {
            for value in 0..=255u8 {
                let mut v = [b'a'; 16];
                v[lane] = value;
                assert_eq!(
                    is_printable_ascii(&v),
                    ref_single(&v),
                    "single lane {lane} byte {value:#04x}"
                );
                assert_eq!(
                    is_printable_ascii_multiline(&v),
                    ref_multi(&v),
                    "multi lane {lane} byte {value:#04x}"
                );
                assert_eq!(
                    is_printable_ascii_or_tab(&v),
                    ref_tab(&v),
                    "tab lane {lane} byte {value:#04x}"
                );
                assert_eq!(
                    classify_ascii(&v),
                    ref_classify(&v),
                    "class lane {lane} byte {value:#04x}"
                );
            }
        }
    }

    #[test]
    fn classify_ascii_matches_predicates_on_random_bytes() {
        let mut seed = 0xA5A5_1234_9ABC_DEF0u64;
        for _ in 0..5000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(11);
            let len = (seed % 97) as usize;
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(11);
                    (seed & 0xFF) as u8
                })
                .collect();
            let class = classify_ascii(&bytes);
            let single = is_printable_ascii(&bytes);
            let tab = is_printable_ascii_or_tab(&bytes);
            let multi = is_printable_ascii_multiline(&bytes);
            // Four classes = the widest accepted set, so each predicate is an
            // exact test of the class: biconditionals, not implications.
            assert_eq!(class == AsciiClass::Printable, single, "{bytes:?}");
            assert_eq!(
                class == AsciiClass::PrintableOrTab,
                tab && !single,
                "{bytes:?}"
            );
            assert_eq!(
                class == AsciiClass::PrintableOrTabNewline,
                multi && !tab,
                "{bytes:?}"
            );
            assert_eq!(class == AsciiClass::Other, !multi, "{bytes:?}");
        }
    }

    #[test]
    fn newline_is_the_fourth_class_and_survives_the_vector_widths() {
        assert_eq!(classify_ascii(b"abc def"), AsciiClass::Printable);
        assert_eq!(classify_ascii(b"a\tb"), AsciiClass::PrintableOrTab);
        assert_eq!(classify_ascii(b"a\nb"), AsciiClass::PrintableOrTabNewline);
        assert_eq!(
            classify_ascii(b"a\tb\nc"),
            AsciiClass::PrintableOrTabNewline
        );
        // Long enough to reach the SWAR body (and any enabled SIMD kernel), so
        // the fourth class is not a scalar-only accident.
        let tabbed = format!("{}\t{}", "x".repeat(40), "y".repeat(40));
        assert_eq!(
            classify_ascii(tabbed.as_bytes()),
            AsciiClass::PrintableOrTab
        );
        let multiline = format!("{}\n{}", "x".repeat(100), "y".repeat(100));
        assert_eq!(
            classify_ascii(multiline.as_bytes()),
            AsciiClass::PrintableOrTabNewline
        );
        // An offender always outranks the newline class.
        assert_eq!(classify_ascii(b"a\n\x01b"), AsciiClass::Other);
        let bad = format!("{}\u{0080}", "x".repeat(100));
        assert_eq!(classify_ascii(bad.as_bytes()), AsciiClass::Other);
        // Every predicate is exactly a test of the class, including the
        // newline cases the old three-class contract could not express.
        for s in [
            &b""[..],
            b"abc def",
            b"a\tb",
            b"a\nb",
            b"a\tb\nc",
            b"a\n\x01b",
            b"\x7f",
            "\u{00e9}".as_bytes(),
        ] {
            let class = classify_ascii(s);
            assert_eq!(is_printable_ascii(s), class == AsciiClass::Printable);
            assert_eq!(
                is_printable_ascii_or_tab(s),
                matches!(class, AsciiClass::Printable | AsciiClass::PrintableOrTab),
                "{s:?}"
            );
            assert_eq!(
                is_printable_ascii_multiline(s),
                class != AsciiClass::Other,
                "{s:?}"
            );
        }
    }

    #[test]
    fn next_non_ascii_covers_all_offsets_through_256() {
        for at in 0..=256 {
            let mut v = vec![b'a'; 320];
            v[at] = 0xC3;
            assert_eq!(next_non_ascii_byte(&v), Some(at), "offset {at}");
        }
    }

    #[test]
    fn swar_del_then_tilde_borrow_chain() {
        let _swar = ForceSwarGuard::new();
        for lane in 0..16 {
            if lane + 1 >= 16 {
                continue;
            }
            let mut v = [b'a'; 16];
            v[lane] = 0x7F;
            v[lane + 1] = 0x7E;
            assert!(!is_printable_ascii(&v), "DEL lane {lane}");
            assert!(!is_printable_ascii_or_tab(&v), "DEL lane {lane}");
            assert!(!is_printable_ascii_multiline(&v), "DEL lane {lane}");
            assert_eq!(classify_ascii(&v), AsciiClass::Other, "DEL lane {lane}");
        }
        let mut ok = [b'a'; 16];
        ok[0] = b' ';
        ok[1] = b'~';
        assert!(is_printable_ascii(&ok));
    }

    /// The u64 SWAR tail after the u128 loop has its own
    /// `trailing_zeros() / 8` arithmetic; the random-length tests only reach it
    /// statistically, so pin it explicitly.  Calls the production
    /// `next_non_ascii_byte` (not the test-local `Option` alias) so the
    /// `usize::MAX` sentinel path is exercised.
    #[test]
    fn forced_swar_u64_tail_matches_scalar() {
        let _swar = ForceSwarGuard::new();
        // 24 bytes = one u128 chunk (0..16) + one u64 chunk (16..24).
        for lane in 0..8 {
            let mut v = [b'a'; 24];
            v[16 + lane] = 0x80;
            assert_eq!(super::next_non_ascii_byte(&v), 16 + lane, "lane {lane}");
        }
        // A high byte in the u128 chunk wins over the tail one.
        let mut v = [b'a'; 24];
        v[3] = 0x80;
        v[20] = 0x80;
        assert_eq!(super::next_non_ascii_byte(&v), 3);
        // Last byte of the u64 chunk still lands in range.
        let mut v = [b'a'; 24];
        v[23] = 0xFF;
        assert_eq!(super::next_non_ascii_byte(&v), 23);
    }

    /// A kernel that consumes the input exactly leaves an empty tail; the
    /// dispatcher must handle that without falling back to a bogus scan.
    #[test]
    fn exact_kernel_widths_leave_a_clean_tail() {
        for len in [16usize, 32, 64, 128, 256, 512] {
            let v = vec![b'a'; len];
            assert_eq!(classify_ascii(&v), AsciiClass::Printable, "len {len}");
            assert_eq!(super::next_non_ascii_byte(&v), usize::MAX, "len {len}");
            let mut w = v.clone();
            w[len - 1] = 0x80;
            assert_eq!(classify_ascii(&w), AsciiClass::Other, "len {len}");
            assert_eq!(super::next_non_ascii_byte(&w), len - 1, "len {len}");
        }
    }

    /// The documented "may return a continuation byte" contract must hold at
    /// every dispatch tier, not only at the scalar entry: the offset arithmetic
    /// is UTF-8-agnostic, so a mid-sequence start must still report the first
    /// high byte.
    ///
    /// The emoji sits at the *start* of a long buffer and the slice keeps the
    /// tail (`len - skip` bytes, ≥ 13), so each case enters a real vector kernel
    /// — placing it last would leave a 1..3-byte slice, which always takes the
    /// scalar short-circuit and would make the name a lie.
    #[test]
    fn next_non_ascii_from_mid_sequence_holds_across_tiers() {
        for &len in &[16usize, 32, 64, 128, 256, 512] {
            let mut v = vec![b'a'; len];
            v[..4].copy_from_slice("😀".as_bytes()); // F0 9F 98 80
            for skip in 1..4 {
                assert_eq!(
                    super::next_non_ascii_byte(&v[skip..]),
                    0,
                    "len {len} skip {skip}"
                );
            }
            // Sanity: the unsliced buffer reports the lead byte.
            assert_eq!(super::next_non_ascii_byte(&v), 0, "len {len}");
            // And a buffer with no high byte reports the sentinel.
            let ascii = vec![b'a'; len];
            assert_eq!(super::next_non_ascii_byte(&ascii), usize::MAX, "len {len}");
        }
    }

    #[test]
    fn next_non_ascii_covers_multi_block_offsets() {
        for at in [8, 15, 16, 31, 32, 127, 128, 255, 256, 383, 512, 600, 1023] {
            let mut v = vec![b'a'; 1088];
            v[at] = 0x80;
            assert_eq!(next_non_ascii_byte(&v), Some(at), "offset {at}");
        }
    }

    #[test]
    fn long_inputs_cover_unrolled_kernels() {
        let mut long = vec![b'a'; 700];
        assert!(is_printable_ascii(&long));
        assert!(is_printable_ascii_or_tab(&long));
        assert!(is_printable_ascii_multiline(&long));
        assert_eq!(classify_ascii(&long), AsciiClass::Printable);

        long[255] = b'\t';
        assert!(!is_printable_ascii(&long));
        assert!(is_printable_ascii_or_tab(&long));
        assert!(is_printable_ascii_multiline(&long));
        assert_eq!(classify_ascii(&long), AsciiClass::PrintableOrTab);

        long[255] = b'a';
        long[600] = 0x7F;
        assert!(!is_printable_ascii(&long));
        assert!(!is_printable_ascii_or_tab(&long));
        assert!(!is_printable_ascii_multiline(&long));
        assert_eq!(classify_ascii(&long), AsciiClass::Other);
    }

    #[test]
    fn forced_swar_matches_scalar_for_all_predicates() {
        let _swar = ForceSwarGuard::new();
        let ref_classify = |s: &[u8]| {
            let mut tab = false;
            let mut newline = false;
            for &b in s {
                if b == b'\t' {
                    tab = true;
                } else if b == b'\n' {
                    newline = true;
                } else if !(0x20..=0x7E).contains(&b) {
                    return AsciiClass::Other;
                }
            }
            if newline {
                AsciiClass::PrintableOrTabNewline
            } else if tab {
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
    }

    /// The multiline predicate is existentially "no control other than
    /// `\t`/`\n`".  Random slices skewed towards `\t`/`\n` (the shape of real
    /// multiline text, where *every* 8-byte chunk trips the `< 0x20` SWAR
    /// mask) must agree with the scalar reference — this pins the
    /// flagged-lane fallback against both borrow-propagation false positives
    /// and tab/newline-dense chunks.
    #[test]
    fn multiline_matches_exists_reference_on_tab_newline_heavy_slices() {
        // Forced so the tab/newline `movemask` tracking and `or_flagged` path
        // under test are the SWAR body's, not a vector kernel's.
        let _swar = ForceSwarGuard::new();
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

    /// Every byte value at every position within a u128 chunk: pins the SWAR
    /// borrow-propagation corner (a non-printable byte's borrow must not corrupt
    /// a neighbouring printable lane's result).  Sixteen bytes is exactly one
    /// chunk and the smallest input that reaches the SWAR body — an 8-byte
    /// buffer would take the `len < 16` scalar short-circuit and pin nothing.
    #[test]
    fn swar_covers_every_byte_value_in_every_lane() {
        let _swar = ForceSwarGuard::new();
        for b in 0u8..=0xFF {
            for lane in 0..16 {
                let mut v = [b'a'; 16];
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
        // Forced: these 16-byte buffers are exactly one u128 chunk, so the
        // borrow shapes below only exist in the SWAR body.
        let _swar = ForceSwarGuard::new();
        for lane in 0..15 {
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
    /// not reject it.  Forced, and every case is ≥ 16 bytes: a shorter buffer is
    /// a scalar short-circuit (no borrows at all), so the hazard this pins would
    /// be untested.
    #[test]
    fn tab_predicate_tolerates_borrow_false_positive() {
        let _swar = ForceSwarGuard::new();
        assert!(is_printable_ascii_or_tab(b"\t abcdefghijklmno"));
        assert!(is_printable_ascii_or_tab(b"\t\t\t  \t  xabcdefghi"));
        assert!(is_printable_ascii_or_tab(b"a\t b\t c\t defghijkl"));
        assert!(is_printable_ascii_or_tab(b"\t abcdefghijklm"));
        // A genuine non-tab control followed by a space must still reject.
        assert!(!is_printable_ascii_or_tab(b"\t\x07 abcdefghijklmn"));
        assert!(!is_printable_ascii_or_tab(b"\x01 abcdefghijklmn"));
        assert!(!is_printable_ascii_or_tab(b"\t abcde\x7Ffghijkl"));
    }

    /// The OR-form's borrow can carry out of the top lane of one chunk into the
    /// next.  Pin that an offender in the **last** lane of a chunk (byte 15 of
    /// the first u128, byte 7 of the u64 tail) is rejected for every possible
    /// next byte, and that the next chunk is not falsely rejected when the
    /// borrow is discarded.
    #[test]
    fn or_form_lane7_borrow_out_covers_every_next_byte() {
        let _swar = ForceSwarGuard::new();
        for b in 0u8..=255 {
            // Offender in the top lane of the first u128 chunk.
            let mut buf = [b'a'; 32];
            buf[15] = 0x07;
            buf[16] = b;
            assert!(
                !is_printable_ascii(&buf),
                "chunk-top offender not rejected with next byte {b:#04x}"
            );
            // A clean chunk whose *next* chunk starts with the same byte must
            // only reject when that byte is itself an offender: the discarded
            // borrow-out cannot leak into lane 0.
            let mut buf2 = [b'a'; 32];
            buf2[16] = b;
            assert_eq!(
                is_printable_ascii(&buf2),
                ref_single(&buf2),
                "borrow-out leaked into the next chunk for {b:#04x}"
            );
            // And the same shape at the u128/u64 seam used by
            // `next_non_ascii_byte` (byte 16 starts the u64 chunk).
            let mut buf3 = [b'a'; 24];
            buf3[16] = b;
            assert_eq!(
                next_non_ascii(&buf3),
                buf3.iter().position(|&x| x >= 0x80),
                "seam {b:#04x}"
            );
        }
    }

    /// `class(s)` is a monoid homomorphism onto the severity lattice:
    /// `class(a ++ b) == max(class(a), class(b))`, with `Printable` the identity
    /// for the empty slice.  One property pins four contracts: the derived
    /// `Ord` order (load-bearing via `compose_classify_tail`'s `max`), the tail
    /// composition, the empty-slice identity, and the chunk-independence claim.
    #[test]
    fn classify_is_a_monoid_homomorphism() {
        let empty = classify_ascii(&[]);
        assert_eq!(empty, AsciiClass::Printable, "empty slice must be identity");
        let mut seed: u64 = 0x0BAD_F00D_1234_5678;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(11);
            seed
        };
        let alphabet = [b'a', b'\t', b'\n', b'\x01', 0x7F, 0x80, b' '];
        // Chunk-boundary lengths (0, 15, 16, 17, 31, 32, 33) matter most: that
        // is where a kernel hands its tail to the next tier.
        for &(left, right) in &[
            (0usize, 0usize),
            (0, 16),
            (16, 0),
            (15, 1),
            (1, 15),
            (16, 17),
            (17, 16),
            (31, 33),
            (33, 31),
        ] {
            for _ in 0..64 {
                let a: Vec<u8> = (0..left).map(|_| alphabet[(next() as usize) % 7]).collect();
                let b: Vec<u8> = (0..right)
                    .map(|_| alphabet[(next() as usize) % 7])
                    .collect();
                let mut ab = a.clone();
                ab.extend_from_slice(&b);
                assert_eq!(
                    classify_ascii(&ab),
                    classify_ascii(&a).max(classify_ascii(&b)),
                    "a={a:?} b={b:?}"
                );
            }
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
