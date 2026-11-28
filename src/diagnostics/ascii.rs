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
static LEVEL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

// Every write to `LEVEL` is the same word (`x86_level_init` is idempotent, and
// the only value besides `0` is the detected level), so the `Relaxed` **store**
// race is benign by construction rather than by luck: racing stores write the
// same byte, and a reader sees either `0` (and re-runs `init`) or the detected
// level.  There is no read-modify-write on this cache line.  Do not "fix" this with a stronger ordering —
// it buys nothing and would cost the hot path a fence.  What *is* load-bearing is
// the ordering inside `x86_level` (`swar_forced()` before the cache read); see
// the comment there.

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
    // a forced-SWAR test straight into AVX2/AVX-512 (the SSE2 arms therefore
    // carry their own `!swar_forced()` conjunct, and the `16..31` band never
    // consults the level at all — its arm is length-gated below 32).  The TLS
    // read this costs exists only under `test`/`bench-hooks`; in production
    // `swar_forced()` folds to `false` and the cache load dominates either way.
    if swar_forced() {
        return 0;
    }
    #[cfg(any(test, feature = "bench-hooks"))]
    {
        let want = LEVEL_OVERRIDE.with(std::cell::Cell::get);
        if want != 0 {
            // Clamped to what the host actually has — see the doc on
            // `set_x86_level_override`.  `min` keeps a 3 request on an AVX2-only
            // box at 2 instead of executing an illegal instruction.
            //
            // Reads the **cache**, not `x86_level_detected()`: this arm was that
            // function's only caller, so any process using the override re-probed
            // CPU features on every predicate call while the production path went
            // through `x86_level_init`.  Two paths to one probe is the drift shape
            // this module documents everywhere else; `init` is idempotent, so the
            // not-yet-populated case simply fills the cache.
            let cached = LEVEL.load(std::sync::atomic::Ordering::Relaxed);
            let detected = if cached != 0 {
                cached
            } else {
                x86_level_init()
            };
            return want.min(detected);
        }
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
    // `avx512f` is probed, not assumed.  `gate_band`'s SAFETY comment already claims
    // "the init probes it explicitly" — and a comment that carries a safety argument
    // has to be literally true, by this module's own standard.  It is also the exact
    // shape of the `has_avx2` probe above: a feature used by the kernels must be
    // tested, not deduced from a sibling bit.
    let has_avx512f = std::arch::is_x86_feature_detected!("avx512f");
    let level = if std::arch::is_x86_feature_detected!("avx512bw") && has_avx512f && has_avx2 {
        3
    } else if has_avx2 {
        2
    } else {
        1
    };
    LEVEL.store(level, std::sync::atomic::Ordering::Relaxed);
    level
}

/// Vector-tier thresholds, **measured** with
/// `cargo bench --bench ascii_dispatch`, comparing the `predicates/` (dispatch)
/// and `swar_forced/` (SWAR body pinned) tiers in the same run.
///
/// The two predicates cross over in opposite directions, so they get their own
/// knobs:
///
/// `classify_ascii` (dispatch vs SWAR, ns):
///
/// | len | vector tier | SWAR  |
/// |-----|-------------|-------|
/// | 16  |  6.87       |  7.15 |
/// | 31  | 16.20       | 12.29 |
/// | 32  | 13.24       |  9.03 |
/// | 63  | 22.74       | 16.32 |
/// | 64  | 13.36       | 12.76 |
/// | 127 | 24.59       | 23.77 |
/// | 128 | 14.71       | 20.41 |
/// | 255 | 27.74       | 38.75 |
///
/// **These numbers predate two changes and must be re-measured before any
/// threshold moves**: the `8..=15` reference now shares the `u64` SWAR step
/// (`classify_word8`) instead of `classify_scalar`, which removes the scalar
/// remainder the `/31`, `/63`, `/127` rows include, and the bounded-classification
/// early exit is a new branch per chunk.  The `/63` note below is a leftover from
/// when AVX2 started at 64.
///
/// What the table did establish, and what the dispatch still encodes: a 32-byte
/// AVX2 step saves little over SSE2's 16, so `classify` uses SSE2 from
/// [`CLASSIFY_SSE2_MIN_LEN`] and the wide tiers from
/// [`CLASSIFY_AVX2_MIN_LEN`] / [`CLASSIFY_AVX512_MIN_LEN`].
///
/// Re-measured 2026-11-27 (`cargo bench --bench ascii_dispatch -- offline`,
/// AVX-512 host, criterion `--save-baseline`/`--baseline` A/B minutes apart so
/// the two configurations share the host's noise):
///
/// | case | 128 floor | 64 floor | change | role |
/// |------|-----------|----------|--------|------|
/// | 127 | 14.98 ns | 6.06 ns | **−59.5%** (p=0.00) | treatment |
/// | 128 | 6.51 ns | 6.53 ns | +0.7% (p=0.44) | control |
/// | 129 | 8.31 ns | 8.31 ns | −6.8% (p=0.00) | control drift |
/// | 64 | 7.11 ns | 5.37 ns | −24% | treatment |
/// | 65 | 8.89 ns | 7.64 ns | −14% (p=0.22) | below the noise floor |
///
/// The control group (128/129, which take the AVX-512 tier under both floors)
/// moves by at most ~7%, and 63 — untouched by the change — drifted 20% between
/// runs.  So the `/127` −59.5% is the signal and the floor is now **64**.
///
/// Caveat, stated rather than hidden: this host is a 4-core WSL2 box and its
/// per-run spread is ~±20%, which is why the `65` row is not significant on its
/// own and why the decision rests on the two large, direction-consistent rows.
/// The same-run `/127` vs `/128` gap was 2.30× in two independent runs before the
/// flip — the confirmation that the two tiers really do differ at this size.
///
/// The gate kernels (`predicates/printable/*`) were re-measured in the same
/// session, and lowering the classify floor exposed a real defect in them: both
/// ended with a **scalar** walk of the 1..=63-byte remainder, which costs ~0.4 ns
/// per byte.  With the floor at 128 that was rare; with it at 64 it made
/// `/127` 30.17 ns against `/128`'s 2.75 ns.  Both now end with a masked overlap
/// load, and the band collapsed:
///
/// | case | scalar tail | masked tail |
/// |------|-------------|-------------|
/// | 31 | ~7.3 ns | **1.91 ns** |
/// | 32 | ~7.7 ns | **1.49 ns** |
/// | 33 | ~8.6 ns | **2.19 ns** |
/// | 63 | 6.92 ns | **2.52 ns** |
/// | 127 | 30.17 ns | **2.88 ns** |
/// | 128 | 2.75 ns | 2.78 ns (control) |
///
/// (`31/32/33` before-columns are derived from criterion's own `change` column,
/// which is why they are approximate; the after-column is the recorded median.)
///
/// # Tail protocol
///
/// Every **x86** kernel — `classify` and `next_non_ascii_byte`, all three tiers —
/// ends with one overlapping load of the final `MIN_CHUNK` bytes, masked down to
/// the lanes it has not scanned.  A kernel never starts below its own threshold
/// (pinned by the `const _` canaries below), which is what keeps
/// `base = len - MIN_CHUNK` in range.
///
/// NEON and wasm cover **both** remainders the same way, via the portable
/// [`tail_overlap_16`]: `next_non_ascii_byte` (bitmask domain) and `classify` (neon masks in
/// the vector domain, wasm in the bitmask domain).  So no backend re-enters the dispatcher
/// for a tail any more.  (An earlier revision of this paragraph said their `classify`
/// kernels still delegated 1..15 bytes to [`classify_ascii_bounded`] and paid the ~1.5 ns/byte
/// re-dispatch tax — true when written, false once the tails landed, and exactly the kind of
/// drift this module's docs are not allowed to have.)
///
/// # Rejected / measurement-gated work, with its blocker
///
/// * **NEON 32-byte group: four `vmaxvq_u8` reductions → two** (OR the tab masks and the
///   NL masks pairwise, then one reduction each, mirroring the already-OR-merged `bad`
///   path).  Free in principle, but aarch64 code cannot be compiled or run on the x86 dev
///   host, so it would ship unverified vector code — the one thing this module's test
///   discipline forbids.  Do it inside the `check-aarch64` / `test-aarch64` (qemu) loop.
/// * **Wide-band gate on AVX2-less hosts** (serve `len >= 128` boolean predicates from
///   `gate_sse2` instead of the fused classify path): the honest answer is the measure-then-
///   decide the doc already asks for — A/B `gate_sse2` against `classify_ascii_avx2` at
///   128..512 on a level-2 host.  Not done because this host is level 3, so any number here
///   would be a guess.
/// * **Fused `(AsciiClass, first-offender offset)` API**: every kernel already computes the
///   violation mask, so `trailing_zeros` on it *is* `next_non_ascii_byte`'s answer for the
///   sanitize-then-locate callers.  Needs a new API, tail-protocol plumbing across five
///   backends and a caller-pattern measurement; the largest remaining win in this file if the
///   sequential classify-then-locate pattern is real.
/// * **Resumable NNA cursor for `char_col`**: the sparse-table build calls
///   `next_non_ascii_byte` per non-ASCII character, each call re-paying the length-gate chain.
///   A `{ptr, remaining, level}` cursor removes that; bounded API change, bench-gated on
///   mixed-script input.
/// * **`u32` SWAR step for 4..7-byte inputs, and the single-compare strict gate**: both
///   measured to no win (the unsigned-compare emulation spends exactly the ops the
///   `lt | eq_del` form does), and the `u32` step would add a third width-copy of the byte-set
///   rule, against the one-copy discipline `classify_word8` exists to enforce.
///
/// An earlier revision of this doc described an "inline SSE2 step, then hand the
/// remainder to the SWAR short path" as a live safety net for all NNA kernels.
/// That step no longer exists in any of them, and the NEON/wasm backends did not
/// have a masked tail at all until `tail_overlap_16` (their tail *was* the
/// re-dispatch), so do not read any canary below as protecting it.
///
/// `next_non_ascii_byte` measures the other way — it returns early, so the
/// kernel's fixed cost dominates and the wide loads pay for themselves: 32 B
/// 5.30 ns (AVX2), 63 B 17.12, 64 B 5.61 (AVX-512), 128 B 6.84.  It therefore
/// keeps AVX2 from 32 and AVX-512 from 64.
///
/// No correctness argument depends on any of these values; they are the knobs.
const CLASSIFY_SSE2_MIN_LEN: usize = 16;
const CLASSIFY_AVX2_MIN_LEN: usize = 128;
// Measured, not inherited: at `/127` the SSE2 path costs 19.19 ns against
// `/128`'s 8.33 ns on the AVX-512 host (`cargo bench --bench ascii_dispatch --
// "classify/printable"`), i.e. the same work is ~2.3x cheaper one byte later.
// The old `/63`-era floor of 128 dated from when AVX-512 only had a 64-byte
// loop; with the masked tail it is correct from 64 up (the `const _` canary
// below is what makes that safe), so the floor follows the measurement.
const CLASSIFY_AVX512_MIN_LEN: usize = 64;
const NNA_AVX2_MIN_LEN: usize = 32;
const NNA_AVX512_MIN_LEN: usize = 64;
/// Upper bound of the SSE2 gate band (`gate_band` serves `16..128`).
///
/// Named separately on purpose: the value *equals* the classify AVX2 floor today, but it is the
/// gate's handoff point, not the classifier's.  Re-measuring `CLASSIFY_AVX2_MIN_LEN` (which the
/// module's own threshold table says is due) must not silently move the gate band with it.
const GATE_SSE2_MAX_LEN: usize = CLASSIFY_AVX2_MIN_LEN;

/// Length floor of the 16-byte NNA kernels (SSE2/NEON/wasm), named for symmetry with
/// [`CLASSIFY_SSE2_MIN_LEN`] so the floor has one home and one canary.
const NNA_SSE2_MIN_LEN: usize = 16;

// Structural, not a debug assert: the AVX-512 kernel's inner loop is one 64-byte
// chunk and its tail re-enters the dispatcher, so a threshold below 64 would
// re-dispatch at the same length and recurse until the stack dies.  In release
// that would be a production stack overflow; here it is a build error.
const _: () = assert!(CLASSIFY_AVX512_MIN_LEN >= 64);
const _: () = assert!(NNA_AVX512_MIN_LEN >= 64);
// Same hazard one tier down: `classify_ascii_avx2`'s masked tail is guarded by
// `s.len() >= 32`, so a threshold below 32 would re-enter the dispatcher at the
// same length and recurse in release.  `NNA_AVX2_MIN_LEN` is pinned for the same
// reason: its sub-32 termination now depends on the 32-byte overlap tail, whose
// `base = len - 32` underflows below the threshold — a release UB only the
// in-kernel `debug_assert` would notice.
const _: () = assert!(CLASSIFY_AVX2_MIN_LEN >= 32);
const _: () = assert!(NNA_AVX2_MIN_LEN >= 32);
/// The classify dispatcher nests its AVX2 arm inside an AVX-512-length gate; that is only sound
/// while the AVX2 floor is the *larger* one (no AVX2 band hides behind the outer gate).  NNA has
/// the relation reversed, so its outer gate is the AVX2 floor — see `next_non_ascii_byte`.
const _: () = assert!(CLASSIFY_AVX2_MIN_LEN >= CLASSIFY_AVX512_MIN_LEN);
const _: () = assert!(CLASSIFY_SSE2_MIN_LEN >= 16);

#[cfg(any(test, feature = "bench-hooks"))]
thread_local! {
    static SWAR_FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Requested vector tier for the current thread; `0` means "detected".
    static LEVEL_OVERRIDE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Pin the vector tier for the current thread (`0` restores detection).
///
/// Why this exists: on an AVX-512 host the dispatch never reaches
/// `classify_ascii_avx2` (128+ goes to AVX-512, 16..127 to SSE2), so that kernel
/// was only tested when CI happened to run on an AVX2-only machine.  The override
/// is **clamped to the detected level** inside `x86_level`: forcing tier 3 on a
/// non-AVX-512 host would execute kernels without their `#[target_feature]`
/// precondition, i.e. exactly the UB the SAFETY comments guard against.
#[cfg(any(test, feature = "bench-hooks"))]
pub(crate) fn set_x86_level_override(level: u8) {
    LEVEL_OVERRIDE.with(|f| f.set(level));
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
/// One gate, three consumers, and **no re-entry**: the widest available vector gate kernel serves
/// the whole input end-to-end (masked overlap tail included), and the `u128` SWAR body below
/// serves the whole input on targets with no vector backend — or under the forced-SWAR test tier.
/// Nothing hands a tail back to this entry.  (This said "the remainder re-enters this entry, which
/// re-selects a narrower tier and reaches the `u128` SWAR body", which described the pre-tail
/// design.)
///
/// The lane proof is width-independent and lives in exactly one place — `classify_swar_wide`'s
/// "SWAR lane proof (canonical)".  This doc carried a second copy of it (`# Chunk independence`:
/// `wrapping_sub` discards the top-lane borrow-out, so lane 0 of every chunk enters with
/// `borrow_in = 0`), which is precisely the drift hazard that proof's own doctrine names: an edit
/// admitting a byte in one copy would pass every test that reads only the other.  A virtual
/// elimination: this predicate never executes the SWAR body on a vector target.
/// Byte-set membership for one gate: printable ASCII, optionally `\t` and `\n`.
///
/// The three gates differ only in which of the two layout controls is *accepted*;
/// everything else is one set, so the membership rule lives here and the three
/// entry points below are instantiations (see `gate_band`).
#[inline(always)]
const fn byte_in_gate<const TAB_OK: bool, const NL_OK: bool>(b: u8) -> bool {
    match b {
        b'\t' => TAB_OK,
        b'\n' => NL_OK,
        _ => b >= 0x20 && b < 0x7F,
    }
}

/// Monomorphic band scanner: `Some(answer)` when a dedicated gate kernel served
/// the input, `None` when the caller must fall back to the fused classify path
/// (short inputs, the 128+ band on a host without AVX-512, and every non-x86
/// target).
///
/// Two bands are served: 16..127 by the SSE2 kernel on every x86_64, and
/// `>= CLASSIFY_AVX512_MIN_LEN` by the AVX-512 kernel where available.  An
/// AVX2-only host keeps the fused path for the wide band on purpose — there is no
/// AVX2 gate kernel, and adding one would need its own measurement before it can
/// claim to beat `classify_ascii_avx2`'s already-merged four-chunk form.
///
/// This band is the hot one for diagnostic rows (16..127 bytes) and it is where
/// `classify_ascii_sse2` computes three masks plus flag bookkeeping to answer a
/// single boolean.  The dedicated kernels need one violation mask per chunk, and
/// `TAB_OK`/`NL_OK` are compile-time, so the unused compares fold away entirely —
/// which is exactly the "finite, determined substitutions" case `duplicate` is
/// for, instead of three hand-copied bodies.
#[duplicate::duplicate_item(
    gate_band        tab_ok   nl_ok;
    [gate_band_strict]    [false]  [false];
    [gate_band_tab]       [true]   [false];
    [gate_band_multiline] [true]   [true];
)]
#[inline]
fn gate_band(s: &[u8]) -> Option<bool> {
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= CLASSIFY_AVX512_MIN_LEN && x86_level() >= 3 {
        // SAFETY: `level >= 3` iff `is_x86_feature_detected!("avx512bw")`
        // succeeded inside `x86_level_init`, and `avx512f` is implied on every
        // part implementing both (the init probes it explicitly).
        return Some(unsafe { x86_64::gate_avx512::<tab_ok, nl_ok>(s) });
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && (CLASSIFY_SSE2_MIN_LEN..GATE_SSE2_MAX_LEN).contains(&s.len()) {
        // SAFETY: SSE2 is baseline on x86_64 — no runtime feature check needed
        // (`classify_ascii_sse2`'s threshold table documents the same fact).
        //
        // `!swar_forced()` belongs here for the same reason it is on every
        // dispatch arm: without it a forced run would still execute SSE2 for
        // this band, so the "the SWAR body is the forced test/bench tier"
        // invariant would be false for exactly the band the predicates are
        // benchmarked in — `ascii_dispatch/swar_forced/printable/127` measured
        // the SSE2 gate (0.66 ns) instead of SWAR (6.3 ns), which is what
        // exposed it.
        return Some(unsafe { x86_64::gate_sse2::<tab_ok, nl_ok>(s) });
    }
    swallow_it::swallow!(s);
    None
}

/// On x86_64/aarch64/wasm32 the `u128` SWAR body below is **unreachable in production**: `len <
/// 16` takes `classify_scalar` and `len >= 16` enters a vector kernel that covers its own tail.  It
/// is the main body only on targets with no vector backend, and on vector targets it runs only
/// under the forced-SWAR test/bench tier.
#[inline]
#[must_use]
pub(crate) fn is_printable_ascii(s: &[u8]) -> bool {
    if let Some(v) = gate_band_strict(s) {
        return v;
    }
    // Fallback: defined by the class, one fused pass.  `stop_at` is the *first*
    // class that already answers this predicate, so a tab or newline anywhere
    // ends the scan at the chunk that contains it.
    classify_ascii_bounded(s, AsciiClass::PrintableOrTab) == AsciiClass::Printable
}

/// `true` when every byte is printable ASCII **or** the two layout controls
/// `\t` (0x09) / `\n` (0x0A). This is the predicate
/// `sanitize_for_terminal_multiline` needs: it preserves tab and newline as
/// layout intent, replaces every other control (CR, C1, DEL) with U+FFFD.
///
/// The SIMD kernels implement this same byte set.  On targets with a vector backend the `u128`
/// SWAR body below is reachable only under the forced-SWAR test/bench tier — production inputs are
/// served end-to-end by the vector kernels and `classify_scalar` at `< 16`, and no kernel hands a
/// tail back.  (This said "production tails are resolved by narrower kernels and `classify_scalar`,
/// never by SWAR", which described the pre-tail design.)
///
/// The lane-level derivation of the `< 0x20` / tab / newline masks lives in one place only,
/// [`classify_swar_wide`]'s "SWAR lane proof"; it is not repeated here so the two cannot drift.
///
/// Precondition: none beyond the slice itself.  The SWAR body rejects high
/// bytes through the `x | (x ^ 0x7F − 1)` mask *before* any signed arithmetic, so it is
/// total over all `u8` values.  (An earlier note cited an overflow in `b + 0x80 - 0x20` as
/// the reason for a "every byte is ≤ 0x7F" precondition — that expression no longer exists
/// in any kernel, and the real invariant lives in [`classify_swar_wide`]'s lane proof.)
#[inline]
#[must_use]
pub(crate) fn is_printable_ascii_multiline(s: &[u8]) -> bool {
    if let Some(v) = gate_band_multiline(s) {
        return v;
    }
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
    if let Some(v) = gate_band_tab(s) {
        return v;
    }
    // The two narrow classes are exactly the tab set, so the first newline (or
    // offender) settles the answer.
    classify_ascii_bounded(s, AsciiClass::PrintableOrTabNewline) <= AsciiClass::PrintableOrTab
}

/// The class implied by the flags accumulated so far.
///
/// One home for the three-arm mapping (`newline` outranks `tab`), so `reached`
/// and `compose_classify_tail` cannot drift apart.  The ordering itself is the
/// `Ord` canary `AsciiClass` documents:
/// `Printable < PrintableOrTab < PrintableOrTabNewline < Other`.
#[inline]
fn running_class(saw_tab: bool, saw_newline: bool) -> AsciiClass {
    if saw_newline {
        AsciiClass::PrintableOrTabNewline
    } else if saw_tab {
        AsciiClass::PrintableOrTab
    } else {
        AsciiClass::Printable
    }
}

/// Would the class accumulated so far already settle a `stop_at`-bounded query?
///
/// Each predicate needs one bit: `is_printable_ascii` only cares whether the
/// class is *exactly* `Printable`, `is_printable_ascii_or_tab` whether it is at
/// most `PrintableOrTab`.  Any class `>= stop_at` therefore answers it, which is
/// what lets a tab-indented line stop at the chunk holding the indentation
/// instead of scanning the whole line.
///
/// The `saw_*` flags are *running*, so a tab seen earlier pins the class at
/// `PrintableOrTab` for every later chunk — monotone, as the `max` in
/// [`compose_classify_tail`] requires.
///
/// `stop_at` stays a **runtime** parameter; monomorphising it was considered and
/// not introduced.
///
/// The proposal was to make it a const generic so `(class >= stop_at)` folds away
/// per backend.  The premise weakened before the change was made: with the gate
/// kernels (`gate_band`) serving every *boolean* predicate call, `classify_ascii`
/// is now reached mostly by callers that need the class, where `stop_at` is
/// `Other` and the compare is a predictable always-false branch inside a loop
/// whose per-chunk vector work dominates — `classify/printable/64` measures
/// 5.37 ns for a *single* 64-byte chunk, so the branch is not the cost.  Three
/// instantiations × five backends of code bloat (the same trade `gate_band` makes
/// deliberately, but there for a measured halving of per-chunk work) is not
/// justified by that.
///
/// Revisit only on a profile that shows this branch, and decide it with an A/B
/// (const `Other` vs runtime) rather than by argument.
#[inline]
fn reached(saw_tab: bool, saw_newline: bool, stop_at: AsciiClass) -> Option<AsciiClass> {
    let class = running_class(saw_tab, saw_newline);
    (class >= stop_at).then_some(class)
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
    running_class(saw_tab, saw_newline).max(tail)
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
/// The byte-set rule these masks implement — including why the `< 0x20`, tab
/// and newline masks are exact while the high-byte/DEL rejection is only
/// existential — is derived once, in [`classify_swar_wide`]'s "SWAR lane
/// proof".  This doc deliberately does not repeat it.
///
/// Load 8 bytes as a little-endian word — the lane order every SWAR step here
/// assumes.
#[inline]
fn load_word8(s: &[u8]) -> u64 {
    u64::from_le_bytes(s[..8].try_into().unwrap())
}

/// One u64 SWAR step of the `classify_ascii` byte-set rule.
///
/// `Some((saw_tab, saw_newline))` when all eight lanes are printable or `\t`/
/// `\n`; `None` when any lane is not.  Two callers share this: the `8..=15`
/// dispatch remainder and the SWAR body's own tail.  The algebra is the u128
/// loop one register narrower — the per-lane borrow proofs never reference the
/// word width, which is what makes this a copy rather than a second rule.
#[inline]
fn classify_word8(word: u64) -> Option<(bool, bool)> {
    const HI64: u64 = 0x8080_8080_8080_8080;
    const SP64: u64 = 0x2020_2020_2020_2020;
    const TAB64: u64 = 0x0909_0909_0909_0909;
    const NL64: u64 = 0x0A0A_0A0A_0A0A_0A0A;
    const DEL64: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES64: u64 = 0x0101_0101_0101_0101;
    let v = word ^ DEL64;
    if (word | v.wrapping_sub(ONES64)) & HI64 != 0 {
        // Non-ASCII (high bit) or exactly DEL.
        return None;
    }
    let lt20 = !((word | HI64).wrapping_sub(SP64)) & HI64;
    if lt20 == 0 {
        // All eight lanes are plain printable ASCII.
        return Some((false, false));
    }
    let tab = !(((word ^ TAB64) | HI64).wrapping_sub(ONES64)) & HI64;
    let nl = !(((word ^ NL64) | HI64).wrapping_sub(ONES64)) & HI64;
    // A `< 0x20` lane that is neither tab nor newline is an offender.
    if (lt20 & !tab & !nl) != 0 {
        return None;
    }
    Some((tab != 0, nl != 0))
}

/// First byte outside `lo..=hi` (`lo`, `hi` ASCII), or `usize::MAX`.
///
/// A SWAR range scan: `position(|b| !lo..=hi .contains(b))` is a per-byte compare that
/// LLVM will not vectorize in the short-circuiting `find` shape, while this walks 8
/// bytes per iteration and only falls back to a byte scan inside a chunk that actually
/// contains an offender.  The caller (`short.rs`'s plain-run scan) runs it once per
/// message byte run, i.e. on every rendered message.
///
/// Exactness: the high-bit test guarantees every lane is `< 0x80`, which is what makes
/// the two `| HI` subtraction masks per-lane exact (OR-ing `0x80` into each lane stops
/// borrows from crossing lanes).  A lane is then `v < lo` iff
/// `!((v|HI) - lo) & HI`, and `v > hi` iff `((v|HI) - (hi+1)) & HI`.
#[must_use]
pub(crate) fn first_byte_outside_range(s: &[u8], lo: u8, hi: u8) -> usize {
    const HI64: u64 = 0x8080_8080_8080_8080;
    const HI128: u128 = 0x8080_8080_8080_8080_8080_8080_8080_8080;
    const ONES128: u128 = 0x0101_0101_0101_0101_0101_0101_0101_0101;
    // A real assert, not a `debug_assert`: `hi >= 0x80` makes `rep(hi) + ONES`
    // carry out of the top lane, after which the two subtraction masks are no
    // longer per-lane exact and the function returns garbage *silently* in
    // release.  The cost is one predictable compare per call against a scan that
    // is already a per-message-run SWAR walk, so the debug-only guard was
    // trading a whole class of silent wrong answers for nothing.
    assert!(lo <= hi && hi < 0x80, "ASCII range required: {lo}..={hi}");
    let rep128 = |b: u8| u128::from(b) * ONES128;
    let (lo128, hi128) = (rep128(lo), rep128(hi) + ONES128);
    let rep = |b: u8| u64::from(b) * 0x0101_0101_0101_0101;
    let (lo_rep, hi_rep) = (rep(lo), rep(hi) + 0x0101_0101_0101_0101);
    let mut i = 0usize;
    // Widest portable step first.  The three masks are per-lane **exact** and their union
    // is exactly the offender set: `word & HI` flags every byte >= 0x80 (which is out of
    // range, since `hi < 0x80`), `lower_bad` flags `v < lo` (plus some high bytes),
    // `upper_bad` flags `v > hi` for `v <= 0x7F` — exact because every minuend lane has its
    // high bit forced, so no cross-lane borrow can occur.  So the **lowest set lane of the
    // union is the first offending byte** and `trailing_zeros() / 8` is its offset: the old
    // shape broke out, re-scanned the same 16 bytes as 2x8, then scanned them again scalar
    // (up to three passes over one chunk, on every rendered message's plain-run scan).
    // `first_byte_outside_range_is_lane_exact_at_every_offset` pins the exactness claim.
    //
    // A vector range kernel would still be faster (the same `+0x80`-stops-the-borrow masks
    // vectorise per lane), but that needs a new tail protocol and a measurement matrix, so it
    // is not added unmeasured.
    while i + 16 <= s.len() {
        let word = u128::from_le_bytes(s[i..i + 16].try_into().unwrap());
        let bad = (word & HI128)
            | (!((word | HI128).wrapping_sub(lo128)) & HI128)
            | (((word | HI128).wrapping_sub(hi128)) & HI128);
        if bad != 0 {
            return i + (bad.trailing_zeros() / 8) as usize;
        }
        i += 16;
    }
    // A 9..=15-byte remainder takes two **overlapping** words (at `i` and `len - 8`) with one
    // branch each, where the sequential loop paid a branch per 8 bytes and then a scalar walk for
    // the last 1..=7 — the same trick [`classify_swar_wide`]'s entry path uses, and the caller
    // (`char_col`'s plain-run scan) is per message run.  Overlap is idempotent: both words compute
    // the same per-lane masks, and the first offender wins by position.
    if s.len() - i >= 9 {
        for at in [i, s.len() - 8] {
            let word = load_word8(&s[at..]);
            let bad = (word & HI64)
                | (!((word | HI64).wrapping_sub(lo_rep)) & HI64)
                | (((word | HI64).wrapping_sub(hi_rep)) & HI64);
            if bad != 0 {
                return at + (bad.trailing_zeros() / 8) as usize;
            }
        }
        return usize::MAX;
    }
    while i + 8 <= s.len() {
        let word = load_word8(&s[i..]);
        let bad = (word & HI64)
            | (!((word | HI64).wrapping_sub(lo_rep)) & HI64)
            | (((word | HI64).wrapping_sub(hi_rep)) & HI64);
        if bad != 0 {
            return i + (bad.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    if i == s.len() {
        return usize::MAX;
    }
    // `1..=7` remainder: one overlapping word at `len - 8` (guarded — this function accepts short
    // inputs, so `len - 8` underflows below 8).  The bytes before `i` are in range by construction,
    // so the lowest flagged lane of this word is the first offender; overlap with the scanned
    // prefix is harmless for the same reason.
    if s.len() >= 8 {
        let word = load_word8(&s[s.len() - 8..]);
        let bad = (word & HI64)
            | (!((word | HI64).wrapping_sub(lo_rep)) & HI64)
            | (((word | HI64).wrapping_sub(hi_rep)) & HI64);
        if bad != 0 {
            return (s.len() - 8) + (bad.trailing_zeros() / 8) as usize;
        }
        return usize::MAX;
    }
    s[i..]
        .iter()
        .position(|&b| !(lo..=hi).contains(&b))
        .map_or(usize::MAX, |off| i + off)
}

#[must_use]
pub(crate) fn classify_ascii(s: &[u8]) -> AsciiClass {
    classify_ascii_bounded(s, AsciiClass::Other)
}

/// [`classify_ascii`] with an early exit as soon as the class reaches `stop_at`.
///
/// `stop_at == AsciiClass::Other` is exactly the old behaviour (the class can
/// only reach `Other` by an offender, which every tier already returns on), so
/// `classify_ascii` stays the full-classification entry.
fn classify_ascii_bounded(s: &[u8], stop_at: AsciiClass) -> AsciiClass {
    // Tail protocol, stated exactly: **no kernel re-enters this dispatcher for its own
    // remainder.**  Every backend — classify and NNA, all five — ends in its own masked overlap
    // tail (`tail_overlap_16` on the 16-byte backends, the 32/64-byte forms in the AVX2/AVX-512
    // kernels, `gate_sse2`'s masked load), and the SWAR body resolves its `< 16` remainder inline
    // via `classify_word8` and `classify_scalar`.  The `CLASSIFY_*_MIN_LEN` / `NNA_*_MIN_LEN`
    // thresholds exist to keep each kernel's `base = len - MIN_CHUNK` overlap load in range (the
    // `const _` canaries pin that), not to structure a delegation chain — a re-dispatch here
    // would reintroduce the ~1.5 ns/byte tax the tails deleted, and the `debug_assert_eq!(i,
    // s.len())` in each kernel only catches it in debug.
    //
    // (This paragraph used to claim a vector kernel's tail "deliberately *does* re-enter", and
    // called the correct wording "an earlier wording [that] contradicted the kernels".  It was the
    // one stale survivor of the doc migration every kernel and both module headers completed, and
    // it sat at the dispatcher — the first place an editor touching thresholds reads — actively
    // authorizing the restoration of the deleted re-dispatch.)
    if s.len() < 16 {
        // `8..=15` gets one u64 SWAR step before falling through to scalar.  Measured on the CI
        // mix: `printable/15` cost 13.09 ns against 6.87 ns at 16 bytes — the cliff is this entry
        // path, not the vector loop.  (The same run's `/31`, `/63`, `/127`, `/255` spikes were the
        // vector tails re-entering here with 15 bytes; no kernel re-enters any more, so this step
        // now serves the SWAR tier — forced on vector targets, production on the rest.)
        let (mut saw_tab, mut saw_newline) = (false, false);
        let mut i = 0usize;
        if s.len() >= 8 {
            match classify_word8(load_word8(s)) {
                Some((tab, nl)) => {
                    saw_tab = tab;
                    saw_newline = nl;
                }
                None => return AsciiClass::Other,
            }
            i = 8;
            if let Some(done) = reached(saw_tab, saw_newline, stop_at) {
                return done;
            }
        }
        if s.len() >= 9 {
            // Second, overlapping word: for `len <= 16` the two windows overlap,
            // so bytes `0..8` plus `len-8..len` cover the whole input and the
            // per-byte scalar tail disappears from the entire `9..=15` band.
            // Overlap is idempotent under the max-semantics of the class, and the
            // `reached` check above stays where it is: for `stop_at < Other` it
            // already settled the answer, for `Other` both words are exact.
            match classify_word8(load_word8(&s[s.len() - 8..])) {
                Some((tab, nl)) => {
                    saw_tab |= tab;
                    saw_newline |= nl;
                }
                None => return AsciiClass::Other,
            }
            // No `reached` check here: `compose_classify_tail(saw, Printable)` returns
            // `max(running, Printable) = running`, which is exactly the value `reached` would
            // return — so the branch was 100% redundant, and its comment claimed a saving that does
            // not exist (`compose` *is* that step).  Only the first word's check is load-bearing: it
            // can skip the second word entirely.
            return compose_classify_tail(saw_tab, saw_newline, AsciiClass::Printable);
        }
        return compose_classify_tail(saw_tab, saw_newline, classify_scalar(&s[i..]));
    }
    // Each arm consults `x86_level()` *behind* its length gate, and `&&`
    // short-circuits: the `16..63` band — the hottest diagnose-line tier — therefore never
    // touches the level cache line, and the cold cpuid probe defers until a row that can
    // actually use a vector kernel arrives.  `64..127` pays one `Relaxed` load: since
    // `CLASSIFY_AVX512_MIN_LEN` moved to 64 (measured: `/127` 19.19 ns on SSE2 vs `/128`
    // 8.33 ns on AVX-512), that band takes the AVX-512 arm, so the level load is bought
    // back many times over.  The old "`16..127` never touches the level" wording dated from
    // a floor of 128.
    // One level load for **both** wide arms.  The old shape tested `x86_level() >= 3` and then,
    // in a separate `>= 2` arm, asked again — two `LEVEL` loads (and two `swar_forced()` TLS
    // reads) per call on exactly the hosts that take the second arm.  Nesting keeps what the flat
    // order was chosen for: both arms are length-gated at `>= CLASSIFY_AVX512_MIN_LEN` (64), so
    // the 16..63 band still never touches the level cache line, and a level-3 host pays nothing
    // extra.
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= CLASSIFY_AVX512_MIN_LEN {
        let level = x86_level();
        if level >= 3 {
            // SAFETY: `level >= 3` iff `is_x86_feature_detected!("avx512bw")` succeeded inside
            // `x86_level_init`; `avx512f` is implied by `avx512bw` on every part implementing
            // both, which is exactly what the kernel's
            // `#[target_feature(enable = "avx512f,avx512bw")]` requires.
            return unsafe { x86_64::classify_ascii_avx512(s, stop_at) };
        }
        if s.len() >= CLASSIFY_AVX2_MIN_LEN && level >= 2 {
            // SAFETY: `level >= 2` iff `is_x86_feature_detected!("avx2")` succeeded; the
            // kernel's 32-byte loads are AVX2-only.  (Its tail is its own masked 32-byte overlap:
            // an earlier note here said the tail was "deferred back to this function", which
            // described the re-dispatch the kernels deleted.)
            return unsafe { x86_64::classify_ascii_avx2(s, stop_at) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= CLASSIFY_SSE2_MIN_LEN {
        // SAFETY: SSE2 is baseline for every x86_64 target, so the 16-byte
        // intrinsics need no runtime detection.  `!swar_forced()` is the
        // test/bench tier switch; it lives on this arm because the `16..31`
        // band never consults `x86_level()` (every level-gated arm is >= 32),
        // so without this conjunct a forced run would execute SSE2 there.
        return unsafe { x86_64::classify_ascii_sse2(s, stop_at) };
    }
    #[cfg(target_arch = "aarch64")]
    if !swar_forced() && s.len() >= 16 {
        // SAFETY: NEON is baseline for every aarch64 target;
        // `!swar_forced()` is the test/bench tier switch.
        return unsafe { neon::classify_ascii(s, stop_at) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if !swar_forced() && s.len() >= 16 {
        // SAFETY: `target_feature = "simd128"` is a compile-time gate; wasm
        // has no runtime feature detection, so the module (and this arm) only
        // exist when the binary is built for a SIMD-capable engine.  A
        // runtime-only gate would be UB on an engine without vectors.
        return unsafe { wasm::classify_ascii(s, stop_at) };
    }
    // The u128 SWAR body is outlined: on vector targets it is
    // production-unreachable (the forced test/bench tier is its only caller),
    // yet it used to sit inline in the dispatcher that every hot call
    // inlines into, paying ~40-60 instructions of icache/layout pollution per
    // call site. `#[cold]` + `#[inline(never)]` keeps it out of the hot body;
    // the forced-SWAR tests keep exercising it.
    classify_swar_wide(s, stop_at)
}

/// The `u128` SWAR classifier: the production body on targets with no vector
/// backend, and the forced test/bench tier everywhere else.
///
/// # SWAR lane proof (canonical)
///
/// Every other SWAR site in this module — [`classify_word8`],
/// [`is_printable_ascii_multiline`], [`first_byte_outside_range`] — implements a
/// slice of this argument and cross-references it here rather than restating it:
/// two copies of a borrow-chain proof drift silently (an edit admitting `\r` in
/// one place would pass every test that only reads the other).
///
/// The high-byte/DEL rejection is deliberately *not* an exact mask: it is
/// existential — any high bit in `x`, or a DEL lane in `x ^ 0x7F`, is itself the
/// proof, and a borrow chain can only originate at such a lane within the same
/// chunk (`wrapping_sub` discards the borrow out of the top lane, so lane 0 of
/// every chunk enters with `borrow_in = 0`).  The `< 0x20`, tab and newline masks
/// **are** exact, because each minuend byte has its high bit forced with
/// `x | HI`, which stops lane-to-lane borrowing:
///
/// * `lt20 = !((x|HI) - SP) & HI` — high bit clear iff `b + 0x60 < 0x80` iff
///   `b < 0x20`, exactly.
/// * `tab = !(((x ^ TAB)|HI) - ONES) & HI` — high bit clear iff `b ^ 0x09 == 0`
///   iff `b == 0x09`, exactly (the 0x08-neighbour trap of a `has_zero` value
///   mask cannot occur here because the minuend never borrows); `nl` likewise
///   with 0x0A.
///
/// The `lo..=hi` range masks in [`first_byte_outside_range`] are the same
/// "OR in 0x80 to stop the borrow" trick applied twice (`v < lo`, `v > hi`),
/// which is why that function *requires* ASCII bounds: `hi >= 0x80` makes
/// `hi + 1` carry into the lane above and the masks stop being per-lane exact.
///
/// Split out of [`classify_ascii_bounded`] (its only caller) so the hot
/// dispatcher does not carry its ~40-60 instructions at every inlined call site.
// Cold **only where a vector backend shadows it**: on riscv/loongarch/etc. this *is* the
// production body for every row, and `#[cold]` there costs placement and prediction for the
// hottest path in the file.
#[cfg_attr(
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    ),
    cold
)]
#[inline(never)]
fn classify_swar_wide(s: &[u8], stop_at: AsciiClass) -> AsciiClass {
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
            // Only this branch can change the flags, so it is the only place the monotone answer
            // can change: on a chunk with no `< 0x20` lane the previous iteration already derived
            // the same value (or the entry path did), so re-asking re-derived a constant — one
            // always-executed compare+branch per clean 16-byte chunk, which is the common case.
            // `Other` never needs the question at all: the full-classification entry cannot stop
            // early.
            if stop_at != AsciiClass::Other
                && let Some(done) = reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
        }
        i += 16;
    }
    // One implementation of the byte-set rule for the tail: `classify_scalar`
    // early-returns `Other` and is `#[inline]`, and `compose_classify_tail` is
    // `max`, so an offending tail propagates exactly as the hand-rolled loop
    // did.  A second copy of "printable ∪ {\t, \n}" here could drift (a future
    // edit admitting `\r` in one place only would be silent).
    //
    // The `8..=15` remainder shares [`classify_word8`] with the dispatch entry:
    // the CI run's `/31`, `/63`, `/127`, `/255` spikes are exactly "k×16 + 15",
    // i.e. a vector tail re-entering this function with 15 bytes, so both sites
    // must take the same step.
    if s.len() - i >= 8 {
        match classify_word8(load_word8(&s[i..])) {
            Some((tab, nl)) => {
                saw_tab |= tab;
                saw_newline |= nl;
            }
            None => return AsciiClass::Other,
        }
        // A 9..=15-byte remainder would still end in a scalar walk of up to 7 bytes (~2.8 ns
        // on the worst tail).  Take the **entry path's** own trick instead: the overlapping
        // word at `len - 8` turns the remainder into two 8-byte steps, and overlap is
        // idempotent under `max`-semantics, so the bytes the first step already covered are
        // simply re-read.  For `len - i == 9` the two words meet at `i + 1`, so their union is
        // exactly `[i, len)`.
        if s.len() - i >= 9 {
            match classify_word8(load_word8(&s[s.len() - 8..])) {
                Some((tab, nl)) => {
                    saw_tab |= tab;
                    saw_newline |= nl;
                }
                None => return AsciiClass::Other,
            }
            return compose_classify_tail(saw_tab, saw_newline, AsciiClass::Printable);
        }
        i += 8;
    }
    if i < s.len() {
        // `1..=7` remainder: one **overlapping** word at `len - 8` instead of a scalar walk
        // (`classify_scalar` on ≤7 bytes costs ~2–5 ns; this is one ~1 ns SWAR step).  Sound
        // because the bytes before `i` are already known clean — an offender there would have
        // returned — and the flags they contribute are re-OR-ed, which is idempotent under the
        // `max`-semantics of [`compose_classify_tail`].  `s.len() >= 16` on entry, so `len - 8`
        // cannot underflow.
        match classify_word8(load_word8(&s[s.len() - 8..])) {
            Some((tab, nl)) => {
                saw_tab |= tab;
                saw_newline |= nl;
            }
            None => return AsciiClass::Other,
        }
    }
    compose_classify_tail(saw_tab, saw_newline, AsciiClass::Printable)
}

/// Overlap-tail parameters for a 16-byte-wide kernel (NEON, wasm).
///
/// A kernel scans `0..i` in whole 16-byte chunks and then covers the `1..=15`
/// remaining bytes with **one load of the final 16 bytes**, at
/// `base = len - 16`.  The first `skip = i - base` lanes of that load were
/// already scanned, so a tail must mask them out before taking
/// `trailing_zeros` / `first_high_lane`.
///
/// Returns `None` when `i == len`: an exact multiple of 16 leaves nothing to
/// scan, and `skip` would be `0`, which the mask forms do not express.
///
/// Extracted rather than duplicated per backend because the arithmetic is the
/// only part of the protocol testable on a host without that backend: both
/// call sites are `#[target_feature]` kernels on aarch64/wasm32, which x86 CI
/// never compiles.  `tail_overlap_16` is plain integer code, so the off-by-one
/// that broke both backends ships with a portable test.
///
/// `base = len - 16` underflows below 16, which is why every caller's arm in
/// [`next_non_ascii_byte`] is gated on `s.len() >= 16`; the `const _` canaries
/// below pin that gate, the in-kernel check is a debug assert.
#[inline]
fn tail_overlap_16(len: usize, i: usize) -> Option<(usize, usize)> {
    if i >= len {
        return None;
    }
    let base = len - 16;
    let skip = i - base; // 1..=15
    debug_assert!((1..16).contains(&skip), "tail skip out of range: {skip}");
    Some((base, skip))
}

/// Lane index of the lowest set bit of a **per-lane** bitmask, or `None`.
///
/// The distinction this function exists to make impossible to get wrong: a SWAR `u64` word
/// carries **8 bits per byte lane**, so its `trailing_zeros() / 8` is a byte offset; a
/// `u8x16_bitmask` / `_mm_movemask_epi8` result carries **one bit per lane**, so
/// `trailing_zeros()` is *already* the lane index.  Mixing the two domains shipped a real bug:
/// the wasm `next_non_ascii_byte` tail divided by 8, so for any length that is not a multiple of
/// 16 with a high byte in the last 1..=15 bytes it returned an already-scanned ASCII offset —
/// wrong character columns on wasm32, with nothing to assert against.
///
/// Extracted and host-tested because the wasm backend cannot execute in CI: this is precisely the
/// conversion that failed silently there, so it must be verifiable on x86.
///
/// `skip` is the count of already-scanned lanes to mask off, per [`tail_overlap_16`]'s contract
/// (`1..=15`).
#[inline]
#[must_use]
pub(crate) fn first_lane_masked(bits: u16, skip: u16) -> Option<usize> {
    // The contract is `1..=15` (what `tail_overlap_16` yields), not `0..=15`: with `skip == 0`
    // the mask keeps **every** lane including the already-scanned ones, which is exactly the
    // silent-wrong-offset class this helper exists to make impossible — and `skip < 16` would not
    // have caught it.
    debug_assert!(
        (1..16).contains(&skip),
        "skip out of the per-lane domain: {skip}"
    );
    let bits = bits & !((1u16 << skip) - 1);
    (bits != 0).then(|| bits.trailing_zeros() as usize)
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
    // build), so a short remaining run must not pay the whole `cfg` chain (and
    // the level load) to reach a `position()`.  `usize::MAX` is the sentinel
    // shared with every kernel tail.
    if s.len() < 16 {
        // `8..=15` gets one u64 high-bit test first: measured `next_non_ascii/15`
        // 10.58 ns against 4.98 ns at 16 bytes, the same entry cliff as
        // `classify_ascii`.  The 8 bytes are re-scanned scalar-ly only when the
        // word does contain a high bit, which is the rare path here.
        if s.len() >= 8 {
            let hi = load_word8(s) & 0x8080_8080_8080_8080;
            if hi != 0 {
                // Same lane mapping as every SWAR loop below: the lowest set
                // high bit divided by 8 is the lowest offending byte.  A scalar
                // `position` here would be branchier and inconsistent with them.
                return (hi.trailing_zeros() / 8) as usize;
            }
            if s.len() == 8 {
                return usize::MAX;
            }
            // Second, overlapping word: bytes `8..len-8` are empty for
            // `len <= 16` and word 0 already ruled out `0..8`, so the first
            // flagged lane here *is* the first non-ASCII byte.
            let hi = load_word8(&s[s.len() - 8..]) & 0x8080_8080_8080_8080;
            if hi != 0 {
                return (s.len() - 8) + (hi.trailing_zeros() / 8) as usize;
            }
            return usize::MAX;
        }
        return s.iter().position(|&b| b >= 0x80).unwrap_or(usize::MAX);
    }
    // Same shape as `classify_ascii_bounded`: the level is consulted behind the
    // length gate, so a short run never loads it.
    // Same nesting as `classify_ascii_bounded`: one `x86_level()` load for both wide arms, so a
    // level-1/2 host does not load it twice for one call.  `x86_level()` returns 0 under the
    // forced-SWAR tier, which is why these arms carry no separate `swar_forced()` conjunct.
    // Outer gate is the **AVX2** floor, not the AVX-512 one.  The classify nesting is safe with a
    // `>= CLASSIFY_AVX512_MIN_LEN` (64) outer gate because `CLASSIFY_AVX2_MIN_LEN` (128) is larger,
    // so no AVX2 band is hidden behind it; NNA is the opposite (`NNA_AVX2_MIN_LEN = 32`,
    // `NNA_AVX512_MIN_LEN = 64`), and gating on 64 made the AVX2 kernel unreachable for 32..63 —
    // that band silently ran SSE2 at every level while the header's measured table recorded the
    // deliberate "AVX2 from 32, AVX-512 from 64" decision (32 B: 5.30 ns AVX2).  The canaries below
    // now pin the precondition instead of leaving it to a copied comment.
    #[cfg(target_arch = "x86_64")]
    if s.len() >= NNA_AVX2_MIN_LEN {
        let level = x86_level();
        if level >= 3 && s.len() >= NNA_AVX512_MIN_LEN {
            // SAFETY: `level >= 3` iff `is_x86_feature_detected!("avx512bw")` succeeded inside
            // `x86_level_init`; `avx512f` is implied on every part implementing both.
            return unsafe { x86_64::next_non_ascii_byte_avx512(s) };
        }
        if level >= 2 {
            // SAFETY: `level >= 2` iff `is_x86_feature_detected!("avx2")` succeeded; the 32-byte
            // loads are AVX2-only, and the length gate is this arm's own floor (`>= 32`, canaried).
            return unsafe { x86_64::next_non_ascii_byte_avx2(s) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    if !swar_forced() && s.len() >= 16 {
        // SAFETY: SSE2 is baseline on x86_64; `!swar_forced()` is the
        // test/bench tier switch: the level-gated arms are all >= 32, so this
        // is the only place that can send a forced run into SSE2.
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
    if i == s.len() {
        return usize::MAX;
    }
    // `1..=7` remainder: one overlapping word at `len - 8` (never underflows — the body is entered
    // at `len >= 16`).  No masking is needed: the bytes before `i` carry no high bit (a flagged one
    // would have returned above), so the lowest flagged lane of this word *is* the first non-ASCII
    // byte.  Replacing the old `position()` scan removes the last scalar walk from this body, which
    // is production code on every non-vector target and the forced tier elsewhere.
    let hi = load_word8(&s[s.len() - 8..]) & HI64;
    if hi == 0 {
        return usize::MAX;
    }
    (s.len() - 8) + (hi.trailing_zeros() / 8) as usize
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
/// SSE2 for 16-byte chunks; `aarch64` uses NEON for 16-byte chunks.
///
/// Every kernel processes only full chunks.  The tail deliberately avoids a
/// masked load, and what it does instead is per family:
///
/// * `classify_ascii_*` re-enters [`super::classify_ascii_bounded`] with the
///   remaining bytes, so the dispatcher picks the next narrower tier (below 16
///   bytes that is the SWAR body's `u64` step and then `classify_scalar`).  The
///   SWAR lane proofs therefore stay the single source of truth below the vector
///   width even though a kernel does not call SWAR directly.
/// * **No kernel re-enters the dispatcher for its own remainder.**  Every backend covers
///   `i..len` with a masked overlap tail ([`tail_overlap_16`] on the 16-byte backends, the
///   32/64-byte forms in the AVX2/AVX-512 kernels, the SSE2 masked load in `gate_sse2`), and
///   each ends in a `debug_assert_eq!(i, s.len())` plus a trivial terminal value.  This
///   paragraph used to claim the opposite for AVX2/AVX-512 ("kept deliberately as the
///   documented safety net") while those kernels' own comments said the re-dispatch had been
///   *deleted as a trap*, and it also claimed NEON/wasm still re-entered — they stopped when
///   their masked tails landed.  The kernels are the truth; do not restore a re-dispatch on
///   the strength of an older revision of this doc.
#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn classify_ascii_avx2(
        s: &[u8],
        stop_at: super::AsciiClass,
    ) -> super::AsciiClass {
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
            // Return the *mask vectors*, not their reductions: OR-ing the two
            // halves before reducing is exact (a lane is set iff it is set in
            // its own half) and halves the layout `movemask`s — the SSE2 kernel
            // already ORs across four 16-byte chunks before reducing.
            let masks = |v: __m256i| {
                // Explicit `unsafe` blocks even though the enclosing fn is
                // `unsafe`: the module already writes its loads that way, and
                // `unsafe_op_in_unsafe_fn` (warn in edition 2024, deny in a
                // later one) would otherwise flag this closure alone the moment
                // the lint level moves — 400 lines of churn for a style
                // inconsistency.
                let lt = unsafe { _mm256_cmpgt_epi8(sp, v) };
                let tab = unsafe { _mm256_cmpeq_epi8(v, tab_v) };
                let nl = unsafe { _mm256_cmpeq_epi8(v, nl_v) };
                // Tab and newline are the two allowed layout controls; every
                // other `< 0x20` lane, and DEL, is an offender.
                let allowed_ctrl = unsafe { _mm256_or_si256(tab, nl) };
                let off = unsafe { _mm256_andnot_si256(allowed_ctrl, lt) };
                let eq = unsafe { _mm256_cmpeq_epi8(v, del) };
                (unsafe { _mm256_or_si256(eq, off) }, tab, nl)
            };
            let (bad_a, tab_a, nl_a) = masks(a);
            let (bad_b, tab_b, nl_b) = masks(b);
            if _mm256_movemask_epi8(_mm256_or_si256(bad_a, bad_b)) != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= _mm256_movemask_epi8(_mm256_or_si256(tab_a, tab_b)) != 0;
            saw_newline |= _mm256_movemask_epi8(_mm256_or_si256(nl_a, nl_b)) != 0;
            i += 64;
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
        }
        while i + 32 <= s.len() {
            // SAFETY: `i + 32 <= len`, and the caller checked AVX2 support.
            let v = _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i);
            let lt = _mm256_cmpgt_epi8(sp, v);
            let tab = _mm256_cmpeq_epi8(v, tab_v);
            let nl = _mm256_cmpeq_epi8(v, nl_v);
            let off = _mm256_andnot_si256(_mm256_or_si256(tab, nl), lt);
            let eq = _mm256_cmpeq_epi8(v, del);
            let bad = _mm256_or_si256(eq, off);
            if _mm256_movemask_epi8(bad) != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= _mm256_movemask_epi8(tab) != 0;
            saw_newline |= _mm256_movemask_epi8(nl) != 0;
            i += 32;
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
        }
        if i == s.len() {
            // An exact multiple of the chunk width leaves nothing to classify:
            // re-entering the dispatcher with an empty slice is pure overhead
            // and it is exactly the 16/32/64/128/256 lengths a benchmark uses.
            return super::compose_classify_tail(
                saw_tab,
                saw_newline,
                super::AsciiClass::Printable,
            );
        }
        // No `if s.len() >= 32` gate: the dispatcher only enters this kernel for
        // `len >= CLASSIFY_AVX2_MIN_LEN` (128), so the fall-through to
        // `compose_classify_tail` below was unreachable.  (The old note also justified this
        // with "a re-entrant tail call from a wider kernel lands in the SSE2 arm" — no wider
        // kernel re-enters the dispatcher any more, every tier has its own masked tail.)  The
        // canary on the constant is asserted next to its definition.
        {
            debug_assert!(s.len() >= 32, "AVX2 kernel entered below its minimum");
            // 32-byte peer of the SSE2 tier's masked tail: one overlapping load
            // covers the 1..=31-byte remainder that used to re-enter the
            // dispatcher (measured on the SSE2 tier as ~1.5 ns per tail byte).
            let base = s.len() - 32;
            let skip = i - base; // 1..=31
            // The `keep` shift is only defined for 1..=31: `skip == 0` would keep
            // lanes before `i` (already scanned) and `skip == 32` is a shift
            // overflow.  Both bounds come from the loop invariant and the early
            // return above; pin them so a future loop rewrite cannot shift into UB.
            debug_assert!((1..32).contains(&skip), "tail skip out of range: {skip}");
            let keep = (!0i32) << skip;
            let v = unsafe { _mm256_loadu_si256(s.as_ptr().add(base) as *const __m256i) };
            let (bad_v, tab_v_, nl_v_) = {
                let lt = _mm256_cmpgt_epi8(sp, v);
                let tab = _mm256_cmpeq_epi8(v, tab_v);
                let nl = _mm256_cmpeq_epi8(v, nl_v);
                let off = _mm256_andnot_si256(_mm256_or_si256(tab, nl), lt);
                let eq = _mm256_cmpeq_epi8(v, del);
                (_mm256_or_si256(eq, off), tab, nl)
            };
            let bad = unsafe { _mm256_movemask_epi8(bad_v) } & keep;
            if bad != 0 {
                // Bit `tz` set means byte `base + tz` is an offender, and the
                // re-dispatched suffix *starts* with it — so every tier would
                // answer `Other`, and `max(running, Other) == Other`.  Returning it
                // directly skips a whole level load + threshold chain + kernel call
                // on the (common) non-ASCII tail of a diagnose line.
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { _mm256_movemask_epi8(tab_v_) } & keep != 0;
            saw_newline |= unsafe { _mm256_movemask_epi8(nl_v_) } & keep != 0;
            return super::compose_classify_tail(
                saw_tab,
                saw_newline,
                super::AsciiClass::Printable,
            );
        }
        // Every path above returns, so the old trailing
        // `compose_classify_tail(.., classify_ascii_bounded(&s[i..], ..))` was dead
        // code (an `unreachable_code` lint under `-D warnings`) and a trap for a
        // future edit that "fixed" the wrong branch.
        unreachable!("classify_ascii_avx2 covers the tail unconditionally")
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
        if i < s.len() && s.len() >= 32 {
            // One overlapping load covers the whole 1..=31-byte remainder.
            // The `i < len` guard is load-bearing: an exact multiple of 32 leaves
            // nothing over, and `(!0i32) << 32` would be a shift overflow.
            let base = s.len() - 32;
            let skip = i - base; // 1..=31
            debug_assert!((1..32).contains(&skip), "tail skip out of range: {skip}");
            let keep = (!0i32) << skip;
            let v = unsafe { _mm256_loadu_si256(s.as_ptr().add(base) as *const __m256i) };
            let mask = unsafe { _mm256_movemask_epi8(v) } & keep;
            return if mask == 0 {
                usize::MAX
            } else {
                base + mask.trailing_zeros() as usize
            };
        }
        // Exact-multiple lengths (32/64/128/256/512 — the sizes a
        // benchmark and a 64-column diagnose row both hit) leave nothing
        // over; re-entering the dispatcher on an empty slice paid the
        // level chain just to compute `usize::MAX`.
        // Every remainder ≥ 1 is covered by the overlap tail above (the kernel is only
        // entered at its own `MIN_LEN`), so reaching here means the slice was an exact
        // multiple.  A `super::next_non_ascii_byte(&s[i..])` re-dispatch used to sit
        // here as a "safety net": it could only fire after a threshold change, and then
        // it would re-enter the dispatcher with a 1..=`MIN_LEN-1` slice — the
        // ~1.5 ns/byte tax the vector tails exist to remove.  Deleted rather than left
        // as a trap; the debug assert keeps the invariant honest if a threshold moves.
        debug_assert_eq!(i, s.len(), "the overlap tail covers every remainder");
        usize::MAX
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn classify_ascii_avx512(
        s: &[u8],
        stop_at: super::AsciiClass,
    ) -> super::AsciiClass {
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
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
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
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
        }
        if i == s.len() {
            // An exact multiple of the chunk width leaves nothing to classify:
            // re-entering the dispatcher with an empty slice is pure overhead
            // and it is exactly the 16/32/64/128/256 lengths a benchmark uses.
            return super::compose_classify_tail(
                saw_tab,
                saw_newline,
                super::AsciiClass::Printable,
            );
        }
        // Overlapping final load — the same protocol the SSE2 and AVX2 classify
        // kernels already use.  This kernel was the only wide one without it, so
        // a 1..=63-byte remainder re-entered the dispatcher (two length gates, a
        // level load, then a whole SSE2 kernel call) instead of one masked load;
        // that is the `/255` spike in the bench table.  The `>= 64` canary above
        // makes `s.len() >= 64` always true here.
        let base = s.len() - 64;
        let skip = i - base; // 1..=63
        debug_assert!((1..64).contains(&skip), "tail skip out of range: {skip}");
        let keep = (!0u64) << skip;
        let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(base).cast()) };
        let lt = unsafe { _mm512_cmpgt_epi8_mask(sp, v) };
        let tab = unsafe { _mm512_cmpeq_epi8_mask(v, tab_v) };
        let nl = unsafe { _mm512_cmpeq_epi8_mask(v, nl_v) };
        let bad = (unsafe { _mm512_cmpeq_epi8_mask(v, del) } | (lt & !(tab | nl))) & keep;
        if bad != 0 {
            // Same argument as the AVX2 arm: the offender is the first byte of the
            // suffix, so the class is `Other` without re-dispatching.
            return super::AsciiClass::Other;
        }
        saw_tab |= tab & keep != 0;
        saw_newline |= nl & keep != 0;
        super::compose_classify_tail(saw_tab, saw_newline, super::AsciiClass::Printable)
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
        if i < s.len() && s.len() >= 64 {
            // One overlapping 64-byte load covers the 1..=63-byte remainder; the
            // `i < len` guard keeps the shift below the lane count.
            //
            // `base = len - 64` underflows below 64, which is why this kernel's
            // dispatch arm is gated on `NNA_AVX512_MIN_LEN` (pinned `>= 64` by
            // the `const _` canary next to the constant).  The `s.len() >= 64`
            // conjunct above is the second, local copy of that gate — it is what
            // a re-entrant tail call would rely on, so it cannot be dropped as
            // redundant.
            let base = s.len() - 64;
            let skip = i - base; // 1..=63
            let keep = (!0u64) << skip;
            let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(base).cast()) };
            let mask = unsafe { _mm512_movepi8_mask(v) } & keep;
            return if mask == 0 {
                usize::MAX
            } else {
                base + mask.trailing_zeros() as usize
            };
        }
        // Exact-multiple lengths (32/64/128/256/512 — the sizes a
        // benchmark and a 64-column diagnose row both hit) leave nothing
        // over; re-entering the dispatcher on an empty slice paid the
        // level chain just to compute `usize::MAX`.
        // Every remainder ≥ 1 is covered by the overlap tail above (the kernel is only
        // entered at its own `MIN_LEN`), so reaching here means the slice was an exact
        // multiple.  A `super::next_non_ascii_byte(&s[i..])` re-dispatch used to sit
        // here as a "safety net": it could only fire after a threshold change, and then
        // it would re-enter the dispatcher with a 1..=`MIN_LEN-1` slice — the
        // ~1.5 ns/byte tax the vector tails exist to remove.  Deleted rather than left
        // as a trap; the debug assert keeps the invariant honest if a threshold moves.
        debug_assert_eq!(i, s.len(), "the overlap tail covers every remainder");
        usize::MAX
    }

    /// Dedicated "every byte belongs to one gate set" kernel.
    ///
    /// `classify_ascii_sse2` computes three masks (signed `lt`, tab, newline) plus
    /// two flag `movemask`s and a `reached` branch per chunk to answer a *class*.
    /// A boolean gate needs exactly one violation mask: `lt` — which under the
    /// signed compare already covers every byte ≥ 0x80 — OR'd with DEL, minus the
    /// layout controls this gate accepts.  `TAB_OK`/`NL_OK` are compile-time, so
    /// the unused `cmpeq`/`andnot` fold away.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn gate_sse2<const TAB_OK: bool, const NL_OK: bool>(s: &[u8]) -> bool {
        let sp = unsafe { _mm_set1_epi8(0x20) };
        let del = unsafe { _mm_set1_epi8(0x7F) };
        let mut i = 0usize;
        // 64-byte group: four loads OR-merged, so the group costs **one** `movemask` and
        // one branch instead of four.  This is the shape `classify_ascii_sse2` already uses
        // for the same band, and the band is the hot one (16..127-byte diagnose rows).
        // Exactness is per-lane: OR-ing violation vectors cannot invent or lose a lane, and
        // a set bit anywhere means "some chunk had that lane offending", which is all this
        // predicate asks.
        while i + 64 <= s.len() {
            let mut acc = unsafe { _mm_setzero_si128() };
            for off in [0usize, 16, 32, 48] {
                let v = unsafe { _mm_loadu_si128(s.as_ptr().add(i + off) as *const __m128i) };
                let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
                let mut bad = unsafe { _mm_or_si128(lt, _mm_cmpeq_epi8(v, del)) };
                if TAB_OK {
                    bad = unsafe { _mm_andnot_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x09)), bad) };
                }
                if NL_OK {
                    bad = unsafe { _mm_andnot_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x0A)), bad) };
                }
                acc = unsafe { _mm_or_si128(acc, bad) };
            }
            if unsafe { _mm_movemask_epi8(acc) } != 0 {
                return false;
            }
            i += 64;
        }
        while i + 16 <= s.len() {
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(i) as *const __m128i) };
            let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
            let mut bad = unsafe { _mm_or_si128(lt, _mm_cmpeq_epi8(v, del)) };
            if TAB_OK {
                bad = unsafe { _mm_andnot_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x09)), bad) };
            }
            if NL_OK {
                bad = unsafe { _mm_andnot_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x0A)), bad) };
            }
            if unsafe { _mm_movemask_epi8(bad) } != 0 {
                return false;
            }
            i += 16;
        }
        if i == s.len() {
            return true;
        }
        // Masked overlap tail over the final 16 bytes.  A scalar walk of the
        // 1..=15-byte remainder costs up to ~6 ns, and 16..63-byte rows are the
        // *hot* diagnostic band: every `len % 16 == 15` row paid it
        // (`predicates/printable/63` measured 6.92 ns against `/64`'s 2.29 ns
        // before this tail existed, while `/32` with no remainder was 0.63).
        //
        // `base = len - 16` underflows below 16, which the dispatch floor
        // (`CLASSIFY_SSE2_MIN_LEN`) guarantees; the `const _` canary pins it.
        debug_assert!(s.len() >= 16, "the floor guarantees a full final chunk");
        let base = s.len() - 16;
        let skip = i - base; // 1..=15
        debug_assert!((1..16).contains(&skip), "tail skip out of range: {skip}");
        let keep = (!0i32) << skip;
        let v = unsafe { _mm_loadu_si128(s.as_ptr().add(base) as *const __m128i) };
        let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
        let mut bad = unsafe { _mm_or_si128(lt, _mm_cmpeq_epi8(v, del)) };
        if TAB_OK {
            bad = unsafe { _mm_andnot_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x09)), bad) };
        }
        if NL_OK {
            bad = unsafe { _mm_andnot_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x0A)), bad) };
        }
        (unsafe { _mm_movemask_epi8(bad) } & keep) == 0
    }

    /// `gate_sse2`'s AVX-512 counterpart, for the `>= CLASSIFY_AVX512_MIN_LEN`
    /// band where the dispatcher would otherwise run the full `classify`.
    ///
    /// `classify_ascii` answers a boolean here by merging four mask vectors, then
    /// reducing three `movemask`s and running the `reached` bookkeeping — work a
    /// gate never needs.  One violation mask per 64-byte chunk and a single
    /// register-wide test is roughly half the per-chunk work, and all-printable
    /// long rows are the common shape for source lines.
    ///
    /// No overlap tail would be wrong *for a 64-byte floor*: the remainder is then
    /// 1..=63 bytes and a scalar walk of it costs ~25 ns on its own, which is why
    /// `predicates/printable/127` measured 30.2 ns against `128`'s 2.75 ns until
    /// this tail was added.  One masked load of the final 64 bytes covers it,
    /// exactly as the `classify` kernels do.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn gate_avx512<const TAB_OK: bool, const NL_OK: bool>(s: &[u8]) -> bool {
        let sp = unsafe { _mm512_set1_epi8(0x20) };
        let del = unsafe { _mm512_set1_epi8(0x7F) };
        let tab_v = unsafe { _mm512_set1_epi8(0x09) };
        let nl_v = unsafe { _mm512_set1_epi8(0x0A) };
        let mut i = 0usize;
        while i + 64 <= s.len() {
            let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(i).cast()) };
            // The signed `sp > v` compare already flags every byte >= 0x80, so the
            // pair below is the whole offender set before the layout exceptions.
            let mut bad = unsafe { _mm512_cmpgt_epi8_mask(sp, v) }
                | unsafe { _mm512_cmpeq_epi8_mask(v, del) };
            if TAB_OK {
                bad &= !unsafe { _mm512_cmpeq_epi8_mask(v, tab_v) };
            }
            if NL_OK {
                bad &= !unsafe { _mm512_cmpeq_epi8_mask(v, nl_v) };
            }
            if bad != 0 {
                return false;
            }
            i += 64;
        }
        if i == s.len() {
            return true;
        }
        // `base = len - 64` underflows below 64, which the dispatch arm
        // (`>= CLASSIFY_AVX512_MIN_LEN`, pinned `>= 64` by its `const _`) and the
        // local `s.len() >= 64` check both guarantee.
        debug_assert!(s.len() >= 64, "the floor guarantees a full final chunk");
        let base = s.len() - 64;
        let skip = i - base; // 1..=63
        debug_assert!((1..64).contains(&skip), "tail skip out of range: {skip}");
        let keep = (!0u64) << skip;
        let v = unsafe { _mm512_loadu_si512(s.as_ptr().add(base).cast()) };
        let mut bad =
            unsafe { _mm512_cmpgt_epi8_mask(sp, v) } | unsafe { _mm512_cmpeq_epi8_mask(v, del) };
        if TAB_OK {
            bad &= !unsafe { _mm512_cmpeq_epi8_mask(v, tab_v) };
        }
        if NL_OK {
            bad &= !unsafe { _mm512_cmpeq_epi8_mask(v, nl_v) };
        }
        (bad & keep) == 0
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn classify_ascii_sse2(
        s: &[u8],
        stop_at: super::AsciiClass,
    ) -> super::AsciiClass {
        let sp = _mm_set1_epi8(0x20);
        let del = _mm_set1_epi8(0x7F);
        let tab_v = _mm_set1_epi8(0x09);
        let nl_v = _mm_set1_epi8(0x0A);
        let mut saw_tab = false;
        let mut saw_newline = false;
        let mut i = 0usize;
        // 64-byte groups: four loads, three OR-merged reductions, one branch.
        //
        // The 16-byte version spent three `movemask`s and a branch per 16 bytes
        // (12 masks per 64).  Merging is exact — a lane is offending iff it is
        // offending in its own vector, and OR-ing the mask vectors before the
        // single `movemask` cannot lose or invent a lane — while the group form
        // turns the same work into 3 masks and 1 branch per 64 bytes.  SSE2 is
        // this dispatcher's tier for 16–127 bytes (the measured vector-tile
        // thresholds put every wider kernel at ≥128), so this loop *is* the hot
        // path for diagnostic line lengths.
        while i + 64 <= s.len() {
            let p = unsafe { s.as_ptr().add(i) } as *const __m128i;
            let mut bad = unsafe { _mm_setzero_si128() };
            let mut tab_acc = unsafe { _mm_setzero_si128() };
            let mut nl_acc = unsafe { _mm_setzero_si128() };
            for k in 0..4 {
                let v = unsafe { _mm_loadu_si128(p.add(k)) };
                let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
                let tab = unsafe { _mm_cmpeq_epi8(v, tab_v) };
                let nl = unsafe { _mm_cmpeq_epi8(v, nl_v) };
                let off = unsafe { _mm_andnot_si128(_mm_or_si128(tab, nl), lt) };
                let eq = unsafe { _mm_cmpeq_epi8(v, del) };
                bad = unsafe { _mm_or_si128(bad, _mm_or_si128(eq, off)) };
                tab_acc = unsafe { _mm_or_si128(tab_acc, tab) };
                nl_acc = unsafe { _mm_or_si128(nl_acc, nl) };
            }
            if unsafe { _mm_movemask_epi8(bad) } != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { _mm_movemask_epi8(tab_acc) } != 0;
            saw_newline |= unsafe { _mm_movemask_epi8(nl_acc) } != 0;
            i += 64;
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
        }
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
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
        }
        if i == s.len() {
            // An exact multiple of the chunk width leaves nothing to classify:
            // re-entering the dispatcher with an empty slice is pure overhead
            // and it is exactly the 16/32/64/128/256 lengths a benchmark uses.
            return super::compose_classify_tail(
                saw_tab,
                saw_newline,
                super::AsciiClass::Printable,
            );
        }
        if s.len() >= 16 {
            // Overlapping final load: re-read the last 16 bytes and mask out the
            // lanes already consumed, so a 1..=15-byte remainder costs one vector
            // step instead of a dispatcher round trip.  The measured cliff is
            // 255 B at 29 ns against 256 B at 13 ns; every one of those breaks is
            // `len % 16` priced at roughly 1.5 ns per byte by the scalar tail.
            //
            // Masking is exact for both outputs here: `compose_classify_tail` is
            // `max` over the classes, so OR-ing the surviving tab/newline lanes
            // gives the same class as visiting them one by one, and a surviving
            // `bad` lane means the remainder holds a non-printable byte — the
            // scalar path then settles *which* byte and the caller only needs the
            // class.
            let base = s.len() - 16;
            let skip = i - base; // 1..=15: lanes already scanned
            debug_assert!((1..16).contains(&skip), "tail skip out of range: {skip}");
            // `_mm_movemask_epi8` yields 16 bits in an `i32`, so the keep mask
            // is built in the same signed type; the bits above 15 are masked off
            // by the AND anyway.
            let keep = !((1i32 << skip) - 1);
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(base) as *const __m128i) };
            let tab = unsafe { _mm_cmpeq_epi8(v, tab_v) };
            let nl = unsafe { _mm_cmpeq_epi8(v, nl_v) };
            let lt = unsafe { _mm_cmpgt_epi8(sp, v) };
            let off = unsafe { _mm_andnot_si128(_mm_or_si128(tab, nl), lt) };
            let eq = unsafe { _mm_cmpeq_epi8(v, del) };
            let bad = unsafe { _mm_movemask_epi8(_mm_or_si128(eq, off)) } & keep;
            if bad != 0 {
                // No re-dispatch: the first set lane *is* the first byte of the
                // suffix, and every tier answers `Other` for a slice beginning with an
                // offender — the argument the AVX2/AVX-512 tails record.  The re-entry
                // cost a level load, three threshold compares and an SSE2 kernel call on
                // the common non-ASCII tail of a diagnose line.
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { _mm_movemask_epi8(tab) } & keep != 0;
            saw_newline |= unsafe { _mm_movemask_epi8(nl) } & keep != 0;
            return super::compose_classify_tail(
                saw_tab,
                saw_newline,
                super::AsciiClass::Printable,
            );
        }
        // Deleted rather than left as a trap, the same policy the AVX2/AVX-512 kernels
        // follow.  The dispatcher enters this kernel only at `len >= 16`
        // (`CLASSIFY_SSE2_MIN_LEN`), and the masked tail above covers every remainder, so
        // the fall-through is unreachable.  It used to re-enter the dispatcher, which is
        // the ~1.5 ns/byte tax this module removed everywhere else — leaving it here kept a
        // live-looking copy of the hazard for the next editor to copy.
        debug_assert_eq!(i, s.len(), "the masked tail covers every remainder");
        super::compose_classify_tail(saw_tab, saw_newline, super::AsciiClass::Printable)
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
        // Exact-multiple lengths (32/64/128/256/512 — the sizes a
        // benchmark and a 64-column diagnose row both hit) leave nothing
        // over; re-entering the dispatcher on an empty slice paid the
        // level chain just to compute `usize::MAX`.
        if i == s.len() {
            return usize::MAX;
        }
        // Overlapping masked tail: load the *last* 16 bytes (they overlap lanes
        // already scanned) and mask off the ones before `i`.  Re-entering the
        // dispatcher for a 1..15-byte remainder is the same ~1.5 ns/byte tax the
        // classify tails removed.  On an AVX2 host this kernel serves only the
        // 16..31-byte band — `classify_ascii_avx2` owns 32+, and AVX-512 owns
        // 128+ — so the band that reaches it is the *narrow* one, not the
        // 16..127 rows later kernels cover.
        if s.len() >= 16 {
            // `s.len() >= 16` is the kernel's own dispatch threshold
            // (`CLASSIFY_SSE2_MIN_LEN`), which is what keeps this subtraction in
            // range: below it `len - 16` would underflow.  The `const _` canary
            // on the threshold is the only other guard.
            let skip = 16 - (s.len() - i);
            debug_assert!(skip > 0 && skip < 16, "tail is a strict sub-chunk");
            let v = unsafe { _mm_loadu_si128(s.as_ptr().add(s.len() - 16) as *const __m128i) };
            let mask = unsafe { _mm_movemask_epi8(v) };
            let keep = mask & !((1i32 << skip) - 1);
            return if keep == 0 {
                usize::MAX
            } else {
                s.len() - 16 + keep.trailing_zeros() as usize
            };
        }
        // Same policy as the classify kernels: the dispatcher's SSE2 arm is gated on
        // `len >= 16`, and the masked tail above covers every remainder, so there is no
        // sub-vector case left to delegate.  The old re-dispatch here was the trap the
        // module's AVX2 kernels explicitly delete; keeping it would let a future
        // `CLASSIFY_SSE2_MIN_LEN` drop reintroduce the tax silently.
        debug_assert_eq!(i, s.len(), "the masked tail covers every remainder");
        usize::MAX
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
        // `v128_any_true` is the single-instruction form of "any lane nonzero";
        // `u8x16_bitmask(v) != 0` was a bitmask extract plus a compare for the same answer.
        unsafe { core::arch::wasm32::v128_any_true(v) }
    }

    #[target_feature(enable = "simd128")]
    pub(super) unsafe fn classify_ascii(s: &[u8], stop_at: super::AsciiClass) -> super::AsciiClass {
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
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
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
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
        }
        if i == s.len() {
            // An exact multiple of the chunk width leaves nothing to classify:
            // re-entering the dispatcher with an empty slice is pure overhead
            // and it is exactly the 16/32/64/128/256 lengths a benchmark uses.
            return super::compose_classify_tail(
                saw_tab,
                saw_newline,
                super::AsciiClass::Printable,
            );
        }
        // Masked overlap tail over the final 16 bytes — same protocol as the NNA
        // kernels (`tail_overlap_16`) and the x86 classify kernels.  Here the
        // lanes are masked in the *bitmask* domain (`& !((1 << skip) - 1)`), which
        // is why no vector can be zeroed: a zeroed lane is NUL, a genuine
        // offender.
        if let Some((base, skip)) = super::tail_overlap_16(s.len(), i) {
            let v = unsafe { v128_load(s.as_ptr().add(base).cast()) };
            let tabm = unsafe { u8x16_eq(v, tab) };
            let nlm = unsafe { u8x16_eq(v, nl) };
            let bad = unsafe {
                v128_or(
                    u8x16_eq(v, del),
                    andnot(i8x16_lt(v, sp), v128_or(tabm, nlm)),
                )
            };
            let keep = !((1u16 << skip) - 1);
            if unsafe { u8x16_bitmask(bad) } & keep != 0 {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { u8x16_bitmask(tabm) } & keep != 0;
            saw_newline |= unsafe { u8x16_bitmask(nlm) } & keep != 0;
            return super::compose_classify_tail(
                saw_tab,
                saw_newline,
                super::AsciiClass::Printable,
            );
        }
        // Unreachable: `tail_overlap_16` answers for every `i < len`, and the
        // exact-multiple case returned above.
        debug_assert_eq!(i, s.len(), "the overlap tail covers every remainder");
        super::compose_classify_tail(saw_tab, saw_newline, super::AsciiClass::Printable)
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
        // Overlap tail: one load of the final 16 bytes covers the whole
        // 1..=15-byte remainder, with the already-scanned lanes masked off.
        // This backend never had the masked tail the x86 kernels did — the
        // re-dispatch below used to *be* its tail — so deleting that re-dispatch
        // (as was correct for AVX2/AVX-512, which do have one) silently stopped
        // scanning bytes `i..len` on aarch64 and wasm32 for every input whose
        // length is not a multiple of 16.  See `tail_overlap_16`.
        if let Some((base, skip)) = super::tail_overlap_16(s.len(), i) {
            // Bitmask domain, matching this backend's `classify` tail: the data lanes
            // are never rewritten here, which is both fewer ops than a lane-index
            // vector and immune to the "a zeroed lane is NUL" trap that only applies
            // to `classify` (a NUL *is* an offender there; it is invisible to a
            // high-bit test).
            //
            // This block must use **wasm32** intrinsics only.  An earlier version of
            // this tail was NEON code pasted into this module (`vld1q_u8`, `LANE_IDX`,
            // `first_high_lane`), which meant the wasm backend had never compiled —
            // no CI target ever checked it, so nothing said so.  `cargo check
            // --target wasm32-unknown-unknown` (and the aarch64 check) now exist for
            // exactly this class of bug.
            let v = unsafe { v128_load(s.as_ptr().add(base).cast()) };
            // Domain check: `u8x16_bitmask` yields **one bit per lane**, so the masked
            // `trailing_zeros()` *is* the lane index.  This block used to divide by 8, copying
            // the SWAR `u64` domain (8 bits per byte lane) into a per-lane bitmask — which
            // returned an already-scanned ASCII offset for every non-multiple-of-16 length with
            // a high byte in the tail.  `first_lane_masked` now owns that conversion and is
            // host-tested, because wasm cannot execute in CI.
            return super::first_lane_masked(unsafe { u8x16_bitmask(v) }, skip as u16)
                .map_or(usize::MAX, |lane| base + lane);
        }
        // Exact-multiple lengths (32/64/128/256/512 — the sizes a
        // benchmark and a 64-column diagnose row both hit) leave nothing
        // over; re-entering the dispatcher on an empty slice paid the
        // level chain just to compute `usize::MAX`.
        debug_assert_eq!(i, s.len(), "the overlap tail covers every remainder");
        usize::MAX
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
        // `vshrn_n_u16` is const-generic in `std::arch` (shift intrinsics have
        // been since Rust 1.61, i.e. long before this crate's MSRV), so the
        // shift count must be a generic argument, not a value argument.
        let packed = unsafe { vshrn_n_u16::<4>(vreinterpretq_u16_u8(high)) };
        let bits = unsafe { vget_lane_u64(vreinterpret_u64_u8(packed), 0) };
        (bits != 0).then_some((bits.trailing_zeros() / 4) as usize)
    }

    /// Lane indices, for masking the already-scanned prefix of an overlap load
    /// (`vcgeq_u8(idx, splat(skip))` keeps lanes `skip..16`).
    const LANE_IDX: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    /// `a & !b` — the convention every backend in this module shares.
    ///
    /// NEON's `vbicq_u8(a, b)` already computes `a & !b`, the **opposite**
    /// argument order from x86's `_mm256_andnot_si256(a, b)` (`!a & b`).  The
    /// wasm backend wraps the same asymmetry; wrapping it here means a future
    /// x86↔NEON port cannot silently swap the operands, which is the port bug the
    /// wasm module's docstring warns about.
    #[inline]
    unsafe fn andnot(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vbicq_u8(a, b) }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn classify_ascii(s: &[u8], stop_at: super::AsciiClass) -> super::AsciiClass {
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
                let off = andnot(lt, vorrq_u8(tab, nl));
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
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
        }
        while i + 16 <= s.len() {
            let v = unsafe { vld1q_u8(s.as_ptr().add(i)) };
            let lt = unsafe { vcltq_s8(vreinterpretq_s8_u8(v), sp) };
            let tab = unsafe { vceqq_u8(v, tab_v) };
            let nl = unsafe { vceqq_u8(v, nl_v) };
            let off = unsafe { andnot(lt, vorrq_u8(tab, nl)) };
            let eq = unsafe { vceqq_u8(v, del) };
            if unsafe { any_nonzero(vorrq_u8(eq, off)) } {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { any_nonzero(tab) };
            saw_newline |= unsafe { any_nonzero(nl) };
            i += 16;
            // `stop_at != Other` is the same conjunct `classify_swar_wide` carries: `Other` can
            // only be produced by an offender, which every kernel returns on regardless, so for the
            // full-classification entry (`stop_at == Other`, the common one) this flags→class→compare
            // chain is dead work in the hot loop.
            if stop_at != super::AsciiClass::Other
                && let Some(done) = super::reached(saw_tab, saw_newline, stop_at)
            {
                return done;
            }
        }
        if i == s.len() {
            // An exact multiple of the chunk width leaves nothing to classify:
            // re-entering the dispatcher with an empty slice is pure overhead
            // and it is exactly the 16/32/64/128/256 lengths a benchmark uses.
            return super::compose_classify_tail(
                saw_tab,
                saw_newline,
                super::AsciiClass::Printable,
            );
        }
        // Masked overlap tail over the final 16 bytes — the same protocol the NNA
        // kernels already use (`tail_overlap_16`), and the one every x86 classify
        // kernel has always had.  Without it a 1..=15-byte remainder re-entered
        // the dispatcher (two length gates plus a level load) to classify at most
        // 15 bytes.
        //
        // The masks are computed on the **raw** vector and then AND-ed with
        // `keep`: zeroing the skipped lanes instead would turn them into NUL,
        // which is a real offender (`0x00 < 0x20`) and would report `Other` for
        // every input with a remainder.
        if let Some((base, skip)) = super::tail_overlap_16(s.len(), i) {
            let v = unsafe { vld1q_u8(s.as_ptr().add(base)) };
            let idx = unsafe { vld1q_u8(LANE_IDX.as_ptr()) };
            let keep = unsafe { vcgeq_u8(idx, vdupq_n_u8(skip as u8)) };
            let lt = unsafe { vcltq_s8(vreinterpretq_s8_u8(v), sp) };
            let tabm = unsafe { vceqq_u8(v, tab_v) };
            let nlm = unsafe { vceqq_u8(v, nl_v) };
            let off = unsafe { andnot(lt, vorrq_u8(tabm, nlm)) };
            let bad = unsafe { vandq_u8(vorrq_u8(vceqq_u8(v, del), off), keep) };
            if unsafe { any_nonzero(bad) } {
                return super::AsciiClass::Other;
            }
            saw_tab |= unsafe { any_nonzero(vandq_u8(tabm, keep)) };
            saw_newline |= unsafe { any_nonzero(vandq_u8(nlm, keep)) };
            return super::compose_classify_tail(
                saw_tab,
                saw_newline,
                super::AsciiClass::Printable,
            );
        }
        // Unreachable: `tail_overlap_16` answers for every `i < len`, and the
        // exact-multiple case returned above.
        debug_assert_eq!(i, s.len(), "the overlap tail covers every remainder");
        super::compose_classify_tail(saw_tab, saw_newline, super::AsciiClass::Printable)
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
        // Overlap tail: same protocol as the wasm backend and the x86 kernels
        // (see `tail_overlap_16`) — one load of the final 16 bytes, with the
        // lanes the main loop already scanned masked to 0x00 so
        // `first_high_lane`'s `>= 0x80` test ignores them.  Without this, any
        // length whose remainder is 1..=15 returned `usize::MAX` in release
        // (debug: the assert below fired) and every non-ASCII byte in the tail
        // was silently missed.
        if let Some((base, skip)) = super::tail_overlap_16(s.len(), i) {
            let v = unsafe { vld1q_u8(s.as_ptr().add(base)) };
            let idx = unsafe { vld1q_u8(LANE_IDX.as_ptr()) };
            let keep = unsafe { vcgeq_u8(idx, vdupq_n_u8(skip as u8)) };
            let v = unsafe { vandq_u8(v, keep) };
            return match unsafe { first_high_lane(v) } {
                Some(lane) => base + lane,
                None => usize::MAX,
            };
        }
        // Exact-multiple lengths (32/64/128/256/512 — the sizes a
        // benchmark and a 64-column diagnose row both hit) leave nothing
        // over; re-entering the dispatcher on an empty slice paid the
        // level chain just to compute `usize::MAX`.
        debug_assert_eq!(i, s.len(), "the overlap tail covers every remainder");
        usize::MAX
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    /// The AVX2 classify kernel is unreachable on an AVX-512 host (128+ goes to
    /// AVX-512, 16..127 to SSE2), so without the level override it is only
    /// exercised when CI happens to run on an AVX2-only machine.  Pin it against
    /// the scalar reference across the 128..=300 band on any host that has AVX2.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_classify_kernel_matches_scalar_under_level_override() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // RAII, like `ForceSwarGuard`: a panic or early return inside this test
        // would otherwise leak the override onto the worker thread.
        let _guard = LevelOverrideGuard::new(2);
        let cases: [&[u8]; 8] = [
            b"plain ascii text, nothing to see here but words and spaces and more words",
            b"with\ttab and\nnewline and plenty of printable padding to reach the tail",
            b"\x01control byte forces Other classification at some position in the buffer",
            &[b'a'; 300],
            &[b'\t'; 200],
            &[b'\n'; 200],
            &[b'\x7f'; 190],
            &[0x80u8; 187],
        ];
        for case in cases {
            for len in [128usize, 129, 150, 191, 256, 257, 300] {
                let slice = &case[..len.min(case.len())];
                let got = super::classify_ascii(slice);
                let want = super::classify_scalar(slice);
                assert_eq!(got, want, "len {} case {:?}", slice.len(), &slice[..8]);
            }
        }
    }

    /// Every dispatch tier must agree with the scalar reference, at **every**
    /// length, for **every** predicate and both the vector and forced-SWAR bodies.
    ///
    /// Why this exists: on an AVX-512 host the dispatch never reaches
    /// `classify_ascii_avx2` (128+ goes to AVX-512, 16..127 to SSE2), so that
    /// kernel's correctness depended on which machine CI happened to run on.  The
    /// override makes the tier a test parameter instead; it is clamped to the
    /// detected capability, so a tier this host cannot execute simply reports the
    /// lower one and the assertions stay true.
    ///
    /// The `129..=191` band is covered by the loop, and it is the band the
    /// AVX-512 overlap tail rewrote (every offset within the final 64-byte block),
    /// so the sweep is the tail's full sweep rather than a sample.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_tier_agrees_with_scalar_across_all_lengths_and_predicates() {
        let mut payloads: Vec<Vec<u8>> = Vec::new();
        for len in 0..=300usize {
            payloads.push(vec![b'a'; len]);
            let mut tab = vec![b'a'; len];
            if len > 0 {
                *tab.last_mut().unwrap() = b'\t';
            }
            payloads.push(tab);
            let mut nl = vec![b'a'; len];
            if len > 1 {
                nl[len / 2] = b'\n';
            }
            payloads.push(nl);
            let mut ctl = vec![b'a'; len];
            if len > 0 {
                ctl[0] = 0x01;
            }
            payloads.push(ctl);
            let mut del = vec![b'a'; len];
            if len > 0 {
                *del.last_mut().unwrap() = 0x7F;
            }
            payloads.push(del);
            // High bytes, without which the `next_non_ascii_byte` assertion below would be
            // vacuous (`usize::MAX` for every payload) and the per-tier coverage the sweep
            // exists for would not exist.
            let mut hi_end = vec![b'a'; len];
            if len > 0 {
                *hi_end.last_mut().unwrap() = 0x80;
            }
            payloads.push(hi_end);
            let mut hi_mid = vec![b'a'; len];
            if len > 1 {
                hi_mid[len / 2] = 0xE4; // CJK lead byte
            }
            payloads.push(hi_mid);
            let mut hi_first = vec![b'a'; len];
            if len > 0 {
                hi_first[0] = 0x80;
            }
            payloads.push(hi_first);
        }
        for level in [0u8, 1, 2, 3] {
            let _level = LevelOverrideGuard::new(level);
            for force_swar in [false, true] {
                let _swar = if force_swar {
                    Some(ForceSwarGuard::new())
                } else {
                    None
                };
                for p in &payloads {
                    let want = super::classify_scalar(p);
                    let ctx = format!("len {} level {level} swar {force_swar}", p.len());
                    assert_eq!(super::classify_ascii(p), want, "classify: {ctx}");
                    assert_eq!(
                        super::is_printable_ascii(p),
                        want == super::AsciiClass::Printable,
                        "is_printable_ascii: {ctx}"
                    );
                    assert_eq!(
                        super::is_printable_ascii_or_tab(p),
                        matches!(
                            want,
                            super::AsciiClass::Printable | super::AsciiClass::PrintableOrTab
                        ),
                        "is_printable_ascii_or_tab: {ctx}"
                    );
                    assert_eq!(
                        super::is_printable_ascii_multiline(p),
                        want != super::AsciiClass::Other,
                        "is_printable_ascii_multiline: {ctx}"
                    );
                    // `next_non_ascii_byte` was absent from this sweep, so its SSE2/AVX2
                    // kernels were covered only by a handful of incidental fixed lengths while
                    // the NEON/wasm tails were already shipped with a silent offset bug (see
                    // `first_lane_masked`).  Same per-tier oracle as the predicates above.
                    assert_eq!(
                        super::next_non_ascii_byte(p),
                        p.iter().position(|&b| b >= 0x80).unwrap_or(usize::MAX),
                        "next_non_ascii_byte: {ctx}"
                    );
                }
            }
        }
    }

    /// Restores the previous per-thread level override on drop (nesting-safe, the
    /// same reason `ForceSwarGuard` restores instead of clearing).
    struct LevelOverrideGuard(u8);

    impl LevelOverrideGuard {
        fn new(level: u8) -> Self {
            let prev = super::LEVEL_OVERRIDE.with(std::cell::Cell::get);
            super::set_x86_level_override(level);
            Self(prev)
        }
    }

    impl Drop for LevelOverrideGuard {
        fn drop(&mut self) {
            super::set_x86_level_override(self.0);
        }
    }

    /// `x86_level()` clamps the requested tier to what the host actually has —
    /// the safety property that keeps a tier-3 request on an AVX2-only box from
    /// executing an AVX-512 kernel without its `#[target_feature]` precondition.
    /// Documented but previously unpinned: dropping the `.min()` would only show
    /// up as an illegal instruction on a restricted host.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn level_override_is_clamped_to_detected_capability() {
        // `x86_level_init` *is* the detection (it fills `LEVEL` as well), so
        // reading its return value keeps one probe path instead of a second
        // `is_x86_feature_detected` wrapper that only tests could reach.
        let detected = super::x86_level_init();
        for want in [0u8, 1, 2, 3, 255] {
            super::set_x86_level_override(want);
            // Assert on the real path.  The old version re-derived the clamp from
            // the same `want.min(detected)` expression it compared against, so it
            // pinned only the thread-local round trip — never the contract that
            // keeps a forced tier from exceeding the CPU's capability (the
            // illegal-instruction hazard `set_x86_level_override` documents).
            let expect = if want == 0 {
                detected
            } else {
                want.min(detected)
            };
            assert_eq!(
                super::x86_level(),
                expect,
                "want {want}, detected {detected}"
            );
        }
        super::set_x86_level_override(0);
    }

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
    struct ForceSwarGuard(bool);

    impl ForceSwarGuard {
        fn new() -> Self {
            // Save the previous value: an unconditional `false` on drop would
            // clobber an outer guard if these scopes ever nest.
            Self(SWAR_FORCED.with(|f| f.replace(true)))
        }
    }

    impl Drop for ForceSwarGuard {
        fn drop(&mut self) {
            SWAR_FORCED.with(|f| f.set(self.0));
        }
    }

    /// The early-exit bound must never change an answer.
    ///
    /// `is_printable_ascii` / `is_printable_ascii_or_tab` now stop at the first
    /// chunk whose class already settles them; this pins that against the
    /// unbounded classification over random bytes *and* over crafted inputs whose
    /// tab/newline/offender sits at every interesting offset (the whole point of
    /// the bound is that those offsets end the scan early).
    #[test]
    fn bounded_predicates_match_full_classification() {
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Alphabet weighted towards the classes the predicates actually split.
        const ALPHABET: &[u8] = b"abcXYZ 019\t\n\r\x00\x7f\x80\xff";

        let mut check = |s: &[u8]| {
            let class = super::classify_ascii(s);
            assert_eq!(
                super::is_printable_ascii(s),
                class == super::AsciiClass::Printable,
                "is_printable_ascii disagrees at {s:?}"
            );
            assert_eq!(
                super::is_printable_ascii_or_tab(s),
                class <= super::AsciiClass::PrintableOrTab,
                "is_printable_ascii_or_tab disagrees at {s:?}"
            );
            assert_eq!(
                super::is_printable_ascii_multiline(s),
                class != super::AsciiClass::Other,
                "is_printable_ascii_multiline disagrees at {s:?}"
            );
        };

        for len in 0..=140 {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(ALPHABET[(next() % ALPHABET.len() as u64) as usize]);
            }
            check(&v);
        }
        // One special byte at every offset of a long line, with the rest plain
        // printable: this is the shift that used to force a full scan.
        for special in [b'\t', b'\n', b'\r', b'\x00', 0x80] {
            for at in [
                0usize, 1, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129,
            ] {
                let mut v = vec![b'a'; 130];
                v[at] = special;
                check(&v);
            }
        }
        // All-printable and empty edges.
        check(b"");
        check(b"the quick brown fox");
    }

    /// Every length in `65..=96` (multi-iteration u128 loop) with offenders at chunk seams.
    ///
    /// The lane-exhaustive suite covers `16..=33` and the random corpus `0..64`, so the second and
    /// third iterations of the 16-byte loop — where a seam lands between the chunks — never ran.
    #[test]
    fn first_byte_outside_range_scans_past_the_first_chunk() {
        let (lo, hi) = (b' ', 0x7Eu8);
        for len in 65..=96usize {
            for bad in [0usize, 15, 16, 17, 31, 32, 33, len - 1] {
                if bad >= len {
                    continue;
                }
                let mut s = vec![b'a'; len];
                s[bad] = 0x1F; // below `lo`: an offender in every position class
                let expect = s.iter().position(|b| !(lo..=hi).contains(b)).unwrap();
                assert_eq!(
                    super::first_byte_outside_range(&s, lo, hi),
                    expect,
                    "len {len}, offender at {bad}"
                );
                // And the clean case returns the sentinel.
                let clean = vec![b'a'; len];
                assert_eq!(
                    super::first_byte_outside_range(&clean, lo, hi),
                    usize::MAX,
                    "len {len} clean"
                );
            }
        }
    }

    #[test]
    fn first_byte_outside_range_matches_the_scalar_predicate() {
        // Every length across the SWAR boundary and every alignment offset, over a
        // corpus that mixes in-range runs, high bytes and both boundary values.
        let corpus: &[u8] = b"abcXYZ 019~!\x7f\x80\xff";
        let mut seed = 0x00A1_1CE5_5EED_1234u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        for len in 0..=64usize {
            for _ in 0..8 {
                let bytes: Vec<u8> = (0..len)
                    .map(|_| corpus[(next() % corpus.len() as u64) as usize])
                    .collect();
                for (lo, hi) in [(0x21u8, 0x7Eu8), (0x20, 0x7F), (0x30, 0x39), (0, 0x7F)] {
                    let want = bytes
                        .iter()
                        .position(|&b| !(lo..=hi).contains(&b))
                        .unwrap_or(usize::MAX);
                    assert_eq!(
                        first_byte_outside_range(&bytes, lo, hi),
                        want,
                        "len {len}, range {lo:#04x}..={hi:#04x}, bytes {bytes:?}"
                    );
                }
            }
        }
    }

    /// Lane-exhaustive differential for the SWAR range scan.
    ///
    /// The direct-`trailing_zeros` rewrite rests on one claim: `(word & HI) | lower_bad |
    /// upper_bad` is per-lane **exact** — no in-range byte is flagged, every offender is.
    /// The fuzz above samples that; this positions it: one probe byte in one lane at a
    /// time, across every lane of the u128 body *and* the u64 step, plus two-offender
    /// cases (the lowest set lane must win).
    #[test]
    fn first_byte_outside_range_is_lane_exact_at_every_offset() {
        for (lo, hi) in [(0x20u8, 0x7Eu8), (0x09, 0x0A), (0x00, 0x7F), (0x41, 0x41)] {
            let probes = [
                lo.wrapping_sub(1),
                lo,
                hi,
                hi.wrapping_add(1),
                0x00,
                0x7F,
                0x80,
                0xFF,
            ];
            // 16..=33 exercises the u128 body (16/32) and leaves 1..=15/17-byte
            // remainders for the u64 step and the scalar tail.
            for len in [16usize, 17, 24, 31, 32, 33] {
                for at in 0..len {
                    for &probe in &probes {
                        let mut s = vec![lo; len]; // in-range filler
                        s[at] = probe;
                        let want = s
                            .iter()
                            .position(|&b| !(lo..=hi).contains(&b))
                            .unwrap_or(usize::MAX);
                        assert_eq!(
                            super::first_byte_outside_range(&s, lo, hi),
                            want,
                            "lo {lo:#04x} hi {hi:#04x} len {len} lane {at} probe {probe:#04x}"
                        );
                    }
                }
            }
            // Two offenders in one chunk: the *first* must be reported, which is what the
            // lane order (`trailing_zeros`) buys over "an offender exists".
            for len in [16usize, 32] {
                for a in 0..len {
                    for b in (a + 1)..len {
                        let mut s = vec![lo; len];
                        s[a] = 0xFF;
                        s[b] = 0x80;
                        assert_eq!(
                            super::first_byte_outside_range(&s, lo, hi),
                            a,
                            "lo {lo:#04x} hi {hi:#04x} len {len} offenders {a} {b}"
                        );
                    }
                }
            }
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

    /// The overlap-tail arithmetic shared by NEON and wasm, plus the two mask
    /// forms the backends build from it.
    ///
    /// This is the only part of those backends x86 CI can execute: the kernels
    /// are behind `#[cfg(target_arch)]` and `#[target_feature]`, so the
    /// off-by-one that made both of them return `usize::MAX` for every
    /// non-multiple-of-16 input shipped without a single failing test.  The
    /// masks are re-derived here from the same `skip`, so a sign/direction slip
    /// in either formula fails on x86 too.
    /// The per-lane bitmask → lane-index conversion, exhaustively.
    ///
    /// This is the conversion the wasm `next_non_ascii_byte` tail got wrong: it divided by 8,
    /// copying the SWAR `u64` domain (8 bits per byte lane) onto a `u8x16_bitmask` result (one
    /// bit per lane), so every non-multiple-of-16 length with a high byte in the tail returned
    /// an already-scanned ASCII offset.  wasm cannot execute in CI, so the conversion lives in a
    /// portable helper and is pinned here, on the host.
    #[test]
    fn first_lane_masked_is_the_lane_index_not_a_bit_offset_over_eight() {
        for lane in 0..16u16 {
            for skip in 1..16u16 {
                let want = (lane >= skip).then_some(lane as usize);
                assert_eq!(
                    super::first_lane_masked(1u16 << lane, skip),
                    want,
                    "lane {lane} skip {skip}"
                );
            }
        }
        assert_eq!(super::first_lane_masked(0, 1), None);
        // Scanned lanes are masked off even when they are set: only lane 15 survives skip 3.
        assert_eq!(super::first_lane_masked((1u16 << 15) | 0b111, 3), Some(15));
        // The concrete regression: `len = 20`, high byte at byte 18 is lane 14 of the tail
        // (base 4, skip 12) and must map to offset 18 — the old `/8` produced `base + 1`.
        assert_eq!(
            super::first_lane_masked(1u16 << 14, 12).map(|lane| 4 + lane),
            Some(18),
            "a tail high byte at byte 18 must map to 18, not to base + 14/8"
        );
    }

    #[test]
    fn tail_overlap_16_and_its_masks_cover_every_remainder() {
        for len in 16..=300usize {
            let i = len / 16 * 16;
            match super::tail_overlap_16(len, i) {
                None => assert_eq!(i, len, "len {len}: only exact multiples have no tail"),
                Some((base, skip)) => {
                    assert_eq!(base, len - 16, "len {len}");
                    assert_eq!(
                        base + skip,
                        i,
                        "len {len}: masked prefix is the scanned prefix"
                    );
                    assert_eq!(
                        i + (16 - skip),
                        len,
                        "len {len}: kept lanes are the remainder"
                    );
                    assert!((1..16).contains(&skip), "len {len}: skip {skip}");
                    // wasm: `bitmask & !((1 << skip) - 1)` keeps lanes `skip..16`.
                    let high = |at: usize| 1u16 << at;
                    let keep = !((1u16 << skip) - 1);
                    for at in 0..16usize {
                        let kept = high(at) & keep != 0;
                        assert_eq!(kept, at >= skip, "len {len} lane {at}");
                        // A lane in the remainder is inside the unscanned region.
                        if kept {
                            assert!((base + at) >= i && (base + at) < len, "len {len} lane {at}");
                        }
                    }
                    // NEON: `vcgeq_u8(lane_idx, skip)` keeps the same lanes.
                    for at in 0..16u8 {
                        assert_eq!((at as usize) >= skip, at >= skip as u8, "len {len}");
                    }
                }
            }
        }
    }

    /// The `classify` overlap tail that NEON and wasm now share with x86.
    ///
    /// Both backends compute their masks on the raw final-16-byte vector and then
    /// mask them (`& keep`, or `& !((1 << skip) - 1)` in the bitmask domain).  The
    /// tempting alternative — zeroing the skipped lanes — turns them into NUL, a
    /// genuine offender, so the last assertion pins that trap rather than leaving
    /// it as a comment.
    #[test]
    fn masked_classify_tail_selects_exactly_the_remainder() {
        use super::AsciiClass;
        for len in [17usize, 20, 31, 33, 63, 65, 127, 255] {
            let i = len / 16 * 16;
            let Some((base, skip)) = super::tail_overlap_16(len, i) else {
                unreachable!("non-multiple lengths always have a tail")
            };
            assert_eq!(base + 16, len, "len {len}: the tail is the final 16 bytes");
            let mut tail = [b'a'; 16];
            for at in 0..16usize {
                // `0x00` is the byte a "zero the skipped lanes" implementation
                // fabricates, so using it as the probe makes that trap explicit.
                tail[at] = 0x00;
                let kept = at >= skip;
                let remainder: &[u8] = if kept { &tail[skip..] } else { &[] };
                // Production scalar rule over the bytes the tail covers.
                let want = super::classify_scalar(remainder);
                assert_eq!(
                    want == AsciiClass::Other,
                    kept,
                    "len {len}: lane {at} (skip {skip}) classified iff in the remainder"
                );
                if !kept {
                    // The tempting alternative — zeroing instead of masking — turns
                    // the skipped lanes into NUL, a genuine offender, and would
                    // report `Other` for every input with a remainder.
                    assert_eq!(
                        super::classify_scalar(&tail[..]),
                        AsciiClass::Other,
                        "len {len}: zeroing skipped lanes would report NUL as an offender"
                    );
                }
                tail[at] = b'a';
            }
        }
    }

    /// Lengths that are multiples of no kernel width.  These are exactly the
    /// inputs the NEON/wasm `next_non_ascii_byte` returned `usize::MAX` for in
    /// release (and aborted on in debug) once the re-dispatch tail was deleted:
    /// a high byte in the final `1..=15` bytes was never scanned, so
    /// `char_col`'s sparse-table build jumped the rest of the line.
    #[test]
    fn next_non_ascii_covers_every_non_multiple_length() {
        for len in [17usize, 20, 31, 33, 63, 65, 127] {
            let clean = vec![b'a'; len];
            assert_eq!(super::next_non_ascii_byte(&clean), usize::MAX, "len {len}");
            for at in [0usize, len / 2, len - 2, len - 1] {
                let mut v = clean.clone();
                v[at] = 0x80;
                assert_eq!(super::next_non_ascii_byte(&v), at, "len {len} at {at}");
            }
            // The tail-only case: nothing but the last byte.
            let mut tail = clean.clone();
            tail[len - 1] = 0xFF;
            assert_eq!(super::next_non_ascii_byte(&tail), len - 1, "len {len}");
        }
    }

    /// The forced-SWAR tier must not execute the SSE2 gate for the 16..127 band:
    /// without the `!swar_forced()` conjunct, `swar_forced/printable/127`
    /// benchmarked the SSE2 gate (0.66 ns) instead of the SWAR body (6.3 ns),
    /// silently invalidating every predicate measurement in that band.
    #[test]
    fn gate_band_declines_under_the_forced_swar_tier() {
        let _swar = ForceSwarGuard::new();
        for len in [16usize, 31, 64, 127] {
            let s = vec![b'a'; len];
            assert_eq!(super::gate_band_strict(&s), None, "len {len}");
            assert_eq!(super::gate_band_tab(&s), None, "len {len}");
            assert_eq!(super::gate_band_multiline(&s), None, "len {len}");
        }
    }

    /// ...and it *is* served when the tier is not forced, so the assertion above
    /// cannot pass by the band being dead in every configuration.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gate_band_serves_the_16_to_127_band_unforced() {
        assert!(
            LEVEL_OVERRIDE.with(|c| c.get()) == 0,
            "another test left the level override set"
        );
        for len in [16usize, 31, 63, 127] {
            let s = vec![b'a'; len];
            assert_eq!(super::gate_band_strict(&s), Some(true), "len {len}");
            assert_eq!(super::gate_band_tab(&s), Some(true), "len {len}");
            let mut del = s.clone();
            del[len - 1] = 0x7F;
            assert_eq!(super::gate_band_strict(&del), Some(false), "len {len}");
            // `_or_tab` accepts the tab `_strict` rejects.
            let mut tab = s.clone();
            tab[len - 1] = b'\t';
            assert_eq!(super::gate_band_strict(&tab), Some(false), "len {len}");
            assert_eq!(super::gate_band_tab(&tab), Some(true), "len {len}");
        }
        // Outside the *narrow* band the answer belongs to another arm: 128 is the
        // first length of the wide AVX-512 gate (when the host has it) and 15 is
        // below the SSE2 floor, where the fused classify path decides.
        let wide = super::x86_level() >= 3;
        assert_eq!(
            super::gate_band_strict(&vec![b'a'; 128]),
            wide.then_some(true)
        );
        assert_eq!(super::gate_band_strict(&vec![b'a'; 15]), None);
    }

    /// The `andnot(a, b) = a & !b` convention is stated host-side, not only in a wasm-only test.
    ///
    /// wasm/NEON `andnot` is `a & !b`; x86's `_mm256_andnot_si256(a, b)` is `!a & b` — the classic
    /// port-bug incubator.  `andnot_is_a_and_not_b` lives in the wasm-gated module, which this
    /// repo's CI never executes, so the *direction* itself was pinned nowhere that runs.  This test
    /// pins the direction on plain integers (any backend whose `andnot` complements the wrong
    /// operand would fail its own `masked_classify_tail_*` test, but the direction deserves one
    /// host-side statement rather than none).
    #[test]
    fn andnot_convention_is_a_and_not_b() {
        let (a, b) = (0xF0u8, 0x0Fu8);
        assert_eq!(a & !b, 0xF0, "the module's convention: a & !b");
        assert_eq!(
            !a & b,
            0x0F,
            "x86's operand order, which this module does not use"
        );
        assert_ne!(a & !b, !a & b, "the conventions must differ on this pair");
        // A distinguishing pair for masks: keep `b`'s low nibble, clear its high nibble.
        let (m, ctrl) = (0xFFu8, 0x0Fu8);
        assert_eq!(m & !ctrl, 0xF0);
        assert_eq!(!m & ctrl, 0x00);
    }

    /// The level-2 wide-band **decline** is a contract, not an accident.
    ///
    /// `gate_band` serves `16..128` from `gate_sse2` at any x86 level and `>= 128` only on level 3
    /// (`gate_avx512`); on an AVX2-only host the wide band must return `None` so the fused
    /// `classify_ascii_avx2` path serves it ("there is no AVX2 gate kernel", as the doc says).
    /// Nothing pinned that boundary: the existing test derives `wide` from the *detected* level, so
    /// an accidental widening of the SSE2 arm past `CLASSIFY_AVX2_MIN_LEN` would route 128..∞
    /// through a 16-byte kernel and every assertion would still pass.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gate_band_declines_the_wide_band_on_avx2_only_hosts() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return; // an SSE2-only host has no AVX2 kernel to decline *toward*
        }
        let _level = LevelOverrideGuard::new(2);
        for len in [1usize, 16, 63, 64, 127, 128, 129, 255, 512] {
            let clean = vec![b'a'; len];
            let expect = |answer: bool| (16..128).contains(&len).then_some(answer);
            assert_eq!(super::gate_band_strict(&clean), expect(true), "len {len}");
            let mut del = clean.clone();
            del[len - 1] = 0x7F;
            assert_eq!(super::gate_band_strict(&del), expect(false), "len {len}");
        }
    }

    /// The wide (`>= 128`) gate band, whichever tier the host has: `Some(..)` must always agree
    /// with the exact byte-set rule, and on an AVX-512 host it must actually be taken (otherwise the
    /// arm is dead and this test would pass while covering nothing).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gate_band_wide_band_agrees_with_the_scalar_rule() {
        assert!(
            LEVEL_OVERRIDE.with(|c| c.get()) == 0,
            "another test left the level override set"
        );
        // `x86_level()` is the detected tier here (override is 0, SWAR not forced).
        let wide = super::x86_level() >= 3;
        for len in [128usize, 129, 191, 255, 256, 333, 512, 1024] {
            let expect = |answer: bool| wide.then_some(answer);
            let clean = vec![b'a'; len];
            assert_eq!(super::gate_band_strict(&clean), expect(true), "len {len}");
            assert_eq!(super::gate_band_tab(&clean), expect(true), "len {len}");
            assert_eq!(
                super::gate_band_multiline(&clean),
                expect(true),
                "len {len}"
            );

            let mut del = clean.clone();
            del[len - len / 3] = 0x7F;
            assert_eq!(super::gate_band_strict(&del), expect(false), "len {len}");

            let mut tab = clean.clone();
            tab[len - 1] = b'\t';
            assert_eq!(super::gate_band_strict(&tab), expect(false), "len {len}");
            assert_eq!(super::gate_band_tab(&tab), expect(true), "len {len}");
            assert_eq!(super::gate_band_multiline(&tab), expect(true), "len {len}");

            let mut nl = clean.clone();
            nl[0] = b'\n';
            assert_eq!(super::gate_band_tab(&nl), expect(false), "len {len}");
            assert_eq!(super::gate_band_multiline(&nl), expect(true), "len {len}");

            // A high byte is an offender for every gate (the signed compare in the
            // kernel and the scalar rule must agree on it).
            let mut high = clean.clone();
            high[len - 1] = 0xC3;
            assert_eq!(
                super::gate_band_multiline(&high),
                expect(false),
                "len {len}"
            );
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
        // `len` must stay ≥ 20: `skip = 3` leaves `len - 3` bytes, and a slice
        // below 16 would take the scalar short-circuit — the case this test is
        // named for is the *vector* path.
        for &len in &[32usize, 64, 128, 256, 512] {
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
        for at in [
            8, 15, 16, 31, 32, 47, 48, 63, 64, 127, 128, 255, 256, 383, 512, 600, 1023,
        ] {
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

    /// The three boolean gates must answer the same byte sets the 4-way
    /// classifier defines — including in the 16..127 band, which now runs the
    /// dedicated SSE2 kernel (`gate_sse2`, one violation mask per chunk instead of
    /// three).  The reference here is derived *directly* from the sets, not from
    /// `classify_ascii`, so a shared misreading cannot hide.
    #[test]
    fn gates_agree_with_the_byte_sets_across_all_lengths() {
        // The reference is the **production** rule, not a third copy of it.  This test used
        // to re-derive the same `match` locally, which is precisely the "two copies of
        // one rule drift silently" failure the module header preaches: an edit that
        // admitted `\r` in `byte_in_gate` would have passed every assertion here.  The
        // const-generic form exists so the rule has one home and this test is its
        // consumer.
        fn ok(b: u8, tab_ok: bool, nl_ok: bool) -> bool {
            match (tab_ok, nl_ok) {
                (false, false) => super::byte_in_gate::<false, false>(b),
                (true, false) => super::byte_in_gate::<true, false>(b),
                (true, true) => super::byte_in_gate::<true, true>(b),
                (false, true) => super::byte_in_gate::<false, true>(b),
            }
        }
        let alphabets: [&[u8]; 5] = [
            b"abcXYZ019 _-./\t",
            b"\x00\x01\x09\x0a\x0d\x1f\x20\x7e\x7f",
            b"\x80\xc3\xa9\xe4\xbd\xa0\xff",
            b"a\t\tb\nc\x1b[02m\x08",
            b"~",
        ];
        for alpha in alphabets {
            // 0..=200 covers the scalar band (<16), the dedicated SSE2 band
            // (16..128) and the wider AVX2/AVX-512 route (>=128).
            for len in 0..=200usize {
                let s: Vec<u8> = (0..len).map(|i| alpha[i % alpha.len()]).collect();
                let strict = s.iter().all(|&b| ok(b, false, false));
                let tab_ok = s.iter().all(|&b| ok(b, true, false));
                let multi = s.iter().all(|&b| ok(b, true, true));
                assert_eq!(
                    super::is_printable_ascii(&s),
                    strict,
                    "strict len {len} alphabet {:?}",
                    String::from_utf8_lossy(alpha)
                );
                assert_eq!(
                    super::is_printable_ascii_or_tab(&s),
                    tab_ok,
                    "or_tab len {len} alphabet {:?}",
                    String::from_utf8_lossy(alpha)
                );
                assert_eq!(
                    super::is_printable_ascii_multiline(&s),
                    multi,
                    "multiline len {len} alphabet {:?}",
                    String::from_utf8_lossy(alpha)
                );
            }
        }
        // One long clean input through the wide path, and one long dirty input.
        let clean = vec![b'x'; 4096];
        assert!(super::is_printable_ascii(&clean));
        let mut dirty = clean.clone();
        dirty[3000] = b'\t';
        assert!(!super::is_printable_ascii(&dirty));
        assert!(super::is_printable_ascii_or_tab(&dirty));
    }

    /// `next_non_ascii_byte` on exact kernel widths, and on the overlapping
    /// masked tail.  Both are places where the scan steps *past* the loop and the
    /// only evidence used to be a benchmark that under-reported the wide tiers.
    #[test]
    fn next_non_ascii_covers_exact_multiples_and_the_masked_tail() {
        for len in [16usize, 32, 48, 63, 64, 65, 127, 128, 256, 512, 1024] {
            let clean = vec![b'a'; len];
            assert_eq!(
                super::next_non_ascii_byte(&clean),
                usize::MAX,
                "pure ASCII of len {len} must report MAX"
            );
            // The last byte is the worst case for a masked tail: every lane before
            // it is masked in, so an off-by-one in `keep` shows up here.
            let mut dirty = clean.clone();
            dirty[len - 1] = 0xC3;
            assert_eq!(
                super::next_non_ascii_byte(&dirty),
                len - 1,
                "tail byte of a len-{len} buffer"
            );
            // First byte, and one in the middle of the final chunk.
            let mut first = clean.clone();
            first[0] = 0xFF;
            assert_eq!(super::next_non_ascii_byte(&first), 0);
            if len >= 20 {
                let mut mid = clean.clone();
                mid[len - 9] = 0xE4;
                assert_eq!(super::next_non_ascii_byte(&mid), len - 9);
            }
        }
    }

    /// The `u128` SWAR body is production-unreachable on vector targets, so its
    /// multi-chunk loop and its `word8` tail seam were only covered statistically
    /// (the forced-SWAR suites cap at ~80 bytes).  Walk the lengths around the
    /// 16-byte chunk boundary and the 8-byte tail explicitly.
    #[test]
    fn forced_swar_wide_loop_and_tail_seam_match_scalar() {
        let _swar = ForceSwarGuard::new();
        let reference = |s: &[u8]| -> AsciiClass {
            let mut tab = false;
            let mut nl = false;
            for &b in s {
                if b == b'\t' {
                    tab = true;
                } else if b == b'\n' {
                    nl = true;
                } else if !(0x20..=0x7E).contains(&b) {
                    return AsciiClass::Other;
                }
            }
            if nl {
                AsciiClass::PrintableOrTabNewline
            } else if tab {
                AsciiClass::PrintableOrTab
            } else {
                AsciiClass::Printable
            }
        };
        for len in [128usize, 129, 135, 136, 143, 144, 151, 152, 255, 256, 257] {
            for clean in [true, false] {
                let mut s: Vec<u8> = (0..len)
                    .map(|i| if i % 7 == 0 { b' ' } else { b'x' })
                    .collect();
                if !clean {
                    // A tab in the middle plus a DEL at the very end: the tab
                    // exercises the wide loop's flags, the DEL lands in the word8
                    // tail.
                    s[len / 2] = b'\t';
                    s[len - 1] = 0x7F;
                }
                assert_eq!(
                    super::classify_ascii(&s),
                    reference(&s),
                    "len {len} clean={clean}"
                );
                assert_eq!(
                    super::is_printable_ascii(&s),
                    reference(&s) == AsciiClass::Printable,
                    "strict len {len} clean={clean}"
                );
            }
        }
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
        // 256 covers the second half of the unrolled 256-byte AVX-512 loop.
        for lane in 0..256 {
            let mut v = vec![b'a'; 256];
            v[lane] = 0x80;
            assert_eq!(next_non_ascii_byte(&v), Some(lane), "lane {lane}");
        }
    }
}
