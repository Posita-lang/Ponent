# Algorithmic optimization opportunities

Short version: the codebase is already near the constant-factor floor in most hot loops (SWAR ASCII predicates, `memchr`/`memchr3`, direct-address tables, cached labels, LRU’d line indexes, min-plus DP in `wrap.rs`, interval-graph lane assignment, `SmallVec`-first storage throughout). What remains are a handful of real constant-factor wins, mostly where an inner loop still carries an extra branch, an extra scan, or a `core::fmt` call that the codebase has already built a primitive to avoid. Below, ordered by expected impact × implementation risk.

---

## 1. `glyph.rs` — highlighter: `CLASS` is `[u8; 128]`, forcing two branches per byte

**File:** `glyph.rs`, `highlight_code_impl`

```rust
// outer loop
if b >= 0x80 {
    next_char(bytes, &mut i);
    continue;
}
match CLASS[b as usize] { ... }

// PASS inner loop
while i < len {
    let nb = bytes[i];
    if nb >= 0x80 || CLASS[nb as usize] != PASS {   // ← two branches
        break;
    }
    i += 1;
}
```

The PASS arm is the hottest (verbatim `= `, ` + `, `(x, y)` runs dominate real source). The separate `nb >= 0x80` test is a second data-dependent branch on every byte, and it also prevents LLVM from vectorising the run scan into a single `CLASS[nb] != PASS` check.

**Fix:** make the table 256 entries and add a `NONASCII` class:

```rust
const NONASCII: u8 = 6;
const CLASS: [u8; 256] = build_class();  // 0x80..=0xFF → NONASCII

const fn build_class() -> [u8; 256] {
    let mut t = [PASS; 256];
    let mut i = 0;
    while i < 128 {
        t[i] = match i as u8 {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => IDENT,
            b'"'  => STRING,
            b'\'' => CHAR,
            b'&' | b'*' => OPER,
            b'`'  => TICK,
            _     => PASS,
        };
        i += 1;
    }
    while i < 256 { t[i] = NONASCII; i += 1; }
    t
}
```

The outer `if b >= 0x80` disappears into `match CLASS[b as usize] { … NONASCII => next_char(…) }`. The PASS inner loop becomes `while i < len && CLASS[bytes[i] as usize] == PASS { i += 1 }` — a single condition, byte-indexed, which LLVM lowers to vectorised compare/`find` for the run. This is the clearest remaining win.

---

## 2. `chain.rs` — `ChainSuffix::write_into` still routes through `core::fmt`

**File:** `chain.rs`

```rust
let _ = write!(out, ": {}:{}", line + 1, col + 1);
```

Every chain entry pays the `core::fmt` machinery (`Formatter`, `Arguments`, trait dispatch) for two `usize`s, while `renderers::fmt::push_uint` exists precisely to avoid it. Same pattern for `{file}:{}:{}`:

```rust
out.push_str(": ");
if let Some(url) = url {
    out.push_str("\x1b]8;;"); out.push_str(url); out.push_str("\x1b\\");
    push_uint(out, (line + 1) as u64);
    // file is already a &str, push_str it before the `:`…
}
```

`ChainSuffix::write_into` is not the hottest path in the crate, but the substitution is free (the primitives already exist and are already used by `glyph`/`json`/`short` for exactly this), removes a `Formatter` allocation-free but non-trivial `format_args!` site, and removes a source of “`core::fmt` bypassed in one place but not this one” inconsistency.

---

## 3. `glyph.rs` — tab expansion inside `render_line_merged_with` does `push(' ')` per cell

**File:** `glyph.rs`, non-ASCII branch

```rust
if g == "\t" {
    let w = grapheme_width_cached(g, col, tab_width, &mut cache);
    for _ in 0..w {
        rendered.push(' ');
    }
}
```

`push(' ')` goes through `String::push`, which checks capacity once per cell. `fmt::push_spaces` already batches 64 bytes per iteration and would reduce this to one `push_str`. For a paragraph indented with tabs (the exact i18n case that motivated `DEFAULT_TAB_WIDTH`), this fires once per tab per row. Free win:

```rust
if g == "\t" {
    let w = grapheme_width_cached(g, col, tab_width, &mut cache);
    super::renderers::fmt::push_spaces(rendered, w);
}
```

(Identical to the `push_spaces` already used in `write_source_section`’s line-number gutter, so this is pure consistency.)

---

## 4. `wrap.rs` — the single-line fast path runs `measure_width` for ASCII over-wide inputs

**File:** `wrap.rs`, `wrap_optimal`

```rust
if !trimmed.is_empty()
    && memchr::memchr(b'\t', trimmed.as_bytes()).is_none()
    && !edges_have_non_ascii_ws(trimmed)
    && (trimmed.len() <= max_width
        || measure_width(trimmed, tab_width) <= max_width)
```

For an ASCII string whose byte length already exceeds `max_width` (so `trimmed.len() <= max_width` fails), `measure_width` runs. Its own ASCII fast path is `is_printable_ascii` (one SWAR pass) followed by `s.len()` — but the result is **known to be `> max_width`** from the byte-length check alone, so the SWAR pass is pure overhead before the slow DP path is taken anyway.

The clean fix is to make the two facts share one scan:

```rust
let fits = if trimmed.len() <= max_width {
    true
} else if is_printable_ascii(trimmed.as_bytes()) {
    false                              // ASCII ⇒ byte-len == display width
} else {
    measure_width(trimmed, tab_width) <= max_width
};
if !trimmed.is_empty() && … && fits { return smallvec![Cow::Borrowed(trimmed)]; }
```

For a long ASCII message that must wrap, this saves one `is_printable_ascii` SWAR scan of the whole message, which is exactly the shape of `wrap_optimal`’s long prose path in its own perf harness.

---

## 5. `LineIndex::new` — three full passes over the source

**File:** `mod.rs`, `LineIndex::new`

```rust
let n_newlines = memchr::memchr_iter(b'\n', source.as_bytes()).count();
let mut starts = Vec::with_capacity(n_newlines + 1);
starts.push(0);
starts.extend(memchr::memchr_iter(b'\n', source.as_bytes()).map(|i| i + 1));
…
ascii: source.is_ascii(),
```

Three passes: `memchr_iter` count, `memchr_iter` fill, `is_ascii` SWAR. The two `memchr_iter` scans are unavoidable for exact-capacity allocation (`memchr_iter` has no O(1) count), but the third (`is_ascii`) can be eliminated with a small `const fn`-friendly helper — a `u64` chunk loop that does both `memchr`’s “any newline” test and “any high bit” test in one pass:

```rust
fn scan(source: &str) -> (Vec<usize>, bool) {
    const HI: u64 = 0x8080_8080_8080_8080;
    const NL: u64 = 0x0A0A_0A0A_0A0A_0A0A;
    const ONES: u64 = 0x0101_0101_0101_0101;

    let bytes = source.as_bytes();
    let mut starts = Vec::with_capacity(16);   // grow-and-shrink; amortised O(n)
    starts.push(0);
    let mut ascii = true;
    let mut i = 0;
    while i + 8 <= bytes.len() {
        let x = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        ascii &= (x & HI) == 0;
        let q = x ^ NL;
        let mut mask = (q.wrapping_sub(ONES)) & !q & HI;   // has_zero(x^NL) = has_nl
        while mask != 0 {
            starts.push(i + (mask.trailing_zeros() as usize / 8) + 1);
            mask &= mask - 1;
        }
        i += 8;
    }
    // tail …
    (starts, ascii)
}
```

The `Vec::with_capacity(16)` grows; the previous exact-capacity pass was a *log-pass* saving. If exact capacity matters more than one fewer scan, use `n_newlines` from the combined pass (the combined scan already sees the count) — you pay one extra full pass but at `u64`-chunk granularity instead of two `memchr` SIMD passes. Benchmark both; the exact-capacity variant is `memchr`’s tuned kernel and hard to beat for the fill; the win here is really the `is_ascii` elimination. For small source files this is noise; for a ≥ 1 MB source it is measurable, and `perf_many_diags_one_source` is the harness to measure against.

---

## 6. `error_code.rs` — `ErrCode::url()` is per-call `format_compact!`

**File:** `error_code.rs`

`url_ansi` already caches by `code_slot` gated on `EXPLAIN_PORT`. But `url()` itself has no cache: a tool that enumerates errors and calls `url()` per diagnostic (outside the `url_ansi` cache path) rebuilds `format_compact!("https://doc.posita-lang.org/error_codes/{}.html", …)` for every one. The `url_ansi` cache could be split so the inner `url` is also memoised, or a parallel `DOC_URL_CACHE` keyed by `code_slot` (same `OnceLock<Box<[OnceLock<CompactString>]>>` shape) avoids the allocation entirely. Not `url_ansi` — `url()`.

This is a smaller win than the earlier ones because the primary consumer (`url_ansi`) already caches, but `ErrCode::url()` is `pub` and the natural thing for tooling to reach for.

---

## 7. `glyph.rs` — `render_line_merged_with`: heap vs. linear for small active sets (consider)

**File:** `glyph.rs`, non-ASCII sweep

The active-set sweep uses `BinaryHeap<Reverse<(usize, usize)>>`. The comment on `LabelWindow` documents exactly this trade-off (fast path for ≤1, burst otherwise), but the grapheme loop doesn’t replicate it: for the overwhelmingly common case of 0–2 active ranges, a linear `SmallVec` scan or a single `(start, end)` tuple avoids heap push/pop entirely. Concretely:

```rust
// inside the per-grapheme loop, only when ranges.len() <= 4 (a typical row)
if k == 0 { … fast path: nothing to resolve, just measure + emit … }
else if k == 1 { … single (start, end): two compares per grapheme … }
else { … heap path as today … }
```

The `active_buf` `BinaryHeap` is already threaded as caller scratch, so the heap path stays for k > threshold. Whether this wins depends on `k` distribution in practice — worth benchmarking `perf_many_labels_spans` and `perf_long_single_line` with the two branch arms added, because at k ≈ 1 the heap’s `Reverse<(usize, usize)>` sift is more instructions than a direct compare.

---

## 8. Micro, but free

These are pure consistency fixes that remove `core::fmt` and extra scans the codebase already avoids elsewhere:

| Location | Change |
|---|---|
| `glyph.rs` `write_header` | The `all_codes_visible` `Cow` is computed for `visible_width` and then moved into `all_codes_display`; in the hyperlink branch it is `Cow::Owned` and re-scanned by the OSC 8 loop. For `all_codes.len() == 1` the value is already a borrow — no action needed; for the ≥ 2 related-error case, joining once and pushing into `out` directly (as the render path already does elsewhere) avoids the intermediate `String`. |
| `error_code.rs` `explain_url_with_port` | The doc-URL branch rebuilds `"https://doc.posita-lang.org/error_codes/"` (a constant 38-byte prefix) per call. Splitting it into a `const DOC_URL_PREFIX: &str` and a `push_str(prefix) + push_str(encoded) + push_str(".html")` saves the `format_compact!` macro plumbing; `encode_url_path_segment` already returns `Cow::Borrowed` for canonical codes, so the whole thing can be one `String::with_capacity` (or `CompactString` when it fits inline). |
| `renderers/short.rs` `push_flattened` | The first `memchr2` is already a fast scan; the `while` re-`memchr2`s per segment. Fine as is, but note `out.push_str(&msg[start..end])` uses `push_str` — good. No change needed; listing to rule out. |
| `render_source_html_into` and `write_source_section` | Both `push_str(&self.s.dim)` etc. per row. LLVM inlines these constant-length pushes to memcpys; no action. |

---

## What not to change

Several things look optimizable but are already optimal for the data shapes they see, and the comments pin the rationale — leave them:

- **`is_printable_ascii_multiline`’s 256-byte LUT** (ascii.rs): a SWAR formulation would need to mask out `0x09`/`0x0A` from a `hasless(x, 0x20)` and add a `has_zero(x ^ 0x7F)` — the LUT is one L1 load per byte and wins.
- **`is_printable_ascii` SWAR body** (ascii.rs): the two-mask form is minimal; the tail is already `iter().all`.
- **`next_json_escape_byte`’s `hasless`/`has_zero`** (json.rs): single-pass SWAR at 8 B/iter; no gain available.
- **`next_html_escape_byte`’s `memchr3` + `memchr2`** (html.rs): AVX2-tuned kernels beat a hand-written 5-char SWAR; don’t combine.
- **`LabelWindow::advance`** (glyph.rs): fast path for ≤1/≤1 is already there; the burst merge is O(active + new + expired) and only fires when the row genuinely changes membership.
- **`wrap_dp`’s min-plus convolution** (wrap.rs): the O(n·k) inner loop with the precomputed `best_cost`/`best_src` is already the textbook reduction; the last-line-free intrinsic and the `JUMP` table are pinned by tests. Advanced-DP evaluation: per-class Monge permits four independent Larmore–Schieber instances, but the block matrix is not Monge for the non-separable `JUMP` table, and the per-comparison constant plus the pinned `i_lo` pointer keep the monotone window ahead in the diagnostic regime. The DP is a DAG shortest path, so an LP would not reduce the state space; an LP fast path would have to check integrality and fall back to the DP on fractional solutions. Decision recorded in `research/wrap-dp/README.md`.
- **`FixSet::apply`** (fix.rs): forward-pass construction with `String::with_capacity(source.len() + extra)`; the old reverse-`replace_range` was the quadratic it replaced.
- **`FilesSeen`** (emitter/mod.rs): inline linear scan for ≤4 files, spill to `FxHashSet` after; the 1–2-file case avoids all hashing.
- **`LineIndex::char_col`’s sparse table** (mod.rs): amortises O(K) build across all M spans on one line; the alternative (per-span `chars().count()`) is the quadratic it replaced.

---

## Summary

If you want to spend engineering effort in order of return:

1. **`highlight_code_impl` — 256-entry `CLASS`.** Highest leverage per line of diff; removes a per-byte branch from the hottest arm.
2. **`ChainSuffix::write_into` and the tab `push_spaces` in `render_line_merged_with`.** Both are “use the primitive the crate already has” one-liners.
3. **`wrap_optimal` — `is_printable_ascii` branch on the ASCII over-wide case.** Removes a redundant SWAR pass on the slow path.
4. **`LineIndex::new` — fused newline/ASCII scan.** Real but only pays for large sources; benchmark against `perf_many_diags_one_source`.
5. **`url()` doc-URL cache.** Only if tooling actually calls `url()` hot; `url_ansi` already caches.

Everything beyond that is already at or near the theoretical floor for the data shapes involved, and the remaining comments in the code document why.
