//! Lexer micro-benchmarks with same-binary old-vs-new references.
//!
//! The root crate is a binary with an empty lib target, so this bench
//! `#[path]`-includes the lexer modules directly (same trick as the octagon
//! bench).  Wall-clock timings on this box drift >1.7x between build/session
//! cycles, so every claim is an A/B *inside one binary*: the `old_*`
//! functions are frozen copies of the pre-change algorithms, benched next to
//! the real (`pub(crate)`) functions in the same criterion run.
//!
//! Groups:
//!   block_comment/*  — old raw scan vs the real logos path (the real side
//!                      carries DFA/token overhead, so the comparison is
//!                      biased *against* the new code)
//!   int_parse/*      — frozen old checked loop vs real `parse_int_literal`
//!   suffixed_int_parse/* — frozen old `replace`+`from_str_radix` (the old
//!                      `expr.rs` path) vs real `parse_suffixed_int_literal`
//!   string_parse/*   — frozen old per-char loop vs real `parse_string_literal`
//!   lexer_e2e/*      — full-source lexing on comment/numeric/escape corpora

#[path = "../src/lexer/mod.rs"]
mod lexer;
#[path = "../src/symbol.rs"]
mod symbol;
#[path = "../src/sync.rs"]
mod sync;

/// Minimal mirror of `crate::ast::IntLit` — the lexer modules' only AST
/// dependency.  The bench crate has no `ast` module and including the real
/// one would drag in the whole AST; the shape must match the real enum
/// (`Small(i128)` / `Large(Box<BigInt>)`) so the scanners' return values
/// are usable here.  `Clone`/`Debug`/`PartialEq` are required by the
/// `Token` derives.
mod ast {
    use num_bigint::BigInt;

    #[derive(Debug, Clone, PartialEq)]
    pub enum IntLit<'input> {
        Small(i128),
        /// Values above `i128::MAX` (up to `u128::MAX`).  Arena-allocated
        /// so the AST stays `Copy`.
        Large(&'input BigInt),
    }

    // `lexer/display.rs` renders token payloads for diagnostics.
    impl<'input> std::fmt::Display for IntLit<'input> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                IntLit::Small(n) => write!(f, "{n}"),
                IntLit::Large(b) => write!(f, "{b}"),
            }
        }
    }
}

/// i128 view of a scanner result for the accumulation loops (`Large`
/// values — above `i128::MAX`, i.e. the `UInt<128>` top half — fold to 0
/// in the bench totals).
fn lit_i128(v: &ast::IntLit<'_>) -> i128 {
    match v {
        ast::IntLit::Small(n) => *n,
        ast::IntLit::Large(_) => 0,
    }
}

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use lexer::Token;
use lexer::literal::{
    LitError, parse_int_literal, parse_string_literal, parse_suffixed_int_literal,
};

// ── Frozen pre-change reference algorithms ─────────────────────────────

/// Old `block_comment` body: byte-at-a-time loop with two branches per
/// byte.  `body` is the text between `/*` and the closing `*/`; returns the
/// total bytes consumed (body + closing `*/`), mirroring `lex.bump`.
fn old_block_comment_scan(body: &[u8]) -> usize {
    let mut rest = body;
    let mut consumed = 0usize;
    let mut depth = 1usize;
    while rest.len() >= 2 {
        let (a, b) = (rest[0], rest[1]);
        if a == b'*' && b == b'/' {
            depth -= 1;
            consumed += 2;
            rest = &rest[2..];
            if depth == 0 {
                return consumed;
            }
        } else if a == b'/' && b == b'*' {
            depth += 1;
            consumed += 2;
            rest = &rest[2..];
        } else {
            consumed += 1;
            rest = &rest[1..];
        }
    }
    consumed + rest.len()
}

/// Exact copy of the pre-change `parse_int_literal` body — including the
/// eager `ok_or(LitError::from(..))` intern per digit, which is part of the
/// measured baseline (it turned out to dominate the arithmetic).
fn old_parse_int_literal(s: &str, radix: u32, overflow_msg: &str) -> Result<i128, LitError> {
    let overflow_err = LitError::from(overflow_msg);
    let mut acc: i128 = 0;
    for c in s.chars() {
        if c == '_' {
            continue;
        }
        let d = c
            .to_digit(radix)
            .ok_or(LitError::from("invalid digit in literal"))? as i128;
        acc = acc
            .checked_mul(radix as i128)
            .and_then(|v| v.checked_add(d))
            .ok_or(overflow_err)?;
    }
    Ok(acc)
}

/// Old `parse_string_literal` body: per-char push loop.  Only the simple
/// escapes are decoded — the `\x` / `\u` arms are byte-identical in both
/// versions (they delegate to the same helpers), so they are not part of
/// this A/B.
fn old_parse_string_literal(s: &str) -> Result<String, ()> {
    let inner = &s[1..s.len() - 1];
    if !inner.contains('\\') {
        return Ok(inner.to_string());
    }
    let mut result = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let next = chars.next().ok_or(())?;
            match next {
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                '\\' => result.push('\\'),
                '"' => result.push('"'),
                '\'' => result.push('\''),
                '0' => result.push('\0'),
                _ => return Err(()),
            }
        } else {
            result.push(c);
        }
    }
    Ok(result)
}

/// Old `expr.rs` suffixed-literal body (all three prefix branches): the
/// digit run goes through `replace('_', "")` — one heap `String` — and is
/// then re-scanned by `from_str_radix` / `parse`.  `None` mirrors the old
/// `Expr::Error` fallback.  The real code derived the radix from the token
/// prefix; here it is a parameter (equivalent for this corpus).
fn old_suffixed_int_literal(s: &str, radix: u32) -> Option<i128> {
    let body = match radix {
        16 | 2 => &s[2..],
        _ => s,
    };
    let num_part = body
        .split(|c: char| c == 'i' || c == 'u')
        .next()
        .unwrap_or("0")
        .replace('_', "");
    match radix {
        16 => i128::from_str_radix(&num_part, 16).ok(),
        2 => i128::from_str_radix(&num_part, 2).ok(),
        _ => num_part.parse::<i128>().ok(),
    }
}

// ── Corpora ────────────────────────────────────────────────────────────

/// License-header-shaped comment body: ~9 KiB of prose, sparse `*`.
fn header_body() -> String {
    let mut s = String::new();
    for i in 0..200 {
        s.push_str("* Section ");
        s.push_str(&i.to_string());
        s.push_str(": the quick brown fox jumps over the lazy dog.\n");
    }
    s
}

/// Marker-dense body: every few bytes is a `*` or `/` (worst case for the
/// skip loop — memchr2 finds a hit immediately, every time).
fn marker_body() -> String {
    "* // * * // ".repeat(300)
}

/// (radix, slice) pairs for the integer A/B: decimal/hex/binary, with and
/// without underscores, plus one over-gate literal per radix so the checked
/// slow path is exercised too.
fn int_literals() -> Vec<(u32, &'static str)> {
    let mut v = Vec::new();
    for _ in 0..64 {
        v.push((10u32, "1234567"));
        v.push((10, "1_000_000"));
        v.push((10, "999999999999999999999999999999999999999"));
        // u128::MAX — slow path, lands in `IntLit::Large` (> i128::MAX).
        v.push((10, "340282366920938463463374607431768211455"));
        v.push((16, "DEADBEEF"));
        v.push((16, "1234_5678_9ABC_DEF0"));
        v.push((2, "101101"));
        v.push((2, "1111_0000_1010_0101"));
    }
    v
}

/// (slice, radix) pairs for the suffixed-literal A/B: decimal/hex/binary,
/// with and without `_`, both `i`/`u` markers, plus over-gate and overflow
/// shapes so the checked slow path is exercised too.
fn suffixed_literals() -> Vec<(&'static str, u32)> {
    let mut v = Vec::new();
    for _ in 0..64 {
        v.push(("42i32", 10u32));
        v.push(("1_000_000u64", 10));
        v.push(("111111111111111111111111111111111111111i8", 10)); // slow path, fits
        v.push(("999999999999999999999999999999999999999i8", 10)); // slow path, overflow
        // u128::MAX — slow path, lands in `IntLit::Large` (> i128::MAX).
        v.push(("340282366920938463463374607431768211455u128", 10));
        v.push(("0xDEADBEEFu32", 16));
        v.push(("0x1234_5678_9ABC_DEF0i64", 16));
        v.push(("0b101101i8", 2));
        v.push(("0b1111_0000_1010_0101u16", 2));
    }
    v
}

/// String literals with simple escapes and long clean runs.
fn escape_strings() -> Vec<String> {
    let mut v = Vec::new();
    for _ in 0..64 {
        v.push(format!(
            "\"{}\\n\\t\\\"{}\"",
            "clean text run with no escapes at all ".repeat(4),
            "more clean text ".repeat(4)
        ));
    }
    v
}

// ── Benches ────────────────────────────────────────────────────────────

fn bench_block_comment(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_comment");
    for (name, body) in [("header", header_body()), ("markers", marker_body())] {
        let body_bytes = body.as_bytes();
        group.throughput(Throughput::Bytes(body_bytes.len() as u64));
        group.bench_function(format!("old_scan/{name}"), |b| {
            b.iter(|| old_block_comment_scan(black_box(body_bytes)))
        });
        // Real path: full source with delimiters, so logos runs the actual
        // `block_comment` callback.  Includes DFA + token overhead.
        let src = format!("/*{body}*/");
        group.bench_function(format!("new_via_logos/{name}"), |b| {
            b.iter(|| black_box(Token::lexer(black_box(src.as_str())).count()))
        });
    }
    group.finish();
}

fn bench_whitespace(c: &mut Criterion) {
    /// Frozen pre-SIMD scalar scan — the body logos used to run
    /// byte-at-a-time inside its DFA for `[ \t\r\n\x0B\x0C]+`.
    fn old_scalar_scan(bytes: &[u8]) -> usize {
        bytes
            .iter()
            .position(|&b| !matches!(b, 0x09..=0x0D | 0x20))
            .unwrap_or(bytes.len())
    }

    // Sparse: many short lines, `\n` runs of length 1 between idents.
    let sparse = {
        let mut s = String::new();
        for _ in 0..4000 {
            s.push_str("x\n");
        }
        s
    };
    // Dense: long leading-space runs (deeply nested indentation).
    let dense = {
        let mut s = String::new();
        for _ in 0..500 {
            s.push_str(&" ".repeat(64));
            s.push_str("x\n");
        }
        s
    };

    let mut group = c.benchmark_group("whitespace");
    for (name, src) in [("sparse", sparse.as_str()), ("dense", dense.as_str())] {
        let bytes = src.as_bytes();
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(format!("old_scalar_scan/{name}"), |b| {
            b.iter(|| {
                // Walk the source the way a scalar lexer would: skip a ws
                // run, then skip one non-ws byte, repeat.
                let mut off = 0usize;
                while off < bytes.len() {
                    off += old_scalar_scan(black_box(&bytes[off..]));
                    off += 1;
                }
                black_box(off)
            })
        });
        // Real path: logos runs the SIMD `whitespace_run` callback.
        group.bench_function(format!("new_via_logos/{name}"), |b| {
            b.iter(|| black_box(Token::lexer(black_box(src)).count()))
        });
    }
    group.finish();
}

fn bench_int_parse(c: &mut Criterion) {
    let lits = int_literals();
    let mut group = c.benchmark_group("int_parse");
    group.throughput(Throughput::Elements(lits.len() as u64));
    group.bench_function("old_checked", |b| {
        b.iter(|| {
            let mut acc = 0i128;
            for &(radix, s) in &lits {
                acc = acc.wrapping_add(
                    old_parse_int_literal(black_box(s), radix, "integer literal overflow")
                        .unwrap_or(0),
                );
            }
            black_box(acc)
        })
    });
    group.bench_function("new_length_gated", |b| {
        b.iter(|| {
            let bump = bumpalo::Bump::new();
            let mut acc = 0i128;
            for &(radix, s) in &lits {
                acc = acc.wrapping_add(
                    parse_int_literal(&bump, black_box(s), radix, "integer literal overflow")
                        .map(|v| lit_i128(&v))
                        .unwrap_or(0),
                );
            }
            black_box(acc)
        })
    });
    group.finish();
}

fn bench_suffixed_int_parse(c: &mut Criterion) {
    let lits = suffixed_literals();
    let mut group = c.benchmark_group("suffixed_int_parse");
    group.throughput(Throughput::Elements(lits.len() as u64));
    group.bench_function("old_replace_from_str_radix", |b| {
        b.iter(|| {
            let mut acc = 0i128;
            for &(s, radix) in &lits {
                acc = acc.wrapping_add(old_suffixed_int_literal(black_box(s), radix).unwrap_or(0));
            }
            black_box(acc)
        })
    });
    group.bench_function("new_zero_alloc", |b| {
        b.iter(|| {
            let bump = bumpalo::Bump::new();
            let mut acc = 0i128;
            for &(s, radix) in &lits {
                // The new scanner returns `(value, width)` — consume both
                // so the width parse cannot be optimized away.
                let (v, w) = parse_suffixed_int_literal(&bump, black_box(s), radix)
                    .unwrap_or((ast::IntLit::Small(0), 0));
                acc = acc.wrapping_add(lit_i128(&v)).wrapping_add(w as i128);
            }
            black_box(acc)
        })
    });
    group.finish();
}

fn bench_string_parse(c: &mut Criterion) {
    let strings = escape_strings();
    let total: usize = strings.iter().map(|s| s.len()).sum();
    let mut group = c.benchmark_group("string_parse");
    group.throughput(Throughput::Bytes(total as u64));
    group.bench_function("old_per_char", |b| {
        b.iter(|| {
            let mut n = 0usize;
            for s in &strings {
                n += old_parse_string_literal(black_box(s.as_str()))
                    .map(|r| r.len())
                    .unwrap_or(0);
            }
            black_box(n)
        })
    });
    group.bench_function("new_bulk_copy", |b| {
        b.iter(|| {
            let mut n = 0usize;
            for s in &strings {
                n += parse_string_literal(black_box(s.as_str()))
                    .map(|r| r.len())
                    .unwrap_or(0);
            }
            black_box(n)
        })
    });
    group.finish();
}

fn bench_lexer_e2e(c: &mut Criterion) {
    let numeric = {
        let mut s = String::new();
        for i in 0..2000 {
            s.push_str("1234567 0xDEADBEEF 0b101101 42 1_000_000 0x1234_5678 9_9_9 ");
            if i % 10 == 0 {
                s.push_str("999999999999999999999999999999999999999 ");
            }
        }
        s
    };
    let escapes = {
        let mut s = String::new();
        for _ in 0..200 {
            s.push('"');
            s.push_str(&"clean text run with no escapes at all ".repeat(4));
            s.push_str("\\n\\t\\x41\\u{1F600}\\\"");
            s.push_str(&"more clean text ".repeat(4));
            s.push_str("\" ");
        }
        s
    };
    let comments = format!("/*{}*/\nident\n/*{}*/\n", header_body(), marker_body());
    let mut group = c.benchmark_group("lexer_e2e");
    for (name, src) in [
        ("numeric_dense", numeric),
        ("escape_heavy", escapes),
        ("comment_heavy", comments),
    ] {
        group.throughput(Throughput::Bytes(src.len() as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(Token::lexer(black_box(src.as_str())).count()))
        });
    }
    group.finish();
}

fn bench_symbol_intern(c: &mut Criterion) {
    // Warm both spellings so the probe (not the first insert) is measured.
    let _ = symbol::Symbol::intern_prefixed('\'', "lifetime");
    let _ = symbol::Symbol::intern("'lifetime");
    let mut group = c.benchmark_group("symbol_intern");
    group.bench_function("old_format_then_intern", |b| {
        b.iter(|| symbol::Symbol::intern(black_box(format!("'{}", "lifetime").as_str())))
    });
    group.bench_function("new_intern_prefixed", |b| {
        b.iter(|| symbol::Symbol::intern_prefixed(black_box('\''), black_box("lifetime")))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_block_comment,
    bench_whitespace,
    bench_int_parse,
    bench_suffixed_int_parse,
    bench_string_parse,
    bench_lexer_e2e,
    bench_symbol_intern
);
criterion_main!(benches);
