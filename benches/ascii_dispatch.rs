//! Dispatch-boundary micro-benchmarks for the ASCII fast-path predicates.
//!
//! The root crate is a binary (empty lib target), so this bench
//! `#[path]`-includes `src/diagnostics/ascii.rs` directly — the same trick the
//! lexer and octagon benches use.  Every length sits exactly on a dispatch
//! boundary (15/16/17, 31/32/33, 63/64/65, 127/128/129, 255/256/257) so a
//! regression in tier selection shows up as a step change instead of noise.
//!
//! This is the harness the constants and comment in `ascii.rs` have been asking
//! for: `AVX512_MIN_LEN` (64 vs 128) and the `< 16` scalar tails are
//! *measurement* decisions, and until now nothing measured them.
//!
//! ```text
//! cargo bench --bench ascii_dispatch                     # native dispatch
//! cargo bench --bench ascii_dispatch --features bench-hooks   # + SWAR-pinned
//! ```
//!
//! `bench-hooks` exposes `set_swar_forced` (the same thread-local the unit
//! tests drive), which is what makes the vector/SWAR comparison possible inside
//! one binary — wall-clock comparisons across binaries on this box drift by
//! more than the effect being measured.
//!
//! Not covered here: `ErrCode::url_ansi`.  `error_code.rs` needs the generated
//! code table and the Fluent locales at build time, so it does not survive a
//! standalone `#[path]` include; that cache lives on the emit path and is
//! exercised by the emitter's own tests.

#[path = "../src/diagnostics/ascii.rs"]
#[allow(dead_code)]
mod ascii;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

#[derive(Clone, Copy)]
enum Kind {
    Printable,
    Tab,
    HighByte,
    Cjk,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Printable => "printable",
            Kind::Tab => "tab",
            Kind::HighByte => "high",
            Kind::Cjk => "cjk",
        }
    }

    /// Cycle a short pattern of this family through `buf`.
    fn fill(self, buf: &mut [u8]) {
        let pattern: &[u8] = match self {
            Kind::Printable => b"abcdefgh",
            Kind::Tab => b"ab\tcd\tef",
            // Latin-1 supplement: high bit set, must be rejected by every
            // printable predicate.
            Kind::HighByte => "éèêë".as_bytes(),
            // 3 bytes per CJK glyph, so `next_non_ascii_byte` sees dense hits.
            Kind::Cjk => "誤り訂正".as_bytes(),
        };
        for (i, b) in buf.iter_mut().enumerate() {
            *b = pattern[i % pattern.len()];
        }
    }
}

const LENS: &[usize] = &[
    15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
];
const KINDS: &[Kind] = &[Kind::Printable, Kind::Tab, Kind::HighByte, Kind::Cjk];

fn buffers() -> Vec<(usize, Vec<u8>)> {
    LENS.iter()
        .map(|&len| {
            // One buffer per length, filled with the printable family; the
            // per-kind groups refill their own copy.
            let mut buf = vec![0u8; len];
            Kind::Printable.fill(&mut buf);
            (len, buf)
        })
        .collect()
}

/// The three "can this be emitted verbatim?" gates plus `classify_ascii`, which
/// is the single primitive the renderers use when they need two answers.
fn predicates(c: &mut Criterion) {
    let mut g = c.benchmark_group("ascii_dispatch/predicates");
    for &kind in KINDS {
        for (len, base) in buffers() {
            let mut data = base.clone();
            kind.fill(&mut data);
            let id = BenchmarkId::new(kind.label(), len);
            g.throughput(Throughput::Bytes(len as u64));
            g.bench_with_input(id, &data, |b, data| {
                b.iter(|| std::hint::black_box(ascii::is_printable_ascii(data)));
            });
        }
    }
    g.finish();
}

fn or_tab_and_multiline(c: &mut Criterion) {
    let mut g = c.benchmark_group("ascii_dispatch/or_tab_multiline");
    for &kind in KINDS {
        for (len, base) in buffers() {
            let mut data = base.clone();
            kind.fill(&mut data);
            g.bench_with_input(
                BenchmarkId::new(format!("or_tab/{}", kind.label()), len),
                &data,
                |b, data| b.iter(|| std::hint::black_box(ascii::is_printable_ascii_or_tab(data))),
            );
            g.bench_with_input(
                BenchmarkId::new(format!("multiline/{}", kind.label()), len),
                &data,
                |b, data| {
                    b.iter(|| std::hint::black_box(ascii::is_printable_ascii_multiline(data)))
                },
            );
        }
    }
    g.finish();
}

fn classify(c: &mut Criterion) {
    let mut g = c.benchmark_group("ascii_dispatch/classify");
    for &kind in KINDS {
        for (len, base) in buffers() {
            let mut data = base.clone();
            kind.fill(&mut data);
            g.bench_with_input(BenchmarkId::new(kind.label(), len), &data, |b, data| {
                b.iter(|| std::hint::black_box(ascii::classify_ascii(data)))
            });
        }
    }
    g.finish();
}

/// `char_col`'s sparse-table build calls this per non-ASCII character, usually
/// on the short remainder of a line — the boundary lengths matter most here.
fn next_non_ascii(c: &mut Criterion) {
    let mut g = c.benchmark_group("ascii_dispatch/next_non_ascii");
    for &kind in KINDS {
        for (len, base) in buffers() {
            let mut data = base.clone();
            kind.fill(&mut data);
            g.bench_with_input(BenchmarkId::new(kind.label(), len), &data, |b, data| {
                b.iter(|| std::hint::black_box(ascii::next_non_ascii_byte(data)))
            });
        }
    }
    g.finish();
}

#[cfg(feature = "bench-hooks")]
fn set_swar(on: bool) {
    ascii::set_swar_forced(on);
}

#[cfg(not(feature = "bench-hooks"))]
fn set_swar(_on: bool) {}

/// Same shapes with the SWAR tier pinned, so vector-vs-scalar can be compared
/// without cross-binary drift.  Without `bench-hooks` this group prints a notice
/// and measures nothing rather than silently reporting native numbers under a
/// misleading name.
fn swar_forced(c: &mut Criterion) {
    if !cfg!(feature = "bench-hooks") {
        eprintln!(
            "ascii_dispatch: skipping the SWAR-pinned group; \
             re-run with --features bench-hooks"
        );
        return;
    }
    set_swar(true);
    let mut g = c.benchmark_group("ascii_dispatch/swar_forced");
    for &kind in KINDS {
        for (len, base) in buffers() {
            let mut data = base.clone();
            kind.fill(&mut data);
            g.throughput(Throughput::Bytes(len as u64));
            g.bench_with_input(
                BenchmarkId::new(format!("printable/{}", kind.label()), len),
                &data,
                |b, data| b.iter(|| std::hint::black_box(ascii::is_printable_ascii(data))),
            );
            g.bench_with_input(
                BenchmarkId::new(format!("multiline/{}", kind.label()), len),
                &data,
                |b, data| {
                    b.iter(|| std::hint::black_box(ascii::is_printable_ascii_multiline(data)))
                },
            );
            g.bench_with_input(
                BenchmarkId::new(format!("classify/{}", kind.label()), len),
                &data,
                |b, data| {
                    b.iter_batched(
                        || (),
                        |()| std::hint::black_box(ascii::classify_ascii(data)),
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }
    g.finish();
    set_swar(false);
}

criterion_group!(
    benches,
    predicates,
    or_tab_and_multiline,
    classify,
    next_non_ascii,
    swar_forced
);
criterion_main!(benches);
