//! SAT solver micro-benchmarks (criterion).
//!
//! Drives both solver backends — **Puto** (CDCL with clause learning) and
//! **Solvo** (plain DPLL) — over workloads shaped like the compiler's `@cfg`
//! domain: binary implication chains, at-most-one groups (Sinz/sequential and
//! commander encodings), at-most-k totalizers, and a random 3-SAT instance
//! near the satisfiability phase transition.
//!
//! Like the other benches in this crate, the module `#[path]`-includes the
//! `src/sat/` family directly (the crate is a binary with an empty lib, so
//! bench targets cannot link it); the modules are self-contained.
//!
//! Run with `cargo bench --bench sat_solver`.

// The path-included module family exposes more surface than the benches
// drive; the bench target compiles it as a standalone crate, so the unused
// remainder would otherwise flood the build with dead-code warnings.
#![allow(dead_code, unused_imports)]

#[path = "../src/sat/mod.rs"]
mod sat;

use criterion::{Criterion, criterion_group, criterion_main};
use sat::{SatSolver, SolveResult};
use std::hint::black_box;

/// Simple LCG for reproducible instances (same generator as the test suite).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0 >> 33
    }
    fn next_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo + 1)
    }
}

fn gen_random_3sat(seed: u64, num_vars: usize, num_clauses: usize) -> Vec<Vec<i32>> {
    let mut rng = Lcg(seed);
    (0..num_clauses)
        .map(|_| {
            (0..3)
                .map(|_| {
                    let var = rng.next_range(1, num_vars) as i32;
                    let sign = if rng.next() % 2 == 0 { 1 } else { -1 };
                    var * sign
                })
                .collect()
        })
        .collect()
}

/// Implication chain a1→a2→…→an plus a falsifying unit — pure binary-clause
/// propagation (the compiler's dominant clause shape).
fn bench_binary_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("sat/binary_chain");
    for &n in &[50usize, 200] {
        group.bench_function(format!("puto/n={n}"), |b| {
            b.iter(|| {
                let mut s = sat::puto::Solver::new();
                let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
                for w in lits.windows(2) {
                    s.add_implies(w[0], w[1]);
                }
                s.add_unit(-lits[n - 1]);
                assert!(black_box(s.solve()).is_sat());
            })
        });
        group.bench_function(format!("solvo/n={n}"), |b| {
            b.iter(|| {
                let mut s = sat::solvo::Solver::new();
                let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
                for w in lits.windows(2) {
                    s.add_implies(w[0], w[1]);
                }
                s.add_unit(-lits[n - 1]);
                assert!(black_box(s.solve()).is_sat());
            })
        });
    }
    group.finish();
}

/// At-most-one over a large group — commander encoding path in Puto, same
/// encoding in Solvo — plus the forced conflict resolution.
fn bench_amo_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("sat/amo_group");
    for &n in &[60usize] {
        group.bench_function(format!("puto/n={n}"), |b| {
            b.iter(|| {
                let mut s = sat::puto::Solver::new();
                let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
                s.add_exactly_one(&lits);
                s.add_unit(lits[7]);
                assert!(black_box(s.solve()).is_sat());
            })
        });
        group.bench_function(format!("solvo/n={n}"), |b| {
            b.iter(|| {
                let mut s = sat::solvo::Solver::new();
                let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
                s.add_exactly_one(&lits);
                s.add_unit(lits[7]);
                assert!(black_box(s.solve()).is_sat());
            })
        });
    }
    group.finish();
}

/// At-most-k over 30 literals (totalizer encoding) with a forced satisfiable
/// assignment.
fn bench_at_most_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("sat/at_most_k");
    let (n, k) = (30usize, 3usize);
    group.bench_function(format!("puto/n={n}/k={k}"), |b| {
        b.iter(|| {
            let mut s = sat::puto::Solver::new();
            let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
            s.add_at_most_k(&lits, k);
            s.add_unit(lits[1]);
            s.add_unit(lits[10]);
            s.add_unit(lits[20]);
            assert!(black_box(s.solve()).is_sat());
        })
    });
    group.bench_function(format!("solvo/n={n}/k={k}"), |b| {
        b.iter(|| {
            let mut s = sat::solvo::Solver::new();
            let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
            s.add_at_most_k(&lits, k);
            s.add_unit(lits[1]);
            s.add_unit(lits[10]);
            s.add_unit(lits[20]);
            assert!(black_box(s.solve()).is_sat());
        })
    });
    group.finish();
}

/// Random 3-SAT near the phase transition (ratio ≈ 4.26) — where clause
/// learning pays for itself; this is the benchmark that separates a CDCL
/// solver from a DPLL solver on hard instances. The reference verdict comes
/// from Puto; Solvo may honestly return `Unknown` (its decision budget is
/// finite), but a definite verdict must agree with the reference.
fn bench_random_3sat(c: &mut Criterion) {
    let mut group = c.benchmark_group("sat/random_3sat");
    let (num_vars, num_clauses) = (60usize, 256usize);
    let clauses = gen_random_3sat(2026, num_vars, num_clauses);

    // Reference verdict (CDCL, comfortably within its budget).
    {
        let mut s = sat::puto::Solver::new();
        for _ in 0..num_vars {
            s.new_var();
        }
        for c in &clauses {
            s.add_clause(c);
        }
        assert!(s.solve().is_sat() || s.solve().is_unsat());
    }
    let reference_sat = {
        let mut s = sat::puto::Solver::new();
        for _ in 0..num_vars {
            s.new_var();
        }
        for c in &clauses {
            s.add_clause(c);
        }
        s.solve().is_sat()
    };

    group.bench_function("puto", |b| {
        b.iter(|| {
            let mut s = sat::puto::Solver::new();
            for _ in 0..num_vars {
                s.new_var();
            }
            for c in &clauses {
                s.add_clause(c);
            }
            let r = black_box(s.solve());
            assert!(
                r.is_sat() == reference_sat,
                "puto verdict drifted from its own reference"
            );
        })
    });
    group.bench_function("solvo", |b| {
        b.iter(|| {
            let mut s = sat::solvo::Solver::new();
            for _ in 0..num_vars {
                s.new_var();
            }
            for c in &clauses {
                s.add_clause(c);
            }
            let r = black_box(s.solve());
            assert!(
                r.is_unknown() || r.is_sat() == reference_sat,
                "solvo produced a definite verdict disagreeing with the CDCL reference"
            );
        })
    });
    group.finish();
}

/// Incremental assumption workloads — the compiler's actual usage pattern:
/// many `solve_assumptions` calls over a growing formula, warm-started
/// (Puto keeps learnts/phases/activity; Solvo keeps phases/activity).
fn bench_incremental_assumptions(c: &mut Criterion) {
    let mut group = c.benchmark_group("sat/incremental");
    group.bench_function("puto/20_assumptions", |b| {
        b.iter(|| {
            let mut s = sat::puto::Solver::new();
            let lits: Vec<i32> = (0..40).map(|_| s.new_var() as i32).collect();
            for w in lits.windows(2) {
                s.add_implies(w[0], w[1]);
            }
            let mut last = sat::SolveResult::Unknown;
            for (i, &lit) in lits.iter().enumerate() {
                last = s.solve_assumptions(&[lit, -lits[(i + 5) % lits.len()]]);
            }
            let done = last.is_sat() || last.is_unsat();
            assert!(black_box(done));
        })
    });
    group.bench_function("solvo/20_assumptions", |b| {
        b.iter(|| {
            let mut s = sat::solvo::Solver::new();
            let lits: Vec<i32> = (0..40).map(|_| s.new_var() as i32).collect();
            for w in lits.windows(2) {
                s.add_implies(w[0], w[1]);
            }
            let mut last = sat::SolveResult::Unknown;
            for (i, &lit) in lits.iter().enumerate() {
                last = s.solve_assumptions(&[lit, -lits[(i + 5) % lits.len()]]);
            }
            let done = last.is_sat() || last.is_unsat();
            assert!(black_box(done));
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_binary_chain,
    bench_amo_group,
    bench_at_most_k,
    bench_random_3sat,
    bench_incremental_assumptions,
);
criterion_main!(benches);
