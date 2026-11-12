//! Dep-graph micro-benchmarks (`cargo bench --bench dep_graph`).
//!
//! The harness this file provides is the "MEASURE first" prerequisite the
//! codebase names in three places (the root-fast-path comment in
//! `dep_graph.rs`, the DEFERRED note in `verify_inner`, and the bench
//! reference on `DEFAULT_CACHE_SHARDS`).  Each scenario below adjudicates a
//! specific optimisation opportunity; run the relevant one before and after
//! touching that subsystem.
//!
//! | Scenario                     | Adjudicates                                  |
//! |------------------------------|----------------------------------------------|
//! | `stamp_hit_*`                | #3 lock-free stamp layer: cost of a cache-hit verification |
//! | `verification_storm`         | #1 per-node revision: O(Σclosure) re-walk after one invalidation |
//! | `fanout_dirty_readers/k`     | #4 fan-out batching: N marks → one linearisation |
//! | `publish_compute_cycle`      | #3/#10: per-compute stamp + rev-edge publish cost |
//! | `deep_chain_walk`            | #1/#6: walk cost vs closure depth / dirty-dep count |
//!
//! Same empty-lib workaround as `octagon.rs`/`lexer.rs`: the crate is a
//! binary, so this bench `#[path]`-includes the dep-graph module family
//! directly.  The shim is FLAT on purpose: dep_graph and its siblings only
//! use `crate::sync` and `super::` names, so declaring the included modules
//! at the bench crate root satisfies both — and a nested inline `mod` would
//! make `#[path]` resolution traverse a phantom module directory (rustc
//! joins the path literally; `open` fails on the missing intermediate
//! component).  Note the shim defines its own `CachePadded` copy — keep it
//! in sync with the one in `src/hir/query/mod.rs` if the alignment ever
//! changes.

#![allow(clippy::all)]

// ── Module shim: recreates the in-crate view dep_graph compiles against ──

/// Stand-in for the crate-root `sync` module (default build: plain
/// parking_lot re-exports; the bench never runs under `--features shuttle`).
mod sync {
    pub use parking_lot::{
        Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
    };
}

use std::ops::Deref;

// Copy of `CachePadded` from src/hir/query/mod.rs (kept private there).
#[repr(align(64))]
#[derive(Debug)]
struct CachePadded<T>(T);

impl<T> Deref for CachePadded<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}

#[path = "../src/hir/query/scheduler.rs"]
pub mod scheduler;

#[path = "../src/hir/query/sync_order.rs"]
mod sync_order;

#[path = "../src/hir/query/ticket.rs"]
mod ticket;

#[path = "../src/hir/query/dep_graph.rs"]
pub mod dep_graph;

// The `super::{...}` surface dep_graph's use statements expect.
use dep_graph::{ContentClass, DepGraph, DepNodeIndex, Quiesced};
pub use scheduler::{NoOpScheduler, SchedulePoint, Scheduler, TestScheduler};

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
fn quiesced() -> Quiesced {
    // Test/bench escape hatch: drive the red-green mutators directly,
    // without `mutate_world` (there is no QuerySystem here).
    Quiesced::assume_for_test()
}

/// a reads b reads c; all verified green by one walk from `a`.
fn setup_chain(dg: &DepGraph<NoOpScheduler>) -> (DepNodeIndex, DepNodeIndex, DepNodeIndex) {
    let a = dg.allocate_node_index(ContentClass::SEMANTIC);
    let b = dg.allocate_node_index(ContentClass::SEMANTIC);
    let c = dg.allocate_node_index(ContentClass::SEMANTIC);
    dg.start_task(a);
    dg.read(b);
    dg.finish_task();
    dg.start_task(b);
    dg.read(c);
    dg.finish_task();
    (a, b, c)
}

/// Fully verify a graph: walk every root once so the whole closure carries
/// `verified_gen` stamps.
fn verify_all(dg: &DepGraph<NoOpScheduler>, roots: &[DepNodeIndex]) {
    for &r in roots {
        assert!(
            dg.verify_green_transitive_with_dirty(r).is_none(),
            "freshly built graph must verify green"
        );
    }
}

// ── #3: the stamp fast path on a cache hit ──────────────────────────

/// A green, stamped, shallow-closure node re-verified in a loop: measures
/// the full hit decision — dirty-bit check + finishing check + verified_gen
/// fast path with zero traversal.  This is the number the lock-free stamp
/// rework (and any future per-node-revision model) must keep or improve.
fn bench_stamp_hit_shallow(c: &mut Criterion) {
    let mut g = c.benchmark_group("stamp_hit");
    g.warm_up_time(Duration::from_millis(500));
    g.bench_function("shallow_chain", |b| {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let (a, _, _) = setup_chain(&dg);
        verify_all(&dg, &[a]);
        b.iter(|| {
            // After invalidation of a leaf this returns Some; here the
            // closure is stamped, so every call is a pure hit.
            assert!(black_box(dg.verify_green_transitive_with_dirty(black_box(a))).is_none());
        });
    });
    g.finish();
}

/// Hit check via `is_green` only (dirty-bit read, no walk at all): the
/// floor every verification design is bounded below by.
fn bench_dirty_bit_floor(c: &mut Criterion) {
    let mut g = c.benchmark_group("stamp_hit");
    g.bench_function("dirty_bit_floor", |b| {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let (a, _, _) = setup_chain(&dg);
        b.iter(|| assert!(black_box(dg.is_green(black_box(a)))));
    });
    g.finish();
}

// ── #1: the verification storm after an invalidation ─────────────────

/// Fan-out root with `k` direct readers, all stamped green; then one
/// `invalidate` + `dirty_readers` storm and a full re-verify of every
/// reader.  The first reader's walk re-verifies and re-stamps the shared
/// root; later readers should stop at its fresh stamp — this measures how
/// well the stamp layer absorbs the storm, which per-node revision (#1)
/// proposes to replace entirely.
fn bench_verification_storm(c: &mut Criterion) {
    let mut g = c.benchmark_group("verification_storm");
    g.throughput(Throughput::Elements(64));
    g.bench_function("fanout_64", |b| {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let q = quiesced();
        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        let readers: Vec<_> = (0..64)
            .map(|_| {
                let r = dg.allocate_node_index(ContentClass::SEMANTIC);
                dg.start_task(r);
                dg.read(root);
                dg.finish_task();
                r
            })
            .collect();
        verify_all(&dg, &readers);
        b.iter(|| {
            dg.invalidate(root, &q);
            dg.dirty_readers(root, &q);
            // Simulate the recompute+publish of the invalidated root: the
            // stamp layer only re-greens a closure whose nodes are clean;
            // a dirty dep makes the walk return it to the caller first.
            dg.mark_green(root, &q);
            for &r in &readers {
                // Each reader was dirtied by the fan-out; recompute+re-verify.
                dg.mark_green(r, &q);
                assert!(dg.verify_green_transitive_with_dirty(r).is_none());
            }
        });
    });
    g.finish();
}

// ── #4: fan-out mark cost vs reader count ────────────────────────────

/// Cost of `dirty_readers` (the N-marks fan-out) at k = 1/16/64/256 direct
/// readers.  Batching (#4) targets the per-reader RMW chain here; the k
/// sweep is the linearity evidence.
fn bench_fanout_dirty_readers(c: &mut Criterion) {
    let mut g = c.benchmark_group("fanout_dirty_readers");
    for k in [1usize, 16, 64, 256] {
        g.throughput(Throughput::Elements(k as u64));
        g.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            let dg = DepGraph::new(Arc::new(NoOpScheduler));
            let q = quiesced();
            let root = dg.allocate_node_index(ContentClass::SEMANTIC);
            for _ in 0..k {
                let r = dg.allocate_node_index(ContentClass::SEMANTIC);
                dg.start_task(r);
                dg.read(root);
                dg.finish_task();
            }
            b.iter(|| {
                dg.dirty_readers(root, &q);
                // Re-green the readers so the next iteration fans out over
                // the same shape (mark_dirty is what we're measuring; the
                // re-green cost sits outside the storm it models).
                for i in 0..k {
                    dg.mark_green(DepNodeIndex(i as u32 + 2), &q);
                }
            });
        });
    }
    g.finish();
}

// ── publish path: per-compute stamp/finishing/rev-edge cost ──────────

/// One full compute-and-publish cycle against a single dependency:
/// `start_task` + `read` + `finish_task_and_publish` (which runs the
/// finishing window, fingerprint publish, and the per-reader fan-out).
/// The per-compute lock/atomic budget that the lock-free stamp layer (#3)
/// and `rev_edges` rework (#10) claim against.
fn bench_publish_compute_cycle(c: &mut Criterion) {
    let mut g = c.benchmark_group("publish");
    g.bench_function("compute_cycle", |b| {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let dep = dg.allocate_node_index(ContentClass::SEMANTIC);
        let node = dg.allocate_node_index(ContentClass::SEMANTIC);
        let mut fp = 0u64;
        // First publish has no prior fingerprint (`changed` is false by
        // contract), so seed it outside the timed loop.
        dg.start_task(node);
        dg.read(dep);
        dg.finish_task_and_publish(node, Some(0));
        b.iter(|| {
            dg.start_task(node);
            dg.read(dep);
            fp = fp.wrapping_add(1);
            // Every later publish changes the fingerprint: true by contract.
            assert!(dg.finish_task_and_publish(node, Some(fp)));
            black_box(&fp);
        });
    });
    g.finish();
}

// ── #1/#6: walk cost vs closure depth ────────────────────────────────

/// A chain of depth `d`, fully stamped, leaf invalidated: each re-verify
/// walks the whole chain (no stamps survive below the dirty node).  This is
/// the transitive-walk cost per-node revision (#1) proposes to collapse to
/// O(out-degree), and the early-abort shape that collect-all-dirty (#6)
/// would change.
fn bench_deep_chain_walk(c: &mut Criterion) {
    let mut g = c.benchmark_group("deep_chain_walk");
    for depth in [16usize, 64, 256] {
        g.throughput(Throughput::Elements(depth as u64));
        g.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &d| {
            let dg = DepGraph::new(Arc::new(NoOpScheduler));
            let q = quiesced();
            // Build chain n_0 -> n_1 -> ... -> n_{d-1} (n_0 is the root).
            let nodes: Vec<_> = (0..d)
                .map(|_| dg.allocate_node_index(ContentClass::SEMANTIC))
                .collect();
            for w in nodes.windows(2) {
                dg.start_task(w[0]);
                dg.read(w[1]);
                dg.finish_task();
            }
            let leaf = *nodes.last().unwrap();
            let root = nodes[0];
            b.iter(|| {
                dg.invalidate(leaf, &q);
                // Recompute the leaf (simulated publish), then one walk from
                // the root traverses the full chain and re-stamps it; the
                // next iteration re-dirties the leaf.
                dg.mark_green(leaf, &q);
                assert!(dg.verify_green_transitive_with_dirty(root).is_none());
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_stamp_hit_shallow,
    bench_dirty_bit_floor,
    bench_verification_storm,
    bench_fanout_dirty_readers,
    bench_publish_compute_cycle,
    bench_deep_chain_walk,
);
criterion_main!(benches);
