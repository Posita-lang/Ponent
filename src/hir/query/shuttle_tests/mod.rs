use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DKey(usize);

impl QueryKey for DKey {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FpValue {
    raw: String,
    semantic: u64,
}

impl StableHash for FpValue {
    fn stable_hash(&self) -> u64 {
        self.semantic
    }
}

/// A fresh `QuerySystem`, leaked to `'static`.
///
/// Leaked on PURPOSE, per `run_check` call: shuttle re-runs the closure
/// many times inside ONE process, and a graph's `NEXT_SYSTEM_ID`-style
/// monotonic ids (node indices, stat-arena ids) must restart from the
/// same baseline on every iteration or the model drifts and later
/// rounds assert against indices the earlier ones left behind.
///
/// The leak used to be described as "one small allocation per
/// iteration", which stopped being true: each graph also owned 64 epoch
/// blocks at 32 KiB apiece — 2 MiB — so hundreds of schedules could push
/// RSS into the gigabytes.  Splitting the head start
/// (`EPOCH_INIT_BLOCKS = 4` = 128 KiB), which restores the claim for any
/// graph that grows past 16 K nodes.  The rationales are independent:
/// this note is about the leak, `EPOCH_INIT_BLOCKS`' doc about the size,
/// and both are needed because the bounded-total argument above is what
/// licenses the leak at all.
///
/// WHY NOT DROPPED after the runner run (verified empirically, then
/// reverted): `QuerySystem::drop` runs the deferred-drain path, which
/// touches shuttle task machinery — outside an active shuttle execution
/// that panics ("Tried to get ExecutionState … NotSet"), the panic
/// cascades into every subsequent test in the process.  The leak is
/// therefore DELIBERATE and total-bounded (one graph per `run_check`
/// call; if calls are ever looped, revisit with a drop strategy that
/// `mem::forget`s the deferred queue first).
fn leak_system() -> &'static QuerySystem<'static, 'static, NoOpScheduler> {
    Box::leak(Box::new(QuerySystem::new()))
}

fn leak_provider<'src, S: Scheduler>(
    system: &'static QuerySystem<'static, 'src, S>,
) -> &'static dyn QueryProvider<'static, 'src, S> {
    Box::leak(Box::new(DefaultQueryProvider::new(system)))
}

fn run_check<F: Fn() + Send + Sync + 'static>(f: F) {
    let mut config = shuttle::Config::default();
    config.stack_size = 2 * 1024 * 1024;
    config.failure_persistence = shuttle::FailurePersistence::File(None);
    let runner = shuttle::Runner::new(shuttle::scheduler::PctScheduler::new(8, 50), config);
    runner.run(f);
}

/// `run_check` with a SEED SWEEP: `PctScheduler::new(8, 50)` with a fixed
/// budget explores one deterministic schedule per run, which is fine for
/// the gate-choreographed tests (their interleavings are hand-built) but
/// thin for the UNCONSTRAINED ones — d1/d2/d11/tstress benefit from seeing
/// several schedules.  Keep the sweep small (8 seeds): each seed is a full
/// model-checking run and these tests already run fast; widen only if a
/// failure class needs more breadth.  Assertions must hold for EVERY seed
/// — a sweep failure reports the seed so `shuttle::replay` can pin it.
/// (Each seed leaks its own system/provider — see `leak_system`'s doc for
/// why they are not dropped.)
fn run_check_swept<F: Fn() + Send + Sync + 'static>(f: F) {
    let f = std::sync::Arc::new(f);
    // Seeds start at 1: `PctScheduler::new(0, …)` asserts `max_depth > 0`
    // (seed 0 is not a valid depth).
    for seed in 1..=8usize {
        let f = std::sync::Arc::clone(&f);
        let mut config = shuttle::Config::default();
        config.stack_size = 2 * 1024 * 1024;
        config.failure_persistence = shuttle::FailurePersistence::File(None);
        let runner =
            shuttle::Runner::new(shuttle::scheduler::PctScheduler::new(seed, 50), config);
        runner.run(move || f());
    }
}

fn spin_until(flag: &AtomicBool, what: &str) {
    for _ in 0..100_000 {
        if flag.load(Ordering::SeqCst) {
            return;
        }
        shuttle::thread::yield_now();
    }
    panic!("shuttle rendezvous failed: {what}");
}

mod aclass;
mod b1;
mod b2;
mod b3;
mod depgraph;
mod tstress;
