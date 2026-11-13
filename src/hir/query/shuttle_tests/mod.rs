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

/// Shuttle version of the batched re-request: under `feature = "shuttle"`
/// the parallel branch is compiled out, so this also proves the sequential
/// fallback still recomputes every dirty dependency exactly once.
#[test]
fn rerequest_recomputes_all_dirty_dependencies() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static K_RUNS: AtomicU64 = AtomicU64::new(0);
    static E_RUNS: AtomicU64 = AtomicU64::new(0);

    struct EQ;
    impl<'q, 'src: 'q> Query<'q, 'src> for EQ {
        type Storage = Owned;
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("shuttle_rerequest_e", "shuttle rerequest leaf")
        }
        fn compute<S: Scheduler>(_key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: String::new(),
                semantic: 7,
            }
        }
    }
    struct KQ;
    impl<'q, 'src: 'q> Query<'q, 'src> for KQ {
        type Storage = Owned;
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("shuttle_rerequest_k", "shuttle rerequest reader")
        }
        fn compute<S: Scheduler>(key: &DKey, provider: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
            K_RUNS.fetch_add(1, Ordering::SeqCst);
            let a = query_get::<EQ, _>(provider, key.clone()).unwrap();
            let b = query_get::<EQ, _>(provider, DKey(key.0 + 1)).unwrap();
            FpValue {
                raw: String::new(),
                semantic: a.semantic + b.semantic,
            }
        }
    }

    run_check(|| {
        K_RUNS.store(0, Ordering::SeqCst);
        E_RUNS.store(0, Ordering::SeqCst);
        let sys = leak_system();
        let prov = leak_provider(sys);
        assert_eq!(sys.get::<KQ>(DKey(0), prov).unwrap().semantic, 14);
        assert_eq!(E_RUNS.load(Ordering::SeqCst), 2);
        assert_eq!(K_RUNS.load(Ordering::SeqCst), 1);
        sys.invalidate::<EQ>(&DKey(0));
        sys.invalidate::<EQ>(&DKey(1));

        // The query itself must stay on ONE shuttle task: TASK_STACK is
        // per-OS-thread, so two concurrent computes interleave frames and
        // trip `finish_task_and_publish`'s owner assert.  A second task that
        // only yields still gives the scheduler real interleavings to
        // explore around the sequential (shuttle) re-request path.
        let release = std::sync::Arc::new(AtomicBool::new(false));
        let release_noise = std::sync::Arc::clone(&release);
        let noise = shuttle::thread::spawn(move || {
            while !release_noise.load(Ordering::SeqCst) {
                shuttle::thread::yield_now();
            }
        });
        let served = sys.get::<KQ>(DKey(0), prov).unwrap();
        release.store(true, Ordering::SeqCst);
        noise.join().unwrap();

        assert_eq!(served.semantic, 14);
        assert_eq!(
            E_RUNS.load(Ordering::SeqCst),
            4,
            "every dirty dependency must be re-requested exactly once"
        );
        assert_eq!(
            K_RUNS.load(Ordering::SeqCst),
            1,
            "the cached reader must serve green after the batched re-request"
        );
    });
}

mod aclass;
mod b1;
mod b2;
mod b3;
mod depgraph;
mod tstress;
