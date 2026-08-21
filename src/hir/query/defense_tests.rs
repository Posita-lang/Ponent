//! Defense tests (cycle / deadlock / poisoning / depth / dedup) and
//! granularity target tests (fingerprint-based invalidation).
//!
//! Every hang-prone test runs the query on a spawned thread and guards it
//! with `recv_timeout`; a timeout is reported as `Err("TIMEOUT")` rather
//! than wedging the test process.  All `QuerySystem` / provider objects are
//! `Box::leak`ed to `'static` so they can be moved into spawned threads.
//!
//! Baseline contract (recorded in the query-system worklog):
//!   - self-recursion / cross-key cycle tests: TIMEOUT on the un-fixed code
//!   - eval_always parallel: Err("query result not found after waiting")
//!   - parallel depth: flaky / false depth-limit errors
//!   - same-key dedup: pass (regression guard, must stay green)
//!   - granularity target: MUST FAIL on the un-fixed eager-BFS invalidation
//!     (that is the target), must go green after lazy invalidation lands.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Timeout for hang-prone tests (2 s, per the worklog contract).
const GUARD: Duration = Duration::from_secs(2);

// ── Shared test scaffolding ───────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DKey(usize);

impl QueryKey for DKey {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DValue {
    raw: String,
    semantic: u64,
}

impl StableHash for DValue {
    fn stable_hash(&self) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        self.raw.hash(&mut h);
        h.finish()
    }
}

impl DValue {
    fn new(semantic: u64) -> Self {
        DValue {
            raw: format!("sem_{semantic}"),
            semantic,
        }
    }
    fn sentinel_cycle() -> Self {
        DValue {
            raw: "cycle_sentinel".into(),
            semantic: 0,
        }
    }
}

fn leak_system() -> &'static QuerySystem {
    Box::leak(Box::new(QuerySystem::new()))
}

fn leak_provider(system: &'static QuerySystem) -> &'static dyn QueryProvider {
    Box::leak(Box::new(DefaultQueryProvider::new(system)))
}

/// Run `f` on a fresh thread; guard with `recv_timeout`.
/// `Err("TIMEOUT")` if `f` does not finish in time, `Err("PANIC")` if the
/// closure panicked.
fn run_with_guard<F, R>(dur: Duration, f: F) -> Result<R, String>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    guard_recv_dur(rx, dur)
}

fn guard_recv<T>(rx: mpsc::Receiver<T>) -> Result<T, String> {
    guard_recv_dur(rx, GUARD)
}

fn guard_recv_dur<T>(rx: mpsc::Receiver<T>, dur: Duration) -> Result<T, String> {
    rx.recv_timeout(dur).map_err(|e| match e {
        mpsc::RecvTimeoutError::Timeout => "TIMEOUT".to_string(),
        mpsc::RecvTimeoutError::Disconnected => "PANIC".to_string(),
    })
}

/// Spin until `flag` is set (bounded, so a wedged barrier fails the test
/// loudly instead of hanging the whole process).
fn wait_flag(flag: &AtomicBool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !flag.load(Ordering::SeqCst) {
        if std::time::Instant::now() > deadline {
            panic!("barrier flag not set within 5s");
        }
        std::hint::spin_loop();
    }
}

// ── Same-thread cycle detection ───────────────────────────────────

/// Same-key direct self-recursion must surface as a cycle error
/// (sentinel), never as a deadlock.
#[test]
fn d1_same_key_self_recursion() {
    static ENTERED: AtomicBool = AtomicBool::new(false);
    ENTERED.store(false, Ordering::SeqCst);

    struct D1Q;
    impl Query for D1Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d1", "self-recursive query")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> DValue {
            ENTERED.store(true, Ordering::SeqCst);
            match query_get::<D1Q>(provider, key.clone()) {
                Ok(v) => v,
                Err(_) => DValue::sentinel_cycle(),
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    let result = run_with_guard(GUARD, move || {
        system.get::<D1Q>(DKey(1), provider).map(|v| v.raw)
    });
    match result {
        Ok(Ok(raw)) => assert_eq!(raw, "cycle_sentinel"),
        Ok(Err(e)) => panic!("expected cycle sentinel, got error: {e:?}"),
        Err(t) => panic!("{t}: same-key self-recursion must not deadlock"),
    }
}

/// Same-thread cross-key cycle k1 -> k2 -> k1 must surface as a cycle
/// error (sentinel), never as a deadlock.
#[test]
fn d2_same_thread_cross_key_cycle() {
    static ENTERED: AtomicBool = AtomicBool::new(false);
    ENTERED.store(false, Ordering::SeqCst);

    struct D2Q;
    impl Query for D2Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d2", "cross-key cyclic query")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> DValue {
            ENTERED.store(true, Ordering::SeqCst);
            let next = if key.0 == 1 { DKey(2) } else { DKey(1) };
            match query_get::<D2Q>(provider, next) {
                Ok(v) => v,
                Err(_) => DValue::sentinel_cycle(),
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    let result = run_with_guard(GUARD, move || {
        system.get::<D2Q>(DKey(1), provider).map(|v| v.raw)
    });
    match result {
        Ok(Ok(raw)) => assert_eq!(raw, "cycle_sentinel"),
        Ok(Err(e)) => panic!("expected cycle sentinel, got error: {e:?}"),
        Err(t) => panic!("{t}: cross-key cycle must not deadlock"),
    }
}

// ── Cross-thread cycle detection ──────────────────────────────────

/// Two threads computing keys that depend on each other must not
/// wedge; at least one side must surface the cycle.
#[test]
fn d3_two_thread_mutual_dependency() {
    static ENTERED_K1: AtomicBool = AtomicBool::new(false);
    static ENTERED_K2: AtomicBool = AtomicBool::new(false);
    ENTERED_K1.store(false, Ordering::SeqCst);
    ENTERED_K2.store(false, Ordering::SeqCst);

    struct D3Q;
    impl Query for D3Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d3", "mutual-dependency query")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> DValue {
            // Barrier: both computes must be in flight before either one
            // requests the other's key.
            if key.0 == 1 {
                ENTERED_K1.store(true, Ordering::SeqCst);
                wait_flag(&ENTERED_K2);
            } else {
                ENTERED_K2.store(true, Ordering::SeqCst);
                wait_flag(&ENTERED_K1);
            }
            let other = if key.0 == 1 { DKey(2) } else { DKey(1) };
            match query_get::<D3Q>(provider, other) {
                Ok(v) => v,
                Err(_) => DValue::sentinel_cycle(),
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Spawn BOTH before awaiting either, so the mutual wait can form.
    let (tx1, rx1) = mpsc::channel();
    let (tx2, rx2) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx1.send(system.get::<D3Q>(DKey(1), provider));
    });
    thread::spawn(move || {
        let _ = tx2.send(system.get::<D3Q>(DKey(2), provider));
    });

    let r1 = guard_recv(rx1);
    let r2 = guard_recv(rx2);
    assert!(
        r1.is_ok() && r2.is_ok(),
        "no thread may wedge: {r1:?} {r2:?}"
    );
    let (r1, r2) = (r1.unwrap(), r2.unwrap());
    assert!(
        r1.is_ok() && r2.is_ok(),
        "both threads must complete Ok: {r1:?} {r2:?}"
    );
    let (v1, v2) = (r1.unwrap(), r2.unwrap());
    assert!(
        v1.raw == "cycle_sentinel" || v2.raw == "cycle_sentinel",
        "at least one side must surface the cycle, got {v1:?} and {v2:?}"
    );
}

// ── Panic poisoning ───────────────────────────────────────────────

/// A compute panic must wake blocked waiters with a poisoned error.
#[test]
fn d4_panic_poisons_latch() {
    static ENTERED: AtomicBool = AtomicBool::new(false);
    static GO: AtomicBool = AtomicBool::new(false);
    ENTERED.store(false, Ordering::SeqCst);
    GO.store(false, Ordering::SeqCst);

    struct D4Q;
    impl Query for D4Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d4", "panicking query")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> DValue {
            ENTERED.store(true, Ordering::SeqCst);
            wait_flag(&GO);
            panic!("d4 compute boom");
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Thread A computes and will panic once GO is set.
    let a_handle = thread::spawn(move || {
        let _ = system.get::<D4Q>(DKey(1), provider);
    });
    wait_flag(&ENTERED);

    // Trigger thread: after B has had time to reach the latch, let A
    // panic.  (Must not run on the main thread — main is blocked awaiting
    // B below.)
    let trigger = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        GO.store(true, Ordering::SeqCst);
    });

    // Thread B requests the same key while A is computing: it must be
    // woken with a poisoned error, not hang.
    let b_result = run_with_guard(GUARD, move || system.get::<D4Q>(DKey(1), provider));
    let _ = trigger.join();

    match b_result {
        Ok(Err(e)) => {
            let msg = e.message.to_lowercase();
            assert!(
                msg.contains("poisoned") || msg.contains("panicked"),
                "waiter error must mention poisoning, got: {e:?}"
            );
        }
        other => panic!("waiter must get Err (poisoned), got: {other:?}"),
    }

    let a_outcome = a_handle.join();
    assert!(a_outcome.is_err(), "compute thread must panic");
}

// ── eval_always parallel same key ────────────────────────────────

/// `eval_always` queries must NOT be deduplicated across threads;
/// each requester computes its own value.
#[test]
fn d5_eval_always_parallel_same_key() {
    static ENTERED: AtomicBool = AtomicBool::new(false);
    ENTERED.store(false, Ordering::SeqCst);

    struct D5Q;
    impl Query for D5Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d5", "eval-always parallel query").with_eval_always(true)
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> DValue {
            ENTERED.store(true, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(100));
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Thread A starts computing; the main thread waits until A is inside
    // compute, then issues a second request for the same key.
    let a_handle = thread::spawn(move || {
        let _ = system.get::<D5Q>(DKey(1), provider);
    });
    wait_flag(&ENTERED);

    let b_result = run_with_guard(GUARD, move || system.get::<D5Q>(DKey(1), provider));

    match b_result {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => {
            panic!("eval_always second request must compute independently, got error: {e:?}")
        }
        Err(t) => panic!("{t}: eval_always second request must not wedge"),
    }

    let _ = a_handle.join();
}

// ── Per-thread depth accounting ───────────────────────────────────

/// Parallel depth-limited queries must account depth per thread;
/// parallel chains must never trip each other's depth limit.
#[test]
fn d6_par_get_depth_limit_per_thread() {
    const ROOTS: usize = 8;
    const DEPTH: usize = 100;

    struct D6Q;
    impl Query for D6Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d6", "depth-limited parallel query").with_depth_limit(true)
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> DValue {
            let d = key.0 % 1000;
            if d > 0 {
                match query_get::<D6Q>(provider, DKey(key.0 - 1)) {
                    Ok(v) => v,
                    Err(_) => DValue {
                        raw: format!("depth_limit_at_{}", key.0),
                        semantic: 0,
                    },
                }
            } else {
                DValue::new(key.0 as u64)
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);
    let keys: Vec<DKey> = (0..ROOTS).map(|r| DKey(r * 1000 + DEPTH)).collect();

    let result = run_with_guard(Duration::from_secs(10), move || {
        system.par_get::<D6Q>(&keys, provider)
    });
    let results = result.expect("par_get must finish");
    for (i, r) in results.iter().enumerate() {
        assert!(r.is_ok(), "root {i}: unexpected error: {r:?}");
        let v = r.as_ref().unwrap();
        assert!(
            !v.raw.contains("depth_limit"),
            "root {i}: depth limit must not trigger for per-thread stacks: {v:?}"
        );
    }
}

// ── Same-key dedup regression guard ───────────────────────────────

/// Two threads requesting the same key must share one computation
/// (dedup).  This is a regression guard: dedup must survive all fixes.
#[test]
fn d7_parallel_same_key_dedup() {
    static ENTERED: AtomicBool = AtomicBool::new(false);
    static GO: AtomicBool = AtomicBool::new(false);
    static COMPUTES: AtomicU64 = AtomicU64::new(0);
    ENTERED.store(false, Ordering::SeqCst);
    GO.store(false, Ordering::SeqCst);
    COMPUTES.store(0, Ordering::SeqCst);

    struct D7Q;
    impl Query for D7Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d7", "dedup guard query")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> DValue {
            COMPUTES.fetch_add(1, Ordering::SeqCst);
            ENTERED.store(true, Ordering::SeqCst);
            wait_flag(&GO);
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    let t1 = thread::spawn(move || system.get::<D7Q>(DKey(7), provider));
    wait_flag(&ENTERED);

    let t2 = thread::spawn(move || system.get::<D7Q>(DKey(7), provider));
    thread::sleep(Duration::from_millis(200)); // let t2 reach the latch
    GO.store(true, Ordering::SeqCst);

    let r1 = t1.join().expect("t1 must not panic");
    let r2 = t2.join().expect("t2 must not panic");
    assert!(
        r1.is_ok() && r2.is_ok(),
        "both requests must succeed: {r1:?} {r2:?}"
    );
    assert_eq!(
        COMPUTES.load(Ordering::SeqCst),
        1,
        "same-key dedup must produce exactly one compute"
    );
}

// ── Stress: random concurrent graph vs naive oracle ───────────────

const STRESS_SLOTS: usize = 16;
const PANIC_SLOT: usize = 15;

static SRC: [AtomicU64; STRESS_SLOTS] = [const { AtomicU64::new(0) }; STRESS_SLOTS];
static PANIC_FLAG: AtomicBool = AtomicBool::new(false);

/// Source (input) query: reads `SRC[slot]`.
struct SrcQ;
impl Query for SrcQ {
    type Key = DKey;
    type Value = FpValue;
    type Cache = DefaultCache<DKey, FpValue>;
    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new("stress_src", "stress input query")
    }
    fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
        let slot = key.0 % STRESS_SLOTS;
        if PANIC_FLAG.load(Ordering::SeqCst) && slot == PANIC_SLOT {
            panic!("stress panic injection");
        }
        FpValue {
            raw: format!("s{slot}_{}", SRC[slot].load(Ordering::SeqCst)),
            semantic: SRC[slot].load(Ordering::SeqCst),
        }
    }
}

/// Sum query: reads `SrcQ(slot)` and `SrcQ((slot+1) % SLOTS)`.
struct SumQ;
impl Query for SumQ {
    type Key = DKey;
    type Value = FpValue;
    type Cache = DefaultCache<DKey, FpValue>;
    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new("stress_sum", "stress sum query")
    }
    fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
        let slot = key.0 % STRESS_SLOTS;
        let a = query_get::<SrcQ>(provider, DKey(slot)).expect("sum reads src");
        let b =
            query_get::<SrcQ>(provider, DKey((slot + 1) % STRESS_SLOTS)).expect("sum reads src");
        FpValue {
            raw: format!("sum{slot}_{}", a.semantic + b.semantic),
            semantic: a.semantic + b.semantic,
        }
    }
}

/// Chain query with random recursion depth: `ChainQ(slot*100 + d)` reads
/// `ChainQ(slot*100 + d-1)` down to `d = 0`, which reads `SrcQ(slot)`.
/// The value is `SRC[slot] + d`, the oracle is `SRC[slot].load() + d`.
struct ChainQ;
impl Query for ChainQ {
    type Key = DKey;
    type Value = FpValue;
    type Cache = DefaultCache<DKey, FpValue>;
    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new("stress_chain", "stress recursive chain query")
    }
    fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
        let slot = key.0 / 100;
        let d = key.0 % 100;
        if d > 0 {
            let inner =
                query_get::<ChainQ>(provider, DKey(slot * 100 + d - 1)).expect("chain recurse");
            FpValue {
                raw: format!("c{slot}_{d}_{}", inner.semantic),
                semantic: inner.semantic + 1,
            }
        } else {
            let src = query_get::<SrcQ>(provider, DKey(slot)).expect("chain base");
            FpValue {
                raw: format!("c{slot}_0_{}", src.semantic),
                semantic: src.semantic,
            }
        }
    }
}

/// Random concurrent query graph vs a naive recompute oracle.
///
/// Phase 1 (value phase): 4 workers × 3000 random ops — gets of `SumQ`,
/// `SrcQ`, `ChainQ` and invalidations with random source mutations.  The
/// sources only ever INCREASE (monotonic), so a sound one-sided oracle
/// invariant holds under races: any returned value must be ≤ the current
/// source snapshot (a value can never be newer than the current source).
/// Phase 2 (exact check, quiesced): invalidate every source slot and
/// re-request — values must EXACTLY equal the (now quiescent) sources.
/// Phase 3 (panic storm): panic-inject on one slot; every outcome must be
/// Ok / Err(poisoned|panicked) / caught panic, within guard timeouts, and
/// the system must stay alive afterwards.
#[test]
fn stress_random_concurrent_graph_oracle() {
    static OP_COUNTER: AtomicU64 = AtomicU64::new(0);
    OP_COUNTER.store(0, Ordering::SeqCst);
    PANIC_FLAG.store(false, Ordering::SeqCst);
    for s in &SRC {
        s.store(0, Ordering::SeqCst);
    }

    let system = leak_system();
    let provider = leak_provider(system);

    fn worker(system: &'static QuerySystem, provider: &'static dyn QueryProvider) {
        let mut local = 0usize;
        loop {
            let op = OP_COUNTER.fetch_add(1, Ordering::SeqCst);
            if op >= 3000 {
                break;
            }
            let slot = ((op as usize).wrapping_mul(2654435761) + local) % STRESS_SLOTS;
            local = local.wrapping_add(1);
            match op % 10 {
                0..=3 => {
                    // get SumQ(slot): v ≤ SRC[slot] + SRC[slot+1] (monotonic).
                    let hi = SRC[slot].load(Ordering::SeqCst)
                        + SRC[(slot + 1) % STRESS_SLOTS].load(Ordering::SeqCst);
                    match std::panic::catch_unwind(AssertUnwindSafe(|| {
                        system.get::<SumQ>(DKey(slot), provider)
                    })) {
                        Ok(Ok(v)) => {
                            assert!(
                                v.semantic <= hi,
                                "sum value must not exceed current sources: {v:?} vs {hi}"
                            );
                        }
                        Ok(Err(e)) => panic!("value phase must not error: {e:?}"),
                        Err(_) => panic!("value phase must not panic"),
                    }
                }
                4..=6 => {
                    // get SrcQ(slot): v ≤ SRC[slot].
                    let hi = SRC[slot].load(Ordering::SeqCst);
                    match std::panic::catch_unwind(AssertUnwindSafe(|| {
                        system.get::<SrcQ>(DKey(slot), provider)
                    })) {
                        Ok(Ok(v)) => {
                            assert!(
                                v.semantic <= hi,
                                "src value must not exceed current source: {v:?} vs {hi}"
                            );
                        }
                        Ok(Err(e)) => panic!("value phase must not error: {e:?}"),
                        Err(_) => panic!("value phase must not panic"),
                    }
                }
                7..=8 => {
                    // get ChainQ(slot, random depth 0..=3): v ≤ SRC[slot] + d.
                    let d = (op as usize % 4) as u64;
                    let hi = SRC[slot].load(Ordering::SeqCst) + d;
                    match std::panic::catch_unwind(AssertUnwindSafe(|| {
                        system.get::<ChainQ>(DKey(slot * 100 + d as usize), provider)
                    })) {
                        Ok(Ok(v)) => {
                            assert!(
                                v.semantic <= hi,
                                "chain value must not exceed current source + depth: {v:?} vs {hi}"
                            );
                        }
                        Ok(Err(e)) => panic!("value phase must not error: {e:?}"),
                        Err(_) => panic!("value phase must not panic"),
                    }
                }
                _ => {
                    // Invalidate; mutate the source with probability 1/2.
                    if op % 4 == 0 {
                        SRC[slot].fetch_add(1, Ordering::SeqCst);
                    }
                    system.invalidate::<SrcQ>(&DKey(slot));
                }
            }
        }
    }

    let phase1 = run_with_guard(Duration::from_secs(20), move || {
        let mut handles = Vec::new();
        for _ in 0..4 {
            handles.push(thread::spawn(move || {
                let r = std::panic::catch_unwind(AssertUnwindSafe(|| worker(system, provider)));
                if let Err(p) = r {
                    std::panic::resume_unwind(p);
                }
            }));
        }
        for h in handles {
            h.join().expect("stress worker must not panic");
        }
    });
    assert!(phase1.is_ok(), "stress phase 1: {phase1:?}");

    // Phase 2: quiesced exact oracle check.
    for slot in 0..STRESS_SLOTS {
        system.invalidate::<SrcQ>(&DKey(slot));
    }
    for slot in 0..STRESS_SLOTS {
        let expect = SRC[slot].load(Ordering::SeqCst);
        let v = system.get::<SrcQ>(DKey(slot), provider).expect("src get");
        assert_eq!(v.semantic, expect, "quiesced src mismatch at slot {slot}");
        let expect_chain = expect + 3;
        let v = system
            .get::<ChainQ>(DKey(slot * 100 + 3), provider)
            .expect("chain get");
        assert_eq!(
            v.semantic, expect_chain,
            "quiesced chain mismatch at slot {slot}"
        );
        let expect_sum = expect + SRC[(slot + 1) % STRESS_SLOTS].load(Ordering::SeqCst);
        let v = system.get::<SumQ>(DKey(slot), provider).expect("sum get");
        assert_eq!(
            v.semantic, expect_sum,
            "quiesced sum mismatch at slot {slot}"
        );
    }

    // Phase 3: panic storm on the dedicated panic slot.
    PANIC_FLAG.store(true, Ordering::SeqCst);
    system.invalidate::<SrcQ>(&DKey(PANIC_SLOT));
    let phase3 = run_with_guard(Duration::from_secs(10), move || {
        let mut handles = Vec::new();
        for _ in 0..8 {
            handles.push(thread::spawn(move || {
                let mut outcomes = Vec::new();
                for _ in 0..5 {
                    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        system.get::<SrcQ>(DKey(PANIC_SLOT), provider)
                    }));
                    match r {
                        Ok(Ok(_)) => outcomes.push("ok"),
                        Ok(Err(e)) => {
                            let m = e.message.to_lowercase();
                            assert!(
                                m.contains("poisoned") || m.contains("panicked"),
                                "panic-phase error must mention poisoning: {e:?}"
                            );
                            outcomes.push("err");
                        }
                        Err(_) => outcomes.push("panic"),
                    }
                }
                outcomes
            }));
        }
        for h in handles {
            let _ = h.join().expect("panic-storm worker must not hang");
        }
    });
    assert!(phase3.is_ok(), "stress phase 3: {phase3:?}");

    // Liveness: other slots still compute fine after the panic storm, and
    // the poisoned slot deterministically reports the previous panic.
    let v = system.get::<SrcQ>(DKey(0), provider).expect("liveness get");
    assert_eq!(v.semantic, SRC[0].load(Ordering::SeqCst));
    let poisoned = system.get::<SrcQ>(DKey(PANIC_SLOT), provider);
    assert!(
        poisoned.is_err(),
        "poisoned key must stay poisoned for the session: {poisoned:?}"
    );
}

// ── Granularity target tests ─────────────────────────────────────

/// Value type for the granularity tests: `semantic` is the stable
/// fingerprint (positional/raw data lives in `raw` and must NOT take part
/// in identity).
#[derive(Debug, Clone, PartialEq, Eq)]
struct FpValue {
    raw: String,
    semantic: u64,
}

/// The fingerprint is the SEMANTIC part only — `raw` (simulating
/// whitespace/comment text) is deliberately excluded.
impl StableHash for FpValue {
    fn stable_hash(&self) -> u64 {
        self.semantic
    }
}

/// Target: an input change whose fingerprint is UNCHANGED (simulated
/// whitespace/comment edit) must recompute the input but NOT any
/// downstream reader: downstream recompute delta == 0.
#[test]
fn g1_semantic_unchanged_input_no_downstream_recompute() {
    static I_COUNT: AtomicU64 = AtomicU64::new(0);
    static D_COUNT: AtomicU64 = AtomicU64::new(0);
    static I_SEMANTIC: AtomicU64 = AtomicU64::new(42);
    I_COUNT.store(0, Ordering::SeqCst);
    D_COUNT.store(0, Ordering::SeqCst);
    I_SEMANTIC.store(42, Ordering::SeqCst);

    struct GIn;
    impl Query for GIn {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("gin", "input query")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            let n = I_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
            FpValue {
                raw: format!("i_{}_{}", key.0, n),
                semantic: I_SEMANTIC.load(Ordering::SeqCst),
            }
        }
    }

    struct GDown;
    impl Query for GDown {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("gdown", "downstream query")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            let n = D_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
            let i = query_get::<GIn>(provider, key.clone()).expect("downstream input read");
            FpValue {
                raw: format!("d_{}_{}", key.0, n),
                semantic: 100 + i.semantic,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // First run: D computes, reads I.
    let _ = system.get::<GDown>(DKey(1), provider).unwrap();
    assert_eq!(I_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(D_COUNT.load(Ordering::SeqCst), 1);

    // "Whitespace-only" edit: invalidate the input node; its value changes
    // (raw differs) but its semantic fingerprint stays 42.
    system.invalidate::<GIn>(&DKey(1));
    let _ = system.get::<GIn>(DKey(1), provider).unwrap();
    assert_eq!(
        I_COUNT.load(Ordering::SeqCst),
        2,
        "input must recompute after invalidation"
    );

    // Downstream must NOT recompute: same-fingerprint change is invisible.
    let _ = system.get::<GDown>(DKey(1), provider).unwrap();
    assert_eq!(
        D_COUNT.load(Ordering::SeqCst),
        1,
        "downstream recompute delta must be 0 when the input fingerprint is unchanged"
    );
}

/// Derived: an input change whose fingerprint CHANGED (real semantic
/// edit) must propagate: the downstream reader must recompute.
#[test]
fn g1b_semantic_changed_input_downstream_recomputes() {
    static I_COUNT: AtomicU64 = AtomicU64::new(0);
    static D_COUNT: AtomicU64 = AtomicU64::new(0);
    static I_SEMANTIC: AtomicU64 = AtomicU64::new(42);
    I_COUNT.store(0, Ordering::SeqCst);
    D_COUNT.store(0, Ordering::SeqCst);
    I_SEMANTIC.store(42, Ordering::SeqCst);

    struct GIn;
    impl Query for GIn {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("gin", "input query")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            let n = I_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
            FpValue {
                raw: format!("i_{}_{}", key.0, n),
                semantic: I_SEMANTIC.load(Ordering::SeqCst),
            }
        }
    }

    struct GDown;
    impl Query for GDown {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("gdown", "downstream query")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            let n = D_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
            let i = query_get::<GIn>(provider, key.clone()).expect("downstream input read");
            FpValue {
                raw: format!("d_{}_{}", key.0, n),
                semantic: 100 + i.semantic,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // First run: D computes, reads I.
    let _ = system.get::<GDown>(DKey(1), provider).unwrap();
    assert_eq!(I_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(D_COUNT.load(Ordering::SeqCst), 1);

    // Real semantic edit: the input fingerprint changes 42 -> 43.
    system.invalidate::<GIn>(&DKey(1));
    I_SEMANTIC.store(43, Ordering::SeqCst);
    let _ = system.get::<GIn>(DKey(1), provider).unwrap();
    assert_eq!(
        I_COUNT.load(Ordering::SeqCst),
        2,
        "input must recompute after invalidation"
    );

    // Downstream MUST recompute: the fingerprint change must propagate.
    let _ = system.get::<GDown>(DKey(1), provider).unwrap();
    assert_eq!(
        D_COUNT.load(Ordering::SeqCst),
        2,
        "changed fingerprint must propagate to downstream readers"
    );
}
