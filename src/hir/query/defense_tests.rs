//! Defense tests (cycle / deadlock / poisoning / depth / dedup) and
//! granularity target tests (fingerprint-based invalidation).
//!
//! Every hang-prone test runs the query on a spawned thread and guards it
//! with `recv_timeout`; a timeout is reported as `Err("TIMEOUT")` rather
//! than wedging the test process.  All `QuerySystem` / provider objects are
//! `Box::leak`ed to `'static` so they can be moved into spawned threads.
//!
//! Baseline contract — HISTORICAL, recorded before the fixes landed
//! (the "un-fixed code" below is the pre-fix state, not the current
//! code; current state: full suite green):
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

fn leak_provider(system: &'static QuerySystem) -> &'static dyn QueryProvider<'static> {
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
    static RUNS: AtomicU64 = AtomicU64::new(0);
    ENTERED.store(false, Ordering::SeqCst);
    RUNS.store(0, Ordering::SeqCst);

    struct D5Q;
    impl Query for D5Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d5", "eval-always parallel query").with_eval_always(true)
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> DValue {
            RUNS.fetch_add(1, Ordering::SeqCst);
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
    // The doc promises "each requester computes its own value" — pin it:
    // no dedup means exactly two compute invocations for the two
    // requesters.
    assert_eq!(
        RUNS.load(Ordering::SeqCst),
        2,
        "eval_always must not deduplicate: each requester computes its \
         own value"
    );
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
    // Deterministic (d9 discipline): wait until t2 has REGISTERED its wait
    // edge before setting GO.  A fixed sleep is racy — under load it can
    // elapse while t2 is still inside `InFlightGuard::acquire`/`probe`, so
    // t1 can publish and drop its `Started` entry first; t2 then becomes a
    // second owner and COMPUTES == 2 (spurious failure).  `wait_graph` is
    // populated by `register_wait` immediately before t2 parks on t1's
    // latch and cleared by `remove_wait` after the wake — and t2 cannot
    // leave the latch until t1 signals, which only happens after GO.
    let mut parked = false;
    for _ in 0..200 {
        if !system.wait_graph.read().is_empty() {
            parked = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(parked, "t2 must be parked on t1's latch before GO is set");
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

/// d9: `clear_all` must be quiesced and must not strand latch waiters,
/// and the post-reset state must be clean (no inherited verified_gen
/// stamps on reused node indices, no stale node_to_key entries).
///
/// Window 1 (latch stranding): owner O computes a slow query for key K;
/// waiter W parks on O's job latch; `clear_all` wipes `active_keys`
/// WITHOUT quiescence, so O's completion path `active.remove(&akey)`
/// finds nothing and `signal_complete` never runs — W blocks forever
/// (the latch is one-shot).
///
/// Window 2 (inherited stamp): after reset, `next_index` restarts at 1,
/// so freshly allocated nodes REUSE indices whose `verified_gen` slots
/// were never cleared in `reset()` — a new node can inherit a stamp
/// claiming its (not yet recorded) closure was verified.
///
/// Window 3 (stale node_to_key): `QuerySystem.node_to_key` is not
/// cleared by `clear_all`; after re-registration the same index may map
/// to a stale (TypeId, key) pair from the pre-reset world.
#[test]
fn d9_clear_all_quiescence_and_reset_hygiene() {
    static SLOW_CONTENT: AtomicU64 = AtomicU64::new(1);
    static SLOW_RUNS: AtomicU64 = AtomicU64::new(0);
    static ENTERED: AtomicBool = AtomicBool::new(false);
    static RELEASE: AtomicBool = AtomicBool::new(false);
    SLOW_CONTENT.store(1, Ordering::SeqCst);
    SLOW_RUNS.store(0, Ordering::SeqCst);
    ENTERED.store(false, Ordering::SeqCst);
    RELEASE.store(false, Ordering::SeqCst);

    struct SlowQ;
    impl Query for SlowQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d9_slow", "slow query")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            SLOW_RUNS.fetch_add(1, Ordering::SeqCst);
            // Only the "slow" key parks; the warm-up key (1) returns
            // immediately so the cached-hit stamping works.
            if key.0 != 9 {
                return FpValue {
                    raw: format!("slow_{}", key.0),
                    semantic: SLOW_CONTENT.load(Ordering::SeqCst),
                };
            }
            ENTERED.store(true, Ordering::SeqCst);
            // Park until the test releases us — but bail out after 2s
            // so a quiesced `clear_all` (which waits for this compute
            // to finish) cannot turn the test itself into a three-way
            // deadlock.  The pre-fix hang is still deterministic: without
            // quiescence, clear_all wipes active_keys while we park;
            // our completion path then finds no entry and never signals
            // the parked waiter.
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            let mut spins: u64 = 0;
            while !RELEASE.load(Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    break;
                }
                spins += 1;
                if spins % (1 << 20) == 0 {
                    std::thread::yield_now();
                }
                std::hint::spin_loop();
            }
            FpValue {
                raw: format!("slow_{}", key.0),
                semantic: SLOW_CONTENT.load(Ordering::SeqCst),
            }
        }
    }

    struct TopQ;
    impl Query for TopQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d9_top", "top reader")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            let s = query_get::<SlowQ>(provider, key.clone()).expect("top reads slow");
            FpValue {
                raw: format!("top_{}", s.semantic),
                semantic: 100 + s.semantic,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Warm: TopQ(1) → SlowQ(1); then a cached hit stamps the closure.
    let v = system.get::<TopQ>(DKey(1), provider).unwrap();
    assert_eq!(v.semantic, 101);
    let v = system.get::<TopQ>(DKey(1), provider).unwrap();
    assert_eq!(v.semantic, 101);

    // ---- Window 1: owner computing + waiter parked + clear_all ----
    let owner = thread::spawn(move || {
        system
            .get::<SlowQ>(DKey(9), provider)
            .map_err(|e| e.to_string())
    });
    wait_flag(&ENTERED);

    let waiter = thread::spawn(move || {
        // The first request parks on the owner's latch until the test
        // releases the compute.
        system
            .get::<SlowQ>(DKey(9), provider)
            .map_err(|e| e.to_string())
    });
    // Give the waiter time to park on the latch (poll for the wait edge
    // appearing in the wait_graph — deterministic, not a sleep guess).
    let mut parked = false;
    for _ in 0..200 {
        if !system.wait_graph.read().is_empty() {
            parked = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(parked, "waiter must have registered its wait edge");

    // Now the critical interleaving: the waiter is parked on the owner's
    // latch.  Call clear_all BEFORE releasing the owner — this is the
    // exact pre-fix hang: without quiescence, clear_all wipes
    // active_keys/vtables while the owner's compute is still parked; the
    // owner's completion path then finds no entry (`active.remove` →
    // None) and never calls signal_complete, so the parked waiter blocks
    // on the one-shot latch FOREVER.  With the fix, clear_all's quiesce
    // write lock queues behind the owner's read lock; the owner's 2s
    // self-timeout lets it finish and signal normally, and clear_all
    // then tears down against empty state.  Either way both threads must
    // come back.
    system.clear_all();
    RELEASE.store(true, Ordering::SeqCst);

    // Both threads must return (bounded); before the fix the waiter
    // hangs forever on the never-signaled latch.
    let r_owner = owner.join().expect("d9 owner thread panicked");
    let r_waiter = waiter.join().expect("d9 waiter thread panicked");
    assert!(
        r_owner.is_ok() && r_waiter.is_ok(),
        "both participants must survive clear_all: {r_owner:?} {r_waiter:?}"
    );

    // ---- Window 3 (stale node_to_key) + Window 2 (inherited stamp) ----
    // Stage a NON-ZERO stamp so inheritance becomes observable: the
    // invalidation marks SlowQ(9) red 0→1 (bumping dirty_generation to
    // 1), the recompute makes it green again, and the following cached
    // hit runs the transitive verifier, which stamps the closure with
    // the current generation.
    system.invalidate::<SlowQ>(&DKey(9));
    let _ = system.get::<SlowQ>(DKey(9), provider);
    let _ = system.get::<SlowQ>(DKey(9), provider);
    let vtable = system
        .find_vtable::<SlowQ>()
        .expect("SlowQ must be registered after its gets");
    let node9 = vtable
        .lookup_node(&DKey(9))
        .expect("SlowQ(9) must have a stable node");
    assert!(
        system.dep_graph.verified_gen_of(node9) >= 1,
        "precondition: the verify pass must have stamped the closure \
         (dirty_generation was bumped by the invalidation)"
    );
    assert!(
        !system.node_to_key.read().is_empty(),
        "precondition: the reverse mapping must be populated pre-clear"
    );

    // The teardown under test.
    system.clear_all();

    // Window 3: `node_to_key` must be wiped with everything else.
    assert!(
        system.node_to_key.read().is_empty(),
        "clear_all must clear node_to_key (Window 3: no stale (TypeId, \
         key) entries may survive the reset)"
    );

    // Window 2: rebuild the same request — `next_index` restarted at 1,
    // so the fresh node REUSES the pre-clear index.  Its stamp must be
    // zero: an inherited `verified_gen >= current_gen` would let its
    // later cached hits skip closure verification entirely.
    let _ = system.get::<SlowQ>(DKey(9), provider);
    let vtable2 = system
        .find_vtable::<SlowQ>()
        .expect("SlowQ must be re-registered after the rebuild get");
    let node9b = vtable2
        .lookup_node(&DKey(9))
        .expect("rebuilt SlowQ(9) must have a node");
    assert_eq!(
        node9b, node9,
        "precondition for Window 2: the rebuilt node reuses the pre-clear \
         index (next_index restarted at 1)"
    );
    assert_eq!(
        system.dep_graph.verified_gen_of(node9b),
        0,
        "a freshly allocated node must not inherit the pre-clear \
         verified_gen stamp (Window 2)"
    );
}

/// d10: the side-door `invalidate_node` bypasses quiescence.
///
/// `QuerySystem::invalidate::<Q>` takes the quiesce write lock exactly
/// because an invalidation landing between a compute's input reads and
/// its publish is ERASED: the publisher's `clear_dirty` wipes the
/// just-set red bit (and `cache.stage` resurrects the value).  The
/// public side door `invalidate_node` (and everything else exposed
/// through the pub `dep_graph` field) skips that lock, so the erase
/// window is wide open: the node ends up permanently green holding the
/// PRE-change value, and every later get serves it.
///
/// Orchestration: A computes PQ(2), snapshots INPUT (=0) BEFORE parking;
/// B lands the side-door invalidation while A is parked (pre-fix: it
/// completes in microseconds and B_DONE is observed immediately).  Main
/// then releases A, whose publish erases B's red bit.  The final get
/// must observe the invalidation (recompute with INPUT=1).  With the
/// quiesced side door, B blocks on the write lock until A finishes, so
/// its red bit lands AFTER the publish and survives — the same oracle
/// holds.  A's 2s self-timeout keeps the quiesced variant from turning
/// the test itself into a three-way cycle (see d9).
#[test]
fn d10_side_door_invalidate_erased_by_publish() {
    static INPUT: AtomicU64 = AtomicU64::new(0);
    static ENTERED: AtomicBool = AtomicBool::new(false);
    static RELEASE: AtomicBool = AtomicBool::new(false);
    static B_DONE: AtomicBool = AtomicBool::new(false);
    INPUT.store(0, Ordering::SeqCst);
    ENTERED.store(false, Ordering::SeqCst);
    RELEASE.store(false, Ordering::SeqCst);
    B_DONE.store(false, Ordering::SeqCst);

    struct PQ;
    impl Query for PQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d10_p", "side-door victim")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            // Snapshot BEFORE parking: the pre-fix race publishes this
            // stale snapshot while the invalidation is already pending.
            let snap = INPUT.load(Ordering::SeqCst);
            ENTERED.store(true, Ordering::SeqCst);
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !RELEASE.load(Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    break;
                }
                std::hint::spin_loop();
            }
            FpValue {
                raw: format!("d10_{}", key.0),
                semantic: snap,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    let a = thread::spawn(move || {
        system
            .get::<PQ>(DKey(2), provider)
            .map_err(|e| e.to_string())
    });
    wait_flag(&ENTERED);

    INPUT.store(1, Ordering::SeqCst);
    let node = system
        .find_vtable::<PQ>()
        .and_then(|vt| vt.lookup_node(&DKey(2)))
        .expect("node must exist once compute started");

    let b = thread::spawn(move || {
        // Side door: NO quiescence in the pre-fix code.
        system.invalidate_node(node);
        B_DONE.store(true, Ordering::SeqCst);
    });

    // Bounded spin for the side-door invalidation to land.
    let mut landed = false;
    for _ in 0..600 {
        if B_DONE.load(Ordering::SeqCst) {
            landed = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }

    RELEASE.store(true, Ordering::SeqCst);
    let r_a = a.join().expect("d10 A panicked");
    b.join().expect("d10 B panicked");
    assert!(r_a.is_ok(), "A must survive: {r_a:?}");

    // Oracle: the side-door invalidation must not be swallowed by A's
    // publish.  Pre-fix: A's clear_dirty erased B's red bit and the
    // cached value 0 is served (assertion fails with 0).  Post-fix: B's
    // red bit lands after A's publish, forcing the recompute to 1.
    let v = system.get::<PQ>(DKey(2), provider).unwrap();
    assert_eq!(
        v.semantic, 1,
        "side-door invalidation must survive the in-flight publish (landed={landed})"
    );
}

/// d11: a mutation called from inside a running query self-deadlocks on
/// the non-reentrant quiesce RwLock: the outermost `get_shared` frame
/// holds `quiesce.read()` for the whole compute, and the mutation path
/// (`with_quiescence`) takes `quiesce.write()` on the SAME thread — a
/// read→write upgrade parking_lot cannot serve (the writer waits for
/// the reader that is itself).  Deterministic hang, pre-fix.
///
/// Post-fix the mutation is DEFERRED to the outermost guard drop —
/// after the read lock is released and after the in-flight publish —
/// so the query completes AND the invalidation still takes effect
/// (OtherQ recomputes with the new input on the next request).
#[test]
fn d11_reentrant_invalidate_inside_compute() {
    static OTHER: AtomicU64 = AtomicU64::new(0);
    static TRIGGER: AtomicBool = AtomicBool::new(false);
    static OTHER_RUNS: AtomicU64 = AtomicU64::new(0);
    static DRIVER_RUNS: AtomicU64 = AtomicU64::new(0);
    OTHER.store(0, Ordering::SeqCst);
    TRIGGER.store(false, Ordering::SeqCst);
    OTHER_RUNS.store(0, Ordering::SeqCst);
    DRIVER_RUNS.store(0, Ordering::SeqCst);

    struct OtherQ;
    impl Query for OtherQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d11_other", "invalidation target")
        }
        fn compute(_: &DKey, _: &dyn QueryProvider) -> FpValue {
            OTHER_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: "other".into(),
                semantic: OTHER.load(Ordering::SeqCst),
            }
        }
    }

    struct DriverQ;
    impl Query for DriverQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d11_driver", "invalidates from inside compute")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            DRIVER_RUNS.fetch_add(1, Ordering::SeqCst);
            if TRIGGER.load(Ordering::SeqCst) {
                // Reentrant mutation: this thread's outermost get_shared
                // frame holds quiesce.read() right now.
                provider.query_system().invalidate::<OtherQ>(&DKey(0));
            }
            FpValue {
                raw: format!("driver{}", key.0),
                semantic: 7,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Warm both queries (TRIGGER false: no invalidation yet).
    let v = system.get::<OtherQ>(DKey(0), provider).unwrap();
    assert_eq!(v.semantic, 0);
    let v = system.get::<DriverQ>(DKey(1), provider).unwrap();
    assert_eq!(v.semantic, 7);
    assert_eq!(OTHER_RUNS.load(Ordering::SeqCst), 1);

    // Fire the reentrant invalidation from inside DriverQ's compute.
    // DriverQ(1) is cached from the warmup — invalidate it so the next
    // get actually RE-RUNS the compute (otherwise the trigger never
    // fires and the reentrant path is never exercised).
    TRIGGER.store(true, Ordering::SeqCst);
    OTHER.store(1, Ordering::SeqCst);
    system.invalidate::<DriverQ>(&DKey(1));
    let r = run_with_guard(Duration::from_secs(5), move || {
        system.get::<DriverQ>(DKey(1), provider).map(|v| v.semantic)
    });
    // Pre-fix this is Err("TIMEOUT"): the compute deadlocks itself.
    let semantic = r
        .expect("reentrant invalidate must not deadlock the query")
        .expect("driver get must succeed");
    assert_eq!(semantic, 7);

    // The deferred invalidation must take effect at the outermost guard
    // drop: OtherQ recomputes with the new input.
    let v = system.get::<OtherQ>(DKey(0), provider).unwrap();
    assert_eq!(
        v.semantic, 1,
        "deferred reentrant invalidation must take effect"
    );
    assert_eq!(OTHER_RUNS.load(Ordering::SeqCst), 2);
}

/// d12: `DefIdCache` must survive sparse / sentinel DefId indices.
///
/// A naive `resize_with(idx + 1)` on `DefId(usize::MAX / 2)` (the
/// "unknown" sentinel used elsewhere in this codebase) would attempt a
/// multi-exabyte allocation and abort.  Indices past the dense limit
/// must go to the overflow map (stage / probe / unstage / for_each).
#[test]
fn d12_defid_cache_sparse_sentinel_no_oom() {
    #[derive(Debug, Clone, PartialEq)]
    struct V(String);
    let cache = DefIdCache::<V>::new();
    let sentinel = crate::hir::types::DefId(usize::MAX / 2);

    // Dense path still works.
    let (dense_node, _, _) = cache.probe(&crate::hir::types::DefId(3), || {
        dep_graph::DepNodeIndex(3)
    });
    cache.stage(
        crate::hir::types::DefId(3),
        dense_node,
        Arc::new(V("dense".into())),
    );
    assert_eq!(
        cache
            .probe(&crate::hir::types::DefId(3), || unreachable!())
            .1
            .unwrap()
            .0,
        "dense"
    );

    // Sparse sentinel: must NOT attempt an unbounded resize.
    let (sentinel_node, _, _) = cache.probe(&sentinel, || dep_graph::DepNodeIndex(999));
    cache.stage(sentinel, sentinel_node, Arc::new(V("sentinel".into())));
    assert_eq!(
        cache.probe(&sentinel, || unreachable!()).1.unwrap().0,
        "sentinel"
    );
    assert!(
        cache
            .probe(
                &crate::hir::types::DefId((1 << 20) + 5),
                || dep_graph::DepNodeIndex(8888),
            )
            .1
            .is_none(),
        "unrelated sparse index misses cleanly"
    );

    // Unstage and enumerate across both paths.  unstage withdraws the
    // VALUE; the NODE stays registered on BOTH paths (merge contract 1).
    assert_eq!(cache.unstage(&sentinel), Some(sentinel_node));
    assert!(
        cache.probe(&sentinel, || unreachable!()).1.is_none(),
        "sentinel value is gone"
    );
    assert_eq!(
        cache.node_of(&sentinel),
        Some(sentinel_node),
        "overflow-path node survives unstage"
    );
    let mut seen = Vec::new();
    cache.for_each(|_k, v| seen.push(v.0.clone()));
    assert_eq!(seen, vec!["dense".to_string()]);
}

// ── Hash-collision hardening (exact ActiveKey) ────────────────────

/// Every key of this type hashes to NOTHING (constant hash): all keys
/// collide on the same `query_key_hash` — exactly the situation the
/// `(TypeId, hash)`-keyed `active_keys` map mishandles. The exact-key
/// map must keep them fully independent. (The constant hash also
/// degenerates the DefaultCache into one bucket — Eq keeps it correct,
/// only slower; test-only type.)
#[derive(Clone, Debug, PartialEq, Eq)]
struct CollideKey(usize);
impl std::hash::Hash for CollideKey {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {}
}
impl QueryKey for CollideKey {}

/// Same-thread nested collision: K1's compute requests K2 — a distinct
/// key with the same hash. Pre-fix, `acquire_key_slot`'s
/// `job.owner == me` guard reported a false "query cycle detected
/// (owner thread)" for K2 (K2_RUNS stayed 0); post-fix, the exact-key
/// map gives K2 its own slot and it computes normally.
#[test]
fn collision_same_thread_nested_key_must_compute() {
    static K2_RUNS: AtomicU64 = AtomicU64::new(0);
    K2_RUNS.store(0, Ordering::SeqCst);

    struct CQ;
    impl Query for CQ {
        type Key = CollideKey;
        type Value = DValue;
        type Cache = DefaultCache<CollideKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("collide_nested", "same-thread hash-collision query")
        }
        fn compute(key: &CollideKey, provider: &dyn QueryProvider) -> DValue {
            match key.0 {
                1 => {
                    // Nested request of the hash-colliding K2. The result
                    // is discarded — what matters is that K2 COMPUTES.
                    let _ = query_get::<CQ>(provider, CollideKey(2));
                }
                2 => {
                    K2_RUNS.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);
    let result = run_with_guard(GUARD, move || {
        system
            .get::<CQ>(CollideKey(1), provider)
            .map(|v| v.semantic)
    });
    match result {
        Ok(Ok(sem)) => assert_eq!(sem, 1, "K1 must compute"),
        other => panic!("K1 must compute despite the nested collision: {other:?}"),
    }
    assert_eq!(
        K2_RUNS.load(Ordering::SeqCst),
        1,
        "the hash-colliding K2 must compute — a false cycle here is the pre-fix bug"
    );
}

/// Poison collision: K1 panics; K2 — a distinct key sharing K1's hash —
/// must still compute. Pre-fix, the `Poisoned` entry was keyed by hash
/// only, so K2 inherited a PERMANENT "query previously panicked" error
/// (nothing ever cleared it except clear_all).
#[test]
fn collision_poison_is_exactly_per_key() {
    static K1_PANICS: AtomicBool = AtomicBool::new(false);
    K1_PANICS.store(false, Ordering::SeqCst);

    struct PQ;
    impl Query for PQ {
        type Key = CollideKey;
        type Value = DValue;
        type Cache = DefaultCache<CollideKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("collide_poison", "poison hash-collision query")
        }
        fn compute(key: &CollideKey, _: &dyn QueryProvider) -> DValue {
            if key.0 == 1 && K1_PANICS.load(Ordering::SeqCst) {
                panic!("k1 compute boom");
            }
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // 1) K1 panics → K1 is poisoned.
    K1_PANICS.store(true, Ordering::SeqCst);
    let r1 = run_with_guard(GUARD, move || system.get::<PQ>(CollideKey(1), provider));
    assert!(
        matches!(r1, Err(ref t) if t == "PANIC"),
        "K1 must panic: {r1:?}"
    );

    // 2) K2 (same hash) must compute fine.
    let r2 = run_with_guard(GUARD, move || {
        system
            .get::<PQ>(CollideKey(2), provider)
            .map(|v| v.semantic)
    });
    assert_eq!(r2.unwrap().unwrap(), 2, "K2 must not inherit K1's poison");

    // 3) K1 stays poisoned for ITS OWN key: the next request gets the
    //    loud poisoned error without running compute again.
    let r3 = run_with_guard(GUARD, move || system.get::<PQ>(CollideKey(1), provider));
    match r3 {
        Ok(Err(e)) => {
            let m = e.message.to_lowercase();
            assert!(m.contains("panicked"), "K1 must stay poisoned: {e:?}");
        }
        other => panic!("K1 must remain poisoned for its own key: {other:?}"),
    }
}

/// Nested par_get must not lose the current task's dependency edges.
/// Pre-fix, rayon work-stealing ran stolen items on workers with an
/// empty TASK_STACK, where `DepGraph::read` is a silent no-op — the
/// outer query's edges to the par_get items never existed, and a later
/// invalidation of an item never dirtied the outer query (stale cache
/// served forever).  Post-fix, a nested par_get runs sequentially on
/// the calling thread, so every item's edge is recorded.  (Pre-fix this
/// test is FLAKY-red: it fails iff DKey(3) happened to be stolen;
/// post-fix it is deterministic-green.)
#[test]
fn nested_par_get_records_deps_on_calling_thread() {
    static INNER_CONTENT: AtomicU64 = AtomicU64::new(1);
    static INNER_RUNS: AtomicU64 = AtomicU64::new(0);
    static OUTER_RUNS: AtomicU64 = AtomicU64::new(0);
    INNER_CONTENT.store(1, Ordering::SeqCst);
    INNER_RUNS.store(0, Ordering::SeqCst);
    OUTER_RUNS.store(0, Ordering::SeqCst);

    struct NInnerQ;
    impl Query for NInnerQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("np_inner", "nested par_get inner")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            INNER_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: format!("np_inner_{}", key.0),
                semantic: INNER_CONTENT.load(Ordering::SeqCst) + key.0 as u64,
            }
        }
    }

    struct NOuterQ;
    impl Query for NOuterQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("np_outer", "nested par_get driver")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            OUTER_RUNS.fetch_add(1, Ordering::SeqCst);
            let keys: Vec<DKey> = (0..16).map(DKey).collect();
            let results = provider.query_system().par_get::<NInnerQ>(&keys, provider);
            let sum: u64 = results
                .iter()
                .filter_map(|r| r.as_ref().ok())
                .map(|v| v.semantic)
                .sum();
            FpValue {
                raw: format!("np_outer_{}", key.0),
                semantic: sum,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // 1) Warm: the outer query computes through the NESTED par_get.
    //    sum = Σ(1+i) for i in 0..16 = 136
    let v = system.get::<NOuterQ>(DKey(0), provider).unwrap();
    assert_eq!(INNER_RUNS.load(Ordering::SeqCst), 16);
    assert_eq!(OUTER_RUNS.load(Ordering::SeqCst), 1);
    assert_eq!(v.semantic, 136);

    // 2) Semantic edit of one inner input + invalidation.
    INNER_CONTENT.store(100, Ordering::SeqCst);
    system.invalidate::<NInnerQ>(&DKey(3));

    // 3) The outer query MUST recompute: its dependency edge to DKey(3)
    //    must have been recorded by the nested par_get.  New sum:
    //    136 - (1+3) + (100+3) = 235.
    let v = system.get::<NOuterQ>(DKey(0), provider).unwrap();
    assert_eq!(
        OUTER_RUNS.load(Ordering::SeqCst),
        2,
        "outer must recompute after an inner edit — the nested par_get \
         must have recorded the dependency edge"
    );
    assert_eq!(v.semantic, 235);
    assert_eq!(
        INNER_RUNS.load(Ordering::SeqCst),
        17,
        "only DKey(3) recomputes"
    );
}

/// d17: the cached path must not serve a stale Arc when the
/// node ITSELF turns dirty between `probe` and the verification walk.
///
/// Chronology: R is cached and green. E is invalidated BEFORE the
/// reader enters (post-quiesce-at-entry the write lock could not be
/// taken while the reader parks holding the read lock).  A reader then
/// enters `get::<R>`'s cached branch — capturing the OLD Arc — and
/// parks in the probe→verify window (d17 hooks).  The main thread
/// recomputes E with a CHANGED fingerprint; E's publish marks R red
/// through the recorded reverse edge (lock-free, no quiesce needed —
/// this works with the in-flight guard at either placement).  When the
/// reader resumes, `verify_green_transitive_with_dirty(R)` returns R
/// ITSELF (`dirty_dep == node_index`).  Pre-fix, the re-request
/// machinery recomputed R (publishing a fresh Arc into the cache) and
/// then served the STALE captured Arc — cache and caller disagreed.
/// Post-fix, the node-itself-dirty case falls through to the compute
/// path, which returns the freshly computed value.
#[test]
fn d17_cached_path_node_itself_dirty_serves_fresh_value() {
    static E_CONTENT: AtomicU64 = AtomicU64::new(1);
    static E_RUNS: AtomicU64 = AtomicU64::new(0);
    static R_RUNS: AtomicU64 = AtomicU64::new(0);
    E_CONTENT.store(1, Ordering::SeqCst);
    E_RUNS.store(0, Ordering::SeqCst);
    R_RUNS.store(0, Ordering::SeqCst);
    D17_PARK.store(true, Ordering::SeqCst);
    D17_TARGET.store(0, Ordering::SeqCst);

    struct D17E;
    impl Query for D17E {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d17_inner", "d17 leaf input")
        }
        fn compute(_key: &DKey, _: &dyn QueryProvider) -> FpValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            let s = E_CONTENT.load(Ordering::SeqCst);
            FpValue {
                raw: format!("d17e_{s}"),
                semantic: s,
            }
        }
    }
    struct D17R;
    impl Query for D17R {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d17_reader", "d17 reader over d17_inner")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            R_RUNS.fetch_add(1, Ordering::SeqCst);
            let e = query_get::<D17E>(provider, key.clone()).unwrap();
            FpValue {
                raw: format!("d17r_{}", e.semantic),
                semantic: e.semantic * 10,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // 1) Warm: R computes once (reads E); both cached; R green.
    let v = system.get::<D17R>(DKey(0), provider).unwrap();
    assert_eq!(v.semantic, 10, "warm value");
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 1);
    assert_eq!(R_RUNS.load(Ordering::SeqCst), 1);

    // 2) Invalidate E while NO reader holds the quiesce lock.
    system.invalidate::<D17E>(&DKey(0));

    // 3) Reader: cached branch (old Arc captured), parks in the window.
    D17_TARGET.store(D17R::descriptor().name.as_ptr() as usize, Ordering::SeqCst);
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let r = system.get::<D17R>(DKey(0), provider);
        let _ = tx.send(r.map(|v| v.semantic));
    });
    // Deterministic wait for the park (bounded, d9 discipline).
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !D17_ARRIVED.load(Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < deadline,
            "reader never reached the cached branch"
        );
        std::thread::yield_now();
    }

    // 4) Main: recompute E with a CHANGED fingerprint → its publish
    //    marks R red through the recorded reverse edge (no quiesce).
    E_CONTENT.store(100, Ordering::SeqCst);
    let e2 = system.get::<D17E>(DKey(0), provider).unwrap();
    assert_eq!(e2.semantic, 100);
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 2);

    // 5) Release the reader and collect its result.
    D17_PARK.store(false, Ordering::SeqCst);
    let got = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("reader must finish");
    D17_TARGET.store(0, Ordering::SeqCst);

    // 6) The reader must observe the NEW value (R recomputed against
    //    the new E), never the stale captured Arc.
    assert_eq!(
        got.unwrap(),
        1000,
        "stale serve: the cached path returned the pre-invalidation Arc"
    );
    assert_eq!(
        R_RUNS.load(Ordering::SeqCst),
        2,
        "R must recompute after its own node turned red"
    );
}

/// d14: a panic during query FINALIZATION (outside the
/// compute `catch_unwind`) must still release every waiter parked on
/// the job's one-shot latch.
///
/// The vehicle is the debug semantic-class assertion: a SEMANTIC query
/// that reads a DOC node trips `assert_semantic_reads` inside
/// `finish_task_and_publish` — which used to run (a) AFTER the reverse
/// edge was inserted (validate-before-mutate violation) and (b) OUTSIDE
/// the compute `catch_unwind`, so the panic skipped the active-key
/// cleanup (`active.remove` + `clear_waiters_of_me` + `latch.poison`).
/// A waiter that had parked on the latch hung forever (TIMEOUT).
/// Post-fix: validation runs BEFORE any graph mutation, finalization
/// panics are caught and routed through the poison-and-resume cleanup,
/// so the waiter wakes with a query error instead of hanging.
#[cfg(debug_assertions)]
#[test]
fn d14_finalize_panic_releases_waiters() {
    static ENTERED: AtomicBool = AtomicBool::new(false);
    static RELEASE: AtomicBool = AtomicBool::new(false);
    ENTERED.store(false, Ordering::SeqCst);
    RELEASE.store(false, Ordering::SeqCst);
    D17_PARK.store(false, Ordering::SeqCst);
    D17_TARGET.store(0, Ordering::SeqCst);

    struct D14E;
    impl Query for D14E {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            // DOC channel node — the thing a semantic query must NOT read.
            QueryDescriptor::new("d14_doc", "doc channel leaf")
                .with_content_class(dep_graph::ContentClass::DOC)
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            FpValue {
                raw: format!("d14doc_{}", key.0),
                semantic: 7,
            }
        }
    }
    struct D14S;
    impl Query for D14S {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            // SEMANTIC by default — but its compute reads the DOC node:
            // a design bug the debug assertion must catch.
            QueryDescriptor::new("d14_sem", "semantic query (buggy: reads doc)")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            ENTERED.store(true, Ordering::SeqCst);
            let doc = query_get::<D14E>(provider, key.clone()).unwrap();
            // Park so the waiter can reach the latch BEFORE the owner's
            // finalization panics (2s self-timeout, d9 discipline).
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !RELEASE.load(Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    break;
                }
                std::thread::yield_now();
            }
            FpValue {
                raw: format!("d14sem_{}", doc.semantic),
                semantic: doc.semantic,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // 1) Owner thread (spawned directly — `run_with_guard` would block
    //    the main thread here while the owner parks in compute awaiting
    //    RELEASE).  Computes S → reads the DOC node E → parks; when
    //    released, its FINALIZATION trips the assertion and panics.
    let (otx, orx) = mpsc::channel();
    {
        let (sys, prov) = (system, provider);
        thread::spawn(move || {
            let _ = otx.send(sys.get::<D14S>(DKey(9), prov).map(|v| v.semantic));
        });
    }
    wait_flag(&ENTERED);

    // 2) Waiter: must reach the owner's latch and park there BEFORE the
    //    owner's finalization runs (wait_graph poll — the edge is
    //    registered just before blocking).
    let (wtx, wrx) = mpsc::channel();
    {
        let (sys, prov) = (system, provider);
        thread::spawn(move || {
            let _ = wtx.send(
                sys.get::<D14S>(DKey(9), prov)
                    .map(|v| v.semantic)
                    .map_err(|e| e.message.to_string()),
            );
        });
    }
    let park_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while system.wait_graph.read().is_empty() {
        assert!(
            std::time::Instant::now() < park_deadline,
            "waiter never reached the wait path"
        );
        std::thread::yield_now();
    }

    // 3) Release the owner → its finalization panics.
    RELEASE.store(true, Ordering::SeqCst);
    match guard_recv_dur(orx, GUARD) {
        Err(t) => assert_eq!(t, "PANIC", "owner must panic in finalization: {t}"),
        other => panic!("owner must panic in finalization: {other:?}"),
    }

    // 4) The waiter MUST terminate promptly (poisoned latch or Poisoned
    //    entry) — pre-fix it hung forever on the never-signaled latch.
    //    Poll the channel instead of joining: pre-fix the thread never
    //    returns, and join() would hang the test instead of failing it.
    match guard_recv_dur(wrx, GUARD) {
        Ok(Err(msg)) => {
            let m = msg.to_lowercase();
            assert!(
                m.contains("poisoned") || m.contains("panicked"),
                "waiter must see a query error, got: {msg}"
            );
        }
        other => panic!("waiter must not hang or succeed: {other:?}"),
    }

    // 5) The key must be poisoned for its next getter (cleanup semantics
    //    identical to a compute panic).
    let (ntx, nrx) = mpsc::channel();
    {
        let (sys, prov) = (system, provider);
        thread::spawn(move || {
            let _ = ntx.send(
                sys.get::<D14S>(DKey(9), prov)
                    .map(|v| v.semantic)
                    .map_err(|e| e.message.to_string()),
            );
        });
    }
    match guard_recv_dur(nrx, GUARD) {
        Ok(Err(msg)) => assert!(
            msg.to_lowercase().contains("panicked"),
            "key must stay poisoned after the finalize panic: {msg}"
        ),
        other => panic!("next getter must see the poisoned key: {other:?}"),
    }
}

/// d15: `clear_all` must not interleave with a getter's
/// PREFLIGHT.  Since the quiesce read guard was moved to the very top
/// of `get_shared` — before vtable resolution, node allocation and key
/// registration — a reader parked between `probe` and the compute path
/// still holds the quiesce read lock and `clear_all`'s write lock
/// cannot be granted.  (Before that move, the guard sat after the
/// preflight, leaving a window in which `clear_all` reset the dep graph
/// (`next_index = 1`) and the registry under the parked reader's feet,
/// letting it publish an old-generation vtable against a recycled node
/// index — cross-generation corruption.)
///
/// The d17 park hook (cached branch, after `probe`) marks the window;
/// `query_names` emptiness is the completion observable for `clear_all`
/// (it is cleared inside `mutate_world`, past the quiesce write lock).
#[test]
fn d15_clear_all_waits_for_quiesced_reader() {
    static E_RUNS: AtomicU64 = AtomicU64::new(0);
    static R_RUNS: AtomicU64 = AtomicU64::new(0);
    E_RUNS.store(0, Ordering::SeqCst);
    R_RUNS.store(0, Ordering::SeqCst);
    D15_PARK.store(true, Ordering::SeqCst);
    D15_TARGET.store(0, Ordering::SeqCst);

    struct D15E;
    impl Query for D15E {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d15_inner", "d15 leaf input")
        }
        fn compute(_key: &DKey, _: &dyn QueryProvider) -> FpValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: "d15e".into(),
                semantic: 3,
            }
        }
    }
    struct D15R;
    impl Query for D15R {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d15_reader", "d15 reader over d15_inner")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            R_RUNS.fetch_add(1, Ordering::SeqCst);
            let e = query_get::<D15E>(provider, key.clone()).unwrap();
            FpValue {
                raw: format!("d15r_{}", e.semantic),
                semantic: e.semantic * 10,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Warm: R and E cached, R green.
    let v = system.get::<D15R>(DKey(0), provider).unwrap();
    assert_eq!(v.semantic, 30, "warm value");
    assert_eq!(R_RUNS.load(Ordering::SeqCst), 1);
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 1);

    // Reader parks in the cached branch (post-probe, pre-serve) via the
    // d15-specific hooks — independent of d17's so the parallel suite
    // cannot desynchronize the two orchestrations.
    D15_TARGET.store(D15R::descriptor().name.as_ptr() as usize, Ordering::SeqCst);
    let (rtx, rrx) = mpsc::channel();
    {
        let (sys, prov) = (system, provider);
        thread::spawn(move || {
            let _ = rtx.send(sys.get::<D15R>(DKey(0), prov).map(|v| v.semantic));
        });
    }
    let park_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !D15_ARRIVED.load(Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < park_deadline,
            "reader never reached the cached branch"
        );
        std::thread::yield_now();
    }

    // While the reader is parked, clear_all must NOT get past its
    // quiesce write lock: the parked reader holds the read guard taken
    // BEFORE its preflight.
    let (ctx, crx) = mpsc::channel::<()>();
    {
        let sys = system;
        thread::spawn(move || {
            sys.clear_all();
            let _ = ctx.send(());
        });
    }
    let clear_deadline = std::time::Instant::now() + Duration::from_millis(300);
    while std::time::Instant::now() < clear_deadline {
        assert!(
            !system.query_names.read().is_empty(),
            "clear_all completed while a quiesced reader was parked in its \
             preflight — the race from the old guard placement is back"
        );
        std::thread::yield_now();
    }

    // Release the reader: it must serve its cached value (its vtable
    // Arc and node index were all resolved before the park, and
    // clear_all never ran under it).
    D15_PARK.store(false, Ordering::SeqCst);
    D15_TARGET.store(0, Ordering::SeqCst);
    let served = guard_recv_dur(rrx, GUARD).expect("reader must finish");
    assert_eq!(
        served.expect("parked reader must succeed"),
        30,
        "parked reader must serve its cached value"
    );

    // clear_all completes only AFTER the reader's guard drops.
    guard_recv_dur(crx, GUARD).expect("clear_all must complete after the reader");

    // Post-clear: a fresh get recomputes from scratch (cache and graph
    // were reset, and the fresh registry is consistent).
    let v2 = run_with_guard(GUARD, move || {
        system.get::<D15R>(DKey(0), provider).map(|v| v.semantic)
    })
    .expect("post-clear get must finish");
    assert_eq!(v2.expect("post-clear get must succeed"), 30);
    assert_eq!(
        R_RUNS.load(Ordering::SeqCst),
        2,
        "post-clear get must recompute"
    );
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 2);
}

/// d16: a FIRST-time reader's
/// reverse edge does not exist until its finish flushes it, so a
/// recompute+republish of its read node inside that window fans out to
/// everyone EXCEPT the reader, and the reader's finish then publishes
/// green with a value built on the PRE-invalidation snapshot — served
/// stale forever (persistent stale).
///
/// Why the DepGraph layer: the QuerySystem-level `invalidate` runs
/// inside quiesced `mutate_world`, which by contract waits for every
/// in-flight compute (including a parked reader) before mutating —
/// the fan-out-miss window is unreachable there by construction.  The
/// window lives one layer down, where `read` / `publish` / `mark`
/// interleave freely; the fix (per-node dirty epoch captured at
/// `read()`, re-checked after the rev-edge flush) lives there too.
///
/// Deterministic single-thread repro: R starts a task and reads green
/// E; E is then invalidated and republished with a CHANGED fingerprint
/// (fan-out misses R — no rev edge yet); R finishes.  Pre-fix R is
/// served green with the stale value; post-fix the epoch re-check
/// restores R red.
#[test]
fn d16_late_fan_out_missed_reader_restored_red() {
    let dg = dep_graph::DepGraph::new();
    let e = dg.allocate_node_index(dep_graph::ContentClass::SEMANTIC);
    let r = dg.allocate_node_index(dep_graph::ContentClass::SEMANTIC);

    // E's initial computation publishes fp 1 and goes green.
    assert!(!dg.publish(e, 1), "first publish of E: nothing changed");
    assert!(dg.is_green(e));

    // R's compute reads green E...
    dg.start_task(r);
    dg.read(e);
    // ...and while it is still in flight, E is invalidated (world
    // changed) and republished with a changed fingerprint.  The
    // publish-time fan-out cannot reach R: its rev edge does not exist
    // yet.
    dg.invalidate(e);
    assert!(!dg.is_green(e), "E invalidated red");
    let changed = dg.publish(e, 7);
    assert!(
        changed,
        "E's fingerprint changed on the invalidating publish"
    );
    assert!(dg.is_green(e), "E republished green");

    // R finishes through the REAL query-system finalize path (which
    // publishes, clears and runs the epoch re-check after the rev-edge
    // flush): its value encoded E's PRE-invalidation snapshot.
    assert!(!dg.finish_task_and_publish(r, Some(1)), "R's fp unchanged");

    // The decisive assertion: R must NOT be green after the fan-out
    // miss.  Pre-fix: green → stale value served forever.  Post-fix:
    // the finish-side epoch re-check restored R red, so the next
    // request recomputes it.
    assert!(
        !dg.is_green(r),
        "late-fan-out missed reader must be restored red (d16: \
         first-time reader's reverse edge not yet flushed)"
    );

    // No false positives: a reader whose read stayed green throughout
    // its compute must remain green.
    let r2 = dg.allocate_node_index(dep_graph::ContentClass::SEMANTIC);
    dg.start_task(r2);
    dg.read(e);
    dg.finish_task();
    assert!(dg.is_green(r2), "clean reader must stay green");
}

/// The cross-thread waiter path serves the cache value it finds after
/// the latch wake with NO green verification at all.  The owner's
/// finalize order makes this a stale-serve hole: `finish_task_and_publish`
/// may restore the node RED (the d16 epoch re-check) while
/// `cache.stage` has already staged the suspect value, and
/// `signal_complete` then hands it to the waiter.  A non-quiesced
/// changed-fingerprint fan-out landing between the owner's publish and
/// the waiter's serve has the same effect.
///
/// Deterministic repro (owner-side half): A computes K, reads
/// eval-always E#1 (epoch 0), parks mid-compute.  B parks on A's job
/// latch (wait_graph observable).  Main recomputes E with a CHANGED
/// value — its publish's fan-out misses K (rev edge not yet flushed),
/// but the epoch-increment from the recomputation is visible.  A finishes: the epoch
/// re-check restores K red, caches the suspect value (semantic 10) and
/// signals.  Pre-fix B serves 10; post-fix B re-verifies, finds K red,
/// recomputes and returns 20.
#[test]
fn waiter_must_not_serve_a_value_the_owner_restored_red() {
    static E_CONTENT: AtomicU64 = AtomicU64::new(1);
    static E_RUNS: AtomicU64 = AtomicU64::new(0);
    static K_RUNS: AtomicU64 = AtomicU64::new(0);
    static K_READ_DONE: AtomicBool = AtomicBool::new(false);
    static GO: AtomicBool = AtomicBool::new(false);
    E_CONTENT.store(1, Ordering::SeqCst);
    E_RUNS.store(0, Ordering::SeqCst);
    K_RUNS.store(0, Ordering::SeqCst);
    K_READ_DONE.store(false, Ordering::SeqCst);
    GO.store(false, Ordering::SeqCst);

    struct EQ;
    impl Query for EQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("wq_e", "eval-always dep").with_eval_always(true)
        }
        fn compute(_key: &DKey, _: &dyn QueryProvider) -> FpValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: String::new(),
                semantic: E_CONTENT.load(Ordering::SeqCst),
            }
        }
    }
    struct KQ;
    impl Query for KQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("wq_k", "reader over eval-always dep")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            K_RUNS.fetch_add(1, Ordering::SeqCst);
            let e = query_get::<EQ>(provider, key.clone()).unwrap();
            // Park AFTER E's read is recorded on K's task (epoch
            // captured) but BEFORE K's finish flushes its rev edge.
            K_READ_DONE.store(true, Ordering::SeqCst);
            while !GO.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            FpValue {
                raw: String::new(),
                semantic: e.semantic * 10,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // A: first compute of K, parks right after reading E#1.
    let a = {
        let (sys, prov) = (system, provider);
        thread::spawn(move || sys.get::<KQ>(DKey(0), prov).map(|v| v.semantic))
    };
    let read_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !K_READ_DONE.load(Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < read_deadline,
            "A never reached its post-read park"
        );
        std::thread::yield_now();
    }

    // B: parks on A's job latch (registered wait edge = observable).
    let b = {
        let (sys, prov) = (system, provider);
        thread::spawn(move || sys.get::<KQ>(DKey(0), prov).map(|v| v.semantic))
    };
    let park_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while system.wait_graph.read().is_empty() {
        assert!(
            std::time::Instant::now() < park_deadline,
            "B never parked on A's job latch"
        );
        std::thread::yield_now();
    }

    // Mid-flight: E recomputes with a CHANGED value.  The publish's
    // fan-out misses K (rev edge not flushed), but the changed publish
    // bumps E's own epoch (the fix for unmarked-value-change), so A's finish re-check restores K
    // red BEFORE staging the suspect value and signalling.
    E_CONTENT.store(2, Ordering::SeqCst);
    let e2 = system.get::<EQ>(DKey(0), provider).unwrap();
    assert_eq!(e2.semantic, 2, "E must have recomputed fresh (eval_always)");
    GO.store(true, Ordering::SeqCst);

    let ra = a.join().unwrap().expect("A must succeed");
    assert_eq!(ra, 10, "A returns its own (pre-change) computed value");
    let rb = b.join().unwrap().expect("B must succeed");

    // Post-fix: B re-verifies, finds K red, recomputes → 20.
    // Pre-fix: B serves the suspect 10 with no verification at all.
    assert_eq!(
        rb, 20,
        "waiter must not serve a value the owner flagged red \
         (cross-thread stale-serve hole)"
    );
    assert_eq!(
        K_RUNS.load(Ordering::SeqCst),
        2,
        "the waiter must recompute K, not serve the stale cache"
    );
    assert_eq!(
        E_RUNS.load(Ordering::SeqCst),
        3,
        "B's recompute must read a fresh E (eval_always)"
    );
}

/// Latch-waiter wake-order invariant (defense test I-1): when a latch
/// waiter wakes AFTER a concurrent `clear_all` fully completed, the
/// waiter's `park_wait` voluntarily released the quiesce read lock, and
/// parking_lot is write-preferring — clear_all's queued write is granted
/// the moment the owner drops its read guard, BEFORE the waiter's
/// `unpark` re-takes the read lock.  Pre-fix the waiter then served
/// through its PRE-wait snapshot: `vtable` is an ORPHANED Arc whose cache
/// still holds the OWNER's just-inserted value — visible only through
/// that orphaned Arc, invisible to the fresh (emptied) registry — and
/// `node_index` is a PRE-RESET index.  The post-reset graph looks
/// trivially green, so `try_serve_verified` serves the stale value AND
/// stamps `verified_gen` for a slot a freshly allocated node reuses (the
/// new node's later hits would then skip closure verification entirely).
///
/// Deterministic harness: the MAIN thread calls `clear_all` itself,
/// blocking on the quiesce write lock while the owner still holds its
/// read guard — the write is therefore QUEUED before the owner is
/// released.  A releaser thread probes `quiesce.try_read()`
/// (parking_lot writer preference makes it fail exactly while a writer
/// is queued) and only then sets RELEASE, forcing the ordering
/// owner-finalize → guard-drop → write-grant → clear_all-completes →
/// waiter-unpark instead of racing spawn-vs-finalize.  (The original
/// spawned-clearer form was racy: if the clearer thread had not reached
/// `quiesce.write()` before the owner dropped its guard, the waiter's
/// unpark read was granted first and serving the pre-clear value was
/// CORRECT for that interleaving — K_RUNS==1 without any bug.)
#[test]
fn d18_waiter_re_resolves_after_clear_all() {
    static K_RUNS: AtomicU64 = AtomicU64::new(0);
    static ENTERED: AtomicBool = AtomicBool::new(false);
    static RELEASE: AtomicBool = AtomicBool::new(false);
    K_RUNS.store(0, Ordering::SeqCst);
    ENTERED.store(false, Ordering::SeqCst);
    RELEASE.store(false, Ordering::SeqCst);

    struct WQ;
    impl Query for WQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d18_w", "clear_all waiter victim")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            K_RUNS.fetch_add(1, Ordering::SeqCst);
            ENTERED.store(true, Ordering::SeqCst);
            // 2s self-timeout so a quiesced clear_all cannot three-way
            // deadlock the test (d9 discipline).
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !RELEASE.load(Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    break;
                }
                std::hint::spin_loop();
            }
            FpValue {
                raw: format!("d18_{}", key.0),
                semantic: 7,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Owner: computes WQ(1) and spins (holding its quiesce read lock).
    let owner = {
        let (sys, prov) = (system, provider);
        thread::spawn(move || {
            sys.get::<WQ>(DKey(1), prov)
                .map(|v| v.semantic)
                .map_err(|e| e.message)
        })
    };
    let entered_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !ENTERED.load(Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < entered_deadline,
            "owner never entered compute"
        );
        std::thread::yield_now();
    }

    // Waiter: parks on the owner's job latch (park_wait released its
    // quiesce read lock).
    let waiter = {
        let (sys, prov) = (system, provider);
        thread::spawn(move || {
            sys.get::<WQ>(DKey(1), prov)
                .map(|v| v.semantic)
                .map_err(|e| e.message)
        })
    };
    let mut parked = false;
    for _ in 0..200 {
        if !system.wait_graph.read().is_empty() {
            parked = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(parked, "waiter must be parked on the owner's latch");

    // Releaser: only release the owner once a WRITER is actually queued —
    // parking_lot writer preference makes `try_read` fail exactly while
    // clear_all's write waits behind the owner's read guard.
    let releaser = {
        let sys = system;
        thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            loop {
                // parking_lot's try_read returns Option<ReadGuard>;
                // writer preference makes it None exactly while a writer
                // (main's clear_all) is queued.  Bind to a let so the
                // guard (if taken) drops immediately.
                let writer_queued = sys.quiesce.try_read().is_none();
                if writer_queued {
                    break; // writer queued (main's clear_all is blocked)
                }
                if std::time::Instant::now() > deadline {
                    break;
                }
                std::thread::yield_now();
            }
            RELEASE.store(true, Ordering::SeqCst);
        })
    };

    // Queue the write from THIS thread: it blocks (the owner still holds
    // its read guard) and is granted the instant the guard drops —
    // strictly BEFORE the waiter's unpark can re-take the read lock.
    system.clear_all();

    releaser.join().expect("releaser panicked");
    let r_owner = owner.join().expect("owner panicked");
    let r_waiter = waiter.join().expect("waiter panicked");
    assert!(
        r_owner.is_ok() && r_waiter.is_ok(),
        "{r_owner:?} {r_waiter:?}"
    );
    assert_eq!(r_owner.unwrap(), 7);
    assert_eq!(r_waiter.unwrap(), 7);
    // Pre-fix: the waiter served the owner's entry from the orphaned
    // Arc'd cache after "verifying" it against the POST-reset graph with
    // a PRE-reset node index — K_RUNS stays 1.
    assert_eq!(
        K_RUNS.load(Ordering::SeqCst),
        2,
        "waiter must recompute after clear_all, not serve the pre-clear \
         orphaned cache entry"
    );
}

/// Cache-mutation-without-quiesce invariant (defense test I-2):
/// `clear::<Q>` mutated the served world (the cache) without holding
/// the quiesce write lock — the module rule says ALL mutation of the
/// red-green world must flow through `mutate_world`, and
/// invalidate/invalidate_node/clear_all comply while clear did not.  The window: thread A computes K holding the quiesce
/// read lock; `clear` runs and returns while A is still in flight; A then
/// `cache.stage`s its (pre-clear) value AFTER clear returned — while its
/// dep node is GREEN (clear sets no red bit).  The next `get` takes the
/// probe fast path (green ⇒ cache hit ⇒ serve) and returns the
/// resurrected pre-clear value WITHOUT recomputing, breaking clear's
/// "next get recomputes" contract.
///
/// Deterministic: clear is spawned BEFORE A is released, so pre-fix it
/// completes while A is parked mid-compute; A's later insert resurrects
/// the entry.  Post-fix clear blocks on the quiesce write lock until A's
/// compute (and its insert) finished, then clears — the next get
/// recomputes.
#[test]
fn d19_clear_must_quiesce_in_flight_publish() {
    static K_RUNS: AtomicU64 = AtomicU64::new(0);
    static ENTERED: AtomicBool = AtomicBool::new(false);
    static RELEASE: AtomicBool = AtomicBool::new(false);
    K_RUNS.store(0, Ordering::SeqCst);
    ENTERED.store(false, Ordering::SeqCst);
    RELEASE.store(false, Ordering::SeqCst);

    struct KQ;
    impl Query for KQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d19_k", "clear resurrection victim")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            K_RUNS.fetch_add(1, Ordering::SeqCst);
            ENTERED.store(true, Ordering::SeqCst);
            // 2s self-timeout so a quiesced clear cannot three-way
            // deadlock the test (d9 discipline).
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !RELEASE.load(Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    break;
                }
                std::hint::spin_loop();
            }
            FpValue {
                raw: format!("d19_{}", key.0),
                semantic: 7,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // A: computes KQ(0), parks mid-compute (holding its quiesce read
    // lock — its insert has NOT happened yet).
    let a = {
        let (sys, prov) = (system, provider);
        thread::spawn(move || {
            sys.get::<KQ>(DKey(0), prov)
                .map(|v| v.semantic)
                .map_err(|e| e.message)
        })
    };
    let entered_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !ENTERED.load(Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < entered_deadline,
            "A never entered compute"
        );
        std::thread::yield_now();
    }

    // C: clear::<KQ> while A is IN FLIGHT.  Spawned before RELEASE so the
    // mutation is issued against the in-flight world.
    let clearer = {
        let sys = system;
        thread::spawn(move || sys.clear::<KQ>())
    };
    RELEASE.store(true, Ordering::SeqCst);

    let ra = a.join().expect("A panicked");
    assert_eq!(ra.unwrap(), 7, "A returns its own computed value");
    clearer.join().expect("clear_all panicked");

    // The decisive probe: the next get must RECOMPUTE, not serve a
    // resurrected pre-clear entry.  Pre-fix A's post-clear
    // `cache.stage` resurrected the value over a still-green node and
    // this get served it (K_RUNS stays 1).
    let r = system
        .get::<KQ>(DKey(0), provider)
        .map(|v| v.semantic)
        .map_err(|e| e.message);
    assert!(r.is_ok(), "post-clear get must succeed: {r:?}");
    assert_eq!(r.unwrap(), 7);
    assert_eq!(
        K_RUNS.load(Ordering::SeqCst),
        2,
        "the next get after clear must recompute — clear must quiesce \
         in-flight publishes"
    );
}

/// Cross-session re-request invariant (defense test I-3):
/// `try_serve_verified` walks `self.dep_graph` / `self.node_to_key`, but
/// the re-request body ran `provider.query_system().get_shared::<Q>` —
/// nothing bound the provider to the system being verified.  With a
/// provider bound to a DIFFERENT system, the dirty dependency was
/// recomputed (and its cache/poison side effects landed) in a foreign
/// session while `self`'s node stayed red, forcing the caller to fall
/// back to its own recompute — the try_mark_green serve never happened.
///
/// Single-threaded, deterministic: K is only ever requested on A; E's
/// execution system is decided entirely by where the re-request lands.
/// The recompute yields the SAME fingerprint (content constant), so
/// post-fix the re-request turns E_A green, the re-verify passes, and
/// the cached K is served WITHOUT recompute (K_RUNS == 1).  Pre-fix the
/// re-request ran on B, E_A stayed red, and K recomputed (K_RUNS == 2).
#[test]
fn d20_rerequest_lands_on_verifying_system() {
    static K_RUNS: AtomicU64 = AtomicU64::new(0);
    static E_RUNS: AtomicU64 = AtomicU64::new(0);
    K_RUNS.store(0, Ordering::SeqCst);
    E_RUNS.store(0, Ordering::SeqCst);

    struct EQ;
    impl Query for EQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d20_e", "rerequest landing probe")
        }
        fn compute(_key: &DKey, _: &dyn QueryProvider) -> FpValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: String::new(),
                semantic: 7,
            }
        }
    }
    struct KQ;
    impl Query for KQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d20_k", "reader whose dep is re-requested")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            K_RUNS.fetch_add(1, Ordering::SeqCst);
            let e = query_get::<EQ>(provider, key.clone()).unwrap();
            FpValue {
                raw: String::new(),
                semantic: e.semantic * 10,
            }
        }
    }

    // Two INDEPENDENT systems; prov_b is bound to B.
    let sys_a = leak_system();
    let prov_a = leak_provider(sys_a);
    let sys_b = leak_system();
    let prov_b = leak_provider(sys_b);

    // Step 1: K on A with A's provider — K_A reads E_A, both on A.
    let v1 = sys_a
        .get::<KQ>(DKey(0), prov_a)
        .map(|v| v.semantic)
        .map_err(|e| e.message);
    assert_eq!(v1.unwrap(), 70);
    assert_eq!(K_RUNS.load(Ordering::SeqCst), 1);
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 1);

    // Step 2: mark E_A red on A (quiesced mutate_world).
    sys_a.invalidate::<EQ>(&DKey(0));

    // Step 3: re-verify K_A on A — but hand it a provider bound to B.
    // The dirty dependency found by A's verifier must be re-requested ON
    // A; pre-fix it was redirected to B.
    let v3 = sys_a
        .get::<KQ>(DKey(0), prov_b)
        .map(|v| v.semantic)
        .map_err(|e| e.message);
    assert_eq!(v3.unwrap(), 70);

    // Exactly one extra E recompute happened (the re-request) — same
    // count pre- and post-fix, so this is a sanity bound, not the
    // verdict.
    assert_eq!(
        E_RUNS.load(Ordering::SeqCst),
        2,
        "the re-request must recompute E exactly once"
    );
    // THE VERDICT: post-fix the re-request ran on A, E_A turned green,
    // the re-verify passed and the cached K was served without
    // recomputing.  Pre-fix it ran on B, E_A stayed red, and A fell back
    // to recomputing K.
    assert_eq!(
        K_RUNS.load(Ordering::SeqCst),
        1,
        "the re-request must land on the VERIFYING system: only a green \
         E_A lets the re-verify pass so the cached K serves without \
         recompute"
    );
}

/// d21: the green bit must never be published before the new cache entry
/// is staged — doing so lets a getter see a GREEN node whose value is
/// still the pre-recompute stale Arc.
///
/// Pre-fix order: `finish_task_and_publish` (clears the dirty bit) →
/// `cache.stage`.  A getter probing the window saw a GREEN node whose
/// cache entry was the PRE-recompute value (fan-out dirtying never
/// removes entries) and `try_serve_verified` served the stale Arc.
/// Post-fix the value is staged BEFORE the green publish, so the same
/// probe returns the fresh value without recomputing.
#[test]
fn d21_green_bit_never_outruns_cache_entry() {
    static E_CONTENT: AtomicU64 = AtomicU64::new(1);
    static E_RUNS: AtomicU64 = AtomicU64::new(0);
    static K_RUNS: AtomicU64 = AtomicU64::new(0);
    E_CONTENT.store(1, Ordering::SeqCst);
    E_RUNS.store(0, Ordering::SeqCst);
    K_RUNS.store(0, Ordering::SeqCst);
    D21_PARK.store(false, Ordering::SeqCst);
    D21_TARGET.store(0, Ordering::SeqCst);
    D21_ARRIVED.store(false, Ordering::SeqCst);

    struct D21E;
    impl Query for D21E {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d21_inner", "d21 leaf input")
        }
        fn compute(_key: &DKey, _: &dyn QueryProvider) -> FpValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: String::new(),
                semantic: E_CONTENT.load(Ordering::SeqCst),
            }
        }
    }
    struct D21K;
    impl Query for D21K {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d21_reader", "d21 reader over d21_inner")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            K_RUNS.fetch_add(1, Ordering::SeqCst);
            let e = query_get::<D21E>(provider, key.clone()).unwrap();
            FpValue {
                raw: String::new(),
                semantic: e.semantic * 10,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Warm: K=10 cached and green.
    assert_eq!(
        system.get::<D21K>(DKey(0), provider).unwrap().semantic,
        10
    );

    // Change E and invalidate E only.  Re-requesting E republishes it
    // with a CHANGED fingerprint (1→2); the publish-time fan-out marks
    // K red WITHOUT removing K's cache entry — the stale precondition.
    E_CONTENT.store(2, Ordering::SeqCst);
    system.invalidate::<D21E>(&DKey(0));
    assert_eq!(
        system.get::<D21E>(DKey(0), provider).unwrap().semantic,
        2
    );

    // Owner recomputes K; parks after its green publish (d21 hook).
    D21_PARK.store(true, Ordering::SeqCst);
    D21_TARGET.store(
        D21K::descriptor().name.as_ptr() as usize,
        Ordering::SeqCst,
    );
    let (otx, orx) = mpsc::channel();
    {
        let (sys, prov) = (system, provider);
        thread::spawn(move || {
            let _ = otx.send(sys.get::<D21K>(DKey(0), prov).map(|v| v.semantic));
        });
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !D21_ARRIVED.load(Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < deadline,
            "owner never reached publish"
        );
        std::thread::yield_now();
    }

    // Probe the green serve path from THIS thread while the owner is
    // parked.  Must return the STAGED value (20) without recomputing.
    // Pre-fix this served the stale pre-recompute Arc (10).  (This get
    // takes only the cached branch, so it never enters the d21 hook.)
    let got = system.get::<D21K>(DKey(0), provider).unwrap();
    assert_eq!(
        got.semantic, 20,
        "green serve returned the pre-recompute entry"
    );

    D21_PARK.store(false, Ordering::SeqCst);
    D21_TARGET.store(0, Ordering::SeqCst);
    assert_eq!(
        guard_recv_dur(orx, GUARD).expect("owner must finish").unwrap(),
        20
    );
    assert_eq!(
        K_RUNS.load(Ordering::SeqCst),
        2,
        "one warm + one recompute; the probe must be a serve, not a recompute"
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
    OP_COUNTER.store(0, Ordering::Relaxed);
    PANIC_FLAG.store(false, Ordering::SeqCst);
    for s in &SRC {
        s.store(0, Ordering::SeqCst);
    }

    let system = leak_system();
    let provider = leak_provider(system);

    fn worker(system: &'static QuerySystem, provider: &'static dyn QueryProvider) {
        let mut local = 0usize;
        loop {
            let op = OP_COUNTER.fetch_add(1, Ordering::Relaxed);
            if op >= 3000 {
                break;
            }
            let slot = ((op as usize).wrapping_mul(2654435761) + local) % STRESS_SLOTS;
            local = local.wrapping_add(1);
            match op % 10 {
                0..=3 => {
                    // get SumQ(slot): v ≤ SRC[slot] + SRC[slot+1] (monotonic).
                    // Sample hi AFTER the get: with monotonic sources, a value
                    // can never be newer than a snapshot taken after it.
                    let v = match std::panic::catch_unwind(AssertUnwindSafe(|| {
                        system.get::<SumQ>(DKey(slot), provider)
                    })) {
                        Ok(Ok(v)) => v,
                        Ok(Err(e)) => panic!("value phase must not error: {e:?}"),
                        Err(_) => panic!("value phase must not panic"),
                    };
                    let hi = SRC[slot].load(Ordering::SeqCst)
                        + SRC[(slot + 1) % STRESS_SLOTS].load(Ordering::SeqCst);
                    assert!(
                        v.semantic <= hi,
                        "sum value must not exceed current sources: {v:?} vs {hi}"
                    );
                }
                4..=6 => {
                    // get SrcQ(slot): v ≤ SRC[slot].
                    let v = match std::panic::catch_unwind(AssertUnwindSafe(|| {
                        system.get::<SrcQ>(DKey(slot), provider)
                    })) {
                        Ok(Ok(v)) => v,
                        Ok(Err(e)) => panic!("value phase must not error: {e:?}"),
                        Err(_) => panic!("value phase must not panic"),
                    };
                    let hi = SRC[slot].load(Ordering::SeqCst);
                    assert!(
                        v.semantic <= hi,
                        "src value must not exceed current source: {v:?} vs {hi}"
                    );
                }
                7..=8 => {
                    // get ChainQ(slot, random depth 0..=3): v ≤ SRC[slot] + d.
                    let d = (op as usize % 4) as u64;
                    let v = match std::panic::catch_unwind(AssertUnwindSafe(|| {
                        system.get::<ChainQ>(DKey(slot * 100 + d as usize), provider)
                    })) {
                        Ok(Ok(v)) => v,
                        Ok(Err(e)) => panic!("value phase must not error: {e:?}"),
                        Err(_) => panic!("value phase must not panic"),
                    };
                    let hi = SRC[slot].load(Ordering::SeqCst) + d;
                    assert!(
                        v.semantic <= hi,
                        "chain value must not exceed current source + depth: {v:?} vs {hi}"
                    );
                }
                _ => {
                    // Invalidate; mutate the source with probability ~1/2.
                    // `op % 10 == 9` means op is always odd here, so
                    // `op % 4 == 0` was unreachable.  Use
                    // `op % 20 >= 10` instead — true for exactly half
                    // of all integers, reachable from any residue.
                    if op % 20 >= 10 {
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
    // Invalidate only SrcQ — transitive green verification ensures
    // downstream readers (SumQ/ChainQ) re-verify their dependencies
    // and recompute if needed, regardless of request order.
    for slot in 0..STRESS_SLOTS {
        system.invalidate::<SrcQ>(&DKey(slot));
    }
    for slot in 0..STRESS_SLOTS {
        let expect = SRC[slot].load(Ordering::SeqCst);
        let v = system.get::<SrcQ>(DKey(slot), provider).expect("src get");
        assert_eq!(v.semantic, expect, "quiesced src mismatch at slot {slot}");
        // Request chain in arbitrary order — transitive verification
        // ensures correct re-evaluation without bottom-up pumping.
        for d in 0..=3u64 {
            let expect_d = expect + d;
            let v = system
                .get::<ChainQ>(DKey(slot * 100 + d as usize), provider)
                .expect("chain get");
            assert_eq!(
                v.semantic, expect_d,
                "quiesced chain mismatch at slot {slot} depth {d}"
            );
        }
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

/// d22: a world-mutation parked on a system that never gets another
/// outermost frame — because the parking thread exits first, or because
/// the system itself is dropped first — must still run, exactly once,
/// against the system it was parked on.
///
/// The queue used to live in the parking thread's own thread-local.  The
/// spawned thread below exits as soon as it returns from `get`, taking
/// the closure with it; no later activity on any system could
/// resurrect it, so the invalidation was silently gone — not
/// misdirected, just lost.  The queue now lives on the TARGET system,
/// partitioned by thread id, and `Drop for QuerySystem` sweeps every
/// partition.
///
/// The first assertion is the other half of the same contract: A's
/// outermost-frame drain must NOT run a mutation parked on B.  Running
/// it there is the original cross-system defect (executed against A's
/// `&QuerySystem` while it belonged to B).  Together the two assertions
/// pin both directions — never lost, and never with the wrong system.
#[test]
fn d22_parked_mutation_survives_target_system_drop() {
    static PARKED: AtomicBool = AtomicBool::new(false);
    static DRIVER_RUNS: AtomicU64 = AtomicU64::new(0);
    PARKED.store(false, Ordering::SeqCst);
    DRIVER_RUNS.store(0, Ordering::SeqCst);

    // B is owned rather than `Box::leak`ed so the test can drop it; it
    // sits in a static so the driver's compute can borrow it while
    // running on another thread.
    static TARGET: std::sync::Mutex<Option<Box<QuerySystem>>> =
        std::sync::Mutex::new(None);

    struct DriverQ;
    impl Query for DriverQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d22_driver", "parks a mutation on another system")
        }
        fn compute(_: &DKey, _: &dyn QueryProvider) -> FpValue {
            DRIVER_RUNS.fetch_add(1, Ordering::SeqCst);
            // This thread's outermost frame belongs to system A, so the
            // mutation must be PARKED: running it now would take B's
            // write lock alongside A's held read lock, and could be
            // erased by a publish still to come (d10/d11).
            let guard = TARGET.lock().unwrap();
            let target = guard.as_ref().expect("target must be alive");
            target.mutate_world(|_| PARKED.store(true, Ordering::SeqCst));
            FpValue {
                raw: "d22".into(),
                semantic: 1,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);
    *TARGET.lock().unwrap() = Some(Box::new(QuerySystem::new()));

    let semantic = run_with_guard(GUARD, move || {
        system
            .get::<DriverQ>(DKey(0), provider)
            .map(|v| v.semantic)
    })
    .expect("driver get must not hang")
    .expect("driver get must succeed");
    assert_eq!(semantic, 1);
    assert_eq!(DRIVER_RUNS.load(Ordering::SeqCst), 1);

    // A's outermost guard drained A's own queue and found nothing of
    // B's.  Executing it here would be the cross-system misdirection.
    assert!(
        !PARKED.load(Ordering::SeqCst),
        "a mutation parked on system B must not run when system A's \
         outermost frame drains"
    );

    // Drop B from a thread that is NOT the one that parked the closure
    // (and, by now, not one that is even alive).
    let target = TARGET.lock().unwrap().take();
    drop(target);
    assert!(
        PARKED.load(Ordering::SeqCst),
        "a parked mutation must run when its target system is dropped, \
         even after the thread that parked it has exited"
    );
}

// ── Per-thread stat shards: merge-on-read exactness ──────────────

/// d23: thread-local stat slots must merge into EXACT totals.
///
/// N threads each bump M hits on instance A and M misses on instance B
/// (the two instances share the TLS slot map but keep separate
/// registries).  `join` provides the happens-before edge, so `stats()` —
/// though it loads each slot `Relaxed` — must observe every bump: a lost
/// update (slot aliasing across instances, a race in first-touch
/// registration) or cross-instance contamination (an address reused as
/// an arena id — the `NEXT_SYSTEM_ID` failure class) would surface as a
/// wrong sum.  Each worker is a FRESH thread, so every slot here was
/// created by the cold path and every bump after the first is the hot
/// path — both pipelines are covered, and the exited workers' slots
/// must still be summed (data lives in the registry, only the handle is
/// thread-local).
#[test]
fn d23_per_thread_stat_merge_exact() {
    const THREADS: usize = 8;
    const M: usize = 500;

    let a: &'static DefaultCache<DKey, DValue> = Box::leak(Box::new(DefaultCache::new()));
    let b: &'static DefaultCache<DKey, DValue> = Box::leak(Box::new(DefaultCache::new()));

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let a = a;
            let b = b;
            thread::spawn(move || {
                for _ in 0..M {
                    a.record_hit();
                    b.record_miss();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("stat worker must not panic");
    }

    let sa = a.stats();
    let sb = b.stats();
    assert_eq!(
        sa.hits,
        (THREADS * M) as u64,
        "instance A hits must merge exactly — no lost slot updates"
    );
    assert_eq!(
        sa.misses,
        0,
        "instance A must not inherit instance B's misses"
    );
    assert_eq!(
        sb.misses,
        (THREADS * M) as u64,
        "instance B misses must merge exactly — no lost slot updates"
    );
    assert_eq!(
        sb.hits,
        0,
        "instance B must not inherit instance A's hits"
    );
    assert_eq!(sa.evictions, 0, "nothing was staged or evicted here");
    assert_eq!(sa.entries, 0, "no staging happened in this test");
}

// ── Shard-distribution audit (degenerate-Hash tripwire) ──────────

/// `DefaultCache::shard_skew_flagged` must stay silent on a real key
/// distribution and fire on a constant-`Hash` key type — the audit exists
/// to catch UNEXPECTED degenerate `Hash` impls on production keys, whose
/// symptom (one hot shard: single-lock contention + a local eviction
/// storm) is otherwise hard to attribute back to `Hash`.  Also pins the
/// global-capacity borrow: a hot shard keeps all 500 values because the
/// GLOBAL budget (1024) was never reached, even though its per-shard
/// share (64) was blown long ago.
///
/// Gated on `debug_assertions` because the audit surface itself
/// (`shard_distribution` / `shard_skew_flagged`) is debug-only — in
/// `cargo test --release` builds the methods do not exist, so the test
/// must be compiled out with them.
#[cfg(debug_assertions)]
#[test]
fn d25_shard_distribution_flags_degenerate_hash() {
    // (a) Uniform keys: FxHash spreads DKey(i) — must NOT flag.
    let uniform = DefaultCache::<DKey, DValue>::with_capacity(1024);
    for i in 0..500u64 {
        uniform.insert(DKey(i as usize), Arc::new(DValue::new(i)));
    }
    let d = uniform.shard_distribution();
    assert_eq!(d.total, 500, "no eviction: global cap 1024 not reached");
    assert!(
        !uniform.shard_skew_flagged(),
        "uniform keys must not trip the audit: {d:?}"
    );

    // (b) Constant-Hash keys: everything lands in ONE shard — must flag,
    // and global borrowing must keep every value alive.
    let collide = DefaultCache::<CollideKey, DValue>::with_capacity(1024);
    for i in 0..500u64 {
        collide.insert(CollideKey(i as usize), Arc::new(DValue::new(i)));
    }
    let d = collide.shard_distribution();
    assert_eq!(
        d.total, 500,
        "global borrow: the hot shard keeps all values — its per-shard \
         share (64) is a floor, not a ceiling, until the global cap bites"
    );
    assert_eq!(d.max, 500, "all keys belong to one shard");
    assert_eq!(d.min, 0, "every other shard is empty");
    assert!(
        collide.shard_skew_flagged(),
        "constant-hash key type must trip the audit: {d:?}"
    );

    // (c) The same audit over `active_keys` shard selection
    // (`query_key_hash` bucketing): a zero-entropy key collapses every
    // lookup into a single bucket.
    struct D25Q;
    impl Query for D25Q {
        type Key = CollideKey;
        type Value = DValue;
        type Cache = DefaultCache<CollideKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d25", "shard-distribution audit query")
        }
        fn compute(_: &CollideKey, _: &dyn QueryProvider) -> DValue {
            unreachable!("audit-only query: never computed")
        }
    }
    let mut buckets = vec![0usize; ACTIVE_SHARDS];
    for i in 0..500usize {
        let h = query_key_hash::<D25Q>(&CollideKey(i));
        buckets[h as usize & (ACTIVE_SHARDS - 1)] += 1;
    }
    let total: usize = buckets.iter().sum();
    let max = *buckets.iter().max().expect("non-empty bucket set");
    let min = *buckets.iter().min().expect("non-empty bucket set");
    assert_eq!(max, 500, "all keys share one active_keys bucket");
    assert!(
        distribution_flagged(total, max, min),
        "query_key_hash collapse must be flagged: {buckets:?}"
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
    // This works regardless of request order thanks to transitive green
    // verification — no more "input-first" masking.
    let _ = system.get::<GDown>(DKey(1), provider).unwrap();
    assert_eq!(
        D_COUNT.load(Ordering::SeqCst),
        1,
        "downstream recompute delta must be 0 when the input fingerprint is unchanged"
    );
}

/// Regression test for the stale-reader bug: request the downstream
/// query FIRST after invalidating the input (before the input is
/// re-requested).  Before the fix, this would serve a stale cached
/// value because D's green hit only checked D's own dirty bit, not
/// I's.  After the fix, transitive verification detects I is dirty,
/// re-requests I (which republishes with unchanged fingerprint),
/// verifies I is now green, and serves D's cached value — all
/// without D recomputing.
#[test]
fn g1c_stale_reader_requested_before_input() {
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

    // "Whitespace-only" edit: invalidate the input; fingerprint stays 42.
    system.invalidate::<GIn>(&DKey(1));

    // Request the DOWNSTREAM query first — a perfectly legal client
    // sequence that previously returned a stale value.
    let v = system.get::<GDown>(DKey(1), provider).unwrap();
    assert_eq!(
        v.semantic, 142,
        "downstream must see the input's current fingerprint (42 -> 142)"
    );
    // I recomputed once (I_COUNT=2), D did NOT recompute (D_COUNT=1).
    assert_eq!(I_COUNT.load(Ordering::SeqCst), 2);
    assert_eq!(D_COUNT.load(Ordering::SeqCst), 1);
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
    // Works regardless of request order thanks to transitive green
    // verification.
    let _ = system.get::<GDown>(DKey(1), provider).unwrap();
    assert_eq!(
        D_COUNT.load(Ordering::SeqCst),
        2,
        "changed fingerprint must propagate to downstream readers"
    );
}

/// Reproduction for the verified_gen memoization soundness hole: red
/// bits created WITHOUT bumping dirty_generation (publish-with-changed-
/// fingerprint after an eval_always recompute, and the dirty_readers
/// utility API) are invisible to the `vg >= current_gen` fast path — a
/// transitively-stamped cached reader keeps serving its stale value.
///
/// Chain: T reads R, R reads E; E is `eval_always` (recomputes on every
/// direct request, never invalidated).
#[test]
fn g2_eval_always_transitive_reader_not_served_stale() {
    static E_RUNS: AtomicU64 = AtomicU64::new(0);
    static R_RUNS: AtomicU64 = AtomicU64::new(0);
    static T_RUNS: AtomicU64 = AtomicU64::new(0);
    static E_CONTENT: AtomicU64 = AtomicU64::new(1);
    E_RUNS.store(0, Ordering::SeqCst);
    R_RUNS.store(0, Ordering::SeqCst);
    T_RUNS.store(0, Ordering::SeqCst);
    E_CONTENT.store(1, Ordering::SeqCst);

    struct EQ;
    impl Query for EQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("g2_e", "eval-always dep").with_eval_always(true)
        }
        fn compute(_: &DKey, _: &dyn QueryProvider) -> FpValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: String::new(),
                semantic: E_CONTENT.load(Ordering::SeqCst),
            }
        }
    }

    struct RQ;
    impl Query for RQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("g2_r", "middle reader")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            R_RUNS.fetch_add(1, Ordering::SeqCst);
            let e = query_get::<EQ>(provider, key.clone()).expect("r reads e");
            FpValue {
                raw: format!("r_{}", e.semantic),
                semantic: 100 + e.semantic,
            }
        }
    }

    struct TQ;
    impl Query for TQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("g2_t", "top reader")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            T_RUNS.fetch_add(1, Ordering::SeqCst);
            let r = query_get::<RQ>(provider, key.clone()).expect("t reads r");
            FpValue {
                raw: format!("t_{}", r.semantic),
                semantic: 1000 + r.semantic,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // t0: compute the chain — E#1, R#1, T#1.
    let v = system.get::<TQ>(DKey(1), provider).unwrap();
    assert_eq!(v.semantic, 1101);
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 1);

    // t1: cached hit that STAMPS the transitive closure at gen G.
    let v = system.get::<TQ>(DKey(1), provider).unwrap();
    assert_eq!(v.semantic, 1101);
    assert_eq!(T_RUNS.load(Ordering::SeqCst), 1);

    // t2: the world changes WITHOUT any invalidate; a direct request of
    // the eval_always dep re-runs it, its fingerprint changes 1→2, and
    // publish marks R dirty WITHOUT bumping dirty_generation.
    E_CONTENT.store(2, Ordering::SeqCst);
    let e = system.get::<EQ>(DKey(1), provider).unwrap();
    assert_eq!(e.semantic, 2);
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 2);

    // t3: T's verified_gen stamp (gen G) is still >= current_gen, so the
    // fast path serves the stale cached value without ever walking the
    // closure that would have found red R.
    let v = system.get::<TQ>(DKey(1), provider).unwrap();
    assert_eq!(
        v.semantic, 1102,
        "T must observe the eval_always change through R"
    );
    assert_eq!(R_RUNS.load(Ordering::SeqCst), 2, "R must recompute");
    assert_eq!(T_RUNS.load(Ordering::SeqCst), 2, "T must recompute");
}

/// g3: stale dependency edges cause over-invalidation.
///
/// `edge_extend`/`edge_insert` are append-only: a node's forward edges
/// are the UNION of its read sets across ALL recomputations, never the
/// read set of the CURRENT one.  When a query's read set changes between
/// recomputations (typeck reading different items), the stale reverse
/// edge keeps the old input pointing at this reader, so changing the
/// old input spuriously recomputes a reader that no longer depends on
/// it — defeating memoization.
///
/// Trace: Leaf(1) reads In0; then reads In1 (edges[Leaf] = [In0, In1],
/// In0's rev edge to Leaf retained).  In0 changes + is invalidated.
/// The transitive verifier walks the STALE In0 edge, re-requests In0,
/// its fingerprint changed (10 -> 99), publish marks Leaf dirty —
/// Leaf recomputes although its CURRENT read set is {In1}.
#[test]
fn g3_stale_edges_cause_over_invalidation() {
    static WHICH: AtomicU64 = AtomicU64::new(0);
    static IN0: AtomicU64 = AtomicU64::new(10);
    static IN1: AtomicU64 = AtomicU64::new(20);
    static LEAF_COUNT: AtomicU64 = AtomicU64::new(0);
    WHICH.store(0, Ordering::SeqCst);
    IN0.store(10, Ordering::SeqCst);
    IN1.store(20, Ordering::SeqCst);
    LEAF_COUNT.store(0, Ordering::SeqCst);

    struct InQ;
    impl Query for InQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("g3_in", "input")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            let v = if key.0 == 0 {
                IN0.load(Ordering::SeqCst)
            } else {
                IN1.load(Ordering::SeqCst)
            };
            FpValue {
                raw: format!("in{}", key.0),
                semantic: v,
            }
        }
    }

    struct LeafQ;
    impl Query for LeafQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("g3_leaf", "reader with shifting read set")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            LEAF_COUNT.fetch_add(1, Ordering::SeqCst);
            let i =
                query_get::<InQ>(provider, DKey(WHICH.load(Ordering::SeqCst) as usize)).unwrap();
            FpValue {
                raw: format!("leaf{}", key.0),
                semantic: 1000 + i.semantic,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // 1) Leaf reads In0.
    let v = system.get::<LeafQ>(DKey(1), provider).unwrap();
    assert_eq!(v.semantic, 1010);
    assert_eq!(LEAF_COUNT.load(Ordering::SeqCst), 1);

    // 2) Leaf now reads In1; recompute so its CURRENT read set is {In1}.
    WHICH.store(1, Ordering::SeqCst);
    system.invalidate::<LeafQ>(&DKey(1));
    let v = system.get::<LeafQ>(DKey(1), provider).unwrap();
    assert_eq!(v.semantic, 1020);
    assert_eq!(LEAF_COUNT.load(Ordering::SeqCst), 2);

    // 3) Change In0 — which Leaf NO LONGER reads — and invalidate it.
    IN0.store(99, Ordering::SeqCst);
    system.invalidate::<InQ>(&DKey(0));
    let v = system.get::<InQ>(DKey(0), provider).unwrap();
    assert_eq!(v.semantic, 99);

    // 4) Re-request Leaf: the stale In0 edge must not force a recompute.
    //    (Before the edge-reconciliation fix this recomputed Leaf:
    //    LEAF_COUNT becomes 3.)
    let v = system.get::<LeafQ>(DKey(1), provider).unwrap();
    assert_eq!(v.semantic, 1020);
    assert_eq!(
        LEAF_COUNT.load(Ordering::SeqCst),
        2,
        "stale edge to In0 must not force a Leaf recompute"
    );
}

// ── d8: trivia / content-class invariance ─────────────────────────

/// d8 (query level).  Two-channel lexing produces one query node per
/// content class; a trivia edit (comment/whitespace) changes the trivia
/// channel's fingerprint but not the semantic channel's.  Guarantees:
///   - semantic readers (typeck stand-in) recompute delta == 0;
///   - declared trivia/doc readers (docs/formatter stand-in) recompute;
///   - a real semantic edit still propagates to its readers.
#[test]
fn d8_trivia_invariance_semantic_queries() {
    static SEM_CONTENT: AtomicU64 = AtomicU64::new(42);
    static TRIV_CONTENT: AtomicU64 = AtomicU64::new(7);
    static SEM_COUNT: AtomicU64 = AtomicU64::new(0);
    static TRIV_COUNT: AtomicU64 = AtomicU64::new(0);
    static TC_COUNT: AtomicU64 = AtomicU64::new(0);
    static DOC_COUNT: AtomicU64 = AtomicU64::new(0);
    SEM_CONTENT.store(42, Ordering::SeqCst);
    TRIV_CONTENT.store(7, Ordering::SeqCst);
    SEM_COUNT.store(0, Ordering::SeqCst);
    TRIV_COUNT.store(0, Ordering::SeqCst);
    TC_COUNT.store(0, Ordering::SeqCst);
    DOC_COUNT.store(0, Ordering::SeqCst);

    struct D8SemQ;
    impl Query for D8SemQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d8_sem", "semantic channel of the lex")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            let n = SEM_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
            FpValue {
                raw: format!("sem#{}_{}", key.0, n),
                semantic: SEM_CONTENT.load(Ordering::SeqCst),
            }
        }
    }

    struct D8TrivQ;
    impl Query for D8TrivQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d8_triv", "trivia channel of the lex")
                .with_content_class(dep_graph::ContentClass::TRIVIA)
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> FpValue {
            let n = TRIV_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
            FpValue {
                raw: format!("triv#{}_{}", key.0, n),
                semantic: TRIV_CONTENT.load(Ordering::SeqCst),
            }
        }
    }

    /// Typeck stand-in: reads ONLY the semantic channel.
    struct D8TypeckQ;
    impl Query for D8TypeckQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d8_typeck", "semantic reader")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            let n = TC_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
            let sem = query_get::<D8SemQ>(provider, key.clone()).expect("typeck reads semantic");
            FpValue {
                raw: format!("tc#{}_{}", key.0, n),
                semantic: 1000 + sem.semantic,
            }
        }
    }

    /// Docs/formatter stand-in: reads ONLY the trivia channel.
    struct D8DocQ;
    impl Query for D8DocQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d8_doc", "trivia/doc reader")
                .with_content_class(dep_graph::ContentClass::DOC)
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            let n = DOC_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
            let triv = query_get::<D8TrivQ>(provider, key.clone()).expect("doc reads trivia");
            FpValue {
                raw: format!("doc#{}_{}", key.0, n),
                semantic: 2000 + triv.semantic,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Run 1: both readers compute.
    let _ = system.get::<D8TypeckQ>(DKey(1), provider).unwrap();
    let _ = system.get::<D8DocQ>(DKey(1), provider).unwrap();
    assert_eq!(SEM_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(TRIV_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(TC_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(DOC_COUNT.load(Ordering::SeqCst), 1);

    // Trivia edit: comment text changes, semantic content does not.
    TRIV_CONTENT.store(8, Ordering::SeqCst);
    system.invalidate::<D8TrivQ>(&DKey(1));
    let _ = system.get::<D8TrivQ>(DKey(1), provider).unwrap();
    assert_eq!(
        TRIV_COUNT.load(Ordering::SeqCst),
        2,
        "trivia channel must recompute after a trivia edit"
    );

    // Semantic reader untouched by a trivia edit.
    let _ = system.get::<D8TypeckQ>(DKey(1), provider).unwrap();
    assert_eq!(
        TC_COUNT.load(Ordering::SeqCst),
        1,
        "typeck recompute delta must be 0 under a trivia-only edit"
    );

    // Declared trivia/doc reader MUST recompute.
    let _ = system.get::<D8DocQ>(DKey(1), provider).unwrap();
    assert_eq!(
        DOC_COUNT.load(Ordering::SeqCst),
        2,
        "doc reader must recompute after a trivia edit"
    );

    // Semantic edit propagates to its reader — and only to it.
    SEM_CONTENT.store(43, Ordering::SeqCst);
    system.invalidate::<D8SemQ>(&DKey(1));
    let _ = system.get::<D8SemQ>(DKey(1), provider).unwrap();
    assert_eq!(SEM_COUNT.load(Ordering::SeqCst), 2);
    let _ = system.get::<D8TypeckQ>(DKey(1), provider).unwrap();
    assert_eq!(
        TC_COUNT.load(Ordering::SeqCst),
        2,
        "semantic edit must propagate to the typeck reader"
    );
    let _ = system.get::<D8DocQ>(DKey(1), provider).unwrap();
    assert_eq!(
        DOC_COUNT.load(Ordering::SeqCst),
        2,
        "semantic edit must not touch the doc reader"
    );
}

/// d8 (lexer level): the SEMANTIC token stream must be invariant under
/// trivia noise — whitespace, line comments, block comments AND doc
/// comments.  Docs are routed to their own channel (the `doc_tokens`
/// query), so doc comments no longer ride the semantic stream and this
/// test passes green.
#[test]
fn d8b_lexer_semantic_stream_invariance() {
    use crate::ast::doc_map::TriviaKind;
    use crate::lexer::Token;

    /// The semantic stream through the TWO-CHANNEL entry point
    /// (Parser's upfront lex): trivia skipped, docs diverted.
    fn semantic_tokens(src: &str) -> Vec<Token> {
        let bump = bumpalo::Bump::new();
        let arena = crate::arena::SyncArena::new(&bump);
        crate::parser::Parser::new(src, &arena).semantic_tokens()
    }

    let base = "def f(x) { return x }";
    let noisy = "/* leading */  def   f(x) { // c\n  return x } /* tail */";
    let docced = "/// @spec A spec\ndef f(x) { return x }";

    let base_t = semantic_tokens(base);
    assert_eq!(
        base_t,
        semantic_tokens(noisy),
        "whitespace/comment noise must not change the semantic stream"
    );
    assert_eq!(
        base_t,
        semantic_tokens(docced),
        "doc-comment insertion must not change the semantic stream \
         (docs belong to the doc channel, not the semantic stream)"
    );

    // Second channel must CAPTURE what the semantic stream excludes:
    // the doc text lands in the doc channel, trivia runs in the trivia
    // channel — nothing is lost, only re-channelled.
    let bump = bumpalo::Bump::new();
    let arena = crate::arena::SyncArena::new(&bump);
    let parser = crate::parser::Parser::new(docced, &arena);
    assert_eq!(
        parser
            .doc_map
            .docs
            .iter()
            .map(|d| d.text.as_str())
            .collect::<Vec<_>>(),
        vec!["@spec A spec"],
        "doc comment must be captured into the doc channel"
    );
    let bump2 = bumpalo::Bump::new();
    let arena2 = crate::arena::SyncArena::new(&bump2);
    let parser2 = crate::parser::Parser::new(noisy, &arena2);
    assert!(
        parser2
            .doc_map
            .trivia
            .iter()
            .any(|t| t.kind == TriviaKind::Comment),
        "comments must be captured into the trivia channel"
    );
    assert!(
        !parser2
            .semantic_tokens()
            .iter()
            .any(|t| matches!(t, Token::DocComment(_) | Token::ModuleDocComment(_))),
        "no doc token may survive into the semantic stream"
    );
}

// ── DocTokens: the `doc_tokens` query (ContentClass::DOC) ────────────

/// The `doc_tokens` query routes item docs through the query system as a
/// first-class DOC-class value: keyed by `FnId`, backed by the doc map's
/// anchor-token lookup (a `///` comment precedes its item, so matching is
/// by the comment's ANCHOR token inside the item span, not the comment's
/// own span), and stable-hashed per line so cache reuse follows doc
/// content.  `//!` module docs never anchor to an item and must not leak
/// into any function's value.
#[test]
fn doctokens_query_returns_item_docs_via_doc_map() {
    use crate::hir::query::queries::{DocTokens, FnId};
    use crate::hir::resolver::ResolutionMap;
    use crate::hir::symbol::SymbolTable;
    use crate::hir::traits::TraitEnv;
    use crate::hir::types::{CrateId, DefId};
    use crate::session::SessionQueryProvider;

    let src = "//! module doc must not leak\n/// @spec adds\n/// second line\ndef add(x, y) { return x; }\ndef bare(x) { return x; }";
    let bump: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
    let arena: &'static crate::arena::SyncArena<'static> =
        Box::leak(Box::new(crate::arena::SyncArena::new(bump)));
    let program = Box::leak(Box::new(
        crate::parser::Parser::new(src, arena)
            .parse_program()
            .expect("test source must parse"),
    ));

    let system: &'static QuerySystem = Box::leak(Box::new(QuerySystem::new()));
    let provider: &'static SessionQueryProvider<'static, 'static> =
        Box::leak(Box::new(SessionQueryProvider::new(
            system,
            Box::leak(Box::new(SymbolTable::new(CrateId(DefId(0))))),
            Box::leak(Box::new(TraitEnv::default())),
            Box::leak(Box::new(ResolutionMap::default())),
            program,
        )));

    // `def add` is item 0 (the `//!` module doc is channel data, not an
    // item); `def bare` is item 1.
    let add = system
        .get::<queries::DocTokens>(
            FnId {
                item: 0,
                method: None,
            },
            provider,
        )
        .expect("doc_tokens for `add` must not error");
    assert_eq!(
        add.lines,
        vec!["@spec adds".to_string(), "second line".to_string()],
        "the item's /// doc lines must come through the query, anchored \
         by token (the comment span itself lies OUTSIDE the item span)"
    );

    let bare = system
        .get::<queries::DocTokens>(
            FnId {
                item: 1,
                method: None,
            },
            provider,
        )
        .expect("doc_tokens for `bare` must not error");
    assert!(
        bare.lines.is_empty(),
        "an undocumented function must get no doc lines"
    );

    // Same lines → same stable hash (cache reuse); different lines →
    // different hash (a doc edit invalidates via fingerprint).
    let same = queries::DocLines {
        lines: add.lines.clone(),
    };
    let other = queries::DocLines {
        lines: vec!["@spec changed".to_string()],
    };
    assert_eq!(add.stable_hash(), same.stable_hash());
    assert_ne!(add.stable_hash(), other.stable_hash());
}

// ── Perf probes (#[ignore]; run: cargo test --bins bench -- --ignored --nocapture) ──
//
// Profile-first discipline: these measure the three candidate hot spots
// from the optimization review BEFORE any code changes:
//   P1  hit-path deep clone   — DefaultCache::lookup clones V under the read lock
//   P2  publish serialization — finish_task + publish are FUSED into one
//                               critical section on the node's own dep-graph
//                               shard (see `finish_task_and_publish`); the
//                               old global-write-lock serial point is gone
//   P3  value-size scaling    — clone cost ∝ sizeof(V) (Vec-heavy values)

// ── Cross-system dep-edge guard (F1) + sticky-poison lift (F2) ────────

/// F1 regression: a nested get served by a FOREIGN system must not leak
/// that system's node indices into the executing system's dep graph.
/// Pre-fix, `sys_b`'s serve-path `dep_graph.read` recorded B's node index
/// into A's task frame; the finish flushed it into A's shards as a
/// phantom node (and dropped the true edge on the owning system).
#[test]
fn f1_cross_system_nested_serve_must_not_leak_foreign_indices() {
    static E_RUNS: AtomicU64 = AtomicU64::new(0);
    E_RUNS.store(0, Ordering::SeqCst);

    struct XE;
    impl Query for XE {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("f1_e", "foreign leaf")
        }
        fn compute(_: &DKey, _: &dyn QueryProvider) -> DValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            DValue::new(7)
        }
    }
    struct XK;
    impl Query for XK {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("f1_k", "reader whose provider is bound elsewhere")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> DValue {
            let e = query_get::<XE>(provider, key.clone()).expect("nested foreign get");
            DValue::new(e.semantic * 10)
        }
    }

    let sys_a = leak_system();
    let _prov_a = leak_provider(sys_a);
    let sys_b = leak_system();
    let prov_b = leak_provider(sys_b);

    // Warm XE on B so the nested get inside XK's compute takes the
    // cached-serve path — the one whose `try_serve_verified` records the
    // read into the CURRENT task frame (A's XK frame).
    let warm = run_with_guard(GUARD, move || sys_b.get::<XE>(DKey(0), prov_b).map(|v| v.semantic))
        .expect("warmup must finish");
    assert_eq!(warm.expect("warmup ok"), 7);

    #[cfg(debug_assertions)]
    {
        // Post-fix the cross-system guard fires loudly on the pairing
        // violation; XK's own compute `catch_unwind` resumes it out of
        // `get`, so the whole call must panic.
        let r = run_with_guard(GUARD, move || {
            let _ = sys_a.get::<XK>(DKey(0), prov_b).map(|v| v.semantic);
            "returned"
        });
        assert_eq!(
            r,
            Err("PANIC".to_string()),
            "debug builds must fail fast on the provider/system pairing violation"
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let v = run_with_guard(GUARD, move || sys_a.get::<XK>(DKey(0), prov_b).map(|v| v.semantic))
            .expect("get must finish")
            .expect("release: the get succeeds, the foreign record is skipped");
        assert_eq!(v, 70);
        // XK's node on A must carry NO edges: the foreign serve's read was
        // refused, so nothing from B's index space may appear in A's graph.
        let vt = sys_a
            .find_vtable::<XK>()
            .and_then(|v| v.lookup_node(&DKey(0)))
            .expect("XK must have a node on A");
        let edges = sys_a.dep_graph.forward_edges_of(vt);
        assert!(
            edges.is_empty(),
            "a foreign system's node index leaked into A's graph: {edges:?}"
        );
    }
}

/// F2 regression: `invalidate` must lift a sticky `Poisoned` entry so the
/// next get recomputes against the new world instead of erroring forever
/// with "query previously panicked".
#[test]
fn f2_invalidate_lifts_sticky_poison() {
    static PANIC_ON: AtomicBool = AtomicBool::new(false);
    PANIC_ON.store(false, Ordering::SeqCst);

    struct PQ;
    impl Query for PQ {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("f2_poison", "poison-lift probe")
        }
        fn compute(_: &DKey, _: &dyn QueryProvider) -> DValue {
            if PANIC_ON.load(Ordering::SeqCst) {
                panic!("f2 boom");
            }
            DValue::new(7)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // 1) Poison the key: the compute panics, the key is marked Poisoned.
    PANIC_ON.store(true, Ordering::SeqCst);
    let r = run_with_guard(GUARD, move || {
        system
            .get::<PQ>(DKey(0), provider)
            .map(|v| v.semantic)
            .map_err(|e| e.message)
    });
    assert_eq!(r, Err("PANIC".to_string()), "compute must panic");

    // 2) The world changed; the caller invalidates to retry.
    system.invalidate::<PQ>(&DKey(0));
    PANIC_ON.store(false, Ordering::SeqCst);

    // 3) The next get must RECOMPUTE, not report "previously panicked".
    let v = run_with_guard(GUARD, move || {
        system
            .get::<PQ>(DKey(0), provider)
            .map(|v| v.semantic)
            .map_err(|e| e.message)
    })
    .expect("get must finish");
    assert_eq!(v.expect("invalidate must lift the sticky poison"), 7);
}

/// Global depth backstop: a chain of nested gets with DISTINCT keys and
/// no `depth_limit` opt-in must be cut off by `MAX_GLOBAL_DEPTH` instead
/// of overflowing the stack.  A single query type suffices — each nested
/// frame increments `IN_FLIGHT_DEPTH` regardless of type, and no per-type
/// check fires (keys are distinct, so no cycle; `depth_limit` is false).
#[test]
fn f3_global_depth_backstop_bounds_unbounded_chain() {
    struct ChainQ;
    impl Query for ChainQ {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("f3_chain", "cross-type depth backstop probe")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> DValue {
            if key.0 >= 600 {
                // Deeper than MAX_GLOBAL_DEPTH — the backstop must fire
                // before this chain gets anywhere near this branch.
                return DValue::new(0);
            }
            match query_get::<ChainQ>(provider, DKey(key.0 + 1)) {
                Ok(v) => v,
                Err(e) => {
                    if e.message.contains("depth") {
                        // Marker propagates up: the outermost get must
                        // report the backstop, not the sentinel at 600.
                        DValue::new(u64::MAX)
                    } else {
                        panic!("unexpected cycle error: {}", e.message);
                    }
                }
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // A big stack so the ~513 nested debug-build frames the backstop
    // allows fit comfortably; the test asserts the ERROR, not a crash.
    let h = thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || system.get::<ChainQ>(DKey(0), provider).map(|v| v.semantic))
        .expect("spawn chain thread");
    let v = h.join().expect("chain thread must not panic (no stack overflow)");
    assert_eq!(
        v.expect("outermost get must succeed"),
        u64::MAX,
        "the global depth backstop must cut the chain off before key 600"
    );
}

/// Test-local cache wrapper whose `insert` panics on demand (T2): it
/// exercises the phase-1 staging guard in `get_shared` — a panic between
/// compute and finalize (here, inside `stage`, standing in for
/// `DefaultCache`'s defensive eviction guards) must run the
/// SAME `finish_task_and_publish(None)` + `poison_active` cleanup as a
/// compute panic, releasing waiters instead of stranding the `Started`
/// entry forever.
///
/// The trap is armed through a type-level static because the vtable owns
/// its cache instance privately.
#[derive(Debug)]
struct PanickingInsertCache<K, V> {
    inner: DefaultCache<K, V>,
    _marker: std::marker::PhantomData<fn(&K, &V)>,
}

/// Armed by the T2 test before the owner runs; consulted by every
/// `PanickingInsertCache::insert`.  Global because `Query::Cache` is
/// constructed by the vtable, not by the test.
static T2_PANIC_ARMED: AtomicBool = AtomicBool::new(false);

impl<K, V> QueryCacheType<K, V> for PanickingInsertCache<K, V>
where
    K: std::fmt::Debug + Clone + Eq + std::hash::Hash + 'static,
    V: std::fmt::Debug + Clone + 'static,
{
    fn probe(
        &self,
        key: &K,
        alloc: impl FnOnce() -> dep_graph::DepNodeIndex,
    ) -> (dep_graph::DepNodeIndex, Option<Arc<V>>, bool) {
        self.inner.probe(key, alloc)
    }
    fn stage(&self, key: K, node: dep_graph::DepNodeIndex, value: Arc<V>) {
        if T2_PANIC_ARMED.load(Ordering::SeqCst) {
            panic!("t2 staged-insert panic (evict_tail unreachable!() stand-in)");
        }
        self.inner.stage(key, node, value);
    }
    fn unstage(&self, key: &K) -> Option<dep_graph::DepNodeIndex> {
        self.inner.unstage(key)
    }
    fn node_of(&self, key: &K) -> Option<dep_graph::DepNodeIndex> {
        self.inner.node_of(key)
    }
    fn clear(&self) {
        self.inner.clear();
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn for_each(&self, f: impl FnMut(&K, &V)) {
        self.inner.for_each(f);
    }
    fn from_capacity(capacity: usize) -> Self {
        PanickingInsertCache {
            inner: DefaultCache::from_capacity(capacity),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<K, V> Default for PanickingInsertCache<K, V>
where
    K: std::fmt::Debug + Clone + Eq + std::hash::Hash + 'static,
    V: std::fmt::Debug + Clone + 'static,
{
    fn default() -> Self {
        Self::from_capacity(0)
    }
}

/// T2 (covers the phase-1 staging guard): a panic INSIDE
/// `cache.stage` — the realistic trigger being a defensive guard inside
/// the cache's staging path — must behave exactly like a compute panic:
/// the owner's task frame is popped, waiters parked on the one-shot latch
/// are released with an error (never TIMEOUT), and the key is left
/// `Poisoned` for the session (until `invalidate` lifts it).
#[test]
fn t2_staging_panic_releases_waiters() {
    struct T2Q;
    impl Query for T2Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = PanickingInsertCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("t2_stage_panic", "staging-panic probe")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> DValue {
            // Wait until a waiter is parked on our latch before the
            // panic fires, so the release path is actually exercised.
            if key.0 == 0 {
                wait_flag(&T2_WAITER_PARKED);
            }
            DValue::new(1)
        }
    }
    static T2_WAITER_PARKED: AtomicBool = AtomicBool::new(false);
    T2_WAITER_PARKED.store(false, Ordering::SeqCst);

    let system = leak_system();
    let provider = leak_provider(system);

    // Arm the trap, then start the owner and a waiter on the same key.
    // The vtable owns its cache instance privately, so the test arms the
    // trap through the type-level static the wrapper consults.
    T2_PANIC_ARMED.store(true, Ordering::SeqCst);

    // Start the owner (its compute BLOCKS on T2_WAITER_PARKED) and the
    // waiter on the same key, then release the owner only once the
    // waiter has registered its wait edge — this makes the panic land
    // while a waiter is genuinely parked on the one-shot latch.
    let owner = {
        let system = system;
        let provider = provider;
        // Same error mapping as the waiter: both joins then carry a
        // `String` error, so the symmetric classification below has one
        // uniform shape.
        thread::spawn(move || {
            system
                .get::<T2Q>(DKey(0), provider)
                .map(|v| v.semantic)
                .map_err(|e| e.message)
        })
    };

    let waiter = {
        let system = system;
        let provider = provider;
        thread::spawn(move || {
            system
                .get::<T2Q>(DKey(0), provider)
                .map(|v| v.semantic)
                .map_err(|e| e.message)
        })
    };
    // Deterministic: wait until the waiter registered its wait edge, THEN
    // let the owner's compute return so the panic fires inside
    // `cache.stage` (phase 1) with the waiter parked on the latch.
    let mut parked = false;
    for _ in 0..200 {
        if !system.wait_graph.read().is_empty() {
            parked = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        parked,
        "waiter must be parked on the owner's latch before the panic fires"
    );
    T2_WAITER_PARKED.store(true, Ordering::SeqCst);

    // The staging panic fired on whichever thread WON the owner slot (the
    // two threads race for it); the other thread is the waiter and is
    // released with the poison error.  The assertions are therefore
    // SYMMETRIC: exactly one thread must have panicked, and the other
    // must have been released with an error mentioning the poison —
    // neither may TIMEOUT (that would mean a stranded latch) or succeed.
    let r1 = run_with_guard(GUARD, move || owner.join());
    let r2 = run_with_guard(GUARD, move || waiter.join());
    let classify = (
        match &r1 {
            Ok(Err(_payload)) => "panicked",
            Ok(Ok(Err(msg)))
                if msg.contains("poisoned") || msg.contains("panicked") =>
            {
                "released"
            }
            Ok(Ok(Ok(_))) => "SUCCEEDED",
            Ok(Ok(Err(msg))) => panic!("unexpected error: {msg}"),
            Ok(Err(_)) => panic!("thread panicked unexpectedly"),
            Err(_) => panic!("TIMEOUT: a latch was stranded"),
        },
        match &r2 {
            Ok(Err(_payload)) => "panicked",
            Ok(Ok(Err(msg)))
                if msg.contains("poisoned") || msg.contains("panicked") =>
            {
                "released"
            }
            Ok(Ok(Ok(_))) => "SUCCEEDED",
            Ok(Ok(Err(msg))) => panic!("unexpected error: {msg}"),
            Ok(Err(_)) => panic!("thread panicked unexpectedly"),
            Err(_) => panic!("TIMEOUT: a latch was stranded"),
        },
    );
    // The two threads RACE for the owner slot, so "panicked" and
    // "released" may land on either handle — the invariant is the SET
    // {panicked, released}, not the order.
    let mut verdict = [classify.0, classify.1];
    verdict.sort_unstable();
    assert_eq!(
        verdict,
        ["panicked", "released"],
        "exactly one thread must panic (the owner) and the other must be \
         released with the poison error: r1={r1:?} r2={r2:?}"
    );

    // The key is left `Poisoned`: the next getter reports the panic
    // instead of recomputing (sticky poison, lifted only by invalidate).
    let r_next = run_with_guard(GUARD, move || {
        system
            .get::<T2Q>(DKey(0), provider)
            .map(|v| v.semantic)
            .map_err(|e| e.message)
    })
    .expect("next get must finish");
    let msg = r_next.expect_err("next getter must see the sticky poison");
    assert!(
        msg.contains("panicked") || msg.contains("poisoned"),
        "next getter must see the sticky poison, got: {msg}"
    );
    T2_PANIC_ARMED.store(false, Ordering::SeqCst);
}

/// T1 stress (covers F1 + F2): N threads hammering a shared key set with
/// randomized timing to force latch waits and wait-edge registration.  The
/// query is a leaf (no compute ever requests another key), so a REAL
/// cross-thread cycle is impossible by construction — every `Err` from a
/// get is a spurious cycle verdict, and every TIMEOUT is a stuck walk or
/// an unreleased latch.  Both are regression surfaces for the completion
/// reordering (F1: latch resolved before edge clearing) and the bounded
/// wait-graph walk (F2: visited set).
#[test]
fn t1_no_spurious_cycle_under_mutual_request_storm() {
    static T1_COMPUTES: AtomicU64 = AtomicU64::new(0);
    T1_COMPUTES.store(0, Ordering::SeqCst);

    struct T1Q;
    impl Query for T1Q {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("t1_storm", "leaf query for the request storm")
        }
        fn compute(key: &DKey, _: &dyn QueryProvider) -> DValue {
            T1_COMPUTES.fetch_add(1, Ordering::SeqCst);
            // A short, key-dependent spin so concurrent same-key requests
            // overlap and produce waiters — without any cross-key
            // dependency (a dependency could form REAL cycles, whose
            // Err results are legitimate and would blur the assertion).
            let spins = 200 * (key.0 as u64 % 7 + 1);
            for _ in 0..spins {
                std::hint::spin_loop();
            }
            DValue::new(key.0 as u64 * 10)
        }
    }

    static KEYS: u64 = 8;
    static THREADS: u64 = 4;
    static GETS_PER_THREAD: u64 = 300;

    let system = leak_system();
    let provider = leak_provider(system);

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let system = system;
            let provider = provider;
            thread::spawn(move || {
                // Per-thread LCG: deterministic but desynchronized.
                let mut state = 0x9E3779B9_7F4A7C15 ^ (tid.wrapping_mul(0x2545F491_4F6CDD1D));
                let mut lcg = move || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                };
                let mut results: Vec<(u64, Result<u64, QueryCycleError>)> = Vec::new();
                for i in 0..GETS_PER_THREAD {
                    let key = DKey((lcg() % KEYS) as usize);
                    // A small randomized pause so requests interleave at
                    // different phases — some hit the cached path, some
                    // race into the compute path and become waiters.
                    let pause = lcg() % 3;
                    if pause > 0 {
                        thread::sleep(Duration::from_micros(pause * 50));
                    }
                    results.push((
                        i,
                        system.get::<T1Q>(key, provider).map(|v| v.semantic),
                    ));
                }
                results
            })
        })
        .collect();

    for (tid, h) in handles.into_iter().enumerate() {
        let results = run_with_guard(GUARD, move || h.join())
            .unwrap_or_else(|_| panic!("storm thread {tid} must finish within the guard"))
            .unwrap_or_else(|e| panic!("storm thread {tid} must not panic: {e:?}"));
        for (i, r) in results.into_iter() {
            assert!(
                r.is_ok(),
                "spurious cycle error or failure at thread {tid} get {i}: {r:?}"
            );
        }
    }
    // Dedup must have engaged at least once under the storm (same keys,
    // concurrent requests); a pure-cache run would suggest the waiters
    // never formed and the test exercised nothing.
    assert!(
        T1_COMPUTES.load(Ordering::SeqCst) < THREADS * GETS_PER_THREAD,
        "the storm must produce same-key dedup waits, not one compute per request"
    );
}

/// T3 (covers Issue A): the finish-side epoch restore must resolve the
/// node's colour INSIDE the publish critical section, before the node is
/// observably green.  The owner computes a FRESH key K whose only read is
/// E (eval_always); while the owner is parked mid-compute, E republishes
/// with a CHANGED fingerprint — the fan-out cannot reach K (K's reverse
/// edge is not flushed yet).  The owner's staged value therefore encodes
/// the PRE-change E.  The D23 hook parks the owner inside
/// `finish_task_and_publish` after the in-section colour decision, and a
/// getter probes K inside that window:
///
/// - pre-fix: `clear_dirty` already ran, so the getter sees GREEN + the
///   staged cache entry and SERVES the suspect value;
/// - post-fix: the in-section suspect check kept the node red, so the
///   getter falls through to the compute path, waits on the still
///   installed `Started` entry, wakes, re-probes red, and RECOMPUTES.
#[test]
fn t3_suspect_value_never_served_green_in_publish_window() {
    static E_RUNS: AtomicU64 = AtomicU64::new(0);
    static K_RUNS: AtomicU64 = AtomicU64::new(0);
    static E_CONTENT: AtomicU64 = AtomicU64::new(1);
    static T3_K_PARK: AtomicBool = AtomicBool::new(false);
    static T3_K_ARRIVED: AtomicBool = AtomicBool::new(false);
    E_RUNS.store(0, Ordering::SeqCst);
    K_RUNS.store(0, Ordering::SeqCst);
    E_CONTENT.store(1, Ordering::SeqCst);
    T3_K_PARK.store(false, Ordering::SeqCst);
    T3_K_ARRIVED.store(false, Ordering::SeqCst);
    dep_graph::D23_PARK.store(false, Ordering::SeqCst);
    dep_graph::D23_TARGET.store(0, Ordering::SeqCst);
    dep_graph::D23_ARRIVED.store(false, Ordering::SeqCst);

    struct T3E;
    impl Query for T3E {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("t3_e", "eval-always leaf whose value changes under K")
                .with_eval_always(true)
        }
        fn compute(_: &DKey, _: &dyn QueryProvider) -> DValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            DValue::new(E_CONTENT.load(Ordering::SeqCst))
        }
    }
    struct T3K;
    impl Query for T3K {
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, DValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("t3_k", "fresh reader of the mid-compute-changed leaf")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> DValue {
            K_RUNS.fetch_add(1, Ordering::SeqCst);
            let e = query_get::<T3E>(provider, key.clone()).expect("K reads E");
            // Park AFTER the read: E's epoch is captured here; the
            // republish below lands inside this park so the fan-out
            // misses K (no rev edge flushed yet).
            T3_K_ARRIVED.store(true, Ordering::SeqCst);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while T3_K_PARK.load(Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    break;
                }
                std::thread::yield_now();
            }
            DValue::new(e.semantic * 10)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // 1) Warm E only: K must stay a FRESH node with no flushed rev edge.
    let warm = run_with_guard(GUARD, move || system.get::<T3E>(DKey(0), provider).map(|v| v.semantic))
        .expect("warm must finish")
        .expect("warm ok");
    assert_eq!(warm, 1);
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 1);

    // 2) Owner computes K and parks mid-compute (epoch of E captured).
    T3_K_PARK.store(true, Ordering::SeqCst);
    let owner = {
        let system = system;
        let provider = provider;
        thread::spawn(move || system.get::<T3K>(DKey(0), provider).map(|v| v.semantic))
    };
    wait_flag(&T3_K_ARRIVED);

    // 3) E republishes with a CHANGED fingerprint: bump_epoch(E) fires,
    //    the fan-out misses K (no rev edge yet).
    E_CONTENT.store(2, Ordering::SeqCst);
    let _ = run_with_guard(GUARD, move || system.get::<T3E>(DKey(0), provider).map(|v| v.semantic))
        .expect("E republish must finish")
        .expect("E republish ok");
    // eval_always recomputes on every get: warm (1) + the owner's nested
    // read inside K's compute (2) + this republish (3).
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 3);

    // 4) Arm the finish-window park on K's node (it exists — the owner's
    //    probe created it before its compute parked), then release the
    //    owner's compute.  The owner stages the SUSPECT value, the
    //    in-section suspect check sees epoch(E) 1 != captured 0 and keeps
    //    K red, then the D23 hook parks it inside the window.
    let k_node = system
        .find_vtable::<T3K>()
        .and_then(|vt| vt.lookup_node(&DKey(0)))
        .expect("K node must exist after the owner's probe");
    dep_graph::D23_TARGET.store(k_node.0, Ordering::SeqCst);
    dep_graph::D23_PARK.store(true, Ordering::SeqCst);
    T3_K_PARK.store(false, Ordering::SeqCst);
    wait_flag(&dep_graph::D23_ARRIVED);

    // 5) A getter probes K INSIDE the window.
    let getter = {
        let system = system;
        let provider = provider;
        thread::spawn(move || {
            system
                .get::<T3K>(DKey(0), provider)
                .map(|v| v.semantic)
                .map_err(|e| e.message)
        })
    };
    // Deterministic: the getter must have registered its wait edge on
    // the owner's still-installed Started entry before we release.
    let mut parked = false;
    for _ in 0..200 {
        if !system.wait_graph.read().is_empty() {
            parked = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        parked,
        "getter must be parked on the owner's latch before the window closes"
    );

    // 6) Close the window: the owner flushes, the late re-check restores
    //    red (idempotent), completes, and signals the getter — which
    //    re-probes RED and recomputes instead of serving the suspect
    //    value.
    dep_graph::D23_PARK.store(false, Ordering::SeqCst);
    let r_owner = run_with_guard(GUARD, move || owner.join());
    let owner_v = match r_owner {
        Ok(Ok(v)) => v.expect("owner's own get must succeed"),
        other => panic!("owner must not panic/TIMEOUT: {other:?}"),
    };
    assert_eq!(owner_v, 10, "the owner's own result is the value it computed");

    let r_getter = run_with_guard(GUARD, move || getter.join());
    let getter_v = match r_getter {
        Ok(Ok(v)) => v.expect("getter must succeed"),
        Ok(Err(msg)) => panic!("getter must not be poisoned: {msg:?}"),
        Err(_) => panic!("getter TIMEOUT: a latch was stranded"),
    };
    assert_eq!(
        getter_v, 20,
        "the getter must RECOMPUTE against the changed E, not serve the \
         suspect staged value (10) from the publish window"
    );
    assert_eq!(K_RUNS.load(Ordering::SeqCst), 2, "K must have recomputed");
    // eval_always recomputes on EVERY get by design (d5): warm (1) + the
    // owner's nested read inside K's compute (2) + the republish (3) +
    // the getter's K recompute reading E again (4).
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 4, "eval_always runs once per get");
    dep_graph::D23_TARGET.store(0, Ordering::SeqCst);
    dep_graph::D23_PARK.store(false, Ordering::SeqCst);
}

/// d24: a mark landing in the finish window [in-section colour decision ..
/// post-flush re-check] whose read node is republished green before a
/// getter's verify must not let the getter serve the staged (stale-based)
/// value.  The per-node `finishing` flag closes the window: a getter that
/// reaches the node while its finish tail is in flight treats it as dirty
/// and takes the compute/wait path instead of serving.
///
/// NOTE on step 4: the mark MUST land inside the window — i.e. while the
/// owner is parked at its finish hook still holding the quiesce read guard.
/// A quiesced `QuerySystem::invalidate` would block on the write lock
/// until the owner's 2s hook deadline expired, closing the window and
/// degenerating the test into an ordinary post-publish fan-out.  The test
/// therefore marks E through the non-quiesced `DepGraph::invalidate` —
/// exactly the mark class the window models (publish-time changed-
/// fingerprint fan-outs and the d16 tooling marks are non-quiesced by
/// design).
#[test]
fn d24_finish_window_republish_never_served() {
    static E_CONTENT: AtomicU64 = AtomicU64::new(1);
    static E_RUNS: AtomicU64 = AtomicU64::new(0);
    static K_RUNS: AtomicU64 = AtomicU64::new(0);
    static K_ARRIVED: AtomicBool = AtomicBool::new(false);
    static K_PARK: AtomicBool = AtomicBool::new(false);
    E_CONTENT.store(1, Ordering::SeqCst);
    E_RUNS.store(0, Ordering::SeqCst);
    K_RUNS.store(0, Ordering::SeqCst);
    K_ARRIVED.store(false, Ordering::SeqCst);
    K_PARK.store(false, Ordering::SeqCst);
    dep_graph::D24_PARK.store(false, Ordering::SeqCst);
    dep_graph::D24_TARGET.store(0, Ordering::SeqCst);
    dep_graph::D24_ARRIVED.store(false, Ordering::SeqCst);

    struct D24E;
    impl Query for D24E {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d24_e", "d24 leaf input")
        }
        fn compute(_key: &DKey, _: &dyn QueryProvider) -> FpValue {
            E_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: String::new(),
                semantic: E_CONTENT.load(Ordering::SeqCst),
            }
        }
    }
    struct D24K;
    impl Query for D24K {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("d24_k", "d24 reader over d24_e")
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            K_RUNS.fetch_add(1, Ordering::SeqCst);
            let e = query_get::<D24E>(provider, key.clone()).unwrap();
            K_ARRIVED.store(true, Ordering::SeqCst);
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while K_PARK.load(Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    break;
                }
                std::thread::yield_now();
            }
            FpValue {
                raw: String::new(),
                semantic: e.semantic * 10,
            }
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // 1) Warm E only (K stays a FRESH node: no flushed rev edge on E).
    let warm = run_with_guard(GUARD, move || {
        system.get::<D24E>(DKey(0), provider).map(|v| v.semantic)
    })
    .expect("warm must finish")
    .expect("warm ok");
    assert_eq!(warm, 1);
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 1);

    // 2) Owner computes K (fresh) and parks in its compute so the test
    //    can arm the D24 finish hook on K's node.
    K_PARK.store(true, Ordering::SeqCst);
    let owner = {
        let (sys, prov) = (system, provider);
        thread::spawn(move || sys.get::<D24K>(DKey(0), prov).map(|v| v.semantic))
    };
    wait_flag(&K_ARRIVED);

    // 3) Arm the D24 finish hook on K's node (it exists — the owner's
    //    probe created it before the compute parked), then release the
    //    owner's compute.  The owner stages the value based on E=1, the
    //    in-section suspect check sees epoch(E) UNCHANGED → clean →
    //    clear_dirty(K) → K observable GREEN, and the D24 hook parks the
    //    owner inside the finish window with `finishing(K)` still set.
    let k_node = system
        .find_vtable::<D24K>()
        .and_then(|vt| vt.lookup_node(&DKey(0)))
        .expect("K node must exist after the owner's probe");
    dep_graph::D24_TARGET.store(k_node.0, Ordering::SeqCst);
    dep_graph::D24_PARK.store(true, Ordering::SeqCst);
    K_PARK.store(false, Ordering::SeqCst);
    wait_flag(&dep_graph::D24_ARRIVED);

    // 4) INSIDE the window: mark E (non-quiesced — see the NOTE above)
    //    and republish it green with a CHANGED fingerprint.  Its fan-out
    //    misses K (K's rev edge is not flushed yet), so K stays green
    //    with a staged value encoding the PRE-change E.
    E_CONTENT.store(2, Ordering::SeqCst);
    let e_node = system
        .find_vtable::<D24E>()
        .and_then(|vt| vt.lookup_node(&DKey(0)))
        .expect("E node must exist after the warm get");
    system.dep_graph.invalidate(e_node);
    let e2 = run_with_guard(GUARD, move || {
        system.get::<D24E>(DKey(0), provider).map(|v| v.semantic)
    })
    .expect("E republish must finish")
    .expect("E republish ok");
    assert_eq!(e2, 2);
    assert_eq!(E_RUNS.load(Ordering::SeqCst), 2);

    // 5) A getter probes K inside the window.  Pre-fix it would serve the
    //    staged 10 immediately (K green, E green after the republish);
    //    post-fix the finishing flag makes verify(K) report dirty, so the
    //    getter falls into the compute path and WAITS on the owner.
    let getter = {
        let (sys, prov) = (system, provider);
        thread::spawn(move || {
            sys.get::<D24K>(DKey(0), prov)
                .map(|v| v.semantic)
                .map_err(|e| e.message)
        })
    };
    let mut parked = false;
    for _ in 0..200 {
        if !system.wait_graph.read().is_empty() {
            parked = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        parked,
        "getter must have taken the wait path, not served the staged value"
    );

    // 6) Close the window: the owner flushes K→E, the post-flush
    //    re-check restores K red (epoch(E) changed), the finishing flag
    //    clears, and the getter wakes, re-probes red and RECOMPUTES
    //    against the fresh E.
    dep_graph::D24_PARK.store(false, Ordering::SeqCst);
    let owner_v = run_with_guard(GUARD, move || owner.join())
        .expect("owner must finish")
        .expect("owner must not panic")
        .expect("owner get must succeed");
    assert_eq!(owner_v, 10, "the owner's own result is the value it computed");

    let getter_v = run_with_guard(GUARD, move || getter.join())
        .expect("getter must finish")
        .expect("getter must not panic")
        .expect("getter get must succeed");
    assert_eq!(getter_v, 20, "getter must recompute against fresh E");
    assert_eq!(K_RUNS.load(Ordering::SeqCst), 2, "K must have recomputed");
    assert_eq!(
        E_RUNS.load(Ordering::SeqCst),
        2,
        "E is green and cached — the getter's K recompute reads it, no recompute"
    );

    dep_graph::D24_TARGET.store(0, Ordering::SeqCst);
    dep_graph::D24_PARK.store(false, Ordering::SeqCst);
}

mod bench {
    use super::*;
    use std::time::Instant;

    /// Value with a `FAT`-element payload (FAT=4096 → 32 KiB deep copy).
    #[derive(Clone, Debug)]
    struct BValue {
        data: Vec<u64>,
        tag: u64,
    }
    impl StableHash for BValue {
        fn stable_hash(&self) -> u64 {
            let mut h = rustc_hash::FxHasher::default();
            self.tag.hash(&mut h);
            h.finish()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct BKey(u32);
    impl QueryKey for BKey {}

    /// Minimal compute: allocate the fat payload and return it.  The
    /// compute cost (one 32 KiB alloc) is identical across thread counts,
    /// so any sub-linear scaling in the publish probe isolates lock
    /// serialization, not compute.
    struct BQuery;
    impl Query for BQuery {
        type Key = BKey;
        type Value = BValue;
        type Cache = DefaultCache<BKey, BValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("bench_probe", "perf probe query")
        }
        fn compute(_key: &BKey, _provider: &dyn QueryProvider) -> BValue {
            BValue {
                data: vec![0u64; 4096],
                tag: 1,
            }
        }
    }

    const KEYS: u32 = 4096;
    const HIT_ITERS: u32 = 20; // × KEYS gets per timing round

    fn setup() -> (&'static QuerySystem, &'static dyn QueryProvider<'static>) {
        let system: &'static QuerySystem = Box::leak(Box::new(QuerySystem::new()));
        let provider = leak_provider(system);
        (system, provider)
    }

    /// P1+P3 — hit path: everything cached; each get pays the read-lock +
    /// 32 KiB clone.  ns/get is the per-hit clone tax; run with both fat
    /// values (this) and check P3 via the ratio to a thin-value variant.
    #[test]
    #[ignore]
    fn bench_hit_path_deep_clone() {
        let (system, provider) = setup();
        // Warm the cache.
        for k in 0..KEYS {
            let _ = system.get::<BQuery>(BKey(k), provider).unwrap();
        }
        // Timing rounds.
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..HIT_ITERS {
                for k in 0..KEYS {
                    let _ = system.get::<BQuery>(BKey(k), provider).unwrap();
                }
            }
            let per = t.elapsed().as_secs_f64() / ((HIT_ITERS * KEYS) as f64);
            best = best.min(per);
        }
        println!(
            "[bench_hit_path] {:>8.1} ns/get (32 KiB value; clone tax = ns - ~50 ns base)",
            best * 1e9
        );
    }

    /// Thin control value (8 B payload).
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct BThin(u64);
    impl StableHash for BThin {
        fn stable_hash(&self) -> u64 {
            let mut h = rustc_hash::FxHasher::default();
            self.0.hash(&mut h);
            h.finish()
        }
    }

    /// P3 control — thin value.  ns/get delta vs the fat run is the
    /// per-hit clone cost of the 32 KiB payload.
    struct ThinQuery;
    impl Query for ThinQuery {
        type Key = BKey;
        type Value = BThin;
        type Cache = DefaultCache<BKey, BThin>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("bench_probe_thin", "perf probe, thin value")
        }
        fn compute(_key: &BKey, _provider: &dyn QueryProvider) -> BThin {
            BThin(1)
        }
    }

    #[test]
    #[ignore]
    fn bench_hit_path_thin_control() {
        let (system, provider) = setup();
        for k in 0..KEYS {
            let _ = system.get::<ThinQuery>(BKey(k), provider).unwrap();
        }
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..HIT_ITERS {
                for k in 0..KEYS {
                    let _ = system.get::<ThinQuery>(BKey(k), provider).unwrap();
                }
            }
            let per = t.elapsed().as_secs_f64() / ((HIT_ITERS * KEYS) as f64);
            best = best.min(per);
        }
        println!("[bench_hit_thin ] {:>8.1} ns/get (8 B value)", best * 1e9);
    }

    /// P4 — PARALLEL hit storm: the probe the node_map/cache merge is
    /// measured against.  All keys warm; N threads hammer the same key set
    /// through the full `get` hit path (probe + transitive verify + cache
    /// read + quiesce guard).  Single-threaded ns/get is dominated by
    /// per-hit hash + a few uncontended atomic RMWs (~10–30 ns of what a
    /// merge saves); under multi-core load the dominant term is instead
    /// cache-line ping-pong on the lock words every hit touches
    /// (node_map shard, dep-graph shard, the cache's ONE lock) — which is
    /// why the single-lock DefaultCache is the merge's real target.
    /// Run pre/post refactor in `--release` and compare.
    const HIT_THREADS: usize = 8;
    const HIT_ROUNDS: u32 = 400; // × KEYS gets per thread per timing round

    #[test]
    #[ignore]
    fn bench_hit_path_parallel() {
        let (system, provider) = setup();
        for k in 0..KEYS {
            let _ = system.get::<ThinQuery>(BKey(k), provider).unwrap();
        }
        static GO: AtomicBool = AtomicBool::new(false);
        let mut best = f64::MAX;
        for _ in 0..3 {
            GO.store(false, Ordering::SeqCst);
            let handles: Vec<_> = (0..HIT_THREADS)
                .map(|_| {
                    let system = system;
                    let provider = provider;
                    thread::spawn(move || {
                        while !GO.load(Ordering::Acquire) {
                            std::hint::spin_loop();
                        }
                        let mut acc = 0u64;
                        for _ in 0..HIT_ROUNDS {
                            for k in 0..KEYS {
                                acc += system.get::<ThinQuery>(BKey(k), provider).unwrap().0;
                            }
                        }
                        acc
                    })
                })
                .collect();
            GO.store(true, Ordering::Release);
            let t = Instant::now();
            let total: u64 = handles.into_iter().map(|h| h.join().expect("hit worker")).sum();
            let per = t.elapsed().as_secs_f64()
                / ((HIT_THREADS * HIT_ROUNDS as usize * KEYS as usize) as f64);
            assert!(total > 0, "workers must observe the values");
            best = best.min(per);
        }
        println!(
            "[bench_hit_par  ] {:>8.1} ns/get ({} threads × {} warm keys)",
            best * 1e9,
            HIT_THREADS,
            KEYS
        );
    }

    /// P3 control for `get_shared` — thin value through the shared-handle
    /// path.  `get_shared` returns `Arc<BThin>` without cloning the payload,
    /// but under parallel hit load each `Arc::clone` is two atomic RMWs
    /// with cache-line ping-pong.  Compare ns/get against
    /// `bench_hit_path_thin_control` (owned path): if the shared path is
    /// SLOWER for 8-byte values, that is the per-query opt-in signal —
    /// `get_shared` belongs on fat values, not thin ones.
    #[test]
    #[ignore]
    fn bench_hit_path_thin_shared() {
        let (system, provider) = setup();
        for k in 0..KEYS {
            let _ = system.get::<ThinQuery>(BKey(k), provider).unwrap();
        }
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..HIT_ITERS {
                for k in 0..KEYS {
                    let _ = system.get_shared::<ThinQuery>(BKey(k), provider).unwrap();
                }
            }
            let per = t.elapsed().as_secs_f64() / ((HIT_ITERS * KEYS) as f64);
            best = best.min(per);
        }
        println!(
            "[bench_hit_thin_sh] {:>8.1} ns/get (8 B via Arc — ping-pong baseline)",
            best * 1e9
        );
    }

    /// P2 — publish serialization: all threads compute+publish DISTINCT
    /// keys of the SAME query (same vtable cache + same DepGraphInner
    /// write lock).  Fixed total work (KEYS computes); if the global
    /// dep-graph write lock is the serial point, T threads will show
    /// sub-linear speedup well before compute becomes the limit.
    #[test]
    #[ignore]
    fn bench_publish_thread_scaling() {
        let (system, provider) = setup();

        for threads in [1usize, 4, 8] {
            // Fresh slate per round: dirty every node so each get takes
            // the full compute + finish_task + publish path.
            for k in 0..KEYS {
                system.invalidate::<BQuery>(&BKey(k));
            }
            let chunk = KEYS / threads as u32;
            let mut handles = Vec::new();
            let t = Instant::now();
            for tid in 0..threads {
                let system = system;
                let provider = provider;
                let lo = tid as u32 * chunk;
                handles.push(thread::spawn(move || {
                    for k in lo..lo + chunk {
                        let _ = system.get::<BQuery>(BKey(k), provider).unwrap();
                    }
                }));
            }
            for h in handles {
                h.join().expect("bench worker panicked");
            }
            let per = t.elapsed().as_secs_f64();
            println!(
                "[bench_publish  ] T={threads}: {per:>7.3} s for {KEYS} publishes ({:>5.1} µs/pub)",
                per / KEYS as f64 * 1e6
            );
        }
        // Interpretation printed by the runner: finish_task + publish are
        // FUSED into one shard-local critical section (see
        // `finish_task_and_publish`), so the old global-write-lock serial
        // point is gone.  Sub-linear T=8/T=1 scaling now points at shard
        // contention (64 shards) or the 32 KiB alloc — cross-check with
        // `bench_publish_thin_scaling`, which removes the alloc factor.
    }

    /// P2c — allocation-pressure control: same publish scaling probe but
    /// with the 8-byte value.  If thin scales ≈2× from T=4 to T=8 while
    /// fat does not, the fat bottleneck is the 32 KiB alloc (malloc/bandwidth),
    /// not the dep-graph write lock — and merging finish_task+publish buys
    /// little.  If thin also stalls near 1.2×, the global write lock is
    /// the serial point.
    #[test]
    #[ignore]
    fn bench_publish_thin_scaling() {
        let (system, provider) = setup();

        for threads in [1usize, 4, 8] {
            for k in 0..KEYS {
                system.invalidate::<ThinQuery>(&BKey(k));
            }
            let chunk = KEYS / threads as u32;
            let mut handles = Vec::new();
            let t = Instant::now();
            for tid in 0..threads {
                let system = system;
                let provider = provider;
                let lo = tid as u32 * chunk;
                handles.push(thread::spawn(move || {
                    for k in lo..lo + chunk {
                        let _ = system.get::<ThinQuery>(BKey(k), provider).unwrap();
                    }
                }));
            }
            for h in handles {
                h.join().expect("bench worker panicked");
            }
            let per = t.elapsed().as_secs_f64();
            println!(
                "[bench_pub_thin ] T={threads}: {per:>7.3} s ({:>5.1} µs/pub)",
                per / KEYS as f64 * 1e6
            );
        }
    }
}

/// The hit path must not re-raise a node's aggregated durability: a
/// HIGH-declared query over a LOW input must observe a LOW-level change
/// of that input even when the change is discovered lazily (input
/// invalidated but not yet recomputed).  Pre-fix, every probe re-stamped
/// the DECLARED durability, so the per-level fast path compared the stamp
/// against `rev[HIGH]` — a counter LOW-level marks never bump — and
/// served the cached value without ever walking the closure that held
/// red L.
#[test]
fn dur_hit_path_preserves_aggregated_durability() {
    static L_RUNS: AtomicU64 = AtomicU64::new(0);
    static H_RUNS: AtomicU64 = AtomicU64::new(0);
    static L_INPUT: AtomicU64 = AtomicU64::new(1);
    L_RUNS.store(0, Ordering::SeqCst);
    H_RUNS.store(0, Ordering::SeqCst);
    L_INPUT.store(1, Ordering::SeqCst);
    struct LQ;
    impl Query for LQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            // LOW is the default durability.
            QueryDescriptor::new("dur_l", "LOW-durability input")
        }
        fn compute(_: &DKey, _: &dyn QueryProvider) -> FpValue {
            L_RUNS.fetch_add(1, Ordering::SeqCst);
            FpValue {
                raw: String::new(),
                semantic: L_INPUT.load(Ordering::SeqCst),
            }
        }
    }
    struct HQ;
    impl Query for HQ {
        type Key = DKey;
        type Value = FpValue;
        type Cache = DefaultCache<DKey, FpValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("dur_h", "HIGH-declared reader over a LOW input")
                .with_durability(dep_graph::Durability::HIGH)
        }
        fn compute(key: &DKey, provider: &dyn QueryProvider) -> FpValue {
            H_RUNS.fetch_add(1, Ordering::SeqCst);
            let l = query_get::<LQ>(provider, key.clone()).expect("h reads l");
            FpValue {
                raw: String::new(),
                semantic: l.semantic * 10,
            }
        }
    }
    let system = leak_system();
    let provider = leak_provider(system);
    // 1) H computes: the finish lowers H's durability to min(HIGH, LOW) = LOW.
    let v = system.get::<HQ>(DKey(0), provider).unwrap();
    assert_eq!(v.semantic, 10);
    assert_eq!(H_RUNS.load(Ordering::SeqCst), 1);
    assert_eq!(L_RUNS.load(Ordering::SeqCst), 1);
    // 2) A cached hit of H — the probe that used to re-raise H's
    //    durability to HIGH.  Must remain a serve, and must not stamp
    //    anything.
    let v = system.get::<HQ>(DKey(0), provider).unwrap();
    assert_eq!(v.semantic, 10);
    assert_eq!(H_RUNS.load(Ordering::SeqCst), 1, "step 2 must be a serve");
    // 3) LOW-level change, LAZY invalidation: only L turns red — no
    //    fan-out, because L is not recomputed here.
    L_INPUT.store(2, Ordering::SeqCst);
    system.invalidate::<LQ>(&DKey(0));
    // 4) THE VERDICT: requesting H (not L!) must transitively detect red
    //    L, re-request it, and recompute H.  Pre-fix the re-raised HIGH
    //    durability fast-pathed past red L and served the stale 10.
    let v = system.get::<HQ>(DKey(0), provider).unwrap();
    assert_eq!(
        v.semantic, 20,
        "a declared-HIGH reader over a LOW input must recompute when the LOW \
         input changes — the hit path must not re-raise the aggregated \
         durability"
    );
    assert_eq!(H_RUNS.load(Ordering::SeqCst), 2);
    assert_eq!(L_RUNS.load(Ordering::SeqCst), 2);
}
