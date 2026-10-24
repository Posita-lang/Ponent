//! Scheduler abstraction for the query system: typed schedule points, a
//! zero-cost production no-op, and a test gate that hard-constructs windows.
//!
//! Production code calls [`Scheduler::suspend`] at four audit-fixed points
//! (see the `SchedulePoint` variant docs for the exact placement contract).
//! `QuerySystem<NoOpScheduler>` monomorphizes each call to an empty inlined
//! body — release-equivalent to the old `cfg(test)` hooks, with zero
//! instructions, branches, or atomic loads.
//!
//! Tests use [`TestScheduler`]: a suspend whose point matches an armed gate
//! parks until [`TestScheduler::release_all`]. This converts the old
//! process-global TARGET/PARK/ARRIVED statics into per-test-instance state
//! (killing the cross-talk root cause) and replaces "race the code to hit
//! the window" with "open the window by gate".

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(not(feature = "shuttle"))]
use std::time::{Duration, Instant};

use crate::sync::{Condvar, Mutex};

/// Mirror of the old test-hook deadline (`test_finish_hook` spun on
/// `Instant::now() + 2s`): a parked suspend degrades to early release after
/// this duration instead of hanging CI, and a test waiting for a park that
/// never happens panics instead of spinning forever.
///
/// Under `feature = "shuttle"` this deadline is compiled out: shuttle's
/// deadlock detection replaces it (wall-clock time conflicts with the
/// deterministic scheduler — see spec §5).
#[cfg(not(feature = "shuttle"))]
const PARK_TIMEOUT: Duration = Duration::from_secs(2);

/// A scheduling observation point inside the query system.
///
/// Each variant documents the exact program position it replaces from the
/// old cfg(test) hook machinery; the placement is part of the contract —
/// assertions in the defense tests depend on it.
#[derive(Clone, Copy, Debug)]
pub enum SchedulePoint {
    /// `finish_task_and_publish`: after the in-section color judgment, with
    /// no shard lock held, before the rev-edge flush.
    /// (Replaces the D23/D24/D26 static triples.)
    FinishWindow { node: u32 },
    /// `verify_green_transitive_with_dirty`: after the fast-path verdict,
    /// before the seqlock re-sample. (Replaces the VERIFY_* statics / d27.)
    VerifyWindow { node: u32 },
    /// `get_shared` cache branch: after the probe captures the cached `Arc`,
    /// before `try_serve_verified`. (Replaces D17/D15; query identity is a
    /// name string instead of a `name.as_ptr()` pointer key.)
    CachedHit { query: &'static str },
    /// `get_shared` finalize closure: after `finish_task_and_publish` and
    /// the `dep_graph.read`, before the latch signal / vtable leave.
    /// (Replaces D21.)
    AfterPublish { query: &'static str, node: u32 },
}

/// Schedule-point sink. The production implementation must be a no-op.
pub trait Scheduler: Send + Sync + 'static {
    /// Observe `point`; implementation may block (test gate) but must not
    /// change query semantics.
    fn suspend(&self, point: SchedulePoint);
}

/// Zero-cost production scheduler: an empty, always-inlined suspend.
///
/// The `SchedulePoint` payload is dead on arrival — monomorphization plus
/// DCE removes the call entirely (prerequisites: this type stays a ZST; the
/// method keeps `#[inline(always)]`).
#[derive(Clone, Copy, Debug, Default)]
pub struct NoOpScheduler;

impl Scheduler for NoOpScheduler {
    #[inline(always)]
    fn suspend(&self, _point: SchedulePoint) {}
}

/// One armed window gate: `pred` selects which schedule points park.
struct Gate {
    pred: Box<dyn Fn(&SchedulePoint) -> bool + Send + Sync>,
}

/// Shared gate state. `gates` is the lock for both the gate list and the
/// condvar hand-off; `armed`/`parked` are lock-free for the fast path and
/// for arrival snapshots.
struct PlanInner {
    /// Fast path: one Acquire load skips all locking when no gate is armed.
    armed: AtomicBool,
    /// Monotonic count of suspend calls that entered a park (guarded by
    /// `gates` on write).
    parked: AtomicU64,
    /// Park count already consumed by `wait_parked` (guarded by `gates`).
    /// A wait returns once `parked` passes this mark, so a park that lands
    /// BEFORE the wait still counts, and each subsequent park is observed
    /// exactly once.
    observed: AtomicU64,
    /// `parked` count as of the most recent `arm`/`release_then_arm`
    /// (guarded by `gates`).  `wait_parked` waits for
    /// `parked > max(observed, arm_mark)`, so a park left over from an
    /// earlier window — released without an intervening `wait_parked`, or
    /// ended by the `PARK_TIMEOUT` degrade — can never satisfy a later
    /// window's wait: the wait blocks for a FRESH park on the new gate.
    arm_mark: AtomicU64,
    /// Bumped under `gates` whenever the gate list is replaced wholesale
    /// (`release_all`, `release_then_arm`). A suspend binds to the
    /// generation it entered under: a bump makes every parked suspend
    /// return, while a gate installed in the same locked step captures the
    /// NEXT suspend — no slip window between "let the old park go" and
    /// "catch the retry".
    generation: AtomicU64,
    gates: Mutex<Vec<Gate>>,
    /// Signalled by `release_all` / `arm`; waiters re-check the gate list.
    open_cv: Condvar,
    /// Signalled whenever a park is entered; `wait_parked` re-checks the
    /// counter.
    state_cv: Condvar,
}

/// Per-test window gate scheduler (parking_lot backend over plain
/// threads; under `feature = "shuttle"` the same shape runs on shuttle
/// primitives — the `#[cfg]` branches below swap the wall-clock deadline
/// machinery for shuttle's deterministic scheduler, which replaces
/// timeouts entirely; see `PARK_TIMEOUT`).
///
/// Usage choreography (mirrors the old TARGET/PARK/ARRIVED sequence):
/// 1. `TestScheduler::new()` + `QuerySystem::with_scheduler(ts.clone())`
/// 2. `ts.arm(pred)` — a matching `suspend` parks the thread
/// 3. `ts.wait_parked()` — the old "wait for ARRIVED" observation
/// 4. test preparation while the target thread is parked
/// 5. `ts.release_all()` — the old "`PARK = false`" release
///
/// Double-window choreography (the d27 retry pattern): between two
/// `wait_parked`s use [`release_then_arm`](Self::release_then_arm), which
/// releases the current park and installs the next gate atomically so the
/// retried suspend cannot slip through ungated.
#[derive(Clone)]
pub struct TestScheduler {
    inner: Arc<PlanInner>,
}

impl Default for TestScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl TestScheduler {
    /// Create a fresh, disarmed gate plan. Every test gets its own
    /// instance — no shared statics, no cross-test interference.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(PlanInner {
                armed: AtomicBool::new(false),
                parked: AtomicU64::new(0),
                observed: AtomicU64::new(0),
                arm_mark: AtomicU64::new(0),
                generation: AtomicU64::new(0),
                gates: Mutex::new(Vec::new()),
                open_cv: Condvar::new(),
                state_cv: Condvar::new(),
            }),
        }
    }

    /// Arm a gate: suspends whose `point` matches `pred` park until
    /// [`release_all`](Self::release_all). One `arm` per window; sequential
    /// windows arm again after a release.
    pub fn arm<F>(&self, pred: F)
    where
        F: Fn(&SchedulePoint) -> bool + Send + Sync + 'static,
    {
        let mut gates = self.inner.gates.lock();
        // "one arm per window" is not enforced, and a
        // double-arm silently re-baselines `arm_mark` while window 1's gate
        // stays installed — the choreography then parks on the WRONG gate
        // and the test fails with a timeout that points at the predicate.
        // A debug assert catches the choreography error at its source.
        debug_assert!(
            gates.is_empty(),
            "TestScheduler::arm called twice without an intervening \
             wait_parked: a second arm re-baselines `arm_mark` while the \
             first gate stays installed, so window 1's park is matched \
             against window 2's predicate.  Arm once per window."
        );
        gates.push(Gate {
            pred: Box::new(pred),
        });
        // Baseline the wait at the most recent arm: any park counted
        // BEFORE this instant belongs to an earlier window and must not
        // satisfy the next wait_parked (see `arm_mark`'s doc).
        self.inner
            .arm_mark
            .store(self.inner.parked.load(Ordering::Acquire), Ordering::Release);
        self.inner.armed.store(true, Ordering::Release);
        self.inner.open_cv.notify_all();
    }

    /// Block until a suspend parks on an armed gate (the old ARRIVED
    /// observation). Panics after [`PARK_TIMEOUT`] — mirroring the old
    /// test-side guard timeouts — if no park arrives (usually a predicate
    /// that matches no schedule point).
    pub fn wait_parked(&self) {
        let mut gates = self.inner.gates.lock();
        #[cfg(not(feature = "shuttle"))]
        let deadline = Instant::now() + PARK_TIMEOUT;
        loop {
            // A park left over from an earlier window (released without an
            // intervening wait_parked, or ended by the PARK_TIMEOUT degrade)
            // sits at or below `arm_mark`; only a FRESH park on the current
            // gate lifts `parked` above max(observed, arm_mark).
            let floor = self
                .inner
                .observed
                .load(Ordering::Acquire)
                .max(self.inner.arm_mark.load(Ordering::Acquire));
            if self.inner.parked.load(Ordering::Acquire) > floor {
                break;
            }
            #[cfg(not(feature = "shuttle"))]
            {
                let now = Instant::now();
                assert!(
                    now < deadline,
                    "TestScheduler::wait_parked: timed out waiting for a park; \
                     no armed gate predicate matched a schedule point"
                );
                self.inner.state_cv.wait_for(&mut gates, deadline - now);
            }
            #[cfg(feature = "shuttle")]
            {
                self.inner.state_cv.wait(&mut gates);
            }
        }
        let mark = self.inner.parked.load(Ordering::Acquire);
        self.inner.observed.store(mark, Ordering::Release);
    }

    /// Open every gate and disarm: parked threads proceed (the old
    /// "`PARK = false`" release). Subsequent suspends pass through.
    pub fn release_all(&self) {
        let mut gates = self.inner.gates.lock();
        gates.clear();
        self.inner.generation.fetch_add(1, Ordering::Release);
        self.inner.armed.store(false, Ordering::Release);
        self.inner.open_cv.notify_all();
    }

    /// Test-only probe of the raw park counter.  The stale-park scenario
    /// cannot be built through `wait_parked` — that is exactly the API
    /// whose watermark the scenario must bypass.
    #[cfg(test)]
    pub fn parked_count(&self) -> u64 {
        self.inner.parked.load(Ordering::Acquire)
    }

    /// Release the current parks and install `pred` as the sole next gate
    /// in one locked step: a suspend parked on the old generation returns,
    /// while any suspend it retries (or any new one) binds to the new
    /// generation and parks on `pred`. The single lock acquisition is what
    /// makes the hand-off race-free — a plain `release_all` + `arm` pair
    /// would leave a window where the retry slips through ungated.
    pub fn release_then_arm<F>(&self, pred: F)
    where
        F: Fn(&SchedulePoint) -> bool + Send + Sync + 'static,
    {
        let mut gates = self.inner.gates.lock();
        gates.clear();
        self.inner.generation.fetch_add(1, Ordering::Release);
        gates.push(Gate {
            pred: Box::new(pred),
        });
        // Same baseline as `arm` — see `arm_mark`'s doc.
        self.inner
            .arm_mark
            .store(self.inner.parked.load(Ordering::Acquire), Ordering::Release);
        self.inner.armed.store(true, Ordering::Release);
        self.inner.open_cv.notify_all();
    }
}

impl Scheduler for TestScheduler {
    fn suspend(&self, point: SchedulePoint) {
        // Fast path: nothing armed → no lock, matching NoOp cost closely.
        //
        // Arm/suspend race: the unlocked `armed` load can read a stale
        // `false` while a concurrent `arm`/`release_then_arm` is already
        // installing its gate, and that suspend would then pass through
        // UNGATED — `wait_parked` would never see a park and the test
        // either trips its timeout (non-shuttle) or hangs (shuttle, whose
        // `wait_parked` has no deadline).  Re-check under `gates` — the
        // very lock `arm` holds across its `armed` store — so the
        // verdict is lock-ordered: still `false` means this pass-through
        // happens-before the gate's installation and is final; `true`
        // means the store already landed and we fall through to the
        // gated path WITH the lock held.  This is safe under the
        // documented arm-before-spawn choreography (a spawn issued after
        // `arm` returns observes the Release store).
        //
        // What it does NOT cover: a task that is already parked (or
        // already spawned and suspended) BEFORE `arm` installs its gate.
        // Such a suspend has passed this fast path — or the gated loop —
        // entirely without binding to the new generation, and a gate
        // installed afterwards never catches it: `wait_parked` then trips
        // its deadline (non-shuttle) or hangs (shuttle, which has no
        // deadline).  So the choreography is arm-BEFORE-spawn, and a
        // test that parks first must arm after it observes the park.
        let mut gates = if !self.inner.armed.load(Ordering::Acquire) {
            let locked = self.inner.gates.lock();
            if !self.inner.armed.load(Ordering::Acquire) {
                return;
            }
            locked
        } else {
            self.inner.gates.lock()
        };
        let bound_gen = self.inner.generation.load(Ordering::Acquire);
        #[cfg(not(feature = "shuttle"))]
        let deadline = Instant::now() + PARK_TIMEOUT;
        let mut counted = false;
        loop {
            let matching = self.inner.generation.load(Ordering::Acquire) == bound_gen
                && gates.iter().any(|gate| (gate.pred)(&point));
            if !matching {
                return;
            }
            #[cfg(not(feature = "shuttle"))]
            {
                // Degrade BEFORE counting: if the window's deadline has
                // already expired when we arrive, we return below without
                // ever parking.  Counting first would lift `parked` above
                // `arm_mark` anyway and let `wait_parked` proceed against
                // a thread that is no longer parked — the exact "prep
                // races an unparked target" failure this scheduler
                // exists to kill.
                if !counted && Instant::now() >= deadline {
                    return;
                }
            }
            if !counted {
                self.inner.parked.fetch_add(1, Ordering::Release);
                self.inner.state_cv.notify_all();
                counted = true;
            }
            #[cfg(not(feature = "shuttle"))]
            {
                let now = Instant::now();
                if now >= deadline {
                    // Deadline degrades to early release — the exact semantics
                    // of the old test_finish_hook spin deadline.
                    return;
                }
                self.inner.open_cv.wait_for(&mut gates, deadline - now);
            }
            #[cfg(feature = "shuttle")]
            {
                self.inner.open_cv.wait(&mut gates);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression (Finding 2): a park released WITHOUT an intervening
    /// `wait_parked` must not satisfy a later window's wait.  Without the
    /// `arm_mark` baseline, window 2's `wait_parked` returns immediately on
    /// the stale count (parked=1 > observed=0) and the test races its
    /// preparation against an UNPARKED target — the failure mode the
    /// TestScheduler exists to eliminate.
    /// Not usable under `feature = "shuttle"`: `PlanInner`'s
    /// `Mutex`/`Condvar` are shuttle's under that feature and demand an
    /// `ExecutionState`, which neither libtest's thread nor a
    /// `std::thread::scope`ed OS thread has — the twin below is the port.
    #[cfg(not(feature = "shuttle"))]
    #[test]
    fn wait_parked_ignores_stale_park_across_windows() {
        let ts = TestScheduler::new();
        // Window 1: arm, let a thread park, release WITHOUT observing.
        ts.arm(|_| true);
        std::thread::scope(|s| {
            let h = s.spawn(|| ts.suspend(SchedulePoint::CachedHit { query: "x" }));
            while ts.parked_count() == 0 {
                std::hint::spin_loop();
            }
            ts.release_all(); // no wait_parked — park unconsumed
            h.join().unwrap();
        });
        assert_eq!(ts.parked_count(), 1, "precondition: the stale park exists");
        // Window 2: wait_parked must block for a FRESH park.  Without the
        // fix it returns on the stale count; with the fix it returns only
        // once parked > arm_mark (= 1).
        ts.arm(|_| true);
        std::thread::scope(|s| {
            let h = s.spawn(|| ts.suspend(SchedulePoint::CachedHit { query: "x" }));
            ts.wait_parked();
            assert!(
                ts.parked_count() >= 2,
                "wait_parked returned on the stale park"
            );
            ts.release_all();
            h.join().unwrap();
        });
    }

    /// Shuttle-variant of the stale-park regression — NOT a copy of the
    /// OS-thread twin above.  Under `feature = "shuttle"` `PlanInner`'s
    /// `Mutex`/`Condvar` come from `crate::sync`, i.e. shuttle's, and they
    /// require an `ExecutionState` installed by the `shuttle::check`
    /// executor.  Neither libtest's test thread nor a `std::thread::spawn`
    /// ed thread has one (`NotSet`), and std threads are invisible to the
    /// deterministic scheduler besides — so the orchestration moves to
    /// `shuttle::thread::spawn`.  The `spin_loop` courtesy-wait becomes
    /// `yield_now`: `parked` is a plain `AtomicU64` that shuttle does not
    /// instrument, so a load is not a scheduling point and spinning would
    /// never hand the parker a turn.
    ///
    /// The deadline / degrade path is compiled out under shuttle
    /// (`state_cv.wait` has no timeout), so the only way `wait_parked`
    /// returns is a genuine fresh park — which is precisely window 2's
    /// claim, now deterministically replayable.
    #[cfg(feature = "shuttle")]
    #[test]
    fn wait_parked_ignores_stale_park_across_windows_shuttle() {
        shuttle::check_random(
            || {
                let ts = TestScheduler::new();
                // Window 1: park, release WITHOUT an intervening wait_parked.
                ts.arm(|_| true);
                let t = ts.clone();
                let h = shuttle::thread::spawn(move || {
                    t.suspend(SchedulePoint::CachedHit { query: "x" });
                });
                while ts.parked_count() == 0 {
                    shuttle::thread::yield_now();
                }
                ts.release_all();
                h.join().expect("window-1 parker must join clean");
                assert_eq!(ts.parked_count(), 1, "the stale park exists");
                // Window 2: `arm` re-bases the watermark AT the stale park
                // (arm_mark = 1), so `wait_parked` must block for a FRESH
                // park.  Returning at all proves the new park was seen
                // above max(observed, arm_mark); returning on the stale
                // count is the regression this twin guards.
                ts.arm(|_| true);
                let t2 = ts.clone();
                let h2 = shuttle::thread::spawn(move || {
                    t2.suspend(SchedulePoint::CachedHit { query: "x" });
                });
                ts.wait_parked();
                assert!(ts.parked_count() >= 2);
                ts.release_all();
                h2.join().expect("window-2 parker must join clean");
            },
            100,
        );
    }
}
