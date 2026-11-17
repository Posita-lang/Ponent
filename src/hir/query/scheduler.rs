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

/// Millisecond default for the injectable park deadline.  Under shuttle the
/// deadline is wall-clock machinery and is compiled out (`park_deadline` does
/// not exist), so the field defaults to a dead 0 — `set_park_timeout` is
/// likewise non-shuttle only.
#[cfg(not(feature = "shuttle"))]
const PARK_TIMEOUT_MS_DEFAULT: u64 = PARK_TIMEOUT.as_millis() as u64;
#[cfg(feature = "shuttle")]
const PARK_TIMEOUT_MS_DEFAULT: u64 = 0;

/// A scheduling observation point inside the query system.
///
/// Each variant documents the exact program position it replaces from the
/// old cfg(test) hook machinery; the placement is part of the contract —
/// assertions in the defense tests depend on it.
#[derive(Clone, Copy, Debug)]
pub enum SchedulePoint {
    /// `finish_task_and_publish`: after the ENTIRE in-section work — the
    /// color judgment AND the deferred fan-out flush (`mark_dirty_batch`)
    /// — with no shard lock held, before the rev-edge flush.
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
    /// Park deadline in milliseconds (`PARK_TIMEOUT`'s injectable face,
    /// millis).  The default mirrors the old 2 s hook; tests exercising
    /// the degrade path shrink it — the path is otherwise unreachable in
    /// a test that must finish well under 2 s.
    park_timeout_ms: AtomicU64,
    /// Degrade-event counter: incremented every time a parked suspend is
    /// released by deadline expiry instead of a `release_all` (see
    /// `degraded_count`).  Observable, deliberately NOT asserted inside
    /// the scheduler — the `deadline_degrade_*` tests legitimately trip
    /// it, so the signal stays consumable.
    degraded_count: AtomicU64,
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
///
/// KNOWN INTERACTION — a predicate panic inside the finalize window
/// (`SchedulePoint::AfterPublish`) is deliberately evaluated inside the
/// finalize `catch_unwind`, so it is routed to `abort_with_poison` and
/// surfaces as key POISON rather than the raw test panic (moving the
/// suspend out would reintroduce d14-style latch stranding on unwind).
/// If a test fails with an unexpected "Poisoned" error and the predicate
/// panics, look here first — the retract-on-panic park bookkeeping is
/// handled; only the error's classification changes.
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
                park_timeout_ms: AtomicU64::new(PARK_TIMEOUT_MS_DEFAULT),
                degraded_count: AtomicU64::new(0),
            }),
        }
    }

    /// Shrink the park deadline (test-only): the degrade path
    /// (`suspend`'s deadline-expiry early release) is otherwise
    /// unreachable in a test that must finish well under 2 s.
    ///
    /// NOT to be unified with [`PARK_TIMEOUT`]: the constant is the
    /// default this injects over, and the `deadline_degrade_*` tests rely
    /// on shrinking it at runtime — see the cross-reference comment on
    /// `PARK_TIMEOUT` itself.
    #[cfg(all(test, not(feature = "shuttle")))]
    pub fn set_park_timeout(&self, d: Duration) {
        self.inner
            .park_timeout_ms
            .store(d.as_millis() as u64, Ordering::Release);
    }

    /// The effective park deadline as an `Instant` offset from now.
    #[cfg(not(feature = "shuttle"))]
    fn park_deadline(&self) -> Instant {
        Instant::now() + Duration::from_millis(self.inner.park_timeout_ms.load(Ordering::Acquire))
    }

    /// How many parks were released by DEADLINE EXPIRY (degrade) instead
    /// of a `release_all` / generation bump.  A nonzero count after a test
    /// that does not shrink the deadline is a choreography smell: the test
    /// proceeded while its target was no longer parked.  Consumable by
    /// design — the scheduler itself never asserts on it (the
    /// `deadline_degrade_*` tests legitimately drive the path); assert
    /// `== 0` from the tests that mean it.
    #[cfg(not(feature = "shuttle"))]
    pub fn degraded_count(&self) -> u64 {
        self.inner.degraded_count.load(Ordering::Acquire)
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
        // Unconditional assert, NOT debug_assert: TestScheduler is
        // test-only infrastructure (the cost of the check is irrelevant)
        // and shuttle CI runs are `--release --features shuttle`, where
        // `debug_assertions` are OFF — a debug_assert would compile this
        // guard out in exactly the configuration that needs it, resurfacing
        // the documented failure mode silently.  Same principle `check`
        // already applies to its own asserts.
        assert!(
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
    ///
    /// DEADLINE SPLIT (deliberate, do not "unify"): `wait_parked` hardcodes
    /// [`PARK_TIMEOUT`], while `suspend`'s degrade path uses the INJECTABLE
    /// knob (`set_park_timeout` / `park_deadline`).  The two answer
    /// different questions — how long a test waits for its choreography to
    /// land (test-harness constant) vs how long a parked suspend may hold
    /// the window before degrading (per-test scenario timing).  The
    /// `deadline_degrade_*` test probes with a 500 ms shrink precisely
    /// BECAUSE this wait stays at the full 2 s; unifying the knobs would
    /// break that shape.  If they are ever unified deliberately, shrink
    /// that test's probe with it.
    ///
    /// NOTE: a successful return does NOT guarantee the park SURVIVES the
    /// window — the parked suspend can still hit its deadline and degrade
    /// afterwards, silently revoking its count.  Non-degrade tests that
    /// care assert `degraded_count() == 0` after `release_all`.
    ///
    /// Requires an ARMED gate.  Calling this without a pending `arm` (the
    /// `arm → park → release_all → wait_parked` shape — the window was
    /// released but never re-armed) is a choreography error: the stale
    /// count from the released window can satisfy the floor check against
    /// an unparked target, silently passing where this scheduler's whole
    /// job is to fail loudly.  The assert converts that silent pass into a
    /// pointed failure at the call site.
    pub fn wait_parked(&self) {
        assert!(
            self.inner.armed.load(Ordering::Acquire),
            "TestScheduler::wait_parked called with no armed gate — arm a \
             window first (a released-but-not-re-armed window leaves a \
             stale park count that would silently satisfy this wait)"
        );
        let mut gates = self.inner.gates.lock();
        #[cfg(not(feature = "shuttle"))]
        let deadline = Instant::now() + PARK_TIMEOUT;
        #[cfg(feature = "shuttle")]
        // No wall clock exists under shuttle, so the deadline is a yield
        // BUDGET: each wake-without-park costs one unit.  It bounds a predicate
        // typo that keeps producing spurious wakeups; a predicate that never
        // parks and never notifies leaves the task blocked in `state_cv.wait`,
        // which shuttle's deadlock detector reports (condvars have no
        // timeout, so this budget cannot cover the no-notify case).
        // Smaller than the old 1M so a typo'd predicate surfaces in bounded
        // steps instead of burning 100 × 1M yields per `check_random`
        // iteration; `set_park_timeout` does NOT control this (it is
        // wall-clock only), so the constant is the single knob.
        let mut yield_budget = 100_000u64;
        // NOTE (test-author contract): a test that legitimately needs more
        // than 100k yields (very large HIT_ROUNDS, deep schedules) will trip
        // the assert below spuriously — that is the trade this constant
        // buys (a typo'd predicate becomes a bounded panic instead of a
        // model-check hang).  If a future stress test needs a bigger
        // budget, raise the constant here and say why in the test.
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
                // Off-by-one fix: assert BEFORE the decrement,
                // so exactly 100_000 iterations are allowed — the constant
                // above documents the budget, and the old shape only
                // permitted 99_999.
                assert!(
                    yield_budget > 0,
                    "TestScheduler::wait_parked: yield budget exhausted waiting \
                     for a park; no armed gate predicate matched a schedule point"
                );
                yield_budget -= 1;
                self.inner.state_cv.wait(&mut gates);
            }
        }
        let mark = self.inner.parked.load(Ordering::Acquire);
        self.inner.observed.store(mark, Ordering::Release);
    }

    /// Open every gate and disarm: parked threads proceed (the old
    /// "`PARK = false`" release). Subsequent suspends pass through.
    ///
    /// Choreography note: a `suspend` that matched its predicate but has
    /// not yet incremented `parked` when `release_all` bumps `generation`
    /// is released WITHOUT incrementing `parked` — `wait_parked` will
    /// therefore not observe that park.  This is why the orchestration
    /// rule is arm-BEFORE-spawn: a thread spawned after `release_all`
    /// re-arms fresh; one released mid-park was never counted and must
    /// not be waited for.
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
        let deadline = self.park_deadline();
        let mut counted = false;
        // The predicate depends only on `point` (fixed for this call) and
        // the installed gate list, which can only change when `generation`
        // bumps (already re-checked).  So the verdict is computed ONCE and
        // remembered: re-wakes re-check `generation == bound_gen` and skip
        // the catch_unwind setup entirely.
        let mut matched = false;
        loop {
            // Predicate evaluation is panic-isolated: a test-authored
            // predicate that panics on a re-check (after this thread has
            // already counted its park) must RETRACT before resuming the
            // panic — an unwinding skip past the explicit retract arms
            // below would leave a phantom count that a same-window
            // `wait_parked` returns against, i.e. the exact "prep races an
            // unparked target" failure this scheduler exists to kill.
            // (First-iteration panics are clean anyway — nothing counted
            // yet.)  Normal returns keep their existing retract/stale-park
            // semantics untouched; only the unwind path is patched.
            if self.inner.generation.load(Ordering::Acquire) != bound_gen {
                return;
            }
            if !matched {
                matched = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gates.iter().any(|gate| (gate.pred)(&point))
                })) {
                    Ok(m) => m,
                    Err(payload) => {
                        if counted {
                            self.inner.parked.fetch_sub(1, Ordering::Release);
                            self.inner.state_cv.notify_all();
                        }
                        std::panic::resume_unwind(payload);
                    }
                };
                if !matched {
                    return;
                }
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
                    // Deadline degrades to early release — the exact
                    // semantics of the old test_finish_hook spin deadline.
                    // The count MUST be revoked: leaving it would let a
                    // same-window `wait_parked` (floor = max(observed,
                    // arm_mark) from before the park) return against a
                    // target that is no longer parked — the `arm_mark`
                    // baseline covers cross-window leftovers only, this
                    // is the same-window leak.
                    if counted {
                        self.inner.parked.fetch_sub(1, Ordering::Release);
                        self.inner.state_cv.notify_all();
                        // OBSERVABILITY, not enforcement: a degrade after
                        // `wait_parked` returned means the test proceeded
                        // against an unparked target — a choreography
                        // break the park count alone no longer shows.
                        // Recorded (counted + one-line stderr) rather
                        // than asserted, because the `deadline_degrade_*`
                        // tests deliberately drive this path.
                        self.inner.degraded_count.fetch_add(1, Ordering::Release);
                        eprintln!(
                            "TestScheduler: park DEGRADED on deadline expiry \
                             (count revoked) — choreography must not rely on \
                             the target staying parked past its deadline"
                        );
                    }
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

    /// Regression (): the deadline degrade must REVOKE the park
    /// count.  Same-window leak: a thread parks (counted, `parked` lifts
    /// above `arm_mark`), the deadline expires mid-wait, and the degrade
    /// returned WITHOUT decrementing — a `wait_parked` issued after the
    /// degrade (same window, so `arm_mark` doesn't cover it) would then
    /// return against a target that is no longer parked: the exact
    /// "prep races an unparked target" race this scheduler kills.
    #[cfg(not(feature = "shuttle"))]
    #[test]
    fn deadline_degrade_revokes_the_park_count_same_window() {
        let ts = TestScheduler::new();
        // Shrink the deadline so the degrade fires quickly but AFTER the
        // park lands (the park happens within microseconds; the degrade
        // needs the deadline to expire while parked).
        ts.set_park_timeout(Duration::from_millis(120));
        ts.arm(|_| true);
        std::thread::scope(|s| {
            let h = s.spawn(|| ts.suspend(SchedulePoint::CachedHit { query: "x" }));
            // Observe the park through the RAW counter (wait_parked here
            // would be the pre-fix correct path — the bug is what a
            // LATER same-window wait sees).
            while ts.parked_count() == 0 {
                std::hint::spin_loop();
            }
            // The degrade fires while parked; joining proves the
            // suspend returned early WITHOUT release_all.
            h.join().unwrap();
            assert!(
                ts.parked_count() == 0,
                "degraded park must be revoked from the counter"
            );
            // The same-window wait_parked after the degrade must BLOCK
            // for a fresh park, not return on the leaked count.  Give it
            // the rest of a normal window: it must time out (no parker
            // will come), which under the pre-fix code never happens.
            let next = ts.clone();
            let w = s.spawn(move || next.wait_parked());
            // wait_parked's own deadline is PARK_TIMEOUT (2s) — a leaked
            // count returns instantly, a revoked one blocks.  Detect the
            // leak by timing: allow 500ms; the fixed path is still
            // blocked there.
            std::thread::sleep(Duration::from_millis(500));
            assert!(!w.is_finished(), "wait_parked returned on the leaked count");
            // Clean up: a FRESH park satisfies the (correctly blocked)
            // wait, then release it.
            let parked_once = ts.clone();
            let p = s.spawn(move || parked_once.suspend(SchedulePoint::CachedHit { query: "y" }));
            let _ = w.join();
            ts.release_all();
            p.join().unwrap();
        });
    }

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
