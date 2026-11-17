//! # Query Job — Thread-safe Query Execution with Suspension/Resumption
//!
//! Analogous to `rustc_middle::query::job`.
//! Provides the primitives for parallel query execution:
//!
//! - [`QueryJobId`] — a unique identifier for an active query job.
//! - [`QueryJob`] — represents an active query execution, with a latch
//!   that other threads can wait on.
//! - [`QueryLatch`] — a thread synchronization primitive that allows
//!   one thread to compute a query while others wait for the result.

use crate::sync::{Condvar, Mutex};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

/// A unique identifier for an active query job.
///
/// Used to match waiters to the job they're waiting on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryJobId(pub u64);

/// A counter for generating unique `QueryJobId` values.
static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

/// Per-thread batch of job ids: `(next, remaining)`.  A global RMW per
/// owner was a shared-line serialization point inside the active-shard
/// write lock under a cold-miss storm.
thread_local! {
    static JOB_ID_LOCAL: std::cell::Cell<(u64, u32)> =
        const { std::cell::Cell::new((0, 0)) };
}

const JOB_ID_BATCH: u64 = 1024;

impl QueryJobId {
    pub fn fresh() -> Self {
        JOB_ID_LOCAL.with(|c| {
            let (next, left) = c.get();
            if left == 0 {
                let base = NEXT_JOB_ID.fetch_add(JOB_ID_BATCH, Ordering::Relaxed);
                c.set((base + 1, (JOB_ID_BATCH - 1) as u32));
                QueryJobId(base)
            } else {
                c.set((next + 1, left - 1));
                QueryJobId(next)
            }
        })
    }
}

/// Identity of the currently executing unit of work: an OS thread in a
/// normal build, a shuttle TASK under feature = "shuttle".
///
/// Shuttle multiplexes every task onto ONE OS thread, so
/// std::thread::current().id() cannot distinguish two tasks that
/// interleave at a schedule point.  `QueryJob::owner` and the wait-graph
/// keys need a per-TASK identity, so they use this type and
/// [`current_exec_id`].  (The deferred-mutation partition stays keyed by
/// OS thread id on purpose — its coalescing is load-bearing; see
/// `QuerySystem::deferred`.)
///
/// ### Known limitation: the rest of the per-thread state is still
/// ### per-OS-thread
///
/// The dep-graph task stack (TASK_STACK), the per-query active partitions
/// (QueryVTable::active), SESSION and the sync_order::HELD tripwire are
/// plain thread_local! (see the "task-multiplexed TLS under shuttle" note
/// in sync_order.rs).  A cross-task same-key get under shuttle therefore
/// still looks like a same-thread cycle to validate_enter (it reads the
/// shared active partition BEFORE acquire_key_slot's owner == me arm),
/// and the task stack is shared across parked tasks.  This type removes
/// the *identity* conflation only; making the whole model per-task is the
/// larger TLS change that note describes.  Until then, do not write a
/// shuttle test that expects two tasks to get the same TRACKED key
/// concurrently (d5 sidesteps it with eval_always, which skips active
/// keys entirely).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecId(
    #[cfg(not(feature = "shuttle"))] std::thread::ThreadId,
    #[cfg(feature = "shuttle")] shuttle::thread::ThreadId,
);

/// The current task's [ExecId] -- see its doc for why this is not
/// std::thread::current().id() under shuttle.
#[inline]
pub fn current_exec_id() -> ExecId {
    #[cfg(not(feature = "shuttle"))]
    {
        ExecId(std::thread::current().id())
    }
    #[cfg(feature = "shuttle")]
    {
        ExecId(shuttle::thread::current().id())
    }
}

/// Represents an active query job.
///
/// When a query starts executing on one task, a `QueryJob` is created
/// and recorded in the `ActiveKeyStatus`.  If another task requests the
/// same key, it will find the `QueryJob` and call `wait_on()` to block
/// until the result is ready.
#[derive(Debug, Clone)]
pub struct QueryJob {
    pub(crate) id: QueryJobId,
    /// The task computing this job.  Used as a double-safety guard for
    /// same-task cycle detection and as the anchor of the cross-task wait
    /// graph.
    ///
    /// Read-only by construction: the owner anchors the whole
    /// cross-thread cycle-detection protocol (`WaitEdge.owner`,
    /// `clear_waiters_of_me`, `poison_active`), so it is set once in
    /// [`QueryJob::new`] and never mutated in production — a stray
    /// assignment would make the walk follow a dead edge or miss a live
    /// one. The field is `pub(crate)` so the crate can read it while external
    /// code cannot reassign it; the raced-cycle test that needs a forged
    /// owner lives in this crate and can still set it.
    pub(crate) owner: ExecId,
    /// The latch that other threads can wait on.
    pub(crate) latch: QueryLatch,
}

impl QueryJob {
    pub fn new() -> Self {
        QueryJob {
            id: QueryJobId::fresh(),
            owner: current_exec_id(),
            latch: QueryLatch::new(),
        }
    }
    // NOTE: there is deliberately no `signal_complete` convenience — the
    // completion order in `get_shared` must resolve the latch BEFORE the
    // active-key entry is removed, and the removed entry's `job.id`
    // is still needed afterwards for `clear_waiters_of_me`, so callers
    // work on the latch and the entry separately.
}

// ── QueryLatch ───────────────────────────────────────────────────

/// Latch state machine — the single canonical state word, merged from
/// the old per-waiter `completed: Mutex<bool>` + `poisoned: Mutex<bool>`
/// pair (two locks → one byte).
const LATCH_WAITING: u8 = 0;
const LATCH_COMPLETED: u8 = 1;
const LATCH_POISONED: u8 = 2;

/// Bounded spin before parking (Perf Book §13.2).
///
/// Most cross-thread waits are short: by the time a waiter reaches
/// `wait_on`, the owner is usually past `compute` and only publishing
/// (fingerprint + cache insert + wake) — microseconds.  Spinning for a
/// bounded number of iterations avoids the futex park/wake syscall pair
/// for exactly those short waits; only genuinely long computations fall
/// through to the condvar.  128, not 64: one spin iteration is a few ns,
/// so the whole budget stays well under a microsecond.
const LATCH_SPIN_LIMIT: u32 = 128;
/// Iterations of a pure `spin_loop` hint before yielding.  On a
/// single-core/oversubscribed CI runner a pure spin burns the only
/// runnable thread's slice while the owner cannot progress.
const LATCH_HOT_SPINS: u32 = 32;
// Cheap invariant: the spin budget must be non-zero.  A zero value would
// silently disable the bounded-spin path entirely and force EVERY wait
// (even the sub-microsecond ones this budget exists for) through the
// condvar park/wake syscall pair; a future edit that "simplifies" the
// constant to 0 should trip here, not in a profiler.
const _: () = assert!(LATCH_SPIN_LIMIT > 0);

/// Park state of a [`QueryLatch`] — ONLY the count of threads currently
/// parked.  The mutex guarding it (`LatchInner::park`) and the condvar
/// the waiters park on (`LatchInner::cv`) both live in `LatchInner`,
/// next to the latch state they synchronize.
#[derive(Debug, Default)]
struct LatchPark;

/// Shared state of a [`QueryLatch`].  ONE heap allocation per latch
/// (the `Arc`) and ZERO per waiter — there is no waiter object at all;
/// parkers share the single condvar and are woken with one
/// `notify_all`.
#[derive(Debug)]
struct LatchInner {
    /// The canonical state: `LATCH_WAITING` while the query runs, then
    /// `LATCH_COMPLETED` or `LATCH_POISONED`, written exactly once.
    state: AtomicU8,
    /// Threads currently parked on `cv`.  Kept OUTSIDE the park mutex so
    /// the RAII decrement guard needs no raw pointer and does not borrow
    /// `park` across `cv.wait(&mut park)`.
    waiters: AtomicU32,
    park: std::sync::OnceLock<(Mutex<LatchPark>, Condvar)>,
}

/// A latch that allows one thread to compute a query while others wait.
///
/// Analogous to Rust's `QueryLatch` in `job.rs`.
/// Waiters spin briefly on the state word, then park on a single shared
/// condvar; the computing thread flips the state word and wakes all
/// parked waiters with ONE notification.  (The old design allocated a
/// `Condvar` + two `Mutex<bool>`s inside a fresh `Arc` per waiter and
/// notified each waiter individually — three synchronisation objects
/// and one allocation per cross-thread wait.)
///
/// # Usage
///
/// ```ignore
/// let latch = QueryLatch::new();
/// // Thread A (compute):
/// let result = compute(key);
/// latch.set();  // resume all waiters
///
/// // Thread B (wait):
/// latch.wait_on();  // blocks until thread A calls set()
/// let result = get_from_cache(key);
/// ```
#[derive(Debug, Clone)]
pub struct QueryLatch {
    inner: Arc<LatchInner>,
}

impl QueryLatch {
    pub fn new() -> Self {
        QueryLatch {
            inner: Arc::new(LatchInner {
                state: AtomicU8::new(LATCH_WAITING),
                waiters: AtomicU32::new(0),
                park: std::sync::OnceLock::new(),
            }),
        }
    }

    /// Block the current thread until the query completes.
    ///
    /// Returns `Ok(())` when the query has completed successfully.
    /// Returns `Err` if the query was poisoned (panicked).
    pub fn wait_on(&self) -> Result<(), ()> {
        // Fast path + bounded spin: one atomic load per iteration — no
        // lock, no allocation, no syscall.  `spin_loop` hints the core
        // to save power instead of burning full tilt.
        for i in 0..LATCH_SPIN_LIMIT {
            match self.inner.state.load(Ordering::Acquire) {
                LATCH_COMPLETED => return Ok(()),
                LATCH_POISONED => return Err(()),
                _ => {
                    if i < LATCH_HOT_SPINS {
                        std::hint::spin_loop();
                    } else {
                        // Yield after the hot sub-budget so a 1-core host
                        // does not starve the owner.  Under shuttle the
                        // scheduler models parking, so keep hinting there.
                        #[cfg(not(feature = "shuttle"))]
                        std::thread::yield_now();
                        #[cfg(feature = "shuttle")]
                        std::hint::spin_loop();
                    }
                }
            }
        }

        // Slow path: park.  The mutex makes "check state, then sleep"
        // atomic with respect to `set`/`poison`, so no wakeup can be
        // lost: the resolver either flips the state word before our
        // check (we see it and never park) or after (it then counts our
        // `waiters` increment and notifies the condvar).
        //
        // Tripwire registration BEFORE the lock (single rule: register,
        // then acquire) — matches `resolve`'s shape.
        let _tw = crate::hir::query::sync_order::Acquired::acquire(
            crate::hir::query::sync_order::Fam::LatchPark,
            "job.latch.wait_on",
        );
        // The waiter count lives in its own atomic, so this RAII guard needs
        // no raw pointer and does not borrow `park` across `cv.wait(&mut park)`
        //.  A leaked increment would make every future
        // `set`/`poison` take the notify path forever; the guard decrements on
        // every exit path, including unwind.
        // The waiter count is NOT a free-floating hint.  Consider a waiter that increments,
        // takes the park lock and observes LATCH_WAITING BEFORE the
        // resolver's CAS: if the resolver's subsequent `waiters` load does
        // not observe that increment, it skips `notify_all` and the waiter
        // parks with no wakeup — a lost-wakeup deadlock.  SeqCst is what
        // rules that interleaving out (it makes the increment and the
        // resolver's load part of one total order with the state CAS);
        // `test_query_latch_set_poison_race` hangs under Relaxed.  This
        // count is only touched by parked waiters, so the fence pair is
        // off every hot path anyway.
        struct WaitersGuard<'a>(&'a std::sync::atomic::AtomicU32);
        impl Drop for WaitersGuard<'_> {
            #[inline]
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        self.inner.waiters.fetch_add(1, Ordering::SeqCst);
        let _guard = WaitersGuard(&self.inner.waiters);
        // #3: the park primitives are created on the first cross-thread wait,
        // not on every cold key.  If `resolve` runs before this `get_or_init`,
        // it sees `None` and does not notify — safe, because this thread then
        // checks `state` under the just-created mutex and sees the transition
        // instead of parking.
        let (park_mu, cv) = self
            .inner
            .park
            .get_or_init(|| (Mutex::new(LatchPark::default()), Condvar::new()));
        let mut park = park_mu.lock();
        loop {
            match self.inner.state.load(Ordering::Acquire) {
                LATCH_COMPLETED => return Ok(()),
                LATCH_POISONED => return Err(()),
                _ => cv.wait(&mut park),
            }
        }
    }

    /// Signal that the query is complete and resume all waiting threads.
    pub fn set(&self) {
        self.resolve(LATCH_COMPLETED);
    }

    /// Poison the latch: wake every blocked waiter and make `wait_on`
    /// return `Err`.  Called when the computing thread panicked,
    /// so blocked threads are woken with an error instead of hanging.
    pub fn poison(&self) {
        self.resolve(LATCH_POISONED);
    }

    /// Transition the latch to `to` (Completed or Poisoned) and wake all
    /// parked waiters with one `notify_all` — never one syscall per
    /// waiter.  The latch is one-shot: the first transition wins.
    fn resolve(&self, to: u8) {
        // AcqRel: a waiter that loads the new state (Acquire) also sees
        // everything published before this point (e.g. the value
        // inserted into the query cache before `signal_complete`).
        if self
            .inner
            .state
            .compare_exchange(LATCH_WAITING, to, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // The latch was already resolved (set racing poison, or a
            // double signal) — whoever won did the waking.
            return;
        }
        // Acquire the park mutex before inspecting `waiters` and notifying.  This
        // is what closes BOTH lost-wakeup interleavings:
        //
        //  (a) racy skip: the waiter's `fetch_add` and the resolver's load are
        //      now serialized by the mutex, so the resolver either observes
        //      the increment (and notifies) or the waiter has not yet taken
        //      the lock — and once it does, its state check sees `to` and it
        //      never parks.  (SeqCst alone does not close this.)
        //  (b) notify-before-register: a waiter that has locked `park` and
        //      checked `WAITING` registers atomically with `cv.wait`'s
        //      unlock; the resolver cannot slip a notify into the gap between
        //      the check and the registration because it must hold the same
        //      mutex.
        //
        // Notifying while holding the mutex means woken waiters briefly
        // re-block on it (`hurry-up-and-wait`), but that is one lock/unlock
        let _tw = crate::hir::query::sync_order::Acquired::acquire(
            crate::hir::query::sync_order::Fam::LatchPark,
            "job.latch.resolve",
        );
        // #3: only a waiter that has already initialized the park pair can be
        // parked, so `None` means there is nothing to wake (the waiter's own
        // state check under the mutex will observe `to`).
        if let Some((park_mu, cv)) = self.inner.park.get() {
            let park = park_mu.lock();
            if self.inner.waiters.load(Ordering::SeqCst) > 0 {
                cv.notify_all();
            }
            drop(park);
        }
    }

    /// Check whether the query has completed.
    pub fn is_complete(&self) -> bool {
        self.inner.state.load(Ordering::Acquire) != LATCH_WAITING
    }
}

impl Default for QueryLatch {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_query_latch_basic() {
        let latch = QueryLatch::new();
        assert!(!latch.is_complete());

        let latch_clone = latch.clone();
        let handle = thread::spawn(move || {
            // Simulate computation.
            thread::sleep(std::time::Duration::from_millis(10));
            latch_clone.set();
        });

        // Wait for the computation to complete.
        latch.wait_on().unwrap();
        assert!(latch.is_complete());
        handle.join().unwrap();
    }

    #[test]
    fn test_query_latch_multiple_waiters() {
        let latch = QueryLatch::new();
        let latch_clone = latch.clone();

        let handle = thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(10));
            latch_clone.set();
        });

        // Spawn multiple threads waiting on the same latch.
        let latch2 = latch.clone();
        let latch3 = latch.clone();
        let h1 = thread::spawn(move || latch2.wait_on().unwrap());
        let h2 = thread::spawn(move || latch3.wait_on().unwrap());

        latch.wait_on().unwrap();
        h1.join().unwrap();
        h2.join().unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn test_query_job_id_unique() {
        let id1 = QueryJobId::fresh();
        let id2 = QueryJobId::fresh();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_signal_complete_before_wait() {
        let latch = QueryLatch::new();
        latch.set(); // Complete before anyone waits.
        assert!(latch.is_complete());
        // Wait should return immediately.
        latch.wait_on().unwrap();
    }

    /// The panic path: `poison` wakes blocked waiters with `Err`, and the
    /// poisoned state is terminal for later waiters too.
    #[test]
    fn test_query_latch_poison_wakes_waiters() {
        let latch = QueryLatch::new();
        let l2 = latch.clone();
        let handle = thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(10));
            l2.poison();
        });
        assert_eq!(latch.wait_on(), Err(()));
        handle.join().unwrap();
        // A waiter arriving after the poison also sees Err.
        assert_eq!(latch.wait_on(), Err(()));
        assert!(latch.is_complete());
    }

    /// Several waiters on one latch resolve with a single `set` — the
    /// shared-condvar / single-notify_all shape.
    #[test]
    fn test_query_latch_many_waiters_one_set() {
        let latch = QueryLatch::new();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let l = latch.clone();
                thread::spawn(move || l.wait_on().unwrap())
            })
            .collect();
        // Give the waiters time to exhaust their spin budget and park, so
        // this exercises the condvar path, not just the spin path.
        thread::sleep(std::time::Duration::from_millis(5));
        latch.set();
        for h in handles {
            h.join().unwrap();
        }
        assert!(latch.is_complete());
    }

    /// `set` racing `poison` must never wedge a waiter: exactly one
    /// transition wins and every waiter observes it.
    #[test]
    fn test_query_latch_set_poison_race() {
        for _ in 0..64 {
            let latch = QueryLatch::new();
            let l2 = latch.clone();
            let l3 = latch.clone();
            let waiter = thread::spawn(move || l2.wait_on());
            thread::sleep(std::time::Duration::from_millis(1));
            let h1 = thread::spawn(move || l3.set());
            latch.poison();
            h1.join().unwrap();
            // Either outcome is acceptable; it must just resolve.
            let _ = waiter.join().unwrap();
        }
    }
}
