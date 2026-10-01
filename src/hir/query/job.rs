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

use parking_lot::{Condvar, Mutex};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// A unique identifier for an active query job.
///
/// Used to match waiters to the job they're waiting on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryJobId(pub u64);

/// A counter for generating unique `QueryJobId` values.
static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

impl QueryJobId {
    pub fn fresh() -> Self {
        QueryJobId(NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Represents an active query job.
///
/// When a query starts executing on one thread, a `QueryJob` is created
/// and recorded in the `ActiveKeyStatus`.  If another thread requests the
/// same key, it will find the `QueryJob` and call `wait_on()` to block
/// until the result is ready.
#[derive(Debug, Clone)]
pub struct QueryJob {
    pub id: QueryJobId,
    /// The thread computing this job.  Used as a double-safety guard for
    /// same-thread cycle detection and as the anchor of the
    /// cross-thread wait graph.
    pub owner: std::thread::ThreadId,
    /// The latch that other threads can wait on.
    pub latch: QueryLatch,
}

impl QueryJob {
    pub fn new() -> Self {
        QueryJob {
            id: QueryJobId::fresh(),
            owner: std::thread::current().id(),
            latch: QueryLatch::new(),
        }
    }
    // NOTE: there is deliberately no `signal_complete` convenience — the
    // completion order in `get_shared` must resolve the latch BEFORE the
    // active-key entry is removed (F1), and the removed entry's `job.id`
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

/// Park state of a [`QueryLatch`] — ONLY the count of threads currently
/// parked.  The mutex guarding it (`LatchInner::park`) and the condvar
/// the waiters park on (`LatchInner::cv`) both live in `LatchInner`,
/// next to the latch state they synchronize.
#[derive(Debug, Default)]
struct LatchPark {
    /// Threads currently parked on `cv`.  Read by `set`/`poison` so the
    /// futex wake syscall is skipped entirely when every waiter was
    /// still in its bounded spin.
    waiters: u32,
}

/// Shared state of a [`QueryLatch`].  ONE heap allocation per latch
/// (the `Arc`) and ZERO per waiter — there is no waiter object at all;
/// parkers share the single condvar and are woken with one
/// `notify_all`.
#[derive(Debug)]
struct LatchInner {
    /// The canonical state: `LATCH_WAITING` while the query runs, then
    /// `LATCH_COMPLETED` or `LATCH_POISONED`, written exactly once.
    state: AtomicU8,
    park: Mutex<LatchPark>,
    cv: Condvar,
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
                park: Mutex::new(LatchPark::default()),
                cv: Condvar::new(),
            }),
        }
    }

    /// Block the current thread until the query completes.
    ///
    /// Returns `Ok(())` when the query has completed successfully.
    /// Returns `Err(())` if the query was poisoned (panicked).
    pub fn wait_on(&self) -> Result<(), ()> {
        // Fast path + bounded spin: one atomic load per iteration — no
        // lock, no allocation, no syscall.  `spin_loop` hints the core
        // to save power instead of burning full tilt.
        for _ in 0..LATCH_SPIN_LIMIT {
            match self.inner.state.load(Ordering::Acquire) {
                LATCH_COMPLETED => return Ok(()),
                LATCH_POISONED => return Err(()),
                _ => std::hint::spin_loop(),
            }
        }

        // Slow path: park.  The mutex makes "check state, then sleep"
        // atomic with respect to `set`/`poison`, so no wakeup can be
        // lost: the resolver either flips the state word before our
        // check (we see it and never park) or after (it then counts our
        // `waiters` increment and notifies the condvar).
        let mut park = self.inner.park.lock();
        park.waiters += 1;
        let result = loop {
            match self.inner.state.load(Ordering::Acquire) {
                LATCH_COMPLETED => break Ok(()),
                LATCH_POISONED => break Err(()),
                _ => self.inner.cv.wait(&mut park),
            }
        };
        park.waiters -= 1;
        result
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
        // Skip the futex wake entirely when nobody is parked: every
        // waiter was still in its bounded spin and already observed the
        // state word above.
        if self.inner.park.lock().waiters > 0 {
            self.inner.cv.notify_all();
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

#[cfg(test)]
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
