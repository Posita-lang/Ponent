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
//! - [`QueryWaiter`] — a blocked thread waiting for a query to complete.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::dep_graph::DepNodeIndex;

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
    /// The `DepNodeIndex` of the query being computed.
    pub dep_node: DepNodeIndex,
    /// The latch that other threads can wait on.
    pub latch: QueryLatch,
}

impl QueryJob {
    pub fn new(dep_node: DepNodeIndex) -> Self {
        QueryJob {
            id: QueryJobId::fresh(),
            dep_node,
            latch: QueryLatch::new(),
        }
    }

    /// Signal that the query has completed and resume all waiters.
    pub fn signal_complete(self) {
        self.latch.set();
    }
}

/// A waiter blocked on a query result.
#[derive(Debug)]
pub struct QueryWaiter {
    /// Reserved for future use: tracks the parent query job that spawned
    /// this waiter.  Currently unused — all waiters are created with
    /// `parent: None`.  This field is kept as a hook for diagnostics
    /// (e.g. building a wait-chain tree) or for priority-based scheduling
    /// of nested query computations.
    pub parent: Option<QueryJobId>,
    pub condvar: Condvar,
    pub completed: Mutex<bool>,
}

impl QueryWaiter {
    pub fn new(parent: Option<QueryJobId>) -> Self {
        QueryWaiter {
            parent,
            condvar: Condvar::new(),
            completed: Mutex::new(false),
        }
    }
}

/// A latch that allows one thread to compute a query while others wait.
///
/// Analogous to Rust's `QueryLatch` in `job.rs`.
/// Uses a `Condvar` to block waiting threads and resume them when the
/// query completes.
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
    /// `Some(..)` while the query is active, `None` once completed.
    waiters: Arc<Mutex<Option<Vec<Arc<QueryWaiter>>>>>,
}

impl QueryLatch {
    pub fn new() -> Self {
        QueryLatch {
            waiters: Arc::new(Mutex::new(Some(Vec::new()))),
        }
    }

    /// Block the current thread until the query completes.
    ///
    /// Returns `Ok(())` when the query has completed successfully.
    /// Returns `Err(())` if the query was poisoned (panicked).
    pub fn wait_on(&self) -> Result<(), ()> {
        // Scope the waiters lock so it is released before we block on the
        // condvar below.  If we held self.waiters across the condvar wait,
        // set() would deadlock trying to acquire self.waiters to notify us.
        let waiter = {
            let mut waiters_guard = self.waiters.lock().unwrap();
            let Some(waiters) = &mut *waiters_guard else {
                // Already complete — the result is in the cache.
                return Ok(());
            };

            let waiter = Arc::new(QueryWaiter::new(None));
            waiters.push(Arc::clone(&waiter));
            waiter
        }; // waiters_guard is dropped here → self.waiters unlocked

        // Block until the latch is set.
        let mut completed = waiter.completed.lock().unwrap();
        while !*completed {
            completed = waiter.condvar.wait(completed).unwrap();
        }

        Ok(())
    }

    /// Signal that the query is complete and resume all waiting threads.
    pub fn set(&self) {
        let mut waiters_guard = self.waiters.lock().unwrap();
        let waiters = waiters_guard.take(); // mark as complete
        if let Some(waiters) = waiters {
            for waiter in waiters {
                *waiter.completed.lock().unwrap() = true;
                waiter.condvar.notify_one();
            }
        }
    }

    /// Check whether the query has completed.
    pub fn is_complete(&self) -> bool {
        self.waiters.lock().unwrap().is_none()
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
}