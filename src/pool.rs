//! Resident diagnostic-render workers with one-batch-at-a-time submission.
//!
//! Rayon remains the general-purpose pool, but this emitter needs one extra
//! guarantee it cannot express: the submitter keeps a non-`Send` sink on the
//! calling thread while resident workers render into a slot table.  That
//! requires submitting borrowed jobs and releasing the batch only after every
//! job has run *and dropped*.  This module confines that lifetime erasure to
//! one audited `transmute`; the diagnostic data itself is still exclusively
//! owned through `Mutex` cells (see `emitter::BatchShared`).

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

/// A `FnOnce` with a caller borrow.  `Send` is a real bound: every closure
/// this pool receives captures only `&BatchShared`, `Arc` clones and other
/// `Send` values.
pub(crate) struct SendJob<'t>(Box<dyn FnOnce() + Send + 't>);

impl<'t> SendJob<'t> {
    pub(crate) fn new(job: impl FnOnce() + Send + 't) -> Self {
        Self(Box::new(job))
    }

    fn run(self) {
        (self.0)();
    }
}

struct State {
    /// `Some` only between `submit` and the completion latch.
    jobs: Option<VecDeque<SendJob<'static>>>,
    /// Jobs enqueued but not yet completed (including panics).
    remaining: usize,
    shutdown: bool,
}

struct Core {
    state: Mutex<State>,
    work: Condvar,
    done: Condvar,
}

/// One-batch-at-a-time permit.
///
/// The zero-sized field is private to this module, so a token can only come from
/// [`WorkerPool::new`] (the first one) or [`BatchWait::wait`] (the one returned
/// by a completed batch).  [`WorkerPool::submit`] demanding one *by value* turns
/// "submit again before waiting" into a compile error.  Previously that was a
/// `debug_assert!` only, which left a release-mode UB path: a second `submit`
/// overwrote `state.jobs` and reset `remaining`, running workers then decremented
/// the new counter, `wait` could return early, and the caller touched `'t`
/// borrows while jobs were still running.
pub(crate) struct BatchToken(());

pub(crate) struct WorkerPool {
    core: Arc<Core>,
    /// `WorkerPool::drop` has `&mut self`, so the `Mutex` that used to wrap this had exactly one
    /// runtime effect: a `if let Ok(..)` that silently skipped the joins on a poisoned lock and let
    /// the workers become detached.  Poisoning is excluded by construction (every worker lock uses
    /// `unwrap_or_else(into_inner)`), which made the fallback dead *and* silent — the worst
    /// combination.  `JoinHandle` is `Send + Sync`, so a plain `Vec` keeps the type `Sync`.
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl WorkerPool {
    /// Spawn the resident workers.  Called lazily by the emitter, so a CLI
    /// run that never reaches the parallel threshold spawns none.
    pub(crate) fn new(workers: usize) -> (Self, BatchToken) {
        let core = Arc::new(Core {
            state: Mutex::new(State {
                jobs: None,
                remaining: 0,
                shutdown: false,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
        });
        let handles = (0..workers.max(1))
            .map(|i| {
                let core = Arc::clone(&core);
                std::thread::Builder::new()
                    .name(format!("ponent-diag-{i}"))
                    .spawn(move || worker_main(core))
                    .expect("failed to spawn diagnostic worker")
            })
            .collect();
        (Self { core, handles }, BatchToken(()))
    }

    /// Enqueue one batch without blocking.  The caller must not touch any data
    /// borrowed by the jobs until [`BatchWait::wait`] returns.
    ///
    /// # Safety
    ///
    /// `SendJob<'t>` is erased to `'static` for storage.  The erasure is sound
    /// under the following protocol:
    ///
    /// * exactly one batch is in flight (`&mut Emitter` + the debug assert);
    /// * each worker removes one job, runs it, and drops it *before*
    ///   decrementing `remaining` (panics included, via `catch_unwind`);
    /// * [`BatchWait::wait`] returns only at `remaining == 0`, and only then
    ///   clears the queue.
    ///
    /// Therefore no `'t` borrow can outlive the frame that created the jobs.
    pub(crate) fn submit<'t>(&self, token: BatchToken, jobs: Vec<SendJob<'t>>) -> BatchWait<'_> {
        // Consuming the token is the whole point: `submit` cannot be reached
        // without one, and `wait` is the only way to get it back.
        let _ = token;
        // The token already proves no batch is in flight; keep the state check as
        // defence in depth (a dropped `BatchWait` loses the token, and this is
        // what would report the protocol violation instead of corrupting state).
        let mut state = self.core.state.lock().unwrap_or_else(|e| e.into_inner());
        debug_assert!(
            state.jobs.is_none() && state.remaining == 0,
            "diagnostic pool: batch already in flight"
        );
        state.remaining = jobs.len();
        if !jobs.is_empty() {
            // SAFETY: see the protocol on `submit`; the jobs are reachable
            // only through `state.jobs` until `remaining` reaches zero.
            let erased: Vec<SendJob<'static>> = unsafe { std::mem::transmute(jobs) };
            state.jobs = Some(erased.into());
            self.core.work.notify_all();
        }
        BatchWait { core: &self.core }
    }
}

#[must_use = "a submitted batch must be waited on before the next submit"]
pub(crate) struct BatchWait<'a> {
    core: &'a Core,
}

impl BatchWait<'_> {
    /// Block until every job has run and been dropped, then release the slot and
    /// return the permit for the next batch.
    #[must_use = "the returned BatchToken is the only way to submit the next batch"]
    pub(crate) fn wait(self) -> BatchToken {
        let mut state = self.core.state.lock().unwrap_or_else(|e| e.into_inner());
        while state.remaining != 0 {
            state = self
                .core
                .done
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
        state.jobs = None;
        drop(state);
        BatchToken(())
    }
}

fn worker_main(core: Arc<Core>) {
    let mut state = core.state.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        while state.jobs.as_ref().is_none_or(VecDeque::is_empty) && !state.shutdown {
            state = core.work.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        if state.shutdown {
            return;
        }
        let Some(job) = state.jobs.as_mut().and_then(VecDeque::pop_front) else {
            continue;
        };
        drop(state);
        // A malformed diagnostic must not kill a worker or stall the latch.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run()));
        state = core.state.lock().unwrap_or_else(|e| e.into_inner());
        // Cannot reach zero twice under the token protocol; saturate so a latent
        // protocol violation stalls the latch instead of wrapping `usize::MAX`.
        state.remaining = state.remaining.saturating_sub(1);
        if state.remaining == 0 {
            core.done.notify_one();
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        {
            let mut state = self.core.state.lock().unwrap_or_else(|e| e.into_inner());
            state.shutdown = true;
        }
        self.core.work.notify_all();
        {
            for handle in self.handles.drain(..) {
                let _ = handle.join();
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn jobs_run_and_workers_are_reused() {
        let (pool, mut token) = WorkerPool::new(1);
        let tids = Arc::new(Mutex::new(Vec::new()));
        for _ in 0..2 {
            let tids = Arc::clone(&tids);
            let jobs = vec![SendJob::new(move || {
                tids.lock().unwrap().push(std::thread::current().id());
            })];
            token = pool.submit(token, jobs).wait();
        }
        let tids = tids.lock().unwrap();
        assert_eq!(tids.len(), 2);
        assert_eq!(tids[0], tids[1], "resident worker must be reused");
    }

    #[test]
    fn panic_does_not_deadlock_the_latch() {
        let (pool, token) = WorkerPool::new(2);
        let ran = Arc::new(AtomicUsize::new(0));
        let first = Arc::clone(&ran);
        let second = Arc::clone(&ran);
        let jobs = vec![
            SendJob::new(move || {
                first.fetch_add(1, Ordering::Relaxed);
                panic!("intentional worker panic");
            }),
            SendJob::new(move || {
                second.fetch_add(1, Ordering::Relaxed);
            }),
        ];
        let _ = pool.submit(token, jobs).wait();
        assert_eq!(
            ran.load(Ordering::Relaxed),
            2,
            "panic must not skip the other job"
        );
    }

    #[test]
    fn empty_batch_is_a_noop() {
        let (pool, token) = WorkerPool::new(1);
        let token = pool.submit(token, Vec::new()).wait();
        let _ = pool.submit(token, vec![SendJob::new(|| {})]).wait();
    }
}
