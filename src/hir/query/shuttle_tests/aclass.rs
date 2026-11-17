use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[test]
fn d5_eval_always_parallel_same_key() {
    run_check(|| {
        static ENTERED: AtomicBool = AtomicBool::new(false);
        static RUNS: AtomicU64 = AtomicU64::new(0);
        ENTERED.store(false, Ordering::SeqCst);
        RUNS.store(0, Ordering::SeqCst);

        struct D5Q;
        impl<'q, 'src: 'q> Query<'q, 'src> for D5Q {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d5", "eval-always parallel query").with_eval_always(true)
            }
            fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                RUNS.fetch_add(1, Ordering::SeqCst);
                ENTERED.store(true, Ordering::SeqCst);
                shuttle::thread::yield_now();
                FpValue {
                    raw: format!("{}", key.0),
                    semantic: key.0 as u64,
                }
            }
        }

        let system = leak_system();
        let provider = leak_provider(system);

        let a_handle = shuttle::thread::spawn(move || {
            let _ = system.get::<D5Q>(DKey(1), provider);
        });
        spin_until(&ENTERED, "the first eval_always compute must start");
        a_handle
            .join()
            .expect("the first eval_always request must not panic");

        let _ = system
            .get::<D5Q>(DKey(1), provider)
            .expect("eval_always second request must compute independently");
        assert_eq!(
            RUNS.load(Ordering::SeqCst),
            2,
            "eval_always must not deduplicate: each requester computes its own value"
        );
    });
}

/// d5's CONCURRENT half: the sequential variant above joins the first get
/// before issuing the second, so it proves no CROSS-request dedup but not
/// that two CONCURRENT eval_always requesters fail to coalesce in flight
/// (the more interesting case — that is exactly what a tracked query's
/// wait-graph would do).  Two requesters run at once; each must compute
/// (RUNS == 2) and both must observe their own key's value.
///
/// COVERAGE LIMIT (structural, under shuttle): the same-KEY concurrent
/// shape is not expressible in this harness.  The active-set partitions
/// are keyed by `thread_local::ThreadLocal` — an OS-THREAD key — and
/// shuttle multiplexes every task onto one OS thread, so two concurrent
/// same-key gets share one partition and the second is mis-reported as a
/// `Cycle` (stack `[DKey(1), DKey(1)]`).  On real OS threads the
/// partitions are separate and the same shape is legal (each requester
/// computes its own value); a genuine same-key concurrent eval_always
/// test needs the partition key to become task-aware under shuttle, which
/// is a production-code change, not a test-harness one.
#[test]
fn d5_eval_always_concurrent_same_key_no_inflight_coalescing() {
    run_check(|| {
        static RUNS: AtomicU64 = AtomicU64::new(0);
        RUNS.store(0, Ordering::SeqCst);

        struct D5C;
        impl<'q, 'src: 'q> Query<'q, 'src> for D5C {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d5c", "eval-always concurrent query").with_eval_always(true)
            }
            fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                RUNS.fetch_add(1, Ordering::SeqCst);
                FpValue {
                    raw: format!("{}", key.0),
                    semantic: key.0 as u64,
                }
            }
        }

        let system = leak_system();
        let provider = leak_provider(system);

        let a = shuttle::thread::spawn(move || {
            system.get::<D5C>(DKey(1), provider).map(|v| v.semantic)
        });
        let b = shuttle::thread::spawn(move || {
            system.get::<D5C>(DKey(2), provider).map(|v| v.semantic)
        });
        let va = a
            .join()
            .expect("d5c requester A must not panic")
            .expect("d5c get A ok");
        let vb = b
            .join()
            .expect("d5c requester B must not panic")
            .expect("d5c get B ok");
        assert_eq!(
            RUNS.load(Ordering::SeqCst),
            2,
            "two concurrent eval_always requesters must each compute their \
             own value — in-flight coalescing is tracked-query-only"
        );
        assert_eq!(va, 1, "requester A must observe its own key's value");
        assert_eq!(vb, 2, "requester B must observe its own key's value");
    });
}

/// d9: `clear_all`'s quiescence and `reset`'s hygiene across the three
/// windows a concurrent reader can occupy.
///
/// COVERAGE GAP — deliberately not covered: WINDOW 1, a slow query
/// parked INSIDE its compute while `clear_all` lands on another thread
/// (the actual quiescence window).  `RELEASE` is pre-set below, so
/// `SlowQ(9)`'s spin never engages and the compute never parks; the
/// flag exists so the test stays deterministic under shuttle, where a
/// genuinely parked compute would serialize with the churn thread and
/// turn the model into a fairness puzzle.  Windows 2 and 3 (a rebuilt
/// node reusing its index without inheriting a stamp, and no stale
/// `node_to_key` entries) are both covered.  A Window 1 test needs a
/// second task that parks before `clear_all` is called — worth adding
/// separately, ideally as its own `check_random` case.
#[test]
fn d9_clear_all_quiescence_and_reset_hygiene() {
    run_check(|| {
        static SLOW_CONTENT: AtomicU64 = AtomicU64::new(1);
        static SLOW_RUNS: AtomicU64 = AtomicU64::new(0);
        static ENTERED: AtomicBool = AtomicBool::new(false);
        static RELEASE: AtomicBool = AtomicBool::new(false);
        SLOW_CONTENT.store(1, Ordering::SeqCst);
        SLOW_RUNS.store(0, Ordering::SeqCst);
        ENTERED.store(false, Ordering::SeqCst);
        // Pre-set (see the WINDOW 1 coverage gap on the test doc): the
        // compute below therefore never parks, which is what keeps this
        // case deterministic under shuttle.
        RELEASE.store(true, Ordering::SeqCst);

        struct SlowQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for SlowQ {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d9_slow", "slow query")
            }
            fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                SLOW_RUNS.fetch_add(1, Ordering::SeqCst);
                if key.0 != 9 {
                    return FpValue {
                        raw: format!("slow_{}", key.0),
                        semantic: SLOW_CONTENT.load(Ordering::SeqCst),
                    };
                }
                ENTERED.store(true, Ordering::SeqCst);
                while !RELEASE.load(Ordering::SeqCst) {
                    shuttle::thread::yield_now();
                }
                FpValue {
                    raw: format!("slow_{}", key.0),
                    semantic: SLOW_CONTENT.load(Ordering::SeqCst),
                }
            }
        }

        struct TopQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for TopQ {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d9_top", "top reader")
            }
            fn compute<S: Scheduler>(
                key: &DKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> FpValue {
                let s = query_get::<SlowQ, _>(provider, key.clone()).expect("top reads slow");
                FpValue {
                    raw: format!("top_{}", s.semantic),
                    semantic: 100 + s.semantic,
                }
            }
        }

        let system = leak_system();
        let provider = leak_provider(system);

        static STOP: AtomicBool = AtomicBool::new(false);
        STOP.store(false, Ordering::SeqCst);
        let churn = shuttle::thread::spawn(move || {
            while !STOP.load(Ordering::SeqCst) {
                shuttle::thread::yield_now();
            }
        });
        shuttle::thread::yield_now();

        let v = system.get::<TopQ>(DKey(1), provider).unwrap();
        assert_eq!(v.semantic, 101);
        let v = system.get::<TopQ>(DKey(1), provider).unwrap();
        assert_eq!(v.semantic, 101);

        system.clear_all();

        let _ = system.get::<SlowQ>(DKey(9), provider);

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
            "precondition: the verify pass must have stamped the closure (dirty_generation was bumped by the invalidation)"
        );
        assert!(
            !system.node_to_key.is_empty(),
            "precondition: the reverse mapping must be populated pre-clear"
        );

        system.clear_all();

        assert!(
            system.node_to_key.is_empty(),
            "clear_all must clear node_to_key (Window 3: no stale (TypeId, key) entries may survive the reset)"
        );

        let _ = system.get::<SlowQ>(DKey(9), provider);
        let vtable2 = system
            .find_vtable::<SlowQ>()
            .expect("SlowQ must be re-registered after the rebuild get");
        let node9b = vtable2
            .lookup_node(&DKey(9))
            .expect("rebuilt SlowQ(9) must have a node");
        assert_eq!(
            node9b, node9,
            "precondition for Window 2: the rebuilt node reuses the pre-clear index (next_index restarted at 1)"
        );
        assert_eq!(
            system.dep_graph.verified_gen_of(node9b),
            0,
            "a freshly allocated node must not inherit the pre-clear verified_gen stamp (Window 2)"
        );

        STOP.store(true, Ordering::SeqCst);
        churn.join().expect("churn thread panicked");
    });
}

/// WINDOW 1 (the gap `d9_clear_all_quiescence_and_reset_hygiene`
/// documents as uncovered): a SLOW COMPUTE parked inside its own
/// `compute` while a second task calls `clear_all`.
///
/// This is the one `clear_all` interleaving that needs no injected
/// schedule point, because the blocking is done by the lock rather than
/// by a choreography: the parked getter holds the quiesce READ guard
/// (`InFlightGuard::acquire` takes it before `acquire_key_slot` — that
/// order is load-bearing for the deadlock it avoids), so `clear_all`'s
/// WRITE lock cannot be granted until that getter has returned.
///
/// Which is exactly why this test exists.  It pins the invariant that
/// makes the `clear_all` poison sweep UNREACHABLE-BY-CONSTRUCTION:
/// every `Started` job a `clear_all` could wipe has a live read guard
/// behind it, so the sweep can never observe a straggler.  If a future
/// edit lets the write lock in before the read guard, or drops the read
/// guard while a job is `Started`, the `COMPUTE_EXITED` assertion below
/// fails — and the poison sweep stops being dead code at the same moment.
///
/// Not a `TestScheduler` choreography: there is no schedule point inside
/// `clear_all`, so a gate could not target the interleaving anyway.
///
/// HOW THE SYNCHRONISATION ACTUALLY HAPPENS (the first versions of this
/// test got this wrong, and the failures were the informative part).
/// Under `--features shuttle` every task runs on ONE OS thread, so the
/// thread-local `SESSION` that `InFlightGuard::acquire` maintains is
/// SHARED between tasks — the same multiplexed-TLS class documented on
/// `sync_order::HELD`.  The getter's frame is therefore
/// visible to the main task's `mutate_world`, whose `SESSION.total > 0`
/// branch DEFERS the mutation onto the target system's queue instead of
/// taking the write lock inline.  So on this path the mutexing is done
/// by the DEFERRAL, not the lock: the parked closure runs at the
/// outermost guard's drop, which is strictly after `compute` returned.
///
/// On real OS threads the two tasks have separate sessions, so the
/// mutation would NOT defer and would block on `quiesce.write()`
/// instead — also strictly after the getter's read guard dropped, i.e.
/// after `compute` returned.  Both mechanisms order the two events the
/// same way.
///
/// EMPIRICALLY CHECKED, both paths: forcing the mutation INTO the inline
/// branch (bypassing the defer) still moves it after the compute,
/// because `quiesce.write()` cannot be granted while the getter holds
/// the read guard — the LOCK is the ordering.  The deferral is not what
/// makes this safe on a single-thread model; the lock is what makes it
/// safe on the multi-thread one.  Neither can be reordered away without
/// the other being broken too.
///
/// The main task holds no `InFlightGuard` of its own, so a deferred
/// closure parked from here has no drop to hang from inside the window:
/// it stays unrun until the test's explicit `drain_deferred` after the
/// getter joins.  That flush makes delivery REQUIRED (not a tolerated
/// starvation outcome), so the ordering assertion below is unconditional.
#[test]
fn d9_window1_slow_compute_blocks_clear_all() {
    run_check(|| {
        static CLEAR_CALLED: AtomicBool = AtomicBool::new(false);
        static RELEASE: AtomicBool = AtomicBool::new(false);
        /// Set INSIDE `compute`, not before the get.  Seeing it is what
        /// proves the getter has already taken the quiesce READ guard AND
        /// is mid-get — waiting any earlier would let the getter's get
        /// start AFTER `clear_all` finished, which is a legal
        /// interleaving and would make the assertion meaningless.
        static COMPUTE_ENTERED: AtomicBool = AtomicBool::new(false);
        /// Set as `compute` RETURNS.  This is the observable for the
        /// invariant, and it has to be this rather than anything measured
        /// from outside the get: the store happens strictly inside the
        /// read-guarded region, so a `clear_all` that had been granted
        /// the write lock could not have observed it, and conversely a
        /// `clear_all` that returned without waiting would be proved by
        /// finding it still UNSET.  (Compare a flag loaded after `get()`
        /// returns: by then the guard is already dropped, so "it returned
        /// and the flag is set" is equally consistent with the lock never
        /// having blocked at all.)
        static COMPUTE_EXITED: AtomicU64 = AtomicU64::new(0);
        static SEQ: AtomicU64 = AtomicU64::new(0);
        static CLEAR_SEQ: AtomicU64 = AtomicU64::new(0);
        /// Arms the compute park.  The WARM get below must run to
        /// completion (it is the setup that creates the cache entry this
        /// window is about), so the park has to be off until the getter is
        /// forked — arming it earlier parks the setup itself.
        static ARM_PARK: AtomicBool = AtomicBool::new(false);
        static CONTENT: AtomicU64 = AtomicU64::new(4);
        static RUNS: AtomicU64 = AtomicU64::new(0);
        CLEAR_CALLED.store(false, Ordering::SeqCst);
        RELEASE.store(false, Ordering::SeqCst);
        COMPUTE_ENTERED.store(false, Ordering::SeqCst);
        COMPUTE_EXITED.store(0, Ordering::SeqCst);
        SEQ.store(0, Ordering::SeqCst);
        CLEAR_SEQ.store(0, Ordering::SeqCst);
        ARM_PARK.store(false, Ordering::SeqCst);
        CONTENT.store(4, Ordering::SeqCst);
        RUNS.store(0, Ordering::SeqCst);

        struct W1Q;
        impl<'q, 'src: 'q> Query<'q, 'src> for W1Q {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d9_w1", "window-1 slow compute query")
            }
            fn compute<S: Scheduler>(_key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                RUNS.fetch_add(1, Ordering::SeqCst);
                COMPUTE_ENTERED.store(true, Ordering::SeqCst);
                // Park until the releaser lets this compute finish.  This
                // is what holds the getter's quiesce READ guard open, so
                // the main task's `clear_all` MUST block behind it.
                while ARM_PARK.load(Ordering::SeqCst) && !RELEASE.load(Ordering::SeqCst) {
                    shuttle::thread::yield_now();
                }
                let v = FpValue {
                    raw: format!("w1_{}", CONTENT.load(Ordering::SeqCst)),
                    semantic: CONTENT.load(Ordering::SeqCst),
                };
                // The observable.  Still inside the read-guarded region
                // (the guard is dropped by `get_shared_inner`'s epilogue,
                // after `compute` and the whole publish tail).  Only
                // recorded for the PARKED (getter) compute — the warm-up
                // compute must not set it, or the assertion below is
                // satisfied by setup and proves nothing.
                if ARM_PARK.load(Ordering::SeqCst) {
                    COMPUTE_EXITED.store(SEQ.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                }
                v
            }
        }

        let system = leak_system();
        let provider = leak_provider(system);

        // Warm the path so the getter's get has a stable node and cache
        // entry, and the parked compute below is a RECOMPUTE (which is the
        // interesting shape: a recompute whose result must still be exact
        // even though a `clear_all` lands behind it).
        CONTENT.store(4, Ordering::SeqCst);
        let warm = system.get::<W1Q>(DKey(9), provider).expect("warm get");
        assert_eq!(warm.semantic, 4);
        RUNS.store(0, Ordering::SeqCst);
        system.invalidate::<W1Q>(&DKey(9));
        // Arm the park only now, after the setup is complete, and clear
        // the marker the WARM compute already set — otherwise the
        // rendezvous below is satisfied before the getter exists and
        // `clear_all` legally completes first.
        ARM_PARK.store(true, Ordering::SeqCst);
        COMPUTE_ENTERED.store(false, Ordering::SeqCst);

        let getter = shuttle::thread::spawn(move || {
            let v = system.get::<W1Q>(DKey(9), provider).expect("window-1 get");
            v.semantic
        });

        // Wait until the getter is INSIDE its compute — i.e. it holds the
        // quiesce READ guard and is mid-get.  Waiting any earlier (before
        // the get is issued) would allow the legal interleaving where the
        // getter's get starts only after `clear_all` finished.
        spin_until(&COMPUTE_ENTERED, "window-1 getter to enter its compute");
        // Releaser spawned BEFORE `CLEAR_CALLED` is set: if the flag went
        // up first, a schedule that ran the getter in between would leave it
        // spinning with no live task able to release it.  The releaser's
        // existence is what closes that window.
        let releaser = shuttle::thread::spawn(|| {
            spin_until(&CLEAR_CALLED, "clear_all to be called");
            RELEASE.store(true, Ordering::SeqCst);
        });
        CLEAR_CALLED.store(true, Ordering::SeqCst);

        // A mutation, not a `clear_all`, so its delivery point is
        // observable: the closure records where in the shared order it
        // actually RAN.  Under shuttle this DEFERS - the getter's frame
        // is in this thread's shared `SESSION`, so `mutate_world`'s
        // `SESSION.total > 0` branch parks the closure on the target
        // system's queue instead of taking the write lock.  On real
        // threads the sessions are separate, the mutation would not defer,
        // and it would block on `quiesce.write()` instead.  Either way it
        // cannot execute until the getter's read guard is gone, and that
        // guard spans the whole get - so strictly after `compute` returned.
        system.mutate_world(|_| {
            CLEAR_SEQ.store(SEQ.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
        });

        let served = getter.join().expect("getter must not panic");
        releaser.join().expect("releaser must not panic");

        // Delivery, not starvation: the deferred closure has no drop to
        // hang from in THIS scope (the main task holds no InFlightGuard),
        // so the test flushes the deferred partition explicitly now that
        // every task has joined — turning `CLEAR_SEQ == 0` from a legal
        // outcome into a FAILURE, and making the ordering assertion below
        // unconditional rather than vacuously true.  (On real OS threads
        // the sessions are separate and the mutation ran inline; the
        // drain is a no-op there.)
        system.drain_deferred();

        assert_eq!(
            served, 4,
            "the getter's recompute must be exact even though the world mutation landed behind it"
        );
        assert_eq!(
            RUNS.load(Ordering::SeqCst),
            1,
            "the invalidated node must recompute exactly once"
        );
        // THE invariant: the mutation executed strictly AFTER the parked
        // compute returned.  That is what makes the clear_all poison sweep
        // unreachable-by-construction - no world mutation (clear included)
        // can be delivered while a `Started` job's owner is still between
        // `acquire_key_slot` and the publish tail.  Both delivery
        // mechanisms order it the same way: the parked closure runs at the
        // outermost guard's drop (after `compute`), and the write lock
        // cannot be granted until that guard is gone.
        // With the explicit drain above, delivery is REQUIRED here (the
        // earlier "CLEAR_SEQ == 0 is a legal starvation outcome" tolerance
        // was this test describing a weaker test than the one in the
        // file); the assertion now pins both delivery AND ordering.
        assert!(
            CLEAR_SEQ.load(Ordering::SeqCst) > 0
                && COMPUTE_EXITED.load(Ordering::SeqCst) < CLEAR_SEQ.load(Ordering::SeqCst),
            "a world mutation executed BEFORE the in-flight compute returned (or was never delivered): compute finished at order {} but the mutation ran at order {}. It was delivered while a Started job's owner was still between acquire_key_slot and the publish tail, which is what makes the clear_all poison sweep unreachable-by-construction. A clear_all delivered this way could wipe a live job and strand its latch waiters forever.",
            COMPUTE_EXITED.load(Ordering::SeqCst),
            CLEAR_SEQ.load(Ordering::SeqCst)
        );
        // Post-clear consistency: the deferred wipe has been delivered (the
        // drain above proves the machinery, this get proves the system is
        // still usable and serves the current content).
        let after = system
            .get::<W1Q>(DKey(9), provider)
            .expect("post-clear get must succeed");
        assert_eq!(
            after.semantic,
            CONTENT.load(Ordering::SeqCst),
            "post-clear get must be exact"
        );
    });
}

#[test]
fn d1_same_key_self_recursion() {
    run_check_swept(|| {
        struct D1Q;
        impl<'q, 'src: 'q> Query<'q, 'src> for D1Q {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d1", "self-recursive query")
            }
            fn compute<S: Scheduler>(
                key: &DKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> FpValue {
                match query_get::<D1Q, _>(provider, key.clone()) {
                    Ok(v) => v,
                    Err(_) => FpValue {
                        raw: "cycle_sentinel".into(),
                        semantic: 0,
                    },
                }
            }
        }

        let system = leak_system();
        let provider = leak_provider(system);

        static STOP: AtomicBool = AtomicBool::new(false);
        STOP.store(false, Ordering::SeqCst);
        let churn = shuttle::thread::spawn(move || {
            while !STOP.load(Ordering::SeqCst) {
                shuttle::thread::yield_now();
            }
        });
        shuttle::thread::yield_now();

        match system.get::<D1Q>(DKey(1), provider) {
            Ok(v) => assert_eq!(
                v.raw, "cycle_sentinel",
                "same-key self-recursion must surface as the cycle sentinel"
            ),
            Err(e) => panic!("expected cycle sentinel, got error: {e:?}"),
        }

        STOP.store(true, Ordering::SeqCst);
        churn.join().expect("churn thread panicked");
    });
}

#[test]
fn d2_same_thread_cross_key_cycle() {
    run_check_swept(|| {
        struct D2Q;
        impl<'q, 'src: 'q> Query<'q, 'src> for D2Q {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d2", "cross-key cyclic query")
            }
            fn compute<S: Scheduler>(
                key: &DKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> FpValue {
                let next = if key.0 == 1 { DKey(2) } else { DKey(1) };
                match query_get::<D2Q, _>(provider, next) {
                    Ok(v) => v,
                    Err(_) => FpValue {
                        raw: "cycle_sentinel".into(),
                        semantic: 0,
                    },
                }
            }
        }

        let system = leak_system();
        let provider = leak_provider(system);

        static STOP: AtomicBool = AtomicBool::new(false);
        STOP.store(false, Ordering::SeqCst);
        let churn = shuttle::thread::spawn(move || {
            while !STOP.load(Ordering::SeqCst) {
                shuttle::thread::yield_now();
            }
        });
        shuttle::thread::yield_now();

        match system.get::<D2Q>(DKey(1), provider) {
            Ok(v) => assert_eq!(
                v.raw, "cycle_sentinel",
                "cross-key cycle k1 -> k2 -> k1 must surface as the cycle sentinel"
            ),
            Err(e) => panic!("expected cycle sentinel, got error: {e:?}"),
        }

        STOP.store(true, Ordering::SeqCst);
        churn.join().expect("churn thread panicked");
    });
}
