use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[test]
fn d10_side_door_invalidate_erased_by_publish() {
    run_check(|| {
        static INPUT: AtomicU64 = AtomicU64::new(0);
        static ENTERED: AtomicBool = AtomicBool::new(false);
        static RELEASE: AtomicBool = AtomicBool::new(false);
        static B_DONE: AtomicBool = AtomicBool::new(false);
        INPUT.store(0, Ordering::SeqCst);
        ENTERED.store(false, Ordering::SeqCst);
        RELEASE.store(false, Ordering::SeqCst);
        B_DONE.store(false, Ordering::SeqCst);

        struct PQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for PQ {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d10_p", "side-door victim")
            }
            fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                let snap = INPUT.load(Ordering::SeqCst);
                ENTERED.store(true, Ordering::SeqCst);
                while !RELEASE.load(Ordering::SeqCst) {
                    shuttle::thread::yield_now();
                }
                FpValue {
                    raw: format!("d10_{}", key.0),
                    semantic: snap,
                }
            }
        }

        let system = leak_system();
        let provider = leak_provider(system);

        let a = shuttle::thread::spawn(move || {
            system
                .get::<PQ>(DKey(2), provider)
                .map_err(|e| e.to_string())
        });
        spin_until(&ENTERED, "d10 A must enter compute");

        INPUT.store(1, Ordering::SeqCst);
        let node = system
            .find_vtable::<PQ>()
            .and_then(|vt| vt.lookup_node(&DKey(2)))
            .expect("node must exist once compute started");

        let b = shuttle::thread::spawn(move || {
            system.invalidate_node(node);
            B_DONE.store(true, Ordering::SeqCst);
        });

        let mut landed = false;
        for _ in 0..1_000 {
            if B_DONE.load(Ordering::SeqCst) {
                landed = true;
                break;
            }
            shuttle::thread::yield_now();
        }

        RELEASE.store(true, Ordering::SeqCst);
        let r_a = a.join().expect("d10 A panicked");
        b.join().expect("d10 B panicked");
        assert!(r_a.is_ok(), "A must survive: {r_a:?}");

        let v = system.get::<PQ>(DKey(2), provider).unwrap();
        assert_eq!(
            v.semantic, 1,
            "side-door invalidation must survive the in-flight publish (landed={landed})"
        );
    });
}

#[test]
fn d17_cached_path_node_itself_dirty_serves_fresh_value() {
    run_check(|| {
        static E_CONTENT: AtomicU64 = AtomicU64::new(1);
        static E_RUNS: AtomicU64 = AtomicU64::new(0);
        static R_RUNS: AtomicU64 = AtomicU64::new(0);
        E_CONTENT.store(1, Ordering::SeqCst);
        E_RUNS.store(0, Ordering::SeqCst);
        R_RUNS.store(0, Ordering::SeqCst);

        struct D17E;
        impl<'q, 'src: 'q> Query<'q, 'src> for D17E {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d17_inner", "d17 leaf input")
            }
            fn compute<S: Scheduler>(_key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                E_RUNS.fetch_add(1, Ordering::SeqCst);
                let s = E_CONTENT.load(Ordering::SeqCst);
                FpValue {
                    raw: format!("d17e_{s}"),
                    semantic: s,
                }
            }
        }
        struct D17R;
        impl<'q, 'src: 'q> Query<'q, 'src> for D17R {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d17_reader", "d17 reader over d17_inner")
            }
            fn compute<S: Scheduler>(
                key: &DKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> FpValue {
                R_RUNS.fetch_add(1, Ordering::SeqCst);
                let e = query_get::<D17E, _>(provider, key.clone()).unwrap();
                FpValue {
                    raw: format!("d17r_{}", e.semantic),
                    semantic: e.semantic * 10,
                }
            }
        }

        let sched = TestScheduler::new();
        let system: &'static QuerySystem<'static, 'static, TestScheduler> =
            Box::leak(Box::new(QuerySystem::with_scheduler(sched.clone())));
        let provider = leak_provider(system);

        let v = system.get::<D17R>(DKey(0), provider).unwrap();
        assert_eq!(v.semantic, 10, "warm value");
        assert_eq!(E_RUNS.load(Ordering::SeqCst), 1);
        assert_eq!(R_RUNS.load(Ordering::SeqCst), 1);

        system.invalidate::<D17E>(&DKey(0));

        sched.arm(|p| matches!(p, SchedulePoint::CachedHit { query } if *query == "d17_reader"));
        let (tx, rx) = shuttle::sync::mpsc::channel();
        shuttle::thread::spawn(move || {
            let r = system.get::<D17R>(DKey(0), provider);
            let _ = tx.send(r.map(|v| v.semantic));
        });
        sched.wait_parked();

        E_CONTENT.store(100, Ordering::SeqCst);
        let e2 = system.get::<D17E>(DKey(0), provider).unwrap();
        assert_eq!(e2.semantic, 100);
        assert_eq!(E_RUNS.load(Ordering::SeqCst), 2);

        sched.release_all();
        let got = rx.recv().expect("reader must finish");

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
    });
}

#[test]
fn d11_reentrant_invalidate_inside_compute() {
    run_check(|| {
        static OTHER: AtomicU64 = AtomicU64::new(0);
        static TRIGGER: AtomicBool = AtomicBool::new(false);
        static OTHER_RUNS: AtomicU64 = AtomicU64::new(0);
        static DRIVER_RUNS: AtomicU64 = AtomicU64::new(0);
        static STOP: AtomicBool = AtomicBool::new(false);
        OTHER.store(0, Ordering::SeqCst);
        TRIGGER.store(false, Ordering::SeqCst);
        OTHER_RUNS.store(0, Ordering::SeqCst);
        DRIVER_RUNS.store(0, Ordering::SeqCst);
        STOP.store(false, Ordering::SeqCst);

        struct OtherQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for OtherQ {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d11_other", "invalidation target")
            }
            fn compute<S: Scheduler>(_: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                OTHER_RUNS.fetch_add(1, Ordering::SeqCst);
                FpValue {
                    raw: "other".into(),
                    semantic: OTHER.load(Ordering::SeqCst),
                }
            }
        }

        struct DriverQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for DriverQ {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d11_driver", "invalidates from inside compute")
            }
            fn compute<S: Scheduler>(
                key: &DKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> FpValue {
                DRIVER_RUNS.fetch_add(1, Ordering::SeqCst);
                if TRIGGER.load(Ordering::SeqCst) {
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

        let churn = shuttle::thread::spawn(move || {
            while !STOP.load(Ordering::SeqCst) {
                shuttle::thread::yield_now();
            }
        });
        shuttle::thread::yield_now();

        let v = system.get::<OtherQ>(DKey(0), provider).unwrap();
        assert_eq!(v.semantic, 0);
        let v = system.get::<DriverQ>(DKey(1), provider).unwrap();
        assert_eq!(v.semantic, 7);
        assert_eq!(OTHER_RUNS.load(Ordering::SeqCst), 1);

        TRIGGER.store(true, Ordering::SeqCst);
        OTHER.store(1, Ordering::SeqCst);
        system.invalidate::<DriverQ>(&DKey(1));
        let semantic = system
            .get::<DriverQ>(DKey(1), provider)
            .expect("reentrant invalidate must not deadlock the query")
            .semantic;
        assert_eq!(semantic, 7);

        let v = system.get::<OtherQ>(DKey(0), provider).unwrap();
        assert_eq!(
            v.semantic, 1,
            "deferred reentrant invalidation must take effect"
        );
        assert_eq!(OTHER_RUNS.load(Ordering::SeqCst), 2);

        STOP.store(true, Ordering::SeqCst);
        churn.join().expect("churn thread panicked");
    });
}

#[test]
fn d15_clear_all_waits_for_quiesced_reader() {
    run_check(|| {
        static E_RUNS: AtomicU64 = AtomicU64::new(0);
        static R_RUNS: AtomicU64 = AtomicU64::new(0);
        E_RUNS.store(0, Ordering::SeqCst);
        R_RUNS.store(0, Ordering::SeqCst);

        struct D15E;
        impl<'q, 'src: 'q> Query<'q, 'src> for D15E {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d15_inner", "d15 leaf input")
            }
            fn compute<S: Scheduler>(_key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                E_RUNS.fetch_add(1, Ordering::SeqCst);
                FpValue {
                    raw: "d15e".into(),
                    semantic: 3,
                }
            }
        }
        struct D15R;
        impl<'q, 'src: 'q> Query<'q, 'src> for D15R {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d15_reader", "d15 reader over d15_inner")
            }
            fn compute<S: Scheduler>(
                key: &DKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> FpValue {
                R_RUNS.fetch_add(1, Ordering::SeqCst);
                let e = query_get::<D15E, _>(provider, key.clone()).unwrap();
                FpValue {
                    raw: format!("d15r_{}", e.semantic),
                    semantic: e.semantic * 10,
                }
            }
        }

        let sched = TestScheduler::new();
        let system: &'static QuerySystem<'static, 'static, TestScheduler> =
            Box::leak(Box::new(QuerySystem::with_scheduler(sched.clone())));
        let provider = leak_provider(system);

        let v = system.get::<D15R>(DKey(0), provider).unwrap();
        assert_eq!(v.semantic, 30, "warm value");
        assert_eq!(R_RUNS.load(Ordering::SeqCst), 1);
        assert_eq!(E_RUNS.load(Ordering::SeqCst), 1);

        sched.arm(|p| matches!(p, SchedulePoint::CachedHit { query } if *query == "d15_reader"));
        let (rtx, rrx) = shuttle::sync::mpsc::channel();
        {
            let (sys, prov) = (system, provider);
            shuttle::thread::spawn(move || {
                let _ = rtx.send(sys.get::<D15R>(DKey(0), prov).map(|v| v.semantic));
            });
        }
        sched.wait_parked();

        let (ctx, crx) = shuttle::sync::mpsc::channel::<()>();
        {
            let sys = system;
            shuttle::thread::spawn(move || {
                sys.clear_all();
                let _ = ctx.send(());
            });
        }
        for _ in 0..100 {
            assert!(
                !system.vtables.load().is_empty(),
                "clear_all completed while a quiesced reader was parked in its preflight"
            );
            shuttle::thread::yield_now();
        }

        sched.release_all();
        let served = rrx.recv().expect("reader must finish");
        assert_eq!(
            served.expect("parked reader must succeed"),
            30,
            "parked reader must serve its cached value"
        );

        crx.recv()
            .expect("clear_all must complete after the reader");

        let v2 = system
            .get::<D15R>(DKey(0), provider)
            .expect("post-clear get must succeed")
            .semantic;
        assert_eq!(v2, 30);
        assert_eq!(
            R_RUNS.load(Ordering::SeqCst),
            2,
            "post-clear get must recompute"
        );
        assert_eq!(E_RUNS.load(Ordering::SeqCst), 2);
    });
}
