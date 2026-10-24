use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn d21_green_bit_never_outruns_cache_entry() {
    run_check(|| {
        static E_CONTENT: AtomicU64 = AtomicU64::new(1);
        static E_RUNS: AtomicU64 = AtomicU64::new(0);
        static K_RUNS: AtomicU64 = AtomicU64::new(0);
        E_CONTENT.store(1, Ordering::SeqCst);
        E_RUNS.store(0, Ordering::SeqCst);
        K_RUNS.store(0, Ordering::SeqCst);

        struct D21E;
        impl<'q, 'src: 'q> Query<'q, 'src> for D21E {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d21_inner", "d21 leaf input")
            }
            fn compute<S: Scheduler>(_key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                E_RUNS.fetch_add(1, Ordering::SeqCst);
                FpValue {
                    raw: String::new(),
                    semantic: E_CONTENT.load(Ordering::SeqCst),
                }
            }
        }
        struct D21K;
        impl<'q, 'src: 'q> Query<'q, 'src> for D21K {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d21_reader", "d21 reader over d21_inner")
            }
            fn compute<S: Scheduler>(
                key: &DKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> FpValue {
                K_RUNS.fetch_add(1, Ordering::SeqCst);
                let e = query_get::<D21E, _>(provider, key.clone()).unwrap();
                FpValue {
                    raw: String::new(),
                    semantic: e.semantic * 10,
                }
            }
        }

        let sched = TestScheduler::new();
        let system: &'static QuerySystem<'static, 'static, TestScheduler> =
            Box::leak(Box::new(QuerySystem::with_scheduler(sched.clone())));
        let provider = leak_provider(system);

        let warm = system
            .get::<D21K>(DKey(0), provider)
            .map(|v| v.semantic)
            .map_err(|e| e.to_string());
        assert_eq!(warm.unwrap(), 10, "warm must serve K=10");

        E_CONTENT.store(2, Ordering::SeqCst);
        system.invalidate::<D21E>(&DKey(0));
        let e2 = system
            .get::<D21E>(DKey(0), provider)
            .map(|v| v.semantic)
            .map_err(|e| e.to_string());
        assert_eq!(
            e2.unwrap(),
            2,
            "E must republish with the changed fingerprint"
        );

        sched.arm(|p| {
            matches!(
                p,
                SchedulePoint::AfterPublish { query, .. } if *query == "d21_reader"
            )
        });
        let (otx, orx) = shuttle::sync::mpsc::channel();
        let owner = {
            let (sys, prov) = (system, provider);
            shuttle::thread::spawn(move || {
                let _ = otx.send(
                    sys.get::<D21K>(DKey(0), prov)
                        .map(|v| v.semantic)
                        .map_err(|e| e.to_string()),
                );
            })
        };
        sched.wait_parked();

        let got = system
            .get::<D21K>(DKey(0), provider)
            .map(|v| v.semantic)
            .unwrap();
        assert_eq!(got, 20, "green serve returned the pre-recompute entry");

        sched.release_all();
        assert_eq!(
            orx.recv().expect("owner must finish").unwrap(),
            20,
            "owner must observe its own recompute"
        );
        owner.join().expect("owner thread must not panic");
        assert_eq!(
            K_RUNS.load(Ordering::SeqCst),
            2,
            "one warm + one recompute; the probe must be a serve, not a recompute"
        );
    });
}

// d23 (`d23_per_thread_stat_merge_exact`) lived here; it now runs as
// `mod tests::test_stat_slots_merge_exactly_per_thread` in `super::mod`
// (OS threads).  Reason, in short: shuttle multiplexes every task onto ONE
// OS thread, so the per-THREAD `STAT_SLOTS` map is shared by all of them —
// the first task to touch an arena registers the slot for everyone else
// and `stats()`' many-slot merge path is never exercised, making the
// test's central assertion vacuous under shuttle.  See the test's own doc
// for the full argument.
