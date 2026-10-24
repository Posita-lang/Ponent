use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[test]
fn d19_clear_must_quiesce_in_flight_publish() {
    run_check(|| {
        static K_RUNS: AtomicU64 = AtomicU64::new(0);
        static ENTERED: AtomicBool = AtomicBool::new(false);
        static RELEASE: AtomicBool = AtomicBool::new(false);
        K_RUNS.store(0, Ordering::SeqCst);
        ENTERED.store(false, Ordering::SeqCst);
        RELEASE.store(false, Ordering::SeqCst);

        struct KQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for KQ {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d19_k", "clear resurrection victim")
            }
            fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                K_RUNS.fetch_add(1, Ordering::SeqCst);
                ENTERED.store(true, Ordering::SeqCst);
                while !RELEASE.load(Ordering::SeqCst) {
                    shuttle::thread::yield_now();
                }
                FpValue {
                    raw: format!("d19_{}", key.0),
                    semantic: 7,
                }
            }
        }

        let system = leak_system();
        let provider = leak_provider(system);

        let a = {
            let (sys, prov) = (system, provider);
            shuttle::thread::spawn(move || {
                sys.get::<KQ>(DKey(0), prov)
                    .map(|v| v.semantic)
                    .map_err(|e| e.to_string())
            })
        };
        spin_until(&ENTERED, "d19 A must enter compute");

        let clearer = {
            let sys = system;
            shuttle::thread::spawn(move || {
                sys.clear::<KQ>();
            })
        };
        RELEASE.store(true, Ordering::SeqCst);

        let ra = a.join().expect("d19 A panicked");
        assert_eq!(ra.unwrap(), 7, "A returns its own computed value");
        clearer.join().expect("d19 clearer panicked");

        let r = system
            .get::<KQ>(DKey(0), provider)
            .map(|v| v.semantic)
            .map_err(|e| e.to_string());
        assert!(r.is_ok(), "post-clear get must succeed: {r:?}");
        assert_eq!(r.unwrap(), 7);
        assert_eq!(
            K_RUNS.load(Ordering::SeqCst),
            2,
            "the next get after clear must recompute - clear must quiesce in-flight publishes"
        );
    });
}
