use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Randomised multi-task stress with a value oracle: churn tasks
/// invalidate and re-get a small key space while the main task checks
/// every served value against the source it must equal, then drives a
/// deliberately panicking compute and asserts the poisoned key stays
/// poisoned.
///
/// Phase 3's `catch_unwind` (below) requires unwinding, so the crate
/// must not set `panic = "abort"` — `Cargo.toml`'s `[profile.release]`
/// deliberately does not, which is what keeps the release-mode run of
/// this test (and of the guard's own abort-free release behaviour)
/// meaningful.
///
/// The panic hook is scoped rather than global: phase 3 injects a panic
/// on purpose and catches it up to 40 times per `run_check` iteration,
/// and shuttle re-runs this closure dozens of times, so the un-muted
/// hook prints ~40 backtraces per iteration for panics that are the
/// expected outcome.  `QuietPanics` swaps in a no-op hook for phase 3
/// ONLY and restores the previous one through `Drop` — so a real
/// assertion failure inside the loop cannot leak a silent hook into a
/// later test, and every other phase keeps printing.  To debug a phase-3
/// failure, set `PONENT_NOISY_PANICS=1` to opt out of the swap.
struct QuietPanics {
    prev: Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send>,
}

impl QuietPanics {
    fn install() -> Option<Self> {
        if std::env::var_os("PONENT_NOISY_PANICS").is_some() {
            return None;
        }
        // `take_hook` returns the previously installed hook; there is no
        // other way to observe it, so this swap-and-restore is the only
        // shape available.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        Some(Self { prev })
    }
}

impl Drop for QuietPanics {
    fn drop(&mut self) {
        std::panic::set_hook(std::mem::replace(
            &mut self.prev,
            Box::new(|_| {}),
        ));
    }
}
#[test]
fn stress_random_concurrent_graph_oracle() {
    run_check(|| {
        const STRESS_SLOTS: usize = 16;
        const PANIC_SLOT: usize = 15;

        static SRC: [AtomicU64; STRESS_SLOTS] = [const { AtomicU64::new(0) }; STRESS_SLOTS];
        static PANIC_FLAG: AtomicBool = AtomicBool::new(false);
        static OP_COUNTER: AtomicU64 = AtomicU64::new(0);
        static STOP: AtomicBool = AtomicBool::new(false);
        OP_COUNTER.store(0, Ordering::SeqCst);
        PANIC_FLAG.store(false, Ordering::SeqCst);
        for s in &SRC {
            s.store(0, Ordering::SeqCst);
        }

        struct SrcQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for SrcQ {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("stress_src", "stress input query")
            }
            fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> FpValue {
                let slot = key.0 % STRESS_SLOTS;
                if PANIC_FLAG.load(Ordering::SeqCst) && slot == PANIC_SLOT {
                    panic!("stress panic injection");
                }
                FpValue {
                    raw: format!("s{slot}_{}", SRC[slot].load(Ordering::SeqCst)),
                    semantic: SRC[slot].load(Ordering::SeqCst),
                }
            }
        }

        struct SumQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for SumQ {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("stress_sum", "stress sum query")
            }
            fn compute<S: Scheduler>(
                key: &DKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> FpValue {
                let slot = key.0 % STRESS_SLOTS;
                let a = query_get::<SrcQ, _>(provider, DKey(slot)).expect("sum reads src");
                let b = query_get::<SrcQ, _>(provider, DKey((slot + 1) % STRESS_SLOTS))
                    .expect("sum reads src");
                FpValue {
                    raw: format!("sum{slot}_{}", a.semantic + b.semantic),
                    semantic: a.semantic + b.semantic,
                }
            }
        }

        struct ChainQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for ChainQ {
            type Storage = Owned;
            type Key = DKey;
            type Value = FpValue;
            type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("stress_chain", "stress recursive chain query")
            }
            fn compute<S: Scheduler>(
                key: &DKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> FpValue {
                let slot = key.0 / 100;
                let d = key.0 % 100;
                if d > 0 {
                    let inner = query_get::<ChainQ, _>(provider, DKey(slot * 100 + d - 1))
                        .expect("chain recurse");
                    FpValue {
                        raw: format!("c{slot}_{d}_{}", inner.semantic),
                        semantic: inner.semantic + 1,
                    }
                } else {
                    let src = query_get::<SrcQ, _>(provider, DKey(slot)).expect("chain base");
                    FpValue {
                        raw: format!("c{slot}_0_{}", src.semantic),
                        semantic: src.semantic,
                    }
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

        let mut local = 0usize;
        loop {
            let op = OP_COUNTER.fetch_add(1, Ordering::Relaxed);
            if op >= 200 {
                break;
            }
            let slot = ((op as usize).wrapping_mul(2654435761) + local) % STRESS_SLOTS;
            local = local.wrapping_add(1);
            match op % 10 {
                0..=3 => {
                    let v = system
                        .get::<SumQ>(DKey(slot), provider)
                        .expect("value phase must not error");
                    let hi = SRC[slot].load(Ordering::SeqCst)
                        + SRC[(slot + 1) % STRESS_SLOTS].load(Ordering::SeqCst);
                    assert!(
                        v.semantic <= hi,
                        "sum value must not exceed current sources: {v:?} vs {hi}"
                    );
                }
                4..=6 => {
                    let v = system
                        .get::<SrcQ>(DKey(slot), provider)
                        .expect("value phase must not error");
                    let hi = SRC[slot].load(Ordering::SeqCst);
                    assert!(
                        v.semantic <= hi,
                        "src value must not exceed current source: {v:?} vs {hi}"
                    );
                }
                7..=8 => {
                    let d = (op as usize % 4) as u64;
                    let v = system
                        .get::<ChainQ>(DKey(slot * 100 + d as usize), provider)
                        .expect("value phase must not error");
                    let hi = SRC[slot].load(Ordering::SeqCst) + d;
                    assert!(
                        v.semantic <= hi,
                        "chain value must not exceed current source + depth: {v:?} vs {hi}"
                    );
                }
                _ => {
                    if op % 20 >= 10 {
                        SRC[slot].fetch_add(1, Ordering::SeqCst);
                    }
                    system.invalidate::<SrcQ>(&DKey(slot));
                }
            }
        }

        for slot in 0..STRESS_SLOTS {
            system.invalidate::<SrcQ>(&DKey(slot));
        }
        for slot in 0..STRESS_SLOTS {
            let expect = SRC[slot].load(Ordering::SeqCst);
            let v = system.get::<SrcQ>(DKey(slot), provider).expect("src get");
            assert_eq!(v.semantic, expect, "quiesced src mismatch at slot {slot}");
            for d in 0..=3u64 {
                let expect_d = expect + d;
                let v = system
                    .get::<ChainQ>(DKey(slot * 100 + d as usize), provider)
                    .expect("chain get");
                assert_eq!(
                    v.semantic, expect_d,
                    "quiesced chain mismatch at slot {slot} depth {d}"
                );
            }
            let expect_sum = expect + SRC[(slot + 1) % STRESS_SLOTS].load(Ordering::SeqCst);
            let v = system.get::<SumQ>(DKey(slot), provider).expect("sum get");
            assert_eq!(
                v.semantic, expect_sum,
                "quiesced sum mismatch at slot {slot}"
            );
        }

        PANIC_FLAG.store(true, Ordering::SeqCst);
        system.invalidate::<SrcQ>(&DKey(PANIC_SLOT));
        // Phase 3 mutes the default hook: every iteration below panics
        // on purpose and catches it, and the backtrace for a panic that
        // IS the expected outcome is noise.  The guard restores the hook
        // on every exit path, including an assertion failure.
        let _quiet = QuietPanics::install();
        for _ in 0..40 {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                system.get::<SrcQ>(DKey(PANIC_SLOT), provider)
            }));
            match r {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    assert!(
                        matches!(e, QueryError::Poisoned { .. }),
                        "panic-phase error must mention poisoning: {e:?}"
                    );
                }
                Err(_) => {}
            }
        }

        let v = system.get::<SrcQ>(DKey(0), provider).expect("liveness get");
        assert_eq!(v.semantic, SRC[0].load(Ordering::SeqCst));
        let poisoned = system.get::<SrcQ>(DKey(PANIC_SLOT), provider);
        assert!(
            poisoned.is_err(),
            "poisoned key must stay poisoned for the session: {poisoned:?}"
        );

        STOP.store(true, Ordering::SeqCst);
        churn.join().expect("churn thread panicked");
    });
}
