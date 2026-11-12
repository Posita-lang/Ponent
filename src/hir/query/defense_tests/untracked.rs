//! Untracked-query regression tests (memo semantics, clear, eviction,
//! digest-gate rejection), split out of `defense_tests` (M4).
//!
//! Shares the parent module's scaffolding (`DKey` / `DValue` /
//! `leak_system` / `leak_provider`) via `use super::*`.

use super::*;

/// An untracked query computes once per key: the second `get_shared` for
/// the same key must hit the memo map and return the identical `Arc`
/// (no recompute, no dep machinery involved).
#[test]
fn u1_untracked_memo_hit_returns_same_arc() {
    static COMPUTES: AtomicUsize = AtomicUsize::new(0);
    COMPUTES.store(0, Ordering::SeqCst);

    struct U1Q;
    impl<'q, 'src: 'q> Query<'q, 'src> for U1Q {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u1", "untracked memo query").with_untracked(true)
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            COMPUTES.fetch_add(1, Ordering::SeqCst);
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    let a = system
        .get_shared::<U1Q>(DKey(7), provider)
        .expect("first get");
    let b = system
        .get_shared::<U1Q>(DKey(7), provider)
        .expect("second get");
    assert_eq!(
        COMPUTES.load(Ordering::SeqCst),
        1,
        "second get must not recompute"
    );
    assert!(
        Arc::ptr_eq(&a, &b),
        "memo hit must return the identical Arc"
    );
}

/// A nested query issued from inside an untracked computation violates the
/// untracked contract: hard panic (not a cycle error — the author broke
/// the promise, the graph is fine).
#[test]
#[should_panic(expected = "issued from inside an untracked computation")]
fn u2_untracked_nested_get_panics() {
    struct U2Inner;
    impl<'q, 'src: 'q> Query<'q, 'src> for U2Inner {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u2inner", "tracked leaf")
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            DValue::new(key.0 as u64)
        }
    }

    struct U2Q;
    impl<'q, 'src: 'q> Query<'q, 'src> for U2Q {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u2", "impure untracked query").with_untracked(true)
        }
        fn compute<S: Scheduler>(key: &DKey, provider: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            // Forbidden: a nested get inside an untracked computation.
            query_get::<U2Inner, _>(provider, key.clone()).unwrap()
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);
    let _ = system.get_shared::<U2Q>(DKey(1), provider);
}

/// Regression: `clear::<Q>` must also drop untracked memo entries — the
/// contract "next get after clear recomputes" applies to the memo-only
/// store too.  Before the fix, `clear` silently no-op'd for untracked
/// queries (their memos live outside the node-ful cache) and the stale
/// memo was served forever between compile iterations.
#[test]
fn u3_clear_untracked_drops_memo_next_get_recomputes() {
    static COMPUTES: AtomicUsize = AtomicUsize::new(0);
    COMPUTES.store(0, Ordering::SeqCst);

    struct U3Q;
    impl<'q, 'src: 'q> Query<'q, 'src> for U3Q {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u3", "untracked memo query for clear").with_untracked(true)
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            COMPUTES.fetch_add(1, Ordering::SeqCst);
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    let a = system
        .get_shared::<U3Q>(DKey(7), provider)
        .expect("first get");
    assert_eq!(COMPUTES.load(Ordering::SeqCst), 1);

    system.clear::<U3Q>();

    let b = system
        .get_shared::<U3Q>(DKey(7), provider)
        .expect("get after clear");
    assert_eq!(
        COMPUTES.load(Ordering::SeqCst),
        2,
        "clear must drop the untracked memo — the next get must recompute"
    );
    assert!(
        !Arc::ptr_eq(&a, &b),
        "the recomputed value must be a fresh Arc"
    );
}

/// d19's untracked twin (P1): an in-flight `get_untracked` compute that
/// started BEFORE `clear::<Q>` must NOT resurrect the cleared memo entry
/// on its way out.  `get_untracked` takes no quiesce read pin — "nothing
/// to quiesce" holds for the compute's inputs, but the memo is
/// served-world state — so pre-fix, the insert landed after the map was
/// wiped with no generation check, and the next get HIT the pre-clear
/// value: the exact "next get recomputes" contract `clear` promises
/// (already enforced for the tracked path by the d19 family).  The
/// memo-generation stamp makes the insert-time validation discard the
/// stale result; only the rare race wastes the compute.
#[test]
fn u8_untracked_clear_races_in_flight_compute_recomputes() {
    static COMPUTES: AtomicUsize = AtomicUsize::new(0);
    static ENTERED: AtomicBool = AtomicBool::new(false);
    static RELEASE: AtomicBool = AtomicBool::new(false);
    COMPUTES.store(0, Ordering::SeqCst);
    ENTERED.store(false, Ordering::SeqCst);
    RELEASE.store(false, Ordering::SeqCst);

    struct U8Q;
    impl<'q, 'src: 'q> Query<'q, 'src> for U8Q {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u8", "untracked clear-race query").with_untracked(true)
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            COMPUTES.fetch_add(1, Ordering::SeqCst);
            ENTERED.store(true, Ordering::SeqCst);
            // Bounded park (the d9/d10/d18/d24 deadline discipline): if a
            // regression ever makes `clear` deadlock instead of racing, the
            // test must FAIL LOUDLY after 2 s instead of wedging the whole
            // `cargo test` process — `RELEASE` may simply never come.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !RELEASE.load(Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    break;
                }
                std::thread::yield_now();
            }
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // 1) Start a compute that parks inside `compute` (no quiesce pin —
    //    the untracked path takes none).
    let h = std::thread::spawn(move || system.get_shared::<U8Q>(DKey(7), provider));
    while !ENTERED.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }

    // 2) Clear WHILE the compute is parked — the exact race.
    system.clear::<U8Q>();

    // 3) Release: the compute finishes and tries to insert; the
    //    generation check must discard the pre-clear result.
    RELEASE.store(true, Ordering::SeqCst);
    let _ = h.join().expect("compute thread must not panic");

    // 4) The next get must RECOMPUTE — the in-flight compute's insert
    //    must not have resurrected the memo entry (d19-class violation,
    //    untracked edition).
    let _ = system
        .get_shared::<U8Q>(DKey(7), provider)
        .expect("post-clear get");
    assert_eq!(
        COMPUTES.load(Ordering::SeqCst),
        2,
        "the post-clear get must recompute — the in-flight compute's \
         insert must be discarded by the memo generation check"
    );
}

/// Regression: untracked memo overflow must evict ONE entry, not clear
/// the whole map (the working-set cliff: at capacity, a wholesale clear
/// makes every refill cycle wipe the window and re-reads of earlier keys
/// recompute).
/// With cap = 2, after overflowing with a third key the map retains the
/// FIFO survivor (key 2 — eviction is by FIRST insertion via the
/// `VecDeque` queue beside the memo map) — so re-reading the survivor is
/// a HIT, and re-reading the evicted key costs exactly ONE recompute.
/// The old wholesale clear would cost TWO; the pre-FIFO bucket-arbitrary
/// victim made the survivor unnameable and the count
/// layout-dependent (a different FxHashMap build could flip it).
#[test]
fn u4_untracked_evict_one_keeps_a_survivor() {
    static COMPUTES: AtomicUsize = AtomicUsize::new(0);
    COMPUTES.store(0, Ordering::SeqCst);

    struct U4Q;
    impl<'q, 'src: 'q> Query<'q, 'src> for U4Q {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u4", "small-capacity untracked memo")
                .with_untracked(true)
                .with_cache_capacity(2)
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            COMPUTES.fetch_add(1, Ordering::SeqCst);
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    let _ = system.get_shared::<U4Q>(DKey(1), provider).expect("cold 1");
    let _ = system.get_shared::<U4Q>(DKey(2), provider).expect("cold 2");
    let _ = system
        .get_shared::<U4Q>(DKey(3), provider)
        .expect("overflow 3");
    assert_eq!(COMPUTES.load(Ordering::SeqCst), 3, "no key read twice yet");

    // FIFO eviction evicted the FIRST-inserted key (1); key 2 is the
    // named survivor.  The survivor's re-read must be a HIT (no
    // recompute), then the evicted key's re-read costs exactly ONE
    // recompute (and itself becomes the next FIFO victim).
    let base = COMPUTES.load(Ordering::SeqCst);
    let _ = system
        .get_shared::<U4Q>(DKey(2), provider)
        .expect("re-read the FIFO survivor");
    assert_eq!(
        COMPUTES.load(Ordering::SeqCst),
        base,
        "the FIFO survivor must still be memoized (a wholesale clear or \
         bucket-order victim would recompute here)"
    );
    let _ = system
        .get_shared::<U4Q>(DKey(1), provider)
        .expect("re-read the evicted key");
    assert_eq!(
        COMPUTES.load(Ordering::SeqCst),
        base + 1,
        "exactly ONE recompute for the evicted key (a wholesale clear would \
         cost one per re-read: {base} -> {})",
        COMPUTES.load(Ordering::SeqCst)
    );
}

/// UNTRACKED MEMO UNDER CONCURRENT SAME-KEY `get_shared`
/// (previously missing coverage).
///
/// `get_untracked` has no in-flight latch: it is a memo map plus a
/// purity fence, nothing else.  So N threads racing on the same cold key
/// can each take the "miss" branch and each run `compute` — a thundering
/// herd, which is duplicated WORK, and each returns its own handle.
/// What must hold is that the racing never produces a WRONG or TORN
/// answer, and that the memo converges afterwards: once any handle is
/// published, every later get — from any thread — must hit the memo,
/// return the identical `Arc`, and stop recomputing.
///
/// This is also the one place `get_shared` can legitimately hand out
/// different handles for one key, so the test pins the two halves of the
/// contract separately and documents that the herd is by design
/// (a "no world to invalidate" query deliberately carries no latch).
#[test]
fn u7_concurrent_same_key_get_shared_never_tears_or_diverges() {
    static COMPUTES: AtomicUsize = AtomicUsize::new(0);
    COMPUTES.store(0, Ordering::SeqCst);

    struct U7Q;
    impl<'q, 'src: 'q> Query<'q, 'src> for U7Q {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            // Capacity is left UNBOUNDED (0) on purpose: a bounded memo
            // would let the storm's inserts evict the published handle
            // and the post-storm convergence assertion would flake.
            QueryDescriptor::new("u7", "untracked memo query, concurrent").with_untracked(true)
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            COMPUTES.fetch_add(1, Ordering::SeqCst);
            // A real compute would be a pure function of the key; the
            // yields just widen the storm window so the herd actually
            // races, rather than serialising on a fast map insert.
            std::thread::yield_now();
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);

    // Warm the query's vtable so the storm measures the MEMO race, not a
    // first-registration race (`vtable_or_init` is the one path here
    // that does take a lock).  The key is deliberately NOT the storm key
    // — warming the storm key would make it a hit and there'd be no
    // herd left to exercise.
    let _ = system.get_shared::<U7Q>(DKey(999), provider).expect("warm");
    let base = COMPUTES.load(Ordering::SeqCst);

    const THREADS: usize = 8;
    const KEY: DKey = DKey(42);
    std::thread::scope(|scope| {
        for t in 0..THREADS {
            scope.spawn(move || {
                let v = system
                    .get_shared::<U7Q>(KEY, provider)
                    .unwrap_or_else(|e| panic!("t{t}: {e:?}"));
                assert_eq!(
                    v.semantic, KEY.0 as u64,
                    "t{t}: served {} for key {} — a torn or wrong untracked value",
                    v.semantic, KEY.0
                );
            });
        }
    });

    // The herd bound: each thread ran `compute` AT MOST once, so the
    // count for this key is in [1, THREADS].  It is 1 when the storm
    // serialises behind the first insert, and THREADS when every thread
    // misses — both are the documented behaviour, so the assertion is a
    // RANGE, and the "no entry lost" property is the UPPER bound (a memo
    // that dropped an entry between insert and read would exceed it).
    let herd = COMPUTES.load(Ordering::SeqCst) - base;
    assert!(
        (1..=THREADS).contains(&herd),
        "the herd must cost between 1 and {THREADS} computes for one key, got {herd}"
    );

    // Convergence: after any handle is published, every later get — from
    // any thread — must hit the memo, return the identical `Arc`, and
    // stop recomputing.  This is the half of the contract that is
    // deterministic, and it is what makes the herd harmless.
    let frozen = COMPUTES.load(Ordering::SeqCst);
    let a = system.get_shared::<U7Q>(KEY, provider).expect("re-get");
    let b = system.get_shared::<U7Q>(KEY, provider).expect("re-get 2");
    assert!(
        Arc::ptr_eq(&a, &b),
        "post-storm re-gets of a memoised key must return the identical Arc"
    );
    assert_eq!(
        COMPUTES.load(Ordering::SeqCst),
        frozen,
        "post-storm re-gets must be memo hits ({frozen} -> {})",
        COMPUTES.load(Ordering::SeqCst)
    );
}
#[test]
#[should_panic(expected = "invalidate_if_changed on untracked query")]
fn u5_digest_gate_rejects_untracked_query() {
    struct U5Q;
    impl<'q, 'src: 'q> Query<'q, 'src> for U5Q {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u5", "untracked digest-reject").with_untracked(true)
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);
    // Register the vtable — the assert only fires when find_vtable is Some.
    let _ = system
        .get_shared::<U5Q>(DKey(1), provider)
        .expect("register the untracked vtable");
    // The gate must reject instead of returning a confident false.
    let _ = system.invalidate_if_changed::<U5Q>(&DKey(1), 42);
}

/// The companion recording API (`note_input_digest`) shares the same fence:
/// it must also reject an untracked query in every build.  It had its own
/// `debug_assert!` for the same contract, so it needs its own pin — a
/// `debug_assert!`-only gate would silently stop firing in release.
#[test]
#[should_panic(expected = "note_input_digest on untracked query")]
fn u6_note_input_digest_rejects_untracked_query() {
    struct U6Q;
    impl<'q, 'src: 'q> Query<'q, 'src> for U6Q {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u6", "untracked digest-record reject").with_untracked(true)
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            DValue::new(key.0 as u64)
        }
    }

    let system = leak_system();
    let provider = leak_provider(system);
    // Pre-register the vtable (historically the assert only fired when
    // find_vtable was Some — see u5b for the never-driven path).
    let _ = system
        .get_shared::<U6Q>(DKey(1), provider)
        .expect("register the untracked vtable");
    system.note_input_digest::<U6Q>(&DKey(1), 42);
}

/// u5's gap (P1): the descriptor-based check must reject an untracked
/// query even when NO vtable has ever been registered.  The old
/// registry-based assert left this path silent — `find_vtable` returned
/// `None`, the assert passed, and the gate returned a confident `false`
/// ("content unchanged") for a query it had nothing to look at.
/// Deliberately NOT `get_shared`ing first: the assert must fire from
/// `Q::descriptor()` alone.
#[test]
#[should_panic(expected = "invalidate_if_changed on untracked query")]
fn u5b_digest_gate_rejects_never_driven_untracked_query() {
    struct U5bQ;
    impl<'q, 'src: 'q> Query<'q, 'src> for U5bQ {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u5b", "untracked, never driven").with_untracked(true)
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            DValue::new(key.0 as u64)
        }
    }
    let system = leak_system();
    let _ = system.invalidate_if_changed::<U5bQ>(&DKey(1), 42);
}

/// u6's gap: same never-driven shape for the recording API.
#[test]
#[should_panic(expected = "note_input_digest on untracked query")]
fn u6b_note_input_digest_rejects_never_driven_untracked_query() {
    struct U6bQ;
    impl<'q, 'src: 'q> Query<'q, 'src> for U6bQ {
        type Storage = Owned;
        type Key = DKey;
        type Value = DValue;
        type Cache = DefaultCache<DKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("u6b", "untracked, never driven").with_untracked(true)
        }
        fn compute<S: Scheduler>(key: &DKey, _: &dyn QueryProvider<'q, 'src, S>) -> DValue {
            DValue::new(key.0 as u64)
        }
    }
    let system = leak_system();
    system.note_input_digest::<U6bQ>(&DKey(1), 42);
}
