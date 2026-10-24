//! # Lock-Order DAG — the written lock-order list + debug tripwire
//!
//! This module is the **single written source of truth** for the query
//! subsystem's lock topology (§0 table). It serves both purposes:
//!
//! 1. [`tripwire::ORD`] — the table-driven lock-order tripwire (machine-readable
//!    face) in debug builds;
//! 2. The transition table for the dep/cache families' typestate in ticket.rs
//!    (mechanism B) (compile-readable face).
//!
//! **Changing any one place (this table / ticket.rs's `impl` blocks / the
//! module docs) requires synchronizing all three.**
//!
//! ## Lock order table (§0)
//!
//! Verdict: ✓ = nesting allowed (row family is held before column family is
//! acquired); ✗ = forbidden; ⊥ = never nests (no constraint, but treated
//! conservatively as Forbid); leaf = no outgoing edges.
//!
//! | #  | Edge                                          | Verdict | Evidence (real call site) |
//! |----|-----------------------------------------------|---------|---------------------------|
//! | 1  | quiesce.{read,write} → everything            | ✓ root  | `InFlightGuard::acquire` / `mutate_world` |
//! | 2  | active_keys[shard].w → wait_graph.w          | ✓       | `get_shared` success path holds active lock and calls `clear_waiters_of_me`; `poison_active` same direction |
//! | 3  | active_keys[shard].w → latch.park            | ✓       | `job.latch.set()`/`poison()` (inside resolve `park.lock()`) under active lock |
//! | 4  | wait_graph ⊥ latch.park                       | ⊥       | `is_complete` is an atomic read; the two only appear sequentially inside the same active section |
//! | 5  | wait_graph / latch.park / node_to_key / deferred / vtable.active | leaf | no outgoing edges |
//! | 6  | node_to_key: same-thread read-lock upgrade to write | ✗ | g1c/stress hang; `try_serve_verified` clones out then releases |
//! | 7  | init_lock → query_names                      | retired (Tier1) | query names now come from the static `for_each_query!` manifest; no lock is acquired on this edge |
//! | 8  | cache.shard → stat_registry                  | ✓       | `stat_bump_cold` may run inside cache write lock (eviction loop) |
//! | 9  | stat_registry → cache.shard                  | ✗       | `stats()` takes `len()` before the registry |
//! | 10 | cache.shard → dep.*                          | ✗ (contract 3) | probe two-phase; alloc outside read lock |
//! | 11 | dep.shard[w] → grow_lock                     | ✓       | `allocate_node_index` (under quiesce.read) |
//! | 12 | grow_lock → anything                         | ✗ (leaf) | growth touches no shard data |
//! | 13 | lowering_lock → dep.shard[r]                 | ✓       | `propagate_lowering` cascade |
//! | 14 | dep.shard → lowering_lock                    | ✗       | both call sites are outside the shard critical section |
//! | 15 | dep.shard ↛ dep.shard (never coexist; reset is the sole exception: ascending, under quiesce.write, privacy-confined) | ✗ | module invariant |
//! | 16 | dep.shard → cache/stat/active/wait_graph     | ✗       | — |
//! | 17 | deferred → quiesce.w                         | ✗       | `drain_deferred` detaches the partition and drops the lock before entering `run_deferred_batch` |
//! | 18 | Atomic ordering (epoch→bit, generation chain, seqlock) | not a lock | comments + shuttle |
//! | 19 | init_lock → active_keys / node_to_key / wait_graph / dep.shard / grow_lock / latch.park | ✓ | `clear_all_impl` (poison sweep reaches latch.park via active → resolve) |
//!
//! quiesce is the root and does not enter the ORD matrix (no lock is held
//! before it — mechanism A's `Quiesced` proof pins this at the type level;
//! reset's multi-shard exception is handled in its body: ascending one at a
//! time through `with_shard_w`, never two shard locks held at once).

use std::cell::RefCell;

/// Lock families. The dep/cache families are managed at compile time by
/// mechanism B's typestate tickets, but guard construction/destruction still
/// registers with the tripwire (this is the runtime gate for "take a lock
/// under an entry" + the exploration signal under shuttle).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fam {
    ActiveShard,
    WaitGraph,
    LatchPark,
    NodeToKey,
    InitLock,
    CacheShard,
    StatRegistry,
    DepShard,
    /// `DepGraph::reset`'s sole multi-shard exception bit.  RESERVED /
    /// unregistered: reset guarantees "ascending one at a time, never two
    /// held at once" in its body, and since the with_shard_w conversion it
    /// registers plain `DepShard` per shard instead — a dedicated reset bit
    /// would need its own row set to stay meaningful.  No `check` runs
    /// against it; kept in the enum only so the FAMS stride and the §0
    /// discussion stay stable.
    DepShardReset,
    GrowLock,
    Lowering,
    /// RESERVED / unregistered: the deferred partition mutex.  Tabled
    /// (#5/#17) but deliberately not registered at its four acquisition
    /// sites (mutate_world's defer push, drain_deferred's take, the DrainGuard
    /// Drop sweep, and the drain fast-path marker) — #17's direction is
    /// already pinned structurally (the guard drops before
    /// run_deferred_batch).  Kept so the table's #5/#17 rows have a named
    /// family; wiring it up is the follow-up if a deferred-path regression
    /// ever justifies the span.
    Deferred,
}
pub(crate) const FAMS: usize = 12;

impl Fam {
    #[inline]
    const fn idx(self) -> usize {
        self as usize
    }

    /// All families in discriminant order, so the tripwire can name the
    /// HELD row it reports instead of printing a bare numeric index.
    const ALL: [Fam; FAMS] = [
        Fam::ActiveShard,
        Fam::WaitGraph,
        Fam::LatchPark,
        Fam::NodeToKey,
        Fam::InitLock,
        Fam::CacheShard,
        Fam::StatRegistry,
        Fam::DepShard,
        Fam::DepShardReset,
        Fam::GrowLock,
        Fam::Lowering,
        Fam::Deferred,
    ];
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Rel {
    /// Taking the column family while holding the row family: legal.
    Allow,
    /// Nesting forbidden (✗ / ⊥ treated conservatively as forbidden).
    Forbid,
}

/// Machine-readable face of the §0 table. `ORD[held][want]`: row = held
/// family, column = desired family. Change in sync with the module-header
/// table; change in sync with ticket.rs's `impl` blocks.
///
/// Construction note: start with the whole table `Forbid`, then allow the
/// ✓ edges cell by cell. quiesce is the root and not in the table; of all
/// ✓ edges only the "held family" appears as a row — during quiesce any lock
/// acquisition is legal, while "enter quiesce while holding another lock" is
/// guaranteed by mechanism A/structure and is not the tripwire's job.
const ORD: [[Rel; FAMS]; FAMS] = {
    let mut o = [[Rel::Forbid; FAMS]; FAMS];
    // #2: active_keys → wait_graph
    o[Fam::ActiveShard.idx()][Fam::WaitGraph.idx()] = Rel::Allow;
    // #3: active_keys → latch.park
    o[Fam::ActiveShard.idx()][Fam::LatchPark.idx()] = Rel::Allow;
    // #7 retired (Tier1): init_lock → query_names no longer exists —
    // query names come from the static manifest, no lock is acquired.
    // #8: cache.shard → stat_registry (cache.shard → dep.* is ✗, contract 3)
    o[Fam::CacheShard.idx()][Fam::StatRegistry.idx()] = Rel::Allow;
    // #11: dep.shard → grow_lock (only outgoing edge; → lowering ✗ #14, →
    //       itself ✗ #15, → cache/active/wait ✗ #16)
    o[Fam::DepShard.idx()][Fam::GrowLock.idx()] = Rel::Allow;
    // #13: lowering → dep.shard[r] (only outgoing edge)
    o[Fam::Lowering.idx()][Fam::DepShard.idx()] = Rel::Allow;
    // #19 (retired): active_keys[shard].w → vtable.active existed while
    //   the per-thread cycle state lived in a sharded mutex map
    //   (`QueryVTable::active`, committed under the shard write guard).
    //   The state is now per-query-type thread-local storage — `commit`
    //   acquires no lock — so the edge and its family are deleted.
    // #20: init_lock → active_keys / node_to_key / wait_graph / dep.shard /
    //   grow_lock / latch.park — all taken by `clear_all_impl` while it
    //   holds init_lock.  The span is load-bearing: a top-level par_get
    //   runs vtable_or_init without the quiesce read lock, so a registry
    //   cloned from the pre-clear map must not be able to win the store
    //   race.  vtable_or_init itself takes init_lock with nothing else
    //   held, so the direction is strictly init → {…} and no cycle exists.
    //   LatchPark is reached via the poison sweep: latch.poison() →
    //   resolve → park.lock() under each active shard write lock, under
    //   init_lock (check() tests EVERY held row, so this cell is required
    //   in addition to InitLock → ActiveShard).
    o[Fam::InitLock.idx()][Fam::ActiveShard.idx()] = Rel::Allow;
    o[Fam::InitLock.idx()][Fam::NodeToKey.idx()] = Rel::Allow;
    o[Fam::InitLock.idx()][Fam::WaitGraph.idx()] = Rel::Allow;
    o[Fam::InitLock.idx()][Fam::DepShard.idx()] = Rel::Allow;
    o[Fam::InitLock.idx()][Fam::GrowLock.idx()] = Rel::Allow;
    o[Fam::InitLock.idx()][Fam::LatchPark.idx()] = Rel::Allow;
    // All other rows (WaitGraph/LatchPark/NodeToKey/StatRegistry/
    // DepShardReset/GrowLock/Deferred) fully Forbid = leaf (#5/#12)
    // or forbidden outgoing edges (#6/#9/#10/#14/#15/#17).
    // (#7 retired: QueryNames no longer a family — see above.)
    o
};

thread_local! {
    /// Current live guard count per "family" (guard construction inc / Drop dec
    /// — reflects real lock hold, not the call instant). Under shuttle a
    /// violation = panic, deterministically replayable; this tripwire is also
    /// the exploration signal.
    ///
    /// Gated: an unread TLS cell still emits registration + a dtor, and
    /// release builds pay that for nothing — every consumer below is a
    /// `debug_assert`.  Shuttle runs are typically
    /// `--release --features shuttle` (the tripwire doubles as the
    /// exploration signal there), hence the disjunction, NOT
    /// `debug_assertions` alone.
    ///
    /// ### Known limitation: task-multiplexed TLS under shuttle
    ///
    /// In a plain `debug_assertions` build one OS thread == one TLS view,
    /// so `h[i] > 0` really does mean *this* thread re-holds family `i`.
    /// Under `feature = "shuttle"` that ceases to hold: shuttle runs every
    /// task on ONE OS thread, so a task that PARKS while a guard is live
    /// leaks its count into whichever task the scheduler runs next. The
    /// consequence is a false positive, never a false negative — the stray
    /// count is only ever read by a DIFFERENT task, which then trips
    /// `check`'s re-held assert on a family it does not itself hold.
    ///
    /// This cannot happen with same-family nesting, only with the
    /// multiplexing, and it is visible only where a task parks INSIDE a
    /// registered window (i.e. a blocking lock acquisition taken while the
    /// guard is live — exactly what the gate is shaped to observe).
    /// Fixing it properly means making the view per-task rather than
    /// per-OS-thread; until then, treat a shuttle re-held report as
    /// tentative and confirm the park/deadlock path before believing it.
    #[cfg(any(debug_assertions, feature = "shuttle"))]
    static HELD: RefCell<[u32; FAMS]> = const { RefCell::new([0; FAMS]) };
}

/// Instant check before lock acquisition.
///
/// - Mechanism-B families (DepShard/GrowLock/Lowering/CacheShard): the sole
///   hook point of `take_*`; same-family re-holding is caught by the guard
///   registration ("take a lock under an entry").
/// - Non-mechanism-B families: `Acquired::acquire` inlines the same check.
#[cfg(any(debug_assertions, feature = "shuttle"))]
#[inline]
pub(crate) fn check(f: Fam, site: &'static str) {
    HELD.with(|h| {
        let h = h.borrow();
        // Unconditional asserts, NOT debug_assert: this fn is already
        // gated on any(debug_assertions, shuttle), and shuttle runs are
        // typically --release --features shuttle, where debug_assertions
        // are OFF.  debug_assert here would compile the whole check out
        // and void the "under shuttle a violation = panic" contract.
        assert_eq!(h[f.idx()], 0, "lock family re-held: {f:?} @ {site}");
        for (row, &n) in h.iter().enumerate() {
            if n > 0 {
                assert!(
                    ORD[row][f.idx()] == Rel::Allow,
                    "lock-order violation: {:?} → {f:?} @ {site} (see sync_order.rs §0 table)",
                    Fam::ALL[row]
                );
            }
        }
    });
}

/// Release no-op: the whole runtime face (TLS access, RefCell flag flips,
/// row scan) only exists to feed the asserts inside the gated `check`, so a
/// build without either gate compiles it out — see the gate note on `HELD`.
#[cfg(not(any(debug_assertions, feature = "shuttle")))]
#[inline]
pub(crate) fn check(_f: Fam, _site: &'static str) {}

/// RAII registration for non-dep/cache families (active_keys/wait_graph/latch/
/// node_to_key/init/stat/vtable-active/deferred, about 12 manual points):
/// bind the guard with the lock guard in the same scope. Hook list = the §0
/// table's "Evidence" column: `get_shared` success/failure paths (active +
/// wait_graph + job latch via job.rs::resolve), `poison_active`,
/// `acquire_key_slot`, `try_serve_verified` (node_to_key), `vtable_or_init`
/// (init), `stat_bump_cold`/`stats()`, `drain_deferred`/`mutate_world`.
pub(crate) struct Acquired(Fam);

impl Acquired {
    #[cfg(any(debug_assertions, feature = "shuttle"))]
    #[inline]
    pub(crate) fn acquire(f: Fam, site: &'static str) -> Acquired {
        check(f, site);
        HELD.with(|h| h.borrow_mut()[f.idx()] += 1);
        Acquired(f)
    }

    /// Release no-op: the `Fam` payload is dead weight past this point,
    /// so the non-gated `acquire` still constructs `Acquired(_f)` but
    /// never reads it and the guard needs no `Drop` — every `_tw`
    /// binding keeps compiling unchanged (see the gate note on `HELD`).
    /// (Not a ZST: it keeps its `Fam` byte, which is why `acquire` takes
    /// the parameter in the ungated arm at all.)
    #[cfg(not(any(debug_assertions, feature = "shuttle")))]
    #[inline]
    pub(crate) fn acquire(_f: Fam, _site: &'static str) -> Acquired {
        Acquired(_f)
    }
}

impl Drop for Acquired {
    #[cfg(any(debug_assertions, feature = "shuttle"))]
    #[inline]
    fn drop(&mut self) {
        HELD.with(|h| h.borrow_mut()[self.0.idx()] -= 1);
    }

    #[cfg(not(any(debug_assertions, feature = "shuttle")))]
    #[inline]
    fn drop(&mut self) {}
}

/// Registration entry for mechanism-B guards (DepShardW/DepShardR/GrowGuard/
/// LoweringGuard/CacheShard*): construction inc, Drop dec.
#[cfg(any(debug_assertions, feature = "shuttle"))]
#[inline]
pub(crate) fn held_inc(f: Fam) {
    HELD.with(|h| h.borrow_mut()[f.idx()] += 1);
}

#[cfg(any(debug_assertions, feature = "shuttle"))]
#[inline]
pub(crate) fn held_dec(f: Fam) {
    HELD.with(|h| h.borrow_mut()[f.idx()] -= 1);
}

#[cfg(not(any(debug_assertions, feature = "shuttle")))]
#[inline]
pub(crate) fn held_inc(_f: Fam) {}

#[cfg(not(any(debug_assertions, feature = "shuttle")))]
#[inline]
pub(crate) fn held_dec(_f: Fam) {}

#[cfg(all(test, any(debug_assertions, feature = "shuttle")))]
mod tests {
    use super::*;

    /// M2 pin: every Allow cell in the machine table must carry a row
    /// number comment tying it to the module-header §0 table (the
    /// human-readable face).  This is enforced by construction — the
    /// cells are written one per line, each preceded by a `// #N` comment
    /// — so this test pins the MEANING instead: the set of allowed edges
    /// matches the §0 verdicts exactly (✓ rows here, and NOTHING else).
    /// If you add an Allow cell without a §0 row, fix the header table
    /// first; if you retire an edge, delete the cell here.
    #[test]
    fn ord_allow_edges_match_the_section0_table() {
        // (held, want) pairs allowed by the §0 table, verbatim:
        //   #2  active → wait_graph          #3  active → latch.park
        //   #8  cache.shard → stat_registry  #11 dep.shard → grow_lock
        //   #13 lowering → dep.shard         #20 init → {active, node_to_key,
        //                                        wait_graph, dep.shard,
        //                                        grow_lock, latch.park}
        let allowed: &[(Fam, Fam)] = &[
            (Fam::ActiveShard, Fam::WaitGraph),
            (Fam::ActiveShard, Fam::LatchPark),
            (Fam::CacheShard, Fam::StatRegistry),
            (Fam::DepShard, Fam::GrowLock),
            (Fam::Lowering, Fam::DepShard),
            (Fam::InitLock, Fam::ActiveShard),
            (Fam::InitLock, Fam::NodeToKey),
            (Fam::InitLock, Fam::WaitGraph),
            (Fam::InitLock, Fam::DepShard),
            (Fam::InitLock, Fam::GrowLock),
            (Fam::InitLock, Fam::LatchPark),
        ];
        let mut extra: Vec<(Fam, Fam)> = Vec::new();
        let mut missing: Vec<(Fam, Fam)> = Vec::new();
        for held in Fam::ALL {
            for want in Fam::ALL {
                let cell = ORD[held.idx()][want.idx()] == Rel::Allow;
                let listed = allowed.contains(&(held, want));
                match (cell, listed) {
                    (true, false) => extra.push((held, want)),
                    (false, true) => missing.push((held, want)),
                    _ => {}
                }
            }
        }
        assert!(
            extra.is_empty() && missing.is_empty(),
            "machine table and §0 header table diverged — extra: {extra:?}, missing: {missing:?}"
        );
    }

    /// M2 pin: self-nesting is never granted by the table.  Same-family
    /// re-holding is caught by the re-held assert in `check`, not by an
    /// Allow diagonal — a diagonal Allow would silently bypass that
    /// assert for any path that reads only the ORD matrix.
    #[test]
    fn ord_diagonal_is_all_forbid() {
        for f in Fam::ALL {
            assert_eq!(
                ORD[f.idx()][f.idx()],
                Rel::Forbid,
                "ORD[{f:?}][{f:?}] must stay Forbid: the re-held assert in \
                 `check` is the same-family guard, and a diagonal Allow would \
                 let an ORD-only reader bypass it"
            );
        }
    }

    /// M2 tripwire self-test: `check` must actually fire on an edge the
    /// table forbids (init_lock → stat_registry is NOT a §0 ✓ row) and
    /// must pass an edge it allows.  This pins that the whole TLS
    /// machinery (HELD counters, row scan, ORD lookup) is wired, not
    /// compiled out — a mis-gated `check` would otherwise make every
    /// "debug tripwire" claim in this module vacuous.
    #[test]
    fn tripwire_fires_on_forbidden_edge_and_passes_allowed_edge() {
        // Allowed edge passes silently.
        let _g = Acquired::acquire(Fam::InitLock, "m2.selftest.allowed");
        check(Fam::ActiveShard, "m2.selftest.allowed");
        drop(_g);

        // Forbidden edge (init → stat) must panic with the violation
        // message, then the TLS counters must be clean again (panic
        // unwinds the acquire).
        let result = std::panic::catch_unwind(|| {
            let _g = Acquired::acquire(Fam::InitLock, "m2.selftest.forbidden");
            check(Fam::StatRegistry, "m2.selftest.forbidden");
        });
        let err = result.expect_err("forbidden edge must trip the wire");
        let msg = panic_message_for_test(&err);
        assert!(
            msg.contains("lock-order violation"),
            "unexpected panic message: {msg}"
        );
        assert!(
            msg.contains("InitLock") && msg.contains("StatRegistry"),
            "violation must name both families: {msg}"
        );
        // HELD is clean again: the same edge must still be grantable.
        let _g = Acquired::acquire(Fam::InitLock, "m2.selftest.retake");
    }

    fn panic_message_for_test(err: &Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = err.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = err.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic>".to_string()
        }
    }
}
