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
//! | 2  | active_keys[shard].w → wait_graph.w          | ✓       | both sweep call sites (success finalize and `abort_with_poison`) now DEFER `clear_waiters_of_me` past the active guard — the nesting is currently DEAD on every path; the ORD cell stays Allow conservatively (a future in-lock site must not renumber) |
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
//! | 13 | lowering_lock → dep.shard[r]                 | retired | `propagate_lowering` takes NO shard lock (log rework #10); kept only as the historical row |
//! | 14 | dep.shard → lowering_lock                    | ✗       | both call sites are outside the shard critical section |
//! | 15 | dep.shard ↛ dep.shard (never coexist; reset is the sole exception: ascending, under quiesce.write, privacy-confined) | ✗ | module invariant |
//! | 16 | dep.shard → cache/stat/active/wait_graph     | ✗       | — |
//! | 17 | deferred → quiesce.w                         | ✗       | `drain_deferred` detaches the partition and drops the lock before entering `run_deferred_batch` |
//! | 18 | Atomic ordering (epoch→bit, generation chain, seqlock) | not a lock | comments + shuttle |
//! | 19 | init_lock → active_keys / node_to_key / wait_graph / dep.shard / grow_lock / latch.park | ✓ | `clear_all_impl` (poison sweep reaches latch.park via active → resolve) |
//!
//! Leaf families have all-Forbid rows unless a path legitimately holds a
//! family while taking them: `rev_log` (`ShardLog`'s mutex) is registered at
//! its four acquisition sites (push / collect tail / compact / reset) and is
//! taken under `init_lock` (clear_all's reset, #20) and under `lowering`
//! (propagate_lowering's reader collection, #21).  `QuiesceGate`'s internal
//! `writer`/`state` mutexes stay unregistered deliberately: they are the
//! root's own plumbing and are never co-held with table families.
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
    /// Leaf: `ShardLog`'s mutex (the append-only op log plus its per-slot
    /// index).  All-Forbid row: it is never co-held with a tabled family, so
    /// a future change that holds it across a shard acquisition (or vice
    /// versa) trips the tripwire instead of shipping silently.
    RevLog,
}
pub(crate) const FAMS: usize = 13;

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
        Fam::RevLog,
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
    // #13 RETIRED (Tier1): lowering_lock → dep.shard no longer exists —
    // `propagate_lowering` takes NO shard lock since the reverse-edge log
    // rework (#10).  The cell is deliberately left Forbid so any future
    // re-introduction of a shard read under a held lowering fails the
    // tripwire instead of passing silently on a stale Allow.
    // #20: init_lock → rev_log — `clear_all` resets each dep shard (and its
    // log) under the init lock.
    o[Fam::InitLock.idx()][Fam::RevLog.idx()] = Rel::Allow;
    // #21: lowering → rev_log — `propagate_lowering` collects readers from
    // the shard log while holding the lowering for the same node.
    o[Fam::Lowering.idx()][Fam::RevLog.idx()] = Rel::Allow;
    // (retired, never renumbered): active_keys[shard].w → vtable.active
    //   existed while the per-thread cycle state lived in a sharded mutex
    //   map (`QueryVTable::active`, committed under the shard write
    //   guard).  The state is now per-query-type thread-local storage —
    //   `commit` acquires no lock — so the edge and its family are
    //   deleted; the slot in the numbering is retired with it.
    // #19: init_lock → active_keys / node_to_key / wait_graph / dep.shard /
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

/// HELD row access.  Non-shuttle: one OS thread == one TLS view, a plain
/// per-thread row.  Under `feature = "shuttle"` the row is keyed by the
/// SHUTTLE TASK id (`shuttle::thread::current().id()`): shuttle
/// multiplexes every task onto one OS thread, so a per-OS-thread row
/// conflates tasks and trips phantoms both ways — a task that blocks
/// inside a registered window (contended `grow_lock` inside the
/// `DepShard` window, the poison sweep's `park.lock()` inside `InitLock`,
/// `wait_on`'s park inside `LatchPark`) leaks its row to the next task,
/// and even plain cross-task contention on one family (two same-key
/// requesters both in a cache-shard write section) reads as a re-hold.
/// Keying by task id makes the view EXACT — `h[i] > 0` means THIS task
/// holds family `i` — which is what kills the entire false-positive
/// class instead of exempting families one by one.  (The exempt-then-
/// generalize approach was tried and reverted: it masked real
/// violations, e.g. the tripwire self-test's own forbidden edge once the
/// holder family itself was exempt.)
///
/// Must be called INSIDE a shuttle execution (`Runner::run`) when the
/// shuttle feature is on: `shuttle::thread::current()` requires the
/// execution state.  The only callers outside one are the module
/// self-tests, which wrap their bodies in an execution — see `tests`.
#[cfg(any(debug_assertions, feature = "shuttle"))]
fn held_with<R>(f: impl FnOnce(&mut [u32; FAMS]) -> R) -> R {
    #[cfg(not(feature = "shuttle"))]
    {
        thread_local! {
            static HELD: RefCell<[u32; FAMS]> = const { RefCell::new([0; FAMS]) };
        }
        HELD.with(|h| f(&mut h.borrow_mut()))
    }
    #[cfg(feature = "shuttle")]
    {
        thread_local! {
            static HELD: RefCell<std::collections::HashMap<shuttle::thread::ThreadId, [u32; FAMS]>> =
                RefCell::new(std::collections::HashMap::new());
        }
        HELD.with(|m| {
            let id = shuttle::thread::current().id();
            let mut m = m.borrow_mut();
            let row = m.entry(id).or_insert_with(|| [0; FAMS]);
            f(row)
        })
    }
}

/// Format the full HELD vector for a panic message: a phantom (leaked row)
/// vs a real violation is diagnosable in one look instead of by schedule
/// replay.
#[cfg(any(debug_assertions, feature = "shuttle"))]
fn held_vector(h: &[u32; FAMS]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    for (i, &n) in h.iter().enumerate() {
        if n > 0 {
            let _ = write!(s, " {:?}={n}", Fam::ALL[i]);
        }
    }
    if s.is_empty() {
        s.push_str(" (none)");
    }
    s
}

/// §0-table pins, run in EVERY profile: these two tests read only the
/// const `ORD` — no TLS, no `check` — so gating them behind
/// `any(debug_assertions, shuttle)` would let a plain
/// `cargo test --release` silently drop the machine-table↔header-table
/// pin.  The tripwire self-test below needs the machinery live and stays
/// gated with the rest.
mod table_pins {
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
        //   #13 lowering → dep.shard is RETIRED (see ORD): no cell.
        //   #19 init → {active, node_to_key, wait_graph, dep.shard,
        //               grow_lock, latch.park}
        //   #20 init → rev_log             #21 lowering → rev_log
        let allowed: &[(Fam, Fam)] = &[
            (Fam::ActiveShard, Fam::WaitGraph),
            (Fam::ActiveShard, Fam::LatchPark),
            (Fam::CacheShard, Fam::StatRegistry),
            (Fam::DepShard, Fam::GrowLock),
            (Fam::InitLock, Fam::RevLog),
            (Fam::Lowering, Fam::RevLog),
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
    if let Some(msg) = check_msg(f, site) {
        panic!("{msg}");
    }
}

/// The tripwire's verdict, WITHOUT panicking: `Some(message)` when taking
/// `f` at `site` would violate the re-held or lock-order rule.  Split out
/// so the self-test can pin the verdict under shuttle, where panics are
/// intercepted by the runner and cannot be caught (`catch_unwind` on the
/// same task never sees them; a spawned task's panic aborts the run
/// before `join` observes it).  `check` is the only production consumer —
/// the message text lives here so the self-test pins exactly what fires.
#[cfg(any(debug_assertions, feature = "shuttle"))]
fn check_msg(f: Fam, site: &'static str) -> Option<String> {
    held_with(|h| {
        // Fast path: `check` runs on EVERY registered lock acquisition, and
        // the overwhelmingly common state is nothing held at all — skip the
        // 12-cell row scan when the vector is empty.  (A u32 bitmask with
        // `trailing_zeros` iteration would also work; FAMS=12 fits, but the
        // zero-skip keeps the code uniform and the scan is already cheap.)
        if h.iter().all(|&n| n == 0) {
            return None;
        }
        // Unconditional asserts, NOT debug_assert (see `check`): the gate
        // covers --release --features shuttle, where debug_assertions are
        // OFF.  (No family exemptions: `held_with` keys the row by SHUTTLE
        // TASK id under the shuttle feature, so the view is exact — a leak
        // or cross-task contention can no longer read as a violation.)
        if h[f.idx()] != 0 {
            return Some(format!(
                "lock family re-held: {f:?} @ {site} (HELD:{})",
                held_vector(h)
            ));
        }
        for (row, &n) in h.iter().enumerate() {
            if n > 0 && ORD[row][f.idx()] != Rel::Allow {
                return Some(format!(
                    "lock-order violation: {:?} → {f:?} @ {site} (see \
                     sync_order.rs §0 table; HELD:{})",
                    Fam::ALL[row],
                    held_vector(h)
                ));
            }
        }
        None
    })
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
#[cfg(any(debug_assertions, feature = "shuttle"))]
pub(crate) struct Acquired(Fam);

/// Release build: ZST companion of [`Acquired`].  The debug/shuttle variant
/// above carries its `Fam` byte and a `Drop` impl; even an empty `Drop`
/// stores the payload and runs a destructor — a few instructions per
/// acquire in an otherwise lock-heavy path.  The ZST keeps every `_tw`
/// binding compiling unchanged while letting LLVM elide the guard whole.
#[cfg(not(any(debug_assertions, feature = "shuttle")))]
pub(crate) struct Acquired;

impl Acquired {
    #[cfg(any(debug_assertions, feature = "shuttle"))]
    #[inline]
    pub(crate) fn acquire(f: Fam, site: &'static str) -> Acquired {
        check(f, site);
        held_with(|h| h[f.idx()] += 1);
        Acquired(f)
    }

    /// Release no-op: the guard is a ZST, so the `Fam` payload is dropped
    /// before it is ever stored — `acquire` keeps the parameters so every
    /// `_tw` binding keeps compiling unchanged (see the gate note on
    /// `HELD`).
    #[cfg(not(any(debug_assertions, feature = "shuttle")))]
    #[inline(always)]
    pub(crate) fn acquire(_f: Fam, _site: &'static str) -> Acquired {
        Acquired
    }
}

#[cfg(any(debug_assertions, feature = "shuttle"))]
impl Drop for Acquired {
    /// INVARIANT: `Acquired` is always a STACK-LOCAL guard — the drop runs
    /// on the same thread that acquired, before thread exit, so the
    /// `held_with` TLS access below never observes a destroyed slot.  If a
    /// future refactor ever stores an `Acquired` in a struct field or
    /// static (any path where Drop could run during TLS destruction),
    /// `held_with`'s `.with()` panics during unwind and ABORTS the
    /// process — do not make that refactor without revisiting this drop.
    #[inline]
    fn drop(&mut self) {
        held_with(|h| {
            // Underflow guard: a double-drop or
            // mismatched acquire/release would wrap the u32 to `u32::MAX`,
            // silently poisoning the tripwire for every later acquisition in
            // the slot (the self-test would then report re-held for legal
            // acquisitions). This must be UNCONDITIONAL: the whole module is
            // compiled in under `--release --features shuttle`, where
            // `debug_assert!` is a no-op — the exact configuration the
            // tripwire exists to police. Same precedent as `check_msg`.
            if h[self.0.idx()] == 0 {
                panic!(
                    "Acquired::drop underflow: double-drop or unbalanced release of {:?}",
                    self.0
                );
            }
            h[self.0.idx()] -= 1;
        });
    }
}

/// Registration entry for mechanism-B guards (DepShardW/DepShardR/GrowGuard/
/// LoweringGuard/CacheShard*): construction inc, Drop dec.
#[cfg(any(debug_assertions, feature = "shuttle"))]
#[inline]
pub(crate) fn held_inc(f: Fam) {
    held_with(|h| h[f.idx()] += 1);
}

#[cfg(any(debug_assertions, feature = "shuttle"))]
#[inline]
pub(crate) fn held_dec(f: Fam) {
    held_with(|h| h[f.idx()] -= 1);
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

    // The §0-table pins (`ord_allow_edges_match_the_section0_table`,
    // `ord_diagonal_is_all_forbid`) live in the UNGATED `table_pins`
    // module above: they read only the const `ORD` and must run in every
    // profile.  Only machinery-dependent tests stay here.

    /// M2 tripwire self-test: `check` must actually fire on an edge the
    /// table forbids (init_lock → stat_registry is NOT a §0 ✓ row) and
    /// must pass an edge it allows.  This pins that the whole TLS
    /// machinery (HELD counters, row scan, ORD lookup) is wired, not
    /// compiled out — a mis-gated `check` would otherwise make every
    /// "debug tripwire" claim in this module vacuous.
    ///
    /// Under the shuttle feature the machinery runs INSIDE a runner
    /// execution: `held_with` keys the row by shuttle task id, which
    /// requires the execution state — so the body is wrapped in a minimal
    /// `Runner::run`.  Non-shuttle profiles run the body directly.
    #[test]
    fn tripwire_fires_on_forbidden_edge_and_passes_allowed_edge() {
        #[cfg(feature = "shuttle")]
        {
            use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
            static FLAG: AtomicBool = AtomicBool::new(false);
            let mut config = shuttle::Config::default();
            config.stack_size = 2 * 1024 * 1024;
            let runner =
                shuttle::Runner::new(shuttle::scheduler::PctScheduler::new(8, 50), config);
            runner.run(|| {
                // PctScheduler only counts STEPS when >1 task is runnable —
                // spawn+immediate-join never overlaps, so the child must
                // coexist with the parent (see tmp_probe note).
                let t = shuttle::thread::spawn(|| {
                    while !FLAG.load(SeqCst) {
                        shuttle::thread::yield_now();
                    }
                });
                for _ in 0..5 {
                    shuttle::thread::yield_now();
                }
                FLAG.store(true, SeqCst);
                t.join().expect("helper task must not panic");
                tripwire_body();
            });
        }
        #[cfg(not(feature = "shuttle"))]
        tripwire_body();
    }

    fn tripwire_body() {
        // Allowed edge passes silently.
        let _g = Acquired::acquire(Fam::InitLock, "m2.selftest.allowed");
        check(Fam::ActiveShard, "m2.selftest.allowed");
        assert!(
            check_msg(Fam::ActiveShard, "m2.selftest.allowed").is_none(),
            "the allowed edge must produce no verdict"
        );
        drop(_g);

        // Forbidden edge (init → stat) must produce a violation naming both
        // families.  Under shuttle, panics are intercepted by the runner
        // and cannot be caught (`catch_unwind` on the same task never sees
        // them; a spawned task's panic aborts the run before `join`
        // observes it), so the verdict is pinned via `check_msg` — the
        // single source `check` panics from.  Non-shuttle additionally
        // pins the panicking path itself via `catch_unwind`.
        #[cfg(feature = "shuttle")]
        {
            let _g = Acquired::acquire(Fam::InitLock, "m2.selftest.forbidden");
            let msg = check_msg(Fam::StatRegistry, "m2.selftest.forbidden")
                .expect("forbidden edge must trip the wire");
            assert_violation_message(&msg);
            drop(_g);
        }
        #[cfg(not(feature = "shuttle"))]
        {
            let result = std::panic::catch_unwind(|| {
                let _g = Acquired::acquire(Fam::InitLock, "m2.selftest.forbidden");
                check(Fam::StatRegistry, "m2.selftest.forbidden");
            });
            let err = result.expect_err("forbidden edge must trip the wire");
            let msg = panic_message_for_test(&err);
            assert_violation_message(&msg);
        }
        // HELD is clean again: the same edge must still be grantable.
        let _g = Acquired::acquire(Fam::InitLock, "m2.selftest.retake");
    }

    fn assert_violation_message(msg: &str) {
        assert!(
            msg.contains("lock-order violation"),
            "unexpected panic message: {msg}"
        );
        assert!(
            msg.contains("InitLock") && msg.contains("StatRegistry"),
            "violation must name both families: {msg}"
        );
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
