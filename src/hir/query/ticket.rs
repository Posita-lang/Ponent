//! # Mechanism B: typestate lock-acquisition tickets for the dep family
//! (linear tickets)
//!
//! The lock topology of the dep family (dep shard / grow_lock /
//! lowering_lock) is pinned at compile time by this module's state types
//! (the compile face of §0 table #11–#15):
//!
//! - `Idle → OneShard` (`CanTakeShard`): a second dep shard = E0277
//!   (`OneShard`/`ShardAndGrow`/`LoweringHeld` deliberately do not impl).
//! - `OneShard → ShardAndGrow` (`CanTakeGrow`): shard→grow ✓;
//!   inside the grow critical section, acquiring any ticket-requiring lock
//!   does not compile (no ticket in the closure body).
//! - `Idle → LoweringHeld` (`CanAcquireLowering`): taking lowering while
//!   holding a shard / holding grow ✗ (table #14).  (The old
//!   lowering→shard-read cascade state is retired — see the note at
//!   `LoweringHeld`.)
//!
//! Tickets are passed linearly by value (move semantics guarantee the old
//! state cannot be reused after acquisition — the stale-ticket hole of
//! "take the old Idle ticket and open a second shard" does not exist by
//! construction). `Held<St>`'s `*mut St` ⇒ !Send + !Sync + invariant over
//! `St`: tickets are thread-bound just like lock holds, and the state cannot
//! be smuggled around the transition table by covariance.
//!
//! ## Minting discipline
//!
//! `Held::fresh` / `DepGraph::ticket` are the family's sole ENTRY minting
//! points: every `dep_graph` method that BEGINS a shard interaction mints on
//! its first line (public API / cross-family entry methods — not just the
//! "first line of entry" wording: practically every shard-touching method
//! qualifies), and tests may mint with a justifying comment.  Every other
//! `Held` value is produced by a TYPED TRANSITION, not a constructor:
//!
//! - the acquisition edges are the `CanTake*` traits' `transition(self)`
//!   method — it CONSUMES the source ticket and can only mint its own
//!   `After` state, so a transition table entry and a minting point are the
//!   same item (`Held<Idle>` cannot mint `ShardAndGrow` — no such impl);
//! - the release edges are per-state `release_*` methods that also consume
//!   the source ticket by value and return the prior state.
//!
//! The former unconstrained constructor `Held::of<U>()` (which could mint
//! ANY state from nothing — e.g. a `Held<OneShard>` without a shard, making
//! `take_grow` callable outside the shard section with no tripwire firing)
//! is DELETED: outside this module the private field makes direct
//! construction impossible, and inside it every mint is either `fresh()` on
//! an entry method or a typed transition whose existence IS the edge.
//!
//! Panic safety: tickets are function-local; every call chain re-mints from
//! the entry, so unwind merely interrupts the current chain, the guard Drop
//! goes through the tripwire's `held_dec`, and no recovery logic is needed.
//! ZST + `PhantomData` guarantees zero cost and full inlinability.
//!
//! **Changing the lock topology requires synchronizing all three places:
//! sync_order.rs §0 table / this module's impl blocks / dep_graph module
//! docs.**

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use super::dep_graph::DepShard;
use super::sync_order::{self, Fam};

/// Linear lock-acquisition ticket (see module header).  The field is
/// PRIVATE: direct `Held(PhantomData)` construction compiles only inside
/// this module — every other site must go through `fresh()` (entry
/// minting, `#[doc(hidden)]`) or a typed transition (the `CanTake*`
/// traits' `transition` and the per-state `release_*` methods, both of
/// which CONSUME the source ticket).
pub(crate) struct Held<St>(PhantomData<*mut St>);

impl<St: std::fmt::Debug> std::fmt::Debug for Held<St> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Held<{}>", std::any::type_name::<St>())
    }
}

impl<St> Held<St> {
    /// Minting discipline: may only appear on the first line of each
    /// family's "entry methods" and in tests with a justifying comment
    /// (grep audit in module header).  An entry method that hands the
    /// ticket back mid-body may RE-mint for each re-entry — e.g.
    /// `DefaultCache::probe`'s two-phase shape releases the phase-1 read
    /// lock and mints fresh again for the alloc and once more for the
    /// phase-2 write lock; every one of those mints enters a state where
    /// nothing else in the family is held, which is the invariant the
    /// discipline is actually protecting.)
    #[doc(hidden)]
    #[inline]
    pub(crate) fn fresh() -> Self {
        Held(PhantomData)
    }
}

/// ∅: no dep-family lock held.
pub(crate) struct Idle;
/// Exactly one dep shard (read/write carried by the guard type).
pub(crate) struct OneShard;
/// Shard + grow (`allocate_node_index` shape, table #11/#12).
pub(crate) struct ShardAndGrow;
/// lowering_lock held.
pub(crate) struct LoweringHeld;
// (The `LoweringShard` state — lowering + one shard read, old table #13 —
// is RETIRED: the reverse-edge log rework (#10) removed the shard lock
// from `propagate_lowering`'s reader walk, so no lock state pairs with a
// held lowering any more.  §0 row #13 is retired with it and its ORD cell
// is left Forbid, so any future re-introduction trips the tripwire.)

/// dep shard: only open to ∅.  `OneShard` / `ShardAndGrow` /
/// `LoweringHeld` deliberately do not impl → second shard = E0277.
///
/// The trait's `transition` IS the acquisition edge: it consumes the
/// source ticket and mints only its own `After` state, so a table entry
/// and a minting point are the same item — there is no constructor that
/// can mint an arbitrary state from nothing.
pub(crate) trait CanTakeShard {
    type After;
}
impl CanTakeShard for Idle {
    type After = OneShard;
}

/// grow: only open to `OneShard` → shard→grow ✓ (table #11); `ShardAndGrow`
/// has no impl → acquiring any ticket-requiring lock inside the grow
/// critical section does not compile (table #12).
pub(crate) trait CanTakeGrow {
    type After;
}
impl CanTakeGrow for OneShard {
    type After = ShardAndGrow;
}

/// lowering_lock: only open to ∅ → taking lowering while holding a shard /
/// holding grow ✗ (table #14).
pub(crate) trait CanAcquireLowering {
    type After;
}
impl CanAcquireLowering for Idle {
    type After = LoweringHeld;
}

// ── cache family states ──────────────────────────────────────────────

/// No cache shard held — the pre-condition proof for cache `probe`'s alloc
/// closure (the compile face of contract 3: the two-phase shape where alloc
/// is always outside the lock).
pub(crate) struct CacheNoShard;
/// Exactly one cache shard inside a read/write critical section.
pub(crate) struct CacheShardHeld;

pub(crate) trait CanTakeCacheShard {
    type After;
}
impl CanTakeCacheShard for CacheNoShard {
    type After = CacheShardHeld;
}

// ── Typed acquisition transitions ────────────────────────────────────
//
// The acquisition edges live as INHERENT methods on `Held<St>` gated by
// the edge traits above: each consumes the source ticket by value and can
// only mint its own `After` state, so a table entry and a minting point
// are the same item — `Held<Idle>` has no method that mints
// `ShardAndGrow`, and a second shard from `OneShard`/`ShardAndGrow`/
// `LoweringHeld` does not compile.  (Inherent, not trait methods: the
// per-edge names keep the `Idle` overlap between the dep-shard and
// lowering tables from colliding, and inherent methods resolve at the
// call site without importing the traits.)

impl<St: CanTakeShard> Held<St> {
    /// dep shard acquisition edge (table #15): `Idle → OneShard`.
    #[inline]
    pub(crate) fn take_dep_shard(self) -> Held<St::After> {
        Held(PhantomData)
    }
}

impl<St: CanTakeGrow> Held<St> {
    /// grow acquisition edge (table #11): `OneShard → ShardAndGrow`.
    #[inline]
    pub(crate) fn take_grow(self) -> Held<St::After> {
        Held(PhantomData)
    }
}

impl<St: CanAcquireLowering> Held<St> {
    /// lowering acquisition edge (table #14): `Idle → LoweringHeld`.
    #[inline]
    pub(crate) fn acquire_lowering(self) -> Held<St::After> {
        Held(PhantomData)
    }
}

impl<St: CanTakeCacheShard> Held<St> {
    /// cache shard acquisition edge (contract 3): `CacheNoShard → CacheShardHeld`.
    #[inline]
    pub(crate) fn take_cache_shard(self) -> Held<St::After> {
        Held(PhantomData)
    }
}

// ── Guards ───────────────────────────────────────────────────────────

/// dep shard write guard: Drop only covers panic/unwind (the normal path uses
/// the explicit `release_*` to return the ticket to its prior state);
/// construction/destruction also registers a live count with the tripwire.
pub(crate) struct DepShardW<'a>(Option<crate::sync::RwLockWriteGuard<'a, DepShard>>);

impl<'a> DepShardW<'a> {
    /// Sole construction path: `dep_graph::take_shard_w`, which pairs this
    /// guard with the `Held<OneShard>` ticket in the same expression —
    /// "guard exists ⇒ ticket exists" is enforced structurally, not just by
    /// grep.  Kept `pub(super)` (visible only inside `hir::query`, where
    /// dep_graph lives) so a stray `new` outside the pairing function is a
    /// visibility error, not an audit miss.
    pub(super) fn new(g: crate::sync::RwLockWriteGuard<'a, DepShard>) -> Self {
        sync_order::held_inc(Fam::DepShard);
        DepShardW(Some(g))
    }
}

impl Deref for DepShardW<'_> {
    type Target = DepShard;
    #[inline]
    fn deref(&self) -> &DepShard {
        self.0.as_ref().expect("guard alive until Drop").deref()
    }
}

impl DerefMut for DepShardW<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut DepShard {
        self.0.as_mut().expect("guard alive until Drop").deref_mut()
    }
}

impl Drop for DepShardW<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.take();
        sync_order::held_dec(Fam::DepShard); // panic path unlock
    }
}

/// dep shard read guard (isomorphic, Deref only).
pub(crate) struct DepShardR<'a>(Option<crate::sync::RwLockReadGuard<'a, DepShard>>);

impl<'a> DepShardR<'a> {
    /// Same pairing contract as [`DepShardW::new`] — constructed only by
    /// `dep_graph::take_shard_r`.
    pub(super) fn new(g: crate::sync::RwLockReadGuard<'a, DepShard>) -> Self {
        sync_order::held_inc(Fam::DepShard);
        DepShardR(Some(g))
    }
}

impl Deref for DepShardR<'_> {
    type Target = DepShard;
    #[inline]
    fn deref(&self) -> &DepShard {
        self.0.as_ref().expect("guard alive until Drop").deref()
    }
}

impl Drop for DepShardR<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.take();
        sync_order::held_dec(Fam::DepShard);
    }
}

/// grow_lock guard (leaf lock, table #12).
pub(crate) struct GrowGuard<'a>(Option<crate::sync::MutexGuard<'a, ()>>);

impl<'a> GrowGuard<'a> {
    /// Same pairing contract as [`DepShardW::new`] — constructed only by
    /// `dep_graph::take_grow`.
    pub(super) fn new(g: crate::sync::MutexGuard<'a, ()>) -> Self {
        sync_order::held_inc(Fam::GrowLock);
        GrowGuard(Some(g))
    }
}

impl Drop for GrowGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.take();
        sync_order::held_dec(Fam::GrowLock);
    }
}

/// lowering_lock guard.
pub(crate) struct LoweringGuard<'a>(Option<crate::sync::MutexGuard<'a, ()>>);

impl<'a> LoweringGuard<'a> {
    /// Same pairing contract as [`DepShardW::new`] — constructed only by
    /// `dep_graph::take_lowering`.
    pub(super) fn new(g: crate::sync::MutexGuard<'a, ()>) -> Self {
        sync_order::held_inc(Fam::Lowering);
        LoweringGuard(Some(g))
    }
}

impl Drop for LoweringGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.take();
        sync_order::held_dec(Fam::Lowering);
    }
}

// ── Explicit release: return the ticket to its prior state ──────────
//
// Each release edge is a method on the HELD state that consumes the
// ticket by value (`self`) and returns the prior state — the same
// typed-transition rule as the `CanTake*` traits.  There is no free
// function taking an ignored `_t` parameter anymore: the source ticket
// must be in hand to mint the prior state, so a release cannot be
// invoked without actually holding the state it claims to release.

impl Held<OneShard> {
    /// Release the shard write guard; ∅ again (table #15 inverse).
    #[inline]
    pub(crate) fn release_shard_w(self, g: DepShardW<'_>) -> Held<Idle> {
        drop(g);
        Held(PhantomData)
    }

    /// Release the shard read guard; ∅ again.
    #[inline]
    pub(crate) fn release_shard_r(self, g: DepShardR<'_>) -> Held<Idle> {
        drop(g);
        Held(PhantomData)
    }
}

impl Held<ShardAndGrow> {
    /// Release grow; one shard still held (table #11 inverse).
    #[inline]
    pub(crate) fn release_grow(self, g: GrowGuard<'_>) -> Held<OneShard> {
        drop(g);
        Held(PhantomData)
    }
}

impl Held<LoweringHeld> {
    /// Release lowering; ∅ again (table #14 inverse).
    #[inline]
    pub(crate) fn release_lowering(self, g: LoweringGuard<'_>) -> Held<Idle> {
        drop(g);
        Held(PhantomData)
    }
}

impl Held<CacheShardHeld> {
    /// Release a cache shard guard; `CacheNoShard` again (the cache
    /// family's single release edge, shared by the read and write guards).
    #[inline]
    pub(crate) fn release_cache_shard(self) -> Held<CacheNoShard> {
        Held(PhantomData)
    }
}

// ── Thread-binding pins ──────────────────────────────────────────────
// The module claims "tickets are thread-bound just like lock holds":
// `Held<St>` is pinned by its `*mut St` field, but the GUARDS' !Send was
// only an emergent property of the inner `crate::sync` guards (std /
// shuttle both make their guard types !Send).  Pin it explicitly so a
// future `Send` guard cannot silently break the thread-binding story —
// these assertions fail to compile the moment a guard becomes Send.
static_assertions::assert_not_impl_all!(DepShardW<'static>: Send);
static_assertions::assert_not_impl_all!(DepShardR<'static>: Send);
static_assertions::assert_not_impl_all!(GrowGuard<'static>: Send);
static_assertions::assert_not_impl_all!(LoweringGuard<'static>: Send);
