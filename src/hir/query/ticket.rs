//! # Mechanism B: typestate lock-acquisition tickets for the dep family
//! (linear tickets)
//!
//! The lock topology of the dep family (dep shard / grow_lock /
//! lowering_lock) is pinned at compile time by this module's state types
//! (the compile face of §0 table #11–#15):
//!
//! - `Idle → OneShard` (`CanTakeShard`): a second dep shard = E0277
//!   (`OneShard`/`ShardAndGrow`/`LoweringShard` deliberately do not impl).
//! - `OneShard → ShardAndGrow` (`CanTakeGrow`): shard→grow ✓;
//!   inside the grow critical section, acquiring any ticket-requiring lock
//!   does not compile (no ticket in the closure body).
//! - `Idle → LoweringHeld` (`CanAcquireLowering`): taking lowering while
//!   holding a shard / holding grow ✗ (table #14); lowering→shard read ✓
//!   (`LoweringHeld` implements `CanTakeShard` → `LoweringShard`, table #13).
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
//! `Held::fresh` / `DepGraph::ticket` are the family's sole minting points:
//! every `dep_graph` method that BEGINS a shard interaction mints on its
//! first line (public API / cross-family entry methods — not just the
//! "first line of entry" wording: practically every shard-touching method
//! qualifies), and tests may mint with a justifying comment.  Closed at
//! compile time outside the crate (pub(crate)); inside the crate, audited
//! by grep:
//!
//! ```text
//! grep -rnE 'Held::<[A-Za-z]+>::fresh\(\)|Held\(|\.ticket\(\)' src/hir/query/
//! ```
//!
//! (`Held\(` stays in the pattern as a tripwire: the transition functions
//! (`release_*`, `take_*`'s state hand-off) route their state moves
//! through `Held::of` — same module, zero cost — so any direct
//! `Held(PhantomData)` construction the grep finds is NEW code outside
//! those functions, needing manual review.  Outside this module the
//! private field makes direct construction impossible anyway; the grep
//! keeps an in-module regression visible.)
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
/// minting) or `of()` (transition state moves), both `#[doc(hidden)]`.
pub(crate) struct Held<St>(PhantomData<*mut St>);

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

    /// Transition constructor for the family's state hand-off functions
    /// (`take_*` returning the next state, `release_*` returning the prior
    /// state).  These are state MOVES, not entry methods — they must not go
    /// through `fresh()` — but direct field construction outside this
    /// module no longer compiles, so they route through here.  The
    /// module-header grep still flags `Held::of` for manual audit.
    #[doc(hidden)]
    #[inline]
    pub(crate) fn of<U>() -> Held<U> {
        Held(PhantomData)
    }
}

// ── dep family states ────────────────────────────────────────────────

/// ∅: no dep-family lock held.
pub(crate) struct Idle;
/// Exactly one dep shard (read/write carried by the guard type).
pub(crate) struct OneShard;
/// Shard + grow (`allocate_node_index` shape, table #11/#12).
pub(crate) struct ShardAndGrow;
/// lowering_lock held.
pub(crate) struct LoweringHeld;
/// lowering_lock + one shard read (`propagate_lowering` cascade shape, table #13).
pub(crate) struct LoweringShard;

/// dep shard: only open to ∅ and lowering-in-hand (tables #13/#15).
/// `OneShard` / `ShardAndGrow` / `LoweringShard` deliberately do not
/// impl → second shard = E0277.
pub(crate) trait CanTakeShard {
    type After;
}
impl CanTakeShard for Idle {
    type After = OneShard;
}
impl CanTakeShard for LoweringHeld {
    type After = LoweringShard;
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

// ── Guards ───────────────────────────────────────────────────────────

/// dep shard write guard: Drop only covers panic/unwind (the normal path uses
/// the explicit `release_*` to return the ticket to its prior state);
/// construction/destruction also registers a live count with the tripwire.
pub(crate) struct DepShardW<'a>(Option<crate::sync::RwLockWriteGuard<'a, DepShard>>);

impl<'a> DepShardW<'a> {
    pub(crate) fn new(g: crate::sync::RwLockWriteGuard<'a, DepShard>) -> Self {
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
    pub(crate) fn new(g: crate::sync::RwLockReadGuard<'a, DepShard>) -> Self {
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
    pub(crate) fn new(g: crate::sync::MutexGuard<'a, ()>) -> Self {
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
    pub(crate) fn new(g: crate::sync::MutexGuard<'a, ()>) -> Self {
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

/// Consume the ticket by value: after the call the old state has been moved,
/// so no stale-ticket hole exists.
#[inline]
pub(crate) fn release_shard_w(g: DepShardW<'_>, _t: Held<OneShard>) -> Held<Idle> {
    drop(g);
    Held::<Idle>::of()
}

#[inline]
pub(crate) fn release_shard_r(g: DepShardR<'_>, _t: Held<OneShard>) -> Held<Idle> {
    drop(g);
    Held::<Idle>::of()
}

#[inline]
pub(crate) fn release_cascade_shard_r(
    g: DepShardR<'_>,
    _t: Held<LoweringShard>,
) -> Held<LoweringHeld> {
    drop(g);
    Held::<LoweringHeld>::of()
}

#[inline]
pub(crate) fn release_grow(g: GrowGuard<'_>, _t: Held<ShardAndGrow>) -> Held<OneShard> {
    drop(g);
    Held::<OneShard>::of()
}

#[inline]
pub(crate) fn release_lowering(g: LoweringGuard<'_>, _t: Held<LoweringHeld>) -> Held<Idle> {
    drop(g);
    Held::<Idle>::of()
}
