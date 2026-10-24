//! # Query System — Cached, Incremental Compiler Queries
//!
//! Analogous to `rustc_middle::query` and `rustc_query_system`.
//! Provides a generic framework for defining compiler queries with
//! automatic memoization (caching) and dependency tracking.
//!
//! ## Architecture
//!
//! - [`Query`] trait — defines a query with a `Key`, `Value`, and
//!   `Cache` type.  Each query selects its own cache strategy.
//! - [`QueryCacheType`] trait — the cache interface.  Different key
//!   types use different cache implementations for optimal performance.
//! - [`DefaultCache`] — HashMap-backed cache with LRU-style (CLOCK) eviction.
//!   Used for most query key types.
//! - [`SingleCache`] — single-entry cache for `()` keys.
//! - [`DefIdCache`] — `Vec`-indexed cache for `DefId` keys (O(1)).
//! - [`QueryDescriptor`] — metadata for a query (name, modifiers).
//! - [`QuerySystem`] — lazily-built cache of per-query vtables, indexed by `TypeId`.
//!   The authoritative query set is the `for_each_query!` manifest.
//! - [`QueryProvider`] trait — implementors provide the `compute` logic.
//!
//! ## Usage
//!
//! ```ignore
//! struct TypeOfQuery;
//! impl<'q, 'src: 'q> Query<'q, 'src> for TypeOfQuery {
//!     type Key = DefId;
//!     type Value = TypeId;
//!     type Cache = DefIdCache<TypeId>;
//!     fn compute<S: Scheduler>(key: &DefId, provider: &dyn QueryProvider<'q, 'src, S>) -> TypeId { ... }
//!     fn descriptor() -> QueryDescriptor { ... }
//! }
//! ```
//!
//! ## Contract-violation policy: fatal vs degraded
//!
//! Every guard in this module reports a provider/system mispair, but they
//! do NOT all respond the same way.  The asymmetry is deliberate, not
//! historical accident:
//!
//! | Violation | Response | Rationale |
//! |---|---|---|
//! | Cross-system frame nesting (`InFlightGuard::acquire`) | **panic**, both builds | Deadlocks in both builds; the only safe end state is to stop now. |
//! | `mutate_world` / `run_deferred_batch` re-entrancy (the `IN_DRAIN` tripwires) | **panic**, both builds | Would self-deadlock on a non-reentrant write lock; `DrainGuard` cannot repair it. |
//! | Foreign deferred mutation with no live target frame | **counted** + release-only `eprintln!` | Starved, not unsafe; the closure still runs when the target drops. |
//! | `DepGraph::read` cross-system edge | **panic** in debug, **skip + count** in release | Recording the edge into the wrong graph is corruption; skipping is permanent staleness for one miss, so release degrades instead of killing the embedder. |
//!
//! An embedder who decides a specific panic is intolerable has to change the
//! ENTRY, not the symptom: the panic at `InFlightGuard::acquire` exists
//! because the mispair has no safe continuation, so suppressing it converts a
//! diagnosable hang into silent corruption.

use arc_swap::ArcSwap;
use std::any::TypeId;
use std::sync::Arc;
// Compiler-internal maps use `FxHasher`, not the default SipHash: keys
// here are `TypeId`s, `DefId`s and interned symbols, never
// attacker-controlled strings, so the DoS resistance of SipHash buys
// nothing while its longer hash costs on every probe.  Same choice
// rustc makes for its own query/dep-graph maps.
use crate::sync::{Mutex, RwLock};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};

pub mod dep_graph;
pub mod erase;
pub mod job;
pub mod queries;
pub mod scheduler;
pub(crate) mod sync_order;
pub(crate) mod ticket;

pub use dep_graph::ContentClass;
pub use erase::{Arena, Erasable, Owned};
pub use scheduler::{NoOpScheduler, SchedulePoint, Scheduler, TestScheduler};

#[cfg(all(test, not(feature = "shuttle")))]
mod defense_tests;

#[cfg(all(test, feature = "shuttle"))]
mod shuttle_tests;

/// Status of a query key that is currently being computed by some thread.
///
/// Analogous to Rust's `ActiveKeyStatus` in `plumbing.rs`.
#[derive(Debug)]
enum ActiveKeyStatus {
    /// Some thread is already evaluating the query for this key.
    /// The enclosed `QueryJob` can be used to wait for it to finish.
    Started(job::QueryJob),
    /// The query panicked.  Other threads waiting on this will get an error.
    Poisoned,
}

/// One cross-thread wait edge: thread `waiter` (the map key) is about to
/// block on the job with id `job_id`, owned by `owner`.  Keeping the job
/// id lets the owner clear exactly the edges for the job it completed —
/// edges pointing at the same owner for its OTHER in-flight jobs (nested
/// computes, e.g. `ChainQ(slot,d)` owning every depth at once) survive,
/// so waiters blocked on those stay visible to the cycle walker.
#[derive(Debug, Clone)]
struct WaitEdge {
    owner: std::thread::ThreadId,
    job_id: job::QueryJobId,
    /// Clone of the waited-on job's latch.  A cycle walk treats an edge
    /// whose latch is already resolved as ABSENT: the job completed, so
    /// the wait it records is over.  This makes a phantom edge (one
    /// registered after its owner already completed, in the window
    /// before the registrant retracts it) invisible to every OTHER
    /// thread's walk, not just to the registrant's own `job_still_active`
    /// retraction — a third thread can no longer traverse a dead edge
    /// into a spurious cycle verdict.
    ///
    /// The completion path must therefore resolve the latch BEFORE it
    /// clears/removes the entry — see `QuerySystem::get_shared` (success
    /// path) and `poison_active` (failure path); the reverse order would
    /// leave a completed job's edges walker-visible with unresolved
    /// latches.
    latch: job::QueryLatch,
}

/// Outcome of active-key acquisition in `QuerySystem::acquire_key_slot`.
enum KeySlot<'q, 'src, Q: Query<'q, 'src>> {
    /// This thread owns the computation — it must run `compute` and publish.
    Compute,
    /// Another thread owns the computation; wait on its job latch.
    Wait(job::QueryJob),
    /// Same-thread cycle with a per-query handler present: the typed
    /// active prefix ferried out from under the cycle check — `get_shared`
    /// runs the handler at the slot match, where no cycle-state borrow is
    /// held.
    CycleContext { stack: Vec<Q::Key> },
}

/// A suspended cross-thread cycle: the breaking (waiter) side parked its
/// latch wait, snapshotted its own committed prefix, and captured the
/// wait-graph chain that closed back on itself.  The waiter *resumes* by
/// running its query's loop processor against this context (local recovery
/// + precise pinpointing) instead of aborting with a bare "cycle across
/// threads" error.
///
/// This is the "resumable waiter" half of the loop-recovery mechanism:
/// `wait_for_owner` suspends here, `resume_suspended_cycle` resumes.
struct SuspendedCycle<'q, 'src, Q: Query<'q, 'src>> {
    /// The re-entered / waited-on key.
    key: Q::Key,
    /// The waiter's own committed active prefix at suspension time,
    /// ferried across the thread boundary (the owner's frames live on
    /// another thread and stay unreachable — this prefix is what makes
    /// the cross-thread path pinpointable instead of `&[]`).
    waiter_prefix: Vec<Q::Key>,
    /// The wait-graph chain (me → owner → …) that closed the loop.
    wait_chain: Vec<std::thread::ThreadId>,
}

/// Full loop context for a per-query cycle processor.
///
/// Same-thread cycles carry the committed active prefix in `stack` (the
/// failed re-entry is NOT pushed) with an empty `wait_chain`.
/// Cross-thread cycles carry the *waiter's* committed prefix in `stack`
/// (ferried across by the resumable waiter) plus the wait-graph
/// `wait_chain` that closed the loop.
#[derive(Debug, Clone)]
pub struct CycleContext<'q, 'src, Q: Query<'q, 'src>> {
    /// The re-entered / waited-on key.
    pub key: Q::Key,
    /// Same-thread: this thread's committed active prefix.
    /// Cross-thread: the waiter's committed prefix at suspension.
    pub stack: Vec<Q::Key>,
    /// Cross-thread only: the wait-graph chain that closed the loop.
    pub wait_chain: Vec<std::thread::ThreadId>,
    /// True when this context crossed the thread boundary.
    pub is_cross_thread: bool,
}

/// What a per-query loop processor decides when it resumes a cycle.
#[derive(Debug, Clone)]
pub enum CycleDecision<V> {
    /// Local recovery: serve this value at the breaking get (same
    /// staging/edge discipline as the legacy handler).
    Recover(V),
    /// Precise pinpointing: abort with a cycle error carrying `note`
    /// as its diagnostic attachment.
    Abort { note: String },
}

/// What `validate_enter` found.  The cycle arm carries the TYPED stack —
/// captured before stringification — so a per-query handler can diagnose
/// from keys rather than from `Vec<String>`.
pub(crate) enum EnterError<'q, 'src, Q: Query<'q, 'src>> {
    /// Same-thread re-entry; `stack` is the committed prefix (the failed
    /// re-entry was not pushed).
    Cycle { stack: Vec<Q::Key> },
    /// Depth limits (global backstop / per-type cap): stringified
    /// immediately — no handler applies.
    Fatal(QueryError),
}

/// Maximum recursive query depth before a depth-limited query returns an error.
const MAX_QUERY_DEPTH: usize = 128;

/// Runaway-recursion diagnostic: cap on nested `get_shared` frames per
/// thread, regardless of query type.  `MAX_QUERY_DEPTH` is per-(thread,
/// query-type, system) and does not bound a chain of DISTINCT query
/// types (`Q1 → Q2 → …` each accumulating exactly one frame per type).
///
/// This is deliberately NOT the stack-overflow guard anymore: every
/// `get_shared` frame runs under `stacker::maybe_grow` (see below), so
/// legal deep chains grow stack segments on demand instead of dying at
/// a frame budget calibrated to one build mode's frame size.  What this
/// bounds is recursion no cycle detection can catch (monotonically
/// fresh keys — the f3 shape): without a cap that is infinite memory
/// consumption with no error.  8192 is diagnostic headroom, not a
/// tuning knob: genuine workloads deeper than this do not exist, and
/// anything that reaches it is a runaway.
const MAX_GLOBAL_DEPTH: usize = 8192;

/// Compute a 64-bit hash for a `(query_type, key)` pair.
/// Used to select the `active_keys` shard only: `active_shard` masks the
/// result to index into the shard array.  Map lookups inside each shard
/// use the full `ActiveKey` (exact `Eq`, keyed by `(TypeId, Q::Key)`),
/// so a hash collision can never alias two distinct keys — it only
/// routes two keys to the same shard lock, which is benign contention.
fn query_key_hash<'src, 'q, Q: Query<'q, 'src>>(key: &Q::Key) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    TypeId::of::<Q>().hash(&mut h);
    key.hash(&mut h);
    h.finish()
}

/// Type-erased query key stored in the `active_keys` map: delegates
/// `Hash`/`Eq` to the concrete `Q::Key`, so map lookups are EXACT and a
/// 64-bit hash collision can never alias two distinct keys.
trait ErasedQueryKey: Send + Sync + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
    fn hash_dyn(&self, state: &mut dyn Hasher);
    fn eq_dyn(&self, other: &dyn ErasedQueryKey) -> bool;
}

impl<K: QueryKey> ErasedQueryKey for K {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn hash_dyn(&self, mut state: &mut dyn Hasher) {
        // `&mut dyn Hasher` itself implements `Hasher` (std blanket impl
        // for `&mut H where H: Hasher + ?Sized`), so route through the
        // extra mutable binding to satisfy the sized `H` bound of
        // `Hash::hash`.
        std::hash::Hash::hash(self, &mut state);
    }
    fn eq_dyn(&self, other: &dyn ErasedQueryKey) -> bool {
        other.as_any().downcast_ref::<K>() == Some(self)
    }
}

/// Exact map key of an `active_keys` entry: query type + the REAL key.
/// `shard_hash` (from [`query_key_hash`]) selects the shard only; the
/// `Hash`/`Eq` impls delegate to the concrete key, so lookups never
/// collide across distinct keys.  One `Arc` allocation per cache-miss
/// acquisition (the `get_shared` hit path constructs no `ActiveKey` at
/// all) — noise next to the compute it precedes.
#[derive(Clone)]
struct ActiveKey {
    shard_hash: u64,
    type_id: TypeId,
    key: Arc<dyn ErasedQueryKey>,
}

impl ActiveKey {
    #[inline]
    fn of<'src, 'q, Q: Query<'q, 'src>>(key: &Q::Key) -> Self {
        ActiveKey {
            shard_hash: query_key_hash::<Q>(key),
            type_id: TypeId::of::<Q>(),
            key: Arc::new(key.clone()),
        }
    }
}

impl Hash for ActiveKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
        self.key.hash_dyn(state);
    }
}

impl PartialEq for ActiveKey {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id && self.key.eq_dyn(other.key.as_ref())
    }
}

impl Eq for ActiveKey {}

impl std::fmt::Debug for ActiveKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The erased key carries no Debug bound; render its shard hash.
        f.debug_struct("ActiveKey")
            .field("shard_hash", &self.shard_hash)
            .field("type_id", &self.type_id)
            .finish_non_exhaustive()
    }
}

/// Monotonic id source for `QuerySystem` instances.  Replaces the
/// address-based id: ids are also the key of a thread's deferred-mutation
/// presence marker, and an id derived from an address would collide
/// whenever a system is dropped — while another thread still holds a
/// marker for it — and a fresh system is later allocated at that address.
static NEXT_SYSTEM_ID: AtomicUsize = AtomicUsize::new(1);

/// Number of shards for the `active_keys` map. Power of 2 for fast modulo.
///
/// 64, not 8: `rayon` uses every core, and with 8 shards a 32-thread miss
/// storm averages 4 threads per lock word.  64 padded locks is 4 KiB.
const ACTIVE_SHARDS: usize = 64;

/// Compute the shard index for an active key: the 64-bit hash selects
/// the SHARD only — map lookups are exact (see [`ActiveKey`]).
#[inline(always)]
fn active_shard(akey: &ActiveKey) -> usize {
    akey.shard_hash as usize & (ACTIVE_SHARDS - 1)
}

// ── Query descriptor ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QueryDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub eval_always: bool,
    pub depth_limit: bool,
    /// Which kind of source content this query's value is derived from.
    /// Declared once per query; the dep graph stamps it onto every node
    /// the query allocates and (in debug builds) asserts that semantic
    /// queries never depend on doc/trivia/position nodes.  Defaults to
    /// [`dep_graph::ContentClass::SEMANTIC`].
    pub content_class: dep_graph::ContentClass,
    /// Declared change-frequency durability of this query's inputs
    /// (salsa-style, opt-in).  A node's effective durability is
    /// `min(declared, min over reads)` — aggregated at every finish and
    /// propagated along the reverse edges on every actual drop (see
    /// [`dep_graph::Durability`] for the structural-soundness story) —
    /// so a dishonest HIGH declaration no longer risks staleness: it
    /// only caps how high the stored level can go, and the fast path
    /// degrades to the walk the moment a lower-level input enters the
    /// picture.  Declare HIGH/MEDIUM to EARN the per-level fast path
    /// (inputs the host treats as immutable per session), not as a
    /// soundness obligation.  Defaults to LOW — every existing query
    /// keeps today's exact invalidation behavior.
    pub durability: dep_graph::Durability,
    /// Capacity of the query's memoization cache, in entries.  Defaults to
    /// [`DEFAULT_CACHE_CAPACITY`]; queries whose working set is known to be
    /// larger should raise it — evicting a green value forces a recompute
    /// even though nothing changed, so an undersized cache defeats
    /// memoization entirely (measured: a 4096-key working set against the
    /// 512 default spends most of its time recomputing).  Best-effort
    /// bound: for a finite capacity below the shard count the effective
    /// Floor is one entry per shard, and `0` means unlimited.
    pub cache_capacity: usize,
    /// Untracked memo query: skips dep-graph recording entirely (no node
    /// index, no active-set entry, no read edges, no durability/fingerprint
    /// publish).  Sound ONLY for deterministic pure functions of
    /// `(key, immutable data)`: the compute MUST NOT call any query
    /// (enforced: nested `get*` inside untracked compute is a hard panic
    /// via `UNTRACKED_DEPTH`) and MUST NOT read mutable session state or
    /// query inputs (author's trust contract — same class as
    /// [`QueryDescriptor::durability`]).
    ///
    /// Untracked queries branch off BEFORE the stack-growth wrapper
    /// (nesting is impossible, so growth is unnecessary) and take no
    /// `InFlightGuard`, no session slot, and no quiesce read guard —
    /// there is nothing to quiesce.  `clear_all` wipes untracked memos
    /// with the vtable registry (vtable lifetime == world generation),
    /// and `invalidate` drops the single memo entry.
    pub untracked: bool,
}

impl QueryDescriptor {
    pub const fn new(name: &'static str, description: &'static str) -> Self {
        QueryDescriptor {
            name,
            description,
            eval_always: false,
            depth_limit: false,
            content_class: dep_graph::ContentClass::SEMANTIC,
            durability: dep_graph::Durability::LOW,
            cache_capacity: DEFAULT_CACHE_CAPACITY,
            untracked: false,
        }
    }
    /// BUG-MAGNET: for an `eval_always` query, a dep node is registered
    /// on first compute and the node half of that registration is
    /// MONOTONE — no eviction and no `clear` ever removes it.  Combined
    /// with `eval_always`'s node-only cache presence (the key's value is
    /// recomputed on every get; only the SURROUNDING node stays in the
    /// graph), each distinct key that was ever computed leaves a
    /// permanent node behind, reachable by neither eviction nor `clear` —
    /// only [`QuerySystem::clear_all`] releases them.  Do not attach
    /// `eval_always` to a query keyed by unbounded data.
    pub const fn with_eval_always(mut self, val: bool) -> Self {
        self.eval_always = val;
        self
    }
    pub const fn with_depth_limit(mut self, val: bool) -> Self {
        self.depth_limit = val;
        self
    }
    pub const fn with_content_class(mut self, val: dep_graph::ContentClass) -> Self {
        self.content_class = val;
        self
    }
    /// Declare the query's input durability (trusted-input-only — see
    /// [`QueryDescriptor::durability`]).
    pub const fn with_durability(mut self, val: dep_graph::Durability) -> Self {
        self.durability = val;
        self
    }
    pub const fn with_cache_capacity(mut self, val: usize) -> Self {
        self.cache_capacity = val;
        self
    }
    /// Opt out of dep-graph recording (memo-only query — see
    /// [`QueryDescriptor::untracked`]).  Contradicts `eval_always`
    /// (recompute-everything needs no memo): combining both panics.
    ///
    /// Two consequences follow from going node-less — both sound, both
    /// worth stating before you opt in:
    ///
    /// * **No cross-thread dedup.**  Two concurrent gets for the same
    ///   key each run `compute` (the memo insert is last-writer-wins).
    ///   Fine for pure functions; a side-effecting compute would run
    ///   twice.  (`par_get`'s dedup does not cover this path.)
    /// * **No dep edges to outer readers.**  A tracked query that reads
    ///   an untracked value records nothing, so it is never invalidated
    ///   when the memo changes — sound ONLY under the immutable-input
    ///   contract (the value is a pure function of the key and frozen
    ///   session data), and it additionally relies on the driver calling
    ///   `clear_all` (or `clear::<Q>`) between sessions: keys minted
    ///   against one `Program` are meaningless in the next, and nothing
    ///   structural enforces the clear.
    pub const fn with_untracked(mut self, val: bool) -> Self {
        self.untracked = val;
        self
    }
}

// ── QueryCacheType trait ──────────────────────────────────────────

/// Cache statistics for observability.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: usize,
}

/// The cache interface for query results.
///
/// Analogous to Rust's `QueryCache` trait in `caches.rs`.
/// Each key type can select its own cache implementation
/// (HashMap, single-entry, Vec-indexed, etc.).
/// The cache abstraction for one query type: key → (node, value) with the
/// NODE half and the VALUE half sharing one storage and one lock.
///
/// This is the merged-storage contract (the node_map/DefaultCache merge):
/// one sharded map per cache holds `Key → (DepNodeIndex, Option<V>)`,
/// so the hit path pays ONE shard read lock where it used to pay two
/// (`node_map` + cache), and the fresh-registration two-phase race lives
/// inside `probe` next to the value read.
///
/// # Hard contracts (every impl MUST hold)
///
/// 1. **The node half is never evicted and never re-created.**  `stage`/
///    `unstage`/`clear`/eviction may drop the stored `V` but MUST leave the
///    `DepNodeIndex` reachable for the key: dropping it would make the
///    next probe allocate a fresh node while the old node's fingerprint,
///    dep edges and epoch history remain in the dep graph — orphaned
///    incremental state behind a `node_to_key` entry that still names the
///    old index (the d9 Window-3 stale-mapping class).  A re-registered
///    fresh node also starts green with no fingerprint, silently cutting
///    the invalidation history.  (`clear_all` is exempt: it tears down the
///    whole registry + dep graph atomically under quiescence.)
/// 2. **The LRU/eviction machinery only covers `value.is_some()` entries.**
///    A node registered by a probe (value `None`), a staged-red entry, or
///    an unstaged entry is invisible to eviction.
/// 3. **`alloc` must never run under a shard write lock.**  It allocates
///    in the dep graph (shard lock + growth lock there); nesting that
///    inside a cache shard lock would create a cache→dep lock ordering the
///    module invariant ("never hold two shard locks") forbids.  `probe`'s
///    two-phase shape — read lock, release, `alloc`, write lock, re-check
///    — keeps it outside; the two-phase race loser discards its allocated
///    index (an orphan node, harmless — same class as the pre-merge loser).
/// 4. **`stage`/`unstage`/`clear` drop replaced payloads OUTSIDE the
///    shard lock** — a fat value's recursive `Drop` must not run while
///    other threads spin on the shard.
/// 5. **Stats are recorded by the CALLER, not by `probe`.**  `probe`
///    returns whatever the storage holds; the caller gates on
///    green/eval_always and calls `record_hit`/`record_miss` (lock-free
///    atomic bumps) only for a real green cache decision — a red node's
///    probe today touches no counters and must not touch them merged.
pub trait QueryCacheType<K, V>: std::fmt::Debug
where
    K: std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Hit-path sole entry: ONE read lock fetches node + value together.
    /// On a miss, release the read lock, call `alloc` (dep-graph node
    /// allocation, outside any cache lock), re-take the write lock and
    /// re-check the registration — the fresh two-phase race, embedded.
    /// `fresh == true` exactly when THIS call registered the key's node
    /// (race losers return the winner's entry with `fresh == false` and
    /// discard their allocated index).  The returned value is whatever
    /// the storage holds — the CALLER gates it on node colour before
    /// serving.
    ///
    /// The compile face of contract 3 (mechanism B): `alloc` takes
    /// `&mut Held<CacheNoShard>` — that state is not present inside the
    /// cache shard critical section (only `CacheShardHeld`), so reintroducing
    /// the "in-lock alloc" bug = E0277. dep shard/grow physically cannot get in.
    ///
    /// "Mechanism B does not cover SingleCache / DefIdCache" —
    /// that premise is WRONG, and it is worth writing down why rather than
    /// adding ticket wrappers to the two single-lock caches:
    ///
    ///  * **E0277, the compile face of contract 3** is enforced for ALL
    ///    THREE impls, because all three obtain their node through the same
    ///    `alloc(&mut Held<CacheNoShard>)` signature.  `SingleCache::probe`
    ///    and `DefIdCache::probe` both build their own
    ///    `Held::<CacheNoShard>::fresh()` and call `alloc` on it, so an
    ///    in-lock alloc fails to compile there exactly as it does in
    ///    `DefaultCache::probe`.
    ///  * **Tripwire visibility** is also complete.  `DefaultCache` takes
    ///    the raw shard lock inside `take_cache_shard_r/w`, which bypasses
    ///    `Acquired::acquire`, so it must call `held_inc`/`held_dec`
    ///    manually (see the comment there).  The single-lock caches pair
    ///    every raw lock with `Acquired::acquire(Fam::CacheShard, ...)`,
    ///    which does the same inc/dec — so `HELD[CacheShard]` rises and
    ///    every `ORD[CacheShard][*]` cell stays testable through them too.
    ///
    /// The residual difference is only WHICH mechanism performs the
    /// bookkeeping, not whether it happens: `DefaultCache` additionally
    /// threads `Held<CacheShardHeld>` through `take`/`release` (an
    /// explicit pairing the single-lock caches express as a scoped
    /// `Acquired` guard).  Both are correct; neither is a gap.
    fn probe(
        &self,
        key: &K,
        alloc: impl FnOnce(&mut ticket::Held<ticket::CacheNoShard>) -> dep_graph::DepNodeIndex,
    ) -> (dep_graph::DepNodeIndex, Option<V>, bool);

    /// Publish a freshly computed value under `key` (was: `insert`).
    /// The `node` must be the one a prior `probe` returned for this key
    /// (the registration already exists).  The replaced payload, if any,
    /// is dropped after the shard lock releases.
    fn stage(&self, key: K, node: dep_graph::DepNodeIndex, v: V);

    /// Withdraw a staged value (was: `remove`) — panic un-stage,
    /// invalidation.  Returns the key's node (which STAYS registered, per
    /// contract 1) so the invalidation path can fuse its former
    /// "node-map read lock + cache write lock" pair into one write lock.
    fn unstage(&self, key: &K) -> Option<dep_graph::DepNodeIndex>;

    /// Allocation-free node lookup (was: `QueryVTable::lookup_node`'s
    /// private node-map read).  Read-lock only; never registers.
    fn node_of(&self, key: &K) -> Option<dep_graph::DepNodeIndex>;

    /// Drop every VALUE (and the LRU order), keep every NODE (was:
    /// `clear`; `QuerySystem::clear::<Q>` relies on the next get
    /// RECOMPUTING against the surviving node, not re-allocating one).
    fn clear(&self);

    /// Number of entries with a staged value (the eviction-managed
    /// population — matches the old `map.len()` semantics).
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Visit the staged values.  One shard lock at a time, mirroring
    /// `DepGraph::reset`'s discipline.
    fn for_each(&self, f: impl FnMut(&K, &V));
    /// Construct the cache with the capacity declared by the query's
    /// descriptor.  Sharded impls split it across shards; `0` = unlimited.
    /// Caches with no meaningful notion of capacity (single-entry,
    /// `Vec`-indexed) ignore the argument; only [`DefaultCache`] honors it.
    fn from_capacity(capacity: usize) -> Self;
    /// Return cache statistics.  Default implementation returns empty stats.
    fn stats(&self) -> CacheStats {
        CacheStats::default()
    }
    /// Lock-free hit bump — the caller invokes it when a probe's value
    /// survives the green gate (a real serve decision).  Default: no-op.
    fn record_hit(&self) {}
    /// Lock-free miss bump — green gate passed but no value staged.
    /// Default: no-op.
    ///
    /// Red-node probes deliberately record NO counters (they never reach
    /// here), so recomputes forced by a red node undercount misses.  Do
    /// not "fix" that by counting inside `probe`: a red probe runs
    /// concurrently with the stage that flips the colour, so the count
    /// would race the very gate that defines hit vs miss.
    fn record_miss(&self) {}
}

/// A value that exclusively occupies its own 64-byte cache line.
///
/// Perf Book §13.7.3 (false sharing): two hot cells — two atomic
/// counters, or two neighbouring lock words — that share a cache line
/// ping-pong that line between cores even though each thread touches
/// disjoint data.  `#[repr(align(64))]` both aligns the value and
/// rounds the struct size up to a full line, so no explicit padding
/// array is needed (adding one would just double the footprint).
#[repr(align(64))]
#[derive(Debug)]
struct CachePadded<T>(T);

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}

// ── Per-thread stat shards (merge-on-read counters) ──────────────

/// One thread's private counters for ONE cache instance.  Each thread
/// writes only its own block, so the block's line stays exclusive to
/// that core — the hit path becomes a local RMW with zero coherence
/// traffic, which padding alone cannot buy: a shared `fetch_add` line
/// still ping-pongs between cores under a parallel hit storm (Perf Book
/// §13.1 — the USL serialisation point).  `AtomicU64` rather than `Cell`
/// because `stats()` sums the blocks cross-thread; `repr(align(64))`
/// because two threads' 24-byte blocks packed adjacently on the heap
/// would share a line — false sharing merely relocated (§13.7.3).
#[repr(align(64))]
#[derive(Debug, Default)]
struct ThreadStats {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

/// Which counter of a thread's slot a stat bump increments.
#[derive(Debug, Clone, Copy)]
enum StatField {
    Hit,
    Miss,
    Eviction,
}

/// Monotonic arena ids — NOT addresses: an address is reused after
/// drop while other threads' TLS may still hold an entry for the dead
/// instance, and a fresh cache allocated there would inherit the dead
/// one's residual counts (the `NEXT_SYSTEM_ID` failure class).
static NEXT_STAT_ARENA_ID: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    /// arena id → THIS thread's slot in that instance's registry.
    /// Hot-path contract: one TLS access + one hash probe + one
    /// uncontended RMW.  The value is a HANDLE only — the DATA lives in
    /// the owning cache's registry, so an exited thread leaves a
    /// finalized slot that `stats()` still sums.  Bumped through the
    /// `Arc`'s `Deref`, NEVER cloned — a per-hit `Arc::clone` would put
    /// the shared refcount line right back on the hit path.  Non-const
    /// init (same as `IN_FLIGHT_BY_SYSTEM`): `RefCell<FxHashMap>` has
    /// no `const` constructor.
    static STAT_SLOTS: std::cell::RefCell<FxHashMap<usize, Arc<ThreadStats>>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// Nesting depth inside untracked computations.  Nonzero while an
/// untracked query's `compute` runs on this thread; any `get` issued
/// there is a purity violation (hard panic at the entries below).
/// `Cell<u32>` (`const`-constructible) — depth only ever reaches 1,
/// the panic fires before a second frame can form.
thread_local! {
    static UNTRACKED_DEPTH: std::cell::Cell<u32> = std::cell::Cell::new(0);
}

/// Restores `UNTRACKED_DEPTH` on scope exit, including unwind: an
/// untracked compute that panics must not leave the thread fenced.
struct UntrackedDepthGuard;
impl Drop for UntrackedDepthGuard {
    fn drop(&mut self) {
        UNTRACKED_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Increment one counter of a registered slot.  `Relaxed` throughout:
/// pure diagnostic counts that enter no happens-before chain (contrast
/// the dep-graph's deliberately `AcqRel` epochs — different contract),
/// and each slot only ever increases, so `stats()`' cross-thread sum is
/// non-decreasing (safe for rate math).
#[inline]
fn bump(slot: &ThreadStats, f: StatField) {
    let c = match f {
        StatField::Hit => &slot.hits,
        StatField::Miss => &slot.misses,
        StatField::Eviction => &slot.evictions,
    };
    c.fetch_add(1, Ordering::Relaxed);
}

// ── DefaultCache (sharded, merged node+value storage, O(1) CLOCK) ──

/// Per-shard state of the merged `DefaultCache`.  One map holds BOTH
/// halves of every key: the never-evicted `DepNodeIndex` (the old
/// per-vtable `node_map`'s job) and the evictable `Option<V>` (the
/// old cache's job) — one shard read lock serves the whole hit path.
#[derive(Debug)]
struct CacheShard<K, V> {
    map: FxHashMap<K, MergedEntry<K, V>>,
    /// LRU chain over the STAGED entries only (`value.is_some()`), doubly
    /// linked by key (query keys are small identity types, `K: Clone`).
    /// `head` = most recent, `tail` = CLOCK sweep start.
    head: Option<K>,
    tail: Option<K>,
    /// Staged-entry count == chain length, maintained by stage/unstage/
    /// evict; `len()` sums it across shards instead of walking the maps.
    valued: usize,
}

/// One merged entry.  The node half is registration-once — NEVER evicted,
/// NEVER re-created (dropping it would orphan the old node's fingerprint,
/// dep edges and epoch history behind a stale `node_to_key` entry, the
/// d9 Window-3 class, and a re-registered fresh node starts green with no
/// fingerprint, cutting the invalidation history).  The value half is
/// evictable.  `prev`/`next` are the LRU links and are only meaningful
/// while `value.is_some()` (valueless entries are chain-detached).
#[derive(Debug)]
struct MergedEntry<K, V> {
    node: dep_graph::DepNodeIndex,
    value: Option<V>,
    /// CLOCK second-chance bit, set by probe hits, consumed by the sweep.
    referenced: AtomicBool,
    prev: Option<K>,
    next: Option<K>,
}

/// HashMap-backed cache with O(1) CLOCK eviction, SHARDED 64 ways and
/// MERGED with the old per-vtable `node_map`: a shard's map holds
/// `Key → MergedEntry` (node + value), so the hit path pays one shard
/// read lock where it used to pay two (node map + cache), and the fresh
/// two-phase registration lives in `probe` next to the value read.
///
/// ### Recency on the read path (CLOCK / second chance)
///
/// `probe` must not reorder the list: it runs under a shared read lock so
/// concurrent readers do not serialize (Perf Book §13.1 USL).  A hit sets
/// the entry's atomic `referenced` bit; the eviction sweep applies the
/// classic CLOCK policy from the tail — clearing a referenced bit and
/// promoting the entry, evicting the first unreferenced one — and the
/// scan is bounded by one revolution because no reader can set bits while
/// the write lock is held.
///
/// ### What eviction evicts (the d9 Window-3 contract)
///
/// ONLY the value.  The node half survives eviction, `unstage` and
/// `clear` for the storage's lifetime: a re-probe of an evicted key finds
/// its node still registered (miss on value, no re-registration) and a
/// recompute re-stages into the same entry.  Dropping the node with the
/// value would re-allocate a fresh green fingerprint-less node over the
/// old one's dep-graph state — never do it.  (`clear_all` is exempt: it
/// tears down the whole registry + dep graph atomically under quiescence.)
///
/// The default capacity is `DEFAULT_CACHE_CAPACITY` (4096 entries), split
/// across the 64 shards (min 1 per shard); `with_capacity(0)` = unlimited.
/// Queries with a larger known working set should raise it via
/// `QueryDescriptor::with_cache_capacity` — evicting a green value forces
/// a full recompute even though nothing changed, so the cache must
/// comfortably hold the hot working set for memoization to pay off.
const DEFAULT_CACHE_CAPACITY: usize = 4096;

/// Number of shards for the merged cache.  Aligned with `ACTIVE_SHARDS`
/// and `DEP_SHARDS` (both 64) so contention domains match the rest of the
/// query engine, chosen by bench adjudication (`bench_publish_thin_scaling`
/// + `bench_hit_path_parallel`, 16 vs 64).  Costs 4 KiB of shard state per
/// cache instance (vs 1 KiB at 16); keys select a shard via FxHash-of-key
/// masking (see `shard_of_key`).
const DEFAULT_CACHE_SHARDS: usize = 64;

/// TinyLFU admission sketch shape: 2 rows × 2048 saturating u8
/// counters (4 KiB per cache instance).  Rows are indexed by two
/// independent low-bit slices of one FxHash of the key (double-hash
/// style); the estimate is the row minimum, Count-Min style.  u8
/// saturates at 255 — frequencies are RELATIVE (newcomer vs victim),
/// so saturation only compresses the top end, and the periodic
/// halving below keeps hot/cold separation.  4 KiB is noise next to
/// the shard maps, and the sketch is lock-free (Relaxed atomics, no
/// mutex), so the admission check costs two loads on the eviction
/// path only — the hit path pays just the increment.
const ADMIT_SKETCH_WIDTH: usize = 2048;
const ADMIT_SKETCH_ROWS: usize = 2;
const ADMIT_SKETCH_SIZE: usize = ADMIT_SKETCH_WIDTH * ADMIT_SKETCH_ROWS;

/// Debug-only occupancy snapshot across a cache's shards — the raw data
/// behind the degenerate-`Hash` audit (`DefaultCache::shard_distribution`).
#[cfg(debug_assertions)]
#[derive(Debug)]
pub struct ShardDistribution {
    /// Staged (valued) entries per shard, in shard order.
    pub valued: Vec<usize>,
    /// Registered entries per shard (node half included), in shard order.
    pub map_sizes: Vec<usize>,
    pub total: usize,
    pub max: usize,
    pub min: usize,
}

/// Shared skew predicate for the shard-distribution audit: a population
/// below `MIN_POPULATION` is statistically guaranteed to look skewed
/// (one idle shard in a handful of keys already trips a 4× max/min
/// ratio), so only a meaningful population is judged.
#[cfg(debug_assertions)]
pub fn distribution_flagged(total: usize, max: usize, min: usize) -> bool {
    const MIN_POPULATION: usize = 64;
    total >= MIN_POPULATION && max > 4 * min.max(1)
}

#[derive(Debug)]
pub struct DefaultCache<K, V> {
    shards: Vec<CachePadded<RwLock<CacheShard<K, V>>>>,
    /// Per-shard staged-entry SOFT cap: `capacity / shards` (min 1) for a
    /// finite budget, `0` = unlimited.  No longer a hard per-shard ceiling
    /// — a hot shard borrows idle shards' share and only the GLOBAL cap
    /// (`capacity` + `staged_global`) governs, so a skewed key
    /// distribution (or a constant-hash key) no longer under-provisions
    /// the budget on the shard it actually uses.  The min-1 floor means a
    /// finite `capacity < DEFAULT_CACHE_SHARDS` still admits one entry
    /// per shard (effective floor = shard count).
    per_shard_capacity: usize,
    /// Raw `max_entries` this cache was constructed with (0 = unlimited).
    /// NOT derivable from `per_shard_capacity`: the `(max/shards).max(1)`
    /// rounding makes `per_shard_capacity * shards != capacity` (e.g.
    /// capacity 10 → per-shard 1 → 16 if back-multiplied).
    capacity: usize,
    /// Global staged-value count, mirrored across every `shard.valued`
    /// mutation (stage/insert ++, evict/unstage --, clear snapshots to 0)
    /// under that shard's own write lock.  Relaxed throughout: transient
    /// skew only over- or under-evicts, and over-eviction is pure perf
    /// loss (direction-safe — the node half is never evicted, contract 1).
    /// Read only on the eviction path, never the hit path.
    staged_global: AtomicUsize,
    /// TinyLFU admission sketch (see `ADMIT_SKETCH_*`): per-key access
    /// frequencies estimating "how hot is this key".  The eviction loop
    /// admits a newcomer only when it is at least as hot as the CLOCK
    /// victim — one-hit scan keys no longer churn green residents out.
    /// A wrong decision is perf-only (rejection just means "recompute
    /// next probe"; the node stays registered per contract 1), so all
    /// atomics are Relaxed and races (a lost increment, a torn halve)
    /// only blur frequencies, never correctness.
    admit: Vec<AtomicU8>,
    /// Stage/insert tick driving periodic sketch halving (TinyLFU
    /// aging): every `capacity.max(64)` ticks all counters halve, so
    /// yesterday's hot keys decay and today's newcomers can win.
    admit_clock: AtomicUsize,
    /// Monotonic instance id keying this cache's entries in the
    /// thread-local stat-slot map (`STAT_SLOTS`).  Never an address —
    /// see `NEXT_STAT_ARENA_ID`.
    arena_id: usize,
    /// Authoritative registry of thread slots.  The DATA lives here, not
    /// in TLS: an exited thread leaves a finalized slot that `stats()`
    /// still sums.  Only the HANDLE is thread-local.
    stat_shards: Mutex<Vec<Arc<ThreadStats>>>,
}

impl<K, V> DefaultCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    pub fn new() -> Self {
        DefaultCache::with_capacity(DEFAULT_CACHE_CAPACITY)
    }
    pub fn with_capacity(max_entries: usize) -> Self {
        let per_shard = if max_entries == 0 {
            0
        } else {
            (max_entries / DEFAULT_CACHE_SHARDS).max(1)
        };
        DefaultCache {
            shards: (0..DEFAULT_CACHE_SHARDS)
                .map(|_| {
                    CachePadded(RwLock::new(CacheShard {
                        map: FxHashMap::default(),
                        head: None,
                        tail: None,
                        valued: 0,
                    }))
                })
                .collect(),
            per_shard_capacity: per_shard,
            capacity: max_entries,
            staged_global: AtomicUsize::new(0),
            admit: (0..ADMIT_SKETCH_SIZE).map(|_| AtomicU8::new(0)).collect(),
            admit_clock: AtomicUsize::new(0),
            arena_id: NEXT_STAT_ARENA_ID.fetch_add(1, Ordering::Relaxed),
            stat_shards: Mutex::new(Vec::new()),
        }
    }

    /// Merge-on-read stat bump: the hit path writes only ITS OWN
    /// thread's block — one TLS access + one FxHash probe + one
    /// uncontended `fetch_add`, with the line parked in this core's M
    /// state and zero coherence traffic (the whole point of the arena;
    /// a shared counter would serialize every hit on one line).
    #[inline]
    fn stat_bump(&self, f: StatField) {
        STAT_SLOTS.with(|m| {
            let fast = {
                // Borrow scoped to the probe so the cold path below can
                // `borrow_mut` without a double-borrow panic.
                let m = m.borrow();
                match m.get(&self.arena_id) {
                    Some(slot) => {
                        bump(slot, f);
                        true
                    }
                    None => false,
                }
            };
            if !fast {
                self.stat_bump_cold(f);
            }
        });
    }

    /// First-touch registration.  MAY run while holding a cache shard
    /// write lock (the eviction loops in `stage`/`insert` bump under the
    /// lock) — so the lock-order rule is: cache-shard lock →
    /// stat-registry lock is legal, the REVERSE is forbidden.  `stats()`
    /// obeys by taking `len()` (shard read locks) BEFORE the registry.
    fn stat_bump_cold(&self, f: StatField) {
        STAT_SLOTS.with(|m| {
            let hit = {
                let m = m.borrow();
                match m.get(&self.arena_id) {
                    Some(slot) => {
                        bump(slot, f);
                        true
                    }
                    None => false,
                }
            };
            if hit {
                return;
            }
            // cache.shard → stat_registry is a legal edge (§0 #8); register
            // here to catch the reverse (#9: stats() takes len before the
            // registry).
            let _tw =
                sync_order::Acquired::acquire(sync_order::Fam::StatRegistry, "stat_bump_cold");
            let slot = Arc::new(ThreadStats::default());
            self.stat_shards.lock().push(Arc::clone(&slot));
            bump(&slot, f);
            m.borrow_mut().insert(self.arena_id, slot);
        });
    }

    /// ONE FxHash pass over `key` — the single source for the shard index
    /// and both TinyLFU sketch rows.  The hit path used to
    /// hash twice (`shard_of_key` + `note_access`) and stage/insert three
    /// times (adding `admit_index`); every hot site now hashes once and
    /// shares the digest.  Same hash function and same bit slices as
    /// before — no behavioral change, just fewer passes.
    #[inline]
    fn hash_key(key: &K) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        key.hash(&mut h);
        h.finish()
    }

    /// Shard selection from a precomputed `hash_key` digest — the same
    /// FxHash-of-key function the old `QueryVTable::node_shard` used for
    /// the node map, so a key's lock word is stable across the merge.
    #[inline]
    fn shard_of_hash(h: u64) -> usize {
        (h as usize) & (DEFAULT_CACHE_SHARDS - 1)
    }

    /// Shard selection for call sites that need nothing else from the
    /// digest (and for the LRU/skew tests): single-pass wrapper over
    /// [`Self::hash_key`] + [`Self::shard_of_hash`].
    #[inline]
    fn shard_of_key(key: &K) -> usize {
        Self::shard_of_hash(Self::hash_key(key))
    }

    /// Sketch row indices from a precomputed digest: two low-bit slices of
    /// the one hash.
    #[inline]
    fn admit_rows_of_hash(h: u64) -> [usize; ADMIT_SKETCH_ROWS] {
        [
            (h as usize) & (ADMIT_SKETCH_WIDTH - 1),
            ((h >> 16) as usize) & (ADMIT_SKETCH_WIDTH - 1),
        ]
    }

    /// Record one access from a precomputed digest: saturating increment
    /// per row (saturation at 255 only compresses the top end — estimates
    /// are relative).  Lock-free; call on hits AND stages.
    ///
    /// The `fetch_update` CAS loop is deliberate and must NOT be "fixed"
    /// to a wrapping `fetch_add`: `admit` is `AtomicU8`, so 256 accesses
    /// to one key would wrap its row counter to 0 and drop a genuinely
    /// hot key to the coldest estimate — TinyLFU would then admit every
    /// newcomer and the cache degenerates to CLOCK.  `checked_add` inside
    /// the loop is what makes it saturate instead.
    #[inline]
    fn note_access_hash(&self, h: u64) {
        for (r, i) in Self::admit_rows_of_hash(h).into_iter().enumerate() {
            let c = &self.admit[r * ADMIT_SKETCH_WIDTH + i];
            let _ = c.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1));
        }
    }

    /// Estimated access frequency from a precomputed digest: row minimum,
    /// Count-Min style.
    #[inline]
    fn estimate_hash(&self, h: u64) -> u8 {
        let mut m = u8::MAX;
        for (r, i) in Self::admit_rows_of_hash(h).into_iter().enumerate() {
            m = m.min(self.admit[r * ADMIT_SKETCH_WIDTH + i].load(Ordering::Relaxed));
        }
        m
    }

    /// TinyLFU aging: every `capacity.max(64)` stages halve all
    /// counters so decayed hot keys stop vetoing newcomers.  Races
    /// with concurrent increments only blur frequencies.
    #[inline]
    fn admit_tick(&self) {
        let n = self.admit_clock.fetch_add(1, Ordering::Relaxed);
        if n % self.capacity.max(64) == 0 {
            for c in self.admit.iter() {
                let _ = c.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| Some(x / 2));
            }
        }
    }

    /// Debug-only occupancy audit: one shard read lock at a time, never
    /// nested.  Catches UNEXPECTED degenerate `Hash` impls on real key
    /// types — their symptom (every key in one shard → single lock
    /// contention plus a local eviction storm) is otherwise very hard to
    /// attribute back to `Hash`.  A deliberately constant key type
    /// (test `CollideKey`) is EXPECTED to trip `shard_skew_flagged`.
    #[cfg(debug_assertions)]
    pub fn shard_distribution(&self) -> ShardDistribution {
        let mut valued = Vec::with_capacity(self.shards.len());
        let mut map_sizes = Vec::with_capacity(self.shards.len());
        for lock in &self.shards {
            let shard = lock.read();
            valued.push(shard.valued);
            map_sizes.push(shard.map.len());
        }
        let total = valued.iter().sum();
        ShardDistribution {
            max: valued.iter().copied().max().unwrap_or(0),
            min: valued.iter().copied().min().unwrap_or(0),
            total,
            valued,
            map_sizes,
        }
    }

    /// `true` when `shard_distribution` looks degenerate (see
    /// `distribution_flagged` for the threshold).  Debug builds only.
    #[cfg(debug_assertions)]
    pub fn shard_skew_flagged(&self) -> bool {
        let d = self.shard_distribution();
        distribution_flagged(d.total, d.max, d.min)
    }

    /// Detach `key`'s entry from the LRU chain: splice its neighbors
    /// together and fix head/tail.  `prev`/`next` are the entry's links
    /// read by the caller BEFORE this call (the entry's own links are
    /// cleared separately — a chain-detached entry keeps None links).
    fn unlink_entry(shard: &mut CacheShard<K, V>, key: &K, prev: Option<K>, next: Option<K>) {
        if let Some(p) = &prev {
            if let Some(pe) = shard.map.get_mut(p) {
                pe.next = next.clone();
            }
        }
        if let Some(n) = &next {
            if let Some(ne) = shard.map.get_mut(n) {
                ne.prev = prev.clone();
            }
        }
        if shard.head.as_ref() == Some(key) {
            shard.head = next;
        }
        if shard.tail.as_ref() == Some(key) {
            shard.tail = prev;
        }
    }

    /// Link `key`'s entry at the chain head.  The entry must exist in the
    /// map and be chain-detached (`prev == next == None`).
    fn link_head(shard: &mut CacheShard<K, V>, key: &K) {
        let old_head = shard.head.clone();
        {
            let entry = shard
                .map
                .get_mut(key)
                .expect("link_head: entry must be registered");
            entry.prev = None;
            entry.next = old_head.clone();
        }
        if let Some(oh) = &old_head {
            if let Some(he) = shard.map.get_mut(oh) {
                he.prev = Some(key.clone());
            }
        }
        shard.head = Some(key.clone());
        if shard.tail.is_none() {
            shard.tail = Some(key.clone());
        }
    }

    /// Promote a CHAIN member to the head (no-op if already head).
    fn move_to_head(shard: &mut CacheShard<K, V>, key: &K) {
        if shard.head.as_ref() == Some(key) {
            return;
        }
        let (prev, next) = {
            let entry = shard
                .map
                .get(key)
                .expect("move_to_head: entry must be registered");
            (entry.prev.clone(), entry.next.clone())
        };
        Self::unlink_entry(shard, key, prev, next);
        Self::link_head(shard, key);
    }

    /// CLOCK sweep: detach ONE staged entry's value — the node half
    /// STAYS (the d9 Window-3 contract) — and RETURN it (key + value)
    /// instead of staging it into `dropped` directly, so the caller can
    /// either drop it post-lock (plain eviction, trait contract 4) or,
    /// under the TinyLFU admission gate, hand it back to a restored
    /// victim.  Returns `None` only when the chain is empty.
    /// Termination matches the old `evict_tail` exactly:
    /// a referenced entry is demoted-and-promoted (its bit is consumed by
    /// the swap), and no reader can set bits while this write lock is
    /// held, so the sweep ends within one revolution.
    fn evict_one(shard: &mut CacheShard<K, V>) -> Option<(K, Option<V>)> {
        let mut cur = shard.tail.clone();
        while let Some(key) = cur {
            let (referenced, prev, next) = {
                let entry = shard.map.get(&key).expect("chain entry must exist");
                (
                    entry.referenced.swap(false, Ordering::Relaxed),
                    entry.prev.clone(),
                    entry.next.clone(),
                )
            };
            if referenced {
                // Second chance: promote and continue the sweep from the
                // (unchanged) tail — the promoted entry now sits at the
                // head, so the tail is its former predecessor.
                Self::unlink_entry(shard, &key, prev, next);
                Self::link_head(shard, &key);
                cur = shard.tail.clone();
                continue;
            }
            // Victim: detach the VALUE, keep the node and the key.
            let old = {
                let entry = shard.map.get_mut(&key).expect("chain entry must exist");
                entry.prev = None;
                entry.next = None;
                entry.value.take()
            };
            Self::unlink_entry(shard, &key, prev, next);
            shard.valued -= 1;
            return Some((key, old));
        }
        None
    }

    /// Shared eviction + TinyLFU-admission tail of [`Self::insert`] and
    /// `stage`.  The two bodies were line-for-line duplicates
    /// of the subtlest accounting in the file — fresh detection, victim
    /// restore, `staged_global` mirroring — so a fix applied to one was
    /// easy to miss in the other.  One shared fn; the copies cannot drift.
    ///
    /// MUST run while holding `shard`'s write lock, after the caller has
    /// staged the new value and updated `shard.valued`, the chain, and
    /// `staged_global`.  `key_hash` is the caller's single hash pass
    /// (3.3) — hoisted out of the loop so one operation compares against
    /// ONE estimate for the newcomer, instead of re-reading a sketch other
    /// threads may bump between iterations.  `fresh` is the caller's
    /// `shard.valued > valued_before`.  Evicted/rejected payloads are
    /// pushed onto `dropped` so the caller drops them AFTER the shard
    /// lock releases (trait contract 4).
    fn evict_with_admission(
        &self,
        shard: &mut CacheShard<K, V>,
        key: &K,
        key_hash: u64,
        fresh: bool,
        dropped: &mut Vec<V>,
    ) {
        // Evict while over the shard's share AND the GLOBAL budget is
        // exceeded (hot shard borrows idle shards' share).  The
        // `per_shard_capacity > 0` guard must stay: with
        // `with_capacity(0)` (unlimited, capacity 0) it is what stops
        // `> self.capacity` from becoming `> 0` and emptying the
        // cache.  Only VALUES leave — nodes and keys stay registered
        // (trait contract 1).
        //
        // TinyLFU admission gate (#5): a FRESH newcomer (this call
        // raised `valued`) colder than the CLOCK victim it displaced
        // is rejected — the victim is restored, the newcomer is
        // unstaged instead, and the loop stops (the budget stays
        // transiently exceeded; the next stage retries with a warmer
        // estimate).  Re-stage-over-staged never gates: an admitted
        // resident keeps its seat.
        while self.per_shard_capacity > 0
            && shard.valued > self.per_shard_capacity
            && self.staged_global.load(Ordering::Relaxed) > self.capacity
        {
            let Some((victim, victim_value)) = Self::evict_one(shard) else {
                break;
            };
            // The victim's own hash is computed ONCE per eviction and then
            // compared by number (re-hashing `&victim` on every
            // loop iteration, plus probing the sketch twice per iteration,
            // quietly undoes the one-hash-per-op refactor on this path).
            let victim_hash = Self::hash_key(&victim);
            self.staged_global.fetch_sub(1, Ordering::Relaxed);
            self.stat_bump(StatField::Eviction);
            if fresh && self.estimate_hash(key_hash) < self.estimate_hash(victim_hash) {
                let victim_hot = victim_value.is_some();
                if let Some(entry) = shard.map.get_mut(&victim) {
                    entry.value = victim_value;
                }
                if victim_hot {
                    Self::link_head(shard, &victim);
                }
                // Both counter bumps are gated on `victim_hot` rather than
                // being unconditional.  Today `victim_value` is
                // always `Some` — chain members hold values only (trait
                // contract 2) — so the gate is currently a no-op, but if that
                // invariant ever breaks the extra `valued += 1` /
                // `staged_global.fetch_add(1)` would leave the two accounting
                // totals one too high per valueless victim and NEVER come
                // back down (nothing subtracts for an entry that never held a
                // value).  Gating here keeps the invariant local instead of
                // borrowed from the chain discipline.
                if victim_hot {
                    shard.valued += 1;
                    self.staged_global.fetch_add(1, Ordering::Relaxed);
                }
                if let Some(entry) = shard.map.get_mut(key) {
                    if let Some(v) = entry.value.take() {
                        let (prev, next) = (entry.prev.clone(), entry.next.clone());
                        Self::unlink_entry(shard, key, prev, next);
                        shard.valued -= 1;
                        self.staged_global.fetch_sub(1, Ordering::Relaxed);
                        dropped.push(v);
                    }
                }
                break;
            }
            dropped.extend(victim_value);
        }
    }

    /// Value-cache surface for callers that use `DefaultCache` WITHOUT
    /// dep-graph nodes — the trait solver's projection / search-graph
    /// caches (they only want the sharded CLOCK LRU).  The node half of
    /// such entries is the unused `DepNodeIndex(0)` sentinel (index 0 is
    /// never allocated: the dep graph's counter starts at 1), so no
    /// query-system code ever observes it and the merged contracts hold
    /// trivially.  Semantics match the pre-merge value cache: eviction
    /// withdraws the value (a later lookup misses) while the map entry —
    /// and its node slot — persists.
    pub fn lookup(&self, key: &K) -> Option<V> {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "cache.lookup");
        let h = Self::hash_key(key);
        let shard = self.shards[Self::shard_of_hash(h)].read();
        let entry = shard.map.get(key)?;
        if let Some(v) = &entry.value {
            entry.referenced.store(true, Ordering::Relaxed);
            let out = v.clone();
            self.note_access_hash(h);
            return Some(out);
        }
        None
    }

    /// See [`DefaultCache::lookup`] — the node-less `insert` twin of
    /// `stage`: insert-or-update with the node half pinned to the
    /// sentinel.  Replaced payloads drop after the shard lock releases
    /// (the same contract-4 discipline).
    pub fn insert(&self, key: K, value: V) {
        // (3.3) ONE hash pass feeds the shard index, the sketch bump and
        // the admission estimate below.
        let h = Self::hash_key(&key);
        let s = Self::shard_of_hash(h);
        self.note_access_hash(h);
        self.admit_tick();
        let mut dropped: Vec<V> = Vec::new();
        // #8: the eviction loop's `stat_bump` may take the stat registry
        // UNDER this shard write lock — the tripwire must see CacheShard
        // held to validate that edge.
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "cache.insert");
        {
            let mut shard = self.shards[s].write();
            let valued_before = shard.valued;
            if let Some(entry) = shard.map.get_mut(&key) {
                let old = entry.value.replace(value);
                let was_staged = old.is_some();
                if let Some(old) = old {
                    dropped.push(old);
                }
                if was_staged {
                    Self::move_to_head(&mut shard, &key);
                } else {
                    Self::link_head(&mut shard, &key);
                    shard.valued += 1;
                    self.staged_global.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                shard.map.insert(
                    key.clone(),
                    MergedEntry {
                        node: dep_graph::DepNodeIndex(0),
                        value: Some(value),
                        referenced: AtomicBool::new(false),
                        prev: None,
                        next: None,
                    },
                );
                Self::link_head(&mut shard, &key);
                shard.valued += 1;
                self.staged_global.fetch_add(1, Ordering::Relaxed);
            }
            // Eviction + TinyLFU admission — see `evict_with_admission`.
            let fresh = shard.valued > valued_before;
            self.evict_with_admission(&mut shard, &key, h, fresh, &mut dropped);
        } // shard lock released — payload Drops run from here on
        drop(dropped);
    }
}

impl<K, V> DefaultCache<K, V>
where
    K: std::fmt::Debug + Clone + Eq + std::hash::Hash,
    V: std::fmt::Debug + Clone,
{
    /// cache shard read lock for the probe/two-phase shape (mechanism B,
    /// the compile face of tables #10/#16): inside the critical section
    /// only the CacheShardHeld ticket is present, so dep shard/grow
    /// physically cannot get in — the formal half of contract 3.  The
    /// ticket path covers exactly this shape; the value-cache surface
    /// (`lookup`/`insert`/`stage`/`unstage`/`node_of`/`clear`/`for_each`)
    /// takes raw shard locks under `Acquired::acquire(CacheShard)`
    /// instead: the tripwire sees every one of those, and mechanism B's
    /// actual job — keeping `alloc` out of the lock — is upheld on the
    /// probe path.
    #[inline]
    fn take_cache_shard_r(
        &self,
        s: usize,
        _t: ticket::Held<ticket::CacheNoShard>,
    ) -> (
        crate::sync::RwLockReadGuard<'_, CacheShard<K, V>>,
        ticket::Held<ticket::CacheShardHeld>,
    ) {
        sync_order::check(sync_order::Fam::CacheShard, "cache.take_r");
        // Unlike the dep family's RAII guards, this inc/dec pair is
        // explicit: the critical sections between take and release must
        // stay panic-free (they are — map ops + Arc clones + stores).
        // Without the inc, `check`'s row scan never sees CacheShard held
        // and every ORD[CacheShard][*] cell is untestable in principle.
        // Inc BEFORE the grant ("register, then acquire" — the single rule
        // stated in job.rs::resolve and vtable_or_init): a buggy re-entrant
        // take then trips the re-held assert with a good diagnostic instead
        // of deadlocking raw on the lock.
        sync_order::held_inc(sync_order::Fam::CacheShard);
        let guard = self.shards[s].read();
        (guard, ticket::Held::<ticket::CacheShardHeld>::of())
    }

    /// cache shard write lock for the same probe/two-phase shape as
    /// [`Self::take_cache_shard_r`] (see there for the scope of the
    /// "sole acquisition point" claim).
    #[inline]
    fn take_cache_shard_w(
        &self,
        s: usize,
        _t: ticket::Held<ticket::CacheNoShard>,
    ) -> (
        crate::sync::RwLockWriteGuard<'_, CacheShard<K, V>>,
        ticket::Held<ticket::CacheShardHeld>,
    ) {
        sync_order::check(sync_order::Fam::CacheShard, "cache.take_w");
        // Inc BEFORE the grant — same "register, then acquire" rule as
        // take_cache_shard_r above.
        sync_order::held_inc(sync_order::Fam::CacheShard);
        let guard = self.shards[s].write();
        (guard, ticket::Held::<ticket::CacheShardHeld>::of())
    }

    /// Explicit release: return the ticket to CacheNoShard (normal path; Drop
    /// covers panic).
    #[inline]
    fn release_cache_shard_r(
        g: crate::sync::RwLockReadGuard<'_, CacheShard<K, V>>,
        _t: ticket::Held<ticket::CacheShardHeld>,
    ) -> ticket::Held<ticket::CacheNoShard> {
        sync_order::held_dec(sync_order::Fam::CacheShard);
        drop(g);
        ticket::Held::<ticket::CacheNoShard>::of()
    }

    #[inline]
    fn release_cache_shard_w(
        g: crate::sync::RwLockWriteGuard<'_, CacheShard<K, V>>,
        _t: ticket::Held<ticket::CacheShardHeld>,
    ) -> ticket::Held<ticket::CacheNoShard> {
        sync_order::held_dec(sync_order::Fam::CacheShard);
        drop(g);
        ticket::Held::<ticket::CacheNoShard>::of()
    }
}

impl<K, V> QueryCacheType<K, V> for DefaultCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    fn from_capacity(capacity: usize) -> Self {
        DefaultCache::with_capacity(capacity)
    }

    fn probe(
        &self,
        key: &K,
        alloc: impl FnOnce(&mut ticket::Held<ticket::CacheNoShard>) -> dep_graph::DepNodeIndex,
    ) -> (dep_graph::DepNodeIndex, Option<V>, bool) {
        // (3.3) ONE hash pass: the shard index here, the sketch bump on
        // the hit below.
        let h = Self::hash_key(key);
        let s = Self::shard_of_hash(h);
        // Phase 1: ONE read lock fetches node + value together — the whole
        // hit path.  The `referenced` bit preserves CLOCK recency exactly
        // like the old `lookup` did; stats are the CALLER's job (trait
        // contract 5 — a red node's probe touches no counters).
        //
        // Mechanism B: inside the critical section only the CacheShardHeld
        // ticket is present; dep shard/grow physically cannot get in (the
        // compile face of contract 3 + the tripwire's cross-family row).
        let hit = {
            let (shard, t1) = self.take_cache_shard_r(s, ticket::Held::fresh());
            let out = shard.map.get(key).map(|entry| {
                if let Some(v) = &entry.value {
                    entry.referenced.store(true, Ordering::Relaxed);
                    (entry.node, Some(v.clone()), false)
                } else {
                    // Registered node, no staged value: fresh registration,
                    // unstaged, or evicted.
                    (entry.node, None, false)
                }
            });
            let _ = Self::release_cache_shard_r(shard, t1);
            out
        };
        if let Some(hit) = hit {
            if hit.1.is_some() {
                self.note_access_hash(h);
            }
            return hit;
        }
        // Phase 2: allocate OUTSIDE any cache lock (trait contract 3 — the
        // dep-graph shard lock must never nest inside a cache shard lock),
        // then re-check under the write lock.  Mechanism B: this is the
        // minting point of the two-phase shape where "alloc is always outside
        // the lock" — reintroducing "in-lock alloc" = E0277 (CacheShardHeld
        // does not satisfy alloc's parameter type).
        let fresh_node = {
            let mut no_shard = ticket::Held::<ticket::CacheNoShard>::fresh();
            alloc(&mut no_shard)
        };
        let (mut shard, t2) = self.take_cache_shard_w(s, ticket::Held::fresh());
        if let Some(entry) = shard.map.get(key) {
            // Race loser: the winner registered first.  Discard our
            // allocated index (an orphan node — the same harmless class
            // as the pre-merge two-phase loser) and return the winner's
            // entry.
            let out = if let Some(v) = &entry.value {
                entry.referenced.store(true, Ordering::Relaxed);
                (entry.node, Some(v.clone()), false)
            } else {
                (entry.node, None, false)
            };
            let _ = Self::release_cache_shard_w(shard, t2);
            return out;
        }
        shard.map.insert(
            key.clone(),
            MergedEntry {
                node: fresh_node,
                value: None,
                referenced: AtomicBool::new(false),
                prev: None,
                next: None,
            },
        );
        let _ = Self::release_cache_shard_w(shard, t2);
        (fresh_node, None, true)
    }

    fn stage(&self, key: K, node: dep_graph::DepNodeIndex, v: V) {
        // (3.3) ONE hash pass — see `insert`.
        let h = Self::hash_key(&key);
        let s = Self::shard_of_hash(h);
        self.note_access_hash(h);
        self.admit_tick();
        let mut dropped: Vec<V> = Vec::new();
        // #8: the eviction loop's `stat_bump` may take the stat registry
        // UNDER this shard write lock — see `insert`.
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "cache.stage");
        {
            let mut shard = self.shards[s].write();
            let valued_before = shard.valued;
            if let Some(entry) = shard.map.get_mut(&key) {
                debug_assert_eq!(
                    entry.node, node,
                    "stage: node must match the probe-registered entry"
                );
                let old = entry.value.replace(v);
                let was_staged = old.is_some();
                if let Some(old) = old {
                    // The replaced payload drops post-lock (trait
                    // contract 4).
                    dropped.push(old);
                }
                if was_staged {
                    // Re-stage over a staged value (a red node's suspect
                    // entry replaced by the recompute): already chained —
                    // just promote.
                    Self::move_to_head(&mut shard, &key);
                } else {
                    // Fresh registration, unstaged, or evicted entry:
                    // (re-)join the chain.
                    Self::link_head(&mut shard, &key);
                    shard.valued += 1;
                    self.staged_global.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                // Defensive totality: `stage` without a prior probe
                // registration.  Upstream never does this (probe
                // precedes staging in `get_shared`), but the entry
                // carries the caller's node so the map stays consistent.
                shard.map.insert(
                    key.clone(),
                    MergedEntry {
                        node,
                        value: Some(v),
                        referenced: AtomicBool::new(false),
                        prev: None,
                        next: None,
                    },
                );
                Self::link_head(&mut shard, &key);
                shard.valued += 1;
                self.staged_global.fetch_add(1, Ordering::Relaxed);
            }
            // Eviction + TinyLFU admission — see `evict_with_admission`.
            let fresh = shard.valued > valued_before;
            self.evict_with_admission(&mut shard, &key, h, fresh, &mut dropped);
        } // shard lock released — payload Drops run from here on
        drop(dropped);
    }

    fn unstage(&self, key: &K) -> Option<dep_graph::DepNodeIndex> {
        let s = Self::shard_of_key(key);
        let mut dropped: Option<V> = None;
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "cache.unstage");
        let node = {
            let mut shard = self.shards[s].write();
            let entry = shard.map.get_mut(key)?;
            let node = entry.node;
            if let Some(old) = entry.value.take() {
                dropped = Some(old);
                let (prev, next) = (entry.prev.take(), entry.next.take());
                Self::unlink_entry(&mut shard, key, prev, next);
                // saturating_sub: hardening for the hypothetical window
                // where a panic landed between `entry.value.replace(v)`
                // and the `valued`/`staged_global` increments in `stage`
                // (today every panic source in stage/evict_one sits
                // either before any mutation or after both increments).
                // A debug overflow here would abort DURING unwind.
                shard.valued = shard.valued.saturating_sub(1);
                self.staged_global.fetch_sub(1, Ordering::Relaxed);
            }
            node
        };
        // Post-lock drop of the withdrawn payload (trait contract 4).
        drop(dropped);
        Some(node)
    }

    fn node_of(&self, key: &K) -> Option<dep_graph::DepNodeIndex> {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "cache.node_of");
        let shard = self.shards[Self::shard_of_key(key)].read();
        shard.map.get(key).map(|e| e.node)
    }

    fn clear(&self) {
        // Values go, nodes stay (trait contract 1): the next get
        // RECOMPUTES against the surviving node (d19) instead of
        // re-allocating one.  One shard lock at a time
        // (DepGraph::reset discipline); payloads drop after ALL locks
        // release.
        let mut dropped: Vec<V> = Vec::new();
        for shard_lock in &self.shards {
            // One shard lock at a time (DepGraph::reset discipline).
            let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "cache.clear");
            let mut shard = shard_lock.write();
            for entry in shard.map.values_mut() {
                if let Some(old) = entry.value.take() {
                    dropped.push(old);
                }
                entry.prev = None;
                entry.next = None;
            }
            shard.head = None;
            shard.tail = None;
            // Snapshot under THIS shard's lock (no concurrent stage can
            // interleave here); a single `store(0)` would swallow
            // concurrent increments on the other shards.
            let cleared = shard.valued;
            shard.valued = 0;
            self.staged_global.fetch_sub(cleared, Ordering::Relaxed);
        }
        drop(dropped);
    }

    fn len(&self) -> usize {
        // #10: `staged_global` mirrors `sum(shard.valued)` on every
        // mutation (stage/insert ++, evict/unstage --, clear snapshots to
        // 0) under that shard's own write lock. Relaxed is sufficient:
        // transient skew only over/under-evicts, and over-eviction is
        // pure perf loss (node half is never evicted). This drops 64
        // read locks (and the CacheShard edge in `stats()`) to one
        // atomic load.
        self.staged_global.load(Ordering::Relaxed)
    }

    fn for_each(&self, mut f: impl FnMut(&K, &V)) {
        for shard_lock in &self.shards {
            // One shard lock at a time.
            let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "cache.for_each");
            let shard = shard_lock.read();
            let mut cur = shard.head.clone();
            while let Some(k) = cur {
                let entry = shard.map.get(&k).expect("chain entry must exist");
                let next = entry.next.clone();
                if let Some(v) = &entry.value {
                    f(&k, v);
                }
                cur = next;
            }
        }
    }

    fn stats(&self) -> CacheStats {
        // `len()` is the embedded `staged_global` load (a Relaxed atomic
        // read), so `entries` is the lock-free half of the function.  What
        // still matters is the lock-ORDER rule on `stat_bump_cold`: never
        // take a shard lock while holding the stat-registry lock.  The
        // registry section below registers `StatRegistry` under that
        // family, which makes #9 genuinely enforced: a future `stats()`
        // that took shard locks AFTER the registry would trip
        // ORD[StatRegistry][CacheShard].  ("Entries FIRST" is not a scheduling
        // statement — nothing blocks on it today.)
        let entries = self.len();
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::StatRegistry, "cache.stats");
        let registry = self.stat_shards.lock();
        let (mut hits, mut misses, mut evictions) = (0u64, 0u64, 0u64);
        for slot in registry.iter() {
            hits += slot.hits.load(Ordering::Relaxed);
            misses += slot.misses.load(Ordering::Relaxed);
            evictions += slot.evictions.load(Ordering::Relaxed);
        }
        CacheStats {
            hits,
            misses,
            evictions,
            entries,
        }
    }

    fn record_hit(&self) {
        self.stat_bump(StatField::Hit);
    }
    fn record_miss(&self) {
        self.stat_bump(StatField::Miss);
    }
}

impl<K, V> Default for DefaultCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ── SingleCache (for unit keys) ───────────────────────────────────

/// Single-entry cache for queries with unit keys (`()`).
///
/// Analogous to Rust's `SingleCache<V>` in `caches.rs`.
/// O(1) with no hashing overhead.  Merged storage: the slot also holds
/// the query's dep-graph node, which — like every cache impl — survives
/// `unstage` and `clear` (only the value leaves; trait contract 1).
#[derive(Debug)]
pub struct SingleCache<V> {
    entry: RwLock<Option<SingleEntry<V>>>,
    stats_hits: AtomicU64,
    stats_misses: AtomicU64,
}

/// The single merged slot: node half (registration-once) + value half.
#[derive(Debug)]
struct SingleEntry<V> {
    node: dep_graph::DepNodeIndex,
    value: Option<V>,
}

impl<V: Clone + std::fmt::Debug> SingleCache<V> {
    pub fn new() -> Self {
        SingleCache {
            entry: RwLock::new(None),
            stats_hits: AtomicU64::new(0),
            stats_misses: AtomicU64::new(0),
        }
    }
}

impl<V: Clone + std::fmt::Debug> QueryCacheType<(), V> for SingleCache<V> {
    fn from_capacity(_capacity: usize) -> Self {
        Self::new()
    }
    fn probe(
        &self,
        _key: &(),
        alloc: impl FnOnce(&mut ticket::Held<ticket::CacheNoShard>) -> dep_graph::DepNodeIndex,
    ) -> (dep_graph::DepNodeIndex, Option<V>, bool) {
        // Single-lock cache: registered under Fam::CacheShard so the whole
        // cache family is visible to the tripwire.  One lock at a time:
        // each critical section registers separately and the guard is
        // dropped BEFORE the fresh-node alloc — holding CacheShard across
        // `alloc` trips the tripwire (CacheShard → DepShard is Forbid,
        // contract 3; see DefaultCache::probe's two-phase shape).
        {
            let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "single.probe");
            let entry = self.entry.read();
            if let Some(e) = entry.as_ref() {
                return (e.node, e.value.clone(), false);
            }
        }
        // Fresh registration two-phase: allocate the node OUTSIDE the
        // lock (trait contract 3), then re-check under the write lock.
        // Mechanism B: alloc takes the CacheNoShard ticket — in-lock alloc
        // = E0277.
        let fresh_node = {
            let mut no_shard = ticket::Held::<ticket::CacheNoShard>::fresh();
            alloc(&mut no_shard)
        };
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "single.probe");
        let mut entry = self.entry.write();
        if let Some(e) = entry.as_ref() {
            // Race loser: discard our allocated index, return the
            // winner's slot.
            return (e.node, e.value.clone(), false);
        }
        *entry = Some(SingleEntry {
            node: fresh_node,
            value: None,
        });
        (fresh_node, None, true)
    }
    fn stage(&self, _key: (), node: dep_graph::DepNodeIndex, v: V) {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "single.stage");
        let old = {
            let mut entry = self.entry.write();
            match entry.as_mut() {
                Some(e) => {
                    debug_assert_eq!(
                        e.node, node,
                        "stage: node must match the probe-registered slot"
                    );
                    std::mem::replace(&mut e.value, Some(v))
                }
                // Defensive totality: stage without a prior probe.
                None => {
                    *entry = Some(SingleEntry {
                        node,
                        value: Some(v),
                    });
                    None
                }
            }
        };
        // Post-lock drop of the replaced payload (trait contract 4).
        drop(old);
    }
    fn unstage(&self, _key: &()) -> Option<dep_graph::DepNodeIndex> {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "single.unstage");
        let mut old: Option<V> = None;
        let node = {
            let mut entry = self.entry.write();
            let e = entry.as_mut()?;
            let node = e.node;
            old = e.value.take();
            node
        };
        drop(old);
        Some(node)
    }
    fn node_of(&self, _key: &()) -> Option<dep_graph::DepNodeIndex> {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "single.node_of");
        self.entry.read().as_ref().map(|e| e.node)
    }
    fn clear(&self) {
        // Value goes, node STAYS (trait contract 1): the next get
        // recomputes against the surviving node instead of re-allocating.
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "single.clear");
        let old = {
            let mut entry = self.entry.write();
            match entry.as_mut() {
                Some(e) => e.value.take(),
                None => None,
            }
        };
        drop(old);
    }
    fn len(&self) -> usize {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "single.len");
        self.entry
            .read()
            .as_ref()
            .map_or(0, |e| usize::from(e.value.is_some()))
    }
    fn for_each(&self, mut f: impl FnMut(&(), &V)) {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "single.for_each");
        if let Some(e) = self.entry.read().as_ref() {
            if let Some(v) = &e.value {
                f(&(), v);
            }
        }
    }
    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.stats_hits.load(Ordering::Relaxed),
            misses: self.stats_misses.load(Ordering::Relaxed),
            evictions: 0,
            entries: self.len(),
        }
    }
    fn record_hit(&self) {
        self.stats_hits.fetch_add(1, Ordering::Relaxed);
    }
    fn record_miss(&self) {
        self.stats_misses.fetch_add(1, Ordering::Relaxed);
    }
}

impl<V: Clone + std::fmt::Debug> Default for SingleCache<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── DefIdCache (Vec-indexed for DefId keys) ───────────────────────

/// Vec-indexed cache for `DefId` keys.
///
/// Analogous to Rust's `DefIdCache<V>` in `caches.rs`.
/// O(1) lookup via `DefId.index` into a `Vec`, no hashing.
/// `DefId` is a `(CrateNum, DefIndex)` pair; this cache uses
/// `DefIndex` as the Vec index (assuming dense indexing).
///
/// Merged storage: each slot holds the key's dep-graph node (the node
/// half is registration-once — a slot's outer `None` means "never
/// probed", so probing allocates the node; the node survives unstage and
/// clear per trait contract 1) plus the evictable value.
///
/// DENSE-INDEX BOUND: `DefId` is a plain `usize` newtype, and callers
/// can produce sparse or sentinel indices (`DefId(usize::MAX / 2)` is
/// used as an "unknown" sentinel elsewhere in this codebase).  A naive
/// `resize_with(idx + 1)` would attempt a multi-exabyte allocation and
/// abort (capacity overflow).  Indices below [`DEFID_DENSE_LIMIT`] take
/// the O(1) dense path; anything above falls back to a hash map, so
/// sparse keys cost a probe instead of an OOM.
#[derive(Debug)]
pub struct DefIdCache<V> {
    entries: RwLock<Vec<Option<DefIdEntry<V>>>>,
    overflow: RwLock<FxHashMap<usize, DefIdEntry<V>>>,
    stats_hits: AtomicU64,
    stats_misses: AtomicU64,
}

/// One merged slot: node half (registration-once) + value half.
#[derive(Debug)]
struct DefIdEntry<V> {
    node: dep_graph::DepNodeIndex,
    value: Option<V>,
}

/// Largest `DefIndex` served by the dense `Vec` path (≈8 MB of slots in
/// the worst case; well above real crate def counts).  Higher indices
/// go to the overflow map.
const DEFID_DENSE_LIMIT: usize = 1 << 20;

impl<V: Clone + std::fmt::Debug> DefIdCache<V> {
    pub fn new() -> Self {
        DefIdCache {
            entries: RwLock::new(Vec::new()),
            overflow: RwLock::new(FxHashMap::default()),
            stats_hits: AtomicU64::new(0),
            stats_misses: AtomicU64::new(0),
        }
    }
}

impl<V: Clone + std::fmt::Debug> QueryCacheType<crate::hir::types::DefId, V> for DefIdCache<V> {
    fn from_capacity(_capacity: usize) -> Self {
        Self::new()
    }
    fn probe(
        &self,
        key: &crate::hir::types::DefId,
        alloc: impl FnOnce(&mut ticket::Held<ticket::CacheNoShard>) -> dep_graph::DepNodeIndex,
    ) -> (dep_graph::DepNodeIndex, Option<V>, bool) {
        // Registered under Fam::CacheShard (single-lock cache discipline;
        // dense/overflow are sequential, never co-held).  Each critical
        // section registers separately: the guard MUST NOT span the
        // fresh-node alloc — CacheShard → DepShard is Forbid (contract 3;
        // see DefaultCache::probe's two-phase shape).
        let idx = key.0;
        if idx < DEFID_DENSE_LIMIT {
            // Phase 1: read lock.
            {
                let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.probe");
                let entries = self.entries.read();
                if let Some(entry) = entries.get(idx).and_then(|e| e.as_ref()) {
                    return (entry.node, entry.value.clone(), false);
                }
            }
            // Phase 2: allocate OUTSIDE the lock (trait contract 3), then
            // re-check under the write lock.  Mechanism B: alloc takes the
            // CacheNoShard ticket — in-lock alloc = E0277.
            let fresh_node = {
                let mut no_shard = ticket::Held::<ticket::CacheNoShard>::fresh();
                alloc(&mut no_shard)
            };
            let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.probe");
            let mut entries = self.entries.write();
            if let Some(entry) = entries.get(idx).and_then(|e| e.as_ref()) {
                // Race loser: discard our index, return the winner's slot.
                return (entry.node, entry.value.clone(), false);
            }
            if idx >= entries.len() {
                // Holes stay outer-`None`: a never-probed DefId has no
                // node, and probing it later allocates one.
                entries.resize_with(idx + 1, || None);
            }
            entries[idx] = Some(DefIdEntry {
                node: fresh_node,
                value: None,
            });
            return (fresh_node, None, true);
        }
        // Sentinel/sparse index: a dense resize would try to allocate up
        // to `idx + 1` slots (multi-exabyte for sentinels like
        // `DefId(usize::MAX / 2)`) — overflow map, same two-phase shape.
        {
            let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.probe");
            let overflow = self.overflow.read();
            if let Some(entry) = overflow.get(&idx) {
                return (entry.node, entry.value.clone(), false);
            }
        }
        let fresh_node = {
            let mut no_shard = ticket::Held::<ticket::CacheNoShard>::fresh();
            alloc(&mut no_shard)
        };
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.probe");
        let mut overflow = self.overflow.write();
        if let Some(entry) = overflow.get(&idx) {
            // Race loser.
            return (entry.node, entry.value.clone(), false);
        }
        overflow.insert(
            idx,
            DefIdEntry {
                node: fresh_node,
                value: None,
            },
        );
        (fresh_node, None, true)
    }

    fn stage(&self, key: crate::hir::types::DefId, node: dep_graph::DepNodeIndex, v: V) {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.stage");
        let idx = key.0;
        let old = if idx < DEFID_DENSE_LIMIT {
            // One write lock, not two: the resize and the store happen in
            // the same critical section (Perf Book §13.1).
            let mut entries = self.entries.write();
            if idx >= entries.len() {
                entries.resize_with(idx + 1, || None);
            }
            match &mut entries[idx] {
                Some(entry) => {
                    debug_assert_eq!(
                        entry.node, node,
                        "stage: node must match the probe-registered slot"
                    );
                    std::mem::replace(&mut entry.value, Some(v))
                }
                // Defensive totality: stage without a prior probe.
                slot @ None => {
                    *slot = Some(DefIdEntry {
                        node,
                        value: Some(v),
                    });
                    None
                }
            }
        } else {
            let mut overflow = self.overflow.write();
            match overflow.get_mut(&idx) {
                Some(entry) => {
                    debug_assert_eq!(
                        entry.node, node,
                        "stage: node must match the probe-registered slot"
                    );
                    std::mem::replace(&mut entry.value, Some(v))
                }
                // Defensive totality: stage without a prior probe.
                None => {
                    overflow.insert(
                        idx,
                        DefIdEntry {
                            node,
                            value: Some(v),
                        },
                    );
                    None
                }
            }
        };
        // Post-lock drop of the replaced payload (trait contract 4).
        drop(old);
    }

    fn unstage(&self, key: &crate::hir::types::DefId) -> Option<dep_graph::DepNodeIndex> {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.unstage");
        let idx = key.0;
        let (old, node) = if idx < DEFID_DENSE_LIMIT {
            let mut entries = self.entries.write();
            let entry = entries.get_mut(idx).and_then(|e| e.as_mut())?;
            (entry.value.take(), entry.node)
        } else {
            // The overflow ENTRY stays (node preserved — trait contract
            // 1); only its value leaves.
            let mut overflow = self.overflow.write();
            let entry = overflow.get_mut(&idx)?;
            (entry.value.take(), entry.node)
        };
        // Post-lock drop (trait contract 4).
        drop(old);
        Some(node)
    }

    fn node_of(&self, key: &crate::hir::types::DefId) -> Option<dep_graph::DepNodeIndex> {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.node_of");
        let idx = key.0;
        if idx < DEFID_DENSE_LIMIT {
            self.entries
                .read()
                .get(idx)
                .and_then(|e| e.as_ref())
                .map(|e| e.node)
        } else {
            self.overflow.read().get(&idx).map(|e| e.node)
        }
    }

    fn clear(&self) {
        // Values go, nodes stay (trait contract 1).  One lock at a time;
        // payloads drop after their lock releases.
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.clear");
        let dense_dropped: Vec<V> = {
            let mut entries = self.entries.write();
            entries
                .iter_mut()
                .filter_map(|slot| slot.as_mut().and_then(|e| e.value.take()))
                .collect()
        };
        drop(dense_dropped);
        let over_dropped: Vec<V> = {
            let mut overflow = self.overflow.write();
            overflow
                .values_mut()
                .filter_map(|e| e.value.take())
                .collect()
        };
        drop(over_dropped);
    }

    fn len(&self) -> usize {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.len");
        let dense = self
            .entries
            .read()
            .iter()
            .filter(|e| matches!(e, Some(s) if s.value.is_some()))
            .count();
        dense
            + self
                .overflow
                .read()
                .values()
                .filter(|e| e.value.is_some())
                .count()
    }

    fn for_each(&self, mut f: impl FnMut(&crate::hir::types::DefId, &V)) {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "defid.for_each");
        for (i, slot) in self.entries.read().iter().enumerate() {
            if let Some(entry) = slot {
                if let Some(v) = &entry.value {
                    f(&crate::hir::types::DefId(i), v);
                }
            }
        }
        for (idx, entry) in self.overflow.read().iter() {
            if let Some(v) = &entry.value {
                f(&crate::hir::types::DefId(*idx), v);
            }
        }
    }

    fn stats(&self) -> CacheStats {
        // No CacheShard registration here: `len()` below registers — a
        // second would trip the same-family re-held assert.
        CacheStats {
            hits: self.stats_hits.load(Ordering::Relaxed),
            misses: self.stats_misses.load(Ordering::Relaxed),
            evictions: 0,
            entries: self.len(),
        }
    }

    fn record_hit(&self) {
        self.stats_hits.fetch_add(1, Ordering::Relaxed);
    }
    fn record_miss(&self) {
        self.stats_misses.fetch_add(1, Ordering::Relaxed);
    }
}

// ── Query trait ───────────────────────────────────────────────────

/// Marker trait for types that may be used as query keys.
///
/// This is the type-level gate for the "identity only" discipline: a query
/// key must be pure *semantic identity* (`DefId`, paths, structural
/// indices).  Positional data — `Span`s, byte offsets, source text — MUST
/// NOT implement this trait, so any attempt to put position into a key or
/// its fingerprint fails to compile (the gate is checked at compile time).
pub trait QueryKey: Clone + Eq + Hash + std::fmt::Debug + Send + Sync + 'static {}

/// Position-free, stable fingerprint of a query value.
///
/// Used for red-green incremental verification: after a node is
/// recomputed, its fingerprint is compared against the previously stored
/// one.  Implementations MUST exclude positional data (spans, byte
/// offsets, source text) — the fingerprint is the value's *semantic*
/// identity.  Spans stay available at runtime for diagnostics but never
/// take part in incremental identity.
pub trait StableHash {
    fn stable_hash(&self) -> u64;
}

/// `StableHash` through a borrow: the fingerprint is the referent's
/// semantic identity, so `&T` hashes exactly like `T`.  This is what lets
/// arena-path values (`Value = &'q V`) fingerprint without a clone.
impl<T: StableHash + ?Sized> StableHash for &T {
    #[inline]
    fn stable_hash(&self) -> u64 {
        (**self).stable_hash()
    }
}

/// What a query's provided value must satisfy: cheap to pass by value
/// (`Clone` — a deep copy for owned values, a pointer copy for `&'q V`),
/// thread-shareable, printable, fingerprintable, and alive for the arena
/// lifetime `'q` — deliberately NOT `'static`, so `Value = &'q V` works.
pub trait QueryValue<'q>: Clone + std::fmt::Debug + Send + Sync + 'q + StableHash {}

impl<'q, T: Clone + std::fmt::Debug + Send + Sync + 'q + StableHash> QueryValue<'q> for T {}

/// Defines a compiler query: a mapping from `Key → Value` that is
/// computed on demand and cached for subsequent lookups.
///
/// `'q` is the arena lifetime: every value the query provides or stores
/// is alive for `'q`.  Owned values satisfy this for every `'q`
/// (`'static: 'q`); borrowed values (`Value = &'q V`) pin the query to
/// systems driven by that arena — enforced by the type system, no
/// transmute (the old `CheckFnBody` bridge is gone).
///
/// Each query selects its own `Cache` type:
/// - `DefaultCache<K, V>` — HashMap-backed, for most keys
/// - `SingleCache<V>` — for unit keys `()`
/// - `DefIdCache<V>` — Vec-indexed, for `DefId` keys
///
/// Analogous to Rust's `QueryConfig` + `QueryKey::Cache`.
pub trait Query<'q, 'src: 'q>: Sized + 'static {
    type Key: QueryKey;
    type Value: QueryValue<'q>;
    /// Storage path selecting the [`Erasable`] two-path split: [`Owned`]
    /// or [`Arena`] (bare `&'q V`, zero-cost).  Required (no default —
    /// associated type defaults are unstable): owned values name
    /// [`Owned`]; a borrowed `Value` names [`Arena`].
    type Storage: Erasable<'q, Self::Value>;
    /// The cache implementation for this query's key type.  It stores
    /// the *erased* form ([`Stored`]), never the provided value.
    type Cache: QueryCacheType<Self::Key, Stored<'q, 'src, Self>> + Default + Send + Sync;

    fn descriptor() -> QueryDescriptor;
    /// Compute the value for `key`.  Nested gets that only READ a
    /// dependency's value should prefer [`query_get_shared`] /
    /// [`QuerySystem::get_shared`]: [`query_get`] / [`QuerySystem::get`]
    /// run the Owned-path `erased_to_provided`, which deep-clones the
    /// payload on every cache hit (the cache always holds one `Arc`
    /// handle of its own, so the cheap unwrap branch can never fire).
    fn compute<S: Scheduler>(
        key: &Self::Key,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Self::Value;

    /// Per-query cycle recovery handler (rustc-style: `fn_sig` → error
    /// type, `layout_of` → E0733, `variances_of` → all-bivariant).
    ///
    /// `key` is the re-entered / waited-on key; `stack` is the current
    /// thread's committed active prefix — the failed re-entry is NOT
    /// pushed (so a self-cycle observes its key exactly once, from the
    /// outer commit).  On the cross-thread path the resumable waiter
    /// ferries its own prefix, so `stack` names the breaking side.
    ///
    /// The handler runs OUTSIDE every query lock (see `acquire_key_slot` /
    /// `get_shared`), so it may nest further gets — those go through the
    /// same cycle machinery and are recursion-safe.  A handler panic is
    /// equivalent to a compute panic (caught by the outer compute's
    /// `catch_unwind` → staging-panic path → poison).  The recovery value
    /// is NOT staged: the breaking side's outer compute publishes
    /// normally and incidentally caches the combined result (rustc's
    /// `FixResult` effect for free); the framework records the recovery
    /// dep edge on the caller's behalf.
    ///
    /// `fn` pointer, not a closure: the handler is stateless and has no
    /// session access (a provider-taking tier-2 form would require
    /// S-ifying `QueryVTable` — deferred until a real `variances_of`
    /// need appears).  Default `None` propagates the cycle error —
    /// today's behavior, unchanged.
    ///
    /// Tier-1 legacy form: when a query defines BOTH this and
    /// [`Query::cycle_processor`], the processor (tier-2) runs first and
    /// this is only a fallback.  New queries should prefer the
    /// processor: it observes the full [`CycleContext`] (waiter prefix +
    /// wait chain on the cross-thread path, where this form sees `&[]`)
    /// and can abort with a pinpointing note instead of recovering.
    fn cycle_recovery() -> Option<CycleRecoveryFn<'q, 'src, Self>> {
        None
    }

    /// Per-query loop processor (tier-2): resumes a cycle — same-thread
    /// or cross-thread — with the full [`CycleContext`] and decides
    /// between local recovery and a pinpointed abort.
    ///
    /// Runs OUTSIDE every query lock with the same re-entrancy, panic
    /// (→ poison), non-staging, and dep-edge discipline as
    /// [`Query::cycle_recovery`].  `fn` pointer, not a closure, for the
    /// same reason as the tier-1 form; per-query state (counts,
    /// diagnostics) lives in the query module's own statics, matching
    /// the existing test pattern.  Default `None` falls back to the
    /// tier-1 handler, then to the cycle error.
    fn cycle_processor() -> Option<CycleProcessorFn<'q, 'src, Self>> {
        None
    }

    /// Erase `key` into the closed [`TaggedQueryKey`] union for the
    /// `node_to_key` reverse map.  Default: the [`TaggedQueryKey::Opaque`]
    /// fallback (type-erased box, `TypeId`-checked on recovery) — keeps
    /// function-local test queries working with zero changes.  Production
    /// queries override this to return their typed variant (no `Any`).
    fn into_tagged(key: &Self::Key) -> TaggedQueryKey {
        TaggedQueryKey::Opaque(TypeId::of::<Self>(), Box::new(key.clone()))
    }

    /// Recover a typed key from a [`TaggedQueryKey`].  Default matches
    /// only the [`TaggedQueryKey::Opaque`] fallback with a `TypeId` check
    /// plus `Any` downcast.  Production queries override this to match
    /// their typed variant (no downcast).
    fn from_tagged(tag: &TaggedQueryKey) -> Option<Self::Key> {
        match tag {
            TaggedQueryKey::Opaque(ty, boxed) if *ty == TypeId::of::<Self>() => {
                let key_dyn: &(dyn CloneAny + Send + Sync) = &**boxed;
                (key_dyn as &dyn std::any::Any)
                    .downcast_ref::<Self::Key>()
                    .cloned()
            }
            _ => None,
        }
    }
}

/// Per-query cycle handler signature.  See [`Query::cycle_recovery`].
// Bound required: without `Q: Query<'q, 'src>` the associated-type projections
// are E0220 (the `type_alias_bounds` warn that comes with it matches the
// pre-existing alias at `node_to_key`'s value type).
pub type CycleRecoveryFn<'q, 'src, Q: Query<'q, 'src>> = fn(&Q::Key, &[Q::Key]) -> Q::Value;

/// Per-query loop-processor signature.  See [`Query::cycle_processor`].
// Same E0220 bound requirement as [`CycleRecoveryFn`].
pub type CycleProcessorFn<'q, 'src, Q: Query<'q, 'src>> =
    fn(&CycleContext<'q, 'src, Q>) -> CycleDecision<Q::Value>;

/// What the framework traffics in for `Q`: the *erased* value handle —
/// `Arc<V>` on the [`Owned`] path, `&'q V` on the [`Arena`] path.
/// Bound required for the same E0220 reason as above.
pub type Stored<'q, 'src, Q: Query<'q, 'src>> =
    <<Q as Query<'q, 'src>>::Storage as Erasable<'q, <Q as Query<'q, 'src>>::Value>>::Erased;

/// Provider side → cache side for `Q` (one `Arc` wrap on [`Owned`],
/// identity on [`Arena`]).  Runs once per fresh compute.
#[inline]
fn erase_value<'q, 'src, Q: Query<'q, 'src>>(provided: Q::Value) -> Stored<'q, 'src, Q> {
    <Q::Storage as Erasable<'q, Q::Value>>::provided_to_erased(provided)
}

/// Cache side → provider side for `Q` (refcount bump on [`Owned`], copy
/// on [`Arena`]).  Runs on every serve — never a deep clone.
#[inline]
fn provide_value<'q, 'src, Q: Query<'q, 'src>>(erased: &Stored<'q, 'src, Q>) -> Q::Value {
    <Q::Storage as Erasable<'q, Q::Value>>::erased_to_provided(erased)
}

impl QueryKey for crate::hir::types::DefId {}

// ── QueryVTable ─────────────────────────────────────────────────

/// One thread's cycle/depth state for a single query type, in exactly one
/// system generation (see `QueryVTable::active_tls`).
///
/// `set` mirrors `stack` so that cycle detection is O(1) instead of an
/// O(depth) scan.  The stack and its set are co-located — one SmallVec
/// slot fetch gets everything the check needs (Perf Book §8.1).
#[derive(Debug)]
struct ThreadActiveState<'q, 'src, Q: Query<'q, 'src>> {
    stack: Vec<Q::Key>,
    set: rustc_hash::FxHashSet<Q::Key>,
}

// Manual `Default` (not derived): the derive would add a spurious
// `Q: Default` bound, but empty state needs nothing from `Q`.
impl<'src, 'q, Q: Query<'q, 'src>> Default for ThreadActiveState<'q, 'src, Q> {
    fn default() -> Self {
        ThreadActiveState {
            stack: Vec::new(),
            set: rustc_hash::FxHashSet::default(),
        }
    }
}

/// This thread's cycle/depth partitions for ONE query type `Q`: one entry
/// per `QuerySystem` the thread has touched, so `Q(k)` legally active in
/// system B never aliases `Q(k)` active in system A (cross-system nesting,
/// the f1/d20/d22 shape — a flat per-`Q` slot would false-positive a
/// cycle there).  The single-system case — every hot path — is one
/// inline SmallVec slot and one `(system_id)` compare.
#[derive(Debug)]
struct ActivePartitions<'q, 'src, Q: Query<'q, 'src>> {
    parts: SmallVec<[(usize, u64, ThreadActiveState<'q, 'src, Q>); 1]>,
}

// Manual `Default` (not derived): same spurious-bound reason as above.
impl<'src, 'q, Q: Query<'q, 'src>> Default for ActivePartitions<'q, 'src, Q> {
    fn default() -> Self {
        ActivePartitions {
            parts: SmallVec::new(),
        }
    }
}

/// Build the "cycle detected" error.  `#[cold]` keeps the formatting
/// (and its allocations) out of `validate_enter`'s hot body, so the
/// common path stays compact and the branch settles as never-taken
/// (Perf Book §11.3).
#[cold]
fn cycle_error<'src, 'q, Q: Query<'q, 'src>>(
    name: &'static str,
    key: &Q::Key,
    stack: &[Q::Key],
    note: Option<String>,
) -> QueryError {
    let mut keys: Vec<String> = stack.iter().map(|k| format!("{:?}", k)).collect();
    keys.push(format!("{:?}", key));
    let rendered = keys
        .iter()
        .enumerate()
        .map(|(i, s)| format!("  {}: {}", i, s))
        .collect::<Vec<_>>()
        .join("\n    ");
    QueryError::Cycle {
        query_name: name,
        key: format!("{:?}", key),
        stack: keys,
        note,
    }
}

/// Cross-thread cycle error with the waiter's ferried prefix rendered
/// into the pinpointing note (the `stack` keeps the wait-graph chain,
/// the `note` names the loop keys on the breaking side).
#[cold]
fn cross_thread_cycle_error<'src, 'q, Q: Query<'q, 'src>>(
    name: &'static str,
    key: &Q::Key,
    waiter_prefix: &[Q::Key],
    chain: &[std::thread::ThreadId],
    note: Option<String>,
) -> QueryError {
    let rendered_prefix = waiter_prefix
        .iter()
        .map(|k| format!("{:?}", k))
        .collect::<Vec<_>>()
        .join(" -> ");
    let pinpoint = format!(
        "cross-thread loop on {:?} (waiter prefix: [{}]){}",
        key,
        rendered_prefix,
        note.map(|n| format!(": {}", n)).unwrap_or_default(),
    );
    QueryError::Cycle {
        query_name: name,
        key: format!("{:?}", key),
        stack: chain.iter().map(|t| format!("{:?}", t)).collect(),
        note: Some(pinpoint),
    }
}

/// Build the "depth limit exceeded" error.  `#[cold]` for the same
/// reason as [`cycle_error`].
#[cold]
fn depth_error<'src, 'q, Q: Query<'q, 'src>>(
    name: &'static str,
    stack: &[Q::Key],
    max: usize,
    total: usize,
) -> QueryError {
    let keys: Vec<String> = stack.iter().map(|k| format!("{:?}", k)).collect();
    QueryError::DepthLimit {
        query_name: name,
        max,
        total,
        stack: keys,
    }
}

/// Type-erased trait for re-requesting a dirty dependency of any query
/// type during transitive green verification.
///
/// The re-request's VALUE is discarded — verification only needs the
/// recompute (and its cache/poison side effects) to run, and the served
/// value is re-read from the cache afterwards.  Returning nothing keeps
/// borrowed values (`&'q V`, which cannot name `dyn Any`) on exactly the
/// same footing as owned ones.
trait QueryVTableDyn<'q, 'src, S: Scheduler>: Send + Sync {
    /// `TypeId::of::<Q>()` for the concrete query — the registry key.
    fn query_type_id(&self) -> TypeId;
    /// The concrete query's `descriptor().name`.  Exists so the
    /// uniqueness `debug_assert!` in `vtable_or_init` can scan the
    /// registry WITHOUT knowing each entry's `Q`: names
    /// are load-bearing identity for `SchedulePoint::{CachedHit,
    /// AfterPublish}` gate matching and for cycle diagnostics, and
    /// neither has a second key to disambiguate.  Called only from that
    /// debug assert, so in release the slot is dead weight.
    fn descriptor_name(&self) -> &'static str;
    /// `self` erased to a thin pointer; may only be cast back to the
    /// concrete vtable named by [`query_type_id`][Self::query_type_id].
    fn as_concrete_ptr(&self) -> *const ();
    /// Execute the query for the given type-erased key ON `system` — the
    /// system whose dep graph the caller's transitive verification
    /// walked.  The provider only supplies compute inputs; a provider
    /// bound to a different `QuerySystem` must never redirect the
    /// re-request (and its cache/poison side effects) into a foreign
    /// session whose dep graph was walked. Redirecting into a foreign
    /// session would leak cache/poison side effects across query systems.
    fn get_shared_erased(
        &self,
        system: &QuerySystem<'q, 'src, S>,
        key: &dyn std::any::Any,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Result<(), QueryError>;
}

/// Checked vtable downcast: `Some` iff `erased` is exactly `Q`'s vtable.
/// Sound by construction — registration inserts each concrete vtable
/// under its own `TypeId::of::<Q>()` and the map never moves entries
/// across keys — so no `Any` (which would demand `'static` values) is
/// needed.  A free function rather than a trait method because generic
/// methods are not dyn-compatible.
fn downcast_vtable<'a, 'src, 'q, Q: Query<'q, 'src>, S: Scheduler>(
    erased: &'a (dyn QueryVTableDyn<'q, 'src, S> + 'q),
) -> Option<&'a QueryVTable<'q, 'src, Q>> {
    if erased.query_type_id() == TypeId::of::<Q>() {
        // SAFETY: the TypeId check above guarantees the concrete
        // type; vtables are Sized so the pointer is thin.
        Some(unsafe { &*(erased.as_concrete_ptr() as *const QueryVTable<'q, 'src, Q>) })
    } else {
        None
    }
}

/// A unified record for a single query type: metadata, the merged cache
/// (whose entries carry the per-key dep-node registry), and active-key
/// state.
///
/// Analogous to Rust's `QueryVTable` in `plumbing.rs`.
///
/// Everything that is keyed by query type lives here rather than in a
/// `TypeId`-indexed side table on `QuerySystem`.  That removes a second
/// hash probe and a `Box<dyn Any>` downcast from the hot path, and — the
/// part that actually matters under `par_get` — it removes a *global*
/// lock: two threads resolving nodes for different query types no longer
/// contend on a single shared `RwLock`.
#[derive(Debug)]
pub(crate) struct QueryVTable<'q, 'src, Q: Query<'q, 'src>> {
    pub descriptor: QueryDescriptor,
    /// `None` for `descriptor.untracked` queries: the cache trait is
    /// node-ful (`stage`, `node_of`) and an untracked key must never own
    /// a node index, so constructing the (64-shard) cache for a memo-only
    /// query would be ~4 KiB of dead memory per query type.  Untracked
    /// gets branch off before the tracked path (`get_shared` →
    /// `get_untracked`), so no tracked access site can see `None` — every
    /// access asserts this.
    pub cache: Option<Q::Cache>,
    /// `Q::cycle_recovery()` captured once at registration — the generic
    /// interception points (`acquire_key_slot` / `get_shared`) read this
    /// instead of going through type-erased dispatch.
    pub cycle_recovery: Option<CycleRecoveryFn<'q, 'src, Q>>,
    /// `Q::cycle_processor()` captured once at registration — the tier-2
    /// loop processor.  Checked BEFORE `cycle_recovery` on both the
    /// same-thread and the cross-thread (resumable-waiter) paths.
    pub cycle_processor: Option<CycleProcessorFn<'q, 'src, Q>>,
    /// This thread's cycle/depth partitions, monomorphized per `Q`:
    /// `ThreadLocal` shards by thread with no hash and no lock — the
    /// slot IS the thread's state.  Each entry is partitioned by
    /// `QuerySystem` id (see [`ActivePartitions`]).
    active: thread_local::ThreadLocal<std::cell::RefCell<ActivePartitions<'q, 'src, Q>>>,
    /// Memo-only store for `descriptor.untracked` queries: `Key → Stored`.
    /// Kept OUTSIDE `Q::Cache` deliberately — the cache trait is node-ful
    /// (`stage(node)`, `node_of`) and an untracked key must never own a
    /// node index.  Dies with the vtable on `clear_all` (registry replace),
    /// so no generation stamp is needed: vtable lifetime == world epoch.
    /// Leaf lock: acquire under `Fam::CacheShard` like the cache shards.
    untracked: crate::sync::RwLock<FxHashMap<Q::Key, Stored<'q, 'src, Q>>>,
}

impl<'src, 'q, Q: Query<'q, 'src>, S: Scheduler> QueryVTableDyn<'q, 'src, S>
    for QueryVTable<'q, 'src, Q>
{
    fn query_type_id(&self) -> TypeId {
        TypeId::of::<Q>()
    }

    fn descriptor_name(&self) -> &'static str {
        self.descriptor.name
    }

    fn as_concrete_ptr(&self) -> *const () {
        self as *const Self as *const ()
    }

    fn get_shared_erased(
        &self,
        system: &QuerySystem<'q, 'src, S>,
        key: &dyn std::any::Any,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Result<(), QueryError> {
        // Downcast the key to Q::Key.
        let key = key
            .downcast_ref::<Q::Key>()
            .ok_or_else(|| QueryError::TypeMismatch {
                expected: std::any::TypeId::of::<Q::Key>(),
                got: key.type_id(),
            })?;
        // Execute on `system` — the system whose dep graph the caller's
        // verification walked — NOT on `provider.query_system()`: the
        // provider only supplies compute inputs, and a provider bound to
        // a different session must never receive this session's
        // recomputes — that would leak cache/poison side effects into
        // the foreign system.  Still goes through the full get_shared
        // logic including waiters; the value is discarded (see the trait
        // docs) and re-read from the cache by the verifier.
        system.get_shared::<Q>(key.clone(), provider)?;
        Ok(())
    }
}

/// Fetch this thread's [`ThreadActiveState`] for `system_id`, creating
/// the partition on first touch.  A partition whose generation stamp
/// predates `epoch` is panic-leak residue: `clear_all` only runs at full
/// thread quiescence (it holds the quiesce write lock, which every live
/// frame's read guard excludes), so no live frame can own it — reset it
/// in place.  Single-system threads never leave slot 0: one compare.
fn active_partition<'src, 'q, 'p, Q: Query<'q, 'src>>(
    parts: &'p mut ActivePartitions<'q, 'src, Q>,
    system_id: usize,
    epoch: u64,
) -> &'p mut ThreadActiveState<'q, 'src, Q> {
    let idx = match parts.parts.iter().position(|(sid, _, _)| *sid == system_id) {
        Some(i) => i,
        None => {
            parts
                .parts
                .push((system_id, epoch, ThreadActiveState::default()));
            parts.parts.len() - 1
        }
    };
    let slot = &mut parts.parts[idx];
    if slot.1 != epoch {
        slot.1 = epoch;
        slot.2.stack.clear();
        slot.2.set.clear();
    }
    &mut slot.2
}

impl<'src, 'q, Q: Query<'q, 'src>> QueryVTable<'q, 'src, Q> {
    pub fn new() -> Self {
        let d = Q::descriptor();
        assert!(
            !(d.untracked && d.eval_always),
            "query '{}': untracked memoizes unconditionally, eval_always recomputes every get — contradictory",
            d.name,
        );
        let cache = if d.untracked {
            None
        } else {
            Some(Q::Cache::from_capacity(d.cache_capacity))
        };
        QueryVTable {
            descriptor: d,
            cache,
            cycle_recovery: Q::cycle_recovery(),
            cycle_processor: Q::cycle_processor(),
            active: thread_local::ThreadLocal::new(),
            untracked: crate::sync::RwLock::new(FxHashMap::default()),
        }
    }

    /// Look up the stable `DepNodeIndex` for `key`, if one exists.
    /// One cache shard read-lock; no system-wide lock; NEVER registers.
    /// Routed to the merged cache's `node_of` — the node registry now
    /// lives inside `Q::Cache` (one map holds node + value per key).
    #[inline]
    fn lookup_node(&self, key: &Q::Key) -> Option<dep_graph::DepNodeIndex> {
        self.cache.as_ref()?.node_of(key)
    }

    /// Read-only hit/miss view for diagnostics and tests — the numbers the
    /// serve verdict records at `try_serve_verified`.  `None` only for an
    /// untracked query, which owns no cache.
    pub fn lookup_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }

    /// Read-only cycle and depth validation against the current thread's
    /// active partition.  Does NOT mutate the stack — use this before deciding
    /// whether the current thread will actually compute the query.
    ///
    /// Returns [`EnterError::Cycle`] with the TYPED stack cloned out; the
    /// caller ferries it to lock-free ground where a per-query handler
    /// may run.  No lock is held on ANY path here (the state is TLS, and
    /// the borrow ends before this returns), so nested gets from a
    /// handler are safe.
    ///
    /// `total` is this thread's session frame count, passed in by the
    /// caller (`acquire_key_slot` receives it from the guard bump) so
    /// the compute path pays zero extra session accesses here.
    #[inline]
    fn validate_enter(
        &self,
        system_id: usize,
        epoch: u64,
        key: &Q::Key,
        total: u32,
    ) -> Result<(), EnterError<'q, 'src, Q>> {
        // Global cross-type depth backstop: `total` counts every nested
        // `get_shared` frame on this thread, across ALL systems (the
        // guard bump runs BEFORE this check, so the outermost frame sees
        // 1), while the per-type check below only bounds THIS query
        // type's stack.  A chain of distinct query types — possibly
        // spread across systems via a mispaired provider — accumulates
        // one frame per type and would otherwise be unbounded.
        if total > MAX_GLOBAL_DEPTH as u32 {
            // The GLOBAL frame count is the depth that fired here — a chain
            // of DISTINCT query types accumulates one frame per type, so the
            // per-type stack can be nearly empty while the total is huge.
            // It goes into `total`, which the Display reports as the firing
            // depth.  The per-type stack is still attached when one exists:
            // it is the useful part for locating the runaway chain (and for
            // a single-type chain it matches the total).
            let tls = self
                .active
                .get_or(|| std::cell::RefCell::new(ActivePartitions::default()));
            let stack: Vec<Q::Key> = tls
                .borrow()
                .parts
                .iter()
                .find(|(sid, ep, _)| *sid == system_id && *ep == epoch)
                .map(|(_, _, part)| part.stack.clone())
                .unwrap_or_default();
            return Err(EnterError::Fatal(depth_error::<Q>(
                self.descriptor.name,
                &stack,
                MAX_GLOBAL_DEPTH,
                total as usize,
            )));
        }

        // One slot fetch, then: O(1) cycle check and the depth
        // check both read the same partition.  Set-before-depth ordering
        // is load-bearing: a cyclic re-entry arriving at the depth
        // boundary reports a cycle (through the handler), never a fatal
        // depth error.
        let tls = self
            .active
            .get_or(|| std::cell::RefCell::new(ActivePartitions::default()));
        {
            let mut guard = tls.borrow_mut();
            let part = active_partition(&mut guard, system_id, epoch);
            if part.set.contains(key) {
                return Err(EnterError::Cycle {
                    stack: part.stack.clone(),
                });
            }
            if self.descriptor.depth_limit && part.stack.len() >= MAX_QUERY_DEPTH {
                return Err(EnterError::Fatal(depth_error::<Q>(
                    self.descriptor.name,
                    &part.stack,
                    MAX_QUERY_DEPTH,
                    total as usize,
                )));
            }
        }
        Ok(())
    }

    /// Push `key` onto the current thread's active stack.  Only called when
    /// this thread is the one computing the query.
    fn commit_enter(&self, system_id: usize, epoch: u64, key: &Q::Key) {
        let tls = self
            .active
            .get_or(|| std::cell::RefCell::new(ActivePartitions::default()));
        let mut guard = tls.borrow_mut();
        let state = active_partition(&mut guard, system_id, epoch);
        state.stack.push(key.clone());
        state.set.insert(key.clone());
    }

    /// Pop the current thread's key off its active stack.  When the stack
    /// becomes empty, the thread's partition is removed entirely.
    pub fn leave(&self, system_id: usize, epoch: u64) {
        let tls = self
            .active
            .get_or(|| std::cell::RefCell::new(ActivePartitions::default()));
        {
            let mut parts = tls.borrow_mut();
            let Some(idx) = parts.parts.iter().position(|(sid, _, _)| *sid == system_id) else {
                return;
            };
            let slot = &mut parts.parts[idx];
            if slot.1 != epoch {
                // Stale generation: panic-leak residue already superseded
                // by a `clear_all` reset — there is nothing live to pop.
                slot.1 = epoch;
                slot.2.stack.clear();
                slot.2.set.clear();
                return;
            }
            if let Some(key) = slot.2.stack.pop() {
                slot.2.set.remove(&key);
            }
            if slot.2.stack.is_empty() {
                parts.parts.swap_remove(idx);
            }
        }
    }

    /// Clone this thread's committed prefix for `system_id` without
    /// mutating it.  The resumable waiter calls this at suspension time
    /// so the cross-thread context ferries the waiter's own loop prefix
    /// to the processor — no lock held on return, so nested gets stay
    /// safe.  Stale-epoch partitions read as empty (same reset
    /// discipline as `leave`).
    fn snapshot_stack(&self, system_id: usize, epoch: u64) -> Vec<Q::Key> {
        let tls = self
            .active
            .get_or(|| std::cell::RefCell::new(ActivePartitions::default()));
        let guard = tls.borrow();
        guard
            .parts
            .iter()
            .find(|(sid, ep, _)| *sid == system_id && *ep == epoch)
            .map(|(_, _, part)| part.stack.clone())
            .unwrap_or_default()
    }
}

// ── IntoQueryKey ───────────────────────────────────────────────────

/// Argument-conversion trait used by `QuerySystem::get`.
///
/// A query that accepts a `Key` of type `DefId` can also be called with
/// a `usize`, since `DefId(usize)` is a trivial conversion.  This avoids
/// forcing callers to wrap every argument.
///
/// Analogous to Rust's `IntoQueryKey` in `into_query_key.rs`.
pub trait IntoQueryKey<K> {
    fn into_query_key(self) -> K;
}

/// Identity conversion — every type can be converted to itself.
impl<K> IntoQueryKey<K> for K {
    #[inline(always)]
    fn into_query_key(self) -> K {
        self
    }
}

/// `usize` → `DefId` (since `DefId` is a newtype around `usize`).
impl IntoQueryKey<crate::hir::types::DefId> for usize {
    #[inline(always)]
    fn into_query_key(self) -> crate::hir::types::DefId {
        crate::hir::types::DefId(self)
    }
}

/// Green-hit probe against an already-resolved vtable: resolve the stable
/// `DepNodeIndex` for `key` (creating it on first encounter) and
/// opportunistically serve a green cache hit.
///
/// Probe the merged cache for `key`'s node + value.
///
/// ONE cache shard read lock fetches node and value together (the
/// node_map/DefaultCache merge's whole point); on a miss the fresh
/// two-phase registration runs inside `cache.probe` with the dep-graph
/// node allocation OUTSIDE any cache lock (trait contract 3 — the
/// dep-graph shard lock must never nest inside a cache shard lock).
///
/// Old gating semantics preserved: `cached` is `Some` only when the node
/// is green *and* the cache still holds a value, so `get` can
/// short-circuit; a red node's staged value is discarded here (one
/// refcount pair — ns-level) and `eval_always` never sees a cache value.
/// Hit/miss stats count under the same gate (a red node's probe touches
/// no counters — trait contract 5); both bumps are lock-free atomic adds.
///
/// Returns `(node, cached_value, fresh)`: `fresh` is true only when THIS
/// call registered the node — the caller registers the `node_to_key`
/// mapping exactly once on that path, keeping the global write lock and
/// the `Box`ed key allocation off the hit path.
fn probe<'src, 'q, Q: Query<'q, 'src>, S: Scheduler>(
    vtable: &QueryVTable<'q, 'src, Q>,
    dep_graph: &dep_graph::DepGraph<S>,
    key: &Q::Key,
) -> (dep_graph::DepNodeIndex, Option<Stored<'q, 'src, Q>>, bool) {
    // Tracked-only entry point: an untracked query branches to the memo
    // map before any cache access (see `get_shared`), so the cache is
    // always constructed here.
    let cache = vtable
        .cache
        .as_ref()
        .expect("untracked query reached the tracked probe path");
    let (node, cached, fresh) = cache.probe(key, |_no_cache_shard| {
        // The node's DECLARED durability is stamped HERE — inside the
        // allocation closure, BEFORE the node becomes reachable through
        // the cache registration.  `set_node_durability` OVERWRITES the
        // durability bits, so it must run exactly once per node:
        // re-running it on cache hits (the old placement) would raise an
        // already-aggregated node back to its declared level, defeating
        // the finish-time `min(declared, reads)` lowering
        // (`DepGraph::lower_durability_to`) and letting the per-level
        // durability fast path serve cached values past lower-durability
        // marks inside the node's own closure — a stale-serve soundness
        // hole.  A registration-race loser stamps only its own orphan
        // index, which nothing ever references.
        let node = dep_graph.allocate_node_index(vtable.descriptor.content_class);
        dep_graph.set_node_durability(node, vtable.descriptor.durability);
        node
    });
    if !vtable.descriptor.eval_always && dep_graph.is_green(node) {
        match cached {
            // NO `record_hit` HERE — see below.
            Some(value) => {
                return (node, Some(value), false);
            }
            None => cache.record_miss(),
        }
    }
    // Red node, eval_always, or a fresh registration (which cannot have
    // a cached value) — `None` is exact here.
    (node, None, fresh)
    //
    // Hit accounting is the SERVE VERDICT's job, not this probe's.  A green
    // node with a staged value is only a CANDIDATE: `serve_cached_verified`
    // (the caller) can still reject it — the walk finds a red transitive
    // dependency, the re-requested dependency recomputes with a CHANGED
    // fingerprint that re-dirties this node — and the lookup turns into a
    // recompute.  Counting that as a hit (the old placement) made the
    // hit-ratio most optimistic exactly where lazy invalidation matters:
    // every first touch of a transitively-stale reader.  So the hit moves
    // to the two points where the verdict is actually in (`serve_cached_verified`,
    // `serve_reprobed`), and each fall-through records the compensating miss.
    //
    // The miss above stays here and stays exact: no candidate was staged.
    // The compute path records nothing, so the compensating miss on the
    // fall-through cannot double-count.
}

// ── QuerySystem ──────────────────────────────────────────────────

/// Error returned when a query cycle, depth limit, poison, or type mismatch
/// is detected.  Use the variants for programmatic checks — never the
/// `Display` text.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[must_use]
pub enum QueryError {
    Cycle {
        query_name: &'static str,
        key: String,
        /// Rendering-only: the formatted cycle chain for humans.  Not a
        /// contract — its shape (separators, ordering) may change at any
        /// time; tests assert on variants and counts, never on the text.
        stack: Vec<String>,
        /// Pinpointing attachment from a tier-2 loop processor's
        /// `CycleDecision::Abort` (or the cross-thread waiter's ferried
        /// prefix rendering).  `None` on every legacy path.
        note: Option<String>,
    },
    DepthLimit {
        query_name: &'static str,
        max: usize,
        /// Total nested `get_shared` frames on the thread across ALL
        /// systems and query types when the limit fired.  This is the
        /// depth the GLOBAL backstop (`MAX_GLOBAL_DEPTH`) actually fires
        /// on — a chain of distinct query types accumulates one frame per
        /// type, so the per-type stack can be nearly empty while the total
        /// is huge.  For the per-type limit it equals (or exceeds) the
        /// per-type depth.  The `Display` reports both so the backstop is
        /// not misreported as "depth 1" with an empty stack.
        /// Rendering-only (same contract as `Cycle::stack`): the chain
        /// text may change at any time; do not parse it.
        total: usize,
        stack: Vec<String>,
    },
    /// A query panicked and its `Started` entry was torn down; later
    /// requests for the same key are vetoed by this sticky marker until
    /// [`QuerySystem::invalidate`] / [`QuerySystem::clear`] /
    /// [`QuerySystem::clear_all`] lifts it.
    ///
    /// Recovery note — what a staging panic actually leaves behind: the
    /// dep node stays registered, green and fingerprint-less, and the
    /// cache entry is unstaged.  So the graph colour never explains a
    /// "green node, empty cache, yet every `get` fails" report — it is
    /// the POISON MARKER that vetoes later gets, and `invalidate` is the
    /// documented way out.
    Poisoned {
        query_name: &'static str,
        key: String,
    },
    TypeMismatch {
        expected: std::any::TypeId,
        got: std::any::TypeId,
    },
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::Cycle {
                query_name,
                key,
                stack,
                note,
            } => {
                write!(
                    f,
                    "query cycle detected: `{}` with key `{}`\n  call stack:\n    {}",
                    query_name,
                    key,
                    stack.join("\n    "),
                )?;
                if let Some(note) = note {
                    write!(f, "\n  note:\n    {}", note)?;
                }
                Ok(())
            }
            QueryError::DepthLimit {
                query_name,
                max,
                total,
                stack,
            } => write!(
                f,
                "query depth limit exceeded: `{}` {} nested frames >= {} \
                 (per-type call-stack depth {})\n  call stack:\n    {}",
                query_name,
                total,
                max,
                stack.len() + 1,
                stack.join("\n    "),
            ),
            QueryError::Poisoned { query_name, key } => {
                write!(f, "query poisoned: `{}` with key `{}`", query_name, key)
            }
            QueryError::TypeMismatch { expected, got } => write!(
                f,
                "type mismatch in transitive verification: expected {:?}, got {:?}",
                expected, got
            ),
        }
    }
}

impl std::error::Error for QueryError {}

/// Lazily-built cache of per-query vtables, indexed by `TypeId`.
///
/// The authoritative query set lives in the `for_each_query!` manifest;
/// this map holds only the queries registered so far.
///
/// Stores a `QueryVTable<'q, Q>` for each query type, bundling metadata,
/// cache, and active-key state together.  Also owns a `DepGraph` for
/// tracking dependencies between query executions, enabling incremental
/// re-evaluation when inputs change.
/// RAII handle for one in-flight query computation.  Holds the system's
/// `quiesce` RwLock for READ from before the thread can become a job
/// owner until its value is in the cache — so an invalidator holding the
/// WRITE lock can never observe a half-published compute.  Released on
/// drop, so the panic unwind path needs no dedicated code.
///
/// Nested gets do NOT take the read lock (parking_lot RwLocks are not
/// reentrant and a nested frame cannot outlive its outer frame's
/// publish); the frame that is outermost FOR A GIVEN SYSTEM holds that
/// system's read guard, tracked via the per-system thread-local depths
/// below.  A frame nested on a DIFFERENT system therefore holds that
/// foreign system's guard too — its depth there is 1 — which is what
/// closes the d10 erase window for provider/system mispairings.
struct InFlightGuard<'q, 'a, 'src, S: Scheduler> {
    _read: Option<crate::sync::RwLockReadGuard<'a, ()>>,
    /// True while the read lock was voluntarily released for a latch
    /// wait (`park_wait`) and must be re-taken by `unpark`.
    parked: bool,
    /// The owning system — needed by `unpark` (re-take the read lock)
    /// and by the outermost drop (drain deferred mutations, d11).
    system: &'a QuerySystem<'q, 'src, S>,
    /// This frame's session total just after its bump — the firing depth
    /// for the validation below, read once here so the compute path pays
    /// no extra session access.
    total: u32,
}

/// One deferred world-mutation (see `QuerySystem::deferred`).  Invoked
/// with exactly the system whose queue it was parked in — guaranteed by
/// the queue living on its target system, never by whichever system owns
/// the frame that drains it.
type DeferredMutation<'q, 'src, S: Scheduler> =
    Box<dyn FnOnce(&QuiescedSystem<'_, 'q, 'src, S>) + Send>;

/// The `&QuerySystem` sub-face handed to deferred closures while the
/// quiesce write lock is held (mechanism A). Deliberately does not
/// implement `Deref<Target = QuerySystem>` and does not expose
/// `get_shared`/`get`/`ensure_ok`/`par_get`: running a query requires an
/// InFlightGuard that re-takes the read lock on the write lock this thread
/// already holds — a read→write self-deadlock (d11). d11's "deferred
/// closures must not run queries" thereby goes from a debug_assert to a
/// type absence; aliases of `&QuerySystem` captured by the closure from
/// outside (the d22 TARGET static is this shape) bypass the type absence
/// and are caught by the two UNCONDITIONAL (release builds included)
/// tripwires instead: `InFlightGuard::acquire`'s `IN_DRAIN` probe for the
/// query shape, `mutate_world`'s for the foreign-mutation shape.  Both
/// drain scopes register the marker — `run_deferred_batch` and
/// `mutate_world`'s inline path — so a captured alias has no unmonitored
/// window.
pub struct QuiescedSystem<'a, 'q, 'src, S: Scheduler> {
    system: &'a QuerySystem<'q, 'src, S>,
    _proof: crate::hir::query::dep_graph::Quiesced,
}

impl<'a, 'src, 'q, S: Scheduler> QuiescedSystem<'a, 'q, 'src, S>
{
    #[inline]
    pub(crate) fn proof(&self) -> &crate::hir::query::dep_graph::Quiesced {
        &self._proof
    }

    // Read-only introspection:
    pub fn find_vtable<Q: Query<'q, 'src>>(&self) -> Option<Arc<QueryVTable<'q, 'src, Q>>> {
        self.system.find_vtable::<Q>()
    }

    pub fn list_queries(&self) -> Vec<&'static str> {
        self.system.list_queries()
    }

    // Red-green world mutations (same semantics as the QuerySystem methods
    // of the same name, but only reachable under quiescence):
    pub fn invalidate<Q: Query<'q, 'src>>(&self, key: &Q::Key) {
        self.system.invalidate_quiesced::<Q>(key, self.proof());
    }

    pub fn invalidate_node(&self, node: dep_graph::DepNodeIndex) {
        self.system.dep_graph.invalidate(node, self.proof());
    }

    pub fn clear<Q: Query<'q, 'src>>(&self) {
        if let Some(vt) = self.find_vtable::<Q>() {
            // Mirror `QuerySystem::clear`: untracked memos live in
            // `untracked`, not the cache.
            if vt.descriptor.untracked {
                vt.untracked.write().clear();
            } else {
                vt.cache
                    .as_ref()
                    .expect("tracked query has a cache")
                    .clear();
            }
        }
        // And mirror the poison lift — a marker left behind would veto
        // every later `get` of `Q` even though the cache has just been
        // emptied (see `QuerySystem::clear`'s doc and `lift_poison_for`).
        self.system.lift_poison_for::<Q>();
    }

    pub fn clear_all(&self) {
        self.system.clear_all_impl(self.proof());
    }

    pub fn note_input_digest<Q: Query<'q, 'src>>(&self, key: &Q::Key, d: u64) {
        self.system
            .note_input_digest_impl::<Q>(key, d, self.proof());
    }
}

type DeferredQueue<'q, 'src, S: Scheduler> = Vec<DeferredMutation<'q, 'src, S>>;

/// RAII for one system's `IN_DRAIN` flag: registers the system id on
/// construction, removes it on drop — including through a panic — so a
/// panicking deferred closure can no longer wedge the flag `true` for
/// the thread's remaining lifetime (the pre-fix leak: every later
/// `mutate_world` on this thread then ran inline, silently bypassing
/// the d10 erase-window protection and the d11 deferral).
struct DrainGuard(usize);

impl DrainGuard {
    fn new(id: usize) -> Self {
        IN_DRAIN.with(|d| {
            d.borrow_mut().insert(id);
        });
        DrainGuard(id)
    }
}

impl Drop for DrainGuard {
    fn drop(&mut self) {
        IN_DRAIN.with(|d| {
            d.borrow_mut().remove(&self.0);
        });
    }
}

/// Best-effort rendering of a panic payload for the deferred-mutation
/// log line.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// This thread's cross-`Q` query session: the merged home of the two
/// counters `IN_FLIGHT_TOTAL` / `IN_FLIGHT_BY_SYSTEM` used to keep in
/// separate TLS slots.  One `RefCell` borrow serves the whole frame
/// boundary (total + per-system depth together), where the old layout
/// paid two TLS accesses plus a hash probe per site.
///
/// `total` preserves the exact legacy semantics: every nested
/// `get_shared` frame on this thread across ALL systems (incremented in
/// `InFlightGuard::acquire`, which runs BEFORE validation, so the
/// outermost frame sees 1).  Two consumers: the native-stack backstop
/// (`validate_enter`'s `MAX_GLOBAL_DEPTH` check — a chain of distinct
/// query types must be bounded regardless of which system each frame
/// ran on) and `mutate_world`'s "inside a running query → defer" test,
/// which must defer for a frame on ANY system.
///
/// `systems` preserves the per-system depths: only the frame that brings
/// THIS system's depth 0 → 1 holds that system's quiesce read guard.
/// Pre-merge a single thread-global counter decided guard ownership, so
/// a thread whose outermost frame was on system A took NO read guard
/// on B when its compute nested into B (provider/system mispairing —
/// the f1/d20 hazard): a concurrent `B.invalidate` could then land
/// between the nested compute's input reads and its publish — the
/// d10 erase window, uncovered on foreign systems.  With per-system
/// depths, the nested frame takes B's read guard and the erase window
/// closes there too.
///
/// 4 × 16 B = 64 B inline, exactly one cache line.  (Lazy init:
/// see the `SESSION` declaration below.)
struct ThreadSession {
    total: u32,
    /// (system_id, depth) active pairs.  Normally 0–2 entries: a linear
    /// scan beats a hash map.  Insert/remove only at the outermost
    /// boundaries 0→1 / 1→0; removal uses `swap_remove`
    /// (order-independent).
    systems: SmallVec<[(usize, u32); 4]>,
}

thread_local! {
    static SESSION: std::cell::RefCell<ThreadSession> = std::cell::RefCell::new(ThreadSession {
        total: 0,
        // NOT const-initialized: `SmallVec::new` is not `const` in the
        // pinned 1.13, so the slot pays std's once-per-thread lazy init.
        // Steady-state access is unchanged (one TLS read); only first
        // touch per thread branches.  Revisit if the pin ever lifts.
        systems: SmallVec::new(),
    });
    /// System ids for which THIS thread has at least one world-mutation
    /// parked in that system's own queue (`QuerySystem::deferred`).
    ///
    /// A PRESENCE MARKER ONLY — the closures themselves live on the
    /// target system, partitioned by this thread's id, because only the
    /// system can outlive its parked work.  (Pre-fix the closures sat
    /// here, keyed by target-system id: they died with this thread-local
    /// whenever the thread exited or the target was dropped first — a
    /// silently lost invalidation.  Before that, the queue was
    /// thread-global with no key at all: with two QuerySystems nested on
    /// one thread, a mutation deferred against system B executed with
    /// system A's `&QuerySystem` at A's outermost drop — mutating the
    /// wrong session.  Both defects are structural, not timing: the queue
    /// now belongs to its target, and its target is what invokes it.)
    ///
    /// The marker exists purely for the fast path.  `drain_deferred`
    /// runs at EVERY outermost guard drop, i.e. after every top-level
    /// compute — the common case by orders of magnitude — so it must
    /// answer "anything parked for me?" without touching the system's
    /// queue lock, which under a multi-threaded compute storm would turn
    /// into one contended word shared by every thread completing work on
    /// that system.  One thread-local hash probe instead; the lock is
    /// taken only when something really is parked.
    ///
    /// A marker may outlive the entries it announces: `Drop for
    /// QuerySystem` sweeps the system's queue without this thread's
    /// participation.  The stale marker then costs one lock acquisition
    /// that finds nothing and clears itself.  Never a missed or a
    /// duplicated execution — membership in the QUEUE, not in this set,
    /// is what makes a closure runnable, and only one drain removes it.
    /// Per (system, thread) the residue is one `usize`, which also means
    /// a dropped system's id can be handed out again while this marker
    /// still names it: the later flush probes the new system's empty
    /// queue, finds nothing, and clears itself, so reuse is harmless.
    static DEFERRED_PENDING: std::cell::RefCell<FxHashSet<usize>> =
        std::cell::RefCell::new(FxHashSet::default());
    /// System ids whose deferred queue is being drained on this thread,
    /// inside that drain's quiesce write lock.  Keyed by system id like
    /// the presence marker: a reentrant `mutate_world` on the DRAINING
    /// system runs inline (the write lock is already held), while a
    /// DIFFERENT system's `mutate_world` still takes its own quiesce
    /// lock normally — the pre-fix global flag let a foreign mutation
    /// skip the other system's quiesce gate entirely.
    static IN_DRAIN: std::cell::RefCell<FxHashSet<usize>> =
        std::cell::RefCell::new(FxHashSet::default());
}

impl<'q, 'src: 'q, 'a, S: Scheduler> InFlightGuard<'q, 'a, 'src, S> {
    fn acquire(system: &'a QuerySystem<'q, 'src, S>) -> Self {
        // Quiescent-entry flush: a mutation parked by an earlier nested
        // frame of this system (its drop skipped the drain — a foreign
        // system's frame was outer; see `Drop`) still sits in this
        // thread's partition.  TOTAL == 0 means no frame of any system is
        // live, so no read guard is held and the write lock cannot
        // self-deadlock; flushing before the compute below keeps the
        // parked invalidation from being served stale.
        //
        // Pre-check and bump are FUSED into a single `borrow_mut` (the
        // outer path pays one access, not two): on the cold TOTAL == 0
        // path the borrow is dropped for the drain — `drain_deferred`'s
        // debug asserts re-read the session, and a `RefCell` re-borrow
        // would panic — then re-taken.  No borrow is outstanding at frame
        // top, so reentrancy is unchanged.
        //
        // One borrow serves the whole boundary: total and per-system
        // depth together.  Nested frames on the SAME system skip the read
        // lock (only the frame that brings 0 → 1 pays one shared read
        // acquisition).
        //
        // The earlier text said here that "a nested frame
        // on a FOREIGN system holds that system's guard instead (closing
        // the d10 erase window there)".  That is dead text and actively
        // harmful: the cross-system mispair tripwire BELOW now panics
        // unconditionally on the first mismatched sid, so no nested foreign
        // frame ever reaches the guard-taking code at all.  A maintainer
        // reading the old sentence could conclude the tripwire is redundant
        // defence and delete it — the exact opposite of the truth.  There
        // is ONE survivor: `DepGraph::read`'s own cross-system check (the
        // second line of defence), which catches a frame recorded with no
        // matching `SESSION` entry at all.
        let (prev, total) = SESSION.with(|s| {
            let mut b = s.borrow_mut();
            // Cross-system mispair tripwire.  Nesting a query on system B
            // from inside a compute already guarded on system A is NOT a
            // supported shape — it is the contract violation `DepGraph::read`
            // guards against (panic in debug, stale edge in release).  Left
            // unguarded here it also DEADLOCKS, in release too: the nested
            // frame is outermost-for-B, so it takes B's read guard; when it
            // then parks on a key owned by another thread, `park_wait`
            // releases only B's guard (`is_outer()` is per-system), leaving
            // A's held.  The owner computes, mispairs into A, and blocks on
            // A's read lock — not behind a lock THIS thread holds (read vs
            // read does not block), but behind any invalidator already
            // queued for A's write, which cannot proceed while this thread
            // holds A.read.  Neither thread moves on.  `register_wait`'s
            // cycle walk cannot see it: it follows latch-wait edges only,
            // and a thread blocked on a lock has no edge.
            //
            // So the FIRST mispair trips here instead, deterministically,
            // in every build — the same "loud panic beats hang" trade the
            // `IN_DRAIN` probe below makes.  The probe stays second: a
            // drain re-entrancy is diagnosed with its own message, and both
            // are bugs, so only the ordering of the two reports is at stake.
            if b.systems.iter().any(|(sid, _)| *sid != system.id) {
                let live: SmallVec<[(usize, u32); 4]> = b.systems.clone();
                drop(b);
                panic!(
                    "query on system {} started while this thread already holds an \
                     InFlightGuard on a different system (live frames: {:?}).  Provider \
                     and executing system must stay paired: `DepGraph::read`'s guard \
                     rejects this shape, and leaving it through here deadlocks — the \
                     nested frame parks holding the outer system's read guard, and the \
                     owner blocks on that system's read lock behind a queued \
                     invalidator.  Route the nested query through the paired provider.",
                    system.id,
                    live,
                );
            }
            if b.total == 0 {
                drop(b);
                // Release-mode re-entrancy guard.  This thread is at
                // quiescent entry, so the flush below would take the
                // quiesce READ lock on the system — and if a
                // `mutate_world` is already draining on THIS thread, its
                // WRITE lock is owned here and the read would self-deadlock
                // with no panic path (parking_lot read-under-write).  The
                // only route in is a captured `&QuerySystem` alias reached
                // from inside a mutate/deferred closure (the d22 shape):
                // `QuiescedSystem` deliberately hides `get*`, so a
                // legitimate closure cannot do this.  Both drain shapes
                // register the marker — `run_deferred_batch` and
                // `mutate_world`'s inline path — so ONE TLS probe here
                // turns the deadlock into a loud panic in release too.
                // (Cost: one TLS set probe per OUTERMOST query.)
                if !IN_DRAIN.with(|d| d.borrow().is_empty()) {
                    panic!(
                        "query started on this thread while a deferred \
                         world-mutation batch is draining (the quiesce write \
                         lock is held here): the `InFlightGuard` read lock \
                         would be taken under it and self-deadlock. The only \
                         route in is a captured &QuerySystem alias reached \
                         from inside a mutate or deferred closure — \
                         `mutate_world`'s inline path registers this very \
                         marker too, so both drain shapes are covered. \
                         Perform the work after the mutation returns, or \
                         route it through the closure's &QuiescedSystem."
                    );
                }
                system.drain_deferred();
                b = s.borrow_mut();
            }
            b.total += 1;
            let total = b.total;
            let prev = match b.systems.iter().position(|(sid, _)| *sid == system.id) {
                Some(i) => {
                    let p = b.systems[i].1;
                    b.systems[i].1 += 1;
                    p
                }
                None => {
                    b.systems.push((system.id, 1));
                    0
                }
            };
            (prev, total)
        });
        // Counted for NESTED frames too: `Drop for QuerySystem` must see
        // every live guard, wherever in the stack it sits.  AcqRel: the
        // release half of the RMW pairs with the `Acquire` load in
        // `Drop for QuerySystem`, so its assert (use-after-free tripwire)
        // can never read a stale non-zero count and fire on a legal
        // teardown.
        system.frames.fetch_add(1, Ordering::AcqRel);
        if prev > 0 {
            return InFlightGuard {
                _read: None,
                parked: false,
                system,
                total,
            };
        }
        // Deadlock-freedom of this acquisition: a thread blocked HERE
        // (invalidator active, we are outermost) holds no active key —
        // the lock sits before `acquire_key_slot` — and every job latch
        // we could later wait on is owned by a thread holding its own
        // read guard, so the wait-for chain terminates at computing
        // threads that publish and release.  parking_lot's
        // write-preferring policy bounds invalidation latency under a
        // compute stream; strict fairness (and thus formal
        // starvation-freedom for readers) is not guaranteed.
        InFlightGuard {
            _read: Some(system.quiesce.read()),
            parked: false,
            system,
            total,
        }
    }

    /// Session total just after this frame's bump — stable for the whole
    /// frame (only this thread's own bump changes its TLS slot, and no
    /// nested frame exists before compute).
    fn total(&self) -> u32 {
        self.total
    }

    /// Temporarily release the read lock.  Used on the WAIT path: a
    /// thread about to block on another thread's job latch must not hold
    /// the read lock — otherwise the invalidator's write lock queues
    /// behind waiters that are blocked on jobs owned by threads queuing
    /// behind the invalidator, closing a three-way cycle
    /// (waiter-read → owner-queued → invalidator-write → waiter-read).
    /// A waiter cannot publish, so releasing its slot is sound; the
    /// guard is re-taken when the wait ends (the thread may then become
    /// a compute owner and publish).
    fn park_wait(&mut self) {
        // Only the outermost frame holds the read lock; nested frames
        // (depth > 1) hold nothing and must NOT touch the lock — their
        // outer frame's guard already covers the wait, and re-taking in
        // `unpark` would acquire the same non-reentrant RwLock twice on
        // one thread (parking_lot read locks are not reentrant).
        //
        // Invariant (why the outer guard is safe to release/re-take, and
        // why the three-way cycle with the invalidator's write lock cannot
        // close): a thread that OWNS a job holds this system's read guard
        // whenever it is parked — owning means being inside this system's
        // compute at depth ≥ 1, so any latch wait it performs is a nested
        // frame (depth ≥ 2) whose `park_wait` is a no-op and whose `unpark`
        // is a no-op; it never queues for the read lock at wake time.
        // Conversely, a thread whose parked frame DID release the guard is
        // a top-level waiter on this system and owns no job here, so no
        // one can wait on its (nonexistent) latch — its wake may queue
        // behind the pending writer, but the writer's blocker set consists
        // only of threads that either publish or are job owners whose
        // latches get published, so the wait-for chain terminates.
        // (A frame that is outermost on system A while owning a job on
        // system B WOULD close a cycle via B — that shape is now
        // structurally impossible: `InFlightGuard::acquire`'s cross-system
        // mispair tripwire panics on the FIRST mismatched sid, before any
        // lock is taken and in BOTH builds.  The earlier text claimed
        // `register_wait`'s cycle walk cuts it; that is wrong — a thread
        // blocked on a quiesce lock has no latch-wait edge, so the walk
        // can never see the cycle.)
        if !self.is_outer() {
            return;
        }
        self._read = None;
        self.parked = true;
    }

    /// Re-take the read lock after a wait (see `park_wait`).
    fn unpark(&mut self) {
        if self.parked {
            self.parked = false;
            self._read = Some(self.system.quiesce.read());
        }
    }

    fn is_outer(&self) -> bool {
        // Outermost FOR THIS SYSTEM: this frame holds this system's
        // quiesce read guard.  (The session `total` is irrelevant here — a
        // frame that is outermost on system B while nested inside system
        // A still holds B's guard, and it is B's guard `park_wait` must
        // release across a latch wait.)
        SESSION.with(|s| {
            s.borrow()
                .systems
                .iter()
                .find(|(sid, _)| *sid == self.system.id)
                .map(|(_, depth)| *depth)
                .unwrap_or(0)
        }) == 1
    }
}

impl<'q, 'src: 'q, 'a, S: Scheduler> Drop for InFlightGuard<'q, 'a, '_, S> {
    fn drop(&mut self) {
        // Decrement THIS system's depth FIRST: the drain below (and its
        // debug assert) must observe the post-drop state — this frame is
        // gone by then.  One borrow yields both values, so the old
        // cross-variable timing argument (decrement per-system first,
        // then read the total across two slots) disappears entirely.
        //
        // The borrow ends before the drain: the drain's asserts re-read
        // the session.  `debug_assert` (not silent saturation) on
        // underflow — a counter running backwards is a logic bug and
        // must surface in tests; release keeps the legacy saturating
        // behavior.
        let (prev, total) = SESSION.with(|s| {
            let mut s = s.borrow_mut();
            let prev = match s.systems.iter().position(|(sid, _)| *sid == self.system.id) {
                Some(i) => {
                    let p = s.systems[i].1;
                    debug_assert!(p > 0, "InFlightGuard dropped below depth 0");
                    s.systems[i].1 = p.saturating_sub(1);
                    if s.systems[i].1 == 0 {
                        s.systems.swap_remove(i);
                    }
                    p
                }
                None => {
                    debug_assert!(false, "InFlightGuard dropped without acquire");
                    0
                }
            };
            debug_assert!(s.total > 0, "session total dropped below 0");
            s.total = s.total.saturating_sub(1);
            (prev, s.total)
        });
        self._read = None; // release the read lock (if outermost for this system)
        self.system.frames.fetch_sub(1, Ordering::AcqRel);
        // Drain ONLY at full thread quiescence: `prev == 1` is outermost
        // for THIS system, but a foreign system's frame may still be live
        // above us (cross-system nesting) holding ITS read guard —
        // draining here takes this system's quiesce.write under that
        // foreign read, the ABBA half of a two-thread deadlock.  A skipped
        // drain stays parked for the next quiescent entry (the `acquire`
        // flush, `mutate_world`'s inline drain-before-write) or teardown
        // `Drop`.  The read lock is GONE by now and this frame's publish
        // is done, so — at TOTAL == 0 — taking the write lock cannot
        // self-deadlock, and the deferred invalidations cannot be erased
        // by this thread's publishes.
        // The drain is panic-safe (see `drain_deferred`).
        if prev == 1 && total == 0 {
            self.system.drain_deferred();
        }
    }
}

/// Object-safe clone for the `node_to_key` map's boxed keys: lets the
/// transitive-verification path copy a key out of the map (under its
/// read lock) before re-requesting the dependency (which needs the write
/// lock) — see the read→write upgrade hazard on `node_to_key`.
trait CloneAny: std::any::Any {
    fn clone_box(&self) -> Box<dyn CloneAny + Send + Sync>;
}

/// Sharded reverse mapping `DepNodeIndex → (TypeId, key)`.
///
/// The map is write-on-fresh-node and read on the dirty-re-request path;
/// both were serialized on ONE global RwLock.  Sharding by the node
/// index's low bits (indices are allocated once and never move — no
/// re-balancing) divides the write contention by `SHARDS`.  All sites
/// keep their `sync_order::Fam::NodeToKey` tripwire acquisition, which
/// guards the operation, not a particular shard.
/// Closed key union for the `node_to_key` reverse map (rustc-style
/// `TaggedQueryKey`): one variant per manifest query plus an `Opaque`
/// fallback for function-local test queries.  Typed variants carry the
/// key by value, so recovery is a match with zero `Any` downcast; only
/// `Opaque` keeps the erased box.
pub enum TaggedQueryKey {
    TypeDefShape(crate::hir::types::DefId),
    CheckFnSignature(queries::FnId),
    CheckFnBody(queries::FnId),
    DocTokens(queries::FnId),
    Opaque(TypeId, Box<dyn CloneAny + Send + Sync>),
}

impl TaggedQueryKey {
    /// The owning query's `TypeId` — the vtable registry key.
    pub fn query_type_id(&self) -> TypeId {
        match self {
            TaggedQueryKey::TypeDefShape(_) => TypeId::of::<queries::TypeDefShape>(),
            TaggedQueryKey::CheckFnSignature(_) => TypeId::of::<queries::CheckFnSignature>(),
            TaggedQueryKey::CheckFnBody(_) => TypeId::of::<queries::CheckFnBody>(),
            TaggedQueryKey::DocTokens(_) => TypeId::of::<queries::DocTokens>(),
            TaggedQueryKey::Opaque(ty, _) => *ty,
        }
    }

    /// Type-erased view for the `Opaque`-only `get_shared_erased` path.
    /// Typed variants have no `Any` identity — returns `None` for them.
    pub fn as_any(&self) -> Option<&dyn std::any::Any> {
        match self {
            TaggedQueryKey::Opaque(_, boxed) => {
                let key_dyn: &(dyn CloneAny + Send + Sync) = &**boxed;
                Some(key_dyn as &dyn std::any::Any)
            }
            _ => None,
        }
    }
}

impl Clone for TaggedQueryKey {
    fn clone(&self) -> Self {
        match self {
            TaggedQueryKey::TypeDefShape(k) => TaggedQueryKey::TypeDefShape(*k),
            TaggedQueryKey::CheckFnSignature(k) => TaggedQueryKey::CheckFnSignature(k.clone()),
            TaggedQueryKey::CheckFnBody(k) => TaggedQueryKey::CheckFnBody(k.clone()),
            TaggedQueryKey::DocTokens(k) => TaggedQueryKey::DocTokens(k.clone()),
            TaggedQueryKey::Opaque(ty, boxed) => {
                let key_dyn: &(dyn CloneAny + Send + Sync) = &**boxed;
                TaggedQueryKey::Opaque(*ty, key_dyn.clone_box())
            }
        }
    }
}

impl std::fmt::Debug for TaggedQueryKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaggedQueryKey::TypeDefShape(k) => f.debug_tuple("TypeDefShape").field(k).finish(),
            TaggedQueryKey::CheckFnSignature(k) => {
                f.debug_tuple("CheckFnSignature").field(k).finish()
            }
            TaggedQueryKey::CheckFnBody(k) => f.debug_tuple("CheckFnBody").field(k).finish(),
            TaggedQueryKey::DocTokens(k) => f.debug_tuple("DocTokens").field(k).finish(),
            TaggedQueryKey::Opaque(ty, _) => {
                f.debug_tuple("Opaque").field(ty).field(&"..").finish()
            }
        }
    }
}

mod node_to_key {
    use super::{TaggedQueryKey, dep_graph};
    use parking_lot::RwLock;
    use rustc_hash::FxHashMap;
    use std::any::TypeId;

    /// Shard count: power of two, small enough that `clear_all`'s sweep
    /// stays a few uncontended locks, large enough to divide the
    /// fresh-node write traffic.
    const SHARDS: usize = 16;
    const SHARD_MASK: u32 = (SHARDS - 1) as u32;

    type Entry = TaggedQueryKey;
    type Map = FxHashMap<dep_graph::DepNodeIndex, Entry>;

    #[inline]
    fn shard_of(node: dep_graph::DepNodeIndex) -> usize {
        (node.0 & SHARD_MASK) as usize
    }

    pub(super) struct ShardedMap([RwLock<Map>; SHARDS]);

    impl Default for ShardedMap {
        fn default() -> Self {
            Self(std::array::from_fn(|_| RwLock::new(FxHashMap::default())))
        }
    }

    impl ShardedMap {
        /// Registration-once insert (fresh node path).
        pub(super) fn insert(&self, node: dep_graph::DepNodeIndex, entry: Entry) {
            self.0[shard_of(node)].write().insert(node, entry);
        }

        /// Clone the mapping out so the caller can drop the lock before
        /// re-requesting (the read→write upgrade hazard — this replaces
        /// the whole-map read guard).
        pub(super) fn clone_out(&self, node: dep_graph::DepNodeIndex) -> Option<TaggedQueryKey> {
            let guard = self.0[shard_of(node)].read();
            guard.get(&node).cloned()
        }

        /// `clear_all` sweep: every shard, one at a time (never two shard
        /// locks coexisting — the §0 multi-shard discipline).
        pub(super) fn clear(&self) {
            for shard in &self.0 {
                shard.write().clear();
            }
        }

        /// Test-only probe: is every shard empty?
        #[cfg(test)]
        pub(super) fn is_empty(&self) -> bool {
            self.0.iter().all(|shard| shard.read().is_empty())
        }
    }
}

impl<T> CloneAny for T
where
    T: std::any::Any + Clone + Send + Sync,
{
    fn clone_box(&self) -> Box<dyn CloneAny + Send + Sync> {
        Box::new(self.clone())
    }
}

pub struct QuerySystem<'q, 'src: 'q, S: Scheduler> {
    /// Stable instance id (monotonic).  Keys this thread's deferred-
    /// mutation presence marker (`DEFERRED_PENDING`) and the `IN_DRAIN`
    /// set — NOT the address, which can be reused after drop (see
    /// `NEXT_SYSTEM_ID`).
    id: usize,
    /// World generation for the per-thread cycle partitions (see
    /// `QuerySystem::epoch`).  Bumped by `clear_all_impl`.
    epoch: AtomicU64,
    /// Queue of world-mutations deferred by computes that ran INSIDE a
    /// query frame (see `mutate_world`), partitioned by the id of the
    /// THREAD that parked them.
    ///
    /// Owned by the SYSTEM, not by a thread-local.  A thread-local queue
    /// is reachable only from its own thread, so when the deferring
    /// thread exited — or when the system got no further outermost frame
    /// before being dropped — the parked closures were destroyed with the
    /// thread-local and never executed: a silently lost invalidation
    /// (the original bug where deferrals parked in per-thread queues
    /// were dropped when the system outlived its threads).  Here the queue lives and
    /// dies with the system, and `Drop for QuerySystem` sweeps every
    /// thread's partition, so each closure still runs — exactly once,
    /// with the `&QuerySystem` it was parked against.
    ///
    /// The per-thread partition is load-bearing, not decoration.  Only
    /// the owning thread may drain its own partition, and only at full
    /// thread quiescence (`IN_FLIGHT_TOTAL == 0` — no frame of any system
    /// live): per-system outermost is NOT that moment under cross-system
    /// nesting, so the guard `Drop` gates on both, and quiescent entries
    /// (the `acquire` flush, `mutate_world`'s inline drain-before-write)
    /// sweep what nested frames left parked.  Draining under a live frame
    /// would either self-deadlock (d11) or take the write lock under a
    /// foreign read guard (cross-system ABBA).  Draining another thread's
    /// partition would violate both, and could publish under a compute
    /// that later overwrites it (d10's erase window).  `Drop` is the
    /// single exception: it runs once the system is unreachable, so no
    /// frame of it can be live anywhere.
    deferred: Mutex<FxHashMap<std::thread::ThreadId, DeferredQueue<'q, 'src, S>>>,
    /// Count of live `InFlightGuard`s for THIS system across all threads.
    ///
    /// Precondition, checked UNCONDITIONALLY rather than under
    /// `debug_assertions`: `Drop` is cold, so an atomic load here costs
    /// nothing, and a precondition whose violation is a use-after-free
    /// should not quietly evaporate in release builds.
    frames: AtomicUsize,
    /// Lock-free registry of per-query vtables.  Readers (i.e. every
    /// query, hit or miss) do one atomic load; writers — the first-ever
    /// registration of a query type — swap in a cloned map under
    /// `init_lock`.
    vtables: ArcSwap<FxHashMap<TypeId, Arc<dyn QueryVTableDyn<'q, 'src, S> + 'q>>>,
    /// Serialises vtable registration (double-checked init).  The hot
    /// path never touches it.
    init_lock: Mutex<()>,
    /// Per-thread vtable resolution cache: one entry per query type this
    /// thread has touched on THIS system (the field lives on the system,
    /// so no system id is needed).  The linear scan IS the registry
    /// lookup — the manifest holds a handful of queries — and the hit
    /// path is one `ThreadLocal` slot fetch + one epoch compare: no
    /// atomic load, no hash probe, no `Arc` refcount bump.  Entries are
    /// validated against the generation stamp (`epoch`, bumped by
    /// `clear_all_impl` before the registry is emptied); a stale entry is
    /// replaced, never served, so a waiter re-resolving after a
    /// `clear_all` (d18) cannot observe an orphaned vtable through here.
    resolve_cache: thread_local::ThreadLocal<std::cell::RefCell<Vec<ResolveEntry<'src, 'q, S>>>>,
    /// Schedule-point sink shared with the dep graph (both hold an `Arc`
    /// pointing at one instance).  For `NoOpScheduler` every `suspend`
    /// call monomorphizes to an empty inlined body — the field costs one
    /// allocation at construction and nothing on the hot path.
    sched: Arc<S>,
    /// Dependency graph for incremental re-evaluation.
    /// The dependency graph.  `pub(crate)`: its red-bit mutators
    /// (`invalidate`, `dirty_readers*`, `mark_green`, `reset`, bare
    /// `publish`) change the world WITHOUT quiescence — user code must
    /// go through the quiesced/deferring gates
    /// (`QuerySystem::invalidate*`, `clear_all`, `mutate_world`),
    /// otherwise a red bit can be erased by an in-flight publish
    /// (d10) or the world can change under a running compute.
    pub(crate) dep_graph: dep_graph::DepGraph<S>,
    /// Active keys: sharded to reduce lock contention under par_get,
    /// each shard with its own RwLock.
    /// Key is [`ActiveKey`] — the query `TypeId` plus the REAL key
    /// (type-erased, Eq-checked).  The 64-bit hash selects the shard
    /// only: two distinct keys of one query type can never alias an
    /// entry (the pre-fix `(TypeId, hash)` keys made a hash collision
    /// report a false same-thread cycle, wait+recompute across threads,
    /// or — worst — inherit a permanent `Poisoned` from an unrelated
    /// key's panic).
    ///
    /// Each shard is `CachePadded`: an unpadded `Vec<RwLock<..>>` puts
    /// neighbouring lock words on the same cache line, so two threads
    /// locking *different* shards still invalidate each other — which
    /// would silently undo the striping (Perf Book §13.7.3).
    active_keys: Vec<CachePadded<RwLock<FxHashMap<ActiveKey, ActiveKeyStatus>>>>,
    /// Cross-thread wait graph: `waiter thread → WaitEdge` for every
    /// thread currently blocked on another thread's job latch.  Each
    /// edge names the owner thread AND the specific job waited on, so
    /// the owner can clear exactly the edges belonging to the job it
    /// just completed (`clear_waiters_of_me`) instead of every edge
    /// pointing at itself.  Registered just before `wait_on` and removed
    /// right after it returns (both Ok and Err), so it always reflects
    /// in-flight waits.
    wait_graph: RwLock<FxHashMap<std::thread::ThreadId, WaitEdge>>,
    /// Quiescence gate for world mutation: `RwLock<()>` — every
    /// outermost query compute holds it for READ from before it can
    /// become a job owner until its value is in the cache (panic
    /// included, via `InFlightGuard`); every mutation of the red-green
    /// world takes it for WRITE (all of them flow through
    /// `mutate_world`: `invalidate*`, `clear`, `clear_all`).
    ///
    /// Why: an invalidation landing between a compute's input reads and
    /// its publish is erased — the publisher's `clear_dirty` wipes the
    /// just-set dirty bit and `cache.insert` resurrects the
    /// pre-invalidation value, leaving the node permanently green with
    /// stale data.  (The same root cause hides a second window: a change
    /// published while a reader is in flight can never mark it — the
    /// reader's reverse edge is not installed until its own publish.)
    /// Making mutation quiescent closes both structurally: no compute is
    /// in flight while the world changes.  Mirrors rustc, where
    /// invalidation only happens between compiler passes.
    ///
    /// parking_lot's write-preferring policy bounds invalidation latency:
    /// an invalidator is served after the current readers drain even under
    /// a continuous compute stream.  Strict fairness (and thus formal
    /// starvation-freedom for readers) is not guaranteed.  Nested gets
    /// skip the read lock
    /// (parking_lot RwLocks are not reentrant; see `InFlightGuard`),
    /// which is sound because a nested frame cannot outlive its outer
    /// frame's publish.  The Wait branch of `get_shared` voluntarily
    /// RELEASES the guard across a latch wait (`park_wait`/`unpark`) —
    /// holding it there would close a three-way cycle with the
    /// invalidator's write lock.
    ///
    /// Deadlock-freedom: a thread blocked on the read lock (invalidator
    /// active) holds no active key — the guard sits before
    /// `acquire_key_slot` — and a thread blocked on a job latch is
    /// covered by the job owner's own read lock, whose chain terminates
    /// at a compute that publishes and releases.  No cycle.
    quiesce: RwLock<()>,
    /// Reverse mapping from `DepNodeIndex` to `(TypeId, key)` for
    /// transitive green verification.  Allows re-requesting dirty
    /// dependencies of any query type when a cached value's dependencies
    /// are found to be stale.  Keys are stored as `CloneAny` so the
    /// re-request path can clone the key out while the read guard is
    /// still held, then release it before the nested `get_shared` (which
    /// takes this map's write lock — same-thread read→write upgrade is a
    /// parking_lot self-deadlock).
    ///
    /// Sharded (P2): the map is written on every FRESH query node
    /// (registration-once) and was a single global write lock on that
    /// path.  Nodes are keyed by their `DepNodeIndex` and never move, so
    /// a fixed stride by the low bits needs no re-balancing.  The
    /// `sync_order::Fam::NodeToKey` tripwire family still covers every
    /// shard: the per-site `Acquired::acquire` guards the *operation*,
    /// and the same-family re-held assert stays correct because no
    /// correct path ever holds two shards at once (the read-out-then-
    /// drop discipline in `try_serve_verified` predates the split and
    /// keeps holding).
    node_to_key: node_to_key::ShardedMap,
}

/// Static manifest of every production query type (Tier1: enumeration
/// only — registration stays lazy under `init_lock`).
///
/// Rust has no reflection, so the authoritative query set lives here, in
/// one macro, instead of being rebuilt at runtime from whichever queries
/// happened to register.  Adding a query = adding one line below; the
/// compiler then forces every `for_each_query!` consumer (today: only
/// `list_queries`) to cover it.
///
/// Deliberately NOT used for registration: test-only queries (the
/// `tests` module's `TestQuery`, `storm_query`, …) are outside the
/// manifest and keep registering lazily, so unit tests need no manifest
/// edits.
macro_rules! for_each_query {
    ($mac:ident) => {
        $mac!(queries::TypeDefShape);
        $mac!(queries::CheckFnSignature);
        $mac!(queries::CheckFnBody);
        $mac!(queries::DocTokens);
    };
}

/// One cached vtable resolution: the generation stamp it was resolved
/// under plus the type-erased `Arc`.  An entry whose stamp predates the
/// system's `epoch` is a `clear_all` orphan — replaced, never served.
struct ResolveEntry<'src, 'q, S: Scheduler> {
    ty: TypeId,
    epoch: u64,
    vtable: Arc<dyn QueryVTableDyn<'q, 'src, S> + 'q>,
}

/// A resolved vtable whose lifetime is fenced by a borrow, not a refcount.
///
/// - `Cached` keeps the `RefCell` shared borrow open for the guard's
///   lifetime (`Ref::filter_map`): the borrow tracker — not convention —
///   forbids a nested refill from reallocating the slot vector under us.
/// - `Owned` is the slow path (miss, stale epoch, contended refill): a
///   plain `Arc` holding no borrow, so nesting freely refills.
///
/// Reassignment drops the old guard after the new one resolves; the
/// re-resolve fast path takes a second *shared* borrow, which legally
/// overlaps, so the wait-branch refresh never trips the owned fallback
/// on its own account.
enum VtableResolve<'q, 'a, 'src, Q: Query<'q, 'src>> {
    Cached(std::cell::Ref<'a, QueryVTable<'q, 'src, Q>>),
    Owned(Arc<QueryVTable<'q, 'src, Q>>),
}

impl<'src, 'a, 'q, Q: Query<'q, 'src>> std::ops::Deref for VtableResolve<'q, 'a, 'src, Q> {
    type Target = QueryVTable<'q, 'src, Q>;
    fn deref(&self) -> &QueryVTable<'q, 'src, Q> {
        match self {
            VtableResolve::Cached(r) => r,
            VtableResolve::Owned(a) => a,
        }
    }
}

/// The result of [`QuerySystem::wait_for_owner`] — the Wait branch
/// extracted from `get_shared_inner`'s attempt loop.
enum WaitOutcome<'q, 'src, Q: Query<'q, 'src>> {
    /// The request has been resolved: return this result as-is from
    /// `get_shared_inner` — a served value, a real cross-thread `Cycle`
    /// error, or the owner's `Poisoned`.
    Done(Result<Stored<'q, 'src, Q>, QueryError>),
    /// Nothing servable — re-enter the attempt loop. The `&mut` out-params
    /// indicate what to re-enter with: untouched on the phantom paths,
    /// re-resolved against the CURRENT world on the wake path (d18).
    Retry,
}

impl<'q, 'src: 'q, S: Scheduler> Drop for QuerySystem<'q, 'src, S> {
    /// Teardown: run every world-mutation still parked on this system,
    /// from every thread's partition.
    ///
    /// The last chance for two classes of work that no outermost frame of
    /// this system will ever come back for: a mutation parked by a thread
    /// that has since exited, and one parked by a NESTED frame of this
    /// system while the outermost frame belonged to another system.
    /// Before the queue lived on the system, both were destroyed with the
    /// thread-local that held them and the invalidation was silently
    /// lost — never misdirected, just gone.
    ///
    /// Sound because `Drop::drop` runs with `&mut self`: no
    /// `&QuerySystem` can exist, hence no `InFlightGuard`, hence no
    /// compute and no partition drain can race this one, and the quiesce
    /// write lock cannot self-deadlock.  Running the closures rather than
    /// discarding them keeps the contract `mutate_world` advertises — a
    /// deferred mutation executes exactly once, against the exact system
    /// it was parked on — instead of weakening it to "unless that system
    /// was dropped first".
    fn drop(&mut self) {
        assert_eq!(
            self.frames.load(Ordering::Acquire),
            0,
            "dropping a QuerySystem while a compute still holds an \
             InFlightGuard for it — a use-after-free regardless of how \
             the parked mutations are swept"
        );
        let pending: Vec<DeferredMutation<'q, 'src, S>> = {
            // `get_mut`, not `lock`: we already hold `&mut self`, so
            // there is no lock to take and no partition to race.
            let partitions = self.deferred.get_mut();
            let mut pending = Vec::new();
            for partition in partitions.values_mut() {
                pending.append(partition);
            }
            partitions.clear();
            pending
        };
        self.run_deferred_batch(pending);
    }
}

impl<'q, 'src: 'q> QuerySystem<'q, 'src, NoOpScheduler> {
    /// Production constructor: a system whose schedule points are
    /// monomorphized away (zero-cost no-op scheduler).
    pub fn new() -> Self {
        Self::with_scheduler(NoOpScheduler)
    }
}

/// Cross-system deferrals observed by [`QuerySystem::mutate_world`].
///
/// Counts the shape where a mutation is deferred from a frame on system A to
/// a target B that has no live frame on the parked thread: the closure then
/// waits for B to be dropped or quiescently touched, and the invalidation
/// looks ignored meanwhile.  A `usize`-per-drop leak is cheap; an invisible
/// one is not, which is the whole reason this counter exists.
///
/// Relaxed: diagnostic-only, exactly like
/// `dep_graph::CROSS_SYSTEM_SKIPS` — no invariant is read off it.
static DEFERRED_CROSS_SYSTEM: AtomicU64 = AtomicU64::new(0);

/// Test-only view of [`DEFERRED_CROSS_SYSTEM`].
#[cfg(test)]
fn deferred_cross_system_count() -> u64 {
    DEFERRED_CROSS_SYSTEM.load(Ordering::Relaxed)
}

impl<'q, 'src: 'q, S: Scheduler> QuerySystem<'q, 'src, S>
{
    /// Build a system that reports schedule points to `sched`.  The
    /// scheduler is stored behind an `Arc` and shared with the dep graph
    /// so both suspend sites observe the same gate plan.
    pub fn with_scheduler(sched: S) -> Self {
        let id = NEXT_SYSTEM_ID.fetch_add(1, Ordering::Relaxed);
        let shared = Arc::new(sched);
        QuerySystem {
            id,
            sched: shared.clone(),
            deferred: Mutex::new(FxHashMap::default()),
            frames: AtomicUsize::new(0),
            epoch: AtomicU64::new(0),
            vtables: ArcSwap::from_pointee(FxHashMap::default()),
            init_lock: Mutex::new(()),
            resolve_cache: thread_local::ThreadLocal::new(),
            dep_graph: dep_graph::DepGraph::with_owner(id, shared),
            active_keys: (0..ACTIVE_SHARDS)
                .map(|_| CachePadded(RwLock::new(FxHashMap::default())))
                .collect(),
            wait_graph: RwLock::new(FxHashMap::default()),
            quiesce: RwLock::new(()),
            node_to_key: node_to_key::ShardedMap::default(),
        }
    }

    /// Stable per-instance id.  Keys this thread's `DEFERRED_PENDING`
    /// marker and the `IN_DRAIN` set — never an address: addresses are
    /// recycled after drop, which once let one system's deferred closure
    /// run against another.
    fn system_id(&self) -> usize {
        self.id
    }

    /// World generation, bumped by `clear_all_impl` under the quiesce
    /// write lock.  The per-thread cycle partitions (`active_tls`) stamp
    /// the generation they were built in; a stale stamp means the
    /// partition is panic-leak residue from before the reset (no live
    /// frame can own it — `clear_all` excludes every read guard), so it
    /// is reset in place.  Relaxed: the stamp is only ever compared for
    /// equality, and the quiesce lock — not this counter — provides the
    /// ordering.
    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Drain the world-mutations THIS thread parked on THIS system.
    ///
    /// PRECONDITION: no compute of ANY system is in flight on this thread
    /// (`IN_FLIGHT_TOTAL == 0`, asserted below).  Per-system outermost is
    /// NOT enough: a foreign frame holding its own read guard is exactly the
    /// shape that used to be permitted, and it is the reason this asserts
    /// TOTAL rather than depth.
    ///
    /// This is also the one REACHABLE route to the
    /// `IN_DRAIN` tripwire's panic from a `Drop`: dropping a second
    /// `QuerySystem` from inside a deferred closure parked on a FIRST one
    /// runs this method on the thread already draining system 1.  It is a
    /// contrived shape (it takes an explicit alias to another system inside
    /// a world-mutation closure), and special-casing it would mean either
    /// dropping the second system's mutations silently or re-entering the
    /// drain machinery — both worse than the panic.  Documented rather than
    /// defended: the panic is the correct outcome, and knowing it is
    /// reachable stops a maintainer from treating every report of it as a
    /// guard bug.
    /// NOT enough — under cross-system nesting a foreign frame above us
    /// still holds its read guard, and the quiesce write lock taken here
    /// would be the ABBA half of a two-thread deadlock.  All three call
    /// sites uphold this: the guard-`Drop` gate (`prev == 1 && total == 0`),
    /// the `acquire` quiescent-entry flush, and `mutate_world`'s inline
    /// drain-before-write.  At quiescence the write lock cannot
    /// self-deadlock (d11) and no later publish of this thread can erase
    /// the mutations made here (d10's window).
    ///
    /// Only this thread's partition is taken — never another's.  That
    /// thread may still be inside a nested frame of this system, and
    /// invalidating under it is exactly the d10 erase window; parking it
    /// here costs latency, violating it costs soundness.
    ///
    /// Fast path: the presence marker is checked before anything else,
    /// so the overwhelming majority of outermost drops (nothing parked)
    /// touch no lock at all.
    ///
    /// Panic-safe (pre-fix this block hand-set and hand-cleared
    /// `IN_DRAIN` around the batch, so ONE panicking closure left the
    /// flag `true` for the thread's remaining lifetime — silently
    /// disabling both the d10 erase-window protection and the d11
    /// deferral for every later mutation — and dropped the remaining
    /// closures on the unwind).  Now:
    /// - `IN_DRAIN` is registered via `DrainGuard` (RAII, panic-safe);
    /// - each closure runs under `catch_unwind`, so one panicking
    ///   closure neither skips the others nor wedges the flag;
    /// - the first panic is re-raised after the batch so the bug stays
    ///   loud — unless the drain itself is unwinding (a panic raised
    ///   from `Drop` while unwinding aborts the process; logging is the
    ///   best we can do there).
    fn drain_deferred(&self) {
        if !DEFERRED_PENDING.with(|p| p.borrow().contains(&self.id)) {
            return;
        }
        debug_assert!(
            SESSION.with(|s| {
                s.borrow()
                    .systems
                    .iter()
                    .find(|(sid, _)| *sid == self.id)
                    .map(|(_, depth)| *depth)
                    .unwrap_or(0)
            }) == 0,
            "drain_deferred called from inside a running query: \
             route world-mutations through QuerySystem::mutate_world \
             (read→write upgrade would self-deadlock, d11)"
        );
        debug_assert!(
            SESSION.with(|s| s.borrow().total) == 0,
            "drain_deferred with a FOREIGN system's frame live: per-system \
             outermost is not quiescence (cross-system nesting holds a \
             foreign read guard; write here is the ABBA half)"
        );
        DEFERRED_PENDING.with(|p| p.borrow_mut().remove(&self.id));
        let pending: Vec<DeferredMutation<S>> = self
            .deferred
            .lock()
            .remove(&std::thread::current().id())
            .unwrap_or_default();
        self.run_deferred_batch(pending);
    }

    /// Run one batch of parked world-mutations under the quiesce WRITE
    /// lock.  Shared by the two drain sites (an outermost guard's drop,
    /// and `Drop for QuerySystem`), so both get identical `IN_DRAIN`
    /// handling, identical per-closure `catch_unwind`, and identical
    /// panic propagation — a divergence between them would make the
    /// teardown path behave differently from the steady-state one for no
    /// reason other than history.
    fn run_deferred_batch(&self, pending: Vec<DeferredMutation<'q, 'src, S>>) {
        if pending.is_empty() {
            return;
        }
        // A deferred closure runs under the quiesce WRITE lock.  If it
        // tried to run a query, `get_shared` would take the READ lock on
        // this same thread — a parking_lot read→write self-deadlock (the
        // d11 shape).  Built-in closures never do this; assert it so a
        // future closure cannot regress silently.  (Same invariant
        // `drain_deferred` asserts; hoisted here to cover the
        // `Drop for QuerySystem` call site too.)
        debug_assert!(
            SESSION.with(|s| {
                s.borrow()
                    .systems
                    .iter()
                    .find(|(sid, _)| *sid == self.id)
                    .map(|(_, depth)| *depth)
                    .unwrap_or(0)
            }) == 0,
            "run_deferred_batch entered with a live in-flight frame: a deferred \
             closure that runs a query would self-deadlock on the quiesce lock"
        );
        // Two-system ABBA guard: this batch holds THIS system's quiesce
        // write lock for its whole duration.  If a closure reaches a
        // DIFFERENT system's mutate_world (only possible via a foreign
        // &QuerySystem alias captured from outside — the d22 TARGET
        // shape), that nested call takes the other system's write lock
        // under ours; two threads doing mirrored drains would close the
        // cycle T1: B.w → A.w / T2: A.w → B.w.  No built-in closure does
        // this; the panic fires (even single-threaded, conservatively)
        // the moment one ever does.  Same-system re-entrancy is fine and
        // intercepted by mutate_world's IN_DRAIN branch, so at entry
        // IN_DRAIN is empty or exactly {self.id}.  (Paired with the
        // inline-path PANIC in mutate_world, which catches the variant
        // where the foreign system has nothing parked and never enters
        // this batch.)  In release builds the tripwire must not vanish:
        // one TLS set probe (the same cost as the DrainGuard's own
        // presence marker below) catches a foreign system's drain
        // already running on this thread — the two-system ABBA shape —
        // which would otherwise deadlock silently on the two quiesce
        // write locks.  Same-system re-entrancy is fine and intercepted
        // by mutate_world's IN_DRAIN branch, so this fires only for a
        // genuinely foreign id.
        if !IN_DRAIN.with(|d| d.borrow().iter().all(|&id| id == self.id)) {
            panic!(
                "run_deferred_batch: a deferred batch is already running for a \
                 DIFFERENT system on this thread — a closure mutating that \
                 foreign system under our quiesce write lock closes a \
                 two-system ABBA"
            );
        }
        let _drain_guard = DrainGuard::new(self.id);
        let _write = self.quiesce.write();
        // Note: Fam::Deferred is not registered here — that family refers to
        // the `deferred` partition mutex itself (leaf, §0 #5/#17), not the
        // drain scope; during the drain, closures legally acquire
        // active/cache locks through the quiesce write lock (root, not in the
        // matrix).
        let qs = QuiescedSystem {
            system: self,
            _proof: dep_graph::Quiesced::mint(),
        }; // minting point 2
        let mut first_panic: Option<Box<dyn std::any::Any + Send>> = None;
        for f in pending {
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| f(&qs))) {
                eprintln!(
                    "query system: deferred world-mutation panicked: {}",
                    panic_message(payload.as_ref())
                );
                if first_panic.is_none() {
                    first_panic = Some(payload);
                }
            }
        }
        if let Some(payload) = first_panic
            && !std::thread::panicking()
        {
            panic::resume_unwind(payload);
        }
    }

    /// Record one cross-system deferral, rate-limited on stderr.
    ///
    /// Mirrors `dep_graph::note_cross_system_skip` exactly — first
    /// occurrence plus each power-of-two after it, release-only stderr
    /// (a debug build's surrounding behaviour already makes the shape
    /// visible), `eprintln!` because the crate has no logging facade.
    /// The `#[cold]` is load-bearing: this is the "no live frame for the
    /// target" arm, which no correct embedding should take.
    #[cold]
    fn note_cross_system_defer(target: usize, live_frames: &[(usize, u32)]) {
        let n = DEFERRED_CROSS_SYSTEM.fetch_add(1, Ordering::Relaxed);
        if !cfg!(debug_assertions) && (n == 0 || (n + 1).is_power_of_two()) {
            eprintln!(
                "warning: world-mutation deferred to query system {target}, which has no live \
                 frame on this thread (live frames on systems {live_frames:?}).  Only a drop of \
                 system {target}, or a later quiescent touch of it on this thread, runs the \
                 parked closure — until then the mutation is NOT applied.  This is the \
                 cross-system pairing shape: keep the provider and the executing system \
                 paired.  {n} earlier defer(s).",
            );
        }
    }

    /// World-mutation entry point: run `f` under quiescence, or — if
    /// called from inside a running query — DEFER it to the outermost
    /// in-flight guard's drop (after that thread's read lock is released
    /// and its publish is done), so a reentrant mutation can neither
    /// self-deadlock (d11) nor be erased by the in-flight publish
    /// (d10's clear_dirty window, but for the caller's own thread).
    ///
    /// ALL mutation of the red-green world must flow through here:
    /// `invalidate::<Q>`, `invalidate_node`, `clear_all`.  The dep-graph
    /// red-bit mutators on `DepGraph` stay `pub(crate)` plumbing.
    ///
    /// Deferred closures cannot return values (they run at an
    /// unspecified later time on the requesting thread), so the
    /// signature is `FnOnce(&QuerySystem)`.
    fn mutate_world(&self, f: impl FnOnce(&QuiescedSystem<'_, 'q, 'src, S>) + Send + 'static) {
        let id = self.system_id();
        // Draining THIS system's deferred mutations: the write lock is
        // already held by this thread's outermost-guard drain — run
        // inline.  Scoped per system: a DIFFERENT system's mutate_world
        // must NOT skip its own quiesce gate (pre-fix the flag was
        // global and let exactly that happen).
        if IN_DRAIN.with(|d| d.borrow().contains(&id)) {
            // The write lock is already on this thread (run_deferred_batch
            // scope): mint in place.
            let qs = QuiescedSystem {
                system: self,
                _proof: dep_graph::Quiesced::mint(),
            };
            f(&qs);
            return;
        }
        if SESSION.with(|s| s.borrow().total) > 0 {
            // Cross-system deferral: `total > 0` says a frame IS live on
            // this thread, and the check below says NONE of them belongs
            // to the target.  This thread's outermost drop will therefore
            // not be the target's — the parked closure stays unrun until
            // the target is dropped or this thread quiescently touches it
            // again (an invalidation that looks ignored).  Same-system
            // deferral (d11) always has the target's own frame live, so it
            // stays silent; a nested A→B shape has both and drains at B's
            // outermost drop.
            let target_has_frame = SESSION
                .with(|s| s.borrow().systems.iter().any(|(sid, _)| *sid == id));
            if !target_has_frame {
                // Second read, cold path only: the ids are needed solely
                // for the message.
                let live: SmallVec<[(usize, u32); 4]> = SESSION.with(|s| s.borrow().systems.clone());
                Self::note_cross_system_defer(id, &live);
            }
            // Parked on the TARGET system, this thread's partition — see
            // `QuerySystem::deferred` for why neither can be omitted.
            self.deferred
                .lock()
                .entry(std::thread::current().id())
                .or_default()
                .push(Box::new(f));
            DEFERRED_PENDING.with(|p| p.borrow_mut().insert(id));
            // NOTE: no re-check of the session total here — the read is
            // thread-local and nothing between it and the defer can drop
            // a frame (no suspension point), so a second `== 0` check is
            // dead code.  If a suspension point is ever introduced between
            // the defer and here, a drain check becomes necessary again.
            return;
        }
        // Quiescent inline path (TOTAL == 0, not in a drain): flush what
        // nested frames left parked BEFORE taking the write lock — taking
        // it first would self-deadlock the flush.  Ordering preserved:
        // earlier-parked mutations apply before this one.
        //
        // Second half of the two-system ABBA guard (run_deferred_batch
        // carries the first), and UNCONDITIONAL — release builds
        // included.  The variant where the foreign system has NOTHING
        // parked never reaches `run_deferred_batch`'s own check, so a
        // `#[cfg(debug_assertions)]` assert left exactly that shape as a
        // nested write→write deadlock in release.  No false positives:
        // same-system re-entrancy returned at the IN_DRAIN branch above,
        // and the TOTAL > 0 defer path returned before this point.  The
        // check runs BEFORE the flush so the flush can never take this
        // system's write lock under a foreign drain's.  Cost: one TLS
        // set probe per top-level mutation.
        if !IN_DRAIN.with(|d| d.borrow().is_empty()) {
            panic!(
                "mutate_world: acquiring the quiesce write lock while a \
                 world-mutation batch for a DIFFERENT system is draining \
                 on this thread — a closure that reached a foreign \
                 &QuerySystem from inside its own batch would take the \
                 foreign write lock under ours; two mirrored drains close \
                 a two-system ABBA"
            );
        }
        self.drain_deferred();
        let _write = self.quiesce.write();
        // Register the drain marker on the inline path too (same set
        // `run_deferred_batch` uses).  Without it a query or a foreign
        // mutation reached from inside `f` via a captured alias saw an
        // EMPTY set — the release-mode holes in the ABBA guard pair.  The
        // guard is panic-safe (RAII) and cannot alias an existing entry:
        // a same-system re-entrancy returned at the IN_DRAIN branch
        // above, so this id is provably absent.
        let _inline_drain = DrainGuard::new(self.id);
        let qs = QuiescedSystem {
            system: self,
            _proof: dep_graph::Quiesced::mint(),
        }; // minting point 1
        f(&qs);
    }

    /// Look up the vtable for `Q`.  One atomic load + one `Arc` refcount
    /// bump — no lock, no system-wide coordination (Perf Book §13.1).
    fn find_vtable<Q: Query<'q, 'src>>(&self) -> Option<Arc<QueryVTable<'q, 'src, Q>>> {
        self.vtables
            .load()
            .get(&TypeId::of::<Q>())
            .map(|arc| {
                downcast_vtable::<Q, S>(arc.as_ref())?;
                let arc_clone = arc.clone();
                // SAFETY: the TypeId lookup guarantees the concrete type
                // is exactly QueryVTable<'q, 'src, Q>, so the raw pointer cast is valid.
                Some(unsafe {
                    Arc::from_raw(Arc::into_raw(arc_clone) as *const QueryVTable<'q, 'src, Q>)
                })
            })
            .flatten()
    }

    /// Look up the vtable as a type-erased executor for the given TypeId.
    fn find_vtable_dyn(&self, ty: TypeId) -> Option<Arc<dyn QueryVTableDyn<'q, 'src, S> + 'q>> {
        self.vtables.load().get(&ty).cloned()
    }

    /// Resolve `Q`'s vtable through the per-thread cache ([`resolve_cache`]).
    /// Hot-path twin of `vtable_or_init`: the hit path serves a borrow out
    /// of the cached `Arc` — one `ThreadLocal` slot fetch, one epoch
    /// compare, one `TypeId` compare — with no atomic load, no hash probe,
    /// and no `Arc` refcount bump.  Takes no lock in any family, so no
    /// tripwire registration.
    ///
    /// The returned guard fences the borrow with the `RefCell` tracker
    /// itself (see `VtableResolve::Cached`): a nested first-ever-use of
    /// another query type fails `try_borrow_mut` and falls back to the
    /// owned path instead of reallocating under a live borrow.  The slow
    /// path (miss, stale epoch, contended refill) holds no borrow at all.
    fn resolve_vtable<Q: Query<'q, 'src>>(&self) -> VtableResolve<'q, '_, 'src, Q> {
        let ty = TypeId::of::<Q>();
        let epoch = self.epoch();
        let slot = self
            .resolve_cache
            .get_or(|| std::cell::RefCell::new(Vec::new()));
        // Fast path: shared borrow only.  `filter_map` hands the borrow
        // back on miss so the slow path starts unborrowed.
        match std::cell::Ref::filter_map(slot.borrow(), |entries| {
            entries
                .iter()
                .find(|e| e.ty == ty && e.epoch == epoch)
                .and_then(|e| downcast_vtable::<Q, S>(e.vtable.as_ref()))
        }) {
            Ok(hit) => return VtableResolve::Cached(hit),
            Err(_borrow) => {}
        }
        // Slow path holds NO borrow (the miss handed it back), so nested
        // resolves refill freely.  Registration still serializes on
        // `init_lock` inside.
        let owned = self.vtable_or_init::<Q>();
        // Best-effort caching: `try_borrow_mut` fails only when an outer
        // frame's `Cached` guard holds the shared borrow (a nested
        // first-ever-use) — then skip caching and serve owned.  Replacing
        // a stale-epoch entry is safe: only this thread ever touches its
        // slot, and a live outer guard would fail the `try` above instead
        // of letting us swap under it.
        if let Ok(mut entries) = slot.try_borrow_mut() {
            let erased: Arc<dyn QueryVTableDyn<'q, 'src, S> + 'q> = owned.clone();
            match entries.iter_mut().find(|e| e.ty == ty) {
                Some(e) => {
                    e.epoch = epoch;
                    e.vtable = erased;
                }
                None => entries.push(ResolveEntry {
                    ty,
                    epoch,
                    vtable: erased,
                }),
            }
        }
        VtableResolve::Owned(owned)
    }

    /// Re-probe the cache and, if the freshly published value verifies
    /// green, serve it.  Shared by the two phantom-wait retry sites in
    /// `get_shared`'s Wait branch (owner completed while we were
    /// registering / before we parked): the value is already in the cache,
    /// so serving it avoids a full recompute.  Same validity discipline as
    /// the wake path — a non-quiesced fan-out can have marked the node red
    /// between the owner's publish and this probe, in which case we fall
    /// through to recompute (`None`).
    /// Re-probe + serve step for the cross-thread waiter path (the latch
    /// wake), mirroring the cached-hit branch of `get_shared`.
    ///
    /// NOTE: stats double-count on this path.  The pre-wait probe already
    /// recorded a hit/miss for this logical request and the re-probe
    /// inside `try_serve_verified` records a second one, so hit ratios
    /// read optimistic whenever a thread actually waited.  Kept simple
    /// deliberately — the wake path cannot always attribute a phantom
    /// wait to its logical request, and the alternative (threading a
    /// "already counted" flag through the waiter) buys accuracy in a
    /// diagnostic counter at the price of bookkeeping on the latency
    /// path.
    fn serve_reprobed<Q: Query<'q, 'src>>(
        &self,
        vtable: &QueryVTable<'q, 'src, Q>,
        key: &Q::Key,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Option<Stored<'q, 'src, Q>> {
        let (node, cached, fresh) = probe(vtable, &self.dep_graph, key);
        if fresh {
            // Third node_to_key registration site (get_shared.register /
            // get_shared.wake_register are the other two): register for
            // the same discipline — a future caller under another lock
            // family must be visible to the tripwire.
            let _tw = sync_order::Acquired::acquire(
                sync_order::Fam::NodeToKey,
                "serve_reprobed.register",
            );
            self.node_to_key.insert(node, Q::into_tagged(key));
        }
        let value = cached?;
        self.try_serve_verified::<Q>(
            vtable
                .cache
                .as_ref()
                .expect("untracked query reached the tracked serve path"),
            node,
            value,
            provider,
        )
    }

    /// The cached-hit serve step shared by the initial probe and the
    /// post-wake re-probe.  `None` cached → `None` (fall through to compute).
    fn serve_cached_verified<Q: Query<'q, 'src>>(
        &self,
        cache: &Q::Cache,
        node_index: dep_graph::DepNodeIndex,
        cached: Option<Stored<'q, 'src, Q>>,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Option<Stored<'q, 'src, Q>> {
        let value = cached?;
        self.try_serve_verified::<Q>(cache, node_index, value, provider)
    }

    /// Re-request a dirty dependency during transitive green verification.
    /// Typed variants dispatch directly to `get_shared` with zero `Any`
    /// downcast; only the `Opaque` fallback goes through the vtable.
    fn rerequest_dirty_dependency(
        &self,
        tag: &TaggedQueryKey,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) {
        // Errors ignored: on failure the re-verify below fails and the
        // caller falls through to compute.
        match tag {
            TaggedQueryKey::TypeDefShape(key) => {
                let _ = self.get_shared::<queries::TypeDefShape>(*key, provider);
            }
            TaggedQueryKey::CheckFnSignature(key) => {
                let _ = self.get_shared::<queries::CheckFnSignature>(key.clone(), provider);
            }
            TaggedQueryKey::CheckFnBody(key) => {
                let _ = self.get_shared::<queries::CheckFnBody>(key.clone(), provider);
            }
            TaggedQueryKey::DocTokens(key) => {
                let _ = self.get_shared::<queries::DocTokens>(key.clone(), provider);
            }
            TaggedQueryKey::Opaque(ty, _) => {
                if let (Some(vtable_dyn), Some(any)) = (self.find_vtable_dyn(*ty), tag.as_any()) {
                    let _ = vtable_dyn.get_shared_erased(self, any, provider);
                }
            }
        }
    }

    /// Shared validity gate for a cached value about to be served, keyed
    /// by `node_index`: transitive green verification with re-request of
    /// dirty dependencies (rustc-style try_mark_green).  Returns
    /// `Some(value)` when the value may be served (closure green, or
    /// green again after re-requesting dirty deps) and records the
    /// current task's read of the node.  Returns `None` when the caller
    /// must fall through to recompute: the node itself turned red (d17,
    /// and the owner-side d16 epoch restore), a dirty dependency could
    /// not be resolved, or a re-requested dependency's changed
    /// fingerprint re-dirtied the node.
    ///
    /// Used by BOTH serve paths:
    /// - the probe-hit (cached) branch of `get_shared`, and
    /// - the cross-thread waiter branch after a latch wake.  The waiter
    ///   branch previously served its cache lookup with NO validity
    ///   hole: the owner's finish can legitimately leave the
    ///   node red (the d16 epoch re-check) while the value is already in
    ///   the cache (`cache.insert` precedes `signal_complete`), and a
    ///   non-quiesced changed-fingerprint fan-out from another compute
    ///   can mark the node red between the owner's publish and the
    ///   waiter's serve — both made the unverified serve a stale-value
    ///   hole.
    /// THE hit-accounting point (contract 5): a hit is a real green cache
    /// decision, and that decision is made HERE, not by the `probe` that
    /// merely supplied a candidate value.  `None` is the caller's signal to
    /// recompute, so it records the compensating miss.
    ///
    /// Kept as a wrapper rather than threading accounting through the
    /// verification body, which has three separate `Some(value)` return
    /// sites (direct-green, green-after-re-request, and the deep-green tail)
    /// and would otherwise need one `record_hit` at each — with the next
    /// added return site silently regressing the contract.
    fn try_serve_verified<Q: Query<'q, 'src>>(
        &self,
        cache: &Q::Cache,
        node_index: dep_graph::DepNodeIndex,
        value: Stored<'q, 'src, Q>,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Option<Stored<'q, 'src, Q>> {
        let out = self.try_serve_verified_inner::<Q>(node_index, value, provider);
        match out {
            Some(_) => cache.record_hit(),
            None => cache.record_miss(),
        }
        out
    }

    /// The verification body: transitive green check, dirty-dependency
    /// re-request, and the read-edge record.  Returns `Some(value)` only
    /// when the value may actually be served.
    fn try_serve_verified_inner<Q: Query<'q, 'src>>(
        &self,
        node_index: dep_graph::DepNodeIndex,
        value: Stored<'q, 'src, Q>,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Option<Stored<'q, 'src, Q>> {
        if let Some(dirty_dep) = self
            .dep_graph
            .verify_green_transitive_with_dirty(node_index)
        {
            // The node ITSELF was marked red between capture and here (a
            // concurrent invalidate, a dependency's changed-fingerprint
            // fan-out, or — waiter case — the owner's own d16 epoch
            // restore): serving the captured value would disagree with
            // the graph — recompute instead (d17).
            if dirty_dep == node_index {
                return None;
            }
            // Clone (TypeId, key) out and DROP the shard lock before
            // re-requesting: the re-request's nested `get_shared` takes
            // a `node_to_key` shard write lock — a same-thread read→write
            // upgrade self-deadlocks parking_lot RwLocks (this hung
            // g1c/stress before the fix).
            let dep_info: Option<TaggedQueryKey> = {
                let _tw = sync_order::Acquired::acquire(
                    sync_order::Fam::NodeToKey,
                    "try_serve_verified.read",
                );
                self.node_to_key.clone_out(dirty_dep)
            };
            let Some(tag) = dep_info else {
                return None; // unresolvable dep: recompute (conservative)
            };
            self.rerequest_dirty_dependency(&tag, provider);
            // If the dependency's fingerprint changed, its publish marked
            // THIS node red — the re-verify below returns the node itself.
            if self
                .dep_graph
                .verify_green_transitive_with_dirty(node_index)
                .is_none()
            {
                // All green now — serve the cached value.  Record that the
                // current task read this node (no-op without an active
                // task): without this edge, a later invalidation of the
                // node never dirties the outer query, causing silent
                // staleness — the waiter's dep edge was not recorded.
                self.dep_graph.read(node_index);
                return Some(value);
            }
            return None;
        }
        // All transitive dependencies green — serve.
        self.dep_graph.read(node_index);
        Some(value)
    }

    /// Get `Q`'s vtable, registering it on first use.  The registration
    /// path runs at most once per query type per session; every later
    /// call is the `find_vtable` fast path.
    fn vtable_or_init<Q: Query<'q, 'src>>(&self) -> Arc<QueryVTable<'q, 'src, Q>> {
        if let Some(vt) = self.find_vtable::<Q>() {
            return vt;
        }
        // Registration BEFORE the lock (single rule: register, then
        // acquire) — matches clear_all_impl's shape.
        let _tw_init = sync_order::Acquired::acquire(sync_order::Fam::InitLock, "vtable_or_init");
        let _guard = self.init_lock.lock();
        if let Some(vt) = self.find_vtable::<Q>() {
            return vt; // lost the registration race
        }
        // The query NAME is load-bearing identity in three
        // places that have no second identity to fall back on —
        // `SchedulePoint::{CachedHit, AfterPublish}` (a `TestScheduler` gate
        // armed on a name parks whichever query matches, so two queries
        // sharing a name silently degrade the d17/d15 choreography to the
        // WRONG query's getter) and `QueryError::{Cycle, ...}::query_name`
        // (diagnostics become ambiguous).  Names are never checked for
        // uniqueness anywhere — not in `for_each_query!`, not here.
        //
        // Nothing in production depends on this, so the check is a
        // `debug_assert!`: the scan is O(registered queries) on a
        // first-touch-only path, but it is still a linear registry walk
        // that buys nothing at runtime.  A release build keeps serving
        // correctly with a duplicate name; a debug build fails the moment a
        // collision is introduced, which is the only time it can still be
        // fixed cheaply.  (`TypeId` would be the better identity for the
        // schedule points — see scheduler.rs's `SchedulePoint`.)
        debug_assert!(
            !self
                .vtables
                .load()
                .values()
                .any(|vt| vt.descriptor_name() == Q::descriptor().name),
            "query name '{}' is already registered by a DIFFERENT query type: \
             names double as `SchedulePoint::CachedHit` / `AfterPublish` identity \
             and as `QueryError` diagnostic identity, and neither has a second \
             key to disambiguate.  Make the two names distinct.",
            Q::descriptor().name,
        );
        let vt: Arc<QueryVTable<'q, 'src, Q>> = Arc::new(QueryVTable::<Q>::new());
        let mut map = FxHashMap::clone(&*self.vtables.load());
        map.insert(TypeId::of::<Q>(), vt.clone());
        self.vtables.store(Arc::new(map));
        vt
    }

    /// Split active-key acquisition into validation + commit.  Returns
    /// `KeySlot::Compute` when this thread will run the query (and pushes
    /// onto its active stack), `KeySlot::CycleContext` when this thread
    /// re-entered a handler-equipped query (typed stack ferried out — no
    /// lock held on return), or `KeySlot::Wait(job)` when another thread
    /// owns the computation — in which case the current thread's stack is
    /// untouched (no speculative enter/leave needed).
    #[inline(always)]
    fn acquire_key_slot<Q: Query<'q, 'src>>(
        &self,
        vtable: &QueryVTable<'q, 'src, Q>,
        key: &Q::Key,
        akey: &ActiveKey,
        total: u32,
    ) -> Result<KeySlot<'q, 'src, Q>, QueryError> {
        let me = std::thread::current().id();

        // Same-thread cycle and depth checks BEFORE any cross-thread wait.
        // The cycle arm ferries the typed stack OUT: the handler runs at
        // the `get_shared` slot match on lock-free ground (the cycle state
        // is TLS — nothing is held here — and the `ActiveShard` write lock
        // taken below is released before the slot match returns).  Running
        // it here would self-deadlock a handler that nests a get.
        let epoch = self.epoch();
        // `total` arrives with the caller (read once from the guard that
        // performed the bump — same zero-extra-access shape as
        // `validate_enter`): the firing depth for both the global check
        // and the error payloads below.
        match vtable.validate_enter(self.system_id(), epoch, key, total) {
            Ok(()) => {}
            Err(EnterError::Cycle { stack }) => {
                if vtable.cycle_processor.is_some() || vtable.cycle_recovery.is_some() {
                    return Ok(KeySlot::CycleContext { stack });
                }
                return Err(cycle_error::<Q>(vtable.descriptor.name, key, &stack, None));
            }
            Err(EnterError::Fatal(e)) => return Err(e),
        }

        if vtable.descriptor.eval_always {
            vtable.commit_enter(self.system_id(), epoch, key);
            return Ok(KeySlot::Compute);
        }

        // Register for the WHOLE critical section: the tripwire must see
        // ActiveShard held for the entire match — not just the acquisition
        // expression.  (The cycle state committed below is TLS and takes
        // no lock, so unlike the old sharded-map design there is no nested
        // acquisition inside this guard.)
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::ActiveShard, "acquire_key_slot");
        let mut active = self.active_keys[active_shard(akey)].write();

        // NOTE: the `Arc` in `QueryJob::new()` is ALLOCATED INSIDE the
        // write-lock below (the `Vacant` arm only).  Hoisting it out
        // would need a read-lock pre-check that could race — a
        // write-lock protocol change for a one-off allocation on a
        // cold-miss path, which is exactly the path about to pay for
        // the compute it precedes.  Not worth it; the wait paths and
        // `Started` hit paths allocate nothing.

        // `entry` hashes the key ONCE for the get-or-insert below (a `get`
        // followed by `insert` would hash twice per miss acquisition).  The
        // `akey.clone()` is an `Arc` bump only; the Wait arms borrow it and
        // drop it, same as before.
        use std::collections::hash_map::Entry;
        match active.entry(akey.clone()) {
            Entry::Occupied(o) => match o.get() {
                ActiveKeyStatus::Started(job) if job.owner == me => {
                    // Double-safety guard: `validate_enter` should already catch
                    // same-thread re-entry, but keep the explicit check.  With
                    // the map keyed by the EXACT key this arm is unreachable
                    // through correct input — a different key can no longer
                    // land on this entry (that was the hash-collision false
                    // cycle).
                    return Err(QueryError::Cycle {
                        query_name: Q::descriptor().name,
                        key: format!("{:?}", key),
                        stack: vec![format!("{:?}", key)],
                        note: None,
                    });
                }
                ActiveKeyStatus::Started(job) => {
                    // Another thread owns this key — we only wait, no commit.
                    return Ok(KeySlot::Wait(job.clone()));
                }
                ActiveKeyStatus::Poisoned => {
                    // Exact-key hit: the poison BELONGS to this key.
                    // Deliberately NOT cleared here — that is the whole
                    // point of a sticky marker.  Only [`Self::invalidate`]
                    // and [`Self::clear`] (and `clear_all`, by wiping
                    // every marker) lift it.  Return `Err` so the caller
                    // STOPS: re-activating the key here would hide the
                    // panic that installed the poison, and the latch watch
                    // in the caller would see no waiter, both making the
                    // failure mode silent instead of remembered.
                    return Err(QueryError::Poisoned {
                        query_name: Q::descriptor().name,
                        key: format!("{:?}", key),
                    });
                }
            },
            Entry::Vacant(v) => {
                // We become the owner — commit the active-stack entry.
                // The key was cloned once for the entry lookup above; the
                // Wait paths borrow it and drop it.
                vtable.commit_enter(self.system_id(), epoch, key);
                let job = job::QueryJob::new();
                v.insert(ActiveKeyStatus::Started(job));
                Ok(KeySlot::Compute)
            }
        }
    }

    /// Execute query `Q` and hand back a shared handle to the value.
    ///
    /// This is the primitive the whole system funnels through: cache hits
    /// return the stored `Arc` without touching the payload, and freshly
    /// computed values are wrapped exactly once (the same handle is
    /// published to the cache, fingerprinted, and returned).  Clone-free
    /// for `Arc` consumers; use [`QuerySystem::get`] for the owned value.
    ///
    /// Stack growth (rustc `ensure_sufficient_stack` shape, same
    /// constants as the checker's `infer_expr` wrapper): every frame —
    /// hit or miss — pays one SP read + compare (~1–2 ns) and grows a
    /// 16 MiB segment only when the remaining stack drops below the
    /// 256 KiB red zone.  Legal deep chains therefore cannot overflow
    /// the worker stack however deep they nest; `MAX_GLOBAL_DEPTH`
    /// remains purely as the runaway-recursion diagnostic.  Recursion
    /// nests through here, so one wrapper at the entry covers the whole
    /// chain — inner frames re-check and grow again as needed.
    pub fn get_shared<Q: Query<'q, 'src>>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Result<Stored<'q, 'src, Q>, QueryError> {
        // Untracked queries bypass the tracked frame entirely: no node,
        // no active key, no dep edges — straight to the memo map.  Before
        // `maybe_grow` (nesting is impossible, no growth ever needed).
        //
        // NOTE: `Q::descriptor()` here is a `const fn`
        // building a small POD struct by value, so the cost is a couple of
        // register moves before the vtable is resolved — measured as
        // negligible, and NOT worth folding into the descriptor: the split
        // is what lets the untracked path skip `maybe_grow` at all, which
        // is the win this route is after.
        if Q::descriptor().untracked {
            return self.get_untracked::<Q>(key, provider);
        }
        // Red zone 256 KiB: when the remaining stack drops below this,
        // grow by a 16 MiB segment (amortized — only triggered on deep
        // chains; the common case is one register read + compare).
        stacker::maybe_grow(256 * 1024, 16 * 1024 * 1024, || {
            self.get_shared_inner::<Q>(key, provider)
        })
    }

    /// Memo-only serve path for [`QueryDescriptor::with_untracked`]
    /// queries.  No `DepNodeIndex`, no active-key slot, no dep edges, no
    /// durability/fingerprint traffic, no `InFlightGuard`/session/read
    /// lock: the memo map on the vtable is the whole served world, and
    /// it dies with the vtable on `clear_all`, so no generation stamp is
    /// needed.  A miss runs `compute` with nesting fenced off, then
    /// memoizes under a capacity bound.
    ///
    /// That quiescence exemption is the one sharp edge: a `clear_all` can
    /// land between this function's first hash and the `commit_*` on the
    /// other side, replacing the whole vtable registry (nothing holds a
    /// write lock here).  The compute's result then memoizes into the
    /// ORPHANED vtable — lost work.  It is never served (every later
    /// hash resolves to the NEW registry), so the failure mode is a
    /// recompute, never staleness.  Untracked queries are the "no world
    /// to invalidate" shape and pay exactly that price for it.
    ///
    /// PANIC SEMANTICS differ deliberately from the tracked path —
    /// a panicking untracked compute propagates RAW
    /// (`catch_unwind` at the call site is the caller's business) and
    /// memoizes nothing — there is no `ActiveKeyStatus::Poisoned` marker
    /// and no `poison_active`, because there is no active-key slot at all.
    /// The tracked path instead poisons the slot and returns
    /// `Err(QueryError::Poisoned)`, so a second get is a loud failure
    /// rather than a recompute.  That asymmetry is safe rather than
    /// sloppy: an untracked query has no edges to invalidate, so the next
    /// get simply recomputes from scratch.  What is NOT free is that a
    /// caller cannot retrieve the poison error — an untracked panic must
    /// be caught at the boundary (e.g. the `catch_unwind` in the
    /// churn-worker harness) or it unwinds through the compiler.
    fn get_untracked<Q: Query<'q, 'src>>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Result<Stored<'q, 'src, Q>, QueryError> {
        // Purity fence (untracked-in-untracked is still a nested get).
        if UNTRACKED_DEPTH.with(|d| d.get()) != 0 {
            panic!(
                "query '{}' issued from inside an untracked computation — untracked queries must be pure functions of (key, immutable data)",
                Q::descriptor().name,
            );
        }
        let key = key.into_query_key();
        let vtable = self.vtable_or_init::<Q>();
        // Fast path: memo hit — one leaf shard read.
        {
            let _tw =
                sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "get_untracked.read");
            if let Some(value) = vtable.untracked.read().get(&key) {
                return Ok(value.clone());
            }
        }
        // Miss: compute with nesting forbidden.  The guard restores the
        // depth on unwind so a panicking compute cannot fence the thread.
        UNTRACKED_DEPTH.with(|d| d.set(d.get() + 1));
        let _depth = UntrackedDepthGuard;
        let value = erase_value::<'q, 'src, Q>(Q::compute(&key, provider));
        drop(_depth);
        // Memoize, bounded by the descriptor's cache capacity (0 =
        // unbounded).  Evict ONE entry on overflow — a wholesale clear
        // is a working-set cliff: once the memo reaches capacity, every
        // refill cycle wipes all entries and any re-read of an earlier
        // key is a full recompute (the exact failure mode the
        // `cache_capacity` doc warns about, re-implemented in the one
        // cache that had no eviction policy).  Keys are small identity
        // types, so "evict one arbitrary entry" is plenty for a pure
        // memo.
        //
        // The old comment here claimed "insertion-order
        // eviction (pseudo-LRU)", which the code does NOT provide:
        // `FxHashMap::keys().next()` is BUCKET order (hash plus capacity
        // dependent), not insertion or recency order.  A hot key can be
        // evicted while cold keys persist, and a working set just over
        // capacity can thrash arbitrarily.  The consequence is
        // RECOMPUTE-ONLY (untracked queries are pure, so a recomputed
        // value is identical), but claiming a property the code lacks is
        // how a future maintainer ends up trusting an ordering that was
        // never there.  If recency ever matters for a caller, the fix is
        // a `VecDeque<K>` insertion queue beside the map, NOT a comment.
        // FxHashMap order is arbitrary and the recompute is
        // deterministic, so this costs only a recomputation, never a
        // wrong answer.
        //
        // Note: the memo map is a single global RwLock — one write per
        // miss insert.  Acceptable at current scale; shard it the same way
        // as `node_to_key` if miss-insert contention ever shows up in a profile.
        {
            let _tw =
                sync_order::Acquired::acquire(sync_order::Fam::CacheShard, "get_untracked.insert");
            let mut map = vtable.untracked.write();
            let cap = vtable.descriptor.cache_capacity;
            if cap != 0
                && map.len() >= cap
                && let Some(k) = map.keys().next().cloned()
            {
                map.remove(&k);
            }
            map.insert(key, value.clone());
        }
        Ok(value)
    }

    /// The recursive query core, called through the `get_shared`
    /// stack-growth wrapper above.
    fn get_shared_inner<Q: Query<'q, 'src>>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Result<Stored<'q, 'src, Q>, QueryError> {
        // Purity fence: a query issued from inside an untracked
        // computation violates the untracked contract (hard panic, not a
        // cycle error — the author broke the promise, the graph is fine).
        if UNTRACKED_DEPTH.with(|d| d.get()) != 0 {
            panic!(
                "query '{}' issued from inside an untracked computation — untracked queries must be pure functions of (key, immutable data)",
                Q::descriptor().name,
            );
        }
        let key = key.into_query_key();

        // Quiesce read lock, taken BEFORE the preflight (vtable
        // resolution, node allocation, key registration) — placement is
        // load-bearing: a `clear_all` interleaved between the
        // preflight and the compute path would reset the dep graph
        // (`next_index = 1`) and the registry under this frame's feet,
        // letting it publish an old-generation vtable against a recycled
        // node index.  The guard is held for the WHOLE frame (early
        // returns in the cached branch drop it), so every preflight step
        // is serialized against world mutations.  Cost: nested same-system
        // gets skip the read lock via the session's per-system depth
        // (only a frame that brings its system's depth 0 → 1 pays one
        // shared read acquisition).
        //
        // The earlier text added "a nested frame on a
        // FOREIGN system holds that system's guard instead", which is now
        // impossible: `InFlightGuard::acquire`'s mispair tripwire panics
        // on the first foreign sid.  Only `DepGraph::read`'s own
        // cross-system check remains as the second line of defence.
        // Deadlock-freedom: a thread blocked HERE (invalidator holds the
        // write lock) owns no active key, so no read-lock holder can
        // ever be waiting on its job; every wait-on-latch chain among
        // read holders terminates at a thread that is computing and will
        // publish+release.  (Placing the guard AFTER `acquire_key_slot`
        // instead deadlocks: the owner queues for the read lock while
        // holding a `Started` job, and a nested reader blocked on that
        // job closes the cycle with the invalidator.)  The Wait branch
        // below voluntarily RELEASES the lock across the latch wait
        // (`park_wait`) — a waiter cannot publish, and holding the read
        // lock across a latch wait would close a three-way cycle with
        // the invalidator's write lock.  Nested frames cannot outlive
        // their outer frame's publish.
        let mut _in_flight = InFlightGuard::acquire(self);

        // Resolve the vtable and the stable DepNodeIndex for
        // (query_type, key), short-circuiting on a green cache hit.  No
        // system-wide lock anywhere on this path: the first-ever use of
        // `Q` registers through `init_lock`, every later call is a
        // thread-local cache hit (`resolve_vtable`).
        // `mut`: the WAIT path re-resolves both after a latch wake — the
        // wait voluntarily released the quiesce read lock, so a completed
        // `clear_all` invalidates the pre-wait snapshot (see the Wait
        // branch, d18).
        let mut vtable = self.resolve_vtable::<Q>();
        let (mut node_index, cached, fresh) = probe(&vtable, &self.dep_graph, &key);

        // Register the key for this node EXACTLY ONCE, on the fresh path
        // only: the mapping is immutable for the node's lifetime, so re-
        // writing it on every hit just burned a global write lock plus a
        // `Box`ed key clone per query execution.  A node only ever needs
        // the mapping after it has been computed (a later request can
        // re-request it as a dirty dependency), and the creator's
        // `get_shared` reaches the compute path before any reader can
        // depend on it — registering here is sufficient.
        if fresh {
            let _tw =
                sync_order::Acquired::acquire(sync_order::Fam::NodeToKey, "get_shared.register");
            self.node_to_key.insert(node_index, Q::into_tagged(&key));
        }

        // If we have a cached value, verify its transitive dependency
        // closure.  If verification finds a dirty dependency, re-request
        // it (nested get) and re-verify.  This implements rustc-style
        // try_mark_green: a cached value is served only if all transitive
        // dependencies are green (or become green after re-requesting).
        if let Some(value) = cached {
            // d17/d15 window (probe captured the stored handle → before
            // `try_serve_verified`): a concurrent mark of THIS node while
            // parked here is exactly the stale-serve race; query identity
            // keys the match so only the armed query parks.
            self.sched.suspend(SchedulePoint::CachedHit {
                query: Q::descriptor().name,
            });
            // Shared validity gate (also used by the waiter branch below):
            // serve only if the transitive closure is green; a dirty
            // dependency is re-requested and re-verified first.  The
            // helper records this task's read of the node on success.
            if let Some(value) = {
                let cache = vtable
                    .cache
                    .as_ref()
                    .expect("untracked query reached the tracked serve path");
                self.serve_cached_verified::<Q>(cache, node_index, Some(value), provider)
            } {
                return Ok(value);
            }
            // The node itself was red, a dirty dependency could not be
            // resolved, or a re-requested dependency's changed fingerprint
            // re-dirtied this node — fall through to the compute path.
        }

        let akey = ActiveKey::of::<Q>(&key);

        // (The quiesce read lock is already held — `InFlightGuard` is
        // acquired at the TOP of this function, before the preflight, so
        // `clear_all` cannot interleave between node allocation and the
        // compute path.)

        // Attempt loop: either become the computing thread, or wait for
        // the owner and consume its published value.  A waiter loops back
        // when a concurrent invalidation removed the value from the cache
        // between the owner's publish and the waiter's lookup — it must
        // recompute, never report "not found after waiting".
        loop {
            // Acquire the key slot — this validates cycle/depth first,
            // then either assigns us as the compute owner or returns a
            // job to wait on. No speculative stack mutation for waiters,
            // and no global lock: the vtable is already in hand.
            let slot = self.acquire_key_slot::<Q>(&vtable, &key, &akey, _in_flight.total())?;

            match slot {
                KeySlot::Compute => break,
                KeySlot::CycleContext { stack } => {
                    // Same-thread cycle on a handler-equipped query: run
                    // the tier-2 loop processor first, then the tier-1
                    // legacy handler — HERE, where no cycle-state borrow
                    // is held (the check is TLS and its borrow ended
                    // inside `acquire_key_slot`) and the `ActiveShard`
                    // write lock was never acquired on this path, so
                    // nested gets are safe.  The
                    // re-entry never committed: active entry, task frame,
                    // and vtable stack are untouched, so there is nothing
                    // to stage, clear, or signal — just serve the
                    // recovery value.  A handler panic unwinds into the
                    // outer compute's `catch_unwind` (≡ compute panic →
                    // poison), reusing existing machinery.
                    if let Some(process) = vtable.cycle_processor {
                        let ctx = CycleContext {
                            key: key.clone(),
                            stack: stack.clone(),
                            wait_chain: Vec::new(),
                            is_cross_thread: false,
                        };
                        match process(&ctx) {
                            CycleDecision::Recover(recovered) => {
                                if self.dep_graph.current_node().is_some() {
                                    self.dep_graph.read(node_index);
                                }
                                return Ok(erase_value::<'q, 'src, Q>(recovered));
                            }
                            CycleDecision::Abort { note } => {
                                return Err(cycle_error::<Q>(
                                    vtable.descriptor.name,
                                    &ctx.key,
                                    &ctx.stack,
                                    Some(note),
                                ));
                            }
                        }
                    }
                    let recover = vtable.cycle_recovery.expect(
                        "acquire_key_slot yields CycleContext only when a handler is present",
                    );
                    let recovered = recover(&key, &stack);
                    // Recovery dep edge: without it, invalidating the
                    // re-entered key never fans out here and the
                    // recovery chain freezes in cache (c5).  Self-cycle
                    // no-ops inside `read` (other == current node).
                    if self.dep_graph.current_node().is_some() {
                        self.dep_graph.read(node_index);
                    }
                    return Ok(erase_value::<'q, 'src, Q>(recovered));
                }
                KeySlot::Wait(job) => {
                    // Extracted into `wait_for_owner` (the old NOTE lives in the
                    // helper function's doc comment). Contract: `Done(r)` resolves
                    // this request — return `r` as-is; `Retry` loops back, with
                    // `vtable` / `node_index` re-resolved against the current world
                    // on the wake path (d18) and unchanged on the phantom paths.
                    match self.wait_for_owner::<Q>(
                        &mut vtable,
                        &mut node_index,
                        &key,
                        &akey,
                        provider,
                        &mut _in_flight,
                        job,
                    ) {
                        WaitOutcome::Done(result) => return result,
                        WaitOutcome::Retry => continue,
                    }
                }
            }
        }

        // (The quiesce read lock is held for this whole function — taken
        // above, before the attempt loop.  The Wait branch voluntarily
        // released it across the latch wait and re-took it on wake, so
        // this compute path is already quiesced.)

        // Start tracking dependencies for this node.
        // ── Unprotected-stretch guard: latch stranding ──
        //
        // Everything between `acquire_key_slot`'s `ActiveKeyStatus::Started`
        // insert and the staging `catch_unwind` below used to run UNGUARDED.
        // `start_task` pushes a TLS frame (a `RefCell` borrow plus a `Vec`
        // push) and `set_finishing` takes the node's shard READ lock, whose
        // `sync_order::check` trips in debug/shuttle builds — so BOTH can
        // panic.  A panic there stranded every waiter parked on this job's
        // ONE-SHOT latch (`QueryLatch::wait_on` has no timeout: it spins,
        // then parks on a condvar until `set`/`poison`) AND left this
        // thread's `TASK_STACK` holding a stale frame.  That is the exact
        // failure mode the staging and finalize guards below were added for,
        // so this stretch is routed through the SAME poison path.
        //
        // LOW probability (a latent TLS-borrow bug or a debug ordering
        // tripwire has to fire right here), SEVERE outcome (an
        // unrecoverable hang with no diagnostic) — cheap insurance.
        //
        // COVERAGE (honest, following the `clear_all` poison-sweep
        // precedent): this guard is UNREACHABLE-BY-CONSTRUCTION
        // today and therefore has NO direct test — a test cannot be made to
        // reach it without faking the panic.
        //
        // `start_task` and `set_finishing` are internal `dep_graph` calls,
        // not trait-dispatched, so there is no injectable panic hook here
        // the way `PanickingInsertCache` works (that one is testable
        // because `stage` IS trait-dispatched).  And the surviving panic
        // source, `set_finishing`'s `with_shard_r`, cannot GROW (growing is
        // `with_shard_and_grow`, so no `GrowLock` is reachable), which
        // leaves only `sync_order::check` — which cannot fire on the normal
        // path because no conflicting lock family is held at this point.
        //
        // So this is defense-in-depth for a future bug, exactly like the
        // poison sweep.  The tripwire is the same in both: the moment it
        // stops being dead code, the observable changes — here, a get
        // that would have hung forever instead returns `Err(Poisoned)`.
        // If `sync_order::check` ever DOES fire in this stretch, the guard
        // is what converts that panic into a poisoning rather than a
        // stranding, and the eprintln above is the diagnostic.

        // (The long Phase-1 comment below documents the staging window; it is
        // unchanged.  The guard goes here so `start_task`'s frame and the d24
        // `set_finishing` flag are both covered before the compute runs.)
        if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| {
            self.dep_graph.start_task(node_index);
            self.dep_graph.set_finishing(node_index);
        })) {
            // Bookkeeping first: pop the frame with no fingerprint — writing a
            // sentinel here would corrupt red-green verification — then poison
            // and resume.  `leave_pending = true`: `vtable.leave` has not run
            // yet on this path (it sits after the finalize guard).
            // If `start_task` itself panicked the frame may never have been
            // pushed, in which case `finish_task_and_publish`'s `tn == node`
            // assert fires WHILE UNWINDING.  That second panic is caught and
            // printed rather than allowed to escape: a panic inside a panic
            // aborts the process, which would lose the poison marker and turn
            // the hang permanent-but-silent.
            let r = panic::catch_unwind(AssertUnwindSafe(|| {
                let _ = self.dep_graph.finish_task_and_publish(node_index, None);
            }));
            if let Err(p) = r {
                eprintln!(
                    "query pre-compute bookkeeping panicked while already unwinding: {}",
                    panic_message(&*p)
                );
            }
            self.abort_with_poison::<Q>(&vtable, &key, &akey, node_index, true, payload)
        }

        // ── Phase 1: compute + fingerprint.  NO dep-graph mutation. ──
        //
        // The cache value is staged HERE, before the publish below, so
        // the system invariant only ever runs in one direction: once the
        // node's green bit is observable, the cache already holds the
        // value the stored fingerprint describes.  Staging the value
        // AFTER `finish_task_and_publish` leaves a window in which
        // `probe` / `try_serve_verified` see GREEN and hand out the
        // PREVIOUS entry — fingerprint fan-out refreshes readers'
        // dirtiness but never evicts their cache entries, so a
        // concurrent getter could serve stale data while the graph
        // claimed the node fresh (d21).
        //
        // Staging first is safe because the entry is unreachable-green
        // for the whole window: this key's `Started` entry is still
        // registered (a world-mutation draining the read lock in this
        // window cannot race this compute on the cache), the node is red
        // until the phase-2 publish clears it — which is the usual red
        // bit, not a special window-closer — and a finalize panic returns
        // before `clear_dirty` — so a staged-but-unpublished value sits
        // behind a permanently red node.  `eval_always` queries skip the
        // insert (they have no cache entry to serve green), but they DO
        // get a fingerprint below: without one, `finish_task_and_publish`
        // skips the `changed` check entirely (fp = None ⇒ `if let
        // Some(fp)` never fires), so a changed eval_always value would
        // never mark its direct readers dirty and they would keep stale
        // green cache entries forever.
        //
        // For a FRESHLY allocated node the red bit is not what closes the
        // window — those nodes are green by default and stay green until
        // `verify_inner` finds something to walk, so there is a window in
        // which the node is fresh-green while a staged value is already
        // published.  What closes that case is d24 in combination with
        // `Started`: `set_finishing` (below) makes the transitive walk
        // treat the node as dirty and take the compute/wait path, the
        // still-registered `Started` entry routes any concurrent getter
        // of this exact key into `Wait`, and the probe fast path would
        // discard a staged entry whose fingerprint gate does not agree —
        // it never hands a staged-but-unpublished value back.
        // `set_finishing` is only opportunistically before the staged
        // value could become reachable; the three guards above are the
        // full set.
        // The staging itself (erase + `cache.stage`) is INSIDE the
        // guard: `cache.insert` can panic (a future cache bug inside
        // `DefaultCache::evict_one`'s eviction internals — its `expect`s
        // all sit before any mutation), and a panic there
        // must run the SAME frame-pop + `poison_active` cleanup as a
        // compute panic — otherwise the `Started` entry stays installed
        // with waiters parked on the one-shot latch forever, and this
        // thread's `TASK_STACK` keeps a stale frame.
        //
        // d24: mark the node as "finish in progress" BEFORE the staged
        // value becomes reachable in the cache.  A transitive verifier
        // that reaches this node while the flag is set treats it as
        // dirty and takes the compute/wait path instead of serving the
        // staged value — closing the residual sub-window
        // [in-section colour decision .. post-flush re-check] where a
        // late mark on a read node could otherwise be invisible to a
        // getter whose verify runs inside the window.  The flag is
        // cleared by `finish_task_and_publish`'s tail (and by every
        // panic arm below).
        // (d24's `set_finishing` ran inside the guard above, together with
        // `start_task` — see the "Unprotected-stretch guard" note.)

        let staged: Result<(Stored<'q, 'src, Q>, u64), Box<dyn std::any::Any + Send>> =
            panic::catch_unwind(AssertUnwindSafe(|| {
                let v = Q::compute(&key, provider);
                let fp = Q::Value::stable_hash(&v);
                // Erase once: the SAME handle is staged into the cache,
                // fingerprinted, and returned to the caller — no deep
                // copy of the freshly computed value on any path (one
                // `Arc` wrap on [`Owned`], identity on [`Arena`]).
                let value = erase_value::<'q, 'src, Q>(v);
                if !vtable.descriptor.eval_always {
                    // Staged while the node's finishing flag (d24) is set:
                    // the merged entry carries node + value in ONE slot.
                    vtable
                        .cache
                        .as_ref()
                        .expect("tracked query has a cache")
                        .stage(key.clone(), node_index, value.clone());
                }
                (value, fp)
            }));

        let (value, fp) = match staged {
            Ok(pair) => pair,
            Err(payload) => {
                // Compute, `stable_hash`, or staging panicked: run the
                // task bookkeeping (frame pop + rev-edge flush of the
                // partial read set) with no fingerprint — writing a
                // sentinel would corrupt red-green verification — then
                // poison and resume via the shared abort helper.  Also
                // guarded: a debug class-assertion on the partial read
                // set must not strand the waiters either.  If the panic
                // came from before the insert, nothing was staged.  If
                // it came from inside the insert itself, the fully
                // computed value IS staged, and the node is a FRESH one
                // — GREEN by default (the dirty bit starts clear and the
                // finishing flag is cleared above) — so the next
                // getter's `probe` would take the hit path and serve
                // that value without ever consulting the poisoned active
                // entry: the same failure mode the finalize-panic arm
                // guards against (d14 step 5).  The node STAYS
                // registered (trait contract 1).
                let r = panic::catch_unwind(AssertUnwindSafe(|| {
                    let _ = self.dep_graph.finish_task_and_publish(node_index, None);
                }));
                if let Err(p) = r {
                    eprintln!(
                        "query finalize bookkeeping panicked while already unwinding: {}",
                        panic_message(&*p)
                    );
                }
                self.abort_with_poison::<Q>(&vtable, &key, &akey, node_index, true, payload)
            }
        };

        // ── Phase 2: publish (guarded — d14) ──
        //
        // FUSED with the fingerprint publish: ONE critical section on
        // the node's dep-graph shard for dep-edge flush + fingerprint
        // publication (the shard striping removed the global write lock
        // that was the measured publish-path serial point).  The task is
        // popped here, so the parent task's `read` below records into
        // the right stack frame.
        //
        // Everything between the compute `catch_unwind` and the cleanup
        // below used to run UNGUARDED: a panic in `finish_task_and_publish`
        // (debug class assertion, d14) skipped the active-key cleanup and
        // left waiters parked on this job's one-shot latch FOREVER.
        // The whole step is therefore guarded, and its panic is routed
        // through the SAME poison-and-resume path as a compute panic.
        // The guard's failure must not leave the phase-1 value
        // servable: the entry is removed on the Err arm below before the
        // key is poisoned, so the next getter falls through `probe`
        // (green + empty cache) into the poisoned active entry.
        let finalize = panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = self.dep_graph.finish_task_and_publish(node_index, Some(fp));
            if self.dep_graph.current_node().is_some() {
                self.dep_graph.read(node_index);
            }
            // d21 window (green publish + staged insert → before the
            // latch signal / `vtable.leave`): a concurrent probe must see
            // the STAGED value while parked here.
            self.sched.suspend(SchedulePoint::AfterPublish {
                query: Q::descriptor().name,
                node: node_index.0,
            });
        }));

        // Deactivate: pop this key off the current thread's stack — on
        // EVERY path (success, compute panic, stable-hash panic,
        // finalize panic).  The active entry was committed in
        // `acquire_key_slot`; skipping the pop on a panicked path left a
        // stale entry that poisoned this thread's vtable state.
        vtable.leave(self.system_id(), self.epoch());

        // Signal completion or poisoning to any waiting threads.
        // Matching on `finalize` BEFORE acquiring the active shard: the
        // Err arm unstages (cache shard write lock), and cache-under-
        // active would be a new §0 edge — the other two unstage sites
        // (staging-panic arm, invalidate_quiesced) also run unstage
        // before the active acquisition.
        let value = match finalize {
            Ok(()) => {
                let _tw_active = sync_order::Acquired::acquire(
                    sync_order::Fam::ActiveShard,
                    "get_shared.finalize",
                );
                let mut active = self.active_keys[active_shard(&akey)].write();
                // Normal completion.  ORDER IS LOAD-BEARING: resolve the
                // job's latch BEFORE removing the active entry and before
                // clearing wait edges.
                //
                // `register_wait` treats a WaitEdge whose latch has already
                // resolved as absent, so from the instant `set()` runs no
                // cycle walk can follow a wait on this job no matter when
                // its edges are retracted.  In the old order
                // (remove → clear → signal) the window
                // [entry removed .. latch resolved] left a completed job's
                // edges registered with UNRESOLVED latches: a third
                // thread's walk could traverse one and — if the chain
                // looped back to it — report a spurious "query cycle
                // detected across threads" for a perfectly valid wait (its
                // own `job_still_active` gate is still true because ITS
                // owner is alive); two overlapping windows could briefly
                // form a cycle in `wait_graph`.
                //
                // A waiter that registers during the residual
                // [signal .. remove] window still finds the entry
                // (`job_still_active` → true), parks on the already-set
                // latch, and falls straight through `wait_on`.
                // `OccupiedEntry::remove` needs no second hash (a `get`
                // followed by `remove` would hash twice per completion)
                // and preserves the load-bearing order: latch resolved
                // BEFORE the entry leaves the map (see the ORDER comment
                // above).
                use std::collections::hash_map::Entry;
                match active.entry(akey.clone()) {
                    Entry::Occupied(mut o) => {
                        if let ActiveKeyStatus::Started(job) = o.get() {
                            job.latch.set();
                        }
                        if let ActiveKeyStatus::Started(job) = o.remove() {
                            // Edges for the completed job are now walker-invisible
                            // (latch resolved); this sweep just reclaims them.
                            self.clear_waiters_of_me(job.id);
                        }
                    }
                    // eval_always queries never insert — tolerate absence,
                    // same as the old get-None/remove-None path.
                    Entry::Vacant(_) => {}
                }
                drop(active);
                value
            }
            Err(payload) => {
                // The whole poison tail (d24 finishing clear → unstage →
                // active poison → resume) is `abort_with_poison`; the
                // compute frame is already popped by the deactivation
                // site above, so no `leave` is pending here.
                self.abort_with_poison::<Q>(&vtable, &key, &akey, node_index, false, payload)
            }
        };

        Ok(value)
    }

    /// The Wait branch of `get_shared_inner`'s attempt loop, extracted.
    /// (Replaces the old inline `KeySlot::Wait` branch and its "maintainer
    /// note" — see the contract below.)
    ///
    /// # Contract
    ///
    /// - **The wake-side re-resolution lives here, and only here.** `*vtable`
    ///   and `*node_index` are both reassigned by it (d18 discipline:
    ///   `clear_all` may happen while we wait, invalidating the pre-wait
    ///   snapshot), so both values are passed across the wait boundary as
    ///   `&mut` out-params and written back on the post-wake path. The
    ///   attempt loop must adopt these values before re-entering
    ///   `acquire_key_slot` — this is what prevents a post-`clear_all`
    ///   recomputation from publishing against an orphaned vtable or a
    ///   recycled node index.
    /// - The phantom paths (tentative cycle / owner left before our edge
    ///   landed) deliberately leave the out-params untouched: they never
    ///   release the quiesce read lock, so the pre-wait snapshot is still the
    ///   current world there. Only the parked path can be overtaken by
    ///   `clear_all` (parking_lot is write-preferring: once the owner releases
    ///   its read lock, the write lock is granted — before our `unpark`
    ///   re-acquires the read lock).
    /// - `'a` binds `&self`, `vtable`, and `in_flight` to one region — the
    ///   caller's world snapshot. This binding is essential: writing the
    ///   newly resolved `VtableResolve<'q, 'a, 'src, Q>` (whose `Cached` arm holds a
    ///   `Ref` borrow of `self.resolve_cache`) back through an out-param only
    ///   type-checks if it is for the same `'a`; with independent elided
    ///   lifetimes it will not compile.
    /// - `WaitOutcome::Retry` = "loop back through the attempt loop";
    ///   `WaitOutcome::Done` = "return this result as-is".
    ///
    /// If you modify this function, re-run the d18 tests (waiter parked
    /// during `clear_all`: orphaned vtable + recycled node index), plus the
    /// cross-thread waiter/cycle suite — they are the regression net for
    /// everything above.
    fn wait_for_owner<'a, Q: Query<'q, 'src>>(
        &'a self,
        vtable: &mut VtableResolve<'q, 'a, 'src, Q>,
        node_index: &mut dep_graph::DepNodeIndex,
        key: &Q::Key,
        akey: &ActiveKey,
        provider: &dyn QueryProvider<'q, 'src, S>,
        in_flight: &mut InFlightGuard<'q, 'a, 'src, S>,
        job: job::QueryJob,
    ) -> WaitOutcome<'q, 'src, Q> {
        // Register the wait edge and detect cross-thread cycles before
        // blocking. Without this, two threads waiting on each other's latch
        // would deadlock forever.
        if let Err(chain) = self.register_wait(&job) {
            // The cycle verdict is TENTATIVE: the task snapshot from
            // `acquire_key_slot` may be stale. Between reading the active
            // entry and walking the wait graph, the owner may have completed
            // the job (removing the entry) and legitimately become blocked on
            // a job owned by us — the walk then sees "the owner is waiting on
            // me", but the key is actually owned by nobody. That is a phantom
            // cycle, not a deadlock. A real cross-thread cycle requires the
            // owner to still hold this key (a thread blocked on our latch can
            // only exist while its own computation is in flight), so re-check
            // under the shard lock: if the entry is gone or replaced, retry
            // instead of failing.
            if self.job_still_active::<Q>(akey, &job) {
                // Real cross-thread cycle (not phantom).  Suspend: ferry
                // the waiter's own committed prefix across the thread
                // boundary, then resume through the query's loop
                // processor (tier-2) or legacy handler (tier-1) — local
                // recovery with a pinpointable context instead of a bare
                // abort.  No wait edge was registered (the registration
                // above was declined), so there is nothing to retract.
                let waiter_prefix = vtable.snapshot_stack(self.system_id(), self.epoch());
                let suspended = SuspendedCycle {
                    key: key.clone(),
                    waiter_prefix,
                    wait_chain: chain,
                };
                return self.resume_suspended_cycle(vtable, *node_index, suspended);
            }
            // Phantom — the owner already finished; loop back
            // (the cache now holds the value, or we recompute).
            if let Some(value) = self.serve_reprobed::<Q>(vtable, key, provider) {
                return WaitOutcome::Done(Ok(value));
            }
            return WaitOutcome::Retry;
        }
        // The owner may also have finished between our slot acquisition and
        // the edge registration (its `clear_waiters_of_me` cannot remove an
        // edge that did not yet exist). In that case we will fast-path on the
        // latch without blocking, and the edge we just inserted is a phantom
        // "blocked thread" other walkers could follow — retract it and retry.
        if !self.job_still_active::<Q>(akey, &job) {
            self.remove_wait();
            // The owner just published; its value is already in the cache —
            // serve it rather than recompute. Same validity discipline as the
            // wake path (a non-quiesced fan-out may already have marked the
            // node red between the owner's publish and this probe).
            if let Some(value) = self.serve_reprobed::<Q>(vtable, key, provider) {
                return WaitOutcome::Done(Ok(value));
            }
            return WaitOutcome::Retry;
        }
        // Release the quiesce read lock while blocked on the latch (see
        // `InFlightGuard::park_wait`): holding it across a latch wait closes
        // the three-way cycle with the invalidator's write lock.
        in_flight.park_wait();
        let wait_result = job.latch.wait_on();
        // Remove the wait edge on both the Ok and Err paths.
        self.remove_wait();
        in_flight.unpark();
        // The owner poisoned the job: the exact error the inline `?` produced.
        if let Err(()) = wait_result {
            return WaitOutcome::Done(Err(QueryError::Poisoned {
                query_name: Q::descriptor().name,
                key: format!("{:?}", key),
            }));
        }
        // ── Re-resolve against the current world ────────
        // `park_wait` released the quiesce read lock, and parking_lot is
        // write-preferring: a `clear_all` requested while we were parked is
        // granted the instant the owner releases its read lock — before our
        // `unpark` re-acquires the read lock. Everything captured before the
        // wait is now stale:
        // * `vtable` may be an ORPHANED Arc: the registry has been cleared,
        //   so its cache is invisible to the fresh world; a lookup would
        //   serve the owner's pre-clear value, which the fresh registry never
        //   saw;
        // * `node_index` is a PRE-RESET index: `next_index` restarted at 1,
        //   so the index may already have been re-allocated by a *new* node —
        //   running transitive verification (and its `verified_gen` stamp)
        //   against the recycled slot would hand the NEW node a "closure
        //   verified" stamp it never earned, and its later hits would skip
        //   closure verification entirely (d18).
        // Re-resolve both, then serve through the normal cache path. Both
        // refreshed values are written back through the out-params: the
        // attempt loop re-enters `acquire_key_slot` against the current world
        // (and publishes when computing).
        *vtable = self.resolve_vtable::<Q>();
        let (node2, cached2, fresh2) = probe(vtable, &self.dep_graph, key);
        *node_index = node2;
        if fresh2 {
            let _tw = sync_order::Acquired::acquire(
                sync_order::Fam::NodeToKey,
                "get_shared.wake_register",
            );
            self.node_to_key.insert(*node_index, Q::into_tagged(&key));
        }
        if let Some(value) = cached2 {
            // Same validity discipline as the cached branch: the value in the
            // cache may predate an invalidation that landed between the
            // owner's publish and our wake (owner-side d16 epoch restore,
            // non-quiesced fan-out). The helper records the current task's
            // read of the node on success — the read edge is recorded so
            // subsequent invalidations propagate to the outer query.
            let cache = vtable
                .cache
                .as_ref()
                .expect("untracked query reached the tracked serve path");
            if let Some(value) = self.serve_cached_verified::<Q>(cache, *node_index, Some(value), provider)
            {
                return WaitOutcome::Done(Ok(value));
            }
        }
        // Nothing cached (e.g. after `clear_all`), the node itself is red,
        // or the re-request failed: loop back and recompute (the owner's
        // active entry is gone, so the next `acquire_key_slot` assigns the
        // computation to us).
        WaitOutcome::Retry
    }

    /// Resume a suspended cross-thread cycle: run the query's tier-2 loop
    /// processor against the ferried waiter context, falling back to the
    /// tier-1 legacy handler (which now observes the waiter's prefix
    /// instead of `&[]`), and finally to a pinpointed cycle error.
    ///
    /// The recovery value is NOT staged and the dep edge is recorded on
    /// the caller's behalf — the same discipline as the same-thread
    /// path.  Runs with no query lock held, so nested gets stay safe.
    fn resume_suspended_cycle<Q: Query<'q, 'src>>(
        &self,
        vtable: &VtableResolve<'q, '_, 'src, Q>,
        node_index: dep_graph::DepNodeIndex,
        suspended: SuspendedCycle<'q, 'src, Q>,
    ) -> WaitOutcome<'q, 'src, Q> {
        if let Some(process) = vtable.cycle_processor {
            let ctx = CycleContext {
                key: suspended.key,
                stack: suspended.waiter_prefix,
                wait_chain: suspended.wait_chain,
                is_cross_thread: true,
            };
            match process(&ctx) {
                CycleDecision::Recover(recovered) => {
                    if self.dep_graph.current_node().is_some() {
                        self.dep_graph.read(node_index);
                    }
                    return WaitOutcome::Done(Ok(erase_value::<'q, 'src, Q>(recovered)));
                }
                CycleDecision::Abort { note } => {
                    return WaitOutcome::Done(Err(cross_thread_cycle_error::<Q>(
                        Q::descriptor().name,
                        &ctx.key,
                        &ctx.stack,
                        &ctx.wait_chain,
                        Some(note),
                    )));
                }
            }
        }
        if let Some(recover) = vtable.cycle_recovery {
            let recovered = recover(&suspended.key, &suspended.waiter_prefix);
            if self.dep_graph.current_node().is_some() {
                self.dep_graph.read(node_index);
            }
            return WaitOutcome::Done(Ok(erase_value::<'q, 'src, Q>(recovered)));
        }
        WaitOutcome::Done(Err(cross_thread_cycle_error::<Q>(
            Q::descriptor().name,
            &suspended.key,
            &suspended.waiter_prefix,
            &suspended.wait_chain,
            None,
        )))
    }

    /// Shared poison tail of [`Self::get_shared_inner`] — the
    /// staging-panic and finalize-panic arms are now unified under this
    /// helper; previously hand-duplicated, every future fix to this
    /// sequence had to be applied in two places, which is exactly how the
    /// d14-class bugs happened.
    ///
    /// ORDER IS LOAD-BEARING (mirror of the success path):
    ///  1. `clear_finishing` (d24) — a leaked flag would make the node
    ///     permanently unverifiable;
    ///  2. un-stage the staged value BEFORE the active shard lock —
    ///     unstage takes a cache shard write lock, and cache-under-active
    ///     would be a new §0 edge (same order as `invalidate_quiesced`).
    ///     A fresh node is GREEN by default, so a staged value left over
    ///     it would be served by the next getter's `probe` without ever
    ///     consulting the poisoned active entry (d14 step 5).  The NODE
    ///     STAYS registered (trait contract 1);
    ///  3. `vtable.leave` pops the compute frame — only when the caller
    ///     has not already done it (`leave_pending`): the finalize path
    ///     deactivates BEFORE the finalize match, the staging path
    ///     unwinds before deactivation;
    ///  4. `poison_active` — the latch is resolved while the entry is
    ///     still present, then the entry is removed and wait edges
    ///     swept;
    ///  5. resume the caller's panic.
    #[cold]
    fn abort_with_poison<Q: Query<'q, 'src>>(
        &self,
        vtable: &QueryVTable<'q, 'src, Q>,
        key: &Q::Key,
        akey: &ActiveKey,
        node_index: dep_graph::DepNodeIndex,
        leave_pending: bool,
        payload: Box<dyn std::any::Any + Send>,
    ) -> ! {
        self.dep_graph.clear_finishing(node_index);
        if !vtable.descriptor.eval_always {
            let _ = vtable
                .cache
                .as_ref()
                .expect("tracked query has a cache")
                .unstage(key);
        }
        if leave_pending {
            vtable.leave(self.system_id(), self.epoch());
        }
        let _tw =
            sync_order::Acquired::acquire(sync_order::Fam::ActiveShard, "get_shared.poison_path");
        let mut active = self.active_keys[active_shard(akey)].write();
        self.poison_active(vtable, akey, &mut active);
        drop(active);
        // Resume the panic — the compiler will abort or catch it higher up.
        panic::resume_unwind(payload)
    }

    /// [`QuerySystem::get_shared`] but handing back an owned value.
    ///
    /// Zero-regression compatibility surface: on the [`Owned`] path the
    /// stored handle is an `Arc`, so when it is uniquely referenced
    /// (cache published one handle, caller holds the other — so in
    /// practice `unwrap_or_clone` clones, keeping semantics identical to
    /// the pre-Arc behaviour), callers that only need the payload by
    /// value are unchanged.  On the [`Arena`] path this is a copy of the
    /// reference.
    pub fn get<Q: Query<'q, 'src>>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Result<Q::Value, QueryError> {
        let value = self.get_shared::<Q>(key, provider)?;
        Ok(provide_value::<Q>(&value))
    }

    /// Record that the current query task depends on the given node.
    pub fn record_dep(&self, other: dep_graph::DepNodeIndex) {
        self.dep_graph.read(other);
    }

    /// Register that the current thread is about to block on the job
    /// owned by `job.owner`, and detect cross-thread wait cycles before
    /// blocking.  The edge is registered only when no cycle is found.
    /// On a cycle, returns the wait chain (me → owner → …) for diagnostics.
    fn register_wait(&self, job: &job::QueryJob) -> Result<(), Vec<std::thread::ThreadId>> {
        let me = std::thread::current().id();
        let owner = job.owner;
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::WaitGraph, "register_wait");
        let mut wg = self.wait_graph.write();
        let mut chain = vec![me];
        let mut cur = owner;
        // Threads already visited on this walk.  The check-and-insert
        // below is atomic under `wg`, so a cycle of LIVE edges involving
        // `me` can never be present here (the thread that would close it
        // detects `cur == me` and declines to insert).  A cycle among
        // OTHER threads' edges can only be transient (a dead edge whose
        // registrant has not yet retracted it), but this walk must still
        // terminate: revisit → stop following, do NOT loop (an unbounded
        // walk could spin forever under the mutex and hang every other
        // wait-graph caller).
        let mut visited: FxHashSet<std::thread::ThreadId> = FxHashSet::default();
        loop {
            if cur == me {
                return Err(chain);
            }
            // Transient cycle not involving us: not our deadlock — stop
            // following and register.  `me`'s own liveness is still
            // guarded by the latch + `job_still_active` pair that follows.
            if !visited.insert(cur) {
                break;
            }
            match wg.get(&cur) {
                // A live edge only: the waited-on job's latch must still be
                // unresolved.  A completed/poisoned job's edge is history,
                // not a wait — skipping it makes a phantom edge invisible
                // to this walk even before its registrant retracts it.
                Some(edge) if !edge.latch.is_complete() => {
                    chain.push(cur);
                    cur = edge.owner;
                }
                _ => break,
            }
        }
        wg.insert(
            me,
            WaitEdge {
                owner,
                job_id: job.id,
                latch: job.latch.clone(),
            },
        );
        Ok(())
    }

    /// Re-check, under the owning shard's lock, that `active[akey]` still
    /// holds the *same* `Started(job)` this thread snapshotted in
    /// `acquire_key_slot`.  Used to distinguish real cross-thread wait
    /// cycles from phantom ones: the snapshot is taken under the shard
    /// lock but the cycle walk is not, so the owner may have completed the
    /// job (and moved on to wait on a job owned by us) in between.  A
    /// vanished or replaced entry means the "owner" no longer owns the
    /// key, so any cycle verdict based on it is stale.
    fn job_still_active<Q: Query<'q, 'src>>(&self, akey: &ActiveKey, job: &job::QueryJob) -> bool {
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::ActiveShard, "job_still_active");
        let active = self.active_keys[active_shard(akey)].read();
        matches!(
            active.get(akey),
            Some(ActiveKeyStatus::Started(current)) if current.id == job.id
        )
    }

    /// Remove the current thread's wait edge after the latch wait finishes.
    fn remove_wait(&self) {
        let me = std::thread::current().id();
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::WaitGraph, "remove_wait");
        self.wait_graph.write().remove(&me);
    }

    /// Remove the wait edges pointing at the CURRENT thread for the one
    /// job just completed (`completed_job_id`).  The owner calls this at
    /// completion/poison time, AFTER resolving the job's latch (see the
    /// success path in `get_shared` and `poison_active`) — the sweep only
    /// reclaims walker-invisible edges, because `register_wait` already
    /// skips edges whose latch has resolved.  Edges are only removed for
    /// THAT job: edges from threads blocked on other jobs this thread
    /// still owns (nested computes) stay in the graph, so the cycle
    /// walker keeps seeing those live waits.
    fn clear_waiters_of_me(&self, completed_job_id: job::QueryJobId) {
        let me = std::thread::current().id();
        let _tw = sync_order::Acquired::acquire(sync_order::Fam::WaitGraph, "clear_waiters_of_me");
        self.wait_graph
            .write()
            .retain(|_, edge| !(edge.owner == me && edge.job_id == completed_job_id));
    }

    /// Tear down a key's `Started` entry after a failed finalize: drop
    /// every incoming wait edge, poison the latch so no waiter parks
    /// forever, then leave a `Poisoned` marker for the next getter.
    ///
    /// The `Started` removal MUST precede the `Poisoned` insert —
    /// inserting over the entry would silently drop the job with waiters
    /// still sleeping on its latch.
    fn poison_active<Q: Query<'q, 'src>>(
        &self,
        vtable: &QueryVTable<'q, 'src, Q>,
        akey: &ActiveKey,
        active: &mut FxHashMap<ActiveKey, ActiveKeyStatus>,
    ) {
        // ORDER IS LOAD-BEARING (mirror of the success path): resolve
        // the latch while the entry is still present, so the window in
        // which the job is finished but its wait edges still look live
        // (unresolved latch) does not exist.  `register_wait` skips
        // edges whose latch has resolved, so every WaitEdge naming
        // this job is walker-invisible from the `poison()` onward.
        if let Some(ActiveKeyStatus::Started(job)) = active.get(akey) {
            job.latch.poison();
        }
        if let Some(entry) = active.remove(akey)
            && let ActiveKeyStatus::Started(job) = entry
        {
            self.clear_waiters_of_me(job.id);
        }
        if !vtable.descriptor.eval_always {
            active.insert(akey.clone(), ActiveKeyStatus::Poisoned);
        }
    }

    /// Invalidate a specific node in the dependency graph.
    ///
    /// QUIESCED (d10): an invalidation landing between an in-flight
    /// compute's input reads and its publish would be erased by the
    /// publisher's `clear_dirty`.  Routed through `mutate_world`, which
    /// takes the quiesce write lock — or DEFERS the invalidation if this
    /// is called from inside a running query (d11), draining it at the
    /// outermost guard's drop where it can neither self-deadlock nor be
    /// erased.
    pub fn invalidate_node(&self, node: dep_graph::DepNodeIndex) {
        self.mutate_world(move |system| {
            system.system.dep_graph.invalidate(node, system.proof());
        });
    }

    /// Invalidate a cache entry for a specific query key.
    pub fn invalidate<Q: Query<'q, 'src>>(&self, key: &Q::Key) {
        let key = key.clone();
        self.mutate_world(move |system| {
            system.invalidate::<Q>(&key);
        });
    }

    /// Digest-gated invalidation: invalidate `Q(key)` only when the
    /// caller-provided `input_digest` differs from the digest recorded at
    /// the previous gated call (or when none was recorded yet — the first
    /// gated call invalidates conservatively).  Returns `true` when the
    /// node was invalidated, `false` when the digest proved the content
    /// unchanged and NOTHING was touched: no red bit, no generation/`rev`
    /// bump, no cache unstage — every downstream `verified_gen`/
    /// `verified_lvl` stamp survives, so the whole dependency subtree
    /// stays served from cache.
    ///
    /// This is the zero-cost first half of the durability proposal: the
    /// caller holds a cheap equivalence proof for the node's INPUT
    /// content (e.g. a source digest), so the correct action for
    /// "content did not change" is to not invalidate at all.  Digest
    /// quality is the caller's trusted-input contract (see
    /// [`dep_graph::Durability`] for the sibling contract) — use a
    /// fingerprint-grade hash.
    ///
    /// The GATE runs synchronously so the returned bool is EXACT on every
    /// path — including the deferral case (d11): when this call is made
    /// inside a running query, `mutate_world` parks the invalidation on
    /// the target system and returns immediately, and a gate hoisted into
    /// that closure would make this function report `false` ("nothing
    /// invalidated") for an invalidation that is queued and WILL run.
    /// The gate itself only reads/writes the node's digest slot under its
    /// dep-graph shard lock — it never mutates the red-green world (no
    /// red bit, no epoch/generation/`rev` change) — so it does not need
    /// quiescence; hoisting it out is the same access pattern `probe`
    /// already uses outside any quiesced section.  The INVALIDATION still
    /// runs under `mutate_world`'s write lock exactly as before.
    ///
    /// Untracked queries: NOT applicable.  Untracked keys never own dep
    /// nodes, so the gate has nothing to compare or record against and
    /// the call would confidently report `false` ("unchanged") while
    /// doing nothing — a debug build asserts instead.  Use `clear::<Q>()`
    /// to drop an untracked memo.
    pub fn invalidate_if_changed<Q: Query<'q, 'src>>(
        &self,
        key: &Q::Key,
        input_digest: u64,
    ) -> bool {
        let key = key.clone();
        let vt = self.find_vtable::<Q>();
        // NOT a `debug_assert!`: the violation is cheap to check (the vtable is
        // already in hand) and unrecoverable either way, so an untracked query
        // must be rejected in EVERY build.  As a debug assert it silently
        // stopped firing in release and the gate returned a confident `false`
        // — the "miswritten condition never triggers" failure mode u5 exists to
        // pin down.  See `defense_tests::untracked::u5`.
        assert!(
            !vt.as_ref().is_some_and(|vt| vt.descriptor.untracked),
            "invalidate_if_changed on untracked query '{}': untracked keys never own \
             dep nodes — the gate would report 'unchanged' without doing anything. \
             Use clear::<Q>() to drop the memo.",
            Q::descriptor().name,
        );
        let changed = vt
            .and_then(|vt| vt.lookup_node(&key))
            .is_some_and(|node| self.dep_graph.digest_gate(node, input_digest));
        if changed {
            self.mutate_world(move |system| system.invalidate::<Q>(&key));
        }
        changed
    }

    /// Record the driver-provided input digest for `Q(key)` WITHOUT any
    /// comparison or invalidation — the companion of
    /// [`Self::invalidate_if_changed`] for the FIRST iteration, where the
    /// gate runs before the node exists (a cold system's `lookup_node`
    /// returns `None`, so the gate has nothing to store against and the
    /// digest would be lost).  The driver calls this AFTER the fan-out
    /// that created the nodes, so the NEXT iteration's gate can compare.
    /// No-op when the key is unregistered.
    ///
    /// Untracked queries: NOT applicable (they have no nodes — nothing
    /// to record against); the check hard-asserts in BOTH builds (see
    /// `defense_tests::untracked::u6`).
    ///
    /// QUIESCED via `mutate_world`, like every red-green-world mutation.
    pub fn note_input_digest<Q: Query<'q, 'src>>(&self, key: &Q::Key, input_digest: u64) {
        // Hard assert for the same reason as `invalidate_if_changed`'s: the
        // check is free (the vtable is already in hand) and unrecoverable, so a
        // debug-only assertion would silently stop firing in release.  See
        // `defense_tests::untracked::u6`.
        //
        // (One assert only — an identical `debug_assert!` DIRECTLY above this
        // one would be pure dead weight: `assert!` always runs, so the
        // `debug_assert!` could never observe anything the assert had not
        // already rejected.)
        let key = key.clone();
        assert!(
            !self
                .find_vtable::<Q>()
                .is_some_and(|vt| vt.descriptor.untracked),
            "note_input_digest on untracked query '{}': untracked keys never own \
             dep nodes — there is nothing to record against.",
            Q::descriptor().name,
        );
        self.mutate_world(move |system| {
            system.note_input_digest::<Q>(&key, input_digest);
        });
    }

    /// The quiesced-channel impl body of `note_input_digest` (`&Quiesced` is
    /// only a proof carrier — `record_digest` itself is not part of the
    /// red-green world; see `digest_gate`'s comment decision).
    fn note_input_digest_impl<Q: Query<'q, 'src>>(
        &self,
        key: &Q::Key,
        input_digest: u64,
        _q: &dep_graph::Quiesced,
    ) {
        if let Some(vtable) = self.find_vtable::<Q>() {
            if let Some(node) = vtable.lookup_node(key) {
                self.dep_graph.record_digest(node, input_digest);
            }
        }
    }

    /// The shared invalidate body: withdraw the staged value (node STAYS
    /// registered — trait contract 1), mark the node red, and lift a
    /// sticky poison for the key.  REQUIRES quiescence: the `&Quiesced`
    /// proof is the compile-time credential (d10) — minted by
    /// `mutate_world` / `run_deferred_batch` (via `QuiescedSystem`), or
    /// `assume_for_test` in tests with an explicit reason comment.
    fn invalidate_quiesced<Q: Query<'q, 'src>>(&self, key: &Q::Key, q: &dep_graph::Quiesced) {
        // Read-only `Arc` lookup: no registry lock.  The node map and the
        // cache both live on the vtable.
        //
        // Quiescence is the CALLER's obligation (d10): we must already run
        // under `mutate_world`'s write lock — no in-flight compute can
        // interleave, so an invalidation landing between a compute's input
        // reads and its publish is impossible, and no `cache.insert` can
        // resurrect the value we just removed.  See the public
        // [`Self::invalidate`], which wraps this in `mutate_world`.
        if let Some(vtable) = self.find_vtable::<Q>() {
            // Untracked queries keep no node, no active key, no poison:
            // the memo map is the whole served world for `Q` — drop the
            // entry only, the tracked machinery below no-ops naturally.
            if vtable.descriptor.untracked {
                let _tw = sync_order::Acquired::acquire(
                    sync_order::Fam::CacheShard,
                    "invalidate_quiesced.untracked",
                );
                vtable.untracked.write().remove(key);
                return;
            }
            // Fused (a merge win): ONE cache shard write lock both
            // withdraws the value and hands back the node — the old
            // "node-map read lock + cache write lock" pair.  The node
            // STAYS registered (trait contract 1); only its dep-graph
            // colour goes red here.
            if let Some(node) = vtable
                .cache
                .as_ref()
                .expect("tracked query has a cache")
                .unstage(key)
            {
                self.dep_graph.invalidate(node, q);
            }
            // Lift a sticky poison for this key: the red-green world
            // now says "recompute" (node red, cache empty), so the
            // active-key layer must agree instead of vetoing every
            // later get with "query previously panicked".  Only
            // `Poisoned` is removed — `Started` belongs to an
            // in-flight compute, which quiescence guarantees is
            // absent here.
            let akey = ActiveKey::of::<Q>(key);
            let _tw =
                sync_order::Acquired::acquire(sync_order::Fam::ActiveShard, "invalidate_quiesced");
            let mut active = self.active_keys[active_shard(&akey)].write();
            if matches!(active.get(&akey), Some(ActiveKeyStatus::Poisoned)) {
                active.remove(&akey);
            }
        }
    }

    /// Lift every sticky [`ActiveKeyStatus::Poisoned`] marker belonging to
    /// query type `Q`, leaving this type's `Started` entries and every
    /// other type's entries alone.  The type-scoped partner of
    /// [`Self::invalidate_quiesced`] for the paths that shake the whole
    /// served world of `Q` (`clear`) rather than a single key.
    ///
    /// REQUIRES quiescence — callers run under `mutate_world`.  The sweep
    /// takes every `active_keys` shard WRITE lock in turn, which is legal
    /// only under the quiesce write lock (the matrix root; the same
    /// pairing `invalidate_quiesced` and `clear_all_impl` already use).
    ///
    /// O(active entries) with one lock acquisition per shard.  Deliberate:
    /// `clear` is rare and quiesced, and the only handle available is a
    /// TYPE — there is no `ActiveKey` to hash, so a marker cannot be
    /// reached directly, only swept.
    ///
    /// `Started` is never touched: that entry belongs to an in-flight
    /// compute, and removing it would take its owner's completion path off
    /// the rails (`signal_complete` never runs → a one-shot-latch waiter
    /// blocks forever — the same argument as `clear_all_impl`'s ordered
    /// sweep).  Quiescence already excludes live computes; this remains
    /// belt-and-braces for a straggler that acquired a slot but has not
    /// taken its read lock yet.
    fn lift_poison_for<Q: Query<'q, 'src>>(&self) {
        // No vtable ⇒ `Q` was never driven on this system ⇒ no marker.
        let Some(vtable) = self.find_vtable::<Q>() else {
            return;
        };
        // `poison_active` installs the marker only for non-eval-always
        // queries (they have no cache entry to serve green), so there is
        // nothing to lift there.
        if vtable.descriptor.eval_always {
            return;
        }
        for shard in &self.active_keys {
            // Re-registered per shard (not one guard for the loop): the
            // §0 rows are per-acquisition, and holding all 64 "at once"
            // would read as one nested-unwritten edge in the matrix.
            let _tw =
                sync_order::Acquired::acquire(sync_order::Fam::ActiveShard, "lift_poison_for");
            let mut active = shard.write();
            active.retain(|akey, status| {
                !(akey.type_id == TypeId::of::<Q>()
                    && matches!(status, ActiveKeyStatus::Poisoned))
            });
        }
    }

    /// Drop every memoized value of `Q`.
    ///
    /// QUIESCED via [`Self::mutate_world`] — same discipline as
    /// [`Self::clear_all`] — all mutations of the red-green world must
    /// flow through `mutate_world`.  `clear`
    /// does not create a red bit, but it mutates the SERVED world (the
    /// cache): without the quiesce write lock, a compute that was in
    /// flight when `clear` ran would `cache.insert` its (pre-clear) value
    /// AFTER this function returns while its dep node is GREEN — the
    /// next `get` would take the probe fast path and serve the
    /// resurrected entry instead of recomputing, breaking clear's "next
    /// get recomputes" contract (see `d19_clear_must_quiesce_in_flight_publish`).
    /// A `clear` requested from INSIDE a running query is deferred to the
    /// outermost guard's drop (d11 semantics), like every mutation.
    ///
    /// `clear` also lifts `Q`'s sticky [`QueryError::Poisoned`] marker,
    /// for the same reason `invalidate` does: the shaken world says
    /// "recompute" (node red, cache empty), so a marker that vetoed every
    /// later `get` with "that query panicked once" would contradict it —
    /// `clear` would empty the cache and still serve `Poisoned` forever.
    pub fn clear<Q: Query<'q, 'src>>(&self) {
        // The closure captures nothing (no key, no data), so
        // `FnOnce(&Self) + Send + 'static` holds for any `Q: Query`.
        self.mutate_world(move |system| {
            if let Some(vtable) = system.find_vtable::<Q>() {
                // Untracked queries memoize in `vtable.untracked`, not in
                // the node-ful cache — clearing only the cache would
                // silently leave the memo intact and serve stale values
                // forever (the d19 "next get recomputes" contract must
                // hold for the memo-only store too).
                if vtable.descriptor.untracked {
                    vtable.untracked.write().clear();
                } else {
                    vtable
                        .cache
                        .as_ref()
                        .expect("tracked query has a cache")
                        .clear();
                }
            }
            // Poison sweep, type-scoped — see `clear`'s doc.  Sweeps
            // `active_keys` rather than the emptied cache because the
            // marker lives there, not in it.
            system.system.lift_poison_for::<Q>();
        });
    }

    pub fn clear_all(&self) {
        // Quiesced via `mutate_world` (d10/d11): the write lock lives in
        // the gate now.  An in-flight owner whose active_keys entry we
        // are about to wipe would find nothing on its completion path
        // (`active.remove(&akey)` -> None) and never call
        // `signal_complete` — every waiter parked on its latch would
        // block FOREVER (the latch is one-shot).  Holding the write lock
        // means no outer compute is in flight while we tear down (a
        // compute that starts after we hold the lock sees empty state);
        // a clear_all called from inside a running query is DEFERRED to
        // the outermost guard's drop instead of self-deadlocking.
        self.mutate_world(|system| system.clear_all());
    }

    /// The quiesced-channel impl body of `clear_all` (`&Quiesced` is the
    /// compile credential for `dep_graph.reset` — reset's privacy-confined
    /// point).
    fn clear_all_impl(&self, _q: &dep_graph::Quiesced) {
        // Register BEFORE the grant — the single rule stated in
        // `job.rs::resolve` and `vtable_or_init` ("Inc BEFORE the
        // grant"): a lock taken while the tripwire cannot see it leaves
        // a window where the InitLock row skips ORD validation.
        let _tw_init = sync_order::Acquired::acquire(sync_order::Fam::InitLock, "clear_all_impl");
        // Held so a concurrent `vtable_or_init` cannot resurrect a
        // stale vtable by storing a map cloned from the pre-clear
        // registry.
        let _init = self.init_lock.lock();
        // Rotate the world generation FIRST: any per-thread cycle
        // partition stamped with an older generation is panic-leak
        // residue (all live frames are excluded by the quiesce write
        // lock this function holds) and resets itself on next touch.
        self.epoch.fetch_add(1, Ordering::Relaxed);
        // Replacing the registry drops every per-query node map with
        // it (except for `Arc`s already held by in-flight queries).
        self.vtables.store(Arc::new(FxHashMap::default()));
        // Poison every started job BEFORE dropping its active entry:
        // with quiescence held there are normally no entries — this
        // is a belt-and-braces sweep for any straggler (e.g. a
        // thread that acquired a slot but has not yet taken the
        // quiesce read lock), so no latch waiter can be stranded by
        // the wipe below.
        //
        // NOTE (verified): the straggler this defends
        // against does not exist.  `InFlightGuard::acquire` takes the
        // quiesce read guard BEFORE `acquire_key_slot` (see the ordering
        // argument in `get_shared_inner`, and the deadlock it avoids at
        // mod.rs:4847-4849), so every `Started` job reachable here has a
        // live read guard behind it — and holding the write lock means no
        // such guard exists.  The sweep is therefore
        // UNREACHABLE-BY-CONSTRUCTION defence-in-depth, which
        // `post_reset_rebuild_racing_slow_first_registration` records
        // (`errors == 0` on every run).  If a future edit moves the read
        // guard after the slot acquisition, this stops being dead code
        // and that test's `errors` branch is the signal to add real
        // coverage for it.
        for shard in &self.active_keys {
            let _tw = sync_order::Acquired::acquire(
                sync_order::Fam::ActiveShard,
                "clear_all_impl.active",
            );
            let mut map = shard.write();
            for status in map.values_mut() {
                if let ActiveKeyStatus::Started(job) = status {
                    job.latch.poison();
                }
            }
            map.clear();
        }
        // Stale (TypeId, key) entries would make transitive green
        // verification fire bogus re-requests after re-registration
        // (index reuse maps new nodes onto old keys).
        {
            let _tw = sync_order::Acquired::acquire(
                sync_order::Fam::NodeToKey,
                "clear_all_impl.node_to_key",
            );
            self.node_to_key.clear();
        }
        {
            let _tw = sync_order::Acquired::acquire(
                sync_order::Fam::WaitGraph,
                "clear_all_impl.wait_graph",
            );
            self.wait_graph.write().clear();
        }
        self.dep_graph.reset(_q);
    }

    /// List all known query names (for debugging).
    ///
    /// Served from the static [`for_each_query!`] manifest, not from the
    /// lazy vtable registry — so this returns the full production set
    /// even before any query has run, and takes no locks (no tripwire
    /// registration needed).
    pub fn list_queries(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        macro_rules! push_name {
            ($q:ty) => {
                names.push(<$q as Query>::descriptor().name);
            };
        }
        for_each_query!(push_name);
        names
    }

    /// Execute a query in `EnsureOk` mode: compute and cache the result,
    /// but discard the value.  Returns `Err(QueryError)` if the query
    /// fails (cycle, depth limit, etc.).
    ///
    /// This is useful for validation passes where you only care whether
    /// the query completes without errors, not what the result is.
    /// Analogous to Rust's `tcx.ensure_ok().$query(..)`.
    pub fn ensure_ok<Q: Query<'q, 'src>>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Result<(), QueryError> {
        // `get_shared`, not `get`: `get` hands back an OWNED value and so
        // runs `Arc::unwrap_or_clone`, deep-cloning the payload we are
        // about to discard.  `get_shared` returns the shared handle — one
        // refcount bump on a query we already computed.
        self.get_shared::<Q>(key, provider).map(|_| ())
    }

    /// Execute multiple queries with the same key type in parallel using
    /// rayon.  Returns a vector of results in the same order as the input
    /// keys.
    ///
    /// Called from INSIDE a running query (nested), it degrades to
    /// sequential execution on the calling thread: rayon work-stealing
    /// would lose the current task's dependency edges and close a
    /// quiesce/join lock cycle (see the comment in the body).
    pub fn par_get<Q: Query<'q, 'src>>(
        &self,
        keys: &[Q::Key],
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Vec<Result<Q::Value, QueryError>>
    where
        Q::Key: Sync,
        Q::Value: Send,
    {
        // ── Nested call: degrade to sequential ─────────────────────
        // A query task is already active on THIS thread.  Rayon
        // work-stealing would steal items onto pool workers whose
        // thread-local TASK_STACK is empty, so `DepGraph::read` inside a
        // stolen item's `get_shared` is a silent no-op: the current
        // task's dependency edges to the stolen nodes are permanently
        // lost, and a later invalidation of those nodes never dirties
        // the outer query (silent staleness — the dep edge is permanently
        // lost on a stolen task, so invalidations of those nodes never
        // dirtied the outer query,  Worse, a stolen task is outermost on
        // its worker, so its `InFlightGuard` drop drains deferred
        // mutations under that worker's `quiesce.write()` while the
        // parent still holds `quiesce.read()` and blocks in the rayon
        // join — a three-way lock cycle.  Sequential execution on the
        // calling thread has neither problem: edges record into the
        // current task naturally, and reentrant mutations follow the
        // normal d11 defer path.
        if self.dep_graph.current_node().is_some() {
            return keys
                .iter()
                .map(|key| self.get::<Q>(key.clone(), provider))
                .collect();
        }
        use rayon::prelude::*;
        // Keyed dedup: each distinct key is computed exactly once and the
        // result fanned out to every input position by index.  Without this
        // a cold duplicate key parks a second rayon task on the first task's
        // latch (condvar + wait-graph traffic) and a warm one still costs a
        // redundant task; with it, duplicates share one compute and clone
        // the (already owned-value) payload per slot — identical to what a
        // cache hit does today.  `Q::Key: QueryKey` gives `Hash + Eq`, and
        // `Result<Q::Value, QueryError>` is `Clone` (`Value: Clone` per the
        // `Query` trait, `QueryError` derives `Clone`), so fan-out needs no
        // new bounds.  Order of `results` still matches `keys`.
        let mut slot_of: FxHashMap<&Q::Key, usize> = FxHashMap::default();
        let mut unique: Vec<Q::Key> = Vec::new();
        let mut slots: Vec<usize> = Vec::with_capacity(keys.len());
        for key in keys {
            let slot = *slot_of.entry(key).or_insert_with(|| {
                unique.push(key.clone());
                unique.len() - 1
            });
            slots.push(slot);
        }
        let computed: Vec<Result<Q::Value, QueryError>> = unique
            .par_iter()
            .map(|key| self.get::<Q>(key.clone(), provider))
            .collect();
        let results: Vec<Result<Q::Value, QueryError>> =
            slots.into_iter().map(|s| computed[s].clone()).collect();
        // Record the parent task's dependency on every node HERE, on the
        // calling thread, after the join — independent of which worker
        // computed what.  A no-op with no active task (the normal
        // top-level case, where `read` finds an empty stack); the exact
        // edge set the outer query needs if this system ever runs
        // par_get under a manually-driven task stack.
        let vtable = self.vtable_or_init::<Q>();
        for key in keys {
            if let Some(node) = vtable.lookup_node(key) {
                self.dep_graph.read(node);
            }
        }
        results
    }

    /// [`par_get`] for callers that want the shared handle, not an owned
    /// value.
    ///
    /// The difference is the payload: `par_get` routes through `get`,
    /// which hands back `Q::Value` and therefore runs
    /// `Arc::unwrap_or_clone` — a DEEP CLONE of the value on every slot,
    /// including cache hits, on top of the one the query already produced.
    /// For the fat values this crate caches (whole function bodies,
    /// check summaries, eval reports) that clone is the dominant cost of a
    /// fan-out.  This variant routes through [`get_shared`] instead and
    /// fans out the `Stored` handle itself, so a hit costs one refcount
    /// bump on `Owned` and a copy on `Arena`.
    ///
    /// Everything else — the nested degrade, the keyed dedup, the
    /// positional fan-out, the post-join dependency recording — is the
    /// same code for the same reasons; read `par_get`'s body for those
    /// arguments, they are not repeated here.
    ///
    /// Bound note: `Q::Value: Send + Sync` (not just `Send`) because the
    /// collected value is `Stored` — `Arc<V>` is `Send` only when `V` is
    /// both, and rayon's collect has to move it across workers.
    pub fn par_get_shared<Q: Query<'q, 'src>>(
        &self,
        keys: &[Q::Key],
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Vec<Result<Stored<'q, 'src, Q>, QueryError>>
    where
        Q::Key: Sync,
        Q::Value: Send + Sync,
    {
        // Nested: sequential on the calling thread, same argument as
        // `par_get` (work-stealing loses the current task's edges and can
        // close a quiesce/join cycle).
        if self.dep_graph.current_node().is_some() {
            return keys
                .iter()
                .map(|key| self.get_shared::<Q>(key.clone(), provider))
                .collect();
        }
        use rayon::prelude::*;
        let mut slot_of: FxHashMap<&Q::Key, usize> = FxHashMap::default();
        let mut unique: Vec<Q::Key> = Vec::new();
        let mut slots: Vec<usize> = Vec::with_capacity(keys.len());
        for key in keys {
            let slot = *slot_of.entry(key).or_insert_with(|| {
                unique.push(key.clone());
                unique.len() - 1
            });
            slots.push(slot);
        }
        let computed: Vec<Result<Stored<'q, 'src, Q>, QueryError>> = unique
            .par_iter()
            .map(|key| self.get_shared::<Q>(key.clone(), provider))
            .collect();
        let results: Vec<Result<Stored<'q, 'src, Q>, QueryError>> =
            slots.into_iter().map(|s| computed[s].clone()).collect();
        let vtable = self.vtable_or_init::<Q>();
        for key in keys {
            if let Some(node) = vtable.lookup_node(key) {
                self.dep_graph.read(node);
            }
        }
        results
    }
}

impl Default for QuerySystem<'static, 'static, NoOpScheduler> {
    fn default() -> Self {
        Self::new()
    }
}

// ── QueryProvider trait ──────────────────────────────────────────

/// Read-only compiler state that a query body may need.
///
/// `Query::compute<S>` receives only `&dyn QueryProvider<'q, 'src, S>` (pinned
/// to the query's arena lifetime), so anything a query body touches has
/// to be reachable from here.
///
/// The lifetime is a **parameter of the trait**, not elided per method, and
/// that is forced: `SymbolTable<'input>` and `TraitEnv<'input>` are
/// *invariant* in `'input` (they hold AST behind interior mutability), so a
/// `&'long SymbolTable<'input>` can never be shortened to
/// `&'short SymbolTable<'short>`.  `dyn Trait<'a>` is likewise invariant in
/// `'a`.  The only workable arrangement is to thread one lifetime through
/// unchanged — hence `'input` here, and on `QueryProvider` below.
///
/// `Any` is not an option either: it requires `'static`, and the session
/// borrows the (stack-local) parse arena.
///
/// `TypeContext` is deliberately absent as a MUTABLE handle.  Everything
/// that uses it wants `&mut TypeContext`, which is the separate sharing
/// problem the integration plan defers to Phase 2.  What IS available
/// (opt-in) is a READ-ONLY borrow via [`CompilerSessionView::ctx_sharing`]:
/// a compute derives its own worker context with `TypeContext::new_sharing`
/// — interner shared, bindings fresh — the same discipline the parallel
/// body-check pass uses.  Direct mutation of the session's context remains
/// impossible from inside a query.
pub trait CompilerSessionView<'input, 'src>: Sync {
    fn symbols(&self) -> &crate::hir::symbol::SymbolTable<'input>;
    fn trait_env(&self) -> &crate::hir::traits::TraitEnv<'input>;
    fn resolution_map(&self) -> &crate::hir::resolver::ResolutionMap<'input>;
    /// The parsed program, so a query can reach the item it is keyed by.
    fn program(&self) -> &crate::ast::Program<'input, 'src>;
    /// Read-only handle for `TypeContext::new_sharing` (opt-in; `None`
    /// unless the provider was built with
    /// `SessionQueryProvider::with_body_check`).  The returned context must
    /// only be cloned, never mutated through.
    fn ctx_sharing(&self) -> Option<&crate::hir::types::TypeContext<'input>> {
        None
    }
    /// Checker config + cross-function tables for body-check workers
    /// (opt-in, same gate as `ctx_sharing`).
    fn body_check_harness(&self) -> Option<&crate::session::BodyCheckHarness<'input>> {
        None
    }
    /// The session's allocation arena, for queries whose `Value` is a
    /// borrow (`&'q V`, [`crate::hir::query::Arena`] storage): `compute`
    /// allocates there and the cache stores the reference itself.
    ///
    /// Defaults to `None` so providers that never drive an arena query —
    /// the tests, and any query with no arena need — are unaffected.  An
    /// arena query whose provider returns `None` treats it as a
    /// programming error and panics rather than silently computing a
    /// value it cannot store.
    fn arena(&self) -> Option<&crate::arena::SyncArena<'input>> {
        None
    }
}

/// Implemented by types that can provide query computations.
///
/// Queries can call other queries through `query_get::<Q>(provider, key)`.
/// Must be `Sync` to support parallel query execution via rayon.
pub trait QueryProvider<'input, 'src: 'input, S: Scheduler>: Sync {
    fn query_system<'i>(&self) -> &QuerySystem<'input, 'src, S>;

    /// The compiler session this provider was built from, if any.
    ///
    /// Defaults to `None` so that providers wrapping a bare `QuerySystem`
    /// — the tests, and any query that needs no compiler state — are
    /// unaffected.  Queries that do need state should treat `None` as a
    /// programming error and panic with a clear message rather than
    /// silently computing something wrong.
    ///
    /// `'input` is the arena lifetime; it is a parameter of the trait
    /// rather than elided because the session's types are invariant in it
    /// (see `CompilerSessionView`).
    fn session(&self) -> Option<&dyn CompilerSessionView<'input, 'src>> {
        None
    }
}

/// Execute a query through a provider, returning the cached or computed value.
/// This is the primary way for queries to call other queries.
///
/// # Cross-system hazard
/// Nested computes follow `provider.query_system()`.  A provider bound
/// to a DIFFERENT `QuerySystem` than the one executing the outer query
/// redirects nested computes (and their cache/poison side effects)
/// into that foreign session, and the dep edges recorded for the outer
/// task then point at foreign node indices.  Only the re-request path
/// (`try_serve_verified`) is pinned to the verifying system (d20);
/// callers must keep the provider and the executing system paired.
pub fn query_get<'q, 'src, Q: Query<'q, 'src>, S: Scheduler>(
    provider: &dyn QueryProvider<'q, 'src, S>,
    key: impl IntoQueryKey<Q::Key>,
) -> Result<Q::Value, QueryError>
where
    'src: 'q,
{
    provider.query_system().get::<Q>(key, provider)
}

/// The shared-handle form of [`query_get`]: returns the cache's stored
/// handle instead of an owned value.
///
/// This is the fix for the hit-path deep-clone tax.  `get` runs the
/// [`Erasable`] load (an `Arc::unwrap_or_clone` on the [`Owned`] path),
/// and a published cache entry always has refcount ≥ 2 (the cache holds
/// one handle, the caller holds the other), so that unwrap DEEP-CLONES
/// the payload on EVERY cache hit — for compiler-sized values (HIR
/// bodies, signature facts) this is the dominant hit-path tax.  A nested
/// compute that only READS the value should call this instead: one
/// refcount bump (a copy on the [`Arena`] path) replaces the clone.
/// Keep [`query_get`] for callers that need owned data or want to mutate
/// a private copy.
pub fn query_get_shared<'q, 'src, Q: Query<'q, 'src>, S: Scheduler>(
    provider: &dyn QueryProvider<'q, 'src, S>,
    key: impl IntoQueryKey<Q::Key>,
) -> Result<Stored<'q, 'src, Q>, QueryError>
where
    'src: 'q,
{
    provider.query_system().get_shared::<Q>(key, provider)
}

// ── DefaultQueryProvider ──────────────────────────────────────────

/// A concrete provider that holds a reference to a `QuerySystem`.
pub struct DefaultQueryProvider<'src, 'a, 'q, S: Scheduler> {
    system: &'a QuerySystem<'q, 'src, S>,
}

impl<'src, 'a, 'q, S: Scheduler> DefaultQueryProvider<'src, 'a, 'q, S> {
    pub fn new(system: &'a QuerySystem<'q, 'src, S>) -> Self {
        DefaultQueryProvider { system }
    }
}

impl<'src, 'a, 'q, S: Scheduler> QueryProvider<'q, 'src, S>
    for DefaultQueryProvider<'src, 'a, 'q, S>
where
    'src: 'q,
{
    fn query_system(&self) -> &QuerySystem<'q, 'src, S> {
        self.system
    }
}

// ── Query handle ──────────────────────────────────────────────────

/// A handle that bundles a `QuerySystem` and a `QueryProvider` together.
pub struct QueryHandle<'q, 'a, 'src, S: Scheduler> {
    pub system: &'a QuerySystem<'q, 'src, S>,
    pub provider: &'a dyn QueryProvider<'q, 'src, S>,
}

impl<'q, 'src: 'q, 'a, S: Scheduler> QueryHandle<'q, 'a, 'src, S> {
    pub fn new(
        system: &'a QuerySystem<'q, 'src, S>,
        provider: &'a dyn QueryProvider<'q, 'src, S>,
    ) -> Self {
        QueryHandle { system, provider }
    }
    pub fn get<Q: Query<'q, 'src>>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
    ) -> Result<Q::Value, QueryError> {
        self.system.get::<Q>(key, self.provider)
    }
    /// Execute a query in `EnsureOk` mode: compute and cache the result,
    /// but discard the value.  Returns `Err(QueryError)` on failure.
    pub fn ensure_ok<Q: Query<'q, 'src>>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
    ) -> Result<(), QueryError> {
        self.system.ensure_ok::<Q>(key, self.provider)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    struct TestQuery;
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestKey(usize);
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestValue(String);

    impl QueryKey for TestKey {}

    impl StableHash for TestValue {
        fn stable_hash(&self) -> u64 {
            let mut h = rustc_hash::FxHasher::default();
            self.0.hash(&mut h);
            h.finish()
        }
    }

    impl<'q, 'src: 'q> Query<'q, 'src> for TestQuery {
        type Storage = Owned;
        type Key = TestKey;
        type Value = TestValue;
        type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("test", "a test query")
        }
        fn compute<S: Scheduler>(key: &TestKey, _: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
            TestValue(format!("value_{}", key.0))
        }
    }

    struct TestProvider;
    impl<'q, 'src, S: Scheduler> QueryProvider<'q, 'src, S> for TestProvider
    where
        'src: 'q,
    {
        fn query_system(&self) -> &QuerySystem<'q, 'src, S> {
            panic!("TestProvider does not have a QuerySystem")
        }
    }

    #[test]
    fn test_default_cache() {
        let cache = DefaultCache::new();
        let key = TestKey(1);
        let value = TestValue("hello".into());
        // Miss: the probe REGISTERS the node (value half empty).
        let (node, cached, fresh) = cache.probe(&key, |_| dep_graph::DepNodeIndex(7777));
        assert!(cached.is_none());
        assert!(fresh);
        assert_eq!(node, dep_graph::DepNodeIndex(7777));
        assert_eq!(cache.node_of(&key), Some(node));
        // Stage, then hit: ONE probe fetches node + value together.
        cache.stage(key.clone(), node, value.clone());
        let (node2, cached2, fresh2) = cache.probe(&key, |_| unreachable!());
        assert!(!fresh2);
        assert_eq!(node2, node, "node half survives staging");
        assert_eq!(cached2.as_ref(), Some(&value));
        // Unstage: the value goes, the node STAYS (merge contract 1).
        assert_eq!(cache.unstage(&key), Some(node));
        assert!(cache.node_of(&key).is_some(), "node survives unstage");
        let (_, cached3, _) = cache.probe(&key, |_| unreachable!());
        assert!(cached3.is_none());
    }

    #[test]
    fn test_default_cache_lru_clock() {
        // Per-shard share floors to 1 at 64 shards (2/64 → max(1)); the
        // GLOBAL cap governs eviction, so pick capacity 2: three keys
        // hashing to the SAME shard (shard assignment is key-hash
        // dependent) exceed the global budget on the third stage, letting
        // the CLOCK behavior stay deterministic under global borrowing.
        let cache = DefaultCache::with_capacity(2);
        let shard = DefaultCache::<TestKey, TestValue>::shard_of_key(&TestKey(0));
        let mut keys: Vec<TestKey> = Vec::new();
        let mut i = 0usize;
        while keys.len() < 3 {
            let k = TestKey(i);
            if DefaultCache::<TestKey, TestValue>::shard_of_key(&k) == shard {
                keys.push(k);
            }
            i += 1;
        }
        let (k1, k2, k3) = (&keys[0], &keys[1], &keys[2]);
        cache.stage(
            k1.clone(),
            dep_graph::DepNodeIndex(1),
            TestValue("a".into()),
        );
        cache.stage(
            k2.clone(),
            dep_graph::DepNodeIndex(2),
            TestValue("b".into()),
        );
        assert!(
            cache.probe(k1, |_| unreachable!()).1.is_some(),
            "k1 was staged"
        );
        cache.stage(
            k3.clone(),
            dep_graph::DepNodeIndex(3),
            TestValue("c".into()),
        );
        assert!(
            cache.probe(k1, |_| unreachable!()).1.is_some(),
            "key 1 was accessed, should survive (CLOCK second chance)"
        );
        assert!(
            cache.probe(k2, |_| unreachable!()).1.is_none(),
            "key 2 was never re-accessed, should be evicted"
        );
        assert!(
            cache.probe(k3, |_| unreachable!()).1.is_some(),
            "key 3 was just inserted"
        );
        // The merge's hard contract: eviction removed key 2's VALUE but
        // its NODE stays registered — dropping it would orphan the old
        // node's fingerprint/edges behind a stale node_to_key entry (the
        // d9 Window-3 class).
        assert_eq!(
            cache.node_of(k2),
            Some(dep_graph::DepNodeIndex(2)),
            "node half survives eviction"
        );
    }

    #[test]
    fn test_single_cache() {
        let cache = SingleCache::new();
        let (node, cached, fresh) = cache.probe(&(), |_| dep_graph::DepNodeIndex(5));
        assert!(cached.is_none());
        assert!(fresh);
        cache.stage((), node, TestValue("singleton".into()));
        let (node2, cached2, _) = cache.probe(&(), |_| unreachable!());
        assert_eq!(node2, node);
        assert_eq!(cached2.as_ref(), Some(&TestValue("singleton".into())));
        assert_eq!(cache.len(), 1);
        // Unstage: value goes, node stays.
        assert_eq!(cache.unstage(&()), Some(node));
        assert_eq!(cache.node_of(&()), Some(node));
        assert_eq!(cache.len(), 0);
    }

    /// PER-THREAD stat merge, on real OS threads — moved here from
    /// `shuttle_tests/b3.rs` (`d23_per_thread_stat_merge_exact`).
    ///
    /// Why it cannot live under shuttle: shuttle multiplexes every task onto
    /// ONE OS thread, so `STAT_SLOTS` — a plain `thread_local!` — is shared
    /// by all of them. The first task to touch an arena registers the slot
    /// for every later task (instrumented: exactly one cold registration
    /// per arena, whichever task arrived first), so the merge path itself —
    /// many per-thread slots summed by `stats()` — is never exercised and
    /// `hits == THREADS * M` degenerates into a check on a single,
    /// uncontended slot. Vacuous.
    ///
    /// Per-ARENA isolation — the contract that actually matters for the
    /// `NEXT_STAT_ARENA_ID` substitution hazard — is covered here just as
    /// well: the arena key, not the task/task-id set, decides whether B's
    /// misses leak into A, and no interleaving changes it.
    #[test]
    fn test_stat_slots_merge_exactly_per_thread() {
        const THREADS: usize = 8;
        const M: usize = 500;

        let a: &'static DefaultCache<TestKey, TestValue> = Box::leak(Box::new(DefaultCache::new()));
        let b: &'static DefaultCache<TestKey, TestValue> = Box::leak(Box::new(DefaultCache::new()));

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in 0..M {
                        a.record_hit();
                        b.record_miss();
                    }
                });
            }
        });

        let sa = a.stats();
        let sb = b.stats();
        assert_eq!(
            sa.hits,
            (THREADS * M) as u64,
            "instance A hits must merge exactly — no lost slot updates"
        );
        assert_eq!(
            sa.misses, 0,
            "instance A must not inherit instance B's misses"
        );
        assert_eq!(
            sb.misses,
            (THREADS * M) as u64,
            "instance B misses must merge exactly — no lost slot updates"
        );
        assert_eq!(sb.hits, 0, "instance B must not inherit instance A's hits");
        assert_eq!(sa.evictions, 0, "nothing was staged or evicted here");
        assert_eq!(sa.entries, 0, "no staging happened in this test");
    }

    #[test]
    fn test_def_id_cache() {
        let cache = DefIdCache::new();
        let d0 = crate::hir::types::DefId(0);
        let d1 = crate::hir::types::DefId(1);
        let (n0, cached0, fresh0) = cache.probe(&d0, |_| dep_graph::DepNodeIndex(10));
        assert!(cached0.is_none() && fresh0);
        let (n1, _, fresh1) = cache.probe(&d1, |_| dep_graph::DepNodeIndex(11));
        assert!(fresh1);
        cache.stage(d0, n0, TestValue("zero".into()));
        cache.stage(d1, n1, TestValue("one".into()));
        assert_eq!(
            cache.probe(&d0, |_| unreachable!()).1.as_ref(),
            Some(&TestValue("zero".into()))
        );
        assert_eq!(
            cache.probe(&d1, |_| unreachable!()).1.as_ref(),
            Some(&TestValue("one".into()))
        );
        // Unstage: value goes, node stays (both paths).
        assert_eq!(cache.unstage(&d1), Some(n1));
        assert_eq!(cache.node_of(&d1), Some(n1));
    }

    #[test]
    fn test_query_system_get() {
        let system = QuerySystem::new();
        struct Q;
        impl<'q, 'src: 'q> Query<'q, 'src> for Q {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("q", "")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                _: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                TestValue(format!("computed_{}", key.0))
            }
        }
        let provider = TestProvider;
        let r = system.get::<Q>(TestKey(42), &provider).unwrap();
        assert_eq!(r, TestValue("computed_42".into()));
        let cached = system.get::<Q>(TestKey(42), &provider).unwrap();
        assert_eq!(cached, TestValue("computed_42".into()));
    }

    #[test]
    fn test_query_handle() {
        let system = QuerySystem::new();
        struct Q;
        impl<'q, 'src: 'q> Query<'q, 'src> for Q {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("q", "")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                _: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                TestValue(format!("handle_{}", key.0))
            }
        }
        let provider = TestProvider;
        let handle = QueryHandle::new(&system, &provider);
        let r = handle.get::<Q>(TestKey(7)).unwrap();
        assert_eq!(r, TestValue("handle_7".into()));
    }

    /// Incremental integration test: verifies that the query system caches
    /// results, and that invalidating a node causes re-computation.
    #[test]
    fn test_incremental_invalidation() {
        let system = QuerySystem::new();
        struct Q;
        impl<'q, 'src: 'q> Query<'q, 'src> for Q {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("q", "")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                _: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                TestValue(format!("computed_{}", key.0))
            }
        }
        let provider = TestProvider;

        // First computation: should compute and cache.
        let r1 = system.get::<Q>(TestKey(1), &provider).unwrap();
        assert_eq!(r1, TestValue("computed_1".into()));

        // Second call with same key: should be cached (same value).
        let r2 = system.get::<Q>(TestKey(1), &provider).unwrap();
        assert_eq!(r2, TestValue("computed_1".into()));

        // Invalidate the node for Q(1) via the dep_graph (the node map
        // lives on Q's vtable — look the vtable up by TypeId, then the
        // real key).
        let node = system
            .find_vtable::<Q>()
            .and_then(|vtable| vtable.lookup_node(&TestKey(1)));
        if let Some(node) = node {
            system.invalidate_node(node);
        }

        // Third call: should be re-computed (same value since Q is deterministic).
        let r3 = system.get::<Q>(TestKey(1), &provider).unwrap();
        assert_eq!(r3, TestValue("computed_1".into()));
    }

    /// Digest-gated invalidation (`invalidate_if_changed`): a matching
    /// digest proves the content unchanged and must invalidate NOTHING —
    /// the next get serves the cached value (compute counter frozen,
    /// downstream stamps survive).  A changed digest invalidates.  The
    /// first gated call invalidates conservatively (no digest recorded
    /// yet — the gate has no proof the cached value matches).
    #[test]
    fn test_invalidate_if_changed_digest_gate() {
        static COMPUTES: AtomicU64 = AtomicU64::new(0);
        static INPUT: AtomicU64 = AtomicU64::new(1);
        let system = QuerySystem::new();
        struct Q;
        impl<'q, 'src: 'q> Query<'q, 'src> for Q {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("digest_gated_q", "")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                _: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                COMPUTES.fetch_add(1, Ordering::SeqCst);
                TestValue(format!(
                    "computed_{}_{}",
                    key.0,
                    INPUT.load(Ordering::SeqCst)
                ))
            }
        }
        let provider = TestProvider;

        // Warm: first get computes and caches.
        let r = system.get::<Q>(TestKey(1), &provider).unwrap();
        assert_eq!(r, TestValue("computed_1_1".into()));
        assert_eq!(COMPUTES.load(Ordering::SeqCst), 1);

        // First gated call: no digest recorded yet → conservative
        // invalidation (and the digest 42 is recorded).
        assert!(system.invalidate_if_changed::<Q>(&TestKey(1), 42));
        let r = system.get::<Q>(TestKey(1), &provider).unwrap();
        assert_eq!(r, TestValue("computed_1_1".into()));
        assert_eq!(COMPUTES.load(Ordering::SeqCst), 2);

        // Same digest: content unchanged → NO invalidation, the cached
        // value is served and nothing recomputes.
        assert!(!system.invalidate_if_changed::<Q>(&TestKey(1), 42));
        let r = system.get::<Q>(TestKey(1), &provider).unwrap();
        assert_eq!(r, TestValue("computed_1_1".into()));
        assert_eq!(COMPUTES.load(Ordering::SeqCst), 2);

        // Changed digest: invalidates; the recompute observes the new
        // input value.
        INPUT.store(2, Ordering::SeqCst);
        assert!(system.invalidate_if_changed::<Q>(&TestKey(1), 43));
        let r = system.get::<Q>(TestKey(1), &provider).unwrap();
        assert_eq!(r, TestValue("computed_1_2".into()));
        assert_eq!(COMPUTES.load(Ordering::SeqCst), 3);

        // Gate is per node: the NEW digest is now recorded — repeating it
        // skips again.
        assert!(!system.invalidate_if_changed::<Q>(&TestKey(1), 43));
        let _ = system.get::<Q>(TestKey(1), &provider).unwrap();
        assert_eq!(COMPUTES.load(Ordering::SeqCst), 3);

        // Unregistered key: nothing to invalidate, gate reports false.
        assert!(!system.invalidate_if_changed::<Q>(&TestKey(99), 42));
    }

    /// Verifies that `eval_always` queries are never cached and always
    /// re-computed, even when the same key is queried repeatedly.
    #[test]
    fn test_eval_always() {
        use std::sync::atomic::{AtomicU64, Ordering};

        static EVAL_COUNTER: AtomicU64 = AtomicU64::new(0);
        EVAL_COUNTER.store(0, Ordering::SeqCst);

        let system = QuerySystem::new();

        struct EvalAlwaysQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for EvalAlwaysQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("eval_always", "always re-computed").with_eval_always(true)
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                _: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                TestValue(format!(
                    "eval_{}_{}",
                    key.0,
                    EVAL_COUNTER.fetch_add(1, Ordering::SeqCst)
                ))
            }
        }
        let provider = TestProvider;

        // First call: compute, counter = 0.
        let r1 = system.get::<EvalAlwaysQ>(TestKey(1), &provider).unwrap();
        assert_eq!(r1, TestValue("eval_1_0".into()));

        // Second call: must ALSO compute (not cached), counter = 1.
        let r2 = system.get::<EvalAlwaysQ>(TestKey(1), &provider).unwrap();
        assert_eq!(r2, TestValue("eval_1_1".into()));
    }

    /// Verifies that `depth_limit` queries return an error when the
    /// recursive query depth exceeds `MAX_QUERY_DEPTH` (128).
    #[test]
    fn test_depth_limit() {
        let system = QuerySystem::new();

        struct DeepQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for DeepQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("deep", "depth-limited query").with_depth_limit(true)
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                if key.0 > 0 {
                    // Recursive call with decremented key.  If the depth limit
                    // is exceeded, `query_get` returns an error — we propagate
                    // it by returning a sentinel value so the test can detect it.
                    match query_get::<DeepQ, S>(provider, TestKey(key.0 - 1)) {
                        Ok(val) => return val,
                        Err(_) => return TestValue(format!("depth_limit_at_{}", key.0)),
                    }
                }
                TestValue(format!("deep_{}", key.0))
            }
        }
        let provider = DefaultQueryProvider::new(&system);
        let handle = QueryHandle::new(&system, &provider);

        // Shallow depth (5 calls, active.len() = 5 < 128): should succeed
        // and return the innermost result (key=0).
        let r = handle.get::<DeepQ>(TestKey(5)).unwrap();
        assert_eq!(r, TestValue("deep_0".into()));

        // Deep depth (200 calls, active.len() = 200 > 128): the depth limit
        // should be triggered at the 129th call (key=72).  The `compute`
        // function for key=73 catches the error and returns a sentinel value,
        // so the final result should be `"depth_limit_at_73"`.
        let r = handle.get::<DeepQ>(TestKey(200)).unwrap();
        assert_eq!(r, TestValue("depth_limit_at_73".into()));
    }

    /// Verifies that `par_get` executes multiple queries in parallel.
    #[test]
    fn test_par_get() {
        let system = QuerySystem::new();
        struct P;
        impl<'q, 'src: 'q> Query<'q, 'src> for P {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("p", "")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                _: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                TestValue(format!("par_{}", key.0))
            }
        }
        let provider = TestProvider;
        let keys: Vec<TestKey> = (0..10).map(TestKey).collect();
        let results = system.par_get::<P>(&keys, &provider);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.as_ref().unwrap(), &TestValue(format!("par_{}", i)));
        }
    }

    #[test]
    fn test_par_get_dedups_duplicate_keys() {
        static COMPUTES: AtomicUsize = AtomicUsize::new(0);
        let system = QuerySystem::new();
        struct D;
        impl<'q, 'src: 'q> Query<'q, 'src> for D {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("d", "")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                _: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                COMPUTES.fetch_add(1, Ordering::SeqCst);
                TestValue(format!("par_{}", key.0))
            }
        }
        let provider = TestProvider;
        let keys: Vec<TestKey> = [1usize, 2, 1, 3, 2, 1].into_iter().map(TestKey).collect();
        COMPUTES.store(0, Ordering::SeqCst);
        let results = system.par_get::<D>(&keys, &provider);
        assert_eq!(COMPUTES.load(Ordering::SeqCst), 3);
        for (key, result) in keys.iter().zip(results.iter()) {
            assert_eq!(
                result.as_ref().unwrap(),
                &TestValue(format!("par_{}", key.0))
            );
        }
    }

    #[test]
    fn test_active_partition_epoch_reset_and_isolation() {
        // Stale generation resets in place (panic-leak residue path).
        let mut parts = ActivePartitions::<TestQuery>::default();
        {
            let p = active_partition(&mut parts, 7, 3);
            p.stack.push(TestKey(1));
            p.set.insert(TestKey(1));
        }
        assert_eq!(parts.parts.len(), 1);
        {
            let p = active_partition(&mut parts, 7, 4);
            assert!(p.stack.is_empty() && p.set.is_empty());
        }
        // Distinct systems stay isolated: same key live in both.
        {
            let a = active_partition(&mut parts, 7, 4);
            a.stack.push(TestKey(9));
            a.set.insert(TestKey(9));
            let b = active_partition(&mut parts, 11, 4);
            assert!(!b.set.contains(&TestKey(9)));
            b.stack.push(TestKey(9));
            b.set.insert(TestKey(9));
        }
        assert_eq!(parts.parts.len(), 2);
        // Draining both pops removes both partitions.
        for sid in [7, 11] {
            let tls_len = parts.parts.len();
            let idx = parts
                .parts
                .iter()
                .position(|(s, _, _)| *s == sid)
                .expect("partition present");
            parts.parts[idx].2.stack.pop();
            parts.parts[idx].2.set.remove(&TestKey(9));
            if parts.parts[idx].2.stack.is_empty() {
                parts.parts.swap_remove(idx);
            }
            assert_eq!(parts.parts.len(), tls_len - 1);
        }
    }

    /// Registration storm: 8 threads × 8 distinct query types, all
    /// first-touching concurrently from different orders.  This is the
    /// `ArcSwap` init path (`vtable_or_init` / `init_lock`): the
    /// double-checked registration must leave exactly ONE surviving vtable
    /// per query type, and every call must get its own type's value back.
    #[test]
    fn test_registration_storm() {
        use std::sync::Barrier;

        macro_rules! storm_query {
            ($name:ident, $tag:literal) => {
                struct $name;
                impl<'q, 'src: 'q> Query<'q, 'src> for $name {
                    type Storage = Owned;
                    type Key = TestKey;
                    type Value = TestValue;
                    type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
                    fn descriptor() -> QueryDescriptor {
                        QueryDescriptor::new($tag, "storm query")
                    }
                    fn compute<S: Scheduler>(
                        key: &TestKey,
                        _: &dyn QueryProvider<'q, 'src, S>,
                    ) -> TestValue {
                        TestValue(format!("{}_{}", $tag, key.0))
                    }
                }
            };
        }
        storm_query!(S0, "s0");
        storm_query!(S1, "s1");
        storm_query!(S2, "s2");
        storm_query!(S3, "s3");
        storm_query!(S4, "s4");
        storm_query!(S5, "s5");
        storm_query!(S6, "s6");
        storm_query!(S7, "s7");

        let system: &'static QuerySystem<'static, 'static, NoOpScheduler> = Box::leak(Box::new(QuerySystem::new()));
        let provider: &'static dyn QueryProvider<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(DefaultQueryProvider::new(system)));

        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let b = barrier.clone();
                std::thread::spawn(move || {
                    b.wait();
                    // Every thread touches every type, each in a rotated
                    // order, so the registration race is genuinely racy.
                    for i in 0..8 {
                        let k = TestKey((t + i) % 8);
                        let r = match (i + t) % 8 {
                            0 => system.get::<S0>(k.clone(), provider),
                            1 => system.get::<S1>(k.clone(), provider),
                            2 => system.get::<S2>(k.clone(), provider),
                            3 => system.get::<S3>(k.clone(), provider),
                            4 => system.get::<S4>(k.clone(), provider),
                            5 => system.get::<S5>(k.clone(), provider),
                            6 => system.get::<S6>(k.clone(), provider),
                            _ => system.get::<S7>(k.clone(), provider),
                        };
                        let tag = match (i + t) % 8 {
                            0 => "s0",
                            1 => "s1",
                            2 => "s2",
                            3 => "s3",
                            4 => "s4",
                            5 => "s5",
                            6 => "s6",
                            _ => "s7",
                        };
                        assert_eq!(r.unwrap(), TestValue(format!("{}_{}", tag, k.0)));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        // Exactly one vtable per type survived the storm — no duplicates,
        // no entries lost to a concurrent store.
        assert_eq!(system.vtables.load().len(), 8);
        for i in 0..8 {
            let k = TestKey(i);
            assert_eq!(
                system.get::<S3>(k.clone(), provider).unwrap(),
                TestValue(format!("s3_{}", i))
            );
        }
    }

    /// Regression test: when thread B waits on thread A's
    /// `QueryJob` and then consumes the published value from the cache,
    /// the dependency edge MUST be recorded.  Without the fix, B's
    /// `TaskDeps` had no edge to the node it read, so invalidating
    /// that node never dirtied B's outer query — silent staleness.
    ///
    /// Structure: A requests `OuterQ` top-level and parks inside its
    /// compute; B requests `WrapperQ`, whose compute calls
    /// `query_get::<OuterQ, S>` — so B is a WAITER on A's job while a
    /// `WrapperQ` task frame is live, and `try_serve_verified`'s
    /// `dep_graph.read` lands in that frame.  (A top-level waiter has
    /// no frame, so the read is a no-op and nothing is recorded.)
    ///
    /// Step 4 then changes `INNER_INPUT` so `InnerQ` republishes with a
    /// CHANGED fingerprint — that is what fans out and marks its
    /// readers red.  `OUTER_COUNT` (not `COMPUTE_COUNT`) is asserted,
    /// because `COMPUTE_COUNT` moves on the explicit
    /// `invalidate::<InnerQ>` + `get::<InnerQ>` regardless of any edge.
    /// Finally `WrapperQ` must recompute too: its edge to `OuterQ`
    /// exists ONLY because the waiter recorded it.
    #[test]
    fn test_waiter_records_dependency() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::thread;

        static COMPUTE_COUNT: AtomicU64 = AtomicU64::new(0);
        static OUTER_COUNT: AtomicU64 = AtomicU64::new(0);
        static WRAPPER_COUNT: AtomicU64 = AtomicU64::new(0);
        /// Drives `InnerQ`'s payload, and therefore its fingerprint.
        /// It must change across step 4 or the fan-out never fires.
        static INNER_INPUT: AtomicU64 = AtomicU64::new(1);
        /// Step-3 orchestration: while set, the `OuterQ` owner parks in
        /// its compute so B is guaranteed to take the WAITER path.
        static OUTER_PARK: AtomicBool = AtomicBool::new(false);
        static OUTER_ENTERED: AtomicBool = AtomicBool::new(false);
        static OUTER_RELEASE: AtomicBool = AtomicBool::new(false);

        COMPUTE_COUNT.store(0, Ordering::SeqCst);
        OUTER_COUNT.store(0, Ordering::SeqCst);
        WRAPPER_COUNT.store(0, Ordering::SeqCst);
        INNER_INPUT.store(1, Ordering::SeqCst);
        OUTER_PARK.store(false, Ordering::SeqCst);
        OUTER_ENTERED.store(false, Ordering::SeqCst);
        OUTER_RELEASE.store(false, Ordering::SeqCst);

        // Inner query: counts recomputations; its value tracks
        // `INNER_INPUT` so a recompute can change the fingerprint.
        struct InnerQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for InnerQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("inner", "inner query")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                _: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                COMPUTE_COUNT.fetch_add(1, Ordering::SeqCst);
                TestValue(format!(
                    "inner_{}_{}",
                    key.0,
                    INNER_INPUT.load(Ordering::SeqCst)
                ))
            }
        }

        // Outer query: reads InnerQ(key).
        struct OuterQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for OuterQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("outer", "outer query")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                OUTER_COUNT.fetch_add(1, Ordering::SeqCst);
                if OUTER_PARK.load(Ordering::SeqCst) {
                    OUTER_ENTERED.store(true, Ordering::SeqCst);
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                    while !OUTER_RELEASE.load(Ordering::SeqCst) {
                        if std::time::Instant::now() > deadline {
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
                let inner = query_get::<InnerQ, S>(provider, key.clone()).unwrap();
                TestValue(format!("outer_{}", inner.0))
            }
        }

        // Wrapper: exists purely to hold a LIVE task frame while the
        // waiter consumes OuterQ's published value.
        struct WrapperQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for WrapperQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("wrapper", "holds a frame while reading outer")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                WRAPPER_COUNT.fetch_add(1, Ordering::SeqCst);
                let outer = query_get::<OuterQ, S>(provider, key.clone()).unwrap();
                TestValue(format!("wrapper_{}", outer.0))
            }
        }

        let system: &'static QuerySystem<'static, 'static, NoOpScheduler> = Box::leak(Box::new(QuerySystem::new()));
        let provider: &'static dyn QueryProvider<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(DefaultQueryProvider::new(system)));

        // Step 1: warm the cache so both InnerQ and OuterQ are cached.
        let r = system.get::<OuterQ>(TestKey(1), provider).unwrap();
        assert_eq!(r, TestValue("outer_inner_1_1".into()));
        assert_eq!(COMPUTE_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(OUTER_COUNT.load(Ordering::SeqCst), 1);

        // Step 2: invalidate both so they must recompute.
        system.invalidate::<InnerQ>(&TestKey(1));
        system.invalidate::<OuterQ>(&TestKey(1));

        // Step 3: A owns OuterQ(1) and parks inside its compute; B asks
        // for WrapperQ(1), whose compute blocks on OuterQ(1) — B is the
        // waiter, and its WrapperQ frame is the recording target.
        OUTER_PARK.store(true, Ordering::SeqCst);
        let (atx, arx) = crossbeam_channel::unbounded();
        {
            let (sys, prov) = (system, provider);
            thread::spawn(move || {
                let _ = atx.send(sys.get::<OuterQ>(TestKey(1), prov));
            });
        }
        let entered = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !OUTER_ENTERED.load(Ordering::SeqCst) {
            assert!(
                std::time::Instant::now() < entered,
                "OuterQ owner never reached its compute"
            );
            std::thread::yield_now();
        }

        let (btx, brx) = crossbeam_channel::unbounded();
        {
            let (sys, prov) = (system, provider);
            thread::spawn(move || {
                let _ = btx.send(sys.get::<WrapperQ>(TestKey(1), prov));
            });
        }
        // B must be parked on A's latch BEFORE A is released — that is
        // the exact race this test covers.
        let blocked = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while system.wait_graph.read().is_empty() {
            assert!(
                std::time::Instant::now() < blocked,
                "waiter never reached the wait path"
            );
            std::thread::yield_now();
        }
        OUTER_RELEASE.store(true, Ordering::SeqCst);
        OUTER_PARK.store(false, Ordering::SeqCst);

        let r1 = arx.recv().unwrap().unwrap();
        let r2 = brx.recv().unwrap().unwrap();
        assert_eq!(r1, TestValue("outer_inner_1_1".into()));
        assert_eq!(r2, TestValue("wrapper_outer_inner_1_1".into()));
        assert_eq!(OUTER_COUNT.load(Ordering::SeqCst), 2);

        // Step 4: change InnerQ's INPUT, invalidate it, re-request it.
        // The recompute now publishes a CHANGED fingerprint, so the
        // fan-out marks its readers — OuterQ among them — red.
        let outer_before = OUTER_COUNT.load(Ordering::SeqCst);
        let wrapper_before = WRAPPER_COUNT.load(Ordering::SeqCst);
        INNER_INPUT.store(2, Ordering::SeqCst);
        system.invalidate::<InnerQ>(&TestKey(1));
        assert_eq!(
            system.get::<InnerQ>(TestKey(1), provider).unwrap(),
            TestValue("inner_1_2".into())
        );

        // OuterQ must recompute — otherwise the fan-out never reached
        // it and the asserted edge does not exist.
        let r = system.get::<OuterQ>(TestKey(1), provider).unwrap();
        assert_eq!(r, TestValue("outer_inner_1_2".into()));
        let outer_after = OUTER_COUNT.load(Ordering::SeqCst);
        assert!(
            outer_after > outer_before,
            "outer query must recompute after the inner fingerprint changed \
             (waiter must record dep edge — without it, the outer query \
             never learns the inner dependency changed). \
             outer_before={outer_before}, outer_after={outer_after}"
        );

        // OuterQ's own republish also changed, so it fans out to ITS
        // readers.  WrapperQ is a reader ONLY because the step-3 waiter
        // recorded WrapperQ→OuterQ from inside its live frame; without
        // that edge this get serves the stale cached value instead of
        // recomputing.
        let r = system.get::<WrapperQ>(TestKey(1), provider).unwrap();
        assert_eq!(
            r,
            TestValue("wrapper_outer_inner_1_2".into()),
            "wrapper served a stale value: the waiter's dep edge is missing"
        );
        let wrapper_after = WRAPPER_COUNT.load(Ordering::SeqCst);
        assert!(
            wrapper_after > wrapper_before,
            "wrapper query must recompute after outer's fingerprint changed \
             (the waiter's read must be recorded into its live frame — \
             otherwise downstream invalidations are silently dropped). \
             wrapper_before={wrapper_before}, wrapper_after={wrapper_after}"
        );
    }

    /// (5.1) TinyLFU admission must REJECT a cold newcomer in favour of a
    /// hotter CLOCK victim: the victim is restored, the newcomer is
    /// unstaged, and `valued` / `staged_global` / the chain all come out
    /// consistent.  This is the subtlest accounting in `DefaultCache` and
    /// had no test of its own.
    ///
    /// Choreography — capacity 1 makes the GLOBAL budget the binding
    /// constraint on every third stage, and all keys must hash to ONE
    /// shard or the CLOCK sweep of a different shard picks a different
    /// victim:
    ///   1. stage H — the hot resident.
    ///   2. probe H 10× — a real sketch lead AND the CLOCK referenced bit.
    ///   3. stage F — the sweep that CONSUMES H's bit (promoting it) and
    ///      evicts cold F.  A self-victim can never gate
    ///      (`estimate(F) < estimate(F)` is false), so F just leaves, and H
    ///      is left alone at the tail with its bit clear.
    ///   4. stage N — now the sweep's victim IS H, and admission sees
    ///      `estimate(N)=1 < estimate(H)=11`: reject, restore, done.
    /// `probe(H)` is deliberately avoided between (3) and (4): it would
    /// re-set H's referenced bit and turn the (4) victim into N itself.
    #[test]
    fn test_default_cache_tinylfu_admission_rejects_cold_newcomer() {
        let cache = DefaultCache::with_capacity(1);
        // Same-shard key set (shard assignment is key-hash dependent).
        let shard = DefaultCache::<TestKey, TestValue>::shard_of_key(&TestKey(0));
        let mut keys: Vec<TestKey> = Vec::new();
        let mut i = 0usize;
        while keys.len() < 3 {
            let k = TestKey(i);
            if DefaultCache::<TestKey, TestValue>::shard_of_key(&k) == shard {
                keys.push(k);
            }
            i += 1;
        }
        let (hot, filler, cold) = (keys[0].clone(), keys[1].clone(), keys[2].clone());
        let node = |i: usize| dep_graph::DepNodeIndex(i as u32 + 1);
        // 1 + 2: the hot resident with a sketch lead and a referenced bit.
        cache.stage(hot.clone(), node(0), TestValue("hot".into()));
        for _ in 0..10 {
            assert!(
                cache.probe(&hot, |_| unreachable!()).1.is_some(),
                "H must be staged"
            );
        }
        // Precondition: the sketch must actually rank `hot` above `cold`
        // (a shared sketch row would make both estimates equal and the
        // admission gate a no-op — the test would then pass for the wrong
        // reason).
        assert!(
            cache.estimate_hash(DefaultCache::<TestKey, TestValue>::hash_key(&hot))
                > cache.estimate_hash(DefaultCache::<TestKey, TestValue>::hash_key(&cold)),
            "sketch lead missing: the TinyLFU gate cannot discriminate these keys"
        );
        assert_eq!(cache.len(), 1);
        // 3: consume H's bit, evict the cold filler.
        cache.stage(filler.clone(), node(1), TestValue("filler".into()));
        assert_eq!(cache.len(), 1, "the filler must have been evicted");
        assert!(
            cache.probe(&filler, |_| unreachable!()).1.is_none(),
            "the filler's VALUE must be gone"
        );
        assert_eq!(
            cache.node_of(&filler),
            Some(node(1)),
            "contract 1: the evicted filler's node stays registered"
        );
        // 4: the cold newcomer.  The sweep picks H; admission rejects.
        cache.stage(cold.clone(), node(2), TestValue("cold".into()));
        assert!(
            cache.probe(&hot, |_| unreachable!()).1.is_some(),
            "the hotter victim must be RESTORED by the admission gate"
        );
        assert!(
            cache.probe(&cold, |_| unreachable!()).1.is_none(),
            "the cold newcomer must be REJECTED by TinyLFU admission"
        );
        assert_eq!(
            cache.node_of(&cold),
            Some(node(2)),
            "contract 1: the rejected newcomer's node stays registered"
        );
        assert_eq!(
            cache.len(),
            1,
            "budget back at capacity after reject + restore"
        );
        assert!(
            cache.stats().evictions >= 2,
            "both sweep victims (the filler and the restored hot entry) \
             must have been counted"
        );
    }

    /// (5.2) `finish_task_and_publish` writes a fingerprint for
    /// `eval_always` precisely so a CHANGED value fans out to its DIRECT
    /// readers.  Without that write a direct reader could serve its
    /// stale-green cache entry forever (the transitive chain case is
    /// already covered by the g2 defense test; the direct-reader shape
    /// was not).
    #[test]
    fn test_eval_always_changed_value_fans_out_to_direct_reader() {
        static E_IN: AtomicU64 = AtomicU64::new(1);
        static E_RUNS: AtomicU64 = AtomicU64::new(0);
        static R_RUNS: AtomicU64 = AtomicU64::new(0);
        E_IN.store(1, Ordering::SeqCst);
        E_RUNS.store(0, Ordering::SeqCst);
        R_RUNS.store(0, Ordering::SeqCst);

        struct EvalInnerQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for EvalInnerQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("eval_inner", "eval_always leaf").with_eval_always(true)
            }
            fn compute<S: Scheduler>(_: &TestKey, _: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
                E_RUNS.fetch_add(1, Ordering::SeqCst);
                TestValue(format!("e_{}", E_IN.load(Ordering::SeqCst)))
            }
        }
        struct ReaderQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for ReaderQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("eval_reader", "reads the eval_always leaf")
            }
            fn compute<S: Scheduler>(
                key: &TestKey,
                provider: &dyn QueryProvider<'q, 'src, S>,
            ) -> TestValue {
                R_RUNS.fetch_add(1, Ordering::SeqCst);
                let e = query_get::<EvalInnerQ, S>(provider, key.clone()).unwrap();
                TestValue(format!("r_{}", e.0))
            }
        }

        let system: &'static QuerySystem<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(QuerySystem::new()));
        let provider: &'static dyn QueryProvider<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(DefaultQueryProvider::new(system)));

        let r = system.get::<ReaderQ>(TestKey(0), provider).unwrap();
        assert_eq!(r, TestValue("r_e_1".into()));
        assert_eq!(E_RUNS.load(Ordering::SeqCst), 1);
        assert_eq!(R_RUNS.load(Ordering::SeqCst), 1);

        // The leaf's input changed.  A direct get recomputes the
        // eval_always node, and the CHANGED published fingerprint must
        // fan out to the reader that recorded its dep edge during warm-up.
        E_IN.store(2, Ordering::SeqCst);
        let e = system.get::<EvalInnerQ>(TestKey(0), provider).unwrap();
        assert_eq!(e, TestValue("e_2".into()));
        assert_eq!(E_RUNS.load(Ordering::SeqCst), 2);

        let r = system.get::<ReaderQ>(TestKey(0), provider).unwrap();
        assert_eq!(r, TestValue("r_e_2".into()));
        assert_eq!(
            R_RUNS.load(Ordering::SeqCst),
            2,
            "the direct reader must recompute: the eval_always leaf's changed \
             fingerprint must fan out to it"
        );
    }

    /// (5.3, fixes 1.1) A deferred closure that reaches a captured FOREIGN
    /// `&QuerySystem` and mutates must PANIC — release builds included —
    /// not deadlock on the nested write→write.  The foreign system here
    /// has nothing parked, which is exactly the variant
    /// `run_deferred_batch`'s own batch-start check never sees.
    #[test]
    fn abba_foreign_mutate_inside_drain_panics() {
        // A nested `fn` cannot capture the test function's locals, so the
        // leaked systems live in a `OnceLock` the compute can reach (same
        // pattern as the defense tests' `PAIR`).
        static PAIR: std::sync::OnceLock<(
            &'static QuerySystem<'static, 'static, NoOpScheduler>,
            &'static QuerySystem<'static, 'static, NoOpScheduler>,
        )> = std::sync::OnceLock::new();
        let s: &'static QuerySystem<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(QuerySystem::new()));
        let t: &'static QuerySystem<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(QuerySystem::new()));
        PAIR.set((s, t)).ok();

        struct MarkerQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for MarkerQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("abba_marker", "")
            }
            fn compute<S: Scheduler>(_: &TestKey, _: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
                TestValue("m".into())
            }
        }
        struct DriverQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for DriverQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("abba_driver", "")
            }
            fn compute<S: Scheduler>(_: &TestKey, _: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
                let (s, t) = *PAIR.get().expect("systems installed");
                // Inside a running frame: the mutation is DEFERRED onto S
                // and runs at the outermost guard's drop, under S's
                // quiesce WRITE lock.
                s.mutate_world(move |_| {
                    // The d22 shape: a captured foreign alias mutating
                    // inside our own drain.  Must panic, never take T's
                    // write lock under S's.
                    t.mutate_world(|_| {});
                });
                TestValue("d".into())
            }
        }

        let provider = DefaultQueryProvider::new(s);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = s.get::<DriverQ>(TestKey(1), &provider);
        }));
        let err = result.expect_err("a foreign mutate inside the drain must panic, not deadlock");
        let msg = panic_message(err.as_ref());
        assert!(
            msg.contains(
                "mutate_world: acquiring the quiesce write lock while a \
                 world-mutation batch for a DIFFERENT system is draining"
            ),
            "unexpected panic payload: {msg}"
        );
        // The guard is RAII, so the sticky `IN_DRAIN` marker is gone: a
        // clean get on the foreign system must still work.
        let m = t.get::<MarkerQ>(TestKey(0), &provider).unwrap();
        assert_eq!(m, TestValue("m".into()));
    }

    /// (5.3, fixes 1.2) A query started from inside a drain (via a
    /// captured alias) must panic in `InFlightGuard::acquire` instead of
    /// taking the quiesce READ lock under the held WRITE lock — a
    /// parking_lot self-deadlock with no panic path at all.
    #[test]
    fn abba_query_inside_drain_panics() {
        static PAIR: std::sync::OnceLock<(
            &'static QuerySystem<'static, 'static, NoOpScheduler>,
            &'static dyn QueryProvider<'static, 'static, NoOpScheduler>,
        )> = std::sync::OnceLock::new();
        let s: &'static QuerySystem<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(QuerySystem::new()));
        let s_provider: &'static dyn QueryProvider<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(DefaultQueryProvider::new(s)));
        PAIR.set((s, s_provider)).ok();

        struct MarkerQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for MarkerQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("abba_query_marker", "")
            }
            fn compute<S: Scheduler>(_: &TestKey, _: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
                TestValue("m".into())
            }
        }
        struct DriverQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for DriverQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("abba_query_driver", "")
            }
            fn compute<S: Scheduler>(_: &TestKey, _: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
                let (s, p) = *PAIR.get().expect("pair installed");
                // Deferred to the drain; then it runs a query on the SAME
                // system through the captured alias → the acquire-time
                // probe must fire (the quiesce write lock is held).
                s.mutate_world(move |_| {
                    let _ = s.get::<MarkerQ>(TestKey(0), p);
                });
                TestValue("d".into())
            }
        }

        let provider = DefaultQueryProvider::new(s);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = s.get::<DriverQ>(TestKey(1), &provider);
        }));
        let err = result.expect_err("a query inside the drain must panic, not deadlock");
        let msg = panic_message(err.as_ref());
        assert!(
            msg.contains("deferred world-mutation batch is draining"),
            "unexpected panic payload: {msg}"
        );
    }

    /// A mutation deferred from a frame on system A to a
    /// target B that has NO live frame on this thread is only observability
    /// tonight: `note_cross_system_defer` must count it, and the closure must
    /// stay unrun until B is quiescently touched (or dropped), not be
    /// silently applied early nor silently lost.
    ///
    /// This is the shape a mismatched provider produces: the provider of A
    /// driving code that reaches a `&QuerySystem` of B.  Step 4 is the point
    /// of the test — pre-fix nothing at all reported the pairing mistake, and
    /// the invalidation looked ignored until B was dropped.
    #[test]
    fn cross_system_defer_is_counted_and_starved() {
        static RAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        static PAIR: std::sync::OnceLock<(
            &'static QuerySystem<'static, 'static, NoOpScheduler>,
            &'static QuerySystem<'static, 'static, NoOpScheduler>,
        )> = std::sync::OnceLock::new();
        let s: &'static QuerySystem<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(QuerySystem::new()));
        let t: &'static QuerySystem<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(QuerySystem::new()));
        PAIR.set((s, t)).ok();
        RAN.store(false, Ordering::SeqCst);

        struct DriverQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for DriverQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("xsys_defer_driver", "")
            }
            fn compute<S: Scheduler>(_: &TestKey, _: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
                let (_s, t) = *PAIR.get().expect("systems installed");
                // Our frame is on S; the target is T.  `mutate_world` parks
                // the closure on T's queue and only T's drop or a later
                // quiescent touch of T runs it.
                t.mutate_world(move |_| {
                    RAN.store(true, Ordering::SeqCst);
                });
                TestValue("d".into())
            }
        }
        struct FlushQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for FlushQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("xsys_defer_flush", "")
            }
            fn compute<S: Scheduler>(_: &TestKey, _: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
                TestValue("f".into())
            }
        }

        let before = deferred_cross_system_count();

        // 1) One cross-system deferral, from S's frame to T.
        let provider = DefaultQueryProvider::new(s);
        let _ = s.get::<DriverQ>(TestKey(1), &provider);

        // Relaxed: another test may bump the global too, so a delta of at
        // least one is what is asserted, never an exact count.
        assert!(
            deferred_cross_system_count() > before,
            "the cross-system deferral must be counted"
        );

        // 2) Starved: S's outermost drop drains S's queue, not T's.
        assert!(
            !RAN.load(Ordering::SeqCst),
            "the closure must stay parked until T is quiescently touched"
        );

        // 3) A quiescent touch of T runs the inline drain and the closure.
        let t_provider = DefaultQueryProvider::new(t);
        let _ = t.get::<FlushQ>(TestKey(0), &t_provider);
        assert!(
            RAN.load(Ordering::SeqCst),
            "touching T must run the parked closure"
        );
    }

    /// The cross-system DEADLOCK window: a compute holding
    /// an outer frame on A that nests a query on system B (the mismatched
    /// provider shape) must panic at acquire time rather than hang.
    ///
    /// Why it hangs if unguarded: the nested frame is outermost-FOR-B, so it
    /// takes B's read guard, and `park_wait` later releases only B's
    /// (`is_outer()` is per-system) — A's guard stays held.  A thread
    /// blocked on a quiesce lock has no latch-wait edge, so
    /// `register_wait`'s cycle walk cannot see the cycle either.  Both
    /// builds hang, and the f1 `dep_graph.read` guard never even runs
    /// (the owner is stuck earlier, in `acquire`).
    ///
    /// So the tripwire fires on the FIRST mispair, deterministically, in
    /// every build.  This is the single-thread version (no second thread
    /// needed to reach the panic): the thread already holds A's frame when
    /// it opens B's, which is precisely the precondition the losing
    /// two-thread cycle needs.
    #[test]
    fn mispaired_provider_nesting_panics_instead_of_deadlocking() {
        static PAIR: std::sync::OnceLock<(
            &'static QuerySystem<'static, 'static, NoOpScheduler>,
            &'static QuerySystem<'static, 'static, NoOpScheduler>,
        )> = std::sync::OnceLock::new();
        let s: &'static QuerySystem<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(QuerySystem::new()));
        let t: &'static QuerySystem<'static, 'static, NoOpScheduler> =
            Box::leak(Box::new(QuerySystem::new()));
        PAIR.set((s, t)).ok();

        struct InnerQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for InnerQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("mispair_inner", "")
            }
            fn compute<S: Scheduler>(_: &TestKey, _: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
                TestValue("i".into())
            }
        }
        struct OuterQ;
        impl<'q, 'src: 'q> Query<'q, 'src> for OuterQ {
            type Storage = Owned;
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, Stored<'q, 'src, Self>>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("mispair_outer", "")
            }
            fn compute<S: Scheduler>(_: &TestKey, provider: &dyn QueryProvider<'q, 'src, S>) -> TestValue {
                let (_s, t) = *PAIR.get().expect("systems installed");
                // The mismatched provider: OUR frame is on S (a read guard
                // is held), and this asks T for a value.  `acquire` must
                // trip on "this thread already holds a different system's
                // guard" before T's read lock is taken and before any wait.
                let t_provider = DefaultQueryProvider::new(t);
                let _ = t.get::<InnerQ>(TestKey(0), &t_provider);
                TestValue("o".into())
            }
        }

        let provider = DefaultQueryProvider::new(s);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = s.get::<OuterQ>(TestKey(1), &provider);
        }));
        let err = result
            .expect_err("a mispaired nested query must panic, not deadlock on the quiesce lock");
        let msg = panic_message(err.as_ref());
        assert!(
            msg.contains("already holds an InFlightGuard on a different system"),
            "unexpected panic payload: {msg}"
        );
        // The frame is RAII, so the session entry was swapped out on the
        // way out: a clean get on the other system must still work.
        let t_provider = DefaultQueryProvider::new(t);
        let i = t.get::<InnerQ>(TestKey(7), &t_provider).unwrap();
        assert_eq!(i, TestValue("i".into()));
    }
}
