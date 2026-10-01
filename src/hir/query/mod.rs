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
//! - [`QuerySystem`] — registry of all query caches, indexed by `TypeId`.
//! - [`QueryProvider`] trait — implementors provide the `compute` logic.
//!
//! ## Usage
//!
//! ```ignore
//! struct TypeOfQuery;
//! impl Query for TypeOfQuery {
//!     type Key = DefId;
//!     type Value = TypeId;
//!     type Cache = DefIdCache<TypeId>;
//!     fn compute(key: &DefId, provider: &dyn QueryProvider<'_>) -> TypeId { ... }
//!     fn descriptor() -> QueryDescriptor { ... }
//! }
//! ```

use arc_swap::ArcSwap;
use std::any::TypeId;
use std::sync::Arc;
// Compiler-internal maps use `FxHasher`, not the default SipHash: keys
// here are `TypeId`s, `DefId`s and interned symbols, never
// attacker-controlled strings, so the DoS resistance of SipHash buys
// nothing while its longer hash costs on every probe.  Same choice
// rustc makes for its own query/dep-graph maps.
use parking_lot::{Mutex, RwLock};
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::{Hash, Hasher};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

pub mod dep_graph;
pub mod job;
pub mod queries;

pub use dep_graph::ContentClass;

#[cfg(test)]
mod defense_tests;

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
enum KeySlot {
    /// This thread owns the computation — it must run `compute` and publish.
    Compute,
    /// Another thread owns the computation; wait on its job latch.
    Wait(job::QueryJob),
}

/// Maximum recursive query depth before a depth-limited query returns an error.
const MAX_QUERY_DEPTH: usize = 128;

/// Hard ceiling on nested `get_shared` frames per thread, regardless of
/// query type.  `MAX_QUERY_DEPTH` is per-(thread, query-type) and does not
/// bound a chain of DISTINCT query types (`Q1 → Q2 → …` each accumulating
/// exactly one frame per type), and every `get_shared` frame is large —
/// an unbounded cross-type chain overflows the worker stack.  This is the
/// backstop for that case: `IN_FLIGHT_TOTAL` counts every nested frame on
/// the thread across all systems, so it is the natural counter.  512
/// frames stays well inside rayon's default 2 MiB worker stack.
const MAX_GLOBAL_DEPTH: usize = 512;

/// Compute a 64-bit hash for a `(query_type, key)` pair.
/// Used to select the `active_keys` shard only: `active_shard` masks the
/// result to index into the shard array.  Map lookups inside each shard
/// use the full `ActiveKey` (exact `Eq`, keyed by `(TypeId, Q::Key)`),
/// so a hash collision can never alias two distinct keys — it only
/// routes two keys to the same shard lock, which is benign contention.
fn query_key_hash<Q: Query>(key: &Q::Key) -> u64 {
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
    fn of<Q: Query>(key: &Q::Key) -> Self {
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

/// Number of shards for each vtable's per-thread cycle/depth map
/// (`QueryVTable::active`). Power of 2: the index is a masked hash of
/// `ThreadId`. 16 is enough — each thread owns exactly one entry, so
/// contention only arises between threads colliding on a shard, and the
/// per-entry critical sections are a few pointer ops.
const ACTIVE_VTABLE_SHARDS: usize = 16;

/// Shard index for `QueryVTable::active`, hashed from the thread id so
/// park-free threads distribute across lock words.
#[inline(always)]
fn active_vtable_shard(tid: std::thread::ThreadId) -> usize {
    // `ThreadId` is `Hash`; fold through FxHasher and mask to a power of 2.
    let mut h = rustc_hash::FxHasher::default();
    std::hash::Hash::hash(&tid, &mut h);
    (h.finish() as usize) & (ACTIVE_VTABLE_SHARDS - 1)
}

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
    /// `min(declared, min over reads)` — aggregated at every finish — so
    /// a HIGH/MEDIUM declaration only buys the per-level fast path when
    /// every transitively read query is declared at least as durable.
    /// Trusted-input-only: a HIGH query whose value actually depends on
    /// LOW-durability inputs is a silent-stale-read bug of the caller's
    /// making (see [`dep_graph::Durability`]).  Defaults to LOW — every
    /// existing query keeps today's exact invalidation behavior.
    pub durability: dep_graph::Durability,
    /// Capacity of the query's memoization cache, in entries.  Defaults to
    /// [`DEFAULT_CACHE_CAPACITY`]; queries whose working set is known to be
    /// larger should raise it — evicting a green value forces a recompute
    /// even though nothing changed, so an undersized cache defeats
    /// memoization entirely (measured: a 4096-key working set against the
    /// 512 default spends most of its time recomputing).  Best-effort
    /// bound: for a finite capacity below the shard count the effective
    /// floor is one entry per shard, and `0` means unlimited.
    pub cache_capacity: usize,
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
        }
    }
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
/// one sharded map per cache holds `Key → (DepNodeIndex, Option<Arc<V>>)`,
/// so the hit path pays ONE shard read lock where it used to pay two
/// (`node_map` + cache), and the fresh-registration two-phase race lives
/// inside `probe` next to the value read.
///
/// # Hard contracts (every impl MUST hold)
///
/// 1. **The node half is never evicted and never re-created.**  `stage`/
///    `unstage`/`clear`/eviction may drop the `Arc<V>` but MUST leave the
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
///    shard lock** — a fat `Arc<V>`'s recursive `Drop` must not run while
///    other threads spin on the shard.
/// 5. **Stats are recorded by the CALLER, not by `probe`.**  `probe`
///    returns whatever the storage holds; the caller gates on
///    green/eval_always and calls `record_hit`/`record_miss` (lock-free
///    atomic bumps) only for a real green cache decision — a red node's
///    probe today touches no counters and must not touch them merged.
pub trait QueryCacheType<K, V>: std::fmt::Debug
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
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
    fn probe(
        &self,
        key: &K,
        alloc: impl FnOnce() -> dep_graph::DepNodeIndex,
    ) -> (dep_graph::DepNodeIndex, Option<Arc<V>>, bool);

    /// Publish a freshly computed value under `key` (was: `insert`).
    /// The `node` must be the one a prior `probe` returned for this key
    /// (the registration already exists).  The replaced payload, if any,
    /// is dropped after the shard lock releases.
    fn stage(&self, key: K, node: dep_graph::DepNodeIndex, v: Arc<V>);

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
/// per-vtable `node_map`'s job) and the evictable `Option<Arc<V>>` (the
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
    value: Option<Arc<V>>,
    /// CLOCK second-chance bit, set by probe hits, consumed by the sweep.
    referenced: AtomicBool,
    prev: Option<K>,
    next: Option<K>,
}

/// HashMap-backed cache with O(1) CLOCK eviction, SHARDED 16 ways and
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
/// across the 16 shards (min 1 per shard); `with_capacity(0)` = unlimited.
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
            let mut m = m.borrow_mut();
            if let Some(slot) = m.get(&self.arena_id) {
                bump(slot, f);
                return;
            }
            let slot = Arc::new(ThreadStats::default());
            self.stat_shards.lock().push(Arc::clone(&slot));
            bump(&slot, f);
            m.insert(self.arena_id, slot);
        });
    }

    /// Shard selection — the same FxHash-of-key function the old
    /// `QueryVTable::node_shard` used for the node map, so a key's lock
    /// word is stable across the merge.
    #[inline]
    fn shard_of_key(key: &K) -> usize {
        let mut h = rustc_hash::FxHasher::default();
        key.hash(&mut h);
        (h.finish() as usize) & (DEFAULT_CACHE_SHARDS - 1)
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

    /// CLOCK sweep: demote ONE staged entry to valueless — the node half
    /// STAYS (the d9 Window-3 contract) — and hand its payload to
    /// `dropped` for the post-lock drop.  Returns false only when the
    /// chain is empty.  Termination matches the old `evict_tail` exactly:
    /// a referenced entry is demoted-and-promoted (its bit is consumed by
    /// the swap), and no reader can set bits while this write lock is
    /// held, so the sweep ends within one revolution.
    fn evict_one(shard: &mut CacheShard<K, V>, dropped: &mut Vec<Arc<V>>) -> bool {
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
            // Victim: drop the VALUE, keep the node and the key.
            let old = {
                let entry = shard.map.get_mut(&key).expect("chain entry must exist");
                entry.prev = None;
                entry.next = None;
                entry.value.take()
            };
            Self::unlink_entry(shard, &key, prev, next);
            shard.valued -= 1;
            if let Some(old) = old {
                dropped.push(old);
            }
            return true;
        }
        false
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
    pub fn lookup(&self, key: &K) -> Option<Arc<V>> {
        let shard = self.shards[Self::shard_of_key(key)].read();
        let entry = shard.map.get(key)?;
        if let Some(v) = &entry.value {
            entry.referenced.store(true, Ordering::Relaxed);
            return Some(Arc::clone(v));
        }
        None
    }

    /// See [`DefaultCache::lookup`] — the node-less `insert` twin of
    /// `stage`: insert-or-update with the node half pinned to the
    /// sentinel.  Replaced payloads drop after the shard lock releases
    /// (the same contract-4 discipline).
    pub fn insert(&self, key: K, value: Arc<V>) {
        let s = Self::shard_of_key(&key);
        let mut dropped: Vec<Arc<V>> = Vec::new();
        {
            let mut shard = self.shards[s].write();
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
            // Evict while over the shard's share AND the GLOBAL budget is
            // exceeded (hot shard borrows idle shards' share).  The
            // `per_shard_capacity > 0` guard must stay: with
            // `with_capacity(0)` (unlimited, capacity 0) it is what stops
            // `> self.capacity` from becoming `> 0` and emptying the
            // cache.  Only VALUES leave — nodes and keys stay registered
            // (trait contract 1).
            while self.per_shard_capacity > 0
                && shard.valued > self.per_shard_capacity
                && self.staged_global.load(Ordering::Relaxed) > self.capacity
            {
                if !Self::evict_one(&mut shard, &mut dropped) {
                    break;
                }
                self.staged_global.fetch_sub(1, Ordering::Relaxed);
                self.stat_bump(StatField::Eviction);
            }
        } // shard lock released — payload Drops run from here on
        drop(dropped);
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
        alloc: impl FnOnce() -> dep_graph::DepNodeIndex,
    ) -> (dep_graph::DepNodeIndex, Option<Arc<V>>, bool) {
        let s = Self::shard_of_key(key);
        // Phase 1: ONE read lock fetches node + value together — the whole
        // hit path.  The `referenced` bit preserves CLOCK recency exactly
        // like the old `lookup` did; stats are the CALLER's job (trait
        // contract 5 — a red node's probe touches no counters).
        {
            let shard = self.shards[s].read();
            if let Some(entry) = shard.map.get(key) {
                if let Some(v) = &entry.value {
                    entry.referenced.store(true, Ordering::Relaxed);
                    return (entry.node, Some(Arc::clone(v)), false);
                }
                // Registered node, no staged value: fresh registration,
                // unstaged, or evicted — the same answer the old
                // (node_map hit + cache miss) pair produced.
                return (entry.node, None, false);
            }
        }
        // Phase 2: allocate OUTSIDE any cache lock (trait contract 3 — the
        // dep-graph shard lock must never nest inside a cache shard lock),
        // then re-check under the write lock.
        let fresh_node = alloc();
        let mut shard = self.shards[s].write();
        if let Some(entry) = shard.map.get(key) {
            // Race loser: the winner registered first.  Discard our
            // allocated index (an orphan node — the same harmless class
            // as the pre-merge two-phase loser) and return the winner's
            // entry.
            if let Some(v) = &entry.value {
                entry.referenced.store(true, Ordering::Relaxed);
                return (entry.node, Some(Arc::clone(v)), false);
            }
            return (entry.node, None, false);
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
        (fresh_node, None, true)
    }

    fn stage(&self, key: K, node: dep_graph::DepNodeIndex, v: Arc<V>) {
        let s = Self::shard_of_key(&key);
        let mut dropped: Vec<Arc<V>> = Vec::new();
        {
            let mut shard = self.shards[s].write();
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
            // Evict while over the shard's share AND the GLOBAL budget is
            // exceeded (hot shard borrows idle shards' share); the
            // `per_shard_capacity > 0` guard keeps `with_capacity(0)`
            // (unlimited) from emptying the cache — see `insert`.  Only
            // VALUES leave — nodes and keys stay registered (trait
            // contract 1).
            while self.per_shard_capacity > 0
                && shard.valued > self.per_shard_capacity
                && self.staged_global.load(Ordering::Relaxed) > self.capacity
            {
                if !Self::evict_one(&mut shard, &mut dropped) {
                    break;
                }
                self.staged_global.fetch_sub(1, Ordering::Relaxed);
                self.stat_bump(StatField::Eviction);
            }
        } // shard lock released — payload Drops run from here on
        drop(dropped);
    }

    fn unstage(&self, key: &K) -> Option<dep_graph::DepNodeIndex> {
        let s = Self::shard_of_key(key);
        let mut dropped: Option<Arc<V>> = None;
        let node = {
            let mut shard = self.shards[s].write();
            let entry = shard.map.get_mut(key)?;
            let node = entry.node;
            if let Some(old) = entry.value.take() {
                dropped = Some(old);
                let (prev, next) = (entry.prev.take(), entry.next.take());
                Self::unlink_entry(&mut shard, key, prev, next);
                shard.valued -= 1;
                self.staged_global.fetch_sub(1, Ordering::Relaxed);
            }
            node
        };
        // Post-lock drop of the withdrawn payload (trait contract 4).
        drop(dropped);
        Some(node)
    }

    fn node_of(&self, key: &K) -> Option<dep_graph::DepNodeIndex> {
        let shard = self.shards[Self::shard_of_key(key)].read();
        shard.map.get(key).map(|e| e.node)
    }

    fn clear(&self) {
        // Values go, nodes stay (trait contract 1): the next get
        // RECOMPUTES against the surviving node (d19) instead of
        // re-allocating one.  One shard lock at a time
        // (DepGraph::reset discipline); payloads drop after ALL locks
        // release.
        let mut dropped: Vec<Arc<V>> = Vec::new();
        for shard_lock in &self.shards {
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
        self.shards.iter().map(|s| s.read().valued).sum()
    }

    fn for_each(&self, mut f: impl FnMut(&K, &V)) {
        for shard_lock in &self.shards {
            // One shard lock at a time.
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
        // Entries FIRST (shard read locks) — the lock-order rule on
        // `stat_bump_cold`: never take a shard lock while holding the
        // stat-registry lock.
        let entries = self.len();
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
    value: Option<Arc<V>>,
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
        alloc: impl FnOnce() -> dep_graph::DepNodeIndex,
    ) -> (dep_graph::DepNodeIndex, Option<Arc<V>>, bool) {
        {
            let entry = self.entry.read();
            if let Some(e) = entry.as_ref() {
                return (e.node, e.value.clone(), false);
            }
        }
        // Fresh registration two-phase: allocate the node OUTSIDE the
        // lock (trait contract 3), then re-check under the write lock.
        let fresh_node = alloc();
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
    fn stage(&self, _key: (), node: dep_graph::DepNodeIndex, v: Arc<V>) {
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
        let mut old: Option<Arc<V>> = None;
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
        self.entry.read().as_ref().map(|e| e.node)
    }
    fn clear(&self) {
        // Value goes, node STAYS (trait contract 1): the next get
        // recomputes against the surviving node instead of re-allocating.
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
        self.entry
            .read()
            .as_ref()
            .map_or(0, |e| usize::from(e.value.is_some()))
    }
    fn for_each(&self, mut f: impl FnMut(&(), &V)) {
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
    value: Option<Arc<V>>,
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
        alloc: impl FnOnce() -> dep_graph::DepNodeIndex,
    ) -> (dep_graph::DepNodeIndex, Option<Arc<V>>, bool) {
        let idx = key.0;
        if idx < DEFID_DENSE_LIMIT {
            // Phase 1: read lock.
            {
                let entries = self.entries.read();
                if let Some(entry) = entries.get(idx).and_then(|e| e.as_ref()) {
                    return (entry.node, entry.value.clone(), false);
                }
            }
            // Phase 2: allocate OUTSIDE the lock (trait contract 3), then
            // re-check under the write lock.
            let fresh_node = alloc();
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
            let overflow = self.overflow.read();
            if let Some(entry) = overflow.get(&idx) {
                return (entry.node, entry.value.clone(), false);
            }
        }
        let fresh_node = alloc();
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

    fn stage(&self, key: crate::hir::types::DefId, node: dep_graph::DepNodeIndex, v: Arc<V>) {
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
                    overflow.insert(idx, DefIdEntry { node, value: Some(v) });
                    None
                }
            }
        };
        // Post-lock drop of the replaced payload (trait contract 4).
        drop(old);
    }

    fn unstage(&self, key: &crate::hir::types::DefId) -> Option<dep_graph::DepNodeIndex> {
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
        let dense_dropped: Vec<Arc<V>> = {
            let mut entries = self.entries.write();
            entries
                .iter_mut()
                .filter_map(|slot| slot.as_mut().and_then(|e| e.value.take()))
                .collect()
        };
        drop(dense_dropped);
        let over_dropped: Vec<Arc<V>> = {
            let mut overflow = self.overflow.write();
            overflow
                .values_mut()
                .filter_map(|e| e.value.take())
                .collect()
        };
        drop(over_dropped);
    }

    fn len(&self) -> usize {
        let dense = self
            .entries
            .read()
            .iter()
            .filter(|e| matches!(e, Some(s) if s.value.is_some()))
            .count();
        dense + self
            .overflow
            .read()
            .values()
            .filter(|e| e.value.is_some())
            .count()
    }

    fn for_each(&self, mut f: impl FnMut(&crate::hir::types::DefId, &V)) {
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

/// Defines a compiler query: a mapping from `Key → Value` that is
/// computed on demand and cached for subsequent lookups.
///
/// Each query selects its own `Cache` type:
/// - `DefaultCache<K, V>` — HashMap-backed, for most keys
/// - `SingleCache<V>` — for unit keys `()`
/// - `DefIdCache<V>` — Vec-indexed, for `DefId` keys
///
/// Analogous to Rust's `QueryConfig` + `QueryKey::Cache`.
pub trait Query: Sized + 'static {
    type Key: QueryKey;
    type Value: Clone + std::fmt::Debug + Send + Sync + 'static + StableHash;
    /// The cache implementation for this query's key type.
    type Cache: QueryCacheType<Self::Key, Self::Value> + Default + Send + Sync;

    fn descriptor() -> QueryDescriptor;
    fn compute(key: &Self::Key, provider: &dyn QueryProvider<'_>) -> Self::Value;
}

impl QueryKey for crate::hir::types::DefId {}

// ── QueryVTable ─────────────────────────────────────────────────

/// One thread's cycle/depth state for a single query type.
///
/// `set` mirrors `stack` so that cycle detection is O(1)
/// instead of an O(depth) scan.  Keeping both behind ONE `ThreadId`
/// key means `enter`/`leave` do a single hash probe instead of two, and
/// the stack and its set are co-located — one bucket lookup fetches
/// everything the check needs (Perf Book §8.1).
#[derive(Debug)]
struct ThreadActiveState<Q: Query> {
    stack: Vec<Q::Key>,
    set: rustc_hash::FxHashSet<Q::Key>,
}

/// Build the "cycle detected" error.  `#[cold]` keeps the formatting
/// (and its allocations) out of `validate_enter`'s hot body, so the
/// common path stays compact and the branch settles as never-taken
/// (Perf Book §11.3).
#[cold]
fn cycle_error<Q: Query>(name: &'static str, key: &Q::Key, stack: &[Q::Key]) -> QueryCycleError {
    let mut keys: Vec<String> = stack.iter().map(|k| format!("{:?}", k)).collect();
    keys.push(format!("{:?}", key));
    let rendered = keys
        .iter()
        .enumerate()
        .map(|(i, s)| format!("  {}: {}", i, s))
        .collect::<Vec<_>>()
        .join("\n    ");
    QueryCycleError {
        query_name: name,
        message: format!(
            "query cycle detected: `{}` with key `{:?}`\n  call stack:\n    {}",
            name, key, rendered,
        ),
        stack: keys,
    }
}

/// Build the "depth limit exceeded" error.  `#[cold]` for the same
/// reason as [`cycle_error`].
#[cold]
fn depth_error<Q: Query>(name: &'static str, stack: &[Q::Key], max: usize) -> QueryCycleError {
    let keys: Vec<String> = stack.iter().map(|k| format!("{:?}", k)).collect();
    let rendered = keys
        .iter()
        .enumerate()
        .map(|(i, s)| format!("  {}: {}", i, s))
        .collect::<Vec<_>>()
        .join("\n    ");
    QueryCycleError {
        query_name: name,
        message: format!(
            "query depth limit exceeded: `{}` depth {} >= {}\n  call stack:\n    {}",
            name,
            stack.len() + 1,
            max,
            rendered,
        ),
        stack: keys,
    }
}

/// Type-erased trait for executing a query given a type-erased key.
/// Used for transitive green verification to re-request dirty dependencies
/// of any query type.
trait QueryVTableDyn: Send + Sync + std::any::Any {
    /// Execute the query for the given type-erased key ON `system` — the
    /// system whose dep graph the caller's transitive verification
    /// walked.  The provider only supplies compute inputs; a provider
    /// bound to a different `QuerySystem` must never redirect the
    /// re-request (and its cache/poison side effects) into a foreign
    /// session whose dep graph was walked. Redirecting into a foreign
    /// session would leak cache/poison side effects across query systems.
    fn get_shared_erased(
        &self,
        system: &QuerySystem,
        key: &dyn std::any::Any,
        provider: &dyn QueryProvider<'_>,
    ) -> Result<Arc<dyn std::any::Any + Send + Sync>, QueryCycleError>;
    fn as_any(&self) -> &dyn std::any::Any;
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
pub(crate) struct QueryVTable<Q: Query> {
    pub descriptor: QueryDescriptor,
    pub cache: Q::Cache,
    /// Per-thread cycle/depth state.  Keyed by `ThreadId` so that
    /// parallel executions on different threads never share cycle/depth
    /// state: depth accounting and cycle diagnostics only ever see the
    /// current thread's keys.
    ///
    /// Sharded by `ThreadId` hash (16 padded mutexes) so that `par_get`
    /// workers running DIFFERENT query types don't contend on one lock
    /// word: each thread only ever reads/writes its own entry, and two
    /// threads hashing to different shards never serialize at all.  The
    /// `active_keys`/`node_map` striping pattern, applied to the last
    /// per-vtable shared lock on the compute path (enter is 2 probes:
    /// validate + commit; leave is 1).
    active: [CachePadded<Mutex<FxHashMap<std::thread::ThreadId, ThreadActiveState<Q>>>>;
        ACTIVE_VTABLE_SHARDS],
}

impl<Q: Query> QueryVTableDyn for QueryVTable<Q> {
    fn get_shared_erased(
        &self,
        system: &QuerySystem,
        key: &dyn std::any::Any,
        provider: &dyn QueryProvider<'_>,
    ) -> Result<Arc<dyn std::any::Any + Send + Sync>, QueryCycleError> {
        // Downcast the key to Q::Key.
        let key = key
            .downcast_ref::<Q::Key>()
            .ok_or_else(|| QueryCycleError {
                query_name: Q::descriptor().name,
                message: format!(
                    "type mismatch in transitive verification: expected {:?}, got {:?}",
                    std::any::TypeId::of::<Q::Key>(),
                    key.type_id()
                ),
                stack: vec![],
            })?;
        // Execute on `system` — the system whose dep graph the caller's
        // verification walked — NOT on `provider.query_system()`: the
        // provider only supplies compute inputs, and a provider bound to
        // a different session must never receive this session's
        // recomputes — that would leak cache/poison side effects into
        // the foreign system.  Still goes through the full get_shared
        // logic including waiters.
        let value = system.get_shared::<Q>(key.clone(), provider)?;
        Ok(value)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<Q: Query> QueryVTable<Q> {
    pub fn new() -> Self {
        QueryVTable {
            descriptor: Q::descriptor(),
            cache: Q::Cache::from_capacity(Q::descriptor().cache_capacity),
            active: std::array::from_fn(|_| CachePadded(Mutex::new(FxHashMap::default()))),
        }
    }

    /// Look up the stable `DepNodeIndex` for `key`, if one exists.
    /// One cache shard read-lock; no system-wide lock; NEVER registers.
    /// Routed to the merged cache's `node_of` — the node registry now
    /// lives inside `Q::Cache` (one map holds node + value per key).
    #[inline]
    fn lookup_node(&self, key: &Q::Key) -> Option<dep_graph::DepNodeIndex> {
        self.cache.node_of(key)
    }

    /// Read-only cycle and depth validation against the current thread's
    /// active stack.  Does NOT mutate the stack — use this before deciding
    /// whether the current thread will actually compute the query.
    #[inline]
    fn validate_enter(&self, key: &Q::Key) -> Result<(), QueryCycleError> {
        let tid = std::thread::current().id();

        // Global cross-type depth backstop: `IN_FLIGHT_TOTAL` counts every
        // nested `get_shared` frame on this thread, across ALL systems
        // (incremented in `InFlightGuard::acquire`, which runs BEFORE this
        // check, so the outermost frame sees 1), while the per-type check
        // below only bounds THIS query type's stack.  A chain of distinct
        // query types — possibly spread across systems via a mispaired
        // provider — accumulates one frame per type and would otherwise be
        // unbounded.
        if IN_FLIGHT_TOTAL.with(|d| d.get()) > MAX_GLOBAL_DEPTH {
            return Err(depth_error::<Q>(
                self.descriptor.name,
                // The per-type stack is the wrong diagnostic here; report
                // the single global count as the "stack" summary.
                &[],
                MAX_GLOBAL_DEPTH,
            ));
        }

        let active = self.active[active_vtable_shard(tid)].lock();
        // One probe, then: O(1) cycle check and the depth
        // check both read the same state.
        if let Some(state) = active.get(&tid) {
            if state.set.contains(key) {
                return Err(cycle_error::<Q>(self.descriptor.name, key, &state.stack));
            }
            if self.descriptor.depth_limit && state.stack.len() >= MAX_QUERY_DEPTH {
                return Err(depth_error::<Q>(
                    self.descriptor.name,
                    &state.stack,
                    MAX_QUERY_DEPTH,
                ));
            }
        }
        Ok(())
    }

    /// Push `key` onto the current thread's active stack.  Only called when
    /// this thread is the one computing the query.
    fn commit_enter(&self, key: &Q::Key) {
        let tid = std::thread::current().id();
        let mut active = self.active[active_vtable_shard(tid)].lock();
        let state = active.entry(tid).or_insert_with(|| ThreadActiveState {
            stack: Vec::new(),
            set: rustc_hash::FxHashSet::default(),
        });
        state.stack.push(key.clone());
        state.set.insert(key.clone());
    }

    /// Pop the current thread's key off its active stack.  When the stack
    /// becomes empty, the thread's entry is removed entirely.
    pub fn leave(&self) {
        let tid = std::thread::current().id();
        let mut active = self.active[active_vtable_shard(tid)].lock();
        if let Some(state) = active.get_mut(&tid) {
            if let Some(key) = state.stack.pop() {
                state.set.remove(&key);
            }
            if state.stack.is_empty() {
                active.remove(&tid);
            }
        }
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
fn probe<Q: Query>(
    vtable: &QueryVTable<Q>,
    dep_graph: &dep_graph::DepGraph,
    key: &Q::Key,
) -> (dep_graph::DepNodeIndex, Option<Arc<Q::Value>>, bool) {
    let (node, cached, fresh) = vtable.cache.probe(key, || {
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
            Some(value) => {
                vtable.cache.record_hit();
                return (node, Some(value), false);
            }
            None => vtable.cache.record_miss(),
        }
    }
    // Red node, eval_always, or a fresh registration (which cannot have
    // a cached value) — `None` is exact here.
    (node, None, fresh)
}

// ── QuerySystem ──────────────────────────────────────────────────

/// Error returned when a query cycle is detected.
#[derive(Debug, Clone)]
#[must_use]
pub struct QueryCycleError {
    pub query_name: &'static str,
    /// Human-readable error message (includes the full call stack).
    pub message: String,
    /// The sequence of keys in the active call stack at the time of the
    /// cycle, from outermost to innermost.  The last entry is the key
    /// that was repeated (the cycle).  Useful for diagnostics and debugging.
    pub stack: Vec<String>,
}

impl std::fmt::Display for QueryCycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Registry of all query caches, indexed by `TypeId`.
///
/// Stores a `QueryVTable<Q>` for each query type, bundling metadata,
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
struct InFlightGuard<'a> {
    _read: Option<parking_lot::RwLockReadGuard<'a, ()>>,
    /// True while the read lock was voluntarily released for a latch
    /// wait (`park_wait`) and must be re-taken by `unpark`.
    parked: bool,
    /// The owning system — needed by `unpark` (re-take the read lock)
    /// and by the outermost drop (drain deferred mutations, d11).
    system: &'a QuerySystem,
}

/// One deferred world-mutation (see `QuerySystem::deferred`).  Invoked
/// with exactly the system whose queue it was parked in — guaranteed by
/// the queue living on its target system, never by whichever system owns
/// the frame that drains it.
type DeferredMutation = Box<dyn FnOnce(&QuerySystem) + Send>;

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

// ── d17 test hooks ────────────────────────────────────────────────
/// Reader sets this on entering the probe→verify window, then spins
/// while [`D17_PARK`] is set — letting the test mark the reader's OWN
/// node red inside the window (via a dependency's changed-fingerprint
/// publish, which needs no quiesce and so works with the guard at
/// either placement).
#[cfg(test)]
pub(crate) static D17_ARRIVED: AtomicBool = AtomicBool::new(false);
/// Test-held gate: while true, a reader parked in the cached branch's
/// probe→verify window stays parked.
#[cfg(test)]
pub(crate) static D17_PARK: AtomicBool = AtomicBool::new(false);
/// `Q::descriptor().name.as_ptr() as usize` of the ONE query the d17
/// test wants parked (0 = disabled).  Pointer-keyed so concurrent
/// tests' cached hits never enter the park.
#[cfg(test)]
pub(crate) static D17_TARGET: AtomicUsize = AtomicUsize::new(0);

// ── d15 test hooks ────────────────────────────────────────────────
/// Same park shape as the d17 hooks but SEPARATE statics: d15 and d17
/// overlap when the full suite runs in parallel, and a shared park flag
/// would release one test's parked reader when the OTHER test cleans
/// up — desynchronizing both orchestrations (the d15 full-run flake).
#[cfg(test)]
pub(crate) static D15_ARRIVED: AtomicBool = AtomicBool::new(false);
/// Test-held gate for the d15 reader (see [`D15_ARRIVED`]).
#[cfg(test)]
pub(crate) static D15_PARK: AtomicBool = AtomicBool::new(false);
/// Pointer-keyed target for the d15 reader (see [`D15_ARRIVED`]).
#[cfg(test)]
pub(crate) static D15_TARGET: AtomicUsize = AtomicUsize::new(0);

// ── d21 test hooks ────────────────────────────────────────────────
/// Owner sets this once the phase-2 publish has turned its node GREEN
/// and its cache entry is in place, then spins while [`D21_PARK`] is
/// set — letting the test issue a CONCURRENT get of the same key inside
/// the green-publish → signal window (the window the old ordering left
/// open between `finish_task_and_publish` and `cache.insert`).
#[cfg(test)]
pub(crate) static D21_ARRIVED: AtomicBool = AtomicBool::new(false);
/// Test-held gate: while true, an owner parked in [`D21_ARRIVED`] stays
/// parked.
#[cfg(test)]
pub(crate) static D21_PARK: AtomicBool = AtomicBool::new(false);
/// `Q::descriptor().name.as_ptr() as usize` of the ONE query the d21
/// test wants parked (0 = disabled).  Pointer-keyed so concurrent
/// tests' publishes never enter the park.
#[cfg(test)]
pub(crate) static D21_TARGET: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Total nested `get_shared` frames on this thread, across ALL
    /// systems.  Two consumers: the native-stack backstop
    /// (`validate_enter`'s `MAX_GLOBAL_DEPTH` check — a chain of distinct
    /// query types must be bounded regardless of which system each frame
    /// ran on) and `mutate_world`'s "inside a running query → defer"
    /// test, which must defer for a frame on ANY system.
    static IN_FLIGHT_TOTAL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Per-system nesting depth.  Only the frame that brings THIS
    /// system's depth 0 → 1 holds that system's quiesce read guard.
    ///
    /// Pre-fix a single thread-global counter decided guard ownership, so
    /// a thread whose outermost frame was on system A took NO read guard
    /// on B when its compute nested into B (provider/system mispairing —
    /// the f1/d20 hazard): a concurrent `B.invalidate` could then land
    /// between the nested compute's input reads and its publish — the
    /// d10 erase window, uncovered on foreign systems.  With per-system
    /// depths, the nested frame takes B's read guard and the erase window
    /// closes there too.
    static IN_FLIGHT_BY_SYSTEM: std::cell::RefCell<FxHashMap<usize, usize>> =
        std::cell::RefCell::new(FxHashMap::default());
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

impl<'a> InFlightGuard<'a> {
    fn acquire(system: &'a QuerySystem) -> Self {
        IN_FLIGHT_TOTAL.with(|d| d.set(d.get() + 1));
        // Per-system depth: the frame that brings THIS system's depth
        // 0 → 1 takes that system's quiesce read guard.  A nested frame on
        // a DIFFERENT system therefore holds ITS guard too — closing the
        // d10 erase window that a provider/system mispairing used to leave
        // open on foreign systems (see `IN_FLIGHT_BY_SYSTEM`).
        let prev = IN_FLIGHT_BY_SYSTEM.with(|m| {
            let mut m = m.borrow_mut();
            let e = m.entry(system.id).or_insert(0);
            let prev = *e;
            *e += 1;
            prev
        });
        // Counted for NESTED frames too: `Drop for QuerySystem` must see
        // every live guard, wherever in the stack it sits.
        system.frames.fetch_add(1, Ordering::Relaxed);
        if prev > 0 {
            return InFlightGuard {
                _read: None,
                parked: false,
                system,
            };
        }
        // Deadlock-freedom of this acquisition: a thread blocked HERE
        // (invalidator active, we are outermost) holds no active key —
        // the lock sits before `acquire_key_slot` — and every job latch
        // we could later wait on is owned by a thread holding its own
        // read guard, so the wait-for chain terminates at computing
        // threads that publish and release.  parking_lot's
        // write-preferring fairness also guarantees the invalidator is
        // eventually served (no starvation under a compute storm).
        InFlightGuard {
            _read: Some(system.quiesce.read()),
            parked: false,
            system,
        }
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
        // quiesce read guard.  (`IN_FLIGHT_TOTAL` is irrelevant here — a
        // frame that is outermost on system B while nested inside system
        // A still holds B's guard, and it is B's guard `park_wait` must
        // release across a latch wait.)
        IN_FLIGHT_BY_SYSTEM
            .with(|m| m.borrow().get(&self.system.id).copied().unwrap_or(0))
            == 1
    }
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        // Decrement THIS system's depth FIRST: the drain below (and its
        // debug assert) must observe the post-drop state — this frame is
        // gone by then.
        let prev = IN_FLIGHT_BY_SYSTEM.with(|m| {
            let mut m = m.borrow_mut();
            let e = m.entry(self.system.id).or_insert(0);
            let prev = *e;
            *e = e.saturating_sub(1);
            if *e == 0 {
                m.remove(&self.system.id);
            }
            prev
        });
        IN_FLIGHT_TOTAL.with(|d| d.set(d.get().saturating_sub(1)));
        self._read = None; // release the read lock (if outermost for this system)
        self.system.frames.fetch_sub(1, Ordering::Relaxed);
        // Outermost frame FOR THIS SYSTEM: drain world-mutations that
        // computes on this thread deferred (d11) — for THIS system only.
        // The queue is partitioned by (system, thread), so mutations
        // parked on a DIFFERENT system wait for that system's own
        // outermost frame — or, if it never gets one, for its `Drop` —
        // instead of being executed with the wrong `&QuerySystem`.  The
        // read lock is GONE by now and this frame's publish is done, so
        // taking the write lock cannot self-deadlock, and the deferred
        // invalidations cannot be erased by this thread's publishes.
        // The drain is panic-safe (see `drain_deferred`).
        if prev == 1 {
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

impl<T> CloneAny for T
where
    T: std::any::Any + Clone + Send + Sync,
{
    fn clone_box(&self) -> Box<dyn CloneAny + Send + Sync> {
        Box::new(self.clone())
    }
}

pub struct QuerySystem {
    /// Stable instance id (monotonic).  Keys this thread's deferred-
    /// mutation presence marker (`DEFERRED_PENDING`) and the `IN_DRAIN`
    /// set — NOT the address, which can be reused after drop (see
    /// `NEXT_SYSTEM_ID`).
    id: usize,
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
    /// the owning thread may drain its own partition, and only at an
    /// outermost frame of THIS system, because that is the one moment
    /// when no compute of any system is in flight on that thread — the
    /// condition `drain_deferred` needs to take the quiesce write lock
    /// without either self-deadlocking (d11) or invalidating nodes under
    /// a compute that could later publish over them (d10's erase
    /// window).  Draining another thread's partition would violate both.
    /// `Drop` is the single exception: it runs once the system is
    /// unreachable, so no frame of it can be live anywhere.
    deferred: Mutex<FxHashMap<std::thread::ThreadId, Vec<DeferredMutation>>>,
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
    vtables: ArcSwap<FxHashMap<TypeId, Arc<dyn QueryVTableDyn>>>,
    /// Serialises vtable registration (double-checked init).  The hot
    /// path never touches it.
    init_lock: Mutex<()>,
    /// Dependency graph for incremental re-evaluation.
    /// The dependency graph.  `pub(crate)`: its red-bit mutators
    /// (`invalidate`, `dirty_readers*`, `mark_green`, `reset`, bare
    /// `publish`) change the world WITHOUT quiescence — user code must
    /// go through the quiesced/deferring gates
    /// (`QuerySystem::invalidate*`, `clear_all`, `mutate_world`),
    /// otherwise a red bit can be erased by an in-flight publish
    /// (d10) or the world can change under a running compute.
    pub(crate) dep_graph: dep_graph::DepGraph,
    /// Registered query names for debugging / introspection.
    query_names: RwLock<FxHashMap<TypeId, &'static str>>,
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
    /// parking_lot's write-preferring fairness solves starvation: an
    /// invalidator is served after the current readers drain even under
    /// a continuous compute stream.  Nested gets skip the read lock
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
    node_to_key:
        RwLock<FxHashMap<dep_graph::DepNodeIndex, (TypeId, Box<dyn CloneAny + Send + Sync>)>>,
}

impl Drop for QuerySystem {
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
        let pending: Vec<DeferredMutation> = {
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

impl<'input> QuerySystem {
    pub fn new() -> Self {
        let id = NEXT_SYSTEM_ID.fetch_add(1, Ordering::Relaxed);
        QuerySystem {
            id,
            deferred: Mutex::new(FxHashMap::default()),
            frames: AtomicUsize::new(0),
            vtables: ArcSwap::from_pointee(FxHashMap::default()),
            init_lock: Mutex::new(()),
            dep_graph: dep_graph::DepGraph::with_owner(id),
            query_names: RwLock::new(FxHashMap::default()),
            active_keys: (0..ACTIVE_SHARDS)
                .map(|_| CachePadded(RwLock::new(FxHashMap::default())))
                .collect(),
            wait_graph: RwLock::new(FxHashMap::default()),
            quiesce: RwLock::new(()),
            node_to_key: RwLock::new(FxHashMap::default()),
        }
    }

    /// Stable per-instance id.  Keys this thread's `DEFERRED_PENDING`
    /// marker and the `IN_DRAIN` set — never an address: addresses are
    /// recycled after drop, which once let one system's deferred closure
    /// run against another.
    fn system_id(&self) -> usize {
        self.id
    }

    /// Drain the world-mutations THIS thread parked on THIS system.
    /// Called from the outermost in-flight guard's drop, after that
    /// frame's read lock is released and its publish is done: no compute
    /// of any system is in flight on this thread, so the quiesce write
    /// lock cannot self-deadlock and no publish of this thread can erase
    /// the mutations made here.
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
            IN_FLIGHT_BY_SYSTEM
                .with(|m| m.borrow().get(&self.id).copied().unwrap_or(0))
                == 0,
            "drain_deferred called from inside a running query: \
             route world-mutations through QuerySystem::mutate_world \
             (read→write upgrade would self-deadlock, d11)"
        );
        DEFERRED_PENDING.with(|p| p.borrow_mut().remove(&self.id));
        let pending: Vec<DeferredMutation> = self
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
    fn run_deferred_batch(&self, pending: Vec<DeferredMutation>) {
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
            IN_FLIGHT_BY_SYSTEM
                .with(|m| m.borrow().get(&self.id).copied().unwrap_or(0))
                == 0,
            "run_deferred_batch entered with a live in-flight frame: a deferred \
             closure that runs a query would self-deadlock on the quiesce lock"
        );
        let _drain_guard = DrainGuard::new(self.id);
        let _write = self.quiesce.write();
        let mut first_panic: Option<Box<dyn std::any::Any + Send>> = None;
        for f in pending {
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| f(self))) {
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
    fn mutate_world(&self, f: impl FnOnce(&Self) + Send + 'static) {
        let id = self.system_id();
        // Draining THIS system's deferred mutations: the write lock is
        // already held by this thread's outermost-guard drain — run
        // inline.  Scoped per system: a DIFFERENT system's mutate_world
        // must NOT skip its own quiesce gate (pre-fix the flag was
        // global and let exactly that happen).
        if IN_DRAIN.with(|d| d.borrow().contains(&id)) {
            f(self);
            return;
        }
        if IN_FLIGHT_TOTAL.with(|d| d.get()) > 0 {
            // Parked on the TARGET system, this thread's partition — see
            // `QuerySystem::deferred` for why neither can be omitted.
            self.deferred
                .lock()
                .entry(std::thread::current().id())
                .or_default()
                .push(Box::new(f));
            DEFERRED_PENDING.with(|p| p.borrow_mut().insert(id));
            return;
        }
        let _write = self.quiesce.write();
        f(self);
    }

    /// Look up the vtable for `Q`.  One atomic load + one `Arc` refcount
    /// bump — no lock, no system-wide coordination (Perf Book §13.1).
    fn find_vtable<Q: Query>(&self) -> Option<Arc<QueryVTable<Q>>> {
        self.vtables
            .load()
            .get(&TypeId::of::<Q>())
            .map(|arc| {
                let any_ref = arc.as_ref().as_any();
                let _ = any_ref.downcast_ref::<QueryVTable<Q>>()?;
                let arc_clone = arc.clone();
                // SAFETY: the TypeId lookup guarantees the concrete type
                // is exactly QueryVTable<Q>, so the raw pointer cast is valid.
                Some(unsafe { Arc::from_raw(Arc::into_raw(arc_clone) as *const QueryVTable<Q>) })
            })
            .flatten()
    }

    /// Look up the vtable as a type-erased executor for the given TypeId.
    fn find_vtable_dyn(&self, ty: TypeId) -> Option<Arc<dyn QueryVTableDyn>> {
        self.vtables.load().get(&ty).cloned()
    }

    /// Re-probe the cache and, if the freshly published value verifies
    /// green, serve it.  Shared by the two phantom-wait retry sites in
    /// `get_shared`'s Wait branch (owner completed while we were
    /// registering / before we parked): the value is already in the cache,
    /// so serving it avoids a full recompute.  Same validity discipline as
    /// the wake path — a non-quiesced fan-out can have marked the node red
    /// between the owner's publish and this probe, in which case we fall
    /// through to recompute (`None`).
    fn serve_reprobed<Q: Query>(
        &self,
        vtable: &Arc<QueryVTable<Q>>,
        key: &Q::Key,
        provider: &dyn QueryProvider<'_>,
    ) -> Option<Arc<Q::Value>> {
        let (node, cached, fresh) = probe::<Q>(vtable, &self.dep_graph, key);
        if fresh {
            self.node_to_key
                .write()
                .insert(node, (TypeId::of::<Q>(), Box::new(key.clone())));
        }
        let value = cached?;
        self.try_serve_verified::<Q>(node, value, provider)
    }

    /// Re-request a dirty dependency during transitive green verification.
    /// Given the dependency's TypeId and type-erased key, look up the
    /// appropriate vtable and execute the query.
    fn rerequest_dirty_dependency(
        &self,
        dep_ty: TypeId,
        dep_key: &dyn std::any::Any,
        provider: &dyn QueryProvider<'_>,
    ) {
        if let Some(vtable_dyn) = self.find_vtable_dyn(dep_ty) {
            // Execute on THIS system — the one whose dep graph the
            // verification walked — regardless of which system the
            // provider is bound to — executing on the provider's system
            // instead would leak cache/poison side effects into a foreign
            // query system.
            let _ = vtable_dyn.get_shared_erased(self, dep_key, provider);
            // Ignore errors: if the re-request fails (cycle, poison, etc.),
            // the verification will fail and we'll fall through to compute.
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
    fn try_serve_verified<Q: Query>(
        &self,
        node_index: dep_graph::DepNodeIndex,
        value: Arc<Q::Value>,
        provider: &dyn QueryProvider<'_>,
    ) -> Option<Arc<Q::Value>> {
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
            // Clone (TypeId, key) out and DROP the read guard before
            // re-requesting: the re-request's nested `get_shared` takes
            // `node_to_key.write()` — a same-thread read→write upgrade
            // self-deadlocks parking_lot RwLocks (this hung g1c/stress
            // before the fix).
            let dep_info: Option<(TypeId, Box<dyn CloneAny + Send + Sync>)> = {
                let guard = self.node_to_key.read();
                match guard.get(&dirty_dep) {
                    // Explicit deref: `k.clone_box()` alone resolves
                    // through the tuple/Box chain and mis-infers a
                    // 'static receiver bound (E0597).
                    Some((t, k)) => {
                        let key_dyn: &(dyn CloneAny + Send + Sync) = &**k;
                        Some((*t, key_dyn.clone_box()))
                    }
                    None => None,
                }
            };
            let Some((dep_ty, dep_key_boxed)) = dep_info else {
                return None; // unresolvable dep: recompute (conservative)
            };
            self.rerequest_dirty_dependency(dep_ty, dep_key_boxed.as_ref(), provider);
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
    fn vtable_or_init<Q: Query>(&self) -> Arc<QueryVTable<Q>> {
        if let Some(vt) = self.find_vtable::<Q>() {
            return vt;
        }
        let _guard = self.init_lock.lock();
        if let Some(vt) = self.find_vtable::<Q>() {
            return vt; // lost the registration race
        }
        let vt: Arc<QueryVTable<Q>> = Arc::new(QueryVTable::<Q>::new());
        let mut map = FxHashMap::clone(&*self.vtables.load());
        map.insert(TypeId::of::<Q>(), vt.clone());
        self.vtables.store(Arc::new(map));
        self.query_names
            .write()
            .insert(TypeId::of::<Q>(), Q::descriptor().name);
        vt
    }

    /// Split active-key acquisition into validation + commit.  Returns
    /// `KeySlot::Compute` when this thread will run the query (and pushes
    /// onto its active stack), or `KeySlot::Wait(job)` when another thread
    /// owns the computation — in which case the current thread's stack is
    /// untouched (no speculative enter/leave needed).
    #[inline(always)]
    fn acquire_key_slot<Q: Query>(
        &self,
        vtable: &QueryVTable<Q>,
        key: &Q::Key,
        akey: &ActiveKey,
    ) -> Result<KeySlot, QueryCycleError> {
        let me = std::thread::current().id();

        // Same-thread cycle and depth checks BEFORE any cross-thread wait.
        vtable.validate_enter(key)?;

        if vtable.descriptor.eval_always {
            vtable.commit_enter(key);
            return Ok(KeySlot::Compute);
        }

        let mut active = self.active_keys[active_shard(akey)].write();

        match active.get(akey) {
            Some(ActiveKeyStatus::Started(job)) if job.owner == me => {
                // Double-safety guard: `validate_enter` should already catch
                // same-thread re-entry, but keep the explicit check.  With
                // the map keyed by the EXACT key this arm is unreachable
                // through correct input — a different key can no longer
                // land on this entry (that was the hash-collision false
                // cycle).
                return Err(QueryCycleError {
                    query_name: Q::descriptor().name,
                    message: format!(
                        "query cycle detected (owner thread): `{}` with key `{:?}`",
                        Q::descriptor().name,
                        key
                    ),
                    stack: vec![format!("{:?}", key)],
                });
            }
            Some(ActiveKeyStatus::Started(job)) => {
                // Another thread owns this key — we only wait, no commit.
                return Ok(KeySlot::Wait(job.clone()));
            }
            Some(ActiveKeyStatus::Poisoned) => {
                // Exact-key hit: the poison BELONGS to this key.
                return Err(QueryCycleError {
                    query_name: Q::descriptor().name,
                    message: format!(
                        "query previously panicked: `{}` with key `{:?}`",
                        Q::descriptor().name,
                        key
                    ),
                    stack: vec![format!("{:?}", key)],
                });
            }
            None => {
                // We become the owner — commit the active-stack entry.
                // The ONLY clone of the ActiveKey happens here: the Wait
                // paths above borrow it and drop it.
                vtable.commit_enter(key);
                let job = job::QueryJob::new();
                active.insert(akey.clone(), ActiveKeyStatus::Started(job));
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
    pub fn get_shared<Q: Query>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
        provider: &dyn QueryProvider<'_>,
    ) -> Result<Arc<Q::Value>, QueryCycleError> {
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
        // gets skip the read lock via `IN_FLIGHT_BY_SYSTEM` (only a frame
        // that brings its system's depth 0 → 1 pays one shared read
        // acquisition); a nested frame on a FOREIGN system holds that
        // system's guard instead.
        //
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
        // `Q` registers through `init_lock`, every later call is one
        // atomic load.
        // `mut`: the WAIT path re-resolves both after a latch wake — the
        // wait voluntarily released the quiesce read lock, so a completed
        // `clear_all` invalidates the pre-wait snapshot (see the Wait
        // branch, d18).
        let mut vtable = self.vtable_or_init::<Q>();
        let (mut node_index, cached, fresh) = probe::<Q>(&vtable, &self.dep_graph, &key);

        // Register the key for this node EXACTLY ONCE, on the fresh path
        // only: the mapping is immutable for the node's lifetime, so re-
        // writing it on every hit just burned a global write lock plus a
        // `Box`ed key clone per query execution.  A node only ever needs
        // the mapping after it has been computed (a later request can
        // re-request it as a dirty dependency), and the creator's
        // `get_shared` reaches the compute path before any reader can
        // depend on it — registering here is sufficient.
        if fresh {
            self.node_to_key
                .write()
                .insert(node_index, (TypeId::of::<Q>(), Box::new(key.clone())));
        }

        // If we have a cached value, verify its transitive dependency
        // closure.  If verification finds a dirty dependency, re-request
        // it (nested get) and re-verify.  This implements rustc-style
        // try_mark_green: a cached value is served only if all transitive
        // dependencies are green (or become green after re-requesting).
        if let Some(value) = cached {
            #[cfg(test)]
            {
                // d17 hook: deterministic park in the probe→verify window —
                // the value `Arc` is already captured here; a concurrent
                // mark of THIS node while parked is exactly the stale-serve
                // race (stale serve from the cached path).  Pointer-targeted so
                // only the test's own query enters the park.
                let target = D17_TARGET.load(Ordering::SeqCst);
                if target != 0 && target == Q::descriptor().name.as_ptr() as usize {
                    D17_ARRIVED.store(true, Ordering::SeqCst);
                    // Bounded like d21: if the orchestrating test unwinds
                    // between ARRIVED and PARK=false (a failed assertion, a
                    // guard timeout), the reader must not spin forever — a
                    // detached thread wedged at teardown can hang the whole
                    // test process.  The deadline degrades to an early
                    // release; the test's assertions then run against a
                    // deterministic state instead of hanging CI.
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                    while D17_PARK.load(Ordering::SeqCst) {
                        if std::time::Instant::now() > deadline {
                            break;
                        }
                        std::thread::yield_now();
                    }
                } else {
                    // d15 hook: same window, independent statics — see
                    // the `D15_ARRIVED` note on why they must not be
                    // shared with d17's.
                    let target = D15_TARGET.load(Ordering::SeqCst);
                    if target != 0 && target == Q::descriptor().name.as_ptr() as usize {
                        D15_ARRIVED.store(true, Ordering::SeqCst);
                        // Bounded like d21 — see the d17 note above.
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(2);
                        while D15_PARK.load(Ordering::SeqCst) {
                            if std::time::Instant::now() > deadline {
                                break;
                            }
                            std::thread::yield_now();
                        }
                    }
                }
            }
            // Shared validity gate (also used by the waiter branch below):
            // serve only if the transitive closure is green; a dirty
            // dependency is re-requested and re-verified first.  The
            // helper records this task's read of the node on success.
            if let Some(value) = self.try_serve_verified::<Q>(node_index, value, provider) {
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
            let slot = self.acquire_key_slot::<Q>(&vtable, &key, &akey)?;

            match slot {
                KeySlot::Compute => break,
                KeySlot::Wait(job) => {
                    // Register the wait edge and detect cross-thread
                    // cycles BEFORE blocking.  Without this, two threads
                    // waiting on each other's latch would wedge forever.
                    if let Err(chain) = self.register_wait(&job) {
                        // The cycle verdict is TENTATIVE: the job snapshot
                        // from `acquire_key_slot` may be stale.  Between
                        // reading the active entry and walking the wait
                        // graph the owner can complete the job (removing
                        // the entry) and legitimately become blocked on a
                        // job owned by US — the walk then sees "owner
                        // waits on me" for a key nobody owns any more.
                        // That is a phantom cycle, not a deadlock.  A real
                        // cross-thread cycle requires the owner to still
                        // hold this key (a thread blocked on our latch can
                        // only exist while its own compute is in flight),
                        // so re-check under the shard lock: if the entry
                        // is gone or was replaced, retry instead of
                        // failing.
                        if self.job_still_active::<Q>(&akey, &job) {
                            return Err(QueryCycleError {
                                query_name: Q::descriptor().name,
                                message: format!(
                                    "query cycle detected across threads: `{}` with key `{:?}`\n  wait chain: {:?}",
                                    Q::descriptor().name,
                                    key,
                                    chain
                                ),
                                stack: chain.iter().map(|t| format!("{:?}", t)).collect(),
                            });
                        }
                        // Phantom — the owner already completed; loop back
                        // (the cache now holds the value, or we recompute).
                        if let Some(value) = self.serve_reprobed::<Q>(&vtable, &key, provider) {
                            return Ok(value);
                        }
                        continue;
                    }
                    // The owner may also have completed between our slot
                    // acquisition and the edge registration (its
                    // `clear_waiters_of_me` cannot remove an edge that did
                    // not exist yet).  In that case we will fast-path on
                    // the latch without blocking, and our just-inserted
                    // edge is a phantom "blocked thread" other walkers
                    // could follow — retract it and retry.
                    if !self.job_still_active::<Q>(&akey, &job) {
                        self.remove_wait();
                        // The owner just published; its value is already
                        // in the cache — serve it rather than recompute.
                        // Same validity discipline as the wake path (a
                        // non-quiesced fan-out can have marked the node
                        // red between the owner's publish and this probe).
                        if let Some(value) = self.serve_reprobed::<Q>(&vtable, &key, provider) {
                            return Ok(value);
                        }
                        continue;
                    }
                    // Release the quiesce read lock while blocked on the
                    // latch (see `InFlightGuard::park_wait`): holding it
                    // across a latch wait closes a three-way cycle with
                    // the invalidator's write lock.
                    _in_flight.park_wait();
                    let wait_result = job.latch.wait_on();
                    // Remove the wait edge on BOTH the Ok and the Err path.
                    self.remove_wait();
                    _in_flight.unpark();
                    wait_result.map_err(|_| QueryCycleError {
                        query_name: Q::descriptor().name,
                        message: format!(
                            "query poisoned: `{}` with key `{:?}`",
                            Q::descriptor().name,
                            key
                        ),
                        stack: vec![format!("{:?}", key)],
                    })?;
                    // ── re-resolve against the CURRENT world ─────────
                    // `park_wait` released the quiesce read lock, and
                    // parking_lot is write-preferring: a `clear_all`
                    // requested while we were parked is granted the moment
                    // the owner drops its read guard — BEFORE our `unpark`
                    // re-takes the read lock.  Everything captured before
                    // the wait is then stale:
                    //   * `vtable` may be an ORPHANED Arc: the registry was
                    //     emptied, so its cache is invisible to the fresh
                    //     world and a lookup would serve the owner's
                    //     pre-clear value without the fresh registry ever
                    //     seeing it;
                    //   * `node_index` is a PRE-RESET index: `next_index`
                    //     restarted at 1, so the index may already have
                    //     been re-allocated by a *new* node — running
                    //     transitive verification (and its `verified_gen`
                    //     stamps) against the recycled slot would hand the
                    //     NEW node a "closure verified" stamp it never
                    //     earned, and its later hits would skip closure
                    //     verification entirely (d18).
                    // Re-resolve both, then serve through the normal cached
                    // path.
                    vtable = self.vtable_or_init::<Q>();
                    let (node2, cached2, fresh2) = probe::<Q>(&vtable, &self.dep_graph, &key);
                    node_index = node2;
                    if fresh2 {
                        self.node_to_key
                            .write()
                            .insert(node_index, (TypeId::of::<Q>(), Box::new(key.clone())));
                    }

                    if let Some(value) = cached2 {
                        // Same validity discipline as the cached branch:
                        // the value in the cache may predate an
                        // invalidation that landed between the owner's
                        // publish and our wake (owner-side d16 epoch
                        // restore, non-quiesced fan-out).  The helper
                        // records the current task's read of the node on
                        // success — the read edge is recorded so subsequent
                        // invalidations propagate to the outer query.
                        if let Some(value) =
                            self.try_serve_verified::<Q>(node_index, value, provider)
                        {
                            return Ok(value);
                        }
                    }

                    // Nothing cached (e.g. post-`clear_all`), the node
                    // itself was red, or a re-request failed: loop back and
                    // recompute (the owner's active entry is gone, so the
                    // next `acquire_key_slot` assigns us the compute).
                    continue;
                }
            }
        }

        // (The quiesce read lock is held for this whole function — taken
        // above, before the attempt loop.  The Wait branch voluntarily
        // released it across the latch wait and re-took it on wake, so
        // this compute path is already quiesced.)

        // Start tracking dependencies for this node.
        self.dep_graph.start_task(node_index);

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
        // registered, the node is red until the phase-2 publish clears
        // it, and a finalize panic returns before `clear_dirty` — so a
        // staged-but-unpublished value sits behind a permanently red
        // node.  `eval_always` queries skip the insert (they have no
        // cache entry to serve green), but they DO get a fingerprint
        // below: without one, `finish_task_and_publish` skips the
        // `changed` check entirely (fp = None ⇒ `if let Some(fp)` never
        // fires), so a changed eval_always value would never mark its
        // direct readers dirty and they would keep stale green cache
        // entries forever.
        // The staging itself (`Arc::new` + `cache.insert`) is INSIDE the
        // guard: `cache.insert` can panic (`DefaultCache::evict_tail`'s
        // `unreachable!()` guard, a future cache bug), and a panic there
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
        self.dep_graph.set_finishing(node_index);

        let staged: Result<(Arc<Q::Value>, u64), Box<dyn std::any::Any + Send>> =
            panic::catch_unwind(AssertUnwindSafe(|| {
                let v = Q::compute(&key, provider);
                let fp = Q::Value::stable_hash(&v);
                // Wrap once: the SAME handle is staged into the cache,
                // fingerprinted, and returned to the caller — no deep
                // copy of the freshly computed value on any path.
                let value = Arc::new(v);
                if !vtable.descriptor.eval_always {
                    // Staged while the node's finishing flag (d24) is set:
                    // the merged entry carries node + value in ONE slot.
                    vtable.cache.stage(key.clone(), node_index, Arc::clone(&value));
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
                // poison and resume.  Also guarded: a debug
                // class-assertion on the partial read set must not strand
                // the waiters either.  If the panic came from before the
                // insert, nothing was staged; if it came from the insert
                // itself, the cache is internally consistent (one node
                // evicted, nothing inserted) and the node is still red
                // (finalize never ran), so the next getter sees the
                // poisoned active entry, exactly like a compute panic.
                let r = panic::catch_unwind(AssertUnwindSafe(|| {
                    let _ = self.dep_graph.finish_task_and_publish(node_index, None);
                }));
                if let Err(p) = r {
                    eprintln!(
                        "query finalize bookkeeping panicked while already unwinding: {}",
                        panic_message(&*p)
                    );
                }
                // d24: clear the finishing flag in case the bookkeeping
                // call panicked before reaching its own clear — a leaked
                // flag would make the node permanently unverifiable.
                self.dep_graph.clear_finishing(node_index);
                vtable.leave();
                let mut active = self.active_keys[active_shard(&akey)].write();
                self.poison_active(&vtable, &akey, &mut active);
                drop(active);
                panic::resume_unwind(payload);
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
            #[cfg(test)]
            {
                // d21 hook: park AFTER the green publish — post-fix this
                // is also post-insert, so a concurrent probe from another
                // thread must see the STAGED value.  Pointer-keyed so
                // only the test's own query enters the park; bounded so a
                // wedged orchestration fails instead of hanging.
                let target = D21_TARGET.load(Ordering::SeqCst);
                if target != 0 && target == Q::descriptor().name.as_ptr() as usize {
                    D21_ARRIVED.store(true, Ordering::SeqCst);
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                    while D21_PARK.load(Ordering::SeqCst) {
                        if std::time::Instant::now() > deadline {
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
            }
        }));

        // Deactivate: pop this key off the current thread's stack — on
        // EVERY path (success, compute panic, stable-hash panic,
        // finalize panic).  The active entry was committed in
        // `acquire_key_slot`; skipping the pop on a panicked path left a
        // stale entry that poisoned this thread's vtable state.
        vtable.leave();

        // Signal completion or poisoning to any waiting threads.
        let mut active = self.active_keys[active_shard(&akey)].write();

        let value = match finalize {
            Ok(()) => {
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
                if let Some(ActiveKeyStatus::Started(job)) = active.get(&akey) {
                    job.latch.set();
                }
                if let Some(entry) = active.remove(&akey)
                    && let ActiveKeyStatus::Started(job) = entry
                {
                    // Edges for the completed job are now walker-invisible
                    // (latch resolved); this sweep just reclaims them.
                    self.clear_waiters_of_me(job.id);
                }
                drop(active);
                value
            }
            Err(payload) => {
                // d24: the finalize panicked before its tail could run —
                // clear the finishing flag FIRST (it may still hold the
                // active_keys write lock below, and a leaked flag would
                // make the node permanently unverifiable).
                self.dep_graph.clear_finishing(node_index);
                // Un-stage the phase-1 entry before poisoning: a fresh
                // node is GREEN by default (dirty bit starts clear), so a
                // staged value left here would be served by the next
                // getter's `probe` (green + cache hit) without ever
                // consulting the poisoned active entry — d14 step 5.
                if !vtable.descriptor.eval_always {
                    // Un-stage the merged entry: the VALUE goes (a fresh
                    // node is GREEN by default — dirty bit starts clear —
                    // so a staged value left here would be served by the
                    // next getter's `probe` without ever consulting the
                    // poisoned active entry, d14 step 5) while the NODE
                    // STAYS registered (trait contract 1).
                    let _ = vtable.cache.unstage(&key);
                }
                self.poison_active(&vtable, &akey, &mut active);
                drop(active);
                // Resume the panic — the compiler will abort or catch it higher up.
                panic::resume_unwind(payload);
            }
        };

        Ok(value)
    }

    /// [`QuerySystem::get_shared`] but handing back an owned value.
    ///
    /// Zero-regression compatibility surface: when the returned `Arc` is
    /// uniquely referenced (cache published one handle, caller holds the
    /// other — so in practice `unwrap_or_clone` clones, keeping semantics
    /// identical to the pre-Arc behaviour), callers that only need the
    /// payload by value are unchanged.
    pub fn get<Q: Query>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
        provider: &dyn QueryProvider<'_>,
    ) -> Result<Q::Value, QueryCycleError> {
        let value = self.get_shared::<Q>(key, provider)?;
        Ok(Arc::unwrap_or_clone(value))
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
    fn job_still_active<Q: Query>(&self, akey: &ActiveKey, job: &job::QueryJob) -> bool {
        let active = self.active_keys[active_shard(akey)].read();
        matches!(
            active.get(akey),
            Some(ActiveKeyStatus::Started(current)) if current.id == job.id
        )
    }

    /// Remove the current thread's wait edge after the latch wait finishes.
    fn remove_wait(&self) {
        let me = std::thread::current().id();
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
    fn poison_active<Q: Query>(
        &self,
        vtable: &QueryVTable<Q>,
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
            system.dep_graph.invalidate(node);
        });
    }

    /// Invalidate a cache entry for a specific query key.
    pub fn invalidate<Q: Query>(&self, key: &Q::Key) {
        let key = key.clone();
        self.mutate_world(move |system| {
            system.invalidate_unquiesced::<Q>(&key);
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
    /// QUIESCED via `mutate_world`, exactly like [`Self::invalidate`] —
    /// the skip decision and the (possible) invalidation run inside the
    /// same quiesced section, so a digest-based skip can never race an
    /// in-flight compute's publish.
    pub fn invalidate_if_changed<Q: Query>(&self, key: &Q::Key, input_digest: u64) -> bool {
        let key = key.clone();
        let invalidated = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&invalidated);
        // ONE quiesced section for both the gate decision and the (possible)
        // invalidation.  The invalidate body is the UNWRAPPED helper — a
        // nested `mutate_world` here would re-take the quiesce write lock
        // this closure already holds and self-deadlock.
        self.mutate_world(move |system| {
            if let Some(vtable) = system.find_vtable::<Q>() {
                if let Some(node) = vtable.lookup_node(&key) {
                    if system.dep_graph.digest_gate(node, input_digest) {
                        system.invalidate_unquiesced::<Q>(&key);
                        flag.store(true, Ordering::Release);
                    }
                }
            }
        });
        invalidated.load(Ordering::Acquire)
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
    /// QUIESCED via `mutate_world`, like every red-green-world mutation.
    pub fn note_input_digest<Q: Query>(&self, key: &Q::Key, input_digest: u64) {
        let key = key.clone();
        self.mutate_world(move |system| {
            if let Some(vtable) = system.find_vtable::<Q>() {
                if let Some(node) = vtable.lookup_node(&key) {
                    system.dep_graph.record_digest(node, input_digest);
                }
            }
        });
    }

    /// The shared invalidate body: withdraw the staged value (node STAYS
    /// registered — trait contract 1), mark the node red, and lift a
    /// sticky poison for the key.  REQUIRES quiescence: called only under
    /// `mutate_world`'s write lock (by [`Self::invalidate`] and the
    /// digest-gated [`Self::invalidate_if_changed`]) — never call it
    /// directly from outside a quiesced section (d10).
    fn invalidate_unquiesced<Q: Query>(&self, key: &Q::Key) {
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
            // Fused (a merge win): ONE cache shard write lock both
            // withdraws the value and hands back the node — the old
            // "node-map read lock + cache write lock" pair.  The node
            // STAYS registered (trait contract 1); only its dep-graph
            // colour goes red here.
            if let Some(node) = vtable.cache.unstage(key) {
                self.dep_graph.invalidate(node);
            }
            // Lift a sticky poison for this key: the red-green world
            // now says "recompute" (node red, cache empty), so the
            // active-key layer must agree instead of vetoing every
            // later get with "query previously panicked".  Only
            // `Poisoned` is removed — `Started` belongs to an
            // in-flight compute, which quiescence guarantees is
            // absent here.
            let akey = ActiveKey::of::<Q>(key);
            let mut active = self.active_keys[active_shard(&akey)].write();
            if matches!(active.get(&akey), Some(ActiveKeyStatus::Poisoned)) {
                active.remove(&akey);
            }
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
    pub fn clear<Q: Query>(&self) {
        // The closure captures nothing (no key, no data), so
        // `FnOnce(&Self) + Send + 'static` holds for any `Q: Query`.
        self.mutate_world(move |system| {
            if let Some(vtable) = system.find_vtable::<Q>() {
                vtable.cache.clear();
            }
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
        self.mutate_world(|system| {
            // Held so a concurrent `vtable_or_init` cannot resurrect a
            // stale vtable by storing a map cloned from the pre-clear
            // registry.
            let _init = system.init_lock.lock();
            // Replacing the registry drops every per-query node map with
            // it (except for `Arc`s already held by in-flight queries).
            system.vtables.store(Arc::new(FxHashMap::default()));
            system.query_names.write().clear();
            // Poison every started job BEFORE dropping its active entry:
            // with quiescence held there are normally no entries — this
            // is a belt-and-braces sweep for any straggler (e.g. a
            // thread that acquired a slot but has not yet taken the
            // quiesce read lock), so no latch waiter can be stranded by
            // the wipe below.
            for shard in &system.active_keys {
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
            system.node_to_key.write().clear();
            system.wait_graph.write().clear();
            system.dep_graph.reset();
        });
    }

    /// List all registered query names (for debugging).
    pub fn list_queries(&self) -> Vec<&'static str> {
        self.query_names.read().values().copied().collect()
    }

    /// Execute a query in `EnsureOk` mode: compute and cache the result,
    /// but discard the value.  Returns `Err(QueryCycleError)` if the query
    /// fails (cycle, depth limit, etc.).
    ///
    /// This is useful for validation passes where you only care whether
    /// the query completes without errors, not what the result is.
    /// Analogous to Rust's `tcx.ensure_ok().$query(..)`.
    pub fn ensure_ok<Q: Query>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
        provider: &dyn QueryProvider<'_>,
    ) -> Result<(), QueryCycleError> {
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
    pub fn par_get<Q: Query>(
        &self,
        keys: &[Q::Key],
        provider: &dyn QueryProvider<'_>,
    ) -> Vec<Result<Q::Value, QueryCycleError>>
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
        // Clone each key for the parallel query. Q::Key is Clone (per QueryKey)
        // but not Copy, so `.copied()` is not an option; explicit `.clone()` is
        // unavoidable here. The clone cost is dominated by the query computation
        // itself, not the key copy.
        let results: Vec<Result<Q::Value, QueryCycleError>> = keys
            .par_iter()
            .map(|key| self.get::<Q>(key.clone(), provider))
            .collect();
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
}

impl Default for QuerySystem {
    fn default() -> Self {
        Self::new()
    }
}

// ── QueryProvider trait ──────────────────────────────────────────

/// Read-only compiler state that a query body may need.
///
/// `Query::compute` receives only `&dyn QueryProvider<'_>`, so anything a query
/// body touches has to be reachable from here.
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
pub trait CompilerSessionView<'input>: Sync {
    fn symbols(&self) -> &crate::hir::symbol::SymbolTable<'input>;
    fn trait_env(&self) -> &crate::hir::traits::TraitEnv<'input>;
    fn resolution_map(&self) -> &crate::hir::resolver::ResolutionMap<'input>;
    /// The parsed program, so a query can reach the item it is keyed by.
    fn program(&self) -> &crate::ast::Program<'input>;
    /// Read-only handle for `TypeContext::new_sharing` (opt-in; `None`
    /// unless the provider was built with
    /// `SessionQueryProvider::with_body_check`).  The returned context must
    /// only be cloned, never mutated through.
    fn ctx_sharing(&self) -> Option<&crate::hir::types::TypeContext<'input>> {
        None
    }
    /// Checker config + cross-function tables for body-check workers
    /// (opt-in, same gate as `ctx_sharing`).
    fn body_check_harness(
        &self,
    ) -> Option<&crate::session::BodyCheckHarness<'input>> {
        None
    }
}

/// Implemented by types that can provide query computations.
///
/// Queries can call other queries through `query_get::<Q>(provider, key)`.
/// Must be `Sync` to support parallel query execution via rayon.
pub trait QueryProvider<'input>: Sync {
    fn query_system<'i>(&self) -> &QuerySystem;

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
    fn session(&self) -> Option<&dyn CompilerSessionView<'input>> {
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
pub fn query_get<Q: Query>(
    provider: &dyn QueryProvider<'_>,
    key: impl IntoQueryKey<Q::Key>,
) -> Result<Q::Value, QueryCycleError> {
    provider.query_system().get::<Q>(key, provider)
}

// ── DefaultQueryProvider ──────────────────────────────────────────

/// A concrete provider that holds a reference to a `QuerySystem`.
pub struct DefaultQueryProvider<'a> {
    system: &'a QuerySystem,
}

impl<'a> DefaultQueryProvider<'a> {
    pub fn new<'input>(system: &'a QuerySystem) -> Self {
        DefaultQueryProvider { system }
    }
}

impl<'a, 'input> QueryProvider<'input> for DefaultQueryProvider<'a> {
    fn query_system(&self) -> &QuerySystem {
        self.system
    }
}

// ── Query handle ──────────────────────────────────────────────────

/// A handle that bundles a `QuerySystem` and a `QueryProvider` together.
pub struct QueryHandle<'a> {
    pub system: &'a QuerySystem,
    pub provider: &'a dyn QueryProvider<'a>,
}

impl<'a> QueryHandle<'a> {
    pub fn new(system: &'a QuerySystem, provider: &'a dyn QueryProvider<'a>) -> Self {
        QueryHandle { system, provider }
    }
    pub fn get<Q: Query>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
    ) -> Result<Q::Value, QueryCycleError> {
        self.system.get::<Q>(key, self.provider)
    }
    /// Execute a query in `EnsureOk` mode: compute and cache the result,
    /// but discard the value.  Returns `Err(QueryCycleError)` on failure.
    pub fn ensure_ok<Q: Query>(
        &self,
        key: impl IntoQueryKey<Q::Key>,
    ) -> Result<(), QueryCycleError> {
        self.system.ensure_ok::<Q>(key, self.provider)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
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

    impl Query for TestQuery {
        type Key = TestKey;
        type Value = TestValue;
        type Cache = DefaultCache<TestKey, TestValue>;
        fn descriptor() -> QueryDescriptor {
            QueryDescriptor::new("test", "a test query")
        }
        fn compute(key: &TestKey, _: &dyn QueryProvider<'_>) -> TestValue {
            TestValue(format!("value_{}", key.0))
        }
    }

    struct TestProvider;
    impl QueryProvider<'_> for TestProvider {
        fn query_system(&self) -> &QuerySystem {
            panic!("TestProvider does not have a QuerySystem")
        }
    }

    #[test]
    fn test_default_cache() {
        let cache = DefaultCache::new();
        let key = TestKey(1);
        let value = TestValue("hello".into());
        // Miss: the probe REGISTERS the node (value half empty).
        let (node, cached, fresh) = cache.probe(&key, || dep_graph::DepNodeIndex(7777));
        assert!(cached.is_none());
        assert!(fresh);
        assert_eq!(node, dep_graph::DepNodeIndex(7777));
        assert_eq!(cache.node_of(&key), Some(node));
        // Stage, then hit: ONE probe fetches node + value together.
        cache.stage(key.clone(), node, Arc::new(value.clone()));
        let (node2, cached2, fresh2) = cache.probe(&key, || unreachable!());
        assert!(!fresh2);
        assert_eq!(node2, node, "node half survives staging");
        assert_eq!(cached2.as_deref(), Some(&value));
        // Unstage: the value goes, the node STAYS (merge contract 1).
        assert_eq!(cache.unstage(&key), Some(node));
        assert!(cache.node_of(&key).is_some(), "node survives unstage");
        let (_, cached3, _) = cache.probe(&key, || unreachable!());
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
            Arc::new(TestValue("a".into())),
        );
        cache.stage(
            k2.clone(),
            dep_graph::DepNodeIndex(2),
            Arc::new(TestValue("b".into())),
        );
        assert!(
            cache.probe(k1, || unreachable!()).1.is_some(),
            "k1 was staged"
        );
        cache.stage(
            k3.clone(),
            dep_graph::DepNodeIndex(3),
            Arc::new(TestValue("c".into())),
        );
        assert!(
            cache.probe(k1, || unreachable!()).1.is_some(),
            "key 1 was accessed, should survive (CLOCK second chance)"
        );
        assert!(
            cache.probe(k2, || unreachable!()).1.is_none(),
            "key 2 was never re-accessed, should be evicted"
        );
        assert!(
            cache.probe(k3, || unreachable!()).1.is_some(),
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
        let (node, cached, fresh) = cache.probe(&(), || dep_graph::DepNodeIndex(5));
        assert!(cached.is_none());
        assert!(fresh);
        cache.stage((), node, Arc::new(TestValue("singleton".into())));
        let (node2, cached2, _) = cache.probe(&(), || unreachable!());
        assert_eq!(node2, node);
        assert_eq!(
            cached2.as_deref(),
            Some(&TestValue("singleton".into()))
        );
        assert_eq!(cache.len(), 1);
        // Unstage: value goes, node stays.
        assert_eq!(cache.unstage(&()), Some(node));
        assert_eq!(cache.node_of(&()), Some(node));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_def_id_cache() {
        let cache = DefIdCache::new();
        let d0 = crate::hir::types::DefId(0);
        let d1 = crate::hir::types::DefId(1);
        let (n0, cached0, fresh0) = cache.probe(&d0, || dep_graph::DepNodeIndex(10));
        assert!(cached0.is_none() && fresh0);
        let (n1, _, fresh1) = cache.probe(&d1, || dep_graph::DepNodeIndex(11));
        assert!(fresh1);
        cache.stage(d0, n0, Arc::new(TestValue("zero".into())));
        cache.stage(d1, n1, Arc::new(TestValue("one".into())));
        assert_eq!(
            cache.probe(&d0, || unreachable!()).1.as_deref(),
            Some(&TestValue("zero".into()))
        );
        assert_eq!(
            cache.probe(&d1, || unreachable!()).1.as_deref(),
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
        impl Query for Q {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("q", "")
            }
            fn compute(key: &TestKey, _: &dyn QueryProvider<'_>) -> TestValue {
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
        impl Query for Q {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("q", "")
            }
            fn compute(key: &TestKey, _: &dyn QueryProvider<'_>) -> TestValue {
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
        impl Query for Q {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("q", "")
            }
            fn compute(key: &TestKey, _: &dyn QueryProvider<'_>) -> TestValue {
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
        impl Query for Q {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("digest_gated_q", "")
            }
            fn compute(key: &TestKey, _: &dyn QueryProvider<'_>) -> TestValue {
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
        impl Query for EvalAlwaysQ {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("eval_always", "always re-computed").with_eval_always(true)
            }
            fn compute(key: &TestKey, _: &dyn QueryProvider<'_>) -> TestValue {
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
        impl Query for DeepQ {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("deep", "depth-limited query").with_depth_limit(true)
            }
            fn compute(key: &TestKey, provider: &dyn QueryProvider<'_>) -> TestValue {
                if key.0 > 0 {
                    // Recursive call with decremented key.  If the depth limit
                    // is exceeded, `query_get` returns an error — we propagate
                    // it by returning a sentinel value so the test can detect it.
                    match query_get::<DeepQ>(provider, TestKey(key.0 - 1)) {
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
        impl Query for P {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("p", "")
            }
            fn compute(key: &TestKey, _: &dyn QueryProvider<'_>) -> TestValue {
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
                impl Query for $name {
                    type Key = TestKey;
                    type Value = TestValue;
                    type Cache = DefaultCache<TestKey, TestValue>;
                    fn descriptor() -> QueryDescriptor {
                        QueryDescriptor::new($tag, "storm query")
                    }
                    fn compute(key: &TestKey, _: &dyn QueryProvider<'_>) -> TestValue {
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

        let system: &'static QuerySystem = Box::leak(Box::new(QuerySystem::new()));
        let provider: &'static dyn QueryProvider =
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
    /// `query_get::<OuterQ>` — so B is a WAITER on A's job while a
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
        impl Query for InnerQ {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("inner", "inner query")
            }
            fn compute(key: &TestKey, _: &dyn QueryProvider<'_>) -> TestValue {
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
        impl Query for OuterQ {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("outer", "outer query")
            }
            fn compute(key: &TestKey, provider: &dyn QueryProvider<'_>) -> TestValue {
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
                let inner = query_get::<InnerQ>(provider, key.clone()).unwrap();
                TestValue(format!("outer_{}", inner.0))
            }
        }

        // Wrapper: exists purely to hold a LIVE task frame while the
        // waiter consumes OuterQ's published value.
        struct WrapperQ;
        impl Query for WrapperQ {
            type Key = TestKey;
            type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("wrapper", "holds a frame while reading outer")
            }
            fn compute(key: &TestKey, provider: &dyn QueryProvider<'_>) -> TestValue {
                WRAPPER_COUNT.fetch_add(1, Ordering::SeqCst);
                let outer = query_get::<OuterQ>(provider, key.clone()).unwrap();
                TestValue(format!("wrapper_{}", outer.0))
            }
        }

        let system: &'static QuerySystem = Box::leak(Box::new(QuerySystem::new()));
        let provider: &'static dyn QueryProvider =
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
        let (atx, arx) = std::sync::mpsc::channel();
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

        let (btx, brx) = std::sync::mpsc::channel();
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
}
