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
//! - [`DefaultCache`] — HashMap-backed cache with LRU eviction.
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
//!     fn compute(key: &DefId, provider: &dyn QueryProvider) -> TypeId { ... }
//!     fn descriptor() -> QueryDescriptor { ... }
//! }
//! ```

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

pub mod dep_graph;
pub mod job;

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

/// Maximum recursive query depth before a depth-limited query returns an error.
const MAX_QUERY_DEPTH: usize = 128;

/// Compute a 64-bit hash for a `(query_type, key)` pair.
/// Used as the key in `node_map` instead of `(TypeId, String)` — avoids
/// the cost of `format!("{:?}", key)` and the fragility of `Debug`
/// output uniqueness.
///
/// # SAFETY (hash collision)
///
/// A 64-bit FxHash collision would cause two different `(query_type, key)`
/// pairs to share the same `DepNodeIndex`, leading to incorrect dependency
/// tracking (incremental pollution).  In practice, for a compiler session
/// with at most ~10^6 query executions, the collision probability is
/// ~10^-8 (birthday bound: ~2^(64/2) = 2^32 ≈ 4×10^9).  This is
/// acceptable for production use.
///
/// If absolute correctness is required, replace the `u64` key in
/// `node_map` with a `HashMap<(TypeId, Q::Key), DepNodeIndex>` using
/// trait-object dispatch — see `node_map` documentation for details.
fn query_key_hash<Q: Query>(key: &Q::Key) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    TypeId::of::<Q>().hash(&mut h);
    key.hash(&mut h);
    h.finish()
}

/// Compute a `(TypeId, u64)` key for the `active_keys` map.
/// The TypeId component eliminates cross-query-type hash collisions.
fn active_key<Q: Query>(key: &Q::Key) -> (TypeId, u64) {
    (TypeId::of::<Q>(), query_key_hash::<Q>(key))
}

// ── Query descriptor ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QueryDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub eval_always: bool,
    pub depth_limit: bool,
}

impl QueryDescriptor {
    pub const fn new(name: &'static str, description: &'static str) -> Self {
        QueryDescriptor { name, description, eval_always: false, depth_limit: false }
    }
    pub const fn with_eval_always(mut self, val: bool) -> Self { self.eval_always = val; self }
    pub const fn with_depth_limit(mut self, val: bool) -> Self { self.depth_limit = val; self }
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
pub trait QueryCacheType<K, V>: std::fmt::Debug
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn lookup(&self, key: &K) -> Option<V>;
    fn insert(&self, key: K, value: V);
    fn remove(&self, key: &K);
    fn clear(&self);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn for_each(&self, f: impl FnMut(&K, &V));
    /// Return cache statistics.  Default implementation returns empty stats.
    fn stats(&self) -> CacheStats { CacheStats::default() }
}

// ── DefaultCache (HashMap-backed, O(1) LRU via linked list) ───────

/// A node in the LRU linked list.
#[derive(Debug, Clone)]
struct LruNode<K, V> {
    key: K,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Internal state of `DefaultCache`, protected by a single `RwLock`.
/// Bundling all mutable fields under one lock avoids multi-lock deadlocks
/// and simplifies the locking discipline.
#[derive(Debug)]
struct CacheInner<K, V> {
    map: HashMap<K, usize>,
    nodes: Vec<LruNode<K, V>>,
    head: Option<usize>,
    tail: Option<usize>,
    free: Vec<usize>,
}

/// HashMap-backed cache with O(1) LRU eviction via a doubly-linked list.
///
/// Uses a `Vec<LruNode<K, V>>` as the node storage pool, a `HashMap<K, usize>`
/// for O(1) key-to-node lookup, and `head`/`tail` pointers for the LRU list.
/// Every `lookup()` moves the accessed node to the **head** (most recently used).
/// On eviction, the node at **tail** (least recently used) is removed.
/// All operations are O(1).
///
/// All mutable state is held in a single `RwLock<CacheInner<K, V>>` to avoid
/// multi-lock deadlocks and simplify the locking discipline.
///
/// The default capacity is `DEFAULT_CACHE_CAPACITY` (512 entries).  Use
/// `with_capacity(0)` for unlimited capacity, or `with_capacity(n)` for a
/// custom limit.
const DEFAULT_CACHE_CAPACITY: usize = 512;

#[derive(Debug)]
pub struct DefaultCache<K, V> {
    inner: RwLock<CacheInner<K, V>>,
    max_entries: usize,
    stats_hits: AtomicU64,
    stats_misses: AtomicU64,
    stats_evictions: AtomicU64,
}

impl<K, V> DefaultCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    pub fn new() -> Self { DefaultCache::with_capacity(DEFAULT_CACHE_CAPACITY) }
    pub fn with_capacity(max_entries: usize) -> Self {
        DefaultCache {
            inner: RwLock::new(CacheInner {
                map: HashMap::default(),
                nodes: Vec::new(),
                head: None,
                tail: None,
                free: Vec::new(),
            }),
            max_entries,
            stats_hits: AtomicU64::new(0),
            stats_misses: AtomicU64::new(0),
            stats_evictions: AtomicU64::new(0),
        }
    }

    /// Allocate a new node slot, reusing a free slot if available.
    fn alloc_node(inner: &mut CacheInner<K, V>, key: K, value: V) -> usize {
        if let Some(idx) = inner.free.pop() {
            inner.nodes[idx] = LruNode { key, value, prev: None, next: None };
            idx
        } else {
            let idx = inner.nodes.len();
            inner.nodes.push(LruNode { key, value, prev: None, next: None });
            idx
        }
    }

    /// Free a node slot (add to free list) and clear its pointers.
    fn free_node(inner: &mut CacheInner<K, V>, idx: usize) {
        if let Some(node) = inner.nodes.get_mut(idx) {
            node.prev = None;
            node.next = None;
        }
        inner.free.push(idx);
    }

    /// Move a node to the head of the LRU list.
    fn move_to_head(inner: &mut CacheInner<K, V>, idx: usize) {
        if Some(idx) == inner.head { return; }

        let (prev, next) = {
            let n = &inner.nodes[idx];
            (n.prev, n.next)
        };

        if let Some(p) = prev { inner.nodes[p].next = next; }
        if let Some(n) = next { inner.nodes[n].prev = prev; }
        if Some(idx) == inner.tail { inner.tail = prev; }

        if let Some(old_head) = inner.head {
            inner.nodes[old_head].prev = Some(idx);
        }
        inner.nodes[idx].prev = None;
        inner.nodes[idx].next = inner.head;
        inner.head = Some(idx);
        if inner.tail.is_none() { inner.tail = Some(idx); }
    }

    /// Evict the tail (least recently used) node.
    fn evict_tail(inner: &mut CacheInner<K, V>) {
        if let Some(tail_idx) = inner.tail {
            let key = inner.nodes[tail_idx].key.clone();
            inner.map.remove(&key);

            let new_tail = inner.nodes[tail_idx].prev;
            if let Some(prev) = new_tail {
                inner.nodes[prev].next = None;
            }
            inner.tail = new_tail;
            if inner.tail.is_none() {
                inner.head = None;
            }

            Self::free_node(inner, tail_idx);
        }
    }
}

impl<K, V> QueryCacheType<K, V> for DefaultCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    fn lookup(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.write().unwrap();
        if let Some(&idx) = inner.map.get(key) {
            Self::move_to_head(&mut inner, idx);
            self.stats_hits.fetch_add(1, Ordering::Relaxed);
            Some(inner.nodes[idx].value.clone())
        } else {
            self.stats_misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    fn insert(&self, key: K, value: V) {
        let mut inner = self.inner.write().unwrap();
        // Check if key already exists — update in place.
        if let Some(&idx) = inner.map.get(&key) {
            inner.nodes[idx].value = value;
            Self::move_to_head(&mut inner, idx);
            return;
        }

        // Evict if at capacity.
        if self.max_entries > 0 && inner.map.len() >= self.max_entries {
            Self::evict_tail(&mut inner);
            self.stats_evictions.fetch_add(1, Ordering::Relaxed);
        }

        // Insert new node.
        let idx = Self::alloc_node(&mut inner, key.clone(), value);
        inner.map.insert(key, idx);
        Self::move_to_head(&mut inner, idx);
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.stats_hits.load(Ordering::Relaxed),
            misses: self.stats_misses.load(Ordering::Relaxed),
            evictions: self.stats_evictions.load(Ordering::Relaxed),
            entries: self.inner.read().unwrap().map.len(),
        }
    }

    fn remove(&self, key: &K) {
        let mut inner = self.inner.write().unwrap();
        if let Some(idx) = inner.map.remove(key) {
            let (prev, next) = {
                let n = &inner.nodes[idx];
                (n.prev, n.next)
            };
            if let Some(p) = prev { inner.nodes[p].next = next; }
            if let Some(n) = next { inner.nodes[n].prev = prev; }
            if Some(idx) == inner.head { inner.head = next; }
            if Some(idx) == inner.tail { inner.tail = prev; }
            Self::free_node(&mut inner, idx);
        }
    }

    fn clear(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.map.clear();
        inner.nodes.clear();
        inner.head = None;
        inner.tail = None;
        inner.free.clear();
    }

    fn len(&self) -> usize { self.inner.read().unwrap().map.len() }

    fn for_each(&self, mut f: impl FnMut(&K, &V)) {
        let inner = self.inner.read().unwrap();
        let mut cur = inner.head;
        while let Some(idx) = cur {
            f(&inner.nodes[idx].key, &inner.nodes[idx].value);
            cur = inner.nodes[idx].next;
        }
    }
}

impl<K, V> Default for DefaultCache<K, V>
where K: Clone + Eq + Hash + std::fmt::Debug, V: Clone + std::fmt::Debug {
    fn default() -> Self { Self::new() }
}

// ── SingleCache (for unit keys) ───────────────────────────────────

/// Single-entry cache for queries with unit keys (`()`).
///
/// Analogous to Rust's `SingleCache<V>` in `caches.rs`.
/// O(1) with no hashing overhead.
#[derive(Debug)]
pub struct SingleCache<V> {
    entry: RwLock<Option<V>>,
    stats_hits: AtomicU64,
    stats_misses: AtomicU64,
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
    fn lookup(&self, _key: &()) -> Option<V> {
        let result = self.entry.read().unwrap().clone();
        if result.is_some() {
            self.stats_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats_misses.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
    fn insert(&self, _key: (), value: V) { *self.entry.write().unwrap() = Some(value); }
    fn remove(&self, _key: &()) { *self.entry.write().unwrap() = None; }
    fn clear(&self) { *self.entry.write().unwrap() = None; }
    fn len(&self) -> usize { if self.entry.read().unwrap().is_some() { 1 } else { 0 } }
    fn for_each(&self, mut f: impl FnMut(&(), &V)) {
        if let Some(ref v) = *self.entry.read().unwrap() { f(&(), v); }
    }
    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.stats_hits.load(Ordering::Relaxed),
            misses: self.stats_misses.load(Ordering::Relaxed),
            evictions: 0,
            entries: self.len(),
        }
    }
}

impl<V: Clone + std::fmt::Debug> Default for SingleCache<V> { fn default() -> Self { Self::new() } }

// ── DefIdCache (Vec-indexed for DefId keys) ───────────────────────

/// Vec-indexed cache for `DefId` keys.
///
/// Analogous to Rust's `DefIdCache<V>` in `caches.rs`.
/// O(1) lookup via `DefId.index` into a `Vec`, no hashing.
/// `DefId` is a `(CrateNum, DefIndex)` pair; this cache uses
/// `DefIndex` as the Vec index (assuming dense indexing).
#[derive(Debug)]
pub struct DefIdCache<V> {
    entries: RwLock<Vec<Option<V>>>,
    stats_hits: AtomicU64,
    stats_misses: AtomicU64,
}

impl<V: Clone + std::fmt::Debug> DefIdCache<V> {
    pub fn new() -> Self {
        DefIdCache {
            entries: RwLock::new(Vec::new()),
            stats_hits: AtomicU64::new(0),
            stats_misses: AtomicU64::new(0),
        }
    }

    fn ensure_index(&self, idx: usize) {
        let mut entries = self.entries.write().unwrap();
        if idx >= entries.len() { entries.resize_with(idx + 1, || None); }
    }
}

impl<V: Clone + std::fmt::Debug> QueryCacheType<crate::hir::types::DefId, V> for DefIdCache<V> {
    fn lookup(&self, key: &crate::hir::types::DefId) -> Option<V> {
        let idx = key.0;
        let result = self.entries.read().unwrap().get(idx).and_then(|e| e.clone());
        if result.is_some() {
            self.stats_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats_misses.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn insert(&self, key: crate::hir::types::DefId, value: V) {
        let idx = key.0;
        self.ensure_index(idx);
        self.entries.write().unwrap()[idx] = Some(value);
    }

    fn remove(&self, key: &crate::hir::types::DefId) {
        let idx = key.0;
        let mut entries = self.entries.write().unwrap();
        if idx < entries.len() { entries[idx] = None; }
    }

    fn clear(&self) { self.entries.write().unwrap().clear(); }

    fn len(&self) -> usize {
        self.entries.read().unwrap().iter().filter(|e| e.is_some()).count()
    }

    fn for_each(&self, mut f: impl FnMut(&crate::hir::types::DefId, &V)) {
        for (i, entry) in self.entries.read().unwrap().iter().enumerate() {
            if let Some(v) = entry {
                f(&crate::hir::types::DefId(i), v);
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
}

// ── Query trait ───────────────────────────────────────────────────

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
    type Key: Clone + Eq + Hash + std::fmt::Debug + Send + Sync + 'static;
    type Value: Clone + std::fmt::Debug + Send + 'static;
    /// The cache implementation for this query's key type.
    type Cache: QueryCacheType<Self::Key, Self::Value> + Default + Send + Sync;

    fn descriptor() -> QueryDescriptor;
    fn compute(key: &Self::Key, provider: &dyn QueryProvider) -> Self::Value;
}

// ── QueryVTable ─────────────────────────────────────────────────

/// A unified record for a single query type, holding metadata, cache,
/// and active-key state together.
///
/// Analogous to Rust's `QueryVTable` in `plumbing.rs`.
#[derive(Debug)]
pub(crate) struct QueryVTable<Q: Query> {
    pub descriptor: QueryDescriptor,
    pub cache: Q::Cache,
    /// Stack of active keys for cycle detection.
    /// Uses the actual `Q::Key` type (with `Eq` comparison) instead of
    /// Debug strings, so cycle detection is exact and avoids the cost
    /// of `format!` on every query entry.
    pub active: Vec<Q::Key>,
}

impl<Q: Query> QueryVTable<Q> {
    pub fn new() -> Self {
        QueryVTable {
            descriptor: Q::descriptor(),
            cache: Q::Cache::default(),
            active: Vec::new(),
        }
    }

    /// Try to enter the query with the given key.
    /// Returns `Err(QueryCycleError)` if the key is already active
    /// (cycle) or if the query is depth-limited and the stack is too deep.
    pub fn enter(&mut self, key: &Q::Key) -> Result<(), QueryCycleError> {
        if let Some(_prev) = self.active.iter().find(|k| *k == key) {
            // Build a full call stack from the active keys.
            let mut stack: Vec<String> = self.active.iter().map(|k| format!("{:?}", k)).collect();
            stack.push(format!("{:?}", key)); // the repeated key
            return Err(QueryCycleError {
                query_name: self.descriptor.name,
                message: format!(
                    "query cycle detected: `{}` with key `{:?}`\n  call stack:\n    {}",
                    self.descriptor.name, key,
                    stack.iter().enumerate().map(|(i, s)| format!("  {}: {}", i, s)).collect::<Vec<_>>().join("\n    "),
                ),
                stack,
            });
        }
        if self.descriptor.depth_limit && self.active.len() >= MAX_QUERY_DEPTH {
            let stack: Vec<String> = self.active.iter().map(|k| format!("{:?}", k)).collect();
            return Err(QueryCycleError {
                query_name: self.descriptor.name,
                message: format!(
                    "query depth limit exceeded: `{}` depth {} >= {}\n  call stack:\n    {}",
                    self.descriptor.name,
                    self.active.len() + 1,
                    MAX_QUERY_DEPTH,
                    stack.iter().enumerate().map(|(i, s)| format!("  {}: {}", i, s)).collect::<Vec<_>>().join("\n    "),
                ),
                stack,
            });
        }
        self.active.push(key.clone());
        Ok(())
    }

    pub fn leave(&mut self) {
        self.active.pop();
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
    fn into_query_key(self) -> K { self }
}

/// `usize` → `DefId` (since `DefId` is a newtype around `usize`).
impl IntoQueryKey<crate::hir::types::DefId> for usize {
    #[inline(always)]
    fn into_query_key(self) -> crate::hir::types::DefId {
        crate::hir::types::DefId(self)
    }
}

// ── QuerySystem ──────────────────────────────────────────────────

/// Query execution mode.
///
/// Analogous to Rust's `QueryMode` in `plumbing.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    /// Normal query execution: compute and cache the result.
    Get,
    /// Ensure the query succeeds without returning the value.
    /// The query is computed and cached, but the caller discards the result.
    /// Useful for validation passes that only care about errors.
    EnsureOk,
}

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
pub struct QuerySystem {
    vtables: RwLock<HashMap<TypeId, Box<dyn std::any::Any + Send + Sync>>>,
    /// Dependency graph for incremental re-evaluation.
    pub dep_graph: dep_graph::DepGraph,
    /// Maps `hash(TypeId, key)` → stable `DepNodeIndex`.
    node_map: RwLock<HashMap<u64, dep_graph::DepNodeIndex>>,
    /// Registered query names for debugging / introspection.
    query_names: RwLock<HashMap<TypeId, &'static str>>,
    /// Active keys: tracks which keys are currently being computed by some
    /// thread.  Used for parallel query execution.
    /// Key is `(TypeId, hash_of_key)` — the TypeId eliminates cross-query-type
    /// hash collisions that could cause one query to wait on another's latch.
    active_keys: RwLock<HashMap<(TypeId, u64), ActiveKeyStatus>>,
}

impl QuerySystem {
    pub fn new() -> Self {
        QuerySystem {
            vtables: RwLock::new(HashMap::default()),
            dep_graph: dep_graph::DepGraph::new(),
            node_map: RwLock::new(HashMap::default()),
            query_names: RwLock::new(HashMap::default()),
            active_keys: RwLock::new(HashMap::default()),
        }
    }

    /// Get or create a stable `DepNodeIndex` for a `(query_type, key)` pair.
    fn get_or_create_node<Q: Query>(&self, key: &Q::Key) -> dep_graph::DepNodeIndex {
        let hash = query_key_hash::<Q>(key);
        let mut node_map = self.node_map.write().unwrap();
        if let Some(&node) = node_map.get(&hash) {
            return node;
        }
        let node = self.dep_graph.allocate_node_index();
        node_map.insert(hash, node);
        node
    }

    pub fn get<Q: Query>(&self, key: impl IntoQueryKey<Q::Key>, provider: &dyn QueryProvider) -> Result<Q::Value, QueryCycleError> {
        let key = key.into_query_key();

        // Register query name for introspection (first call only).
        self.query_names.write().unwrap().entry(TypeId::of::<Q>()).or_insert_with(|| Q::descriptor().name);

        // Get or create a stable DepNodeIndex for this (query_type, key).
        let node_index = self.get_or_create_node::<Q>(&key);

        // Check if the node is green (up-to-date) and cached.
        if self.dep_graph.is_green(node_index) {
            // Check cache.
            let vtables = self.vtables.read().unwrap();
            if let Some(boxed) = vtables.get(&TypeId::of::<Q>()) {
                let vtable = boxed.downcast_ref::<QueryVTable<Q>>().unwrap();
                if !vtable.descriptor.eval_always {
                    if let Some(value) = vtable.cache.lookup(&key) {
                        // Record that the current task (if any) read this node.
                        self.dep_graph.read(node_index);
                        return Ok(value);
                    }
                }
            }
        }

        // Get or create the vtable for cycle detection.
        let mut vtables = self.vtables.write().unwrap();

        // Check if another thread is already computing this key.
        // (Parallel query execution: wait on the latch instead of re-computing.)
        let akey = active_key::<Q>(&key);
        {
            let mut active = self.active_keys.write().unwrap();
            match active.get(&akey) {
                Some(ActiveKeyStatus::Started(job)) => {
                    // Another thread is computing this key.  Wait for it.
                    let latch = job.latch.clone();
                    drop(active);
                    drop(vtables);
                    latch.wait_on().map_err(|_| QueryCycleError {
                        query_name: Q::descriptor().name,
                        message: format!("query poisoned: `{}` with key `{:?}`", Q::descriptor().name, key),
                        stack: vec![format!("{:?}", key)],
                    })?;
                    // Now the result should be in the cache.
                    let vtables = self.vtables.read().unwrap();
                    if let Some(boxed) = vtables.get(&TypeId::of::<Q>()) {
                        let vtable = boxed.downcast_ref::<QueryVTable<Q>>().unwrap();
                        if let Some(value) = vtable.cache.lookup(&key) {
                            return Ok(value);
                        }
                    }
                    return Err(QueryCycleError {
                        query_name: Q::descriptor().name,
                        message: format!("query result not found after waiting: `{}` with key `{:?}`", Q::descriptor().name, key),
                        stack: vec![format!("{:?}", key)],
                    });
                }
                Some(ActiveKeyStatus::Poisoned) => {
                    return Err(QueryCycleError {
                        query_name: Q::descriptor().name,
                        message: format!("query previously panicked: `{}` with key `{:?}`", Q::descriptor().name, key),
                        stack: vec![format!("{:?}", key)],
                    });
                }
                None => {
                    // We'll be the one computing this key.
                    let job = job::QueryJob::new(node_index);
                    active.insert(akey, ActiveKeyStatus::Started(job));
                }
            }
        }
        let vtable: &mut QueryVTable<Q> = vtables
            .entry(TypeId::of::<Q>())
            .or_insert_with(|| Box::new(QueryVTable::<Q>::new()))
            .downcast_mut::<QueryVTable<Q>>()
            .expect("vtable type mismatch: the query system's internal type registry is corrupted — this is a compiler bug");

        // Enter the query: cycle detection + depth_limit (if configured).
        vtable.enter(&key)?;

        // Start tracking dependencies for this node.
        self.dep_graph.start_task(node_index);

        // Drop the mutable borrows so that compute can use QuerySystem.
        drop(vtables);

        // Compute the value, wrapped in catch_unwind for panic safety.
        // If the compute panics, we must clean up:
        //   - active_keys (signal Poisoned to waiting threads)
        //   - vtable active stack (leave)
        //   - dep_graph task stack (finish_task)
        // Otherwise waiting threads would deadlock and subsequent queries
        // would get false-positive cycle detection.
        let compute_result = panic::catch_unwind(AssertUnwindSafe(|| {
            Q::compute(&key, provider)
        }));

        // Clean up dep_graph task stack (must happen regardless of panic).
        self.dep_graph.finish_task();

        // Record that the current task (the caller) reads this node.
        if self.dep_graph.current_node().is_some() {
            self.dep_graph.read(node_index);
        }

        // Re-borrow the vtable to cache and deactivate.
        let mut vtables = self.vtables.write().unwrap();
        let vtable: &mut QueryVTable<Q> = vtables
            .get_mut(&TypeId::of::<Q>())
            .expect("vtable not found: vtable was created earlier in this function but get_mut returned None")
            .downcast_mut::<QueryVTable<Q>>()
            .expect("vtable type mismatch: the query system's internal type registry is corrupted — this is a compiler bug");
        vtable.leave();

        // Signal completion or poisoning to any waiting threads.
        let hash = query_key_hash::<Q>(&key);
        let mut active = self.active_keys.write().unwrap();

        // Handle the computation result.
        let value = match compute_result {
            Ok(v) => v,
            Err(panic_payload) => {
                // Mark the query as poisoned so waiting threads get an error.
                active.insert(akey, ActiveKeyStatus::Poisoned);
                drop(active);
                drop(vtables);
                // Resume the panic — the compiler will abort or catch it higher up.
                panic::resume_unwind(panic_payload);
            }
        };

        // Normal completion: remove from active keys and signal waiters.
        if let Some(entry) = active.remove(&akey) {
            if let ActiveKeyStatus::Started(job) = entry {
                job.signal_complete();
            }
        }
        drop(active);

        // Cache the result (unless eval_always).
        if !vtable.descriptor.eval_always {
            vtable.cache.insert(key, value.clone());
        }

        // Mark the node as green after successful re-computation.
        if !vtable.descriptor.eval_always {
            self.dep_graph.mark_green(node_index);
        }

        Ok(value)
    }

    /// Record that the current query task depends on the given node.
    pub fn record_dep(&self, other: dep_graph::DepNodeIndex) {
        self.dep_graph.read(other);
    }

    /// Invalidate a specific node in the dependency graph.
    pub fn invalidate_node(&self, node: dep_graph::DepNodeIndex) {
        self.dep_graph.invalidate(node);
    }

    /// Invalidate a cache entry for a specific query key.
    pub fn invalidate<Q: Query>(&self, key: &Q::Key) {
        let hash = query_key_hash::<Q>(key);
        if let Some(&node) = self.node_map.read().unwrap().get(&hash) {
            self.dep_graph.invalidate(node);
        }
        let mut vtables = self.vtables.write().unwrap();
        if let Some(vtable) = vtables.get_mut(&TypeId::of::<Q>()) {
            let vtable = vtable.downcast_mut::<QueryVTable<Q>>()
                .expect("vtable type mismatch: the query system's internal type registry is corrupted — this is a compiler bug");
            vtable.cache.remove(key);
        }
    }

    pub fn clear<Q: Query>(&self) {
        let mut vtables = self.vtables.write().unwrap();
        if let Some(vtable) = vtables.get_mut(&TypeId::of::<Q>()) {
            let vtable = vtable.downcast_mut::<QueryVTable<Q>>()
                .expect("vtable type mismatch: the query system's internal type registry is corrupted — this is a compiler bug");
            vtable.cache.clear();
        }
    }

    pub fn clear_all(&self) {
        self.vtables.write().unwrap().clear();
        self.node_map.write().unwrap().clear();
        self.active_keys.write().unwrap().clear();
        self.dep_graph.reset();
    }

    /// List all registered query names (for debugging).
    pub fn list_queries(&self) -> Vec<&'static str> {
        self.query_names.read().unwrap().values().copied().collect()
    }

    /// Execute a query in `EnsureOk` mode: compute and cache the result,
    /// but discard the value.  Returns `Err(QueryCycleError)` if the query
    /// fails (cycle, depth limit, etc.).
    ///
    /// This is useful for validation passes where you only care whether
    /// the query completes without errors, not what the result is.
    /// Analogous to Rust's `tcx.ensure_ok().$query(..)`.
    pub fn ensure_ok<Q: Query>(&self, key: impl IntoQueryKey<Q::Key>, provider: &dyn QueryProvider) -> Result<(), QueryCycleError> {
        self.get::<Q>(key, provider).map(|_| ())
    }

    /// Execute multiple queries with the same key type in parallel using
    /// rayon.  Returns a vector of results in the same order as the input
    /// keys.
    ///
    /// Each key is processed by a separate rayon task.  The `QuerySystem`
    /// is thread-safe (RwLock + thread-local task stacks), so concurrent
    /// access is safe.  If two threads request the same key, one will wait
    /// on the `QueryLatch` while the other computes it.
    pub fn par_get<Q: Query>(
        &self,
        keys: &[Q::Key],
        provider: &dyn QueryProvider,
    ) -> Vec<Result<Q::Value, QueryCycleError>>
    where
        Q::Key: Sync,
        Q::Value: Send,
    {
        use rayon::prelude::*;
        keys.par_iter()
            .map(|key| self.get::<Q>(key.clone(), provider))
            .collect()
    }
}

impl Default for QuerySystem { fn default() -> Self { Self::new() } }

// ── QueryProvider trait ──────────────────────────────────────────

/// Implemented by types that can provide query computations.
///
/// Queries can call other queries through `query_get::<Q>(provider, key)`.
/// Must be `Sync` to support parallel query execution via rayon.
pub trait QueryProvider: Sync {
    fn query_system(&self) -> &QuerySystem;
}

/// Execute a query through a provider, returning the cached or computed value.
/// This is the primary way for queries to call other queries.
pub fn query_get<Q: Query>(provider: &dyn QueryProvider, key: impl IntoQueryKey<Q::Key>) -> Result<Q::Value, QueryCycleError> {
    provider.query_system().get::<Q>(key, provider)
}

// ── DefaultQueryProvider ──────────────────────────────────────────

/// A concrete provider that holds a reference to a `QuerySystem`.
pub struct DefaultQueryProvider<'a> {
    system: &'a QuerySystem,
}

impl<'a> DefaultQueryProvider<'a> {
    pub fn new(system: &'a QuerySystem) -> Self { DefaultQueryProvider { system } }
}

impl<'a> QueryProvider for DefaultQueryProvider<'a> {
    fn query_system(&self) -> &QuerySystem { self.system }
}

// ── Query handle ──────────────────────────────────────────────────

/// A handle that bundles a `QuerySystem` and a `QueryProvider` together.
pub struct QueryHandle<'a> {
    pub system: &'a QuerySystem,
    pub provider: &'a dyn QueryProvider,
}

impl<'a> QueryHandle<'a> {
    pub fn new(system: &'a QuerySystem, provider: &'a dyn QueryProvider) -> Self {
        QueryHandle { system, provider }
    }
    pub fn get<Q: Query>(&self, key: impl IntoQueryKey<Q::Key>) -> Result<Q::Value, QueryCycleError> {
        self.system.get::<Q>(key, self.provider)
    }
    /// Execute a query in `EnsureOk` mode: compute and cache the result,
    /// but discard the value.  Returns `Err(QueryCycleError)` on failure.
    pub fn ensure_ok<Q: Query>(&self, key: impl IntoQueryKey<Q::Key>) -> Result<(), QueryCycleError> {
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

    impl Query for TestQuery {
        type Key = TestKey;
        type Value = TestValue;
        type Cache = DefaultCache<TestKey, TestValue>;
        fn descriptor() -> QueryDescriptor { QueryDescriptor::new("test", "a test query") }
        fn compute(key: &TestKey, _: &dyn QueryProvider) -> TestValue {
            TestValue(format!("value_{}", key.0))
        }
    }

    struct TestProvider;
    impl QueryProvider for TestProvider {
        fn query_system(&self) -> &QuerySystem {
            panic!("TestProvider does not have a QuerySystem")
        }
    }

    #[test]
    fn test_default_cache() {
        let cache = DefaultCache::new();
        let key = TestKey(1);
        let value = TestValue("hello".into());
        assert!(cache.lookup(&key).is_none());
        cache.insert(key.clone(), value.clone());
        assert_eq!(cache.lookup(&key), Some(value));
    }

    #[test]
    fn test_default_cache_lru() {
        let cache = DefaultCache::with_capacity(2);
        cache.insert(TestKey(1), TestValue("a".into()));
        cache.insert(TestKey(2), TestValue("b".into()));
        assert_eq!(cache.lookup(&TestKey(1)), Some(TestValue("a".into())));
        cache.insert(TestKey(3), TestValue("c".into()));
        assert!(cache.lookup(&TestKey(1)).is_some(), "key 1 was accessed, should survive");
        assert!(cache.lookup(&TestKey(2)).is_none(), "key 2 was never re-accessed, should be evicted");
        assert!(cache.lookup(&TestKey(3)).is_some(), "key 3 was just inserted");
    }

    #[test]
    fn test_single_cache() {
        let cache = SingleCache::new();
        assert!(cache.lookup(&()).is_none());
        cache.insert((), TestValue("singleton".into()));
        assert_eq!(cache.lookup(&()), Some(TestValue("singleton".into())));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_def_id_cache() {
        let cache = DefIdCache::new();
        let d0 = crate::hir::types::DefId(0);
        let d1 = crate::hir::types::DefId(1);
        assert!(cache.lookup(&d0).is_none());
        cache.insert(d0, TestValue("zero".into()));
        cache.insert(d1, TestValue("one".into()));
        assert_eq!(cache.lookup(&d0), Some(TestValue("zero".into())));
        assert_eq!(cache.lookup(&d1), Some(TestValue("one".into())));
    }

    #[test]
    fn test_query_system_get() {
        let system = QuerySystem::new();
        struct Q;
        impl Query for Q {
            type Key = TestKey; type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor { QueryDescriptor::new("q", "") }
            fn compute(key: &TestKey, _: &dyn QueryProvider) -> TestValue {
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
            type Key = TestKey; type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor { QueryDescriptor::new("q", "") }
            fn compute(key: &TestKey, _: &dyn QueryProvider) -> TestValue {
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
            type Key = TestKey; type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor { QueryDescriptor::new("q", "") }
            fn compute(key: &TestKey, _: &dyn QueryProvider) -> TestValue {
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

        // Invalidate the node for Q(1) via the dep_graph.
        let hash = query_key_hash::<Q>(&TestKey(1));
        if let Some(&node) = system.node_map.read().unwrap().get(&hash) {
            system.invalidate_node(node);
        }

        // Third call: should be re-computed (same value since Q is deterministic).
        let r3 = system.get::<Q>(TestKey(1), &provider).unwrap();
        assert_eq!(r3, TestValue("computed_1".into()));
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
            type Key = TestKey; type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("eval_always", "always re-computed")
                    .with_eval_always(true)
            }
            fn compute(key: &TestKey, _: &dyn QueryProvider) -> TestValue {
                TestValue(format!("eval_{}_{}", key.0, EVAL_COUNTER.fetch_add(1, Ordering::SeqCst)))
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
            type Key = TestKey; type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor {
                QueryDescriptor::new("deep", "depth-limited query")
                    .with_depth_limit(true)
            }
            fn compute(key: &TestKey, provider: &dyn QueryProvider) -> TestValue {
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
            type Key = TestKey; type Value = TestValue;
            type Cache = DefaultCache<TestKey, TestValue>;
            fn descriptor() -> QueryDescriptor { QueryDescriptor::new("p", "") }
            fn compute(key: &TestKey, _: &dyn QueryProvider) -> TestValue {
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
}