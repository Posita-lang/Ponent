//! # DepGraph — Dependency Tracking for Incremental Re-evaluation
//!
//! Analogous to `rustc_middle::dep_graph`.
//! Records which query executions read which other query executions,
//! forming a dependency graph.  When an input changes, the graph
//! enables incremental re-evaluation.
//!
//! ## Thread safety
//!
//! - `edges`, `rev_edges`, `fingerprints` and `classes` are sharded by
//!   node index: node `idx` lives in `shards[idx % 64]` at compact slot
//!   `idx / 64`, behind one `CachePadded` `RwLock` per shard (the same
//!   striping shape as the query system's 64-way `active_keys`).  A
//!   per-node operation touches exactly one shard; a task's reverse-edge
//!   flush and class-filtered dirtying touch the read nodes' shards one
//!   lock at a time.  Concurrent finishes collide only when they touch
//!   the same shard (the measured global serial point of the pre-shard
//!   design).  No path holds two shard locks at once except `reset`,
//!   which takes them one at a time in ascending order — consistent
//!   order, so no deadlock.
//! - `dirty` is a lock-free chunked bitmap of `Arc`-shared `AtomicU64`
//!   blocks: `is_green` (the hot path, hit on every cached query) reads
//!   it with a single atomic load and takes no lock at all.  Regrowth
//!   only ever APPENDS fresh blocks (old blocks are never copied or
//!   moved), so a dirty mark racing a regrowth stays visible in every
//!   snapshot — no lost mark, no stale cached value (see `dirty`).
//! - `task_stack` is thread-local (`thread_local!`), so each thread
//!   has its own execution context.  This is the same pattern used
//!   by Rust's `DepGraph` for implicit task deps.

use arc_swap::ArcSwap;
use bitflags::bitflags;
use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use rustc_hash::FxHashSet as HashSet;

use super::CachePadded;

// ── DepNodeIndex ─────────────────────────────────────────────────

/// A unique ID for each query execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DepNodeIndex(pub u32);
impl DepNodeIndex {
    pub const ZERO: DepNodeIndex = DepNodeIndex(0);
}

// ── ContentClass ─────────────────────────────────────────────────

bitflags! {
    /// Which kind of source content a query's value is derived from.
    ///
    /// Every `DepNodeIndex` is allocated with exactly one *declared*
    /// class (the `content_class` of the query that owns the node).
    /// Classes are a *declaration and enforcement* mechanism, not a
    /// differential-invalidation mechanism: fingerprints are whole-node,
    /// so a change in any class flips the node's fingerprint and dirtying
    /// must always consider every real reader (see `publish` — it does
    /// NOT filter).  The classes buy three things:
    ///
    /// 1. **Documentation** — each query states what content it may
    ///    consume.
    /// 2. **Debug bookkeeping** ([`DepGraph::read`]) — a reader whose
    ///    declared class does not cover the class of a node it reads is
    ///    a design bug (e.g. typeck reading the doc channel); caught in
    ///    debug builds, free in release.
    /// 3. **Scoped invalidation API** ([`DepGraph::dirty_readers_by_class`])
    ///    — mark only the readers of a given class, for tooling that
    ///    invalidates one channel deliberately (capsa).
    ///
    /// Edges are deliberately stored WITHOUT a per-edge class: a reader's
    /// class is reachable via the reader node's own declared class
    /// (`classes`), so a per-edge mask would duplicate state while
    /// enabling nothing — per-class fingerprints (the only thing that
    /// could make masks drive differential dirtying) are deliberately
    /// not kept.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ContentClass: u8 {
        /// Program semantics: tokens, items, types, bodies.
        const SEMANTIC = 1 << 0;
        /// Documentation text harvested from the doc channel.
        const DOC = 1 << 1;
        /// Whitespace/comments trivia (layout, formatting).
        const TRIVIA = 1 << 2;
        /// Byte offsets / line numbers / spans.  Never hashed into a
        /// query value's fingerprint (stable-hash audit) —
        /// trivia edits must not invalidate semantic queries.
        const POSITION = 1 << 3;
    }
}

impl Default for ContentClass {
    fn default() -> Self {
        // Almost every query is a semantic query; `QueryDescriptor::new`
        // defaults its `content_class` to this.
        ContentClass::SEMANTIC
    }
}

// ── Durability ───────────────────────────────────────────────────

/// How rarely a query's *inputs* are expected to change — the
/// change-frequency axis, orthogonal to [`ContentClass`] (the
/// content-kind axis).  Salsa-style trusted-input contract:
///
/// - `LOW` (default): may change at any time (user source, editor
///   state).  A LOW mark invalidates every durability fast path.
/// - `MEDIUM`: changes rarely (dependency manifests, config files).
/// - `HIGH`: effectively immutable within a session (stdlib, prelude,
///   builtins).  A HIGH node's fast path survives LOW/MEDIUM marks.
///
/// # Trusted-input-only (opt-in)
///
/// The durability fast path is the one serve path in this system that
/// skips transitive verification BY DESIGN.  Its soundness rests on the
/// `min(declared, reads)` aggregation: a node is only ever treated as
/// HIGH/MEDIUM when every query it transitively read was declared at
/// least that durable — so a LOW-level change cannot reach it.  A query
/// declared HIGH whose value actually depends on LOW-durability inputs
/// is a silent-stale-read bug of the caller's making, the same class of
/// accident this dep graph's defense tests otherwise close.  Declare
/// HIGH/MEDIUM only for inputs the host guarantees immutable per
/// session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Durability {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2,
}

impl Durability {
    #[inline]
    fn from_bits(bits: u64) -> Durability {
        match bits {
            1 => Durability::MEDIUM,
            2 => Durability::HIGH,
            _ => Durability::LOW,
        }
    }

    /// Number of per-level counters: durability `d` bumps levels
    /// `0..=d`, so a durability-`d` node's fast path is invalidated
    /// exactly by marks of durability `>= d`.
    pub(crate) const LEVELS: usize = 3;
}

impl Default for Durability {
    fn default() -> Self {
        // Every existing query keeps today's exact behavior: LOW bumps
        // the LOW counter on every mark, so the per-level fast path
        // degenerates to the existing `verified_gen >= dirty_generation`
        // check.
        Durability::LOW
    }
}

// ── TaskDeps ─────────────────────────────────────────────────────

/// Records which `DepNodeIndex` values are read during a single
/// query's execution.
///
/// Typical tasks read a handful of nodes, so the reads live inline in a
/// `SmallVec` — a heap hash set per query execution is pure overhead
/// (Perf Book §8.1.5).  `spilled` is materialised only past
/// `LINEAR_DEDUP_MAX` reads so that wide queries keep O(1) dedup instead
/// of degrading the linear scan to O(k^2).
#[derive(Debug)]
pub struct TaskDeps {
    pub current_node: DepNodeIndex,
    /// Identity of the `QuerySystem` whose `start_task` pushed this frame.
    /// A frame may only record reads of nodes belonging to the SAME
    /// system: a foreign index flushed into this system's shards becomes
    /// a phantom node (its slot here is unallocated and may later be
    /// aliased by a fresh allocation) and replaces the true dependency
    /// edge, so invalidations on the owning system never reach this
    /// graph — permanent staleness: a phantom foreign index overwrites a
    /// true dependency edge, so invalidations from the owning system never
    /// reach this graph.
    system_id: usize,
    reads: SmallVec<[DepNodeIndex; 8]>,
    /// Parallel to `reads`: each read's dirty epoch captured at `read()`
    /// time — the read's "start revision".  The finish re-checks these
    /// after its rev-edge flush: if a read node was dirtied during this
    /// task's compute, the freshly published value may encode a pre-
    /// invalidation snapshot that the publish-time fan-out could not
    /// reach (the reader's reverse edge did not yet exist), so the
    /// reader must be restored red instead of being served green stale.
    read_epochs: SmallVec<[u64; 8]>,
    spilled: Option<HashSet<DepNodeIndex>>,
}

/// Above this many reads, dedup switches from a linear scan to a hash set.
const LINEAR_DEDUP_MAX: usize = 64;

impl TaskDeps {
    pub fn new(current_node: DepNodeIndex, system_id: usize) -> Self {
        TaskDeps {
            current_node,
            system_id,
            reads: SmallVec::new(),
            read_epochs: SmallVec::new(),
            spilled: None,
        }
    }

    pub fn read(&mut self, other: DepNodeIndex, epoch: u64) {
        if other == self.current_node {
            return;
        }
        if let Some(spilled) = self.spilled.as_mut() {
            if spilled.insert(other) {
                self.reads.push(other);
                self.read_epochs.push(epoch);
            }
            return;
        }
        if self.reads.contains(&other) {
            return;
        }
        self.reads.push(other);
        self.read_epochs.push(epoch);
        if self.reads.len() >= LINEAR_DEDUP_MAX {
            self.spilled = Some(self.reads.iter().copied().collect());
        }
    }

    /// Consume the recorded reads together with their start-revision
    /// epochs (parallel vectors, same order).  Deliberately unordered:
    /// every consumer feeds an unordered `HashSet` (`edges` /
    /// `rev_edges`), so sorting here was O(k log k) of pure waste on
    /// every query execution.  Sort at a serialization boundary if
    /// deterministic order is ever needed.
    pub fn into_reads_with_epochs(self) -> (SmallVec<[DepNodeIndex; 8]>, SmallVec<[u64; 8]>) {
        (self.reads, self.read_epochs)
    }
}

// ── Thread-local task stack ──────────────────────────────────────

// T3 test hooks (cfg-test): park a finishing task inside
// `finish_task_and_publish` between the in-section colour decision and
// the rev-edge flush — the window the in-section suspect check closes.
// Keyed by NODE INDEX (0 = disabled); the dep graph has no query-type
// identity, so unlike the d17/d21 hooks the target is an index, not a
// name pointer.
#[cfg(test)]
pub(crate) static D23_ARRIVED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
pub(crate) static D23_PARK: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
pub(crate) static D23_TARGET: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
// d24's OWN hook slot set: the parallel test harness runs every test in
// ONE process, so two tests sharing the D23 statics would cross-talk
// (one test's arm/release clobbering the other's park window).  Each
// finish-window test gets its own (target, park, arrived) triple; see
// `test_finish_hook`.
#[cfg(test)]
pub(crate) static D24_ARRIVED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
pub(crate) static D24_PARK: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
pub(crate) static D24_TARGET: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Test-only finish hook (T3 / d24): park AFTER the in-section colour
/// decision but BEFORE the rev-edge flush — exactly the window the
/// in-section suspect check (T3) and the per-node `finishing` flag
/// (d24) close.  Keyed by node index (the dep graph does not know query
/// types).  Consults every slot set above; a node matches at most one.
/// No shard lock is held at the park site, so the park cannot block
/// concurrent getters.
#[cfg(test)]
fn test_finish_hook(node: DepNodeIndex) {
    for (target, park, arrived) in [
        (&D23_TARGET, &D23_PARK, &D23_ARRIVED),
        (&D24_TARGET, &D24_PARK, &D24_ARRIVED),
    ] {
        let t = target.load(Ordering::SeqCst);
        if t != 0 && t == node.0 {
            arrived.store(true, Ordering::SeqCst);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while park.load(Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    break;
                }
                std::thread::yield_now();
            }
            return;
        }
    }
}

thread_local! {
    /// Per-thread stack of active `TaskDeps`.  Each query thread has its
    /// own execution context, so the task stack is thread-local rather
    /// than shared.  This is the same pattern as Rust's implicit task
    /// deps (`TaskDepsRef` in `dep_graph/mod.rs`).
    static TASK_STACK: std::cell::RefCell<Vec<TaskDeps>> = const { std::cell::RefCell::new(Vec::new()) };
}

// ── DepGraph ─────────────────────────────────────────────────────

/// Number of shards.  Power of two for fast `&`-modulo; 64 matches the
/// query system's `active_keys` striping (Perf Book §13.7.2): `rayon`
/// uses every core, and with 8 shards a 32-thread miss storm averages 4
/// threads per lock word.
const DEP_SHARDS: usize = 64;

/// One shard of the dep graph: the compact slots of every node whose
/// index ≡ shard (mod `DEP_SHARDS`).  Node `idx` → shard `idx % 64`,
/// slot `idx / 64`, so each shard's vectors stay ~1/64th the length of
/// the old global vectors.
#[derive(Debug)]
struct DepShard {
    /// Inline edge lists (Perf Book §8.1.5): typical nodes read 1–8
    /// other nodes, so a `SmallVec<[u32; 4]>` keeps the edges in the
    /// shard vector itself — no per-node heap hash set, no pointer
    /// chase in `dirty_readers` scans.  Dedup happens on write (O(k)
    /// linear probe, k ≤ a handful) so repeated recomputes of a node
    /// cannot grow the list unboundedly the way blind `extend` would.
    /// Duplicate readers would be semantically harmless (`mark_dirty`
    /// is an idempotent atomic bit) — the dedup is a memory bound, not
    /// a correctness requirement.
    edges: Vec<SmallVec<[DepNodeIndex; 4]>>,
    rev_edges: Vec<SmallVec<[DepNodeIndex; 4]>>,
    fingerprints: Vec<Option<u64>>,
    /// Declared `ContentClass` of each node (single bit, fixed at
    /// allocation from the owning query's descriptor).  Class is a
    /// property of the NODE, not of an edge: any reader's class is one
    /// lookup away via this vector, so edges stay plain `DepNodeIndex`
    /// pairs and the hot read path stays lock-free.
    classes: Vec<ContentClass>,
    /// Verified generation stamp for transitive green verification.
    /// When a node's transitive dependencies are verified green, we
    /// stamp it with the current `dirty_generation`.  Subsequent hits
    /// can skip re-walking edges as long as `verified_gen >=
    /// dirty_generation`.  This keeps steady-state hits O(1).
    verified_gen: Vec<u64>,
    /// Per-level durability stamp: when a node's closure verifies green,
    /// this is set to `rev[node.dur]` (the node's durability at stamp
    /// time).  The durability fast path serves without re-walking while
    /// `verified_lvl >= rev[node.dur]` — see the `rev` field.  A node
    /// whose durability later lowers (finish-time aggregation) compares
    /// its old HIGH stamp against a faster-moving lower counter, which
    /// can only FAIL the fast path — the conservative direction.
    verified_lvl: Vec<u64>,
    /// Per-node caller-provided input digest, recorded by the digest-gated
    /// invalidation API ([`DepGraph::digest_gate`]).  `None` = no digest
    /// recorded yet (the gate then invalidates conservatively).  Plain
    /// storage under the shard lock: the gate runs under
    /// `mutate_world` quiescence and nothing else reads it, so no
    /// lock-free discipline is needed.
    digests: Vec<Option<u64>>,
    /// Per-node "finish in progress" flag (d24).  Set before a freshly
    /// computed value is staged into the cache and cleared after the
    /// finish tail (post-flush re-check) completes.  A transitive
    /// verifier that reaches a node with this flag set treats it as
    /// dirty: the node's colour is not yet final, and serving the staged
    /// value could hand out a pre-invalidation snapshot — the residual
    /// sub-window `[in-section colour decision .. post-flush re-check]`
    /// where a mark landing on a read node could otherwise be invisible
    /// to a getter whose verify runs inside the window.
    finishing: Vec<AtomicBool>,
}

#[inline(always)]
fn shard_of(node: DepNodeIndex) -> usize {
    (node.0 as usize) & (DEP_SHARDS - 1)
}

#[inline(always)]
fn slot_of(node: DepNodeIndex) -> usize {
    (node.0 as usize) / DEP_SHARDS
}

/// Insert `v` into an inline edge list if not already present.  Edge
/// lists are tiny (typical k ≤ 8), so a linear probe beats hashing and
/// keeps repeated recomputes of a node from growing the list.
#[inline]
fn edge_insert(list: &mut SmallVec<[DepNodeIndex; 4]>, v: DepNodeIndex) {
    if !list.contains(&v) {
        list.push(v);
    }
}

/// Replace a forward-edge list with the node's CURRENT read set and
/// return the stale entries it dropped.
///
/// This replaces the old append-only `edge_extend`: forward edges were
/// the UNION of a node's read sets across ALL recomputations, never the
/// read set of the current one — a reader that stopped depending on an
/// input kept a stale edge to it forever, so changing that input
/// spuriously invalidated the reader (g3), memoization decayed toward
/// eager invalidation, and the monotonically growing graph could grow
/// cycles under an unguarded recursive verifier.
#[inline]
fn reconcile_forward_edges(
    list: &mut SmallVec<[DepNodeIndex; 4]>,
    reads: &[DepNodeIndex],
) -> SmallVec<[DepNodeIndex; 4]> {
    // Current read set, deduplicated, in first-read order.
    let mut new_list: SmallVec<[DepNodeIndex; 4]> = SmallVec::new();
    for &r in reads {
        if !new_list.contains(&r) {
            new_list.push(r);
        }
    }
    // Stale edges: entries of the previous list the current read set no
    // longer contains.
    let mut removed: SmallVec<[DepNodeIndex; 4]> = SmallVec::new();
    for &old in list.iter() {
        if !new_list.contains(&old) {
            removed.push(old);
        }
    }
    *list = new_list;
    removed
}

/// Grow a shard's slot vectors so `slot` is a valid index in all four.
#[inline]
fn ensure_shard_len(shard: &mut DepShard, slot: usize) {
    if slot >= shard.edges.len() {
        shard.edges.resize_with(slot + 1, SmallVec::new);
        shard.rev_edges.resize_with(slot + 1, SmallVec::new);
        shard.fingerprints.resize_with(slot + 1, || None);
        shard.classes.resize_with(slot + 1, ContentClass::default);
        shard.verified_gen.resize_with(slot + 1, || 0);
        shard.verified_lvl.resize_with(slot + 1, || 0);
        shard.digests.resize_with(slot + 1, || None);
        shard.finishing.resize_with(slot + 1, || AtomicBool::new(false));
    }
}

/// Debug-only: a SEMANTIC-declared task must only depend on SEMANTIC
/// nodes.  Typeck reading the doc or trivia channel is a design bug — it
/// couples semantic re-verification to prose/layout edits even when the
/// value's fingerprint is unaffected.  Release builds never check.
/// (Checked at the rev-edge flush rather than in `read`: the read nodes'
/// shard locks are already being taken there, so the hot read path
/// stays untouched.)
#[cfg(debug_assertions)]
fn assert_semantic_reads(
    reads: &[DepNodeIndex],
    shard: &DepShard,
    task_node: DepNodeIndex,
    task_class: ContentClass,
) {
    if task_class != ContentClass::SEMANTIC {
        return;
    }
    for &r in reads {
        if let Some(&c) = shard.classes.get(slot_of(r)) {
            assert!(
                c == ContentClass::SEMANTIC,
                "semantic node {task_node:?} recorded a dependency on a {c:?} node {r:?}: \
                 semantic queries must not read the doc/trivia/position channels"
            );
        }
    }
}

/// Blocks of the dirty bitmap allocated up front: 64 blocks × 4096 bits
/// covers 262144 nodes before the first regrowth.
const DIRTY_INIT_BLOCKS: usize = 64;

/// One 4096-node bit block of the dirty bitmap: 64 `AtomicU64` words ×
/// 64 bits.  Blocks are individually `Arc`-allocated so that regrowth
/// only ever APPENDS: existing blocks are never copied, rewritten or
/// moved, which is what makes marks racing a regrowth safe (see
/// [`DepGraph::dirty`]).
type DirtyBlock = [AtomicU64; 64];

#[inline]
fn zero_block() -> Arc<DirtyBlock> {
    Arc::new([const { AtomicU64::new(0) }; 64])
}

/// Which block of the dirty bitmap owns `idx`: one block per 4096 nodes
/// (64 words × 64 bits).
#[inline]
fn block_of(idx: u32) -> usize {
    (idx >> 12) as usize
}

/// Sort `reads` — and, when provided, the parallel `read_epochs` vector —
/// by shard, applying the IDENTICAL permutation to both.
///
/// The finish-side late-fan-out re-check (`finish_task_and_publish`)
/// compares `epoch_of(reads[i])` against `read_epochs[i]` POSITIONALLY;
/// the two vectors are parallel by construction (`TaskDeps::read` pushes
/// them in lockstep).  `flush_rev_edges` groups reads by shard, which
/// requires shard-sorted input — sorting `reads` alone silently scrambles
/// the pairing (epoch of one node compared against another node's
/// captured revision), turning the re-check into a false-positive
/// generator: every task with ≥2 reads whose captured epochs differ is
/// re-dirtied immediately after publishing, and every spurious 0→1 mark
/// bumps `dirty_generation`, invalidating every `verified_gen` stamp in
/// the system.  Sorting both vectors with one permutation keeps the
/// pairing and still yields the shard-sorted order `flush_rev_edges`
/// needs.
fn sort_reads_by_shard(
    reads: &mut SmallVec<[DepNodeIndex; 8]>,
    mut epochs: Option<&mut SmallVec<[u64; 8]>>,
) {
    if reads.len() <= 1 {
        return; // nothing to permute; pairing trivially intact
    }
    let mut order: SmallVec<[usize; 8]> = (0..reads.len()).collect();
    order.sort_by_key(|&i| shard_of(reads[i]));
    let sorted_reads: SmallVec<[DepNodeIndex; 8]> = order.iter().map(|&i| reads[i]).collect();
    *reads = sorted_reads;
    if let Some(epochs) = epochs {
        let sorted_epochs: SmallVec<[u64; 8]> = order.iter().map(|&i| epochs[i]).collect();
        *epochs = sorted_epochs;
    }
}

/// Which `AtomicU64` word within a block owns `idx`.
#[inline]
fn word_of(idx: u32) -> usize {
    ((idx >> 6) & 63) as usize
}

#[inline]
fn mask_of(idx: u32) -> u64 {
    1u64 << (idx & 63)
}

/// One block of per-node dirty epochs: one `AtomicU64` per node, 4096
/// nodes per block (same blocking as the dirty bitmap, so `block_of`
/// indexes both).  Grown in lockstep with `dirty` under the same
/// append-only discipline: blocks are `Arc`-shared and only ever
/// appended, so a bump racing a regrowth lands on a block every later
/// snapshot still points at.
type EpochBlock = [AtomicU64; 4096];

#[inline]
fn zero_epoch_block() -> Arc<EpochBlock> {
    Arc::new([const { AtomicU64::new(0) }; 4096])
}

/// Which `AtomicU64` within an [`EpochBlock`] owns `idx`.
#[inline]
fn epoch_word_of(idx: u32) -> usize {
    (idx & 4095) as usize
}

/// One epoch word packs the node's per-node `Durability` in the top two
/// bits and its value-revision counter in the low 62: durability is
/// read lock-free on the same word `bump_epoch` already touches, so
/// `mark_dirty` can pick the per-level counters to invalidate without
/// taking a shard lock (it is called under shard write locks at some
/// fan-out sites — a shard read there would self-deadlock).  An epoch
/// would have to wrap 2^62 bumps for the packing to corrupt, which no
/// real session reaches.
const EPOCH_MASK: u64 = (1u64 << 62) - 1;
const DUR_SHIFT: u64 = 62;

#[inline]
fn dur_bits(d: Durability) -> u64 {
    (d as u64) << DUR_SHIFT
}

#[inline]
fn word_dur(word: u64) -> Durability {
    Durability::from_bits(word >> DUR_SHIFT)
}

/// The dependency graph: records which query execution read which
/// other query executions, and tracks which nodes are dirty.
///
/// Maintains both forward edges (node -> reads) and reverse edges
/// (read -> readers) so that `invalidate()` can find all transitive
/// readers in O(1) per dirty node rather than O(N) per scan.
///
/// Thread-safe: `edges` / `rev_edges` / `fingerprints` / `classes` are
/// striped across `DEP_SHARDS` `CachePadded` `RwLock`s by node index, so
/// concurrent task finishes collide only when they touch the same shard
/// (the measured serial point of the old single-`RwLock` design); no
/// path ever holds two shard locks at once.
///
/// `dirty` is deliberately NOT inside any shard: it is a lock-free
/// bitmap (see [`DepGraph::is_green`]), because `is_green` runs on every
/// single cache hit and any lock there serialised the whole system (Perf
/// Book §13.1).
pub struct DepGraph {
    next_index: AtomicU32,
    /// One `CachePadded` lock per shard, exactly like the query system's
    /// `active_keys`: an unpadded `Vec<RwLock<..>>` puts neighbouring
    /// lock words on the same cache line, so two threads locking
    /// *different* shards would still invalidate each other's line —
    /// silently undoing the striping (Perf Book §13.7.3).
    shards: Vec<CachePadded<RwLock<DepShard>>>,
    /// Serializes bitmap regrowths so two growers cannot race their
    /// clone-append-store cycles and drop a block range.  A leaf lock:
    /// never held while acquiring a shard lock (allocation holds its
    /// node's shard lock when it grows, but growth touches nothing
    /// sharded), so no ordering constraints.
    grow_lock: Mutex<()>,
    /// Lock-free dirty bitmap, one bit per `DepNodeIndex`: `Arc`-shared
    /// 64-bit blocks behind an `ArcSwap`.
    ///
    /// ## Growth discipline — this is what makes it correct
    ///
    /// Pre-sharding, regrowth *snapshots and copied* the old blocks, and
    /// every mutator held the single global `RwLock`, which excluded
    /// regrowth entirely — a mark racing the copy was impossible, and a
    /// lost dirty bit (a stale cached query value, P0 class) could not
    /// happen.  Sharding removed that exclusion, so the bitmap is now
    /// APPEND-ONLY: [`DepGraph::grow_dirty`] clones the outer vec (cheap
    /// `Arc` clones) and appends fresh zero blocks; existing blocks are
    /// never copied, rewritten or moved.  A `mark_dirty`/`clear_dirty`
    /// therefore works on ANY snapshot — a mark racing a regrowth lands
    /// on an `Arc`-shared block the new snapshot still points at, so it
    /// can never be lost.  A node's block is appended by its own
    /// `allocate_node_index`, which happens-before any mark of that node
    /// (the index only escapes the allocator through the node-map shard
    /// lock), so marks always find their block.  A missing block is only
    /// possible after `reset()`, where the whole graph state (caches,
    /// active keys) was cleared anyway and dropping the mark is the safe
    /// direction.  `is_green` — the hot path — takes no lock at all.
    dirty: ArcSwap<Vec<Arc<DirtyBlock>>>,
    /// Per-node dirty epochs, one `AtomicU64` per node, blocked and grown
    /// in lockstep with `dirty` (same `block_of` indexing, same
    /// append-only `ArcSwap` discipline — see `dirty` for why that makes
    /// lock-free access racing a regrowth safe).
    ///
    /// This guards against the late-fan-out failure mode: when a query
    /// reads another query for the first time, its reverse edge does not
    /// exist until its finish flushes it.  A republish of the read node
    /// inside that window fans out to every existing reader EXCEPT the
    /// new one.  Each read captures its node's epoch at record time; the
    /// finish re-checks it after the flush — any mark of a read node in
    /// between (the bump is ordered before the bit set) means the
    /// reader's value may encode a pre-invalidation snapshot, and the
    /// reader is restored red instead of being published green stale.
    epochs: ArcSwap<Vec<Arc<EpochBlock>>>,
    /// Global dirty generation counter.  Incremented on every
    /// `invalidate()`.  Nodes verified transitively green are stamped
    /// with this generation; hits skip re-verification when
    /// `node.verified_gen >= dirty_generation`.
    dirty_generation: AtomicU64,
    /// Per-durability-level mark counters.  `rev[l]` counts every 0→1
    /// mark of a node whose durability is `>= l` (a durability-`d` mark
    /// bumps levels `0..=d`; a HIGH change bumps all three).  The
    /// durability fast path serves a verified node without re-walking
    /// when its per-level stamp is `>= rev[node.dur]` — a HIGH-dur
    /// node's stamp survives LOW/MEDIUM marks because they never bump
    /// its level's counter.
    ///
    /// Soundness shape: `rev[d]`'s RMW is the mark's linearization point
    /// FOR LEVEL-`d` FAST SERVES — self-contained, no cross-counter
    /// consistency with `dirty_generation` is required (sharing the
    /// generation's value space would need the per-level record to land
    /// atomically with the bump, which two atomics cannot do; a lagging
    /// record opens a serve-past-the-mark window).  A fast-path reader
    /// that loads `rev[d]` before a mark's add linearizes before the
    /// mark (the same overlap doctrine as the global stamp check); one
    /// that loads after sees the bumped value and walks.  The global
    /// `dirty_generation` mechanism keeps its own linearization point
    /// (its bump) and its own doctrine, unchanged.
    rev: [AtomicU64; Durability::LEVELS],
    /// Id of the owning `QuerySystem` (0 for a standalone/test graph).
    /// Every task frame pushed through this graph is tagged with it, and
    /// [`DepGraph::read`] refuses to record a node of a different system
    /// (see `TaskDeps::system_id`).
    owner_id: usize,
}

impl DepGraph {
    pub fn new() -> Self {
        DepGraph {
            next_index: AtomicU32::new(1),
            shards: (0..DEP_SHARDS)
                .map(|_| {
                    CachePadded(RwLock::new(DepShard {
                        edges: Vec::new(),
                        rev_edges: Vec::new(),
                        fingerprints: Vec::new(),
                        classes: Vec::new(),
                        verified_gen: Vec::new(),
                        verified_lvl: Vec::new(),
                        digests: Vec::new(),
                        finishing: Vec::new(),
                    }))
                })
                .collect(),
            grow_lock: Mutex::new(()),
            dirty: ArcSwap::from_pointee((0..DIRTY_INIT_BLOCKS).map(|_| zero_block()).collect()),
            epochs: ArcSwap::from_pointee(
                (0..DIRTY_INIT_BLOCKS).map(|_| zero_epoch_block()).collect(),
            ),
            dirty_generation: AtomicU64::new(0),
            rev: [const { AtomicU64::new(0) }; Durability::LEVELS],
            owner_id: 0,
        }
    }

    /// A graph owned by the `QuerySystem` with id `owner_id`.  All task
    /// frames pushed through it are tagged so `read` can reject
    /// cross-system edges (see `TaskDeps::system_id`).
    pub fn with_owner(owner_id: usize) -> Self {
        let mut g = Self::new();
        g.owner_id = owner_id;
        g
    }

    /// Grow the bitmap so it covers `idx`.
    ///
    /// APPEND-ONLY: clone the outer vec (cheap `Arc` clones — the blocks
    /// themselves are untouched) and append fresh zero blocks.  Existing
    /// blocks are never copied, rewritten or moved, so a mark/clear/is_green
    /// racing this regrowth stays correct on any snapshot (see `dirty`).
    /// Growers serialize on `grow_lock` so a racing grower cannot drop the
    /// appended block range; growth is rare (the block count doubles), so
    /// this is not a hot path.
    fn grow_dirty(&self, idx: u32) {
        let need = block_of(idx) + 1;
        if self.dirty.load().len() >= need {
            return;
        }
        let _growth = self.grow_lock.lock();
        let cur = self.dirty.load(); // re-load under the lock
        if cur.len() >= need {
            return;
        }
        let new_len = need
            .next_power_of_two()
            .max(cur.len() * 2)
            .max(DIRTY_INIT_BLOCKS);
        // Store the EPOCH blocks first.  `grow_dirty` only ever grows, so
        // the only possible transient skew is `dirty.len() < epochs.len()`;
        // any operation on a not-yet-covered slot then degrades to the
        // conservative path (a missing dirty block makes `mark_dirty` bump
        // `dirty_generation` instead of silently dropping the epoch bump).
        // The reverse order would let a mark of a just-covered slot find its
        // dirty block but miss its epoch block — `bump_epoch` would then
        // silently drop the bump that the finish-side epoch re-check relies
        // on.
        let ecur = self.epochs.load(); // re-load under the lock
        let mut ev: Vec<Arc<EpochBlock>> = Vec::with_capacity(new_len);
        ev.extend(ecur.iter().cloned());
        ev.extend((ecur.len()..new_len).map(|_| zero_epoch_block()));
        self.epochs.store(Arc::new(ev));

        let mut v: Vec<Arc<DirtyBlock>> = Vec::with_capacity(new_len);
        v.extend(cur.iter().cloned());
        v.extend((cur.len()..new_len).map(|_| zero_block()));
        self.dirty.store(Arc::new(v));
    }

    /// Current dirty epoch of `node` — its value's revision counter,
    /// bumped on every dirty mark AND on every changed-fingerprint
    /// republish (`mark_dirty` / `publish` / `finish_task_and_publish`).
    /// `Acquire` load, paired with the `AcqRel` bumps.
    #[inline]
    fn epoch_of(&self, node: DepNodeIndex) -> u64 {
        self.with_epoch_word(node, |w| {
            w.map(|w| w.load(Ordering::Acquire) & EPOCH_MASK).unwrap_or(0)
        })
    }

    /// Run `f` on `node`'s epoch word from the current snapshot, if its
    /// block exists (post-`reset` it may not).  The `ArcSwap` guard must
    /// never escape this frame — the word reference borrows it — so
    /// every access goes through this closure instead of returning the
    /// reference.
    #[inline]
    fn with_epoch_word<T>(
        &self,
        node: DepNodeIndex,
        f: impl FnOnce(Option<&AtomicU64>) -> T,
    ) -> T {
        let snap = self.epochs.load();
        f(snap.get(block_of(node.0)).map(|b| &b[epoch_word_of(node.0)]))
    }

    /// The node's CURRENT durability: its declared durability lowered by
    /// every finish-time `min(declared, reads)` aggregation since.  A
    /// missing block (post-`reset`) reads LOW — the conservative
    /// direction (LOW marks bump the most counters).
    #[inline]
    pub(crate) fn durability_of(&self, node: DepNodeIndex) -> Durability {
        self.with_epoch_word(node, |w| {
            w.map_or(Durability::LOW, |w| word_dur(w.load(Ordering::Acquire)))
        })
    }

    /// Set a freshly allocated node's DECLARED durability (from its
    /// query's descriptor).  Runs once per node before the index escapes
    /// the allocator, so a plain CAS-store cannot race a real conflict.
    pub(crate) fn set_node_durability(&self, node: DepNodeIndex, d: Durability) {
        self.with_epoch_word(node, |w| {
            let Some(w) = w else { return };
            debug_assert_eq!(
                word_dur(w.load(Ordering::Relaxed)),
                Durability::LOW,
                "set_node_durability is allocation-time only — re-stamping would \
                 raise an aggregated durability, which must be monotone downward"
            );
            let mut cur = w.load(Ordering::Relaxed);
            loop {
                let next = (cur & EPOCH_MASK) | dur_bits(d);
                match w.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => break,
                    Err(c) => cur = c,
                }
            }
        });
    }

    /// Lower `node`'s durability to `d` if it is currently higher — the
    /// finish-time aggregation `min(declared, min(reads))`.  Monotone
    /// downward: re-lowering is a no-op, never a raise.
    fn lower_durability_to(&self, node: DepNodeIndex, d: Durability) {
        self.with_epoch_word(node, |w| {
            let Some(w) = w else { return };
            let mut cur = w.load(Ordering::Relaxed);
            loop {
                if word_dur(cur) <= d {
                    return;
                }
                let next = (cur & EPOCH_MASK) | dur_bits(d);
                match w.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => break,
                    Err(c) => cur = c,
                }
            }
        });
    }

    /// Bump `node`'s dirty epoch (its value-revision counter).  Called
    /// from `mark_dirty` BEFORE the bit is set — that ordering is what
    /// makes the finish-side re-check sound (see `mark_dirty`) — and from
    /// the changed-fingerprint publish paths (`publish`,
    /// `finish_task_and_publish`) where a value changes WITHOUT any mark
    /// (the eval-always / direct-publish shape where a value changes
    /// on every request without any dirty mark).  A redundant mark (bit already
    /// set) over-bumps the epoch; harmless: an over-bumped epoch only
    /// makes a concurrent reader's restore check MORE conservative (one
    /// extra recomputation), never less.
    #[inline]
    fn bump_epoch(&self, node: DepNodeIndex) {
        self.with_epoch_word(node, |w| {
            let Some(w) = w else { return };
            // CAS loop instead of `fetch_add`: the durability bits live
            // in the same word and must not absorb the carry.  An
            // epoch-counter overflow into bit 62 is unreachable in
            // practice (see `EPOCH_MASK`).
            let mut cur = w.load(Ordering::Relaxed);
            loop {
                let next = ((cur & EPOCH_MASK).wrapping_add(1) & EPOCH_MASK) | (cur & !EPOCH_MASK);
                match w.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => break,
                    Err(c) => cur = c,
                }
            }
        });
    }

    /// Set the dirty bit for `node`.
    ///
    /// No shard lock: the node's block was appended by its own allocation
    /// (happens-before any mark — see `dirty`), so the block is present in
    /// every snapshot this load can observe.  A missing block is only
    /// possible after `reset()` cleared the whole world — dropping the
    /// mark there is the safe direction (nothing is cached any more).
    #[inline]
    fn mark_dirty(&self, node: DepNodeIndex) {
        if let Some(block) = self.dirty.load().get(block_of(node.0)) {
            // Epoch bump BEFORE the bit set: any Acquire observer of the
            // red bit then also observes the bump (the bump is a prior
            // access of the Release bit-set, so it can never be reordered
            // after it).  A finish-side epoch re-check therefore never
            // misses a mark whose red bit was observable — that ordering
            // is precisely what prevents a fan-out-missed reader from
            // staying green stale.  The bump precedes the 0→1 gate, so
            // a redundant mark (bit already set) over-bumps the epoch:
            // harmless — an over-bumped epoch only makes a concurrent
            // reader's restore check MORE conservative (one extra
            // recomputation), never less.
            self.bump_epoch(node);
            // 0→1 transition gate: only a bit that WAS CLEAR carries new
            // information, so only it invalidates stamps.  A redundant
            // mark (bit already set) would otherwise thrash every
            // verified_gen stamp in the system on each publish fan-out
            // (one reader mark each → O(closure) verification instead of
            // O(1) memoization under write storms).
            let mask = mask_of(node.0);
            let prev = block[word_of(node.0)].fetch_or(mask, Ordering::Release);
            if prev & mask != 0 {
                return; // already red — stamps were invalidated when it turned red
            }
        } else {
            // Block missing (post-`reset`): nothing is cached any more,
            // but treat it conservatively as a new red bit for stamp
            // invalidation.  The epoch is left alone there: any in-flight
            // capture also read the epoch as 0 (`epoch_of` maps a missing
            // block to 0), so captures compare equal and no spurious
            // restore fires.
            self.dirty_generation.fetch_add(1, Ordering::AcqRel);
            return;
        }
        // 0→1: a red bit may postdate any verified_gen stamp: the stamp
        // means "closure was all-green at generation G", so CREATING a
        // red bit must invalidate all stamps.  This closes the soundness
        // hole where a red bit created without a bump — publish with a
        // changed fingerprint after an eval_always recompute, or the
        // dirty_readers{,_by_class} tooling APIs — was invisible to the
        // `vg >= current_gen` fast path and a transitively-stamped cached
        // reader served its stale value forever (g2).  Every transition
        // MUST bump (not batched per fan-out): a concurrent walk can
        // legitimately miss this bit — its Release-set is not ordered
        // before another mark's generation RMW — and an unbumped stamp
        // written by such a walk would never be re-checked.
        //
        // The linearization point of a mark is its `dirty_generation`
        // bump.  A walk that loads the generation before the bump may
        // serve a value that is, in wall-clock time, already invalid —
        // that is a valid linearization (the walk overlaps the marking
        // thread's operation).  A walk that loads the generation after
        // the bump is guaranteed by the release/acquire chain (the bump
        // is `AcqRel` and sequenced after the `Release` bit-set above) to
        // see the bit and to re-walk.  Reversing the order (bump → bit)
        // would break the chain — a walk seeing the bumped generation
        // could miss the bit, stamp with the bumped generation, and the
        // stamp would survive (no new bump) as a permanent stale serve —
        // and is unsound.  Concurrency:
        // a mark racing a walk's generation load may or may not be
        // reflected in that walk; any mark it misses makes the stamp it
        // writes stale relative to FUTURE lookups (the bump lands before
        // the stamp's reader loads the generation on its next visit) —
        // the conservative direction.  Two racing 0→1 marks on one bit
        // may double-bump: harmless over-invalidation.
        self.dirty_generation.fetch_add(1, Ordering::AcqRel);
        // Per-level invalidation (durability): this mark's effect on the
        // level-`l` fast paths is its `rev[l]` add — the linearization
        // point documented on the `rev` field.  Sequenced after the bit
        // set (a fast-path serve never reads the bit, but the global
        // bump-order doctrine stays intact).  A missing block reads LOW
        // below, which bumps only level 0 — the same conservative skew
        // the global bump just took (post-reset nothing is cached).
        let d = self.durability_of(node);
        for l in 0..=(d as usize) {
            self.rev[l].fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Clear the dirty bit for `node`.  Same append-only-snapshot
    /// discipline as [`DepGraph::mark_dirty`].
    #[inline]
    fn clear_dirty(&self, node: DepNodeIndex) {
        if let Some(block) = self.dirty.load().get(block_of(node.0)) {
            block[word_of(node.0)].fetch_and(!mask_of(node.0), Ordering::Release);
        }
    }

    /// Allocate a node with its DECLARED content class (fixed for the
    /// node's lifetime; taken from the owning query's descriptor).
    pub fn allocate_node_index(&self, class: ContentClass) -> DepNodeIndex {
        // `fetch_add` wraps at u32::MAX: a wrapped index would alias a live
        // node (node_to_key overwrite, shared fingerprints/edges).  2^32
        // nodes is far beyond any real session — fail loudly instead of
        // corrupting.
        let prev = self.next_index.fetch_add(1, Ordering::Relaxed);
        assert!(
            prev != u32::MAX,
            "DepNodeIndex address space exhausted ({} nodes allocated)",
            prev as u64 + 1,
        );
        let idx = DepNodeIndex(prev);
        let slot = slot_of(idx);
        {
            let mut shard = self.shards[shard_of(idx)].write();
            self.grow_dirty(idx.0);
            ensure_shard_len(&mut shard, slot);
            // Written unconditionally, NOT via `resize_with(.., || class)`:
            // a concurrent allocator's resize must never leave THIS node's
            // slot holding a foreign class — the class is fixed at
            // allocation and drives the semantic-purity debug checks and
            // class-scoped invalidation.
            shard.classes[slot] = class;
        }
        idx
    }

    /// The declared content class of a node, if the node exists.
    pub fn class_of(&self, node: DepNodeIndex) -> Option<ContentClass> {
        self.shards[shard_of(node)]
            .read()
            .classes
            .get(slot_of(node))
            .copied()
    }

    /// Begin recording reads for a query execution.
    /// Push a new `TaskDeps` onto the current thread's task stack.
    pub fn start_task(&self, node: DepNodeIndex) {
        TASK_STACK.with(|s| s.borrow_mut().push(TaskDeps::new(node, self.owner_id)));
    }

    /// Record that the current task (top of current thread's stack) read
    /// another query.  If no task is active, this is a no-op.
    pub fn read(&self, other: DepNodeIndex) {
        // One borrow_mut instead of borrow → drop → borrow_mut: halves the
        // RefCell traffic on the hit path, which calls read() per node.
        // Deliberately NOT `try_borrow_mut`: silently skipping the push
        // would drop a dependency edge, and a dropped edge yields a stale
        // query result — the silent-staleness failure mode where a waiter
        // never records its dep edge.  A genuine double borrow
        // must keep panicking loudly.
        TASK_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            let Some(task) = stack.last_mut() else {
                return;
            };
            // Cross-system guard: a node index of a DIFFERENT QuerySystem
            // must never be recorded into this system's task frames.
            // Flushed into this system's shards it becomes a phantom node
            // (its slot here is unallocated and may later be aliased by a
            // fresh allocation), and it replaces the true dependency edge,
            // so invalidations on the owning system never reach this graph
            // (permanent staleness).  Debug builds fail loudly; release
            // builds skip the record — the documented "keep provider and
            // executing system paired" contract then degrades to a missing
            // edge, never to graph corruption.
            if task.system_id != self.owner_id {
                #[cfg(debug_assertions)]
                panic!(
                    "cross-system dependency edge: node {other:?} of query system \
                     {owner} recorded into a task frame of query system {frame} \
                     (provider/system pairing violated)",
                    owner = self.owner_id,
                    frame = task.system_id,
                );
                #[cfg(not(debug_assertions))]
                {
                    return;
                }
            }
            // Capture the read node's dirty epoch BEFORE recording: this is
            // the read's "start revision" against which the finish re-checks
            // after the reverse-edge flush.
            let epoch = self.epoch_of(other);
            task.read(other, epoch);
        });
    }

    /// Get the `DepNodeIndex` of the current task (top of current thread's stack).
    #[inline]
    pub fn current_node(&self) -> Option<DepNodeIndex> {
        TASK_STACK.with(|s| s.borrow().last().map(|t| t.current_node))
    }

    /// Test/diagnostic path only: pops the task frame, reconciles the
    /// forward edges and flushes the reverse edges, but publishes NO
    /// fingerprint and runs NO stale-restore — not a valid production
    /// finalize.  Production finalization MUST go through
    /// [`DepGraph::finish_task_and_publish`], which additionally publishes
    /// the fingerprint (or `None` on a panic path) and runs the in-section
    /// colour decision plus the post-flush epoch re-check.
    #[cfg(test)]
    pub fn finish_task(&self) {
        let task = TASK_STACK.with(|s| s.borrow_mut().pop());
        let Some(task) = task else { return };
        let task_node = task.current_node;
        // Diagnostic/test path (no value is published here), so the
        // finish-side stale-restore does not apply; the epochs are still
        // consumed to keep the parallel-vec invariant in one place.
        let (reads, _read_epochs) = task.into_reads_with_epochs();
        let mut reads = reads;

        // This node's shard: forward edges, RECONCILED to the current
        // read set (stale edges removed — see `reconcile_forward_edges`).
        let removed = {
            let tslot = slot_of(task_node);
            let mut shard = self.shards[shard_of(task_node)].write();
            ensure_shard_len(&mut shard, tslot);
            reconcile_forward_edges(&mut shard.edges[tslot], &reads)
        };

        // The reads' shards: reverse edges, one lock per shard.
        sort_reads_by_shard(&mut reads, None);
        self.flush_rev_edges(task_node, &mut reads);
        // Drop the reverse edges of reads we no longer depend on.
        self.prune_rev_edges(task_node, removed);
    }

    /// Remove `task_node` from the reverse-edge lists of `removed` deps —
    /// the counterpart of [`Self::flush_rev_edges`] for edges the node's
    /// CURRENT recomputation no longer has.  One lock per shard, grouped,
    /// never two at once (module invariant); runs after the node-shard
    /// critical section that produced `removed` has been released.
    fn prune_rev_edges(&self, task_node: DepNodeIndex, mut removed: SmallVec<[DepNodeIndex; 4]>) {
        if removed.is_empty() {
            return;
        }
        removed.sort_by_key(|&r| shard_of(r));
        let mut i = 0;
        while i < removed.len() {
            let s = shard_of(removed[i]);
            let start = i;
            while i < removed.len() && shard_of(removed[i]) == s {
                i += 1;
            }
            let group = &removed[start..i];
            let max_slot = group
                .iter()
                .map(|&r| slot_of(r))
                .max()
                .expect("group is non-empty");
            let mut shard = self.shards[s].write();
            ensure_shard_len(&mut shard, max_slot);
            for &r in group {
                if let Some(list) = shard.rev_edges.get_mut(slot_of(r)) {
                    list.retain(|n| *n != task_node);
                }
            }
        }
    }

    /// Insert the reverse edges `read → task_node` into the READ nodes'
    /// shards — one write lock at a time, never two at once (module
    /// invariant).  Reads are grouped by shard so each shard's lock is
    /// taken at most once: with the old single global lock one
    /// acquisition covered all reads; with sharding, acquiring a lock
    /// per read would multiply the CAS traffic by the read count.
    ///
    /// Deliberately unordered across shards: every consumer is an
    /// unordered `HashSet`, so grouping order changes nothing
    /// semantically.
    ///
    /// Takes `reads` by `&mut` (sorted in place) instead of by value: the
    /// caller re-checks the reads' captured dirty epochs AFTER this flush
    /// — the re-check must run after the reverse edges are visible so
    /// any LATER fan-out reaches this task normally.
    fn flush_rev_edges(&self, task_node: DepNodeIndex, reads: &mut SmallVec<[DepNodeIndex; 8]>) {
        // Callers MUST pre-sort `reads` by shard (`sort_reads_by_shard`)
        // so the grouping below takes each shard's lock exactly once — and
        // so any parallel epoch vector stays paired (see
        // `sort_reads_by_shard`).
        #[cfg(debug_assertions)]
        debug_assert!(
            reads.windows(2).all(|w| shard_of(w[0]) <= shard_of(w[1])),
            "flush_rev_edges requires shard-sorted reads"
        );
        let mut i = 0;
        while i < reads.len() {
            let s = shard_of(reads[i]);
            let start = i;
            while i < reads.len() && shard_of(reads[i]) == s {
                i += 1;
            }
            let group = &reads[start..i];
            let max_slot = group
                .iter()
                .map(|&r| slot_of(r))
                .max()
                .expect("group is non-empty");
            let mut shard = self.shards[s].write();
            ensure_shard_len(&mut shard, max_slot);
            for &r in group {
                edge_insert(&mut shard.rev_edges[slot_of(r)], task_node);
            }
        }
    }

    /// DEBUG-ONLY semantic-class validation of a task's read set, run
    /// BEFORE any graph mutation ("validate before mutate").
    ///
    /// The assert used to live inside `flush_rev_edges` — run AFTER the
    /// violating reverse edge had already been inserted, so the panic
    /// unwound with a partially mutated graph AND (originating outside
    /// the compute `catch_unwind`) skipped the caller's active-key
    /// cleanup, stranding waiters on the job latch.  Both halves are
    /// fixed: validation happens here, before `finish_task_and_publish`
    /// mutates anything, and the whole finalize step is guarded in
    /// `QuerySystem::get_shared`.
    ///
    /// Classes are immutable per node (fixed at `allocate_node_index`),
    /// so reading them under a short read lock per shard group is
    /// race-free.
    #[cfg(debug_assertions)]
    fn validate_semantic_reads(&self, task_node: DepNodeIndex, reads: &[DepNodeIndex]) {
        let task_class = self.class_of(task_node).unwrap_or_default();
        if task_class != ContentClass::SEMANTIC {
            return;
        }
        // Group by shard: one read lock per run of equal shard (same
        // batching discipline as `flush_rev_edges`, read instead of
        // write).
        let mut sorted: SmallVec<[DepNodeIndex; 8]> = reads.iter().copied().collect();
        if sorted.len() > 1 {
            sorted.sort_by_key(|&r| shard_of(r));
        }
        let mut i = 0;
        while i < sorted.len() {
            let s = shard_of(sorted[i]);
            let start = i;
            while i < sorted.len() && shard_of(sorted[i]) == s {
                i += 1;
            }
            let group = &sorted[start..i];
            let shard = self.shards[s].read();
            assert_semantic_reads(group, &shard, task_node, task_class);
        }
    }

    /// `finish_task` + `publish` fused, sharded.
    ///
    /// The normal miss path used to take the global write lock twice —
    /// once to flush the task's dependency edges, once to publish the
    /// fingerprint.  Under parallel load that lock is the measured serial
    /// point (thin-value publish scaling saturates at T=4), so the two
    /// sections are fused — now into ONE critical section on the NODE's
    /// own shard (its forward edges, fingerprint and reader dirtying all
    /// live in shard `node % 64`), with the task's reverse-edge flush
    /// taking one lock per READ shard afterwards (never two shard locks
    /// at once — see `flush_rev_edges`).
    ///
    /// `fp: None` means "no value to publish" (the `eval_always` and
    /// panic paths): only the task bookkeeping runs, and NO fingerprint
    /// is written — writing a sentinel would corrupt red-green
    /// verification.  The reverse-edge flush still runs: a panicking
    /// node's dependency edges are just as load-bearing for later
    /// invalidation.
    ///
    /// Ordering inside the node's section: dependency edges first,
    /// fingerprint second — a concurrent reader that observes the green
    /// fingerprint (lock-free `is_green`) is serialized against this
    /// section on the node's own shard, so it already sees the edges.
    ///
    /// Returns `true` when a fingerprint was published AND it changed
    /// (direct readers were marked dirty for lazy re-verification).
    pub fn finish_task_and_publish(&self, node: DepNodeIndex, fp: Option<u64>) -> bool {
        // Pop the task stack outside any shard lock: RefCell work must
        // not run while other threads spin on a shard lock.
        let task = TASK_STACK.with(|s| s.borrow_mut().pop());
        let (task_node, mut reads, mut read_epochs) = match task {
            Some(t) => {
                let node = t.current_node;
                let (reads, epochs) = t.into_reads_with_epochs();
                (Some(node), reads, epochs)
            }
            None => (None, SmallVec::new(), SmallVec::new()),
        };

        // The popped frame MUST belong to `node`: the fused section below
        // writes the task's forward edges into `node`'s shard at
        // `slot_of(task_node)`.  A mismatch in RELEASE would corrupt a
        // foreign shard — that failure mode does not justify a
        // debug-only check, so this is a hard assert (runs once per query
        // execution, never in a hot loop).
        if let Some(tn) = task_node {
            assert!(
                tn == node,
                "finish_task_and_publish: popped task frame does not belong to `node`"
            );
        }

        // ---- validate before mutate (debug) ----
        // The class assertion runs BEFORE the forward-edge reconcile, the
        // reverse-edge insert, the fingerprint write and the reader
        // dirtying — a panicking assert must never leave a partially
        // mutated graph.
        #[cfg(debug_assertions)]
        if let Some(tn) = task_node {
            self.validate_semantic_reads(tn, &reads);
        }

        // ---- durability aggregation: node.dur = min(declared, reads) ----
        // `finish_task_and_publish` already holds this task's read set,
        // so the aggregation costs one lock-free load per read.  Monotone
        // downward: a node never gains durability from a recompute, so a
        // stale-high stamp can only fail the fast path (conservative).
        if !reads.is_empty() {
            let mut min_dur = Durability::HIGH;
            for &r in reads.iter() {
                let d = self.durability_of(r);
                if d < min_dur {
                    min_dur = d;
                }
            }
            self.lower_durability_to(node, min_dur);
        }

        // ---- finish_task part (forward edges) + publish part: ONE
        // critical section on the node's own shard ----
        let mut changed = false;
        // Stale forward edges dropped by the reconciliation below; the
        // pruning of their reverse edges runs AFTER the critical section
        // releases (the removed deps may live in OTHER shards — never
        // two shard locks at once).
        let mut removed: SmallVec<[DepNodeIndex; 4]> = SmallVec::new();
        {
            let mut shard = self.shards[shard_of(node)].write();
            let slot = slot_of(node);
            ensure_shard_len(&mut shard, slot);

            // ---- finish_task part: forward edges, RECONCILED to the
            // current read set (stale edges removed — see
            // `reconcile_forward_edges`).  The removed entries' reverse
            // edges are pruned AFTER this critical section releases (the
            // removed deps may live in OTHER shards — never two shard
            // locks at once).
            removed = if let Some(tn) = task_node {
                let tslot = slot_of(tn);
                ensure_shard_len(&mut shard, tslot);
                reconcile_forward_edges(&mut shard.edges[tslot], &reads)
            } else {
                SmallVec::new()
            };

            // ---- publish part ----
            if let Some(fp) = fp {
                changed = shard.fingerprints[slot].is_some_and(|old| old != fp);
                shard.fingerprints[slot] = Some(fp);
                // NEW: resolve the colour BEFORE the node can be observed
                // green.  A read whose epoch advanced during this compute
                // may have been republished green already (its fan-out
                // missed us because our rev edge is not flushed yet), so
                // the transitive walk of a concurrent getter would find
                // the whole closure green and serve our stale-based value
                // in the [clear_dirty .. late re-check] window.  The
                // in-section check is lock-free (`epoch_of` is an atomic
                // load) and positionally correct: `reads`/`read_epochs`
                // are still the unsorted, parallel-by-construction
                // vectors here.
                //
                // d24: the residual sub-window [this check .. the
                // post-flush re-check] is closed by the per-node
                // `finishing` flag: `get_shared` sets it before the
                // staged value becomes reachable in the cache and this
                // method clears it after the tail below — so a getter
                // whose transitive verify runs inside the window sees the
                // node as not-yet-verifiable and takes the compute/wait
                // path instead of serving.
                let suspect = task_node.is_some()
                    && reads
                        .iter()
                        .zip(read_epochs.iter())
                        .any(|(&r, &e)| self.epoch_of(r) != e);
                if suspect {
                    // Keep/put the node red.  The value stays staged in
                    // the cache, but `probe` gates on `is_green`, so it
                    // is unreachable: the next getter takes the compute
                    // path (the `Started` entry is still installed until
                    // the completion path in `get_shared` runs, so it
                    // waits, then re-probes and recomputes).
                    self.mark_dirty(node);
                } else {
                    self.clear_dirty(node);
                }
                if changed {
                    // Same unmarked-value-change bump as `publish` (see
                    // there): an in-flight reader's finish re-check is
                    // epoch-based and otherwise blind to value changes
                    // that arrive without a mark.
                    self.bump_epoch(node);
                    if let Some(readers) = shard.rev_edges.get(slot) {
                        for &r in readers {
                            self.mark_dirty(r);
                        }
                    }
                }
            }
        }

        // ---- reverse-edge flush for this task's reads: one lock per
        // READ shard ----
        if let Some(tn) = task_node {
            // Test-only finish hook (T3 / d24, cfg-test): each test gets
            // its own (target, park, arrived) slot set — see
            // `test_finish_hook`.
            #[cfg(test)]
            test_finish_hook(tn);
            // Sort reads and their captured epochs TOGETHER (identical
            // permutation) before the flush — the epoch re-check below is
            // positional and must compare each node against ITS OWN
            // start-of-task revision.
            sort_reads_by_shard(&mut reads, Some(&mut read_epochs));
            self.flush_rev_edges(tn, &mut reads);
            // Drop the reverse edges of reads this recomputation no
            // longer has (see `reconcile_forward_edges` / g3).
            self.prune_rev_edges(tn, removed);
            // ---- late-fan-out detection ----
            // The publish-time fan-out above reaches only readers whose
            // reverse edges ALREADY existed when `node` republished.  A
            // task that read `node` while computing has its reverse edge
            // inserted only by the flush above — a fan-out that ran
            // before that insert never reached it, yet this task's
            // freshly published value may encode the pre-invalidation
            // snapshot of such a read (d16).  Each read captured its
            // node's dirty epoch at record time; any mark of a read node
            // since then (bump ordered before the bit set, so this
            // Acquire load cannot miss a mark whose red bit was
            // observable) means the read went red during this task's
            // compute: restore this task's dirty bit so the suspect
            // value is never served green.  From the next request on,
            // the task's rev edge participates in fan-out normally.
            // Deliberately conservative: a mark whose republished
            // fingerprint is unchanged also costs one recomputation.
            for (i, &r) in reads.iter().enumerate() {
                if self.epoch_of(r) != read_epochs[i] {
                    self.mark_dirty(tn);
                    break;
                }
            }
        }
        // d24: the finish tail is complete — the node's colour is final.
        // Runs UNCONDITIONALLY (also for `task_node == None` defensive
        // calls) so a flag set by `get_shared` before staging can never
        // leak past its finish.  Ordering is load-bearing: this store
        // precedes `get_shared`'s completion `latch.set()`, so a waiter
        // woken from the job latch observes `finishing == false` and its
        // re-probe sees the final colour.
        self.clear_finishing(node);

        changed
    }

    /// Digest gate for call-side invalidation.  Records `input_digest`
    /// for `node` and returns whether the caller MUST invalidate:
    /// `true` when no digest was recorded yet (conservative first call —
    /// the gate has no proof the cached value matches this content) or
    /// the recorded digest differs; `false` when the digests match, in
    /// which case the caller must NOT invalidate (that skip is the whole
    /// point: content unchanged → nothing invalidated, no red bit, no
    /// generation/`rev` bump, downstream stamps all survive).
    ///
    /// Digest quality is the CALLER's contract: this is a cheap
    /// equivalence proof for the node's INPUT content (e.g. a source
    /// digest).  A collision on a weak hash is the same trusted-input
    /// class of bug as a mislabelled `Durability` — use a
    /// fingerprint-grade hash.
    ///
    /// Shard write lock, single small critical section: the gate runs
    /// under `mutate_world` quiescence and nothing else reads `digests`,
    /// so plain storage is enough here.
    pub(crate) fn digest_gate(&self, node: DepNodeIndex, input_digest: u64) -> bool {
        let mut shard = self.shards[shard_of(node)].write();
        ensure_shard_len(&mut shard, slot_of(node));
        let slot = slot_of(node);
        let changed = match shard.digests[slot] {
            Some(old) => old != input_digest,
            None => true,
        };
        shard.digests[slot] = Some(input_digest);
        changed
    }

    /// Record `input_digest` for `node` WITHOUT any comparison or
    /// invalidation — the driver-side companion to [`DepGraph::digest_gate`]
    /// for the first iteration, where the node does not exist yet when the
    /// gate runs (a cold system's `lookup_node` returns `None`, so the
    /// gate has nothing to store against).  Called after the fan-out that
    /// created the nodes, so the NEXT iteration's gate can compare.
    pub(crate) fn record_digest(&self, node: DepNodeIndex, input_digest: u64) {
        let mut shard = self.shards[shard_of(node)].write();
        ensure_shard_len(&mut shard, slot_of(node));
        shard.digests[slot_of(node)] = Some(input_digest);
    }

    /// Mark a node as dirty.  LAZY invalidation (replaces the old eager
    /// transitive BFS): only the node itself is marked red.  Readers are
    /// not touched here — propagation happens lazily, after the node is
    /// recomputed and its fingerprint compared: an UNCHANGED fingerprint
    /// keeps readers green (they stay cached); a CHANGED fingerprint
    /// marks only the DIRECT readers dirty (see [`DepGraph::dirty_readers`]),
    /// and they re-verify in turn when next requested.
    pub(crate) fn invalidate(&self, node: DepNodeIndex) {
        // O(1) atomic RMW, seqlock-guarded: invalidators serialise on
        // nothing at all — not even a shard read lock.  (The generation
        // bump lives in `mark_dirty`: every red-bit creator bumps, not
        // just this path.)
        self.mark_dirty(node);
    }

    pub(crate) fn dirty_readers(&self, node: DepNodeIndex) {
        // Read lock, not write: we only read `rev_edges`.  The dirty
        // marks themselves need no shard lock (seqlock-guarded atomics).
        let shard = self.shards[shard_of(node)].read();
        if let Some(readers) = shard.rev_edges.get(slot_of(node)) {
            for &r in readers {
                self.mark_dirty(r);
            }
        }
    }

    /// Like [`DepGraph::dirty_readers`], but marks only the DIRECT
    /// readers whose declared class intersects `class`.  Tooling API for
    /// channel-scoped invalidation (e.g. capsa re-lexing only the trivia
    /// channel wakes doc/format readers without touching semantic ones).
    ///
    /// Deliberately NOT used by [`DepGraph::publish`]: a fingerprint
    /// change can affect every real reader regardless of class — the
    /// per-query-class split already gives differential invalidation
    /// naturally, so filtering in `publish` would only risk staleness.
    ///
    /// Two phases, one shard lock at a time (module invariant): first
    /// snapshot the reader list under the READ node's shard, then filter
    /// by each reader's declared class under the READER's own shard
    /// (classes live with their node).  A reader committing its edge
    /// between the phases is missed — exactly like a reader committing
    /// after the old single-lock scan.
    pub(crate) fn dirty_readers_by_class(&self, node: DepNodeIndex, class: ContentClass) {
        let mut readers: SmallVec<[DepNodeIndex; 8]> = {
            let shard = self.shards[shard_of(node)].read();
            shard
                .rev_edges
                .get(slot_of(node))
                .map(|r| r.iter().copied().collect())
                .unwrap_or_default()
        };
        if readers.len() > 1 {
            readers.sort_by_key(|&r| shard_of(r));
        }
        let mut i = 0;
        while i < readers.len() {
            let s = shard_of(readers[i]);
            let start = i;
            while i < readers.len() && shard_of(readers[i]) == s {
                i += 1;
            }
            let shard = self.shards[s].read();
            for &r in &readers[start..i] {
                if shard
                    .classes
                    .get(slot_of(r))
                    .is_some_and(|&c| c.intersects(class))
                {
                    self.mark_dirty(r);
                }
            }
        }
    }

    /// Publish a freshly computed value in ONE critical section — on the
    /// node's own shard: compare the new fingerprint against the stored
    /// one, store it, mark the node green, and — only if the fingerprint
    /// actually changed — mark its direct readers dirty.
    ///
    /// Returns `true` when the fingerprint changed (i.e. readers must
    /// re-verify).  This replaces what was four separate lock
    /// acquisitions on the hot miss path (`fingerprint`,
    /// `set_fingerprint`, `mark_green`, `dirty_readers`).
    /// Test-only direct publish.  It writes the fingerprint, marks the
    /// node green and fans out on a CHANGED fingerprint, but it never
    /// reconciles forward edges, flushes reverse edges, or runs the
    /// stale-restore — using it as a finalize path would regress g3 and
    /// d16.  Production finalization MUST go through
    /// [`DepGraph::finish_task_and_publish`].
    #[cfg(test)]
    pub(crate) fn publish(&self, node: DepNodeIndex, fp: u64) -> bool {
        let mut shard = self.shards[shard_of(node)].write();
        let slot = slot_of(node);
        ensure_shard_len(&mut shard, slot);
        let changed = shard.fingerprints[slot].is_some_and(|old| old != fp);
        shard.fingerprints[slot] = Some(fp);
        self.clear_dirty(node);
        if changed {
            // The value SEMANTICALLY changed without any mark (the
            // eval_always / direct-publish shape).  Readers whose captured
            // epoch predates this point must not publish green over it —
            // bump our own epoch so any in-flight reader's finish-side
            // re-check restores itself red).  `bump_epoch` takes
            // no shard lock (lock-free epoch blocks), so calling it under
            // this write lock breaks no ordering.
            self.bump_epoch(node);
            if let Some(readers) = shard.rev_edges.get(slot) {
                for &r in readers {
                    self.mark_dirty(r);
                }
            }
        }
        changed
    }

    pub(crate) fn set_fingerprint(&self, node: DepNodeIndex, fp: u64) {
        let mut shard = self.shards[shard_of(node)].write();
        let slot = slot_of(node);
        ensure_shard_len(&mut shard, slot);
        shard.fingerprints[slot] = Some(fp);
    }

    pub fn fingerprint(&self, node: DepNodeIndex) -> Option<u64> {
        self.shards[shard_of(node)]
            .read()
            .fingerprints
            .get(slot_of(node))
            .copied()
            .flatten()
    }

    pub(crate) fn reset(&self) {
        self.next_index.store(1, Ordering::Relaxed);
        // Per-level mark counters restart at 0, in lockstep with the
        // cleared stamps.  Safe under quiescence: no mark can be in
        // flight (`reset` runs under `mutate_world`), so no `rev` add
        // can land after the zeroing.
        for r in &self.rev {
            r.store(0, Ordering::Relaxed);
        }
        // Clear shard by shard — one write lock at a time (the only
        // multi-shard path; ascending order, and never two locks held at
        // once, so no deadlock).
        for shard in &self.shards {
            let mut sh = shard.write();
            sh.edges.clear();
            sh.rev_edges.clear();
            sh.fingerprints.clear();
            sh.classes.clear();
            // verified_gen MUST be cleared too: `next_index` restarts at
            // 1, so freshly allocated nodes reuse indices whose slots
            // still hold pre-reset stamps — a new node would inherit a
            // claim that its (not yet recorded) closure was verified,
            // and the `vg >= current_gen` fast path would serve stale
            // cache entries without ever walking the closure.
            sh.verified_gen.clear();
            // Per-level durability stamps die with the generation stamps:
            // post-reset every `rev` counter restarts at 0 below, so a
            // surviving stamp (== some pre-reset counter value) would
            // compare `>= 0` and fast-path every reused node without a
            // walk — the per-level twin of the `verified_gen` trap above.
            sh.verified_lvl.clear();
            // Digest gate state dies with everything else: a reused node
            // index must not inherit a pre-reset digest (it would let the
            // gate skip an invalidation for content that never fed the
            // old node's value).
            sh.digests.clear();
            // Same for the finish-in-progress flags: post-reset no node
            // is finishing (quiescence guarantees no compute is in
            // flight), and a stale `true` would make the reused index
            // unverifiable until something cleared it.
            sh.finishing.clear();
        }
        // Append-only bitmap: swap in a fresh zero bitmap under the
        // growth lock.  A mark racing this reset lands on the replaced
        // snapshot and is dropped — the safe direction, because reset
        // also cleared every cache and active-key entry (nothing cached
        // can be stale).
        let _growth = self.grow_lock.lock();
        // Dirty first, epochs after: the safe transient here is
        // `dirty.len() < epochs.len()` — which is exactly the conservative
        // skew (the comparison is the flipped form of grow_dirty's).
        self.dirty.store(Arc::new(
            (0..DIRTY_INIT_BLOCKS).map(|_| zero_block()).collect(),
        ));
        // Epoch blocks reset in lockstep: post-reset every node is fresh
        // (next_index restarts at 1), so fresh captures must read 0 —
        // matching the pre-reset captures of any task that survives the
        // reset only ensures compares are EQUAL (no spurious restore), and
        // fresh captures/finish compares are consistent with the zeroed
        // dirty bitmap.
        self.epochs.store(Arc::new(
            (0..DIRTY_INIT_BLOCKS).map(|_| zero_epoch_block()).collect(),
        ));
        // Thread-local: this clears ONLY the calling thread's stack.
        // Sound because `reset` runs under quiescence (`clear_all` →
        // `mutate_world`): no outermost compute is in flight on ANY
        // thread, and a task frame exists only between a compute's
        // `start_task` and its finalize pop — so every other thread's
        // stack is already empty.  The clear is belt-and-braces for the
        // calling thread.
        TASK_STACK.with(|s| s.borrow_mut().clear());
    }

    /// Test-only view of a node's `verified_gen` stamp.  d9's Window 2
    /// asserts that a freshly allocated node (post-`reset`, index reused)
    /// has stamp 0 — i.e. it did NOT inherit the pre-reset closure stamp
    /// that would let its later hits skip the closure walk entirely.
    #[cfg(test)]
    pub(crate) fn verified_gen_of(&self, node: DepNodeIndex) -> u64 {
        let shard = self.shards[shard_of(node)].read();
        shard.verified_gen.get(slot_of(node)).copied().unwrap_or(0)
    }

    /// Test-only view of a per-level mark counter.
    #[cfg(test)]
    pub(crate) fn rev_of(&self, d: Durability) -> u64 {
        self.rev[d as usize].load(Ordering::Acquire)
    }

    /// Test-only view of a node's per-level durability stamp.
    #[cfg(test)]
    pub(crate) fn verified_lvl_of(&self, node: DepNodeIndex) -> u64 {
        let shard = self.shards[shard_of(node)].read();
        shard.verified_lvl.get(slot_of(node)).copied().unwrap_or(0)
    }

    /// Test hook for the cross-system leak regression: the forward edges
    /// of `node` (its recorded dependency read set).
    #[cfg(test)]
    pub(crate) fn forward_edges_of(&self, node: DepNodeIndex) -> Vec<DepNodeIndex> {
        self.shards[shard_of(node)]
            .read()
            .edges
            .get(slot_of(node))
            .map(|l| l.to_vec())
            .unwrap_or_default()
    }

    /// Check whether a node is green (up-to-date, not dirty).
    ///
    /// Lock-free: one atomic load.  This is THE hot path — every cached
    /// query hit goes through it, so even a *shared* lock here costs a
    /// coherence transfer per hit (Perf Book §13.1).  The `Acquire` load
    /// pairs with the `Release` RMWs in `mark_dirty` / `clear_dirty`, so
    /// a dirty mark published before a dependent `get()` is visible to
    /// it.  A missing block is only possible after `reset()` cleared the
    /// whole world — "green" (nothing cached to be stale) is the safe
    /// answer there.
    #[inline]
    pub fn is_green(&self, node: DepNodeIndex) -> bool {
        self.dirty
            .load()
            .get(block_of(node.0))
            .is_none_or(|b| b[word_of(node.0)].load(Ordering::Acquire) & mask_of(node.0) == 0)
    }

    /// Mark a node as re-computed (green again).  Lock-free: the bitmap
    /// RMW is seqlock-guarded against swaps, so no shard lock is needed.
    #[inline]
    pub(crate) fn mark_green(&self, node: DepNodeIndex) {
        self.clear_dirty(node);
    }

    // ── Finish-in-progress marker (d24) ─────────────────────────────

    /// Mark `node` as "finish in progress": its colour is not final until
    /// `clear_finishing` runs.  Called before staging the freshly computed
    /// value into the cache, so a concurrent transitive verifier cannot
    /// serve the staged value while a late mark on a read node could still
    /// invalidate it (d24).
    ///
    /// Takes a shard READ lock: the flag lives in an `AtomicBool`, so
    /// interior mutability makes a read guard sufficient.  No write lock
    /// is ever held across `set_finishing`/`clear_finishing`, so the
    /// unwind-safety argument holds.
    #[inline]
    pub(crate) fn set_finishing(&self, node: DepNodeIndex) {
        let shard = self.shards[shard_of(node)].read();
        if let Some(f) = shard.finishing.get(slot_of(node)) {
            f.store(true, Ordering::Release);
        }
    }

    /// Clear the "finish in progress" flag: the node's colour is now
    /// final (green or red per the in-section decision and the post-flush
    /// re-check).
    #[inline]
    pub(crate) fn clear_finishing(&self, node: DepNodeIndex) {
        let shard = self.shards[shard_of(node)].read();
        if let Some(f) = shard.finishing.get(slot_of(node)) {
            f.store(false, Ordering::Release);
        }
    }

    /// Test-only view of a node's finishing flag.
    #[cfg(test)]
    pub(crate) fn is_finishing(&self, node: DepNodeIndex) -> bool {
        let shard = self.shards[shard_of(node)].read();
        shard
            .finishing
            .get(slot_of(node))
            .is_some_and(|f| f.load(Ordering::Acquire))
    }

    /// Returns the first dirty
    /// dependency found (if any) so the caller can re-request it.
    /// Returns `None` if the entire transitive closure is green.
    ///
    /// Iterative post-order DFS over an explicit heap stack: the old
    /// plain-recursive walk had no visited set and no depth bound, so a
    /// cycle among (individually green) nodes — reachable once edge sets
    /// grew stale — recursed until the real stack overflowed.  Here a
    /// revisited node is treated as green: its own `is_green` check
    /// already ran on its first visit, and its closure is verified by
    /// its own exit frame deeper in the walk.  Stamping stays POST-order
    /// (only subtrees that fully verified get a `verified_gen` stamp),
    /// and an early `Some(dirty)` abort discards the pending exit
    /// frames — exactly the stamps the recursive version skipped.
    pub fn verify_green_transitive_with_dirty(&self, node: DepNodeIndex) -> Option<DepNodeIndex> {
        let current_gen = self.dirty_generation.load(Ordering::Acquire);

        // Root fast paths, BEFORE the walk state is built.  Every
        // steady-state cached hit runs this function, and the vast
        // majority of hits land here: allocating `visited` + `stack` just
        // to discover the root is already green-and-stamped cost two
        // allocator round-trips per hit.  Ordering matches the walk's own
        // per-node checks exactly: the red bit is ground truth (checked
        // first), the stamp only skips the subtree walk.
        if !self.is_green(node) {
            return Some(node);
        }
        {
            let shard = self.shards[shard_of(node)].read();
            // NEW (d24): a finish tail in flight — the colour is not
            // final yet, and the staged value must not be served.  This
            // check MUST precede the `verified_gen` fast path: a
            // finishing node must never be fast-pathed on a stamp.
            if shard
                .finishing
                .get(slot_of(node))
                .is_some_and(|f| f.load(Ordering::Acquire))
            {
                return Some(node);
            }
            if shard.verified_gen.get(slot_of(node)).copied().unwrap_or(0) >= current_gen {
                return None;
            }
            // Durability fast path (opt-in, trusted-input contract — see
            // `Durability`): the closure was verified at per-level stamp
            // `verified_lvl`, and no mark of durability >= this node's
            // has linearized since (`rev`'s RMW is the level's
            // linearization point — a reader loading a lower value
            // linearizes before the mark, the same overlap doctrine as
            // the global check above).  Every existing query is LOW, so
            // this disjunct only ever fires for queries that opted in.
            let vl = shard.verified_lvl.get(slot_of(node)).copied().unwrap_or(0);
            if vl >= self.rev[self.durability_of(node) as usize].load(Ordering::Acquire) {
                return None;
            }
        }

        let mut visited: HashSet<DepNodeIndex> = HashSet::default();
        // `(node, exiting)` frames: `false` = enter, `true` = all deps
        // verified, stamp post-order.
        let mut stack: Vec<(DepNodeIndex, bool)> = Vec::with_capacity(16);
        stack.push((node, false));
        while let Some((n, exiting)) = stack.pop() {
            if exiting {
                // All green: stamp this node with the current generation
                // AND the per-level mark counter for its durability — the
                // durability fast path's valid_rev.  `verified_lvl` is
                // written AFTER `verified_gen` so a reader that takes the
                // per-level fast path does so against a fully written
                // pair; the extra global check on the slow path costs
                // nothing.
                let mut shard = self.shards[shard_of(n)].write();
                ensure_shard_len(&mut shard, slot_of(n));
                shard.verified_gen[slot_of(n)] = current_gen;
                let d = self.durability_of(n);
                shard.verified_lvl[slot_of(n)] = self.rev[d as usize].load(Ordering::Acquire);
                continue;
            }
            // Visited on this walk (another path, sibling dedup, or a
            // cycle back-edge): green by the first visit.
            if !visited.insert(n) {
                continue;
            }
            // The node itself must be green — checked BEFORE the stamp fast
            // path.  The stamp is an optimistic summary written under a
            // generation snapshot that this walk's `dirty_generation` load
            // can legitimately miss (there is no happens-before edge from
            // the marking thread to this walk), so a "current" stamp must
            // never override the node's own red bit.  One lock-free atomic
            // load; the O(1) subtree skip below (the actual win) is
            // retained.
            //
            // Linearization note (see `mark_dirty`): a walk that loads the
            // generation before a mark's bump overlaps the marking
            // operation and linearizes before it — its all-green verdict
            // (and every stamp it writes at the stale generation) is then
            // validly stale and rejected by any future reader that loads
            // the bumped generation.  The mark's linearization point is
            // its generation bump, not its bit-set.
            if !self.is_green(n) {
                return Some(n);
            }
            // ONE read lock for the whole enter branch.  The d24
            // finishing check, both fast paths and the dependency-edge
            // fetch all read shard `shard_of(n)`, so the two `.read()`
            // acquisitions this branch used to pay collapse into a
            // single guard.  Nothing between the old first and second
            // lock touched that shard — only lock-free loads
            // (`is_green`, `durability_of`, `rev`) and local walk state
            // (`visited`, `stack`) — so the merged section observes
            // exactly the interleavings the split version could, and
            // the fast-path verdict and the edge list now come from ONE
            // shard snapshot.  Still strictly read-only: the shard's
            // write lock stays exclusive to the exit frame's post-order
            // stamp.
            let edges = {
                let shard = self.shards[shard_of(n)].read();
                // NEW (d24): finishing nodes are not verifiable — treat
                // as dirty.  Returning (instead of `continue`) also
                // guarantees the node is never exit-stamped while its
                // colour is still unfinalized.
                if shard
                    .finishing
                    .get(slot_of(n))
                    .is_some_and(|f| f.load(Ordering::Acquire))
                {
                    return Some(n);
                }
                // Fast path: the CLOSURE was already verified at this
                // generation, so skip re-walking its edges.
                if let Some(&vg) = shard.verified_gen.get(slot_of(n)) {
                    if vg >= current_gen {
                        continue;
                    }
                }
                // Per-level durability fast path for interior nodes —
                // same trusted-input contract as the root check above:
                // skips the subtree walk when no mark of this node's
                // durability or higher has linearized since its stamp.
                // Defaults keep every existing query on the global check.
                if let Some(&vl) = shard.verified_lvl.get(slot_of(n)) {
                    if vl >= self.rev[self.durability_of(n) as usize].load(Ordering::Acquire) {
                        continue;
                    }
                }
                // Both fast paths missed: this node's CURRENT edge set,
                // read in the same critical section as the checks above —
                // the second `.read()` is gone.
                shard
                    .edges
                    .get(slot_of(n))
                    .cloned()
                    .unwrap_or_default()
            };
            // Guard released.  Defer the stamp until every dependency has
            // verified, then enter all direct dependencies (reversed so
            // LIFO pops them in declaration order — keeps the "first dirty
            // found" stable with the old recursive walk).  Pure local Vec
            // work, no locks: it does not re-widen the critical section
            // above, and pushing the exit frame before or after the edge
            // read is immaterial — it sits below every dep frame on the
            // LIFO stack either way, and an early `Some(dirty)` abort
            // discards it exactly as before.
            stack.push((n, true));
            for &dep in edges.iter().rev() {
                stack.push((dep, false));
            }
        }
        None
    }
}

impl Default for DepGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_chain(dg: &DepGraph) -> (DepNodeIndex, DepNodeIndex, DepNodeIndex) {
        let a = dg.allocate_node_index(ContentClass::SEMANTIC);
        let b = dg.allocate_node_index(ContentClass::SEMANTIC);
        let c = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.start_task(a);
        dg.read(b);
        dg.finish_task();
        dg.start_task(b);
        dg.read(c);
        dg.finish_task();
        (a, b, c)
    }

    /// Lazy invalidation: `invalidate(c)` marks ONLY `c` red.  Readers are
    /// dirtied lazily via `dirty_readers` when a recomputed fingerprint
    /// changes (the old eager transitive BFS was removed by design).
    #[test]
    fn test_invalidate_marks_node_only_not_readers() {
        let dg = DepGraph::new();
        let (a, b, c) = setup_chain(&dg);
        dg.invalidate(c);
        assert!(!dg.is_green(c));
        assert!(dg.is_green(b), "lazy invalidation: readers must stay green");
        assert!(dg.is_green(a));
    }

    /// A changed fingerprint marks only DIRECT readers dirty (one level).
    #[test]
    fn test_fingerprint_change_dirties_direct_readers_only() {
        let dg = DepGraph::new();
        let (a, b, c) = setup_chain(&dg); // a reads b reads c
        dg.set_fingerprint(c, 1);
        dg.dirty_readers(c);
        assert!(!dg.is_green(b), "b reads c directly, must be dirtied");
        assert!(
            dg.is_green(a),
            "a reads b only transitively, must stay green"
        );
    }

    #[test]
    fn test_fingerprint_roundtrip() {
        let dg = DepGraph::new();
        let n = dg.allocate_node_index(ContentClass::SEMANTIC);
        assert_eq!(dg.fingerprint(n), None);
        dg.set_fingerprint(n, 42);
        assert_eq!(dg.fingerprint(n), Some(42));
        dg.set_fingerprint(n, 43);
        assert_eq!(dg.fingerprint(n), Some(43));
    }

    #[test]
    fn test_rev_edges_independent_nodes() {
        let dg = DepGraph::new();
        let a = dg.allocate_node_index(ContentClass::SEMANTIC);
        let b = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.start_task(a);
        dg.read(b);
        dg.finish_task();
        dg.invalidate(a);
        assert!(!dg.is_green(a));
        assert!(dg.is_green(b));
    }

    /// Fan-out under lazy invalidation: a fingerprint change on the root
    /// dirties ALL direct readers (the fan-out is one level).
    #[test]
    fn test_rev_edges_fan_out() {
        let dg = DepGraph::new();
        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        let x = dg.allocate_node_index(ContentClass::SEMANTIC);
        let y = dg.allocate_node_index(ContentClass::SEMANTIC);
        let z = dg.allocate_node_index(ContentClass::SEMANTIC);
        for &reader in &[x, y, z] {
            dg.start_task(reader);
            dg.read(root);
            dg.finish_task();
        }
        dg.invalidate(root);
        assert!(!dg.is_green(root));
        assert!(dg.is_green(x), "invalidate alone must not propagate");
        assert!(dg.is_green(y));
        assert!(dg.is_green(z));
        dg.dirty_readers(root);
        assert!(!dg.is_green(x));
        assert!(!dg.is_green(y));
        assert!(!dg.is_green(z));
    }

    #[test]
    fn test_allocate_node_resizes_rev_edges() {
        let dg = DepGraph::new();
        let a = dg.allocate_node_index(ContentClass::SEMANTIC);
        let b = dg.allocate_node_index(ContentClass::SEMANTIC);
        assert!(a.0 < b.0);
        let shard = dg.shards[shard_of(b)].read();
        assert!(shard.rev_edges.len() > slot_of(b));
    }

    /// Sharding layout: node `idx` lives in shard `idx % 64` at compact
    /// slot `idx / 64`; per-node class and fingerprint lookups work for
    /// every node, and no single shard holds the whole graph.
    #[test]
    fn test_shard_layout_compact_slots() {
        let dg = DepGraph::new();
        let mut nodes = Vec::new();
        for i in 0..200 {
            let class = if i % 2 == 0 {
                ContentClass::SEMANTIC
            } else {
                ContentClass::DOC
            };
            nodes.push((dg.allocate_node_index(class), class));
        }
        for (i, &(n, class)) in nodes.iter().enumerate() {
            assert_eq!(n.0 as usize, i + 1);
            assert_eq!(dg.class_of(n), Some(class), "class for node {n:?}");
            let shard = dg.shards[shard_of(n)].read();
            assert!(shard.classes.len() > slot_of(n));
            assert!(shard.rev_edges.len() > slot_of(n));
        }
        // Each shard holds only its 1/64th share of the nodes.
        let per_shard_cap = nodes.len() / DEP_SHARDS + 1;
        assert!(
            dg.shards
                .iter()
                .all(|s| s.read().classes.len() <= per_shard_cap)
        );
        // Fingerprints are per-shard too.
        let last = nodes.last().unwrap().0;
        dg.set_fingerprint(last, 7);
        assert_eq!(dg.fingerprint(last), Some(7));
    }

    /// Concurrent allocation + invalidation across shards exercises the
    /// bitmap regrowth / seqlock path: after every thread finishes its
    /// mark-invalidate-mark-green cycles, every node must be green (a
    /// lost clear is the stale-value failure mode) and carry its own
    /// class (no cross-shard mislabelling).
    #[test]
    fn test_parallel_allocate_and_invalidate() {
        let dg = Arc::new(DepGraph::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let dg = dg.clone();
                std::thread::spawn(move || {
                    // 40_000 × 8 = 320_000 nodes > the 262_144-node initial
                    // bitmap capacity (64 blocks × 4096 bits), so
                    // `grow_dirty` regrows the bitmap mid-storm while other
                    // threads are marking — exercising append-only
                    // regrowth under concurrent marks.  (The old 2500/
                    // thread = 20 000 nodes never left the initial
                    // capacity, so the regrowth path was never taken.)
                    for _ in 0..40_000 {
                        let n = dg.allocate_node_index(ContentClass::SEMANTIC);
                        assert!(dg.is_green(n), "fresh node is green");
                        dg.invalidate(n);
                        assert!(!dg.is_green(n));
                        dg.mark_green(n);
                        assert!(dg.is_green(n));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // 320 000 nodes pushed the bitmap well past its initial 64-block
        // capacity — the regrowth happened while other threads were
        // marking, and every node must still be green.
        assert!(dg.next_index.load(Ordering::Relaxed) > 300_000);
        for idx in 1..dg.next_index.load(Ordering::Relaxed) {
            let n = DepNodeIndex(idx);
            assert!(dg.is_green(n), "node {n:?} lost its green mark");
            assert_eq!(dg.class_of(n), Some(ContentClass::SEMANTIC));
        }
    }

    /// `dirty_readers_by_class` marks only readers whose declared class
    /// intersects the given class; other readers stay green.
    #[test]
    fn test_dirty_readers_by_class_filters() {
        let dg = DepGraph::new();
        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        let sem_reader = dg.allocate_node_index(ContentClass::SEMANTIC);
        let doc_reader = dg.allocate_node_index(ContentClass::DOC);
        let trivia_reader = dg.allocate_node_index(ContentClass::TRIVIA);
        for &reader in &[sem_reader, doc_reader, trivia_reader] {
            dg.start_task(reader);
            dg.read(root);
            dg.finish_task();
        }
        // A TRIVIA-channel change wakes the trivia reader (same class)
        // but not the semantic one.
        dg.dirty_readers_by_class(root, ContentClass::TRIVIA);
        assert!(
            dg.is_green(sem_reader),
            "semantic reader must not be woken by a trivia-scoped invalidation"
        );
        assert!(dg.is_green(doc_reader));
        assert!(!dg.is_green(trivia_reader));

        // A class mask covering DOC+TRIVIA wakes both of those readers.
        dg.dirty_readers_by_class(root, ContentClass::DOC | ContentClass::TRIVIA);
        assert!(dg.is_green(sem_reader));
        assert!(!dg.is_green(doc_reader));

        // The unfiltered API still marks every direct reader.
        dg.dirty_readers(root);
        assert!(!dg.is_green(sem_reader));
    }

    /// Declared classes are recorded per node and queryable.
    #[test]
    fn test_class_of() {
        let dg = DepGraph::new();
        let s = dg.allocate_node_index(ContentClass::SEMANTIC);
        let d = dg.allocate_node_index(ContentClass::DOC);
        assert_eq!(dg.class_of(s), Some(ContentClass::SEMANTIC));
        assert_eq!(dg.class_of(d), Some(ContentClass::DOC));
        assert_eq!(dg.class_of(DepNodeIndex(9_999)), None);
    }

    /// Finish-side late-fan-out re-check pairing bug: the finish-side
    /// late-fan-out re-check compares `epoch_of(reads[i])` against
    /// `read_epochs[i]` POSITIONALLY, but `flush_rev_edges` sorts
    /// `reads` by shard WITHOUT permuting the parallel epoch vector.
    /// A genuine permutation (recorded shard order [1, 0]) scrambles
    /// the pairing, so the re-check compares one node's current epoch
    /// against ANOTHER node's captured revision.  The reader here is
    /// honest (each read is green relative to its OWN capture) yet gets
    /// spuriously re-dirtied — and every spurious 0→1 mark bumps
    /// `dirty_generation`, invalidating every `verified_gen` stamp
    /// system-wide (memoization collapse).
    #[test]
    fn finish_epoch_recheck_pairing_survives_shard_sort() {
        let dg = DepGraph::new();
        // Nodes 1..=64: shard_of(n) = n & 63, so node 64 sits in shard 0.
        for _ in 0..64 {
            let _ = dg.allocate_node_index(ContentClass::SEMANTIC);
        }
        // Node 65 lives in shard 1; give it a dirty history (epoch 1)
        // while node 64 stays at 0 — the unequal pair a scrambled
        // comparison miscouples.
        let later = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.invalidate(later);
        assert!(!dg.publish(later, 1), "first publish: nothing changed");
        assert!(dg.is_green(later));

        let reader = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.start_task(reader);
        dg.read(later); // shard 1, epoch 1 — recorded FIRST
        dg.read(DepNodeIndex(64)); // shard 0, epoch 0 — recorded SECOND
        assert!(!dg.finish_task_and_publish(reader, Some(7)), "fresh fp");

        assert!(
            dg.is_green(reader),
            "epoch re-check must compare each read against its own captured \
             epoch (reads/epochs pairing must survive the shard sort)"
        );
    }

    /// The d16 guard is epoch-based and only `mark_dirty` bumps epochs —
    /// but a node's value can change WITHOUT ever being marked (eval-always
    /// republishes on every direct request; direct `publish` with a new
    /// fingerprint).  In the late-fan-out window (reader's rev edge not yet
    /// flushed) the fan-out misses the reader AND the epoch guard is blind,
    /// so the reader publishes green over a stale base.  Fix: a
    /// changed-fingerprint publish bumps the node's own epoch, so any
    /// in-flight reader's finish re-check restores itself red.
    #[test]
    fn finish_epoch_recheck_catches_unmarked_value_change() {
        let dg = DepGraph::new();
        let e = dg.allocate_node_index(ContentClass::SEMANTIC);
        let r = dg.allocate_node_index(ContentClass::SEMANTIC);

        // E's initial value: fp 1, epoch 0.
        assert!(!dg.publish(e, 1));

        // R reads green E (captures E's current epoch).
        dg.start_task(r);
        dg.read(e);

        // E republished with a CHANGED fingerprint — no mark, no
        // invalidation (the eval_always shape).
        assert!(dg.publish(e, 7), "fp changed");

        // R finishes through the real finalize path.
        assert!(!dg.finish_task_and_publish(r, Some(1)), "R's fp unchanged");

        assert!(
            !dg.is_green(r),
            "reader over an unmarked value change must be restored red"
        );

        // No false positive: a reader that read E AFTER the change stays
        // green.
        let r2 = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.start_task(r2);
        dg.read(e);
        assert!(!dg.finish_task_and_publish(r2, Some(1)));
        assert!(dg.is_green(r2));
    }

    // ── Durability fast path (per-level rev / verified_lvl) ─────────

    /// Declared HIGH reader + HIGH dep, both verified: a LOW-level mark
    /// must NOT invalidate their per-level fast path (verify returns
    /// None without any walk), while a HIGH-level mark must (verify
    /// returns the marked node).  The g1 analog: the stdlib subtree
    /// survives user-source invalidation.
    #[test]
    fn durability_low_mark_does_not_invalidate_high_fast_path() {
        let dg = DepGraph::new();
        // Warm-up mark: at generation 0 the GLOBAL stamp check is
        // trivially true (everything is "already verified", no walk, no
        // per-level stamp) — force a real walk by raising the generation
        // with a HIGH-level invalidate.
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm);

        let dep = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(dep, Durability::HIGH);
        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(root, Durability::HIGH);
        dg.start_task(root);
        dg.read(dep);
        assert!(!dg.finish_task_and_publish(root, Some(1)));

        // First verify walks (gen stamps are stale) and stamps both
        // nodes per-level.
        assert_eq!(dg.verify_green_transitive_with_dirty(root), None);
        assert!(dg.verified_lvl_of(root) > 0);
        assert_eq!(dg.rev_of(Durability::HIGH), 1);

        // A LOW-level mark (user source): must not bump rev[HIGH].
        let low = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.invalidate(low);
        assert_eq!(dg.rev_of(Durability::LOW), 2);
        assert_eq!(dg.rev_of(Durability::HIGH), 1);
        // Fast path: still all-green at level HIGH, no walk needed.
        assert_eq!(dg.verify_green_transitive_with_dirty(root), None);

        // A HIGH-level mark (stdlib): bumps all three levels — the
        // stamp no longer covers rev[HIGH], so the walk runs and finds
        // the red dep.
        dg.invalidate(dep);
        assert_eq!(
            dg.verify_green_transitive_with_dirty(root),
            Some(dep)
        );
    }

    /// The min(declared, reads) aggregation: a declared-HIGH node that
    /// reads a LOW node is aggregated DOWN to LOW, so a LOW mark DOES
    /// invalidate its fast path.  Declared durability never flows
    /// upward.
    #[test]
    fn durability_aggregates_down_over_reads() {
        let dg = DepGraph::new();
        let low_dep = dg.allocate_node_index(ContentClass::SEMANTIC); // LOW
        let high_reader = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(high_reader, Durability::HIGH);
        dg.start_task(high_reader);
        dg.read(low_dep);
        assert!(!dg.finish_task_and_publish(high_reader, Some(1)));
        assert_eq!(dg.durability_of(high_reader), Durability::LOW);

        // Now a LOW mark invalidates the reader's (LOW) fast path.
        assert_eq!(dg.verify_green_transitive_with_dirty(high_reader), None);
        dg.invalidate(low_dep);
        assert_eq!(
            dg.verify_green_transitive_with_dirty(high_reader),
            Some(low_dep)
        );
    }

    /// g2 analog, per-level: a red bit on the node ITSELF must never be
    /// fast-pathed past, even with a current level stamp — the `is_green`
    /// root check precedes the durability disjunct.
    #[test]
    fn durability_fast_path_never_bypasses_own_red_bit() {
        let dg = DepGraph::new();
        // Warm-up mark: raise the generation so the first verify actually
        // walks (at generation 0 the global stamp check is trivially
        // true and no per-level stamp is written).
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm);

        let dep = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(dep, Durability::HIGH);
        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(root, Durability::HIGH);
        dg.start_task(root);
        dg.read(dep);
        assert!(!dg.finish_task_and_publish(root, Some(1)));
        assert_eq!(dg.verify_green_transitive_with_dirty(root), None);
        assert_eq!(dg.verified_lvl_of(root), 1);

        // Mark the root itself (a HIGH-level mark — also bumps rev[HIGH],
        // so BOTH checks fail): verify must return the red root.
        dg.invalidate(root);
        assert!(!dg.is_green(root));
        assert_eq!(
            dg.verify_green_transitive_with_dirty(root),
            Some(root)
        );

        // Recompute (fresh publish, same fingerprint): the root is green
        // again but its level stamp (1) predates the mark's rev[HIGH]
        // bump (2), so the fast path must NOT serve — the walk re-runs.
        // The closure is all green (dep was never invalidated), so the
        // walk re-stamps and returns None.
        assert!(!dg.finish_task_and_publish(root, Some(1)));
        assert!(dg.is_green(root));
        assert_eq!(dg.verify_green_transitive_with_dirty(root), None);
        assert_eq!(dg.verified_lvl_of(root), 2);

        // Now mark the DEP (level HIGH again): the freshly re-stamped
        // root must not fast-path past its red dependency.
        dg.invalidate(dep);
        assert_eq!(
            dg.verify_green_transitive_with_dirty(root),
            Some(dep),
            "red dep must be found by the walk, not served past"
        );
    }

    /// d24 analog, per-level: a node whose `finishing` flag is set is
    /// not verifiable even when its level stamp is current — the
    /// finishing check precedes the durability fast path.
    #[test]
    fn durability_fast_path_respects_finishing_flag() {
        let dg = DepGraph::new();
        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(root, Durability::HIGH);
        dg.start_task(root);
        dg.finish_task();
        assert!(!dg.finish_task_and_publish(root, Some(1)));
        assert_eq!(dg.verify_green_transitive_with_dirty(root), None);

        // Simulate the d24 window: finishing set, stamp current.
        dg.set_finishing(root);
        assert_eq!(
            dg.verify_green_transitive_with_dirty(root),
            Some(root),
            "finishing node must not be fast-pathed on a level stamp"
        );
        dg.clear_finishing(root);
        assert_eq!(dg.verify_green_transitive_with_dirty(root), None);
    }

    /// Reset clears the per-level counters AND the per-level stamps in
    /// lockstep: a reused node index must not inherit a pre-reset level
    /// stamp that would fast-path its (not yet recorded) closure.
    #[test]
    fn durability_reset_clears_counters_and_stamps() {
        let dg = DepGraph::new();
        // Warm-up mark: raise the generation so the first verify actually
        // walks and writes a per-level stamp.
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm);

        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(root, Durability::HIGH);
        dg.start_task(root);
        dg.finish_task();
        assert!(!dg.finish_task_and_publish(root, Some(1)));
        assert_eq!(dg.verify_green_transitive_with_dirty(root), None);
        assert!(dg.verified_lvl_of(root) > 0);

        let low = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.invalidate(low);
        dg.reset();
        assert_eq!(dg.rev_of(Durability::LOW), 0);
        assert_eq!(dg.rev_of(Durability::HIGH), 0);
        assert_eq!(dg.verified_lvl_of(root), 0);
    }
}
