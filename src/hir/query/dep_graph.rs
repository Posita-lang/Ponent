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

use crate::sync::{Mutex, RwLock};
use arc_swap::ArcSwap;
use bitflags::bitflags;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use rustc_hash::FxHashSet as HashSet;

use super::sync_order;
use super::ticket;
use super::{CachePadded, NoOpScheduler, SchedulePoint, Scheduler};

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
/// content-kind axis).  Salsa-style change-frequency levels:
///
/// - `LOW` (default): may change at any time (user source, editor
///   state).  A LOW mark invalidates every durability fast path.
/// - `MEDIUM`: changes rarely (dependency manifests, config files).
/// - `HIGH`: effectively immutable within a session (stdlib, prelude,
///   builtins).  A HIGH node's fast path survives LOW/MEDIUM marks.
///
/// # Soundness is structural; declarations are a cap
///
/// The durability fast path is the one serve path in this system that
/// skips transitive verification BY DESIGN.  Its soundness condition is
/// the invariant
///
///     stored(X) <= min over closure(X) of stored
///
/// (a mark of any closure node then bumps `rev[stored(X)]` and kills
/// the stamp, forcing the walk).  That invariant is maintained
/// STRUCTURALLY, not trusted:
///
/// - the finish-time `min(declared, reads)` aggregation keeps each
///   node's own level at most `min` over its DIRECT reads;
/// - every actual lowering propagates along the reverse edges
///   ([`DepGraph::lower_and_propagate`]): readers — transitively — are
///   lowered with it, so a read whose durability drops after a reader
///   finished cannot leave the reader stale-high;
/// - the finish-side late re-aggregation (post-flush, the durability
///   twin of d16) catches a read lowering during the reader's own
///   compute; the `lowering_epoch` seqlock seals a verification against
///   an in-flight cascade.
///
/// Consequence: declaring HIGH for a query that (transitively, later)
/// reads LOW inputs costs only fast-path MISSES, never staleness — the
/// aggregation plus propagation push the stored level down and the
/// fast path degrades to the walk.  The declaration is an optimization
/// cap, not a soundness obligation; declare HIGH/MEDIUM only for inputs
/// the host treats as immutable per session to actually earn the fast
/// path.  The remaining trusted-input surface is the digest gate
/// (`invalidate_if_changed`'s input digest quality) and the host's duty
/// to actually mark/invalidate when an input changes — both outside
/// this mechanism.
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

    /// Insert `other` unless it is already recorded, and report which
    /// happened.  The epoch is NOT taken here: [`Self::read_with_epoch`]
    /// captures it only on the path that actually kept the entry, so a
    /// duplicate read pays the dedup (a contains scan / hash probe) and
    /// nothing else.  A wide query that re-reads a hot node in a loop
    /// used to pay a TLS `epochs_snapshot()` round trip plus an atomic
    /// load per duplicate and then discard both.
    ///
    /// Returns `true` when the caller must now supply the epoch.
    pub fn read(&mut self, other: DepNodeIndex) -> bool {
        if other == self.current_node {
            return false;
        }
        if let Some(spilled) = self.spilled.as_mut() {
            if spilled.insert(other) {
                self.reads.push(other);
                return true;
            }
            return false;
        }
        if self.reads.contains(&other) {
            return false;
        }
        self.reads.push(other);
        if self.reads.len() >= LINEAR_DEDUP_MAX {
            self.spilled = Some(self.reads.iter().copied().collect());
        }
        true
    }

    /// Supply `epoch` for the entry [`Self::read`] just reported as
    /// freshly inserted.  Kept SEPARATE from `read` so the epoch capture
    /// (a TLS snapshot plus an atomic load) is paid once per DISTINCT
    /// read, not once per call.  The two must stay paired — the
    /// `reads`/`read_epochs` vectors are positional-by-construction, and
    /// `sort_reads_by_shard` relies on that pairing surviving.
    pub fn read_with_epoch(&mut self, other: DepNodeIndex, epoch: u64) {
        debug_assert_eq!(
            self.reads.last().copied(),
            Some(other),
            "read_with_epoch must immediately follow a `read` that reported \
             the node as freshly inserted"
        );
        self.read_epochs.push(epoch);
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

// ── Observability for dropped dependency edges ────────────────────

/// Count of dependency edges rejected by
/// [`DepGraph::read`]'s cross-system guard — in EVERY build, taken
/// before the mode-specific outcome (see `note_cross_system_skip`).
/// Debug builds panic on the violation (a pairing error is a
/// programming error); a release build cannot, because the contract is
/// documented as "keep provider and executing system paired", not
/// enforced by the type system — so it skips the record instead.
///
/// The skip is the SAFE direction (a missing edge never corrupts the
/// graph; the alternative would be a phantom node), but it means
/// permanent staleness: invalidations on the owning system can no
/// longer reach this graph.  Silence makes that indistinguishable from
/// correct behaviour, hence the counter.  `Relaxed` — a diagnostic
/// count that enters no happens-before chain (the same contract the
/// cache's `StatField` counters use).
///
/// NOTE the counter is PROCESS-GLOBAL, not per `DepGraph`: two graphs in
/// one process share it, so a count tells you "somewhere, a mispairing
/// happened" without telling you which graph.  That is enough for the
/// only question it answers — "is this process mis-pairing at all?" —
/// and it is deliberately not keyed by `owner_id`, because the correct
/// attribution (which graph, and which call site) belongs in a
/// structured log/sink rather than a single atomic.  If this ever needs
/// per-graph attribution, the shape is a `DashMap<usize, AtomicU64>` (or
/// a `Vec` beside the shards, sized like them) keyed on `owner_id`; note
/// that `owner_id` is now minted uniquely per graph (see
/// `next_owner_id`), so such a key would be well defined — which was not
/// true when the id was the shared sentinel 0.
static CROSS_SYSTEM_SKIPS: AtomicU64 = AtomicU64::new(0);

/// Record one dropped cross-system edge, rate-limited on stderr.
///
/// Called from the guard in BOTH modes — before the debug panic and
/// before the release skip — so the count is taken no matter which
/// outcome follows.  Only the stderr note is release-only: a debug build
/// panics with the full detail on the very next statement, so a warning
/// line here would be pure noise.
///
/// The crate has no logging facade (`hir/` uses `eprintln!` directly —
/// see `hir/target/mod.rs`'s platform fallbacks), so this matches that
/// convention rather than adding a `log` dependency for one line.  Only
/// the first occurrence and each power-of-two after it print: a
/// systematic pairing bug would otherwise flood stderr with one line
/// per dropped edge, and a single line per threshold keeps the volume
/// logarithmic while still making a persistent leak visible.
#[cold]
fn note_cross_system_skip(other: DepNodeIndex, owner: usize, frame: usize) {
    let n = CROSS_SYSTEM_SKIPS.fetch_add(1, Ordering::Relaxed);
    if !cfg!(debug_assertions) && (n == 0 || (n + 1).is_power_of_two()) {
        eprintln!(
            "warning: dropped cross-system dependency edge: node {other:?} of query system \
             {owner} recorded into a task frame of query system {frame} \
             (provider/system pairing violated).  The edge is NOT recorded, so this graph \
             will not be invalidated by changes on the owning system and may serve a stale \
             value.  {n} earlier drop(s).",
        );
    }
}

/// Test-only view of [`CROSS_SYSTEM_SKIPS`].
#[cfg(test)]
fn cross_system_skip_count() -> u64 {
    CROSS_SYSTEM_SKIPS.load(Ordering::Relaxed)
}

impl<S: Scheduler> DepGraph<S> {
    /// Test-only view of the graph's owner id, so a test can assert the
    /// mint vs. `with_owner` distinction.  Not `pub`: production code has no
    /// reason to read an id it handed out itself.
    #[cfg(test)]
    pub(crate) fn owner_id_for_test(&self) -> usize {
        self.owner_id
    }
}

// ── Thread-local task stack ──────────────────────────────────────

thread_local! {
    /// Per-thread stack of active `TaskDeps`.  Each query thread has its
    /// own execution context, so the task stack is thread-local rather
    /// than shared.  This is the same pattern as Rust's implicit task
    /// deps (`TaskDepsRef` in `dep_graph/mod.rs`).
    static TASK_STACK: std::cell::RefCell<Vec<TaskDeps>> = const { std::cell::RefCell::new(Vec::new()) };
}

// ── Buffered verification stamps (durability seqlock) ────────────

/// One buffered post-order stamp of a verification walk (see
/// `verify_green_transitive_with_dirty`'s seqlock loop).  (`generation`,
/// not `gen` — `gen` is a reserved keyword in edition 2024.)
struct PendingStamp {
    node: DepNodeIndex,
    generation: u64,
    lvl: u64,
}

/// How many times a verification may retry when a durability lowering
/// overlaps its seqlock window before falling back to the no-shortcut
/// walk.  Lowerings are rare finish-side events and levels only ever
/// drop `Durability::LEVELS - 1` times per node EVER, so an overlap
/// storm is effectively unreachable; the bound is belt-and-braces.
const VERIFY_SEQLOCK_RETRIES: usize = 3;

/// Cap on pooled `verify_inner` walk buffers (#11): one giant walk must
/// not pin megabytes on its thread forever — oversized buffers are
/// dropped instead of returned to TLS.
const VERIFY_SCRATCH_CAP: usize = 2048;

/// Thread-local reuse for `verify_inner`'s walk state (#11): the
/// unstamped traversal allocated a fresh `HashSet` + `Vec` per call —
/// pure allocator churn after an invalidation storm, when every request
/// walks until stamps are rebuilt.  Buffers are taken out of TLS on
/// entry and restored cleared on `Drop`, so every exit path returns
/// them.  TLS is per-thread and `verify_inner` never re-enters itself
/// on the same thread, so a failed borrow (TLS destruction) falls back
/// to fresh locals instead of deadlocking.
thread_local! {
    static VERIFY_SCRATCH: RefCell<(HashSet<DepNodeIndex>, Vec<(DepNodeIndex, bool)>)> =
        RefCell::new((HashSet::default(), Vec::with_capacity(16)));
}

/// Pooled `verify_inner` walk state: `take` it, use `.visited`/`.stack`
/// directly, and `Drop` returns the cleared buffers to TLS.
struct VerifyScratch {
    visited: HashSet<DepNodeIndex>,
    stack: Vec<(DepNodeIndex, bool)>,
    pooled: bool,
}

impl VerifyScratch {
    fn take() -> Self {
        let pooled = VERIFY_SCRATCH
            .try_with(|c| {
                c.try_borrow_mut()
                    .ok()
                    .map(|mut b| (std::mem::take(&mut b.0), std::mem::take(&mut b.1)))
            })
            .unwrap_or(None);
        match pooled {
            Some((visited, stack)) => Self {
                visited,
                stack,
                pooled: true,
            },
            None => Self {
                visited: HashSet::default(),
                stack: Vec::with_capacity(16),
                pooled: false,
            },
        }
    }
}

impl Drop for VerifyScratch {
    fn drop(&mut self) {
        if !self.pooled {
            return;
        }
        if self.visited.capacity() > VERIFY_SCRATCH_CAP
            || self.stack.capacity() > VERIFY_SCRATCH_CAP
        {
            return;
        }
        self.visited.clear();
        self.stack.clear();
        let mut visited = std::mem::take(&mut self.visited);
        let mut stack = std::mem::take(&mut self.stack);
        // `take` left empty defaults behind, so TLS holds valid buffers;
        // swap the grown ones back in and drop the defaults.  Failure
        // (TLS destruction) just drops the buffers.
        let _ = VERIFY_SCRATCH.try_with(|c| {
            if let Ok(mut b) = c.try_borrow_mut() {
                std::mem::swap(&mut b.0, &mut visited);
                std::mem::swap(&mut b.1, &mut stack);
            }
        });
    }
}

// ── Pooled verification stamp buffer (same rationale as
//    `VERIFY_SCRATCH`) ─────────────────────────────────────────────

/// Cap on a pooled stamp buffer: one giant walk must not pin megabytes
/// on its thread forever, so an oversized buffer is dropped rather than
/// returned.  A stamp is 16 bytes, so 2048 entries ≈ 32 KiB.
const STAMP_POOL_CAP: usize = 2048;

thread_local! {
    static STAMP_POOL: RefCell<Vec<PendingStamp>> = RefCell::new(Vec::new());
}

/// Pooled `verify_inner` stamp buffer (perf item 2).  The walk used to
/// allocate a fresh `Vec` per call — and a fresh one per seqlock retry,
/// so an overlap storm paid up to `VERIFY_SEQLOCK_RETRIES` allocations
/// for nothing.  `Vec::new()` does not allocate until the first push, so
/// this costs nothing on the pass-through hits; the win is on the WALK
/// path after an invalidation storm, when stamps grow to the closure
/// size and every node on it re-runs the walk.
///
/// Same shape as [`VerifyScratch`]: the buffer is taken OUT of TLS (so
/// TLS always holds a valid, empty `Vec`) and restored cleared on `Drop`,
/// which makes every exit path return it.  Taking it out is also what
/// makes same-thread re-entry safe — a nested verification finds an
/// empty TLS slot and simply gets fresh locals instead of pooling.
struct StampBuf {
    stamps: Vec<PendingStamp>,
    pooled: bool,
}

impl StampBuf {
    fn take() -> Self {
        let pooled = STAMP_POOL
            .try_with(|c| c.try_borrow_mut().ok().map(|mut b| std::mem::take(&mut *b)))
            .unwrap_or(None);
        match pooled {
            Some(stamps) => Self { stamps, pooled: true },
            None => Self {
                stamps: Vec::new(),
                pooled: false,
            },
        }
    }

    /// The buffer, cleared: every verification must start from an empty
    /// stamp list even when reusing a buffer from an earlier call.
    #[inline]
    fn cleared(&mut self) -> &mut Vec<PendingStamp> {
        self.stamps.clear();
        &mut self.stamps
    }
}

impl Drop for StampBuf {
    fn drop(&mut self) {
        if !self.pooled || self.stamps.capacity() > STAMP_POOL_CAP {
            return;
        }
        self.stamps.clear();
        // `take` left an empty `Vec` behind, so TLS holds a valid
        // buffer; swap the grown one back in and drop the empty one.
        // Failure (TLS destruction) just drops the buffer.
        let mut s = std::mem::take(&mut self.stamps);
        let _ = STAMP_POOL.try_with(|c| {
            if let Ok(mut b) = c.try_borrow_mut() {
                std::mem::swap(&mut *b, &mut s);
            }
        });
    }
}

/// Cap on a pooled fan-out buffer.  `DepNodeIndex` is 4 bytes, so 4096
/// entries ≈ 16 KiB — four times the entry count of the stamp pool for
/// the same footprint, because a node's reverse-edge list is far more
/// often large than a walk's stamp list is deep.
const FANOUT_POOL_CAP: usize = 4096;

thread_local! {
    static FANOUT_POOL: RefCell<Vec<DepNodeIndex>> = RefCell::new(Vec::new());
}

/// Pooled publish-time fan-out reader list.
///
/// The marks themselves run AFTER the node's shard write lock releases
/// (perf item 1), so the list is pure scratch — but it is FILLED under the
/// lock, and an inline `SmallVec` heap-allocates as soon as the node has
/// more than four direct readers.  That is exactly the wrong place for it:
/// the lock being held is the one every reader's own `finish` must take, so
/// a hot node's tail latency would carry a malloc.  Same shape as
/// [`StampBuf`] — taken out of TLS, returned cleared on `Drop`, so the
/// allocation is amortized across finishes on the thread and every exit
/// path (including a panic unwind) hands the buffer back.
struct FanoutBuf {
    readers: Vec<DepNodeIndex>,
    pooled: bool,
}

impl FanoutBuf {
    fn take() -> Self {
        let pooled = FANOUT_POOL
            .try_with(|c| c.try_borrow_mut().ok().map(|mut b| std::mem::take(&mut *b)))
            .unwrap_or(None);
        match pooled {
            Some(readers) => Self {
                readers,
                pooled: true,
            },
            None => Self {
                readers: Vec::new(),
                pooled: false,
            },
        }
    }

    /// FILL side only — called INSIDE the shard section.
    ///
    /// Deliberately takes the reader slice as an ARGUMENT and returns
    /// nothing: there is no accessor on this type that hands out a
    /// `&mut Vec`, so a consume site has nothing to call that could wipe
    /// the list.  That is the whole point of the shape —
    /// the previous API exposed `cleared() -> &mut Vec` for the fill side,
    /// and the consume site reached for it too, which CLEARED the list
    /// about to be walked and turned the entire publish-time fan-out into
    /// a silent no-op (readers of a changed node were
    /// never marked red, so a transitively stale value could be served
    /// forever).  Keeping `cleared` but documenting the rule was a
    /// convention; this makes it structural.
    #[inline]
    fn fill(&mut self, readers: &[DepNodeIndex]) {
        self.readers.clear();
        self.readers.extend_from_slice(readers);
    }

    /// CONSUME side — called AFTER the section releases, to walk the list
    /// the section filled.
    ///
    /// Returns a newtype rather than `&[DepNodeIndex]` so the two halves
    /// are distinguishable at a call site, and so that no `&mut Vec` is in
    /// scope here at all: [`Self::fill`] takes the source data as an
    /// argument, so the only mutable access is the one the fill side
    /// needs.
    #[inline]
    fn filled(&self) -> FilledFanout<'_> {
        FilledFanout(&self.readers)
    }
}

/// The FILLED half of a [`FanoutBuf`] — the value the consume side walks.
///
/// Zero cost (a slice newtype).  Its purpose is purely to make the
/// fill/consume split a TYPE-level rule: a consume site holding this has no
/// access to anything mutable, so the `cleared()`-on-the-consume-side bug
/// cannot be written even by accident.
struct FilledFanout<'a>(&'a [DepNodeIndex]);

impl<'a> FilledFanout<'a> {
    #[inline]
    fn iter(&self) -> std::iter::Copied<std::slice::Iter<'a, DepNodeIndex>> {
        self.0.iter().copied()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Drop for FanoutBuf {
    fn drop(&mut self) {
        if !self.pooled || self.readers.capacity() > FANOUT_POOL_CAP {
            return;
        }
        self.readers.clear();
        let mut s = std::mem::take(&mut self.readers);
        let _ = FANOUT_POOL.try_with(|c| {
            if let Ok(mut b) = c.try_borrow_mut() {
                std::mem::swap(&mut *b, &mut s);
            }
        });
    }
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
pub(crate) struct DepShard {
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
    // NO dedup pass: `reads` arrives ALREADY deduplicated, in first-read
    // order — `TaskDeps::read` guarantees it (linear `contains` up to
    // `LINEAR_DEDUP_MAX`, a `HashSet` past it).  The first version of this
    // function rebuilt `new_list` with its own `contains` scan, paying
    // O(k^2) on every finish to re-derive what `TaskDeps` already
    // guarantees: a 50-read query paid 1275 redundant compares.  Readers
    // of this function rely on that invariant, so re-adding a dedup here
    // must be justified against `TaskDeps::read` first.
    //
    // Stale edges: entries of the previous list the current read set no
    // longer contains.  O(|list| × k) is unavoidable without hashing;
    // both sides are normally a handful.
    let mut removed: SmallVec<[DepNodeIndex; 4]> = SmallVec::new();
    for &old in list.iter() {
        if !reads.contains(&old) {
            removed.push(old);
        }
    }
    *list = reads.iter().copied().collect();
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
        shard
            .finishing
            .resize_with(slot + 1, || AtomicBool::new(false));
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

/// Blocks of the epoch table allocated up front — sized SEPARATELY from
/// [`DIRTY_INIT_BLOCKS`].
///
/// The two tables used to share one constant, which cost far more than it
/// looked: a dirty block is 512 bytes (4096 bits), but an epoch block is
/// `[AtomicU64; 4096]` = 32 KiB — 64 of each meant **2 MiB per `DepGraph`**
/// to cover the same 262144 nodes that 32 KiB of bitmap would cover.
/// Every `QuerySystem` paid that upfront, and `reset` reallocated the whole
/// 2 MiB on every `clear_all`.
///
/// 4 blocks = 16384 nodes = 128 KiB, which is still far above any real
/// node count; growth is append-only, rare (the count doubles) and cheap
/// (it `Arc`-clones the outer vec and leaves the blocks themselves
/// untouched), so a smaller head start costs one growth earlier on the
/// path to a genuinely enormous graph.  Do not "unify" this back with
/// `DIRTY_INIT_BLOCKS` — the asymmetry is the whole point.
const EPOCH_INIT_BLOCKS: usize = 4;

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
///
/// Zero-allocation on the common paths (perf item 4): an already-sorted
/// input is detected with one O(k) pass, and the small fan-out is sorted
/// in place.  Both are provably the SAME permutation the old
/// sort-of-indices produced — a stable sort (the old
/// `order.sort_by_key`) and an insertion sort with a STRICT `>`
/// comparator agree exactly, which is what keeps the positional pairing
/// intact rather than merely "still sorted".
fn sort_reads_by_shard(
    reads: &mut SmallVec<[DepNodeIndex; 8]>,
    mut epochs: Option<&mut SmallVec<[u64; 8]>>,
) {
    if reads.len() <= 1 {
        return; // nothing to permute; pairing trivially intact
    }
    // Already shard-sorted (common: a query reading a contiguous index
    // run, or one whose reads landed in allocation order): no permutation
    // and no allocation at all.
    let mut sorted = true;
    for pair in reads.iter().zip(reads.iter().skip(1)) {
        if shard_of(*pair.0) > shard_of(*pair.1) {
            sorted = false;
            break;
        }
    }
    if sorted {
        return;
    }
    if reads.len() <= 16 {
        // Small fan-outs: in-place insertion sort, stable, with the
        // epochs swapped in lockstep so the parallel pairing follows the
        // same swaps.  Beats the old allocate-an-index-vec + two collects:
        // k = 3 (a typical query) went from three heap round-trips to
        // none, and it stays inline up to k = 16.
        for i in 1..reads.len() {
            let mut j = i;
            while j > 0 && shard_of(reads[j - 1]) > shard_of(reads[j]) {
                reads.swap(j - 1, j);
                if let Some(e) = epochs.as_mut() {
                    e.swap(j - 1, j);
                }
                j -= 1;
            }
        }
        return;
    }
    // Large fan-outs: counting sort over the DEP_SHARDS buckets,
    // O(k + DEP_SHARDS).  Stable (equal shards keep declaration
    // order), and the same permutation is applied to both vectors
    // below either way, so the positional pairing is preserved.
    //
    // The permutation is applied IN PLACE by cycle rotation, so
    // the two `sorted_reads`/`sorted_epochs` output buffers are gone.
    // `order` is the one remaining allocation and it is unavoidable: the
    // counting path is reached only for k > 16, so no inline capacity
    // below 17 is ever useful here.
    let mut order: SmallVec<[usize; 8]> = (0..reads.len()).collect();
    let mut starts = [0usize; DEP_SHARDS];
    for &r in reads.iter() {
        starts[shard_of(r)] += 1;
    }
    let mut cursor = 0usize;
    for s in starts.iter_mut() {
        let bucket_start = cursor;
        cursor += *s;
        *s = bucket_start;
    }
    for (i, &r) in reads.iter().enumerate() {
        let b = shard_of(r);
        order[starts[b]] = i;
        starts[b] += 1;
    }
    // Apply `dst[i] = src[order[i]]` by rotating each cycle once.  The
    // walk runs along `order` itself (which is the inverse of the value's
    // destination direction), holding the cycle entry point out of the
    // vector and pulling each successor into place; overwriting
    // `order[i] = i` as we go marks the position placed, so each cycle is
    // visited exactly once and every later `start` in it is skipped.
    // `sort_reads_by_shard_keeps_the_epoch_pairing` pins the exact
    // resulting permutation against a reference stable sort.
    let mut eps = epochs;
    for start in 0..reads.len() {
        if order[start] == start {
            continue;
        }
        let tmp_r = reads[start];
        let tmp_e = eps.as_ref().map(|e| e[start]);
        let mut i = start;
        loop {
            let src = order[i];
            order[i] = i;
            if src == start {
                reads[i] = tmp_r;
                if let Some(e) = eps.as_mut() {
                    e[i] = tmp_e.expect("parity: epochs present iff starts[i]");
                }
                break;
            }
            reads[i] = reads[src];
            if let Some(e) = eps.as_mut() {
                e[i] = e[src];
            }
            i = src;
        }
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

/// Cached view of the dirty bitmap for one use, mirroring `VtableResolve`:
/// the `Cached` arm fences a `RefCell` borrow through `Ref::filter_map`
/// (no raw reference ever escapes the borrow tracker); the `Owned` arm
/// serves first uses, post-grow/reset staleness, and slot contention
/// without touching the slot at all.
enum DirtySnap<'a> {
    Cached(std::cell::Ref<'a, Vec<Arc<DirtyBlock>>>),
    Owned(Arc<Vec<Arc<DirtyBlock>>>),
}

impl std::ops::Deref for DirtySnap<'_> {
    type Target = Vec<Arc<DirtyBlock>>;
    fn deref(&self) -> &Self::Target {
        match self {
            DirtySnap::Cached(r) => r,
            DirtySnap::Owned(a) => a,
        }
    }
}

/// Cached view of the epochs table for one use, mirroring `DirtySnap`
/// (same borrow-fencing protocol — see `DirtySnap`).
enum EpochSnap<'a> {
    Cached(std::cell::Ref<'a, Vec<Arc<EpochBlock>>>),
    Owned(Arc<Vec<Arc<EpochBlock>>>),
}

impl std::ops::Deref for EpochSnap<'_> {
    type Target = Vec<Arc<EpochBlock>>;
    fn deref(&self) -> &Self::Target {
        match self {
            EpochSnap::Cached(r) => r,
            EpochSnap::Owned(a) => a,
        }
    }
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
pub struct DepGraph<S: Scheduler> {
    next_index: AtomicU32,
    /// One `CachePadded` lock per shard, exactly like the query system's
    /// `active_keys`: an unpadded `Vec<RwLock<..>>` puts neighbouring
    /// lock words on the same cache line, so two threads locking
    /// *different* shards would still invalidate each other's line —
    /// silently undoing the striping (Perf Book §13.7.3).
    ///
    /// AUDIT: `with_shard_w`/`with_shard_r`/`with_shard_and_grow` and the
    /// `take_*` combinators are the SOLE acquisition points — every grant
    /// must go through them so the tripwire (`sync_order::check` +
    /// `held_inc`) sees the hold and validates it against the §0 table,
    /// and so the ticket typestates stay exhaustive.  Direct
    /// `self.shards[i].read()/.write()` compiles but is an audit
    /// violation: the acquisition is invisible to the tripwire (no
    /// `check`, no `held_inc`) and the grep audit only covers ticket
    /// minting.  (Test code may read shards directly for white-box
    /// assertions — it holds no other family; production code may not.)
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
    /// Monotonic version of the dirty/epochs guard identity.
    ///
    /// # Sustaining invariant
    ///
    /// **Every `dirty.store(...)` AND every `epochs.store(...)` site MUST
    /// `fetch_add(1, Release)` this counter after the LAST store of the
    /// sequence.** There are exactly two store sequences (the grow path that
    /// appends a block to each table, and the quiesced [`Self::reset`] that
    /// swaps in fresh tables). The store-then-bump order lets snapshot
    /// readers treat a matching generation as proof that their cached guards
    /// observe every block the live tables had at some instant within the
    /// generation, and treat a mismatch as a signal to reload. See
    /// `dirty_snapshot` for the soundness argument. One counter covers both
    /// tables because every sequence stores both (grow) or swaps both
    /// (reset) — the bump always comes after ALL of the sequence's stores.
    /// The two sites are tagged `INVARIANT ANCHOR — table_gen::{grow,reset}`
    /// so a grep audit can confirm nothing escaped untagged; a future third
    /// table or one-table store sequence would silently break
    /// `dirty_snapshot`/`epochs_snapshot` otherwise.
    table_gen: AtomicU64,
    /// Per-thread cache of the [`Self::dirty`] and [`Self::epochs`] guards,
    /// keyed by generation.
    ///
    /// A fresh `ArcSwap` load costs an atomic pointer load plus an [`Arc`]
    /// refcount increment/decrement pair on a hot shared line.
    /// `is_green` runs once per cache hit *and* once per miss level, and
    /// `with_epoch_word` runs several times per miss level, so the cache
    /// holds `(generation, dirty guard, epochs guard)` per thread and the
    /// snapshot helpers validate with a single read-only `Acquire` load of
    /// `table_gen`. Field (not `thread_local!`): `DepGraph<S>` is
    /// generic over the scheduler, and the macro cannot capture outer
    /// generics.
    dirty_table_cache: thread_local::ThreadLocal<
        std::cell::RefCell<(
            u64,
            Option<Arc<Vec<Arc<DirtyBlock>>>>,
            Option<Arc<Vec<Arc<EpochBlock>>>>,
        )>,
    >,
    /// Per-durability-level mark counters.  `rev[l]` counts every 0→1
    /// mark of a node whose durability is `>= l` (a durability-`d` mark
    /// bumps levels `0..=d`; a HIGH change bumps all three).  The
    /// durability fast path serves a verified node without re-walking
    /// when its per-level stamp is `>= rev[node.dur]` — a HIGH-dur
    /// node's stamp survives LOW/MEDIUM marks because they never bump
    /// its level's counter.
    ///
    /// Soundness shape: `rev[d]`'s RMW is the mark's linearization point
    /// FOR LEVEL-`d` FAST SERVES.  It is NOT self-contained: because the
    /// global `verified_gen >= dirty_generation` check is consulted
    /// FIRST and only its failure routes into this check, the per-level
    /// add MUST be sequenced BEFORE the mark's `dirty_generation` bump
    /// (`mark_dirty` does exactly that).  Otherwise a verifier — which
    /// loads `current_gen` once at the top of `verify_inner` and
    /// `rev[dur]` later, in the root and interior fast paths — could see
    /// the generation bump while MISSING the add (two RMWs on different
    /// lines are not ordered into one another, even on x86-TSO), pass
    /// `vl >= rev[dur]` on the stale value, and re-admit a node whose
    /// closure was just marked without ever walking.  `mark_dirty` also
    /// keeps the add AFTER the `Release` bit-set, so a verifier that
    /// loads a bumped `rev[d]` is guaranteed to observe the red bit it
    /// belongs to.  Both orders are required; do not reorder either RMW.
    ///
    /// A fast-path reader that loads `rev[d]` before a mark's add
    /// linearizes before the mark (the same overlap doctrine as the
    /// global stamp check); one that loads after sees the bumped value
    /// and walks.  The global `dirty_generation` mechanism keeps its own
    /// linearization point (its bump, the last store of the mark) and its
    /// own doctrine, unchanged — only the ordering between the two is
    /// now specified rather than incidental.
    rev: [AtomicU64; Durability::LEVELS],
    /// Single-writer mutex for lowering events.  A lowering writes
    /// per-node level bits via lock-free CAS and then cascades along the
    /// reverse edges; while one is in flight the stored levels are
    /// non-uniform and a verification's fast-path verdict could rest on
    /// a stale-high level.  The writer brackets each event with an
    /// ODD→EVEN pair on [`DepGraph::lowering_epoch`]; the mutex
    /// guarantees at most one writer in flight, which is what makes the
    /// parity meaningful (two overlapping writers would scramble it:
    /// the second's even entry would hide the first's unfinished
    /// cascade).  Lowerings are rare finish-side events, so serializing
    /// them costs nothing; the mutex is a leaf lock (never taken while
    /// holding a shard lock — both call sites run outside any shard
    /// critical section), and the cascade inside holds only one shard
    /// READ lock at a time, so no deadlock cycle is possible.
    lowering_lock: Mutex<()>,
    /// Lowering seqlock: ODD = a lowering event (level CAS + reverse-edge
    /// cascade) is in flight, EVEN = stable.  `verify_green_transitive_with_dirty`
    /// samples the epoch before and after its fast-path verdict and only
    /// trusts a GREEN verdict taken in a stable, even window (see the
    /// retry loop there).  Reset zeroes it under quiescence.
    lowering_epoch: AtomicU64,
    /// Id of the owning `QuerySystem`.  A graph built by `new()` (a
    /// standalone/test graph) is minted a process-unique id instead — a
    /// shared sentinel such as 0 would let two independent graphs accept each
    /// other's node indices, which is the phantom-node corruption `read`'s
    /// guard exists to prevent (see `next_owner_id`).  Every task frame
    /// pushed through this graph is tagged with it, and [`DepGraph::read`]
    /// refuses to record a node of a different system (see
    /// `TaskDeps::system_id`).
    owner_id: usize,
    sched: Arc<S>,
}

// ── Mechanism A: the Quiesced proof token ─────────────────────────────

/// Zero-sized proof: "this thread currently holds the quiesce write lock
/// (⇒ no in-flight compute)".  `*mut ()` ⇒ !Send + !Sync: the proof is
/// thread-bound and cannot be transferred (otherwise another thread's lock
/// hold would be forged as this thread's).
///
/// Private construction: the only minting sites in the whole codebase are
/// the bodies of `QuerySystem::mutate_world` / `run_deferred_batch`, plus an
/// explicit escape hatch [`Quiesced::assume_for_test`] — grepping
/// `assume_for_test` lists every bypass point. The d10 family ("all
/// red-green changes must go through mutate_world") becomes a compile rule
/// instead of a documentation rule: all red-green mutation methods on
/// dep_graph take `&Quiesced`.
pub struct Quiesced(std::marker::PhantomData<*mut ()>);

impl Quiesced {
    /// Minting discipline: only allowed inside the bodies of
    /// `QuerySystem::mutate_world` and `run_deferred_batch` (the two
    /// chokepoints where the quiesce write lock is held).
    #[inline]
    pub(crate) fn mint() -> Self {
        Quiesced(std::marker::PhantomData)
    }

    /// Escape hatch: the legitimate bypass for non-quiesced channels (d24
    /// tests, dep_graph unit tests). Discipline: every call site must carry
    /// a one-line justifying comment.
    #[cfg(test)]
    pub fn assume_for_test() -> Self {
        Quiesced(std::marker::PhantomData)
    }
}

impl std::fmt::Debug for Quiesced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Quiesced")
    }
}

// ── Mechanism B: typestate lock-acquisition confinement (the compile face
// of ticket.rs's transition table) ─────────────────────────────────────

impl<S: Scheduler> DepGraph<S> {
    /// Entry minting. Allowed locations: the first line of each dep_graph
    /// entry-method body + tests with a justifying comment (grep audit in
    /// ticket.rs's module header).
    #[doc(hidden)]
    #[inline]
    pub(crate) fn ticket(&self) -> ticket::Held<ticket::Idle> {
        ticket::Held::fresh()
    }

    /// dep shard write lock: the sole acquisition point. The transition
    /// `St::After` is determined by the `CanTakeShard` table — `OneShard`/
    /// `ShardAndGrow`/`LoweringShard` do not implement that trait, so a
    /// second shard = E0277 (table #15).
    #[inline]
    fn take_shard_w<St: ticket::CanTakeShard>(
        &self,
        i: usize,
        _t: ticket::Held<St>,
    ) -> (ticket::DepShardW<'_>, ticket::Held<St::After>) {
        sync_order::check(sync_order::Fam::DepShard, "dep.take_shard_w");
        (
            ticket::DepShardW::new(self.shards[i].write()),
            ticket::Held::<St::After>::of(),
        )
    }

    /// dep shard read lock: the sole acquisition point (same as above).
    #[inline]
    fn take_shard_r<St: ticket::CanTakeShard>(
        &self,
        i: usize,
        _t: ticket::Held<St>,
    ) -> (ticket::DepShardR<'_>, ticket::Held<St::After>) {
        sync_order::check(sync_order::Fam::DepShard, "dep.take_shard_r");
        (
            ticket::DepShardR::new(self.shards[i].read()),
            ticket::Held::<St::After>::of(),
        )
    }

    /// grow_lock: only open to `OneShard` (tables #11/#12).
    #[inline]
    fn take_grow(
        &self,
        _t: ticket::Held<ticket::OneShard>,
    ) -> (ticket::GrowGuard<'_>, ticket::Held<ticket::ShardAndGrow>) {
        sync_order::check(sync_order::Fam::GrowLock, "dep.take_grow");
        (
            ticket::GrowGuard::new(self.grow_lock.lock()),
            ticket::Held::<ticket::ShardAndGrow>::of(),
        )
    }

    /// lowering_lock: only open to ∅ (table #14).
    #[inline]
    fn take_lowering(
        &self,
        _t: ticket::Held<ticket::Idle>,
    ) -> (
        ticket::LoweringGuard<'_>,
        ticket::Held<ticket::LoweringHeld>,
    ) {
        sync_order::check(sync_order::Fam::Lowering, "dep.take_lowering");
        (
            ticket::LoweringGuard::new(self.lowering_lock.lock()),
            ticket::Held::<ticket::LoweringHeld>::of(),
        )
    }

    /// Cascade shape: take one shard read while holding lowering (table #13).
    #[inline]
    fn take_cascade_shard_r(
        &self,
        i: usize,
        t: ticket::Held<ticket::LoweringHeld>,
    ) -> (ticket::DepShardR<'_>, ticket::Held<ticket::LoweringShard>) {
        self.take_shard_r(i, t)
    }

    /// Single-shard write critical-section combinator: no ticket and no
    /// minting authority inside the closure body → a second lock in the
    /// family does not compile. (Difference from the rejected closure-based
    /// API: there the guard was lent out via &self + interior mutability,
    /// and a shared borrow stacked on the closure could take another; here
    /// the sole acquisition entry consumes a ticket, and the ticket is not
    /// in the closure body.)
    #[inline]
    pub(crate) fn with_shard_w<R>(
        &self,
        i: usize,
        t: ticket::Held<ticket::Idle>,
        f: impl FnOnce(&mut DepShard) -> R,
    ) -> (R, ticket::Held<ticket::Idle>) {
        let (mut g, t1) = self.take_shard_w(i, t);
        let r = f(&mut g);
        (r, ticket::release_shard_w(g, t1))
    }

    /// Single-shard read critical-section combinator (same, read shape).
    #[inline]
    pub(crate) fn with_shard_r<R>(
        &self,
        i: usize,
        t: ticket::Held<ticket::Idle>,
        f: impl FnOnce(&DepShard) -> R,
    ) -> (R, ticket::Held<ticket::Idle>) {
        let (g, t1) = self.take_shard_r(i, t);
        let r = f(&g);
        (r, ticket::release_shard_r(g, t1))
    }

    /// allocate shape (the compile face of tables #11/#12):
    /// Idle → OneShard → ShardAndGrow → OneShard → Idle.
    #[inline]
    fn with_shard_and_grow<R>(
        &self,
        i: usize,
        t: ticket::Held<ticket::Idle>,
        f: impl FnOnce(&mut DepShard, &ticket::GrowGuard<'_>) -> R,
    ) -> (R, ticket::Held<ticket::Idle>) {
        let (mut g, t1) = self.take_shard_w(i, t);
        let (gg, t2) = self.take_grow(t1);
        let r = f(&mut g, &gg);
        let t3 = ticket::release_grow(gg, t2); // → OneShard (shard still held)
        (r, ticket::release_shard_w(g, t3)) // → Idle
    }
}

impl<S: Scheduler> DepGraph<S> {
    /// Mint a unique owner id for a standalone graph.
    ///
    /// Owner ids are the ONLY thing that makes [`Self::read`]'s cross-system
    /// guard reject a foreign node index: it compares the frame's
    /// `TaskDeps::system_id` against this graph's `owner_id`, so two graphs
    /// that share an id silently accept EACH OTHER'S node indices — exactly the
    /// phantom-node corruption (slot unallocated here, later aliased by a fresh
    /// allocation, plus a missing edge on the owning side) the guard exists to
    /// prevent.  `owner_id = 0` as a shared sentinel made every pair of
    /// independently constructed "standalone" graphs one such collision.
    ///
    /// This is the `NEXT_SYSTEM_ID` failure class one layer down, and the same
    /// reason applies: ids must not be a name two live graphs can both claim.
    /// The start offset sits far above `QuerySystem`'s `NEXT_SYSTEM_ID` range
    /// so a standalone graph can never alias a system id either, and
    /// `with_owner` still overrides it explicitly (a `QuerySystem` hand-tags
    /// its own graph).
    fn next_owner_id() -> usize {
        static NEXT_GRAPH_ID: AtomicUsize = AtomicUsize::new(1 << 48);
        NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn new(sched: Arc<S>) -> Self {
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
                (0..EPOCH_INIT_BLOCKS).map(|_| zero_epoch_block()).collect(),
            ),
            dirty_generation: AtomicU64::new(0),
            table_gen: AtomicU64::new(0),
            dirty_table_cache: thread_local::ThreadLocal::new(),
            rev: [const { AtomicU64::new(0) }; Durability::LEVELS],
            lowering_lock: Mutex::new(()),
            lowering_epoch: AtomicU64::new(0),
            // Minted, not `0`: a shared sentinel would let two standalone
            // graphs freely accept each other's node indices (see
            // `next_owner_id`).  `with_owner` overwrites it for the graphs a
            // `QuerySystem` owns.
            owner_id: Self::next_owner_id(),
            sched,
        }
    }

    /// A graph owned by the `QuerySystem` with id `owner_id`.  All task
    /// frames pushed through it are tagged so `read` can reject
    /// cross-system edges (see `TaskDeps::system_id`).
    pub fn with_owner(owner_id: usize, sched: Arc<S>) -> Self {
        let mut g = Self::new(sched);
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
    ///
    /// The compile face of table #12: `&GrowGuard` is the sole credential —
    /// without holding grow you cannot call it; inside the grow critical
    /// section acquiring any ticket-requiring lock does not compile (the
    /// ticket is in the caller's hands).
    fn grow_dirty(&self, idx: u32, _g: &ticket::GrowGuard<'_>) {
        let need = block_of(idx) + 1;
        // The caller (with_shard_and_grow / allocate_node_index) already
        // holds grow, so a single load is authoritative: the redundant
        // pre-check this used to do first could only re-read the same
        // table the caller's lock already pinned.
        let cur = self.dirty.load(); // caller holds the grow lock
        let ecur = self.epochs.load(); // same critical section
        // Both tables must be adequate for the early return to fire.
        // `EPOCH_INIT_BLOCKS < DIRTY_INIT_BLOCKS` means the two start out
        // asymmetric (the epochs head start is small on purpose — 32 KiB a
        // block), so a dirty-only check used to return with the epoch table
        // still short.  That is the UNSAFE skew direction: `mark_dirty`
        // would find the node's dirty block and then `bump_epoch` would
        // find no epoch word and drop the bump WITHOUT touching the
        // generation — which is precisely the mark the finish-side epoch
        // re-check depends on.  Grow here instead of silently degrading.
        if cur.len() >= need && ecur.len() >= need {
            return;
        }
        let new_len = need
            .next_power_of_two()
            .max(cur.len().max(ecur.len()) * 2)
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
        // Every table store must bump the shared generation (Release)
        // AFTER the last store of the sequence — see `table_gen`. Both
        // tables were stored above, so one bump covers both.
        //
        // INVARIANT ANCHOR — table_gen::grow
        // Grep this tag (with `table_gen::reset` below) to audit every
        // `dirty.store` / `epochs.store` site in this file: each site must
        // be followed by exactly one `table_gen.fetch_add(1, Release)`
        // placed after ALL of that sequence's stores.  A new site touching
        // only ONE table would break the invariant silently — the cached
        // slot's `(gen, dirty, epochs)` triplet would then be accepted
        // with one table at the new generation and the other at the old.
        self.table_gen.fetch_add(1, Ordering::Release);
    }

    /// Current dirty epoch of `node` — its value's revision counter,
    /// bumped on every dirty mark AND on every changed-fingerprint
    /// republish (`mark_dirty` / `publish` / `finish_task_and_publish`).
    /// `Acquire` load, paired with the `AcqRel` bumps.
    #[inline]
    fn epoch_of(&self, node: DepNodeIndex) -> u64 {
        self.with_epoch_word(node, |w| {
            w.map(|w| w.load(Ordering::Acquire) & EPOCH_MASK)
                .unwrap_or(0)
        })
    }

    /// Run `f` on `node`'s epoch word from the current snapshot, if its
    /// block exists (post-`reset` it may not).  The `ArcSwap` guard must
    /// never escape this frame — the word reference borrows it — so
    /// every access goes through this closure instead of returning the
    /// reference.
    #[inline]
    fn with_epoch_word<T>(&self, node: DepNodeIndex, f: impl FnOnce(Option<&AtomicU64>) -> T) -> T {
        let snap = self.epochs_snapshot();
        f(snap
            .get(block_of(node.0))
            .map(|b| &b[epoch_word_of(node.0)]))
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
    /// downward: re-lowering is a no-op, never a raise.  Returns `true`
    /// when the level ACTUALLY dropped — the trigger for reverse-edge
    /// propagation ([`DepGraph::lower_and_propagate`]); a no-op must not
    /// cascade (it would revisit settled readers forever).
    fn lower_durability_to(&self, node: DepNodeIndex, d: Durability) -> bool {
        self.with_epoch_word(node, |w| {
            let Some(w) = w else { return false };
            let mut cur = w.load(Ordering::Relaxed);
            loop {
                if word_dur(cur) <= d {
                    return false;
                }
                let next = (cur & EPOCH_MASK) | dur_bits(d);
                match w.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => return true,
                    Err(c) => cur = c,
                }
            }
        })
    }

    /// Reverse-edge propagation of a durability LOWERING: `start` has
    /// dropped to `to`; every direct reader's stored level must now be
    /// `<= to`, and every reader that itself dropped cascades further.
    ///
    /// This is what upgrades the finish-time `min(declared, reads)`
    /// aggregation from a per-node snapshot to a structural invariant
    /// `stored(X) <= min over closure(X) of stored` — the condition the
    /// per-level fast path's soundness rests on (a mark of any closure
    /// node then bumps `rev[stored(X)]` and kills the stamp).  Without
    /// it, a reader that finished before a read lowered keeps a
    /// stale-high level, and marks of the newly-low closure nodes
    /// bypass its stamp forever.
    ///
    /// Termination: levels are monotone downward with only
    /// [`Durability::LEVELS`] values, so each node drops at most
    /// `LEVELS - 1` times EVER; the `lower_durability_to` success rule
    /// doubles as the revisit filter (a second path arriving at an
    /// already-settled node finds `word_dur <= to` and stops), which
    /// also makes cycles harmless.  Explicit heap worklist: the reader
    /// graph can be deep, recursion could overflow.
    ///
    /// Lock discipline: one shard READ lock at a time (module invariant); the
    /// level writes are lock-free epoch-word CASes. Called only under
    /// [`DepGraph::lowering_lock`] (single writer). The compile face of
    /// table #13: LoweringHeld → LoweringShard → LoweringHeld.
    fn propagate_lowering(
        &self,
        start: DepNodeIndex,
        to: Durability,
        mut t: ticket::Held<ticket::LoweringHeld>,
    ) -> ticket::Held<ticket::LoweringHeld> {
        let mut work: Vec<(DepNodeIndex, Durability)> = vec![(start, to)];
        while let Some((node, level)) = work.pop() {
            // Walk this node's reverse edges IN PLACE under the same
            // guard — no `SmallVec` clone, hence no allocation under the
            // lock.  This mirrors `verify_inner`'s `take_shard_r` +
            // explicit `release_shard_r` shape: the old
            // `.cloned()` heap-allocated for reader sets wider than the
            // inline four, and a popular node (every query reads it) is
            // exactly the case where that matters.
            //
            // Holding the guard across the per-reader work is sound
            // because that work is LOCK-FREE: `lower_durability_to` is a
            // `with_epoch_word` CAS loop, and `with_epoch_word` only takes
            // the `epochs` table snapshot (a TLS cache probe plus an
            // index) — no nested dep-shard lock, and the `GrowLock` is
            // only reachable from `mark_dirty`, which is not on this path.
            // So the module invariant "one shard READ lock at a time" is
            // unchanged, and the only extra latency is against a WRITER
            // of this shard (a finisher publishing into it); readers take
            // the same lock shared and never block on us.
            //
            // The guard is also re-taken per node, so a deep cascade does
            // not pin any shard across the whole walk.
            let (guard, t1) = self.take_cascade_shard_r(shard_of(node), t);
            if let Some(readers) = (&*guard).rev_edges.get(slot_of(node)) {
                for &r in readers.iter() {
                    if self.lower_durability_to(r, level) {
                        work.push((r, level));
                    }
                }
            }
            t = ticket::release_cascade_shard_r(guard, t1);
        }
        t
    }

    /// One complete lowering EVENT: lower `node` to `to` and, if it
    /// actually dropped, cascade along the reverse edges — bracketed by
    /// the ODD→EVEN pair on [`DepGraph::lowering_epoch`] under
    /// [`DepGraph::lowering_lock`].
    ///
    /// The lock-free pre-check keeps the writer mutex cold on the
    /// overwhelmingly common no-op path (honest declarations, stable
    /// read sets).  It is racy, so the check is REPEATED under the
    /// lowering ticket below: two racing threads can both pass the
    /// pre-check, serialize on the ticket, and the loser would then find
    /// nothing to lower — yet toggle `lowering_epoch` odd all the same,
    /// discarding the buffered stamps of every concurrent verification
    /// and burning a `VERIFY_SEQLOCK_RETRIES` attempt on a lowering that
    /// never existed.  Only a thread that is about to REALLY lower pays
    /// the seal.
    ///
    /// Coverage of the propagation-cascade window W2: a verification
    /// racing this event samples `lowering_epoch` before and after its
    /// fast-path verdict and re-runs on any change or on odd — see the
    /// retry loop in [`DepGraph::verify_green_transitive_with_dirty`].
    /// Without the seal, a serve between the level CAS and the cascade
    /// landing on the reader could fast-path past a mark on the newly
    /// less-durable closure.  Coarse alternatives are unsound or
    /// wasteful: bumping `dirty_generation` merely forces one walk whose
    /// re-stamp is taken at the STILL stale-high level (the cascade has
    /// not landed), re-arming the hole for the next mark; bumping
    /// `rev[l]` for the abandoned levels has the same re-arm shape.
    pub(crate) fn lower_and_propagate(&self, node: DepNodeIndex, to: Durability) -> bool {
        if self.durability_of(node) <= to {
            return false;
        }
        let (writer, mut t_l) = self.take_lowering(self.ticket()); // Idle → LoweringHeld
        // Re-check under the ticket.  Another thread may have won the
        // race and already lowered this node (or its closure) to `to`;
        // bailing out BEFORE the odd bump keeps the seqlock quiet.
        if self.durability_of(node) <= to {
            drop(ticket::release_lowering(writer, t_l)); // → Idle
            return false;
        }
        self.lowering_epoch.fetch_add(1, Ordering::AcqRel); // → odd: in flight
        let lowered = self.lower_durability_to(node, to);
        if lowered {
            t_l = self.propagate_lowering(node, to, t_l);
        }
        self.lowering_epoch.fetch_add(1, Ordering::AcqRel); // → even: stable
        drop(ticket::release_lowering(writer, t_l)); // → Idle
        lowered
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
    ///
    /// Returns the node's durability as of the successful CAS — free of
    /// charge: the CAS only rewrote the low counter bits, so the swapped
    /// value already carries the durability bits unchanged.  `mark_dirty`
    /// uses this to merge its bump and its level read into ONE word
    /// access; every other caller ignores the return.
    #[inline]
    fn bump_epoch(&self, node: DepNodeIndex) -> Durability {
        self.with_epoch_word(node, |w| {
            let Some(w) = w else { return Durability::LOW };
            // CAS loop instead of `fetch_add`: the durability bits live
            // in the same word and must not absorb the carry.  An
            // epoch-counter overflow into bit 62 is unreachable in
            // practice (see `EPOCH_MASK`).
            let mut cur = w.load(Ordering::Relaxed);
            loop {
                let next = ((cur & EPOCH_MASK).wrapping_add(1) & EPOCH_MASK) | (cur & !EPOCH_MASK);
                match w.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire) {
                    // `next` preserved `cur`'s high bits verbatim, so
                    // this IS the durability at the successful swap —
                    // no second word read.
                    Ok(_) => return word_dur(next),
                    Err(c) => cur = c,
                }
            }
        })
    }

    /// Generation-validated snapshot of the dirty table: one `Acquire`
    /// load of `table_gen` plus a TLS slot hit replaces the
    /// `ArcSwap::load` (atomic load + refcount inc/dec RMWs) on every
    /// use.  Soundness: blocks are append-only and `Arc`-shared across
    /// guard versions, so a block PRESENT in a generation-matching slot
    /// is always current; a missing block (post-grow/post-reset world)
    /// falls through to the caller's existing missing-block arm, and a
    /// generation mismatch takes the seqlock slow path (`g1/load/g2`)
    /// that also refreshes the slot best-effort.  All borrows use
    /// `try_*` so unforeseen reentrancy degrades to `Owned`, never
    /// panics.
    #[inline]
    fn dirty_snapshot(&self) -> DirtySnap<'_> {
        let tbl_gen = self.table_gen.load(Ordering::Acquire);
        let cell = self
            .dirty_table_cache
            .get_or(|| std::cell::RefCell::new((u64::MAX, None, None)));
        if let Ok(r) = cell.try_borrow() {
            if r.0 == tbl_gen {
                if let Ok(mapped) = std::cell::Ref::filter_map(r, |t| t.1.as_ref().map(|a| &**a)) {
                    return DirtySnap::Cached(mapped);
                }
            }
        }
        loop {
            let g1 = self.table_gen.load(Ordering::Acquire);
            let full = self.dirty.load_full();
            let g2 = self.table_gen.load(Ordering::Acquire);
            if g1 == g2 {
                if let Ok(mut w) = cell.try_borrow_mut() {
                    w.0 = g2;
                    w.1 = Some(Arc::clone(&full));
                }
                return DirtySnap::Owned(full);
            }
        }
    }

    /// Generation-validated snapshot of the epochs table: same protocol
    /// as [`Self::dirty_snapshot`] (shared `table_gen`, shared slot, same
    /// soundness argument — epoch blocks are append-only and `Arc`-shared
    /// exactly like dirty blocks, and every caller already degrades a
    /// missing block to its conservative arm).
    #[inline]
    fn epochs_snapshot(&self) -> EpochSnap<'_> {
        let tbl_gen = self.table_gen.load(Ordering::Acquire);
        let cell = self
            .dirty_table_cache
            .get_or(|| std::cell::RefCell::new((u64::MAX, None, None)));
        if let Ok(r) = cell.try_borrow() {
            if r.0 == tbl_gen {
                if let Ok(mapped) = std::cell::Ref::filter_map(r, |t| t.2.as_ref().map(|a| &**a)) {
                    return EpochSnap::Cached(mapped);
                }
            }
        }
        loop {
            let g1 = self.table_gen.load(Ordering::Acquire);
            let full = self.epochs.load_full();
            let g2 = self.table_gen.load(Ordering::Acquire);
            if g1 == g2 {
                if let Ok(mut w) = cell.try_borrow_mut() {
                    w.0 = g2;
                    w.2 = Some(Arc::clone(&full));
                }
                return EpochSnap::Owned(full);
            }
        }
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
        // Declared here, assigned in the block branch: the 0→1 tail below
        // (after the early returns) consumes the level the bump's CAS
        // returned.  The missing-block branch returns without touching it.
        let d;
        if let Some(block) = self.dirty_snapshot().get(block_of(node.0)) {
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
            // Bump and level read merged (ONE word access instead of
            // `bump_epoch` + `durability_of`): the CAS's swapped value
            // already carries the durability.  Taking the level EARLIER
            // (before the bit set instead of after the generation bump)
            // is the conservative direction: the level only ever drops,
            // so an earlier read yields a level >= a later one and the
            // `0..=d` bump set below is a superset — over-bumping kills
            // extra fast paths, only MISSING a bump would be unsound.
            d = self.bump_epoch(node);
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
            // Per-level add BEFORE the generation bump, the same
            // load-bearing order as the live-block branch below (see
            // there for the straddle argument): a verifier that observes
            // the generation bump must also observe this add, or the
            // durability fast path could re-admit a node whose closure
            // was just marked.  Nothing is cached post-`reset` today, so
            // the order is doctrine rather than a live bug here.
            //
            // The per-level bump is conservative at LOW — the level every
            // real mark bumps, and the same skew the missing-durability
            // read takes everywhere else.  The early return used to skip
            // it, leaving this branch as the one place the "every 0→1
            // mark bumps the per-level counters" invariant was not
            // upheld (benign today — post-`reset` zeroes `rev` and every
            // stamp — but the invariant must not depend on that).
            self.rev[Durability::LOW as usize].fetch_add(1, Ordering::AcqRel);
            self.dirty_generation.fetch_add(1, Ordering::AcqRel);
            return;
        }
        // Per-level invalidation (durability): this mark's effect on the
        // level-`l` fast paths is its `rev[l]` add — the linearization
        // point documented on the `rev` field.  It is sequenced BETWEEN
        // the bit-set above and the generation bump below, and BOTH
        // orders are load-bearing:
        //
        //  * bit-set (Release) -> rev add (AcqRel).  A verifier that
        //    loads the bumped `rev[d]` also observes the red bit, so the
        //    `vl >= rev[d]` re-admission can never hide a bit this mark
        //    just set.
        //  * rev add -> generation bump.  THIS ONE IS THE FIX FOR THE
        //    gen/rev STRADDLE: the durability fast path is consulted only
        //    after the global `verified_gen >= current_gen` check has
        //    already failed, and it re-admits the node WITHOUT walking.
        //    `verify_inner` loads `current_gen` once at the top and
        //    `rev[dur]` later (root fast path, interior fast path), so
        //    with the bump first a verifier could load the bumped
        //    `current_gen`, then load a stale `rev[dur]` — on x86-TSO
        //    stores retire in program order, but two RMWs on different
        //    lines are not ordered into one another, so the generation
        //    store can be visible while the per-level add is not — and
        //    re-admit a node whose closure was just marked, writing a
        //    `verified_gen` stamp that then passes the global check
        //    forever.  With the add first the straddle is impossible:
        //    seeing the generation bump implies the add already landed.
        //    The reverse pattern (miss the bump, see the add) is the safe
        //    direction — it forces the walk.
        // The level `d` came free from the bump's CAS above (see there
        // for the earlier-read-is-conservative argument).  The
        // missing-block branch bumps only level 0 — the same conservative
        // skew the global bump just took (post-reset nothing is cached).
        //
        // Why `0..=(d as usize)`: the COUNT of levels bumped is exactly
        // this node's durability, which is the value `bump_epoch`'s CAS
        // promoted — the SAME word that the durability fast path reads in
        // `verify_inner` (`rev[dur_of(n)]`).  The contract is "every mark
        // of a node bumps every counter a verifier of the node could be
        // comparing against": a reader whose closure includes this node
        // reads the level from the same epoch word, so `d` is by
        // construction the highest `rev` index it can consult.  Bumping
        // fewer would let a stamp survive; bumping more would be legal
        // but would invalidate stamps of nodes that can no longer contain
        // this one.  `Durability::LEVELS` is the bound: `d as usize` is a
        // `LEVELS`-ranged discriminant and `rev` has `LEVELS` entries, so
        // the range can never run off the array — but if `Durability`
        // ever gains a level, `rev`'s length must grow with it (see the
        // field doc).
        for l in 0..=(d as usize) {
            self.rev[l].fetch_add(1, Ordering::AcqRel);
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
        // bump — the LAST store of the three, so everything a verifier
        // needs to see (the red bit, the per-level adds) is already
        // visible to anyone who sees it.  A walk that loads the
        // generation before the bump may serve a value that is, in
        // wall-clock time, already invalid — that is a valid
        // linearization (the walk overlaps the marking thread's
        // operation).  A walk that loads the generation after the bump is
        // guaranteed by the release/acquire chain (the bump is `AcqRel`
        // and sequenced after both the `Release` bit-set and the `AcqRel`
        // per-level adds above) to see the bit AND the bumped counters,
        // and to re-walk.  Reversing either order (bump → bit, or bump →
        // rev adds) would break a chain — a walk seeing the bumped
        // generation could miss the bit or the add, stamp with the bumped
        // generation, and the stamp would survive (no new bump) as a
        // permanent stale serve — and is unsound.  Concurrency:
        // a mark racing a walk's generation load may or may not be
        // reflected in that walk; any mark it misses makes the stamp it
        // writes stale relative to FUTURE lookups (the bump lands before
        // the stamp's reader loads the generation on its next visit) —
        // the conservative direction.  Two racing 0→1 marks on one bit
        // may double-bump: harmless over-invalidation.
        self.dirty_generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Clear the dirty bit for `node`.  Same append-only-snapshot
    /// discipline as [`DepGraph::mark_dirty`].
    #[inline]
    fn clear_dirty(&self, node: DepNodeIndex) {
        if let Some(block) = self.dirty_snapshot().get(block_of(node.0)) {
            block[word_of(node.0)].fetch_and(!mask_of(node.0), Ordering::Release);
        }
    }

    /// Allocate a node with its DECLARED content class (fixed for the
    /// node's lifetime; taken from the owning query's descriptor).
    ///
    /// The compile face of tables #11/#12: inside `with_shard_and_grow`,
    /// Idle → OneShard → ShardAndGrow → OneShard → Idle; `grow_dirty` only
    /// accepts `&GrowGuard`.
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
        let (_, _t) = self.with_shard_and_grow(shard_of(idx), self.ticket(), |shard, growth| {
            self.grow_dirty(idx.0, growth);
            ensure_shard_len(shard, slot);
            // Written unconditionally, NOT via `resize_with(.., || class)`:
            // a concurrent allocator's resize must never leave THIS node's
            // slot holding a foreign class — the class is fixed at
            // allocation and drives the semantic-purity debug checks and
            // class-scoped invalidation.
            shard.classes[slot] = class;
        });
        idx
    }

    /// The declared content class of a node, if the node exists.
    pub fn class_of(&self, node: DepNodeIndex) -> Option<ContentClass> {
        let (r, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            shard.classes.get(slot_of(node)).copied()
        });
        r
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
                // Counted in BOTH modes and BEFORE the outcome: the debug
                // panic and the release skip are two different fates for
                // the same contract violation, and the count is what
                // makes the release one observable.  See
                // `note_cross_system_skip`.
                note_cross_system_skip(other, self.owner_id, task.system_id);
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
            // after the reverse-edge flush.  Taken only when `read` reports
            // the node as freshly inserted, so a duplicate read does not pay
            // a TLS snapshot round trip for a value that would be discarded.
            if task.read(other) {
                task.read_with_epoch(other, self.epoch_of(other));
            }
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
            let (r, _t) = self.with_shard_w(shard_of(task_node), self.ticket(), |shard| {
                ensure_shard_len(shard, tslot);
                reconcile_forward_edges(&mut shard.edges[tslot], &reads)
            });
            r
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
            let (_, _t) = self.with_shard_w(s, self.ticket(), |shard| {
                ensure_shard_len(shard, max_slot);
                for &r in group {
                    if let Some(list) = shard.rev_edges.get_mut(slot_of(r)) {
                        list.retain(|n| *n != task_node);
                    }
                }
            });
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
            let (_, _t) = self.with_shard_w(s, self.ticket(), |shard| {
                ensure_shard_len(shard, max_slot);
                for &r in group {
                    edge_insert(&mut shard.rev_edges[slot_of(r)], task_node);
                }
            });
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
            let (_cls, _t) = self.with_shard_r(s, self.ticket(), |shard| {
                assert_semantic_reads(group, shard, task_node, task_class);
            });
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
    ///
    /// NOTE: EVERY production caller discards this with
    /// `let _ =` (`get_shared_inner`'s two paths) — the caller has no
    /// decision to make, because the marking already happened here.  It
    /// is kept because ~30 test call sites assert on it, and a return
    /// type change would churn all of them for no behavioural gain.
    /// Do not "fix" the `let _ =` into a real use: there is nothing to
    /// branch on.  (If that ever changes, the natural consumer is a
    /// metric — e.g. an invalidation fan-out counter.)
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
        // An ACTUAL drop propagates along the reverse edges immediately
        // (`lower_and_propagate`): every direct reader — and transitively
        // every reader of a reader — must not keep a level higher than
        // min over its reads, or the per-level fast path serves past
        // marks on the newly-less-durable closure.  Readers whose own
        // finish is still in flight are not in the snapshot; their own
        // aggregation re-reads the levels at THEIR finish, and the
        // post-flush re-check below covers the residual window.
        if !reads.is_empty() {
            // ONE snapshot for the whole pass — and it is the
            // TLS-cached `epochs_snapshot()`, not a raw `ArcSwap::load`
            // The promotion buys back the full load +
            // refcount pair that aggregation #1 used to pay.  The
            // in-section suspect check below ALREADY uses this form, and
            // the safety argument is identical (see its comment).
            let snap = self.epochs_snapshot();
            let mut min_dur = Durability::HIGH;
            for &r in reads.iter() {
                let d = match snap.get(block_of(r.0)) {
                    Some(b) => word_dur(b[epoch_word_of(r.0)].load(Ordering::Acquire)),
                    None => Durability::LOW,
                };
                if d < min_dur {
                    min_dur = d;
                }
            }
            drop(snap);
            self.lower_and_propagate(node, min_dur);
        }

        // Stale forward edges dropped by the reconciliation below; the
        // pruning of their reverse edges runs AFTER the critical section
        // releases (the removed deps may live in OTHER shards — never
        // two shard locks at once).  The third element is the publish-time
        // fan-out reader list, cloned into a POOLED buffer: an inline
        // SmallVec heap-allocates once the node has more
        // than four direct readers — under this very lock, which every
        // reader's own finish must take) so the marks can run lock-free
        // outside the section (see the comment at its only writer).
        // The in-section suspect re-check's epoch snapshot is taken
        // HERE, before the write lock, rather than inside the closure.  The
        // closure captures it by reference; the soundness argument for
        // taking it early (the late re-check below is a strict superset and
        // restores anything this one misses, with `finishing` covering the
        // whole method) is written at the use site.
        let suspect_epochs = self.epochs_snapshot();
        let (changed_in_section, removed, mut fanout): (
            bool,
            SmallVec<[DepNodeIndex; 4]>,
            FanoutBuf,
        ) = {
            let (out, _t) = self.with_shard_w(shard_of(node), self.ticket(), |shard| {
                let slot = slot_of(node);
                ensure_shard_len(shard, slot);
                let mut fanout = FanoutBuf::take();

                // ---- finish_task part: forward edges, RECONCILED to the
                // current read set (stale edges removed — see
                // `reconcile_forward_edges`).  The removed deps' reverse
                // edges are pruned AFTER this critical section releases
                // (the removed deps may live in OTHER shards — never two
                // shard locks at once).
                let removed = if let Some(tn) = task_node {
                    let tslot = slot_of(tn);
                    ensure_shard_len(shard, tslot);
                    reconcile_forward_edges(&mut shard.edges[tslot], &reads)
                } else {
                    SmallVec::new()
                };

                // ---- publish part ----
                let changed = if let Some(fp) = fp {
                    let changed = shard.fingerprints[slot].is_some_and(|old| old != fp);
                    shard.fingerprints[slot] = Some(fp);
                    // NEW: resolve the colour BEFORE the node can be observed
                    // green.  A read whose epoch advanced during this compute
                    // may have been republished green already (its fan-out
                    // missed us because our rev edge is not flushed yet), so
                    // the transitive walk of a concurrent getter would find
                    // the whole closure green and serve our stale-based value
                    // in the [clear_dirty .. late re-check] window.  The
                    // in-section check is lock-free (one snapshot, then an
                    // atomic load per read) and positionally correct:
                    // `reads`/`read_epochs` are still the unsorted,
                    // parallel-by-construction vectors here.
                    //
                    // d24: the residual sub-window [this check .. the
                    // post-flush re-check] is closed by the per-node
                    // `finishing` flag: `get_shared` sets it before the
                    // staged value becomes reachable in the cache and this
                    // method clears it after the tail below — so a getter
                    // whose transitive verify runs inside the window sees the
                    // node as not-yet-verifiable and takes the compute/wait
                    // path instead of serving.
                    let suspect = task_node.is_some() && {
                        // The snapshot is HOISTED out of the shard
                        // write closure (declared just above it) so the TLS
                        // probe, the `table_gen` load and the `Arc` refcount
                        // pair do not run under this node's write lock — the
                        // lock every reader's own `flush_rev_edges` and every
                        // verifier of the node must take.
                        //
                        // Taking it earlier is SOUND because this check is a
                        // strict subset of the late re-check below, which runs
                        // at a strictly later point (after the fan-out, the
                        // rev-edge flush and the pruning) against the same
                        // positional comparison.  A mark that lands between
                        // this hoisted snapshot and the closure's lock
                        // acquisition therefore costs at most one spurious
                        // `clear_dirty`, which the late re-check restores
                        // BEFORE `clear_finishing` — and `finishing` (d24)
                        // already closes the whole method, so no verifier can
                        // serve across that repair either way.
                        reads.iter().zip(read_epochs.iter()).any(|(&r, &e)| {
                            suspect_epochs.get(block_of(r.0)).map_or(0, |b| {
                                b[epoch_word_of(r.0)].load(Ordering::Acquire) & EPOCH_MASK
                            }) != e
                        })
                    };
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
                    // ─── CONCURRENT FAN-OUT MARK OF THIS NODE ───
                    //
                    // The colour decision just taken is NOT the end of the
                    // story for THIS node, and the reason is that the
                    // publish-time fan-out below marks readers LOCK-FREE.
                    // Concretely: N is finishing cleanly, N read R, and R
                    // republishes with a CHANGED fingerprint while N is
                    // inside this section.  R's fan-out then calls
                    // `mark_dirty(N)` — no shard lock, so it can land
                    // anywhere, including in the window between the suspect
                    // check above and the `clear_dirty(N)` that erases it.
                    //
                    // That erase is not a lost mark.  Two mechanisms repair
                    // it, and BOTH are needed because neither covers the
                    // other's window:
                    //
                    //  (a) THE LATE RE-CHECK (step 5 below) re-marks N.
                    //      A fan-out only ever runs in the `if changed` arm,
                    //      and that same arm calls `bump_epoch(R)` inside
                    //      the section — so every fan-out mark is
                    //      ACCOMPANIED by a bump of the republished node's
                    //      own epoch.  N captured R's epoch at record time
                    //      (`read_epochs[i]`), hence before the bump, so
                    //      the positional re-check below sees
                    //      `epoch_of(R) != read_epochs[i]` and restores N's
                    //      dirty bit.  Anything in
                    //      [this clear_dirty .. that re-check] is repaired
                    //      here.
                    //
                    //  (b) THE RESIDUAL WINDOW (that re-check ..
                    //      `clear_finishing`) is closed by `finishing`.
                    //      A fan-out mark landing there survives — nothing
                    //      after this point clears the bit — and a verifier
                    //      of R that walks into N inside the window is told
                    //      N is "not yet verifiable" (d24) and takes the
                    //      compute/wait path, so R's staged value is never
                    //      served across it.
                    //
                    // The chain spans `finish_task_and_publish` (both the
                    // in-section decision and the late re-check) and
                    // `get_shared` (which installs the `finishing` flag
                    // before the staged value becomes reachable).  Do not
                    // simplify either half in isolation: dropping the late
                    // re-check re-opens (a), and dropping `finishing`
                    // re-opens (b).
                    if changed {
                        // Same unmarked-value-change bump as `publish` (see
                        // there): an in-flight reader's finish re-check is
                        // epoch-based and otherwise blind to value changes
                        // that arrive without a mark.
                        self.bump_epoch(node);
                // Perf item 1: the fan-out is O(readers) x 4–6
                // atomic RMWs, and it used to run here — under THIS
                // NODE'S SHARD WRITE LOCK, which every reader's own
                // finish (`flush_rev_edges` → `with_shard_w` on the
                // read's shard) and every verifier of the node must
                // take.  A popular node's republish therefore
                // serialized the whole graph behind its own
                // invalidation.  `mark_dirty` is lock-free (dirty
                // bitmap + `rev` counters + generation bump), so the
                // marks only need the READER LIST, not the lock:
                // clone it into a POOLED buffer (`FanoutBuf`) and mark below,
                // after the section
                // releases.  The pool matters precisely here: the
                // buffer is FILLED under this lock, and an inline
                // `SmallVec` heap-allocates as soon as the node has
                // more than four direct readers — a malloc inside the
                // critical section, on the tail-latency path of the
                // hottest nodes.
                        //
                        // Soundness of the move, in the order a reader
                        // checking it would want:
                        //
                        // 1. THE WINDOW IS CLOSED BY `finishing`.  The gap
                        //    this deferral opens — [fingerprint stored here
                        //    .. `mark_dirty(R)` below] — lies entirely
                        //    inside `finish_task_and_publish`, and `clear_finishing(node)`
                        //    is the LAST thing this method does (after the
                        //    fan-out, `flush_rev_edges`, the late re-check
                        //    and the latch set).  `verify_inner` treats a
                        //    finishing node as dirty and returns it, so no
                        //    concurrent verifier of R can walk into N, see
                        //    the new fingerprint and serve R from cache
                        //    across the gap: it gets "dirty" and takes the
                        //    compute/wait path (d24).  The deferral therefore
                        //    cannot widen the observable window at all.
                        //
                        // 2. The ordering that makes a verifier which DOES
                        //    walk observe R as red is carried by `mark_dirty`'s
                        //    memory order, not by this lock: the bit is set
                        //    (Release) BEFORE the per-level adds and the
                        //    generation bump, so any thread that sees a
                        //    bumped `dirty_generation` also sees the bit.
                        //    R is also often in a DIFFERENT shard, which this
                        //    write lock never covered.
                        //
                        // 3. `bump_epoch(node)` deliberately STAYS inside the
                        //    section: moving it out would open a window where
                        //    the new fingerprint is visible but the node's
                        //    epoch is still the old one.
                        if let Some(readers) = shard.rev_edges.get(slot) {
                            fanout.fill(readers);
                        }
                    }
                    changed
                } else {
                    // `fp: None` — the eval_always and panic paths (see the
                    // `fp: None` doc above).  The colour decision is SKIPPED
                    // here, and that is deliberate: with no fingerprint there
                    // is nothing a reader could have observed, so "changed")
                    // is meaningless and there are no direct readers to
                    // fan out to.  The node simply KEEPS whatever colour it
                    // already had — a node that was red when the task started
                    // stays red, which is exactly what keeps a panicking
                    // node's never-published value unreachable.  In
                    // particular, do NOT add a `clear_dirty` here to
                    // "finalize" the node.  Nothing else in this method
                    // depends on `changed`: the fan-out list stays empty, the
                    // rev-edge flush still runs, and the late re-check below
                    // is driven by `read_epochs`, not by `changed`.
                    false
                };
                (changed, removed, fanout)
            });
            out
        };

        // Publish-time fan-out, now that the section has released: one
        // lock-free `mark_dirty` per direct reader of this node.  Same
        // marks, same order as before the deferral (perf item 1).
        //
        // The binding is a `FilledFanout` newtype, so nothing
        // mutable is in scope here.  The old shape exposed
        // `cleared() -> &mut Vec` for the fill side and the consume side
        // reached for it by mistake — see `FanoutBuf::fill` for the bug
        // that caused.
        let fanout = fanout.filled();
        for r in fanout.iter() {
            self.mark_dirty(r);
        }

        // ---- reverse-edge flush for this task's reads: one lock per
        // READ shard ----
        if let Some(tn) = task_node {
            // Finish window: in-section colour decision taken, rev-edge
            // flush not yet run — no shard lock is held at this park site.
            self.sched
                .suspend(SchedulePoint::FinishWindow { node: tn.0 });
            // Sort reads and their captured epochs TOGETHER (identical
            // permutation) before the flush — the epoch re-check below is
            // positional and must compare each node against ITS OWN
            // start-of-task revision.
            sort_reads_by_shard(&mut reads, Some(&mut read_epochs));
            self.flush_rev_edges(tn, &mut reads);
            // Drop the reverse edges of reads this recomputation no
            // longer has (see `reconcile_forward_edges` / g3).
            self.prune_rev_edges(tn, removed);
            // ---- late-fan-out detection + late-propagation re-check ----
            // (a) LATE FAN-OUT (d16): the publish-time fan-out above
            //     reaches only readers whose reverse edges ALREADY
            //     existed when `node` republished.  A task that read
            //     `node` while computing has its reverse edge inserted
            //     only by the flush above — a fan-out that ran before
            //     that insert never reached it, yet this task's freshly
            //     published value may encode the pre-invalidation
            //     snapshot of such a read.  Each read captured its
            //     node's dirty epoch at record time; any mark of a read
            //     node since then (bump ordered before the bit set, so
            //     this Acquire load cannot miss a mark whose red bit was
            //     observable) means the read went red during this task's
            //     compute: restore this task's dirty bit so the suspect
            //     value is never served green.  Deliberately
            //     conservative: a mark whose republished fingerprint is
            //     unchanged also costs one recomputation.
            //
            // (b) LATE PROPAGATION (the durability twin of (a)): a read
            //     whose LEVEL lowered during this task's compute is
            //     missed twice — by aggregation #1 (it read the
            //     pre-lowering level) and by that read's own cascade
            //     (this task's rev edge was not flushed yet).  Now that
            //     the edge is flushed, take the reads' CURRENT levels
            //     and re-aggregate; any drop propagates from here, and
            //     the mark below fires with the SETTLED level so its
            //     per-level bumps are exact.  The early-exit `break` of
            //     the old loop is gone: the min collection needs the
            //     full pass anyway.
            //
            // Both checks share ONE lock-free pass over `reads` (the
            // epoch load and the level load hit the same epoch word — one
            // combined read per read) and ONE snapshot guard, taken here
            // instead of per read: same append-only/`reset`-excluded
            // safety argument as aggregation #1 above.  `reads` is
            // shard-sorted at this point, so the block index walk is
            // ascending — cache friendly on top of the load savings.  The
            // snapshot is the TLS-cached `epochs_snapshot()`:
            // same argument as aggregation #1 above.
            let snap = self.epochs_snapshot();
            let mut min_dur = Durability::HIGH;
            let mut suspect = false;
            for (i, &r) in reads.iter().enumerate() {
                let (epoch, d) = match snap.get(block_of(r.0)) {
                    Some(b) => {
                        let v = b[epoch_word_of(r.0)].load(Ordering::Acquire);
                        (v & EPOCH_MASK, word_dur(v))
                    }
                    None => (0, Durability::LOW),
                };
                if d < min_dur {
                    min_dur = d;
                }
                if epoch != read_epochs[i] {
                    suspect = true;
                }
            }
            drop(snap);
            self.lower_and_propagate(tn, min_dur);
            if suspect {
                self.mark_dirty(tn);
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

        changed_in_section
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
    /// Shard write lock, single small critical section: plain storage is
    /// enough here.
    ///
    /// QUIESCENCE: the gate itself runs OUTSIDE `mutate_world` —
    /// `QuerySystem::invalidate_if_changed` gates first (so its bool is
    /// exact on the d11 deferral path) and only the INVALIDATION enters
    /// the quiesced channel.  This is sound because (a) `digests` is
    /// written by no one else (the gate and `record_digest` are the sole
    /// writers, both under this shard lock) and (b) the gate touches no
    /// red-green state (no red bit, no epoch/generation/`rev`).  The
    /// invariant to preserve: a digest write must never race a compute
    /// that READS this digest inside its closure — if a query ever starts
    /// reading digests, the gate must move under quiescence.
    pub(crate) fn digest_gate(&self, node: DepNodeIndex, input_digest: u64) -> bool {
        let slot = slot_of(node);
        let (changed, _t) = self.with_shard_w(shard_of(node), self.ticket(), |shard| {
            ensure_shard_len(shard, slot);
            let changed = match shard.digests[slot] {
                Some(old) => old != input_digest,
                None => true,
            };
            shard.digests[slot] = Some(input_digest);
            changed
        });
        changed
    }

    /// Record `input_digest` for `node` WITHOUT any comparison or
    /// invalidation — the driver-side companion to [`DepGraph::digest_gate`]
    /// for the first iteration, where the node does not exist yet when the
    /// gate runs (a cold system's `lookup_node` returns `None`, so the
    /// gate has nothing to store against).  Called after the fan-out that
    /// created the nodes, so the NEXT iteration's gate can compare.
    pub(crate) fn record_digest(&self, node: DepNodeIndex, input_digest: u64) {
        let slot = slot_of(node);
        let (_, _t) = self.with_shard_w(shard_of(node), self.ticket(), |shard| {
            ensure_shard_len(shard, slot);
            shard.digests[slot] = Some(input_digest);
        });
    }

    /// Mark a node as dirty.  LAZY invalidation (replaces the old eager
    /// transitive BFS): only the node itself is marked red.  Readers are
    /// not touched here — propagation happens lazily, after the node is
    /// recomputed and its fingerprint compared: an UNCHANGED fingerprint
    /// keeps readers green (they stay cached); a CHANGED fingerprint
    /// marks only the DIRECT readers dirty (see [`DepGraph::dirty_readers`]),
    /// and they re-verify in turn when next requested.
    /// d10 family: the compile gate for all "external" red-green mutations —
    /// the caller must present `&Quiesced` (minted inside
    /// `mutate_world`/`run_deferred_batch` scope, or via
    /// `assume_for_test` in tests).
    pub(crate) fn invalidate(&self, node: DepNodeIndex, _q: &Quiesced) {
        // O(1) atomic RMW, seqlock-guarded: invalidators serialise on
        // nothing at all — not even a shard read lock.  (The generation
        // bump lives in `mark_dirty`: every red-bit creator bumps, not
        // just this path.)
        self.mark_dirty(node);
    }

    pub(crate) fn dirty_readers(&self, node: DepNodeIndex, _q: &Quiesced) {
        // Read lock, not write: we only read `rev_edges`.  The dirty
        // marks themselves need no shard lock (seqlock-guarded atomics).
        let (_, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            if let Some(readers) = shard.rev_edges.get(slot_of(node)) {
                for &r in readers {
                    self.mark_dirty(r);
                }
            }
        });
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
    pub(crate) fn dirty_readers_by_class(
        &self,
        node: DepNodeIndex,
        class: ContentClass,
        _q: &Quiesced,
    ) {
        let mut readers: SmallVec<[DepNodeIndex; 8]> = {
            let (r, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
                shard
                    .rev_edges
                    .get(slot_of(node))
                    .map(|r| r.iter().copied().collect())
                    .unwrap_or_default()
            });
            r
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
            let (_, _t) = self.with_shard_r(s, self.ticket(), |shard| {
                for &r in &readers[start..i] {
                    if shard
                        .classes
                        .get(slot_of(r))
                        .is_some_and(|&c| c.intersects(class))
                    {
                        self.mark_dirty(r);
                    }
                }
            });
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
    pub(crate) fn publish(&self, node: DepNodeIndex, fp: u64, _q: &Quiesced) -> bool {
        let slot = slot_of(node);
        let (changed, _t) = self.with_shard_w(shard_of(node), self.ticket(), |shard| {
            ensure_shard_len(shard, slot);
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
        });
        changed
    }

    pub(crate) fn set_fingerprint(&self, node: DepNodeIndex, fp: u64) {
        let slot = slot_of(node);
        let (_, _t) = self.with_shard_w(shard_of(node), self.ticket(), |shard| {
            ensure_shard_len(shard, slot);
            shard.fingerprints[slot] = Some(fp);
        });
    }

    pub fn fingerprint(&self, node: DepNodeIndex) -> Option<u64> {
        let (r, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            shard.fingerprints.get(slot_of(node)).copied().flatten()
        });
        r
    }

    /// The sole multi-shard function. The ticket mechanism (mechanism B) does
    /// not cover it: ascending one at a time, never two held at once is
    /// guaranteed by this function body; must be under quiesce.write
    /// (`&Quiesced` is the compile credential, privacy-confined — only one
    /// call site: `clear_all_impl`).
    pub(crate) fn reset(&self, _q: &Quiesced) {
        self.next_index.store(1, Ordering::Relaxed);
        // Per-level mark counters restart at 0, in lockstep with the
        // cleared stamps.  Safe under quiescence: no mark can be in
        // flight (`reset` runs under `mutate_world`), so no `rev` add
        // can land after the zeroing.  Same for the lowering seqlock:
        // quiescence guarantees no lowering event is in flight, so the
        // epoch restarts at an EVEN (stable) value.
        for r in &self.rev {
            r.store(0, Ordering::Relaxed);
        }
        self.lowering_epoch.store(0, Ordering::Relaxed);
        // Clear shard by shard — one write lock at a time (the only
        // multi-shard path; ascending order, and never two locks held at
        // once, so no deadlock).  Routed through `with_shard_w` so the
        // tripwire validates the InitLock → DepShard edge (#20) at this
        // real call site (clear_all_impl holds init_lock above us);
        // `reset` is a cross-family entry, so minting here is legal per
        // the ticket.rs minting discipline.
        for i in 0..self.shards.len() {
            // `mem::take` under the lock, drop AFTER it is released: the
            // deallocation (the only O(capacity) work here) runs outside
            // the critical section — the same take-and-drop-outside shape
            // the dirty/epoch swap below already uses.  `with_shard_w`
            // returns the taken vectors, so the guard is released before
            // the loop-end `drop`.
            let (
                (
                    taken_edges,
                    taken_rev,
                    taken_fp,
                    taken_classes,
                    taken_vgen,
                    taken_vlvl,
                    taken_digests,
                    taken_finishing,
                ),
                _t,
            ) = self.with_shard_w(i, self.ticket(), |sh| {
                (
                    std::mem::take(&mut sh.edges),
                    std::mem::take(&mut sh.rev_edges),
                    std::mem::take(&mut sh.fingerprints),
                    std::mem::take(&mut sh.classes),
                    // verified_gen MUST be cleared too: `next_index` restarts at
                    // 1, so freshly allocated nodes reuse indices whose slots
                    // still hold pre-reset stamps — a new node would inherit a
                    // claim that its (not yet recorded) closure was verified,
                    // and the `vg >= current_gen` fast path would serve stale
                    // cache entries without ever walking the closure.
                    std::mem::take(&mut sh.verified_gen),
                    // Per-level durability stamps die with the generation stamps:
                    // post-reset every `rev` counter restarts at 0 below, so a
                    // surviving stamp (== some pre-reset counter value) would
                    // compare `>= 0` and fast-path every reused node without a
                    // walk — the per-level twin of the `verified_gen` trap above.
                    std::mem::take(&mut sh.verified_lvl),
                    // Digest gate state dies with everything else: a reused node
                    // index must not inherit a pre-reset digest (it would let the
                    // gate skip an invalidation for content that never fed the
                    // old node's value).
                    std::mem::take(&mut sh.digests),
                    // Same for the finish-in-progress flags: post-reset no node
                    // is finishing (quiescence guarantees no compute is in
                    // flight), and a stale `true` would make the reused index
                    // unverifiable until something cleared it.
                    std::mem::take(&mut sh.finishing),
                )
            });
            drop((
                taken_edges,
                taken_rev,
                taken_fp,
                taken_classes,
                taken_vgen,
                taken_vlvl,
                taken_digests,
                taken_finishing,
            ));
        }
        // Append-only bitmap: swap in a fresh zero bitmap under the
        // growth lock.  A mark racing this reset lands on the replaced
        // snapshot and is dropped — the safe direction, because reset
        // also cleared every cache and active-key entry (nothing cached
        // can be stale).
        let _tw_grow = sync_order::Acquired::acquire(sync_order::Fam::GrowLock, "reset.grow");
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
        // dirty bitmap.  `EPOCH_INIT_BLOCKS`, not `DIRTY_INIT_BLOCKS`: an
        // epoch block is 32 KiB against a dirty block's 512 bytes, so
        // sizing the head start together cost 2 MiB per `clear_all`.
        // Growth re-syncs the pair (`grow_dirty` now checks BOTH tables
        // before its early return, so the epoch table can never be left
        // short — which would be the unsafe skew: a missing epoch word
        // makes `bump_epoch` drop the bump and the generation with it).
        self.epochs.store(Arc::new(
            (0..EPOCH_INIT_BLOCKS).map(|_| zero_epoch_block()).collect(),
        ));
        // Shared-generation bump AFTER the last store of the sequence —
        // see `table_gen`. Both tables were swapped above, so one bump
        // covers both and no thread can cache a mixed old/new pair.
        //
        // INVARIANT ANCHOR — table_gen::reset
        // Second and last store sequence in this file.  A grep audit
        // (`dirty.store`, `epochs.store`) must find exactly two anchors.
        // See the grow anchor's note for why the pairing matters.
        self.table_gen.fetch_add(1, Ordering::Release);
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
        let (r, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            shard.verified_gen.get(slot_of(node)).copied().unwrap_or(0)
        });
        r
    }

    /// Test-only view of a per-level mark counter.
    #[cfg(test)]
    pub(crate) fn rev_of(&self, d: Durability) -> u64 {
        self.rev[d as usize].load(Ordering::Acquire)
    }

    /// Test-only view of a node's per-level durability stamp.
    #[cfg(test)]
    pub(crate) fn verified_lvl_of(&self, node: DepNodeIndex) -> u64 {
        let (r, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            shard.verified_lvl.get(slot_of(node)).copied().unwrap_or(0)
        });
        r
    }

    /// Test hook for the cross-system leak regression: the forward edges
    /// of `node` (its recorded dependency read set).
    #[cfg(test)]
    pub(crate) fn forward_edges_of(&self, node: DepNodeIndex) -> Vec<DepNodeIndex> {
        let (r, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            shard
                .edges
                .get(slot_of(node))
                .map(|l| l.to_vec())
                .unwrap_or_default()
        });
        r
    }

    /// Check whether a node is green (up-to-date, not dirty).
    ///
    /// Lock-free: a generation-validated TLS snapshot replaces the
    /// per-hit `ArcSwap::load` (one `Acquire` gen load on the fast path;
    /// see `dirty_snapshot`).  This is THE hot path — every cached
    /// query hit goes through it, so even a *shared* lock here costs a
    /// coherence transfer per hit (Perf Book §13.1).  The `Acquire` loads
    /// pair with the `Release` RMWs in `mark_dirty` / `clear_dirty`, so
    /// a dirty mark published before a dependent `get()` is visible to
    /// it.  A missing block is only possible after `reset()` cleared the
    /// whole world — "green" (nothing cached to be stale) is the safe
    /// answer there.
    #[inline]
    pub fn is_green(&self, node: DepNodeIndex) -> bool {
        self.dirty_snapshot()
            .get(block_of(node.0))
            .is_none_or(|b| b[word_of(node.0)].load(Ordering::Acquire) & mask_of(node.0) == 0)
    }

    /// Mark a node as re-computed (green again).  Lock-free: the bitmap
    /// RMW is seqlock-guarded against swaps, so no shard lock is needed.
    #[inline]
    pub(crate) fn mark_green(&self, node: DepNodeIndex, _q: &Quiesced) {
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
        let (_, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            if let Some(f) = shard.finishing.get(slot_of(node)) {
                f.store(true, Ordering::Release);
            }
        });
    }

    /// Clear the "finish in progress" flag: the node's colour is now
    /// final (green or red per the in-section decision and the post-flush
    /// re-check).
    #[inline]
    pub(crate) fn clear_finishing(&self, node: DepNodeIndex) {
        let (_, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            if let Some(f) = shard.finishing.get(slot_of(node)) {
                f.store(false, Ordering::Release);
            }
        });
    }

    /// Test-only view of a node's finishing flag.
    #[cfg(test)]
    pub(crate) fn is_finishing(&self, node: DepNodeIndex) -> bool {
        let (r, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            shard
                .finishing
                .get(slot_of(node))
                .is_some_and(|f| f.load(Ordering::Acquire))
        });
        r
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
    ///
    /// Ticket traversal (mechanism B): `verify_inner` does one Idle round-trip
    /// read critical section per branch; the ticket travels with the loop,
    /// and every early return carries the ticket back (ZST, no cost).
    pub fn verify_green_transitive_with_dirty(&self, node: DepNodeIndex) -> Option<DepNodeIndex> {
        // ── Lowering seqlock (durability propagation) ────────────────
        // A durability lowering (`lower_and_propagate`) writes per-node
        // level bits via lock-free CAS and then cascades along the
        // reverse edges.  While one is in flight the stored levels are
        // NON-UNIFORM: a fast-path verdict computed against a stale-high
        // level could serve past a mark on the newly-less-durable closure
        // (window W2 — the mark already happened, the reader's level
        // repair has not landed).  The writer brackets each event with an
        // ODD→EVEN pair on `lowering_epoch`; this side samples the epoch
        // before and after the verdict and only trusts a GREEN verdict
        // taken in a stable, even window — such a serve linearizes at the
        // re-sample, after every lowering whose cascade completed earlier
        // and before any lowering that starts later.  DIRTY verdicts need
        // no seal: a recompute is sound under any interleaving, so they
        // return immediately.
        //
        // The GLOBAL (generation) fast path is inherently lowering-safe
        // and is NOT gated by the seal: a passing `verified_gen >=
        // current_gen` means no mark at all since the stamp — lowerings
        // never mark, so no closure node can be red in a mark-free
        // window, regardless of levels.  Only the per-LEVEL disjunct
        // (root and interior) rests on the `stored(X) <= min closure
        // levels` invariant that an in-flight cascade is mid-repair, and
        // only it is disabled in the fallback walk.
        //
        // ── Why the stamps are BUFFERED (and a deliberate
        //    simplification left on the table) ──
        // The exit frames below push their stamps into a buffer that is
        // flushed only after the stable-window check, and a verdict whose
        // window overlapped a lowering discards the buffer with the
        // verdict.  Strictly speaking the buffering is NOT necessary: a
        // stamp is a "true at stamp time" claim ("this closure verified
        // green at gen G / level counter K"), and its FORWARD validity is
        // carried by the level counters themselves — once the cascade
        // lands, the stamped node's stored level is lowered, so the stamp
        // fails the faster counter on its next check even if some
        // closure node was marked before the stamp was written.  Writing
        // the two fields inline (the pre-propagation direct write)
        // therefore also sound, and `PendingStamp`/`flush_stamps` could
        // be deleted; the only thing that MUST stay is the seqlock over
        // the fast-path VERDICTS.  The buffer is kept because it makes
        // the seal argument local — no stamp enshrines a verdict computed
        // inside an unstable window — and because the walk is not the
        // hot path.  Implementers of the simplification: delete
        // `PendingStamp`/`flush_stamps`, restore the inline write in the
        // exit frame, keep everything else; this paragraph plus the
        // `lower_and_propagate` doc is the proof obligation.
        //
        // One pooled stamp buffer per verification (perf item 2): taken
        // out of TLS here and cleared before each walk, so the retry loop
        // reuses it and an overlap storm no longer allocates a fresh
        // `Vec` per attempt.  `Drop` returns it to TLS on every exit path.
        let mut buf = StampBuf::take();
        for _ in 0..VERIFY_SEQLOCK_RETRIES {
            let s1 = self.lowering_epoch.load(Ordering::Acquire);
            let stamps = buf.cleared();
            let verdict = self.verify_inner(node, true, stamps);
            self.sched
                .suspend(SchedulePoint::VerifyWindow { node: node.0 });
            let s2 = self.lowering_epoch.load(Ordering::Acquire);
            if s1 & 1 == 1 || s1 != s2 {
                // A lowering overlapped this verification: the level
                // reads inside are not uniformly pre- or post-cascade.
                // Discard the buffered stamps with the verdict and
                // re-run — on the retry the levels have settled (or the
                // window is stable and the verdict seals).
                continue;
            }
            if verdict.is_none() {
                self.flush_stamps(stamps);
            }
            return verdict;
        }
        // Pathological fallback: lowerings kept overlapping every retry
        // (effectively unreachable — see `VERIFY_SEQLOCK_RETRIES`).  A
        // walk with the durability fast path DISABLED (root and
        // interior): every red bit is read directly, sound under any
        // interleaving.  The generation shortcut stays — it is
        // lowering-safe (see above).  No seqlock seal is needed here:
        // without the level shortcuts the walk reads actual colours, and
        // a recompute verdict is conservative under any interleaving.
        let stamps = buf.cleared();
        let verdict = self.verify_inner(node, false, stamps);
        if verdict.is_none() {
            self.flush_stamps(stamps);
        }
        verdict
    }

    /// The verification walk proper: root fast paths + iterative
    /// post-order DFS, factored out of
    /// [`DepGraph::verify_green_transitive_with_dirty`] so the seqlock
    /// retry loop can re-run it.  `allow_durability_fast_path` gates ONLY
    /// the per-LEVEL shortcuts (root and interior) — the generation
    /// shortcuts are lowering-safe and stay on in both modes.  Exit
    /// frames push into `stamps` instead of writing the shards directly;
    /// [`DepGraph::flush_stamps`] commits them after the seqlock window
    /// closes.
    fn verify_inner(
        &self,
        node: DepNodeIndex,
        allow_durability_fast_path: bool,
        stamps: &mut Vec<PendingStamp>,
    ) -> Option<DepNodeIndex> {
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
        // (3.2) Lazily-hoisted epochs snapshot for EVERY durability read
        // of this walk: the root fast path, the interior per-level fast
        // path and the exit-frame stamping all read the SAME table, and a
        // per-node `durability_of` paid a fresh TLS round-trip each time.
        // Lazy rather than eager because the dominant hit (root
        // `verified_gen`) touches no epoch word at all — a snapshot taken
        // unconditionally here would be a pure loss on the hottest path.
        // Same append-only / `reset`-excluded argument as
        // `finish_task_and_publish`; a missing block reads LOW, the
        // conservative direction.  Lowerings overlapping the walk are
        // sealed by the seqlock retry in
        // `verify_green_transitive_with_dirty` exactly as before — every
        // level read now linearizes at snapshot time, still inside the
        // sealed s1..s2 window.
        let mut epochs: Option<EpochSnap<'_>> = None;
        let mut dur_of = |n: DepNodeIndex| -> Durability {
            let snap = epochs.get_or_insert_with(|| self.epochs_snapshot());
            snap.get(block_of(n.0)).map_or(Durability::LOW, |b| {
                word_dur(b[epoch_word_of(n.0)].load(Ordering::Acquire))
            })
        };
        {
            let (verdict, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
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
                // Durability fast path (structural soundness — see
                // `Durability`): the closure was verified at per-level stamp
                // `verified_lvl`, and no mark of durability >= this node's
                // has linearized since (`rev`'s RMW is the level's
                // linearization point — a reader loading a lower value
                // linearizes before the mark, the same overlap doctrine as
                // the global check above).  Sound because the finish-time
                // aggregation plus the reverse-edge propagation
                // (`propagate_lowering`) maintain `stored(X) <= min over
                // closure(X) of stored`: a mark of any closure node then
                // bumps this level's counter and kills the stamp.  Every
                // existing query is LOW, so this disjunct only ever fires for
                // queries that opted in; the seqlock-fallback walk disables
                // it.
                if allow_durability_fast_path {
                    let vl = shard.verified_lvl.get(slot_of(node)).copied().unwrap_or(0);
                    if vl >= self.rev[dur_of(node) as usize].load(Ordering::Acquire) {
                        return None;
                    }
                }
                None
            });
            if let Some(dirty) = verdict {
                return Some(dirty);
            }
        }

        // NOTE (DEFERRED — measured-first) the root fast
        // path above takes the NODE'S shard READ lock on every cached hit.
        // A read lock is a shared, uncontended-taken atomic in the common
        // case, so this is cheap, but it is still far from the lock-free
        // load the `epochs` word offers.  A possible move was
        // to pack `finishing` + `verified_gen` + `verified_lvl` into one
        // lock-free word alongside the durability bits (or a small array
        // of them), so a hit costs three loads and no lock.
        //
        // NOT DONE, for two reasons.  (1) The repo has no benchmark
        // harness, so "MEASURE first" has nothing to measure with — the
        // only available signal is a full-suite wall-clock, which is
        // dominated by compile time, not query traffic.  (2) d24 makes it
        // non-trivial: `set_finishing` writes the flag through
        // `with_shard_r`, so moving the flag out of the shard means
        // either a second, differently-ordered store on the finish path
        // (and d24's whole point is that the write is serialized against
        // the shard section that publishes the fingerprint) or a lock-free
        // word whose store order must be re-derived from scratch.
        //
        // So this is a real opportunity that needs a benchmark to justify
        // the risk.  If it is ever done, the invariant to preserve is
        // that the three reads stay in this order: `finishing` FIRST
        // (a finishing node must never be fast-pathed on a stamp), then
        // `verified_gen` against `current_gen`, then the durability level
        // against `rev`.
        // across traversals via TLS instead of reallocated per walk.
        // `(node, exiting)` frames: `false` = enter, `true` = all deps
        // verified, stamp post-order.
        let mut scratch = VerifyScratch::take();
        let visited = &mut scratch.visited;
        let stack = &mut scratch.stack;
        stack.push((node, false));
        while let Some((n, exiting)) = stack.pop() {
            if exiting {
                // All green: stamp this node with the current generation
                // AND the per-level mark counter for its durability — the
                // durability fast path's valid_rev.  BUFFERED (see the
                // seqlock loop in
                // `verify_green_transitive_with_dirty`): the pair is
                // committed by `flush_stamps` only after the seqlock
                // window closes, so no stamp enshrines a verdict computed
                // inside an unstable window — deliberate conservatism,
                // not a soundness requirement (the correction argument
                // for deleting the buffer lives in that loop's comment).
                stamps.push(PendingStamp {
                    node: n,
                    generation: current_gen,
                    lvl: self.rev[dur_of(n) as usize].load(Ordering::Acquire),
                });
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
            // ONE read lock for the whole enter branch, held in the open
            // (not lent to a `with_shard_r` closure) so it outlives the
            // edge walk below.  The d24 finishing check, both fast paths
            // and the dependency-edge fetch all read shard `shard_of(n)`,
            // and the slow path now iterates the edge list IN PLACE under
            // the same guard — no `SmallVec` clone, hence no allocation
            // under the lock (the old `.cloned()` heap-allocated for edge
            // sets wider than the inline four).  Releasing explicitly on
            // every exit path is the same ticket hand-off `with_shard_r`
            // performs internally; the ticket is a fresh `Held<Idle>` per
            // node, so no state threads across iterations.  Still strictly
            // read-only: the shard's write lock stays exclusive to the
            // exit frame's post-order stamp.
            //
            // Widening note: the guard now also covers the local
            // `stack.push`es.  They touch no shard (no nested lock, so no
            // deadlock cycle).  The per-edge work is the same work the old
            // code did — it just moves from after the lock to under it —
            // but the `SmallVec` allocation, and its malloc/free under the
            // shard lock, is gone.
            let (guard, tkt) = self.take_shard_r(shard_of(n), self.ticket());
            let shard = &*guard;
            let slot = slot_of(n);
            // NEW (d24): finishing nodes are not verifiable — treat as
            // dirty.  Returning (instead of `continue`) also guarantees
            // the node is never exit-stamped while its colour is still
            // unfinalized.
            if shard
                .finishing
                .get(slot)
                .is_some_and(|f| f.load(Ordering::Acquire))
            {
                let _ = ticket::release_shard_r(guard, tkt);
                return Some(n);
            }
            // Fast path: the CLOSURE was already verified at this
            // generation, so skip re-walking its edges.
            if shard.verified_gen.get(slot).copied().unwrap_or(0) >= current_gen {
                let _ = ticket::release_shard_r(guard, tkt);
                continue;
            }
            // Per-level durability fast path for interior nodes — same
            // structural contract as the root check above: skips the
            // subtree walk when no mark of this node's durability or
            // higher has linearized since its stamp.  Defaults keep every
            // existing query on the global check; the seqlock-fallback
            // walk disables it.
            if allow_durability_fast_path {
                if let Some(&vl) = shard.verified_lvl.get(slot) {
                    if vl >= self.rev[dur_of(n) as usize].load(Ordering::Acquire) {
                        let _ = ticket::release_shard_r(guard, tkt);
                        continue;
                    }
                }
            }
            // Both fast paths missed: defer the stamp until every
            // dependency has verified, then enter all direct dependencies
            // (reversed so LIFO pops them in declaration order — keeps the
            // "first dirty found" stable with the old recursive walk),
            // reading this node's CURRENT edge set in place.  Pushing the
            // exit frame before the dep frames is immaterial: it sits
            // below them on the LIFO stack either way, and an early
            // `Some(dirty)` abort discards it exactly as before.
            stack.push((n, true));
            if let Some(edges) = shard.edges.get(slot) {
                for &dep in edges.iter().rev() {
                    stack.push((dep, false));
                }
            }
            let _ = ticket::release_shard_r(guard, tkt);
        }
        None
    }

    /// Commit the buffered post-order stamps of a completed,
    /// seqlock-sealed verification: grouped by shard, one write lock at a
    /// time (module invariant).  `verified_gen` and `verified_lvl` land
    /// under the same lock, so a reader taking the per-level fast path
    /// always sees a fully written pair.
    fn flush_stamps(&self, stamps: &mut Vec<PendingStamp>) {
        if stamps.is_empty() {
            return;
        }
        stamps.sort_unstable_by_key(|s| shard_of(s.node));
        let mut i = 0;
        while i < stamps.len() {
            let shard_idx = shard_of(stamps[i].node);
            let mut j = i;
            while j < stamps.len() && shard_of(stamps[j].node) == shard_idx {
                j += 1;
            }
            let (_, _t) = self.with_shard_w(shard_idx, self.ticket(), |shard| {
                for s in &stamps[i..j] {
                    ensure_shard_len(shard, slot_of(s.node));
                    shard.verified_gen[slot_of(s.node)] = s.generation;
                    shard.verified_lvl[slot_of(s.node)] = s.lvl;
                }
            });
            i = j;
        }
        stamps.clear();
    }
}

impl Default for DepGraph<NoOpScheduler> {
    fn default() -> Self {
        Self::new(Arc::new(NoOpScheduler))
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::super::TestScheduler;
    use super::*;

    /// Test escape hatch: dep_graph unit tests drive the red-green mutators
    /// directly, without `mutate_world` (there is no QuerySystem there). See
    /// `Quiesced::assume_for_test`.
    fn q() -> Quiesced {
        Quiesced::assume_for_test()
    }

    fn setup_chain(dg: &DepGraph<NoOpScheduler>) -> (DepNodeIndex, DepNodeIndex, DepNodeIndex) {
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
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let (a, b, c) = setup_chain(&dg);
        dg.invalidate(c, &q());
        assert!(!dg.is_green(c));
        assert!(dg.is_green(b), "lazy invalidation: readers must stay green");
        assert!(dg.is_green(a));
    }

    /// A changed fingerprint marks only DIRECT readers dirty (one level).
    #[test]
    fn test_fingerprint_change_dirties_direct_readers_only() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let (a, b, c) = setup_chain(&dg); // a reads b reads c
        dg.set_fingerprint(c, 1);
        dg.dirty_readers(c, &q());
        assert!(!dg.is_green(b), "b reads c directly, must be dirtied");
        assert!(
            dg.is_green(a),
            "a reads b only transitively, must stay green"
        );
    }

    #[test]
    fn test_fingerprint_roundtrip() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let n = dg.allocate_node_index(ContentClass::SEMANTIC);
        assert_eq!(dg.fingerprint(n), None);
        dg.set_fingerprint(n, 42);
        assert_eq!(dg.fingerprint(n), Some(42));
        dg.set_fingerprint(n, 43);
        assert_eq!(dg.fingerprint(n), Some(43));
    }

    #[test]
    fn test_rev_edges_independent_nodes() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let a = dg.allocate_node_index(ContentClass::SEMANTIC);
        let b = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.start_task(a);
        dg.read(b);
        dg.finish_task();
        dg.invalidate(a, &q());
        assert!(!dg.is_green(a));
        assert!(dg.is_green(b));
    }

    /// Fan-out under lazy invalidation: a fingerprint change on the root
    /// dirties ALL direct readers (the fan-out is one level).
    #[test]
    fn test_rev_edges_fan_out() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        let x = dg.allocate_node_index(ContentClass::SEMANTIC);
        let y = dg.allocate_node_index(ContentClass::SEMANTIC);
        let z = dg.allocate_node_index(ContentClass::SEMANTIC);
        for &reader in &[x, y, z] {
            dg.start_task(reader);
            dg.read(root);
            dg.finish_task();
        }
        dg.invalidate(root, &q());
        assert!(!dg.is_green(root));
        assert!(dg.is_green(x), "invalidate alone must not propagate");
        assert!(dg.is_green(y));
        assert!(dg.is_green(z));
        dg.dirty_readers(root, &q());
        assert!(!dg.is_green(x));
        assert!(!dg.is_green(y));
        assert!(!dg.is_green(z));
    }

    #[test]
    fn test_allocate_node_resizes_rev_edges() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
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
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
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
        let dg = Arc::new(DepGraph::new(Arc::new(NoOpScheduler)));
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
                        dg.invalidate(n, &q());
                        assert!(!dg.is_green(n));
                        dg.mark_green(n, &q());
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
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
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
        dg.dirty_readers_by_class(root, ContentClass::TRIVIA, &q());
        assert!(
            dg.is_green(sem_reader),
            "semantic reader must not be woken by a trivia-scoped invalidation"
        );
        assert!(dg.is_green(doc_reader));
        assert!(!dg.is_green(trivia_reader));

        // A class mask covering DOC+TRIVIA wakes both of those readers.
        dg.dirty_readers_by_class(root, ContentClass::DOC | ContentClass::TRIVIA, &q());
        assert!(dg.is_green(sem_reader));
        assert!(!dg.is_green(doc_reader));

        // The unfiltered API still marks every direct reader.
        dg.dirty_readers(root, &q());
        assert!(!dg.is_green(sem_reader));
    }

    /// Declared classes are recorded per node and queryable.
    #[test]
    fn test_class_of() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
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
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        // Nodes 1..=64: shard_of(n) = n & 63, so node 64 sits in shard 0.
        for _ in 0..64 {
            let _ = dg.allocate_node_index(ContentClass::SEMANTIC);
        }
        // Node 65 lives in shard 1; give it a dirty history (epoch 1)
        // while node 64 stays at 0 — the unequal pair a scrambled
        // comparison miscouples.
        let later = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.invalidate(later, &q());
        assert!(
            !dg.publish(later, 1, &q()),
            "first publish: nothing changed"
        );
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
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let e = dg.allocate_node_index(ContentClass::SEMANTIC);
        let r = dg.allocate_node_index(ContentClass::SEMANTIC);

        // E's initial value: fp 1, epoch 0.
        assert!(!dg.publish(e, 1, &q()));

        // R reads green E (captures E's current epoch).
        dg.start_task(r);
        dg.read(e);

        // E republished with a CHANGED fingerprint — no mark, no
        // invalidation (the eval_always shape).
        assert!(dg.publish(e, 7, &q()), "fp changed");

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
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        // Warm-up mark: at generation 0 the GLOBAL stamp check is
        // trivially true (everything is "already verified", no walk, no
        // per-level stamp) — force a real walk by raising the generation
        // with a HIGH-level invalidate.
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm, &q());

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
        dg.invalidate(low, &q());
        assert_eq!(dg.rev_of(Durability::LOW), 2);
        assert_eq!(dg.rev_of(Durability::HIGH), 1);
        // Fast path: still all-green at level HIGH, no walk needed.
        assert_eq!(dg.verify_green_transitive_with_dirty(root), None);

        // A HIGH-level mark (stdlib): bumps all three levels — the
        // stamp no longer covers rev[HIGH], so the walk runs and finds
        // the red dep.
        dg.invalidate(dep, &q());
        assert_eq!(dg.verify_green_transitive_with_dirty(root), Some(dep));
    }

    /// The min(declared, reads) aggregation: a declared-HIGH node that
    /// reads a LOW node is aggregated DOWN to LOW, so a LOW mark DOES
    /// invalidate its fast path.  Declared durability never flows
    /// upward.
    #[test]
    fn durability_aggregates_down_over_reads() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let low_dep = dg.allocate_node_index(ContentClass::SEMANTIC); // LOW
        let high_reader = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(high_reader, Durability::HIGH);
        dg.start_task(high_reader);
        dg.read(low_dep);
        assert!(!dg.finish_task_and_publish(high_reader, Some(1)));
        assert_eq!(dg.durability_of(high_reader), Durability::LOW);

        // Now a LOW mark invalidates the reader's (LOW) fast path.
        assert_eq!(dg.verify_green_transitive_with_dirty(high_reader), None);
        dg.invalidate(low_dep, &q());
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
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        // Warm-up mark: raise the generation so the first verify actually
        // walks (at generation 0 the global stamp check is trivially
        // true and no per-level stamp is written).
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm, &q());

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
        dg.invalidate(root, &q());
        assert!(!dg.is_green(root));
        assert_eq!(dg.verify_green_transitive_with_dirty(root), Some(root));

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
        dg.invalidate(dep, &q());
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
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
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
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        // Warm-up mark: raise the generation so the first verify actually
        // walks and writes a per-level stamp.
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm, &q());

        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(root, Durability::HIGH);
        dg.start_task(root);
        dg.finish_task();
        assert!(!dg.finish_task_and_publish(root, Some(1)));
        assert_eq!(dg.verify_green_transitive_with_dirty(root), None);
        assert!(dg.verified_lvl_of(root) > 0);

        let low = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.invalidate(low, &q());
        dg.reset(&q());
        assert_eq!(dg.rev_of(Durability::LOW), 0);
        assert_eq!(dg.rev_of(Durability::HIGH), 0);
        assert_eq!(dg.verified_lvl_of(root), 0);
    }

    /// A mark whose dirty block is MISSING (beyond the initial
    /// bitmap, e.g. post-`reset`) must still uphold the per-level
    /// invariant: the `rev[LOW]` counter is bumped conservatively, not
    /// skipped.  Benign today (post-`reset` zeroes `rev` and every
    /// stamp), but the invariant must not depend on that.
    #[test]
    fn mark_dirty_missing_block_still_bumps_low_level() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let before = dg.rev_of(Durability::LOW);
        // Node index 9_999_999 lives in block 2441 — far beyond the
        // 64 initial blocks, so mark_dirty takes the missing-block
        // branch.
        dg.invalidate(DepNodeIndex(9_999_999), &q());
        assert_eq!(
            dg.rev_of(Durability::LOW),
            before + 1,
            "a missing-block mark must bump rev[LOW] (conservative level)"
        );
        // HIGH is NOT bumped — the node's durability is unknown (LOW is
        // the conservative read), and over-bumping HIGH would needlessly
        // kill the HIGH fast path.
        assert_eq!(dg.rev_of(Durability::HIGH), 0);
    }

    /// The cross-system guard's release-mode skip must be OBSERVABLE,
    /// not silent: the guard counts in every build (before the mode's
    /// outcome), so a release build that quietly drops edges leaves a
    /// trail.
    ///
    /// Drives the guard directly through two graphs with different owner
    /// ids — cheaper than going through a whole `QuerySystem` pair (see
    /// `defense_tests::tls_partition_cross_system_same_key_no_false_cycle`
    /// for the end-to-end shape, whose release arm asserts the value
    /// still flows).  The count is global and process-wide, so the
    /// assertion is on the DELTA; no other test violates the pairing
    /// contract, so the delta is exactly the number of reads issued
    /// here.
    #[test]
    fn cross_system_skip_is_counted_and_reported() {
        let a = DepGraph::<NoOpScheduler>::with_owner(1, Arc::new(NoOpScheduler));
        let b = DepGraph::<NoOpScheduler>::with_owner(2, Arc::new(NoOpScheduler));
        let foreign = a.allocate_node_index(ContentClass::SEMANTIC);

        let before = super::cross_system_skip_count();
        // A task frame owned by `b` reading into `a` is the violation.
        b.start_task(foreign);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            a.read(foreign);
        }));
        a.finish_task();
        // Debug panics (the pairing guard's loud mode); release skips.
        // Either way the edge must not be recorded.
        #[cfg(debug_assertions)]
        r.err().expect("debug builds must panic on a cross-system edge");
        #[cfg(not(debug_assertions))]
        r.expect("release builds skip, never panic");
        let after = super::cross_system_skip_count();
        assert!(
            after > before,
            "a dropped cross-system edge must be counted (was {before}, now {after})"
        );
    }

    /// A graph must never be left with the dirty table LONGER than the
    /// epoch table (safety side of `EPOCH_INIT_BLOCKS`).
    ///
    /// The epoch table's head start is deliberately smaller than the dirty
    /// table's (an epoch block is 32 KiB, a dirty block 512 bytes), so the
    /// pair starts asymmetric.  `grow_dirty`'s old early return looked only
    /// at the dirty table's length, which meant a graph crossing the epoch
    /// boundary but not the dirty one could allocate nodes whose dirty
    /// block exists while their EPOCH BLOCK DOES NOT.  That is the unsafe
    /// skew: `mark_dirty` would find the bit, `bump_epoch` would find no
    /// word and silently drop the bump — while still taking the
    /// `dirty_generation` path — so a finish-side epoch re-check comparing
    /// `epoch_of(node)` against a captured value would see 0 vs 0 and
    /// conclude "unchanged".
    ///
    /// Allocates past `EPOCH_INIT_BLOCKS` (well inside `DIRTY_INIT_BLOCKS`)
    /// and then proves a mark of the LAST node really does bump its epoch.
    #[test]
    fn epoch_table_never_lags_the_dirty_table() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let n = super::EPOCH_INIT_BLOCKS * 4096 + 8;
        let nodes: Vec<DepNodeIndex> = (0..n)
            .map(|_| dg.allocate_node_index(ContentClass::SEMANTIC))
            .collect();
        let late = *nodes.last().expect("nonempty");

        // The safe direction is `dirty.len() <= epochs.len()`.
        assert!(
            dg.dirty_snapshot().len() <= dg.epochs_snapshot().len(),
            "the epoch table must never be the shorter one: dirty={} epochs={}",
            dg.dirty_snapshot().len(),
            dg.epochs_snapshot().len(),
        );

        // And the real consequence: a mark of a LAST-BLOCK node must bump
        // its epoch word.  Before the fix this read 0 — the bump was
        // dropped because the block was never allocated.
        assert_eq!(
            dg.epoch_of(late),
            0,
            "a freshly allocated node's epoch starts at 0"
        );
        dg.mark_dirty(late);
        assert_eq!(
            dg.epoch_of(late),
            1,
            "mark_dirty must bump the epoch of a node in the last-grown \
             block — 0 means the bump was dropped for a missing block"
        );
    }

    /// Two independently built standalone graphs must NOT share an owner id
    /// (regression for `owner_id = 0` as a shared sentinel).  The cross-system
    /// guard in `read` compares `TaskDeps::system_id` against `owner_id`, so a
    /// shared id let graph A's task frames swallow graph B's node indices:
    /// the slot is unallocated here and a later allocation aliases it, while
    /// the real edge on B is never recorded — permanent staleness, silently.
    ///
    /// A `QuerySystem` owns its graph via `with_owner(id)`, so hand-tagged ids
    /// still collide with each other; the minted path is what closes the
    /// hole, and this test asserts the minted ids differ across many graphs
    /// (not just two: a counter that restarted, or a once-only guard, would
    /// still pass the pair case).
    #[test]
    fn standalone_graphs_get_unique_owner_ids() {
        let mut ids = std::collections::HashSet::new();
        for _ in 0..8 {
            let g = DepGraph::<NoOpScheduler>::new(Arc::new(NoOpScheduler));
            assert!(
                ids.insert(g.owner_id_for_test()),
                "two standalone graphs minted the same owner id {}",
                g.owner_id_for_test()
            );
        }
        // And a minted id must not alias the `QuerySystem` id range either:
        // `NEXT_SYSTEM_ID` starts at 1 and climbs, so a system's graph built
        // through `with_owner` stays distinguishable from any standalone one.
        let standalone = DepGraph::<NoOpScheduler>::new(Arc::new(NoOpScheduler));
        assert!(
            standalone.owner_id_for_test() > 1 << 40,
            "a minted graph id must sit outside the QuerySystem id range, got {}",
            standalone.owner_id_for_test()
        );
    }

    /// `sort_reads_by_shard`: shard-sorted output, epochs carrying the
    /// IDENTICAL permutation, and all three internal paths (already
    /// sorted, small fan-out, large fan-out) agreeing with one reference
    /// stable sort.
    ///
    /// The pairing is load-bearing: the finish-side late re-check compares
    /// `epoch_of(reads[i])` against `read_epochs[i]` POSITIONALLY, so a
    /// permutation that touches `reads` without touching `read_epochs` the
    /// same way turns that re-check into a false-positive generator.
    #[test]
    fn sort_reads_by_shard_keeps_the_epoch_pairing() {
        // Six index/shard pattern families, some already sorted (early
        // return), some not (small insertion sort), some > 16 long
        // (counting sort), plus equal-shard families to pin stability.
        // Shard = idx % 64, so the low bits carry the shard.
        let patterns: Vec<Vec<u32>> = vec![
            vec![],
            vec![0],
            vec![5],
            vec![5, 70],                       // 5, 6 — already sorted
            vec![70, 5],                       // 6, 5 — swapped
            vec![7, 71, 135, 9, 3],            // all shard 7 — sorted, dup shards
            vec![9, 70, 3, 135, 5],            // 9, 6, 3, 7, 5 — small, unsorted
            vec![1, 2, 130, 3, 8, 4],          // 1, 2, 2, 3, 8, 4 — equal shards
            vec![7, 8, 1, 2, 130, 3],          // 7, 8, 1, 2, 2, 3 — equal shards
            (0..40).map(|i| i * 64 + i).collect(), // large, already sorted
            (0..40).rev().map(|i| i * 64 + i).collect(), // large, reversed
            (0..40).map(|i| ((i * 7) % 5) * 64 + i).collect(), // large, cycling shards
        ];
        for pat in patterns {
            let n = pat.len();
            let mut reads: SmallVec<[DepNodeIndex; 8]> =
                pat.iter().map(|&i| DepNodeIndex(i)).collect();
            // epochs[i] == i: after sorting, epochs[i] is the reader's
            // original index, so the pairing is directly checkable.
            let mut epochs: SmallVec<[u64; 8]> = (0..n as u64).collect();
            let orig: Vec<DepNodeIndex> = reads.to_vec();

            super::sort_reads_by_shard(&mut reads, Some(&mut epochs));

            // (a) shard-sorted, non-decreasing.
            for w in reads.iter().zip(reads.iter().skip(1)) {
                assert!(
                    super::shard_of(*w.0) <= super::shard_of(*w.1),
                    "reads not shard-sorted: {reads:?}"
                );
            }
            // (b) pairing preserved and stable: the sequence of
            //     (shard, original index) pairs must be exactly the
            //     reference stable sort's.
            let mut expected: Vec<(usize, usize)> =
                (0..n).map(|i| (super::shard_of(orig[i]), i)).collect();
            expected.sort_by_key(|&(s, _)| s);
            let mut got: Vec<(usize, usize)> = Vec::with_capacity(n);
            for (i, &e) in epochs.iter().enumerate() {
                // epochs[i] is the original index of reads[i]; the slot
                // positions must line up one-to-one.
                let src = e as usize;
                assert!(src < n, "epoch {e} out of range for n = {n}");
                got.push((super::shard_of(orig[src]), src));
                assert_eq!(
                    got.len(),
                    i + 1,
                    "epoch list length disagrees with the read list"
                );
            }
            assert_eq!(
                got, expected,
                "permutation differs from a stable sort for pattern {pat:?}"
            );
        }
    }

    /// `reconcile_forward_edges` replaces the list with the current read
    /// set verbatim (the read set is ALREADY deduplicated by
    /// `TaskDeps::read` — re-deduping here was O(k^2) of waste) and
    /// returns exactly the edges the read set dropped.
    #[test]
    fn reconcile_forward_edges_replaces_list_and_reports_stale() {
        // Previous edges: 5, 7, 9.  Current read set: 7, 5, 11 — note it
        // arrives in first-read order (7 BEFORE 5), so the replacement is
        // verbatim, not sorted and not reordered.
        let mut list: SmallVec<[DepNodeIndex; 4]> = [5u32, 7, 9]
            .into_iter()
            .map(DepNodeIndex)
            .collect();
        let reads = [7u32, 5, 11].map(DepNodeIndex);
        let removed = super::reconcile_forward_edges(&mut list, &reads);
        assert_eq!(
            list.to_vec(),
            reads.to_vec(),
            "the list must become the read set verbatim (first-read order)"
        );
        assert_eq!(
            removed.to_vec(),
            vec![DepNodeIndex(9)],
            "only the edge the read set no longer has is stale"
        );
        // Idempotent: reconciling the same read set again is a no-op.
        let removed2 = super::reconcile_forward_edges(&mut list, &reads);
        assert_eq!(removed2.to_vec(), vec![], "second reconciliation is a no-op");
    }

    /// F(dur-prop): the reverse-edge cascade terminates on a cycle among
    /// individually-green nodes and settles every member at the lowered
    /// level.  Termination comes from monotone-downward levels (each
    /// node drops at most `LEVELS - 1` times ever) with the
    /// `lower_durability_to` success rule as the revisit filter; the
    /// explicit worklist keeps a deep reader graph off the real stack.
    #[test]
    fn dur_prop_lowering_cascade_terminates_on_cycle() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        // Build the cycle a -> b -> c -> a (forward edges), all declared
        // HIGH and finished green.
        let mut idxs = [DepNodeIndex::ZERO; 3];
        for i in 0..3 {
            idxs[i] = dg.allocate_node_index(ContentClass::SEMANTIC);
            dg.set_node_durability(idxs[i], Durability::HIGH);
        }
        for i in 0..3 {
            let n = idxs[i];
            let next = idxs[(i + 1) % 3];
            dg.start_task(n);
            dg.read(next);
            assert!(!dg.finish_task_and_publish(n, Some(1)));
        }
        // Warm-up mark at HIGH level: raise the generation so the first
        // verify actually walks and writes per-level stamps (at
        // generation 0 the global check is trivially true).
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm, &q());
        assert_eq!(dg.verify_green_transitive_with_dirty(idxs[0]), None);
        assert!(dg.verified_lvl_of(idxs[0]) > 0);

        // Lower `a` with a full lowering event.  The cascade must walk
        // the cycle (a's readers = c, c's readers = b, b's readers = a)
        // and terminate; every member settles at LOW.
        assert!(dg.lower_and_propagate(idxs[0], Durability::LOW));
        for i in 0..3 {
            assert_eq!(
                dg.durability_of(idxs[i]),
                Durability::LOW,
                "every cycle member must be lowered by the cascade"
            );
        }
        // The graph still verifies (the lowering never marks anything —
        // a mark-free window stays green); run the walk twice to also
        // exercise the buffered-stamp flush on a re-verified cycle.
        assert_eq!(dg.verify_green_transitive_with_dirty(idxs[0]), None);
        assert_eq!(dg.verify_green_transitive_with_dirty(idxs[0]), None);
    }

    /// D26 (dur-prop): the post-flush late re-aggregation — the
    /// durability twin of d16 — must repair a reader whose read lowered
    /// DURING the reader's own finish.  In that window the read's own
    /// cascade cannot reach the reader (the reader's rev edge is not
    /// flushed yet) and aggregation #1 already ran against the
    /// pre-lowering level; only the post-flush re-check sees the
    /// lowered read.
    #[test]
    fn dur_prop_late_recheck_catches_mid_compute_lowering() {
        let sched = TestScheduler::new();
        let dg = DepGraph::new(Arc::new(sched.clone()));
        let m = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(m, Durability::HIGH);
        let x = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(x, Durability::HIGH);

        // Park X's finish after aggregation #1, before the flush.
        sched.arm(move |p| matches!(p, SchedulePoint::FinishWindow { node } if *node == x.0));
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                // The task stack is thread-local: start/read/finish must
                // run on ONE thread.
                dg.start_task(x);
                dg.read(m);
                dg.finish_task_and_publish(x, Some(1))
            });
            sched.wait_parked();
            // X's #1 aggregation saw M at HIGH.  Lower M now: the
            // cascade CANNOT reach X — X's rev edge into M is not
            // flushed until the parked finish resumes.
            assert!(dg.lower_and_propagate(m, Durability::LOW));
            assert_eq!(
                dg.durability_of(x),
                Durability::HIGH,
                "the cascade must miss the un-flushed reader"
            );
            sched.release_all();
            let changed = h.join().expect("finish thread must not panic");
            assert!(!changed, "M's level drop never changed X's fingerprint");
        });
        // The post-flush re-aggregation must have re-taken the reads'
        // CURRENT levels and repaired X.
        assert_eq!(dg.durability_of(m), Durability::LOW);
        assert_eq!(
            dg.durability_of(x),
            Durability::LOW,
            "the late re-aggregation must catch the mid-compute lowering"
        );
    }

    /// D27 (dur-prop): the lowering seqlock seals GREEN verdicts.  A
    /// verification parked between its verdict and the re-sample must
    /// RETRY (not serve) when a lowering event brackets the park — even
    /// though the parked verdict itself computed green.  Observed via
    /// the verify hook firing a second time and via the retry's
    /// re-stamp being taken at the LOWERED level's counter.
    #[test]
    fn dur_prop_seqlock_seals_green_verdict_against_inflight_lowering() {
        let sched = TestScheduler::new();
        let dg = DepGraph::new(Arc::new(sched.clone()));
        let x = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(x, Durability::HIGH);
        dg.start_task(x);
        dg.finish_task();
        assert!(!dg.finish_task_and_publish(x, Some(1)));
        // Warm-up mark at HIGH level: force a real walk (generation > 0)
        // and make rev[HIGH] non-zero so the first stamp is observable.
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm, &q());
        assert_eq!(dg.verify_green_transitive_with_dirty(x), None);
        let vl_before = dg.verified_lvl_of(x);
        assert!(vl_before > 0);
        assert_eq!(dg.durability_of(x), Durability::HIGH);

        // Park a verification after its verdict, then bracket a lowering
        // event (odd bump, level drop — the real writer's cascade would
        // run here — even bump) while the verdict is parked in hand.
        sched.arm(move |p| matches!(p, SchedulePoint::VerifyWindow { node } if *node == x.0));
        std::thread::scope(|s| {
            let h = s.spawn(|| dg.verify_green_transitive_with_dirty(x));
            sched.wait_parked();
            // Writer bracket.  (The real writer holds `lowering_lock`;
            // the mutex is irrelevant to this single-threaded simulation
            // — what the seal reads is the epoch parity.)
            dg.lowering_epoch.fetch_add(1, Ordering::AcqRel); // → odd
            assert!(dg.lower_durability_to(x, Durability::LOW));
            dg.lowering_epoch.fetch_add(1, Ordering::AcqRel); // → even
            // Hand off: release the first park and catch the retry on a
            // fresh gate — the re-sample must see the bracket and RETRY,
            // not serve.
            sched.release_then_arm(
                move |p| matches!(p, SchedulePoint::VerifyWindow { node } if *node == x.0),
            );
            sched.wait_parked();
            sched.release_all();
            assert_eq!(
                h.join().expect("verify thread must not panic"),
                None,
                "the retry's verdict is still green (a lowering never marks)"
            );
        });
        assert_eq!(dg.durability_of(x), Durability::LOW);
        // A subsequent LOW-level mark must keep forcing the walk (the
        // re-stamp after the retry is taken at the lowered level's
        // counter, which now moves with LOW marks).
        let low = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.invalidate(low, &q());
        assert_eq!(dg.verify_green_transitive_with_dirty(x), None);
        assert!(
            dg.verified_lvl_of(x) > vl_before,
            "the post-lowering re-stamp must ride the LOW counter, not the abandoned HIGH one"
        );
    }

    /// A popular node's publish-time fan-out: 64 readers,
    /// far past the 4-entry inline `SmallVec` the buffer used to be.
    ///
    /// The reader list is FILLED under the node's shard write lock (the
    /// marks are deferred out of it by perf item 1), so an inline SmallVec
    /// heap-allocated right there — under the very lock every reader's own
    /// `finish` must take.  The pooled `FanoutBuf` moves that allocation
    /// off the critical section, amortized across finishes on the thread.
    ///
    /// The correctness assertion is that every one of the 64 readers is
    /// actually marked red on the republish: the reader list is the
    /// rev-edge set of the published node, so this is also the regression
    /// guard for "the pooled buffer must be cleared between finishes" — a
    /// stale non-empty buffer would mark readers of an EARLIER node too.
    /// This is ALSO the regression guard for the consume side: the buffer's
    /// `fill` takes the reader slice as an argument precisely so there is
    /// no `cleared()` call available here, because an earlier consumer
    /// reached for it and the whole fan-out became a silent no-op.
    ///
    /// PUBLISH-WINDOW FAMILY (grep `publish_window` and `FinishWindow`):
    /// this test covers the fan-out mechanism at unit level; the END-TO-END
    /// version is `defense_tests::publish_window_suspect_value_never_served_green`
    /// (OS threads — the getter-side symptom) and
    /// `shuttle_tests::depgraph::finish_window_erased_fanout_mark_is_repaired`
    /// (exhaustive interleaving — the late-recheck repair for a reader with
    /// no rev edge yet).
    #[test]
    fn fanout_marks_every_reader_of_a_popular_node() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        // One popular node read by 64 distinct readers.
        let popular = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(popular, Durability::HIGH);
        // Publish it once so the republish below has a prior fingerprint
        // to differ from (`changed` is "a fingerprint existed and it
        // differed" — a never-published node returns false, no fan-out).
        dg.start_task(popular);
        assert!(!dg.finish_task_and_publish(popular, Some(1)));
        let readers: Vec<DepNodeIndex> = (0..64)
            .map(|_| dg.allocate_node_index(ContentClass::SEMANTIC))
            .collect();
        for &r in &readers {
            dg.start_task(r);
            dg.read(popular);
            assert!(!dg.finish_task_and_publish(r, Some(1)));
        }
        // Warm the generation so the verify walk runs rather than taking
        // the trivially-true global fast path at generation 0.
        let warm = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(warm, Durability::HIGH);
        dg.invalidate(warm, &q());
        assert_eq!(dg.verify_green_transitive_with_dirty(readers[0]), None);

        // Republish the popular node with a CHANGED fingerprint so the
        // publish-time fan-out runs.
        dg.start_task(popular);
        assert!(
            dg.finish_task_and_publish(popular, Some(2)),
            "the changed fingerprint must drive the fan-out"
        );

        for (i, &r) in readers.iter().enumerate() {
            assert!(
                !dg.is_green(r),
                "reader {i} must be red after the popular node's fan-out"
            );
        }
    }
}
