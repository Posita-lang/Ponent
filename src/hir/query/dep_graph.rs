//! # DepGraph — Dependency Tracking for Incremental Re-evaluation
//!
//! Analogous to `rustc_middle::dep_graph`.
//! Records which query executions read which other query executions,
//! forming a dependency graph.  When an input changes, the graph
//! enables incremental re-evaluation.
//!
//! ## Why not per-node revisions (the salsa-style redesign — CLOSED)
//!
//! A per-node `changed_at`/`verified_at` revision model (salsa 3.x /
//! rustc-incremental style) was evaluated as the top optimisation
//! candidate and REJECTED, deliberately and by design.  The two models
//! are mutually exclusive here:
//!
//! - In THIS graph, invalidation is LAZY: a mark fans out exactly ONE
//!   level (direct readers, at fingerprint-change publish time), and the
//!   transitive walk (`verify_inner`) is not merely a hit check — it is
//!   the engine that DRIVES recomputation up the dependency chain.  A
//!   revision-model fast path ("serve when no direct dep's `changed_at`
//!   exceeds my `verified_at`") would serve a node whose TRANSITIVE dep
//!   was bare-invalidated (marked without a value change yet) without
//!   ever recomputing that dep — permanently stale if no one else
//!   requests it.  The walk returning the dirty dep to the caller is
//!   what triggers the re-request that recomputes it.
//! - Salsa-family designs are sound because invalidation propagates
//!   EAGERLY through the transitive reverse-edge graph at input-write
//!   time.  Adopting that here means restoring the eager transitive BFS
//!   that was removed BY DESIGN (see `dirty_readers`), plus revision
//!   tables, plus re-deriving d16 (late fan-out epoch re-check), d24
//!   (`finishing`), and the `lowering_epoch` seqlock window under
//!   revision comparison.
//!
//! The measured walk is linear at ~30 Melem/s (`benches/dep_graph.rs`,
//! `deep_chain_walk`), a full hit decision is ~17 ns, and the collect-
//! all-dirty + batch re-request rework (#6) already collapses the storm
//! from O(d) walks to two.  The revision model would buy a constant-
//! factor walk removal at the cost of a semantic redesign — not taken.
//! Revisit ONLY together with the eager-vs-lazy invalidation decision.
//!
//! ## Thread safety
//!
//! - `edges`, `fingerprints` and `classes` are sharded by
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
    /// Inline open-addressing dedup table over `reads` (BEFORE the
    /// `spilled` fallback).  The linear `contains` scan this replaces was
    /// O(k²) per dedup query-batch — 4096 compares at k=64 — and wide
    /// queries in the 16–64 zone landed exactly in that worst segment;
    /// probing is O(1) and removes the performance cliff at the bucket
    /// fallback.  Only membership testing changes: `reads` keeps its
    /// first-read push order, so the `reads`/`read_epochs` positional
    /// pairing (`sort_reads_by_shard`) and `reconcile_forward_edges`'s
    /// "already deduplicated" contract are untouched.
    dedup: DedupTable,
    /// Owned epochs-table snapshot taken lazily on the FIRST fresh read
    ///.  Every later read in the same task
    /// reads its epoch word from this snapshot instead of paying a TLS
    /// `epochs_snapshot()` probe per distinct read.  Append-only + shared
    /// blocks make a stale snapshot conservative (a grow adds blocks we
    /// simply see as missing → epoch 0); `reset` cannot race a live task
    /// (quiescence).
    epochs: Option<Arc<Vec<Arc<EpochBlock>>>>,
}

/// Outcome of a [`DedupTable::insert`].  The enum is load-bearing: the
/// caller must distinguish "stored" from "already present" from "full",
/// because only the first may push the entry onto `reads`, and the third
/// must NOT be conflated with the second.  A `bool` conflating them made
/// the full-table bail-out lie "newly inserted" without storing, which
/// appended a duplicate to `reads` and broke the "already deduplicated"
/// invariant `reconcile_forward_edges` documents (see its contract note).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InsertOutcome {
    /// Entry was stored in the table.
    Inserted,
    /// Entry was already present.
    AlreadyPresent,
    /// Table was full (unreachable while the spill-before-full contract
    /// holds — debug builds assert); the caller must spill.
    Full,
}

/// One inline dedup table: 64 slots of linear-probed open addressing.
/// The empty marker is `u32::MAX`; node indices are sequential from 1
/// (`allocate_node_index`), so a real node can never collide with it.
#[derive(Debug)]
struct DedupTable {
    slots: [DepNodeIndex; LINEAR_DEDUP_MAX],
}

impl DedupTable {
    fn new() -> Self {
        DedupTable {
            slots: [DepNodeIndex(u32::MAX); LINEAR_DEDUP_MAX],
        }
    }

    /// Fibonacci-hash the sequential index into the table.  Consecutive
    /// allocations differ by 1; the golden-ratio multiplier spreads them
    /// over the whole table instead of clustering on adjacent probes.
    ///
    /// Why THIS constant (verified exactly by
    /// `research/fibonacci-occupancy/scripts/golden_ratio_occupancy.py`): 1/φ is the worst-case optimum —
    /// its continued fraction is [0;1,1,...], so it maximizes
    /// `min_{q<=n} ||q·alpha||` among all constants (badly approximable).
    /// A constant with a large partial quotient collapses the sequence onto
    /// few "strands": alpha_2 = 165902777/2^32 ≈ 1/26 drops the 64-point
    /// bucket occupancy from 55 to 37.  The Knuth constant's effective
    /// rotation 2654435769/2^32 sits 1.2e-10 from 1/φ, deep inside the
    /// proven stability radius 9.4e-6, so its bucket assignment at n=64 is
    /// identical to the exact golden one (Occ = 55, a theorem not luck).
    #[inline]
    fn probe_start(idx: DepNodeIndex) -> usize {
        // 0x9E37_79B9 = 2654435769 = floor(2^32 / φ), the 32-bit
        // golden-ratio constant.  (2654435761 is a DIFFERENT number —
        // 0x9E3779B1-adjacent lookalikes abound; floor(2^32/φ) is the
        // standard Knuth multiplier.  Both spread identically at this
        // table size — the lookalike also passes the pin below — but only
        // floor(2^32/φ) is canonical.)
        // (`>> 26` on a u32 already leaves a value in [0, 63], so the
        // trailing mask is technically redundant — kept as a defensive
        // idiom; it is NOT load-bearing.)
        (idx.0.wrapping_mul(0x9E37_79B9) >> 26) as usize & (LINEAR_DEDUP_MAX - 1)
    }

    /// Insert unless present; [`InsertOutcome::Inserted`] only when the
    /// entry was actually STORED.  Callers must guarantee the table is not
    /// yet full — `TaskDeps::read` spills to the hash set at
    /// `LINEAR_DEDUP_MAX` entries before this can return to a full table.
    ///
    /// The return value is EXACTLY "I stored this entry": a full-table
    /// bail-out returns [`InsertOutcome::Full`] (never `Inserted`), so a
    /// caller that mishandles it fails loudly instead of appending a
    /// duplicate to `reads`.  (The old `bool` bail-out returned `true`
    /// ("newly inserted") without storing — unsound for the `reads`
    /// dedup invariant; the path is unreachable today, but the fallback
    /// shape is now impossible to misuse.)
    #[inline]
    fn insert(&mut self, other: DepNodeIndex) -> InsertOutcome {
        debug_assert_ne!(other.0, u32::MAX, "index collides with the empty marker");
        let empty = DepNodeIndex(u32::MAX);
        let mut i = Self::probe_start(other);
        // Step bound: linear probing on a FULL table would wrap forever —
        // a silent infinite loop inside the hot miss path is the worst
        // failure mode available.  The only caller maintains "spill before
        // full" today, but that guarantee depends on the spill check never
        // moving; the bound converts a hypothetical hang into a loud
        // failure (debug_assert) with a release-side bail-out (`Full`,
        // which the caller routes to the spill path).
        let mut steps = 0usize;
        loop {
            let slot = &mut self.slots[i];
            if *slot == other {
                return InsertOutcome::AlreadyPresent;
            }
            if *slot == empty {
                *slot = other;
                return InsertOutcome::Inserted;
            }
            steps += 1;
            if steps > LINEAR_DEDUP_MAX {
                // Full table (or caller-contract break): see the step-bound
                // comment above.  Debug builds fail loudly; release reports
                // `Full` so the caller spills and STORES the entry — the
                // return value stays exactly "I stored this".
                debug_assert!(
                    false,
                    "dedup table full; TaskDeps must spill to the hash set \
                     before inserting (LINEAR_DEDUP_MAX)"
                );
                return InsertOutcome::Full;
            }
            i = (i + 1) & (LINEAR_DEDUP_MAX - 1);
        }
    }
}

/// Inline open-addressing dedup capacity; above this many reads the
/// dedup set spills to a hash table.
const LINEAR_DEDUP_MAX: usize = 64;

/// A finished task's read set, kept in the membership structure it already
/// had at record time (reuse-DedupTable follow-up): `TaskDeps::read` built
/// this index while deduplicating — re-deriving the same containment
/// relation with a linear scan on the finish side was pure waste, and
/// `reconcile_forward_edges`' stale-edge scan pays it for EVERY `old` edge.
///
/// - `Table` — k < `LINEAR_DEDUP_MAX`: the inline open-addressing table,
///   exactly the reads (every `DedupTable::insert` returned
///   `InsertOutcome::Inserted` — the table cannot fill without spilling,
///   and a `Full` outcome routes the caller to the spill path).  O(1)
///   probe per membership test.
/// - `Spilled` — k ≥ `LINEAR_DEDUP_MAX`: the hash set, which by
///   construction holds the FULL read set (spill seeds it from `reads`,
///   and post-spill inserts go through it).
/// - `Empty` — no reads at all (the `finish_task_and_publish` no-frame
///   fallback): every membership test is `false`, no table to build.
pub(crate) enum ReadsIndex {
    Table(DedupTable),
    Spilled(HashSet<DepNodeIndex>),
    Empty,
}

impl ReadsIndex {
    /// Build an index from an existing read slice (tests / callers that
    /// hold only the slice): `Spilled` at capacity, `Table` below it.
    #[cfg(test)]
    fn build(reads: &[DepNodeIndex]) -> Self {
        if reads.is_empty() {
            return ReadsIndex::Empty;
        }
        if reads.len() >= LINEAR_DEDUP_MAX {
            return ReadsIndex::Spilled(reads.iter().copied().collect());
        }
        let mut table = DedupTable::new();
        for &r in reads {
            let _ = table.insert(r);
        }
        ReadsIndex::Table(table)
    }

    /// O(1) membership test against the finished read set.
    #[inline]
    fn contains(&self, n: DepNodeIndex) -> bool {
        match self {
            ReadsIndex::Table(t) => {
                let empty = DepNodeIndex(u32::MAX);
                let mut i = DedupTable::probe_start(n);
                let mut steps = 0usize;
                loop {
                    let slot = t.slots[i];
                    if slot == n {
                        return true;
                    }
                    if slot == empty {
                        return false;
                    }
                    steps += 1;
                    if steps > LINEAR_DEDUP_MAX {
                        // Unreachable for an index built from a set smaller
                        // than the table (a full table implies `spilled` is
                        // already `Some`, so `Table` is never probed full).
                        // The sound direction is `false`: `n` was not found
                        // in a probe that walked the whole table, and
                        // `reconcile_forward_edges` treats a membership miss
                        // as "the old edge is stale, remove it" — reporting
                        // a false PRESENT here would keep a dead edge, and
                        // reporting on a corrupt table at all is the loud-
                        // assert's job, not a widening heuristic's.
                        debug_assert!(false, "ReadsIndex::contains: table full (corrupt)");
                        return false;
                    }
                    i = (i + 1) & (LINEAR_DEDUP_MAX - 1);
                }
            }
            ReadsIndex::Spilled(s) => s.contains(&n),
            ReadsIndex::Empty => false,
        }
    }
}

impl TaskDeps {
    pub fn new(current_node: DepNodeIndex, system_id: usize) -> Self {
        TaskDeps {
            current_node,
            system_id,
            reads: SmallVec::new(),
            read_epochs: SmallVec::new(),
            spilled: None,
            dedup: DedupTable::new(),
            epochs: None,
        }
    }

    /// Epoch of `node` read from this task's own snapshot, if one has been
    /// taken.  `None` means the caller should take it lazily.
    #[inline]
    fn cached_epoch(&self, node: DepNodeIndex) -> Option<u64> {
        self.epochs
            .as_ref()
            .map(|snap| epoch_from_owned(snap.as_slice(), node))
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
        match self.dedup.insert(other) {
            InsertOutcome::AlreadyPresent => false,
            InsertOutcome::Inserted => {
                self.reads.push(other);
                if self.reads.len() >= LINEAR_DEDUP_MAX {
                    self.spilled = Some(self.reads.iter().copied().collect());
                }
                true
            }
            InsertOutcome::Full => {
                // Unreachable while "spill before full" holds (debug builds
                // already asserted inside `insert`), but handled SOUNDLY:
                // spill NOW, into a set seeded from the reads taken so far,
                // and store `other` — the outcome contract is "Full means
                // NOT stored", so the caller must not report a kept entry
                // that no structure holds (the old `bool` bail-out did
                // exactly that, appending a duplicate to `reads`).
                let mut set: HashSet<DepNodeIndex> = self.reads.iter().copied().collect();
                if set.insert(other) {
                    self.reads.push(other);
                    self.spilled = Some(set);
                    true
                } else {
                    false
                }
            }
        }
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
    /// epochs (parallel vectors, same order) and the read set's membership
    /// index (`ReadsIndex`): the dedup structure `TaskDeps::read` already
    /// maintained, handed to the finish side so `reconcile_forward_edges`
    /// probes it instead of re-deriving containment with an O(k) scan per
    /// stale-edge candidate.  Deliberately unordered: every consumer feeds
    /// an unordered `HashSet` (`edges` / `rev_edges`), so sorting here was
    /// O(k log k) of pure waste on every query execution.  Sort at a
    /// serialization boundary if deterministic order is ever needed.
    pub fn into_reads_with_epochs(
        self,
    ) -> (
        SmallVec<[DepNodeIndex; 8]>,
        SmallVec<[u64; 8]>,
        ReadsIndex,
    ) {
        // The two vectors are POSITIONAL by construction (every `read`
        // pushes both in lockstep) and `sort_reads_by_shard` swaps them in
        // lockstep — a length drift would mispair the finish-side epoch
        // re-check, so pin it in debug builds.
        debug_assert_eq!(
            self.reads.len(),
            self.read_epochs.len(),
            "TaskDeps: reads and read_epochs must stay positionally paired"
        );
        let index = match self.spilled {
            Some(spilled) => ReadsIndex::Spilled(spilled),
            None if self.reads.is_empty() => ReadsIndex::Empty,
            // Below the spill threshold the inline table holds EXACTLY the
            // reads: every `read` took the `dedup.insert` path, `Inserted`
            // pushed the entry (and full implies spilled), `AlreadyPresent`
            // meant a member, and `Full` (unreachable) spilled-and-stored —
            // so the table can never hold a false member nor miss one.
            None => ReadsIndex::Table(self.dedup),
        };
        (self.reads, self.read_epochs, index)
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

/// Generation-stamped visited bitmap for `verify_inner` (#3 of the perf
/// pass): replaces the pooled `HashSet<DepNodeIndex>` — one hash insert +
/// one hash probe per visit (~10–30 ns) — with two bit ops (~1–2 ns) on a
/// per-thread `Vec<u64>` pair.
///
/// The `stamps` word per bit-word records the walk generation a word was
/// last (re)written at; on a new walk `current` bumps and any word whose
/// stamp is stale is cleared lazily on first touch — zero clearing cost per
/// walk, O(words/64) growth amortized across walks.  Revisit = "green by
/// the first visit" carries over trivially: the visited test is one
/// `bits[w] & (1 << b)` load.
struct VisitedBitmap {
    bits: Vec<u64>,
    stamps: Vec<u64>,
    current: u64,
}

impl VisitedBitmap {
    fn new() -> Self {
        Self {
            bits: Vec::new(),
            stamps: Vec::new(),
            current: 0,
        }
    }
}

impl Default for VisitedBitmap {
    fn default() -> Self {
        Self::new()
    }
}

impl VisitedBitmap {
    /// Begin a new walk: bump the generation; stale words clear lazily.
    ///
    /// Generation WRAP (theoretical): after exactly 2⁶⁴ `clear()`s,
    /// `current` collides with a stale word's stamp and that word's
    /// visited bits survive into the new walk — a false "already visited",
    /// which for a verifier only bounds the walk's precision, never its
    /// soundness.  Unreachable in practice (a 10 ns walk per ns for 584
    /// years); documented rather than defended.
    #[inline]
    fn clear(&mut self) {
        self.current = self.current.wrapping_add(1);
    }

    /// Mark `n` visited; returns `false` if it was already visited on this
    /// walk (same contract as `HashSet::insert`'s return).
    #[inline]
    fn insert(&mut self, n: DepNodeIndex) -> bool {
        let i = n.0 as usize;
        let w = i >> 6;
        let b = i & 63;
        if w >= self.bits.len() {
            self.bits.resize(w + 1, 0);
            // New words are stamped with the PREVIOUS generation so their
            // first touch this walk clears them.
            self.stamps
                .resize(w + 1, self.current.wrapping_sub(1));
        }
        if self.stamps[w] != self.current {
            self.bits[w] = 0;
            self.stamps[w] = self.current;
        }
        let mask = 1u64 << b;
        if self.bits[w] & mask != 0 {
            return false;
        }
        self.bits[w] |= mask;
        true
    }
}

/// Thread-local reuse for `verify_inner`'s walk state (#11): the
/// unstamped traversal allocated a fresh visited-set + `Vec` per call —
/// pure allocator churn after an invalidation storm, when every request
/// walks until stamps are rebuilt.  Buffers are taken out of TLS on
/// entry and restored cleared on `Drop`, so every exit path returns
/// them.  TLS is per-thread and `verify_inner` never re-enters itself
/// on the same thread, so a failed borrow (TLS destruction) falls back
/// to fresh locals instead of deadlocking.
thread_local! {
    static VERIFY_SCRATCH: RefCell<(
        VisitedBitmap,
        [Vec<DepNodeIndex>; 2],
        Vec<DepNodeIndex>,
        [SmallVec<[DepNodeIndex; 8]>; DEP_SHARDS],
    )> = RefCell::new((
        VisitedBitmap::new(),
        [Vec::new(), Vec::new()],
        Vec::new(),
        std::array::from_fn(|_| SmallVec::new()),
    ));
}

/// Pooled `verify_inner` walk state: `take` it, use `.visited`,
/// `.level[..]` and `.to_stamp` directly, and `Drop` returns the cleared
/// buffers to TLS.  The DFS `(node, exiting)` stack is dead — the walk is
/// now a two-phase level-by-level BFS, so the pooled buffers are the
/// ping-pong level pair and the to-stamp list.
struct VerifyScratch {
    visited: VisitedBitmap,
    level: [Vec<DepNodeIndex>; 2],
    to_stamp: Vec<DepNodeIndex>,
    /// Per-level `shard → nodes` expansion buckets: 64 `SmallVec`s the old
    /// walk rebuilt on every call (≈2–3 KB of stack init on the storm path).
    shards_to_expand: [SmallVec<[DepNodeIndex; 8]>; DEP_SHARDS],
    pooled: bool,
}

impl VerifyScratch {
    fn take() -> Self {
        let pooled = VERIFY_SCRATCH
            .try_with(|c| {
                c.try_borrow_mut().ok().map(|mut b| {
                    (
                        std::mem::take(&mut b.0),
                        std::mem::take(&mut b.1),
                        std::mem::take(&mut b.2),
                        std::mem::replace(
                            &mut b.3,
                            std::array::from_fn(|_| SmallVec::new()),
                        ),
                    )
                })
            })
            .unwrap_or(None);
        match pooled {
            Some((visited, level, to_stamp, shards_to_expand)) => Self {
                visited,
                level,
                to_stamp,
                shards_to_expand,
                pooled: true,
            },
            None => Self {
                visited: VisitedBitmap::new(),
                level: [Vec::new(), Vec::new()],
                to_stamp: Vec::new(),
                shards_to_expand: std::array::from_fn(|_| SmallVec::new()),
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
        // 2 bits per node (bits + level/stamp words) — compare the byte
        // footprint against the same cap the HashSet version used.
        let level_cap = self
            .level
            .iter()
            .map(|v| v.capacity())
            .max()
            .unwrap_or(0);
        if self.visited.bits.len() * 16 > VERIFY_SCRATCH_CAP
            || level_cap > VERIFY_SCRATCH_CAP
            || self.to_stamp.capacity() > VERIFY_SCRATCH_CAP
            || self
                .shards_to_expand
                .iter()
                .any(|v| v.capacity() > VERIFY_SCRATCH_CAP / DEP_SHARDS)
        {
            return;
        }
        self.visited.clear();
        for v in self.level.iter_mut() {
            v.clear();
        }
        self.to_stamp.clear();
        for v in self.shards_to_expand.iter_mut() {
            v.clear();
        }
        let mut visited = std::mem::take(&mut self.visited);
        let mut level = std::mem::take(&mut self.level);
        let mut to_stamp = std::mem::take(&mut self.to_stamp);
        let mut shards = std::mem::replace(
            &mut self.shards_to_expand,
            std::array::from_fn(|_| SmallVec::new()),
        );
        // `take` left empty defaults behind, so TLS holds valid buffers;
        // swap the grown ones back in and drop the defaults.  Failure
        // (TLS destruction) just drops the buffers.
        let _ = VERIFY_SCRATCH.try_with(|c| {
            if let Ok(mut b) = c.try_borrow_mut() {
                std::mem::swap(&mut b.0, &mut visited);
                std::mem::swap(&mut b.1, &mut level);
                std::mem::swap(&mut b.2, &mut to_stamp);
                std::mem::swap(&mut b.3, &mut shards);
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
            Some(stamps) => Self {
                stamps,
                pooled: true,
            },
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
    fingerprints: Vec<Option<u64>>,
    /// Declared `ContentClass` of each node (single bit, fixed at
    /// allocation from the owning query's descriptor).  Class is a
    /// property of the NODE, not of an edge: any reader's class is one
    /// lookup away via this vector, so edges stay plain `DepNodeIndex`
    /// pairs and the hot read path stays lock-free.
    classes: Vec<ContentClass>,
    /// Per-node caller-provided input digest, recorded by the digest-gated
    /// invalidation API ([`DepGraph::digest_gate`]).  `None` = no digest
    /// recorded yet (the gate then invalidates conservatively).  Plain
    /// storage under the shard lock: the gate runs under
    /// `mutate_world` quiescence and nothing else reads it, so no
    /// lock-free discipline is needed.
    digests: Vec<Option<u64>>,
}

#[inline(always)]
fn shard_of(node: DepNodeIndex) -> usize {
    (node.0 as usize) & (DEP_SHARDS - 1)
}

#[inline(always)]
fn slot_of(node: DepNodeIndex) -> usize {
    (node.0 as usize) / DEP_SHARDS
}

/// Reverse-edge list for ONE node: inline `SmallVec` up to
/// [`REV_INLINE_LIMIT`] entries, then a hash set.
///
/// `rev_edges[n]` is the list of ALL readers of node `n`, and every finish
/// of a reader inserts into the same list.  The old shape — a `SmallVec`
/// with a linear `contains` probe — is O(k) PER insert, i.e. O(N²) TOTAL
/// for a node with N readers (N=1000 → ~500K compares per republish
/// cycle).  Popular nodes are exactly the ones memoization exists for, so
/// they are the ones that fan out; the hybrid keeps the inline path
/// allocation-free for the typical k ≤ 8 case and makes the popular tail
/// O(1) amortized per insert.
///
/// Iteration order of a SPILLED list is non-deterministic.  That is
/// deliberate and safe: every consumer feeds idempotent `mark_dirty`
/// marks or level-lowering (`lower_durability_to` is monotone), so the
/// order only affects which marks coalesce, never the outcome.
const REV_INLINE_LIMIT: usize = 16;

#[derive(Debug, Clone)]
enum RevEdges {
    Inline(SmallVec<[DepNodeIndex; 4]>),
    Spilled(HashSet<DepNodeIndex>),
}

impl RevEdges {
    #[inline]
    fn new() -> Self {
        RevEdges::Inline(SmallVec::new())
    }

    /// Idempotent insert — O(1) amortized once spilled, linear (tiny k)
    /// while inline.
    fn insert(&mut self, v: DepNodeIndex) {
        match self {
            RevEdges::Inline(list) => {
                if list.contains(&v) {
                    return;
                }
                if list.len() >= REV_INLINE_LIMIT {
                    let mut set: HashSet<DepNodeIndex> =
                        list.iter().copied().collect();
                    set.insert(v);
                    *self = RevEdges::Spilled(set);
                    return;
                }
                list.push(v);
            }
            RevEdges::Spilled(set) => {
                set.insert(v);
            }
        }
    }

    fn retain(&mut self, mut f: impl FnMut(DepNodeIndex) -> bool) {
        match self {
            RevEdges::Inline(list) => list.retain(|n| f(*n)),
            RevEdges::Spilled(set) => set.retain(|n| f(*n)),
        }
    }

    fn iter(&self) -> RevEdgesIter<'_> {
        match self {
            RevEdges::Inline(list) => RevEdgesIter(RevEdgesIterInner::Inline(list.iter())),
            RevEdges::Spilled(set) => RevEdgesIter(RevEdgesIterInner::Spilled(set.iter())),
        }
    }
}

/// Iterator over a [`RevEdges`] list — yields `&DepNodeIndex` either way,
/// so existing `.iter().copied()` consumers keep working.
struct RevEdgesIter<'a>(RevEdgesIterInner<'a>);

enum RevEdgesIterInner<'a> {
    Inline(std::slice::Iter<'a, DepNodeIndex>),
    Spilled(std::collections::hash_set::Iter<'a, DepNodeIndex>),
}

impl<'a> Iterator for RevEdgesIter<'a> {
    type Item = &'a DepNodeIndex;
    #[inline]
    fn next(&mut self) -> Option<&'a DepNodeIndex> {
        match &mut self.0 {
            RevEdgesIterInner::Inline(it) => it.next(),
            RevEdgesIterInner::Spilled(it) => it.next(),
        }
    }
}

// ── Reverse-edge log (lock-free rework, #10) ─────────────────────────

/// One entry in the per-shard reverse-edge log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeOp {
    /// `(node, reader)` — reader now reads node.
    Insert(DepNodeIndex, DepNodeIndex),
    /// `(node, reader)` — reader no longer reads node.
    Tombstone(DepNodeIndex, DepNodeIndex),
}

impl EdgeOp {
    #[inline]
    fn node(self) -> DepNodeIndex {
        match self {
            EdgeOp::Insert(n, _) | EdgeOp::Tombstone(n, _) => n,
        }
    }
    #[inline]
    fn reader(self) -> DepNodeIndex {
        match self {
            EdgeOp::Insert(_, r) | EdgeOp::Tombstone(_, r) => r,
        }
    }
    #[inline]
    fn is_tombstone(self) -> bool {
        matches!(self, EdgeOp::Tombstone(..))
    }
}

/// Log-length threshold at which a quiesced `mutate_world` compacts a
/// shard. Bounds the append-only log between compactions; also bounds the
/// tail a reader scans. Lowering it trades more compaction work for
/// smaller scans.
const SHARD_LOG_COMPACT_THRESHOLD: usize = 4096;

/// Per-shard reverse-edge log + compacted index (the lock-free rework of
/// the old `Vec<RevEdges>` field — the full safety argument lives on the
/// `DepShard::rev_log` field doc).
///
/// ## Safety direction (over-marks, never under-marks)
///
/// - **Tombstones over-mark.** Until compaction drains them, a deleted
///   edge remains visible to readers. Over-marking is sound (`mark_dirty`
///   is an idempotent bit set; `lower_durability_to` is
///   monotone-downward).
/// - **Inserts may under-mark.** A reader that loads the index snapshot
///   and a stale log tail can miss a recent `Insert` — but only a push
///   that has not landed yet, and that reader's `finishing` flag is still
///   set, so its own d16 late-fan-out re-check repairs it. This is why
///   compaction is quiesced: a quiesced compaction cannot interleave with
///   a reader's scan.
///
/// ## Per-slot tail index
///
/// The log tail is ALSO indexed by `slot_of(op.node())`: `by_slot[s]`
/// lists the log positions of every op touching slot `s`, in log order.
/// `collect_readers` folds only the target node's ops instead of scanning
/// the whole tail — a batch compile with no invalidations (which never
/// compacts) previously made the i-th publish scan ~`i·reads/64` ops, an
/// Θ(E²) blowup on exactly the workload the query system exists to
/// accelerate. `by_slot` is maintained under the SAME mutex as `ops`
/// (one struct, one critical section), so per-node op order — the
/// tombstone-over-insert property — is preserved by construction, and the
/// over-mark/never-under-mark argument above is unchanged: a fresh node
/// simply has no entry in `by_slot`, and every op it does have is folded
/// in the same order the flat scan would.
///
/// `#[doc(hidden)]`: implementation detail of `DepShard`.
pub(crate) struct ShardLog {
    /// Append-only operation log PLUS its per-slot index, guarded by one
    /// mutex. Short critical section: one `Vec::push` + one `SmallVec`
    /// push per edge. `crate::sync::Mutex` (shuttle-aware), like every
    /// other lock in the graph.
    log: Mutex<ShardLogOps>,
    /// Compacted index snapshot. Readers `load()` this lock-free (a debt
    /// guard, no refcount RMW).  Indices are slots (`slot_of(node)`); each
    /// slot's edge list is its OWN `Arc`, so compaction reuses untouched
    /// slots by pointer copy and deep-clones only slots that received ops
    ///.  `ArcSwap<Vec<_>>` wraps the snapshot in its own
    /// `Arc` — no explicit second layer.
    index: ArcSwap<Vec<Arc<RevEdges>>>,
    /// Mirror of `log.ops.len()` readable without the log mutex, so the
    /// common post-compaction `collect_readers` skips the lock entirely
    ///.  Written under the log mutex; a stale read can only
    /// miss a just-pushed edge, which the d16 epoch re-check repairs.
    ops_len: AtomicUsize,
    /// Set when a push crosses the compaction threshold; lets
    /// `compact_rev_edges` skip scanning all 64 shard logs on every
    /// mutation when none is due.
    compact_needed: AtomicBool,
}

/// The mutex-guarded half of [`ShardLog`]: ops plus their per-slot index.
#[derive(Default)]
struct ShardLogOps {
    ops: Vec<EdgeOp>,
    /// `by_slot[s]` = positions (indices into `ops`) of every op whose
    /// node lives in slot `s`, in log order. Cleared at compaction/reset
    /// together with `ops`.
    by_slot: Vec<SmallVec<[u32; 8]>>,
}

impl ShardLogOps {
    #[inline]
    fn push(&mut self, op: EdgeOp) {
        let slot = slot_of(op.node());
        self.ops.push(op);
        if slot >= self.by_slot.len() {
            self.by_slot.resize_with(slot + 1, SmallVec::new);
        }
        self.by_slot[slot].push((self.ops.len() - 1) as u32);
    }
}

/// Pick between `collect_readers`' linear in-place fold and the set-based
/// fold.  Cost model, in arbitrary time units: the linear
/// fold is ≈`3·P·R` (a scan/compare per op per reader), the set fold is
/// ≈`100·R + 50·P` (one `HashSet` build plus one probe per op).
///
/// The previous fixed gate (`P > 8 && R > 16`) left the linear path in
/// charge for shapes like `P = 40, R = 1000`, where it is the worst-case
/// ABSOLUTE cost.  This predicate is the algebraic crossover; it is cheap
/// integer arithmetic on two length fields.
#[inline]
fn prefer_set_path(ops: usize, readers: usize) -> bool {
    let (p, r) = (ops as u64, readers as u64);
    3 * p * r > 100 * r + 50 * p
}

thread_local! {
    /// Per-thread pool of reader fan-out buffers.  The republish fan-out is
    /// hot, and a node with more than the inline `SmallVec` capacity used to
    /// malloc on every publish; the switchboard below reuses one buffer per
    /// nested call.  Bounded so an exceptional fan-out cannot pin memory.
    static READERS_POOL: std::cell::RefCell<Vec<SmallVec<[DepNodeIndex; 8]>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Buffers kept per thread.
const READERS_POOL_SLOTS: usize = 8;
/// A pooled buffer that grew past this is dropped rather than retained.
const READERS_POOL_MAX_CAP: usize = 64;

impl ShardLog {
    fn new() -> Self {
        Self {
            log: Mutex::new(ShardLogOps::default()),
            index: ArcSwap::from_pointee(Vec::new()),
            ops_len: AtomicUsize::new(0),
            compact_needed: AtomicBool::new(false),
        }
    }

    /// Append one edge operation. Short critical section: one `Vec::push`
    /// plus one `SmallVec::push` into the per-slot index. This is the
    /// whole write path — no shard write lock.
    #[inline]
    fn push(&self, op: EdgeOp) {
        let _tw_rev_log =
            sync_order::Acquired::acquire(sync_order::Fam::RevLog, "rev_log.push");
        let mut log = self.log.lock();
        log.push(op);
        let len = log.ops.len();
        self.ops_len.store(len, Ordering::Release);
        if len >= SHARD_LOG_COMPACT_THRESHOLD {
            self.compact_needed.store(true, Ordering::Release);
        }
    }

    /// Collect the current readers of `node` from `(compacted index) +
    /// (log tail)`, applying inserts and tombstones in log order. Returns
    /// a deduplicated list.
    ///
    /// The tail fold visits ONLY this node's ops (via `by_slot`), so its
    /// cost is O(this node's recent ops), not O(log tail) — the P0 rework.
    /// The mutex is released before the caller processes the readers, so
    /// callers can take other locks (class shards, epoch words) without
    /// deadlock — no log → shard lock ordering exists anywhere.
    fn collect_readers(&self, node: DepNodeIndex) -> SmallVec<[DepNodeIndex; 8]> {
        let mut readers = SmallVec::new();
        self.collect_readers_into(node, &mut readers);
        readers
    }

    /// [`collect_readers`](Self::collect_readers) into a caller-supplied
    /// buffer.  The pooled [`with_readers`](Self::with_readers) path uses it
    /// so a node with more than 8 direct readers no longer mallocs on every
    /// republish — the exact hot path memoization exists for.
    fn collect_readers_into(
        &self,
        node: DepNodeIndex,
        readers: &mut SmallVec<[DepNodeIndex; 8]>,
    ) {
        readers.clear();
        // `load()` takes a debt guard instead of `load_full()`'s refcount
        // pair on a shared line that every reader of every shard contends
        // on.
        let index = self.index.load();
        // Compacted readers for this slot.
        if let Some(list) = index.get(slot_of(node)) {
            readers.extend(list.iter().copied());
        }
        // Empty-tail fast path: after a compaction the log
        // is empty and the index is the whole answer.  A concurrent push
        // racing this load can be missed; that reader is still finishing,
        // so the d16 epoch re-check repairs it (the same tolerance the
        // pre-lock collection relied on).
        //
        // #6: this tolerance is a precondition, not a bug — callers that need
        // a *complete* reader set must be finishing-task (d16 re-check runs)
        // or quiesced.  A future completeness-expecting caller needs a
        // lock-taking variant, not this fast path.
        if self.ops_len.load(Ordering::Acquire) == 0 {
            return;
        }
        // Apply the log tail on top, IN ORDER: a Tombstone after an Insert
        // must win, and vice versa.
        let _tw_rev_log =
            sync_order::Acquired::acquire(sync_order::Fam::RevLog, "rev_log.collect");
        let log = self.log.lock();
        if let Some(positions) = log.by_slot.get(slot_of(node)) {
            if prefer_set_path(positions.len(), readers.len()) {
                // Hot/churned node: the linear `retain`/`contains` fold is
                // O(readers × ops).  Fold through ONE set
                // in log order: the previous shape built
                // `seen`, `order`, `out` and `emitted` — five allocations on
                // the very path that exists because the node is hot.
                //
                // ORDER IS DELIBERATELY UNSPECIFIED: every consumer
                // (`mark_dirty_batch`, `lower_durability_to`,
                // `dirty_readers_by_class`) iterates unordered or re-sorts
                // by shard, so a direct set drain is equivalent.
                let mut seen: HashSet<DepNodeIndex> = readers.iter().copied().collect();
                for &i in positions.iter() {
                    let op = log.ops[i as usize];
                    debug_assert_eq!(op.node(), node);
                    if op.is_tombstone() {
                        // Bounded set: a tombstone for a reader that was
                        // never present is a no-op (same as the linear
                        // `retain` arm below).
                        seen.remove(&op.reader());
                    } else {
                        seen.insert(op.reader());
                    }
                }
                *readers = seen.into_iter().collect();
            } else {
                for &i in positions.iter() {
                    let op = log.ops[i as usize];
                    debug_assert_eq!(op.node(), node);
                    if op.is_tombstone() {
                        readers.retain(|x| *x != op.reader());
                    } else if !readers.contains(&op.reader()) {
                        readers.push(op.reader());
                    }
                }
            }
        }
        drop(log);
    }

    /// Take a pooled reader buffer, fill it with `node`'s live readers, run
    /// `f`, then check the buffer back in.  Per-thread and bounded; a buffer
    /// that grew past [`READERS_POOL_MAX_CAP`] is simply dropped.
    fn with_readers<R>(
        &self,
        node: DepNodeIndex,
        f: impl FnOnce(&mut SmallVec<[DepNodeIndex; 8]>) -> R,
    ) -> R {
        let mut readers = READERS_POOL
            .try_with(|c| c.try_borrow_mut().ok().and_then(|mut b| b.pop()))
            .unwrap_or(None)
            .unwrap_or_default();
        self.collect_readers_into(node, &mut readers);
        let out = f(&mut readers);
        if readers.capacity() <= READERS_POOL_MAX_CAP {
            let _ = READERS_POOL.try_with(|c| {
                if let Ok(mut pool) = c.try_borrow_mut()
                    && pool.len() < READERS_POOL_SLOTS
                {
                    pool.push(readers);
                }
            });
        }
        out
    }

    /// Compaction: rebuild the index from `index + log`, publish it, and
    /// clear the log (and its per-slot index). MUST run under quiescence
    /// (write lock) — clearing the log tail out from under any reader
    /// otherwise loses its inserts permanently (an under-mark that the
    /// finishing flag does NOT repair: the missed reader may have finished
    /// long ago).
    fn compact(&self) {
        let _tw_rev_log =
            sync_order::Acquired::acquire(sync_order::Fam::RevLog, "rev_log.compact");
        let guard = self.index.load();
        let mut log = self.log.lock();
        if log.ops.is_empty() {
            return;
        }
        // The mutex is held across the rebuild: pushes serialize on it, so
        // no op is lost between the take and the clear.  Only the Vec's
        // POINTERS are copied here; `Arc::make_mut` deep-clones a slot at
        // most once, and only if that slot received an op.
        let mut new_index: Vec<Arc<RevEdges>> = guard.iter().cloned().collect();
        for op in log.ops.iter().copied() {
            let slot = slot_of(op.node());
            match op {
                EdgeOp::Insert(_, r) => {
                    if slot >= new_index.len() {
                        new_index.resize_with(slot + 1, || Arc::new(RevEdges::new()));
                    }
                    Arc::make_mut(&mut new_index[slot]).insert(r);
                }
                EdgeOp::Tombstone(_, r) => {
                    if let Some(list) = new_index.get_mut(slot) {
                        Arc::make_mut(list).retain(|x| x != r);
                    }
                }
            }
        }
        self.index.store(Arc::new(new_index));
        log.ops.clear();
        log.by_slot.clear();
        self.ops_len.store(0, Ordering::Release);
        self.compact_needed.store(false, Ordering::Release);
    }

    /// Reset: clear both log (with its per-slot index) and index. Quiesced
    /// (called from `reset`).
    fn reset(&self) {
        let _tw_rev_log =
            sync_order::Acquired::acquire(sync_order::Fam::RevLog, "rev_log.reset");
        let mut log = self.log.lock();
        log.ops.clear();
        log.by_slot.clear();
        drop(log);
        self.ops_len.store(0, Ordering::Release);
        self.compact_needed.store(false, Ordering::Release);
        self.index.store(Arc::new(Vec::new()));
    }

    /// `true` when the log has grown past the compaction threshold.
    #[inline]
    fn needs_compaction(&self) -> bool {
        self.compact_needed.load(Ordering::Acquire)
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
    reads_index: &ReadsIndex,
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
    // Fast path 1 — SAME LIST, same order: a deterministic `compute`
    // re-records exactly the reads it recorded last time, so elementwise
    // equality is the overwhelmingly common shape.  `removed` is trivially
    // empty and the list is left untouched (skipping even the O(k)
    // clear-and-extend write).  One memcmp instead of the O(|list| × k)
    // containment scan below.
    if list.as_slice() == reads {
        return SmallVec::new();
    }
    // Stale edges: entries of the previous list the current read set no
    // longer contains.  The membership test probes the read set's OWN
    // index (reuse-DedupTable follow-up): `TaskDeps::read` already built
    // the inline table / spill set while deduplicating, so the old
    // `reads.contains(&old)` linear scan — O(|list| × k) — collapses to
    // O(|list|) probes.  `ReadsIndex::Table`/`Spilled` are exact sets of
    // `reads`, `Empty` (no reads) rejects everything, so the output is
    // identical to the linear scan.
    let mut removed: SmallVec<[DepNodeIndex; 4]> = SmallVec::new();
    for &old in list.iter() {
        if !reads_index.contains(old) {
            removed.push(old);
        }
    }
    // Reuse the old list's allocation instead of replacing it: when the
    // read set is larger than the four inline slots the SmallVec is on
    // the heap, and copying into its existing capacity allocates
    // nothing.  `reads` is a separate task-deps slice at every call
    // site, never `list` itself, so clearing first cannot lose input.
    list.clear();
    list.extend_from_slice(reads);
    removed
}

/// Grow a shard's five slot vectors so `slot` is a valid index in all of
/// them.  The stamp/finishing slots live in the append-only `stamps`
/// table, so they are not resized here.
#[inline]
fn ensure_shard_len(shard: &mut DepShard, slot: usize) {
    if slot >= shard.edges.len() {
        shard.edges.resize_with(slot + 1, SmallVec::new);
        // NOTE: no `rev_log` growth here — the log is a flat `Vec` of ops
        // and the compacted index grows during `ShardLog::compact`.
        shard.fingerprints.resize_with(slot + 1, || None);
        shard.classes.resize_with(slot + 1, ContentClass::default);
        shard.digests.resize_with(slot + 1, || None);
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

/// Blocks of the per-node stamp table allocated up front.  A stamp slot
/// carries two `AtomicU64`s plus the d24 `AtomicBool` (24 bytes), so one
/// 4096-slot block is ~96 KiB — comparable to an epoch block, and far
/// smaller than the 2 MiB the old in-shard vectors could reach.  Sized
/// separately from [`DIRTY_INIT_BLOCKS`] for the same reason as
/// [`EPOCH_INIT_BLOCKS`]: the tables have different per-node costs and a
/// shared constant over-allocates the expensive ones.  Growth is
/// append-only and doubles, so one block is enough head start.
const STAMP_INIT_BLOCKS: usize = 1;

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
/// TLS pool for `sort_reads_by_shard`'s >64-band permutation buffer
///: take-and-return, cleared on the way back in. Cap mirrors
/// the other finish-path pools loosely — a pathological wide query's
/// buffer is dropped rather than pooled past this size.
// 4096 entries (32 KiB per thread) covers any plausible wide query; a
// genuinely gigantic one re-allocates once instead of pinning 512 KiB of
// TLS per worker.
const SORT_ORDER_POOL_CAP: usize = 1 << 12;

thread_local! {
    static SORT_ORDER_POOL: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

fn sort_reads_by_shard(
    reads: &mut SmallVec<[DepNodeIndex; 8]>,
    mut epochs: Option<&mut SmallVec<[u64; 8]>>,
) {
    // Lockstep-swap contract: the epochs vector (when present) must cover
    // exactly the reads it permutes alongside.
    debug_assert!(
        epochs.as_ref().is_none_or(|e| e.len() == reads.len()),
        "sort_reads_by_shard: epochs length must match reads length"
    );
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
    if reads.len() <= 32 {
        // Small/medium fan-outs: in-place insertion sort, stable, with the
        // epochs swapped in lockstep so the parallel pairing follows the
        // same swaps.  Both paths are allocation-free in this band, and
        // insertion sort still beats counting sort's `starts[64]` init plus
        // second pass up to about k = 32 — which is where realistic wide
        // read sets cluster.
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
    // `order` is the one remaining allocation — and only for k > 64
    // (#7 of the perf pass): the counting path is reached for k > 16,
    // so the `SmallVec<[usize; 8]>` it used ALWAYS heap-allocated
    // (8 < 17) for the whole 17–64 band.  A `[usize; 64]` stack
    // buffer (512 B) covers that band allocation-free; `Vec` beyond.
    // `[0; 64]` instead of `from_fn(|i| i)`: the counting phase below
    // overwrites `order[0..reads.len()]` completely, so any initial value is
    // dead — both compile to the same memset, but zero is the honest one
    //.
    let mut stack_order: [usize; 64] = [0; 64];
    let mut pooled_order: Option<Vec<usize>> = None;
    let order: &mut [usize] = if reads.len() <= 64 {
        &mut stack_order[..reads.len()]
    } else {
        // Wide tail (k > 64): the permutation buffer is TLS-POOLED
        // instead of a fresh `Vec` collect per finish — a
        // wide-read query paid a malloc/free on every publish otherwise.
        // Take-and-return after the permutation below, cleared on the way
        // back in.
        let v = SORT_ORDER_POOL
            .try_with(|c| {
                c.try_borrow_mut().ok().map(|mut b| {
                    let mut v = std::mem::take(&mut *b);
                    v.clear();
                    v.resize(reads.len(), 0);
                    v
                })
            })
            .unwrap_or(None)
            .unwrap_or_else(|| vec![0usize; reads.len()]);
        pooled_order = Some(v);
        pooled_order.as_mut().unwrap()
    };
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
                    e[i] = tmp_e.expect("parity: epochs present iff eps was Some");
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
    // Hand the wide-band permutation buffer back to the TLS pool
    //: the borrow through `order` ends above, so the Vec can
    // be moved into the pool for the next wide finish on this thread.
    // Oversized buffers are dropped, not pooled.
    if let Some(mut buf) = pooled_order.take() {
        if buf.capacity() <= SORT_ORDER_POOL_CAP {
            let _ = SORT_ORDER_POOL.try_with(|c| {
                if let Ok(mut b) = c.try_borrow_mut() {
                    std::mem::swap(&mut *b, &mut buf);
                }
            });
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

/// Cached view of an append-only per-node table for one use, mirroring
/// `VtableResolve`: the `Cached` arm fences a `RefCell` borrow through
/// `Ref::filter_map` (no raw reference ever escapes the borrow tracker);
/// the `Owned` arm serves first uses, post-grow/reset staleness, and slot
/// contention without touching the slot at all.
///
/// One generic replaces the three byte-identical `DirtySnap`/`EpochSnap`/
/// `StampSnap` enums: the borrow-fencing argument now lives
/// in ONE place instead of three copies that could drift.
enum Snap<'a, T> {
    Cached(std::cell::Ref<'a, Vec<Arc<T>>>),
    Owned(Arc<Vec<Arc<T>>>),
}

impl<T> std::ops::Deref for Snap<'_, T> {
    type Target = Vec<Arc<T>>;
    fn deref(&self) -> &Self::Target {
        match self {
            Snap::Cached(r) => r,
            Snap::Owned(a) => a,
        }
    }
}

type DirtySnap<'a> = Snap<'a, DirtyBlock>;
type EpochSnap<'a> = Snap<'a, EpochBlock>;
type StampSnap<'a> = Snap<'a, StampBlock>;

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

/// Read one node's epoch word from an owned epochs snapshot (the
/// per-task path).  A missing block is post-grow/post-reset and reads 0 —
/// the same conservative arm as [`DepGraph::epoch_of`].
#[inline]
fn epoch_from_owned(snap: &[Arc<EpochBlock>], node: DepNodeIndex) -> u64 {
    snap.get(block_of(node.0))
        .map(|b| b[epoch_word_of(node.0)].load(Ordering::Acquire) & EPOCH_MASK)
        .unwrap_or(0)
}

/// One per-node stamp slot in the append-only `stamps` table.
///
/// `verified_gen` is stamped with `dirty_generation` after a transitive
/// green verification; a hit skips the closure walk while it is `>=` the
/// reader's current generation.  `verified_lvl` is the per-level
/// durability stamp (`rev[node.dur]` at stamp time); the durability fast
/// path serves while it is `>= rev[node.dur]`.  A later durability
/// lowering compares an old HIGH stamp against a faster lower counter,
/// which only fails conservatively.
///
/// The two counters are SEPARATE words, deliberately not packed.  The
/// generation check is a complete verdict on its own, and a stale level
/// stamp can only cause an extra walk.  A reader can see a new generation
/// with an old level (the generation is sufficient) or an old generation
/// with a new level (the level certificate is independently valid:
/// `rev` advances only with a generation bump, and a level stamp is
/// written only after a stable green walk).  Neither torn combination is
/// unsound, so pair-atomicity buys nothing and would cost counter width.
///
/// `finishing` is d24: set before a freshly computed value is staged into
/// the cache, cleared after the finish tail.  A verifier that reaches a
/// finishing node treats it as dirty; the node's colour is not final and
/// serving the staged value could hand out a pre-invalidation snapshot in
/// the residual window `[in-section colour decision .. post-flush
/// re-check]`.  The read order in `verify_inner` is load-bearing:
/// finishing FIRST, then `verified_gen`, then `verified_lvl`.
#[derive(Debug)]
struct StampSlot {
    finishing: AtomicBool,
    verified_gen: AtomicU64,
    verified_lvl: AtomicU64,
}

impl StampSlot {
    const fn new() -> Self {
        Self {
            finishing: AtomicBool::new(false),
            verified_gen: AtomicU64::new(0),
            verified_lvl: AtomicU64::new(0),
        }
    }
}

/// One block of the per-node stamp table: same 4096-node blocking as the
/// dirty bitmap and epoch table, so [`block_of`] indexes all three.
/// Blocks are `Arc`-shared and append-only; a stamp write racing a
/// regrowth lands on a block every later snapshot still points at.
type StampBlock = [StampSlot; 4096];

#[inline]
fn zero_stamp_block() -> Arc<StampBlock> {
    Arc::new([const { StampSlot::new() }; 4096])
}

/// Which [`StampSlot`] within a [`StampBlock`] owns `idx`.
#[inline]
fn stamp_word_of(idx: u32) -> usize {
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
/// Thread-safe: `edges` / `fingerprints` / `classes` are
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
    /// REVERSE-EDGE LOG (#10) — IMPLEMENTED. Parallel to `shards` by shard
    /// index, deliberately OUTSIDE the shard `RwLock`: the finish path
    /// (`flush_rev_edges` / `prune_rev_edges`) must touch NO shard lock —
    /// putting the log inside `DepShard` would force even a log push to
    /// take the shard's read lock and queue behind the publish critical
    /// section, defeating the rework. Each entry is a `ShardLog`: an
    /// append-only `Mutex<Vec<EdgeOp>>` log plus an
    /// `ArcSwap<Vec<Arc<RevEdges>>>` compacted index.
    ///
    /// ## Write path
    ///
    /// `flush_rev_edges` pushes `Insert(node, reader)`; `prune_rev_edges`
    /// pushes `Tombstone(node, reader)`. The critical section is one
    /// `Vec::push` under the log's short mutex — no shard lock anywhere on
    /// the finish path.
    ///
    /// ## Read path
    ///
    /// `dirty_readers{,_by_class}`, the fan-out fill in
    /// `finish_task_and_publish` (collected BEFORE the shard write lock —
    /// never nested), and `propagate_lowering` call
    /// `ShardLog::collect_readers(node)`, which loads the index snapshot
    /// lock-free, scans the log tail under the short mutex, and returns
    /// the deduplicated reader list. The mutex is released before any
    /// other lock is taken (no log → shard ordering anywhere).
    ///
    /// ## Safety direction (over-marks, never under-marks)
    ///
    /// - **Tombstones over-mark**: a deleted edge remains visible to
    ///   readers until compaction drains it. `mark_dirty` is idempotent;
    ///   `lower_durability_to` is monotone-downward — both tolerate stale
    ///   marks.
    /// - **Inserts may under-mark** a concurrent reader that missed a
    ///   recent push. The only possible miss is a reader whose `flush`
    ///   push has not landed yet — and that reader's `finishing` flag is
    ///   still set (the push precedes `clear_finishing`), so no verifier
    ///   can serve its value across the gap; its own d16 late-fan-out
    ///   epoch re-check recomputes it. This is the same tolerance the old
    ///   in-shard `rev_edges` design documented ("a reader committing its
    ///   edge between the phases is missed").
    ///
    /// ## Compaction
    ///
    /// `ShardLog::compact()` rebuilds the index from `index + log` and
    /// clears the log. It runs under quiescence (write lock):
    /// - **Trigger**: `mutate_world` calls `compact_rev_edges` when any
    ///   shard's log exceeds `SHARD_LOG_COMPACT_THRESHOLD` (4096 entries).
    /// - **`reset`**: `clear_all` clears the log and index to empty.
    /// - **No mid-scan races**: the quiesce write lock excludes every
    ///   reader (they hold either the `InFlightGuard` read pin or
    ///   `&Quiesced`), so compaction can safely clear the log — readers
    ///   never run during it.
    ///
    /// ## Known bounded lags
    ///
    /// - **Tombstone accumulation** is bounded by
    ///   `SHARD_LOG_COMPACT_THRESHOLD`. Between compactions, a node with
    ///   many churned readers pays the fan-out cost of the stale entries
    ///   (idempotent `mark_dirty` per stale reader).
    /// - **Log-tail growth between mutations**: a query-heavy phase with
    ///   no `mutate_world` (e.g. one full compile pass) lets the log grow
    ///   with every recorded read, and each `collect_readers` scans the
    ///   whole tail. The next `mutate_world` (invalidate / `clear_all`)
    ///   compacts it back under the threshold — for edit-driven sessions
    ///   that is every keystroke; a pure batch compile pays the tail scan
    ///   for that one pass.
    rev_logs: Vec<CachePadded<ShardLog>>,
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
    /// WITH `dirty` (same `block_of` indexing, same append-only
    /// `ArcSwap` discipline — see `dirty` for why that makes lock-free
    /// access racing a regrowth safe).
    ///
    /// The two tables are NOT kept at equal lengths, and the safety
    /// invariant is NOT a global length order.  It is per ALLOCATED node:
    /// every node that can be marked must have an epoch word.  Allocation
    /// enforces exactly that — `grow_dirty(idx)` returns early only when
    /// BOTH tables already cover `idx`, and the index does not escape the
    /// allocator until it has — while `reset` restores the deliberately
    /// asymmetric resting state (`DIRTY_INIT_BLOCKS` = 64 dirty blocks
    /// against `EPOCH_INIT_BLOCKS` = 4 epoch blocks).  A dirty table
    /// longer than the epoch table therefore covers only indices that were
    /// never allocated; no valid caller can mark one.  See `reset` for why
    /// the asymmetry is kept and `grow_dirty` for the growth order.
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
    /// Per-node verification stamps and d24 finishing flags, blocked and
    /// grown with `dirty`/`epochs` (same `block_of` indexing, same
    /// append-only `ArcSwap` discipline — see `dirty` for why that makes
    /// lock-free access racing a regrowth safe).
    ///
    /// This is the lock-free replacement for the old per-shard
    /// `verified_gen`/`verified_lvl`/`finishing` vectors: a cache hit
    /// reads one stable [`StampSlot`] instead of taking the node shard's
    /// read lock.  See [`StampSlot`] for the counter and d24 invariants.
    stamps: ArcSwap<Vec<Arc<StampBlock>>>,
    /// Fast-path cover bound for [`Self::tables_cover`]: the
    /// number of leading BLOCKS for which all three tables are known
    /// covered. Growth is monotone within a generation, so this is updated
    /// at the END of `grow_dirty` (after all three stores, before the
    /// `table_gen` bump) and re-derived in `reset`. A stale-low value is
    /// only a slow-path fallback (the authoritative grow-locked check);
    /// a too-high value is impossible — it is only ever raised from
    /// `min(dirty, epochs)` block counts observed after the stores.
    /// Relaxed: correctness does not depend on freshness, only on the
    /// value never exceeding the true covered prefix.
    covered_upto: CachePadded<AtomicU32>,
    /// Global dirty generation counter.  Incremented on every
    /// `invalidate()`.  Nodes verified transitively green are stamped
    /// with this generation; hits skip re-verification when
    /// `node.verified_gen >= dirty_generation`.
    dirty_generation: CachePadded<AtomicU64>,
    /// Monotonic version of the dirty/epochs/stamps guard identity.
    ///
    /// # Sustaining invariant
    ///
    /// **Every `dirty.store(...)`, `epochs.store(...)` AND
    /// `stamps.store(...)` site MUST `fetch_add(1, Release)` this counter
    /// after the LAST store of the sequence.** There are exactly two store
    /// sequences (the grow path that appends blocks to each table, and the
    /// quiesced [`Self::reset`] that swaps in fresh tables). The
    /// store-then-bump order lets snapshot readers treat a matching
    /// generation as proof that their cached guards observe every block the
    /// live tables had at some instant within the generation, and treat a
    /// mismatch as a signal to reload. See `dirty_snapshot` for the
    /// soundness argument. One counter covers all three tables because
    /// every sequence stores all three (grow) or swaps all three (reset) —
    /// the bump always comes after ALL of the sequence's stores. The two
    /// sites are tagged `INVARIANT ANCHOR — table_gen::{grow,reset}` so a
    /// grep audit can confirm nothing escaped untagged; a future fourth
    /// table or one-table store sequence would silently break
    /// `dirty_snapshot`/`epochs_snapshot`/`stamps_snapshot` otherwise.
    table_gen: CachePadded<AtomicU64>,
    /// Per-thread cache of the [`Self::dirty`], [`Self::epochs`] and
    /// [`Self::stamps`] guards, keyed by generation.
    ///
    /// A fresh `ArcSwap` load costs an atomic pointer load plus an [`Arc`]
    /// refcount increment/decrement pair on a hot shared line.
    /// `is_green` runs once per cache hit *and* once per miss level,
    /// `with_epoch_word` runs several times per miss level, and
    /// `verify_inner` now reads a stamp slot on every hit, so the cache
    /// holds `(generation, dirty guard, epochs guard, stamps guard)` per
    /// thread and the snapshot helpers validate with a single read-only
    /// `Acquire` load of `table_gen`. Field (not `thread_local!`):
    /// `DepGraph<S>` is generic over the scheduler, and the macro cannot
    /// capture outer generics.
    dirty_table_cache: thread_local::ThreadLocal<
        std::cell::RefCell<(
            u64,
            Option<Arc<Vec<Arc<DirtyBlock>>>>,
            Option<Arc<Vec<Arc<EpochBlock>>>>,
            Option<Arc<Vec<Arc<StampBlock>>>>,
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
    rev: [CachePadded<AtomicU64>; Durability::LEVELS],
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
            _t.take_dep_shard(),
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
            _t.take_dep_shard(),
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
            _t.take_grow(),
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
            _t.acquire_lowering(),
        )
    }

    // (The old "cascade shape" combinator — `take_cascade_shard_r`,
    // `LoweringHeld → LoweringShard`, table #13 — is retired: the
    // reverse-edge log rework (#10) removed the shard lock from
    // `propagate_lowering`'s reader walk, so there is no shard lock for
    // the ticket to scope while lowering is held.)

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
        (r, t1.release_shard_w(g))
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
        (r, t1.release_shard_r(g))
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
        let t3 = t2.release_grow(gg); // → OneShard (shard still held)
        (r, t3.release_shard_w(g)) // → Idle
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
        // 64-bit only: `1 << 48` does not compile on a
        // 32-bit target (shift ≥ width). Rather than derive the offset
        // from `usize::BITS` — which would silently shrink the safety
        // margin against `NEXT_SYSTEM_ID`'s range to something the
        // "can never alias" argument no longer supports — fail the build
        // with a clear message.
        #[cfg(not(target_pointer_width = "64"))]
        compile_error!(
            "DepGraph::next_owner_id assumes a 64-bit target: the \
             NEXT_GRAPH_ID start offset (1 << 48) needs the width to keep \
             standalone graph ids out of NEXT_SYSTEM_ID's range"
        );
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
                        fingerprints: Vec::new(),
                        classes: Vec::new(),
                        digests: Vec::new(),
                    }))
                })
                .collect(),
            rev_logs: (0..DEP_SHARDS).map(|_| CachePadded(ShardLog::new())).collect(),
            grow_lock: Mutex::new(()),
            dirty: ArcSwap::from_pointee((0..DIRTY_INIT_BLOCKS).map(|_| zero_block()).collect()),
            epochs: ArcSwap::from_pointee(
                (0..EPOCH_INIT_BLOCKS).map(|_| zero_epoch_block()).collect(),
            ),
            stamps: ArcSwap::from_pointee(
                (0..STAMP_INIT_BLOCKS).map(|_| zero_stamp_block()).collect(),
            ),
            // Resting cover bound = the SMALLEST of the three init block
            // counts (epochs has the smallest head start on purpose).
            covered_upto: CachePadded(AtomicU32::new(
                (DIRTY_INIT_BLOCKS.min(EPOCH_INIT_BLOCKS)).min(STAMP_INIT_BLOCKS) as u32,
            )),
            dirty_generation: CachePadded(AtomicU64::new(0)),
            table_gen: CachePadded(AtomicU64::new(0)),
            dirty_table_cache: thread_local::ThreadLocal::new(),
            rev: std::array::from_fn(|_| CachePadded(AtomicU64::new(0))),
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
        let scur = self.stamps.load(); // same critical section
        // All three tables must be adequate for the early return to fire.
        // `EPOCH_INIT_BLOCKS < DIRTY_INIT_BLOCKS` means the two start out
        // asymmetric (the epochs head start is small on purpose — 32 KiB a
        // block), so a dirty-only check used to return with the epoch table
        // still short.  That is the UNSAFE skew direction: `mark_dirty`
        // would find the node's dirty block and then `bump_epoch` would
        // find no epoch word and drop the EPOCH bump (the `dirty_generation`
        // bump still happens — the missing-block arm is not taken) — and
        // the epoch bump is precisely what the finish-side re-check
        // compares.  Grow here instead of silently degrading.
        //
        // The invariant this protects is per ALLOCATED node, not a global
        // length order: the caller is allocating `idx`, so covering `idx`
        // in both tables is exactly "the node about to escape has an epoch
        // word".  `reset` leaves `dirty` longer than `epochs` at rest, but
        // those extra blocks cover only never-allocated indices.
        if cur.len() >= need && ecur.len() >= need && scur.len() >= need {
            return;
        }
        let dirty_new_len = need
            .next_power_of_two()
            .max(cur.len().max(ecur.len()) * 2)
            .max(DIRTY_INIT_BLOCKS);
        // The epoch table has its own per-block cost (`EpochBlock` is
        // `[AtomicU64; 4096]` = 32 KiB) and its own resting length
        // (`EPOCH_INIT_BLOCKS` = 4).  Growing it with `dirty_new_len` (which
        // has a 64-block floor) jumped it straight to 128 blocks ≈ 4 MiB on
        // the first growth past the epoch head start — defeating the reason
        // `EPOCH_INIT_BLOCKS` exists.  Size it geometrically from its *own*
        // length instead; the invariant is per-allocated-node coverage, not a
        // shared length order, and `grow_dirty` re-runs until all three tables
        // cover `need`.
        let epoch_new_len = need
            .next_power_of_two()
            .max(ecur.len() * 2)
            .max(EPOCH_INIT_BLOCKS);
        // The stamp table has its own per-node cost (two `AtomicU64`s plus
        // the d24 flag), so it must NOT inherit the dirty table's
        // `DIRTY_INIT_BLOCKS` floor.  Size it independently and let it
        // double from its own resting length.
        let stamp_new_len = need
            .next_power_of_two()
            .max(scur.len() * 2)
            .max(STAMP_INIT_BLOCKS);
        // Store the EPOCH blocks first.  `grow_dirty` only ever grows, so
        // the only possible transient skew is `dirty.len() < epochs.len()`;
        // any operation on a not-yet-covered slot then degrades to the
        // conservative path (a missing dirty block makes `mark_dirty` bump
        // `dirty_generation` instead of silently dropping the epoch bump).
        // The reverse order would let a mark of a just-covered slot find its
        // dirty block but miss its epoch block — `bump_epoch` would then
        // silently drop the bump that the finish-side epoch re-check relies
        // on.  The `ecur` snapshot from the top is reused — the grow lock
        // the caller holds pins the table for this whole function, so a
        // re-load could only see the same blocks (the old "re-load under
        // the lock" line was stale pre-lock-era text).
        let mut ev: Vec<Arc<EpochBlock>> = Vec::with_capacity(epoch_new_len);
        ev.extend(ecur.iter().cloned());
        ev.extend((ecur.len()..epoch_new_len).map(|_| zero_epoch_block()));
        self.epochs.store(Arc::new(ev));

        let mut v: Vec<Arc<DirtyBlock>> = Vec::with_capacity(dirty_new_len);
        v.extend(cur.iter().cloned());
        v.extend((cur.len()..dirty_new_len).map(|_| zero_block()));
        self.dirty.store(Arc::new(v));

        // Stamps last: a missing stamp slot degrades to "no stamp" (a
        // walk), so its position relative to dirty/epochs is not
        // safety-critical.  It still must be stored before the shared
        // generation bump below, or a snapshot could accept a stale stamp
        // guard.  The top-of-function `scur` snapshot is authoritative
        // for the same grow-lock reason as `ecur` above.
        let mut sv: Vec<Arc<StampBlock>> = Vec::with_capacity(stamp_new_len);
        sv.extend(scur.iter().cloned());
        sv.extend((scur.len()..stamp_new_len).map(|_| zero_stamp_block()));
        self.stamps.store(Arc::new(sv));

        // Raise the cover bound AFTER all three stores.  `Release`
        // (not `Relaxed`): a thread that observes the new bound via the
        // matching `Acquire` load must also observe the table stores above,
        // otherwise it could take the fast path without the grown tables and
        // then hit the pre-growth table via its TLS snapshot — the
        // consumer-side skew that fired spurious `BUG` reports.
        self.covered_upto.store(
            dirty_new_len
                .min(epoch_new_len)
                .min(stamp_new_len) as u32,
            Ordering::Release,
        );

        // Every table store must bump the shared generation (Release)
        // AFTER the last store of the sequence — see `table_gen`. All
        // three tables were stored above, so one bump covers all three.
        //
        // INVARIANT ANCHOR — table_gen::grow
        // Grep this tag (with `table_gen::reset` below) to audit every
        // `dirty.store` / `epochs.store` / `stamps.store` site in this
        // file: each site must be followed by exactly one
        // `table_gen.fetch_add(1, Release)` placed after ALL of that
        // sequence's stores.  A new site touching only ONE table would
        // break the invariant silently — the cached slot's
        // `(gen, dirty, epochs, stamps)` tuple would then be accepted
        // with one table at the new generation and another at the old.
        self.table_gen.fetch_add(1, Ordering::Release);
    }

    /// Current dirty epoch of `node` — its value's revision counter,
    /// bumped on every dirty mark AND on every changed-fingerprint
    /// republish (`mark_dirty` / `publish` / `finish_task_and_publish`).
    /// `Acquire` load, paired with the `AcqRel` bumps.
    ///
    /// `pub(crate)`: the cached-serve path in `mod.rs` captures the epoch
    /// at PROBE time (C1) and hands it back through `read_with_epoch`, so
    /// the capture is made from outside this module.
    #[inline]
    pub(crate) fn epoch_of(&self, node: DepNodeIndex) -> u64 {
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
        debug_assert_ne!(
            node.0, DepNodeIndex::ZERO.0,
            "set_node_durability called with DepNodeIndex::ZERO — index 0 is the \
             DefaultCache 'no node' sentinel and is never allocated"
        );
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
        let snap = self.epochs_snapshot();
        self.lower_durability_to_on(&snap, node, d)
    }

    /// `lower_durability_to` against a CALLER-PROVIDED epochs snapshot
    ///.  `propagate_lowering` hoists ONE snapshot for the
    /// whole cascade: a popular node with N readers otherwise paid N TLS
    /// borrows + `table_gen` loads, exactly what `mark_dirty_batch` already
    /// avoids by using `bump_epoch_on`.  Same append-only argument: a block
    /// in a generation-matching snapshot is current, and a node allocated
    /// AFTER the snapshot degrades to the missing-block arm below.
    fn lower_durability_to_on(
        &self,
        snap: &EpochSnap<'_>,
        node: DepNodeIndex,
        d: Durability,
    ) -> bool {
        let w = snap
            .get(block_of(node.0))
            .map(|b| &b[epoch_word_of(node.0)]);
        let Some(w) = w else {
            // A node allocated after the snapshot: its own finish-side
            // aggregation re-reads the (now lowered) levels of its reads and
            // settles it, so skipping it here is sound — the same
            // conservative direction `bump_epoch_on`'s missing-block arm
            // takes.  Keep the release-mode tripwire for the OTHER reason a
            // block can be missing (an allocated node the table never
            // covered), which would silently disable the cascade.
            debug_assert!(
                node.0 >= self.next_index.load(Ordering::Relaxed),
                "lower_durability_to_on: missing epoch word for ALLOCATED node {node:?} \
                 (next_index = {}) — allocate_node_index must have grown the \
                 epoch table before the index escaped",
                self.next_index.load(Ordering::Relaxed),
            );
            if node.0 < self.next_index.load(Ordering::Relaxed) {
                eprintln!(
                    "BUG: lower_durability_to_on: ALLOCATED node {node:?} has no epoch \
                     word (next_index = {}); the lowering cascade is blind",
                    self.next_index.load(Ordering::Relaxed),
                );
            }
            return false;
        };
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
    /// Lock discipline: NO shard lock at all (log rework #10) — the reader
    /// lists come from `ShardLog::collect_readers` (index snapshot +
    /// short log mutex, released before any other lock), and the level
    /// writes are lock-free epoch-word CASes. Called only under
    /// [`DepGraph::lowering_lock`] (single writer). The old cascade shape
    /// (`LoweringHeld → LoweringShard`, table #13) is retired: with no
    /// shard lock on this path there is nothing for the ticket to scope —
    /// it is held and returned unchanged.
    fn propagate_lowering(
        &self,
        start: DepNodeIndex,
        to: Durability,
        mut t: ticket::Held<ticket::LoweringHeld>,
    ) -> ticket::Held<ticket::LoweringHeld> {
        let _ = &mut t; // held unchanged across the cascade (see doc)
        // ONE growing worklist instead of a fresh `Vec` per level
        //.  Termination is unchanged: levels
        // are monotone downward with only [`Durability::LEVELS`] values, so
        // each node drops at most `LEVELS - 1` times EVER; the
        // `lower_durability_to` success rule doubles as the revisit filter,
        // which also makes cycles harmless.  Stale (tombstoned-but-
        // uncompacted) edges over-lower at worst — monotone-downward, so
        // sound; see the `rev_logs` doc's "known bounded lags".
        let mut work: Vec<(DepNodeIndex, Durability)> = vec![(start, to)];
        // ONE epochs snapshot for the whole cascade: the
        // previous per-reader `lower_durability_to` re-paid the TLS borrow +
        // generation load for every reader of every popped node.
        let snap = self.epochs_snapshot();
        let mut i = 0;
        while i < work.len() {
            let (node, level) = work[i];
            i += 1;
            self.rev_logs[shard_of(node)].with_readers(node, |readers| {
                for &r in readers.iter() {
                    if self.lower_durability_to_on(&snap, r, level) {
                        work.push((r, level));
                    }
                }
            });
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
            drop(t_l.release_lowering(writer)); // → Idle
            return false;
        }
        // Panic-safe odd→even bracket: the old explicit
        // second bump was skipped on unwind, leaving `lowering_epoch`
        // permanently odd and silently retiring the per-level fast path.
        struct LoweringSeqlock<'a>(&'a std::sync::atomic::AtomicU64);
        impl Drop for LoweringSeqlock<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::AcqRel);
            }
        }
        self.lowering_epoch.fetch_add(1, Ordering::AcqRel); // → odd: in flight
        let _seqlock = LoweringSeqlock(&self.lowering_epoch);
        let lowered = self.lower_durability_to(node, to);
        if lowered {
            t_l = self.propagate_lowering(node, to, t_l);
        }
        drop(_seqlock); // → even: stable, on the normal path
        drop(t_l.release_lowering(writer)); // → Idle
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
        let snap = self.epochs_snapshot();
        self.bump_epoch_on(&snap, node)
    }

    /// `bump_epoch` against a CALLER-PROVIDED epochs snapshot.  The
    /// batch fan-out (`mark_dirty_batch`) hoists ONE snapshot for the
    /// whole loop: per-reader `epochs_snapshot()` re-paid the Acquire
    /// `table_gen` load plus the TLS borrow probe on every reader of a
    /// hot node's publish.  Same append-only/allocated-before argument
    /// as the shared dirty snapshot: a block present in a
    /// generation-matching snapshot is current, and a node allocated
    /// AFTER the snapshot degrades to the missing-block arm (`LOW`) —
    /// the same conservative direction a fresh snapshot would take.
    fn bump_epoch_on(&self, snap: &EpochSnap<'_>, node: DepNodeIndex) -> Durability {
        let w = snap
            .get(block_of(node.0))
            .map(|b| &b[epoch_word_of(node.0)]);
        let Some(w) = w else {
            // Unreachable for an ALLOCATED node: `allocate_node_index`
            // grows the epoch table to cover `idx` before the index
            // escapes (see `grow_dirty`).  The silent `LOW` degradation is
            // the exact "epoch bump dropped → finish-side re-check compares
            // 0 vs 0" failure mode `grow_dirty`'s doc warns about — assert
            // the only valid reason (never-allocated, post-`reset`
            // resting-skew index) instead of trusting it.
            debug_assert!(
                node.0 >= self.next_index.load(Ordering::Relaxed),
                "bump_epoch_on: missing epoch word for ALLOCATED node {node:?} \
                 (next_index = {}) — allocate_node_index must have grown the \
                 epoch table before the index escaped; dropping this bump \
                 makes the finish-side epoch re-check compare 0 vs 0",
                self.next_index.load(Ordering::Relaxed),
            );
            // Release-mode tripwire: the debug assert above
            // vanishes in release, where this arm would otherwise drop the
            // epoch bump silently and make the finish-side re-check compare
            // 0 against 0.  The arm is unreachable-by-construction, so the
            // log line costs nothing on the hot path and makes any future
            // violation observable in production.
            if node.0 < self.next_index.load(Ordering::Relaxed) {
                eprintln!(
                    "BUG: bump_epoch_on: ALLOCATED node {node:?} has no epoch word \
                     (next_index = {}); the d16 re-check is blind",
                    self.next_index.load(Ordering::Relaxed),
                );
            }
            return Durability::LOW;
        };
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
    /// All-or-nothing slow-path refresh of the shared TLS slot, shared by
    /// `dirty_snapshot` / `epochs_snapshot` / `stamps_snapshot` /
    /// `dirty_stamps_snapshot`.  The slot is keyed by the single
    /// `table_gen`, which advances only after all three stores — a partial
    /// refresh would leave the other fields at the previous generation
    /// while stamping the slot with the new one, and a later `Cached` fast
    /// path would then hand back a pre-reset table.  Loading all three
    /// here (gen-stable between the two `table_gen` loads) is what makes
    /// "all three or none" STRUCTURAL instead of a grep-audit contract;
    /// it is cheap because this slow path runs only on grow/reset, never
    /// on a hit.  Returns freshly loaded `Arc`s of all three tables at
    /// the stable point; the caller wraps the one it needs.
    fn refresh_shared_slot(
        &self,
        cell: &std::cell::RefCell<(
            u64,
            Option<std::sync::Arc<Vec<std::sync::Arc<DirtyBlock>>>>,
            Option<std::sync::Arc<Vec<std::sync::Arc<EpochBlock>>>>,
            Option<std::sync::Arc<Vec<std::sync::Arc<StampBlock>>>>,
        )>,
    ) -> (
        std::sync::Arc<Vec<std::sync::Arc<DirtyBlock>>>,
        std::sync::Arc<Vec<std::sync::Arc<EpochBlock>>>,
        std::sync::Arc<Vec<std::sync::Arc<StampBlock>>>,
    ) {
        loop {
            let g1 = self.table_gen.load(Ordering::Acquire);
            let dirty = self.dirty.load_full();
            let epochs = self.epochs.load_full();
            let stamps = self.stamps.load_full();
            let g2 = self.table_gen.load(Ordering::Acquire);
            if g1 == g2 {
                if let Ok(mut w) = cell.try_borrow_mut() {
                    w.0 = g2;
                    w.1 = Some(std::sync::Arc::clone(&dirty));
                    w.2 = Some(std::sync::Arc::clone(&epochs));
                    w.3 = Some(std::sync::Arc::clone(&stamps));
                }
                return (dirty, epochs, stamps);
            }
        }
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
            .get_or(|| std::cell::RefCell::new((u64::MAX, None, None, None)));
        if let Ok(r) = cell.try_borrow() {
            if r.0 == tbl_gen {
                if let Ok(mapped) = std::cell::Ref::filter_map(r, |t| t.1.as_ref().map(|a| &**a)) {
                    return DirtySnap::Cached(mapped);
                }
            }
        }
        let (full, _, _) = self.refresh_shared_slot(&cell);
        DirtySnap::Owned(full)
    }

    /// Batched dirty + stamps snapshot (#6 of the perf pass): ONE
    /// `table_gen` load and one TLS slot fetch serve BOTH snapshots, where
    /// `dirty_snapshot()` + `stamps_snapshot()` back-to-back paid two of
    /// each.  `verify_inner` calls this on every uncached verify — the
    /// three `RefCell` shared borrows coexist legally, and the slow path is
    /// the same gen-stable refresh loop the individual helpers run.  The
    /// epochs snapshot stays lazy in the caller: the dominant verify touches
    /// no epoch word, so taking it eagerly here would be a pure loss.
    fn dirty_stamps_snapshot(&self) -> (DirtySnap<'_>, StampSnap<'_>) {
        let tbl_gen = self.table_gen.load(Ordering::Acquire);
        let cell = self
            .dirty_table_cache
            .get_or(|| std::cell::RefCell::new((u64::MAX, None, None, None)));
        // Fast path: the slot is current — ONE shared borrow serves both.
        // (Split borrows so inference on the closure params flows from the
        // cell's tuple type; the two `Ref`s coexist legally.)
        if let Ok(r) = cell.try_borrow() {
            if r.0 == tbl_gen {
                let dirty_ref =
                    std::cell::Ref::filter_map(std::cell::Ref::clone(&r), |t| {
                        t.1.as_ref().map(|a| &**a)
                    });
                let stamp_ref =
                    std::cell::Ref::filter_map(r, |t| t.3.as_ref().map(|a| &**a));
                if let (Ok(d), Ok(s)) = (dirty_ref, stamp_ref) {
                    return (DirtySnap::Cached(d), StampSnap::Cached(s));
                }
            }
        }
        // Slow path (grow/reset only): one gen-stable refresh of the shared
        // slot filling BOTH halves — see `refresh_shared_slot` for why a
        // partial refresh is forbidden.
        let (dirty, _, stamps) = self.refresh_shared_slot(&cell);
        (DirtySnap::Owned(dirty), StampSnap::Owned(stamps))
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
            .get_or(|| std::cell::RefCell::new((u64::MAX, None, None, None)));
        if let Ok(r) = cell.try_borrow() {
            if r.0 == tbl_gen {
                if let Ok(mapped) = std::cell::Ref::filter_map(r, |t| t.2.as_ref().map(|a| &**a)) {
                    return EpochSnap::Cached(mapped);
                }
            }
        }
        let (_, full, _) = self.refresh_shared_slot(&cell);
        EpochSnap::Owned(full)
    }

    /// Owned variant of [`Self::epochs_snapshot`] for [`TaskDeps`]: same
    /// generation-validated TLS slot hit, but always returns a cloneable
    /// `Arc` so the task can keep it across reads.  One `Arc` bump per
    /// task (first fresh read), replacing one TLS probe per read.
    fn epochs_owned(&self) -> Arc<Vec<Arc<EpochBlock>>> {
        let tbl_gen = self.table_gen.load(Ordering::Acquire);
        let cell = self
            .dirty_table_cache
            .get_or(|| std::cell::RefCell::new((u64::MAX, None, None, None)));
        if let Ok(r) = cell.try_borrow() {
            if r.0 == tbl_gen {
                if let Some(epochs) = &r.2 {
                    return Arc::clone(epochs);
                }
            }
        }
        let (_, full, _) = self.refresh_shared_slot(&cell);
        full
    }

    /// Generation-validated snapshot of the stamps table: same protocol
    /// as [`Self::dirty_snapshot`] (shared `table_gen`, shared slot, same
    /// append-only soundness argument).  A missing stamp block degrades
    /// to the conservative "no stamp" arm — the verifier walks — and a
    /// missing slot after `reset` cannot inherit a pre-reset stamp.
    #[inline]
    fn stamps_snapshot(&self) -> StampSnap<'_> {
        let tbl_gen = self.table_gen.load(Ordering::Acquire);
        let cell = self
            .dirty_table_cache
            .get_or(|| std::cell::RefCell::new((u64::MAX, None, None, None)));
        if let Ok(r) = cell.try_borrow() {
            if r.0 == tbl_gen {
                if let Ok(mapped) = std::cell::Ref::filter_map(r, |t| t.3.as_ref().map(|a| &**a)) {
                    return StampSnap::Cached(mapped);
                }
            }
        }
        let (_, _, full) = self.refresh_shared_slot(&cell);
        StampSnap::Owned(full)
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
        let dirty_snap = self.dirty_snapshot();
        let epochs_snap = self.epochs_snapshot();
        self.mark_dirty_on(&dirty_snap, &epochs_snap, node);
    }

    /// Abort-path colour repair: a panic after the publish section can
    /// leave a node green with no staged value, and a transitive reader
    /// would then verify a green closure and serve its own stale value.
    /// `mark_dirty` is lock-free and idempotent, so it is safe under the
    /// quiesce read pin the abort path holds.
    pub(crate) fn mark_dirty_for_abort(&self, node: DepNodeIndex) {
        self.mark_dirty(node);
    }

    /// Snapshot-parameterized body of [`Self::mark_dirty`]: the caller
    /// supplies BOTH append-only snapshots, so a hot in-section caller (the
    /// colour decision inside `finish_task_and_publish`'s shard write
    /// closure) pays no TLS probe or ArcSwap load under the lock.  Same
    /// append-only/`reset`-excluded validity argument as the batch path: a
    /// stale snapshot is conservative (an already-red early return, or a
    /// redundant re-set the idempotent `fetch_or` absorbs; a missing block
    /// takes the LOW/conservative arm).
    fn mark_dirty_on(
        &self,
        dirty_snap: &DirtySnap<'_>,
        epochs_snap: &EpochSnap<'_>,
        node: DepNodeIndex,
    ) {
        debug_assert_ne!(
            node.0, DepNodeIndex::ZERO.0,
            "mark_dirty called with DepNodeIndex::ZERO — index 0 is the \
             DefaultCache 'no node' sentinel and is never allocated; marking \
             it would set a phantom bit nothing can clear"
        );
        // Declared here, assigned in the block branch: the 0→1 tail below
        // (after the early returns) consumes the level the bump's CAS
        // returned.  The missing-block branch returns without touching it.
        // (The singular wrapper takes fresh snapshots per call; hot
        // in-section callers use `mark_dirty_on` with a hoisted pair.)
        let d;
        if let Some(block) = dirty_snap.get(block_of(node.0)) {
            // For an ALLOCATED node the epoch word is guaranteed present:
            // `allocate_node_index` grows BOTH tables to cover `idx` before
            // the index escapes (see `grow_dirty`), and every caller here
            // passes an allocated node.  The post-`reset` resting skew
            // (dirty longer than epochs) covers only never-allocated
            // indices, so `bump_epoch` below is a real bump — the
            // missing-word degradation the missing-block arm documents is
            // unreachable from this branch for valid input.
            //
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
            // unreachable from this branch for valid input.
            d = self.bump_epoch_on(&epochs_snap, node);
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
            // Same assumption `mark_dirty_batch`'s missing-block arm
            // asserts: the only valid missing-block reason is a
            // never-allocated node.  If `grow_dirty` ever failed to cover
            // an ALLOCATED node, this branch silently drops the epoch bump
            // and only bumps `rev[LOW]` + `dirty_generation` — the
            // finish-side epoch re-check then compares 0 against 0 and can
            // miss the mark.
            debug_assert!(
                node.0 >= self.next_index.load(Ordering::Relaxed),
                "mark_dirty: missing dirty block for ALLOCATED node {node:?} \
                 (next_index = {}): the only valid missing-block reason is \
                 that the node was never allocated",
                self.next_index.load(Ordering::Relaxed),
            );
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

    /// Batch form of [`DepGraph::mark_dirty`] for the publish-time fan-out:
    /// N marks → one linearization.  The per-reader work that CANNOT batch
    /// stays per-reader (the `bump_epoch` CAS and the `fetch_or` bit — both
    /// per-node words), but the three SYSTEM-WIDE effects collapse:
    ///
    ///  * ONE `dirty_snapshot` for the whole batch — the append-only
    ///    discipline that validates `mark_dirty`'s single snapshot covers a
    ///    batch unchanged: a block present at snapshot time stays valid
    ///    forever, and every batch reader was allocated before this node's
    ///    rev-edge list (the list the fan-out reads) was built.
    ///  * ONE `dirty_snapshot` AND ONE `epochs_snapshot` for the whole
    ///    batch — both tables' append-only discipline validates them (the
    ///    same argument as the shared dirty snapshot: a block present in a
    ///    generation-matching snapshot is current).  The epochs snapshot
    ///    feeds `bump_epoch_on`; without it, per-reader `bump_epoch` would
    ///    re-pay the Acquire `table_gen` load plus the TLS borrow probe on
    ///    every reader of a hot node's publish.  An auditor counting "what
    ///    does one batch pay?" must count BOTH snapshot round-trips.
    ///  * ONE `rev[l]` add PER LEVEL (delta = count of 0→1 transitions at
    ///    durability ≥ l) instead of d+1 adds per mark — the shared counter
    ///    line stops taking N RMWs per republish.
    ///  * ONE `dirty_generation` bump for the whole batch, LAST.
    ///
    /// The batching is a CLEANER linearization INSIDE `mark_dirty`'s own
    /// argument (do not weaken it — see the MUST-bump comment there): what
    /// that argument forbids is bits whose Release-set is not ordered
    /// before the batch's bump.  Here every bit set (Release) and every
    /// rev add (AcqRel) is sequenced-before the single bump, so a walk
    /// that loads the bumped generation sees ALL bits and ALL adds — a
    /// strictly stronger guarantee than N sequential marks, where a walk
    /// could interleave between two marks.  A walk that loads the
    /// generation before the bump misses the whole batch: the same valid
    /// "walk overlaps the marking thread" linearization as before, and the
    /// stamp it writes is stale relative to future lookups (the bump lands
    /// first) — the conservative direction.
    ///
    /// The 0→1 transition gate is preserved PER READER: an already-red
    /// reader contributes no rev adds (a redundant mark must not thrash
    /// every stamp — see the gate comment in `mark_dirty`), though its
    /// epoch CAS still runs (over-bumping an epoch is conservative).  The
    /// bump is skipped entirely when NO reader transitioned — the batch
    /// form of the early return.
    fn mark_dirty_batch<I: IntoIterator<Item = DepNodeIndex>>(&self, readers: I) {
        // Empty fan-out early-out: an unchanged-fingerprint
        // republish reaches here with an empty list — take the two TLS
        // snapshots (dirty + epochs) only when there is at least one reader.
        let mut readers = readers.into_iter();
        let Some(first) = readers.next() else {
            return;
        };
        let mut level_counts = [0usize; Durability::LEVELS];
        let mut any = false;
        let snap = self.dirty_snapshot();
        // ONE epochs snapshot for the whole fan-out loop (see
        // `bump_epoch_on`): per-reader `bump_epoch` calls re-paid the
        // Acquire `table_gen` load + TLS borrow probe each — the publish
        // of a hot node is exactly the path this file optimizes
        // elsewhere.
        let epochs_snap = self.epochs_snapshot();
        // The first reader runs through the same body as the iterator
        // tail (one loop, `first` folded back in).
        for node in std::iter::once(first).chain(readers) {
            debug_assert_ne!(
                node.0, DepNodeIndex::ZERO.0,
                "mark_dirty_batch called with DepNodeIndex::ZERO — index 0 is \
                 the DefaultCache 'no node' sentinel and is never allocated"
            );
            let Some(block) = snap.get(block_of(node.0)) else {
                // Post-`reset` reader: no cached value can exist, but keep
                // `mark_dirty`'s missing-block accounting — a rev[LOW] add
                // and (below) the generation bump, no bit, no epoch CAS.
                // The epoch-silence here is safe ONLY because the reader was
                // never allocated (post-`reset`, `next_index` restarted at
                // 1 and `reset` cleared every `rev_edges` list, so no
                // allocated reader can be in this set): assert that
                // assumption instead of trusting it — if a future change
                // ever lets a stale reader survive `reset`, this fires
                // instead of silently dropping the epoch bump.
                debug_assert!(
                    node.0 >= self.next_index.load(Ordering::Relaxed),
                    "mark_dirty_batch: missing dirty block for ALLOCATED node \
                     {node:?} (next_index = {}): the only valid missing-block \
                     reason is that the node was never allocated; an allocated \
                     reader whose epoch bump is dropped here stays stale \
                     forever (the finish-side re-check compares 0 vs 0)",
                    self.next_index.load(Ordering::Relaxed),
                );
                // Release-mode tripwire: the debug assert
                // vanishes in release, where this arm would otherwise drop
                // the reader's epoch bump and the bit set SILENTLY — the
                // exact permanent-staleness mode `bump_epoch_on` already
                // guards with a release arm.  Unreachable by construction,
                // so the log line costs nothing on the hot path.
                if node.0 < self.next_index.load(Ordering::Relaxed) {
                    eprintln!(
                        "BUG: mark_dirty_batch: ALLOCATED node {node:?} has no dirty \
                         block (next_index = {}); its mark is dropped and the \
                         d16 re-check is blind",
                        self.next_index.load(Ordering::Relaxed),
                    );
                }
                level_counts[0] += 1;
                any = true;
                continue;
            };
            // Epoch bump BEFORE the bit set, per reader (see `mark_dirty`):
            // an Acquire observer of this reader's red bit must also
            // observe its epoch bump, or the finish-side epoch re-check
            // (d16) can miss a mark whose bit was observable.
            //
            // NO red pre-check: the old `load(Acquire)` +
            // `continue` skipped BOTH the epoch bump and the bit set for an
            // already-red reader — but a concurrent `clear_dirty_on` (the
            // reader's own finish) could clear the bit between that load and
            // this point, silently dropping the mark entirely. The
            // finishing node's post-flush late re-check cannot repair that:
            // its fresh epoch load has no happens-before edge with this
            // thread's skipped CAS. Always running `bump_epoch_on` +
            // `fetch_or` is the sound shape: each reader has its own epoch
            // word, so the extra CAS on an already-red reader is
            // uncontended, and `prev & mask != 0` (below) still gates the
            // per-level rev adds. An over-bumped epoch is conservative —
            // never an under-mark.
            let mask = mask_of(node.0);
            let d = self.bump_epoch_on(&epochs_snap, node);
            let prev = block[word_of(node.0)].fetch_or(mask, Ordering::Release);
            if prev & mask != 0 {
                continue; // already red — stamps were invalidated when it turned red
            }
            any = true;
            for l in 0..=(d as usize) {
                level_counts[l] += 1;
            }
        }
        if !any {
            return; // no 0→1 transition in the batch — no invalidation at all
        }
        // rev adds BEFORE the generation bump, all adds after ALL bits
        // (the per-mark straddle argument applies to the batch as a
        // whole: seeing a bumped rev[l] implies every bit sequenced before
        // the rev phase; seeing the bump implies every add).
        for (l, &c) in level_counts.iter().enumerate() {
            if c > 0 {
                self.rev[l].fetch_add(c as u64, Ordering::AcqRel);
            }
        }
        self.dirty_generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Clear the dirty bit for `node`.  Same append-only-snapshot
    /// discipline as [`DepGraph::mark_dirty`].
    #[inline]
    fn clear_dirty(&self, node: DepNodeIndex) {
        let dirty_snap = self.dirty_snapshot();
        self.clear_dirty_on(&dirty_snap, node);
    }

    /// Snapshot-parameterized body of [`Self::clear_dirty`] — see
    /// `mark_dirty_on` for the validity argument.
    #[inline]
    fn clear_dirty_on(&self, dirty_snap: &DirtySnap<'_>, node: DepNodeIndex) {
        debug_assert_ne!(
            node.0, DepNodeIndex::ZERO.0,
            "clear_dirty called with DepNodeIndex::ZERO"
        );
        if let Some(block) = dirty_snap.get(block_of(node.0)) {
            block[word_of(node.0)].fetch_and(!mask_of(node.0), Ordering::Release);
        }
    }

    /// Lock-free pre-check: do all three append-only tables already cover
    /// `idx`?  Safe without the grow lock because the tables only ever
    /// GROW (`grow_dirty`) or are swapped by a quiesced `reset` (which
    /// runs under the quiesce write lock and cannot overlap an allocator,
    /// which holds the quiesce read guard for its whole frame).  A
    /// "covered" verdict is therefore final: no other thread can shrink
    /// the table out from under us.  A "not covered" verdict is advisory —
    /// the caller falls back to the authoritative grow-locked path.
    ///
    /// This is what keeps the global `grow_lock` cold: without it, every
    /// node allocation serialized on the grow mutex even though growth is
    /// a rare event — the exact global serial point this file otherwise
    /// removes (sharded locks, deferred fan-out).
    #[inline]
    fn tables_cover(&self, idx: u32) -> bool {
        // Fast path: one `Acquire` load replaces three
        // `ArcSwap` loads per node allocation.  `Acquire` pairs with the
        // `Release` store in `grow` so a covered verdict also observes the
        // grown tables (a stale-low value still falls through to the
        // grow-locked check, which is the safe direction).
        if block_of(idx) + 1 <= self.covered_upto.load(Ordering::Acquire) as usize {
            return true;
        }
        let need = block_of(idx) + 1;
        if self.dirty.load().len() < need {
            return false;
        }
        if self.epochs.load().len() < need {
            return false;
        }
        if self.stamps.load().len() < need {
            return false;
        }
        true
    }

    /// Allocate a node with its DECLARED content class (fixed for the
    /// node's lifetime; taken from the owning query's descriptor).
    ///
    /// The compile face of tables #11/#12: inside `with_shard_and_grow`,
    /// Idle → OneShard → ShardAndGrow → OneShard → Idle; `grow_dirty` only
    /// accepts `&GrowGuard`.
    pub fn allocate_node_index(&self, class: ContentClass) -> DepNodeIndex {
        // `checked_add` inside `try_update`, NOT `fetch_add`: the old
        // shape wrapped the counter to 0 BEFORE the assert fired, so a
        // caller that caught the panic would keep allocating from
        // `DepNodeIndex(0)` — the `DefaultCache` sentinel for "no node"
        // (index 0 is never allocated: the counter starts at 1) — and
        // alias every live slot.
        //
        // The bound is `u32::MAX`, not `u32::MAX - 1`: `try_update` returns
        // the value BEFORE the update, so `prev` here IS the minted index —
        // indices run 1, 2, …, `u32::MAX - 1`.  `u32::MAX` itself is never
        // minted: it is `DedupTable`'s inline dedup EMPTY marker (see
        // `TaskDeps`) — handing it out would fire its `debug_assert_ne`
        // inside the hot read path in debug builds, and in release builds
        // silently treat every read of that node as "freshly inserted"
        // WITHOUT storing it — duplicates in `reads`, breaking the
        // documented "`reconcile_forward_edges` receives an
        // already-deduplicated list" invariant for that one node.  (The
        // previous bound refused at `u32::MAX - 1`, which wasted one index
        // and misdescribed `try_update`'s result as the post-update value.)
        // `try_update` leaves the counter SATURATED at `u32::MAX`, so every
        // later call panics the same way.  2^32 nodes is far beyond any
        // real session — fail loudly, fail closed.
        let prev = self
            .next_index
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                if cur >= u32::MAX {
                    None
                } else {
                    Some(cur + 1)
                }
            })
            .unwrap_or_else(|cur| {
                panic!("DepNodeIndex address space exhausted (next_index = {cur})")
            });
        // `prev` is the pre-update counter value, i.e. the minted index.
        let idx = DepNodeIndex(prev);
        let slot = slot_of(idx);
        if self.tables_cover(idx.0) {
            // Fast path: all three tables already cover `idx` — skip the
            // global grow lock entirely.  Sequential allocations almost
            // never grow; taking `grow_lock` on every allocation turned a
            // rare-event mutex into a global serial point on the cold-miss
            // path.  The pre-check is final (see `tables_cover`): a
            // "covered" verdict cannot be invalidated under us.
            let (_, _t) = self.with_shard_w(shard_of(idx), self.ticket(), |shard| {
                ensure_shard_len(shard, slot);
                // Written unconditionally, NOT via `resize_with(.., || class)`:
                // a concurrent allocator's resize must never leave THIS node's
                // slot holding a foreign class — the class is fixed at
                // allocation and drives the semantic-purity debug checks and
                // class-scoped invalidation.
                shard.classes[slot] = class;
            });
        } else {
            let (_, _t) =
                self.with_shard_and_grow(shard_of(idx), self.ticket(), |shard, growth| {
                    self.grow_dirty(idx.0, growth);
                    ensure_shard_len(shard, slot);
                    shard.classes[slot] = class;
                });
        }
        idx
    }

    /// The declared content class of a node, if the node exists.
    ///
    /// `DepNodeIndex(0)` is the never-allocated sentinel (the allocation
    /// counter starts at 1): a class read for it would return whichever
    /// class the FIRST allocated node (slot 0) happens to carry — the
    /// classic sentinel-aliasing bug, so make it loud in debug builds.
    pub fn class_of(&self, node: DepNodeIndex) -> Option<ContentClass> {
        debug_assert_ne!(
            node.0, DepNodeIndex::ZERO.0,
            "class_of called with DepNodeIndex::ZERO — index 0 is the \
             'no node' sentinel and never allocated"
        );
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
    /// another query.  If no task is active, this is a no-op.  The read's
    /// "start revision" epoch is captured HERE, at record time — use
    /// [`Self::read_with_epoch`] when the caller captured the epoch
    /// EARLIER (the cached-serve path captures it at probe time, before
    /// the value can become stale: C1).
    pub fn read(&self, other: DepNodeIndex) {
        self.read_impl(other, None);
    }

    /// Record a read with a CALLER-CAPTURED epoch.  The epoch must have
    /// been captured at or before the moment the value was obtained (probe
    /// time on the cached-serve path).  Using a pre-capture is STRICTLY
    /// CONSERVATIVE: any bump landing between capture and this call is
    /// detected by the finish-side re-check and restores the reader red,
    /// even if the bump did not actually change the value — one extra
    /// recomputation against a swallowed permanent-stale window (C1: a
    /// thread preempted between "value received" and the read-edge record
    /// used to capture the epoch AFTER the mark, compare equal, and
    /// publish green over the pre-mark value).
    pub fn read_with_epoch(&self, other: DepNodeIndex, epoch: u64) {
        self.read_impl(other, Some(epoch));
    }

    fn read_impl(&self, other: DepNodeIndex, captured: Option<u64>) {
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
                let e = match captured {
                    Some(e) => e,
                    None => task.cached_epoch(other).unwrap_or_else(|| {
                        // First fresh read of this task: take the owned
                        // snapshot once, then serve every subsequent read
                        // from it.
                        let snap = self.epochs_owned();
                        let e = epoch_from_owned(snap.as_slice(), other);
                        task.epochs = Some(snap);
                        e
                    }),
                };
                task.read_with_epoch(other, e);
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
    ///
    /// NOTE: this path deliberately discards `read_epochs` — no fingerprint
    /// is published, so there is no stale-restore to run.  Tests that intend
    /// to exercise the epoch re-check (d16/d24) must use
    /// `finish_task_and_publish`; using `finish_task` here means the
    /// stale-reader detection never fires.
    #[cfg(test)]
    pub fn finish_task(&self) {
        let task = TASK_STACK.with(|s| s.borrow_mut().pop());
        let Some(task) = task else { return };
        let task_node = task.current_node;
        // Diagnostic/test path (no value is published here), so the
        // finish-side stale-restore does not apply; the epochs are still
        // consumed to keep the parallel-vec invariant in one place.
        let (reads, _read_epochs, reads_index) = task.into_reads_with_epochs();
        let mut reads = reads;

        // This node's shard: forward edges, RECONCILED to the current
        // read set (stale edges removed — see `reconcile_forward_edges`).
        let removed = {
            let tslot = slot_of(task_node);
            let (r, _t) = self.with_shard_w(shard_of(task_node), self.ticket(), |shard| {
                ensure_shard_len(shard, tslot);
                reconcile_forward_edges(&mut shard.edges[tslot], &reads, &reads_index)
            });
            r
        };

        // The reads' shards: reverse edges, one lock per shard.
        sort_reads_by_shard(&mut reads, None);
        self.flush_rev_edges(task_node, &mut reads);
        // Drop the reverse edges of reads we no longer depend on.
        self.prune_rev_edges(task_node, removed);
    }

    /// Remove `task_node` from the reverse edges of `removed` deps — the
    /// counterpart of [`Self::flush_rev_edges`] for edges the node's
    /// CURRENT recomputation no longer has. Runs after the node-shard
    /// critical section that produced `removed` has been released.
    ///
    /// Log rework (#10): each removed edge is one `Tombstone` push onto the
    /// read node's shard log — a single `Vec::push` under the log's short
    /// mutex, NO shard write lock. The tombstone over-marks until the next
    /// quiesced compaction drains it (sound: `mark_dirty` is idempotent,
    /// `lower_durability_to` is monotone-downward — see `ShardLog`'s doc).
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
            let log = &self.rev_logs[s];
            for &r in group {
                log.push(EdgeOp::Tombstone(r, task_node));
            }
        }
    }

    /// Insert the reverse edges `read → task_node` into the READ nodes'
    /// shard logs — one `Insert` push per read, grouped by shard.  Log
    /// rework (#10): the critical section is a single `Vec::push` under
    /// the log's short mutex; NO shard lock is taken anywhere (see
    /// `rev_logs`' doc for why the log lives outside the shard `RwLock`).
    ///
    /// Deliberately unordered across shards: every consumer feeds
    /// idempotent `mark_dirty` marks, so grouping order changes nothing
    /// semantically.
    ///
    /// Takes `reads` by `&mut` (sorted in place) instead of by value: the
    /// caller re-checks the reads' captured dirty epochs AFTER this flush
    /// — the re-check must run after the reverse edges are visible so
    /// any LATER fan-out reaches this task normally.
    fn flush_rev_edges(&self, task_node: DepNodeIndex, reads: &mut SmallVec<[DepNodeIndex; 8]>) {
        // Callers MUST pre-sort `reads` by shard (`sort_reads_by_shard`)
        // so the grouping below touches each shard's log once — and so
        // any parallel epoch vector stays paired (see
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
            let log = &self.rev_logs[s];
            for &r in group {
                log.push(EdgeOp::Insert(r, task_node));
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
        let (task_node, mut reads, mut read_epochs, reads_index) = match task {
            Some(t) => {
                let node = t.current_node;
                let (reads, epochs, index) = t.into_reads_with_epochs();
                (Some(node), reads, epochs, index)
            }
            None => (
                None,
                SmallVec::new(),
                SmallVec::new(),
                ReadsIndex::Empty,
            ),
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

        // Sort reads and their captured epochs TOGETHER (identical
        // permutation) BEFORE the durability aggregation and the shard
        // write section.  The order is by SHARD
        // (`shard_of = idx & 63`), not by block (`block_of = idx >> 12`), so
        // it does NOT make the epoch lookups walk ascending block indices
        // — what it buys is that `flush_rev_edges` touches
        // each shard's log mutex once per group, and the shard-grouped
        // aggregation keeps the epoch-word reads within one shard's slot
        // run.  The previous finish stored the forward-edge list in this
        // same order, so `reconcile_forward_edges`'s memcmp fast path still
        // hits.  The epoch re-check below stays positional because the
        // permutation is shared.
        sort_reads_by_shard(&mut reads, Some(&mut read_epochs));

        // ---- validate before mutate (debug) ----
        // The class assertion runs BEFORE the forward-edge reconcile, the
        // reverse-edge insert, the fingerprint write, the reader dirtying,
        // AND the durability aggregation — `lower_and_propagate` (the
        // aggregation's propagation tail) CASes epoch words and can
        // cascade taking shard read locks, so it counts as a mutation
        // here too: a panicking assert must never leave a partially
        // mutated graph.  The task-frame pop above necessarily precedes
        // it (the read set IS the popped frame's), but it is TLS-local
        // and atomic-in-effect — no graph state is touched before this
        // point.
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
        // Post-flush re-check below covers the residual window.
        //
        // fp: None (the poison-cleanup path — both production callers are
        // panic unwinds in get_shared_inner) skips aggregation #1, which
        // exists only to keep the PUBLISHED value's level correct, and a
        // poisoned node is red regardless: its next successful compute
        // re-aggregates from its own read set.
        //
        // The LATE re-check below still runs on this path and is
        // load-bearing, not an oversight: a reader of the poisoned node X
        // can verify BEFORE X's next recompute and still reach X's
        // surviving `verified_lvl` stamp (only the gen stamp is killed).
        // Without the late propagation lowering `stored(X)`, a mark on a
        // newly-less-durable closure node m with `dur(m) < stored(X)`
        // would never bump `rev[stored(X)]` and the stamp would survive
        // past m's red bit.  So do NOT "complete" the skip by dropping the
        // late re-check.  The stated tail-latency benefit of skipping
        // aggregation #1 is real but partial: `lower_and_propagate`'s
        // racy pre-check keeps `lowering_lock` cold unless a lowering is
        // genuinely due.
        //
        // The in-section suspect re-check's epoch snapshot is taken
        // HERE, before the write lock, rather than inside the closure.  The
        // closure captures it by reference; the soundness argument for
        // taking it early (the late re-check below is a strict superset and
        // restores anything this one misses, with `finishing` covering the
        // whole method) is written at the use site.  The durability
        // aggregation below REUSES this same snapshot: both
        // passes read the same epoch words, so one TLS setup serves both.
        let suspect_epochs = self.epochs_snapshot();
        if fp.is_some() && !reads.is_empty() {
            let snap = &suspect_epochs;
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
            self.lower_and_propagate(node, min_dur);
        }

        // Stale forward edges dropped by the reconciliation below; the
        // pruning of their reverse edges runs AFTER the critical section
        // releases (the removed deps may live in OTHER shards — never
        // two shard locks at once).  The publish-time fan-out reader list
        // is collected BEFORE the lock (see below) — no pooled buffer copy
        // anymore.
        // The dirty-table twin of `suspect_epochs`: the in-section colour
        // decision (`mark_dirty_on` / `clear_dirty_on` below) reads it
        // under the node's shard write lock without paying a TLS probe or
        // ArcSwap load there.  A stale snapshot is conservative (see
        // `mark_dirty_on`): an already-red reader early-returns and a
        // redundant re-set is absorbed by the idempotent RMW.
        //
        // Taken only on the publish path (`fp.is_some()`): the colour
        // decision itself is gated on `fp`, so the panic-cleanup path used
        // to pay a TLS snapshot probe for a value it never reads.
        let suspect_dirty = fp.is_some().then(|| self.dirty_snapshot());
        // Fan-out reader list: collected AFTER the shard write section,
        // gated on `changed_in_section`.  Collecting before
        // the section left the window `collect < flush(R) < recheck(R) <
        // bump(this node)`: R is absent from the list AND R's epoch
        // re-check ran before the bump, so R stayed green-stale forever.
        // Collecting after the bump closes it: an R already in the list is
        // marked; an R that flushes after the collect necessarily
        // re-checks after its flush, hence after the bump, and restores
        // itself.  This also skips the collect entirely for unchanged
        // republishes.
        let (changed_in_section, removed): (bool, SmallVec<[DepNodeIndex; 4]>) = {
            let (out, _t) = self.with_shard_w(shard_of(node), self.ticket(), |shard| {
                let slot = slot_of(node);
                ensure_shard_len(shard, slot);

                // ---- finish_task part: forward edges, RECONCILED to the
                // current read set (stale edges removed — see
                // `reconcile_forward_edges`).  The removed deps' reverse
                // edges are pruned AFTER this critical section releases
                // (the removed deps may live in OTHER shards — never two
                // shard locks at once).
                let removed = if let Some(tn) = task_node {
                    let tslot = slot_of(tn);
                    ensure_shard_len(shard, tslot);
                    reconcile_forward_edges(&mut shard.edges[tslot], &reads, &reads_index)
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
                    // `reads`/`read_epochs` are already shard-sorted by the sort above;
                    // the pairing is maintained by the SAME permutation being
                    // applied to both vectors in lockstep, not by original
                    // order.
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
                    let suspect_dirty = suspect_dirty
                        .as_ref()
                        .expect("dirty snapshot is taken exactly when fp.is_some()");
                    if suspect {
                        // Keep/put the node red.  The value stays staged in
                        // the cache, but `probe` gates on `is_green`, so it
                        // is unreachable: the next getter takes the compute
                        // path (the `Started` entry is still installed until
                        // the completion path in `get_shared` runs, so it
                        // waits, then re-probes and recomputes).
                        //
                        // Snapshot-parameterized forms (`suspect_dirty` /
                        // `suspect_epochs`, hoisted before the section):
                        // taking fresh snapshots here would re-pay a TLS
                        // probe + ArcSwap load under THIS node's write lock
                        // — the same cost the hoisted suspect check above
                        // exists to avoid, and the same append-only
                        // validity argument applies.
                        self.mark_dirty_on(suspect_dirty, &suspect_epochs, node);
                    } else {
                        self.clear_dirty_on(suspect_dirty, node);
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
                        // that arrive without a mark.  Uses the hoisted
                        // `suspect_epochs` (via `bump_epoch_on`, the batch
                        // path's helper) instead of a fresh
                        // `bump_epoch` snapshot under the write lock; the
                        // CAS target word is the same either way — the
                        // snapshot only decides block presence, and a stale
                        // snapshot degrades to the conservative LOW arm.
                        self.bump_epoch_on(&suspect_epochs, node);
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
                        //
                        // The reader list itself is NOT touched here
                        // anymore: `fanout_readers` was
                        // collected BEFORE the section, and copying it
                        // into a pooled buffer under the write lock was a
                        // vestigial memcpy from the era when collection
                        // ran in-lock. The marks below consume
                        // `fanout_readers` directly.
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
                (changed, removed)
            });
            out
        };

        // Publish-time fan-out, now that the section has released: one
        // lock-free `mark_dirty` per direct reader of this node.  Same
        // marks, same order as before the deferral (perf item 1).
        //
        // GATED on `changed_in_section`: the old
        // `FanoutBuf` was filled ONLY inside the `if changed` arm, so an
        // unchanged-fingerprint republish fanned out to nobody. The
        // pre-lock `fanout_readers` list is collected whenever `fp` is
        // present, so feeding it to `mark_dirty_batch` unconditionally
        // would dirty every reader of every clean republish (g1's exact
        // contract).
        //
        // Batched marks: one snapshot, per-level rev adds, ONE generation
        // bump for the whole fan-out (see `mark_dirty_batch` — the same
        // marks, same per-reader epoch/bit work, one linearization).
        // The reader list is collected HERE, after the section: the epoch
        // bump ran inside it, so collecting now closes the d16 window
        // where a reader that flushed after the old pre-lock collect but
        // re-checked before the bump stayed green-stale.
        if changed_in_section {
            self.rev_logs[shard_of(node)]
                .with_readers(node, |readers| self.mark_dirty_batch(readers.iter().copied()));
        }

        // ---- reverse-edge flush for this task's reads: one lock per
        // READ shard ----
        if let Some(tn) = task_node {
            // Finish window: the ENTIRE in-section work has run — the
            // colour decision is taken AND the deferred fan-out flush
            // (`mark_dirty_batch` above) has been applied; only the
            // rev-edge flush is still ahead.  No shard lock is held at
            // this park site.  (Kept in sync with `SchedulePoint::
            // FinishWindow`'s doc in scheduler.rs — §0 mandate.)
            self.sched
                .suspend(SchedulePoint::FinishWindow { node: tn.0 });
            // `reads`/`read_epochs` were already sorted together before
            // the aggregation/write section, so the flush needs no second
            // permutation.
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
            // snapshot is the SAME `suspect_epochs` taken before the shard write
            // section: `finish_task_and_publish` runs under the
            // caller's quiesce read pin, so `clear_all` cannot reset the table
            // under it and one snapshot serves the aggregation, the in-section
            // re-check and this late re-check — the previous second
            // `epochs_snapshot()` here paid a TLS borrow + generation load for
            // identical data.
            let snap = &suspect_epochs;
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

    /// Read-only twin of [`DepGraph::digest_gate`]: reports whether
    /// `input_digest` differs from the recorded digest WITHOUT storing it.
    /// `invalidate_if_changed` uses this so the digest is written only
    /// inside the (possibly deferred) invalidation closure — otherwise a
    /// deferred invalidation left the new digest visible and a concurrent
    /// caller concluded "unchanged" while the invalidation was still
    /// queued.
    pub(crate) fn digest_would_change(&self, node: DepNodeIndex, input_digest: u64) -> bool {
        let slot = slot_of(node);
        let (changed, _t) = self.with_shard_r(shard_of(node), self.ticket(), |shard| {
            match shard.digests.get(slot).copied().flatten() {
                Some(old) => old != input_digest,
                None => true,
            }
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
    /// The package is a binary (empty lib), so `pub` here does not escape
    /// the crate: these four mutators are widened from `pub(crate)` so the
    /// `benches/dep_graph.rs` harness — a separate crate `#[path]`-including
    /// this file — can drive the red-green machinery directly.
    pub fn invalidate(&self, node: DepNodeIndex, _q: &Quiesced) {
        // O(1) atomic RMW, seqlock-guarded: invalidators serialise on
        // nothing at all — not even a shard read lock.  (The generation
        // bump lives in `mark_dirty`: every red-bit creator bumps, not
        // just this path.)
        self.mark_dirty(node);
    }

    /// Batch form of [`invalidate`](Self::invalidate).  Same per-node red-bit
    /// work, but the system-wide effects (generation bump, level counters)
    /// collapse to one linearization — strictly stronger than N independent
    /// marks, and N−1 fewer RMWs on the hot counters.
    pub fn invalidate_many(&self, nodes: &[DepNodeIndex], _q: &Quiesced) {
        self.mark_dirty_batch(nodes.iter().copied());
    }

    pub fn dirty_readers(&self, node: DepNodeIndex, _q: &Quiesced) {
        // Log rework (#10): the reader list comes from the shard log
        // (index snapshot + short log mutex, released before the marks);
        // the dirty marks themselves need no shard lock (seqlock-guarded
        // atomics).
        self.rev_logs[shard_of(node)]
            .with_readers(node, |readers| self.mark_dirty_batch(readers.iter().copied()));
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
    /// Two phases: first collect the reader list via the log (index
    /// snapshot + short log mutex, no shard lock), then filter by each
    /// reader's declared class under the READER's own shard (classes live
    /// with their node).  A reader committing its edge between the phases
    /// is missed — the same tolerance as before (see `rev_logs`' doc).
    pub(crate) fn dirty_readers_by_class(
        &self,
        node: DepNodeIndex,
        class: ContentClass,
        _q: &Quiesced,
    ) {
        self.rev_logs[shard_of(node)].with_readers(node, |readers| {
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
                // Filter under the READER's own shard, then batch: one
                // `mark_dirty_batch` per shard mirrors `dirty_readers`'s
                // single-linearization publish (one generation bump per
                // batch, one shared snapshot round-trip) instead of N
                // per-reader `mark_dirty` calls — same invalidation
                // semantics, N× fewer linearization points.
                let (_, _t) = self.with_shard_r(s, self.ticket(), |shard| {
                    let matching = readers[start..i]
                        .iter()
                        .copied()
                        .filter(|&r| {
                            shard
                                .classes
                                .get(slot_of(r))
                                .is_some_and(|&c| c.intersects(class))
                        })
                        .collect::<SmallVec<[DepNodeIndex; 8]>>();
                    self.mark_dirty_batch(matching);
                });
            }
        });
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
    ///
    /// NOTE: its fan-out runs IN-LOCK (per-reader `mark_dirty` inside the
    /// shard write section), while production `finish_task_and_publish`
    /// defers the fan-out outside the lock (`mark_dirty_batch` on a pooled
    /// `FanoutBuf`).  The fan-out SEMANTICS are identical; the structural
    /// difference is a perf concern only, which is why tests using
    /// `publish` must not be read as exercising the deferred fan-out path.
    /// Log rework (#10): the reader list is collected from the shard log
    /// BEFORE the write section (never nested under it); the per-reader
    /// marks inside are lock-free.
    #[cfg(test)]
    pub(crate) fn publish(&self, node: DepNodeIndex, fp: u64, _q: &Quiesced) -> bool {
        let readers = self.rev_logs[shard_of(node)].collect_readers(node);
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
                for r in &readers {
                    self.mark_dirty(*r);
                }
            }
            changed
        });
        changed
    }

    pub fn set_fingerprint(&self, node: DepNodeIndex, fp: u64) {
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
        // Restart the global generation at the same value as the freshly
        // zeroed stamps.  Leaving it at its pre-reset value
        // made every post-`clear_all` first hit fail `verified_gen >=
        // current_gen` and pay the seqlock walk until re-stamped.
        self.dirty_generation.store(0, Ordering::Relaxed);
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
                    taken_fp,
                    taken_classes,
                    taken_digests,
                ),
                _t,
            ) = self.with_shard_w(i, self.ticket(), |sh| {
                (
                    std::mem::take(&mut sh.edges),
                    std::mem::take(&mut sh.fingerprints),
                    std::mem::take(&mut sh.classes),
                    // Digest gate state dies with everything else: a reused node
                    // index must not inherit a pre-reset digest (it would let the
                    // gate skip an invalidation for content that never fed the
                    // old node's value).
                    std::mem::take(&mut sh.digests),
                )
            });
            drop((taken_edges, taken_fp, taken_classes, taken_digests));
            // Clear the reverse-edge log + index (log rework #10). `reset`
            // runs under quiescence, so no reader holds a log guard or an
            // index snapshot — the clear cannot strand a scan.
            self.rev_logs[i].reset();
        }
        // Append-only bitmap: swap in a fresh zero bitmap under the
        // growth lock.  A mark racing this reset lands on the replaced
        // snapshot and is dropped — the safe direction, because reset
        // also cleared every cache and active-key entry (nothing cached
        // can be stale).
        let _tw_grow = sync_order::Acquired::acquire(sync_order::Fam::GrowLock, "reset.grow");
        let _growth = self.grow_lock.lock();
        // Store order carries no safety weight HERE: reset runs under
        // `&Quiesced`, so no mark, capture or verifier can observe the
        // window between the two stores.  (grow_dirty's order DOES matter —
        // there a concurrent allocator can mark immediately.)
        //
        // The RESTING state is deliberately asymmetric: dirty is restored
        // to `DIRTY_INIT_BLOCKS` (64 blocks = 262144 nodes) while epochs
        // goes back to `EPOCH_INIT_BLOCKS` (4 blocks = 16384 nodes), so
        // `dirty.len() > epochs.len()` until the first allocation past
        // block 3 grows both.  That is safe because the invariant is per
        // ALLOCATED node, not a global length order: `next_index` restarts
        // at 1, no node exists yet, and `allocate_node_index` calls
        // `grow_dirty(idx)`, whose early return requires BOTH tables to
        // cover `idx`.  The extra dirty blocks cover only indices that
        // were never allocated, which no valid caller can mark.
        // Do NOT "fix" this by restoring 64 epoch blocks: that is the
        // 2 MiB-per-`clear_all` cost `EPOCH_INIT_BLOCKS` exists to avoid.
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
        // Growth re-syncs the pair (`grow_dirty` checks BOTH tables before
        // its early return, so the epoch table can never be left short for
        // an ALLOCATED node — a missing epoch word makes `bump_epoch` drop
        // the epoch bump while the dirty bit and the `dirty_generation`
        // bump still happen, so the finish-side epoch re-check would
        // compare 0 against 0 and miss the mark).
        self.epochs.store(Arc::new(
            (0..EPOCH_INIT_BLOCKS).map(|_| zero_epoch_block()).collect(),
        ));
        // Stamp slots reset with the generation and epoch tables.  This is
        // what prevents a reused index from inheriting a pre-reset
        // `verified_gen` (the `vg >= current_gen` fast path would serve a
        // stale cache entry without walking), a pre-reset `verified_lvl`
        // (every `rev` restarts at 0, so `vl >= 0` would fast-path every
        // reused node), or a stale `finishing == true` (the reused index
        // would be unverifiable until something cleared it).  Quiescence
        // guarantees no compute or verification is in flight, so the swap
        // is safe.
        self.stamps.store(Arc::new(
            (0..STAMP_INIT_BLOCKS).map(|_| zero_stamp_block()).collect(),
        ));
        // Cover bound tracks the resting tables: the same
        // resting min as the constructor.
        self.covered_upto.store(
            (DIRTY_INIT_BLOCKS.min(EPOCH_INIT_BLOCKS)).min(STAMP_INIT_BLOCKS) as u32,
            Ordering::Relaxed,
        );
        // Shared-generation bump AFTER the last store of the sequence —
        // see `table_gen`. All three tables were swapped above, so one bump
        // covers all three and no thread can cache a mixed old/new tuple.
        //
        // INVARIANT ANCHOR — table_gen::reset
        // Second and last store sequence in this file.  A grep audit
        // (`dirty.store`, `epochs.store`, `stamps.store`) must find exactly
        // two anchors.  See the grow anchor's note for why the pairing
        // matters.
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

    /// Compact any shard whose reverse-edge log has grown past
    /// `SHARD_LOG_COMPACT_THRESHOLD`.  Called under quiescence (write
    /// lock) from `mutate_world` — the ONLY scheduled compaction site.
    /// Readers cannot be scanning during this call: the quiesce write lock
    /// excludes every reader (they hold either the `InFlightGuard` read
    /// pin or `&Quiesced`), so `ShardLog::compact` can safely clear the
    /// log — the lost-tail race that would otherwise strand inserts.
    pub(crate) fn compact_rev_edges(&self, _q: &Quiesced) {
        for shard in &self.rev_logs {
            if shard.needs_compaction() {
                shard.compact();
            }
        }
    }

    /// Test-only view of a node's `verified_gen` stamp.  d9's Window 2
    /// asserts that a freshly allocated node (post-`reset`, index reused)
    /// has stamp 0 — i.e. it did NOT inherit the pre-reset closure stamp
    /// that would let its later hits skip the closure walk entirely.
    #[cfg(test)]
    pub(crate) fn verified_gen_of(&self, node: DepNodeIndex) -> u64 {
        let snap = self.stamps_snapshot();
        snap.get(block_of(node.0))
            .map_or(0, |b| {
                b[stamp_word_of(node.0)]
                    .verified_gen
                    .load(Ordering::Acquire)
            })
    }

    /// Test-only view of a per-level mark counter.
    #[cfg(test)]
    pub(crate) fn rev_of(&self, d: Durability) -> u64 {
        self.rev[d as usize].load(Ordering::Acquire)
    }

    /// Test-only view of a node's per-level durability stamp.
    #[cfg(test)]
    pub(crate) fn verified_lvl_of(&self, node: DepNodeIndex) -> u64 {
        let snap = self.stamps_snapshot();
        snap.get(block_of(node.0))
            .map_or(0, |b| {
                b[stamp_word_of(node.0)]
                    .verified_lvl
                    .load(Ordering::Acquire)
            })
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
        debug_assert_ne!(
            node.0, DepNodeIndex::ZERO.0,
            "is_green called with DepNodeIndex::ZERO — index 0 is the \
             DefaultCache 'no node' sentinel; block 0 bit 0 would report a \
             phantom green"
        );
        self.dirty_snapshot()
            .get(block_of(node.0))
            .is_none_or(|b| b[word_of(node.0)].load(Ordering::Acquire) & mask_of(node.0) == 0)
    }

    /// `is_green` against a CALLER-HELD dirty snapshot: identical
    /// predicate, but the snapshot's TLS round-trip (`table_gen` Acquire
    /// + borrow probe) is paid once per walk instead of once per node —
    /// `verify_inner`'s 10k-node storm walks paid two snapshot
    /// round-trips per node (`stamps_snapshot` + `is_green`'s
    /// `dirty_snapshot`).  Holding the snapshot across the walk is sound
    /// by the tables' own discipline: blocks are `Arc`-stable and
    /// append-only, every reachable node was allocated before the walk
    /// began.
    ///
    /// A MISSING BLOCK reads green — and that is the conservative
    /// direction HERE, which is non-obvious.  A missing block means the
    /// node was allocated after the snapshot (or not yet allocated at
    /// all).  If such a node were RED, the invariant
    /// `reader green ⟹ read green` forces the reader that reaches it to
    /// be red too: a red node's `mark_dirty` fans out to its direct
    /// readers, so any reader whose rev edge was flushed before the mark
    /// is marked red itself, and a reader whose rev edge was not yet
    /// flushed is caught by the finish-side epoch re-check (d16).  So a
    /// walk that arrives at a missing block from a green reader is either
    /// looking at a node that cannot be red, or the invariant already
    /// forced the reader red and the walk would not have started.  Any
    /// LATER mark bumps `dirty_generation` and invalidates the stamp, so
    /// a post-snapshot miss cannot hide behind a green verdict forever.
    /// Do NOT "fix" the missing arm to `false`: post-`reset` walks would
    /// spuriously recompute everything.  And a future node type that does
    /// not fan out on mark (e.g. a "constant" eval_always node) would
    /// silently break the invariant above — it needs its own soundness
    /// argument first.
    #[inline]
    fn is_green_on(snap: &DirtySnap<'_>, node: DepNodeIndex) -> bool {
        snap.get(block_of(node.0)).is_none_or(|b| {
            b[word_of(node.0)].load(Ordering::Acquire) & mask_of(node.0) == 0
        })
    }

    /// Mark a node as re-computed (green again).  Lock-free: the bitmap
    /// RMW is seqlock-guarded against swaps, so no shard lock is needed.
    #[inline]
    pub fn mark_green(&self, node: DepNodeIndex, _q: &Quiesced) {
        self.clear_dirty(node);
    }

    // ── Finish-in-progress marker (d24) ─────────────────────────────

    /// Mark `node` as "finish in progress": its colour is not final until
    /// `clear_finishing` runs.  Called before staging the freshly computed
    /// value into the cache, so a concurrent transitive verifier cannot
    /// serve the staged value while a late mark on a read node could still
    /// invalidate it (d24).
    ///
    /// The flag lives in the append-only `stamps` table, so this is a
    /// lock-free `Release` store into a stable slot — no shard read lock
    /// and no write lock held across `set_finishing`/`clear_finishing`,
    /// so the unwind-safety argument still holds.  A missing slot is a
    /// no-op: that can only happen after `reset`, which is quiesced and
    /// clears every cached value anyway.
    #[inline]
    pub(crate) fn set_finishing(&self, node: DepNodeIndex) {
        let snap = self.stamps_snapshot();
        if let Some(b) = snap.get(block_of(node.0)) {
            b[stamp_word_of(node.0)]
                .finishing
                .store(true, Ordering::Release);
        }
    }

    /// Clear the "finish in progress" flag: the node's colour is now
    /// final (green or red per the in-section decision and the post-flush
    /// re-check).
    #[inline]
    pub(crate) fn clear_finishing(&self, node: DepNodeIndex) {
        let snap = self.stamps_snapshot();
        if let Some(b) = snap.get(block_of(node.0)) {
            b[stamp_word_of(node.0)]
                .finishing
                .store(false, Ordering::Release);
        }
    }

    /// Test-only view of a node's finishing flag.
    #[cfg(test)]
    pub(crate) fn is_finishing(&self, node: DepNodeIndex) -> bool {
        let snap = self.stamps_snapshot();
        snap.get(block_of(node.0))
            .is_some_and(|b| b[stamp_word_of(node.0)].finishing.load(Ordering::Acquire))
    }

    /// Returns the first dirty dependency found (if any) so the caller can
    /// re-request it.  Returns `None` if the entire transitive closure is
    /// green.  Thin compatibility wrapper over
    /// [`DepGraph::verify_green_collect_dirty`] — `try_serve_verified_inner`
    /// uses the collect variant to batch re-requests; this first-dirty form
    /// remains for tests and single-dep callers.
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
        let mut out = SmallVec::new();
        if self.verify_green_collect_dirty(node, &mut out) {
            None
        } else {
            Some(out[0])
        }
    }

    /// Transitive green check collecting EVERY dirty dependency, not just
    /// the first (see `verify_inner`).  Returns `true` when the entire
    /// closure is green — and only then are the buffered post-order stamps
    /// committed.  The batch shape lets the caller re-request all dirty
    /// deps and run ONE final re-verify, instead of paying a full-closure
    /// walk per dirty dep after an invalidation storm.
    /// Lock-free global-generation pre-check for
    /// [`Self::verify_green_collect_dirty`]'s fast path: does `node`'s
    /// closure stamp certify the whole closure green at the CURRENT
    /// generation?
    ///
    /// One `dirty_stamps_snapshot` (the same single-TLS-probe pair
    /// `verify_inner` uses) plus four Acquire loads — no `StampBuf`, no
    /// walk state, no `lowering_epoch` samples.  A `true` verdict is
    /// lowering-safe by construction (see the call-site comment): a stamp
    /// can only be current if no mark has linearized since it was written,
    /// and lowerings never mark.  A `false` verdict is advisory — the
    /// caller falls through to the full seqlock walk.
    fn is_globally_verified(&self, node: DepNodeIndex) -> bool {
        let current_gen = self.dirty_generation.load(Ordering::Acquire);
        let (dirty_snap, stamps_all) = self.dirty_stamps_snapshot();
        let slot = stamps_all
            .get(block_of(node.0))
            .map(|b| &b[stamp_word_of(node.0)]);
        // d24 read order: `finishing` FIRST — a node whose finish tail is
        // in flight must never be fast-pathed, its colour is not final.
        if slot.is_some_and(|s| s.finishing.load(Ordering::Acquire)) {
            return false;
        }
        if slot.map_or(0, |s| s.verified_gen.load(Ordering::Acquire)) < current_gen {
            return false;
        }
        // Ground truth: the node's own red bit.  A current stamp should
        // imply green (the stamp was written post-order over a fully
        // verified subtree, and any later mark would have bumped the
        // generation) — this load is belt-and-braces that keeps this
        // path's contract byte-identical to `verify_inner`'s first
        // per-node check.
        Self::is_green_on(&dirty_snap, node)
    }

    pub fn verify_green_collect_dirty(
        &self,
        node: DepNodeIndex,
        dirty_out: &mut SmallVec<[DepNodeIndex; 8]>,
    ) -> bool {
        self.verify_green_impl(node, Some(dirty_out))
    }

    /// Verdict-only variant used for the second (post-re-request) walk:
    /// same global fast path and seqlock, but dirty deps are counted, not
    /// collected — the caller only needs "green or not".
    pub fn verify_green_only(&self, node: DepNodeIndex) -> bool {
        self.verify_green_impl(node, None)
    }

    fn verify_green_impl(
        &self,
        node: DepNodeIndex,
        mut dirty_out: Option<&mut SmallVec<[DepNodeIndex; 8]>>,
    ) -> bool {
        // The contract is "green means no dirty deps", so a reused buffer must
        // not retain entries from a previous call.  Clear it here, *before*
        // the global fast path below can return without touching it.
        if let Some(d) = dirty_out.as_deref_mut() {
            d.clear();
        }
        // ── Global-generation fast path (BEFORE the seqlock) ──────────
        // A node whose closure stamp is current does not need the
        // durability seqlock at all: `verified_gen >= current_gen` means
        // NO mark has occurred since the stamp (every 0→1 mark bumps
        // `dirty_generation` — see `mark_dirty`), and lowerings never
        // mark, so the closure cannot contain a red bit regardless of
        // level state.  The seqlock exists to seal the per-LEVEL verdicts
        // against mid-repair cascades; this verdict rests only on the
        // generation, which an in-flight lowering does not touch.  The
        // common fully-green hit therefore skips `StampBuf` take/return,
        // the `verify_inner` call and both `lowering_epoch` samples.
        //
        // Read order matches `verify_inner`'s root fast path exactly and
        // is load-bearing for the same reason: `finishing` FIRST (a
        // finishing node is not verifiable, d24), then `verified_gen`,
        // then the node's own red bit as ground truth (belt-and-braces —
        // a stamp can never legitimately coexist with a red bit, but the
        // check costs one load and keeps this path's contract identical
        // to the walk's first per-node check).
        //
        // The `VerifyWindow` suspension is kept so scheduler-driven test
        // choreography observes the same park point it always did.
        if self.is_globally_verified(node) {
            self.sched
                .suspend(SchedulePoint::VerifyWindow { node: node.0 });
            return true;
        }

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
            // Fresh collect buffer per attempt: a discarded (overlapped)
            // attempt's dirty deps must not leak into the verdict.
            if let Some(d) = dirty_out.as_deref_mut() {
                d.clear();
            }
            let s1 = self.lowering_epoch.load(Ordering::Acquire);
            let stamps = buf.cleared();
            let clean = self.verify_inner(node, true, stamps, dirty_out.as_deref_mut());
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
            // Stamps commit ONLY on a fully-green walk: with dirty deps
            // recorded, exit frames above them ran but their subtrees are
            // NOT verified (see `verify_inner`'s collect-all note).
            if clean {
                self.flush_stamps(stamps);
                return true;
            }
            return false;
        }
        // Pathological fallback: lowerings kept overlapping every retry
        // (effectively unreachable — see `VERIFY_SEQLOCK_RETRIES`).  A
        // walk with the durability fast path DISABLED (root and
        // interior): every red bit is read directly, sound under any
        // interleaving.  The generation shortcut stays — it is
        // lowering-safe (see above).  No seqlock seal is needed here:
        // without the level shortcuts the walk reads actual colours, and
        // a recompute verdict is conservative under any interleaving.
        if let Some(d) = dirty_out.as_deref_mut() {
            d.clear();
        }
        let stamps = buf.cleared();
        let clean = self.verify_inner(node, false, stamps, dirty_out.as_deref_mut());
        if clean {
            self.flush_stamps(stamps);
            return true;
        }
        false
    }

    /// The verification walk proper: root fast paths + iterative
    /// post-order DFS, factored out of [`DepGraph::verify_green_collect_dirty`]
    /// so the seqlock retry loop can re-run it.  `allow_durability_fast_path`
    /// gates ONLY the per-LEVEL shortcuts (root and interior) — the generation
    /// shortcuts are lowering-safe and stay on in both modes.  Exit
    /// frames push into `stamps` instead of writing the shards directly;
    /// [`DepGraph::flush_stamps`] commits them after the seqlock window
    /// closes.
    ///
    /// COLLECT-ALL-DIRTY: every dirty dependency found is appended to
    /// `dirty_out` and the walk CONTINUES (a dirty node is recorded once —
    /// the visited set stops later paths from re-recording it), instead of
    /// the old first-dirty early abort.  This turns `d` dirty deps from
    /// `d` full-closure walks (caller re-requests → re-walks) into ONE walk
    /// + one batch re-request + one final re-verify.  When `dirty_out` is
    /// non-empty the caller must NOT flush `stamps`: exit frames above a
    /// recorded dirty node still ran, but their subtree is NOT verified —
    /// the buffer is discarded wholesale (pure subtrees lose their stamps,
    /// the conservative direction: a storm is recomputing anyway).
    #[allow(unused_assignments)] // `dirty_seen` is not read on the early-return arms
    fn verify_inner(
        &self,
        node: DepNodeIndex,
        allow_durability_fast_path: bool,
        stamps: &mut Vec<PendingStamp>,
        mut dirty_out: Option<&mut SmallVec<[DepNodeIndex; 8]>>,
    ) -> bool {
        let mut dirty_seen = false;
        // Record a dirty dep, or (no-collect mode) just remember that one
        // was seen.  The second re-verify after re-requests only needs the
        // verdict, so skipping the buffer writes is free.
        macro_rules! mark_dirty {
            ($n:expr) => {{
                dirty_seen = true;
                if let Some(out) = dirty_out.as_mut() {
                    out.push($n);
                } else {
                    // No-collect mode (`verify_green_only`): the verdict is
                    // already decided and the stamps are discarded, so
                    // expanding the rest of the closure is pure waste.
                    return false;
                }
            }};
        }
        let current_gen = self.dirty_generation.load(Ordering::Acquire);
        // F1 (soundness): sample the per-level mark counters ONCE at walk
        // start and use this sample for every emitted stamp.  Sampling
        // `rev[l]` at *commit* time (the old emit loop) let any mark landing
        // in `[evidence .. flush_stamps]` — a window that includes the
        // arbitrarily long `VerifyWindow` suspend — be absorbed into the
        // stamp instead of killing it: a later verifier could then re-certify
        // over a value computed from the pre-edit node.  Any mark after this
        // sample pushes `rev[l]` above `rev_ws[l]` and fails the durability
        // check; marks before it are reflected in the sample.  (The root
        // re-certification inside the fast path keeps using the *checked*
        // `vl`, which is equally sound and more permissive.)
        let rev_ws: [u64; Durability::LEVELS] =
            std::array::from_fn(|i| self.rev[i].load(Ordering::Acquire));

        // ONE batched dirty + stamps snapshot for the WHOLE walk (#6 of the
        // perf pass; perf): the loop body used to pay `is_green`'s and the
        // stamp checks' TLS round-trips per node; the tables' own discipline
        // (Arc-stable append-only blocks, `table_gen`-certified all-or-nothing
        // refresh, reachable nodes allocated pre-walk) makes the held
        // snapshot equivalent — see `is_green_on`.  One `table_gen` load and
        // one TLS slot fetch now serve both snapshots (`dirty_stamps_snapshot`);
        // the seqlock retry loop re-hoists per attempt.  The epochs snapshot
        // stays lazy below — the dominant verify touches no epoch word.
        let (dirty_snap, stamps_all) = self.dirty_stamps_snapshot();

        // Root fast paths, BEFORE the walk state is built.  Every
        // steady-state cached hit runs this function, and the vast
        // majority of hits land here: allocating `visited` + `stack` just
        // to discover the root is already green-and-stamped cost two
        // allocator round-trips per hit.  Ordering matches the walk's own
        // per-node checks exactly: the red bit is ground truth (checked
        // first), the stamp only skips the subtree walk.
        if !Self::is_green_on(&dirty_snap, node) {
            mark_dirty!(node);
            return false;
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
        // Lazy snapshot: taken on first `dur_of` use so the dominant hit
        // (root `verified_gen`) touches no epoch word.  Borrow safety with
        // the walk's reentrancy: `get_or_insert_with` re-borrows the
        // `Option` on each call, and the `Ref` inside an `EpochSnap::Cached`
        // is dropped at the end of the SAME `dur_of` invocation — nothing
        // escapes into the walk's state — so a later `dur_of` call simply
        // re-borrows the already-taken snapshot.  (A parallel-vec epoch
        // capture like `finish_task_and_publish`'s forbids a nested refill;
        // this closure makes no such demand.)  The seqlock window
        // (`s1..s2`) brackets every access, so all epoch reads inside one
        // attempt linearize consistently.
        let mut dur_of = |n: DepNodeIndex| -> Durability {
            let snap = epochs.get_or_insert_with(|| self.epochs_snapshot());
            snap.get(block_of(n.0)).map_or(Durability::LOW, |b| {
                word_dur(b[epoch_word_of(n.0)].load(Ordering::Acquire))
            })
        };
        {
            let slot = stamps_all
                .get(block_of(node.0))
                .map(|b| &b[stamp_word_of(node.0)]);
            // NEW (d24): a finish tail in flight — the colour is not final
            // yet, and the staged value must not be served.  This check MUST
            // precede the `verified_gen` fast path: a finishing node must
            // never be fast-pathed on a stamp.  The stamp slot is stable
            // (append-only table), so these are three lock-free Acquire
            // loads; the read order is load-bearing.
            if slot.is_some_and(|s| s.finishing.load(Ordering::Acquire)) {
                mark_dirty!(node);
                return false;
            }
            if slot.map_or(0, |s| s.verified_gen.load(Ordering::Acquire)) >= current_gen {
                return true;
            }
            // Durability fast path (structural soundness — see
            // `Durability`): the closure was verified at per-level stamp
            // `verified_lvl`, and no mark of durability >= this node's has
            // linearized since (`rev`'s RMW is the level's linearization
            // point — a reader loading a lower value linearizes before the
            // mark, the same overlap doctrine as the global check above).
            // Sound because the finish-time aggregation plus the reverse-edge
            // propagation (`propagate_lowering`) maintain `stored(X) <= min
            // over closure(X) of stored`: a mark of any closure node then
            // bumps this level's counter and kills the stamp.  Every existing
            // query is LOW, so this disjunct only ever fires for queries that
            // opted in; the seqlock-fallback walk disables it.
            if allow_durability_fast_path {
                let vl = slot.map_or(0, |s| s.verified_lvl.load(Ordering::Acquire));
                if vl >= self.rev[dur_of(node) as usize].load(Ordering::Acquire) {
                    // Re-certify at the CURRENT generation.
                    // The level stamp proves the closure is unaffected by
                    // the lower-durability marks that bumped
                    // `dirty_generation`, so promoting `verified_gen` here
                    // restores the O(1) global fast path instead of making
                    // every subsequent hit walk to this same check.
                    stamps.push(PendingStamp {
                        node,
                        generation: current_gen,
                        lvl: vl,
                    });
                    return true;
                }
            }
        }

        // The stable append-only `stamps` table makes the root fast path
        // lock-free: no shard read lock is taken here, and the three
        // checks above are plain Acquire loads.  The read order
        // `finishing` -> `verified_gen` -> `verified_lvl` is load-bearing;
        // see `StampSlot` for why the three words need not be packed.
        //
        // Walk state is TLS-pooled across traversals instead of
        // reallocated per walk.
        //
        // TWO-PHASE BFS, SHARD-GROUPED: the old DFS took
        // one shard read lock PER NODE whose fast paths missed. A wide
        // closure scattered across all 64 shards paid up to 64 lock
        // acquisitions per level where a shard-grouped expansion pays one
        // per DISTINCT shard that level touches. The post-order DFS
        // stamps are replaced by two-phase collect-then-stamp: equivalent
        // because (a) `flush_stamps` writes all buffered stamps in one
        // loop with independent `Release` stores (`verified_gen`/
        // `verified_lvl` are independent certificates — see `StampSlot`),
        // and (b) the caller discards the whole buffer whenever
        // `dirty_out` is non-empty, so per-node exit ordering carried no
        // verdict weight.  `dirty_out` order becomes BFS order; the
        // collect-all contract ("every dirty dep") is order-free, and
        // `try_serve_verified_inner` re-requests each dirty dep
        // independently before one final full re-verify.
        let mut scratch = VerifyScratch::take();
        let visited = &mut scratch.visited;

        // Nodes whose fast paths missed.  Collected separately from
        // `stamps` so a dirty walk emits nothing (matches the DFS
        // per-exit gate, minus the wasted `PendingStamp` constructions).
        let to_stamp = &mut scratch.to_stamp;
        to_stamp.push(node);
        // The root is the only node not deduped at push time (it enters
        // the walk without a parent's expansion loop).
        visited.insert(node);

        // Expand the root's edges into level 1 — one shard read lock,
        // same acquisition count as the DFS's first node.
        // The ping-pong level pair is swapped BY VALUE each round (two
        // simultaneous `&mut scratch.level[..]` would alias — E0499);
        // each iteration owns one full buffer and hands it back cleared
        // on the next round.
        let mut current = std::mem::take(&mut scratch.level[0]);
        {
            let (guard, tkt) = self.take_shard_r(shard_of(node), self.ticket());
            let shard = &*guard;
            if let Some(edges) = shard.edges.get(slot_of(node)) {
                for &dep in edges.iter() {
                    // Push-time dedup: first arrival wins.
                    if visited.insert(dep) {
                        current.push(dep);
                    }
                }
            }
            let _ = tkt.release_shard_r(guard);
        }
        // Hoisted out of the level loop: rebuilding
        // 64 SmallVecs on EVERY BFS level was pure stack traffic, dominant
        // on deep single-node chains.  `touched` lets a level clear only
        // the shards it used.
        const _: () = assert!(DEP_SHARDS == u64::BITS as usize);
        let shards_to_expand = &mut scratch.shards_to_expand;
        let mut touched: u64 = 0;
        while !current.is_empty() {
            let mut next = std::mem::take(&mut scratch.level[1]);
            // The taken buffer holds whatever the previous round left in
            // the scratch slot (the old level's nodes) — clear it before
            // pushing, or stale nodes would ride along.
            next.clear();
            // Phase A — fast-path checks, no locks.  First visit is
            // guaranteed by push-time dedup, so no `visited` re-check
            // runs here: the check runs exactly once per node, on its
            // sole arrival ("green by first visit" survives the DFS→BFS
            // move unchanged).
            for &n in current.iter() {
                // The node itself must be green — checked BEFORE the
                // stamp fast path.  The stamp is an optimistic summary
                // written under a generation snapshot that this walk's
                // `dirty_generation` load can legitimately miss (no
                // happens-before edge from the marking thread to this
                // walk), so a "current" stamp must never override the
                // node's own red bit.  Linearization: a walk that loads
                // the generation before a mark's bump overlaps the mark
                // and linearizes before it — its verdict is validly
                // stale and rejected by the next reader that loads the
                // bumped generation.
                if !Self::is_green_on(&dirty_snap, n) {
                    mark_dirty!(n);
                    continue;
                }
                let sslot = stamps_all
                    .get(block_of(n.0))
                    .map(|b| &b[stamp_word_of(n.0)]);
                // d24: finishing nodes are not verifiable — treat as
                // dirty (read order finishing → gen → lvl, load-bearing).
                if sslot.is_some_and(|s| s.finishing.load(Ordering::Acquire)) {
                    mark_dirty!(n);
                    continue;
                }
                // Fast path: the CLOSURE was already verified at this
                // generation — do NOT expand, do NOT re-stamp.
                if sslot.map_or(0, |s| s.verified_gen.load(Ordering::Acquire)) >= current_gen {
                    continue;
                }
                // Per-level durability fast path — same structural
                // contract as the root check above.
                if allow_durability_fast_path {
                    let vl = sslot.map_or(0, |s| s.verified_lvl.load(Ordering::Acquire));
                    if vl >= self.rev[dur_of(n) as usize].load(Ordering::Acquire) {
                        // Same re-certification as the root path: the node
                        // was not expanded because its level stamp proves
                        // its subtree unaffected, so stamp it at the
                        // current generation.
                        to_stamp.push(n);
                        continue;
                    }
                }
                // Fast paths missed: needs expansion; needs a fresh
                // stamp iff the closure stays clean.
                to_stamp.push(n);
                let s = shard_of(n);
                if shards_to_expand[s].is_empty() {
                    touched |= 1u64 << s;
                }
                shards_to_expand[s].push(n);
            }
            // Phase B — expand: ONE shard read lock per DISTINCT shard
            // this level touches, held across every node the shard owns
            // (the same shape `flush_rev_edges` uses on the write side).
            let mut bits = touched;
            while bits != 0 {
                let s = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let group = &shards_to_expand[s];
                let (guard, tkt) = self.take_shard_r(s, self.ticket());
                let shard = &*guard;
                for &n in group.iter() {
                    if let Some(edges) = shard.edges.get(slot_of(n)) {
                        for &dep in edges.iter() {
                            // Push-time dedup: later arrivals (from other
                            // parents in this level, or already-seen
                            // ancestors) are dropped here.
                            if visited.insert(dep) {
                                next.push(dep);
                            }
                        }
                    }
                }
                let _ = tkt.release_shard_r(guard);
            }
            // Clear only the shards this level used; capacity is retained
            // for the next level.
            let mut bits = touched;
            while bits != 0 {
                let s = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                shards_to_expand[s].clear();
            }
            touched = 0;
            // Ping-pong: `next` becomes `current`; the consumed old-level
            // buffer (its nodes all processed above) goes back into the
            // scratch slot.  Its stale contents are cleared on the next
            // round's `take` (see the clear above).
            let consumed = std::mem::replace(&mut next, Vec::new());
            scratch.level[1] = current;
            current = consumed;
        }
        // Walk over: return both level buffers to the scratch for the
        // Drop to pool (cap-checked, cleared there).
        scratch.level[0] = current;

        // Emit stamps iff the closure is clean.  Matches the DFS gate's
        // caller-visible behavior: when any dirty dep exists, the whole
        // buffer is discarded by the caller, so constructing
        // `PendingStamp`s in that case is pure waste.  Reading
        // `rev[dur_of(n)]` here is safe: the seqlock (`s1`/`s2` around
        // `verify_inner`) retries the whole walk if `lowering_epoch`
        // moved, so every read inside this window linearizes
        // consistently.
        if !dirty_seen {
            // Take the epoch snapshot ONCE for the whole emit loop
            //: `dur_of` re-borrowed the `Option`/`Ref` per
            // stamped node.
            let dur_snap = epochs.get_or_insert_with(|| self.epochs_snapshot());
            let mut lvls: SmallVec<[usize; 16]> = SmallVec::new();
            lvls.reserve(to_stamp.len());
            for &n in to_stamp.iter() {
                let d = dur_snap.get(block_of(n.0)).map_or(Durability::LOW, |b| {
                    word_dur(b[epoch_word_of(n.0)].load(Ordering::Acquire))
                });
                lvls.push(d as usize);
            }
            for (&n, &l) in to_stamp.iter().zip(lvls.iter()) {
                stamps.push(PendingStamp {
                    node: n,
                    generation: current_gen,
                    lvl: rev_ws[l],
                });
            }
        }
        !dirty_seen
    }

    /// Commit the buffered post-order stamps of a completed,
    /// seqlock-sealed verification.  The stable append-only stamp slot
    /// makes this a set of lock-free Release stores: no shard sort, no
    /// shard write lock, no `ensure_shard_len`.  The old same-lock
    /// pair-visibility requirement is gone; `verified_gen` and
    /// `verified_lvl` are independently valid certificates (see
    /// `StampSlot`).  `finishing` is untouched — it belongs to the
    /// compute/finish protocol, not to verification.
    fn flush_stamps(&self, stamps: &mut Vec<PendingStamp>) {
        if stamps.is_empty() {
            return;
        }
        let snap = self.stamps_snapshot();
        for s in stamps.iter() {
            if let Some(b) = snap.get(block_of(s.node.0)) {
                let slot = &b[stamp_word_of(s.node.0)];
                slot.verified_gen.store(s.generation, Ordering::Release);
                slot.verified_lvl.store(s.lvl, Ordering::Release);
            }
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

    /// Consecutive `DepNodeIndex` values — the allocation shape this
    /// table exists to dedup — must spread across the table.  The
    /// original constant was `0x9E3_79B9` (one hex digit short of the
    /// 32-bit golden ratio): `K >> 26 ≈ 2` collapsed the spread to 2–3
    /// slots apart, so a task reading a contiguous run of freshly
    /// allocated nodes filled the table in near-linear order and linear
    /// probing degraded toward O(k) per insert — reviving the O(k²)
    /// cliff this table removed.  There was no test on `probe_start`,
    /// which is why the typo survived.
    ///
    /// The pinned bound is MEASURED, not aspirational: the first 64
    /// multiples of the golden ratio hit 55 of 64 top-six-bit buckets
    /// (a three-distance-theorem artifact — they do not biject onto the
    /// 64 buckets; the gap multiset is φ⁻⁸×24, φ⁻⁹×31, φ⁻¹⁰×10), while
    /// the typo'd constant hit only 37 with adjacent clustering.  52
    /// sits well clear of both.  The full proof — why 1/φ maximizes
    /// `min ||q·alpha||`, the exact occupancy ledger (9 empty / 9 double
    /// buckets), the stability wall 2215/3584, and the alpha_2 = 37
    /// counterexample — lives in `research/fibonacci-occupancy/scripts/golden_ratio_occupancy.py`,
    /// whose every claim is proven in integer arithmetic, not sampled.
    #[test]
    fn dedup_probe_start_spreads_consecutive_indices() {
        let mut seen = [false; LINEAR_DEDUP_MAX];
        for i in 1..=LINEAR_DEDUP_MAX as u32 {
            seen[DedupTable::probe_start(DepNodeIndex(i))] = true;
        }
        let distinct = seen.iter().filter(|&&s| s).count();
        assert_eq!(
            distinct, 55,
            "consecutive DepNodeIndex values must spread to exactly 55/64 slots \
             under 0x9E37_79B9.  This is a theorem, not a measurement: the \
             constant sits 1.2e-10 from 1/phi, far inside the proven pattern \
             radius 9.4e-6, so its full bucket assignment equals the golden \
             one and Occ(64,64) = 55 (derivations.md §3.2; \
             research/fibonacci-occupancy/scripts/golden_ratio_occupancy.py::check_q3 asserts it exactly). \
             The 0x09E379B9 typo collapses to 37/64.  Got {distinct}/64"
        );
    }

    /// Wide-closure regression guard for the BFS verify walk
    /// §4.2): a root with 128 direct leaves (past the inline `SmallVec`
    /// capacity, spread over many shards) whose closure is verified,
    /// partially invalidated, collected, re-requested (re-published) and
    /// re-verified.  Pins, as SETS (never sorted — the contract is
    /// "every dirty dep", not an order):
    ///   1. a clean wide closure verifies green in one pass;
    ///   2. the collect reports EXACTLY the marked subset — no dropped
    ///      node (shard-grouping bug between group construction and
    ///      expansion), no spurious extra (stale `visited` reuse from
    ///      the TLS pool);
    ///   3. after every dirty leaf republishes with an UNCHANGED
    ///      fingerprint, the final re-verify passes and re-stamps.
    #[test]
    fn verify_wide_closure_reports_every_dirty_dep_before_restamp() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        const N: usize = 128;
        // N leaves, each published with its own fingerprint.
        let mut leaves = Vec::with_capacity(N);
        for i in 0..N {
            let l = dg.allocate_node_index(ContentClass::SEMANTIC);
            dg.start_task(l);
            dg.finish_task();
            dg.finish_task_and_publish(l, Some(i as u64));
            leaves.push(l);
        }
        // Root reads all of them in one task, then publishes.
        let root = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.start_task(root);
        for &l in &leaves {
            dg.read(l);
        }
        dg.finish_task();
        dg.finish_task_and_publish(root, Some(0xDEAD_BEEF));

        // 1. Clean wide closure: green in one pass.
        let mut dirty: SmallVec<[DepNodeIndex; 8]> = SmallVec::new();
        assert!(
            dg.verify_green_collect_dirty(root, &mut dirty),
            "a fully published wide closure must verify green"
        );
        assert!(dirty.is_empty());

        // 2. Mark a scattered subset dirty; the collect must report it
        //    exactly (as a set).
        let subset: Vec<DepNodeIndex> =
            leaves.iter().step_by(3).copied().collect();
        for &l in &subset {
            dg.invalidate(l, &q());
        }
        dirty.clear();
        assert!(
            !dg.verify_green_collect_dirty(root, &mut dirty),
            "a closure containing red leaves must come back dirty"
        );
        let mut got: Vec<DepNodeIndex> = dirty.iter().copied().collect();
        got.sort_by_key(|n| n.0);
        let mut want: Vec<DepNodeIndex> = subset.clone();
        want.sort_by_key(|n| n.0);
        assert_eq!(got, want, "collect-all must report exactly the marked subset");

        // 3. Re-request = republish each dirty leaf with its UNCHANGED
        //    fingerprint (changed == false ⇒ the leaf is cleared, the
        //    root is NOT fanned out to).  The final re-verify must then
        //    pass and re-stamp every clean leaf plus the root.
        for (i, &l) in leaves.iter().enumerate() {
            if subset.contains(&l) {
                dg.finish_task_and_publish(l, Some(i as u64));
            }
        }
        dirty.clear();
        assert!(
            dg.verify_green_collect_dirty(root, &mut dirty),
            "after every dirty leaf re-publishes unchanged, the final \
             re-verify must pass"
        );
        assert!(dirty.is_empty());
        // The root's stamp is back: the global fast path now serves it.
        assert!(dg.is_globally_verified(root));
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
    fn test_allocate_node_resizes_shard_slots() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let a = dg.allocate_node_index(ContentClass::SEMANTIC);
        let b = dg.allocate_node_index(ContentClass::SEMANTIC);
        assert!(a.0 < b.0);
        let shard = dg.shards[shard_of(b)].read();
        // Log rework (#10): the reverse-edge log is a flat `Vec` of ops
        // (its compacted index grows only at compaction), but the shard's
        // per-slot vectors must still cover the new node.
        assert!(shard.edges.len() > slot_of(b));
        assert!(shard.fingerprints.len() > slot_of(b));
        assert!(shard.classes.len() > slot_of(b));
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
            assert!(shard.edges.len() > slot_of(n));
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
        // The durability fast path must RE-CERTIFY at the current
        // generation: otherwise every later hit keeps walking
        // to the level check instead of taking the O(1) global stamp.
        assert!(
            dg.is_globally_verified(root),
            "durability fast path must refresh verified_gen"
        );

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
        r.err()
            .expect("debug builds must panic on a cross-system edge");
        #[cfg(not(debug_assertions))]
        r.expect("release builds skip, never panic");
        let after = super::cross_system_skip_count();
        assert!(
            after > before,
            "a dropped cross-system edge must be counted (was {before}, now {after})"
        );
    }

    /// Every ALLOCATED node must have an epoch word — the precise form of
    /// the "epoch table never lags" invariant (safety side of
    /// `EPOCH_INIT_BLOCKS`).
    ///
    /// The epoch table's head start is deliberately smaller than the dirty
    /// table's (an epoch block is 32 KiB, a dirty block 512 bytes), so the
    /// pair starts asymmetric.  `grow_dirty`'s old early return looked only
    /// at the dirty table's length, which meant a graph crossing the epoch
    /// boundary but not the dirty one could allocate nodes whose dirty
    /// block exists while their EPOCH BLOCK DOES NOT.  That is the unsafe
    /// skew: `mark_dirty` would find the bit, `bump_epoch` would find no
    /// word and silently drop the epoch bump (the bit-set and the
    /// `dirty_generation` bump still happen), so a finish-side epoch
    /// re-check comparing `epoch_of(node)` against a captured value would
    /// see 0 vs 0 and conclude "unchanged".
    ///
    /// The invariant is PER NODE, not a global length order: after `reset`
    /// the dirty table is deliberately LONGER than the epoch table while no
    /// node is allocated (see `reset`), and that is safe.  On a graph that
    /// has only ever grown — this test — the grow path also happens to keep
    /// `dirty.len() <= epochs.len()`; that assertion pins the grow-path
    /// shape, while the mark of the LAST node pins the invariant that
    /// actually matters.  `reset_resting_skew_is_benign_for_allocated_nodes`
    /// covers the post-reset side.
    ///
    /// Allocates past `EPOCH_INIT_BLOCKS` (well inside `DIRTY_INIT_BLOCKS`)
    /// and then proves a mark of the LAST node really does bump its epoch.
    #[test]
    fn epoch_table_covers_every_allocated_node() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        let n = super::EPOCH_INIT_BLOCKS * 4096 + 8;
        let nodes: Vec<DepNodeIndex> = (0..n)
            .map(|_| dg.allocate_node_index(ContentClass::SEMANTIC))
            .collect();
        let late = *nodes.last().expect("nonempty");

        // Node coverage, not a length order: the epoch table now grows
        // geometrically from its own (small) resting size, so it is expected
        // to be the *shorter* table on a grown graph.  The safety invariant is
        // that every allocated node's block exists in both tables.
        let allocated_blocks = n.div_ceil(4096) as usize;
        assert!(
            dg.epochs_snapshot().len() >= allocated_blocks,
            "epoch table does not cover every allocated node: epochs={} need={allocated_blocks}",
            dg.epochs_snapshot().len(),
        );
        assert!(
            dg.dirty_snapshot().len() >= allocated_blocks,
            "dirty table does not cover every allocated node: dirty={} need={allocated_blocks}",
            dg.dirty_snapshot().len(),
        );
        // The independent sizing must actually pay off: the epoch table (32
        // KiB per block) stays far smaller than the dirty table (512 B per
        // block) after the same allocation.
        assert!(
            dg.epochs_snapshot().len() < dg.dirty_snapshot().len(),
            "epoch table was sized like the dirty table again: dirty={} epochs={}",
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

    /// Post-`reset` resting state: the dirty table is deliberately LONGER
    /// than the epoch table (`DIRTY_INIT_BLOCKS` vs `EPOCH_INIT_BLOCKS`),
    /// which is safe BECAUSE the invariant is per allocated node, not a
    /// global length order.
    ///
    /// This is the trap the old wording hid: `dirty.len() <= epochs.len()`
    /// does not hold at rest, but every index the dirty table covers past
    /// the epoch table is unallocated (`next_index` restarts at 1), so no
    /// valid caller can mark one.  The test pins BOTH halves: the exact
    /// resting lengths (so "unifying" the two constants trips here) and
    /// the per-node guarantee across the epoch head-start boundary.
    #[test]
    fn reset_resting_skew_is_benign_for_allocated_nodes() {
        let dg = DepGraph::new(Arc::new(NoOpScheduler));
        // Grow the pre-reset tables past the epoch head start, so `reset`
        // demonstrably SHRINKS them back to the resting shape instead of
        // merely leaving a fresh graph untouched.
        for _ in 0..(EPOCH_INIT_BLOCKS * 4096 + 8) {
            let _ = dg.allocate_node_index(ContentClass::SEMANTIC);
        }
        dg.reset(&q());

        // Resting state: dirty is the longer table, by design.
        assert_eq!(dg.dirty_snapshot().len(), DIRTY_INIT_BLOCKS);
        assert_eq!(dg.epochs_snapshot().len(), EPOCH_INIT_BLOCKS);
        assert!(dg.dirty_snapshot().len() > dg.epochs_snapshot().len());

        // No node exists yet, so the skew covers only unallocated indices.
        // The FIRST post-reset node is inside the epoch head start: its
        // mark must bump the epoch normally.
        let first = dg.allocate_node_index(ContentClass::SEMANTIC);
        assert_eq!(first.0, 1, "reset must restart the allocator at 1");
        assert_eq!(dg.epoch_of(first), 0);
        dg.mark_dirty(first);
        assert_eq!(dg.epoch_of(first), 1, "first post-reset mark must bump");

        // The node just past the epoch head start forces the joint grow;
        // after it, the epoch table must cover the node it handed out.
        for _ in 0..(EPOCH_INIT_BLOCKS * 4096 - 1) {
            let _ = dg.allocate_node_index(ContentClass::SEMANTIC);
        }
        let late = dg.allocate_node_index(ContentClass::SEMANTIC);
        assert!(late.0 as usize > EPOCH_INIT_BLOCKS * 4096);
        assert!(
            dg.epochs_snapshot().len() > EPOCH_INIT_BLOCKS,
            "allocating past the epoch head start must grow the epoch table"
        );
        assert_eq!(dg.epoch_of(late), 0);
        dg.mark_dirty(late);
        assert_eq!(
            dg.epoch_of(late),
            1,
            "mark_dirty must bump a post-reset node past the epoch head start"
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
            vec![5, 70],                                       // 5, 6 — already sorted
            vec![70, 5],                                       // 6, 5 — swapped
            vec![7, 71, 135, 9, 3],                            // all shard 7 — sorted, dup shards
            vec![9, 70, 3, 135, 5],                            // 9, 6, 3, 7, 5 — small, unsorted
            vec![1, 2, 130, 3, 8, 4],                          // 1, 2, 2, 3, 8, 4 — equal shards
            vec![7, 8, 1, 2, 130, 3],                          // 7, 8, 1, 2, 2, 3 — equal shards
            (0..40).map(|i| i * 64 + i).collect(),             // large, already sorted
            (0..40).rev().map(|i| i * 64 + i).collect(),       // large, reversed
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
        let mut list: SmallVec<[DepNodeIndex; 4]> =
            [5u32, 7, 9].into_iter().map(DepNodeIndex).collect();
        let reads = [7u32, 5, 11].map(DepNodeIndex);
        let index = super::ReadsIndex::build(&reads);
        let removed = super::reconcile_forward_edges(&mut list, &reads, &index);
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
        let removed2 = super::reconcile_forward_edges(&mut list, &reads, &index);
        assert_eq!(
            removed2.to_vec(),
            vec![],
            "second reconciliation is a no-op"
        );
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

        // Force the SEQLOCK path, not the global fast path: an unrelated
        // LOW-durability mark bumps `dirty_generation` past x's closure
        // stamp (so `is_globally_verified` fails and the seqlock loop runs)
        // while leaving `rev[HIGH]` untouched (so x's per-level stamp still
        // passes inside the walk and the verdict stays green).  Without
        // this, the VerifyWindow park/retry choreography below would be
        // short-circuited by the generation shortcut.
        let unverified = dg.allocate_node_index(ContentClass::SEMANTIC);
        dg.set_node_durability(unverified, Durability::LOW);
        dg.invalidate(unverified, &q());
        assert!(dg.verified_lvl_of(x) > 0);

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
