//! The emitter: the only layer that performs I/O.
//!
//! An [`Emitter`] pairs a pure [`Renderer`] with a [`MakeSink`] factory.  It
//! owns the batch loop — `begin`, one `render` per diagnostic, the summary,
//! `end`, flush — and is the sole place that touches a sink.  Renderers never
//! see an `io::Write`; sinks never see a `Diagnostic`.

use crate::diagnostics::render::{
    PRELOAD_SLOTS, PreloadedSource, RenderCtx, Renderer, SharedRenderState, SharedSources,
    source_fingerprint,
};
use crate::diagnostics::sink::{DiagnosticSink, MakeSink};
use crate::diagnostics::{DiagFilter, DiagOpts, Diagnostic, LineIndex};
use crate::smallvec::SmallVec;
use compact_str::CompactString;
use std::io;

/// Test-only arrival counter: proves the resident-pool path was actually taken, so
/// a "parallel == serial" test cannot pass vacuously through the serial branch.
#[cfg(test)]
static PARALLEL_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Written for a diagnostic whose render (or worker setup) panicked.
///
/// The **flat-text** form: correct for every human-oriented format, and wrong
/// for a structured one — [`Renderer::render_internal_error`] exists so JSON can
/// override it, because prose inside a JSON array yields a stream no consumer
/// can parse.
pub(crate) const RENDER_PANIC_PLACEHOLDER: &str = "<internal: diagnostic rendering panicked>\n";

/// Fill `buf` with the renderer's internal-error placeholder, without letting the
/// placeholder itself unwind.
///
/// `render_internal_error` is user-written UI code just like `render_into`, so it
/// can panic too.  In the streaming/dynamic paths a panic here would unwind the
/// worker *before* it fills its slot, and the consumer waits on a slot no other
/// worker can fill (claims are unique) — a permanent hang.  The fallback is the
/// flat-text constant on purpose: a fallback for the fallback must be total, and
/// flat text in a JSON stream is degraded but bounded.
fn fill_internal_error<R: Renderer>(renderer: &R, buf: &mut String) {
    buf.clear();
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        renderer.render_internal_error(buf);
    }))
    .is_ok();
    if !ok {
        buf.clear();
        buf.push_str(RENDER_PANIC_PLACEHOLDER);
    }
}

/// Restores the one-batch permit on unwind, or retires the pool if the permit
/// was already handed to it.
///
/// `submit` consumes the permit by value, so any unwind after that point leaves
/// the pool's in-flight state unknown *and* the permit unobtainable.  Restoring
/// only on the success path meant one recoverable panic made every later
/// parallel batch panic; retiring instead degrades to the serial path, which is
/// always correct.
struct BatchPermit<'a> {
    slot: &'a std::sync::Mutex<Option<crate::pool::BatchToken>>,
    retired: &'a std::sync::atomic::AtomicBool,
    token: Option<crate::pool::BatchToken>,
}

impl<'a> BatchPermit<'a> {
    fn take(
        slot: &'a std::sync::Mutex<Option<crate::pool::BatchToken>>,
        retired: &'a std::sync::atomic::AtomicBool,
    ) -> Self {
        let token = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
        Self {
            slot,
            retired,
            token,
        }
    }
}

impl Drop for BatchPermit<'_> {
    fn drop(&mut self) {
        match self.token.take() {
            Some(token) => *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(token),
            // Handed to `submit` and never returned: never start another
            // parallel batch on this pool.
            None => self
                .retired
                .store(true, std::sync::atomic::Ordering::Relaxed),
        }
    }
}

pub mod stylesheet;

/// Distinct-file set used by the batch summary, shaped for the access
/// pattern: the overwhelmingly common batch draws from 1–2 files, where a
/// handful of `eq` checks against a `SmallVec` beat hashing a `CompactString`
/// and maintaining a bucket array.  Once the distinct-file count crosses the
/// inline capacity the set spills to an `FxHashSet`, preserving the O(1)
/// amortized dedup that large multi-file batches need.
///
/// A `last` memo rides on top of both shapes: diagnostics from one
/// compilation unit arrive grouped by file, so most `insert` calls repeat the
/// previous name and exit on a single string compare — before the linear
/// scan, and before any hashing once spilled.
/// Stable identity of a file in the summary's distinct-file count.
///
/// `FileId` is the real identity: two diagnostics can carry the same display
/// name but different `FileId`s (or the same `FileId` under a renamed view),
/// and the old `CompactString` key conflated the former while allocating for
/// long paths.  A diagnostic with no primary span has no `FileId`, so it falls
/// back to its display name.
#[derive(Clone, PartialEq, Eq, Hash)]
enum FileKey {
    Id(crate::ast::FileId),
    Name(CompactString),
}

struct FilesSeen {
    /// Index of the previous key in `seen` — no second ownership of a
    /// `FileKey::Name` (and no `CompactString` clone) on repeat-heavy batches.
    last: Option<usize>,
    /// The `(FileId, display name)` pair recorded most recently.  Repeat-heavy
    /// batches re-mention one file; the memo turns the common case into a pair
    /// of comparisons instead of a linear scan over every distinct pair (that
    /// scan is `O(diagnostics x files)` across a batch).  Invalidated wherever
    /// `id_names` is mutated from another path (`claim_anonymous`).
    last_pair: Option<(crate::ast::FileId, CompactString)>,
    seen: FilesSeenInner,
}

/// Inline capacity of [`FilesSeenInner::Small`].  Used in both the array type
/// and the spill check below so the two cannot silently drift apart (changing
/// one without the other would move the spill point).
///
/// **Measurement-gated, deliberately not raised.**  A 5-file batch pays the
/// `IndexSet` spill, and the common batch is 1–2 files, so 4 is tighter than ideal;
/// 8 costs ~500 B of stack per `FilesSeen` and only wins for batches of 5–8.  The
/// decision needs a benchmark over batch shapes (1/2/4/8/16 distinct files), not an
/// argument — raising it also changes `FilesSeen`'s size, which the emitter holds per
/// batch.
const INLINE_FILES: usize = 4;

enum FilesSeenInner {
    Small {
        seen: SmallVec<[FileKey; INLINE_FILES]>,
        /// Display names of `FileKey::Id` entries while still inline.
        id_names: SmallVec<[(crate::ast::FileId, CompactString); INLINE_FILES]>,
    },
    Large {
        seen: indexmap::IndexSet<FileKey, rustc_hash::FxBuildHasher>,
        /// Present by construction in Large mode; no `Option` invariant.
        names: NameIndex,
    },
}

struct NameIndex {
    /// Names currently present as anonymous `FileKey::Name` entries.
    anon: rustc_hash::FxHashSet<CompactString>,
    /// Names claimed by at least one `FileKey::Id`.
    claimed: rustc_hash::FxHashSet<CompactString>,
    /// `FileId` → its display aliases.
    ///
    /// Keyed by the integer id so the membership probe is one hash plus (in the
    /// common no-alias case) an empty bucket: the previous
    /// `FxHashSet<(FileId, CompactString)>` had no borrowed-key projection, so it was
    /// probed **linearly** — O(diagnostics × distinct files) tuple compares in exactly
    /// the mode `INLINE_FILES` exists to hand over to, i.e. batches spanning hundreds
    /// of files.
    id_names: rustc_hash::FxHashMap<crate::ast::FileId, SmallVec<[CompactString; 2]>>,
}

/// Test-only count of `last_pair` buffer allocations, per thread.
///
/// Thread-local, not a process-global atomic: the assertion is an exact count, and any
/// other test emitting from another thread would have bumped a shared counter (it did —
/// this became order-dependent the moment two emitter tests ran in parallel).
#[cfg(test)]
thread_local! {
    pub(crate) static LAST_PAIR_ALLOCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl FilesSeen {
    fn new() -> Self {
        FilesSeen {
            last: None,
            last_pair: None,
            seen: FilesSeenInner::Small {
                seen: SmallVec::new(),
                id_names: SmallVec::new(),
            },
        }
    }

    fn key_at(&self, index: usize) -> Option<&FileKey> {
        match &self.seen {
            FilesSeenInner::Small { seen, .. } => seen.get(index),
            FilesSeenInner::Large { seen, .. } => seen.get_index(index),
        }
    }

    fn last_id(&self) -> Option<crate::ast::FileId> {
        match self.last.and_then(|i| self.key_at(i)) {
            Some(FileKey::Id(id)) => Some(*id),
            _ => None,
        }
    }

    fn last_name(&self) -> Option<&str> {
        match self.last.and_then(|i| self.key_at(i)) {
            Some(FileKey::Name(name)) => Some(name.as_str()),
            _ => None,
        }
    }

    fn insert(&mut self, name: &str, id: Option<crate::ast::FileId>) {
        // Grouped batches repeat the same file: check the last index before
        // constructing `FileKey::Name`, which would allocate for long paths.
        //
        // The memo hit must still consult `anonymous_name_seen`: it used to
        // return immediately, so `anon("a.rs") → id(X,"b.rs") → id(X,"a.rs")`
        // left `Name("a.rs")` in `seen` (and, in Large mode, in `names.anon`)
        // while `Id(X)` was registered too — the summary then counted two
        // distinct files for one, and a later span-less repeat of `"a.rs"` took
        // the anonymous path instead of the claimed one.  `claim_anonymous` is
        // idempotent, so the probe costs nothing when there is nothing to claim.
        if let Some(id) = id {
            if self.last_id() == Some(id) {
                if self.anonymous_name_seen(name) {
                    self.claim_anonymous(name, id);
                } else {
                    self.add_id_name(id, name);
                }
                return;
            }
        } else {
            if self.last_name() == Some(name) {
                return;
            }
        }
        // A span-less diagnostic falls back to the display name; if a
        // spanned diagnostic already registered that name, this is the same
        // file for summary purposes.  Check before constructing
        // `FileKey::Name`, which would allocate for long paths.
        if id.is_none() && self.name_seen(name) {
            return;
        }
        let key = match id {
            Some(id) => FileKey::Id(id),
            None => FileKey::Name(CompactString::from(name)),
        };
        // A span-less name was seen first, then a spanned diagnostic with the
        // same display name arrives: claim the anonymous entry in place.  This
        // is essential: leaving `Name` in `seen` would make a later, distinct
        // `FileId` merge into the same anonymous name again.
        if let Some(id) = id
            && self.anonymous_name_seen(name)
        {
            self.claim_anonymous(name, id);
            return;
        }
        let last = self.insert_seen_key(key);
        if let Some(id) = id {
            self.add_id_name(id, name);
        } else {
            self.add_anonymous_name(name);
        }
        self.last = Some(last);
    }

    /// Insert `key` and return its index in the active store.  Moving the key
    /// into the store is what removes the old `last` clone in Small mode.
    fn insert_seen_key(&mut self, key: FileKey) -> usize {
        let mut spill = None;
        let index = match &mut self.seen {
            FilesSeenInner::Small { seen, id_names } => {
                if let Some(pos) = seen.iter().position(|known| *known == key) {
                    pos
                } else if seen.len() < INLINE_FILES {
                    let pos = seen.len();
                    seen.push(key);
                    pos
                } else {
                    let mut set: indexmap::IndexSet<FileKey, rustc_hash::FxBuildHasher> =
                        seen.drain(..).collect();
                    let (index, _) = set.insert_full(key);
                    let names = NameIndex::from_seen(&set, id_names);
                    spill = Some((set, names));
                    index
                }
            }
            FilesSeenInner::Large { seen, .. } => seen.insert_full(key).0,
        };
        if let Some((seen, names)) = spill {
            self.seen = FilesSeenInner::Large { seen, names };
        }
        index
    }

    /// Replace an anonymous `Name(name)` entry with `Id(id)` without changing
    /// the summary length, and update the name indexes.
    fn claim_anonymous(&mut self, name: &str, id: crate::ast::FileId) {
        let old = FileKey::Name(CompactString::from(name));
        let last_index;
        match &mut self.seen {
            FilesSeenInner::Small { seen, id_names } => {
                let Some(pos) = seen.iter().position(|key| *key == old) else {
                    unreachable!("claim_anonymous called without an anonymous entry")
                };
                // Remove first, then insert with de-duplication — exactly what the
                // Large arm gets from `IndexSet::insert_full`.  Overwriting in
                // place counted one file twice when the claiming `FileId` was
                // already present under another display name (the "renamed view"
                // alias case), so the batch summary said "2 files" for one file.
                seen.swap_remove(pos);
                last_index = match seen
                    .iter()
                    .position(|key| matches!(key, FileKey::Id(existing) if *existing == id))
                {
                    Some(existing) => existing,
                    None => {
                        seen.push(FileKey::Id(id));
                        seen.len() - 1
                    }
                };
                if !id_names
                    .iter()
                    .any(|(known_id, known_name)| *known_id == id && known_name == name)
                {
                    id_names.push((id, CompactString::from(name)));
                }
            }
            FilesSeenInner::Large { seen, names } => {
                // Order is irrelevant: `last` is reassigned from `insert_full` and
                // no consumer iterates `seen` in order (only `len`/`get_index`).
                // `swap_remove` may move the tail into `old`'s slot, so
                // `self.last` is untrustworthy between here and the final
                // reassignment.  Do not read last_id/last_name inside this
                // function.
                let _ = seen.swap_remove(&old);
                last_index = seen.insert_full(FileKey::Id(id)).0;
                names.anon.remove(name);
                // Probe before allocating, as `add_id_name` does: a repeated claim of an
                // already-claimed name (the common shape in a repeat-heavy batch) used to
                // build and immediately drop a `CompactString`.
                if !names.claimed.contains(name) {
                    names.claimed.insert(CompactString::from(name));
                }
                // Same probe-before-allocate rule as `add_id_name`: a repeated
                // file must not cost a `CompactString` allocation per
                // diagnostic.
                // Bucket probe first, so the common (already-seen) case allocates
                // nothing: `get` is an integer hash, and the bucket is empty or holds
                // this file's aliases.
                if !names
                    .id_names
                    .get(&id)
                    .is_some_and(|bucket| bucket.iter().any(|known| known == name))
                {
                    names
                        .id_names
                        .entry(id)
                        .or_default()
                        .push(CompactString::from(name));
                }
            }
        }
        self.last = Some(last_index);
    }

    fn add_id_name(&mut self, id: crate::ast::FileId, name: &str) {
        if let Some((last_id, last_name)) = &self.last_pair
            && *last_id == id
            && last_name == name
        {
            return;
        }
        match &mut self.seen {
            FilesSeenInner::Small { id_names, .. } => {
                if !id_names
                    .iter()
                    .any(|(known_id, known_name)| *known_id == id && known_name == name)
                {
                    id_names.push((id, CompactString::from(name)));
                }
            }
            FilesSeenInner::Large { names, .. } => {
                // Probe before allocating: `CompactString::from` heap-allocates
                // once a name outgrows the inline store, and a batch that keeps
                // re-seeing the same files paid that allocation only for
                // `insert` to drop it.  `claimed` is probed borrow-free
                // (`CompactString: Borrow<str>`).
                if !names.claimed.contains(name) {
                    names.claimed.insert(CompactString::from(name));
                }
                // `(FileId, name)` has no borrowed-key projection (a tuple of
                // two borrowed halves is not `Borrow`-able from the owned key),
                // so the membership probe is a linear scan instead of a hash
                // lookup.  That is the right trade here: the set is bounded by
                // the number of *distinct* files in a batch (a handful), while
                // an unconditional `insert` allocated a `CompactString` for
                // every single diagnostic that re-mentioned a file.
                // Bucket probe first, so the common (already-seen) case allocates
                // nothing: `get` is an integer hash, and the bucket is empty or holds
                // this file's aliases.
                if !names
                    .id_names
                    .get(&id)
                    .is_some_and(|bucket| bucket.iter().any(|known| known == name))
                {
                    names
                        .id_names
                        .entry(id)
                        .or_default()
                        .push(CompactString::from(name));
                }
            }
        }
        // Reuse the stored buffer instead of building a fresh `CompactString` on every
        // non-early-return call.  The memo is a one-entry cache, so it is overwritten on
        // essentially every miss — and `CompactString::from` heap-allocates once a name
        // outgrows the inline store, which a batch alternating two file names paid twice
        // per diagnostic.  `clear` + `push_str` keeps the existing capacity, so the
        // steady state allocates nothing.
        match &mut self.last_pair {
            Some((last_id, last_name)) => {
                *last_id = id;
                last_name.clear();
                last_name.push_str(name);
            }
            None => {
                // Test-only: makes "the one-entry memo reuses its buffer" measurable
                // instead of a claim in a comment.
                #[cfg(test)]
                LAST_PAIR_ALLOCS.with(|c| c.set(c.get() + 1));
                self.last_pair = Some((id, CompactString::from(name)));
            }
        }
    }

    fn add_anonymous_name(&mut self, name: &str) {
        if let FilesSeenInner::Large { names, .. } = &mut self.seen {
            names.anon.insert(CompactString::from(name));
        }
    }

    fn anonymous_name_seen(&self, name: &str) -> bool {
        match &self.seen {
            FilesSeenInner::Small { seen, .. } => seen
                .iter()
                .any(|key| matches!(key, FileKey::Name(known) if known == name)),
            FilesSeenInner::Large { names, .. } => names.anon.contains(name),
        }
    }

    fn name_seen(&self, name: &str) -> bool {
        match &self.seen {
            FilesSeenInner::Small { seen, id_names } => {
                id_names.iter().any(|(_, known)| known == name)
                    || seen
                        .iter()
                        .any(|key| matches!(key, FileKey::Name(known) if known == name))
            }
            FilesSeenInner::Large { names, .. } => {
                names.claimed.contains(name) || names.anon.contains(name)
            }
        }
    }

    fn len(&self) -> usize {
        match &self.seen {
            FilesSeenInner::Small { seen, .. } => seen.len(),
            FilesSeenInner::Large { seen, .. } => seen.len(),
        }
    }
}

impl NameIndex {
    fn from_seen(
        seen: &indexmap::IndexSet<FileKey, rustc_hash::FxBuildHasher>,
        id_names: &SmallVec<[(crate::ast::FileId, CompactString); INLINE_FILES]>,
    ) -> Self {
        let mut index = NameIndex {
            anon: rustc_hash::FxHashSet::default(),
            claimed: rustc_hash::FxHashSet::default(),
            id_names: rustc_hash::FxHashMap::default(),
        };
        for key in seen {
            if let FileKey::Name(name) = key {
                index.anon.insert(name.clone());
            }
        }
        for (id, name) in id_names {
            index.claimed.insert(name.clone());
            let bucket = index.id_names.entry(*id).or_default();
            if !bucket.iter().any(|known| known == name) {
                bucket.push(name.clone());
            }
        }
        index
    }
}

/// Non-borrowing per-batch scratch.  `filtered`/`items` hold `&mut
/// Diagnostic` and therefore cannot live here; weights and claim order are
/// plain integers and are reused across emits.
#[derive(Default)]
struct BatchScratch {
    weights: Vec<usize>,
    order: Vec<usize>,
    /// Reused claim slots.
    ///
    /// Every parallel batch used to allocate `n * 64 B` of `Padded<Mutex<_>>` and free
    /// it again (256 KB at n=4096).  Hoisting it is safe because a batch provably ends
    /// with every slot empty — the batch stitch takes each one, and the streaming error
    /// path drains the rest after the workers are joined — which the `debug_assert` in
    /// `prepare_slots` checks.  The stitch reads them through `&mut` (`get_mut`), so
    /// hoisting is a pure win rather than trading an allocation for `n` locks.
    slots: Vec<Padded<std::sync::Mutex<Option<(String, usize)>>>>,
}

impl BatchScratch {
    /// Size and clear the claim slots for a batch of `n`.
    fn prepare_slots(&mut self, n: usize) {
        self.slots
            .resize_with(n, || Padded(std::sync::Mutex::new(None)));
        for slot in &mut self.slots[..n] {
            // A slot that survived its batch means the stitch or the error drain
            // missed it; clearing is correct, but the tripwire is worth having.
            debug_assert!(
                slot.0
                    .get_mut()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_none(),
                "a claim slot survived its batch — the stitch or the stream drain missed it"
            );
            *slot.0.get_mut().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
}

/// Prebuilt line-table entry shared by all workers.
type Prebuilt = crate::diagnostics::render::PreloadedSource;

/// Full-content hash for the cross-batch line-table cache key.
///
/// Single definition lives in [`crate::diagnostics::content_hash_of`], because
/// `LineIndex::new_hashed` fuses the same accumulation into the line scan; two
/// copies of the mixer could silently disagree and turn every miss into a
/// permanent cache miss.  Kept as a named wrapper for the tests below and for
/// callers that only have the text.
#[inline]
fn source_content_hash(s: &str) -> u64 {
    crate::diagnostics::content_hash_of(s.as_bytes())
}

#[cfg(test)]
mod content_hash_tests {
    use super::source_content_hash;

    #[test]
    fn middle_edit_changes_the_hash() {
        let a = format!("{}{}{}", "a".repeat(64), 'b', "a".repeat(64));
        let b = format!("{}{}{}", "a".repeat(64), 'c', "a".repeat(64));
        assert_ne!(source_content_hash(&a), source_content_hash(&b));
    }

    #[test]
    fn equal_content_hashes_equal() {
        let a = format!("{}{}", "x".repeat(37), "yz");
        assert_eq!(source_content_hash(&a), source_content_hash(&a.clone()));
        assert_eq!(source_content_hash(""), source_content_hash(""));
    }

    #[test]
    fn length_and_tail_are_part_of_the_hash() {
        assert_ne!(source_content_hash(""), source_content_hash("\0"));
        assert_ne!(
            source_content_hash("abcdefg"),
            source_content_hash("abcdefgh")
        );
        assert_ne!(
            source_content_hash("abcdefgh"),
            source_content_hash("abcdefgi")
        );
    }
}

/// One batch shared with resident workers.  No lifetime erasure: the
/// diagnostics live behind `Mutex` cells, so Rust's borrow checker proves the
/// exclusive access that the GLM plan obtained with unsafe raw pointers.
struct BatchShared<'a, 'src, R: Renderer> {
    next: std::sync::atomic::AtomicUsize,
    /// The slot the streaming consumer is currently blocked on, published by the
    /// consumer **while it holds that slot's mutex** and read by a worker after it fills.
    ///
    /// The mutex is what makes this sound: the consumer's store happens-before it releases
    /// the lock to wait, and a worker's fill acquires that same lock, so a worker that
    /// filled the awaited slot always observes the matching value and wakes the consumer.
    /// Without the cursor every fill woke the single waiter, costing a futex pair per
    /// out-of-order completion — O(n) wasted wakeups on a weight-skewed batch.
    next_needed: std::sync::atomic::AtomicUsize,
    /// Set when the streaming consumer's sink fails: workers stop claiming instead
    /// of rendering the rest of a batch whose output nothing will read.  `Relaxed` is
    /// enough — the flag is advisory (any interleaving still yields a correct aborted
    /// stream), and only the *streaming* path ever sets it.
    cancelled: std::sync::atomic::AtomicBool,
    /// Claim order (weights descending for throughput, index order for
    /// streaming).
    order: &'a [usize],
    /// The batch's diagnostics themselves.  These used to be
    /// `Vec<Padded<Mutex<&Diagnostic>>>` — one lock/unlock (two atomic RMWs) per
    /// diagnostic plus a 64-byte-per-entry allocation, protecting a *shared
    /// immutable reference*.  Claims are unique (`fetch_add`) and the batch path is
    /// explicitly read-only ("already humanized by the caller: the emitter has no
    /// way to mutate a diagnostic"), so no synchronization is needed at all; the
    /// slice borrow is what makes that a compile-time fact.
    cells: &'a [&'a Diagnostic<'src>],
    /// The batch's claim slots, borrowed from `BatchScratch` (see its field doc).
    slots: &'a [Padded<std::sync::Mutex<Option<(String, usize)>>>],
    slot_cv: std::sync::Condvar,
    renderer: &'a R,
    opts: &'a DiagOpts,
    prebuilt: &'a [Prebuilt],
    /// The worker LRU seed, heaviest-first — deliberately **not** a slice of
    /// `prebuilt`, which is sorted by `(ptr, len)` for the tier-0 binary search (a
    /// prefix of that would be four arbitrary sources, not the four heaviest).
    preload: &'a [Prebuilt],
    /// One buffer pool per resident worker.  A buffer is returned to the pool
    /// of the worker that produced it, so the hot path has no shared mutex.
    string_pools: &'a [std::sync::Mutex<StringPool>],
    streaming: bool,
}

/// Largest buffer the pool keeps unconditionally.
/// Rejected / measured-gated designs for the parallel path.
///
/// Recorded rather than re-derived — the same discipline `render.rs`'s "Decision
/// record" block established.  Each entry names the blocker or the number.
///
/// * **Phase 1 (filter + weight + files) parallelization.** Not done.  The
///   order-dependent `max_errors` gate needs chunk-prefix arithmetic plus a
///   `FilesSeen` union that preserves the distinct-file count and the suppressed
///   tally *bit-identically*, and the payoff only starts around 10^4 diagnostics.  No
///   workload here reaches that; reopen with a rustc-scale batch as an explicit target,
///   and differential-test the merged Phase 1 against the serial one exactly as
///   `human_parallel_matches_serial` does for rendering.
/// * **Folding the prebuild phase into the resident pool.** Not done.  It needs a
///   two-phase `submit` (prebuild jobs, latch, render jobs) under one `BatchToken`,
///   with `BatchPermit` retired across the whole window.  The scoped spawns are now
///   chunked and byte-gated (`prebuild_chunk_len`), so a batch pays `min(P, files)`
///   spawns; reopen only if a profile shows that cost for a frequently-parallel batch
///   (an LSP re-publish loop), not for CLI one-shots.
/// * **One process-wide pool serving humanize + prebuild + render.** Not done.  The
///   blocker is plumbing, not concept: `DiagCtxt` holds no emitter reference, and a
///   global pool needs shutdown semantics for tests.  Worth it only if LSP-style
///   long-lived processes are a first-class target — a product decision.
/// * **Lock-free per-worker string pool.** Rejected with numbers.  The two
///   uncontended lock/unlock pairs cost ~40 ns per diagnostic.  True disjoint
///   ownership is blocked by the borrow checker, not by preference: jobs are
///   `SendJob<'a>` closures handed to a pool that outlives the batch borrow, so
///   `split_at_mut`-style `&mut StringPool` handles cannot be proven across the job
///   indirection.  The sound alternatives (worker-local storage inside the pool's
///   thread lifecycle, or an intrusive freelist) are high effort for that 40 ns.
/// * **A "pre-filtered" flag threaded into `emit_ordered`** (skipping the Phase-1
///   `filter.matches` re-check for the two `&mut` entry points that already filtered).
///   Not done.  The win is one match per diagnostic and only materialises with
///   finder-heavy filters (`--only-files` `memmem` scans) on 10^4-diagnostic batches,
///   while the failure mode of a stale flag is the worst one this subsystem has:
///   *rendering a dropped diagnostic*.  A one-bit invariant guarding the filter contract
///   needs a measurement putting the double check on the critical path; reopen with that
///   bench row, and keep the flag `pub(crate)` set in exactly two places.
/// * **An inlined small-batch source table in `prebuild_entries`** (linear `SmallVec` up
///   to 4 sources, spilling to the hash structures).  Not done.  The common batch has 1-2
///   distinct sources, so the saving is one `FxHashSet` + `FxHashMap` construction per
///   batch — real, but unmeasured, and the rewrite would have to preserve the
///   first-appearance order the cache merge depends on.  Reopen with a `benches/` row on a
///   two-source batch.
const POOL_SMALL_CAP: usize = 64 * 1024;
/// Largest *oversize* buffer the pool keeps at all: one per pool, so a single
/// huge diagnostic cannot evict its size class permanently while worst-case
/// retention stays `POOL_BYTE_BUDGET + POOL_OVERSIZE_CAP` per pool.
const POOL_OVERSIZE_CAP: usize = 1024 * 1024;

/// Byte budget for the *small* buffers one pool retains.
///
/// The old rule kept up to 64 buffers of up to `POOL_SMALL_CAP` (64 KiB) each —
/// up to 4 MiB **per pool**, with one pool per worker, so a finished compile
/// could sit on tens of megabytes of idle RSS.  A byte budget keeps the reuse
/// that matters (a worker re-rendering batches of similar size) while bounding
/// retention at roughly eight typical buffers per pool.
const POOL_BYTE_BUDGET: usize = 512 * 1024;

/// Floor for a single pool's budget after dividing `POOL_BYTE_BUDGET` by the
/// worker count.
///
/// The floor exists because a pool with a few hundred bytes of budget retains
/// nothing worth reusing: reaching a 2 KiB buffer costs the doubling chain (~4
/// allocations), so the budget has to cover at least a few typical buffers.
///
/// **The floor is what makes the aggregate arithmetic exact rather than aspirational.**
/// With `per_pool = max(512 KiB / P, 8 KiB)` the aggregate is
///
/// ```text
/// P <= 64:  512 KiB            (the documented target, exactly)
/// P >  64:  8 KiB * P          (1 MiB at 128 workers)
/// ```
///
/// It used to be 64 KiB, which made the aggregate 4 MiB at 64 workers — 8x the
/// target the surrounding comment claimed, because `max()` hides the multiplication.
/// 8 KiB still retains four typical 2 KiB render buffers, so the reuse the pool exists
/// for survives; only the idle ceiling changes.  `const _` below pins the crossover so
/// a future bump cannot silently reintroduce the 8x.
const POOL_BYTE_BUDGET_MIN: usize = 8 * 1024;
const _: () = assert!(POOL_BYTE_BUDGET % POOL_BYTE_BUDGET_MIN == 0);
const _: () = assert!(POOL_BYTE_BUDGET / POOL_BYTE_BUDGET_MIN == 64);

/// Below this much total source, prebuilding serially beats fanning out.
///
/// A scoped thread costs ~50 µs to spawn and join; a line table for a few hundred
/// bytes is built in ~1 µs.  The gate matters most for the pathological batch: a
/// 1,000-file lint run of small files used to spawn 1,000 threads (and hold 1,000
/// line tables at once) before the render weight gate then chose the serial path
/// anyway.  `prebuild_chunk_len` additionally caps the worker count at
/// `available_parallelism`, so a large batch chunks instead of storming.
const PREBUILD_MIN_TOTAL_BYTES: usize = 64 * 1024;

/// Return `buf` to `pool` under the retained-bytes budget.
///
/// One implementation for both the batch and the streaming path: they carried
/// hand-copied versions of this policy, which is exactly the duplicate that
/// drifts.
/// One resident worker's buffer pool.
///
/// Admission used to recompute `sum(capacities)` and scan for an oversize slot
/// on *every* return — a linear pass per rendered diagnostic on the stitch path.
/// Both quantities only change on push/pop, so they are maintained here.
struct StringPool {
    bufs: Vec<String>,
    /// Sum of `bufs` capacities, oversize slot included (the budget check has
    /// always covered the whole pool).
    retained: usize,
    /// Whether an oversize (`capacity > POOL_SMALL_CAP`) buffer is resident.
    oversize: bool,
    /// Bytes this pool may retain.  Scaled down by the worker count at
    /// construction: the old *per-pool* budget multiplied by the core count, so a
    /// 64-core host kept ~96 MB of idle buffers alive.  Each pool is also used
    /// proportionally less often when there are more of them.
    budget: usize,
}

impl StringPool {
    fn new(budget: usize) -> Self {
        StringPool {
            bufs: Vec::new(),
            retained: 0,
            oversize: false,
            budget,
        }
    }

    fn take(&mut self) -> String {
        match self.bufs.pop() {
            Some(buf) => {
                self.retained -= buf.capacity();
                if buf.capacity() > POOL_SMALL_CAP {
                    self.oversize = false;
                }
                buf
            }
            None => String::new(),
        }
    }
}

fn return_buffer_to_pool(pool: &mut StringPool, buf: String) {
    let cap = buf.capacity();
    if cap > POOL_SMALL_CAP {
        // One oversize slot per pool, never above the absolute cap.
        if cap > POOL_OVERSIZE_CAP || pool.oversize {
            return;
        }
        pool.oversize = true;
    } else if pool.retained.saturating_add(cap) > pool.budget {
        return;
    }
    pool.retained += cap;
    pool.bufs.push(buf);
}

/// Cache-line-padded cell.
///
/// Workers claim in *weight* order, so adjacent indices are routinely written by
/// different cores — the textbook false-sharing shape for a `Vec<Mutex<_>>` whose
/// entries are ~40 bytes.  Memory grows to one line per slot (a 4k-diagnostic
/// batch goes from ~0.2 MB to ~2 MB of mutexes), which is trivial next to the
/// rendered output the batch holds at the same time.
#[repr(align(64))]
struct Padded<T>(T);

/// Resident-pool worker for one batch.  Claims are atomic and each cell is
/// locked exactly for its own render, so weight-model lies only affect claim
/// order, not load balance.
fn worker_render_dynamic<R: Renderer>(
    shared: &BatchShared<'_, '_, R>,
    sources: std::sync::Arc<SharedSources>,
    urls: crate::diagnostics::render::FileUrlMap,
    worker_idx: usize,
) {
    let cx = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        RenderCtx::for_parallel_worker(
            shared.opts,
            None,
            sources,
            urls,
            shared.preload,
            shared.prebuilt,
        )
    }))
    .ok();
    loop {
        // Checked before the claim: after a sink error the main thread breaks and
        // joins, so a claim taken here rendered a diagnostic into a slot nobody reads
        // — potentially the whole remaining batch after an LSP client disconnected.
        if shared.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let claim = shared
            .next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if claim >= shared.order.len() {
            break;
        }
        let orig = shared.order[claim];
        // If anything between the claim and the slot fill unwinds, the consumer's
        // `slot_cv` wait for *this* slot would block forever — nothing else fills it.  The
        // batch path catches that shape with a loud `expect`; streaming had only prose, and
        // the region is not provably panic-free (`fill_internal_error` runs inside it).
        struct FillOnUnwind<'a> {
            slot: &'a std::sync::Mutex<Option<(String, usize)>>,
            cv: Option<&'a std::sync::Condvar>,
            worker_idx: usize,
            armed: bool,
        }
        impl Drop for FillOnUnwind<'_> {
            fn drop(&mut self) {
                if !self.armed {
                    return;
                }
                let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
                if slot.is_none() {
                    *slot = Some((String::from(RENDER_PANIC_PLACEHOLDER), self.worker_idx));
                    drop(slot);
                    if let Some(cv) = self.cv {
                        cv.notify_one();
                    }
                }
            }
        }
        let mut fill_guard = FillOnUnwind {
            slot: &shared.slots[orig].0,
            cv: shared.streaming.then_some(&shared.slot_cv),
            worker_idx,
            armed: true,
        };
        // `worker_idx` is already reduced modulo the pool count by the
        // dispatcher (`worker_idx % pool_count` at the submit site) and
        // `shared.string_pools.len() == pool_count`, so this index is in range
        // by construction — the repeated `%` was a division per diagnostic.
        debug_assert!(worker_idx < shared.string_pools.len());
        let mut buf = shared.string_pools[worker_idx]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        buf.clear();
        let buf = match &cx {
            Some(cx) => {
                let diag: &Diagnostic<'_> = shared.cells[orig];
                let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // Already humanized by the caller: batches are rendered from
                    // `&Diagnostic`, so the emitter has no way to mutate a
                    // diagnostic (and therefore no way to desynchronise a
                    // collector's tallies).
                    shared.renderer.render_into(diag, cx, &mut buf);
                }));
                if rendered.is_err() {
                    fill_internal_error(&*shared.renderer, &mut buf);
                }
                buf
            }
            // Setup panic: keep draining the queue so slots/latch cannot stall.
            None => {
                fill_internal_error(&*shared.renderer, &mut buf);
                buf
            }
        };
        *shared.slots[orig]
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some((buf, worker_idx));
        fill_guard.armed = false;
        if shared.streaming {
            // Streaming has exactly one waiter (the main consumer), and it is blocked on
            // one specific slot: waking it for any other completion just costs a futex
            // wake + a re-check + a futex wait.
            if orig
                == shared
                    .next_needed
                    .load(std::sync::atomic::Ordering::Relaxed)
            {
                shared.slot_cv.notify_one();
            }
        }
    }
}

/// Composes a renderer with a sink factory and emits batches of diagnostics.
pub struct Emitter<R, M> {
    renderer: R,
    make_sink: M,
    opts: DiagOpts,
    /// Filter view built once from `opts` in [`Emitter::new`] and reused by
    /// every emit path, so a caller looping on [`Emitter::emit`] does not
    /// rebuild the filter's hash sets per diagnostic.
    filter: DiagFilter,
    /// Render state hoisted out of the per-call/per-batch `RenderCtx`: the
    /// `file://` URL cache and the grow-once section scratch (see
    /// `SharedRenderState`).  The per-source line-index cache is deliberately
    /// *not* hoisted — its `(ptr, len)` keys are only sound while the sources
    /// are alive, which a batch ctx guarantees by being dropped with the
    /// batch; a hoisted key could alias a reused allocation in a later call.
    shared: SharedRenderState,
    /// Host parallelism, resolved once in [`Emitter::new`].  `available_parallelism`
    /// is a syscall and host affinity effectively does not change over a
    /// compiler process's lifetime, so a per-batch query is pure overhead.
    parallelism: usize,
    /// Resident render pool.  Lazily built on the first parallel batch, so a
    /// CLI run that never reaches the threshold pays no threads.
    pool: std::sync::OnceLock<crate::pool::WorkerPool>,
    /// One-batch-at-a-time permit for `pool`.  `submit` demands it by value, so a
    /// second submit before `wait` is a compile error rather than a release-mode
    /// UB path (see `pool::BatchToken`).
    pool_token: std::sync::Mutex<Option<crate::pool::BatchToken>>,
    /// Set when a panic unwound through a batch *after* the permit was handed to
    /// the pool: its in-flight state is then unknown, so parallel rendering is
    /// retired for the rest of this emitter's life and every later batch renders
    /// serially.  A recoverable panic must not become a panic on every subsequent
    /// batch, which is what `expect`ing a swallowed permit produced.
    pool_retired: std::sync::atomic::AtomicBool,
    /// Reused across batch emits: weights and claim order.
    batch: BatchScratch,
    /// One reusable `String` pool per resident worker.  Workers only ever push
    /// into and pop from their own pool, so the old single hot mutex is gone.
    string_pools: Vec<std::sync::Mutex<StringPool>>,
    /// Cross-batch line tables keyed by `(FileId, full-content hash)`, newest
    /// at the front with FIFO eviction past 8 entries (a recent-files window,
    /// not a timestamp LRU — the per-batch `line_cache` is that).
    /// `Prebuilt` owns its `Arc` tables, so no source lifetime is captured.
    /// Keyed by `(FileId, content hash, content length)`: the hash is
    /// non-cryptographic (64-bit), so two *different* contents of one file could
    /// in principle collide and serve a stale line table — wrong rendering.
    /// Length is free to carry and makes that require a same-length collision.
    file_index_cache: std::cell::RefCell<
        crate::smallvec::SmallVec<[(crate::ast::FileId, u64, usize, Prebuilt); 8]>,
    >,
}

impl<R: Renderer, M: MakeSink> Emitter<R, M> {
    pub fn new(renderer: R, make_sink: M, opts: DiagOpts) -> Self {
        let filter = DiagFilter::from_opts(&opts);
        let parallelism = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            renderer,
            make_sink,
            opts,
            filter,
            shared: SharedRenderState::new(),
            parallelism,
            pool: std::sync::OnceLock::new(),
            pool_token: std::sync::Mutex::new(None),
            pool_retired: std::sync::atomic::AtomicBool::new(false),
            batch: BatchScratch::default(),
            string_pools: {
                // Keep the *aggregate* resident budget roughly constant instead of
                // growing it by a factor of `parallelism`.
                let per_pool = (POOL_BYTE_BUDGET / parallelism.max(1)).max(POOL_BYTE_BUDGET_MIN);
                (0..parallelism)
                    .map(|_| std::sync::Mutex::new(StringPool::new(per_pool)))
                    .collect()
            },
            file_index_cache: std::cell::RefCell::new(crate::smallvec::SmallVec::new()),
        }
    }

    /// Force a worker count (the test-reachable shape of the `available_parallelism`
    /// default).
    ///
    /// The parallel path is where the render-panic containment, the pool-retention
    /// invariant and the claim-order stitch live, and a 1-core runner would skip all
    /// three — the tests used to `return` early instead of exercising them.  The pools are
    /// rebuilt with the same budget formula `new` uses, so the per-worker pool slot stays
    /// in range (which `worker_jobs` now asserts in debug).
    #[cfg(test)]
    pub(crate) fn with_parallelism(mut self, workers: usize) -> Self {
        let workers = workers.max(1);
        let per_pool = (POOL_BYTE_BUDGET / workers).max(POOL_BYTE_BUDGET_MIN);
        self.parallelism = workers;
        self.string_pools = (0..workers)
            .map(|_| std::sync::Mutex::new(StringPool::new(per_pool)))
            .collect();
        self
    }

    /// Line table for one source, reusing the cross-batch `(FileId, content
    /// hash)` cache.
    ///
    /// [`Self::emit`] used to pass no prebuilt table at all, so an LSP loop
    /// re-emitting diagnostics for one file rebuilt the whole line index per
    /// diagnostic — an O(source) scan per keystroke.  The batch path already
    /// fills this cache; a hit still needs `ptr`/`len`/`fingerprint` refreshed
    /// from *this* call's source, because the downstream lookups match prebuilt
    /// tables by address.
    fn prebuilt_for_single(
        &self,
        src: &str,
        file_id: Option<crate::ast::FileId>,
    ) -> SmallVec<[PreloadedSource; 1]> {
        let ptr = src.as_ptr() as usize;
        let len = src.len();
        let fingerprint = source_fingerprint(src);
        // The hash is only meaningful when there is a `FileId` to key on *and* the
        // cache holds something to compare against; computing it unconditionally
        // put a full pass over the source on the single-emit (LSP) path for
        // diagnostics that could never hit.
        //
        // One scan, not two: the same-`(file, len)` entries are collected as the
        // candidates, and that single borrow answers both "is hashing worth it" and
        // "which entry matches".  The length is part of the cache key and costs
        // nothing to compare, while `source_content_hash` is a full O(n) pass; nearly
        // every real edit changes the length, so the hash is computed only when a
        // same-length candidate exists.  (`PreloadedSource` is `Arc`-backed, so the
        // clones are refcount bumps.)
        let mut candidates: SmallVec<[(usize, u64, PreloadedSource); 2]> = SmallVec::new();
        if let Some(id) = file_id {
            for (idx, (f, ch, n, p)) in self.file_index_cache.borrow().iter().enumerate() {
                if *f == id && *n == len {
                    candidates.push((idx, *ch, p.clone()));
                }
            }
        }
        let mut hash: Option<u64> = None;
        let cached = if candidates.is_empty() {
            None
        } else {
            let h = source_content_hash(src);
            hash = Some(h);
            match candidates.into_iter().find(|(_, ch, _)| *ch == h) {
                Some((idx, _, p)) => {
                    // Promote the hit to the front.  The window is documented as "recent
                    // files", but a FIFO never reordered on a hit, so an LSP session touching
                    // more than 8 files evicted the very file it re-emits on every keystroke —
                    // and the miss costs a full `LineIndex::new` scan of that source.
                    let mut cache = self.file_index_cache.borrow_mut();
                    if idx < cache.len() {
                        let hit = cache.remove(idx);
                        cache.insert(0, hit);
                    }
                    Some(p)
                }
                None => None,
            }
        };
        let entry = match cached {
            // Carried from the cached entry **explicitly**: `starts`/`ascii`/
            // `char_prefixes`/`generation` all describe one `LineIndex` instance, and a
            // hash+len hit proves the content is byte-identical, so reusing them —
            // including the generation, which is that instance's id — is what the
            // cache is for.  `ptr`/`len`/`fingerprint` must come from *this* call's
            // buffer, which is why they are re-stamped.  The previous `..p` made the
            // carried set invisible, so a future field would be inherited by accident.
            Some(p) => PreloadedSource {
                ptr,
                len,
                fingerprint,
                starts: std::sync::Arc::clone(&p.starts),
                ascii: p.ascii,
                char_prefixes: p.char_prefixes.clone(),
                generation: p.generation,
            },
            None => {
                // Fuse the line table and the cache key into one pass, exactly as
                // the batch path already does: a cold single emit used to scan the
                // source twice (`LineIndex::new` + `source_content_hash`).
                let (idx, computed) = match (file_id, hash) {
                    (Some(_), None) => {
                        let (idx, h) = LineIndex::new_hashed(src);
                        (idx, Some(h))
                    }
                    _ => (LineIndex::new(src), None),
                };
                let entry = PreloadedSource {
                    ptr,
                    len,
                    starts: std::sync::Arc::clone(idx.starts_arc()),
                    ascii: idx.ascii(),
                    char_prefixes: idx.char_prefixes_arc().clone(),
                    generation: idx.generation(),
                    fingerprint,
                };
                if let Some(id) = file_id {
                    // Same bound and eviction as the batch path: newest at the
                    // front, oldest dropped from the back (FIFO by insertion —
                    // the *render* path's `line_cache` is the real LRU, keyed by
                    // a timestamp; this one is a small recent-files window).
                    let mut cache = self.file_index_cache.borrow_mut();
                    if cache.len() >= 8 {
                        cache.pop();
                    }
                    let h = hash
                        .or(computed)
                        .unwrap_or_else(|| source_content_hash(src));
                    cache.insert(0, (id, h, len, entry.clone()));
                }
                entry
            }
        };
        let mut out: SmallVec<[PreloadedSource; 1]> = SmallVec::new();
        out.push(entry);
        out
    }

    /// Emit a single diagnostic through a fresh sink.
    ///
    /// This is the render boundary: `humanize()` materializes the
    /// structured `kind` (message/labels/help/suggestions) against the
    /// *current* i18n `Lang` right before rendering, so a diagnostic's
    /// language is never frozen at construction time.
    ///
    /// The sink is built fresh per call (that is the contract — a sink never
    /// outlives the batch method), but the reusable render state behind the
    /// `RenderCtx` — the `file://` URL cache and the grow-once section
    /// scratch — is hoisted onto the emitter, so an LSP-style loop does not
    /// pay a 16 KB scratch allocation or a fresh `canonicalize` per
    /// diagnostic.  Batch callers should still prefer [`Self::emit_all_iter`],
    /// which also shares the per-source line-index cache across the batch
    /// (newest-first, FIFO eviction — see `file_index_cache`).
    ///
    /// Note: `max_errors` is a *batch* budget — [`Self::emit`] enforces no
    /// cross-call limit, so an LSP-style loop must track its own if it needs
    /// one.  A filtered-out diagnostic returns before `begin`/`end`; batch
    /// paths always run them, so an empty-document renderer must use a batch
    /// API when it needs the header/footer for an empty set.  A panic during
    /// `humanize` **or** `render_into` becomes the renderer's internal-error
    /// placeholder (see [`Renderer::render_internal_error`]) — the same policy
    /// as the batch paths, so one broken diagnostic never aborts an LSP batch.
    pub fn emit(&mut self, diag: &mut Diagnostic<'_>) -> io::Result<()> {
        if !self.filter.matches(diag) {
            return Ok(());
        }
        // `humanize` must run *before* the prebuilt lookup: it derives the primary
        // spans from the kind's labels, and `prebuilt_for_single` keys the line
        // table on the first span's file.  It is guarded (rather than simply
        // moved inside the render closure) because the documented policy is that
        // a `humanize` panic becomes the placeholder instead of escaping to the
        // caller on the LSP path.
        let mut humanized = true;
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| diag.humanize())).is_err() {
            humanized = false;
        }
        let mut sink = self.make_sink.make();
        self.renderer.begin(&mut sink)?;
        // Reuse the line table this emitter already built for the file, when it
        // has one: an LSP loop emits diagnostics for the same file repeatedly,
        // and the table is content-addressed, so a rebuild is pure waste.
        // A failed `humanize` already forces the internal-error placeholder, so a
        // line table built for it would be pure waste (the old order did exactly
        // that).
        // Span-less diagnostics are never line-queried, and they cannot even enter the
        // file-keyed cache (`file_id == None`), so building their table was pure
        // O(source) waste per emit — the batch path skips them for exactly this reason.
        let has_primary = diag.spans.primary_ref().first().is_some();
        let mut prebuilt = if humanized && has_primary {
            // `primary_ref()`, the same accessor the batch paths use: `spans()` is the
            // full collection, so keying the *file* cache off it could disagree with
            // the batch path for the same diagnostic and miss across the boundary.
            diag.source()
                .map(|src| {
                    self.prebuilt_for_single(src, diag.spans.primary_ref().first().map(|s| s.file))
                })
                .unwrap_or_default()
        } else {
            SmallVec::new()
        };
        // `RenderCtx` binary-searches this slice (see `line_index`), and the
        // tier-0 hit path must not depend on the prebuild's emission order.
        // The single path always has exactly one entry (`prebuilt_for_single`), so this
        // is a one-element no-op there; it is load-bearing for the batch path, whose
        // tier-0 hit ordering must not depend on the prebuild's emission order.
        prebuilt.sort_unstable_by_key(|e| (e.ptr, e.len));
        let cx = if prebuilt.is_empty() {
            RenderCtx::hoisted(&self.opts, None, &self.shared)
        } else {
            RenderCtx::hoisted_with_prebuilt(&self.opts, None, &self.shared, &prebuilt)
        };
        // Reuse the hoisted scratch buffer: the rendered text is copied into
        // the sink and dropped, so a fresh `String` per diagnostic is a pure
        // allocation/free in the documented LSP re-emit loop.
        let mut buf = self.shared.render_buf.borrow_mut();
        buf.clear();
        let rendered = humanized
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Idempotent on the already-humanized path (the latch makes it a
                // `u8` compare); kept so the render closure is self-contained.
                diag.humanize();
                self.renderer.render_into(diag, &cx, &mut buf);
            }))
            .is_ok();
        if !rendered {
            fill_internal_error(&self.renderer, &mut buf);
        }
        sink.write_str(&buf)?;
        self.renderer.end(&mut sink)?;
        sink.flush()
    }

    /// Emit a batch of diagnostics plus a summary through a fresh sink.
    ///
    /// Honours [`DiagOpts::reverse_errors`] exactly like [`Self::emit_all_iter`].
    ///
    /// **`reverse_errors` + `max_errors`:** the gate admits the first `N`
    /// errors *in render order*, so reversing the batch keeps the last `N`
    /// errors of the original order and suppresses the first ones.  This is
    /// deliberate (the budget follows what the user sees first) but is spelled
    /// out here because "same set, reversed" would otherwise be the natural
    /// reading.
    /// The filter this emitter applies before rendering.
    ///
    /// Exposed so the collector can humanize only the diagnostics that will actually be
    /// rendered: filtering is a pure predicate over already-set fields (`code`,
    /// `file_name`, `category`), so it can run before humanization, and a build with a
    /// suppression list or a `max_errors` cap used to humanize every dropped diagnostic.
    pub(crate) fn filter(&self) -> &DiagFilter {
        &self.filter
    }

    pub fn emit_all(&mut self, diags: &[Diagnostic<'_>]) -> io::Result<()> {
        let mut sink = self.make_sink.make();
        self.emit_ordered(diags.iter(), &mut sink, false)
    }

    /// Dispatch a double-ended diagnostic iterator through the batch loop in
    /// the configured order.  One home for the reverse/non-reverse split: the
    /// iterator and its reverse are distinct concrete types, so the branch
    /// cannot be erased, but it need not be repeated at every entry point.
    fn emit_ordered<'r, 'src, S, I>(
        &mut self,
        iter: I,
        sink: &mut S,
        streaming: bool,
    ) -> io::Result<()>
    where
        S: DiagnosticSink,
        'src: 'r,
        I: IntoIterator<Item = &'r Diagnostic<'src>>,
        I::IntoIter: DoubleEndedIterator,
    {
        let iter = iter.into_iter();
        if self.opts.reverse_errors {
            self.emit_batch_iter(iter.rev(), sink, streaming)
        } else {
            self.emit_batch_iter(iter, sink, streaming)
        }
    }

    /// Emit a batch from any double-ended iterator of borrowed diagnostics —
    /// avoids cloning a `DiagCtxt`'s contents into a `Vec` just to pass a
    /// slice to [`Self::emit_all`].
    pub fn emit_all_iter<'r, 'src, I>(&mut self, diags: I) -> io::Result<()>
    where
        'src: 'r,
        I: IntoIterator<Item = &'r Diagnostic<'src>>,
        I::IntoIter: DoubleEndedIterator,
    {
        let mut sink = self.make_sink.make();
        self.emit_ordered(diags, &mut sink, false)
    }

    /// Streaming batch emit: write diagnostics to the sink in batch order as
    /// soon as their render completes.  Intended for interactive/LSP sinks
    /// where first-token latency matters; ordinary compiler batches should
    /// use [`Self::emit_all_iter`].
    pub fn emit_all_streaming<'r, 'src, I>(&mut self, diags: I) -> io::Result<()>
    where
        'src: 'r,
        I: IntoIterator<Item = &'r Diagnostic<'src>>,
        I::IntoIter: DoubleEndedIterator,
    {
        let mut sink = self.make_sink.make();
        self.emit_ordered(diags, &mut sink, true)
    }

    /// Streaming emit to a caller-provided sink, mirroring
    /// [`Self::emit_all_to`] but consuming slots as they complete.
    /// Humanize the filter-kept diagnostics, in parallel above the collector's threshold.
    ///
    /// Humanization is the last **serial** stage in front of the parallel renderer: at 10^4
    /// diagnostics × ~1-5 µs (Fluent rendering, label derivation, suggestion mapping) that is
    /// 10-50 ms of serial work, the same Amdahl term `DiagCtxt::humanize_pending` already
    /// removes for the collector path.  These entry points took `&mut [Diagnostic]` and
    /// therefore bypassed it.
    ///
    /// Sound for the same reasons the collector documents: `Lang` is a process-global atomic,
    /// the chunks are disjoint `&mut`, every per-diagnostic cache is diagnostic-local, and
    /// `DiagFilter` is `Sync` (a bitset, an `FxHashSet` and `memmem::Finder`s).  Output is
    /// untouched because each `humanize` writes only its own diagnostic.
    ///
    /// The containment from `humanize_contained` is what makes the fan-out safe: a panicking
    /// diagnostic costs one placeholder, and `catch_unwind` per diagnostic also keeps a worker
    /// from taking the whole scope down.
    fn humanize_kept(diags: &mut [Diagnostic<'_>], filter: &DiagFilter) {
        const PARALLEL_HUMANIZE_MIN: usize = 64;
        // `available_parallelism` is a `sched_getaffinity` syscall on Linux; a process's
        // affinity mask does not change under it, so the answer is cached like the collector's.
        static AVAILABLE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let threads = if diags.len() >= PARALLEL_HUMANIZE_MIN {
            (*AVAILABLE.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get())))
                .min(diags.len() / PARALLEL_HUMANIZE_MIN)
                .max(1)
        } else {
            1
        };
        if threads > 1 {
            let chunk = diags.len().div_ceil(threads);
            std::thread::scope(|scope| {
                for slice in diags.chunks_mut(chunk) {
                    scope.spawn(move || {
                        for diag in slice {
                            if filter.matches(diag) {
                                diag.humanize_contained();
                            }
                        }
                    });
                }
            });
        } else {
            for diag in diags.iter_mut() {
                if filter.matches(diag) {
                    diag.humanize_contained();
                }
            }
        }
    }

    pub fn emit_all_streaming_to<S: DiagnosticSink>(
        &mut self,
        diags: &mut [Diagnostic<'_>],
        sink: &mut S,
    ) -> io::Result<()> {
        // Same contract and same reason as `emit_all_to`.
        // Humanize only what the filter keeps.  The filter reads pre-humanize data
        // (the Phase-1 invariant), so this cannot change which diagnostics survive —
        // it stops the dropped ones paying Fluent rendering, label derivation and
        // their allocations.  Same policy as `DiagCtxt::humanize_pending_for`, which
        // this path bypassed because it takes `&mut [Diagnostic]` directly.
        Self::humanize_kept(diags, &self.filter);
        self.emit_ordered(diags.iter(), sink, true)
    }

    /// Escape hatch: write to a caller-provided sink without a `MakeSink`.
    ///
    /// Honours [`DiagOpts::reverse_errors`] exactly like [`Self::emit_all_iter`].
    pub fn emit_all_to<S: DiagnosticSink>(
        &mut self,
        diags: &mut [Diagnostic<'_>],
        sink: &mut S,
    ) -> io::Result<()> {
        // Self-sufficient, like `emit`: the lazy-humanization policy requires the
        // caller to run `DiagCtxt::humanize_pending` first, and taking `&mut` while
        // *not* using it made a contract violation silently render pre-materialization
        // spans and an empty message.  Idempotent — the latch makes this a `bool`
        // compare for already-humanized input.
        // Humanize only what the filter keeps.  The filter reads pre-humanize data
        // (the Phase-1 invariant), so this cannot change which diagnostics survive —
        // it stops the dropped ones paying Fluent rendering, label derivation and
        // their allocations.  Same policy as `DiagCtxt::humanize_pending_for`, which
        // this path bypassed because it takes `&mut [Diagnostic]` directly.
        Self::humanize_kept(diags, &self.filter);
        self.emit_ordered(diags.iter(), sink, false)
    }

    fn emit_batch_iter<'r, 'src, S, I>(
        &mut self,
        iter: I,
        sink: &mut S,
        streaming: bool,
    ) -> io::Result<()>
    where
        S: DiagnosticSink,
        'src: 'r,
        I: Iterator<Item = &'r Diagnostic<'src>>,
    {
        self.renderer.begin(sink)?;

        // Count *every* diagnostic up front — including those `max_errors`
        // drops — so the summary reflects the true total.  This is a
        // deliberate divergence from rustc, whose summary is bounded by
        // `--error-limit`; the suppressed count is reported separately via
        // `render_suppressed_note`.
        let mut errors = 0usize;
        let mut warnings = 0usize;
        let mut suppressed = 0usize;
        let mut emitted_errors = 0usize;
        // Owned names: the batch loop may not borrow `&str` views from the
        // diagnostics, because the `&mut` entry points humanize before calling in (and
        // a future mid-iteration humanize would invalidate the borrows).  This comment
        // used to claim the loop *did* humanize on `&mut`; it never did — the `&mut`
        // use now lives in `emit_all_to`/`emit_all_streaming_to`.  `FilesSeen`
        // starts as a `SmallVec` linear scan (1–2 files is the common
        // case, where a few `eq` checks beat hashing) and spills to an
        // `FxHashSet` only once the distinct-file count crosses the inline
        // capacity — preserving the O(1) amortized dedup of large
        // multi-file batches without taxing the small ones.
        let mut files = FilesSeen::new();

        // ── Phase 1: filter + count.  Serial: the `max_errors` gate is
        //    order-dependent and trivially O(n) per diagnostic. ────────────
        //
        // Humanization contract for this loop: it renders `&Diagnostic` and never
        // mutates, so every diagnostic reaching here must already be humanized.  The
        // `&mut` entry points (`emit_all_to`, `emit_all_streaming_to`) now do that
        // themselves; the borrowed ones (`emit_all`, `emit_all_iter`) require
        // `DiagCtxt::humanize_pending` first, which the assert inside the loop
        // enforces in debug builds.  Consequences for `max_errors`: a suppressed
        // diagnostic is never rendered, so it pays no renderer work — the property
        // that was previously only claimed.
        // Pre-size from the iterator's lower bound.  Distinguish an exact
        // hint (`slice::IterMut`, `vec::IntoIter`, and their `.rev()`s are
        // `ExactSizeIterator`) from a loose one: for an exact hint the cap is
        // exact and the growth chain disappears; for a loose hint the
        // `.min(64)` cap keeps a pathological over-estimate from
        // over-allocating.
        let (lo, hi) = iter.size_hint();
        // Trust a usable upper bound (Map/Filter often provide one), but keep
        // a hard cap so a pathological hint cannot over-allocate.
        const HARD_CAP: usize = 4096;
        let exact = hi == Some(lo);
        let cap = if exact {
            lo
        } else {
            hi.unwrap_or(lo.min(64)).min(HARD_CAP)
        };
        let mut batch = std::mem::take(&mut self.batch);
        batch.weights.clear();
        batch.weights.reserve(cap);
        let mut filtered: SmallVec<[&'r Diagnostic<'src>; 64]> = SmallVec::with_capacity(cap);
        // Cheap render-cost proxy summed over the batch.  The parallel
        // decision gates on this as well as the count: 64 one-line notes cost
        // less to render than the ~50 µs/thread spawn, so they stay serial.
        let mut total_weight = 0usize;
        // Per-diagnostic weights, parallel to `filtered`.  Collected once here
        // so the parallel split consumes this single model instead of
        // recomputing the closure (and risking a drift between two copies).

        // The distinct-file set feeds only the summary line, which is gated
        // on `errors > 0 || warnings > 0` — so tracking files for a batch of
        // only Info/Note diagnostics (or an all-filtered one) is pure waste:
        // a `CompactString` per file plus the set machinery for a count that
        // is never printed.  The insert is therefore gated on the diagnostic
        // actually contributing to the counters.  Suppressed-by-`max_errors`
        // diagnostics still count (`errors` is incremented before the gate),
        // so they still track their file and the printed file count stays
        // exactly what it was.
        let track_files = !self.opts.quiet;

        // Invariant: `DiagFilter::matches`, `is_error` and `is_warning` inspect only
        // pre-humanize data (code, error category, file id/name, level) —
        // never the message — so evaluating them before `humanize()` below is
        // sound and avoids Fluent lookups for filtered-out diagnostics.  If a
        // future filter predicate reads `message()`, it must move after
        // `humanize()` in Phase 2.
        for diag in iter {
            // The silent-wrong-output class this closes: an un-humanized diagnostic
            // keys `FilesSeen`/the line tables on *constructor-default* spans and
            // renders `message_unchecked()` against an unmaterialized kind — wrong file
            // attribution, wrong line numbers, possibly an empty message, with no
            // signal at all.  Debug builds abort here instead; release behaviour is
            // unchanged (and the `&mut` entry points make it moot for the common path).
            // The check applies to **survivors only**: a diagnostic the filter rejects is
            // never rendered, so requiring it to be humanized was a precondition with no
            // purpose — and it became a hard failure once `DiagCtxt::emit` started
            // humanizing only what the filter keeps (which is the whole point of that
            // change: a `--only-files` run must not pay a Fluent render per rejected
            // diagnostic).
            if !self.filter.matches(diag) {
                continue;
            }
            debug_assert!(
                diag.is_humanized(),
                "batch entry points require humanized diagnostics — run `DiagCtxt::humanize_pending` first (the `&mut` variants do it for you)"
            );
            let is_warning = diag.is_warning();
            let is_error = diag.is_error();
            // `warn_is_error` promotes warnings into the error budget: they
            // count toward the summary and the `max_errors` gate exactly like
            // real errors, but are still rendered at their original level.
            // A promoted warning counts as an error *only* — not also as a
            // warning — so the summary never double-counts it, and the
            // suppressed / non-suppressed paths agree (the `continue` below
            // skips the warning counter too).
            let promoted = is_warning && self.opts.warn_is_error;
            if track_files && (is_error || is_warning) {
                files.insert(
                    diag.file_name(),
                    diag.spans.primary_ref().first().map(|s| s.file),
                );
            }
            if is_error || promoted {
                errors += 1;
                if let Some(limit) = self.opts.max_errors
                    && emitted_errors >= limit
                {
                    suppressed += 1;
                    continue;
                }
                emitted_errors += 1;
            }
            if is_warning && !promoted {
                warnings += 1;
            }
            // `kind` is retained before `humanize()`, so this is a pre-humanize
            // cost proxy.  Labels/suggestions derived *from* the kind are not
            // materialized yet; the kind term plus child/related counts keep
            // expensive diagnostics above the parallel weight threshold.
            let weight = 1
                + diag.labels.len()
                + diag.suggestions.len()
                + diag.spans.primary_ref().len()
                + usize::from(diag.kind().is_some()) * 4
                + diag.children.len() * 2
                + diag.related_errors.len() * 2
                + diag.message_unchecked().len() / 32;
            total_weight += weight;
            batch.weights.push(weight);
            filtered.push(diag);
        }
        let num_renderable = filtered.len();

        // ── Phase 2: render.  Parallel when the batch is large enough that
        //    thread-spawn overhead (~50 µs/thread) is amortised by the render
        //    work itself.  Small batches and single-core hosts stay serial.
        //    (`num_renderable` was tallied inside the Phase 1 loop — no extra
        //    O(n) pass here.)
        const PARALLEL_THRESHOLD: usize = 64;
        // A batch of trivial diagnostics (bare notes) can render faster than
        // the spawn/join cost even above the count threshold; require enough
        // total weight that the work actually amortises the threads.
        const WEIGHT_THRESHOLD: usize = 256;
        // Keep the worker count bounded so each worker handles at least this many
        // diagnostics on average; dynamic claiming eliminates fixed chunks.
        const MIN_CHUNK: usize = 16;
        let num_threads = self.parallelism;
        // Cap the spawn count: at the 64-diagnostic threshold this yields
        // 4 threads (≥16 diags each).  Without the cap, a 128-core host
        // would spawn 64 threads for one diagnostic each — ~3 ms of
        // `spawn`/`join` for work that barely amortises.
        let num_threads = num_threads.min((num_renderable / MIN_CHUNK).max(1));
        let pool_usable = !self.pool_retired.load(std::sync::atomic::Ordering::Relaxed);
        let render_result = if streaming
            && pool_usable
            && num_renderable >= PARALLEL_THRESHOLD
            && total_weight >= WEIGHT_THRESHOLD
            && num_threads > 1
        {
            self.render_streaming(filtered.as_slice(), &mut batch, sink, num_threads)
        } else if pool_usable
            && num_renderable >= PARALLEL_THRESHOLD
            && total_weight >= WEIGHT_THRESHOLD
            && num_threads > 1
        {
            self.render_dynamic(filtered.as_slice(), &mut batch, sink, num_threads)
        } else {
            // Opportunistic content-keyed reuse, on the serial path too.
            //
            // This used to be gated on the cache already being non-empty, which
            // made the mechanism dead for workloads that never reach the parallel
            // threshold and never call `emit` (an LSP re-publish loop over a small
            // batch): it stayed cold forever and rescanned every source per batch.
            // The gate's premise was also wrong — the cold branch below uses the
            // fused `LineIndex::new_hashed`, so prebuilding is scan-neutral and
            // strictly better once a batch touches one file several times.
            // Entries only: this path never shares them across threads, so the
            // `SharedSources` wrapper (one `Arc` allocation + a refcount per entry)
            // was pure waste on the most common branch.
            let mut prebuilt = self.prebuild_entries(&filtered[..], &batch.weights);
            // Sorted for the tier-0 `binary_search` in `RenderCtx::line_index`.
            // The single path always has exactly one entry (`prebuilt_for_single`), so this
            // is a one-element no-op there; it is load-bearing for the batch path, whose
            // tier-0 hit ordering must not depend on the prebuild's emission order.
            prebuilt.sort_unstable_by_key(|e| (e.ptr, e.len));
            let cx = RenderCtx::hoisted_with_prebuilt(&self.opts, None, &self.shared, &prebuilt);
            self.render_sequential(filtered.as_slice(), sink, &cx)
        };
        self.batch = batch;
        render_result?;

        // ── Phase 3: trailer.  Always single-threaded. ─────────────────────
        // The suppressed note is a human-facing trailer: like the summary, it
        // is gated on `quiet`, which `HumanRenderer`'s note would otherwise
        // leak through (machine renderers return "" here anyway).
        if suppressed > 0 && !self.opts.quiet {
            // Into the batch's reused render buffer: the note is copied into the sink and
            // dropped, so a fresh `String` per batch was a pure allocation/free in the
            // documented LSP re-emit loop.  `render_suppressed_note_into` already existed; this
            // path just never called it.
            let mut note = self.shared.render_buf.borrow_mut();
            note.clear();
            self.renderer
                .render_suppressed_note_into(suppressed, &mut note);
            sink.write_str(&note)?;
        }
        if !self.opts.quiet && (errors > 0 || warnings > 0) {
            sink.write_str(
                &self
                    .renderer
                    .render_summary_full(errors, warnings, files.len()),
            )?;
        }

        self.renderer.end(sink)?;
        sink.flush()
    }

    /// Single-threaded batch renderer — the fallback for small batches and
    /// single-core hosts.  `filtered` contains only rendered diagnostics.
    fn render_sequential<S: DiagnosticSink>(
        &self,
        filtered: &[&Diagnostic<'_>],
        sink: &mut S,
        cx: &RenderCtx<'_>,
    ) -> io::Result<()> {
        let separator = self.renderer.render_separator();
        // Reuse the emitter-hoisted scratch buffer: `emit`'s own doc calls a
        // per-call `String` a pure allocation/free in the LSP re-emit loop,
        // and the serial batch path would otherwise pay exactly that per
        // batch.  `render_sequential` takes `&self`, so the hoisted buffer is
        // borrowable; the renderer never re-enters the emitter, so there is
        // no nested borrow.
        let mut first = true;
        for diag in filtered.iter() {
            // Render boundary: materialize `kind` only for diagnostics we
            // are actually about to render.  Suppressed entries never get
            // here, so a filter-skipped diagnostic costs no Fluent lookup
            // and no derived strings.  `humanize` is idempotent, so an
            // already-materialized diagnostic is a free no-op here.
            // Render *before* deciding on the separator: the separator is
            // inserted between *rendered* diagnostics only, so a diagnostic
            // whose renderer emits nothing contributes no separator either —
            // exactly the dynamic renderer's stitcher semantics
            // (the two paths are byte-identical for every `Renderer`, not
            // just the built-ins).
            let mut buf = self.shared.render_buf.borrow_mut();
            buf.clear();
            let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // No `humanize()` here: the batch entry points take
                // `&Diagnostic`, so humanization is the caller's step (see
                // `DiagCtxt::humanize_pending`) and the emitter cannot mutate a
                // diagnostic — hence no channel to desynchronise tallies.
                self.renderer.render_into(diag, &cx, &mut buf);
            }));
            if rendered.is_err() {
                // Same panic containment as `emit` and the worker path: the
                // fallback's fallback must be global, and this was the one route
                // where a panic inside `render_internal_error` escaped into the
                // caller (aborting an LSP batch, or killing a pooled thread).
                fill_internal_error(&self.renderer, &mut buf);
            }
            if buf.is_empty() {
                continue;
            }
            if first {
                first = false;
            } else if !separator.is_empty() {
                // Write the separator on its own instead of splicing it into
                // the buffer: `insert_str(0, ..)` memmoves the whole rendered
                // diagnostic (up to a 64 KiB pooled buffer) per emission, and
                // may reallocate when capacity == len.  A sink is a
                // concatenation of `write_str` calls, so the bytes are
                // identical.
                sink.write_str(separator)?;
            }
            sink.write_str(&buf)?;
        }
        Ok(())
    }

    /// Prebuild every distinct source's line tables once per batch and sort
    /// them by aggregate render weight, so resident workers preload their
    /// small LRU with the sources they are most likely to touch.
    /// [`Self::prebuild_sources`] without the shared-source wrapper.
    ///
    /// The serial path renders through one hoisted `RenderCtx` and throws the
    /// `Arc<SharedSources>` away, so building it was an allocation plus a refcount
    /// per entry on the *most common* path (every sub-threshold batch, every
    /// single-core host).
    /// Build one source's table against a **snapshot** of the cross-batch cache.
    ///
    /// A free function (not a method) on purpose: the parallel phase must not touch
    /// `self.file_index_cache`, which is a `RefCell` and therefore neither `Sync` nor
    /// reachable from a scoped thread.  The snapshot is taken once, before fanning out.
    fn build_prebuilt(
        snapshot: &[(crate::ast::FileId, u64, usize, Prebuilt)],
        file_id: Option<crate::ast::FileId>,
        key: (usize, usize),
        src: &str,
    ) -> (u64, Prebuilt) {
        let fingerprint = crate::diagnostics::render::source_fingerprint(src);
        // Warm means "a hit is *possible*", which needs the length to match: the lookup
        // requires `n == key.1`, so keying on the id alone made every length-changing edit
        // (the common edit) pay `source_content_hash` *and* the fused `new_hashed` pass.
        let warm =
            file_id.is_some_and(|id| snapshot.iter().any(|(f, _, n, _)| *f == id && *n == key.1));
        let (prebuilt_idx, content_hash) = if warm {
            (None, source_content_hash(src))
        } else {
            let (idx, h) = LineIndex::new_hashed(src);
            (Some(idx), h)
        };
        let cached = file_id.and_then(|id| {
            snapshot.iter().find_map(|(f, h, n, p)| {
                (*f == id && *h == content_hash && *n == key.1).then(|| p.clone())
            })
        });
        let entry = if let Some(p) = cached {
            // The hit proves *content* identity, but `p`'s identity fields address a
            // previous batch's allocation; every downstream lookup matches by `(ptr, len)`,
            // so they are refreshed from this batch (`starts`/`char_prefixes` are
            // content-derived and stay valid).
            Prebuilt {
                ptr: key.0,
                len: key.1,
                fingerprint,
                starts: std::sync::Arc::clone(&p.starts),
                ascii: p.ascii,
                char_prefixes: p.char_prefixes.clone(),
                generation: p.generation,
            }
        } else {
            let idx = prebuilt_idx.unwrap_or_else(|| LineIndex::new(src));
            Prebuilt {
                ptr: key.0,
                len: key.1,
                starts: std::sync::Arc::clone(idx.starts_arc()),
                ascii: idx.ascii(),
                char_prefixes: idx.char_prefixes_arc().clone(),
                generation: idx.generation(),
                fingerprint,
            }
        };
        (content_hash, entry)
    }

    /// Line tables for every distinct source in the batch, heaviest first.
    ///
    /// Three phases: a serial discovery pass (O(diagnostics) — it always was), the
    /// O(source) table builds **in parallel** for a multi-source batch, and a serial cache
    /// merge that preserves the serial version's insertion order exactly.  The build phase
    /// used to be serial too, so a cold multi-file batch paid Σ O(source) (a hash plus a
    /// line-start scan per file; ~20 ms for 100 × 100 KB) *before* any worker was
    /// dispatched — the parallelism the batch path exists for, minus its prologue.
    /// One prebuild job, with the emitter's containment policy applied.
    ///
    /// The rest of this module degrades a failing diagnostic to a placeholder and
    /// never aborts the emit ("a fallback for the fallback must be total"), but the
    /// prebuild phase used to `join().expect(...)`: a panic in
    /// `LineIndex::new_hashed` (its >4 GiB `assert!` is release-enforced) took the
    /// whole batch down.  Dropping the entry instead leaves the render path building
    /// the index per query — correct, just slower — and no `(hash, entry)` pair is
    /// produced, so nothing half-built can reach `file_index_cache`.
    fn build_prebuilt_contained(
        snapshot: &[(crate::ast::FileId, u64, usize, Prebuilt)],
        id: Option<crate::ast::FileId>,
        key: (usize, usize),
        src: &str,
    ) -> Option<(Option<crate::ast::FileId>, (usize, usize), u64, Prebuilt)> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (hash, entry) = Self::build_prebuilt(snapshot, id, key, src);
            (id, key, hash, entry)
        }))
        .ok()
    }

    /// How many entries one prebuild worker takes, or `None` for the serial path.
    ///
    /// The render path caps its workers and gates on total weight; the prebuild path
    /// used to spawn **one scoped thread per distinct source**, unbounded — a
    /// 1,000-file batch that then failed the render weight gate still paid 1,000
    /// spawn/join pairs and held 1,000 line tables at once.  Same policy as the
    /// render gate: gate on bytes, then chunk to the machine's parallelism.
    fn prebuild_chunk_len(len: usize, total_bytes: usize) -> Option<usize> {
        // A thread per file is worse than useless when each file is a few hundred
        // bytes: ~50 µs of spawn buys a table built in ~1 µs.
        if len < 2 || total_bytes < PREBUILD_MIN_TOTAL_BYTES {
            return None;
        }
        let workers = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(len);
        Some(len.div_ceil(workers))
    }

    fn prebuild_entries(&self, filtered: &[&Diagnostic<'_>], weights: &[usize]) -> Vec<Prebuilt> {
        let mut seen: rustc_hash::FxHashSet<(usize, usize)> = rustc_hash::FxHashSet::default();
        let mut source_weight: rustc_hash::FxHashMap<(usize, usize), usize> =
            rustc_hash::FxHashMap::default();
        // `(key, FileId, source)`, in first-appearance order — the order the cache merge
        // below must keep.
        let mut pending: Vec<((usize, usize), Option<crate::ast::FileId>, &str)> = Vec::new();
        for (i, diag) in filtered.iter().enumerate() {
            let Some(src) = diag.source() else {
                continue;
            };
            let key = (src.as_ptr() as usize, src.len());
            *source_weight.entry(key).or_default() += weights[i];
            // A source with no span is never line-queried and could not enter the cache
            // (no `FileId`), so the fused build was pure O(source) waste.
            if diag.spans().first().is_none() {
                continue;
            }
            if seen.insert(key) {
                pending.push((key, diag.spans.primary_ref().first().map(|s| s.file), src));
            }
        }

        // Snapshot before fanning out: `file_index_cache` is a `RefCell`, so the workers
        // cannot consult it, and a mid-batch mutation would make the warm/cold decision
        // depend on thread timing.
        // `iter().cloned()` rather than `.clone()`: the cache is a `SmallVec`, and this
        // phase needs a plain `Vec` it can share with scoped threads.
        let snapshot: Vec<(crate::ast::FileId, u64, usize, Prebuilt)> =
            self.file_index_cache.borrow().iter().cloned().collect();
        let total_bytes: usize = pending.iter().map(|(_, _, src)| src.len()).sum();
        let built: Vec<(Option<crate::ast::FileId>, (usize, usize), u64, Prebuilt)> =
            match Self::prebuild_chunk_len(pending.len(), total_bytes) {
                Some(chunk_len) => std::thread::scope(|scope| {
                    // One task per *chunk*, in order; `flat_map` over the joined
                    // handles keeps the merge below in first-appearance order, which
                    // is the order the cache and the prebuilt slice must have.
                    let handles: Vec<_> = pending
                        .chunks(chunk_len)
                        .map(|chunk| {
                            let snapshot = snapshot.as_slice();
                            scope.spawn(move || {
                                let mut part: Vec<_> = Vec::with_capacity(chunk.len());
                                for (key, id, src) in chunk {
                                    part.extend(Self::build_prebuilt_contained(
                                        snapshot, *id, *key, src,
                                    ));
                                }
                                part
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        // A worker panic can only come from outside the contained job
                        // body, so treat it as "this chunk produced nothing" rather
                        // than aborting the batch.
                        .flat_map(|h| h.join().unwrap_or_default())
                        .collect()
                }),
                None => pending
                    .iter()
                    .filter_map(|(key, id, src)| {
                        Self::build_prebuilt_contained(snapshot.as_slice(), *id, *key, src)
                    })
                    .collect(),
            };

        let mut out: Vec<Prebuilt> = Vec::with_capacity(built.len());
        for (id, _key, hash, entry) in built {
            out.push(entry.clone());
            if let Some(id) = id {
                let mut cache = self.file_index_cache.borrow_mut();
                if cache.len() >= 8 {
                    cache.pop();
                }
                cache.insert(0, (id, hash, entry.len, entry));
            }
        }
        out.sort_by_key(|entry| {
            std::cmp::Reverse(*source_weight.get(&(entry.ptr, entry.len)).unwrap_or(&0))
        });
        out
    }

    /// [`Self::prebuild_entries`] plus the `Arc<SharedSources>` the parallel path
    /// shares across workers.
    ///
    /// **Two consumers, two orders, and they are different.**  `RenderCtx::line_index`'s
    /// tier-0 lookup is a `binary_search_by_key((ptr, len))`, so the list it searches
    /// must be sorted by that key; the worker LRU preload wants the batch's *heaviest*
    /// sources first.  `prebuild_entries` produces the weight order, so this captures
    /// its prefix for the preload and then sorts the list for the search.
    ///
    /// Before this split the weight order went straight to the workers, whose binary
    /// search then missed every entry it did not happen to find by luck.  That was
    /// *correct* — `binary_search` returning `Err` just rebuilds the table, and the
    /// fingerprint check covers a false `Ok` — but it silently turned the prebuild into
    /// dead weight, i.e. exactly the "silent, not loud" failure this cache stack exists
    /// to exclude.
    fn prebuild_sources(
        &self,
        filtered: &[&Diagnostic<'_>],
        weights: &[usize],
    ) -> (
        Vec<Prebuilt>,
        smallvec::SmallVec<[Prebuilt; PRELOAD_SLOTS]>,
        std::sync::Arc<SharedSources>,
    ) {
        let mut out = self.prebuild_entries(filtered, weights);
        let preload: smallvec::SmallVec<[Prebuilt; PRELOAD_SLOTS]> =
            out.iter().take(PRELOAD_SLOTS).cloned().collect();
        // Same key the tier-0 `binary_search` below uses; `sort_unstable` because
        // duplicate keys (one pointer, one length) are indistinguishable to it.
        out.sort_unstable_by_key(|e| (e.ptr, e.len));
        let shared = SharedSources::from_entries(&out);
        (out, preload, shared)
    }

    /// The per-batch shared state both parallel renderers hand to their workers.
    ///
    /// The two renderers' prologues were verbatim copies differing only in the
    /// `streaming` flag — precisely the hand-copied shape `return_buffer_to_pool` was
    /// created to remove, and the dynamic path's unused `Condvar` was residue of it.
    /// This keeps the invariant-dense literal (`Padded` slots, the pool borrow, the
    /// claim-order slice) in one place.
    fn make_batch_shared<'a, 'src>(
        &'a self,
        cells: &'a [&'a Diagnostic<'src>],
        order: &'a [usize],
        prebuilt: &'a [Prebuilt],
        preload: &'a [Prebuilt],
        slots: &'a [Padded<std::sync::Mutex<Option<(String, usize)>>>],
        streaming: bool,
    ) -> BatchShared<'a, 'src, R> {
        BatchShared {
            next: std::sync::atomic::AtomicUsize::new(0),
            next_needed: std::sync::atomic::AtomicUsize::new(0),
            cancelled: std::sync::atomic::AtomicBool::new(false),
            order,
            cells,
            slots,
            slot_cv: std::sync::Condvar::new(),
            renderer: &self.renderer,
            opts: &self.opts,
            prebuilt,
            preload,
            string_pools: &self.string_pools,
            streaming,
        }
    }

    /// The worker jobs for one batch: one per worker, each with its own source/URL
    /// handles and its pool slot.
    ///
    /// Also deduplicated: both renderers built this vector line for line, and a drift
    /// here (a wrong `% pool_count`, a missed `Arc::clone`) is a data race, not a
    /// cosmetic bug.
    fn worker_jobs<'a, 'src>(
        shared: &'a BatchShared<'a, 'src, R>,
        shared_sources: &std::sync::Arc<SharedSources>,
        shared_urls: &crate::diagnostics::render::FileUrlMap,
        num_workers: usize,
        pool_count: usize,
    ) -> Vec<crate::pool::SendJob<'a>> {
        let mut jobs = Vec::with_capacity(num_workers);
        for worker_idx in 0..num_workers {
            let sources = std::sync::Arc::clone(shared_sources);
            let urls = std::sync::Arc::clone(shared_urls);
            // The dispatcher reduces the index modulo the pool count and
            // `string_pools.len()` is that count, so the worker's pool slot is in
            // range by construction.
            // `worker_idx < num_workers <= pool_count` (the dispatcher takes
            // `min(parallelism, ..)` and `string_pools.len() == parallelism`), so the
            // `%` that used to sit here was a division per diagnostic that could only
            // return its own input.  Keep the bound as a debug assertion instead.
            debug_assert!(worker_idx < pool_count, "worker slot out of range");
            let worker_slot = worker_idx;
            jobs.push(crate::pool::SendJob::new(move || {
                worker_render_dynamic(shared, sources, urls, worker_slot)
            }));
        }
        jobs
    }

    /// Resident-pool dynamic-claim renderer.  Replaces the LPT bin split:
    /// claims are atomic, so a wrong weight model only changes claim order,
    /// never load balance.  Slots are indexed by original position and
    /// stitched in order, preserving byte-identical output.
    fn render_dynamic<S: DiagnosticSink>(
        &self,
        filtered: &[&Diagnostic<'_>],
        scratch: &mut BatchScratch,
        sink: &mut S,
        num_workers: usize,
    ) -> io::Result<()> {
        #[cfg(test)]
        PARALLEL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let separator = self.renderer.render_separator();
        let (prebuilt, preload, shared_sources) = self.prebuild_sources(filtered, &scratch.weights);
        let shared_urls = std::sync::Arc::clone(&self.shared.file_urls);
        let n = filtered.len();
        scratch.order.clear();
        scratch.order.extend(0..n);
        // No claim-order sort.  Claims are atomic (`next.fetch_add`), so which worker
        // renders which diagnostic is invisible: the stitch is by slot index, which is
        // why `render_streaming` has always used identity order.  The weight *sum* still
        // gates the parallel path (`emit_batch_iter`); only the O(n log n) reordering was
        // vestigial work here — it did not even improve locality, since `order` is read
        // once per claim from a cold slice.
        let mut slots = std::mem::take(&mut scratch.slots);
        slots.resize_with(n, || Padded(std::sync::Mutex::new(None)));
        for slot in &mut slots[..n] {
            debug_assert!(
                slot.0
                    .get_mut()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_none(),
                "a claim slot survived its batch — the stitch missed it"
            );
            *slot.0.get_mut().unwrap_or_else(|e| e.into_inner()) = None;
        }
        let pool = self.pool.get_or_init(|| {
            let (pool, token) = crate::pool::WorkerPool::new(self.parallelism);
            *self.pool_token.lock().unwrap_or_else(|e| e.into_inner()) = Some(token);
            pool
        });
        let mut permit = BatchPermit::take(&self.pool_token, &self.pool_retired);
        let token = permit
            .token
            .take()
            .expect("batch token is absent only while a batch is in flight");
        // Scoped so the shared borrow of `slots` ends before the stitch, which reads
        // them through `&mut` (`get_mut`, no per-slot lock).  That is what makes reusing
        // the scratch's slot vector a win instead of trading one allocation for `n`
        // mutex round-trips.
        let _shared_wrap = crate::diagnostics::wrap::enter_shared_wrap();
        let pool_count = self.string_pools.len();
        {
            let shared = self.make_batch_shared(
                filtered,
                &scratch.order,
                &prebuilt,
                &preload,
                &slots[..n],
                false,
            );
            let jobs = Self::worker_jobs(
                &shared,
                &shared_sources,
                &shared_urls,
                num_workers,
                pool_count,
            );
            let finished = pool.submit(token, jobs).wait();
            permit.token = Some(finished);
        }
        let mut first = true;
        for slot in &mut slots[..n] {
            let (mut buf, owner) = slot
                .0
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .expect("every claimed render slot must be filled before stitch");
            // Return the buffer even when the render was empty: dropping it here
            // evicted half the pool on an every-other-empty workload, forcing a
            // fresh allocation on the next pop — the churn the pool exists to
            // remove.  A cleared buffer is worth exactly as much as before.
            if buf.is_empty() {
                let mut pool = self.string_pools[owner]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                return_buffer_to_pool(&mut pool, buf);
                continue;
            }
            if first {
                first = false;
            } else if !separator.is_empty() {
                // Write the separator on its own instead of splicing it into
                // the buffer: `insert_str(0, ..)` memmoves the whole rendered
                // diagnostic (up to a 64 KiB pooled buffer) per emission, and
                // may reallocate when capacity == len.  A sink is a
                // concatenation of `write_str` calls, so the bytes are
                // identical.
                sink.write_str(separator)?;
            }
            sink.write_str(&buf)?;
            // Keep one oversize buffer per pool (bounded by
            // `POOL_OVERSIZE_CAP`): a single huge diagnostic used to evict its
            // size class from the pool permanently, so every later batch
            // re-allocated.
            {
                let mut pool = self.string_pools[owner]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                buf.clear();
                return_buffer_to_pool(&mut pool, buf);
            }
        }
        // Hand the (now all-empty) slot vector back to the scratch; its allocation is
        // what the next batch reuses.
        scratch.slots = slots;
        Ok(())
    }

    /// Streaming renderer: workers claim in index order, the main thread
    /// consumes slots in index order and writes each completed diagnostic
    /// immediately.  The ordinary non-streaming path remains the default.
    fn render_streaming<S: DiagnosticSink>(
        &self,
        filtered: &[&Diagnostic<'_>],
        scratch: &mut BatchScratch,
        sink: &mut S,
        num_workers: usize,
    ) -> io::Result<()> {
        #[cfg(test)]
        PARALLEL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let separator = self.renderer.render_separator();
        let (prebuilt, preload, shared_sources) = self.prebuild_sources(filtered, &scratch.weights);
        let shared_urls = std::sync::Arc::clone(&self.shared.file_urls);
        let n = filtered.len();
        scratch.order.clear();
        scratch.order.extend(0..n);
        scratch.prepare_slots(n);
        let shared = self.make_batch_shared(
            filtered,
            &scratch.order,
            &prebuilt,
            &preload,
            &scratch.slots[..n],
            true,
        );
        let pool = self.pool.get_or_init(|| {
            let (pool, token) = crate::pool::WorkerPool::new(self.parallelism);
            *self.pool_token.lock().unwrap_or_else(|e| e.into_inner()) = Some(token);
            pool
        });
        let mut permit = BatchPermit::take(&self.pool_token, &self.pool_retired);
        let token = permit
            .token
            .take()
            .expect("batch token is absent only while a batch is in flight");
        let _shared_wrap = crate::diagnostics::wrap::enter_shared_wrap();
        let pool_count = self.string_pools.len();
        let jobs = Self::worker_jobs(
            &shared,
            &shared_sources,
            &shared_urls,
            num_workers,
            pool_count,
        );
        let wait = pool.submit(token, jobs);
        let mut result: io::Result<()> = Ok(());
        let mut first = true;
        for cursor in 0..n {
            let (mut buf, owner) = {
                let mut slot = shared.slots[cursor]
                    .0
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                // Published *under the slot mutex, before the wait*: that ordering is what
                // lets a worker that filled this slot see the matching value after its own
                // acquire and wake us (see `next_needed`).
                // Published *under the slot mutex, before the wait*: that ordering is what
                // lets a worker that filled this slot see the matching value after its own
                // acquire and wake us (see `next_needed`).  Skip the store when the slot is
                // already filled — a fast consumer drains slots faster than workers fill
                // them, and a filled slot has no waiter to publish to.
                if slot.is_none() {
                    shared
                        .next_needed
                        .store(cursor, std::sync::atomic::Ordering::Relaxed);
                    while slot.is_none() {
                        slot = shared.slot_cv.wait(slot).unwrap_or_else(|e| e.into_inner());
                    }
                }
                slot.take().expect("slot filled")
            };
            // Same as the batch stitch: a fast consumer drains slots faster than
            // workers fill them, so most buffers arriving here are empty — losing
            // them to the allocator defeats the pool.
            if buf.is_empty() {
                let mut pool = shared.string_pools[owner]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                return_buffer_to_pool(&mut pool, buf);
                continue;
            }
            if first {
                first = false;
            } else if !separator.is_empty() {
                // Write the separator on its own instead of splicing it into
                // the buffer: `insert_str(0, ..)` memmoves the whole rendered
                // diagnostic (up to a 64 KiB pooled buffer) per emission, and
                // may reallocate when capacity == len.  A sink is a
                // concatenation of `write_str` calls, so the bytes are
                // identical.
                if let Err(err) = sink.write_str(separator) {
                    // The same protocol as the buffer write below.  `?` here skipped the
                    // `wait.wait()` after the loop, so the batch token never returned,
                    // `BatchPermit::drop` saw `None` and **retired the pool for the
                    // emitter's lifetime** — one mid-publish disconnect silently degraded
                    // every later batch to serial — and the workers kept claiming and
                    // rendering into slots this thread would never read.
                    shared
                        .cancelled
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    result = Err(err);
                    break;
                }
            }
            if let Err(err) = sink.write_str(&buf) {
                // Tell the workers to stop: without this they keep claiming and
                // rendering every remaining diagnostic into slots this thread will
                // never read (an LSP sink whose client disconnected mid-publish is
                // the real shape of that).
                shared
                    .cancelled
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                result = Err(err);
                break;
            }
            {
                let mut pool = shared.string_pools[owner]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                buf.clear();
                return_buffer_to_pool(&mut pool, buf);
            }
        }
        let token = wait.wait();
        permit.token = Some(token);
        // After a sink error the remaining slots still hold rendered buffers — including
        // the ones workers filled while the consumer was unwinding out of the loop
        // (`cancelled` is checked before a *claim*, not during a render).  They used to
        // drop with `shared`, handing the pool's warm buffers to the allocator at the
        // one moment a re-publish (an LSP client that reconnects) wants them back.
        //
        // Only after `wait.wait()`: the workers are joined, so nothing can fill a slot
        // underneath us.  Guarded on `result.is_err()` because the success path leaves
        // every slot empty and this loop is `n` uncontended lock/unlock pairs.
        if result.is_err() {
            for slot in shared.slots {
                let Some((buf, owner)) = slot.0.lock().unwrap_or_else(|e| e.into_inner()).take()
                else {
                    continue;
                };
                let mut pool = shared.string_pools[owner]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                return_buffer_to_pool(&mut pool, buf);
            }
        }
        result
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::{FileId, Span};
    use crate::diagnostics::JsonRenderer;
    use crate::diagnostics::sink::{StringSink, string};

    /// The parallel prebuild feeds two consumers with **different** orders: the tier-0
    /// `binary_search_by_key((ptr, len))` in `RenderCtx::line_index`, and the worker LRU
    /// seed (heaviest source first).  The weight order used to go to both, so the binary
    /// search missed every entry it did not find by luck — correct (a miss just rebuilds)
    /// but silently pointless, which is the failure mode this cache stack exists to
    /// exclude.
    #[test]
    fn parallel_prebuild_sorts_for_the_search_and_keeps_weight_for_the_preload() {
        let id = FileId::DUMMY;
        let light = "a\nb\nc";
        let heavy = "a\nb\nc\nd\ne\nf\ng";
        let mut emitter = Emitter::new(JsonRenderer::new(false), string(), DiagOpts::default());
        let diags = vec![
            warn_diag()
                .with_span(Span::new(id, 1, 2))
                .with_source(light),
            warn_diag()
                .with_span(Span::new(id, 1, 2))
                .with_source(heavy),
        ];
        let refs: Vec<&Diagnostic<'_>> = diags.iter().collect();

        // Weight the second diagnostic so weight order and pointer order cannot
        // coincide by accident (whichever way the allocator ordered the two strings).
        let (prebuilt, preload, _shared) = emitter.prebuild_sources(&refs, &[1, 9]);

        let keys: Vec<(usize, usize)> = prebuilt.iter().map(|e| (e.ptr, e.len)).collect();
        assert!(
            keys.windows(2).all(|w| w[0] <= w[1]),
            "the tier-0 binary search needs (ptr, len) order, got {keys:?}"
        );
        assert_eq!(preload.len(), 2, "every buildable source fits the seed");
        assert_eq!(
            preload[0].len,
            heavy.len(),
            "the preload prefix is chosen by weight, not by pointer order"
        );
    }

    fn warn_diag() -> Diagnostic<'static> {
        Diagnostic::warning("boom")
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_source("x")
    }

    fn emit(opts: DiagOpts) -> String {
        let mut diags = [warn_diag(), warn_diag()];
        let mut sink = StringSink::new();
        Emitter::new(JsonRenderer::new(false), string(), opts)
            .emit_all_to(&mut diags, &mut sink)
            .unwrap();
        sink.into_string()
    }

    /// Alternating file names must not allocate per diagnostic.
    ///
    /// `add_id_name` used to end with `last_pair = Some((id, CompactString::from(name)))`,
    /// so a batch alternating two files rebuilt and dropped a `CompactString` for *every*
    /// diagnostic — the exact pattern such a batch produces.  The memo now reuses its
    /// buffer (`clear` + `push_str`), which this counts.
    #[test]
    fn alternating_file_names_reuse_the_last_pair_buffer() {
        LAST_PAIR_ALLOCS.with(|c| c.set(0));
        let mut seen = FilesSeen::new();
        for i in 0..64 {
            seen.add_id_name(FileId::DUMMY, if i % 2 == 0 { "a.pn" } else { "b.pn" });
        }
        assert_eq!(
            LAST_PAIR_ALLOCS.with(std::cell::Cell::get),
            1,
            "the one-entry memo must reuse its buffer, not rebuild it per diagnostic"
        );
    }

    /// A single `emit` must reuse the line table a batch already built.
    ///
    /// The batch path fills `file_index_cache`; the single-diagnostic path (`emit`) used
    /// to pass no prebuilt table at all and rebuilt the whole line index per keystroke.
    /// `generation` is the `LineIndex` instance id, so equality proves the *same* table.
    #[test]
    fn a_single_emit_reuses_the_line_table_a_batch_cached() {
        let src = "a\nb\nc\nd\ne";
        let id = FileId::DUMMY;
        let mut emitter = Emitter::new(JsonRenderer::new(false), string(), DiagOpts::default());
        let mut diags = vec![
            warn_diag().with_span(Span::new(id, 2, 3)).with_source(src),
            warn_diag().with_span(Span::new(id, 4, 5)).with_source(src),
        ];
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut diags[..], &mut sink).unwrap();

        let batch = emitter.prebuild_entries(&[&diags[0]], &[1]);
        let single = emitter.prebuilt_for_single(src, Some(id));
        assert_eq!(batch.len(), 1);
        assert_eq!(single.len(), 1);
        assert_eq!(
            single[0].generation, batch[0].generation,
            "the single path must hit the batch's cached table instead of rebuilding it"
        );
    }

    /// A sink that fails on the **separator** write must abort the batch the same way a
    /// failed buffer write does: workers stop, and the pool survives.
    ///
    /// The separator write used to propagate with `?`, which returned before the
    /// `wait.wait()` at the end of the stitch.  The batch token then never returned, so
    /// `BatchPermit::drop` retired the pool for the emitter's lifetime — one
    /// disconnect-shaped sink error silently degraded every later batch to serial — and the
    /// workers kept rendering into slots nobody would read.
    #[test]
    fn separator_write_failure_cancels_workers_and_keeps_the_pool() {
        struct FailAfter {
            budget: usize,
            writes: usize,
            out: String,
        }
        impl std::io::Write for FailAfter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if self.writes >= self.budget {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "test sink budget exhausted",
                    ));
                }
                self.writes += 1;
                if let Ok(text) = std::str::from_utf8(buf) {
                    self.out.push_str(text);
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let mut diags: Vec<Diagnostic<'_>> = (0..512).map(|_| warn_diag()).collect();
        // One successful write (the first buffer), then the separator write fails.
        let mut sink = FailAfter {
            budget: 1,
            writes: 0,
            out: String::new(),
        };
        let mut emitter = Emitter::new(
            crate::diagnostics::HumanRenderer::plain(),
            string(),
            DiagOpts::default(),
        )
        .with_parallelism(4);
        let result = emitter.emit_all_streaming_to(&mut diags, &mut sink);
        assert!(result.is_err(), "the sink error must surface");
        assert!(
            PARALLEL_CALLS.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "the streaming path must actually have run"
        );
        assert!(
            !emitter
                .pool_retired
                .load(std::sync::atomic::Ordering::Relaxed),
            "a sink failure must not retire the pool for the emitter's lifetime"
        );
    }

    /// `reverse_errors` + `max_errors` must admit the *last* diagnostics
    /// in render order, and the slot stitch must not disturb that order.
    #[test]
    fn reverse_errors_max_errors_emit_last_in_original_order() {
        let mut diags: Vec<Diagnostic<'static>> = (0..96)
            .map(|i| {
                Diagnostic::error(format!(
                    "error number {i} with a deliberately long message to carry weight"
                ))
                .with_span(Span::new(FileId::DUMMY, 0, 1))
                .with_source("line one\nline two\nline three\n")
            })
            .collect();
        let opts = DiagOpts {
            reverse_errors: true,
            max_errors: Some(1),
            ..DiagOpts::default()
        };
        let mut sink = StringSink::new();
        Emitter::new(JsonRenderer::new(false), string(), opts)
            .emit_all_to(&mut diags, &mut sink)
            .unwrap();
        let out = sink.into_string();
        assert!(out.contains("error number 95"), "{out}");
        assert!(!out.contains("error number 0 "), "{out}");
    }

    #[test]
    fn cross_batch_cache_rebuilds_for_middle_line_shift() {
        // Same FileId, same length, same first/last 8 bytes, but line starts
        // differ: the old first/last-8 fingerprint reused a stale table.
        let src1 = "HEADER00\na\nb\nTAIL0000";
        let src2 = "HEADER00\nab\n\nTAIL0000";
        assert_eq!(src1.len(), src2.len());
        assert_eq!(
            source_content_hash(src1) == source_content_hash(src2),
            false
        );

        let mk = |src: &'static str| -> Vec<Diagnostic<'static>> {
            (0..64)
                .map(|i| {
                    Diagnostic::warning(format!("{i} {}", "x".repeat(64)))
                        .with_span(Span::new(FileId::DUMMY, 11, 12))
                        .with_source(src)
                        .with_file_name("f.ps")
                })
                .collect()
        };
        let opts = DiagOpts::default();
        let mut emitter = Emitter::new(JsonRenderer::new(false), string(), opts);
        emitter.parallelism = 4;

        let mut first = mk(src1);
        let mut sink1 = StringSink::new();
        emitter.emit_all_to(&mut first, &mut sink1).unwrap();
        let out1 = sink1.into_string();
        assert!(out1.contains("\"line\": 3"), "{out1}");

        let mut second = mk(src2);
        let mut sink2 = StringSink::new();
        emitter.emit_all_to(&mut second, &mut sink2).unwrap();
        let out2 = sink2.into_string();
        assert!(
            out2.contains("\"line\": 2"),
            "stale cross-batch line table survived a middle edit: {out2}"
        );
    }

    /// Renders the message for "real" diagnostics and *nothing* for messages
    /// named "empty" — the minimal reproduction of the byte-identical
    /// parallel/sequential contract, with a visible separator so a stray
    /// `<sep><sep>` cannot hide.
    struct EveryOtherEmpty;

    impl Renderer for EveryOtherEmpty {
        fn render_into(&self, diag: &Diagnostic<'_>, _cx: &RenderCtx<'_>, out: &mut String) {
            if diag.message() != "empty" {
                out.push_str(diag.message());
            }
        }
        fn render_separator(&self) -> &str {
            "|"
        }
        fn render_summary(&self, _errors: usize, _warnings: usize) -> String {
            String::new()
        }
    }

    fn alternating_diags(n: usize) -> Vec<Diagnostic<'static>> {
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    Diagnostic::warning("full")
                } else {
                    Diagnostic::warning("empty")
                }
            })
            .collect()
    }

    /// The parallel stitcher must be byte-identical to the serial renderer
    /// even when individual diagnostics render empty: an empty render
    /// contributes no separator, whether it sits inside a chunk or *starts*
    /// one right after a non-empty chunk (the double-separator case).  The batch
    /// is large enough that both the count (≥ 64) and total-weight (≥ 256)
    /// gates pass, and the test asserts that the parallel path was actually
    /// entered on multi-core hosts — the previous 128 one-line diagnostics
    /// stayed under the weight threshold and silently tested the serial path.
    #[test]
    fn human_parallel_matches_serial() {
        const N: usize = 192;
        let mk = || -> Vec<Diagnostic<'static>> {
            (0..N)
                .map(|i| {
                    Diagnostic::error(format!("error {i} {}", "x".repeat(96)))
                        .with_code_str("E030")
                        .with_span(Span::new(FileId::DUMMY, 0, 1))
                        .with_source("fn main() { let x = 1; }\n")
                })
                .collect()
        };
        let opts = DiagOpts {
            quiet: true,
            ..DiagOpts::default()
        };
        let mut emitter = Emitter::new(
            crate::diagnostics::HumanRenderer::plain(),
            string(),
            opts.clone(),
        );
        emitter.parallelism = 4;
        let before = PARALLEL_CALLS.load(std::sync::atomic::Ordering::Relaxed);

        let mut parallel_diags = mk();
        let mut parallel_sink = StringSink::new();
        emitter
            .emit_all_to(&mut parallel_diags, &mut parallel_sink)
            .unwrap();
        let parallel = parallel_sink.into_string();
        if emitter.parallelism > 1 {
            assert!(
                PARALLEL_CALLS.load(std::sync::atomic::Ordering::Relaxed) > before,
                "the assertion must exercise the parallel glyph path"
            );
        }

        let mut serial_diags = mk();
        let refs: Vec<&Diagnostic<'static>> = serial_diags.iter().collect();
        let mut serial_sink = StringSink::new();
        let cx = RenderCtx::new(&opts, None);
        emitter
            .render_sequential(&mut { refs }, &mut serial_sink, &cx)
            .unwrap();
        let serial = serial_sink.into_string();
        assert_eq!(
            parallel, serial,
            "parallel glyph output diverged from serial"
        );
    }

    #[test]
    fn parallel_matches_sequential_when_renders_are_empty() {
        const N: usize = 512;
        let opts = DiagOpts::default();
        let mut emitter = Emitter::new(EveryOtherEmpty, string(), opts.clone());
        let before = PARALLEL_CALLS.load(std::sync::atomic::Ordering::Relaxed);

        let mut parallel_diags = alternating_diags(N);
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut parallel_diags, &mut sink).unwrap();
        let parallel = sink.into_string();

        if emitter.parallelism > 1 {
            assert!(
                PARALLEL_CALLS.load(std::sync::atomic::Ordering::Relaxed) > before,
                "test must exercise the parallel path, not the serial fallback"
            );
        }

        let mut serial_diags = alternating_diags(N);
        let serial_refs: Vec<&Diagnostic<'static>> = serial_diags.iter().collect();
        let mut serial_sink = StringSink::new();
        let cx = RenderCtx::new(&opts, None);
        emitter
            .render_sequential(&mut { serial_refs }, &mut serial_sink, &cx)
            .unwrap();
        let serial = serial_sink.into_string();

        let expected = std::iter::repeat("full")
            .take(N / 2)
            .collect::<Vec<_>>()
            .join("|");
        assert_eq!(serial, expected, "serial path regressed:\n{serial}");
        assert_eq!(parallel, serial, "parallel output diverged from serial");
    }

    /// Renders every message as `ok`, panicking on `boom`; used to pin the
    /// per-diagnostic unwind boundary in the resident-pool path.
    struct PanicOnBoom;

    impl Renderer for PanicOnBoom {
        fn render_into(&self, diag: &Diagnostic<'_>, _cx: &RenderCtx<'_>, out: &mut String) {
            assert!(diag.message() != "boom", "intentional render panic");
            out.push_str("ok");
        }
        fn render_separator(&self) -> &str {
            "|"
        }
        fn render_summary(&self, _errors: usize, _warnings: usize) -> String {
            String::new()
        }
    }

    #[test]
    fn dynamic_claims_isolate_render_panics() {
        let opts = DiagOpts::default();
        let mut emitter = Emitter::new(PanicOnBoom, string(), opts);
        // Forced: on a 1-core runner this used to `return`, so the only coverage of the
        // parallel path's invariants never ran.
        let mut emitter = if emitter.parallelism <= 1 {
            emitter.with_parallelism(4)
        } else {
            emitter
        };
        const N: usize = 512;
        let mut diags: Vec<Diagnostic<'static>> = (0..N)
            .map(|i| {
                if i % 128 == 0 {
                    Diagnostic::warning("boom")
                } else {
                    Diagnostic::warning("ok")
                }
            })
            .collect();
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut diags, &mut sink).unwrap();
        let out = sink.into_string();
        assert_eq!(out.matches("ok").count(), N - N / 128);
        assert_eq!(
            out.matches("diagnostic rendering panicked").count(),
            N / 128,
            "every failed diagnostic must get exactly one placeholder: {out}"
        );
    }

    /// A permit that goes into `submit` and never comes back must retire
    /// the pool rather than leave an `expect` that panics on every later batch.
    #[test]
    fn lost_batch_permit_retires_the_pool() {
        let slot = std::sync::Mutex::new(None);
        let retired = std::sync::atomic::AtomicBool::new(false);
        {
            let permit = BatchPermit::take(&slot, &retired);
            assert!(permit.token.is_none());
            // Drop with no permit: the unwound-through-`submit` shape.
        }
        assert!(
            retired.load(std::sync::atomic::Ordering::Relaxed),
            "a swallowed permit must retire the pool instead of panicking later"
        );
    }

    /// A workload whose renders are mostly empty must not drain the buffer
    /// pool — the empty stitch used to drop the buffer to the allocator.
    #[test]
    fn empty_renders_keep_the_buffer_pool_resident() {
        let opts = DiagOpts::default();
        let mut emitter = Emitter::new(EveryOtherEmpty, string(), opts);
        // Below the parallel threshold the pools are never touched.
        // Forced: on a 1-core runner this used to `return`, so the only coverage of the
        // parallel path's invariants never ran.
        let mut emitter = if emitter.parallelism <= 1 {
            emitter.with_parallelism(4)
        } else {
            emitter
        };
        let mut diags = alternating_diags(512);
        let mut sink = StringSink::new();
        emitter.emit_all_to(&mut diags, &mut sink).unwrap();
        let resident: usize = emitter
            .string_pools
            .iter()
            .map(|pool| pool.lock().unwrap_or_else(|e| e.into_inner()).bufs.len())
            .sum();
        assert!(
            resident > 0,
            "an every-other-empty batch must leave buffers pooled, not drop them"
        );
    }

    /// 5.5: the single-emit path (the LSP hot path) contains a render panic to a
    /// placeholder instead of unwinding into the caller.
    #[test]
    fn single_emit_contains_render_panics() {
        let mut emitter = Emitter::new(PanicOnBoom, string(), DiagOpts::default());
        let mut diag = Diagnostic::warning("boom");
        emitter
            .emit(&mut diag)
            .expect("a panic must not escape emit");
    }

    #[test]
    fn streaming_matches_sequential_output() {
        const N: usize = 512;
        let opts = DiagOpts::default();
        let mut emitter = Emitter::new(EveryOtherEmpty, string(), opts.clone());
        let before = PARALLEL_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        let mut diags = alternating_diags(N);
        let mut sink = StringSink::new();
        emitter
            .emit_all_streaming_to(&mut diags, &mut sink)
            .unwrap();
        let streaming = sink.into_string();
        if emitter.parallelism > 1 {
            assert!(
                PARALLEL_CALLS.load(std::sync::atomic::Ordering::Relaxed) > before,
                "streaming test must exercise the parallel path"
            );
        }

        let mut serial_diags = alternating_diags(N);
        let serial_refs: Vec<&Diagnostic<'static>> = serial_diags.iter().collect();
        let mut serial_sink = StringSink::new();
        let cx = RenderCtx::new(&opts, None);
        emitter
            .render_sequential(&mut { serial_refs }, &mut serial_sink, &cx)
            .unwrap();
        assert_eq!(
            streaming,
            serial_sink.into_string(),
            "streaming output must be byte-identical to serial"
        );
    }

    /// `quiet` must gate the suppressed note exactly like the summary: the
    /// human-facing `= note: N additional errors suppressed` trailer used to
    /// leak through `--quiet`.
    #[test]
    fn quiet_gates_the_suppressed_note() {
        let opts = DiagOpts {
            quiet: true,
            max_errors: Some(1),
            ..DiagOpts::default()
        };
        let mut diags = [Diagnostic::error("first"), Diagnostic::error("second")];
        let mut sink = StringSink::new();
        Emitter::new(EveryOtherEmpty, string(), opts)
            .emit_all_to(&mut diags, &mut sink)
            .unwrap();
        let out = sink.into_string();
        assert_eq!(
            out, "first",
            "quiet must suppress the note and summary:\n{out}"
        );

        let opts = DiagOpts {
            quiet: false,
            max_errors: Some(1),
            ..DiagOpts::default()
        };
        let mut diags = [Diagnostic::error("first"), Diagnostic::error("second")];
        let mut sink = StringSink::new();
        Emitter::new(EveryOtherEmpty, string(), opts)
            .emit_all_to(&mut diags, &mut sink)
            .unwrap();
        let out = sink.into_string();
        assert!(
            out.contains("note: 1 additional error suppressed"),
            "non-quiet must still emit the note:\n{out}"
        );
    }

    /// `warn_is_error` promotes warnings into the error budget: the second
    /// warning is dropped by `max_errors` even though it renders at its
    /// original level.  Without the flag the budget is untouched.
    #[test]
    fn warnings_consume_the_error_budget_only_when_promoted() {
        let promoted = DiagOpts {
            warn_is_error: true,
            max_errors: Some(1),
            ..DiagOpts::default()
        };
        let out = emit(promoted);
        assert_eq!(
            out.matches("\"boom\"").count(),
            1,
            "promoted warning must be cut by max_errors:\n{out}"
        );

        let plain = DiagOpts {
            warn_is_error: false,
            max_errors: Some(1),
            ..DiagOpts::default()
        };
        let out = emit(plain);
        assert_eq!(
            out.matches("\"boom\"").count(),
            2,
            "plain warnings must not consume the error budget:\n{out}"
        );
    }

    #[test]
    fn files_seen_name_then_id_counts_one_file() {
        let mut seen = FilesSeen::new();
        seen.insert("a.rs", None);
        seen.insert("a.rs", Some(FileId::DUMMY));
        assert_eq!(seen.len(), 1, "logical file was counted twice");
    }

    /// The alias case both arms must handle the same way: a file rendered under
    /// two display names, where the second name is first seen on a span-less
    /// diagnostic and is then claimed by an `FileId` that is *already* present.
    /// The Small arm used to overwrite in place and report two files.
    #[test]
    fn files_seen_claim_onto_an_existing_id_dedups_in_both_modes() {
        // Small (inline) mode.
        let mut small = FilesSeen::new();
        small.insert("a.rs", Some(FileId::from_raw(1)));
        small.insert("b.rs", None);
        small.insert("b.rs", Some(FileId::from_raw(1)));
        assert_eq!(small.len(), 1, "Small mode double-counted one file");

        // Large mode: force the spill, then replay the same sequence.
        let mut large = FilesSeen::new();
        for i in 0..(INLINE_FILES + 2) {
            large.insert("filler.rs", Some(FileId::from_raw(0x100 + i as u32)));
        }
        large.insert("a.rs", Some(FileId::from_raw(1)));
        large.insert("b.rs", None);
        large.insert("b.rs", Some(FileId::from_raw(1)));
        let dedup_len = large.len();
        let mut large_ref = FilesSeen::new();
        for i in 0..(INLINE_FILES + 2) {
            large_ref.insert("filler.rs", Some(FileId::from_raw(0x100 + i as u32)));
        }
        large_ref.insert("a.rs", Some(FileId::from_raw(1)));
        assert_eq!(
            dedup_len,
            large_ref.len(),
            "claiming an existing id must not add an entry"
        );
    }

    #[test]
    fn files_seen_distinct_ids_with_same_name_stay_distinct() {
        let mut seen = FilesSeen::new();
        seen.insert("module.rs", Some(FileId::from_raw(1)));
        seen.insert("module.rs", Some(FileId::from_raw(2)));
        assert_eq!(seen.len(), 2, "distinct FileIds were merged by name");
    }

    #[test]
    fn files_seen_large_spill_preserves_last_and_claim() {
        let mut seen = FilesSeen::new();
        for i in 1..=6u32 {
            seen.insert(&format!("f{i}.rs"), Some(FileId::from_raw(i)));
        }
        assert_eq!(seen.len(), 6);
        // `last` memo must still hit after the Small→Large spill.
        seen.insert("f6.rs", Some(FileId::from_raw(6)));
        assert_eq!(seen.len(), 6);
        seen.insert("f3.rs", Some(FileId::from_raw(3)));
        assert_eq!(seen.len(), 6);
        // Anonymous entry spilled first, then claimed by a real id.
        seen.insert("anon.rs", None);
        assert_eq!(seen.len(), 7);
        seen.insert("anon.rs", Some(FileId::from_raw(7)));
        assert_eq!(seen.len(), 7, "wrong span-less claim in Large mode");
    }
}
