//! The emitter: the only layer that performs I/O.
//!
//! An [`Emitter`] pairs a pure [`Renderer`] with a [`MakeSink`] factory.  It
//! owns the batch loop — `begin`, one `render` per diagnostic, the summary,
//! `end`, flush — and is the sole place that touches a sink.  Renderers never
//! see an `io::Write`; sinks never see a `Diagnostic`.

use crate::diagnostics::render::{
    CharPrefixMap, PRELOAD_SLOTS, RenderCtx, Renderer, SharedRenderState, SharedSources,
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
const RENDER_PANIC_PLACEHOLDER: &str = "<internal: diagnostic rendering panicked>\n";

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
    seen: FilesSeenInner,
}

/// Inline capacity of [`FilesSeenInner::Small`].  Used in both the array type
/// and the spill check below so the two cannot silently drift apart (changing
/// one without the other would move the spill point).
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
    /// Exact `(FileId, display name)` pairs (aliases included).
    id_names: rustc_hash::FxHashSet<(crate::ast::FileId, CompactString)>,
}

impl FilesSeen {
    fn new() -> Self {
        FilesSeen {
            last: None,
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
        if let Some(id) = id {
            if self.last_id() == Some(id) {
                self.add_id_name(id, name);
                return;
            }
        } else {
            if self.last_name() == Some(name) {
                return;
            }
            if let Some(prev) = self.last_id()
                && self.id_name_known(prev, name)
            {
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
                seen[pos] = FileKey::Id(id);
                last_index = pos;
                if !id_names.iter().any(|(known_id, known_name)| {
                    *known_id == id && known_name == name
                }) {
                    id_names.push((id, CompactString::from(name)));
                }
            }
            FilesSeenInner::Large { seen, names } => {
                let _ = seen.shift_remove(&old);
                last_index = seen.insert_full(FileKey::Id(id)).0;
                names.anon.remove(name);
                names.claimed.insert(CompactString::from(name));
                names.id_names.insert((id, CompactString::from(name)));
            }
        }
        self.last = Some(last_index);
    }

    fn add_id_name(&mut self, id: crate::ast::FileId, name: &str) {
        match &mut self.seen {
            FilesSeenInner::Small { id_names, .. } => {
                if !id_names.iter().any(|(known_id, known_name)| {
                    *known_id == id && known_name == name
                }) {
                    id_names.push((id, CompactString::from(name)));
                }
            }
            FilesSeenInner::Large { names, .. } => {
                names.claimed.insert(CompactString::from(name));
                names.id_names.insert((id, CompactString::from(name)));
            }
        }
    }

    fn add_anonymous_name(&mut self, name: &str) {
        if let FilesSeenInner::Large { names, .. } = &mut self.seen {
            names.anon.insert(CompactString::from(name));
        }
    }

    fn id_name_known(&self, id: crate::ast::FileId, name: &str) -> bool {
        match &self.seen {
            FilesSeenInner::Small { id_names, .. } => id_names
                .iter()
                .any(|(known_id, known_name)| *known_id == id && known_name == name),
            FilesSeenInner::Large { names, .. } => names
                .id_names
                .contains(&(id, CompactString::from(name))),
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
            id_names: rustc_hash::FxHashSet::default(),
        };
        for key in seen {
            if let FileKey::Name(name) = key {
                index.anon.insert(name.clone());
            }
        }
        for (id, name) in id_names {
            index.claimed.insert(name.clone());
            index.id_names.insert((*id, name.clone()));
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
}

/// Prebuilt line-table entry shared by all workers.
type Prebuilt = (
    usize,
    usize,
    std::sync::Arc<[u32]>,
    bool,
    std::sync::Arc<CharPrefixMap>,
    u64,
    u64,
);

/// One batch shared with resident workers.  No lifetime erasure: the
/// diagnostics live behind `Mutex` cells, so Rust's borrow checker proves the
/// exclusive access that the GLM plan obtained with unsafe raw pointers.
struct BatchShared<'a, 'src, R: Renderer> {
    next: std::sync::atomic::AtomicUsize,
    /// Claim order (weights descending for throughput, index order for
    /// streaming).
    order: &'a [usize],
    cells: Vec<std::sync::Mutex<&'a mut Diagnostic<'src>>>,
    slots: Vec<std::sync::Mutex<Option<String>>>,
    slot_cv: std::sync::Condvar,
    renderer: &'a R,
    opts: &'a DiagOpts,
    prebuilt: &'a [Prebuilt],
    string_pool: &'a std::sync::Mutex<Vec<String>>,
    streaming: bool,
}

/// Resident-pool worker for one batch.  Claims are atomic and each cell is
/// locked exactly for its own render, so weight-model lies only affect claim
/// order, not load balance.
fn worker_render_dynamic<R: Renderer>(
    shared: &BatchShared<'_, '_, R>,
    sources: std::sync::Arc<SharedSources>,
    urls: crate::diagnostics::render::FileUrlMap,
) {
    let cx = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        RenderCtx::for_parallel_worker(
            shared.opts,
            None,
            sources,
            urls,
            &shared.prebuilt[..shared.prebuilt.len().min(PRELOAD_SLOTS)],
        )
    }))
    .ok();
    loop {
        let claim = shared
            .next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if claim >= shared.order.len() {
            break;
        }
        let orig = shared.order[claim];
        let mut buf = shared
            .string_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default();
        buf.clear();
        let buf = match &cx {
            Some(cx) => {
                let mut guard = shared.cells[orig]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let diag: &mut Diagnostic<'_> = &mut **guard;
                let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    diag.humanize();
                    shared.renderer.render_into(diag, cx, &mut buf);
                }));
                if rendered.is_err() {
                    buf.clear();
                    buf.push_str(RENDER_PANIC_PLACEHOLDER);
                }
                buf
            }
            // Setup panic: keep draining the queue so slots/latch cannot stall.
            None => {
                buf.clear();
                buf.push_str(RENDER_PANIC_PLACEHOLDER);
                buf
            }
        };
        *shared.slots[orig]
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(buf);
        if shared.streaming {
            shared.slot_cv.notify_all();
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
    /// `SharedRenderState`).  The per-source line-index LRU is deliberately
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
    /// Reused across batch emits: weights and claim order.
    batch: BatchScratch,
    /// Rendered `String` buffers returned by earlier batches, reused by
    /// resident workers instead of a fresh allocation per diagnostic.
    string_pool: std::sync::Mutex<Vec<String>>,
    /// Cross-batch line tables keyed by `(FileId, source_fingerprint)`.
    /// `Prebuilt` owns its `Arc` tables, so no source lifetime is captured.
    file_index_cache:
        std::cell::RefCell<crate::smallvec::SmallVec<[(crate::ast::FileId, u64, Prebuilt); 8]>>,
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
            batch: BatchScratch::default(),
            string_pool: std::sync::Mutex::new(Vec::new()),
            file_index_cache: std::cell::RefCell::new(crate::smallvec::SmallVec::new()),
        }
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
    /// which also shares the per-source line-index LRU across the batch.
    ///
    /// Note: `max_errors` is a *batch* budget — [`Self::emit`] enforces no
    /// cross-call limit, so an LSP-style loop must track its own if it needs
    /// one.
    pub fn emit(&mut self, diag: &mut Diagnostic<'_>) -> io::Result<()> {
        if !self.filter.matches(diag) {
            return Ok(());
        }
        diag.humanize();
        let mut sink = self.make_sink.make();
        self.renderer.begin(&mut sink)?;
        let cx = RenderCtx::hoisted(&self.opts, None, &self.shared);
        // Reuse the hoisted scratch buffer: the rendered text is copied into
        // the sink and dropped, so a fresh `String` per diagnostic is a pure
        // allocation/free in the documented LSP re-emit loop.
        let mut buf = self.shared.render_buf.borrow_mut();
        buf.clear();
        let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            diag.humanize();
            self.renderer.render_into(diag, &cx, &mut buf);
        }));
        if rendered.is_err() {
            buf.clear();
            buf.push_str(RENDER_PANIC_PLACEHOLDER);
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
    pub fn emit_all(&mut self, diags: &mut [Diagnostic<'_>]) -> io::Result<()> {
        let mut sink = self.make_sink.make();
        self.emit_ordered(diags.iter_mut(), &mut sink, false)
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
        I: IntoIterator<Item = &'r mut Diagnostic<'src>>,
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
        I: IntoIterator<Item = &'r mut Diagnostic<'src>>,
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
        I: IntoIterator<Item = &'r mut Diagnostic<'src>>,
        I::IntoIter: DoubleEndedIterator,
    {
        let mut sink = self.make_sink.make();
        self.emit_ordered(diags, &mut sink, true)
    }

    /// Streaming emit to a caller-provided sink, mirroring
    /// [`Self::emit_all_to`] but consuming slots as they complete.
    pub fn emit_all_streaming_to<S: DiagnosticSink>(
        &mut self,
        diags: &mut [Diagnostic<'_>],
        sink: &mut S,
    ) -> io::Result<()> {
        self.emit_ordered(diags.iter_mut(), sink, true)
    }

    /// Escape hatch: write to a caller-provided sink without a `MakeSink`.
    ///
    /// Honours [`DiagOpts::reverse_errors`] exactly like [`Self::emit_all_iter`].
    pub fn emit_all_to<S: DiagnosticSink>(
        &mut self,
        diags: &mut [Diagnostic<'_>],
        sink: &mut S,
    ) -> io::Result<()> {
        self.emit_ordered(diags.iter_mut(), sink, false)
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
        I: Iterator<Item = &'r mut Diagnostic<'src>>,
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
        // Owned names: the batch loop now humanizes (`&mut`) each
        // diagnostic mid-iteration, so the file list cannot hold `&str`
        // views borrowed from the diagnostics themselves.  `FilesSeen`
        // starts as a `SmallVec` linear scan (1–2 files is the common
        // case, where a few `eq` checks beat hashing) and spills to an
        // `FxHashSet` only once the distinct-file count crosses the inline
        // capacity — preserving the O(1) amortized dedup of large
        // multi-file batches without taxing the small ones.
        let mut files = FilesSeen::new();

        // ── Phase 1: filter + count.  Serial: the `max_errors` gate is
        //    order-dependent and trivially O(n) per diagnostic. ────────────
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
        let mut filtered: SmallVec<[&'r mut Diagnostic<'src>; 64]> =
            SmallVec::with_capacity(cap);
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
            if !self.filter.matches(diag) {
                continue;
            }
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
        let render_result = if streaming
            && num_renderable >= PARALLEL_THRESHOLD
            && total_weight >= WEIGHT_THRESHOLD
            && num_threads > 1
        {
            self.render_streaming(filtered.as_mut_slice(), &mut batch, sink, num_threads)
        } else if num_renderable >= PARALLEL_THRESHOLD
            && total_weight >= WEIGHT_THRESHOLD
            && num_threads > 1
        {
            self.render_dynamic(filtered.as_mut_slice(), &mut batch, sink, num_threads)
        } else {
            let cx = RenderCtx::hoisted(&self.opts, None, &self.shared);
            self.render_sequential(filtered.as_mut_slice(), sink, &cx)
        };
        self.batch = batch;
        render_result?;

        // ── Phase 3: trailer.  Always single-threaded. ─────────────────────
        // The suppressed note is a human-facing trailer: like the summary, it
        // is gated on `quiet`, which `HumanRenderer`'s note would otherwise
        // leak through (machine renderers return "" here anyway).
        if suppressed > 0 && !self.opts.quiet {
            sink.write_str(&self.renderer.render_suppressed_note(suppressed))?;
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
        filtered: &mut [&mut Diagnostic<'_>],
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
        for diag in filtered.iter_mut() {
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
                diag.humanize();
                self.renderer.render_into(diag, &cx, &mut buf);
            }));
            if rendered.is_err() {
                buf.clear();
                buf.push_str(RENDER_PANIC_PLACEHOLDER);
            }
            if buf.is_empty() {
                continue;
            }
            if first {
                first = false;
            } else if !separator.is_empty() {
                buf.insert_str(0, separator);
            }
            sink.write_str(&buf)?;
        }
        Ok(())
    }

    /// Prebuild every distinct source's line tables once per batch and sort
    /// them by aggregate render weight, so resident workers preload their
    /// small LRU with the sources they are most likely to touch.
    fn prebuild_sources(
        &self,
        filtered: &[&mut Diagnostic<'_>],
        weights: &[usize],
    ) -> (Vec<Prebuilt>, std::sync::Arc<SharedSources>) {
        let mut seen: rustc_hash::FxHashSet<(usize, usize)> =
            rustc_hash::FxHashSet::default();
        let mut out: Vec<Prebuilt> = Vec::new();
        let mut source_weight: rustc_hash::FxHashMap<(usize, usize), usize> =
            rustc_hash::FxHashMap::default();
        for (i, diag) in filtered.iter().enumerate() {
            if let Some(src) = diag.source() {
                let key = (src.as_ptr() as usize, src.len());
                *source_weight.entry(key).or_default() += weights[i];
                if seen.insert(key) {
                    let file_id = diag.spans.primary_ref().first().map(|s| s.file);
                    let fingerprint =
                        crate::diagnostics::render::source_fingerprint(src);
                    let cached = file_id.and_then(|id| {
                        self.file_index_cache.borrow().iter().find_map(|(f, h, p)| {
                            (*f == id && *h == fingerprint).then(|| p.clone())
                        })
                    });
                    if let Some(p) = cached {
                        out.push((key.0, key.1, p.2, p.3, p.4, p.5, fingerprint));
                    } else {
                        let idx = LineIndex::new(src);
                        let entry = (
                            key.0,
                            key.1,
                            std::sync::Arc::clone(idx.starts_arc()),
                            idx.ascii(),
                            std::sync::Arc::clone(idx.char_prefixes_arc()),
                            idx.generation(),
                            fingerprint,
                        );
                        out.push(entry.clone());
                        if let Some(id) = file_id {
                            let mut cache = self.file_index_cache.borrow_mut();
                            if cache.len() >= 8 {
                                cache.pop();
                            }
                            cache.insert(0, (id, fingerprint, entry));
                        }
                    }
                }
            }
        }
        out.sort_by_key(|entry| {
            std::cmp::Reverse(*source_weight.get(&(entry.0, entry.1)).unwrap_or(&0))
        });
        let shared = SharedSources::from_entries(&out);
        (out, shared)
    }

    /// Resident-pool dynamic-claim renderer.  Replaces the LPT bin split:
    /// claims are atomic, so a wrong weight model only changes claim order,
    /// never load balance.  Slots are indexed by original position and
    /// stitched in order, preserving byte-identical output.
    fn render_dynamic<S: DiagnosticSink>(
        &self,
        filtered: &mut [&mut Diagnostic<'_>],
        scratch: &mut BatchScratch,
        sink: &mut S,
        num_workers: usize,
    ) -> io::Result<()> {
        #[cfg(test)]
        PARALLEL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let separator = self.renderer.render_separator();
        let (prebuilt, shared_sources) = self.prebuild_sources(filtered, &scratch.weights);
        let shared_urls = std::sync::Arc::clone(&self.shared.file_urls);
        let n = filtered.len();
        scratch.order.clear();
        scratch.order.extend(0..n);
        let weights = &scratch.weights;
        scratch.order.sort_unstable_by(|&a, &b| {
            weights[b].cmp(&weights[a]).then(a.cmp(&b))
        });
        let cells: Vec<std::sync::Mutex<&mut Diagnostic<'_>>> = filtered
            .iter_mut()
            .map(|diag| std::sync::Mutex::new(&mut **diag))
            .collect();
        let shared = BatchShared {
            next: std::sync::atomic::AtomicUsize::new(0),
            order: &scratch.order,
            cells,
            slots: (0..n).map(|_| std::sync::Mutex::new(None)).collect(),
            slot_cv: std::sync::Condvar::new(),
            renderer: &self.renderer,
            opts: &self.opts,
            prebuilt: &prebuilt,
            string_pool: &self.string_pool,
            streaming: false,
        };
        let pool = self
            .pool
            .get_or_init(|| crate::pool::WorkerPool::new(self.parallelism));
        let _shared_wrap = crate::diagnostics::wrap::enter_shared_wrap();
        let shared_ref = &shared;
        let mut jobs = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let sources = std::sync::Arc::clone(&shared_sources);
            let urls = std::sync::Arc::clone(&shared_urls);
            jobs.push(crate::pool::SendJob::new(move || {
                worker_render_dynamic(shared_ref, sources, urls)
            }));
        }
        pool.submit(jobs).wait();
        let string_pool = shared.string_pool;
        let BatchShared { slots, .. } = shared;
        let mut first = true;
        for slot in slots {
            let mut buf = slot
                .into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .unwrap_or_else(|| RENDER_PANIC_PLACEHOLDER.to_string());
            if buf.is_empty() {
                continue;
            }
            if first {
                first = false;
            } else if !separator.is_empty() {
                buf.insert_str(0, separator);
            }
            sink.write_str(&buf)?;
            if buf.capacity() <= 64 * 1024 {
                let mut pool = string_pool
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if pool.len() < 64 {
                    buf.clear();
                    pool.push(buf);
                }
            }
        }
        Ok(())
    }

    /// Streaming renderer: workers claim in index order, the main thread
    /// consumes slots in index order and writes each completed diagnostic
    /// immediately.  The ordinary non-streaming path remains the default.
    fn render_streaming<S: DiagnosticSink>(
        &self,
        filtered: &mut [&mut Diagnostic<'_>],
        scratch: &mut BatchScratch,
        sink: &mut S,
        num_workers: usize,
    ) -> io::Result<()> {
        #[cfg(test)]
        PARALLEL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let separator = self.renderer.render_separator();
        let (prebuilt, shared_sources) = self.prebuild_sources(filtered, &scratch.weights);
        let shared_urls = std::sync::Arc::clone(&self.shared.file_urls);
        let n = filtered.len();
        scratch.order.clear();
        scratch.order.extend(0..n);
        let cells: Vec<std::sync::Mutex<&mut Diagnostic<'_>>> = filtered
            .iter_mut()
            .map(|diag| std::sync::Mutex::new(&mut **diag))
            .collect();
        let shared = BatchShared {
            next: std::sync::atomic::AtomicUsize::new(0),
            order: &scratch.order,
            cells,
            slots: (0..n).map(|_| std::sync::Mutex::new(None)).collect(),
            slot_cv: std::sync::Condvar::new(),
            renderer: &self.renderer,
            opts: &self.opts,
            prebuilt: &prebuilt,
            string_pool: &self.string_pool,
            streaming: true,
        };
        let pool = self
            .pool
            .get_or_init(|| crate::pool::WorkerPool::new(self.parallelism));
        let _shared_wrap = crate::diagnostics::wrap::enter_shared_wrap();
        let shared_ref = &shared;
        let mut jobs = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let sources = std::sync::Arc::clone(&shared_sources);
            let urls = std::sync::Arc::clone(&shared_urls);
            jobs.push(crate::pool::SendJob::new(move || {
                worker_render_dynamic(shared_ref, sources, urls)
            }));
        }
        let wait = pool.submit(jobs);
        let mut result: io::Result<()> = Ok(());
        let mut first = true;
        for cursor in 0..n {
            let mut buf = {
                let mut slot = shared.slots[cursor]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                while slot.is_none() {
                    slot = shared
                        .slot_cv
                        .wait(slot)
                        .unwrap_or_else(|e| e.into_inner());
                }
                slot.take().expect("slot filled")
            };
            if buf.is_empty() {
                continue;
            }
            if first {
                first = false;
            } else if !separator.is_empty() {
                buf.insert_str(0, separator);
            }
            if let Err(err) = sink.write_str(&buf) {
                result = Err(err);
                break;
            }
            if buf.capacity() <= 64 * 1024 {
                let mut pool = shared
                    .string_pool
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if pool.len() < 64 {
                    buf.clear();
                    pool.push(buf);
                }
            }
        }
        wait.wait();
        result
    }

}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::{FileId, Span};
    use crate::diagnostics::JsonRenderer;
    use crate::diagnostics::sink::{StringSink, string};

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

    /// 1.4: `reverse_errors` + `max_errors` must admit the *last* diagnostics
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

    /// Renders the message for "real" diagnostics and *nothing* for messages
    /// named "empty" — the minimal reproduction of the byte-identical
    /// parallel/sequential contract, with a visible separator so a stray
    /// `<sep><sep>` cannot hide.
    struct EveryOtherEmpty;

    impl Renderer for EveryOtherEmpty {
        fn render(&self, diag: &Diagnostic<'_>, _cx: &RenderCtx<'_>) -> String {
            if diag.message() == "empty" {
                String::new()
            } else {
                diag.message().to_string()
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
    fn parallel_matches_sequential_when_renders_are_empty() {
        const N: usize = 512;
        let opts = DiagOpts::default();
        let mut emitter = Emitter::new(EveryOtherEmpty, string(), opts.clone());
        let before = PARALLEL_CALLS.load(std::sync::atomic::Ordering::Relaxed);

        let mut parallel_diags = alternating_diags(N);
        let mut sink = StringSink::new();
        emitter
            .emit_all_to(&mut parallel_diags, &mut sink)
            .unwrap();
        let parallel = sink.into_string();

        if emitter.parallelism > 1 {
            assert!(
                PARALLEL_CALLS.load(std::sync::atomic::Ordering::Relaxed) > before,
                "test must exercise the parallel path, not the serial fallback"
            );
        }

        let mut serial_diags = alternating_diags(N);
        let mut serial_refs: Vec<&mut Diagnostic<'static>> = serial_diags.iter_mut().collect();
        let mut serial_sink = StringSink::new();
        let cx = RenderCtx::new(&opts, None);
        emitter
            .render_sequential(&mut serial_refs, &mut serial_sink, &cx)
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
        fn render(&self, diag: &Diagnostic<'_>, _cx: &RenderCtx<'_>) -> String {
            assert!(diag.message() != "boom", "intentional render panic");
            "ok".to_string()
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
        if emitter.parallelism <= 1 {
            return;
        }
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
        let mut serial_refs: Vec<&mut Diagnostic<'static>> = serial_diags.iter_mut().collect();
        let mut serial_sink = StringSink::new();
        let cx = RenderCtx::new(&opts, None);
        emitter
            .render_sequential(&mut serial_refs, &mut serial_sink, &cx)
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
        let mut diags = [
            Diagnostic::error("first"),
            Diagnostic::error("second"),
        ];
        let mut sink = StringSink::new();
        Emitter::new(EveryOtherEmpty, string(), opts)
            .emit_all_to(&mut diags, &mut sink)
            .unwrap();
        let out = sink.into_string();
        assert_eq!(out, "first", "quiet must suppress the note and summary:\n{out}");

        let opts = DiagOpts {
            quiet: false,
            max_errors: Some(1),
            ..DiagOpts::default()
        };
        let mut diags = [
            Diagnostic::error("first"),
            Diagnostic::error("second"),
        ];
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

    #[test]
    fn files_seen_distinct_ids_with_same_name_stay_distinct() {
        let mut seen = FilesSeen::new();
        seen.insert("module.rs", Some(FileId::from_raw(1)));
        seen.insert("module.rs", Some(FileId::from_raw(2)));
        assert_eq!(seen.len(), 2, "distinct FileIds were merged by name");
    }

    #[test]
    fn files_seen_anonymous_name_claimed_by_one_id_then_another_counts_two() {
        let mut seen = FilesSeen::new();
        seen.insert("module.rs", None);
        seen.insert("module.rs", Some(FileId::from_raw(1)));
        assert_eq!(seen.len(), 1, "anonymous name should merge with first id");
        seen.insert("module.rs", Some(FileId::from_raw(2)));
        assert_eq!(
            seen.len(),
            2,
            "the anonymous entry was consumed by Id(1); Id(2) is a second file"
        );
    }
}
