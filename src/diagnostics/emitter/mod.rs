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

/// Test-only arrival counter: proves `render_parallel` was actually taken, so
/// a "parallel == serial" test cannot pass vacuously through the serial branch.
#[cfg(test)]
static PARALLEL_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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
    /// Previous key — the grouped-batch fast path skips the scan on a repeat.
    last: Option<FileKey>,
    seen: FilesSeenInner,
}

/// Inline capacity of [`FilesSeenInner::Small`].  Used in both the array type
/// and the spill check below so the two cannot silently drift apart (changing
/// one without the other would move the spill point).
const INLINE_FILES: usize = 4;

enum FilesSeenInner {
    Small(SmallVec<[FileKey; INLINE_FILES]>),
    Large(rustc_hash::FxHashSet<FileKey>),
}

impl FilesSeen {
    fn new() -> Self {
        FilesSeen {
            last: None,
            seen: FilesSeenInner::Small(SmallVec::new()),
        }
    }

    fn insert(&mut self, name: &str, id: Option<crate::ast::FileId>) {
        // Grouped batches repeat the same file: check the last key before
        // constructing `FileKey::Name`, which would allocate for long paths.
        match (self.last.as_ref(), id) {
            (Some(FileKey::Id(prev)), Some(id)) if *prev == id => return,
            (Some(FileKey::Name(prev)), None) if prev.as_str() == name => return,
            _ => {}
        }
        let key = match id {
            Some(id) => FileKey::Id(id),
            None => FileKey::Name(CompactString::from(name)),
        };
        if self.last.as_ref() == Some(&key) {
            return;
        }
        let (last, seen) = (&mut self.last, &mut self.seen);
        match seen {
            FilesSeenInner::Small(v) => {
                if !v.contains(&key) {
                    if v.len() < INLINE_FILES {
                        v.push(key.clone());
                    } else {
                        let mut set: rustc_hash::FxHashSet<FileKey> = v.drain(..).collect();
                        set.insert(key.clone());
                        *seen = FilesSeenInner::Large(set);
                    }
                }
            }
            FilesSeenInner::Large(s) => {
                s.insert(key.clone());
            }
        }
        *last = Some(key);
    }

    fn len(&self) -> usize {
        match &self.seen {
            FilesSeenInner::Small(v) => v.len(),
            FilesSeenInner::Large(s) => s.len(),
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
        self.renderer.render_into(diag, &cx, &mut buf);
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
        self.emit_ordered(diags.iter_mut(), &mut sink)
    }

    /// Dispatch a double-ended diagnostic iterator through the batch loop in
    /// the configured order.  One home for the reverse/non-reverse split: the
    /// iterator and its reverse are distinct concrete types, so the branch
    /// cannot be erased, but it need not be repeated at every entry point.
    fn emit_ordered<'r, 'src, S, I>(&mut self, iter: I, sink: &mut S) -> io::Result<()>
    where
        S: DiagnosticSink,
        'src: 'r,
        I: IntoIterator<Item = &'r mut Diagnostic<'src>>,
        I::IntoIter: DoubleEndedIterator,
    {
        let iter = iter.into_iter();
        if self.opts.reverse_errors {
            self.emit_batch_iter(iter.rev(), sink)
        } else {
            self.emit_batch_iter(iter, sink)
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
        self.emit_ordered(diags, &mut sink)
    }

    /// Escape hatch: write to a caller-provided sink without a `MakeSink`.
    ///
    /// Honours [`DiagOpts::reverse_errors`] exactly like [`Self::emit_all_iter`].
    pub fn emit_all_to<S: DiagnosticSink>(
        &mut self,
        diags: &mut [Diagnostic<'_>],
        sink: &mut S,
    ) -> io::Result<()> {
        self.emit_ordered(diags.iter_mut(), sink)
    }

    fn emit_batch_iter<'r, 'src, S, I>(&mut self, iter: I, sink: &mut S) -> io::Result<()>
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
        let cap = if hi == Some(lo) { lo } else { lo.min(64) };
        let mut filtered: SmallVec<[&'r mut Diagnostic<'src>; 16]> =
            SmallVec::with_capacity(cap);
        // Cheap render-cost proxy summed over the batch.  The parallel
        // decision gates on this as well as the count: 64 one-line notes cost
        // less to render than the ~50 µs/thread spawn, so they stay serial.
        let mut total_weight = 0usize;
        // Per-diagnostic weights, parallel to `filtered`.  Collected once here
        // so the parallel split consumes this single model instead of
        // recomputing the closure (and risking a drift between two copies).
        let mut weights: SmallVec<[usize; 16]> = SmallVec::with_capacity(cap);

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
                    diag.spans.first().map(|s| s.file),
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
            let weight =
                1 + diag.labels.len() + diag.suggestions.len() + diag.spans.primary_ref().len();
            total_weight += weight;
            weights.push(weight);
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
        // Keep each parallel chunk at least this large so spawn/join
        // overhead (~50 µs/thread) is amortised by real render work.
        const MIN_CHUNK: usize = 16;
        let num_threads = self.parallelism;
        // Cap the spawn count: at the 64-diagnostic threshold this yields
        // 4 threads (≥16 diags each).  Without the cap, a 128-core host
        // would spawn 64 threads for one diagnostic each — ~3 ms of
        // `spawn`/`join` for work that barely amortises.
        let num_threads = num_threads.min((num_renderable / MIN_CHUNK).max(1));
        if num_renderable >= PARALLEL_THRESHOLD
            && total_weight >= WEIGHT_THRESHOLD
            && num_threads > 1
        {
            self.render_parallel(filtered.as_mut_slice(), &weights, total_weight, sink, num_threads)?;
        } else {
            // Serial path: one batch ctx, with the URL cache and scratch
            // hoisted onto the emitter so they survive across batches.
            let cx = RenderCtx::hoisted(&self.opts, None, &self.shared);
            self.render_sequential(filtered.as_mut_slice(), sink, &cx)?;
        }

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
        let mut buf = self.shared.render_buf.borrow_mut();
        let mut first = true;
        for diag in filtered.iter_mut() {
            // Render boundary: materialize `kind` only for diagnostics we
            // are actually about to render.  Suppressed entries never get
            // here, so a filter-skipped diagnostic costs no Fluent lookup
            // and no derived strings.  `humanize` is idempotent, so an
            // already-materialized diagnostic is a free no-op here.
            diag.humanize();
            // Render *before* deciding on the separator: the separator is
            // inserted between *rendered* diagnostics only, so a diagnostic
            // whose renderer emits nothing contributes no separator either —
            // exactly the stitcher's semantics in [`Self::render_parallel`]
            // (the two paths are byte-identical for every `Renderer`, not
            // just the built-ins).
            buf.clear();
            self.renderer.render_into(diag, &cx, &mut buf);
            if buf.is_empty() {
                continue;
            }
            if first {
                first = false;
            } else if !separator.is_empty() {
                sink.write_str(separator)?;
            }
            sink.write_str(&buf)?;
        }
        Ok(())
    }

    /// Parallel batch renderer.  Splits `filtered` (rendered diagnostics
    /// only) into `num_threads` chunks, renders each into its own `String`
    /// on its own scoped thread (each thread has its own `RenderCtx`, so
    /// `char_prefixes`/scratch never cross thread boundaries; the line
    /// tables and the `file://` URL cache *are* shared, via `SharedSources`
    /// and one `Arc<RwLock<…>>` URL map), and stitches the chunk buffers
    /// back in batch order with the renderer's separator.
    ///
    /// # Correctness
    ///
    /// * `split_at_mut` gives each thread exclusive ownership of its
    ///   `&mut Diagnostic` slice, so the internal `humanize` mutations are
    ///   disjoint.
    /// * Every chunk holds ≥ `MIN_CHUNK` real diagnostics (the spawn cap is
    ///   `num_renderable / MIN_CHUNK`), but a renderer may still emit
    ///   nothing for a diagnostic — so an empty chunk `String` stays
    ///   possible.  Empty chunks contribute no separator, and each chunk
    ///   inserts its separator only between *rendered* diagnostics, so the
    ///   final stream is byte-identical to [`Self::render_sequential`] for
    ///   every `Renderer`, including ones that render empty output.
    /// * `renderer` and `opts` are `Sync` (`Renderer: Send + Sync`, `DiagOpts`
    ///   is plain data), so `&R` and `&DiagOpts` are safely shared.
    fn render_parallel<S: DiagnosticSink>(
        &self,
        filtered: &mut [&mut Diagnostic<'_>],
        weights: &[usize],
        total_weight: usize,
        sink: &mut S,
        num_threads: usize,
    ) -> io::Result<()> {
        // The LPT split above uses the per-diagnostic weights directly; the
        // aggregate is retained in the signature for the caller's gate.
        let _ = total_weight;
        #[cfg(test)]
        PARALLEL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // LPT (longest processing time first): assign each diagnostic, heaviest
        // first, to the currently lightest bin.  This is the classic
        // 4/3-approximation makespan heuristic and prevents one monster
        // render from starving a worker (the greedy contiguous split could
        // strand the whole heavy tail in one chunk).  Bins are non-contiguous;
        // workers tag each rendered piece with its original index and the
        // stitch below restores serial order, so the stream stays
        // byte-identical.  Weights were computed once in Phase 1 (one model,
        // so the gate and this split cannot drift).
        let n = filtered.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| weights[b].cmp(&weights[a]).then(a.cmp(&b)));
        let mut bins: Vec<Vec<usize>> = (0..num_threads).map(|_| Vec::new()).collect();
        let mut heap = std::collections::BinaryHeap::new();
        for bin in 0..num_threads {
            heap.push(std::cmp::Reverse((0usize, bin)));
        }
        for i in order {
            let std::cmp::Reverse((load, bin)) = heap.pop().expect("non-empty bin heap");
            let next = load + weights[i];
            bins[bin].push(i);
            heap.push(std::cmp::Reverse((next, bin)));
        }
        let mut bin_of = vec![0usize; n];
        let mut bin_lens: Vec<usize> = Vec::with_capacity(num_threads);
        for (b, bin) in bins.iter().enumerate() {
            if bin.is_empty() {
                continue;
            }
            bin_lens.push(bin.len());
            for &i in bin {
                bin_of[i] = b;
            }
        }
        // Fewer than two non-empty bins (single-core or one diagnostic):
        // spawn/join overhead buys nothing, render serially.
        if bin_lens.len() < 2 {
            let cx = RenderCtx::hoisted(&self.opts, None, &self.shared);
            return self.render_sequential(filtered, sink, &cx);
        }
        let separator = self.renderer.render_separator();
        let renderer = &self.renderer;
        let opts = &self.opts;

        // Pre-build the line tables once in the *main* thread, before
        // spawning: without this, every worker lazily rebuilds a
        // `LineIndex::new(source)` on its first diagnostic — `(T-1) ×
        // O(source_len)` redundant SWAR scans per batch, which on a large
        // single source can rival the render itself.  Sources are keyed by
        // `(ptr, len)` (every `Diagnostic<'src>` in a batch borrows a source
        // alive for the whole call, so the key is unambiguous).  The batch is
        // finite and its sources are alive for the whole call, so *every*
        // distinct source is prebuilt — not just the `CACHE_SLOTS` the
        // per-worker LRU keeps warm; sources beyond those slots are shared
        // via the `SharedSources` map instead of being re-scanned T times.
        // Workers inject the first `CACHE_SLOTS` entries directly into their
        // LRU (preserving the preload semantics for the covered slots) and
        // consult the shared map for the rest.  Each entry also shares one
        // `char_prefixes` map (the hint is an `AtomicUsize` and a racy store
        // can only cost a re-anchor, never a wrong answer), so a non-ASCII
        // source's per-line sparse tables are built once per *batch*, not
        // once per thread.  So is the `file://` URL cache: one
        // `canonicalize` per file per batch, not once per worker per file.
        let prebuilt: Vec<(
            usize,
            usize,
            std::sync::Arc<[u32]>,
            bool,
            std::sync::Arc<CharPrefixMap>,
            u64,
        )> = {
            // The prebuild is *not* a batched render: a `RenderCtx` here would
            // grow its `source_index` LRU cache only for it to be dropped at
            // the end of this block, so the raw constructor does the same work
            // with none of the bookkeeping.
            let mut seen: rustc_hash::FxHashSet<(usize, usize)> =
                rustc_hash::FxHashSet::default();
            let mut out = Vec::new();
            for diag in filtered.iter() {
                if let Some(src) = diag.source() {
                    let key = (src.as_ptr() as usize, src.len());
                    if seen.insert(key) {
                        let idx = LineIndex::new(src);
                        out.push((
                            key.0,
                            key.1,
                            std::sync::Arc::clone(idx.starts_arc()),
                            idx.ascii(),
                            std::sync::Arc::clone(idx.char_prefixes_arc()),
                            idx.generation(),
                        ));
                    }
                }
            }
            out
        };
        let shared_sources = SharedSources::from_entries(&prebuilt);
        // Share the emitter's hoisted URL cache across all workers *and*
        // across batches: cloning the `Arc` (not the map) is legal even though
        // `SharedRenderState` is `!Sync` via its `RefCell`s, because the map
        // itself is `Arc<RwLock<…>>`.  Without this the parallel path paid one
        // `canonicalize` per file per batch even when a serial `emit` had
        // already warmed the cache.
        let shared_urls = std::sync::Arc::clone(&self.shared.file_urls);
        // A slice reference (Copy) so each `move` closure shares the prebuilt
        // tables instead of the first `spawn` moving the `Vec` itself.
        let prebuilt: &[(usize, usize, std::sync::Arc<[u32]>, bool, std::sync::Arc<CharPrefixMap>, u64)] =
            &prebuilt;

        // LPT bins are non-contiguous, so materialise each bin as a contiguous run
        // for `split_at_mut`, tagging every piece with its original index; the
        // final sort restores serial order byte-for-byte.
        let mut items: Vec<(usize, &mut Diagnostic<'_>)> = filtered
            .iter_mut()
            .enumerate()
            .map(|(i, d)| (i, &mut **d))
            .collect();
        items.sort_by_key(|(i, _)| (bin_of[*i], *i));

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(bin_lens.len());
            let mut rest_items: &mut [(usize, &mut Diagnostic<'_>)] = &mut items;

            for &take in bin_lens.iter() {
                let (chunk, remaining) = rest_items.split_at_mut(take);
                rest_items = remaining;

                let shared_sources = std::sync::Arc::clone(&shared_sources);
                let shared_urls = std::sync::Arc::clone(&shared_urls);
                handles.push(s.spawn(move || {
                    // One-shot worker constructor: shared line tables, shared
                    // URL cache, and the preloaded LRU in a single call, so a
                    // worker cannot silently miss one of the setup steps.
                    let cx = RenderCtx::for_parallel_worker(
                        opts,
                        None,
                        shared_sources,
                        shared_urls,
                        &prebuilt[..prebuilt.len().min(PRELOAD_SLOTS)],
                    );
                    // LPT bins are out of order, so return (original index,
                    // rendered piece) and let the stitch sort.  An empty
                    // render is dropped here (it contributes no separator).
                    let mut pieces: Vec<(usize, String)> = Vec::with_capacity(chunk.len());
                    for (orig, diag) in chunk.iter_mut() {
                        let orig = *orig;
                        // One malformed diagnostic must not abort the whole
                        // batch: contain the panic and emit a placeholder for
                        // that slot, keeping every other worker's output.
                        let rendered = std::panic::catch_unwind(
                            std::panic::AssertUnwindSafe(|| {
                                diag.humanize();
                                let mut buf = String::new();
                                renderer.render_into(*diag, &cx, &mut buf);
                                buf
                            }),
                        );
                        match rendered {
                            Ok(buf) if !buf.is_empty() => pieces.push((orig, buf)),
                            Ok(_) => {}
                            Err(_) => pieces.push((
                                orig,
                                "<ponent: diagnostic rendering panicked>\n".to_string(),
                            )),
                        }
                    }
                    pieces
                }));
            }

            // Gather in bin order, then restore the batch's original order.
            // Streaming per chunk is impossible with LPT bins; the retained
            // memory is the same Σ(rendered) as the old collect-then-stitch
            // shape.
            let mut all: Vec<(usize, String)> = Vec::new();
            for h in handles {
                all.extend(h.join().expect("render thread panicked"));
            }
            all.sort_by_key(|(i, _)| *i);
            let mut first = true;
            for (_, buf) in all {
                if first {
                    first = false;
                } else if !separator.is_empty() {
                    sink.write_str(separator)?;
                }
                sink.write_str(&buf)?;
            }
            Ok(())
        })
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
    /// in render order, and the LPT stitch must not disturb that order.
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
    /// gates pass, and the test asserts that `render_parallel` was actually
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
}
