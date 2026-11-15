//! Pure diagnostic formatting.
//!
//! A [`Renderer`] knows the *output format* and nothing about I/O: it returns
//! `String`s and lets the emitter decide where they go.  Document-level
//! framing (HTML `<head>`, JSON array brackets) lives in `begin`/`end` so the
//! emitter stays format-agnostic.

use crate::ast::FileId;
use crate::diagnostics::sink::DiagnosticSink;
use crate::diagnostics::term::path_to_file_url;
use crate::diagnostics::glyph::SectionScratch;
use crate::diagnostics::{DiagOpts, Diagnostic, LineIndex};
use crate::smallvec::SmallVec;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::cell::RefCell;
use std::io;
use std::sync::Arc;

/// Maximum number of distinct source files whose line tables are cached
/// across one batch (the per-thread LRU capacity).  Split from the parallel
/// preload count so the *serial* path can keep more files warm: a batch
/// alternating over 5+ files would otherwise thrash a 4-slot LRU, paying a
/// full `LineIndex::new` O(source_len) scan per miss.  The hit-path scan is a
/// `position()` over N small entries — at 16 that is trivial next to one
/// avoided scan.
pub(crate) const CACHE_SLOTS: usize = 16;
/// Largest source copied into the cross-emit stable line cache.  The cache
/// keeps a source copy so a hit can verify content equality — pointer-keyed
/// global caching would be unsound once a caller frees the buffer.
const STABLE_CACHE_MAX_SOURCE: usize = 512 * 1024;

/// How many prebuilt tables the parallel prebuild injects directly into each
/// worker's LRU.  Must not exceed [`CACHE_SLOTS`] (excess inserts would evict
/// each other); the const assertion pins the coupling that used to be a single
/// constant.  Sources beyond this reach workers via [`SharedSources`].
pub(crate) const PRELOAD_SLOTS: usize = 4;
const _: () = assert!(PRELOAD_SLOTS <= CACHE_SLOTS);

/// One `OnceLock` per file name (2.1): concurrent workers racing on the same
/// name pay one `canonicalize`, not one each.
pub(crate) type FileUrlMap = Arc<
    std::sync::RwLock<
        FxHashMap<String, Arc<std::sync::OnceLock<Arc<str>>>>,
    >,
>;

/// One cached per-line sparse table plus its last-access timestamp.
///
/// The timestamp lives behind an atomic so a cache *hit* can refresh LRU
/// recency while holding only the map's read lock.  Eviction (which needs a
/// write lock) scans for the smallest timestamp; this happens only when the
/// soft cap is reached, so the hot path stays a single atomic load/store.
#[derive(Debug)]
pub(crate) struct CharPrefixEntry {
    pub table: Option<(Arc<[(u32, u32)]>, std::sync::atomic::AtomicUsize)>,
    last_used: std::sync::atomic::AtomicU64,
}

impl CharPrefixEntry {
    pub(crate) fn new(
        table: Option<(Arc<[(u32, u32)]>, std::sync::atomic::AtomicUsize)>,
    ) -> Self {
        Self {
            table,
            last_used: std::sync::atomic::AtomicU64::new(next_char_prefix_tick()),
        }
    }

    pub(crate) fn touch(&self) {
        self.last_used
            .store(next_char_prefix_tick(), std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn last_used(&self) -> u64 {
        self.last_used.load(std::sync::atomic::Ordering::Relaxed)
    }
}

static CHAR_PREFIX_TICK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

fn next_char_prefix_tick() -> u64 {
    CHAR_PREFIX_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Shared sparse non-ASCII char-extra cache (see
/// [`LineIndex::char_prefixes_arc`]): per-line tables behind an atomic hint,
/// so a racy store can only cost a re-anchor, never a wrong answer.
/// `Send + Sync`, so the parallel path shares one map across workers.
pub(crate) type CharPrefixMap =
    std::sync::RwLock<FxHashMap<usize, CharPrefixEntry>>;

/// One cached line index, keyed by source identity `(ptr, len)`.
struct CachedSourceIndex {
    ptr: usize,
    len: usize,
    starts: Arc<[u32]>,
    ascii: bool,
    /// Shared sparse non-ASCII char-extra cache, threaded through so a cache
    /// hit reuses the per-line tables a previous render built instead of
    /// rebuilding them per diagnostic.
    char_prefixes: Arc<CharPrefixMap>,
    /// Generation id from the `LineIndex::new` that built these tables —
    /// threaded through so `from_shared` indexes keep the ABA-proof memo key.
    generation: u64,
    /// Debug-only content fingerprint, checked on every hit.  The `(ptr, len)`
    /// liveness argument is airtight (every `Diagnostic<'src>` in a batch
    /// borrows a source alive for the whole batch), but it is an invariant on
    /// callers enforced by nothing; if it ever breaks, the failure mode is
    /// *wrong line numbers with no crash*.  This converts that into a loud
    /// debug panic at the exact render that would have corrupted output.
    #[cfg(debug_assertions)]
    fingerprint: u64,
}

/// Cross-emit line-index entry.  Unlike [`CachedSourceIndex`], the key is a
/// *content* fingerprint plus the stored source text: the source buffer may be
/// freed between `emit` calls, so a pointer key would be unsound.  Equality is
/// re-checked on every hit, making hash collisions a miss rather than wrong
/// line numbers.
struct StableSourceIndex {
    text: String,
    fingerprint: u64,
    starts: Arc<[u32]>,
    ascii: bool,
    char_prefixes: Arc<CharPrefixMap>,
    generation: u64,
}

/// Cheap content fingerprint for the debug ABA tripwire: mixes the first and
/// last 8 bytes plus the length.  Two *live* same-length sources cannot share
/// a `(ptr, len)` key by the liveness argument; the fingerprint only has to
/// distinguish a cached table from a *different* source that reused the
/// allocation, which this does with overwhelming probability.  Identical
/// content (merged literals) produces the same fingerprint — correct, since
/// the tables would be identical anyway.
pub(crate) fn source_fingerprint(s: &str) -> u64 {
    let b = s.as_bytes();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |x: u8| {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for &x in b.iter().take(8) {
        feed(x);
    }
    for &x in b.iter().rev().take(8) {
        feed(x);
    }
    for x in (b.len() as u64).to_le_bytes() {
        feed(x);
    }
    h
}

/// The line-table half of a [`CachedSourceIndex`], shareable across the
/// workers of one parallel batch.
#[derive(Clone)]
struct SharedSourceIndex {
    starts: Arc<[u32]>,
    ascii: bool,
    char_prefixes: Arc<CharPrefixMap>,
    generation: u64,
}

/// Cross-worker share of line tables for one parallel batch.
///
/// The per-worker LRU holds only [`CACHE_SLOTS`] sources; sources beyond that
/// would otherwise be re-scanned O(source_len) by *every* worker.  This map is
/// built once in the main thread (or grown lazily by whichever worker first
/// renders a source) and consulted by every worker, so each distinct source
/// builds at most once per batch.  It is dropped with the batch, so a
/// `(ptr, len)` key can never outlive its source.
pub(crate) struct SharedSources {
    map: std::sync::RwLock<FxHashMap<(usize, usize), SharedSourceIndex>>,
}

impl SharedSources {
    /// Build from pre-scanned `(ptr, len, starts, ascii, char_prefixes)`
    /// entries (the shape [`LineIndex::new`] + the accessors produce).
    pub(crate) fn from_entries(
        entries: &[(
            usize,
            usize,
            Arc<[u32]>,
            bool,
            Arc<CharPrefixMap>,
            u64,
            u64,
        )],
    ) -> Arc<Self> {
        let mut map = FxHashMap::default();
        for &(ptr, len, ref starts, ascii, ref prefixes, generation, _fingerprint) in entries {
            map.insert(
                (ptr, len),
                SharedSourceIndex {
                    starts: Arc::clone(starts),
                    ascii,
                    char_prefixes: Arc::clone(prefixes),
                    generation,
                },
            );
        }
        Arc::new(SharedSources {
            map: std::sync::RwLock::new(map),
        })
    }
}

/// Scratch cell that either owns its value (a batch `RenderCtx`) or borrows a
/// longer-lived one from the `Emitter` (the single-diagnostic `emit` path), so
/// both shapes share the renderers' `cx.scratch.borrow_mut()` access pattern.
pub(crate) enum CtxCell<'a, T> {
    Owned(RefCell<T>),
    Borrowed(&'a RefCell<T>),
}

impl<T> CtxCell<'_, T> {
    pub(crate) fn borrow_mut(&self) -> std::cell::RefMut<'_, T> {
        match self {
            CtxCell::Owned(c) => c.borrow_mut(),
            CtxCell::Borrowed(c) => c.borrow_mut(),
        }
    }
}

/// Context shared by every render call in a batch.
///
/// Carries configuration, the per-batch line-index LRU, the `file://` URL
/// cache, and section scratch.  Constructed once per batch — an owned ctx
/// allocates the ~16 KB `SectionScratch` and a URL map, so "free" would be
/// false; the hoisted [`Self::hoisted`] variant borrows both from the
/// emitter for the per-call path.
pub struct RenderCtx<'a> {
    pub opts: &'a DiagOpts,
    pub uri: Option<&'a str>,
    /// Per-batch LRU of line indexes, most-recently-used first.
    ///
    /// Every `Diagnostic<'src>` in a batch borrows a source that is alive for
    /// the whole batch (`'src` outlives the call), so two distinct live
    /// sources can never share a `(ptr, len)` key: a hit always means the
    /// same source.  The cache is dropped with the ctx at the end of the
    /// batch, so no key can outlive its source.
    source_index: RefCell<SmallVec<[CachedSourceIndex; CACHE_SLOTS]>>,
    /// Cross-emit content-verified cache, present only on the hoisted
    /// single-diagnostic path.  Entries carry a source copy so a hit cannot
    /// alias a freed/reused allocation.
    line_cache: Option<&'a RefCell<SmallVec<[StableSourceIndex; CACHE_SLOTS]>>>,
    /// Per-batch cache of `file://` URLs keyed by filename, so a batch of M
    /// diagnostics over one file pays one `canonicalize` (3–6 `stat`
    /// syscalls) instead of M.  Only touched on the interactive hyperlink
    /// path; machine formats never call [`Self::file_url`].  An
    /// `Arc<RwLock<…>>` so the workers of one parallel batch share a single
    /// cache — `canonicalize` runs once per batch per file, not once per
    /// worker per file (uncontended, the lock costs two atomic ops).
    file_urls: FileUrlMap,
    /// Per-batch scratch buffers reused across every `render_diagnostic_into`
    /// call made through this ctx, so the 13 `SectionScratch` buffers
    /// (`rendered`, `row_buf`, `connector`, `lanes`, `ranges_buf`, …) grow
    /// once on the first section and are reused by every later diagnostic
    /// instead of being dropped and reallocated per render.  Borrowed from a
    /// hoisted [`SharedRenderState`] on the single-diagnostic `emit` path
    /// ([`Self::hoisted`]); owned otherwise.  The render path is
    /// single-threaded per ctx and the borrow lives only for one render; it
    /// touches only this cell, so the `source_index`/`file_urls`/`shared`
    /// borrows the foreign-file path makes stay conflict-free.
    pub(crate) scratch: CtxCell<'a, SectionScratch>,
    /// Optional cross-worker share of line tables for a parallel batch (see
    /// [`SharedSources`]).  `None` on the serial path.
    shared: Option<Arc<SharedSources>>,
}

impl<'a> RenderCtx<'a> {
    /// Build a batch context.  One per batch: the per-source line index is
    /// reused across every `render` call made through this ctx.
    pub fn new(opts: &'a DiagOpts, uri: Option<&'a str>) -> Self {
        Self {
            opts,
            uri,
            source_index: RefCell::new(SmallVec::new()),
            line_cache: None,
            file_urls: Arc::default(),
            scratch: CtxCell::Owned(RefCell::new(SectionScratch::new())),
            shared: None,
        }
    }

    /// Build a batch ctx whose `file://` URL cache and section scratch are
    /// hoisted into `shared` and survive across calls — the single-diagnostic
    /// [`crate::diagnostics::emitter::Emitter::emit`] path, which would
    /// otherwise pay a fresh 16 KB scratch allocation and a full
    /// `canonicalize` per diagnostic.  The per-source line-index LRU stays
    /// per-call on purpose: its `(ptr, len)` keys are only valid while the
    /// sources are alive, and a batch ctx is dropped with its batch.
    pub(crate) fn hoisted(
        opts: &'a DiagOpts,
        uri: Option<&'a str>,
        shared: &'a SharedRenderState,
    ) -> Self {
        Self {
            opts,
            uri,
            source_index: RefCell::new(SmallVec::new()),
            line_cache: Some(&shared.line_cache),
            file_urls: Arc::clone(&shared.file_urls),
            scratch: CtxCell::Borrowed(&shared.scratch),
            shared: None,
        }
    }

    /// The line index for `source`, built at most once per source per batch.
    ///
    /// The returned index borrows only `source`, so callers can hold it for
    /// the duration of a render without borrowing the ctx.
    pub(crate) fn line_index<'s>(&self, source: &'s str) -> LineIndex<'s> {
        let ptr = source.as_ptr() as usize;
        let len = source.len();
        {
            let mut cache = self.source_index.borrow_mut();
            if let Some(pos) = cache.iter().position(|c| c.ptr == ptr && c.len == len) {
                // Debug ABA tripwire: a hit means the same `(ptr, len)`, but if
                // the cached table belonged to a *different* source that
                // reused the allocation, every line number would be silently
                // wrong.  `0` is the "unknown" sentinel set by `preload`
                // (which has no source text).
                #[cfg(debug_assertions)]
                if cache[pos].fingerprint != 0 {
                    debug_assert_eq!(
                        cache[pos].fingerprint,
                        source_fingerprint(source),
                        "(ptr,len) collision — cached line table belongs to a different source"
                    );
                }
                // LRU: promote the hit to MRU so it survives eviction longest.
                // N ≤ 4, so the shifts are a handful of pointer moves.
                if pos > 0 {
                    let entry = cache.remove(pos);
                    cache.insert(0, entry);
                }
                return LineIndex::from_shared(
                    source,
                    Arc::clone(&cache[0].starts),
                    cache[0].ascii,
                    Arc::clone(&cache[0].char_prefixes),
                    cache[0].generation,
                );
            }
        }
        // Cross-worker share: a hit means some other worker (or the main
        // thread's prebuild) already paid the O(source_len) scan.  Backfill
        // the local LRU with the shared entry, so subsequent renders of the
        // same source stop taking the shared `RwLock` read — with more than
        // `CACHE_SLOTS` distinct sources per worker this is the difference
        // between one lock acquisition per source and one per render.
        if let Some(shared) = &self.shared {
            // Poison is impossible-in-practice and harmless here: the guard
            // only ever spans infallible map operations, so a poisoned map's
            // contents are still valid.  Recovering instead of `expect` keeps
            // one panicking render from wedging the emitter for the rest of
            // the process.
            if let Some(e) = shared
                .map
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .get(&(ptr, len))
            {
                let mut cache = self.source_index.borrow_mut();
                if cache.len() >= CACHE_SLOTS {
                    cache.pop();
                }
                // Insert first, then build the returned index from the cached
                // entry, so the hit path clones two `Arc`s instead of four.
                cache.insert(
                    0,
                    CachedSourceIndex {
                        ptr,
                        len,
                        starts: Arc::clone(&e.starts),
                        ascii: e.ascii,
                        char_prefixes: Arc::clone(&e.char_prefixes),
                        generation: e.generation,
                        #[cfg(debug_assertions)]
                        fingerprint: source_fingerprint(source),
                    },
                );
                let e = &cache[0];
                return LineIndex::from_shared(
                    source,
                    Arc::clone(&e.starts),
                    e.ascii,
                    Arc::clone(&e.char_prefixes),
                    e.generation,
                );
            }
        }
        if len <= STABLE_CACHE_MAX_SOURCE {
            if let Some(cache_ref) = self.line_cache {
                let fingerprint = source_fingerprint(source);
                let mut cache = cache_ref.borrow_mut();
                if let Some(pos) = cache.iter().position(|entry| {
                    entry.fingerprint == fingerprint
                        && entry.text.len() == len
                        && entry.text == source
                }) {
                    if pos > 0 {
                        let entry = cache.remove(pos);
                        cache.insert(0, entry);
                    }
                    let entry = &cache[0];
                    return LineIndex::from_shared(
                        source,
                        Arc::clone(&entry.starts),
                        entry.ascii,
                        Arc::clone(&entry.char_prefixes),
                        entry.generation,
                    );
                }
            }
        }
        let idx = LineIndex::new(source);
        // Publish to the cross-worker share (if any) before the LRU insert so
        // a worker rendering a 5th+ distinct source still shares its scan.
        if let Some(shared) = &self.shared {
            shared
                .map
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .entry((ptr, len))
                .or_insert_with(|| SharedSourceIndex {
                    starts: Arc::clone(idx.starts_arc()),
                    ascii: idx.ascii(),
                    char_prefixes: Arc::clone(idx.char_prefixes_arc()),
                    generation: idx.generation(),
                });
        }
        let mut cache = self.source_index.borrow_mut();
        // Full: evict the LRU slot (the tail) to keep the cache bounded.
        if cache.len() >= CACHE_SLOTS {
            cache.pop();
        }
        cache.insert(
            0,
            CachedSourceIndex {
                ptr,
                len,
                starts: Arc::clone(idx.starts_arc()),
                ascii: idx.ascii(),
                char_prefixes: Arc::clone(idx.char_prefixes_arc()),
                generation: idx.generation(),
                #[cfg(debug_assertions)]
                fingerprint: source_fingerprint(source),
            },
        );
        if len <= STABLE_CACHE_MAX_SOURCE {
            if let Some(cache_ref) = self.line_cache {
                let mut cache = cache_ref.borrow_mut();
                if cache.len() >= CACHE_SLOTS {
                    cache.pop();
                }
                cache.insert(
                    0,
                    StableSourceIndex {
                        text: source.to_owned(),
                        fingerprint: source_fingerprint(source),
                        starts: Arc::clone(idx.starts_arc()),
                        ascii: idx.ascii(),
                        char_prefixes: Arc::clone(idx.char_prefixes_arc()),
                        generation: idx.generation(),
                    },
                );
            }
        }
        idx
    }

    /// Attach the cross-worker line-table share for a parallel batch (see
    /// [`SharedSources`]).  After this, [`Self::line_index`] consults the
    /// shared map before building, and publishes anything it builds itself.
    pub(crate) fn set_shared_sources(&mut self, shared: Arc<SharedSources>) {
        self.shared = Some(shared);
    }

    /// One-shot constructor for a parallel worker context, wiring the shared
    /// line tables, the shared URL cache, and the preloaded LRU in a single
    /// call.  This exists so the three `set_*`/`preload` steps cannot be
    /// forgotten independently — a worker that got `set_shared_sources` but
    /// not `set_file_urls` still compiles and runs, it just silently pays a
    /// `canonicalize` per file (the failure mode that already fired once).
    pub(crate) fn for_parallel_worker(
        opts: &'a DiagOpts,
        uri: Option<&'a str>,
        shared_sources: Arc<SharedSources>,
        file_urls: FileUrlMap,
        preload: &[(usize, usize, Arc<[u32]>, bool, Arc<CharPrefixMap>, u64, u64)],
    ) -> Self {
        let mut cx = Self::new(opts, uri);
        cx.set_shared_sources(shared_sources);
        cx.set_file_urls(file_urls);
        cx.preload_line_index(preload);
        cx
    }

    /// Replace the `file://` URL cache with a caller-supplied (shared) map —
    /// used by the parallel render path so all workers share one
    /// `canonicalize` per file per batch.
    pub(crate) fn set_file_urls(
        &mut self,
        urls: FileUrlMap,
    ) {
        self.file_urls = urls;
    }

    /// Seed the cache with pre-built line tables, keyed by source identity
    /// like the LRU above.
    ///
    /// Used by the parallel render path: the *same* source is scanned once
    /// in the main thread and every worker then injects the shared
    /// `Arc<[u32]>` line-start table instead of each re-running the
    /// O(source_len) SWAR scan.  Each entry also shares one `char_prefixes`
    /// map (the hint is an `AtomicUsize` and a racy store can only cost a
    /// re-anchor, never a wrong answer), so a non-ASCII source's per-line
    /// sparse tables are built once per *batch*, not once per thread.
    /// Entries are inserted MRU-first from the tail of `entries`
    /// so the caller can pass its list in natural order.
    ///
    /// Order matters for hit rate: the seeded slots are the first `entries`,
    /// so a `reverse_errors` batch should pass its list in *render* order or
    /// the first renders miss and backfill from the shared map.  `&mut self`
    /// (not `&self`) makes a mid-batch reseed a compile error — the pop-tail
    /// eviction below is live, so reseeding between renders would silently
    /// evict warm tables.
    pub(crate) fn preload_line_index(
        &mut self,
        entries: &[(usize, usize, Arc<[u32]>, bool, Arc<CharPrefixMap>, u64, u64)],
    ) {
        let mut cache = self.source_index.borrow_mut();
        for &(ptr, len, ref starts, ascii, ref prefixes, generation, fingerprint) in entries.iter().rev() {
            if cache.len() >= CACHE_SLOTS {
                cache.pop();
            }
            cache.insert(
                0,
                CachedSourceIndex {
                    ptr,
                    len,
                    starts: Arc::clone(starts),
                    ascii,
                    char_prefixes: Arc::clone(prefixes),
                    generation,
                    #[cfg(debug_assertions)]
                    fingerprint,
                },
            );
        }
    }

    /// The `file://` URL for `name`, cached per batch so a batch of M
    /// diagnostics over one file pays one `canonicalize` instead of M.
    ///
    /// Callers gate this on `use_color && supports_hyperlinks() &&
    /// is_real_file_name(name)`; the cache itself is unconditional so the
    /// hit/miss logic stays here and the caller stays mechanical.
    pub(crate) fn file_url(&self, name: &str) -> Arc<str> {
        // 2.1: one `OnceLock` per name.  The read lock covers the hit path and
        // the write lock is held only to obtain/insert the slot, never across
        // `canonicalize`; the per-name `OnceLock` serializes the expensive
        // syscall so a batch with W workers and F files pays F calls, not W·F.
        let slot = {
            let map = self
                .file_urls
                .read()
                .unwrap_or_else(|e| e.into_inner());
            map.get(name).map(Arc::clone)
        };
        let slot = match slot {
            Some(slot) => slot,
            None => {
                let mut map = self
                    .file_urls
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                Arc::clone(map.entry(name.to_string()).or_default())
            }
        };
        Arc::clone(slot.get_or_init(|| {
            Arc::from(path_to_file_url(std::path::Path::new(name)).as_str())
        }))
    }
}

/// Long-lived render state an [`crate::diagnostics::emitter::Emitter`] hoists
/// out of its per-call [`RenderCtx`]s: the `file://` URL cache and the
/// section scratch.  Both are keyed by / sized for content that stays valid
/// across calls (a name→URL mapping; grow-once buffers), unlike the
/// per-source line-index LRU whose `(ptr, len)` keys are only sound while
/// the sources are alive — that one stays per-batch.
pub struct SharedRenderState {
    pub(crate) file_urls: FileUrlMap,
    pub(crate) scratch: RefCell<SectionScratch>,
    /// Content-verified line tables that survive across `emit` calls.  Bounded
    /// by `CACHE_SLOTS`; entries over `STABLE_CACHE_MAX_SOURCE` are not copied.
    pub(crate) line_cache: RefCell<SmallVec<[StableSourceIndex; CACHE_SLOTS]>>,
    /// Grow-once scratch buffer for the single-diagnostic [`Emitter::emit`]
    /// path: the rendered text is copied into the sink and dropped, so one
    /// reused `String` avoids a heap allocation/free per diagnostic in an
    /// LSP-style re-emit loop.  Cleared before each render.
    pub(crate) render_buf: RefCell<String>,
}

impl SharedRenderState {
    pub(crate) fn new() -> Self {
        Self {
            file_urls: Arc::default(),
            scratch: RefCell::new(SectionScratch::new()),
            line_cache: RefCell::new(SmallVec::new()),
            render_buf: RefCell::new(String::new()),
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// The whole point of the cache: the second render of the same source
    /// reuses the first index's table (an `Arc` clone), not a rebuild.
    #[test]
    fn line_index_is_reused_for_the_same_source() {
        let opts = DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);

        let src = "a\nb\nc\nd\n";
        let first = cx.line_index(src);
        let second = cx.line_index(src);
        assert!(
            Arc::ptr_eq(first.starts_arc(), second.starts_arc()),
            "same source must reuse the cached line table",
        );

        // A different source is a different table — and, since the cache is
        // multi-slot, it does *not* evict the existing one.
        let other = "x\ny\n";
        let third = cx.line_index(other);
        assert!(
            !Arc::ptr_eq(first.starts_arc(), third.starts_arc()),
            "a different source must not alias the cached table",
        );
    }

    #[test]
    fn hoisted_line_index_is_reused_across_contexts() {
        let opts = DiagOpts::default();
        let shared = SharedRenderState::new();
        let src = "a\nb\nc\n";
        let first = {
            let cx = RenderCtx::hoisted(&opts, None, &shared);
            Arc::clone(cx.line_index(src).starts_arc())
        };
        let second = {
            let cx = RenderCtx::hoisted(&opts, None, &shared);
            Arc::clone(cx.line_index(src).starts_arc())
        };
        assert!(
            Arc::ptr_eq(&first, &second),
            "hoisted ctx must reuse the content-verified line table",
        );

        let changed = "a\nB\nc\n";
        let third = {
            let cx = RenderCtx::hoisted(&opts, None, &shared);
            Arc::clone(cx.line_index(changed).starts_arc())
        };
        assert!(
            !Arc::ptr_eq(&first, &third),
            "changed content must not alias the cached table",
        );
    }

    /// Multi-file alternation must keep every warm source in the cache — the
    /// exact case the old single slot rebuilt on every switch.
    #[test]
    fn line_index_keeps_multiple_sources_warm() {
        let opts = DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);

        let a = "a\nb\nc\n";
        let b = "x\ny\nz\n";

        let a1 = cx.line_index(a);
        let b1 = cx.line_index(b);
        // Alternate several times; every hit should reuse the primed table.
        let a2 = cx.line_index(a);
        let b2 = cx.line_index(b);
        let a3 = cx.line_index(a);
        let b3 = cx.line_index(b);

        assert!(Arc::ptr_eq(a1.starts_arc(), a2.starts_arc()));
        assert!(Arc::ptr_eq(a1.starts_arc(), a3.starts_arc()));
        assert!(Arc::ptr_eq(b1.starts_arc(), b2.starts_arc()));
        assert!(Arc::ptr_eq(b1.starts_arc(), b3.starts_arc()));
        assert!(!Arc::ptr_eq(a1.starts_arc(), b1.starts_arc()));
    }

    /// Exceeding `CACHE_SLOTS` distinct sources evicts the LRU one.  Pins
    /// the eviction *policy*, not just the presence of a multi-slot cache.
    #[test]
    fn line_index_evicts_the_least_recently_used() {
        let opts = DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);

        // Distinct lengths guarantee distinct `(ptr, len)` keys even if the
        // compiler deduplicates equal-length literals.
        let sources: Vec<String> = (0..=CACHE_SLOTS).map(|i| "a\n".repeat(i + 1)).collect();

        let first: Vec<LineIndex<'_>> = sources[..CACHE_SLOTS]
            .iter()
            .map(|s| cx.line_index(s))
            .collect();

        // Insert one more — `sources[0]` is the LRU and must be evicted.
        let extra = cx.line_index(&sources[CACHE_SLOTS]);
        assert!(
            !first
                .iter()
                .any(|f| Arc::ptr_eq(f.starts_arc(), extra.starts_arc())),
            "new source must not alias any existing cached table"
        );

        // `sources[1]` is still warm (MRU slot after the insert).
        let refetch1 = cx.line_index(&sources[1]);
        assert!(
            Arc::ptr_eq(first[1].starts_arc(), refetch1.starts_arc()),
            "sources[1] should still be cached"
        );

        // `sources[0]` was evicted: rebuilding yields a fresh `Arc`.
        let refetch0 = cx.line_index(&sources[0]);
        assert!(
            !Arc::ptr_eq(first[0].starts_arc(), refetch0.starts_arc()),
            "sources[0] should have been evicted"
        );
    }

    /// The cache must never change the answer: a supplied index renders
    /// exactly like one built locally.
    #[test]
    fn supplied_index_matches_local_build() {
        use crate::ast::Span;
        use crate::diagnostics::glyph::GlyphRenderer;

        let src = "def a(x: Bool)\n  { set j = \"0xFFFF\"; return j; }\ndef main(){}\n";
        let diag = crate::diagnostics::Diagnostic::error("boom")
            .with_span(Span::new(FileId::DUMMY, 4, 9))
            .with_source(src)
            .with_file_name("<input>");

        let renderer = GlyphRenderer::new(false);
        let opts = DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);
        let idx = cx.line_index(src);

        let local = renderer.render_diagnostic(&diag);
        let cached = renderer.render_diagnostic_with_index(&diag, Some(&idx));
        assert_eq!(local, cached);
    }
}

/// Formats diagnostics into strings.  Pure: no I/O.
///
/// `Sync` (not just `Send`) lets a batch be rendered on several scoped
/// threads that share `&R` — each thread carries its own `RenderCtx`, so no
/// renderer-internal state is ever mutated concurrently.
pub trait Renderer: Send + Sync {
    /// Called once per batch, before any diagnostics.
    ///
    /// `&mut dyn DiagnosticSink` (not `impl DiagnosticSink`) keeps `Renderer`
    /// dyn-compatible, so a runtime format selection can use
    /// `Box<dyn Renderer>`.  The vtable dispatch happens twice per batch —
    /// immeasurable against the render work.
    fn begin(&self, out: &mut dyn DiagnosticSink) -> io::Result<()> {
        let _ = out;
        Ok(())
    }

    /// Render one diagnostic.  No sink I/O; may `stat` the filesystem for
    /// hyperlink targets (`file_url`), cached once per file per batch.
    fn render(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String;

    /// Render one diagnostic into a caller-supplied buffer.
    ///
    /// The default appends [`render`](Self::render); renderers that already
    /// build into an internal buffer override this to skip the intermediate
    /// `String`, so the emitter's batch loop reuses one scratch buffer across
    /// the whole batch instead of allocating one `String` per diagnostic.
    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
        out.push_str(&self.render(diag, cx));
    }

    /// Text written between two adjacent rendered diagnostics in a batch.
    ///
    /// The human renderer inserts a blank line so successive diagnostics do
    /// not run together; JSON inserts a `,` (or `,\n` in pretty mode) so the
    /// array stays well-formed; HTML leaves the default empty separator.
    fn render_separator(&self) -> &str {
        ""
    }

    /// Render the summary line.  Empty string means "no summary".
    fn render_summary(&self, errors: usize, warnings: usize) -> String;

    /// Render the summary line with a distinct-file count for the
    /// "N errors in M files" overview (checklist item 8).
    ///
    /// Default impl ignores `files` and delegates to [`render_summary`](Self::render_summary)
    /// so existing renderers keep working; human/HTML overrides include
    /// the file count while JSON keeps the empty summary (machine-consumed).
    fn render_summary_full(&self, errors: usize, warnings: usize, files: usize) -> String {
        let _ = files;
        self.render_summary(errors, warnings)
    }

    /// Render the "N errors suppressed" trailer when `max_errors` cut the
    /// batch short.  Empty string means "emit nothing".  Formats that are
    /// machine-consumed (JSON, LSP) override this to `""` so the trailer
    /// cannot corrupt the document.
    ///
    /// Routed through the i18n machinery so the count gets the locale's CLDR
    /// plural categories (Russian has one/few/many, not a binary —s).
    fn render_suppressed_note(&self, count: usize) -> String {
        let mut s = String::new();
        s.push_str("  = note: ");
        crate::diagnostics::i18n::render_into(
            "suppressed-note",
            &[("count", Cow::Owned(count.to_string()))],
            &mut s,
        );
        s.push('\n');
        s
    }

    /// Called once per batch, after all diagnostics and the summary.
    fn end(&self, out: &mut dyn DiagnosticSink) -> io::Result<()> {
        let _ = out;
        Ok(())
    }
}
