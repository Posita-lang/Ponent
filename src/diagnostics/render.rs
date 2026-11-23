//! Pure diagnostic formatting.
//!
//! A [`Renderer`] knows the *output format* and nothing about I/O: it returns
//! `String`s and lets the emitter decide where they go.  Document-level
//! framing (HTML `<head>`, JSON array brackets) lives in `begin`/`end` so the
//! emitter stays format-agnostic.

use crate::ast::FileId;
use crate::diagnostics::glyph::SectionScratch;
use crate::diagnostics::sink::DiagnosticSink;
use crate::diagnostics::term::path_to_file_url;
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
pub(crate) type FileUrlMap =
    Arc<std::sync::RwLock<FxHashMap<String, Arc<std::sync::OnceLock<Arc<str>>>>>>;

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
    pub(crate) fn new(table: Option<(Arc<[(u32, u32)]>, std::sync::atomic::AtomicUsize)>) -> Self {
        Self {
            table,
            last_used: std::sync::atomic::AtomicU64::new(next_char_prefix_tick()),
        }
    }

    pub(crate) fn touch(&self) {
        self.last_used.store(
            next_char_prefix_tick(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    pub(crate) fn last_used(&self) -> u64 {
        self.last_used.load(std::sync::atomic::Ordering::Relaxed)
    }
}

static CHAR_PREFIX_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_char_prefix_tick() -> u64 {
    CHAR_PREFIX_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Shared sparse non-ASCII char-extra cache (see
/// [`LineIndex::char_prefixes_arc`]): per-line tables behind an atomic hint,
/// so a racy store can only cost a re-anchor, never a wrong answer.
/// `Send + Sync`, so the parallel path shares one map across workers.
pub(crate) type CharPrefixMap = std::sync::RwLock<FxHashMap<usize, CharPrefixEntry>>;

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
    // Middle sample: first/last-8-only fingerprints collide for two
    // same-length sources that share a header/footer and differ in the body.
    let mid = b.len() / 2;
    for &x in b.iter().skip(mid).take(8) {
        feed(x);
    }
    for x in (b.len() as u64).to_le_bytes() {
        feed(x);
    }
    h
}

#[derive(Clone)]
pub(crate) struct PreloadedSource {
    pub ptr: usize,
    pub len: usize,
    pub starts: Arc<[u32]>,
    pub ascii: bool,
    pub char_prefixes: Arc<CharPrefixMap>,
    pub generation: u64,
    pub fingerprint: u64,
}

/// The line-table half of a [`CachedSourceIndex`], shareable across the
/// workers of one parallel batch.
#[derive(Clone)]
struct SharedSourceIndex {
    starts: Arc<[u32]>,
    ascii: bool,
    char_prefixes: Arc<CharPrefixMap>,
    generation: u64,
    #[cfg(debug_assertions)]
    fingerprint: u64,
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
    pub(crate) fn from_entries(entries: &[PreloadedSource]) -> Arc<Self> {
        let mut map = FxHashMap::default();
        for e in entries {
            map.insert(
                (e.ptr, e.len),
                SharedSourceIndex {
                    starts: Arc::clone(&e.starts),
                    ascii: e.ascii,
                    char_prefixes: Arc::clone(&e.char_prefixes),
                    generation: e.generation,
                    #[cfg(debug_assertions)]
                    fingerprint: e.fingerprint,
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
        let cell = match self {
            CtxCell::Owned(c) => c,
            CtxCell::Borrowed(c) => c,
        };
        // The section scratch is borrowed for exactly one render; a nested
        // render (foreign-file recursion) reaching it again is the invariant
        // violation.  `RefCell`'s generic "already borrowed" panic forces the
        // next person to bisect the borrow sites, so name the invariant here
        // (the check itself is the same flag test either way).
        cell.try_borrow_mut().unwrap_or_else(|_| {
            panic!(
                "section scratch re-entered: a renderer held the scratch borrow across a \
                 nested render"
            )
        })
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
    /// Per-batch display-column table for the location header's and the note
    /// locator's `byte offset → column` conversion, keyed by
    /// `(line_no, tab_width)`.
    ///
    /// That conversion walks the line prefix grapheme by grapheme, and a
    /// diagnostic performs it once for the header plus once per note locator.
    /// Measured on a 200 KB single line it costs ~55 ms per call in a debug
    /// build — roughly a hundred times a whole diagnostic's render — so one
    /// table per line per batch is what makes long-line sources tractable.  A
    /// single slot suffices: a render's sections sit on the same line range.
    line_columns: RefCell<
        Option<(
            usize,
            usize,
            usize,
            usize,
            crate::diagnostics::glyph::LineColumns,
        )>,
    >,
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
    /// Content-addressed tables prebuilt for this batch, consulted *before* the
    /// per-ctx LRU.  The serial path fills it opportunistically once
    /// `Emitter::file_index_cache` holds content keys: an LSP re-publish is the
    /// same file in a *new* source allocation, so all three `(ptr, len)` tiers
    /// miss and the whole file would be rescanned for every publish.
    /// Sorted by `(ptr, len)` at every construction site, so the tier-0 lookup
    /// is `binary_search` instead of a linear `find`.  The emitter's
    /// content-addressed cache grows across LSP publishes, so `P` can reach
    /// hundreds and every `line_index` call would otherwise pay the scan.
    prebuilt: Option<&'a [PreloadedSource]>,
}

impl<'a> RenderCtx<'a> {
    /// Build a batch context.  One per batch: the per-source line index is
    /// reused across every `render` call made through this ctx.
    pub fn new(opts: &'a DiagOpts, uri: Option<&'a str>) -> Self {
        Self {
            opts,
            uri,
            source_index: RefCell::new(SmallVec::new()),
            line_columns: RefCell::new(None),
            line_cache: None,
            file_urls: Arc::default(),
            scratch: CtxCell::Owned(RefCell::new(SectionScratch::new())),
            shared: None,
            prebuilt: None,
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
            line_columns: RefCell::new(None),
            line_cache: Some(&shared.line_cache),
            file_urls: Arc::clone(&shared.file_urls),
            scratch: CtxCell::Borrowed(&shared.scratch),
            shared: None,
            prebuilt: None,
        }
    }

    /// [`Self::hoisted`] plus the content-addressed line tables `Emitter`
    /// prebuilt for this batch (see the `prebuilt` field).  Byte output is
    /// unaffected: the tables are exactly what `LineIndex::new` would build.
    pub(crate) fn hoisted_with_prebuilt(
        opts: &'a DiagOpts,
        uri: Option<&'a str>,
        shared: &'a SharedRenderState,
        prebuilt: &'a [PreloadedSource],
    ) -> Self {
        Self {
            opts,
            uri,
            source_index: RefCell::new(SmallVec::new()),
            line_columns: RefCell::new(None),
            line_cache: Some(&shared.line_cache),
            file_urls: Arc::clone(&shared.file_urls),
            scratch: CtxCell::Borrowed(&shared.scratch),
            shared: None,
            prebuilt: Some(prebuilt),
        }
    }

    /// Display column of `byte_in_line` inside `line` (whose 0-based line number
    /// is `line_no`), sharing one [`LineColumns`] table per `(line, tab width)`
    /// for the whole batch.
    ///
    /// Falls back to the one-shot walk only if the table cannot be built, which
    /// it always can; the fallback exists so a future line representation can
    /// decline without changing call sites.
    pub(crate) fn display_col_in_line(
        &self,
        line: &str,
        line_no: usize,
        byte_in_line: usize,
        tab_width: usize,
    ) -> usize {
        // The key must include *which* line, not merely its number.  A batch ctx
        // renders diagnostics over several files, so "line 5" of file A and
        // "line 5" of file B used to hit the same slot and the second one got the
        // first one's tab-expanded columns — silent, user-visible wrong headers.
        // `(ptr, len)` of the line slice identifies the buffer (all sources are
        // alive for the batch, the same invariant `line_index` already relies on),
        // and a re-published file gets a new allocation, hence a new pointer.
        let line_ptr = line.as_ptr() as usize;
        let line_len = line.len();
        let mut slot = self.line_columns.borrow_mut();
        if let Some((p, l, no, tab, cols)) = slot.as_ref()
            && *p == line_ptr
            && *l == line_len
            && *no == line_no
            && *tab == tab_width
        {
            return cols.display(byte_in_line);
        }
        let cols = crate::diagnostics::glyph::LineColumns::new(line, tab_width);
        let col = cols.display(byte_in_line);
        *slot = Some((line_ptr, line_len, line_no, tab_width, cols));
        col
    }

    /// The line index for `source`, built at most once per source per batch.
    ///
    /// The returned index borrows only `source`, so callers can hold it for
    /// the duration of a render without borrowing the ctx.
    pub(crate) fn line_index<'s>(&self, source: &'s str) -> LineIndex<'s> {
        let ptr = source.as_ptr() as usize;
        let len = source.len();
        // Tier 0: tables prebuilt from the content-addressed cache.  The pair is
        // exact for this batch — the prebuild recorded the pointer of the very
        // source string being rendered — *provided* the prebuilt list is this
        // batch's.  It is cross-batch data keyed by pointer, so a stale list can
        // see an address the allocator recycled for a different source; that is
        // the "wrong line numbers, no crash" failure mode this cache stack is
        // built to exclude.
        //
        // The fingerprint is therefore verified in **every** profile, not only
        // under `debug_assertions`: ~24 sampled bytes plus the length against an
        // avoided O(source_len) rescan.  A mismatch is treated as a miss (the
        // LRU/shared/rebuild path is always correct).
        //
        // The old condition also accepted `e.fingerprint == 0`, the "unknown"
        // sentinel `preload_line_index` documents for tables built without source
        // text.  That bypass voided this tier's central invariant: a recycled
        // allocation at the same `(ptr, len)` behind a sentinel entry returned
        // silently wrong line numbers in release.  Every *prebuilt* entry is built
        // from the source text (`source_fingerprint(source)` at the construction
        // sites), so a sentinel here is a producer bug: loud in debug, a miss
        // (never wrong output) in release.
        if let Some(prebuilt) = self.prebuilt
            && let Ok(pos) = prebuilt.binary_search_by_key(&(ptr, len), |e| (e.ptr, e.len))
            && let Some(e) = prebuilt.get(pos)
            && e.fingerprint != 0
            && e.fingerprint == source_fingerprint(source)
        {
            // Backfill the per-ctx LRU: without this every render re-pays the
            // tier-0 `find` plus the release-mode fingerprint, while an LRU hit is
            // an O(4) scan with no fingerprint at all.  The entry is already
            // content-verified here, so the LRU's tripwire invariant holds.
            {
                let mut cache = self.source_index.borrow_mut();
                if let Some(pos) = cache.iter().position(|c| c.ptr == ptr && c.len == len) {
                    cache.remove(pos);
                } else if cache.len() >= CACHE_SLOTS {
                    cache.pop();
                }
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
                        fingerprint: e.fingerprint,
                    },
                );
            }
            return LineIndex::from_shared(
                source,
                Arc::clone(&e.starts),
                e.ascii,
                Arc::clone(&e.char_prefixes),
                e.generation,
            );
        }
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
                #[cfg(debug_assertions)]
                let fingerprint = {
                    let fp = source_fingerprint(source);
                    if e.fingerprint != 0 {
                        debug_assert_eq!(
                            e.fingerprint, fp,
                            "SharedSources entry belongs to a different source — \
                             batch-scoped lifetime violated"
                        );
                    }
                    fp
                };
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
                        fingerprint,
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
                    let (starts, ascii, char_prefixes, generation) = (
                        Arc::clone(&entry.starts),
                        entry.ascii,
                        Arc::clone(&entry.char_prefixes),
                        entry.generation,
                    );
                    drop(cache);
                    // Backfill the per-ctx LRU: without this, every later render
                    // of the same source re-pays this whole tier walk — including
                    // the O(len) `entry.text == source` memcmp, up to 512 KB per
                    // render — before reaching this branch again.  The entry is
                    // content-verified above, so the LRU tripwire invariant holds.
                    {
                        let mut lru = self.source_index.borrow_mut();
                        if let Some(pos) = lru.iter().position(|c| c.ptr == ptr && c.len == len) {
                            lru.remove(pos);
                        } else if lru.len() >= CACHE_SLOTS {
                            lru.pop();
                        }
                        lru.insert(
                            0,
                            CachedSourceIndex {
                                ptr,
                                len,
                                starts: Arc::clone(&starts),
                                ascii,
                                char_prefixes: Arc::clone(&char_prefixes),
                                generation,
                                #[cfg(debug_assertions)]
                                fingerprint,
                            },
                        );
                    }
                    return LineIndex::from_shared(
                        source,
                        starts,
                        ascii,
                        char_prefixes,
                        generation,
                    );
                }
            }
        }
        let idx = LineIndex::new(source);
        // Publish to the cross-worker share (if any) before the LRU insert so
        // a worker rendering a 5th+ distinct source still shares its scan.
        // The losing racer must use the winner's Arcs; keeping its own would
        // retain a second full `starts`/prefix table until LRU eviction.
        let shared_entry = if let Some(shared) = &self.shared {
            let mut map = shared.map.write().unwrap_or_else(|e| e.into_inner());
            let e = map.entry((ptr, len)).or_insert_with(|| SharedSourceIndex {
                starts: Arc::clone(idx.starts_arc()),
                ascii: idx.ascii(),
                char_prefixes: Arc::clone(idx.char_prefixes_arc()),
                generation: idx.generation(),
                #[cfg(debug_assertions)]
                fingerprint: source_fingerprint(source),
            });
            (
                Arc::clone(&e.starts),
                e.ascii,
                Arc::clone(&e.char_prefixes),
                e.generation,
            )
        } else {
            (
                Arc::clone(idx.starts_arc()),
                idx.ascii(),
                Arc::clone(idx.char_prefixes_arc()),
                idx.generation(),
            )
        };
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
                starts: Arc::clone(&shared_entry.0),
                ascii: shared_entry.1,
                char_prefixes: Arc::clone(&shared_entry.2),
                generation: shared_entry.3,
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
                        starts: Arc::clone(&shared_entry.0),
                        ascii: shared_entry.1,
                        char_prefixes: Arc::clone(&shared_entry.2),
                        generation: shared_entry.3,
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
        preload: &[PreloadedSource],
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
    pub(crate) fn set_file_urls(&mut self, urls: FileUrlMap) {
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
    pub(crate) fn preload_line_index(&mut self, entries: &[PreloadedSource]) {
        // The documented coupling (`PRELOAD_SLOTS <= CACHE_SLOTS`, excess inserts
        // would evict each other) is now executed: without this, an emitter
        // passing more than the seeded slots loses the tail entries silently.
        debug_assert!(
            entries.len() <= PRELOAD_SLOTS,
            "preload passed {} entries but only {PRELOAD_SLOTS} seeded slots exist — \
             the tail would be evicted before the first render",
            entries.len()
        );
        let mut cache = self.source_index.borrow_mut();
        for e in entries.iter().rev() {
            if cache.len() >= CACHE_SLOTS {
                cache.pop();
            }
            cache.insert(
                0,
                CachedSourceIndex {
                    ptr: e.ptr,
                    len: e.len,
                    starts: Arc::clone(&e.starts),
                    ascii: e.ascii,
                    char_prefixes: Arc::clone(&e.char_prefixes),
                    generation: e.generation,
                    #[cfg(debug_assertions)]
                    fingerprint: e.fingerprint,
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
            let map = self.file_urls.read().unwrap_or_else(|e| e.into_inner());
            map.get(name).map(Arc::clone)
        };
        let slot = match slot {
            Some(slot) => slot,
            None => {
                let mut map = self.file_urls.write().unwrap_or_else(|e| e.into_inner());
                // Re-check under the write lock: a racing worker may have
                // inserted the slot between the read-lock miss and here, and in
                // that case the `to_string()` key would be built and dropped.
                if let Some(slot) = map.get(name) {
                    Arc::clone(slot)
                } else {
                    Arc::clone(map.entry(name.to_string()).or_default())
                }
            }
        };
        Arc::clone(slot.get_or_init(|| path_to_file_url(std::path::Path::new(name))))
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

    #[test]
    fn source_fingerprint_samples_the_middle() {
        let a = "HEADER00\na\nb\nTAIL0000";
        let b = "HEADER00\nab\n\nTAIL0000";
        assert_eq!(a.len(), b.len());
        assert_ne!(source_fingerprint(a), source_fingerprint(b));
    }

    /// The single-slot column cache must be keyed by the line *slice*, not just
    /// its line number.  Before that, two sources rendered through one batch ctx
    /// whose line 2 both had a leading tab run shared the slot, so the second
    /// diagnostic's header used the first file's tab-expanded columns — plausible
    /// output with no crash, the exact class this cache stack exists to exclude.
    #[test]
    fn display_col_in_line_is_keyed_by_the_line_slice() {
        let opts = DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);
        // Same line number, same tab width, different indent: the old key
        // `(line_no, tab_width)` matched across sources.
        let a = "\t\tcode_a";
        let b = "\tcode_b";
        let want_a = crate::diagnostics::glyph::LineColumns::new(a, 4).display(a.len());
        let want_b = crate::diagnostics::glyph::LineColumns::new(b, 4).display(b.len());
        assert_ne!(want_a, want_b, "the fixture must distinguish the two");

        assert_eq!(cx.display_col_in_line(a, 2, a.len(), 4), want_a);
        // Interleaved: this is the cross-source hit that used to alias.
        assert_eq!(cx.display_col_in_line(b, 2, b.len(), 4), want_b);
        assert_eq!(cx.display_col_in_line(a, 2, a.len(), 4), want_a);
    }

    /// Tier-0 pins.  A `fingerprint == 0` sentinel must degrade to a *miss*
    /// (never wrong numbers), and the binary-searched lookup must still resolve
    /// the right entry when two prebuilt sources share a length.
    #[test]
    fn prebuilt_tier_falls_back_on_a_sentinel_and_disambiguates_by_pointer() {
        let opts = DiagOpts::default();
        let a = "aa
bb
";
        let b = "cc
dd
ee
";
        assert_ne!(a.len(), b.len());
        let same_len_a = "x1
y2
";
        let same_len_b = "p1
q2
";
        assert_eq!(same_len_a.len(), same_len_b.len());

        let entry = |src: &str, fingerprint: u64| PreloadedSource {
            ptr: src.as_ptr() as usize,
            len: src.len(),
            starts: Arc::from(LineIndex::new(src).starts_arc().to_vec()),
            ascii: true,
            char_prefixes: Arc::default(),
            generation: 7,
            fingerprint,
        };
        let mut prebuilt = vec![
            entry(a, 0), // sentinel: must never be trusted
            entry(b, source_fingerprint(b)),
            entry(same_len_a, source_fingerprint(same_len_a)),
            entry(same_len_b, source_fingerprint(same_len_b)),
        ];
        prebuilt.sort_unstable_by_key(|e| (e.ptr, e.len));
        let shared = SharedRenderState::new();
        let cx = RenderCtx::hoisted_with_prebuilt(&opts, None, &shared, &prebuilt);

        // Sentinel → miss → rebuilt table (correct, just not prebuilt).
        assert_eq!(cx.line_index(a).line_of(a.len()), 2);
        // Correct entry at the right pointer.
        assert_eq!(cx.line_index(b).line_of(b.len()), 3);
        // Two entries of equal length must not be confused by the search.
        assert_eq!(cx.line_index(same_len_b).line_of(3), 1);
        assert_eq!(cx.line_index(same_len_a).line_of(3), 1);
    }

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

    /// Append a placeholder for a diagnostic whose render panicked.
    ///
    /// The emitter catches render panics and substitutes this so one broken
    /// diagnostic cannot abort a whole batch.  The default is the flat-text
    /// marker the human and HTML formats want.
    ///
    /// A **structured** format must override this: prose spliced into a JSON
    /// array produces output no consumer can parse, which is strictly worse than
    /// omitting the item — the batch would fail at the parser instead of
    /// surfacing one broken diagnostic.
    fn render_internal_error(&self, out: &mut String) {
        out.push_str(super::emitter::RENDER_PANIC_PLACEHOLDER);
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
        // rustc-style annotation prefixes (`= note:`) stay English by
        // convention; only the payload goes through Fluent.  This keeps the
        // terminal summary lexically stable for tools while localized counts
        // still follow CLDR plurals.
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

/// Measurement probes for the two decisions the review says must be *measured*
/// before any code: whether `linecol_display` re-walks the line per query (the
/// monotone-cursor optimization is only worth it if it does), and what the
/// serial batch prebuild actually costs (it is the candidate for `thread::scope`
/// parallelism).
///
/// `#[ignore]`d so ordinary runs stay fast; CI's `benchmark` job runs every
/// `diagnostics::`-scoped ignored probe and prints the numbers.  Nothing here
/// asserts a wall-clock threshold — shared runners are too noisy to gate on, and
/// the point is a recorded baseline, not a tripwire.
#[cfg(all(test, not(feature = "shuttle")))]
mod perf_probes {
    use super::*;

    fn line_of_len(n: usize) -> String {
        // Tab-indented so the display-column path (not the ASCII fast path) runs.
        let mut s = String::from("\t\t");
        s.push_str(&"x".repeat(n));
        s
    }

    /// Per-query cost of `linecol_display` at several in-line offsets.
    ///
    /// If the column is recomputed by walking the line prefix, cost grows with
    /// the offset (and with span count per line); if it is cached/`O(log)`, the
    /// numbers stay flat and the monotone-cursor idea in the review is dropped.
    #[test]
    #[ignore = "measurement only"]
    fn perf_linecol_display_repeated_queries() {
        let opts = DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);
        for len in [64usize, 512, 4096] {
            let src = line_of_len(len);
            let idx = cx.line_index(&src);
            let probes: Vec<usize> = (0..16).map(|i| 2 + len * i / 16).collect();
            // Warm the lazily-built per-line tables so the measurement is steady
            // state, which is what a batch actually sees.
            for &p in &probes {
                std::hint::black_box(idx.linecol_display(p));
            }
            let iters = 2_000;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                for &p in &probes {
                    std::hint::black_box(idx.linecol_display(p));
                }
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / (iters * probes.len()) as f64;
            println!(
                "[perf] linecol_display len={len:>5} x{}: {ns:.1} ns/query",
                probes.len()
            );
        }
    }

    /// Cost of the tier walk on a repeat render of the same source: the LRU hit
    /// path (no fingerprint), the tier-0 prebuilt path, and the rebuild.
    #[test]
    #[ignore = "measurement only"]
    fn perf_line_index_lookup_tiers() {
        let opts = DiagOpts::default();
        let src = "a\nb\nc\n".repeat(4_000);
        let cx = RenderCtx::new(&opts, None);
        std::hint::black_box(cx.line_index(&src)); // cold: rebuild + publish
        let iters = 20_000;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(cx.line_index(&src));
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
        println!(
            "[perf] line_index LRU hit ({} bytes source): {ns:.1} ns/lookup",
            src.len()
        );
    }
}
