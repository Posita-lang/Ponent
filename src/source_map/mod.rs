//! The compiler's concurrent [`SourceMap`]: registry of source files
//! addressed by [`FileId`], with secondary indexes by physical path and
//! by module path.
//!
//! Slot 0 of the registry is a pre-registered dummy file backing
//! [`FileId::DUMMY`], so [`SourceMap::file_or_dummy`] is infallible and
//! never panics — diagnostics is a cold path that must stay safe even
//! with a stale or synthetic id in hand.  [`FileId::DUMMY`] is a
//! sentinel, not an index; real ids start at [`FileId::FIRST_REAL`].
//!
//! # Architecture
//!
//! - One `RwLock<Registry>` for **all** mutations (three maps collapsed
//!   into one critical section — no lock ordering, no double-check race).
//! - One [`ContentStore`] for content interning (`content.rs`): files
//!   with identical bytes share one `Arc<Content>` (text, line starts,
//!   fingerprint, parsed header edition).
//! - **Invariant: never call `contents.intern` while holding the
//!   registry write lock.**  `intern` is O(n) hashing / possible O(n)
//!   build; the whole point of the split is that it runs *outside* the
//!   publish critical section.
//!
//! Concurrency model: every method takes `&self` (the map is shared,
//! e.g. with parallel file loading).  A `FileId` is allocated *inside*
//! the write critical section, after every fallible step, so a failed
//! insert consumes no id and leaves no dummy placeholder.

mod content;
mod source_file;

pub use source_file::{Fingerprint, MAX_FILE_LEN, SourceFile, SourceFileError};

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::SystemTime;

use compact_str::CompactString;
use rustc_hash::FxHashMap;

use crate::ast::FileId;
use crate::hir::types::Edition;
use crate::sync::RwLock;

use content::ContentStore;

/// A language-level module path (e.g. `std::collections`) in its
/// canonical `::`-joined string form — the shape `import` paths parse
/// to.  One module path maps to at most one registered file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModulePath(CompactString);

impl ModulePath {
    /// The empty path backing [`FileId::DUMMY`].
    pub fn dummy() -> Self {
        Self(CompactString::new(""))
    }

    /// Build from a `::`-joined path string (or anything convertible to
    /// a [`CompactString`]).
    pub fn new(path: impl Into<CompactString>) -> Self {
        Self(path.into())
    }

    /// The canonical `::`-joined form.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModulePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Error returned by [`SourceMap::insert`] / [`SourceMap::set_file_contents`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceMapError {
    /// A file with this physical path is already registered.
    DuplicatePath(PathBuf),
    /// A file for this module path is already registered.
    DuplicateModule(ModulePath),
    /// No file has been (or successfully) registered under this id.
    UnknownFileId(FileId),
    /// The file-id space is exhausted (or wrapped into a reserved value).
    TooManyFiles,
    /// The raw bytes could not be turned into a [`SourceFile`].
    File(SourceFileError),
}

impl From<SourceFileError> for SourceMapError {
    fn from(e: SourceFileError) -> Self {
        SourceMapError::File(e)
    }
}

impl std::fmt::Display for SourceMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceMapError::DuplicatePath(p) => {
                write!(f, "file `{}` is already registered", p.display())
            }
            SourceMapError::DuplicateModule(m) => write!(f, "module `{m}` is already registered"),
            SourceMapError::UnknownFileId(id) => write!(f, "unknown FileId({})", id.as_u32()),
            SourceMapError::TooManyFiles => {
                // Valid ids are 1 ..= u32::MAX - 1 (0 is the Option niche,
                // u32::MAX is FileId::DUMMY).
                write!(f, "cannot register more than {} source files", u32::MAX - 1)
            }
            SourceMapError::File(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SourceMapError {}

// ── Registry ───────────────────────────────────────────────────

/// The three maps as one unit: `files` is the source of truth, the two
/// path maps are its derived indexes.  Mutated under one `RwLock`.
struct Registry {
    files: Vec<Arc<SourceFile>>,
    by_path: FxHashMap<PathBuf, FileId>,
    by_module: FxHashMap<ModulePath, FileId>,
}

impl Registry {
    fn new() -> Self {
        Registry {
            // Slot 0 = the dummy for `FileId::DUMMY`.  `files` is never
            // empty, so `files[0]` in the fallback paths is infallible.
            files: vec![Arc::new(SourceFile::dummy())],
            by_path: FxHashMap::default(),
            by_module: FxHashMap::default(),
        }
    }
}

// ── SourceMap ──────────────────────────────────────────────────

/// Registry of source files, indexed by [`FileId`], with secondary
/// indexes by physical path and module path, and content deduplication
/// across files with identical bytes.
///
/// Invariants:
/// - `files[0]` is always the dummy file ([`FileId::DUMMY`]);
/// - `FileId::DUMMY` maps to `files[0]` via [`Self::file_or_dummy`]
///   without being used as an index;
/// - real ids are dense from [`FileId::FIRST_REAL`] and equal their
///   vector index (`files[id]`) — no failed insert consumes an id;
/// - a `FileId`, once allocated, always points to the same
///   `module_path` — content may be replaced, identity may not;
/// - two files with identical bytes share one `Arc<Content>`.
pub struct SourceMap {
    registry: RwLock<Registry>,
    contents: ContentStore,
    /// Starts at `FileId::FIRST_REAL` (1); `fetch_add` allocates ids
    /// under the registry write lock so no failed insert leaks an id.
    next_id: AtomicU32,
}

// The whole point of this design is that the map is shared across
// parallel file loading — pin that contract here, at the type site.
static_assertions::assert_impl_all!(SourceMap: Send, Sync);

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceMap {
    /// An empty map whose slot 0 is the dummy file for [`FileId::DUMMY`].
    pub fn new() -> Self {
        SourceMap {
            registry: RwLock::new(Registry::new()),
            contents: ContentStore::new(),
            next_id: AtomicU32::new(FileId::FIRST_REAL),
        }
    }

    /// Register a file and return its [`FileId`].
    ///
    /// Failure modes (none of which consume an id):
    /// - duplicate physical path or module path → `DuplicatePath` /
    ///   `DuplicateModule`;
    /// - raw bytes fail size / UTF-8 checks → `File(...)`;
    /// - id space exhausted → `TooManyFiles`.
    ///
    /// On success the id is dense (`files[id] == this file`).  Content is
    /// interned before the id is allocated, so two files with identical
    /// bytes share one `Arc<Content>` (text, line starts, fingerprint,
    /// parsed header edition).
    pub fn insert(
        &self,
        physical_path: PathBuf,
        module_path: ModulePath,
        raw: &[u8],
        default_edition: Edition,
        mtime: Option<SystemTime>,
    ) -> Result<FileId, SourceMapError> {
        // (1) Read-lock duplicate pre-check — cheap, skips the O(n)
        //     intern on the common error path.
        {
            let r = self.registry.read();
            if r.by_path.contains_key(&physical_path) {
                return Err(SourceMapError::DuplicatePath(physical_path));
            }
            if r.by_module.contains_key(&module_path) {
                return Err(SourceMapError::DuplicateModule(module_path));
            }
        }

        // (2) Intern content FIRST, outside every registry lock.  On the
        //     hot path (content seen before) this is one xxh3 (inherent)
        //     + one shard read-lock hit.  A failure here returns before
        //     any id is allocated — no hole, no dummy filler.
        let (content, had_bom) = self.contents.intern(raw)?;

        // (3) Single write critical section: recheck duplicates, allocate
        //     the id, build the file (infallible), publish atomically.
        let mut w = self.registry.write();
        if w.by_path.contains_key(&physical_path) {
            return Err(SourceMapError::DuplicatePath(physical_path));
        }
        if w.by_module.contains_key(&module_path) {
            return Err(SourceMapError::DuplicateModule(module_path));
        }
        let id = self.alloc_id_locked()?;
        let file = Arc::new(SourceFile::from_parts(
            id,
            physical_path.clone(),
            module_path.clone(),
            content,
            had_bom,
            default_edition,
            mtime,
        ));
        // Density invariant: `files.len() == id` before push.  Because
        // `alloc_id_locked` runs inside the write lock (serializing id
        // allocation with the push) and never leaks an id, this holds
        // unconditionally.
        debug_assert_eq!(
            w.files.len(),
            id.as_u32() as usize,
            "files must be dense; next_id must not have advanced"
        );
        w.files.push(file);
        w.by_path.insert(physical_path, id);
        w.by_module.insert(module_path, id);
        Ok(id)
    }

    /// Allocate an id **while holding the registry write lock**.
    /// Guarantees: no id is returned unless the caller is about to
    /// publish, and no id is ever `0` (Option niche) or
    /// `FileId::DUMMY` (u32::MAX).
    fn alloc_id_locked(&self) -> Result<FileId, SourceMapError> {
        let raw = self.next_id.fetch_add(1, Ordering::Relaxed);
        if raw == 0 || raw == FileId::DUMMY.as_u32() {
            return Err(SourceMapError::TooManyFiles);
        }
        Ok(FileId::from_raw(raw))
    }

    // ── Lookup ───────────────────────────────────────────────────

    /// The file registered under `id`, **only if** `id` is a real,
    /// successful registration.  Returns `None` for the sentinel
    /// `FileId::DUMMY` and for any id never successfully registered —
    /// the caller decides whether to fall back.
    pub fn file(&self, id: FileId) -> Option<Arc<SourceFile>> {
        let r = self.registry.read();
        r.files.get(id.as_u32() as usize).cloned()
    }

    /// The file registered under `id`, or the dummy sentinel — safe
    /// fallback for diagnostics rendering, where an unknown id must
    /// still produce *some* source (an empty one) rather than panic.
    pub fn file_or_dummy(&self, id: FileId) -> Arc<SourceFile> {
        let r = self.registry.read();
        r.files
            .get(id.as_u32() as usize)
            .cloned()
            .unwrap_or_else(|| r.files[0].clone())
    }

    /// The id registered under `physical_path`, if any.
    pub fn file_id_by_path(&self, path: &std::path::Path) -> Option<FileId> {
        self.registry.read().by_path.get(path).copied()
    }

    /// The id registered under `module_path`, if any.
    pub fn file_id_by_module(&self, module: &ModulePath) -> Option<FileId> {
        self.registry.read().by_module.get(module).copied()
    }

    /// All registered module paths, sorted (stable order for
    /// did-you-mean style suggestions).
    pub fn module_paths(&self) -> Vec<ModulePath> {
        let mut paths: Vec<ModulePath> = self.registry.read().by_module.keys().cloned().collect();
        paths.sort();
        paths
    }

    /// Total registered files (including the dummy at slot 0).
    pub fn len(&self) -> usize {
        self.registry.read().files.len()
    }

    /// True when only the dummy file is registered.
    pub fn is_empty(&self) -> bool {
        self.registry.read().files.len() <= 1
    }

    // ── Idempotent insert ────────────────────────────────────────

    /// Register a file, or return the existing id when the path is
    /// already registered — an idempotent insert for watch / incremental
    /// reload scenarios.
    ///
    /// "Duplicate path" is a *success* here: under a race another thread
    /// may have won the insert, in which case the winner's id is
    /// returned.  A duplicate *module* mapping onto a different path is
    /// still an error.  Callers that need to react to content changes
    /// on an already-registered path follow up with
    /// [`Self::set_file_contents`], whose `Arc::ptr_eq` fast path filters
    /// no-op updates in O(1) beyond one `xxh3_128`.
    pub fn insert_or_get(
        &self,
        physical_path: PathBuf,
        module_path: ModulePath,
        raw: &[u8],
        default_edition: Edition,
        mtime: Option<SystemTime>,
    ) -> Result<FileId, SourceMapError> {
        {
            let r = self.registry.read();
            if let Some(&id) = r.by_path.get(&physical_path) {
                return Ok(id);
            }
        }
        match self.insert(
            physical_path.clone(),
            module_path,
            raw,
            default_edition,
            mtime,
        ) {
            Ok(id) => Ok(id),
            // A concurrent insert won the race — read back its id.
            Err(SourceMapError::DuplicatePath(_)) => {
                let r = self.registry.read();
                r.by_path
                    .get(&physical_path)
                    .copied()
                    .ok_or(SourceMapError::DuplicatePath(physical_path))
            }
            Err(e) => Err(e),
        }
    }

    // ── Reload ──────────────────────────────────────────────────

    /// Replace the content of a registered file, returning whether the
    /// file changed.
    ///
    /// Fast path: `Arc::ptr_eq` on the shared content **plus** matching
    /// `had_bom` and `mtime` — the common "touched but unchanged" event
    /// in watch mode (formatting save, editor save with trailing
    /// whitespace) costs one `xxh3_128` and returns `Ok(false)`.
    ///
    /// No mtime-only fast path is kept: "same mtime" is a false negative
    /// for edits within filesystem timestamp granularity — the `ptr_eq`
    /// path already makes a no-op reload cheap, and correctness on the
    /// granularity window matters more than shaving one hash.
    pub fn set_file_contents(
        &self,
        id: FileId,
        new_raw: &[u8],
        new_mtime: Option<SystemTime>,
    ) -> Result<bool, SourceMapError> {
        if id == FileId::DUMMY {
            return Err(SourceMapError::UnknownFileId(id));
        }

        // (1) Intern **outside** every registry lock.
        let (new_content, new_had_bom) = self.contents.intern(new_raw)?;

        // (2) Snapshot the old file under a read lock.
        let old_file = {
            let r = self.registry.read();
            r.files
                .get(id.as_u32() as usize)
                .cloned()
                .ok_or(SourceMapError::UnknownFileId(id))?
        };

        // (3) Fast path: content Arc unchanged + metadata unchanged.
        //     The `had_bom` and `mtime` terms are load-bearing — BOM is
        //     stripped before interning, so a BOM-only change produces
        //     the same `Arc<Content>` but a different `had_bom`; without
        //     the check the flag would go stale silently.
        if Arc::ptr_eq(&old_file.content, &new_content)
            && old_file.had_bom == new_had_bom
            && old_file.mtime == new_mtime
        {
            return Ok(false);
        }

        // (4) Slow path: rebuild identity around the (possibly shared)
        //     new content.  `from_parts` is infallible — interning has
        //     already validated everything.  Preserve the caller's
        //     original default edition so a header-bearing reload cannot
        //     make its header the sticky fallback for a later
        //     header-less reload.
        let new_file = Arc::new(SourceFile::from_parts(
            id,
            old_file.physical_path.clone(),
            old_file.module_path.clone(),
            new_content,
            new_had_bom,
            old_file.default_edition,
            new_mtime,
        ));
        let mut w = self.registry.write();
        if let Some(slot) = w.files.get_mut(id.as_u32() as usize) {
            *slot = new_file;
            Ok(true)
        } else {
            Err(SourceMapError::UnknownFileId(id))
        }
    }

    // ── Manual sweep ────────────────────────────────────────────

    /// Sweep dead `Weak` slots from every content shard.  Optional — the
    /// store already sweeps oversized shards opportunistically on write,
    /// but an LSP that goes idle can call this to reclaim the tail.
    pub fn sweep_dead_contents(&self) {
        self.contents.sweep_all();
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use std::path::Path;

    const EDITION: Edition = Edition::Year2026;

    fn insert_ok(map: &SourceMap, path: &str, module: &str, raw: &[u8]) -> FileId {
        map.insert(
            PathBuf::from(path),
            ModulePath::new(module),
            raw,
            EDITION,
            None,
        )
        .unwrap_or_else(|e| panic!("insert {path}: {e}"))
    }

    #[test]
    fn slot_zero_is_dummy() {
        let map = SourceMap::new();
        let dummy = map.file_or_dummy(FileId::DUMMY);
        assert_eq!(dummy.id, FileId::DUMMY);
        assert_eq!(dummy.text(), "");
        assert_eq!(dummy.module_path, ModulePath::dummy());
        // `file(DUMMY)` returns None — DUMMY is not an index.
        assert!(map.file(FileId::DUMMY).is_none());
    }

    #[test]
    fn file_unknown_id_returns_none() {
        let map = SourceMap::new();
        assert!(map.file(FileId::from_raw(99)).is_none());
        // `file_or_dummy` still gives the sentinel for diagnostics.
        assert_eq!(map.file_or_dummy(FileId::from_raw(99)).id, FileId::DUMMY);
    }

    #[test]
    fn insert_and_file_roundtrip() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a/main.pos", "main", b"def main() {}");
        assert_eq!(id, FileId::from_raw(FileId::FIRST_REAL));
        let f = map.file(id).expect("just-inserted file must resolve");
        assert_eq!(f.text(), "def main() {}");
        assert_eq!(f.edition, EDITION);
        assert_eq!(f.default_edition, EDITION);
        assert_eq!(f.module_path, ModulePath::new("main"));
        assert_eq!(f.physical_path, PathBuf::from("/a/main.pos"));
    }

    #[test]
    fn duplicate_path_rejected() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a/x.pos", "a::x", b"x");
        let err = map
            .insert(
                PathBuf::from("/a/x.pos"),
                ModulePath::new("other"),
                b"x",
                EDITION,
                None,
            )
            .unwrap_err();
        assert_eq!(
            err,
            SourceMapError::DuplicatePath(PathBuf::from("/a/x.pos"))
        );
        // The first registration is untouched.
        assert_eq!(map.file(id).unwrap().module_path, ModulePath::new("a::x"));
    }

    #[test]
    fn duplicate_module_rejected() {
        let map = SourceMap::new();
        insert_ok(&map, "/a/x.pos", "m", b"x");
        let err = map
            .insert(
                PathBuf::from("/b/x.pos"),
                ModulePath::new("m"),
                b"y",
                EDITION,
                None,
            )
            .unwrap_err();
        assert_eq!(err, SourceMapError::DuplicateModule(ModulePath::new("m")));
    }

    /// A failed insert must not consume an id (no hole, no dummy filler).
    #[test]
    fn failed_insert_leaves_no_hole() {
        let map = SourceMap::new();
        let id1 = insert_ok(&map, "/a", "a", b"v1");
        assert_eq!(id1, FileId::from_raw(1));
        // Duplicate insert fails BEFORE alloc_id_locked runs.
        let err = map
            .insert(
                PathBuf::from("/a"),
                ModulePath::new("other"),
                b"v2",
                EDITION,
                None,
            )
            .unwrap_err();
        assert!(matches!(err, SourceMapError::DuplicatePath(_)));
        // A UTF-8 failure fails BEFORE alloc_id_locked runs too.
        let bad = map
            .insert(
                PathBuf::from("/bad"),
                ModulePath::new("bad"),
                b"\xff",
                EDITION,
                None,
            )
            .unwrap_err();
        assert_eq!(bad, SourceMapError::File(SourceFileError::InvalidUtf8));
        // Next insert gets id 2 — dense, no dummy filler.
        let id2 = insert_ok(&map, "/b", "b", b"v3");
        assert_eq!(id2, FileId::from_raw(2));
        assert!(map.file(FileId::from_raw(3)).is_none(), "no hole-3");
        assert_eq!(map.file(id1).unwrap().text(), "v1");
        assert_eq!(map.file(id2).unwrap().text(), "v3");
    }

    /// Content-dedup smoke test: two files with identical bytes share
    /// one `Arc<Content>`.
    #[test]
    fn identical_content_shares_payload() {
        let map = SourceMap::new();
        let a = insert_ok(&map, "/a", "a", b"same bytes");
        let b = insert_ok(&map, "/b", "b", b"same bytes");
        let fa = map.file(a).unwrap();
        let fb = map.file(b).unwrap();
        assert!(Arc::ptr_eq(&fa.content, &fb.content));
        assert_eq!(fa.text(), fb.text());
    }

    /// Regression: a header-bearing reload must not make its header the
    /// sticky fallback for a later header-less reload.  Uses Year2026 as
    /// the caller default and Year2024 as the header value (the only two
    /// editions the language knows).
    #[test]
    fn edition_falls_back_to_default_after_headered_reload() {
        let map = SourceMap::new();
        // Initial insert: no header, default = 2026.
        let id = map
            .insert(
                PathBuf::from("/a"),
                ModulePath::new("a"),
                b"def f() {}",
                Edition::Year2026,
                None,
            )
            .unwrap();
        assert_eq!(map.file(id).unwrap().edition, Edition::Year2026);

        // Reload with header = 2024.
        let t1 = SystemTime::now();
        assert!(
            map.set_file_contents(id, b"edition = \"2024\";\ndef f() {}", Some(t1))
                .unwrap()
        );
        assert_eq!(map.file(id).unwrap().edition, Edition::Year2024);
        assert_eq!(map.file(id).unwrap().default_edition, Edition::Year2026);

        // Reload again with NO header — must fall back to the ORIGINAL
        // default (2026), not the previous resolved 2024.
        let t2 = SystemTime::now();
        assert!(
            map.set_file_contents(id, b"def f() { /* new body */ }", Some(t2))
                .unwrap()
        );
        assert_eq!(map.file(id).unwrap().edition, Edition::Year2026);
    }

    /// A BOM-only change (same canonical text, different `had_bom`) must
    /// be reported as changed.
    #[test]
    fn bom_only_change_reports_changed() {
        let map = SourceMap::new();
        let id = map
            .insert(
                PathBuf::from("/a"),
                ModulePath::new("a"),
                b"def f() {}",
                EDITION,
                None,
            )
            .unwrap();
        assert!(!map.file(id).unwrap().had_bom);
        let t = SystemTime::now();
        assert!(
            map.set_file_contents(id, b"\xEF\xBB\xBFdef f() {}", Some(t))
                .unwrap()
        );
        assert!(map.file(id).unwrap().had_bom);
        // Content Arc is unchanged (BOM stripped before interning) but
        // had_bom / mtime changed, so the swap did happen.
        assert_eq!(map.file(id).unwrap().text(), "def f() {}");
    }

    /// Reload fast path: unchanged content + unchanged metadata ⇒ no-op.
    #[test]
    fn ptr_eq_reload_is_no_op() {
        let map = SourceMap::new();
        let t = SystemTime::now();
        let id = map
            .insert(
                PathBuf::from("/a"),
                ModulePath::new("a"),
                b"same",
                EDITION,
                Some(t),
            )
            .unwrap();
        assert!(!map.set_file_contents(id, b"same", Some(t)).unwrap());
        // A second no-op reload is still a no-op.
        assert!(!map.set_file_contents(id, b"same", Some(t)).unwrap());
    }

    /// mtime change with unchanged content still swaps.
    #[test]
    fn mtime_change_with_unchanged_content_swaps() {
        let map = SourceMap::new();
        let t1 = SystemTime::now();
        let id = map
            .insert(
                PathBuf::from("/a"),
                ModulePath::new("a"),
                b"same",
                EDITION,
                Some(t1),
            )
            .unwrap();
        let t2 = t1 + std::time::Duration::from_secs(1);
        assert!(map.set_file_contents(id, b"same", Some(t2)).unwrap());
        assert_eq!(map.file(id).unwrap().mtime, Some(t2));
    }

    #[test]
    fn set_file_contents_rejects_dummy() {
        let map = SourceMap::new();
        let err = map
            .set_file_contents(FileId::DUMMY, b"x", None)
            .unwrap_err();
        assert_eq!(err, SourceMapError::UnknownFileId(FileId::DUMMY));
        let d = map.file_or_dummy(FileId::DUMMY);
        assert_eq!(d.text(), "");
    }

    #[test]
    fn set_file_contents_unknown_id() {
        let map = SourceMap::new();
        assert_eq!(
            map.set_file_contents(FileId::from_raw(50), b"new", None),
            Err(SourceMapError::UnknownFileId(FileId::from_raw(50)))
        );
    }

    #[test]
    fn insert_or_get_is_idempotent() {
        let map = SourceMap::new();
        let first = map
            .insert_or_get(
                PathBuf::from("/a"),
                ModulePath::new("a"),
                b"v1",
                EDITION,
                None,
            )
            .unwrap();
        // Same path again — even with different content — returns the
        // SAME id; content updates go through set_file_contents.
        let second = map
            .insert_or_get(
                PathBuf::from("/a"),
                ModulePath::new("a"),
                b"v2",
                EDITION,
                None,
            )
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(map.file(first).unwrap().text(), "v1");
    }

    #[test]
    fn insert_or_get_still_rejects_duplicate_module() {
        let map = SourceMap::new();
        insert_ok(&map, "/a/x.pos", "m", b"x");
        let err = map
            .insert_or_get(
                PathBuf::from("/b/x.pos"),
                ModulePath::new("m"),
                b"y",
                EDITION,
                None,
            )
            .unwrap_err();
        assert_eq!(err, SourceMapError::DuplicateModule(ModulePath::new("m")));
    }

    #[test]
    fn concurrent_inserts_get_distinct_ids() {
        let map = Arc::new(SourceMap::new());
        std::thread::scope(|s| {
            for n in 0..16u32 {
                let map = Arc::clone(&map);
                s.spawn(move || insert_ok(&map, &format!("/t/{n}"), &format!("m{n}"), b""));
            }
        });
        // All 16 files registered with dense ids from FIRST_REAL.
        for n in 0..16u32 {
            let f = map.file(FileId::from_raw(1 + n)).expect("dense ids");
            assert_ne!(f.id, FileId::DUMMY, "hole at id {}", 1 + n);
            assert_eq!(f.text(), "");
        }
        assert_eq!(map.len(), 17); // dummy + 16 files
    }

    #[test]
    fn alloc_id_skips_zero_and_dummy() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a", "a", b"x");
        assert_ne!(id.as_u32(), 0);
        assert_ne!(id.as_u32(), FileId::DUMMY.as_u32());
    }

    #[test]
    fn lookup_by_path_and_module() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a/main.pos", "main", b"def main() {}");
        assert_eq!(map.file_id_by_path(Path::new("/a/main.pos")), Some(id));
        assert_eq!(map.file_id_by_path(Path::new("/other")), None);
        assert_eq!(map.file_id_by_module(&ModulePath::new("main")), Some(id));
        assert_eq!(map.file_id_by_module(&ModulePath::new("nope")), None);
    }

    /// `sweep_dead_contents` is callable and preserves live content.
    #[test]
    fn sweep_dead_contents_is_safe() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a", "a", b"live");
        map.sweep_dead_contents();
        assert_eq!(map.file(id).unwrap().text(), "live");
    }
}
