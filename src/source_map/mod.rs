//! The compiler's concurrent [`SourceMap`]: registry of source files
//! addressed by [`FileId`], with secondary indexes by physical path and
//! by module path.
//!
//! Slot 0 of the registry is a pre-registered dummy file backing
//! [`FileId::DUMMY`], so [`SourceMap::file`] is infallible and never
//! panics — diagnostics is a cold path that must stay safe even with a
//! stale or synthetic id in hand.
//!
//! Concurrency model: every method takes `&self` (the map is shared, e.g.
//! with parallel file loading).  A `FileId` is allocated from an atomic
//! counter BEFORE the file is built, so a build failure leaves a dummy
//! placeholder at that id; successful ids are dense in load order,
//! starting at [`FileId::FIRST_REAL`].  Locks are always acquired in the
//! order `files` → `by_path` → `by_module`.

mod source_file;

pub use source_file::{Fingerprint, MAX_FILE_LEN, SourceFile, SourceFileError};

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use compact_str::CompactString;
use rustc_hash::FxHashMap;

use crate::ast::FileId;
use crate::hir::types::Edition;
use crate::sync::RwLock;

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
            SourceMapError::File(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SourceMapError {}

/// Registry of source files, indexed by [`FileId`], with secondary
/// indexes by physical path and module path.
///
/// Invariants:
/// - `files[0]` is always the dummy file ([`FileId::DUMMY`]);
/// - a `FileId`, once allocated, always points to the same
///   `module_path` — content may be replaced, identity may not;
/// - [`SourceMap::file`] never panics: unknown or placeholder ids
///   return the dummy.
pub struct SourceMap {
    files: RwLock<Vec<Arc<SourceFile>>>,
    by_path: RwLock<FxHashMap<PathBuf, FileId>>,
    by_module: RwLock<FxHashMap<ModulePath, FileId>>,
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
            files: RwLock::new(vec![Arc::new(SourceFile::dummy())]),
            by_path: RwLock::new(FxHashMap::default()),
            by_module: RwLock::new(FxHashMap::default()),
            next_id: AtomicU32::new(1),
        }
    }

    /// Register a file and return its [`FileId`].
    ///
    /// The id is allocated before the file is built, so a build failure
    /// (bad UTF-8, oversized file) leaves a dummy placeholder at that id
    /// — [`Self::file`] maps it back to the dummy.  Duplicate physical
    /// paths and duplicate module paths are rejected (checked under
    /// read locks first, rechecked under the write locks to close the
    /// race window).
    pub fn insert(
        &self,
        physical_path: PathBuf,
        module_path: ModulePath,
        raw: &[u8],
        default_edition: Edition,
        mtime: Option<SystemTime>,
    ) -> Result<FileId, SourceMapError> {
        // Fast duplicate rejection under read locks.
        if self.by_path.read().contains_key(&physical_path) {
            return Err(SourceMapError::DuplicatePath(physical_path));
        }
        if self.by_module.read().contains_key(&module_path) {
            return Err(SourceMapError::DuplicateModule(module_path));
        }
        let id = FileId::from_raw(self.next_id.fetch_add(1, Ordering::Relaxed));
        let file = SourceFile::from_bytes(
            id,
            physical_path.clone(),
            module_path.clone(),
            raw,
            default_edition,
            mtime,
        )?;
        // Recheck under the write locks: a concurrent insert may have
        // registered the same path or module between the checks above.
        let mut files = self.files.write();
        let mut by_path = self.by_path.write();
        let mut by_module = self.by_module.write();
        if by_path.contains_key(&physical_path) {
            return Err(SourceMapError::DuplicatePath(physical_path));
        }
        if by_module.contains_key(&module_path) {
            return Err(SourceMapError::DuplicateModule(module_path));
        }
        while files.len() <= id.as_u32() as usize {
            files.push(Arc::new(SourceFile::dummy()));
        }
        files[id.as_u32() as usize] = Arc::new(file);
        by_path.insert(physical_path, id);
        by_module.insert(module_path, id);
        Ok(id)
    }

    /// The file registered under `id`.
    ///
    /// Never panics: an unknown id (or a placeholder left by a failed
    /// [`Self::insert`]) returns the dummy file.
    pub fn file(&self, id: FileId) -> Arc<SourceFile> {
        let files = self.files.read();
        files
            .get(id.as_u32() as usize)
            .cloned()
            .unwrap_or_else(|| files[0].clone())
    }

    /// The id registered under `physical_path`, if any.
    pub fn file_id_by_path(&self, path: &std::path::Path) -> Option<FileId> {
        self.by_path.read().get(path).copied()
    }

    /// The id registered under `module_path`, if any.
    pub fn file_id_by_module(&self, module: &ModulePath) -> Option<FileId> {
        self.by_module.read().get(module).copied()
    }

    /// All registered module paths, sorted (stable order for
    /// did-you-mean style suggestions).
    pub fn module_paths(&self) -> Vec<ModulePath> {
        let by_module = self.by_module.read();
        let mut paths: Vec<ModulePath> = by_module.keys().cloned().collect();
        paths.sort();
        paths
    }

    /// Register a file, or return the existing id when the path is
    /// already registered — an idempotent insert for watch / incremental
    /// reload scenarios.
    ///
    /// "Duplicate path" is a *success* here: under a race another thread
    /// may have won the insert, in which case the winner's id is
    /// returned (the lock order `files → by_path → by_module` makes the
    /// read-back safe).  A duplicate *module* mapping onto a different
    /// path is still an error.  Callers that need to react to content
    /// changes on an already-registered path follow up with
    /// [`Self::set_file_contents`], whose fingerprint fast path filters
    /// no-op updates.
    pub fn insert_or_get(
        &self,
        physical_path: PathBuf,
        module_path: ModulePath,
        raw: &[u8],
        default_edition: Edition,
        mtime: Option<SystemTime>,
    ) -> Result<FileId, SourceMapError> {
        if let Some(id) = self.file_id_by_path(&physical_path) {
            return Ok(id);
        }
        match self.insert(
            physical_path.clone(),
            module_path.clone(),
            raw,
            default_edition,
            mtime,
        ) {
            Ok(id) => Ok(id),
            // A concurrent insert won the race — read back its id.
            Err(SourceMapError::DuplicatePath(_)) => self
                .file_id_by_path(&physical_path)
                .ok_or(SourceMapError::DuplicatePath(physical_path)),
            Err(e) => Err(e),
        }
    }

    /// Replace the content of a registered file, returning whether the
    /// file changed.
    ///
    /// Fast path: an unchanged `mtime` short-circuits without reading
    /// bytes.  Slow path: rebuild the file and compare fingerprints — a
    /// matching fingerprint means unchanged content, so the slot is left
    /// untouched and `false` is returned.
    pub fn set_file_contents(
        &self,
        id: FileId,
        new_raw: &[u8],
        new_mtime: Option<SystemTime>,
    ) -> Result<bool, SourceMapError> {
        // The dummy slot is a sentinel, never a registered file — replacing
        // it would poison every `file()` fallback in the whole map.
        if id == FileId::DUMMY {
            return Err(SourceMapError::UnknownFileId(id));
        }
        let mut files = self.files.write();
        let Some(slot) = files.get_mut(id.as_u32() as usize) else {
            return Err(SourceMapError::UnknownFileId(id));
        };
        let old = slot.clone();
        // Fast path: mtime unchanged.
        if old.mtime.is_some() && new_mtime.is_some() && old.mtime == new_mtime {
            return Ok(false);
        }
        // Slow path: rebuild and compare fingerprints.
        let new_file = SourceFile::from_bytes(
            id,
            old.physical_path.clone(),
            old.module_path.clone(),
            new_raw,
            old.edition,
            new_mtime,
        )?;
        if new_file.fingerprint == old.fingerprint {
            return Ok(false); // content unchanged
        }
        *slot = Arc::new(new_file);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const EDITION: Edition = Edition::Year2026;

    fn insert_ok(map: &SourceMap, path: &str, module: &str, raw: &[u8]) -> FileId {
        map.insert(PathBuf::from(path), ModulePath::new(module), raw, EDITION, None)
            .unwrap_or_else(|e| panic!("insert {path}: {e}"))
    }

    #[test]
    fn slot_zero_is_dummy() {
        let map = SourceMap::new();
        let dummy = map.file(FileId::DUMMY);
        assert_eq!(dummy.id, FileId::DUMMY);
        assert_eq!(dummy.text.as_ref(), "");
        assert_eq!(dummy.module_path, ModulePath::dummy());
    }

    #[test]
    fn insert_and_file_roundtrip() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a/main.pos", "main", b"def main() {}");
        assert_eq!(id, FileId::from_raw(FileId::FIRST_REAL));
        let f = map.file(id);
        assert_eq!(f.text.as_ref(), "def main() {}");
        assert_eq!(f.edition, EDITION);
        assert_eq!(f.module_path, ModulePath::new("main"));
        assert_eq!(f.physical_path, PathBuf::from("/a/main.pos"));
    }

    #[test]
    fn duplicate_path_rejected() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a/x.pos", "a::x", b"x");
        let err = map
            .insert(PathBuf::from("/a/x.pos"), ModulePath::new("other"), b"x", EDITION, None)
            .unwrap_err();
        assert_eq!(err, SourceMapError::DuplicatePath(PathBuf::from("/a/x.pos")));
        // The first registration is untouched.
        assert_eq!(map.file(id).module_path, ModulePath::new("a::x"));
    }

    #[test]
    fn duplicate_module_rejected() {
        let map = SourceMap::new();
        insert_ok(&map, "/a/x.pos", "m", b"x");
        let err = map
            .insert(PathBuf::from("/b/x.pos"), ModulePath::new("m"), b"y", EDITION, None)
            .unwrap_err();
        assert_eq!(err, SourceMapError::DuplicateModule(ModulePath::new("m")));
    }

    #[test]
    fn file_unknown_id_returns_dummy() {
        let map = SourceMap::new();
        let f = map.file(FileId::from_raw(99));
        assert_eq!(f.id, FileId::DUMMY);
    }

    #[test]
    fn failed_insert_leaves_dummy_placeholder() {
        let map = SourceMap::new();
        let bad = map
            .insert(PathBuf::from("/bad"), ModulePath::new("bad"), b"\xff", EDITION, None)
            .unwrap_err();
        assert_eq!(bad, SourceMapError::File(SourceFileError::InvalidUtf8));
        // The consumed id holds a dummy placeholder; the next insert gets
        // the following id.
        assert_eq!(map.file(FileId::from_raw(1)).id, FileId::DUMMY);
        let good = insert_ok(&map, "/good", "good", b"ok");
        assert_eq!(good, FileId::from_raw(2));
        assert_eq!(map.file(good).text.as_ref(), "ok");
    }

    #[test]
    fn lookup_by_path_and_module() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a/main.pos", "main", b"def main() {}");
        assert_eq!(map.file_id_by_path(Path::new("/a/main.pos")), Some(id));
        assert_eq!(map.file_id_by_path(Path::new("/a/other.pos")), None);
        assert_eq!(map.file_id_by_module(&ModulePath::new("main")), Some(id));
        assert_eq!(map.file_id_by_module(&ModulePath::new("nope")), None);
    }

    #[test]
    fn insert_or_get_is_idempotent_for_same_path() {
        let map = SourceMap::new();
        let first = map
            .insert_or_get(
                PathBuf::from("/a/main.pos"),
                ModulePath::new("main"),
                b"v1",
                EDITION,
                None,
            )
            .expect("first insert");
        // Same path again — even with different content — returns the
        // SAME id; content updates go through set_file_contents.
        let second = map
            .insert_or_get(
                PathBuf::from("/a/main.pos"),
                ModulePath::new("main"),
                b"v2",
                EDITION,
                None,
            )
            .expect("idempotent re-insert");
        assert_eq!(first, second);
        assert_eq!(map.file(first).text.as_ref(), "v1");
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
    fn set_file_contents_rejects_dummy_slot() {
        let map = SourceMap::new();
        let err = map.set_file_contents(FileId::DUMMY, b"evil", None).unwrap_err();
        assert_eq!(err, SourceMapError::UnknownFileId(FileId::DUMMY));
        // Slot 0 must remain the dummy: identity, text, fingerprint.
        let d = map.file(FileId::DUMMY);
        assert_eq!(d.id, FileId::DUMMY);
        assert_eq!(d.text.as_ref(), "");
        assert_eq!(d.fingerprint, Fingerprint::of(""));
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
    fn set_file_contents_changed() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a", "a", b"v1");
        let t = SystemTime::now();
        assert!(map.set_file_contents(id, b"v2", Some(t)).unwrap());
        let f = map.file(id);
        assert_eq!(f.text.as_ref(), "v2");
        assert_eq!(f.mtime, Some(t));
        // Identity is preserved: same id → same module path.
        assert_eq!(f.module_path, ModulePath::new("a"));
    }

    #[test]
    fn set_file_contents_same_content_unchanged() {
        let map = SourceMap::new();
        let id = insert_ok(&map, "/a", "a", b"same");
        let t = SystemTime::now();
        // mtime changed but the fingerprint matches → no replacement.
        assert!(!map.set_file_contents(id, b"same", Some(t)).unwrap());
        assert_eq!(map.file(id).fingerprint, Fingerprint::of("same"));
    }

    #[test]
    fn set_file_contents_mtime_fast_path() {
        let map = SourceMap::new();
        let t = SystemTime::now();
        let id = map
            .insert(PathBuf::from("/a"), ModulePath::new("a"), b"v1", EDITION, Some(t))
            .unwrap();
        // Same mtime → unchanged, even if the content differs.
        assert!(!map.set_file_contents(id, b"completely different", Some(t)).unwrap());
        assert_eq!(map.file(id).text.as_ref(), "v1");
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
            let f = map.file(FileId::from_raw(1 + n));
            assert_ne!(f.id, FileId::DUMMY, "hole at id {}", 1 + n);
            assert_eq!(f.text.as_ref(), "");
        }
    }
}
