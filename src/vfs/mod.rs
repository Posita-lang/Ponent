use crate::ast::Program;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::parser::Parser;
use logos::Logos;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Kind of a VFS node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfsNodeKind {
    File,
    Directory,
    Root,
    SrcDirectory,
}

/// A directory entry returned by a VFS backend.
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub kind: VfsNodeKind,
}

/// Trait for VFS backends that abstract file system access.
///
/// A backend provides file content reading and directory listing,
/// allowing the VFS tree to operate over real filesystems, in-memory
/// test fixtures, or any other storage medium.
pub trait VfsBackend: fmt::Debug {
    /// Read the full contents of a file at `path`.
    ///
    /// Backends never carry diagnostics, so their error type is always
    /// `VfsError<'static>` — callers that need a shorter lifetime coerce
    /// it via a `return` (see `ensure_ast`).
    fn read_file(&self, path: &str) -> Result<String, VfsError<'static>>;

    /// List entries in a directory at `path`.
    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError<'static>>;

    /// Return the kind of the entry at `path`, or `None` if it doesn't exist.
    fn path_type(&self, path: &str) -> Option<VfsNodeKind>;

    /// Canonical identity of `path`: resolve symlinks and (on
    /// case-insensitive platforms) normalize case.  Two visible paths
    /// whose canonical forms are equal denote the same file — the
    /// caller relies on this to keep `SourceMap`'s DuplicatePath check
    /// meaningful across symlinked or aliased paths.
    ///
    /// Default: normalize the path lexically (no filesystem access).
    /// Real-filesystem backends override this with true resolution.
    fn canonicalize(&self, path: &str) -> Result<PathBuf, VfsError<'static>> {
        Ok(normalize_lexical(Path::new(path)))
    }

    /// Read a file together with its modification time.
    ///
    /// `mtime == None` tells the `SourceMap` fast path to stand down and
    /// let the fingerprint comparison decide whether content changed —
    /// backends whose content can change without an observable mtime
    /// (editor overlays, test doubles) return `None` or a fresh stamp.
    ///
    /// Default: plain `read_file` with no mtime (always fingerprint path).
    fn read_file_with_mtime(
        &self,
        path: &str,
    ) -> Result<(String, Option<std::time::SystemTime>), VfsError<'static>> {
        self.read_file(path).map(|s| (s, None))
    }
}

/// Collapse `.` / `..` segments and duplicate separators lexically.
///
/// This is the fallback canonical form for backends that have no
/// filesystem to consult (memory, overlay buffers).  A leading `/` is
/// preserved; `..` above the root clamps to the root.
fn normalize_lexical(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() {
        out.push("/");
    }
    out
}

/// Filesystem backend — reads from the real OS filesystem.
///
/// This is the default backend used in production builds.
#[derive(Debug, Clone)]
pub struct FsBackend;

impl VfsBackend for FsBackend {
    fn read_file(&self, path: &str) -> Result<String, VfsError<'static>> {
        std::fs::read_to_string(path)
            .map_err(|e| VfsError::Io(format!("failed to read '{}': {}", path, e)))
    }

    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError<'static>> {
        let dir = std::fs::read_dir(path)
            .map_err(|e| VfsError::Io(format!("failed to read dir '{}': {}", path, e)))?;
        let mut entries = Vec::new();
        for entry in dir {
            let entry = entry.map_err(|e| VfsError::Io(format!("entry error: {}", e)))?;
            let ft = entry.file_type().map_err(|e| VfsError::Io(e.to_string()))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let kind = if ft.is_dir() {
                VfsNodeKind::Directory
            } else {
                VfsNodeKind::File
            };
            entries.push(DirEntry { name, kind });
        }
        entries.sort_unstable_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
        Ok(entries)
    }

    fn path_type(&self, path: &str) -> Option<VfsNodeKind> {
        let p = Path::new(path);
        if p.is_dir() {
            Some(VfsNodeKind::Directory)
        } else if p.is_file() {
            Some(VfsNodeKind::File)
        } else {
            None
        }
    }

    fn canonicalize(&self, path: &str) -> Result<PathBuf, VfsError<'static>> {
        // True resolution: symlinks followed, case normalized by the OS.
        // Fall back to the lexical form when the target doesn't exist yet
        // (the caller may canonicalize before the first read).
        std::fs::canonicalize(path)
            .or_else(|_| Ok::<PathBuf, VfsError<'static>>(normalize_lexical(Path::new(path))))
            .map_err(|e| VfsError::Io(format!("failed to canonicalize '{}': {:?}", path, e)))
    }

    fn read_file_with_mtime(
        &self,
        path: &str,
    ) -> Result<(String, Option<std::time::SystemTime>), VfsError<'static>> {
        let content = self.read_file(path)?;
        let mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok();
        Ok((content, mtime))
    }
}

/// In-memory backend — backed by a `HashMap`, useful for testing.
///
/// All files are stored in a flat map keyed by absolute path string
/// (e.g. `/src/main.ps`).  Directories are created implicitly when
/// a file is inserted under a prefix.
#[derive(Debug, Clone)]
pub struct MemoryBackend {
    files: HashMap<String, String>,
    dirs: HashMap<String, Vec<DirEntry>>,
}

impl MemoryBackend {
    pub fn new() -> Self {
        MemoryBackend {
            files: HashMap::new(),
            dirs: HashMap::new(),
        }
    }

    /// Insert a file into the in-memory backend.
    /// `path` must be an absolute path (e.g. `/src/main.ps`).
    /// Intermediate directories are created automatically.
    pub fn insert_file(&mut self, path: &str, content: &str) {
        self.files.insert(path.to_string(), content.to_string());
        // Create implicit directory entries for each ancestor
        if let Some(parent) = Path::new(path).parent() {
            let parent_str = parent.to_string_lossy().into_owned();
            let fname = Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let entries = self.dirs.entry(parent_str.clone()).or_default();
            if !entries
                .iter()
                .any(|e| e.name == fname && e.kind == VfsNodeKind::File)
            {
                entries.push(DirEntry {
                    name: fname,
                    kind: VfsNodeKind::File,
                });
            }
            // Ensure all ancestor directories exist
            self.ensure_dir_entries(&parent_str);
        }
    }

    /// Recursively ensure all ancestor directories have entries.
    fn ensure_dir_entries(&mut self, path: &str) {
        if path.is_empty() {
            return;
        }
        // Register the current path itself as a directory so that
        // root "/" is always known to `path_type` / `read_dir`.
        self.dirs.entry(path.to_string()).or_default();

        if path == "/" {
            return;
        }
        if let Some(parent) = Path::new(path).parent() {
            let parent_str = parent.to_string_lossy().into_owned();
            let dirname = Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let entries = self.dirs.entry(parent_str.clone()).or_default();
            if !entries.iter().any(|e| e.name == dirname) {
                entries.push(DirEntry {
                    name: dirname,
                    kind: VfsNodeKind::Directory,
                });
            }
            self.ensure_dir_entries(&parent_str);
        }
    }
}

impl VfsBackend for MemoryBackend {
    fn read_file(&self, path: &str) -> Result<String, VfsError<'static>> {
        self.files
            .get(path)
            .cloned()
            .ok_or_else(|| VfsError::Io(format!("file '{}' not found in memory backend", path)))
    }

    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError<'static>> {
        self.dirs.get(path).cloned().ok_or_else(|| {
            VfsError::Io(format!("directory '{}' not found in memory backend", path))
        })
    }

    fn path_type(&self, path: &str) -> Option<VfsNodeKind> {
        if self.files.contains_key(path) {
            Some(VfsNodeKind::File)
        } else if self.dirs.contains_key(path) {
            Some(VfsNodeKind::Directory)
        } else {
            None
        }
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Overlay backend — editor unsaved buffers shadow a base backend.
///
/// A path with a buffer entry reads the buffer text; everything else
/// (including `read_dir`, `path_type`, `canonicalize`) falls through to
/// the base.  Buffered reads report a fresh mtime: the stamp is
/// guaranteed to differ from the previous compile's, so the
/// `SourceMap` mtime fast path always misses and the fingerprint
/// comparison makes the final change/no-change call.
#[derive(Debug)]
pub struct OverlayBackend {
    base: std::sync::Arc<dyn VfsBackend>,
    buffers: std::sync::RwLock<HashMap<String, String>>,
}

impl Clone for OverlayBackend {
    fn clone(&self) -> Self {
        OverlayBackend {
            base: self.base.clone(),
            // Share the buffer table: clones observe the same unsaved state.
            buffers: std::sync::RwLock::new(
                self.buffers.read().expect("overlay buffers lock").clone(),
            ),
        }
    }
}

impl OverlayBackend {
    /// Layer unsaved buffers over `base`.
    pub fn new(base: std::sync::Arc<dyn VfsBackend>) -> Self {
        OverlayBackend {
            base,
            buffers: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Install (or replace) the unsaved buffer for `path`.
    pub fn set_buffer(&self, path: &str, content: String) {
        self.buffers.write().expect("overlay buffers lock").insert(path.to_string(), content);
    }

    /// Drop the buffer for `path` — the file is saved, the base shows
    /// through again.
    pub fn clear_buffer(&self, path: &str) {
        self.buffers.write().expect("overlay buffers lock").remove(path);
    }

    /// True while `path` has an unsaved buffer.
    pub fn has_buffer(&self, path: &str) -> bool {
        self.buffers.read().expect("overlay buffers lock").contains_key(path)
    }
}

impl VfsBackend for OverlayBackend {
    fn read_file(&self, path: &str) -> Result<String, VfsError<'static>> {
        if let Some(buf) = self.buffers.read().expect("overlay buffers lock").get(path) {
            return Ok(buf.clone());
        }
        self.base.read_file(path)
    }

    fn read_file_with_mtime(
        &self,
        path: &str,
    ) -> Result<(String, Option<std::time::SystemTime>), VfsError<'static>> {
        if let Some(buf) = self.buffers.read().expect("overlay buffers lock").get(path) {
            // Fresh stamp on every read ⇒ mtime fast path always misses
            // ⇒ the fingerprint decides whether content truly changed.
            return Ok((buf.clone(), Some(std::time::SystemTime::now())));
        }
        self.base.read_file_with_mtime(path)
    }

    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError<'static>> {
        self.base.read_dir(path)
    }

    fn path_type(&self, path: &str) -> Option<VfsNodeKind> {
        self.base.path_type(path)
    }

    fn canonicalize(&self, path: &str) -> Result<PathBuf, VfsError<'static>> {
        self.base.canonicalize(path)
    }
}

/// A node in the virtual file system tree.
/// Each node can lazily load its source text, tokens, and AST.
/// Nodes form a tree via `children`; parent relationships are tracked by
/// `parent_id` rather than an owned `parent` pointer, avoiding recursive
/// cloning of the entire ancestor chain.
#[derive(Debug, Clone)]
pub struct VfsNode<'input, 'src> {
    pub kind: VfsNodeKind,
    pub name: String,
    pub id: usize,
    pub src: Option<String>,
    pub tokens: Option<Vec<Token<'input>>>,
    pub ast: Option<Program<'input, 'src>>,
    pub children: Option<Vec<VfsNode<'input, 'src>>>,
    /// Absolute virtual path in the VFS tree (e.g. `/src/main.ps`).
    pub abs_path: Option<String>,
    /// Absolute filesystem path this node maps to (if backed by a real file).
    pub fs_path: Option<String>,
    /// Shared arena backing every node's parsed AST — owned by the caller
    /// of `scan` and threaded through the whole tree, so re-parsing a node
    /// never leaks a fresh arena.  A `SyncArena` (not a bare `&Bump`) so the
    /// parser allocates through the arena's lock and the handle stays `Send`.
    pub arena: &'input crate::arena::SyncArena<'input>,
}

/// Errors that can occur during VFS operations.
#[derive(Debug, Clone)]
pub enum VfsError<'src> {
    Io(String),
    Lex(String),
    Parse(String),
    Diagnostics(Vec<Diagnostic<'src>>),
}

impl<'input, 'src> VfsNode<'input, 'src>
where
    'input: 'src,
{
    /// Create a new file node.
    pub fn new_file(id: usize, name: &str, arena: &'input crate::arena::SyncArena<'input>) -> Self {
        VfsNode {
            kind: VfsNodeKind::File,
            name: name.to_string(),
            id,
            src: None,
            tokens: None,
            ast: None,
            children: None,
            abs_path: None,
            fs_path: None,
            arena,
        }
    }

    /// Create a new directory node.
    pub fn new_dir(
        id: usize,
        name: &str,
        kind: VfsNodeKind,
        arena: &'input crate::arena::SyncArena<'input>,
    ) -> Self {
        VfsNode {
            kind,
            name: name.to_string(),
            id,
            src: None,
            tokens: None,
            ast: None,
            children: Some(Vec::new()),
            abs_path: None,
            fs_path: None,
            arena,
        }
    }

    /// Ensure the source text is loaded via the given backend.
    ///
    /// # Errors
    ///
    /// Returns `Err(VfsError)` if the source file cannot be read from the backend.
    #[must_use]
    pub fn ensure_src(&mut self, backend: &dyn VfsBackend) -> Result<(), VfsError<'static>> {
        if self.src.is_some() {
            return Ok(());
        }
        if let Some(ref path) = self.fs_path {
            let text = backend.read_file(path)?;
            self.src = Some(text);
            Ok(())
        } else {
            Err(VfsError::Io(format!(
                "no source path for node '{}'",
                self.name
            )))
        }
    }

    /// Ensure tokens are lexed from the source text.
    ///
    /// # Errors
    ///
    /// Returns `Err(VfsError)` if the source text cannot be lexed.
    #[must_use]
    pub fn ensure_tokens(&mut self, backend: &dyn VfsBackend) -> Result<(), VfsError<'static>> {
        if self.tokens.is_some() {
            return Ok(());
        }
        self.ensure_src(backend)?;
        let arena = self.arena;
        let src: &'input str = arena.alloc_str(
            self.src
                .as_deref()
                .ok_or(VfsError::Io("source not loaded".to_string()))?,
        );
        let lexer = Token::lexer(src);
        let mut tokens = Vec::new();
        for result in lexer {
            match result {
                Ok(token) => tokens.push(token),
                Err(()) => return Err(VfsError::Lex(format!("invalid token in '{}'", self.name))),
            }
        }
        self.tokens = Some(tokens);
        Ok(())
    }

    /// Ensure the AST is parsed from the source text.
    ///
    /// # Errors
    ///
    /// Returns `Err(VfsError)` if the token stream cannot be parsed.
    #[must_use]
    pub fn ensure_ast(&mut self, backend: &dyn VfsBackend) -> Result<(), VfsError<'input>> {
        if self.ast.is_some() {
            return Ok(());
        }
        // `ensure_src` yields `VfsError<'static>`; a plain `return` performs
        // the `'static → 'input` subtyping coercion that `?`'s `From` cannot.
        if let Err(e) = self.ensure_src(backend) {
            return Err(e);
        }
        let arena = self.arena;
        // Copy the source into the TREE-level shared arena (owned by the
        // caller of `scan`), so the parsed `Program<'input, 'src>` may borrow it
        // and re-parsing a node never leaks a fresh arena.  A borrow of the
        // node-local `String` could not escape into `VfsError<'input>`.
        let raw = self
            .src
            .as_deref()
            .ok_or(VfsError::Io("source not loaded".to_string()))?;
        let mut source_map = crate::sourcemap::SourceMap::new();
        let file_id = source_map
            .add_file(self.name.clone(), raw)
            .map_err(|e| VfsError::Io(e.to_string()))?;
        let src: &'input str = arena.alloc_str(raw);
        let mut parser = Parser::new(src, file_id, arena);
        match parser.parse_program() {
            Ok(program) => {
                self.ast = Some(program);
                Ok(())
            }
            Err(diags) => Err(VfsError::Diagnostics(diags)),
        }
    }

    /// Return the absolute virtual path in the VFS tree.
    /// Return the absolute virtual path in the VFS tree, if set.
    pub fn absolute_path(&self) -> Option<&str> {
        self.abs_path.as_deref()
    }

    /// Recursively scan a directory tree using the given backend and build
    /// a VFS tree from it.  Only `.ps` files are included; hidden files
    /// (dot-prefixed) and common build directories are skipped.
    ///
    /// The caller owns `arena` and must keep it alive for as long as the
    /// returned tree is used: every node's parsed `Program<'input, 'src>` borrows
    /// from it, so re-parsing a node never leaks a fresh arena.
    pub fn scan(
        backend: &dyn VfsBackend,
        root: &str,
        id_counter: &mut usize,
        arena: &'input crate::arena::SyncArena<'input>,
    ) -> Result<VfsNode<'input, 'input>, VfsError<'static>> {
        let path = Path::new(root);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| root.to_string());

        match backend.path_type(root) {
            Some(VfsNodeKind::Directory) => {
                let mut node = VfsNode::new_dir(*id_counter, &name, VfsNodeKind::Directory, arena);
                *id_counter += 1;
                node.fs_path = Some(root.to_string());
                node.abs_path = Some(root.to_string());

                let entries = backend.read_dir(root)?;
                let mut children = Vec::new();
                for entry in entries {
                    // Skip hidden files and common build artifacts
                    if entry.name.starts_with('.')
                        || entry.name == "target"
                        || entry.name == "node_modules"
                    {
                        continue;
                    }

                    let child_path = Path::new(root).join(&entry.name);
                    let child_path_str = child_path.to_string_lossy().into_owned();

                    match entry.kind {
                        VfsNodeKind::Directory => {
                            let child = Self::scan(backend, &child_path_str, id_counter, arena)?;
                            children.push(child);
                        }
                        VfsNodeKind::File => {
                            // Only include .ps files
                            if child_path.extension().is_some_and(|ext| ext == "ps") {
                                let mut f = VfsNode::new_file(*id_counter, &entry.name, arena);
                                *id_counter += 1;
                                f.abs_path = Some(child_path_str.clone());
                                f.fs_path = Some(child_path_str);
                                children.push(f);
                            }
                        }
                        _ => {}
                    }
                }

                node.children = Some(children);
                Ok(node)
            }
            Some(VfsNodeKind::File) => {
                let mut node = VfsNode::new_file(*id_counter, &name, arena);
                *id_counter += 1;
                node.abs_path = Some(root.to_string());
                node.fs_path = Some(root.to_string());
                Ok(node)
            }
            _ => Err(VfsError::Io(format!("path '{}' does not exist", root))),
        }
    }
}

impl<'input, 'src> fmt::Display for VfsNode<'input, 'src> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.absolute_path() {
            Some(path) => write!(f, "{} ({:?})", path, self.kind),
            None => write!(f, "<unnamed {:?}>", self.kind),
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// Defensive regression: `scan` must NOT leak an arena via
    /// `Box::leak`; the caller owns the `Bump` and threads it through the
    /// whole tree, so repeated scans stay bounded. This locks in the
    /// caller-owned-arena API contract.
    #[test]
    fn scan_borrows_caller_owned_arena() {
        let mut backend = MemoryBackend::new();
        backend.insert_file("/src/main.ps", "def main() -> Int<32> { return 42; }");

        // The arena is owned HERE, by the caller, and outlives the tree.
        let bump = bumpalo::Bump::new();
        let arena = crate::arena::SyncArena::new(&bump);
        let mut id_counter = 0;
        let mut root =
            VfsNode::scan(&backend, "/src", &mut id_counter, &arena).expect("scan should succeed");
        assert_eq!(id_counter, 2, "one dir + one file node allocated");

        // Parsing borrows from the caller-owned arena — no fresh arena.
        let file = root
            .children
            .as_ref()
            .expect("dir has children")
            .iter()
            .find(|c| c.kind == VfsNodeKind::File)
            .expect("one file child");
        let mut file = file.clone();
        file.ensure_ast(&backend).expect("ast should parse");
        assert!(file.ast.is_some(), "parsed program is retained");
    }

    #[test]
    fn overlay_shadows_base_and_falls_through() {
        let mut mem = MemoryBackend::new();
        mem.insert_file("/src/main.ps", "from disk");
        let overlay = OverlayBackend::new(std::sync::Arc::new(mem));

        // No buffer yet: base shows through, mtime is None (fingerprint path).
        assert!(!overlay.has_buffer("/src/main.ps"));
        let (text, mtime) = overlay.read_file_with_mtime("/src/main.ps").expect("read");
        assert_eq!(text, "from disk");
        assert!(mtime.is_none());

        // Buffer installed: it shadows the base and stamps a fresh mtime
        // so the SourceMap fast path always misses.
        overlay.set_buffer("/src/main.ps", "unsaved edit".to_string());
        let (text, mtime) = overlay.read_file_with_mtime("/src/main.ps").expect("read");
        assert_eq!(text, "unsaved edit");
        assert!(mtime.is_some());

        // Clearing the buffer exposes the base again.
        overlay.clear_buffer("/src/main.ps");
        assert_eq!(overlay.read_file("/src/main.ps").expect("read"), "from disk");
    }

    #[test]
    fn canonicalize_is_idempotent_and_normalizes() {
        let overlay =
            OverlayBackend::new(std::sync::Arc::new(MemoryBackend::new()));
        let once = overlay.canonicalize("/src/./a/../main.ps").expect("canonicalize");
        let twice = overlay.canonicalize(once.to_str().expect("utf8")).expect("canonicalize");
        assert_eq!(once, twice);
        assert_eq!(once.to_str().expect("utf8"), "/src/main.ps");
    }
}
