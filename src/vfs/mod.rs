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
    fn read_file(&self, path: &str) -> Result<String, VfsError>;

    /// List entries in a directory at `path`.
    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError>;

    /// Return the kind of the entry at `path`, or `None` if it doesn't exist.
    fn path_type(&self, path: &str) -> Option<VfsNodeKind>;
}

/// Filesystem backend — reads from the real OS filesystem.
///
/// This is the default backend used in production builds.
#[derive(Debug, Clone)]
pub struct FsBackend;

impl VfsBackend for FsBackend {
    fn read_file(&self, path: &str) -> Result<String, VfsError> {
        std::fs::read_to_string(path)
            .map_err(|e| VfsError::Io(format!("failed to read '{}': {}", path, e)))
    }

    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError> {
        let dir = std::fs::read_dir(path)
            .map_err(|e| VfsError::Io(format!("failed to read dir '{}': {}", path, e)))?;
        let mut entries = Vec::new();
        for entry in dir {
            let entry = entry.map_err(|e| VfsError::Io(format!("entry error: {}", e)))?;
            let ft = entry.file_type().map_err(|e| VfsError::Io(e.to_string()))?;
            let name = entry
                .file_name()
                .to_string_lossy()
                .into_owned();
            let kind = if ft.is_dir() {
                VfsNodeKind::Directory
            } else {
                VfsNodeKind::File
            };
            entries.push(DirEntry { name, kind });
        }
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
            if !entries.iter().any(|e| e.name == fname && e.kind == VfsNodeKind::File) {
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
        if path.is_empty() || path == "/" {
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
    fn read_file(&self, path: &str) -> Result<String, VfsError> {
        self.files
            .get(path)
            .cloned()
            .ok_or_else(|| VfsError::Io(format!("file '{}' not found in memory backend", path)))
    }

    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError> {
        self.dirs
            .get(path)
            .cloned()
            .ok_or_else(|| VfsError::Io(format!("directory '{}' not found in memory backend", path)))
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

/// A node in the virtual file system tree.
/// Each node can lazily load its source text, tokens, and AST.
/// Nodes form a tree via `children`; parent relationships are tracked by
/// `parent_id` rather than an owned `parent` pointer, avoiding recursive
/// cloning of the entire ancestor chain.
#[derive(Debug, Clone)]
pub struct VfsNode {
    pub kind: VfsNodeKind,
    pub name: String,
    pub id: usize,
    pub src: Option<String>,
    pub tokens: Option<Vec<Token>>,
    pub ast: Option<Program>,
    pub children: Option<Vec<VfsNode>>,
    /// Absolute virtual path in the VFS tree (e.g. `/src/main.ps`).
    pub abs_path: Option<String>,
    /// Absolute filesystem path this node maps to (if backed by a real file).
    pub fs_path: Option<String>,
}

/// Errors that can occur during VFS operations.
#[derive(Debug, Clone)]
pub enum VfsError {
    Io(String),
    Lex(String),
    Parse(String),
    Diagnostics(Vec<Diagnostic>),
}

impl VfsNode {
    /// Create a new file node.
    pub fn new_file(id: usize, name: &str) -> Self {
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
        }
    }

    /// Create a new directory node.
    pub fn new_dir(id: usize, name: &str, kind: VfsNodeKind) -> Self {
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
        }
    }

    /// Ensure the source text is loaded via the given backend.
    pub fn ensure_src(&mut self, backend: &dyn VfsBackend) -> Result<(), VfsError> {
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
    pub fn ensure_tokens(&mut self, backend: &dyn VfsBackend) -> Result<(), VfsError> {
        if self.tokens.is_some() {
            return Ok(());
        }
        self.ensure_src(backend)?;
        let src = self
            .src
            .as_ref()
            .ok_or(VfsError::Io("source not loaded".to_string()))?;
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
    pub fn ensure_ast(&mut self, backend: &dyn VfsBackend) -> Result<(), VfsError> {
        if self.ast.is_some() {
            return Ok(());
        }
        self.ensure_src(backend)?;
        let src = self
            .src
            .as_ref()
            .ok_or(VfsError::Io("source not loaded".to_string()))?
            .clone();
        let mut parser = Parser::new(&src);
        match parser.parse_program() {
            Ok(program) => {
                self.ast = Some(program);
                Ok(())
            }
            Err(diags) => Err(VfsError::Diagnostics(diags)),
        }
    }

    /// Return the absolute virtual path in the VFS tree.
    /// Panics if `abs_path` was not set during construction.
    pub fn absolute_path(&self) -> &str {
        self.abs_path.as_deref().unwrap_or("<unknown>")
    }

    /// Recursively scan a directory tree using the given backend and build
    /// a VFS tree from it.  Only `.ps` files are included; hidden files
    /// (dot-prefixed) and common build directories are skipped.
    pub fn scan(
        backend: &dyn VfsBackend,
        root: &str,
        id_counter: &mut usize,
    ) -> Result<VfsNode, VfsError> {
        let path = Path::new(root);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| root.to_string());

        match backend.path_type(root) {
            Some(VfsNodeKind::Directory) => {
                let mut node = VfsNode::new_dir(*id_counter, &name, VfsNodeKind::Directory);
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
                            let child =
                                Self::scan(backend, &child_path_str, id_counter)?;
                            children.push(child);
                        }
                        VfsNodeKind::File => {
                            // Only include .ps files
                            if child_path.extension().map_or(false, |ext| ext == "ps") {
                                let mut f = VfsNode::new_file(*id_counter, &entry.name);
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
                let mut node = VfsNode::new_file(*id_counter, &name);
                *id_counter += 1;
                node.abs_path = Some(root.to_string());
                node.fs_path = Some(root.to_string());
                Ok(node)
            }
            _ => Err(VfsError::Io(format!("path '{}' does not exist", root))),
        }
    }
}

impl fmt::Display for VfsNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = self.absolute_path();
        write!(f, "{} ({:?})", path, self.kind)
    }
}