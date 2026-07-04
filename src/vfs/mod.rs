use crate::ast::Program;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::parser::Parser;
use logos::Logos;
use std::path::{Path, PathBuf};

/// Kind of a VFS node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfsNodeKind {
    File,
    Directory,
    Root,
    SrcDirectory,
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

    /// Ensure the source text is loaded (from the filesystem if `fs_path` is set,
    /// or from inline content already stored in `src`).
    pub fn ensure_src(&mut self) -> Result<(), VfsError> {
        if self.src.is_some() {
            return Ok(());
        }
        if let Some(ref path) = self.fs_path {
            match std::fs::read_to_string(path) {
                Ok(text) => {
                    self.src = Some(text);
                    Ok(())
                }
                Err(e) => Err(VfsError::Io(format!("failed to read '{}': {}", path, e))),
            }
        } else {
            Err(VfsError::Io(format!(
                "no source path for node '{}'",
                self.name
            )))
        }
    }

    /// Ensure tokens are lexed from the source text.
    pub fn ensure_tokens(&mut self) -> Result<(), VfsError> {
        if self.tokens.is_some() {
            return Ok(());
        }
        self.ensure_src()?;
        let src = self.src.as_ref().unwrap();
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

    /// Ensure the AST is parsed from the token stream.
    pub fn ensure_ast(&mut self) -> Result<(), VfsError> {
        if self.ast.is_some() {
            return Ok(());
        }
        self.ensure_src()?;
        let src = self.src.as_ref().unwrap().clone();
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

    /// Recursively scan a real directory and build a VFS tree from it.
    /// `prefix` is the virtual path prefix accumulated from ancestors.
    pub fn scan_fs(root: &Path, id_counter: &mut usize) -> Result<VfsNode, VfsError> {
        let name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| root.to_string_lossy().to_string());
        let abs_path = root.to_string_lossy().to_string();

        if root.is_dir() {
            let mut node = VfsNode::new_dir(*id_counter, &name, VfsNodeKind::Directory);
            *id_counter += 1;
            node.fs_path = Some(abs_path.clone());
            node.abs_path = Some(abs_path);

            let mut children = Vec::new();
            let entries = std::fs::read_dir(root)
                .map_err(|e| VfsError::Io(format!("failed to read dir: {}", e)))?;

            for entry in entries {
                let entry = entry.map_err(|e| VfsError::Io(format!("entry error: {}", e)))?;
                let path = entry.path();
                let child_name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();

                // Skip hidden files and common build artifacts
                if child_name.starts_with('.')
                    || child_name == "target"
                    || child_name == "node_modules"
                {
                    continue;
                }

                let mut child = if path.is_dir() {
                    Self::scan_fs(&path, id_counter)?
                } else if path.extension().map_or(false, |ext| ext == "ps") {
                    let mut f = VfsNode::new_file(*id_counter, &child_name);
                    *id_counter += 1;
                    let child_abs = path.to_string_lossy().to_string();
                    f.abs_path = Some(child_abs.clone());
                    f.fs_path = Some(child_abs);
                    f
                } else {
                    continue; // skip non-.ps files
                };

                children.push(child);
            }

            node.children = Some(children);
            Ok(node)
        } else {
            let mut node = VfsNode::new_file(*id_counter, &name);
            *id_counter += 1;
            node.abs_path = Some(abs_path.clone());
            node.fs_path = Some(abs_path);
            Ok(node)
        }
    }
}

impl std::fmt::Display for VfsNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path = self.absolute_path();
        write!(f, "{} ({:?})", path, self.kind)
    }
}
