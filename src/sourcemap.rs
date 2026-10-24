//! The compiler's [`SourceMap`]: registry of source files addressed by
//! [`FileId`](crate::ast::FileId).
//!
//! Every file the compiler reads is registered exactly once, in load order.
//! `FileId` values are dense indices into this registry, which makes span
//! ordering (`Span`'s `Ord` sorts by file first) equivalent to registration
//! order and lets [`SourceMap::file`] be infallible.

use crate::ast::FileId;
use crate::diagnostics::floor_char_boundary;
use crate::diagnostics::label::SourcePos;
use memchr::memchr_iter;

/// Largest file this compiler can address: spans store byte offsets in
/// `u32`, so a source file may be at most `u32::MAX` (4 GiB) bytes.
pub const MAX_FILE_LEN: usize = u32::MAX as usize;

/// Error returned by [`SourceMap::add_file`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceMapError {
    /// The file's text exceeds [`MAX_FILE_LEN`]; spans could not
    /// address its tail.
    FileTooLarge { len: usize },
    /// More than `u32::MAX` files registered; new ids would not fit
    /// in a [`FileId`].
    TooManyFiles,
}

impl std::fmt::Display for SourceMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceMapError::FileTooLarge { len } => write!(
                f,
                "source file is {len} bytes; at most {MAX_FILE_LEN} (4 GiB) are addressable"
            ),
            SourceMapError::TooManyFiles => {
                write!(f, "cannot register more than {} source files", u32::MAX)
            }
        }
    }
}

impl std::error::Error for SourceMapError {}

/// One registered source file: its display name, full text, and the
/// precomputed byte offset of each line start (always begins with `0`).
#[derive(Debug, Clone)]
pub struct SourceFile {
    name: String,
    source: String,
    line_starts: Vec<usize>,
}

impl SourceFile {
    fn new(name: String, source: String) -> Self {
        let mut line_starts = Vec::with_capacity(source.len() / 40 + 1);
        line_starts.push(0);
        for i in memchr_iter(b'\n', source.as_bytes()) {
            line_starts.push(i + 1);
        }
        SourceFile {
            name,
            source,
            line_starts,
        }
    }

    /// The file's display name (typically its path, as given at registration).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The file's full text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// 0-based (line, byte-column) for a byte offset into this file.
    ///
    /// The offset is floored to a char boundary (never panics on slicing)
    /// and clamped to the file length. `col` is a byte offset from the
    /// start of the line, matching [`SourcePos`]'s contract.
    pub fn line_col(&self, offset: u32) -> SourcePos {
        let clamped = floor_char_boundary(&self.source, offset as usize);
        let line = match self.line_starts.binary_search(&clamped) {
            Ok(line) => line,
            Err(0) => 0,
            Err(next) => next - 1,
        };
        SourcePos {
            line,
            col: clamped - self.line_starts[line],
        }
    }
}

/// Registry of source files, indexed by [`FileId`].
///
/// Slot 0 is a pre-registered dummy file (empty, name `"<dummy>"`)
/// backing [`FileId::DUMMY`], so [`SourceMap::file`] is infallible:
/// every valid `FileId` — including the sentinel — has an entry.
#[derive(Debug, Clone)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceMap {
    /// An empty map whose slot 0 is the dummy file for [`FileId::DUMMY`].
    pub fn new() -> Self {
        SourceMap {
            files: vec![SourceFile::new("<dummy>".to_owned(), String::new())],
        }
    }

    /// Register a file and return its [`FileId`].
    ///
    /// Ids are assigned in load order starting at [`FileId::FIRST_REAL`]
    /// (the dummy occupies id 0, so a file's id equals its vector index).
    pub fn add_file(
        &mut self,
        name: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<FileId, SourceMapError> {
        let source = source.into();
        if source.len() > MAX_FILE_LEN {
            return Err(SourceMapError::FileTooLarge {
                len: source.len(),
            });
        }
        // The dummy keeps `files.len() >= 1`, so the raw id is `>= 1`
        // (i.e. `>= FIRST_REAL`). A `u32` index would require 4 billion
        // registered files — far beyond addressable memory — but fall
        // back to an error rather than a truncating cast.
        let raw = u32::try_from(self.files.len()).map_err(|_| SourceMapError::TooManyFiles)?;
        self.files.push(SourceFile::new(name.into(), source));
        Ok(FileId::from_raw(raw))
    }

    /// The file registered under `id`.
    ///
    /// Infallible: slot 0 holds the dummy file, and every other id is
    /// only ever produced by [`SourceMap::add_file`].
    pub fn file(&self, id: FileId) -> &SourceFile {
        &self.files[id.as_u32() as usize]
    }

    /// How many files are registered (including the dummy).
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// True when only the dummy file is registered.
    pub fn is_empty(&self) -> bool {
        self.files.len() <= 1
    }
}
