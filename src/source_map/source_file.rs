//! One registered source file: identity plus a shared content payload.
//!
//! The mutable-in-time identity (`id`, paths, `mtime`, `had_bom`, the
//! caller-supplied default edition) is split from the immutable derived
//! payload (`text`, line starts, fingerprint, parsed header edition) —
//! the latter lives on [`super::content::Content`] and is shared across
//! every file with identical bytes.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use crate::ast::FileId;
use crate::hir::types::Edition;

use super::ModulePath;
use super::content::Content;

/// Largest file this compiler can address: spans store byte offsets in
/// `u32`, so a source file may be at most `u32::MAX` (4 GiB) bytes.
pub const MAX_FILE_LEN: usize = u32::MAX as usize;

/// Failure to build a [`SourceFile`] from raw bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFileError {
    /// The raw bytes exceed [`MAX_FILE_LEN`]; spans could not address
    /// the file's tail.
    TooLarge { len: usize },
    /// The bytes (after BOM stripping) are not valid UTF-8.
    InvalidUtf8,
}

impl std::fmt::Display for SourceFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceFileError::TooLarge { len } => write!(
                f,
                "source file is {len} bytes; at most {MAX_FILE_LEN} (4 GiB) are addressable"
            ),
            SourceFileError::InvalidUtf8 => write!(f, "source file is not valid UTF-8"),
        }
    }
}

impl std::error::Error for SourceFileError {}

/// 128-bit content fingerprint (xxHash3-128).
///
/// Different texts get different fingerprints with overwhelming
/// probability; the value is a cache key, not a security primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fingerprint([u64; 2]);

impl Fingerprint {
    /// The fingerprint of exactly the bytes `text` holds (callers pass
    /// the BOM-stripped text).
    pub fn of(text: &str) -> Self {
        let value = xxhash_rust::xxh3::xxh3_128(text.as_bytes());
        Fingerprint([value as u64, (value >> 64) as u64])
    }

    /// The 128-bit value (`hi` is the high 64 bits).
    pub const fn as_u128(self) -> u128 {
        ((self.0[1] as u128) << 64) | self.0[0] as u128
    }

    /// Test-only: construct a fingerprint from arbitrary bits.  Used by
    /// the shard-index / collision tests in `content.rs`.
    #[cfg(all(test, not(feature = "shuttle")))]
    pub(in crate::source_map) const fn from_u128_for_tests(v: u128) -> Self {
        Fingerprint([v as u64, (v >> 64) as u64])
    }
}

impl std::fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // High half first — the big-endian form of `as_u128`.
        write!(f, "{:016x}{:016x}", self.0[1], self.0[0])
    }
}

/// One registered source file: identity plus a shared content payload.
///
/// `text`, `line_starts`, and `fingerprint` live on the shared
/// [`Content`]; use the accessor methods below rather than reaching
/// into `content` (which is `pub(crate)` and not part of the public
/// API).
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub id: FileId,
    pub physical_path: PathBuf,
    pub module_path: ModulePath,
    /// The caller-supplied **default** edition.  Preserved across
    /// reloads so that a header-bearing reload cannot make the previous
    /// header sticky when the header is subsequently removed.
    pub default_edition: Edition,
    /// The **resolved** edition:
    /// `content.header_edition.unwrap_or(default_edition)`.
    pub edition: Edition,
    /// Per-raw-bytes metadata; cannot live on `Content` because it
    /// describes the on-disk bytes, not their canonical text.
    pub had_bom: bool,
    /// Shared dedup handle.  Two files with identical bytes point at the
    /// same `Arc<Content>`.
    pub(crate) content: Arc<Content>,
    pub mtime: Option<SystemTime>,
}

impl SourceFile {
    // ── Accessors over the shared payload ──────────────────────────

    /// The canonical (BOM-stripped) text, shared across files with
    /// identical bytes.
    #[inline]
    pub fn text(&self) -> &str {
        &self.content.text
    }

    /// Cheap `Arc<str>` clone — for callers that need to hold the text
    /// independently of the `SourceFile`.
    #[inline]
    pub fn text_arc(&self) -> Arc<str> {
        Arc::clone(&self.content.text)
    }

    /// `line_starts[0] == 0`, strictly increasing, byte offsets.
    #[inline]
    pub fn line_starts(&self) -> &[u32] {
        &self.content.line_starts
    }

    #[inline]
    pub fn fingerprint(&self) -> Fingerprint {
        self.content.fingerprint
    }

    // ── Construction ───────────────────────────────────────────────

    /// Assemble a `SourceFile` from an already-interned [`Content`].
    /// Infallible — all fallible work (size, BOM, UTF-8) has already
    /// happened in the store's `intern`.  Called only by
    /// `SourceMap::insert` / `SourceMap::set_file_contents` and by
    /// tests within the crate.
    pub(crate) fn from_parts(
        id: FileId,
        physical_path: PathBuf,
        module_path: ModulePath,
        content: Arc<Content>,
        had_bom: bool,
        default_edition: Edition,
        mtime: Option<SystemTime>,
    ) -> Self {
        // Resolve the edition now so `file.edition` is O(1) at every
        // read.  The header parse result (`header_edition`) is cached on
        // the shared `Content`; only the fallback default is per-file.
        let edition = content.header_edition.unwrap_or(default_edition);
        SourceFile {
            id,
            physical_path,
            module_path,
            default_edition,
            edition,
            had_bom,
            content,
            mtime,
        }
    }

    /// Direct construction from raw bytes — used only by tests and by
    /// code that does not own a [`super::SourceMap`].  Production
    /// callers should use `SourceMap::insert`, which shares payloads
    /// across files with identical bytes.
    pub fn from_bytes(
        id: FileId,
        physical_path: PathBuf,
        module_path: ModulePath,
        raw: &[u8],
        default_edition: Edition,
        mtime: Option<SystemTime>,
    ) -> Result<Self, SourceFileError> {
        if raw.len() > MAX_FILE_LEN {
            return Err(SourceFileError::TooLarge { len: raw.len() });
        }
        let (text_bytes, had_bom) = match raw {
            [0xEF, 0xBB, 0xBF, rest @ ..] => (rest, true),
            other => (other, false),
        };
        let text = std::str::from_utf8(text_bytes).map_err(|_| SourceFileError::InvalidUtf8)?;
        let fp = Fingerprint::of(text);
        let content = Arc::new(Content::build(text, fp));
        Ok(Self::from_parts(
            id,
            physical_path,
            module_path,
            content,
            had_bom,
            default_edition,
            mtime,
        ))
    }

    /// The synthetic empty file backing `FileId::DUMMY` (slot 0 of the
    /// registry).  Its `Content` is private to this file — not the
    /// store's empty singleton — because the dummy is a per-`SourceMap`
    /// sentinel, not interned content.
    pub fn dummy() -> Self {
        SourceFile {
            id: FileId::DUMMY,
            physical_path: PathBuf::from("<dummy>"),
            module_path: ModulePath::dummy(),
            default_edition: Edition::latest(),
            edition: Edition::latest(),
            had_bom: false,
            content: Arc::new(Content::empty()),
            mtime: None,
        }
    }

    // ── Line arithmetic ────────────────────────────────────────────

    /// 0-based (line, byte-column) for a byte offset into this file.
    ///
    /// The offset is clamped to the file length (never panics on
    /// past-EOF offsets); `col` is a byte offset from the line start,
    /// matching the diagnostics layer's contract.
    pub fn line_col(&self, byte_offset: u32) -> (u32, u32) {
        let text_len = self.content.text.len() as u32;
        let offset = byte_offset.min(text_len);
        let line = match self.content.line_starts.binary_search(&offset) {
            Ok(i) => i as u32,
            Err(i) => (i - 1) as u32,
        };
        let col = offset - self.content.line_starts[line as usize];
        (line, col)
    }

    /// The text of one line, without its terminator: a trailing `'\n'`
    /// (and the `'\r'` before it in CRLF) is stripped.
    ///
    /// `line` must index `line_starts` (`< line_starts.len()`); the
    /// phantom entry after a trailing newline yields `""`.
    pub fn line_text(&self, line: u32) -> &str {
        let start = self.content.line_starts[line as usize] as usize;
        let end = self
            .content
            .line_starts
            .get(line as usize + 1)
            .map(|&x| x as usize)
            .unwrap_or(self.content.text.len());
        let raw = &self.content.text[start..end];
        // Strip the terminator: a trailing '\n', and the '\r' before it
        // in CRLF.  (Chain on the already-stripped value — falling back
        // to `raw` here would leave bare-LF lines unterminated.)
        let without_nl = raw.strip_suffix('\n').unwrap_or(raw);
        without_nl.strip_suffix('\r').unwrap_or(without_nl)
    }

    /// Number of text lines (valid `line` values for [`Self::line_text`],
    /// excluding the phantom line after a trailing newline).  An empty
    /// file counts as one line — consistent with `LineIndex`.
    pub fn text_line_count(&self) -> u32 {
        let entries = self.content.line_starts.len() as u32;
        if self.content.text.ends_with('\n') {
            entries - 1
        } else {
            entries
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    const DEFAULT: Edition = Edition::Year2024;

    fn file(id: u32, raw: &[u8]) -> SourceFile {
        SourceFile::from_bytes(
            FileId::from_raw(id),
            PathBuf::from(format!("test/{id}.pos")),
            ModulePath::new(format!("test::m{id}")),
            raw,
            DEFAULT,
            None,
        )
        .unwrap_or_else(|e| panic!("test input {raw:?} must build: {e}"))
    }

    /// The same boundary matrix `diagnostics::LineIndex` pins: empty
    /// input, single byte, with and without trailing newlines, multi-byte
    /// leading char, and CRLF.  Every conversion must stay panic-free and
    /// self-consistent.
    #[test]
    fn line_starts_boundaries_stay_consistent() {
        let cases: [(&str, Vec<u32>, u32); 7] = [
            ("", vec![0], 1),
            ("a", vec![0], 1),
            ("a\n", vec![0, 2], 1),
            ("a\n\n", vec![0, 2, 3], 2),
            ("中文\n", vec![0, 7], 1),
            ("\t\n", vec![0, 2], 1),
            ("abc\r\ndef\r\n", vec![0, 5, 10], 2),
        ];
        for (text, expected_starts, text_lines) in cases {
            let f = file(1, text.as_bytes());
            assert_eq!(
                f.line_starts(),
                expected_starts.as_slice(),
                "line_starts for {text:?}"
            );
            assert_eq!(
                f.text_line_count(),
                text_lines,
                "text_line_count for {text:?}"
            );
            for (line, &start) in f.line_starts().iter().enumerate() {
                assert_eq!(
                    f.line_col(start),
                    (line as u32, 0),
                    "round-trip for {text:?} @ {start}"
                );
            }
            let (line, col) = f.line_col(u32::MAX);
            assert!(line < f.line_starts().len() as u32);
            assert_eq!(col, f.text().len() as u32 - f.line_starts()[line as usize]);
        }
    }

    #[test]
    fn line_text_strips_terminator() {
        let f = file(1, b"abc\r\ndef\r\n");
        assert_eq!(f.line_text(0), "abc");
        assert_eq!(f.line_text(1), "def");

        let f = file(2, b"one\ntwo");
        assert_eq!(f.line_text(0), "one");
        assert_eq!(f.line_text(1), "two");

        // Isolated '\r' is content, not a terminator.
        let f = file(3, b"lone\rcr");
        assert_eq!(f.line_text(0), "lone\rcr");
        assert_eq!(f.text_line_count(), 1);

        let f = file(4, b"");
        assert_eq!(f.line_text(0), "");
        assert_eq!(f.text_line_count(), 1);
    }

    #[test]
    fn bom_stripped_and_flagged() {
        let f = file(1, b"\xEF\xBB\xBFdef main() -> Int<32> { }");
        assert!(f.had_bom);
        assert_eq!(f.text(), "def main() -> Int<32> { }");
        assert_eq!(
            f.fingerprint(),
            Fingerprint::of("def main() -> Int<32> { }")
        );

        let f = file(2, b"def main() {}");
        assert!(!f.had_bom);

        // A BOM-only file becomes empty.
        let f = file(3, b"\xEF\xBB\xBF");
        assert!(f.had_bom);
        assert_eq!(f.text(), "");
        assert_eq!(f.line_starts(), [0u32].as_slice());
    }

    #[test]
    fn invalid_utf8_rejected() {
        let err = SourceFile::from_bytes(
            FileId::from_raw(1),
            PathBuf::from("bad.pos"),
            ModulePath::new("bad"),
            b"ok \xff\xfe",
            DEFAULT,
            None,
        )
        .unwrap_err();
        assert_eq!(err, SourceFileError::InvalidUtf8);
        assert_eq!(err.to_string(), "source file is not valid UTF-8");
    }

    #[test]
    fn edition_header_variants() {
        assert_eq!(
            file(1, b"edition = \"2026\";\ndef main() {}").edition,
            Edition::Year2026
        );
        assert_eq!(file(2, b"edition = \"2024\";").edition, Edition::Year2024);
        // Absent → the caller's default.
        assert_eq!(file(3, b"def main() {}").edition, Edition::Year2024);
        // Leading trivia: line comment, nested block comment, module doc.
        assert_eq!(
            file(4, b"// hi\nedition = \"2026\";\n").edition,
            Edition::Year2026
        );
        assert_eq!(
            file(5, b"/* a /* b */ c */\nedition = \"2026\";\n").edition,
            Edition::Year2026
        );
        assert_eq!(
            file(6, b"//! module docs\nedition = \"2024\";\n").edition,
            Edition::Year2024
        );
        // Not the first token → no header.
        assert_eq!(
            file(7, b"def f();\nedition = \"2026\";\n").edition,
            Edition::Year2024
        );
        // Unknown value → default (the resolver reports E070 with a span).
        assert_eq!(file(8, b"edition = \"9999\";\n").edition, Edition::Year2024);
        // Keyword-shaped identifiers are not the keyword.
        assert_eq!(
            file(9, b"edition2 = \"2026\";\n").edition,
            Edition::Year2024
        );
        assert_eq!(
            file(10, b"my_edition = \"2026\";\n").edition,
            Edition::Year2024
        );
        // Whitespace freedom.
        assert_eq!(
            file(11, b"  edition   =   \"2026\"  ;\n").edition,
            Edition::Year2026
        );
        // Missing `=` or the opening quote → no header.
        assert_eq!(file(12, b"edition \"2026\";\n").edition, Edition::Year2024);
        assert_eq!(file(13, b"edition = 2026;\n").edition, Edition::Year2024);
    }

    #[test]
    fn default_edition_preserved_through_from_parts() {
        // A header-less content with default = 2026 resolves to 2026.
        let f = SourceFile::from_bytes(
            FileId::from_raw(1),
            PathBuf::from("/a"),
            ModulePath::new("a"),
            b"def f() {}",
            Edition::Year2026,
            None,
        )
        .unwrap();
        assert_eq!(f.default_edition, Edition::Year2026);
        assert_eq!(f.edition, Edition::Year2026);
    }

    #[test]
    fn fingerprint_separates_content() {
        assert_eq!(Fingerprint::of("a"), Fingerprint::of("a"));
        assert_ne!(Fingerprint::of("a"), Fingerprint::of("b"));
        assert_eq!(Fingerprint::of(""), Fingerprint::of(""));
        // 128 bits → 32 hex digits in the Display form.
        assert_eq!(Fingerprint::of("x").to_string().len(), 32);
        // The Display form is the hex of as_u128.
        let fp = Fingerprint::of("hex-check");
        assert_eq!(format!("{fp}"), format!("{:032x}", fp.as_u128()));
    }

    #[test]
    fn dummy_is_empty() {
        let d = SourceFile::dummy();
        assert_eq!(d.id, FileId::DUMMY);
        assert_eq!(d.text(), "");
        assert_eq!(d.line_starts(), [0u32].as_slice());
        assert!(!d.had_bom);
        assert!(d.mtime.is_none());
    }
}
