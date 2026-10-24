//! One registered source file: its text (BOM-stripped, UTF-8-validated),
//! the file's own edition, and the precomputed line index.
//!
//! `SourceFile` is immutable after construction — `from_bytes` is the
//! only way to make one and does everything once, in a fixed order
//! (size check → BOM strip → UTF-8 validation → line index → edition
//! header → fingerprint).  The registry hands out `Arc<SourceFile>`
//! clones, so a content update is published as a *replaced* `Arc`, never
//! a mutation of a shared file.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use memchr::memchr_iter;

use crate::ast::FileId;
use crate::hir::types::Edition;

use super::ModulePath;

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

/// 128-bit content fingerprint of a file's text (xxHash3-128).
///
/// Different texts get different fingerprints with overwhelming
/// probability; the value is a cache-invalidation key, not a security
/// primitive.
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
}

impl std::fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // High half first — the big-endian form of `as_u128`.
        write!(f, "{:016x}{:016x}", self.0[1], self.0[0])
    }
}

/// One registered source file: identity, text, and line index.
///
/// Line convention (consistent with the lexer and
/// `diagnostics::LineIndex`): the terminator is `'\n'`; in `"\r\n"` the
/// `'\r'` is line *content* and only the `'\n'` terminates; an isolated
/// `'\r'` does not terminate; `line_starts` starts with `0` and holds
/// the byte offset after every `'\n'` (a trailing `'\n'` therefore
/// leaves a final entry equal to `text.len()`); an empty file is
/// `line_starts == [0]` with one text line.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub id: FileId,
    pub physical_path: PathBuf,
    pub module_path: ModulePath,
    /// BOM-stripped, UTF-8-validated text.
    pub text: Arc<str>,
    /// The file's own edition (from its `edition = "..."` header, or
    /// the default the caller supplied).
    pub edition: Edition,
    /// `line_starts[0] == 0`, strictly increasing.
    pub line_starts: Box<[u32]>,
    pub had_bom: bool,
    /// `xxh3_128(text)` — of the BOM-stripped text.
    pub fingerprint: Fingerprint,
    pub mtime: Option<SystemTime>,
}

impl SourceFile {
    /// Build a file from raw on-disk bytes.
    ///
    /// Steps run in a fixed order:
    /// 1. size check against [`MAX_FILE_LEN`];
    /// 2. BOM strip (setting `had_bom`);
    /// 3. UTF-8 validation;
    /// 4. `line_starts` build;
    /// 5. file-header `edition = "..."` parse (absent → `default_edition`);
    /// 6. `fingerprint` of the BOM-stripped text.
    pub fn from_bytes(
        id: FileId,
        physical_path: PathBuf,
        module_path: ModulePath,
        raw: &[u8],
        default_edition: Edition,
        mtime: Option<SystemTime>,
    ) -> Result<Self, SourceFileError> {
        // 1. Addressability: every byte offset and line start must fit u32.
        if raw.len() > MAX_FILE_LEN {
            return Err(SourceFileError::TooLarge { len: raw.len() });
        }
        // 2. Strip a leading UTF-8 BOM if present.
        let (text_bytes, had_bom) = match raw {
            [0xEF, 0xBB, 0xBF, rest @ ..] => (rest, true),
            other => (other, false),
        };
        // 3. Validate UTF-8 (the compiler has no byte-oriented sources).
        let text = std::str::from_utf8(text_bytes).map_err(|_| SourceFileError::InvalidUtf8)?;
        // 4. Line starts: 0, then the byte offset after every '\n'.
        let mut line_starts = Vec::with_capacity(text.len() / 40 + 1);
        line_starts.push(0u32);
        for i in memchr_iter(b'\n', text.as_bytes()) {
            line_starts.push((i + 1) as u32);
        }
        let line_starts = line_starts.into_boxed_slice();
        // 5. File header `edition = "..."` (absent → `default_edition`).
        let edition = parse_edition_header(text, default_edition);
        // 6. Fingerprint the BOM-stripped text.
        let fingerprint = Fingerprint::of(text);
        Ok(SourceFile {
            id,
            physical_path,
            module_path,
            text: Arc::from(text.to_string()),
            edition,
            line_starts,
            had_bom,
            fingerprint,
            mtime,
        })
    }

    /// The synthetic empty file backing [`FileId::DUMMY`] (slot 0 of the
    /// registry).
    pub fn dummy() -> Self {
        SourceFile {
            id: FileId::DUMMY,
            physical_path: PathBuf::from("<dummy>"),
            module_path: ModulePath::dummy(),
            text: Arc::from(""),
            edition: Edition::latest(),
            line_starts: Box::new([0u32]),
            had_bom: false,
            fingerprint: Fingerprint::of(""),
            mtime: None,
        }
    }

    /// 0-based (line, byte-column) for a byte offset into this file.
    ///
    /// The offset is clamped to the file length (never panics on
    /// past-EOF offsets); `col` is a byte offset from the line start,
    /// matching the diagnostics layer's contract.
    pub fn line_col(&self, byte_offset: u32) -> (u32, u32) {
        let offset = byte_offset.min(self.text.len() as u32);
        let line = match self.line_starts.binary_search(&offset) {
            Ok(i) => i as u32,
            Err(i) => (i - 1) as u32,
        };
        let col = offset - self.line_starts[line as usize];
        (line, col)
    }

    /// The text of one line, without its terminator: a trailing `'\n'`
    /// (and the `'\r'` before it in CRLF) is stripped.
    ///
    /// `line` must index `line_starts` (`< line_starts.len()`); the
    /// phantom entry after a trailing newline yields `""`.
    pub fn line_text(&self, line: u32) -> &str {
        let start = self.line_starts[line as usize] as usize;
        let end = self
            .line_starts
            .get(line as usize + 1)
            .map(|&x| x as usize)
            .unwrap_or(self.text.len());
        let raw = &self.text[start..end];
        // Strip the terminator: a trailing '\n', and the '\r' before it in
        // CRLF.  (Chain on the already-stripped value — falling back to
        // `raw` here would leave bare-LF lines unterminated.)
        let without_nl = raw.strip_suffix('\n').unwrap_or(raw);
        without_nl.strip_suffix('\r').unwrap_or(without_nl)
    }

    /// Number of text lines (valid `line` values for [`Self::line_text`],
    /// excluding the phantom line after a trailing newline).  An empty
    /// file counts as one line — consistent with `LineIndex`.
    pub fn text_line_count(&self) -> u32 {
        let entries = self.line_starts.len() as u32;
        if self.text.ends_with('\n') {
            entries - 1
        } else {
            entries
        }
    }
}

/// Parse the file-header `edition = "..."` declaration, if present.
///
/// The header is the first thing in the file after whitespace and
/// comments (`//` to end-of-line, `/* … */` with nesting — the same
/// trivia the lexer skips).  Anything that does not match
/// `edition = "<value>"` — including a value [`Edition::from_str`] does
/// not know — falls back to `default`; the resolver later reports
/// unknown editions with a proper span.  Cold path (once per file
/// load), so clarity beats speed here.
fn parse_edition_header(text: &str, default: Edition) -> Edition {
    const WS: [u8; 6] = [b' ', b'\t', b'\r', b'\n', 0x0B, 0x0C];
    let bytes = text.as_bytes();
    let mut i = 0usize;
    loop {
        match bytes.get(i) {
            Some(&b) if WS.contains(&b) => i += 1,
            Some(b'/') if bytes.get(i + 1) == Some(&b'/') => {
                // Line comment — also covers `///` doc and `//!` module docs.
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            Some(b'/') if bytes.get(i + 1) == Some(&b'*') => {
                let mut depth = 1usize;
                i += 2;
                while i + 1 < bytes.len() && depth > 0 {
                    if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                        depth += 1;
                        i += 2;
                    } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if depth > 0 {
                    return default; // unterminated comment: no header
                }
            }
            _ => break,
        }
    }
    let rest = text[i..].trim_start();
    let Some(after_kw) = rest.strip_prefix("edition") else {
        return default;
    };
    if after_kw
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return default; // `edition2`, `my_edition`, … — not the keyword
    }
    let Some(after_eq) = after_kw.trim_start().strip_prefix('=') else {
        return default;
    };
    let Some(after_quote) = after_eq.trim_start().strip_prefix('"') else {
        return default;
    };
    let value = after_quote.split('"').next().unwrap_or(after_quote);
    Edition::from_str(value).unwrap_or(default)
}

#[cfg(test)]
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
            assert_eq!(f.line_starts.as_ref(), expected_starts.as_slice(), "line_starts for {text:?}");
            assert_eq!(f.text_line_count(), text_lines, "text_line_count for {text:?}");
            // Every line start maps back to (line, 0).
            for (line, &start) in f.line_starts.iter().enumerate() {
                assert_eq!(
                    f.line_col(start),
                    (line as u32, 0),
                    "round-trip for {text:?} @ {start}"
                );
            }
            // Past-EOF offsets clamp to the file length and stay valid.
            let (line, col) = f.line_col(u32::MAX);
            assert!(line < f.line_starts.len() as u32, "past-EOF line {line} for {text:?}");
            assert_eq!(col, f.text.len() as u32 - f.line_starts[line as usize]);
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
        assert_eq!(f.text.as_ref(), "def main() -> Int<32> { }");
        // The fingerprint covers the BOM-stripped text.
        assert_eq!(f.fingerprint, Fingerprint::of("def main() -> Int<32> { }"));

        let f = file(2, b"def main() {}");
        assert!(!f.had_bom);

        // A BOM-only file becomes empty.
        let f = file(3, b"\xEF\xBB\xBF");
        assert!(f.had_bom);
        assert_eq!(f.text.as_ref(), "");
        assert_eq!(f.line_starts.as_ref(), [0u32].as_slice());
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
        assert_eq!(file(1, b"edition = \"2026\";\ndef main() {}").edition, Edition::Year2026);
        assert_eq!(file(2, b"edition = \"2024\";").edition, Edition::Year2024);
        // Absent → the caller's default.
        assert_eq!(file(3, b"def main() {}").edition, Edition::Year2024);
        // Leading trivia: line comment, nested block comment, module doc.
        assert_eq!(file(4, b"// hi\nedition = \"2026\";\n").edition, Edition::Year2026);
        assert_eq!(file(5, b"/* a /* b */ c */\nedition = \"2026\";\n").edition, Edition::Year2026);
        assert_eq!(file(6, b"//! module docs\nedition = \"2024\";\n").edition, Edition::Year2024);
        // Not the first token → no header.
        assert_eq!(file(7, b"def f();\nedition = \"2026\";\n").edition, Edition::Year2024);
        // Unknown value → default (the resolver reports E070 with a span).
        assert_eq!(file(8, b"edition = \"9999\";\n").edition, Edition::Year2024);
        // Keyword-shaped identifiers are not the keyword.
        assert_eq!(file(9, b"edition2 = \"2026\";\n").edition, Edition::Year2024);
        assert_eq!(file(10, b"my_edition = \"2026\";\n").edition, Edition::Year2024);
        // Whitespace freedom.
        assert_eq!(file(11, b"  edition   =   \"2026\"  ;\n").edition, Edition::Year2026);
        // Missing `=` or the opening quote → no header.
        assert_eq!(file(12, b"edition \"2026\";\n").edition, Edition::Year2024);
        assert_eq!(file(13, b"edition = 2026;\n").edition, Edition::Year2024);
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
        assert_eq!(d.text.as_ref(), "");
        assert_eq!(d.line_starts.as_ref(), [0u32].as_slice());
        assert!(!d.had_bom);
        assert!(d.mtime.is_none());
    }
}
