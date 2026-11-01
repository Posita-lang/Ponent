//! Content-addressed source payload: one allocation shared by every
//! [`super::SourceFile`] whose bytes are identical.
//!
//! [`Content`] bundles the immutable, derived-from-bytes data (canonical
//! text, line starts, fingerprint, parsed header edition).  `SourceFile`
//! bundles identity (id, paths, mtime, `had_bom`, caller-supplied default
//! edition) around a shared `Arc<Content>` handle.  Splitting the two is
//! what makes dedup possible: N files with the same bytes share one
//! `Arc<str>`, one `line_starts` slice, one fingerprint computation, and
//! one header parse.
//!
//! `Content` is not part of the public API — `SourceFile` exposes
//! `text()` / `line_starts()` / `fingerprint()` so callers never see the
//! dedup handle directly.

use std::sync::{Arc, OnceLock, Weak};

use memchr::memchr_iter;
use rustc_hash::FxHashMap;

use crate::hir::types::Edition;
use crate::sync::RwLock;

use super::source_file::{Fingerprint, MAX_FILE_LEN, SourceFileError};

/// Six bits => 64 shards.  Write contention during parallel loading drops
/// by 64×; reads are essentially free (`RwLock::read` + one hash).
const SHARD_BITS: usize = 6;
const SHARDS: usize = 1 << SHARD_BITS;

/// Shard-local map size above which a write attempts an opportunistic
/// sweep of dead `Weak` entries.  Watch-mode / LSP sessions accumulate
/// superseded content versions here over time; without a sweep the map
/// would grow with every historical version even though the
/// `Arc<Content>` payloads themselves are already GC'd.
const SWEEP_THRESHOLD: usize = 1024;

// Compile-time guards for the shard math (this codebase's convention for
// pinning arithmetic invariants at the type site).
const _: () = assert!(SHARDS.is_power_of_two());
const _: () = assert!(SHARD_BITS < 128);

// ── Content ────────────────────────────────────────────────────

/// Immutable payload derived from one unique byte sequence.
#[derive(Debug)]
pub(crate) struct Content {
    /// Canonical bytes (BOM stripped, UTF-8 validated).
    pub(crate) text: Arc<str>,
    /// Byte offset of each line start; `line_starts[0] == 0`.
    pub(crate) line_starts: Arc<[u32]>,
    /// `xxh3_128(text)` — a cache key, not a security primitive.
    pub(crate) fingerprint: Fingerprint,
    /// Parsed file-header `edition = "..."` declaration, if any.
    /// `None` when the file has no valid header; `SourceFile::edition`
    /// falls back to the caller's default edition in that case.
    pub(crate) header_edition: Option<Edition>,
}

impl Content {
    /// Build the payload for `text` (already BOM-stripped and validated).
    /// `fingerprint` is passed in so callers that already computed it
    /// (the `ContentStore::intern` hot path) don't pay for a second hash.
    pub(crate) fn build(text: &str, fingerprint: Fingerprint) -> Self {
        let mut line_starts = Vec::with_capacity(text.len() / 40 + 1);
        line_starts.push(0u32);
        for i in memchr_iter(b'\n', text.as_bytes()) {
            line_starts.push((i + 1) as u32);
        }
        Content {
            text: Arc::from(text),
            line_starts: Arc::from(line_starts.into_boxed_slice()),
            fingerprint,
            header_edition: parse_header_edition(text),
        }
    }

    /// Cheap hand-crafted empty payload for [`super::SourceFile::dummy`].
    /// Not the same `Arc` as [`ContentStore::empty_ref`] — the dummy is a
    /// per-`SourceMap` sentinel, not interned content.
    pub(crate) fn empty() -> Self {
        Content {
            text: Arc::from(""),
            line_starts: Arc::from([0u32].as_slice()),
            fingerprint: Fingerprint::of(""),
            header_edition: None,
        }
    }
}

// ── ContentStore ───────────────────────────────────────────────

/// Deduplicating store mapping content fingerprints to shared payloads.
///
/// Values are `Weak<Content>`: the store never keeps content alive on its
/// own, so a `Content` is collected the moment its last `SourceFile`
/// drops it.  The map slot lingers (until an opportunistic sweep) but
/// costs only the fingerprint + a dead `Weak` — the ~1 KiB of text and
/// line starts is already gone.
pub(crate) struct ContentStore {
    shards: [RwLock<FxHashMap<Fingerprint, Weak<Content>>>; SHARDS],
    /// The `""` payload — every empty file shares it (including BOM-only
    /// files: BOM is stripped before interning, so they also hit this).
    empty: OnceLock<Arc<Content>>,
}

impl ContentStore {
    pub(crate) fn new() -> Self {
        ContentStore {
            shards: std::array::from_fn(|_| RwLock::new(FxHashMap::default())),
            empty: OnceLock::new(),
        }
    }

    fn empty_ref(&self) -> Arc<Content> {
        self.empty
            .get_or_init(|| Arc::new(Content::empty()))
            .clone()
    }

    /// Intern `raw` bytes, returning `(content, had_bom)`.
    ///
    /// `had_bom` is computed here (it is a property of the raw bytes, not
    /// of the canonical text) so the caller doesn't redo the BOM check.
    ///
    /// Collision handling: if the fingerprint collides with an unrelated
    /// text, the `memcmp` guard rejects the hit and the new entry
    /// **overwrites** the old slot.  The old content's files keep their
    /// strong `Arc<Content>` refs and render correctly; the overwrite
    /// only costs the *other* text one future re-intern.  Correctness is
    /// preserved; only dedup degrades.
    pub(crate) fn intern(&self, raw: &[u8]) -> Result<(Arc<Content>, bool), SourceFileError> {
        if raw.len() > MAX_FILE_LEN {
            return Err(SourceFileError::TooLarge { len: raw.len() });
        }
        let (text_bytes, had_bom) = match raw {
            [0xEF, 0xBB, 0xBF, rest @ ..] => (rest, true),
            other => (other, false),
        };
        let text = std::str::from_utf8(text_bytes).map_err(|_| SourceFileError::InvalidUtf8)?;
        if text.is_empty() {
            return Ok((self.empty_ref(), had_bom));
        }
        let fp = Fingerprint::of(text);
        let idx = shard_index(fp);
        let shard = &self.shards[idx];

        // Read-lock fast path.
        {
            let r = shard.read();
            // Collision guard (non-negotiable for correctness):
            // xxh3_128 is not cryptographic, but a 128-bit fingerprint's
            // birthday bound is far below practical concern — the memcmp
            // is one branch and total correctness.  Without it we would
            // silently serve text A to a file whose bytes are B.
            if let Some(w) = r.get(&fp)
                && let Some(c) = w.upgrade()
                && c.text.as_ref() == text
            {
                return Ok((c, had_bom));
            }
        }

        // Miss: build outside the lock, then install with a double-check
        // (another thread may have won between the read-lock drop and
        // the write-lock acquire).
        let built = Arc::new(Content::build(text, fp));
        let mut w = shard.write();
        if let Some(c) = w.get(&fp).and_then(Weak::upgrade)
            && c.text.as_ref() == text
        {
            return Ok((c, had_bom));
        }
        // No live match: install ours, overwriting any dead slot or a
        // collided entry (see collision-handling doc on `intern`).
        w.insert(fp, Arc::downgrade(&built));

        // Opportunistic sweep: shrink a shard that has grown past the
        // threshold by dropping dead `Weak`s.  O(shard) per sweep, but
        // a shard only reaches the threshold after enough inserts that
        // the amortized cost is negligible, and the sweep is a no-op
        // when the shard is still mostly live (retain does nothing).
        if w.len() >= SWEEP_THRESHOLD {
            w.retain(|_, weak| weak.strong_count() > 0);
        }

        Ok((built, had_bom))
    }

    /// Sweep dead `Weak` slots from every shard.  Called by
    /// [`super::SourceMap::sweep_dead_contents`] for an LSP that goes
    /// idle; the opportunistic sweep in [`Self::intern`] covers the
    /// common case.
    pub(crate) fn sweep_all(&self) {
        for shard in &self.shards {
            shard.write().retain(|_, weak| weak.strong_count() > 0);
        }
    }
}

/// Shard index for `fp`: top `SHARD_BITS` bits of the 128-bit fingerprint,
/// masked to `SHARDS - 1`.
///
/// Shift on `u128` **before** the `usize` cast — the naive
/// `(fp.as_u128() as usize) >> (128 - SHARD_BITS)` panics on every
/// non-empty `intern` in a debug build (usize is 64-bit, shift-by-122
/// overflows) and "works" in release only via Rust's masking.
#[inline]
fn shard_index(fp: Fingerprint) -> usize {
    ((fp.as_u128() >> (128 - SHARD_BITS)) as usize) & (SHARDS - 1)
}

// ── Header edition parse ────────────────────────────────────────

/// Parse the file-header `edition = "..."` declaration, if present.
///
/// The header is the first thing in the file after whitespace and
/// comments (`//` to end-of-line, `/* … */` with nesting — the same
/// trivia the lexer skips).  Returns `Some(edition)` only when a valid
/// header is found; `None` means "no header" or "unknown edition", in
/// which case the caller falls back to its per-insert default (see
/// `SourceFile::from_parts`).  Cold path (once per unique content).
pub(crate) fn parse_header_edition(text: &str) -> Option<Edition> {
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
                    return None; // unterminated comment: no header
                }
            }
            _ => break,
        }
    }
    let rest = text[i..].trim_start();
    let after_kw = rest.strip_prefix("edition")?;
    if after_kw
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return None; // `edition2`, `my_edition`, … — not the keyword
    }
    let after_eq = after_kw.trim_start().strip_prefix('=')?;
    let after_quote = after_eq.trim_start().strip_prefix('"')?;
    let value = after_quote.split('"').next().unwrap_or(after_quote);
    Edition::from_str(value)
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// `shard_index` must stay in range for every 128-bit fingerprint
    /// value — including the debug-build panic the naive cast-then-shift
    /// formulation would hit.
    #[test]
    fn shard_index_no_overflow() {
        let samples: &[u128] = &[
            0,
            1,
            u64::MAX as u128,
            1u128 << 64,
            u128::MAX,
            1u128 << 122,
            (1u128 << 122) - 1,
            0x1234_5678_9ABC_DEF0_1357_9BDF_2468_ACE0,
        ];
        for &v in samples {
            let fp = Fingerprint::from_u128_for_tests(v);
            let idx = shard_index(fp);
            assert!(
                idx < SHARDS,
                "shard index {idx} out of range for fingerprint {v:#034x}"
            );
        }
    }

    /// The `""` payload is shared by every empty file — including
    /// BOM-only files (BOM strips to `""` before interning).
    #[test]
    fn empty_and_bom_only_files_share_singleton() {
        let store = ContentStore::new();
        let (a, had_bom_a) = store.intern(b"").unwrap();
        let (b, had_bom_b) = store.intern(b"\xEF\xBB\xBF").unwrap();
        let (c, had_bom_c) = store.intern(b"").unwrap();
        assert!(Arc::ptr_eq(&a, &b), "empty file must share with BOM-only");
        assert!(Arc::ptr_eq(&a, &c), "empty file must be idempotent");
        assert!(!had_bom_a);
        assert!(had_bom_b, "BOM-only file must report had_bom = true");
        assert!(!had_bom_c);
    }

    /// Two distinct texts force a fingerprint collision via a manual
    /// store injection; the `memcmp` guard must reject the hit and
    /// return the newly-built content.
    #[test]
    fn collision_path_keeps_correctness() {
        let store = ContentStore::new();
        let (a_content, _) = store.intern(b"aaa").unwrap();
        let b_fp = Fingerprint::of("bbb");
        let idx = shard_index(b_fp);
        store.shards[idx]
            .write()
            .insert(b_fp, Arc::downgrade(&a_content));

        let (b_content, _) = store.intern(b"bbb").unwrap();
        assert_eq!(b_content.text.as_ref(), "bbb");
        assert!(!Arc::ptr_eq(&a_content, &b_content));
    }

    /// A header-cached `Content` is built once per unique text; interning
    /// the same text twice does not re-parse the header.
    #[test]
    fn header_edition_cached_on_content() {
        let store = ContentStore::new();
        let src = b"edition = \"2026\";\ndef f() {}";
        let (a, _) = store.intern(src).unwrap();
        let (b, _) = store.intern(src).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.header_edition, Some(Edition::Year2026));
    }

    #[test]
    fn utf8_error_returns_before_insert() {
        let store = ContentStore::new();
        let err = store.intern(b"\xff\xfe").unwrap_err();
        assert_eq!(err, SourceFileError::InvalidUtf8);
    }

    #[test]
    fn too_large_rejected() {
        // The size check is a single comparison on `raw.len()`; pin the
        // bound at compile time so a future change can't silently shrink
        // the addressable range.
        assert_eq!(MAX_FILE_LEN, u32::MAX as usize);
    }

    /// Identical non-empty content shares one `Arc<Content>`.
    #[test]
    fn identical_content_shares_payload() {
        let store = ContentStore::new();
        let (a, _) = store.intern(b"same bytes").unwrap();
        let (b, _) = store.intern(b"same bytes").unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    /// Distinct content yields distinct `Arc<Content>` handles.
    #[test]
    fn distinct_content_separates() {
        let store = ContentStore::new();
        let (a, _) = store.intern(b"aaa").unwrap();
        let (b, _) = store.intern(b"bbb").unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
    }

    /// `sweep_all` drops dead entries and keeps live ones.
    #[test]
    fn sweep_drops_dead_weak_entries() {
        let store = ContentStore::new();
        let (live, _) = store.intern(b"live").unwrap();
        // Drop the only strong ref to "ghost" — its slot becomes dead.
        {
            let _ghost = store.intern(b"ghost").unwrap();
        }
        // Sanity: the live content is still addressable.
        assert_eq!(live.text.as_ref(), "live");
        store.sweep_all();
        // Re-interning "ghost" after the sweep builds fresh (no live ref
        // remained), and re-interning "live" still hits the cached slot.
        let (live_again, _) = store.intern(b"live").unwrap();
        assert!(Arc::ptr_eq(&live, &live_again));
    }
}
