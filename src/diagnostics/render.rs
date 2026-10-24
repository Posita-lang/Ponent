//! Pure diagnostic formatting.
//!
//! A [`Renderer`] knows the *output format* and nothing about I/O: it returns
//! `String`s and lets the emitter decide where they go.  Document-level
//! framing (HTML `<head>`, JSON array brackets) lives in `begin`/`end` so the
//! emitter stays format-agnostic.

use crate::ast::FileId;
use crate::diagnostics::sink::DiagnosticSink;
use crate::diagnostics::{DiagOpts, Diagnostic, LineIndex};
use crate::smallvec::SmallVec;
use std::cell::RefCell;
use std::io;
use std::sync::Arc;

/// Maximum number of distinct source files whose line tables are cached
/// across one batch.  A batch typically draws from at most a handful of
/// files; 4 covers the common "primary + a few includes" shape and keeps
/// two-file alternation from thrashing.
const CACHE_SLOTS: usize = 4;

/// One cached line index, keyed by source identity `(ptr, len)`.
struct CachedSourceIndex {
    ptr: usize,
    len: usize,
    starts: Arc<[usize]>,
}

/// Context shared by every render call in a batch.
///
/// Carries configuration and (in the future) source/URI maps and a serial
/// number.  Borrowed, so constructing one per batch is free.
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
}

impl<'a> RenderCtx<'a> {
    /// Build a batch context.  One per batch: the per-source line index is
    /// reused across every `render` call made through this ctx.
    pub fn new(opts: &'a DiagOpts, uri: Option<&'a str>) -> Self {
        Self {
            opts,
            uri,
            source_index: RefCell::new(SmallVec::new()),
        }
    }

    /// The line index for `source`, built at most once per source per batch.
    ///
    /// The returned index borrows only `source`, so callers can hold it for
    /// the duration of a render without borrowing the ctx.
    pub(crate) fn line_index<'s>(&self, source: &'s str) -> LineIndex<'s> {
        let ptr = source.as_ptr() as usize;
        let len = source.len();
        {
            let mut cache = self.source_index.borrow_mut();
            if let Some(pos) = cache.iter().position(|c| c.ptr == ptr && c.len == len) {
                // LRU: promote the hit to MRU so it survives eviction longest.
                // N ≤ 4, so the shifts are a handful of pointer moves.
                if pos > 0 {
                    let entry = cache.remove(pos);
                    cache.insert(0, entry);
                }
                return LineIndex::from_shared(source, Arc::clone(&cache[0].starts));
            }
        }
        let idx = LineIndex::new(source);
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
                starts: Arc::clone(idx.starts_arc()),
            },
        );
        idx
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

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
        let sources: Vec<String> = (0..=CACHE_SLOTS)
            .map(|i| "a\n".repeat(i + 1))
            .collect();

        let first: Vec<LineIndex<'_>> = sources[..CACHE_SLOTS]
            .iter()
            .map(|s| cx.line_index(s))
            .collect();

        // Insert one more — `sources[0]` is the LRU and must be evicted.
        let extra = cx.line_index(&sources[CACHE_SLOTS]);
        assert!(
            !first.iter().any(|f| Arc::ptr_eq(f.starts_arc(), extra.starts_arc())),
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
pub trait Renderer: Send {
    /// Called once per batch, before any diagnostics.
    fn begin(&mut self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        let _ = out;
        Ok(())
    }

    /// Render one diagnostic.  Pure: no I/O.
    fn render(&mut self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String;

    /// Text written between two adjacent rendered diagnostics in a batch.
    ///
    /// The human renderer inserts a blank line so successive diagnostics do
    /// not run together; machine-consumed formats (JSON, HTML) leave the
    /// default empty separator so the document stays well-formed.
    fn render_separator(&mut self) -> &str {
        ""
    }

    /// Render the summary line.  Empty string means "no summary".
    fn render_summary(&mut self, errors: usize, warnings: usize) -> String;

    /// Render the summary line with a distinct-file count for the
    /// "N errors in M files" overview (checklist item 8).
    ///
    /// Default impl ignores `files` and delegates to [`render_summary`](Self::render_summary)
    /// so existing renderers keep working; human/HTML overrides include
    /// the file count while JSON keeps the empty summary (machine-consumed).
    fn render_summary_full(&mut self, errors: usize, warnings: usize, files: usize) -> String {
        let _ = files;
        self.render_summary(errors, warnings)
    }

    /// Render the "N errors suppressed" trailer when `max_errors` cut the
    /// batch short.  Empty string means "emit nothing".  Formats that are
    /// machine-consumed (JSON, LSP) override this to `""` so the trailer
    /// cannot corrupt the document.
    fn render_suppressed_note(&mut self, count: usize) -> String {
        format!(
            "  = note: {count} additional error{} suppressed\n",
            if count == 1 { "" } else { "s" },
        )
    }

    /// Called once per batch, after all diagnostics and the summary.
    fn end(&mut self, out: &mut impl DiagnosticSink) -> io::Result<()> {
        let _ = out;
        Ok(())
    }
}
