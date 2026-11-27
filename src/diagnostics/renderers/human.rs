//! Human-readable terminal renderer.
//!
//! Replaces the former `ColoredEmitter` / `PlainEmitter`, whose only
//! difference was the `use_color` flag passed to [`GlyphRenderer`].

use crate::diagnostics::glyph::GlyphRenderer;
use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::{DiagOpts, Diagnostic};

/// Renders diagnostics in the Rust-inspired box-drawing format, with or
/// without ANSI colour.
pub struct HumanRenderer {
    glyph: GlyphRenderer,
}

impl HumanRenderer {
    /// Colour output for interactive terminals, with **default** layout options.
    ///
    /// Convenience constructor for tests and for callers that genuinely have no
    /// [`DiagOpts`] (there is one, in `main.rs`'s colour-aware path, and it uses
    /// [`Self::with_opts`]).  It deliberately passes `None`, so a caller that
    /// *does* have CLI settings must use [`Self::with_opts`] — constructing this
    /// one after parsing `--max-width`/context options silently renders with the
    /// defaults instead.
    #[inline]
    #[must_use]
    pub fn colored() -> Self {
        Self::with_opts(true, None)
    }

    /// ASCII-only output for CI or non-interactive terminals, with **default**
    /// layout options.  Same caveat as [`Self::colored`]: use [`Self::with_opts`]
    /// when a [`DiagOpts`] is available.
    #[inline]
    #[must_use]
    pub fn plain() -> Self {
        Self::with_opts(false, None)
    }

    /// Build a renderer whose `context_lines` and `max_width` come from
    /// `opts` (falling back to the [`DiagOpts`] defaults when `None`), so the
    /// CLI can drive the rendered width without the renderer owning a copy.
    #[inline]
    #[must_use]
    pub fn with_opts(use_color: bool, opts: Option<&DiagOpts>) -> Self {
        Self {
            glyph: GlyphRenderer::with_opts(use_color, opts),
        }
    }
}

impl Renderer for HumanRenderer {
    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
        // Same empty-buffer contract as `json.rs`/`short.rs`: a stale non-empty
        // buffer splices two boxes together, and nothing else asserts it.
        debug_assert!(
            out.is_empty(),
            "human renderer expects an empty buffer, got {} bytes",
            out.len()
        );
        // Build the source's line index at most once per batch and share it
        // between the diagnostic body and its call-chain footer (both used to
        // rescan the file independently).
        //
        // Only build it when something consumes it: a source-attached diagnostic
        // with no spans, no labels and an empty call chain (summary-level errors
        // do occur) would otherwise pay an O(file) scan on an LRU miss.
        //
        // This deliberately does NOT mirror the JSON renderer's `needs_index`,
        // which is `anchor_file.is_some()` only.  JSON narrowed to the anchor
        // because none of its other consumers (labels, suggestions, children)
        // ever read line/column, so the extra disjuncts only forced an O(n) scan
        // for data it discards.  The glyph renderer *does* draw label markers
        // and the call-chain footer against the source excerpt, so it needs the
        // index for a span-less diagnostic that carries labels or a chain —
        // dropping those disjuncts here would silently render such a diagnostic
        // without line numbers.  Keep the policies divergent on purpose: this
        // disjunction IS the pin, so narrowing it to match JSON is a rendering
        // regression, not a cleanup.
        //
        // `related_errors` is here for the same reason, and it was the missing
        // disjunct: glyph turns every related error that has a span into a
        // synthetic label drawn against the excerpt, so a diagnostic carrying
        // only related errors would otherwise render unplaceable.  Children and
        // suggestions are message-and-diff only — no coordinates, so they stay
        // out.
        //
        // Recorded, not adopted: a review proposed narrowing this disjunct to related spans in
        // the *anchor* file (`Some(s.file) == diag.spans.first().map(|p| p.file)`) to skip the
        // O(file) index for diagnostics whose related errors are all foreign.  Not done, for the
        // reason the review itself gave — it defers to glyph internals ("verify first"), and
        // this comment records glyph's contract as "every related error with a span becomes a
        // synthetic label", with no anchor-file qualifier.  Trigger for reopening: a glyph test
        // proving a foreign-file related error still renders (it resolves against its own
        // excerpt) — then the narrowing is safe and the index build is skipped.
        // One `call_chain()` walk per diagnostic: the `needs_index` gate and the footer
        // below both need it, and every other accessor in this file is bound once for the
        // same reason ("the renderer's contract is one accessor call each").
        let call_chain = diag.call_chain();
        let needs_index = diag.spans.primary_ref().first().is_some()
            || !diag.labels().is_empty()
            || call_chain.is_some_and(|c| !c.is_empty())
            || diag.related_errors.iter().any(|r| r.span.is_some());
        let idx = if needs_index {
            diag.source().map(|s| cx.line_index(s))
        } else {
            None
        };
        // Both halves append into one buffer: the glyph body first, then the
        // Zig-style `referenced by` footer.  `render_into` is a no-op for an
        // empty chain, so no emptiness check is needed here.
        self.glyph
            .render_diagnostic_into(diag, idx.as_ref(), Some(cx), out);
        if let Some(chain) = call_chain {
            chain.render_into(
                &super::super::chain::ChainRenderCtx {
                    use_color: self.glyph.use_color(),
                    source: diag.source(),
                    file_name: diag.file_name(),
                    idx: idx.as_ref(),
                    max_width: self.glyph.max_width(),
                    tab_width: self.glyph.tab_width(),
                    cx: Some(cx),
                    // Anchors foreign-file span detection to the diagnostic's
                    // own file, so a chain entry from another file is rendered as
                    // its kind label instead of bogus coordinates.
                    anchor: diag.spans.primary_ref().first().map(|s| s.file),
                },
                out,
            );
        }
    }

    /// A blank line between successive diagnostics keeps the boxes
    /// visually distinct.
    fn render_separator(&self) -> &str {
        "\n"
    }

    fn render_summary(&self, errors: usize, warnings: usize) -> String {
        let mut s = self.glyph.render_summary(errors, warnings);
        if !s.is_empty() {
            s.push('\n');
        }
        s
    }

    fn render_summary_full(&self, errors: usize, warnings: usize, files: usize) -> String {
        let mut s = self.glyph.render_summary_full(errors, warnings, files);
        if !s.is_empty() {
            s.push('\n');
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A related error pointing into a *different* file must be drawn against **that
    /// file**, never against the anchor excerpt.
    ///
    /// The audit behind this test expected to find the foreign span simply dropped; it is
    /// stronger than that.  `glyph.rs` gates every coordinate consumer on
    /// `span.file == anchor_file` (labels, additional spans, related-error synthetic
    /// labels, underlines), so a foreign span gets no marker on the anchor excerpt — and
    /// the human renderer then renders it as its own excerpt resolved through *its own*
    /// source.  Both halves matter and both used to be unpinned here: without the gate a
    /// foreign offset is mapped through the anchor buffer and silently prints a wrong
    /// line/column on the wrong text (`json.rs` documents the same hazard for its
    /// coordinates).
    ///
    /// The foreign file is deliberately shaped so a leak is visible: its span sits on a
    /// line number that does not exist in the anchor file.
    #[test]
    fn a_foreign_related_span_is_drawn_against_its_own_file_only() {
        use crate::ast::{FileId, Span};
        use crate::diagnostics::RelatedError;
        use crate::diagnostics::render::{RenderCtx, Renderer as _};

        // Two lines only in the anchor; the foreign span is on line 3 of its own file, so
        // "line 3" can only come from the foreign source.
        let anchor_src = "fn main() {\n    bad();\n}\n";
        let foreign_src = "fn other() {\n    unrelated();\n    very_long_line_here();\n}\n";
        let anchor = FileId::from_raw(1);
        let other = FileId::from_raw(2);
        let foreign_span = Span::new(other, 34, 55);

        let diag = Diagnostic::error("boom")
            .with_span(Span::new(anchor, 20, 23))
            .with_source(anchor_src)
            .with_file_name("anchor.pn")
            .with_foreign_source(other, "other.pn", foreign_src)
            .with_related(RelatedError {
                code: None,
                message: "see also".into(),
                span: Some(foreign_span),
                label: None,
            });

        let opts = crate::diagnostics::DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);
        let mut out = String::new();
        HumanRenderer::with_opts(false, None).render_into(&diag, &cx, &mut out);

        assert_eq!(
            out.matches("anchor.pn:").count(),
            1,
            "only the anchor body may be attributed to the anchor file: {out}"
        );
        assert!(
            out.contains("other.pn:3:"),
            "the foreign span must report its own line 3: {out}"
        );
        assert!(
            out.contains("very_long_line_here"),
            "the foreign excerpt must come from the foreign source: {out}"
        );
        assert!(
            !out.contains("anchor.pn:3:"),
            "a foreign offset must never be mapped through the anchor buffer: {out}"
        );
    }
}
