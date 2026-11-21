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
    pub fn colored() -> Self {
        Self::with_opts(true, None)
    }

    /// ASCII-only output for CI or non-interactive terminals, with **default**
    /// layout options.  Same caveat as [`Self::colored`]: use [`Self::with_opts`]
    /// when a [`DiagOpts`] is available.
    #[inline]
    pub fn plain() -> Self {
        Self::with_opts(false, None)
    }

    /// Build a renderer whose `context_lines` and `max_width` come from
    /// `opts` (falling back to the [`DiagOpts`] defaults when `None`), so the
    /// CLI can drive the rendered width without the renderer owning a copy.
    #[inline]
    pub fn with_opts(use_color: bool, opts: Option<&DiagOpts>) -> Self {
        Self {
            glyph: GlyphRenderer::with_opts(use_color, opts),
        }
    }
}

impl Renderer for HumanRenderer {
    fn render_into(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>, out: &mut String) {
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
        let needs_index = diag.spans().first().is_some()
            || !diag.labels().is_empty()
            || diag.call_chain().is_some_and(|c| !c.is_empty());
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
        if let Some(chain) = diag.call_chain() {
            chain.render_into(
                &super::super::chain::ChainRenderCtx {
                    use_color: self.glyph.use_color(),
                    source: diag.source(),
                    file_name: diag.file_name(),
                    idx: idx.as_ref(),
                    max_width: self.glyph.max_width(),
                    tab_width: self.glyph.tab_width(),
                    cx: Some(cx),
                    // C6: anchors foreign-file span detection to the diagnostic's
                    // own file, so a chain entry from another file is rendered as
                    // its kind label instead of bogus coordinates.
                    anchor: diag.spans().first().map(|s| s.file),
                },
                out,
            );
        }
    }

    fn render(&self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String {
        let mut s = String::new();
        self.render_into(diag, cx, &mut s);
        s
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
