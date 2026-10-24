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
    use_color: bool,
}

impl HumanRenderer {
    /// Colour output for interactive terminals.
    #[inline]
    pub fn colored() -> Self {
        Self::with_opts(true, None)
    }

    /// ASCII-only output for CI or non-interactive terminals.
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
            use_color,
        }
    }
}

impl Renderer for HumanRenderer {
    fn render(&mut self, diag: &Diagnostic<'_>, cx: &RenderCtx<'_>) -> String {
        // Build the source's line index at most once per batch and share it
        // between the diagnostic body and its call-chain footer (both used to
        // rescan the file independently).
        let idx = diag.source().map(|s| cx.line_index(s));
        let mut s = self.glyph.render_diagnostic_with_index(diag, idx.as_ref());
        // Call chain (Zig-style `referenced by`) — separate from the box.
        if let Some(chain) = diag.call_chain() {
            let r = chain.render_with_index(
                self.use_color,
                diag.source(),
                diag.file_name(),
                idx.as_ref(),
                self.glyph.max_width(),
                self.glyph.tab_width(),
            );
            if !r.is_empty() {
                s.push_str(&r);
            }
        }
        s
    }

    /// A blank line between successive diagnostics keeps the boxes
    /// visually distinct.
    fn render_separator(&mut self) -> &str {
        "\n"
    }

    fn render_summary(&mut self, errors: usize, warnings: usize) -> String {
        let mut s = self.glyph.render_summary(errors, warnings);
        if !s.is_empty() {
            s.push('\n');
        }
        s
    }

    fn render_summary_full(&mut self, errors: usize, warnings: usize, files: usize) -> String {
        let mut s = self.glyph.render_summary_full(errors, warnings, files);
        if !s.is_empty() {
            s.push('\n');
        }
        s
    }
}
