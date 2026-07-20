use crate::diagnostics::{Diagnostic, glyph::GlyphRenderer};

/// Plain text emitter — uses the custom `GlyphRenderer` with ASCII-only box-drawing
/// for non-interactive terminals or CI environments.
pub struct PlainEmitter {
    renderer: GlyphRenderer,
}

impl PlainEmitter {
    pub fn new() -> Self {
        PlainEmitter {
            renderer: GlyphRenderer::new(false),
        }
    }
}

impl Default for PlainEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl super::DiagnosticEmitter for PlainEmitter {
    fn emit(&mut self, diag: &Diagnostic) {
        let rendered = self.renderer.render_diagnostic(diag);
        if !rendered.is_empty() {
            eprint!("{}", rendered);
        }

        // Call chain (Zig-style referenced by) — separate from the box
        if let Some(ref chain) = diag.call_chain {
            let rendered = chain.render(false);
            if !rendered.is_empty() {
                eprintln!("{}", rendered);
            }
        }
    }

    fn emit_summary(&mut self, error_count: usize, warning_count: usize) {
        let rendered = self.renderer.render_summary(error_count, warning_count);
        if !rendered.is_empty() {
            eprintln!("{}", rendered);
        }
    }
}
