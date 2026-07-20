use crate::diagnostics::{Diagnostic, glyph::GlyphRenderer};

/// Colored terminal emitter — uses the custom `GlyphRenderer` for rich,
/// Rust-inspired box-drawing diagnostic output.
pub struct ColoredEmitter {
    renderer: GlyphRenderer,
}

impl ColoredEmitter {
    pub fn new() -> Self {
        ColoredEmitter {
            renderer: GlyphRenderer::new(true),
        }
    }
}

impl Default for ColoredEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl super::DiagnosticEmitter for ColoredEmitter {
    fn emit(&mut self, diag: &Diagnostic) {
        let rendered = self.renderer.render_diagnostic(diag);
        if !rendered.is_empty() {
            eprint!("{}", rendered);
        }

        // Call chain (Zig-style referenced by) — separate from the box
        if let Some(ref chain) = diag.call_chain {
            let rendered = chain.render(true);
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