//! The emitter: the only layer that performs I/O.
//!
//! An [`Emitter`] pairs a pure [`Renderer`] with a [`MakeSink`] factory.  It
//! owns the batch loop — `begin`, one `render` per diagnostic, the summary,
//! `end`, flush — and is the sole place that touches a sink.  Renderers never
//! see an `io::Write`; sinks never see a `Diagnostic`.

use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::sink::{DiagnosticSink, MakeSink};
use crate::diagnostics::{DiagOpts, Diagnostic};
use smallvec::SmallVec;
use std::io;

pub mod stylesheet;

/// Composes a renderer with a sink factory and emits batches of diagnostics.
pub struct Emitter<R, M> {
    renderer: R,
    make_sink: M,
    opts: DiagOpts,
}

impl<R: Renderer, M: MakeSink> Emitter<R, M> {
    pub fn new(renderer: R, make_sink: M, opts: DiagOpts) -> Self {
        Self {
            renderer,
            make_sink,
            opts,
        }
    }

    /// Emit a single diagnostic through a fresh sink.
    ///
    /// This is the render boundary: `humanize()` materializes the
    /// structured `kind` (message/labels/help/suggestions) against the
    /// *current* i18n `Lang` right before rendering, so a diagnostic's
    /// language is never frozen at construction time.
    pub fn emit(&mut self, diag: &mut Diagnostic<'_>) -> io::Result<()> {
        diag.humanize();
        let mut sink = self.make_sink.make();
        self.renderer.begin(&mut sink)?;
        let cx = RenderCtx::new(&self.opts, None);
        sink.write_str(&self.renderer.render(diag, &cx))?;
        self.renderer.end(&mut sink)?;
        sink.flush()
    }

    /// Emit a batch of diagnostics plus a summary through a fresh sink.
    ///
    /// Honours [`DiagOpts::reverse_errors`] exactly like [`Self::emit_all_iter`]:
    /// the iterator and its reverse are distinct concrete types, so each branch
    /// calls the batch loop directly rather than unifying them.
    pub fn emit_all(&mut self, diags: &mut [Diagnostic<'_>]) -> io::Result<()> {
        let mut sink = self.make_sink.make();
        let iter = diags.iter_mut();
        if self.opts.reverse_errors {
            self.emit_batch_iter(iter.rev(), &mut sink)
        } else {
            self.emit_batch_iter(iter, &mut sink)
        }
    }

    /// Emit a batch from any double-ended iterator of borrowed diagnostics —
    /// avoids cloning a `DiagCtxt`'s contents into a `Vec` just to pass a
    /// slice to [`Self::emit_all`].
    pub fn emit_all_iter<'r, 'src, I>(&mut self, diags: I) -> io::Result<()>
    where
        'src: 'r,
        I: IntoIterator<Item = &'r mut Diagnostic<'src>>,
        I::IntoIter: DoubleEndedIterator,
    {
        let mut sink = self.make_sink.make();
        let iter = diags.into_iter();
        // `iter` and `iter.rev()` are different concrete iterator types, so
        // each branch calls the batch loop directly rather than unifying them.
        if self.opts.reverse_errors {
            self.emit_batch_iter(iter.rev(), &mut sink)
        } else {
            self.emit_batch_iter(iter, &mut sink)
        }
    }

    /// Escape hatch: write to a caller-provided sink without a `MakeSink`.
    ///
    /// Honours [`DiagOpts::reverse_errors`] exactly like [`Self::emit_all_iter`]:
    /// the iterator and its reverse are distinct concrete types, so each branch
    /// calls the batch loop directly rather than unifying them.
    pub fn emit_all_to<S: DiagnosticSink>(
        &mut self,
        diags: &mut [Diagnostic<'_>],
        sink: &mut S,
    ) -> io::Result<()> {
        let iter = diags.iter_mut();
        if self.opts.reverse_errors {
            self.emit_batch_iter(iter.rev(), sink)
        } else {
            self.emit_batch_iter(iter, sink)
        }
    }

    fn emit_batch_iter<'r, 'src, S, I>(&mut self, iter: I, sink: &mut S) -> io::Result<()>
    where
        S: DiagnosticSink,
        'src: 'r,
        I: Iterator<Item = &'r mut Diagnostic<'src>>,
    {
        self.renderer.begin(sink)?;

        // Count *every* diagnostic up front — including those `max_errors`
        // drops — so the summary reflects the true total.  This is a
        // deliberate divergence from rustc, whose summary is bounded by
        // `--error-limit`; the suppressed count is reported separately via
        // `render_suppressed_note`.
        let mut errors = 0usize;
        let mut warnings = 0usize;
        let mut suppressed = 0usize;
        let mut emitted_errors = 0usize;
        let mut first = true;
        // Owned names: the batch loop now humanizes (`&mut`) each
        // diagnostic mid-iteration, so the file list cannot hold `&str`
        // views borrowed from the diagnostics themselves.
        let mut files: SmallVec<[String; 4]> = SmallVec::new();
        let cx = RenderCtx::new(&self.opts, None);

        for diag in iter {
            let name = diag.file_name();
            if !files.iter().any(|f| f == name) {
                files.push(name.to_string());
            }
            if diag.is_error() {
                errors += 1;
                if let Some(limit) = self.opts.max_errors
                    && emitted_errors >= limit
                {
                    suppressed += 1;
                    continue;
                }
                emitted_errors += 1;
            } else if diag.is_warning() && self.opts.warn_is_error {
                // `warn_is_error` promotes warnings into the error budget:
                // they count toward the summary and the `max_errors` gate
                // exactly like real errors, but are still rendered at their
                // original level.
                errors += 1;
                if let Some(limit) = self.opts.max_errors
                    && emitted_errors >= limit
                {
                    suppressed += 1;
                    continue;
                }
                emitted_errors += 1;
            }
            if diag.is_warning() {
                warnings += 1;
            }
            // Render boundary: materialize `kind` only for diagnostics we
            // are actually about to render.  Suppressed entries have
            // already `continue`d above, so a filter-skipped diagnostic
            // costs no Fluent lookup and no derived strings.  `humanize`
            // is idempotent, so an already-materialized diagnostic is a
            // free no-op here.
            diag.humanize();
            // A non-empty separator (the human renderer's blank line) keeps
            // successive diagnostics visually distinct; machine formats
            // return the default empty string, so this is a no-op there.
            if first {
                first = false;
            } else {
                let sep = self.renderer.render_separator();
                if !sep.is_empty() {
                    sink.write_str(sep)?;
                }
            }
            sink.write_str(&self.renderer.render(diag, &cx))?;
        }

        if suppressed > 0 {
            sink.write_str(&self.renderer.render_suppressed_note(suppressed))?;
        }
        if errors > 0 || warnings > 0 {
            sink.write_str(&self.renderer.render_summary_full(errors, warnings, files.len()))?;
        }

        self.renderer.end(sink)?;
        sink.flush()
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::{FileId, Span};
    use crate::diagnostics::sink::{string, StringSink};
    use crate::diagnostics::JsonRenderer;

    fn warn_diag() -> Diagnostic<'static> {
        Diagnostic::warning("boom")
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_source("x")
    }

    fn emit(opts: DiagOpts) -> String {
        let mut diags = [warn_diag(), warn_diag()];
        let mut sink = StringSink::new();
        Emitter::new(JsonRenderer::new(false), string(), opts)
            .emit_all_to(&mut diags, &mut sink)
            .unwrap();
        sink.into_string()
    }

    /// `warn_is_error` promotes warnings into the error budget: the second
    /// warning is dropped by `max_errors` even though it renders at its
    /// original level.  Without the flag the budget is untouched.
    #[test]
    fn warnings_consume_the_error_budget_only_when_promoted() {
        let promoted = DiagOpts {
            warn_is_error: true,
            max_errors: Some(1),
            ..DiagOpts::default()
        };
        let out = emit(promoted);
        assert_eq!(
            out.matches("\"boom\"").count(),
            1,
            "promoted warning must be cut by max_errors:\n{out}"
        );

        let plain = DiagOpts {
            warn_is_error: false,
            max_errors: Some(1),
            ..DiagOpts::default()
        };
        let out = emit(plain);
        assert_eq!(
            out.matches("\"boom\"").count(),
            2,
            "plain warnings must not consume the error budget:\n{out}"
        );
    }
}
