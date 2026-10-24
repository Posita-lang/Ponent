//! Custom source-code renderer for diagnostics.
//!
//! Produces Rust-inspired output with `┌─└─│ ^^^` box-drawing,
//! replacing the previous miette-based renderer.
//!
//! NOTE on layout architecture (decided: keep hand-rolled column
//! arithmetic, do NOT introduce a Wadler-style `Doc` algebra or an
//! external snippet crate): the layout here is a flat, row-shaped
//! grammar — header line, source rows, underline rows, gutter footer —
//! where every row's width is derived from ONE value (`display_width` /
//! `line_num_width`) already computed on the shared `LineIndex`.  A `Doc`
//! combinator layer (vcat/beside/indent/flat-breaks) pays off when
//! nesting or alignment groups multiply; here there is exactly one
//! alignment group per row and the box-drawing prefix is fixed, so the
//! algebra would add an IR + a second rendering pass to translate the
//! same three column values.  Revisit only if a format needs genuinely
//! nested layout (e.g. folded multi-level source trees).
//!
//! Format (the renderer is charset-configurable via
//! [`DiagOpts::use_unicode`](crate::diagnostics::DiagOpts::use_unicode);
//! this is the Unicode charset):
//! ```text
//! ┌─ error[E003]: unexpected identifier `dem` at top level
//! │  ┌─ <input>:1:1
//! │  │
//! │1 │ dem main(){}
//! │  │ ^^^
//! │  │ ^^^
//! │  │ |
//! │  │ | only declarations are allowed at the top level.
//! │  └─
//! │  = note: `dem` is not a keyword. Did you mean `def`?
//! ```
//!
//! `render_diagnostic` emits the frame above; the emitter appends the
//! one-line summary (`└─ aborting due to previous error`) below it when it
//! reports one.

use crate::ast::FileId;
use crate::ast::Span;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::Applicability;
use crate::diagnostics::DisplayArg;
use crate::diagnostics::LineIndex;
use crate::diagnostics::Suggestion;
use crate::diagnostics::label::{AnnotationKind, Label};
use crate::diagnostics::level::DiagnosticLevel;
use crate::smallvec::SmallVec;
use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Write;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

// ── Box-drawing characters ──────────────────────────────────────

struct BoxChars {
    tl: &'static str,     // top-left corner
    bl: &'static str,     // bottom-left corner
    h: &'static str,      // horizontal line
    v: &'static str,      // vertical line
    sub_tl: &'static str, // sub-box top-left
    sub_bl: &'static str, // sub-box bottom-left
    sub_v: &'static str,  // sub-box vertical
    sub_h: &'static str,  // sub-box horizontal
}

const UNICODE: BoxChars = BoxChars {
    tl: "┌",
    bl: "└",
    h: "─",
    v: "│",
    sub_tl: "┌",
    sub_bl: "└",
    sub_v: "│",
    sub_h: "─",
};

const ASCII: BoxChars = BoxChars {
    tl: "+",
    bl: "\\",
    h: "-",
    v: "|",
    sub_tl: "+",
    sub_bl: "\\",
    sub_v: "|",
    sub_h: "-",
};

// ── Style system (analogous to rustc's anstyle) ───────────────────

/// Semantic style tokens for diagnostic output.
/// Each variant maps to an ANSI style; color is resolved through `Styles`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Error,
    Fatal,
    Warning,
    Note,
    Help,
    Bold,
    Dim,
    BrightRed,
    BrightBlue,
    Blue,
    Green,
    Yellow,
}

/// Resolves `Style` tokens to ANSI escape codes.
/// When `use_color` is false, all styles resolve to the empty string.
struct Styles {
    use_color: bool,
    reset: &'static str,
    bold: &'static str,
    dim: &'static str,
    error: &'static str, // bold red (Error / BrightRed)
    fatal: &'static str, // bold red-bg, white text (Fatal)
    help: &'static str,  // bright blue (Help / BrightBlue)
    blue: &'static str,
    note: &'static str,    // bright green (Note / Green)
    warning: &'static str, // bright yellow (Warning / Yellow)
    /// Diff-removal colour for structural suggestion `-` lines.
    diff_del: &'static str, // bright red (`\x1b[91m`)
    /// Diff-addition colour for structural suggestion `+` lines.
    diff_add: &'static str, // bright green (`\x1b[92m`)
    /// Background colour for source-highlight spans.  256-colour terminals
    /// get `\x1b[48;5;236m` (dark gray); 16-colour terminals get `\x1b[7m`
    /// (reverse — universally available, visually distinct).
    highlight_bg: &'static str,
}

impl Styles {
    const fn new(use_color: bool, color_depth: super::term::ColorDepth) -> Self {
        if use_color {
            let highlight_bg = match color_depth {
                super::term::ColorDepth::TwoFiftySix | super::term::ColorDepth::TrueColor => {
                    "\x1b[48;5;236m"
                }
                super::term::ColorDepth::Sixteen => "\x1b[7m",
            };
            Styles {
                use_color: true,
                reset: "\x1b[0m",
                bold: "\x1b[1m",
                dim: "\x1b[2m",
                error: "\x1b[1;91m",
                fatal: "\x1b[1;41;97m",
                help: "\x1b[94m",
                blue: "\x1b[34m",
                note: "\x1b[92m",
                warning: "\x1b[93m",
                diff_del: "\x1b[91m",
                diff_add: "\x1b[92m",
                highlight_bg,
            }
        } else {
            Styles {
                use_color: false,
                reset: "",
                bold: "",
                dim: "",
                error: "",
                fatal: "",
                help: "",
                blue: "",
                note: "",
                warning: "",
                diff_del: "",
                diff_add: "",
                highlight_bg: "",
            }
        }
    }

    fn get(&self, style: Style) -> &'static str {
        if !self.use_color {
            return "";
        }
        match style {
            Style::Error => self.error,
            Style::Fatal => self.fatal,
            Style::Warning => self.warning,
            Style::Note => self.note,
            Style::Help => self.help,
            Style::Bold => self.bold,
            Style::Dim => self.dim,
            Style::BrightRed => self.error,
            Style::BrightBlue => self.help,
            Style::Blue => self.blue,
            Style::Green => self.note,
            Style::Yellow => self.warning,
        }
    }
}

// ── Box-drawing characters ──────────────────────────────────────

// ── GlyphRenderer ───────────────────────────────────────────────

/// Renders diagnostics in the Rust-inspired box-drawing format.
pub struct GlyphRenderer {
    use_color: bool,
    context_lines: usize,
    max_width: usize,
    bc: &'static BoxChars,
    s: Styles,
    /// Visible width of the `│  = ` prefix plus the trailing `: ` — i.e.
    /// everything in the `│  = <label>: ` child/note/help gutter except the
    /// label text itself. Measured once at construction so the per-call
    /// width does not re-`format!` + re-scan a constant string.
    prefix_base: usize,
    /// Compact layout mode (see [`DiagOpts::compact`]): skips the sub-box
    /// spacing and close rows and drops the sub-box header's `┌─ ` prefix.
    compact: bool,
    /// Tab stop width for display-column calculations (tabs expand to the
    /// next `tab_width`-column stop).  From [`DiagOpts::tab_width`];
    /// defaults to [`DEFAULT_TAB_WIDTH`].
    tab_width: usize,
}

impl GlyphRenderer {
    /// Create a new renderer with the given options.
    /// When `opts` is provided, its `context_lines`, `max_width` and
    /// `use_color` are used; otherwise defaults to `use_color` with the
    /// `DiagOpts` default context lines and width.
    /// Minimum supported render width.  [`Self::with_opts`] clamps
    /// narrower configurations up to this (a user CLI flag or a
    /// mis-detecting terminal must not panic the renderer); `wrap_msg`
    /// keeps a `.max(1)` floor as defence-in-depth against pathological
    /// indents.
    pub(crate) const MIN_WIDTH: usize = 20;

    pub fn new(use_color: bool) -> Self {
        Self::with_opts(use_color, None)
    }

    /// Create a new renderer that reads `context_lines`, `max_width`,
    /// `use_unicode`, and `compact` from the given
    /// [`DiagOpts`](crate::diagnostics::DiagOpts), falling back to the
    /// `DiagOpts` defaults when `opts` is `None`.
    ///
    /// With `opts` supplied, the box-drawing charset follows
    /// [`DiagOpts::use_unicode`](crate::diagnostics::DiagOpts::use_unicode);
    /// a bare `use_color` renderer (no options) keeps the historical
    /// coupling where Unicode implies colour, so existing callers that
    /// only care about colour keep their exact output.
    pub fn with_opts(use_color: bool, opts: Option<&crate::diagnostics::DiagOpts>) -> Self {
        let fallback = crate::diagnostics::DiagOpts::default();
        let (context_lines, max_width, compact) = match opts {
            Some(o) => (o.context_lines, o.max_width, o.compact),
            None => (fallback.context_lines, fallback.max_width, fallback.compact),
        };
        // Clamp, don't panic: `max_width` can come from a user CLI flag
        // (`--width 15`) or a mis-detecting terminal, and a diagnostics
        // renderer must never abort the compiler over its own width.
        let max_width = max_width.max(Self::MIN_WIDTH);
        // Charset: options carry an explicit `use_unicode`; a bare
        // `use_color` renderer has no charset knob and keeps the legacy
        // coupling (colour-capable ⇒ Unicode).
        let use_unicode = match opts {
            Some(o) => o.use_unicode,
            None => use_color,
        };
        let bc: &'static BoxChars = if use_unicode { &UNICODE } else { &ASCII };
        // `│  = ` + `: ` — the box char is one cell in both charsets; only
        // the label between them varies, and it is measured per level below.
        let prefix_base = visible_width(bc.v) + "  = ".len() + ": ".len();
        let color_depth = match opts {
            Some(o) => o.color_depth,
            None => crate::diagnostics::term::ColorDepth::TwoFiftySix,
        };
        let tab_width = match opts {
            Some(o) => o.tab_width,
            None => DEFAULT_TAB_WIDTH,
        };
        GlyphRenderer {
            use_color,
            context_lines,
            max_width,
            bc,
            s: Styles::new(use_color, color_depth),
            prefix_base,
            compact,
            tab_width,
        }
    }

    /// Visible width of the `│  = <label>: ` gutter that precedes child /
    /// note / help messages, for a given level. The box char and fixed
    /// separators are folded into `prefix_base`; only the label length varies.
    fn prefix_cols(&self, level: DiagnosticLevel) -> usize {
        self.prefix_base + level.label().len()
    }

    pub fn with_context_lines(mut self, n: usize) -> Self {
        self.context_lines = n;
        self
    }

    /// Configured maximum render width (after clamping to `MIN_WIDTH`).
    #[inline]
    pub(crate) fn max_width(&self) -> usize {
        self.max_width
    }

    /// Configured tab stop width.
    #[inline]
    pub(crate) fn tab_width(&self) -> usize {
        self.tab_width
    }

    /// Word-wrap `msg` to `self.max_width` display columns measured from the
    /// *start* of the rendered line (`prefix_cols` = visible width of
    /// everything already on the first line, e.g. `│ = note: `).
    ///
    /// Returns the segments to emit, one per output line; the caller
    /// re-prints the prefix on each continuation.  Styling is applied by
    /// the caller *per segment*, so every wrapped line gets a full
    /// open/reset pair and no SGR run is ever cut in half.
    fn wrap_msg<'a>(&self, msg: &'a str, prefix_cols: usize) -> Vec<Cow<'a, str>> {
        // `max(1)` only guarantees forward progress for absurdly deep
        // indents.  Narrow *configurations* are clamped at construction
        // (`with_opts` raises `max_width` to `MIN_WIDTH`), so the old
        // `max(20)` floor — which silently made `max_width < 20` a lie —
        // is gone.
        let avail = self.max_width.saturating_sub(prefix_cols).max(1);
        super::wrap::wrap_optimal(msg, avail, self.tab_width)
    }

    /// Map a diagnostic level to its semantic [`Style`] token so colours are
    /// resolved through [`Styles`] (and therefore disabled when `use_color`
    /// is false).
    fn level_style(level: DiagnosticLevel) -> Style {
        match level {
            DiagnosticLevel::Fatal => Style::Fatal,
            DiagnosticLevel::Error => Style::Error,
            DiagnosticLevel::Warning => Style::Warning,
            DiagnosticLevel::Help => Style::Help,
            DiagnosticLevel::Note | DiagnosticLevel::Info => Style::Note,
        }
    }

    // ── Public API ──────────────────────────────────────────────

    /// Render a single diagnostic to its full formatted string.
    /// Returns an empty string if the diagnostic has no source text.
    ///
    /// Builds its own [`LineIndex`]; batch callers that already have one for
    /// this source should use [`Self::render_diagnostic_with_index`] to avoid
    /// rescanning the file per diagnostic.
    pub fn render_diagnostic(&self, diag: &Diagnostic<'_>) -> String {
        self.render_diagnostic_with_index(diag, None)
    }

    /// Like [`Self::render_diagnostic`], but reuses a caller-provided line
    /// index for `diag`'s source (the emitter's per-batch cache) instead of
    /// rebuilding it.
    pub(crate) fn render_diagnostic_with_index<'src>(
        &self,
        diag: &Diagnostic<'src>,
        idx: Option<&LineIndex<'_>>,
    ) -> String {
        // Rendered output is dominated by the context windows (a handful of
        // source rows), not the whole file, so derive a hint from the source
        // size but cap it — a 1 MB file must not reserve tens of KB of buffer.
        let hint = diag
            .source
            .as_ref()
            .map_or(256, |s| (s.len() / 8).clamp(256, 8 * 1024));
        let mut out = String::with_capacity(hint);

        // Per-suggestion "rendered as a diff block" flags, written back by
        // `write_source_section` so the fallback loop below can skip them.
        let mut suggestion_rendered = vec![false; diag.suggestions.len()];

        // ┌─ error[E003]: message
        self.write_header(&mut out, diag);

        if let Some(ref source) = diag.source {
            // Borrowed view of the primary labels plus a small local stash for
            // the synthetic ones built from related errors.  The old code
            // cloned every `Label` into a `Vec<Label>`, which allocates a
            // `String` for each `Cow::Owned` message once per rendered
            // diagnostic; the renderer only reads them, so references suffice.
            let mut synthetic: SmallVec<[Label<'src>; 2]> = SmallVec::new();
            for rel in &diag.related_errors {
                if let Some(rel_span) = rel.span {
                    let label_text = match rel.code.as_ref() {
                        Some(code) => format!("{} [{}]", rel.message, code.code()).into(),
                        None => rel.message.clone(),
                    };
                    synthetic.push(Label::secondary(rel_span, label_text));
                }
            }
            // Synthetic `^` annotations for merged groups: when two or more
            // primary spans share one section, each real span keeps its own
            // caret (the earliest acts as the section primary), so
            // intermediate lines between them get no union caret.
            // Declared before `merged_labels` so the borrows nest cleanly.
            let mut extras_labels: Vec<Label<'src>> = Vec::new();
            // Per-section scratch for the annotation line ranges.
            let mut ranges_scratch: Vec<(usize, usize)> = Vec::new();
            // Per-section scratch for `write_source_section`'s rebuilt
            // annotation lists (extra-primary line ranges and the combined
            // label line ranges): threaded from here so a render with
            // several sections allocates them once.
            let mut extra_lines_scratch: Vec<(usize, usize)> = Vec::new();
            let mut all_label_lines_scratch: Vec<(usize, usize)> = Vec::new();
            // `merged_labels` borrows from both `diag.labels` and
            // `synthetic`; declaring `synthetic` first keeps its drop after
            // `merged_labels`.
            let mut merged_labels: SmallVec<[&Label<'src>; 4]> = SmallVec::new();
            merged_labels.extend(diag.labels.iter());
            merged_labels.extend(synthetic.iter());
            // Render source sections for each cluster of overlapping primary spans.
            // Sort spans by position, then merge overlapping context windows into
            // a single source section so the output reads top-to-bottom without
            // redundant line ranges (e.g. lines 1-3 and 3-6 → 1-6).
            let sorted_spans = diag.spans.primary();
            debug_assert!(
                sorted_spans.windows(2).all(|w| w[0] <= w[1]),
                "MultiSpan::primary must be sorted by (file, start)",
            );

            // Line index shared by every conversion in this render: each
            // lookup is O(log n) instead of rescanning the file.  The batch
            // path supplies one (built once per source, not per diagnostic);
            // the local fallback builds it once here.
            let owned;
            let idx: &LineIndex = match idx {
                Some(provided) => provided,
                None => {
                    owned = LineIndex::new(source);
                    &owned
                }
            };
            // Each merged label's (start_line, end_line), converted once
            // per render and shared by every source section below.
            let merged_label_lines: Vec<(usize, usize)> = merged_labels
                .iter()
                .map(|lbl| idx.span_lines(lbl.span))
                .collect();

            // ── Multi-file guard ───────────────────────────────────
            // Source sections can only locate spans inside `diag.source`;
            // the emitter contract is that the source belongs to the first
            // primary span's file, so that file is the anchor.  Line/col
            // numbers of spans from other files would be meaningless
            // against this buffer, so only anchor-file spans and labels
            // render in sections; the rest stay visible as byte-range
            // notes after the sections.
            let anchor_file = sorted_spans.first().map(|sp| sp.file);
            let foreign_spans: Vec<Span> = sorted_spans
                .iter()
                .copied()
                .filter(|sp| anchor_file.is_some_and(|anchor| sp.file != anchor))
                .collect();
            // Anchor-file view of `merged_labels`; each kept label also
            // remembers its index into `label_rendered`, so a section can
            // mark exactly the label it rendered even when earlier labels
            // were filtered out.
            let mut section_labels: Vec<&Label<'src>> = Vec::new();
            let mut section_label_lines: Vec<(usize, usize)> = Vec::new();
            let mut section_label_sources: Vec<usize> = Vec::new();
            for (i, lbl) in merged_labels.iter().enumerate() {
                if anchor_file.is_none_or(|anchor| lbl.span.file == anchor) {
                    section_labels.push(lbl);
                    section_label_lines.push(merged_label_lines[i]);
                    section_label_sources.push(i);
                }
            }

            // Build (line_range, span) pairs for merging. The line count
            // comes from the shared index (`str::lines`-equivalent) so the
            // whole file is never materialized as a `Vec<&str>` just to take
            // its length.
            let last_text_line = idx.text_line_count().saturating_sub(1);
            let ranges: Vec<((usize, usize), Span)> = sorted_spans
                .iter()
                .filter(|sp| anchor_file.is_none_or(|anchor| sp.file == anchor))
                .map(|&sp| {
                    let start_pos = idx.to_linecol(sp.start_as_usize());
                    let end_pos = idx.to_linecol(sp.end_as_usize());
                    let first = start_pos.line.saturating_sub(self.context_lines);
                    let last = std::cmp::min(end_pos.line + 1 + self.context_lines, last_text_line);
                    ((first, last), sp)
                })
                .collect();

            // Merge overlapping ranges. `ranges` is the upper bound on the
            // merged count, so reserve it once instead of regrowing.
            // Each group keeps its real spans (order inherited from
            // `sorted_spans`); the byte-union is derived at use time for the
            // location header and line range only.
            let mut merged: Vec<((usize, usize), Vec<Span>)> = Vec::with_capacity(ranges.len());
            for (range, sp) in &ranges {
                if let Some(last) = merged.last_mut() {
                    // Merge only same-file ranges: line numbers collide
                    // across files, so overlap in (first,last) means
                    // nothing when `sp.file` differs.
                    if last.1[0].file == sp.file && range.0 <= last.0.1.saturating_add(1) {
                        last.0.1 = last.0.1.max(range.1);
                        last.1.push(*sp);
                        continue;
                    }
                }
                merged.push((*range, vec![*sp]));
            }

            // Global line-number width: the largest line number any section
            // can show is the file's last text line, so the width is shared
            // by every section (previously each section sized its own
            // column from its local window, and wide context lines drifted
            // the gutters of neighbouring sections out of alignment).
            let line_num_width =
                last_text_line.checked_ilog10().unwrap_or(0) as usize + 1;

            // Per-label "rendered inline somewhere" flags, indexed like
            // `merged_labels` (and therefore `diag.labels`), set by the
            // sections as they emit each label's underline or message.  A
            // line-range test is not enough: a label whose line lies inside
            // a rendered range can still be dropped by the per-line byte
            // intersection (e.g. a point on the phantom line after a
            // trailing newline, which no section ever draws), and would then
            // never be shown at all.
            let mut label_rendered = vec![false; merged_labels.len()];
            // ── Fold markers between disjoint sections ─────────────────
            //
            // A diagnostic can span more than one disjoint source region; each
            // merged range is rendered as its own section.  Without a marker
            // the sections sit back-to-back and the reader has to infer the
            // omitted rows from the line-number jump.  Insert a single `…`
            // line between them — one row, always cheaper than drawing even a
            // 1-line gap, and the same visual convention as `diff` hunk
            // separators.
            //
            // Cost model: showing a gap costs `gap` rows; folding costs 1.
            // The cost is additive and independent per gap, so the optimal
            // layout is separable and the DP collapses to "fold every gap".
            // The structure is kept here so a future *coupled* cost (e.g.
            // "one big fold beats two small ones") can slot in without
            // touching the call site.
            for (i, (_, group)) in merged.iter().enumerate() {
                // The group's earliest span is the section primary (caret +
                // location header); the other real primaries are passed as
                // extra `^` annotations, and `write_source_section` extends
                // its window to cover them.  The section therefore shows the
                // same line range the old byte-union did, but intermediate
                // lines between the real spans get no caret.
                extras_labels.clear();
                extras_labels.extend(group[1..].iter().map(|&sp| Label::new(sp, "")));
                // Fold-marker gap: lines hidden between the previous
                // section's window end and this section's window start
                // (0 when the sections are contiguous — no fold is
                // drawn).
                let folded_lines = i
                    .checked_sub(1)
                    .and_then(|prev_i| merged.get(prev_i))
                    .map(|prev| merged[i].0 .0.saturating_sub(prev.0 .1).saturating_sub(1));
                self.write_source_section(
                    &mut out,
                    source,
                    idx,
                    group[0],
                    &section_labels,
                    &section_label_lines,
                    &section_label_sources,
                    &extras_labels,
                    &mut ranges_scratch,
                    &mut extra_lines_scratch,
                    &mut all_label_lines_scratch,
                    &diag.file_name,
                    diag.level,
                    folded_lines,
                    i == 0,
                    &mut label_rendered,
                    line_num_width,
                    &diag.suggestions,
                    &mut suggestion_rendered,
                );
            }

            // ── Fallback: render labels not shown in any source section ──
            // A label may fall outside every rendered line range (e.g. a
            // "previous definition here" on a different function), in which
            // case it is rendered as a note so it stays visible.  The test is
            // per label and per emission — see `label_rendered` above — so a
            // label that a section could not draw, even inside its range,
            // still appears instead of being silently dropped.
            // Every fallback label shares the same `│  = note: ` gutter, so
            // compute its width once instead of per label.
            let fallback_prefix_cols = self.prefix_cols(DiagnosticLevel::Note);
            for (i, lbl) in diag.labels.iter().enumerate() {
                if label_rendered.get(i).copied().unwrap_or(true) {
                    continue;
                }
                // Multi-file guard: a label from another file cannot be
                // located in this buffer; keep it visible via its byte
                // range instead of a bogus line:col.
                let foreign = anchor_file.is_some_and(|anchor| lbl.span.file != anchor);
                let pos = (!foreign).then(|| idx.to_linecol(lbl.span.start_as_usize()));
                let sanitized = super::sanitize_for_terminal_multiline(&lbl.message);
                let segs = self.wrap_msg(&sanitized, fallback_prefix_cols);
                let last = segs.len().saturating_sub(1);
                for (i, seg) in segs.iter().enumerate() {
                    // Keep the `at line:col` locator attached to the final
                    // line — written inline so no `suffix` buffer per line.
                    if i == last {
                        if let Some(pos) = pos {
                            let _ = writeln!(
                                out,
                                "{v}  {cyan}= {bold}note{reset}: {msg} at {line}:{col}",
                                v = self.bc.v,
                                cyan = self.s.help,
                                bold = self.s.bold,
                                reset = self.s.reset,
                                msg = seg,
                                line = pos.line + 1,
                                col = pos.col + 1,
                            );
                        } else {
                            let _ = writeln!(
                                out,
                                "{v}  {cyan}= {bold}note{reset}: {msg} at {span}",
                                v = self.bc.v,
                                cyan = self.s.help,
                                bold = self.s.bold,
                                reset = self.s.reset,
                                msg = seg,
                                span = lbl.span,
                            );
                        }
                    } else {
                        let _ = writeln!(
                            out,
                            "{v}  {cyan}= {bold}note{reset}: {msg}",
                            v = self.bc.v,
                            cyan = self.s.help,
                            bold = self.s.bold,
                            reset = self.s.reset,
                            msg = seg,
                        );
                    }
                }
            }

            // Multi-file guard: primary spans from other files cannot be
            // drawn in this buffer either; surface each as a location-only
            // note so the diagnostic's other sites stay visible.  Prefix the
            // byte range with the filename when it is a real file (not the
            // `<input>` placeholder) so the site stays identifiable.
            let fname = diag.file_name();
            let has_fname = !fname.is_empty() && fname != "<input>";
            for sp in &foreign_spans {
                let loc = if has_fname {
                    format!("{fname} {sp}")
                } else {
                    sp.to_string()
                };
                let _ = writeln!(
                    out,
                    "{v}  {cyan}= {bold}note{reset}: at {loc}",
                    v = self.bc.v,
                    cyan = self.s.help,
                    bold = self.s.bold,
                    reset = self.s.reset,
                    loc = loc,
                );
            }
        } else {
            // No source attached: surface the byte ranges of the primary
            // spans so the diagnostic's location is not silently lost.
            // Prefix with the filename when it is a real file.
            let fname = diag.file_name();
            let has_fname = !fname.is_empty() && fname != "<input>";
            for sp in diag.spans.primary() {
                let loc = if has_fname {
                    format!("{fname} {sp}")
                } else {
                    sp.to_string()
                };
                let _ = writeln!(
                    out,
                    "{v}  {cyan}= {bold}note{reset}: at {loc}",
                    v = self.bc.v,
                    cyan = self.s.help,
                    bold = self.s.bold,
                    reset = self.s.reset,
                    loc = loc,
                );
            }
        }

        // Children (sub-diagnostics: notes, help)
        for child in &diag.children {
            self.write_child(&mut out, child);
        }

        // Suggestions: `help:` for machine-applicable, `note:` otherwise
        // (with a `(maybe incorrect)` / `(has placeholders)` tag via
        // `Suggestion::Display`).  Suggestions already rendered as a diff
        // block inside a source section are skipped here.
        for (i, suggestion) in diag.suggestions.iter().enumerate() {
            if suggestion_rendered[i] {
                continue;
            }
            let text = suggestion.to_string();
            match suggestion.applicability {
                Applicability::MachineApplicable => {
                    self.write_help_line(&mut out, &text);
                }
                _ => {
                    self.write_note_line(&mut out, &text);
                }
            }
        }

        // Help
        if let Some(ref help) = diag.help {
            self.write_help_line(&mut out, help);
        }

        out
    }

    /// Render the summary line (e.g. "aborting due to N previous errors").
    pub fn render_summary(&self, error_count: usize, warning_count: usize) -> String {
        if error_count == 0 && warning_count == 0 {
            return String::new();
        }

        let mut out = String::with_capacity(64);
        let accent = if error_count > 0 { self.s.error } else { self.s.warning };
        let _ = write!(
            out,
            "{dim}{bl}{h}{reset} {accent}",
            dim = self.s.dim,
            bl = self.bc.bl,
            h = self.bc.h,
            reset = self.s.reset,
            accent = accent,
        );
        match (error_count, warning_count) {
            (0, 0) => unreachable!("returned early above"),
            (0, w) => {
                let _ = write!(
                    out,
                    "{}",
                    crate::tr!("summary-warnings", count = DisplayArg(w))
                );
            }
            (e, 0) => {
                let _ = write!(
                    out,
                    "{}",
                    crate::tr!("summary-errors", count = DisplayArg(e))
                );
            }
            (e, w) => {
                let _ = write!(
                    out,
                    "{}; ",
                    crate::tr!("summary-mixed-errors", count = DisplayArg(e))
                );
                let _ = write!(
                    out,
                    "{}",
                    crate::tr!("summary-warnings", count = DisplayArg(w))
                );
            }
        }
        let _ = write!(out, "{}", self.s.reset);
        out
    }

    /// Render the summary line with a file count (e.g. "aborting due to
    /// 3 previous errors in 2 files").  Falls back to [`render_summary`](Self::render_summary)
    /// when `file_count <= 1` so single-file output is byte-identical.
    pub fn render_summary_full(
        &self,
        error_count: usize,
        warning_count: usize,
        file_count: usize,
    ) -> String {
        if file_count <= 1 {
            return self.render_summary(error_count, warning_count);
        }
        let mut s = self.render_summary(error_count, warning_count);
        if s.is_empty() {
            return s;
        }
        let _ = write!(
            s,
            " {}",
            crate::tr!("summary-files", count = DisplayArg(file_count))
        );
        s
    }

    // ── Header ──────────────────────────────────────────────────

    fn write_header(&self, out: &mut String, diag: &Diagnostic<'_>) {
        let level_label = diag.level.label();
        let level_color = self.s.get(Self::level_style(diag.level));
        // `with_code_str` validates through `try_new`, but `ErrCode::unknown` is
        // the documented escape hatch for untrusted strings — sanitize every code
        // before it enters the SGR run below, the same way `url_ansi` sanitizes
        // its visible text.  Each code keeps its `ErrCode` so the header can
        // wrap it in an OSC 8 hyperlink when the terminal supports one.
        // The `None` fallback owns an `ErrCode::unknown("?")` (borrowed via the
        // `?` literal), since a codeless diagnostic still shows `[]`.
        let fallback_code = crate::diagnostics::ErrCode::unknown("?");
        let all_codes: Vec<&crate::diagnostics::ErrCode> = if diag.related_errors.is_empty() {
            match diag.code.as_ref() {
                Some(c) => vec![c],
                None => vec![&fallback_code],
            }
        } else {
            let mut codes: Vec<&crate::diagnostics::ErrCode> = Vec::new();
            if let Some(c) = diag.code.as_ref() {
                codes.push(c);
            }
            for rel in &diag.related_errors {
                if let Some(c) = rel.code.as_ref() {
                    if !codes.iter().any(|k| k.code() == c.code()) {
                        codes.push(c);
                    }
                }
            }
            if codes.is_empty() {
                codes.push(&fallback_code);
            }
            codes
        };
        // Visible comma-joined form, used for the width of the fixed
        // `┌─ error[E030]: ` prefix (sanitization can inject U+FFFD).
        let all_codes_visible: Cow<'_, str> = if all_codes.len() == 1 {
            super::sanitize_for_terminal(all_codes[0].code())
        } else {
            let mut joined = String::new();
            for (i, c) in all_codes.iter().enumerate() {
                if i > 0 {
                    joined.push(',');
                }
                joined.push_str(&super::sanitize_for_terminal(c.code()));
            }
            Cow::Owned(joined)
        };
        // OSC 8 hyperlink form of the code list, used only when the terminal
        // supports hyperlinks and color is on — each code becomes its own
        // `url_ansi` link so Ctrl+click opens the matching `--explain` page.
        let all_codes_display = if self.s.use_color && super::term::supports_hyperlinks() {
            let mut s = String::new();
            for (i, c) in all_codes.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&c.url_ansi());
            }
            Cow::Owned(s)
        } else {
            all_codes_visible.clone()
        };
        // Visible width of the fixed `┌─ error[E030]: ` prefix, computed
        // directly from its parts (layout: `{tl}{h} {level}[{codes}]: `)
        // instead of building the string only to measure it.  Box chars are
        // one cell each; the level label is ASCII; `all_codes` goes through
        // `visible_width` because sanitization can inject U+FFFD.
        //
        // With related errors the primary is item `1.`; its continuation
        // lines reserve the same gutter but leave the number off so it does
        // not collide with the related items' `2.`, `3.`, ….
        let numbered = !diag.related_errors.is_empty();
        let header_cols = self.bc.tl.chars().count() // ┌
            + self.bc.h.chars().count()              // ─
            + 1                                      // space
            + level_label.len()                      // error
            + 1                                      // [
            + visible_width(&all_codes_visible)      // E030
            + "]: ".len()                            // ]: 
            + if numbered { 3 } else { 0 };
        // Continuation gutter for wrapped header segments: the `│` replaces
        // the `┌`, and the spaces fill the rest of the prefix so the segment
        // starts at the same column.  `header_cols` already includes the
        // 3-column number slot when numbered, so the no-number prefix width
        // is `header_cols - (if numbered { 3 } else { 0 })`.
        let prefix_no_num = header_cols - usize::from(numbered) * 3;
        let cont_pad: Cow<'_, str> = Cow::Owned(" ".repeat(prefix_no_num - 1));
        let cont_number = if numbered { "   " } else { "" };
        let msg_src: Cow<'_, str> = if diag.message.trim().is_empty() {
            // Empty message: fall back to the code's title when a code
            // exists, otherwise a neutral placeholder.  Without this
            // `wrap_msg` returns a single empty segment and the header
            // ends in a bare trailing colon.
            diag.code
                .as_ref()
                .map(|c| Cow::Borrowed(c.title()))
                .unwrap_or_else(|| crate::tr!("unknown-error"))
        } else {
            Cow::Borrowed(&diag.message)
        };
        let sanitized_msg = super::sanitize_for_terminal_multiline(&msg_src);
        for (i, seg) in self.wrap_msg(&sanitized_msg, header_cols).iter().enumerate() {
            if i == 0 {
                // First line: the full `┌─ error[...]: ` prefix.
                let _ = write!(
                    out,
                    "{dim}{tl}{h}{reset} {level_color}{bold}{level}{reset}[{bold}{all_codes}{reset}]: ",
                    dim = self.s.dim,
                    tl = self.bc.tl,
                    h = self.bc.h,
                    reset = self.s.reset,
                    level_color = level_color,
                    bold = self.s.bold,
                    level = level_label,
                    all_codes = all_codes_display,
                );
                let lead = if numbered { "1. " } else { "" };
                let _ = writeln!(
                    out,
                    "{}{}",
                    lead,
                    crate::diagnostics::glyph::highlight_code(seg, self.s.use_color)
                );
            } else {
                // Continuation: `│` + padding so the segment starts at the
                // same column as the first line's message.
                let _ = write!(
                    out,
                    "{dim}{v}{reset}{pad}{num}",
                    dim = self.s.dim,
                    v = self.bc.v,
                    reset = self.s.reset,
                    pad = cont_pad,
                    num = cont_number,
                );
                let _ = writeln!(
                    out,
                    "{}",
                    crate::diagnostics::glyph::highlight_code(seg, self.s.use_color)
                );
            }
        }
        // Calculate the indentation for related error items so they align
        // with the "1." on the header line:
        //   ┌─ error[E019,E030]: 1. duplicate definition of `i`
        //   │                    2. type mismatch
        let indent = " ".repeat(header_cols - 1);
        // List related errors as numbered items under the header.
        for (i, rel) in diag.related_errors.iter().enumerate() {
            let num = i + 2;
            let item_prefix = format!("{}{}{}. ", self.bc.v, indent, num);
            let item_cols = visible_width(&item_prefix);
            let rel_msg = super::sanitize_for_terminal_multiline(&rel.message);
            for (j, seg) in self.wrap_msg(&rel_msg, item_cols).iter().enumerate() {
                // Gutter pieces written straight into `out`; the continuation
                // lines carry no number so they stay clear of the related
                // items' `2.`, `3.`, ….
                let _ = write!(out, "{v}{indent}", v = self.bc.v, indent = indent);
                if j == 0 {
                    let _ = write!(out, "{num}. ", num = num);
                } else {
                    let _ = write!(out, "   ");
                }
                let _ = writeln!(
                    out,
                    "{}",
                    crate::diagnostics::glyph::highlight_code_with_backticks(seg, self.s.use_color)
                );
            }
        }
    }

    // ── Source section ───────────────────────────────────────────

    fn write_source_section<'a, 'src>(
        &self,
        out: &mut String,
        source: &str,
        idx: &LineIndex,
        primary_span: Span,
        labels: &[&'a Label<'src>],
        label_lines: &[(usize, usize)],
        // Position → index in `label_rendered` for each entry of `labels`.
        // The two differ when earlier labels were filtered out (foreign
        // files, see the multi-file guard in `render_diagnostic_with_index`).
        label_sources: &[usize],
        extras: &'a [Label<'src>],
        // Per-section line-range scratch, reused across the render's
        // sections instead of reallocated (same discipline as the row
        // scratch buffers below).
        scratch_ranges: &mut Vec<(usize, usize)>,
        // Per-section annotation list scratch: the section-local
        // `extras_lines` / `all_label_lines` rebuilds below, threaded from
        // the caller so a render with many sections allocates them once.
        // (`all_labels` cannot be threaded: it holds references into
        // per-section `extras`, which do not outlive a section call.)
        scratch_extra_lines: &mut Vec<(usize, usize)>,
        scratch_all_label_lines: &mut Vec<(usize, usize)>,
        filename: &str,
        level: crate::diagnostics::level::DiagnosticLevel,
        // Lines hidden between the previous section's window end and this
        // section's window start; `None` (first section) draws no fold
        // marker, 0 draws a bare `...` (sections contiguous in the
        // merged range but not in the file), N > 0 draws `... (N lines
        // omitted)`.
        folded_lines: Option<usize>,
        // When `false` (subsequent section of a multi-span diagnostic), the
        // location header omits the filename and shows just `:line:col`.
        show_filename: bool,
        // Per-label "already emitted inline" flags shared across the render's
        // sections (indexed like `merged_labels`); see `render_diagnostic`
        // for why a line-range test is not enough.
        label_rendered: &mut [bool],
        // Global line-number column width, shared across all sections of
        // this render so the gutters stay aligned (computed by the caller
        // from the file's last text line).
        global_line_num_width: usize,
        // Structural suggestions to attempt diff-rendering in this section.
        // `suggestion_rendered` is written back so the caller's fallback
        // loop knows which ones were already drawn inline.
        suggestions: &[crate::diagnostics::Suggestion<'src>],
        suggestion_rendered: &mut [bool],
    ) {
        // Empty source has no rows to draw. `text_line_count()` reports 1
        // for `""` (str::lines-style), so guard on the source itself rather
        // than materializing every line just to test for emptiness.
        if source.is_empty() {
            return;
        }
        // Shared line index (built once per render); the per-line
        // underline helpers index into it instead of rescanning.
        let starts = idx.starts();

        // Extra real primaries of a merged group (all but the earliest,
        // which is `primary_span`) join the annotation list so each keeps
        // its own `^` caret.  The display window below is extended to
        // cover them, so the section shows the same line range the old
        // byte-union primary did.  All three lists live in caller-owned
        // scratch buffers (rebuilt per section, allocated once per render).
        scratch_extra_lines.clear();
        scratch_extra_lines.extend(extras.iter().map(|l| idx.span_lines(l.span)));
        let extras_lines: &[(usize, usize)] = scratch_extra_lines;

        // `all_labels` is rebuilt per section: it holds references into
        // `extras`, which are per-section data and cannot be stored in a
        // caller-owned scratch across sections.
        let mut all_labels: Vec<&'a Label<'src>> = Vec::with_capacity(labels.len() + extras.len());
        all_labels.extend(labels.iter().copied());
        all_labels.extend(extras.iter());

        scratch_all_label_lines.clear();
        scratch_all_label_lines.extend_from_slice(label_lines);
        scratch_all_label_lines.extend_from_slice(extras_lines);
        let all_label_lines: &[(usize, usize)] = scratch_all_label_lines;

        let start_pos = idx.to_linecol(primary_span.start_as_usize());
        let end_pos = idx.to_linecol(primary_span.end_as_usize());

        // Determine the line range to display (with context)
        let first_line = start_pos.line.saturating_sub(self.context_lines);
        let text_last = idx.text_line_count().saturating_sub(1);
        let mut last_line = std::cmp::min(end_pos.line + 1 + self.context_lines, text_last);
        // Extras come after `primary_span` (spans are sorted by start), so
        // only the window's end needs extending.
        for &(_, extra_end) in extras_lines {
            last_line = last_line.max(std::cmp::min(extra_end + 1 + self.context_lines, text_last));
        }

        // Width of the line number column: the max of this section's local
        // need and the render-wide column, so every section's gutter stays
        // aligned (computed by the caller from the file's last text line).
        let local_line_num_width = (last_line + 1)
            .checked_ilog10()
            .unwrap_or(0) as usize
            + 1;
        let line_num_width = std::cmp::max(local_line_num_width, global_line_num_width);
        // Indent for lines that don't have a line number (location header, spacing, explanation, fold marker)
        let indent = " ".repeat(line_num_width + 1);

        // ── Fold marker ────────────────────────────────────────────
        // Drawn with this section's own gutter (`│` + indent) so it lines
        // up with the location header directly below it.  Emitted before
        // the header so the reader sees the omission *above* the section
        // it precedes.  The marker text is ASCII (`...`) in both the
        // Unicode and ASCII box-character modes; when lines were hidden
        // the gap count makes the omission concrete
        // (`... (37 lines omitted)`).
        if let Some(gap) = folded_lines {
            let marker = if gap > 0 {
                Cow::Owned(format!("... ({gap} lines omitted)"))
            } else {
                Cow::Borrowed("...")
            };
            let _ = writeln!(
                out,
                "{dim}{v}{indent}{marker}{reset}",
                dim = self.s.dim,
                v = self.bc.v,
                indent = indent,
                marker = marker,
                reset = self.s.reset,
            );
        }

        // Location header: ┌─ input:1:1  (with OSC 8 hyperlink when filename is a real path)
        //
        // The filename is external text: control characters would be
        // interpreted by the terminal (forging links, changing the title,
        // clearing the screen), so strip them from the *displayed* path.  The
        // URL is separately percent-encoded below.
        let filename_display = super::sanitize_filename(filename);
        let location_str = if show_filename {
            format!(
                "{filename}:{line}:{col}",
                filename = filename_display,
                line = start_pos.line + 1,
                col = start_pos.col + 1,
            )
        } else {
            format!(
                ":{line}:{col}",
                line = start_pos.line + 1,
                col = start_pos.col + 1,
            )
        };
        let location_display = if show_filename
            && self.s.use_color
            && super::term::supports_hyperlinks()
            && filename != "<input>"
            && !filename.is_empty()
        {
            // OSC 8 hyperlink — Ctrl+click opens the file in the editor.
            // Convert to absolute path so the `file://` URL works correctly.
            let abs_path = std::path::Path::new(filename)
                .canonicalize()
                .unwrap_or_else(|_| std::path::PathBuf::from(filename));
            // Percent-encode the path so control characters in a filename
            // cannot terminate the OSC 8 sequence early or inject further
            // terminal escapes.
            let file_url = format!(
                "file://{}",
                encode_file_url_path(&abs_path.to_string_lossy())
            );
            format!("\x1b]8;;{file_url}\x1b\\{location_str}\x1b]8;;\x1b\\")
        } else {
            location_str
        };
        // Compact mode drops the sub-box header's `┌─ ` prefix and the
        // spacing row: the location line sits directly above the first
        // source row.
        let sub_box_prefix = if self.compact {
            Cow::Borrowed("")
        } else {
            Cow::Owned(format!("{}{}", self.bc.sub_tl, self.bc.sub_h))
        };
        let _ = writeln!(
            out,
            "{dim}{v}{indent}{prefix}{reset} {cyan}{location_display}{reset}",
            dim = self.s.dim,
            v = self.bc.v,
            indent = indent,
            prefix = sub_box_prefix,
            reset = self.s.reset,
            cyan = self.s.help,
            location_display = location_display,
        );
        // Spacing line (skipped in compact mode)
        if !self.compact {
            let _ = writeln!(
                out,
                "{dim}{v}{indent}{sub_v}{reset}",
                dim = self.s.dim,
                v = self.bc.v,
                indent = indent,
                sub_v = self.bc.sub_v,
                reset = self.s.reset
            );
        }

        // Render each line in the range
        // Per-label (start_line, end_line): the primary span converts here
        // (O(log n)); the rest were precomputed once per render and arrive
        // via `label_lines`, so sections never rescan the file.
        debug_assert_eq!(all_label_lines.len(), all_labels.len());
        // Line range per annotation: index 0 is this section's primary
        // (the synthetic underline), index i+1 is `all_labels[i]` (the
        // caller's `labels`, then this group's extras) — the same keying
        // the underline `label_idx` uses below.  Built into the caller's
        // reused scratch buffer.
        scratch_ranges.clear();
        scratch_ranges.reserve(all_labels.len() + 1);
        scratch_ranges.push(idx.span_lines(primary_span));
        scratch_ranges.extend_from_slice(&all_label_lines);
        let label_line_ranges: &[(usize, usize)] = scratch_ranges;

        // Display-column-aware source line rendering loop.  `rendered` is
        // hoisted out of the row loop to avoid per-iteration allocation.
        let mut rendered = String::with_capacity(256);
        // Underline lane allocator — owns the interval-graph colouring scratch
        // (heap, order, column buffer) so the row loop reuses it across lines.
        let mut lanes = UnderlineLanes::new();
        // `multiline_flags` holds only `(display_col, part)` — the two fields
        // read back below — so no per-line `String` is cloned out of
        // `underlines_buf`, and the flags outlive that reused buffer.
        let mut multiline_flags: Vec<(usize, &'static str)> = Vec::new();
        let mut connector = String::new();
        // Scratch buffers — reused each iteration; `clear()` preserves capacity.
        let mut underlines_buf: Vec<(usize, usize, char, &'a str, Option<usize>)> =
            Vec::with_capacity(all_labels.len() + 1);
        let mut ranges_buf: Vec<(usize, usize)> = Vec::with_capacity(all_labels.len() + 1);
        let mut meta_buf: Vec<(char, &'a str, Option<usize>)> =
            Vec::with_capacity(all_labels.len() + 1);
        let mut display_buf: Vec<(usize, usize)> = Vec::with_capacity(all_labels.len() + 1);
        let mut order_buf: Vec<usize> = Vec::with_capacity(all_labels.len() + 1);
        let mut done_buf: Vec<bool> = Vec::with_capacity(all_labels.len() + 1);
        let mut active_buf: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
        // Sliding window over `all_labels`, ordered by start line once here;
        // each row then only visits the labels that can cover it, not all
        // of them.
        let mut label_window = LabelWindow::new(&all_label_lines);
        // Row-invariant: the primary-highlight escapes depend only on
        // `use_color` and `color_depth`, so resolve them once instead of per
        // source row.
        let highlight_on = self.s.highlight_bg;
        let highlight_off = if self.s.use_color { "\x1b[49m" } else { "" };

        for line_idx in first_line..=last_line {
            // O(1) slice via the shared index instead of indexing a
            // whole-file `Vec<&str>` that was collected only for this.
            let line = idx.line(line_idx);
            // One grapheme walk converts this row's byte ranges to display
            // columns, measures the line's width, and emits the highlighted
            // source text — replacing the three separate passes over the
            // line the row used to run.
            let display_width = compute_line_underlines_and_render(
                primary_span,
                &all_labels,
                &mut label_window,
                source,
                line,
                line_idx,
                &starts,
                &mut underlines_buf,
                &mut rendered,
                highlight_on,
                highlight_off,
                &mut ranges_buf,
                &mut meta_buf,
                &mut display_buf,
                &mut order_buf,
                &mut done_buf,
                &mut active_buf,
                self.tab_width,
            );
            let underlines = &underlines_buf;

            // Line number + source
            let _ = writeln!(
                out,
                "{dim}{v}{reset}{dim}{line_num:>width$}{reset} {dim}{sub_v}{reset} {rendered}",
                dim = self.s.dim,
                v = self.bc.v,
                reset = self.s.reset,
                line_num = line_idx + 1,
                width = line_num_width,
                sub_v = self.bc.sub_v,
                rendered = rendered,
            );

            // ── Detect multi-line labels ───────────────────────────
            // Keyed by label index, not message text: two labels can carry
            // the same message while covering different lines, and a
            // message-keyed map would merge them. `label_line_ranges` is
            // indexed like `labels` offset by +1 (index 0 is the
            // per-section primary).
            multiline_flags.clear();
            for (col, _ulen, _underline_char, msg, label_idx) in underlines {
                let Some(i) = label_idx else {
                    continue;
                };
                let (start, end) = label_line_ranges[i + 1];
                // A label counts as shown once something for it is actually
                // emitted: its message (message loop below, on the final
                // covered line) or — for a single-line or message-less label
                // — this underline row alone.  Multi-line labels are marked
                // only from the message loop: an underline-only tail (end
                // line outside the shown window) still deserves the fallback
                // note rather than being counted as fully shown.
                if (msg.is_empty() || start == end)
                    && let Some(&src_idx) = label_sources.get(*i)
                    && let Some(flag) = label_rendered.get_mut(src_idx)
                {
                    *flag = true;
                }
                if msg.is_empty() {
                    continue;
                }
                let part = if line_idx == start {
                    "start"
                } else if line_idx == end {
                    "end"
                } else {
                    "middle"
                };
                multiline_flags.push((*col, part));
            }

            let annotation_style = Self::level_style(level);
            // Per-character colours: `^` uses the diagnostic's level colour,
            // `~` (secondary) uses blue, ` ` (help) uses the help colour,
            // `-` and blank padding use dim.
            let level_color = self.s.get(annotation_style);
            let blue = self.s.get(Style::Note);
            let dim_c = self.s.dim;
            let help_color = self.s.help;
            let connector_color = level_color;
            let connector_reset = self.s.reset;
            // ── Underline lines (one row per overlap depth) ──────────
            if !underlines.is_empty() {
                let lane_count = lanes.assign(underlines);
                let mut row_buf = String::new();
                for lane_idx in 0..lane_count {
                    let visible = lanes.row(
                        underlines,
                        lane_idx,
                        display_width,
                        &mut row_buf,
                        |ch| match ch {
                            '^' => level_color,
                            '~' => blue,
                            ' ' => help_color,
                            _ => dim_c,
                        },
                        self.s.reset,
                    );
                    if visible {
                        let _ = writeln!(
                            out,
                            "{dim}{v}{indent}{sub_v}{reset} {row}",
                            dim = self.s.dim,
                            v = self.bc.v,
                            indent = indent,
                            sub_v = self.bc.sub_v,
                            reset = self.s.reset,
                            row = row_buf,
                        );
                    }
                }

                // Fast path: when every visible message on this row is
                // single-line, non-multiline, and non-help, the message
                // goes inline after the underline row (`^^^^ message`),
                // rustc's common compact form — no separate vertical-bar
                // connector line.  A connector row is needed when a
                // message would otherwise be visually ambiguous: a
                // help-style annotation, a multi-line label, or more
                // than one message sharing the row.  Multi-line
                // NON-FINAL labels are skipped — their message is
                // deliberately not printed on this row, so they must
                // not influence the layout of the messages that are.
                let needs_connector = {
                    let mut visible = 0usize;
                    let mut force = false;
                    for (_, _, ch, msg, label_idx) in underlines {
                        if msg.is_empty() {
                            continue;
                        }
                        if label_idx.is_some_and(|i| label_line_ranges[i + 1].1 != line_idx) {
                            continue;
                        }
                        visible += 1;
                        if *ch == ' '
                            || label_idx.is_some_and(|i| {
                                let (s, e) = label_line_ranges[i + 1];
                                s != e
                            })
                        {
                            force = true;
                        }
                    }
                    force || visible > 1
                };
                for (col, _ulen, ch, msg, label_idx) in
                    underlines.iter().filter(|(_, _, _, msg, _)| !msg.is_empty())
                {
                    // For a multi-line label, render its message only on the
                    // final covered line; earlier lines get the connector
                    // below, so the same text is not repeated on every row.
                    let is_multiline_nonfinal = label_idx
                        .is_some_and(|i| label_line_ranges[i + 1].1 != line_idx);
                    if is_multiline_nonfinal {
                        continue;
                    }
                    if let Some(i) = label_idx
                        && let Some(&src_idx) = label_sources.get(*i)
                        && let Some(flag) = label_rendered.get_mut(src_idx)
                    {
                        // This message is the label's rendering; block the
                        // fallback from emitting a duplicate note.
                        *flag = true;
                    }
                    // `col` is a display column; pad with spaces to align.
                    connector.clear();
                    for _ in 0..*col {
                        connector.push(' ');
                    }
                    connector.push('|');
                    // Visible width of the `│<indent>│ <spaces>| ` prefix the
                    // message text follows.  Summed from its parts (the same
                    // arithmetic `header_cols` uses) instead of building the
                    // string only to measure it — one `format!` per label
                    // otherwise.  A help label prepends `help: ` on its first
                    // line only, so reserve that gutter too.
                    let prefix_cols = visible_width(self.bc.v)
                        + indent.len() // ASCII spaces
                        + visible_width(self.bc.sub_v)
                        + 2 // the spaces around the connector
                        + connector.len() // ASCII spaces + `|`
                        + if *ch == ' ' { "help: ".len() } else { 0 };
                    if !needs_connector {
                        // Inline: `^^^^ message` on the underline row's
                        // gutter — no separate connector line.  The
                        // underline rows were already emitted above, so
                        // the message follows them directly, indented to
                        // the source-column position.
                        let sanitized_msg = super::sanitize_for_terminal_multiline(msg);
                        // Gutter width: `│` + indent + `│` + one space,
                        // then the message aligns under the source text.
                        let msg_prefix = visible_width(self.bc.v)
                            + indent.len()
                            + visible_width(self.bc.sub_v)
                            + 1;
                        let segments = self.wrap_msg(&sanitized_msg, msg_prefix);
                        for (i, seg) in segments.iter().enumerate() {
                            let highlighted =
                                crate::diagnostics::glyph::highlight_code_with_backticks(
                                    seg,
                                    self.s.use_color,
                                );
                            let gutter = if i == 0 {
                                format!(
                                    "{dim}{v}{reset}{indent}{dim}{sub_v}{reset} ",
                                    dim = self.s.dim,
                                    v = self.bc.v,
                                    reset = self.s.reset,
                                    indent = indent,
                                    sub_v = self.bc.sub_v,
                                )
                            } else {
                                format!(
                                    "{dim}{v}{reset}{pad}{dim}{sub_v}{reset} ",
                                    dim = self.s.dim,
                                    v = self.bc.v,
                                    reset = self.s.reset,
                                    pad = " ".repeat(indent.len()),
                                    sub_v = self.bc.sub_v,
                                )
                            };
                            let _ = writeln!(
                                out,
                                "{gutter}{level_color}{highlighted}{reset}",
                                level_color = level_color,
                                highlighted = highlighted,
                                reset = self.s.reset,
                            );
                        }
                        continue;
                    }
                    // Connector line, emitted once; each wrapped segment then
                    // repeats the connector so the label reads continuously.
                    let _ = writeln!(
                        out,
                        "{dim}{v}{reset}{indent}{dim}{sub_v}{reset} {connector_color}{connector}{connector_reset}",
                        dim = self.s.dim,
                        v = self.bc.v,
                        indent = indent,
                        reset = self.s.reset,
                        sub_v = self.bc.sub_v,
                        connector_color = connector_color,
                        connector = connector,
                        connector_reset = connector_reset,
                    );
                    let sanitized_msg = super::sanitize_for_terminal_multiline(msg);
                    let segments = self.wrap_msg(&sanitized_msg, prefix_cols);
                    for (i, seg) in segments.iter().enumerate() {
                        // Fixed gutter first, written straight into `out`
                        // (reusing its capacity); the segment text follows.
                        let _ = write!(
                            out,
                            "{dim}{v}{reset}{indent}{dim}{sub_v}{reset} ",
                            dim = self.s.dim,
                            v = self.bc.v,
                            reset = self.s.reset,
                            indent = indent,
                            sub_v = self.bc.sub_v,
                        );
                        let _ = write!(
                            out,
                            "{connector_color}{connector} {connector_reset}",
                            connector_color = connector_color,
                            connector = connector,
                            connector_reset = connector_reset,
                        );
                        if *ch == ' ' {
                            if i == 0 {
                                let _ = write!(
                                    out,
                                    "{cyan}help:{reset} ",
                                    cyan = self.s.help,
                                    reset = self.s.reset,
                                );
                            }
                            let _ = writeln!(
                                out,
                                "{cyan}{seg}{reset}",
                                cyan = self.s.help,
                                reset = self.s.reset,
                                seg = seg,
                            );
                        } else {
                            let highlighted =
                                crate::diagnostics::glyph::highlight_code_with_backticks(
                                    seg,
                                    self.s.use_color,
                                );
                            // Re-apply the connector color after the reset
                            // pair `highlight` injects.  The pieces are written
                            // straight into `out`; the old `replace` allocated
                            // a second `String` per wrapped segment.  The
                            // opening color and the trailing reset stay
                            // outside the loop so the byte layout matches.
                            let _ = write!(out, "{connector_color}", connector_color = connector_color);
                            let pat = "\x1b[22m\x1b[39m";
                            let mut rest: &str = highlighted.as_ref();
                            while let Some(p) = rest.find(pat) {
                                let _ = write!(
                                    out,
                                    "{}{pat}{connector_color}",
                                    &rest[..p],
                                    pat = pat,
                                    connector_color = connector_color,
                                );
                                rest = &rest[p + pat.len()..];
                            }
                            let _ = writeln!(
                                out,
                                "{rest}{connector_reset}",
                                rest = rest,
                                connector_reset = connector_reset,
                            );
                        }
                    }
                }
            }

            // ── Multi-line label connectors ─────────────────────────
            for (col, part) in &multiline_flags {
                if *part == "middle" || *part == "end" {
                    connector.clear();
                    for _ in 0..*col {
                        connector.push(' ');
                    }
                    connector.push('|');
                    let _ = writeln!(
                        out,
                        "{dim}{v}{reset}{indent}{dim}{sub_v}{reset} {connector_color}{connector}{connector_reset}",
                        dim = self.s.dim,
                        v = self.bc.v,
                        indent = indent,
                        reset = self.s.reset,
                        sub_v = self.bc.sub_v,
                        connector_color = connector_color,
                        connector = connector,
                        connector_reset = connector_reset,
                    );
                }
            }
        }

        // Explanation box — now redundant since labels are rendered inline.
        // Kept as an empty visual terminal to close the source section
        // (skipped in compact mode, which has no box to close).

        // ── Structural suggestion diff blocks ─────────────────────
        // Each suggestion carrying a `span` + `replacement` whose
        // first line falls inside this section's window is drawn as a
        // two-line `-`/`+` diff with the message above, so the reader
        // sees exactly what would change.  Suggestions whose span
        // crosses lines or is not char-boundary-aligned are left
        // unrendered here and fall through to the caller's text-only
        // loop.
        //
        // The diff block is emitted after the entire source-row loop,
        // so when `context_lines > 0` it may sit several rows below
        // the `-`/`+` anchor line (rustc-style: the whole context
        // window is shown first, then the diff).
        debug_assert_eq!(suggestions.len(), suggestion_rendered.len());
        for (i, suggestion) in suggestions.iter().enumerate() {
            let (Some(span), Some(replacement)) = (suggestion.span, &suggestion.replacement) else {
                continue;
            };
            // Multi-file guard: `idx` is built from `diag.source`, which
            // belongs to the anchor file (`primary_span.file`).  A span
            // from another file cannot be located in this buffer; leave
            // it to the caller's text-only fallback.
            if span.file != primary_span.file {
                continue;
            }
            let (sug_first, sug_last) = idx.span_lines(span);
            // Single-line spans only: a span that crosses lines would
            // silently drop the intermediate lines from the `-`/`+` diff,
            // producing semantically wrong output.  The text-only fallback
            // remains the correct path for those.
            if sug_first != sug_last {
                continue;
            }
            if sug_first < first_line || sug_first > last_line {
                continue;
            }
            let old_line = idx.line(sug_first);
            let line_start = starts[sug_first];
            let span_start_in_line = span.start_as_usize().saturating_sub(line_start).min(old_line.len());
            let span_end_in_line = span.end_as_usize().saturating_sub(line_start).min(old_line.len());
            // Defensive: a malformed span with `end < start` would make the
            // slice below panic; bail to the text-only fallback instead.
            if span_end_in_line < span_start_in_line
                || !old_line.is_char_boundary(span_start_in_line)
                || !old_line.is_char_boundary(span_end_in_line)
            {
                continue;
            }
            let new_line = format!("{}{}{}", &old_line[..span_start_in_line], replacement, &old_line[span_end_in_line..]);
            // Raw source and replacement text are external input: route
            // them through the same sanitizer the label-message loop uses
            // so control characters (ESC, C1) and bidi overrides cannot
            // be injected into the terminal via the diff block.
            let old_display = super::sanitize_for_terminal_multiline(old_line);
            let new_display = super::sanitize_for_terminal_multiline(&new_line);

            let applicability = suggestion.applicability;
            let glyph = match applicability {
                crate::diagnostics::Applicability::MachineApplicable => "√",
                crate::diagnostics::Applicability::MaybeIncorrect => "?",
                _ => "",
            };
            let label: &'static str = match applicability {
                crate::diagnostics::Applicability::MachineApplicable => "help",
                _ => "note",
            };
            let label_color = match applicability {
                crate::diagnostics::Applicability::MachineApplicable => self.s.help,
                _ => self.s.note,
            };
            let glyph_pad = if glyph.is_empty() {
                String::new()
            } else {
                format!("{dim}{glyph}{reset}", dim = self.s.dim, reset = self.s.reset)
            };
            // `Unspecified` / `HasPlaceholders` render no glyph at all, but
            // the pad still occupies 1 column in the gutter (a blank cell),
            // so measure the *actual* rendered width instead of the glyph's
            // intrinsic width (which is 0 in that case).
            let glyph_rendered_width = if glyph.is_empty() { 1 } else { visible_width(glyph) };

            let sanitized_msg = super::sanitize_for_terminal_multiline(&suggestion.message);
            let first_prefix = visible_width(self.bc.v)
                + indent.len()
                + visible_width(self.bc.sub_v)
                + 1
                + glyph_rendered_width
                + label.len()
                + 2;
            let wrapped = self.wrap_msg(&sanitized_msg, first_prefix);
            let mut wrapped_iter = wrapped.iter();
            let first_wrapped = wrapped_iter.next().unwrap_or(&std::borrow::Cow::Borrowed(""));

            // Blank gutter separator line (no line number — matches the
            // label-connector convention of not carrying a number on
            // non-source rows).
            let _ = writeln!(
                out,
                "{dim}{v}{reset}{dim}{blank_sep}{reset} {dim}{sub_v}{reset}",
                dim = self.s.dim,
                v = self.bc.v,
                reset = self.s.reset,
                sub_v = self.bc.sub_v,
                blank_sep = " ".repeat(line_num_width),
            );
            // Message line: `│ 12  √ help: message`
            // Gutter: `│` + `12` (line_num_width) + ` ` + `│` = 4 visible
            // columns when width is 2; the `+`/`-` rows below share the
            // same gutter so the two diff rows align.
            let _ = writeln!(
                out,
                "{dim}{v}{reset}{dim}{line_no:>width$}{reset} {dim}{sub_v}{reset} {glyph_pad}{label_color}{label}:{reset} {first_wrapped}",
                dim = self.s.dim,
                v = self.bc.v,
                reset = self.s.reset,
                sub_v = self.bc.sub_v,
                line_no = sug_first + 1,
                width = line_num_width,
                glyph_pad = glyph_pad,
                label_color = label_color,
                label = label,
                first_wrapped = first_wrapped,
            );
            for line in wrapped_iter {
                let _ = writeln!(
                    out,
                    "{dim}{v}{reset}{dim}{blank}{reset} {dim}{sub_v}{reset} {glyph_pad}{line}",
                    dim = self.s.dim,
                    v = self.bc.v,
                    reset = self.s.reset,
                    sub_v = self.bc.sub_v,
                    blank = " ".repeat(line_num_width),
                    glyph_pad = glyph_pad,
                    line = line,
                );
            }

            // `-` diff line: line number + old source text (sanitized).
            let _ = writeln!(
                out,
                "{dim}{v}{reset}{dim}{line_no:>width$}{reset} {dim}{sub_v}{reset} {glyph_pad}{del}- {old_line}{reset}",
                dim = self.s.dim,
                v = self.bc.v,
                reset = self.s.reset,
                sub_v = self.bc.sub_v,
                line_no = sug_first + 1,
                width = line_num_width,
                glyph_pad = glyph_pad,
                del = self.s.diff_del,
                old_line = old_display,
            );
            // `+` diff line: line number + new source text (sanitized).
            // Carries the same line number as the `-` line (an inline
            // replacement, not an insertion), so the two rows share the
            // exact gutter prefix: `│` + line_num_width + ` ` + `│` = 4
            // visible columns when width is 2.  No stray leading space.
            let _ = writeln!(
                out,
                "{dim}{v}{reset}{dim}{line_no:>width$}{reset} {dim}{sub_v}{reset} {glyph_pad}{add}+ {new_line}{reset}",
                dim = self.s.dim,
                v = self.bc.v,
                reset = self.s.reset,
                sub_v = self.bc.sub_v,
                line_no = sug_first + 1,
                width = line_num_width,
                glyph_pad = glyph_pad,
                add = self.s.diff_add,
                new_line = new_display,
            );

            suggestion_rendered[i] = true;
        }

        if !self.compact {
            let _ = writeln!(
                out,
                "{dim}{v}{indent}{sub_bl}{sub_h}{reset}",
                dim = self.s.dim,
                v = self.bc.v,
                indent = indent,
                sub_bl = self.bc.sub_bl,
                sub_h = self.bc.sub_h,
                reset = self.s.reset,
            );
        }
    }

    // ── Children ─────────────────────────────────────────────────

    /// Render the portion of `styled` corresponding to `seg` — one wrapped
    /// segment of the plain concatenation `plain`.  `search_from` is the
    /// byte offset in `plain` to start locating `seg`; returns the styled
    /// fragment and the offset just past it.  Highlighted parts never
    /// straddle a break unless they contain a space (type names such as
    /// `Int<32>` do not), so slicing at the segment boundary is exact.
    /// `parts` must be the styled parts *after* the same sanitization that
    /// produced `plain` (see `write_child`): the offsets below are byte
    /// offsets into their concatenation, so unsanitized parts would slice
    /// mid-character once a control char was replaced by U+FFFD.
    ///
    /// The fallback (whitespace normalization broke the plain↔styled
    /// mapping) borrows `seg` rather than copying it: the only consumer is
    /// a `Display` sink, so the old `seg.to_string()` was an unconditional
    /// heap allocation on a path that has nothing to style.
    fn styled_segment<'a>(
        &self,
        parts: &[crate::diagnostics::StringPart<'_>],
        plain: &str,
        seg: &'a str,
        search_from: usize,
    ) -> (Cow<'a, str>, usize) {
        let Some(rel) = plain.get(search_from..).and_then(|rest| rest.find(seg)) else {
            // Whitespace normalization broke the mapping: emit unstyled.
            return (Cow::Borrowed(seg), search_from);
        };
        let start = search_from + rel;
        let end = start + seg.len();
        let mut rendered = String::with_capacity(seg.len() + 16);
        let mut pos = 0usize;
        for part in parts {
            let pstart = pos;
            let pend = pos + part.content.len();
            pos = pend;
            let os = pstart.max(start);
            let oe = pend.min(end);
            if os >= oe {
                continue;
            }
            let slice = &part.content[os - pstart..oe - pstart];
            match part.style {
                crate::diagnostics::StringPartStyle::Normal => rendered.push_str(slice),
                crate::diagnostics::StringPartStyle::Highlighted => {
                    // Highlighted parts in bright blue (matching type name
                    // color) — three push_str calls instead of a `format!`
                    // whose result is immediately copied again.
                    rendered.push_str(self.s.help);
                    rendered.push_str(slice);
                    rendered.push_str(self.s.reset);
                }
            }
        }
        (Cow::Owned(rendered), end)
    }

    fn write_child(&self, out: &mut String, child: &crate::diagnostics::Subdiag) {
        let prefix: Cow<'_, str> = child.level.label();
        let prefix_cols = self.prefix_cols(child.level);
        // Wrap the plain message; when a styled version is present, map each
        // wrapped segment back onto its parts so highlights survive per line.
        // `child.message` already holds the plain text (set at construction
        // time via `styled.to_plain()`), so no additional allocation here.
        // User text written between SGR runs: sanitize before wrapping and
        // before styled mapping.  A control char becomes U+FFFD, which
        // changes *byte* lengths (+2 per C0/DEL, +1 per C1), so the styled
        // parts are sanitized in lockstep whenever the plain text changed —
        // otherwise the `search_from` byte offsets into their concatenation
        // drift and `styled_segment` can slice mid-character.
        let sanitized = super::sanitize_for_terminal_multiline(child.message.as_ref());
        let plain = sanitized.as_ref();
        // `plain` is the sanitized concatenation of the raw parts, and
        // replacement strictly grows the text, so equal lengths mean the
        // sanitization was a no-op and the raw parts can be used as-is.
        let sanitized_parts: Vec<crate::diagnostics::StringPart<'_>>;
        let styled_parts: Option<&[crate::diagnostics::StringPart<'_>]> = match child.styled_message
        {
            Some(ref styled) => {
                let total: usize = styled.0.iter().map(|p| p.content.len()).sum();
                if total == plain.len() {
                    Some(&styled.0)
                } else {
                    sanitized_parts = styled
                        .0
                        .iter()
                        .map(|p| crate::diagnostics::StringPart {
                            content: super::sanitize_for_terminal_multiline(&p.content),
                            style: p.style,
                        })
                        .collect();
                    Some(&sanitized_parts)
                }
            }
            None => None,
        };
        let mut search_from = 0usize;
        let cont_pad: String = " ".repeat(prefix_cols - 1); // `│` (1 col) + this pad = prefix_cols
        for (i, seg) in self.wrap_msg(plain, prefix_cols).iter().enumerate() {
            // Keep the wrapped segment as a `Cow`: the unstyled path (no
            // `styled_message`) then borrows straight through, and the styled
            // fallback inside `styled_segment` borrows too — neither needs the
            // old `into_owned()`/`to_string()` heap allocation.
            let msg: Cow<'_, str> = match styled_parts {
                Some(parts) => {
                    let (rendered, next) =
                        self.styled_segment(parts, plain, seg.as_ref(), search_from);
                    search_from = next;
                    rendered
                }
                None => seg.to_owned(),
            };
            if i == 0 {
                let _ = writeln!(
                    out,
                    "{v}  {prefix_color}= {bold}{prefix}{reset}: {msg}",
                    v = self.bc.v,
                    prefix_color = self.s.get(Self::level_style(child.level)),
                    bold = self.s.bold,
                    prefix = prefix,
                    reset = self.s.reset,
                    msg = msg,
                );
            } else {
                // Continuation: `│` + pad aligning the segment to column
                // `prefix_cols` (where the first line's message started).
                let _ = writeln!(out, "{v}{pad}{msg}", v = self.bc.v, pad = cont_pad, msg = msg);
            }
        }
    }

    fn write_note_line(&self, out: &mut String, msg: &str) {
        let prefix_cols = self.prefix_cols(DiagnosticLevel::Note);
        let sanitized = super::sanitize_for_terminal_multiline(msg);
        let cont_pad: String = " ".repeat(prefix_cols - 1);
        let level_label = DiagnosticLevel::Note.label();
        for (i, seg) in self.wrap_msg(&sanitized, prefix_cols).iter().enumerate() {
            if i == 0 {
                let _ = writeln!(
                    out,
                    "{v}  {cyan}= {bold}{level_label}{reset}: {green}{msg}{reset}",
                    v = self.bc.v,
                    cyan = self.s.help,
                    bold = self.s.bold,
                    reset = self.s.reset,
                    green = self.s.note,
                    msg = seg,
                );
            } else {
                let _ = writeln!(out, "{v}{pad}{msg}", v = self.bc.v, pad = cont_pad, msg = seg);
            }
        }
    }

    fn write_help_line(&self, out: &mut String, msg: &str) {
        let prefix_cols = self.prefix_cols(DiagnosticLevel::Help);
        let sanitized = super::sanitize_for_terminal_multiline(msg);
        let cont_pad: String = " ".repeat(prefix_cols - 1);
        let level_label = DiagnosticLevel::Help.label();
        for (i, seg) in self.wrap_msg(&sanitized, prefix_cols).iter().enumerate() {
            if i == 0 {
                let _ = writeln!(
                    out,
                    "{v}  {cyan}= {bold}{level_label}{reset}: {cyan}{msg}{reset}",
                    v = self.bc.v,
                    cyan = self.s.help,
                    bold = self.s.bold,
                    reset = self.s.reset,
                    msg = seg,
                );
            } else {
                let _ = writeln!(out, "{v}{pad}{msg}", v = self.bc.v, pad = cont_pad, msg = seg);
            }
        }
    }
}

// ── Helper: get the 0-based line number for a span's start ───────

/// Returns the 0‑based line index of the start of a span.  The offset is
/// floored to a char boundary so a span landing inside a multi-byte
/// character cannot panic the slice.
pub(crate) fn span_line(span: Span, source: &str) -> usize {
    let clamped = crate::diagnostics::floor_char_boundary(source, span.start_as_usize());
    memchr::memchr_iter(b'\n', source[..clamped].as_bytes()).count()
}

// ── Helper: sliding window over a section's labels ─────────────

/// Sliding-window selector over a section's labels, driven by the
/// monotonically advancing rendered line index.
///
/// The old per-row scan used to test every label against every row
/// (O(lines × labels)); here the labels are sorted once by `start_line`, and
/// as the row index advances each label enters the window at its start line
/// and leaves it after its end line, so a row only touches the labels that
/// can actually cover it — O(labels·log labels + lines + active).
///
/// `active` is kept in **original label order**, so the per-row output keeps
/// the exact ordering it had before the sort (message/connector order is
/// observable and snapshot-pinned).
///
/// `label_lines` is the per-label `(start_line, end_line)` slice (indexed
/// exactly like the label slice) and must outlive the window.
pub(crate) struct LabelWindow<'a> {
    label_lines: &'a [(usize, usize)],
    /// Label indices sorted by `start_line` (stable: ties keep original order).
    order: Vec<usize>,
    /// Labels covering the current row, ordered by original index.
    active: Vec<usize>,
    /// Index into `order` of the first label not yet folded in.
    cursor: usize,
}

impl<'a> LabelWindow<'a> {
    /// Build the start-line-sorted order for `label_lines`.
    pub(crate) fn new(label_lines: &'a [(usize, usize)]) -> Self {
        let mut order: Vec<usize> = (0..label_lines.len()).collect();
        order.sort_by_key(|&i| label_lines[i].0);
        Self {
            label_lines,
            order,
            active: Vec::new(),
            cursor: 0,
        }
    }

    /// Fold `line_idx` into the window. Rows must be visited in ascending
    /// order (the renderers iterate `first_line..=last_line`).
    pub(crate) fn advance(&mut self, line_idx: usize) {
        let lines = self.label_lines;
        while self.cursor < self.order.len() {
            let i = self.order[self.cursor];
            if lines[i].0 > line_idx {
                break;
            }
            // Keep `active` sorted by original index so the emitted order is
            // independent of the sort by start line.
            let pos = self.active.partition_point(|&a| a < i);
            self.active.insert(pos, i);
            self.cursor += 1;
        }
        // Drop labels whose range ended before this row. After the
        // `span_lines` end-attribution fix, `lines[i].1` is the last line
        // containing a covered byte, so anything that can still overlap this
        // row is retained; the exact per-annotation intersection below stays
        // as a belt-and-braces guard for out-of-source spans.
        self.active.retain(|&i| lines[i].1 >= line_idx);
    }

    /// Label indices covering the current row, in original order.
    pub(crate) fn active(&self) -> &[usize] {
        &self.active
    }
}

// ── Helper: compute underlines for a single line ───────────────

/// Returns a list of `(display_col, display_width, underline_char, message)`
/// for the primary span plus every label that covers the given line index.
/// `display_col` and `display_width` are measured in terminal cells (tabs
/// expanded, CJK/emoji = 2). Results are written into `out` which is cleared
/// before use so callers can reuse it as a scratch buffer across iterations.
/// `ranges_buf`/`meta_buf`/`display_buf`/`order_buf`/`done_buf`/`active_buf`
/// are internal scratch of the same kind — pass buffers that live across the
/// caller's row loop so the per-line walk allocates nothing.
///
/// `window` is advanced to `line_idx` here and supplies only the labels whose
/// line range can cover this row (in original order), so the per-row cost is
/// O(active) rather than O(labels); callers must drive rows in ascending
/// order and pass the same `label_lines` the window was built from.
///
/// `primary` is the (per-section) span that carries the `^` underline; it is
/// taken by value and iterated ahead of the labels so callers never have to
/// materialize a combined `Vec<Label>` (which would clone every label once
/// per source section).
/// Build the byte ranges and metadata for the annotations covering
/// `line_idx`: the per-section `primary` first, then the labels whose line
/// range can cover the row (in original order), each intersected with the
/// line.  `ranges_buf` and `meta_buf` are cleared and pushed in lockstep, so
/// index `i` in either always refers to the same annotation.  Pure
/// span-to-line intersection — no grapheme walk — shared by the glyph and
/// HTML renderers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_line_annotation_ranges<'a, 'src>(
    primary: Span,
    labels: &[&'a Label<'src>],
    window: &mut LabelWindow<'_>,
    source: &str,
    line: &str,
    line_idx: usize,
    line_starts: &[usize],
    ranges_buf: &mut Vec<(usize, usize)>,
    meta_buf: &mut Vec<(char, &'a str, Option<usize>)>,
) {
    ranges_buf.clear();
    meta_buf.clear();

    let line_start = line_starts
        .get(line_idx)
        .copied()
        .unwrap_or_else(|| source.len());
    let line_end = line_start + line.len();

    // Narrow the candidate labels to those whose line range covers this row.
    window.advance(line_idx);

    // The synthetic primary's message is empty (`""`), matching the old
    // `Label { message: String::new(), kind: Primary, .. }` placeholder.
    let primary_ann = (primary, AnnotationKind::Primary.underline_char(), "", None);
    let label_anns = window.active().iter().map(|&i| {
        let l = labels[i];
        (l.span, l.underline_char(), l.message.as_ref(), Some(i))
    });
    for (span, underline_char, msg, label_idx) in std::iter::once(primary_ann).chain(label_anns) {
        let span_start = span.start_as_usize();
        let span_end = span.end_as_usize();

        // Does this annotation's span overlap this line? A zero-width
        // annotation (an insertion point) belongs to the line whose
        // `[line_start, line_end]` range contains the point — including
        // both ends of an empty line.
        let intersects = if span_start == span_end {
            span_start >= line_start && span_start <= line_end
        } else {
            span_end > line_start && span_start < line_end
        };
        if !intersects {
            continue;
        }

        let start_offset = if span_start > line_start {
            span_start - line_start
        } else {
            0
        };
        let end_offset = if span_end < line_end {
            span_end - line_start
        } else {
            line.len()
        };
        ranges_buf.push((start_offset, end_offset));
        meta_buf.push((underline_char, msg, label_idx));
    }
}

/// Merge the `^` intervals from `ranges`/`meta` for one line.
///
/// The intervals are clipped to the line and sorted, and adjacent or
/// overlapping runs are fused, so highlighting can sweep a single monotone
/// cursor instead of scanning every interval per byte or grapheme (which
/// made lines with several annotations O(positions × intervals)).
///
/// Ranges are clipped to `len` (clip **after** filtering on the original
/// `start`, so a range beginning past the line is dropped as effectively
/// empty).  Zero-width ranges are deliberately **kept**: a zero-width range
/// strictly inside a multi-byte grapheme still has to light up that whole
/// cluster, and the byte-overlap formula (`gend > start && byte_idx < end`)
/// reproduces the expansion for `start < gend`.
fn merged_primary_ranges<'a>(
    ranges: &[(usize, usize)],
    meta: &[(char, &'a str, Option<usize>)],
    len: usize,
) -> SmallVec<[(usize, usize); 4]> {
    let mut sorted: SmallVec<[(usize, usize); 4]> = ranges
        .iter()
        .enumerate()
        .filter(|&(i, _)| meta[i].0 == '^')
        .filter(|&(_, &(start, _))| start <= len)
        .map(|(_, &(start, end))| (start, end.min(len)))
        .collect();
    sorted.sort_unstable_by_key(|&(start, end)| (start, end));
    let mut merged: SmallVec<[(usize, usize); 4]> = SmallVec::new();
    for &(start, end) in sorted.iter() {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

/// Advance `cursor` past every interval ending at or before byte position
/// `i`, then report whether the cursor's interval still covers `i`.
///
/// Requires monotonically non-decreasing `i` across calls (the render loops
/// walk left to right) and merged, start-sorted intervals.
#[inline]
fn primary_at(intervals: &[(usize, usize)], cursor: &mut usize, i: usize) -> bool {
    while let Some(&(_, end)) = intervals.get(*cursor) {
        if end > i {
            break;
        }
        *cursor += 1;
    }
    intervals.get(*cursor).is_some_and(|&(start, _)| start <= i)
}

/// Single-pass per-line render: converts byte ranges to display columns,
/// accumulates the line's total display width, and emits the source line
/// with `^`-span highlighting — all in one grapheme walk.
///
/// Replaces the three separate grapheme walks
/// (`byte_ranges_to_display_into`, `line_display_width`, and the source-line
/// print loop) that `write_source_section` used to run per rendered row.
///
/// Fills `out` with `(start_col, width, underline_char, message, label_idx)`
/// tuples in the same order as `ranges`/`meta`.  Returns the line's total
/// display width in terminal cells.
///
/// `emit` receives every chunk of sanitized source text (the highlight
/// escapes and the tab/control replacements around it are written directly
/// to `rendered`).  Terminal callers pass the identity, so plain text is
/// appended verbatim; the HTML caller passes an escaper.  The callback is
/// invoked once per contiguous same-highlight run on the ASCII fast path
/// and once per grapheme otherwise, never per byte.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_line_merged_with<'a, F: FnMut(&mut String, &str)>(
    line: &str,
    ranges: &[(usize, usize)],
    meta: &[(char, &'a str, Option<usize>)],
    out: &mut Vec<(usize, usize, char, &'a str, Option<usize>)>,
    rendered: &mut String,
    highlight_on: &str,
    highlight_off: &str,
    display_buf: &mut Vec<(usize, usize)>,
    order_buf: &mut Vec<usize>,
    done_buf: &mut Vec<bool>,
    active_buf: &mut BinaryHeap<Reverse<(usize, usize)>>,
    tab_width: usize,
    mut emit: F,
) -> usize {
    rendered.clear();
    display_buf.clear();
    out.clear();

    let len = line.len();

    // Pre-filter and fuse `^` ranges so the per-position highlight check is
    // one cursor sweep.  The byte-overlap formula (`gend > b1 && byte_idx <
    // b2`) covers *all* the cases the display-column check did — including
    // zero-width ranges strictly inside a multi-byte grapheme, which
    // `byte_range_to_display` expands to the whole cluster.
    let primary_ranges = merged_primary_ranges(ranges, meta, len);

    // ── ASCII fast path: byte offset ≡ display column ──
    if line.bytes().all(|b| (0x20..0x7F).contains(&b)) {
        display_buf.reserve(ranges.len());
        for &(start, end) in ranges {
            let end_c = end.min(len);
            display_buf.push(if end > start && start < len {
                (start, end_c - start)
            } else {
                (end_c, 0)
            });
        }

        let mut in_primary = false;
        let mut primary_cursor = 0usize;
        let mut i = 0usize;
        while i < len {
            let is_primary = primary_at(&primary_ranges, &mut primary_cursor, i);
            let start = i;
            i += 1;
            while i < len {
                let ip = primary_at(&primary_ranges, &mut primary_cursor, i);
                if ip != is_primary {
                    break;
                }
                i += 1;
            }
            if is_primary != in_primary {
                rendered.push_str(if is_primary {
                    highlight_on
                } else {
                    highlight_off
                });
                in_primary = is_primary;
            }
            emit(rendered, &line[start..i]);
        }
        if in_primary {
            rendered.push_str(highlight_off);
        }

        out.reserve(display_buf.len());
        for (&(col, width), &(ch, msg, label_idx)) in display_buf.iter().zip(meta.iter()) {
            // Insertion points (`width == 0`) get a one-cell marker.
            out.push((col, width.max(1), ch, msg, label_idx));
        }
        return len;
    }

    // ── Non-ASCII: single grapheme walk doing all three jobs ──
    order_buf.clear();
    order_buf.extend(0..ranges.len());
    order_buf.sort_by_key(|&i| (ranges[i].0, ranges[i].1));

    display_buf.resize(ranges.len(), (0, 0));
    done_buf.clear();
    done_buf.resize(ranges.len(), false);
    active_buf.clear();

    let mut col = 0usize;
    let mut byte_idx = 0usize;
    let mut next = 0usize;
    let mut in_primary = false;
    let mut primary_cursor = 0usize;

    for g in line.graphemes(true) {
        let glen = g.len();
        let gend = byte_idx + glen;

        // (1) Close active ranges ending at or before this grapheme's start.
        // The heap is keyed by range end, so the earliest-ending active range
        // is always on top: expiring is a peek + pops, not a linear scan.
        while let Some(&Reverse((end, i))) = active_buf.peek() {
            if end > byte_idx {
                break;
            }
            active_buf.pop();
            display_buf[i].1 = col;
            done_buf[i] = true;
        }

        // (2) Activate ranges starting before this grapheme's end.
        while next < order_buf.len() && ranges[order_buf[next]].0 < gend {
            let i = order_buf[next];
            next += 1;
            if ranges[i].1 <= byte_idx {
                display_buf[i] = (col, col);
                done_buf[i] = true;
            } else {
                display_buf[i].0 = col;
                active_buf.push(Reverse((ranges[i].1, i)));
            }
        }

        // (3) Primary highlight — byte overlap via the merged cursor: skip
        // intervals that ended at or before this grapheme, then test the
        // first one that can still overlap.
        while let Some(&(_, end)) = primary_ranges.get(primary_cursor) {
            if end > byte_idx {
                break;
            }
            primary_cursor += 1;
        }
        let is_primary = primary_ranges
            .get(primary_cursor)
            .is_some_and(|&(start, end)| gend > start && byte_idx < end);
        if is_primary != in_primary {
            rendered.push_str(if is_primary {
                highlight_on
            } else {
                highlight_off
            });
            in_primary = is_primary;
        }

        // (4) Append the sanitized grapheme.
        if g == "\t" {
            let w = grapheme_width(g, col, tab_width);
            for _ in 0..w {
                rendered.push(' ');
            }
        } else if g.chars().any(|c| c.is_control()) {
            // Source rows are untrusted text written between SGR runs: a
            // control grapheme becomes U+FFFD (tab was expanded above) and
            // the highlight escapes below are added by this function only.
            rendered.push('\u{FFFD}');
        } else if g.chars().all(|c| super::is_bidi_override(c)) {
            // Bidi formatting characters (U+202A–U+202E, U+2066–U+2069) are
            // invisible but reorder surrounding display text — a path like
            // `evil.cod\u{202E}exe` can spoof the extension.  Remove them
            // outright (like `sanitize_filename` does) instead of replacing
            // with U+FFFD, which would inject noise.  They are zero-width,
            // so `col` does not advance and the column arithmetic stays
            // correct.
        } else {
            emit(rendered, g);
        }

        col += grapheme_width(g, col, tab_width);
        byte_idx = gend;
    }
    if in_primary {
        rendered.push_str(highlight_off);
    }

    // Close remaining active ranges.
    for &Reverse((_, i)) in active_buf.iter() {
        display_buf[i].1 = col;
        done_buf[i] = true;
    }
    // Ranges that never activated: (total_col, 0).
    for (i, d) in done_buf.iter().enumerate() {
        if !d {
            display_buf[i] = (col, col);
        }
    }
    // (start_col, end_col) → (start_col, width).
    for (start, end) in display_buf.iter_mut() {
        *end = end.saturating_sub(*start);
    }

    out.reserve(display_buf.len());
    for (&(col, width), &(ch, msg, label_idx)) in display_buf.iter().zip(meta.iter()) {
        // Insertion points (`width == 0`) get a one-cell marker.
        out.push((col, width.max(1), ch, msg, label_idx));
    }

    col
}

/// Terminal-facing wrapper over [`render_line_merged_with`]: source text is
/// appended verbatim (the terminal escape codes a row needs are the highlight
/// runs the core writes itself, nothing else).
#[allow(clippy::too_many_arguments)]
fn render_line_merged<'a>(
    line: &str,
    ranges: &[(usize, usize)],
    meta: &[(char, &'a str, Option<usize>)],
    out: &mut Vec<(usize, usize, char, &'a str, Option<usize>)>,
    rendered: &mut String,
    highlight_on: &str,
    highlight_off: &str,
    display_buf: &mut Vec<(usize, usize)>,
    order_buf: &mut Vec<usize>,
    done_buf: &mut Vec<bool>,
    active_buf: &mut BinaryHeap<Reverse<(usize, usize)>>,
    tab_width: usize,
) -> usize {
    render_line_merged_with(
        line,
        ranges,
        meta,
        out,
        rendered,
        highlight_on,
        highlight_off,
        display_buf,
        order_buf,
        done_buf,
        active_buf,
        tab_width,
        |out, s| out.push_str(s),
    )
}

/// Fused replacement for the `compute_line_underlines` + `line_display_width`
/// + source-print trio: folds this row's annotation spans into byte ranges,
/// then resolves, measures, and renders them in one grapheme walk.  Returns
/// the line's total display width in terminal cells.
#[allow(clippy::too_many_arguments)]
fn compute_line_underlines_and_render<'a, 'src>(
    primary: Span,
    labels: &[&'a Label<'src>],
    window: &mut LabelWindow<'_>,
    source: &str,
    line: &str,
    line_idx: usize,
    line_starts: &[usize],
    out: &mut Vec<(usize, usize, char, &'a str, Option<usize>)>,
    rendered: &mut String,
    highlight_on: &str,
    highlight_off: &str,
    ranges_buf: &mut Vec<(usize, usize)>,
    meta_buf: &mut Vec<(char, &'a str, Option<usize>)>,
    display_buf: &mut Vec<(usize, usize)>,
    order_buf: &mut Vec<usize>,
    done_buf: &mut Vec<bool>,
    active_buf: &mut BinaryHeap<Reverse<(usize, usize)>>,
    tab_width: usize,
) -> usize {
    out.clear();
    compute_line_annotation_ranges(
        primary,
        labels,
        window,
        source,
        line,
        line_idx,
        line_starts,
        ranges_buf,
        meta_buf,
    );
    render_line_merged(
        line,
        ranges_buf,
        meta_buf,
        out,
        rendered,
        highlight_on,
        highlight_off,
        display_buf,
        order_buf,
        done_buf,
        active_buf,
        tab_width,
    )
}

/// Find the byte offset of the start of a given line (0-based).
pub(crate) fn line_start_byte(source: &str, line_idx: usize) -> usize {
    let mut offset = 0;
    for _ in 0..line_idx {
        if let Some(pos) = source[offset..].find('\n') {
            offset += pos + 1;
        } else {
            break;
        }
    }
    offset
}

/// Percent-encode the bytes that are unsafe in a `file://` URL path.
///
/// Keeps unreserved characters plus `/`, `:`, `-`, `_`, `.`, `~`; encodes
/// everything else (including `\x1b`, `\x07`, space, `#`, `?`, `%`).  This
/// keeps a filename from terminating the enclosing OSC 8 hyperlink early or
/// injecting terminal control codes.
pub(crate) fn encode_file_url_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b':' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                let _ = write!(out, "%{:02X}", b);
            }
        }
    }
    out
}

/// Precompute the byte offset of the start of every line in one O(n) pass.
///
/// Callers that render several lines should build this once and index it,
/// instead of re-scanning from the start of the file for each line (which
/// made source rendering O(lines × file_size)).
///
/// Thin alias for [`LineIndex::new`], which owns the scan; kept because
/// the line-start shape is the only part this helper's callers need.
pub(crate) fn line_starts(source: &str) -> Vec<usize> {
    LineIndex::new(source).starts().to_vec()
}

/// Fill `primary_cols` (resized to `display_width`) with `true` for every
/// display column covered by a `^` underline.  Used only by the tests as the
/// oracle for the fused renderer's byte-overlap highlight rule.
#[cfg(test)]
pub(crate) fn compute_primary_cols(
    underlines: &[(usize, usize, char, &str, Option<usize>)],
    display_width: usize,
    primary_cols: &mut Vec<bool>,
) {
    primary_cols.clear();
    primary_cols.resize(display_width, false);
    for (col, ulen, underline_char, _msg, _label_idx) in underlines {
        if *underline_char == '^' {
            let end = (*col + *ulen).min(display_width);
            for i in *col..end {
                primary_cols[i] = true;
            }
        }
    }
}

/// Non-overlapping underline lanes for one source row.
///
/// Built by [`assign`](Self::assign), which colours the row's annotations
/// as an interval graph.  Interval graphs are perfect (χ = ω), so the
/// greedy sweep — sort by `(start, end)`, reuse the lane whose end is
/// smallest and already ≤ `start`, otherwise open a new lane — uses
/// exactly the maximum overlap depth.  No colouring uses fewer.
///
/// Intervals are half-open `[col, col + ulen)`: annotations touching at a
/// boundary do not overlap and may share a lane.
///
/// A lane owns one output row; callers render lane `0..count` in order.
/// The struct owns the sweep scratch (heap and order vector) so a row loop
/// reuses the allocations across lines.
pub(crate) struct UnderlineLanes {
    /// `lanes[i]` = indices into the row's `underlines` on lane `i`.
    lanes: Vec<Vec<usize>>,
    /// Min-heap of `(end_col, lane)` driving the greedy sweep.
    active: BinaryHeap<Reverse<(usize, usize)>>,
    /// Start-sorted index scratch.
    order: Vec<usize>,
    /// Lanes used by the last `assign`.
    count: usize,
}

impl UnderlineLanes {
    pub(crate) fn new() -> Self {
        Self {
            lanes: Vec::new(),
            active: BinaryHeap::new(),
            order: Vec::new(),
            count: 0,
        }
    }

    /// Colour `underlines` into lanes and return the number of lanes used.
    ///
    /// Identical `(start, end, char)` annotations collapse to one lane
    /// entry before the sweep: overlapping spans of the same underline
    /// character already render as a single fused row (contiguous runs are
    /// emitted back-to-back), so duplicates would only add a redundant
    /// stacked lane.
    pub(crate) fn assign<'a>(
        &mut self,
        underlines: &[(usize, usize, char, &'a str, Option<usize>)],
    ) -> usize {
        let Self {
            lanes,
            active,
            order,
            count,
        } = self;
        active.clear();
        for lane in lanes.iter_mut() {
            lane.clear();
        }
        if underlines.is_empty() {
            *count = 0;
            return 0;
        }
        order.clear();
        order.extend(0..underlines.len());
        // Stable, so equal `(start, end)` keep their original order.
        order.sort_by_key(|&i| {
            let (col, ulen, ..) = underlines[i];
            (col, col + ulen)
        });

        let mut n = 0usize;
        // Collapse duplicate annotations to a single lane entry. `order` is
        // sorted by `(start, end)` (stable), so entries sharing
        // `(start, ulen, ch)` are adjacent in `order` order; the later
        // duplicates would only stack identical underline rows. The entry
        // stays in `underlines`, so its message (if any) is still emitted by
        // the caller's separate message loop.
        let mut prev_key: Option<(usize, usize, char)> = None;
        for &i in order.iter() {
            let (col, ulen, ch, ..) = underlines[i];
            let start = col;
            let end = col + ulen;
            let key = (start, ulen, ch);
            if prev_key == Some(key) {
                continue;
            }
            prev_key = Some(key);
            let lane = match active.peek() {
                // Earliest-ending lane already past `start` → reuse it.
                Some(&Reverse((lane_end, lane_idx))) if lane_end <= start => {
                    active.pop();
                    lane_idx
                }
                // Every lane still covers `start` → open a new one.
                _ => {
                    let idx = n;
                    n += 1;
                    idx
                }
            };
            if lane >= lanes.len() {
                lanes.push(Vec::new());
            }
            lanes[lane].push(i);
            active.push(Reverse((end, lane)));
        }
        *count = n;
        n
    }

    /// Render lane `index` into `out` as a trimmed, leading-padded
    /// underline row, applying `color_for` to every emitted character and
    /// `reset` between colour runs.  Returns `false` when the lane is
    /// blank.  Call only after [`assign`](Self::assign).
    ///
    /// Annotations within a lane never overlap and arrive in ascending
    /// column order, so the row is built directly from the lane's intervals
    /// (gaps become padding) instead of materialising a full-width column
    /// buffer per lane per row.  Zero-width annotations arrive already
    /// promoted to one-cell markers (see [`render_line_merged_with`]), so an
    /// insertion point at the end of the line renders one cell past the
    /// last character.
    ///
    /// Padding cells inherit the colour of the LAST emitted annotation char
    /// (or `color_for(' ')` before the first one), so a lane's runs read as
    /// one contiguous row rather than alternating colour gaps.
    pub(crate) fn row<'a, F: FnMut(char) -> &'a str>(
        &mut self,
        underlines: &[(usize, usize, char, &'a str, Option<usize>)],
        index: usize,
        display_width: usize,
        out: &mut String,
        mut color_for: F,
        reset: &'a str,
    ) -> bool {
        let lane = &self.lanes[index];
        out.clear();
        // First column not yet emitted.
        let mut cursor = 0usize;
        // `' '` is a real underline character (help-level annotations), so
        // blankness must be tracked separately from the buffer contents.
        let mut visible = false;
        // Colour of the last emitted annotation char; pads inherit it.
        let mut pad_color: &'a str = color_for(' ');
        // One cell past the line end is reachable: an insertion-point marker
        // at the end of the line sits there.
        let limit = display_width + 1;
        for &i in lane {
            let (col, ulen, ch, ..) = underlines[i];
            let end = (col + ulen).min(limit);
            if end <= cursor {
                continue;
            }
            let start = col.min(limit).max(cursor);
            if start >= end {
                continue;
            }
            let color = color_for(ch);
            if start > cursor {
                for _ in cursor..start {
                    out.push_str(color);
                    out.push(' ');
                    out.push_str(reset);
                }
            }
            for _ in start..end {
                out.push_str(color);
                out.push(ch);
                out.push_str(reset);
            }
            if ch != ' ' {
                visible = true;
            }
            pad_color = color;
            cursor = end;
        }
        if !visible {
            out.clear();
        }
        visible
    }
}

// ── Helper: display width & column conversion ──────────────────

/// Default tab stop width when no explicit [`DiagOpts::tab_width`] is
/// provided (tests, internal helpers).
pub(crate) const DEFAULT_TAB_WIDTH: usize = 4;

/// Display width of a grapheme cluster at the given (zero-based) display
/// column.  Tabs expand to the next `tab_width`-column tab stop; single-byte
/// ASCII clusters are resolved inline; every other cluster is memoised and, on
/// a miss, delegates to `unicode-width` (0.2), which handles ZWJ / presentation
/// sequence / composition rules correctly at the cluster level.  A cluster
/// containing a C0/DEL control measures one cell — the source-row printer
/// replaces it with a single U+FFFD, and measurement must stay aligned with
/// what is printed.
///
/// The memo exists because a render walks each source line several times
/// (the glyph renderer fuses those walks into one pass, but the HTML
/// renderer still walks the line once per job — `line_display_width`, the
/// underline sweep, then the highlight loop) and the same clusters — CJK,
/// emoji, combining sequences — recur across those walks and across
/// diagnostics.  It is deliberately scoped to *non-ASCII* clusters:
/// a printable ASCII byte's width is one range check, cheaper than any hash or
/// TLS access, so routing ASCII through the cache would only tax the common
/// case.
pub(crate) fn grapheme_width(g: &str, current_col: usize, tab_width: usize) -> usize {
    if g == "\t" {
        return tab_width - (current_col % tab_width);
    }
    // Fast path: an empty cluster is zero cells; any single-byte cluster is
    // one cell — printable ASCII directly, a C0 control / DEL via the U+FFFD
    // the printer substitutes for it (tab is handled above).  No cache, no
    // TLS, no `unicode-width` call.
    if g.len() <= 1 {
        return match g.as_bytes().first() {
            Some(_) => 1,
            None => 0,
        };
    }
    // Multi-byte cluster: the width depends only on `g` (tabs are the sole
    // column-dependent case and were handled above), so it can be memoised.
    if g.len() > WIDTH_CACHE_MAX_KEY {
        return grapheme_width_uncached(g);
    }
    WIDTH_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(w) = cache.get(g) {
            return w;
        }
        let w = grapheme_width_uncached(g);
        cache.put(g, w);
        w
    })
}

/// Uncached cluster width: `unicode-width` plus the RI-pair / ZWJ corrections.
///
/// Kept separate so the cache in [`grapheme_width`] can be differential-tested
/// against it (the two must agree on every non-tab cluster — controls
/// included, since both measure a control-containing cluster as the one cell
/// its U+FFFD replacement occupies).
#[inline]
fn grapheme_width_uncached(g: &str) -> usize {
    let mut w = g.width();
    // Regional Indicator pairs: unicode-width sums each RI (2 each),
    // but terminals render a flag (🇺🇸) as 2 cells — correct here.
    let is_ri = |c: char| ('\u{1F1E6}'..='\u{1F1FF}').contains(&c);
    let mut it = g.chars();
    if it.next().is_some_and(is_ri) && it.all(is_ri) {
        return 2;
    }
    // Non-RGI ZWJ joins (e.g. 👨‍👩): unicode-width already returns 2 for
    // well-formed RGI sequences; for malformed left-over spans cap at 2.
    if g.contains('\u{200D}') {
        w = w.min(2);
    }
    // A cluster containing a control char prints as a single U+FFFD (one
    // cell); `unicode-width` credits controls with zero, so floor the width.
    if g.chars().any(|c| c.is_control()) {
        w = w.max(1);
    }
    w
}

// ── Grapheme width cache ────────────────────────────────────────

/// Longest cluster (in UTF-8 bytes) the cache stores.  ZWJ family emoji top
/// out near 25 bytes; longer clusters are rare enough to compute uncached.
const WIDTH_CACHE_MAX_KEY: usize = 32;
/// `SETS × WAYS` entries, LRU within each set.  256 entries ≈ 12 KiB per
/// thread — ample for the small set of clusters that recur in one source file.
const WIDTH_CACHE_SETS: usize = 64;
const WIDTH_CACHE_WAYS: usize = 4;

/// One cache slot.  `len == 0` marks an empty slot; the key is stored inline
/// so a lookup never allocates.  `stamp` is the last-access tick that drives
/// per-set LRU eviction.
#[derive(Clone, Copy)]
struct WidthEntry {
    key: [u8; WIDTH_CACHE_MAX_KEY],
    len: u8,
    width: u8,
    stamp: u64,
}

impl WidthEntry {
    const EMPTY: Self = Self {
        key: [0; WIDTH_CACHE_MAX_KEY],
        len: 0,
        width: 0,
        stamp: 0,
    };

    #[inline]
    fn matches(&self, g: &str) -> bool {
        self.len as usize == g.len() && &self.key[..g.len()] == g.as_bytes()
    }
}

/// Thread-local, fixed-capacity, set-associative LRU of grapheme widths.
///
/// Only non-ASCII clusters reach it (see [`grapheme_width`]), so the ASCII hot
/// path pays nothing for it, and one thread renders a batch at a time, so no
/// locking is needed.
struct WidthCache {
    sets: [[WidthEntry; WIDTH_CACHE_WAYS]; WIDTH_CACHE_SETS],
    clock: u64,
}

impl WidthCache {
    const fn new() -> Self {
        Self {
            sets: [[WidthEntry::EMPTY; WIDTH_CACHE_WAYS]; WIDTH_CACHE_SETS],
            clock: 0,
        }
    }

    /// Look up `g`, refreshing its recency on a hit.
    #[inline]
    fn get(&mut self, g: &str) -> Option<usize> {
        self.clock += 1;
        let set = &mut self.sets[width_cache_hash(g) & (WIDTH_CACHE_SETS - 1)];
        for entry in set.iter_mut() {
            if entry.matches(g) {
                entry.stamp = self.clock;
                return Some(entry.width as usize);
            }
        }
        None
    }

    /// Insert `g` → `width`, evicting the least-recently-used way of its set.
    #[inline]
    fn put(&mut self, g: &str, width: usize) {
        // A single cluster is at most a couple of cells wide; anything larger
        // would not fit the byte slot, so skip rather than truncate.
        if width > u8::MAX as usize {
            return;
        }
        self.clock += 1;
        let set = &mut self.sets[width_cache_hash(g) & (WIDTH_CACHE_SETS - 1)];
        // Prefer an empty way; otherwise evict the least-recently-used one.
        let mut victim = 0;
        for (i, entry) in set.iter().enumerate() {
            if entry.len == 0 {
                victim = i;
                break;
            }
            if entry.stamp < set[victim].stamp {
                victim = i;
            }
        }
        let entry = &mut set[victim];
        entry.key[..g.len()].copy_from_slice(g.as_bytes());
        entry.len = g.len() as u8;
        entry.width = width as u8;
        entry.stamp = self.clock;
    }
}

/// FNV-1a over the cluster's bytes.  Cheap and dependency-free; the result
/// only needs to spread the (small) working set across the cache's sets.
#[inline]
fn width_cache_hash(g: &str) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in g.as_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h as usize
}

thread_local! {
    /// Per-thread grapheme-width memo.  `const`-initialised so the lookup
    /// path has no lazy-init branch.
    static WIDTH_CACHE: RefCell<WidthCache> = const { RefCell::new(WidthCache::new()) };
}

/// Single-char fallback for code that reasons per-character rather than
/// per-grapheme.  Rendering loops should prefer [`grapheme_width`] +
/// `graphemes(true)`.
#[inline]
pub(crate) fn char_width_at(c: char, current_col: usize, tab_width: usize) -> usize {
    if c == '\t' {
        tab_width - (current_col % tab_width)
    } else if c.is_control() {
        // Matches the printer: a control char becomes one U+FFFD cell.
        1
    } else {
        c.width().unwrap_or(0)
    }
}

/// Total display width of a line in terminal cells (tabs expanded).
///
/// Test-only: the production renders measure width inside the fused
/// [`render_line_merged_with`] walk.
#[cfg(test)]
pub(crate) fn line_display_width(line: &str, tab_width: usize) -> usize {
    // Fast path: every byte printable ASCII (0x20..=0x7E). No tabs, no
    // control chars, no multi-byte clusters — each byte contributes exactly
    // one cell (`grapheme_width`'s single-byte arm), so the byte length is
    // the display width and the grapheme iterator is never constructed.
    if line.bytes().all(|b| (0x20..0x7F).contains(&b)) {
        return line.len();
    }
    let mut w = 0usize;
    for g in line.graphemes(true) {
        w += grapheme_width(g, w, tab_width);
    }
    w
}

/// Visible display width of `s`, skipping CSI (`\x1b[…m`) and OSC
/// (`\x1b]…\x1b\\`) escape sequences.  Reuses [`grapheme_width`] for the
/// per-cluster arithmetic (tabs, CJK, ZWJ) already used everywhere else.
pub(crate) fn visible_width(s: &str) -> usize {
    visible_width_with_tab(s, DEFAULT_TAB_WIDTH)
}

pub(crate) fn visible_width_with_tab(s: &str, tab_width: usize) -> usize {
    // Fast path: every byte printable ASCII. ESC is 0x1b (a control char,
    // outside this range), so a string that reaches this branch contains no
    // CSI/OSC sequences to skip, no tabs to expand, and no multi-byte
    // clusters — one cell per byte, i.e. `s.len()`.
    if s.bytes().all(|b| (0x20..0x7F).contains(&b)) {
        return s.len();
    }
    let mut w = 0usize;
    let bytes = s.as_bytes();
    // Byte offset before which clusters lie inside a skipped escape
    // sequence.  One `graphemes_indices` walk over the whole string replaces
    // the old per-cluster `s[i..].graphemes(true).next()`, which re-sliced
    // and rebuilt the segmentation state machine at every step.
    let mut skip_until = 0usize;
    for (i, g) in s.grapheme_indices(true) {
        if i < skip_until {
            continue;
        }
        if bytes[i] == 0x1b {
            // CSI: ESC [ … final byte in 0x40..=0x7E.
            if i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                let mut j = i + 2;
                while j < bytes.len() && !(0x40..=0x7e).contains(&bytes[j]) {
                    j += 1;
                }
                skip_until = (j + 1).min(bytes.len());
                continue;
            }
            // OSC: ESC ] … terminated by BEL or ST (ESC \).
            if i + 1 < bytes.len() && bytes[i + 1] == b']' {
                let mut j = i + 2;
                while j < bytes.len() && bytes[j] != 0x07 {
                    if bytes[j] == 0x1b && j + 1 < bytes.len() && bytes[j + 1] == b'\\' {
                        j += 1;
                        break;
                    }
                    j += 1;
                }
                skip_until = (j + 1).min(bytes.len());
                continue;
            }
        }
        // `grapheme_width` keeps CJK / combining sequences correct.
        w += grapheme_width(g, w, tab_width);
    }
    w
}

/// Convert a byte range `[start_offset, end_offset)` within `line`
/// into `(start_display_col, display_width)`.
///
/// Span boundaries falling inside a grapheme cluster are expanded to the
/// cluster boundary so an emoji (or combining sequence) is never half-
/// underlined.
///
/// Test-only oracle for the fused [`render_line_merged_with`] walk.
#[cfg(test)]
pub(crate) fn byte_range_to_display(
    line: &str,
    start_offset: usize,
    end_offset: usize,
    tab_width: usize,
) -> (usize, usize) {
    let mut col = 0usize;
    let mut byte_idx = 0usize;
    let mut start_col: Option<usize> = None;
    let mut end_col = 0usize;
    let mut found_end = false;

    for g in line.graphemes(true) {
        if byte_idx >= end_offset {
            if !found_end {
                end_col = col;
                found_end = true;
            }
            break;
        }
        if start_col.is_none() && byte_idx + g.len() > start_offset {
            start_col = Some(col);
        }
        byte_idx += g.len();
        col += grapheme_width(g, col, tab_width);
    }
    if !found_end {
        end_col = col;
    }
    let start_col = start_col.unwrap_or(end_col);
    (start_col, end_col.saturating_sub(start_col))
}

/// Convert many `[start_offset, end_offset)` byte ranges within `line`
/// into `(start_display_col, display_width)` in a SINGLE grapheme walk.
///
/// Each element matches what `byte_range_to_display` would return for
/// that range alone (same cluster-boundary expansion, same zero-width
/// rules); sharing the walk turns per-row underline work from
/// O(labels × line) into O(line + labels log labels).
///
/// Test-only oracle; the renderers now use [`render_line_merged_with`].
#[cfg(test)]
pub(crate) fn byte_ranges_to_display(line: &str, ranges: &[(usize, usize)], tab_width: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(ranges.len());
    let mut order = Vec::with_capacity(ranges.len());
    let mut done = Vec::with_capacity(ranges.len());
    let mut active: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    byte_ranges_to_display_into(line, ranges, &mut out, &mut order, &mut done, &mut active, tab_width);
    out
}

/// In-place variant of [`byte_ranges_to_display`] that accepts pre-allocated
/// scratch buffers so callers can reuse them across many calls without
/// re-allocation.
///
/// Test-only oracle; see [`byte_ranges_to_display`].
#[cfg(test)]
pub(crate) fn byte_ranges_to_display_into(
    line: &str,
    ranges: &[(usize, usize)],
    out: &mut Vec<(usize, usize)>,
    order: &mut Vec<usize>,
    done: &mut Vec<bool>,
    active: &mut BinaryHeap<Reverse<(usize, usize)>>,
    tab_width: usize,
) {
    out.clear();
    if ranges.is_empty() {
        return;
    }
    // ASCII fast path: every byte is printable ASCII, so every grapheme is
    // exactly one byte and one cell — byte offset ≡ display column.  The
    // cluster-boundary expansion and the zero-width rules collapse to a
    // closed form per range: no sort, no scratch buffers, no grapheme
    // iterator (mirrors the `line_display_width` / `visible_width` fast
    // paths).  The equality test against `byte_range_to_display` covers
    // this branch via its all-ASCII line.
    if line.bytes().all(|b| (0x20..0x7F).contains(&b)) {
        let len = line.len();
        out.reserve(ranges.len());
        for &(start, end) in ranges {
            let end_c = end.min(len);
            out.push(if end > start && start < len {
                (start, end_c - start)
            } else {
                // end <= start (reversed/empty) or start past the line:
                // serial code yields a zero-width range at min(end, len).
                (end_c, 0)
            });
        }
        return;
    }
    // Single-range fast path — the common shape (one label under a `^`):
    // no sort, no active/done bookkeeping. This function is contractually
    // identical to `byte_range_to_display` per range (see the equality test),
    // so reuse it directly.
    if ranges.len() == 1 {
        let (start, end) = ranges[0];
        out.push(byte_range_to_display(line, start, end, tab_width));
        return;
    }
    // Sweep in (start, end) order so one left-to-right walk resolves all
    // ranges. The sort is stable; results are per-range independent.
    order.clear();
    order.extend(0..ranges.len());
    order.sort_by_key(|&i| (ranges[i].0, ranges[i].1));

    out.resize(ranges.len(), (0, 0));
    let mut col = 0usize;
    let mut byte_idx = 0usize;
    let mut next = 0usize;
    active.clear();
    done.clear();
    done.resize(ranges.len(), false);

    for g in line.graphemes(true) {
        let glen = g.len();
        // Ending check first, mirroring the serial loop-top check, so a
        // zero-width range exactly on a grapheme boundary resolves to
        // width 0 instead of activating and leaking into the cluster.
        // The heap is keyed by range end, so the earliest-ending active
        // range is always on top: expiring is a peek + pops, not a scan.
        while let Some(&Reverse((end, i))) = active.peek() {
            if end > byte_idx {
                break;
            }
            active.pop();
            out[i].1 = col;
            done[i] = true;
        }
        // Activate every range overlapping this grapheme. A range whose
        // end is already behind ends here with width 0 (serial break).
        while next < order.len() && ranges[order[next]].0 < byte_idx + glen {
            let i = order[next];
            next += 1;
            if ranges[i].1 <= byte_idx {
                out[i] = (col, col);
                done[i] = true;
            } else {
                out[i].0 = col;
                active.push(Reverse((ranges[i].1, i)));
            }
        }
        byte_idx += glen;
        col += grapheme_width(g, col, tab_width);
    }
    for &Reverse((_, i)) in active.iter() {
        out[i].1 = col;
        done[i] = true;
    }
    // Ranges starting past the last byte never activated: serial code
    // yields (total_col, 0) for them.
    for (i, d) in done.iter().enumerate() {
        if !d {
            out[i] = (col, col);
        }
    }
    // (start_col, end_col) → (start_col, width).
    for (start, end) in out.iter_mut() {
        *end = end.saturating_sub(*start);
    }
}

// ── Syntax highlighting ─────────────────────────────────────────

/// Posita keywords to highlight in bold.
const KEYWORDS: &[&str] = &[
    "def",
    "set",
    "let",
    "return",
    "if",
    "else",
    "while",
    "for",
    "loop",
    "break",
    "leave",
    "continue",
    "true",
    "false",
    "import",
    "type",
    "trait",
    "impl",
    "ensures",
    "requires",
    "invariant",
    "decreases",
    "match",
    "with",
    "struct",
    "enum",
    "pub",
    "mut",
    "ref",
    "comptime",
    "extern",
    "edition",
    "constraint",
    "where",
    "in",
    "is",
    "as",
    "and",
    "or",
    "not",
    "fn",
];

/// Groups by first byte to match [`KEYWORDS`]; the
/// `keyword_dispatch_matches_contains` test asserts they stay in sync.
#[inline]
fn is_keyword(word: &str) -> bool {
    let Some(&first) = word.as_bytes().first() else {
        return false;
    };
    let hit = match first {
        b'a' => matches!(word, "as" | "and"),
        b'b' => matches!(word, "break"),
        b'c' => matches!(word, "comptime" | "constraint" | "continue"),
        b'd' => matches!(word, "decreases" | "def"),
        b'e' => matches!(word, "edition" | "else" | "ensures" | "enum" | "extern"),
        b'f' => matches!(word, "false" | "fn" | "for"),
        b'i' => matches!(word, "if" | "impl" | "import" | "in" | "invariant" | "is"),
        b'l' => matches!(word, "leave" | "let" | "loop"),
        b'm' => matches!(word, "match" | "mut"),
        b'n' => matches!(word, "not"),
        b'o' => matches!(word, "or"),
        b'p' => matches!(word, "pub"),
        b'r' => matches!(word, "ref" | "requires" | "return"),
        b's' => matches!(word, "set" | "struct"),
        b't' => matches!(word, "trait" | "true" | "type"),
        b'w' => matches!(word, "where" | "while" | "with"),
        _ => false,
    };
    debug_assert_eq!(
        hit,
        KEYWORDS.contains(&word),
        "is_keyword disagrees with KEYWORDS for {word:?}"
    );
    hit
}

/// Posita built-in type names to highlight in bright blue.
const TYPES: &[&str] = &[
    "Int", "UInt", "Float", "Bool", "Char", "Byte", "USize", "Str", "Unit", "Never", "String",
];

/// Advance `pos` past a matching bracket pair at `bytes[pos]`, if any.
/// Handles `<...>`, `(...)`, `[...]` with proper nesting.
/// Returns the position past the closing bracket, or `pos` if no bracket
/// or if the nesting depth exceeds `MAX_BRACKET_DEPTH` (guards against
/// unmatched brackets scanning to the end of the input).
fn skip_brackets(bytes: &[u8], pos: usize) -> usize {
    const MAX_BRACKET_DEPTH: usize = 10;
    const MAX_SKIP_LEN: usize = 120; // max chars to scan; guards against
    // a single unclosed bracket scanning
    // to the end of the input.
    if pos >= bytes.len() {
        return pos;
    }
    let (open, close) = match bytes[pos] {
        b'<' => (b'<', b'>'),
        b'(' => (b'(', b')'),
        b'[' => (b'[', b']'),
        _ => return pos,
    };
    let mut p = pos + 1;
    let mut depth: usize = 1;
    while p < bytes.len() && depth > 0 && (p - pos) <= MAX_SKIP_LEN {
        if bytes[p] == open {
            depth += 1;
            if depth > MAX_BRACKET_DEPTH {
                return pos;
            }
        } else if bytes[p] == close {
            depth -= 1;
        }
        p += 1;
    }
    if depth > 0 { pos } else { p }
}

/// Simple syntax highlighter that wraps select tokens in ANSI color codes.
/// Only keywords (bold) and types (bright blue) are highlighted — everything else
/// stays in the default terminal color for a clean, readable output.
/// When `use_color` is false, the input is returned unchanged.
/// Highlight code in a line, optionally adding backticks around type names.
/// When `add_backtick_quotes` is true, type names like `Int<32>` are wrapped
/// in backticks (`` `Int<32>` ``) so that downstream consumers (e.g.
/// `write_header`) can render them with consistent styling.
pub fn highlight_code(line: &str, use_color: bool) -> Cow<'_, str> {
    highlight_code_impl(line, use_color, false)
}

pub(crate) fn highlight_code_with_backticks(line: &str, use_color: bool) -> Cow<'_, str> {
    highlight_code_impl(line, use_color, true)
}

fn highlight_code_impl<'a>(
    line: &'a str,
    use_color: bool,
    add_backtick_quotes: bool,
) -> Cow<'a, str> {
    if !use_color {
        return Cow::Borrowed(line);
    }
    // Every token this pass rewrites (keywords, type names, backtick-quoted
    // type names, and `&`/`*` prefixes on those) contains at least one ASCII
    // letter, so a line with none can only be passed through verbatim.
    // Punctuation-only and digit-only lines (common in stack-trace snippets)
    // now skip the allocation entirely.
    if !line.bytes().any(|b| b.is_ascii_alphabetic()) {
        return Cow::Borrowed(line);
    }

    // Reset only foreground color and bold — preserve background color
    // so that callers can wrap the output in a background color block.
    let reset = "\x1b[22m\x1b[39m";
    let bold = "\x1b[1m";
    let hl = "\x1b[94m";

    let mut out = String::with_capacity(line.len() + 32);
    let mut i = 0;
    let bytes = line.as_bytes();
    let len = bytes.len();
    // When `add_backtick_quotes` is true, type names are wrapped in
    // backticks (`` `Int<32>` ``) so that downstream consumers like
    // `write_header` can render them with consistent styling.
    let backtick = if add_backtick_quotes { "`" } else { "" };

    while i < len {
        // Skip string literals entirely — no highlighting inside strings.
        if bytes[i] == b'"' {
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < len {
                    i += 1;
                }
                i += 1;
            }
            if i < len {
                i += 1;
            }
            out.push_str(&line[start..i]);
            continue;
        }
        if bytes[i] == b'\'' {
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'\'' {
                if bytes[i] == b'\\' && i + 1 < len {
                    i += 1;
                }
                i += 1;
            }
            if i < len {
                i += 1;
            }
            out.push_str(&line[start..i]);
            continue;
        }

        // Identifiers: check for keywords and types.
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &line[start..i];
            if TYPES.contains(&word) {
                // Consume optional `<...>`, `(...)`, `[...]` after the type name
                // so the full type like `Int<32>` or `&Str` is highlighted together.
                let type_end = skip_brackets(bytes, i);
                let _ = std::fmt::write(
                    &mut out,
                    format_args!(
                        "{hl}{}{}{}{reset}",
                        backtick,
                        &line[start..type_end],
                        backtick
                    ),
                );
                i = type_end;
                continue;
            }
            if is_keyword(word) {
                let _ = std::fmt::write(&mut out, format_args!("{bold}{}{reset}", word));
                continue;
            }
            out.push_str(word);
            continue;
        }

        // `&` followed by a type name: highlight the `&` too.
        if (bytes[i] == b'&' || bytes[i] == b'*')
            && i + 1 < len
            && (bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'_')
        {
            let op = bytes[i] as char;
            i += 1;
            let start = i;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &line[start..i];
            if TYPES.contains(&word) {
                let type_end = skip_brackets(bytes, i);
                let _ = std::fmt::write(
                    &mut out,
                    format_args!(
                        "{hl}{}{}{}{}{reset}",
                        backtick,
                        op,
                        &line[start..type_end],
                        backtick
                    ),
                );
                i = type_end;
                continue;
            }
            out.push(op);
            out.push_str(word);
            continue;
        }

        // Backtick-quoted type names: `Int<32>`, `Bool`, etc.
        // Highlight the entire backtick-quoted segment in bright blue
        // when the content matches a known type name.
        if bytes[i] == b'`' {
            let start = i;
            i += 1;
            // Collect the content between backticks.
            let content_start = i;
            while i < len && bytes[i] != b'`' {
                i += 1;
            }
            if i >= len {
                // Unterminated backtick: emit the remainder verbatim instead
                // of slicing out of bounds with `..=i`.
                out.push_str(&line[start..]);
                break;
            }
            let content = &line[content_start..i];
            // Check if the content (without the trailing `>`) is a type
            // name, e.g. `Int<32>` → check `Int`, `&Str` → check `Str`.
            let type_name = content
                .split(&['<', '(', '['][..])
                .next()
                .unwrap_or(content);
            // Strip leading `&` or `*` for reference/pointer types.
            let type_name = type_name.trim_start_matches(|c| c == '&' || c == '*');
            if TYPES.contains(&type_name) {
                let _ = std::fmt::write(&mut out, format_args!("{hl}{}{reset}", &line[start..=i]));
            } else {
                out.push_str(&line[start..=i]);
            }
            i += 1;
            continue;
        }

        // Everything else: pass through as-is, decoding the full UTF-8
        // character (never bytes) so multi-byte chars are not mangled.
        // `next_char` decodes straight from the byte slice, avoiding a
        // `line[i..].chars().next()` re-slice + iterator rebuild per byte —
        // this fallthrough is the hottest path in the scanner.
        // This function is not the terminal-safety boundary: callers must
        // sanitize before it, and the highlight escapes added here are the
        // only escapes it ever emits.
        let c = next_char(bytes, &mut i);
        out.push(c);
    }

    Cow::Owned(out)
}

/// Decode the UTF-8 scalar at `bytes[*i]` (a lead byte on a char boundary)
/// and advance `*i` past it.  Valid-UTF-8 input makes every `unwrap`
/// infallible; this is the decoder `highlight_code_impl` uses to avoid
/// re-slicing the string for every ordinary character.
#[inline]
fn next_char(bytes: &[u8], i: &mut usize) -> char {
    let b0 = bytes[*i];
    let (c, n) = if b0 < 0x80 {
        (b0 as char, 1)
    } else if b0 < 0xE0 {
        let cp = ((b0 & 0x1F) as u32) << 6 | (bytes[*i + 1] & 0x3F) as u32;
        (char::from_u32(cp).unwrap(), 2)
    } else if b0 < 0xF0 {
        let cp = ((b0 & 0x0F) as u32) << 12
            | ((bytes[*i + 1] & 0x3F) as u32) << 6
            | (bytes[*i + 2] & 0x3F) as u32;
        (char::from_u32(cp).unwrap(), 3)
    } else {
        let cp = ((b0 & 0x07) as u32) << 18
            | ((bytes[*i + 1] & 0x3F) as u32) << 12
            | ((bytes[*i + 2] & 0x3F) as u32) << 6
            | (bytes[*i + 3] & 0x3F) as u32;
        (char::from_u32(cp).unwrap(), 4)
    };
    *i += n;
    c
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::diagnostics::Diagnostic;

    #[test]
    fn test_basic_error_render() {
        let source = "def a(x: Bool)\n  ensures @s > 1\n  {\n    set j = \"0xFFFF\";\n    set i = j + 1;\n    return @s @r i;\n  }\ndef main(){}";
        let diag = Diagnostic::error(
            "trait solver error: no trait implementation found for `Ord` on type `Int`",
        )
        .with_code_str("E030")
        .with_span(Span::new(FileId::DUMMY, 0, 106))
        .with_source(source);
        let renderer = GlyphRenderer::new(false);
        let output = renderer.render_diagnostic(&diag);
        println!("{}", output);
        assert!(output.contains("error[E030]"));
        assert!(output.contains("trait solver error"));
        assert!(output.contains("<input>"));
    }

    #[test]
    fn test_keyword_suggestion_format() {
        let source = "dem main(){}";
        let diag = Diagnostic::error("unexpected identifier `dem` at top level")
            .with_code_str("E003")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_source(source)
            .with_suggestion("`dem` is not a keyword. Did you mean `def`?");
        let renderer = GlyphRenderer::new(false);
        let output = renderer.render_diagnostic(&diag);
        println!("{}", output);
        assert!(output.contains("error[E003]"));
        assert!(output.contains("dem"));
        assert!(output.contains("Did you mean"));
    }

    #[test]
    fn test_summary() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        let renderer = GlyphRenderer::new(false);
        let s = renderer.render_summary(1, 0);
        assert!(s.contains("aborting due to previous error"));
        let s = renderer.render_summary(3, 0);
        assert!(s.contains("aborting due to 3 previous errors"));
        let s = renderer.render_summary(0, 1);
        assert!(s.contains("1 warning emitted"));
        let s = renderer.render_summary(0, 4);
        assert!(s.contains("4 warnings emitted"));
        let s = renderer.render_summary(2, 1);
        assert!(s.contains("aborting due to 2 previous errors; 1 warning emitted"));
        assert!(!s.contains("emitted emitted"), "double `emitted`: {s:?}");
        let s = renderer.render_summary(1, 2);
        assert!(s.contains("aborting due to 1 previous error; 2 warnings emitted"));
    }

    /// `is_keyword` re-derives `KEYWORDS` by hand, so every keyword — plus
    /// strings exercising the empty case, case sensitivity, prefixes and
    /// every unhandled first byte — must agree with `KEYWORDS::contains`.
    #[test]
    fn keyword_dispatch_matches_contains() {
        let mut cases: Vec<String> = KEYWORDS.iter().map(|k| (*k).to_string()).collect();
        cases.extend(
            [
                "", "x", "Def", "DEF", "define", "defs", "iff", "let_", "zzz", "αβ", "def2",
            ]
            .iter()
            .map(|s| (*s).to_string()),
        );
        for b in b'a'..=b'z' {
            cases.push(format!("{}zzz", b as char));
        }

        for word in &cases {
            assert_eq!(
                is_keyword(word),
                KEYWORDS.contains(&word.as_str()),
                "is_keyword disagrees with KEYWORDS::contains for {word:?}"
            );
        }
    }

    /// Measurement/printing alignment for control characters: the source-row
    /// printer emits one U+FFFD (1 cell) per control grapheme, so every
    /// width function must credit controls with one cell too — otherwise
    /// underlines and primary highlights shift left of the replaced glyph
    /// for every preceding control on the line.
    #[test]
    fn control_chars_measure_as_printed() {
        assert_eq!(grapheme_width("\x07", 0, DEFAULT_TAB_WIDTH), 1);
        assert_eq!(grapheme_width("\x7f", 0, DEFAULT_TAB_WIDTH), 1); // DEL
        assert_eq!(line_display_width("a\x07b", DEFAULT_TAB_WIDTH), 3);
        // `b` sits at display column 2 — after `a` and the U+FFFD cell.
        assert_eq!(byte_range_to_display("a\x07b", 2, 3, DEFAULT_TAB_WIDTH), (2, 1));
    }

    #[test]
    fn grapheme_widths_match_terminal() {
        assert_eq!(grapheme_width("a", 0, DEFAULT_TAB_WIDTH), 1);
        assert_eq!(grapheme_width("中", 0, DEFAULT_TAB_WIDTH), 2);
        assert_eq!(grapheme_width("🚀", 0, DEFAULT_TAB_WIDTH), 2); // previously missed (outside 1F300–1F64F)
        assert_eq!(grapheme_width("e\u{0301}", 0, DEFAULT_TAB_WIDTH), 1); // combining acute = 0 width
        assert_eq!(grapheme_width("👨\u{200D}👩\u{200D}👧\u{200D}👦", 0, DEFAULT_TAB_WIDTH), 2); // ZWJ family
        assert_eq!(grapheme_width("🇺🇸", 0, DEFAULT_TAB_WIDTH), 2); // RI pair
        assert_eq!(grapheme_width("\u{25B6}\u{FE0F}", 0, DEFAULT_TAB_WIDTH), 2); // presentation sequence
        assert_eq!(grapheme_width("👍\u{1F3FD}", 0, DEFAULT_TAB_WIDTH), 2); // skin-tone modifier
        assert_eq!(grapheme_width("\t", 1, DEFAULT_TAB_WIDTH), 3); // tab stop at col 4
        assert_eq!(grapheme_width("\x07", 0, DEFAULT_TAB_WIDTH), 1); // bell control → one U+FFFD cell
        assert_eq!(line_display_width("e\u{0301}中", DEFAULT_TAB_WIDTH), 3); // é + 中 = 1+2
    }

    #[test]
    fn ascii_width_fast_path_covers_every_byte() {
        // Every single-byte cluster is one cell: printable ASCII directly,
        // C0/DEL via the U+FFFD the printer substitutes for it — tab
        // excepted, it expands to the next stop.
        for b in 0u8..0x80 {
            let s = (b as char).to_string();
            if b == b'\t' {
                assert_eq!(grapheme_width(&s, 1, DEFAULT_TAB_WIDTH), 3, "tab from col 1");
                continue;
            }
            assert_eq!(grapheme_width(&s, 0, DEFAULT_TAB_WIDTH), 1, "byte {b:#04x}");
        }
    }

    #[test]
    fn width_cache_agrees_with_uncached_across_eviction() {
        // More distinct clusters than the cache can hold, so the first pass
        // fills and evicts and the second re-queries entries that are gone.
        // Both passes must equal the uncached width, and a repeated hit must
        // not drift.
        let clusters: Vec<String> = (0x4E00u32..0x4E00 + 1000)
            .filter_map(char::from_u32)
            .map(|c| c.to_string())
            .collect();
        for g in &clusters {
            assert_eq!(
                grapheme_width(g, 0, DEFAULT_TAB_WIDTH),
                grapheme_width_uncached(g),
                "first pass {g:?}"
            );
        }
        for g in &clusters {
            assert_eq!(
                grapheme_width(g, 0, DEFAULT_TAB_WIDTH),
                grapheme_width_uncached(g),
                "second pass {g:?}"
            );
        }
        let family = "👨\u{200D}👩\u{200D}👧\u{200D}👦";
        assert_eq!(grapheme_width(family, 0, DEFAULT_TAB_WIDTH), 2);
        assert_eq!(grapheme_width(family, 0, DEFAULT_TAB_WIDTH), 2);
    }

    #[test]
    fn byte_range_display_respects_clusters() {
        let line = "e\u{0301}f"; // graphemes: "e" | "e\u{0301}" | "f", bytes: 0 | 1-2 | 3
        // span covering whole é cluster (bytes 0..2) → start_col 0, width 1
        assert_eq!(byte_range_to_display(line, 0, 2, DEFAULT_TAB_WIDTH), (0, 1));
        // span starting inside cluster (byte 1, the combining accent) → expand to whole cluster
        assert_eq!(byte_range_to_display(line, 1, 3, DEFAULT_TAB_WIDTH), (0, 1));
        // span covering only 'f' (bytes 3..4) — start falls inside 2nd cluster → start_col 1
        assert_eq!(byte_range_to_display(line, 3, 4, DEFAULT_TAB_WIDTH), (1, 1));
    }

    #[test]
    fn byte_ranges_batch_matches_serial() {
        // The batched sweep must agree with the serial walk on every
        // range, including zero-width ranges on cluster boundaries.
        let lines = ["e\u{0301}f", "hello world", "a\tb中c", "", "👍\u{1F3FD}x"];
        for line in lines {
            let n = line.len();
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            let mut i = 0;
            while i <= n {
                for e in [0, 1, 2, 3, 5, 8, n, n.saturating_add(4)] {
                    let (s, e) = if i <= e { (i, e.min(n)) } else { (e.min(n), i) };
                    ranges.push((s, s));
                    ranges.push((s, e));
                }
                i += 1;
            }
            let expected: Vec<(usize, usize)> = ranges
                .iter()
                .map(|&(s, e)| byte_range_to_display(line, s, e, DEFAULT_TAB_WIDTH))
                .collect();
            assert_eq!(
                byte_ranges_to_display(line, &ranges, DEFAULT_TAB_WIDTH),
                expected,
                "line={line:?}"
            );
        }
    }

    /// Old three-walk pipeline kept as the oracle for [`render_line_merged`]:
    /// batch byte→display conversion, separate width measure, primary-column
    /// marks, then the standalone source-line print loop (a verbatim copy of
    /// the code `write_source_section` ran before the walks were fused).
    ///
    /// The one deliberate change is the insertion-point policy: zero-width
    /// annotations are promoted to one-cell markers in the returned underline
    /// list (`width.max(1)`), while the primary highlight is computed from
    /// the raw widths — exactly matching the fused renderer's byte-overlap
    /// rule (a point on a cluster boundary highlights nothing) — so the
    /// equality assertions hold.
    fn legacy_render<'a>(
        line: &str,
        ranges: &[(usize, usize)],
        meta: &[(char, &'a str, Option<usize>)],
        highlight_on: &str,
        highlight_off: &str,
    ) -> (
        usize,
        String,
        Vec<(usize, usize, char, &'a str, Option<usize>)>,
    ) {
        let display_buf = byte_ranges_to_display(line, ranges, DEFAULT_TAB_WIDTH);
        let display_width = line_display_width(line, DEFAULT_TAB_WIDTH);
        let raw_underlines: Vec<(usize, usize, char, &'a str, Option<usize>)> = display_buf
            .iter()
            .zip(meta.iter())
            .map(|(&(col, width), &(ch, msg, label_idx))| (col, width, ch, msg, label_idx))
            .collect();
        // Primary marks come from the raw widths, matching the fused
        // renderer's byte-overlap rule: a point on a cluster boundary
        // highlights nothing, while a point inside a cluster has already
        // expanded to the whole cluster in the conversion above.
        let mut primary_cols = Vec::new();
        compute_primary_cols(&raw_underlines, display_width, &mut primary_cols);
        // Insertion-point policy (mirrors the fused pipeline's underline
        // rows): zero-width annotations reach lane assignment one cell wide.
        let underlines: Vec<(usize, usize, char, &'a str, Option<usize>)> = raw_underlines
            .iter()
            .map(|&(col, width, ch, msg, label_idx)| (col, width.max(1), ch, msg, label_idx))
            .collect();

        let mut rendered = String::new();
        let mut in_primary = false;
        let mut disp_col = 0usize;
        if line.bytes().all(|b| (0x20..0x7F).contains(&b)) {
            let len = line.len();
            let mut i = 0usize;
            while i < len {
                let is_primary = primary_cols.get(i).copied().unwrap_or(false);
                let start = i;
                while i < len && primary_cols.get(i).copied().unwrap_or(false) == is_primary {
                    i += 1;
                }
                if is_primary != in_primary {
                    if in_primary {
                        rendered.push_str(highlight_off);
                    }
                    if is_primary {
                        rendered.push_str(highlight_on);
                    }
                    in_primary = is_primary;
                }
                rendered.push_str(&line[start..i]);
            }
        } else {
            for g in line.graphemes(true) {
                let is_primary = primary_cols.get(disp_col).copied().unwrap_or(false);
                if is_primary != in_primary {
                    if in_primary {
                        rendered.push_str(highlight_off);
                    }
                    if is_primary {
                        rendered.push_str(highlight_on);
                    }
                    in_primary = is_primary;
                }
                if g == "\t" {
                    let w = grapheme_width(g, disp_col, DEFAULT_TAB_WIDTH);
                    for _ in 0..w {
                        rendered.push(' ');
                    }
                } else if g.chars().any(|c| c.is_control()) {
                    rendered.push('\u{FFFD}');
                } else if g.chars().all(|c| crate::diagnostics::is_bidi_override(c)) {
                    // Bidi overrides removed — see `render_line_merged`.
                } else {
                    rendered.push_str(g);
                }
                disp_col += grapheme_width(g, disp_col, DEFAULT_TAB_WIDTH);
            }
        }
        if in_primary {
            rendered.push_str(highlight_off);
        }
        (display_width, rendered, underlines)
    }

    #[test]
    fn merged_render_matches_three_walk_pipeline() {
        // The fused walk must reproduce the old three-walk pipeline byte for
        // byte: same `(col, width)` conversion (and therefore same synthetic
        // underlines), same measured display width, and the same highlight
        // escape placement — on ASCII and non-ASCII lines, across annotation
        // sets that include zero-width and out-of-bounds ranges.
        let highlight_on = "\x1b[48;5;236m";
        let highlight_off = "\x1b[49m";
        let lines = [
            "",
            "hello world",
            "a\tb中c",
            "e\u{0301}f",
            "👍\u{1F3FD}x",
            "a\x07b",
            "\x7f中",
            "中中中",
        ];
        let check = |line: &str, ranges: &[(usize, usize)]| {
            // Annotation chars alternate between `^` (primary) and `~` so the
            // fused prefilter and the old primary-column marks are both
            // exercised; messages/label indices verify straight pass-through.
            let meta: Vec<(char, &str, Option<usize>)> = ranges
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i % 2 == 0 {
                        ('^', "", None)
                    } else {
                        ('~', "some label", Some(i))
                    }
                })
                .collect();
            let (want_width, want_rendered, want_out) =
                legacy_render(line, ranges, &meta, highlight_on, highlight_off);

            let mut out = Vec::new();
            let mut rendered = String::new();
            let mut display_buf = Vec::new();
            let mut order_buf = Vec::new();
            let mut done_buf = Vec::new();
            let mut active_buf = BinaryHeap::new();
            let got_width = render_line_merged(
                line,
                ranges,
                &meta,
                &mut out,
                &mut rendered,
                highlight_on,
                highlight_off,
                &mut display_buf,
                &mut order_buf,
                &mut done_buf,
                &mut active_buf,
                DEFAULT_TAB_WIDTH,
            );

            assert_eq!(got_width, want_width, "width for line={line:?}");
            assert_eq!(rendered, want_rendered, "render for line={line:?}");
            assert_eq!(out, want_out, "underlines for line={line:?}");
        };

        for line in lines {
            let n = line.len();
            let mut marks: Vec<usize> = vec![0, n, n.saturating_add(2)];
            marks.extend(0..n);
            marks.sort_unstable();
            marks.dedup();
            // Every normalized pair over the boundary marks — starts and ends
            // land on cluster boundaries, inside multi-byte clusters, at the
            // line end, and past it.
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            for &s in &marks {
                for &e in &marks {
                    if s <= e {
                        ranges.push((s, e));
                    }
                }
            }
            check(line, &ranges);
            // ...plus each pair alone: single-range rows are the common shape
            // and the old pipeline short-circuits them to the serial walk.
            for &(s, e) in &ranges {
                check(line, &[(s, e)]);
            }
        }
    }

    #[test]
    fn test_plain_render_has_no_ansi() {
        let source = "def x() {}";
        let diag = Diagnostic::fatal("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_label(Span::new(FileId::DUMMY, 4, 5), "here")
            .with_help("do better")
            .with_file_name("test.ps")
            .with_source(source);
        let output = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            !output.contains('\x1b'),
            "plain output leaked ANSI: {output:?}"
        );
    }

    #[test]
    fn span_suggestion_renders_diff() {
        let src = "dem main(){}\n";
        let diag = Diagnostic::error("unexpected identifier `dem` at top level")
            .with_code_str("E003")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_span_suggestion(
                Span::new(FileId::DUMMY, 0, 3),
                "def",
                "replace `dem` with `def`",
                Applicability::MachineApplicable,
            )
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(out.contains("- dem main(){}"), "missing - line: {out:?}");
        assert!(out.contains("+ def main(){}"), "missing + line: {out:?}");
        assert!(
            out.contains("help: replace `dem` with `def`"),
            "missing message line: {out:?}"
        );
        // No duplicate text-only fallback for the same suggestion.
        let text_fallback = out.matches("replace `dem` with `def`").count();
        assert_eq!(
            text_fallback, 1,
            "suggestion emitted both as diff and as fallback: {out:?}"
        );
        // Gutter alignment: everything up to (not including) the `-`/`+`
        // marker must be identical between the two diff rows — same gutter
        // bars, line number, sub-bar, and glyph cell.  The payloads after
        // the marker legitimately differ (old vs new source).
        let lines: Vec<&str> = out.lines().collect();
        let minus = lines
            .iter()
            .find(|l| l.contains("- dem main(){}"))
            .unwrap();
        let plus = lines.iter().find(|l| l.contains("+ def main(){}")).unwrap();
        let prefix_before = |l: &str, marker: char| -> String {
            l.find(marker)
                .map(|i| &l[..i])
                .unwrap_or(l)
                .to_string()
        };
        assert_eq!(
            prefix_before(minus, '-'),
            prefix_before(plus, '+'),
            "diff rows do not share the same gutter prefix:\n  {minus:?}\n  {plus:?}"
        );
        // The `+` line carries the same source line number as the `-` line
        // (inline replacement, spec conformance).
        assert!(
            plus.contains("1"),
            "the + line must keep the source line number: {plus:?}"
        );
    }

    #[test]
    fn span_suggestion_multi_line_span_falls_back_to_text() {
        // A span crossing two lines must not produce a diff; the message
        // still reaches the reader via the text-only fallback path.
        let src = "dem\nmain(){}\n";
        let diag = Diagnostic::error("unexpected identifier `dem`")
            .with_code_str("E003")
            .with_span(Span::new(FileId::DUMMY, 0, 5))
            .with_span_suggestion(
                // Span (0, 5) covers `dem\nm` — it crosses into line 2, so
                // diff rendering must fall back to the text-only message.
                Span::new(FileId::DUMMY, 0, 5),
                "def\nmain(){}",
                "replace `dem` with `def`",
                Applicability::MachineApplicable,
            )
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            !out.contains("- dem"),
            "multi-line span must not produce a diff: {out:?}"
        );
        assert!(
            !out.contains("+ def"),
            "multi-line span must not produce a diff: {out:?}"
        );
        // The text-only fallback (the `Display` impl carries the tag) keeps
        // the suggestion visible.
        assert!(
            out.contains("replace `dem` with `def`"),
            "text fallback missing: {out:?}"
        );
    }

    #[test]
    fn span_suggestion_foreign_file_falls_back_to_text() {
        // A suggestion span from a different file than the source anchor
        // must not slice the wrong buffer; fall back to text only.
        let src = "dem main(){}\n";
        let diag = Diagnostic::error("unexpected identifier `dem`")
            .with_code_str("E003")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_span_suggestion(
                Span::new(FileId::from_raw(2), 0, 3),
                "def",
                "replace `dem` with `def`",
                Applicability::MachineApplicable,
            )
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            !out.contains("- dem main(){}"),
            "foreign-file suggestion must not produce a diff: {out:?}"
        );
        assert!(
            out.contains("replace `dem` with `def`"),
            "text fallback missing: {out:?}"
        );
    }

    #[test]
    fn span_suggestion_malformed_span_does_not_panic() {
        // `span.end < span.start` would panic the unsanitized slice
        // `&old_line[start..end]`; the guard must skip to the fallback.
        let src = "dem main(){}\n";
        let sug = Suggestion {
            message: "replace".into(),
            applicability: Applicability::MachineApplicable,
            style: crate::diagnostics::SuggestionStyle::ShowAlways,
            span: Some(Span::new(FileId::DUMMY, 5, 2)), // end < start
            replacement: Some("def".into()),
        };
        let diag = Diagnostic::error("unexpected identifier `dem`")
            .with_code_str("E003")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_suggestions(vec![sug])
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            out.contains("replace"),
            "text fallback missing for malformed span: {out:?}"
        );
    }

    #[test]
    fn span_suggestion_escapes_source_controls() {
        // A source line carrying an ESC sequence must not leak it verbatim
        // into the diff block: sanitize_for_terminal_multiline replaces
        // control characters with U+FFFD.
        let src = "\x1b]0;pwned\x07dem main(){}\n";
        let diag = Diagnostic::error("unexpected identifier")
            .with_code_str("E003")
            .with_span(Span::new(FileId::DUMMY, 17, 20))
            .with_span_suggestion(
                Span::new(FileId::DUMMY, 17, 20),
                "def",
                "replace `dem` with `def`",
                Applicability::MachineApplicable,
            )
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            !out.contains('\x1b'),
            "unescaped control char in plain output: {out:?}"
        );
        assert!(
            out.contains('\u{FFFD}'),
            "expected U+FFFD replacement in diff: {out:?}"
        );
    }

    #[test]
    fn test_unterminated_backtick_does_not_panic() {
        // `highlight_code` only does anything when colors are enabled.
        let out = highlight_code("see `Int", true);
        assert!(out.contains("Int"), "got: {out:?}");
    }

    #[test]
    fn test_non_char_boundary_span_does_not_panic() {
        let src = "é"; // two bytes
        assert_eq!(span_line(Span::new(FileId::DUMMY, 1, 1), src), 0);
        let pos = crate::diagnostics::byte_to_linecol(src, 1);
        assert_eq!(pos.line, 0);
    }

    #[test]
    fn test_multiline_label_message_rendered_once() {
        let source = "aaa\nbbb\nccc\n";
        let diag = Diagnostic::error("multi")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_label(Span::new(FileId::DUMMY, 0, 11), "spans three lines")
            .with_source(source);
        let output = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert_eq!(
            output.matches("spans three lines").count(),
            1,
            "multi-line label repeated:\n{output}"
        );
    }

    #[test]
    fn test_multi_primary_span_label_not_duplicated() {
        // A label near the *second* primary span must not also be emitted as
        // a fallback note.  The second span is on line 4, outside the first
        // span's context window (lines 0-3).
        let source = "line one\nline two\nline three\nline four\nline five\n";
        let diag = Diagnostic::error("two spans")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 4))
            .with_additional_span(Span::new(FileId::DUMMY, 39, 43))
            .with_label(Span::new(FileId::DUMMY, 39, 43), "second site")
            .with_source(source);
        let output = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert_eq!(
            output.matches("second site").count(),
            1,
            "label duplicated across primary spans:\n{output}"
        );
    }

    #[test]
    fn line_starts_matches_naive() {
        let src = "a\nbb\n\nccc";
        let starts = line_starts(src);
        assert_eq!(starts, vec![0, 2, 5, 6]);
        for (i, &s) in starts.iter().enumerate() {
            assert_eq!(line_start_byte(src, i), s, "mismatch at line {i}");
        }
    }

    #[test]
    fn osc8_url_is_percent_encoded() {
        let src = "def x() {}\n";
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_source(src)
            .with_file_name("a b\x1b]8;;evil\x07.ps");
        let out = GlyphRenderer::new(true).render_diagnostic(&diag);
        assert!(
            !out.contains("\x1b]8;;evil"),
            "control char leaked into OSC 8 URL: {out:?}"
        );
        assert!(out.contains("%20"), "space not encoded: {out:?}");
    }

    // ── Message wrapping ────────────────────────────────────────

    /// Strip CSI and OSC escape sequences so two renderings can be compared
    /// by their *visible* text.
    fn strip_ansi(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i] == 0x1b && i + 1 < b.len() && b[i + 1] == b'[' {
                i += 2;
                while i < b.len() && !(0x40..=0x7e).contains(&b[i]) {
                    i += 1;
                }
                i = (i + 1).min(b.len());
                continue;
            }
            if b[i] == 0x1b && i + 1 < b.len() && b[i + 1] == b']' {
                i += 2;
                while i < b.len() && b[i] != 0x07 {
                    if b[i] == 0x1b && i + 1 < b.len() && b[i + 1] == b'\\' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                i = (i + 1).min(b.len());
                continue;
            }
            let c = s[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
        out
    }

    #[test]
    fn visible_width_skips_ansi() {
        assert_eq!(visible_width("abc"), 3);
        assert_eq!(visible_width("\x1b[1;91mabc\x1b[0m"), 3);
        assert_eq!(visible_width("\x1b]8;;http://x\x1b\\ab\x1b]8;;\x1b\\"), 2);
        assert_eq!(visible_width("中\x1b[0m"), 2);
    }

    /// The printable-ASCII fast path must equal the byte length and agree
    /// with the slow path; any byte outside `0x20..=0x7E` (tab, DEL, ESC,
    /// non-ASCII) must fall through to the grapheme walk.
    #[test]
    fn ascii_fast_path_matches_byte_len() {
        for s in ["E030", "  = note: ", "hello world", "a]8;;x\\b", ""] {
            assert_eq!(line_display_width(s, DEFAULT_TAB_WIDTH), s.len(), "line_display_width fast path: {s:?}");
            assert_eq!(visible_width(s), s.len(), "visible_width fast path: {s:?}");
        }
        // Fall-through cases: not all bytes printable ASCII.
        assert_eq!(line_display_width("\t", DEFAULT_TAB_WIDTH), 4); // tab → TAB_WIDTH at col 0
        assert_eq!(line_display_width("\x7f", DEFAULT_TAB_WIDTH), 1); // DEL → one U+FFFD cell
        assert_eq!(line_display_width("中", DEFAULT_TAB_WIDTH), 2); // CJK → 2 cells
        assert_eq!(visible_width("\t"), 4);
        assert_eq!(visible_width("\x7f"), 1);
    }

    /// Non-ASCII characters fall through the token scanner's `next_char`
    /// decoder; every UTF-8 length class (2/3/4 bytes) must survive verbatim.
    #[test]
    fn highlight_code_preserves_multibyte_chars() {
        for s in ["é", "中", "🙂", "a中b", "é中🙂", "xé中🙂y"] {
            assert_eq!(highlight_code(s, true), s, "mangled {s:?}");
        }
    }

    /// A label message longer than `max_width` wraps, and every continuation
    /// line repeats the `|` connector so the annotation reads continuously.
    #[test]
    fn long_label_message_wraps_with_repeated_connector() {
        let src = "set i = 1;\n";
        let long = "this label repeats the connector over and over again to force a wrap";
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 5))
            .with_label(Span::new(FileId::DUMMY, 0, 5), long)
            .with_source(src);
        // ASCII charset pinned: the assertion below checks the `|` connector
        // literally, and explicit options no longer derive ASCII from
        // `use_color == false`.
        let opts = crate::diagnostics::DiagOpts {
            max_width: 40,
            use_unicode: false,
            ..Default::default()
        };
        let out = GlyphRenderer::with_opts(false, Some(&opts)).render_diagnostic(&diag);
        // Words unique to the label; each must appear on a line whose visible
        // text starts with the connector column.
        let mut wrapped_lines = 0;
        for line in out.lines() {
            if line.contains("this") || line.contains("over") || line.contains("wrap") {
                wrapped_lines += 1;
                assert!(
                    line.trim_start().starts_with('|'),
                    "continuation lost the box connector: {line:?}\n{out}"
                );
            }
        }
        assert!(
            wrapped_lines >= 2,
            "expected the label to wrap across lines, got:\n{out}"
        );
    }

    /// The wrapper runs on plain text before styling, so ANSI escapes are
    /// zero-width: coloured and plain output break at the same words and
    /// their visible text is identical line for line.
    #[test]
    fn colored_and_plain_wrap_at_the_same_word() {
        let src = "set i = 1;\n";
        let long = "a long label message that must wrap identically with and without colour";
        let mk = || {
            Diagnostic::error("boom")
                .with_code_str("E030")
                .with_span(Span::new(FileId::DUMMY, 0, 5))
                .with_label(Span::new(FileId::DUMMY, 0, 5), long)
                .with_source(src)
        };
        // Charsets are now independent of colour: pin the plain renderer to
        // ASCII (it has no colour to imply Unicode) while the coloured side
        // keeps the default Unicode charset, so the normalisation below
        // bridges the intended box-drawing difference.
        let ascii = crate::diagnostics::DiagOpts {
            max_width: 44,
            use_unicode: false,
            ..Default::default()
        };
        let unicode = crate::diagnostics::DiagOpts {
            max_width: 44,
            ..Default::default()
        };
        let plain = GlyphRenderer::with_opts(false, Some(&ascii)).render_diagnostic(&mk());
        let colored = GlyphRenderer::with_opts(true, Some(&unicode)).render_diagnostic(&mk());
        // Normalise the Unicode box-drawing characters to their ASCII
        // equivalents (the only intended visible difference between modes).
        let normalized = strip_ansi(&colored)
            .replace('│', "|")
            .replace('┌', "+")
            .replace('└', "\\")
            .replace('─', "-");
        assert_eq!(
            normalized, plain,
            "coloured/plain wrapping diverged:\n--- colored ---\n{colored}\n--- plain ---\n{plain}"
        );
    }

    #[test]
    fn error_code_is_sanitized_in_header() {
        let out = GlyphRenderer::new(true)
            .render_diagnostic(&Diagnostic::error("boom").with_code_str("E0\x1b]8;;x\x07"));
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced: {out:?}"
        );
    }

    #[test]
    fn related_error_codes_are_sanitized_in_header() {
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_related(crate::diagnostics::RelatedError {
                code: Some(crate::diagnostics::ErrCode::unknown("E0\x1b]8;;evil\x07")),
                message: "related".into(),
                span: None,
                label: None,
            });
        let out = GlyphRenderer::new(true).render_diagnostic(&diag);
        assert!(!out.contains('\x07'), "BEL leaked from related: {out:?}");
    }

    #[test]
    fn header_message_is_sanitized() {
        let out = GlyphRenderer::new(true).render_diagnostic(&Diagnostic::error("bo\x1b]0;om\x07"));
        assert!(!out.contains('\x07'), "BEL leaked from header: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced: {out:?}"
        );
    }

    #[test]
    fn label_message_is_sanitized() {
        let src = "set i = 1;\n";
        let diag = Diagnostic::error("boom")
            .with_span(Span::new(FileId::DUMMY, 0, 5))
            .with_label(Span::new(FileId::DUMMY, 0, 5), "bad\x1b]0;label\x07")
            .with_source(src);
        let out = GlyphRenderer::new(true).render_diagnostic(&diag);
        assert!(!out.contains('\x07'), "BEL leaked from label: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced: {out:?}"
        );
    }

    #[test]
    fn source_line_control_chars_are_replaced() {
        // The ESC/BEL sit inside the rendered source row, between the SGR
        // runs the renderer itself emits.
        let src = "let x = \x1b\x07;\n";
        let diag = Diagnostic::error("boom")
            .with_span(Span::new(FileId::DUMMY, 8, 12))
            .with_label(Span::new(FileId::DUMMY, 8, 12), "here")
            .with_source(src);
        let out = GlyphRenderer::new(true).render_diagnostic(&diag);
        // Strip the renderer's own highlight escapes, then no raw control
        // character may survive.
        let plain = strip_ansi(&out);
        assert!(
            !plain.contains('\x1b') && !plain.contains('\x07'),
            "control char leaked into source row: {plain:?}"
        );
    }

    #[test]
    fn same_message_labels_are_not_merged() {
        // Two distinct labels carry the same message: a multi-line label
        // spanning both rows and a single-line label on the FIRST row only.
        // Identity must come from the label index: the single-line label
        // renders its message on its own row even though a same-message
        // multi-line label does not end there (a message-keyed map treated
        // the single-line label as "non-final" and suppressed it entirely).
        let src = "aaaa\nbbbb\n";
        let diag = Diagnostic::error("boom")
            .with_span(Span::new(FileId::DUMMY, 0, 4))
            .with_label(Span::new(FileId::DUMMY, 0, 9), "dup")
            .with_label(Span::new(FileId::DUMMY, 0, 3), "dup")
            .with_source(src);
        let out = GlyphRenderer::new(true).render_diagnostic(&diag);
        let rendered: Vec<&str> = out
            .lines()
            .filter(|line| line.contains("dup"))
            .collect();
        assert_eq!(
            rendered.len(),
            2,
            "same-message labels were merged or suppressed: {rendered:?}\n{out}"
        );
        // The two label lines: one may use the inline fast path (no `|`),
        // the other keeps the connector; both must show the message text.
        assert!(
            rendered.join(" ").contains("dup"),
            "label message text lost: {rendered:?}\n{out}"
        );
    }

    #[test]
    fn eof_span_renders_without_panic() {
        // Regression guard: a span whose end sits exactly at EOF (and one that
        // runs one byte past it) used to be a boundary case for line-range
        // computation.  Rendering must produce the diagnostic header and the
        // label message, and must not panic.
        let src = "let x = 1;\n";
        let src_len = src.len() as u32;
        for end in [src_len, src_len + 1] {
            let diag = Diagnostic::error("boom")
                .with_code_str("E030")
                .with_span(Span::new(FileId::DUMMY, src_len, end))
                .with_label(Span::new(FileId::DUMMY, src_len, end), "at eof")
                .with_source(src);
            let out = GlyphRenderer::new(false).render_diagnostic(&diag);
            assert!(out.contains("boom"), "missing header for end={end}: {out}");
            assert!(
                out.contains("at eof"),
                "missing label message for end={end}: {out}"
            );
        }
    }

    /// Regression (#1): a label whose span end sits exactly on a line start
    /// was attributed to the *following* line by `span_lines`, which both
    /// suppressed its message on the line that drew the underline and made
    /// the fallback believe it had already been shown. The message vanished.
    #[test]
    fn boundary_ended_label_message_is_still_rendered() {
        let src = "aaa\nbbb\nccc\n";
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_label(Span::new(FileId::DUMMY, 0, 4), "msg")
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(out.contains("msg"), "label message dropped:\n{out}");
        assert_eq!(out.matches("msg").count(), 1, "message duplicated:\n{out}");
    }

    /// a zero-width label at a line end used to vanish
    /// entirely — the zero-width range drew nothing and the fallback
    /// skipped it because its line was inside a shown range.  It now gets a
    /// one-cell caret just past the last character, with its message inline
    /// on the connector row.
    #[test]
    fn zero_width_label_at_line_end_renders_one_cell_marker() {
        let src = "aaa\nbbb\nccc\n";
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 8, 11))
            .with_label(Span::new(FileId::DUMMY, 11, 11), "here")
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(out.contains("here"), "zero-width label dropped:\n{out}");
        assert_eq!(out.matches("here").count(), 1, "message duplicated:\n{out}");
        // The label no longer falls back to an `at line:col` note.
        assert!(!out.contains("at 3:4"), "label fell back to a note:\n{out}");
        // Primary `^^^` plus the insertion-point caret share one lane.
        assert!(out.contains("^^^^"), "missing one-cell marker:\n{out}");
        assert!(out.contains("| here"), "message not inline:\n{out}");
    }

    /// zero-width annotations render as one-cell markers at
    /// the point itself — line start, line middle, and line end (including
    /// the final character position of a file without a trailing newline).
    #[test]
    fn zero_width_annotations_render_one_cell_markers() {
        let src = "aaa\nbbb\nccc";
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_label(Span::new(FileId::DUMMY, 4, 4), "line start")
            .with_label(Span::new(FileId::DUMMY, 9, 9), "middle")
            .with_label(Span::new(FileId::DUMMY, 11, 11), "end")
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        // Every point renders inline next to its one-cell marker...
        for msg in ["line start", "middle", "end"] {
            assert!(
                out.contains(&format!("| {msg}")),
                "marker message {msg:?} not inline:\n{out}"
            );
            assert_eq!(out.matches(msg).count(), 1, "{msg:?} duplicated:\n{out}");
        }
        // ...and none falls back to an `at line:col` note.
        for locator in ["at 2:1", "at 3:2", "at 3:4"] {
            assert!(
                !out.contains(locator),
                "unexpected fallback {locator:?}:\n{out}"
            );
        }
    }

    /// Multi-file guard: a primary span from another file has meaningless
    /// offsets in this diagnostic's buffer; it must not draw a section or a
    /// caret here, but still stays visible as a location-only note.
    #[test]
    fn foreign_primary_spans_render_as_location_only_notes() {
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::from_raw(1), 0, 3))
            .with_additional_span(Span::new(FileId::from_raw(2), 0, 3))
            .with_source("def x\nset y\n");
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert_eq!(
            out.matches("<input>:1:1").count(),
            1,
            "foreign span drew its own section:\n{out}"
        );
        assert_eq!(
            out.matches("^^^").count(),
            1,
            "foreign caret drawn in this buffer:\n{out}"
        );
        assert!(
            out.contains("at 0..3"),
            "foreign location note missing:\n{out}"
        );
    }

    /// Multi-file guard: a label from another file cannot be located in this
    /// buffer, so it falls back to a range-located note; anchor-file labels
    /// keep their inline rendering.
    #[test]
    fn foreign_labels_fall_back_to_range_notes() {
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::from_raw(1), 0, 3))
            .with_label(Span::new(FileId::from_raw(1), 0, 3), "here")
            .with_label(Span::new(FileId::from_raw(2), 500, 505), "elsewhere")
            .with_source("def x\nset y\n");
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(out.contains("here"), "anchor label dropped:\n{out}");
        assert!(
            out.contains("elsewhere at 500..505"),
            "range-located note missing:\n{out}"
        );
        assert_eq!(
            out.matches("elsewhere").count(),
            1,
            "foreign label rendered twice:\n{out}"
        );
    }

    /// Regression (#2): `write_child` sanitizes the plain message (control →
    /// U+FFFD changes byte lengths) but used to slice the *unsanitized*
    /// styled parts with offsets derived from the sanitized text — a
    /// mid-char slice panic. The long CJK run forces wrapping past the
    /// replaced control, which is what put a drifted offset on a slice
    /// boundary.
    #[test]
    fn control_char_in_styled_note_keeps_styled_offsets() {
        let styled = crate::diagnostics::StyledString::new(vec![
            crate::diagnostics::StringPart {
                content: format!("x\u{7} {}", "中".repeat(300)).into(),
                style: crate::diagnostics::StringPartStyle::Highlighted,
            },
            crate::diagnostics::StringPart {
                content: " tail".into(),
                style: crate::diagnostics::StringPartStyle::Normal,
            },
        ]);
        let child = crate::diagnostics::Subdiag {
            level: DiagnosticLevel::Note,
            message: styled.to_plain().into(),
            styled_message: Some(styled),
            span: None,
            labels: Vec::new(),
        };
        let mut out = String::new();
        GlyphRenderer::new(false).write_child(&mut out, &child);
        assert!(out.contains('x'), "styled text mangled:\n{out}");
        assert!(out.contains('中'), "styled text mangled:\n{out}");
        assert!(out.contains("tail"), "styled text mangled:\n{out}");
    }

    /// A diagnostic whose primary spans are far apart renders each span in
    /// its own source section; the sections are joined by a `…` marker so
    /// the omitted rows are explicit (previously the reader had to infer
    /// the jump from the line-number column alone).
    #[test]
    fn distant_spans_are_joined_by_a_fold_marker() {
        // 200 lines, each 9 bytes ("line NNN\n"), so line N starts at N*9.
        let mut source = String::with_capacity(200 * 9);
        for i in 0..200 {
            source.push_str(&format!("line {i:03}\n"));
        }
        let diag = Diagnostic::error("two distant spans")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 4))
            .with_additional_span(Span::new(FileId::DUMMY, 100 * 9, 100 * 9 + 4))
            .with_source(&source);
        let out = GlyphRenderer::new(true).render_diagnostic(&diag);

        // One main header (`┌─ error…`) plus two section headers (`┌─ <input>…`).
        let header_positions: Vec<usize> = out.match_indices("┌─").map(|(p, _)| p).collect();
        assert_eq!(
            header_positions.len(),
            3,
            "expected 3 headers (1 main + 2 sections):\n{out}"
        );
        assert_eq!(
            out.matches("...").count(),
            1,
            "expected exactly one fold marker:\n{out}"
        );
        // The marker must sit between the two section headers, not before
        // the first one.  header_positions[0] is the main header,
        // [1] is section 1, [2] is section 2.
        let marker = out.find("...").expect("fold marker");
        assert!(
            header_positions[1] < marker && marker < header_positions[2],
            "fold marker out of place:\n{out}"
        );
    }

    /// The common single-region case must not gain a stray `...` line.
    #[test]
    fn single_region_has_no_fold_marker() {
        let diag = Diagnostic::error("one region")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_source("aaa\nbbb\nccc\n");
        let out = GlyphRenderer::new(true).render_diagnostic(&diag);
        assert!(!out.contains("..."), "single region must not fold:\n{out}");
    }

    /// Two spans close enough to be merged into one section (their context
    /// windows overlap) also must not produce a fold marker — the merge
    /// already eliminated the gap.
    #[test]
    fn merged_spans_do_not_fold() {
        // Spans on lines 2 and 4 of a short file: `context_lines = 2` makes
        // both context windows overlap, so they merge into a single section.
        let source = "l0\nl1\nl2\nl3\nl4\nl5\nl6\n";
        let diag = Diagnostic::error("merged")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 6, 8))
            .with_additional_span(Span::new(FileId::DUMMY, 12, 14))
            .with_source(source);
        let out = GlyphRenderer::new(true).render_diagnostic(&diag);
        // One main header + one section header (the two spans merged).
        assert_eq!(
            out.matches("┌─").count(),
            2,
            "expected 2 headers (1 main + 1 merged section):\n{out}"
        );
        assert!(!out.contains("..."), "merged spans must not fold:\n{out}");
    }

    /// Regression for merged-section carets: each real primary keeps its
    /// own `^` row; the intermediate line between two merged spans (e.g.
    /// a "duplicate definition" with this/previous locations) must NOT
    /// get a full-width caret for the byte-union of the two spans.
    #[test]
    fn merged_spans_keep_separate_carets() {
        // Spans on lines 2 and 4, with the untouched line 3 between them.
        let source = "l0\nl1\nl2\nl3\nl4\nl5\nl6\n";
        let diag = Diagnostic::error("merged")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 6, 8)) // `l2` on line 2
            .with_additional_span(Span::new(FileId::DUMMY, 12, 14)) // `l4` on line 4
            .with_source(source);
        let out = GlyphRenderer::new(true).render_diagnostic(&diag);
        let lines: Vec<&str> = out.lines().collect();
        // Exactly two caret rows — one per real span, none for a union.
        assert_eq!(
            lines.iter().filter(|l| l.contains('^')).count(),
            2,
            "expected one caret row per real span:\n{out}"
        );
        // Each real span's line is followed by its caret row.
        for want in ["l2", "l4"] {
            let i = lines.iter().position(|l| l.contains(want)).expect(want);
            assert!(
                lines[i + 1].contains('^'),
                "span on `{want}` must keep its own caret:\n{out}"
            );
        }
        // The intermediate line has no caret row of its own.
        let i = lines.iter().position(|l| l.contains("l3")).expect("l3");
        assert!(
            !lines[i + 1].contains('^'),
            "intermediate line must not get a union caret:\n{out}"
        );
    }

    // ── P2 layout tests (compact mode, fold counts, width unify, inline labels) ──

    /// Item 16: compact mode drops the sub-box header prefix, the spacing
    /// row, and the close row; the full layout keeps all three.
    #[test]
    fn compact_mode_drops_box_furniture() {
        let source = "let x = 1;\n";
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_source(source);

        let full = GlyphRenderer::with_opts(
            false,
            Some(&crate::diagnostics::DiagOpts {
                compact: false,
                ..Default::default()
            }),
        )
        .render_diagnostic(&diag);
        // Full layout: the sub-box header (`┌─ <input>:1:1` in Unicode,
        // `+- <input>:1:1` in ASCII) and the close row survive.
        let plain_full = strip_ansi(&full);
        assert!(
            plain_full.contains("┌─") || plain_full.contains("+-"),
            "full layout must keep the sub-box header: {plain_full}"
        );
        assert!(
            plain_full.contains("└─") || plain_full.contains("\\-"),
            "full layout must keep the close row: {plain_full}"
        );
        assert!(
            plain_full.contains("<input>"),
            "location missing in full layout: {plain_full}"
        );
        // The spacing row is `│` + indent + `│` (indent = `line_num_width + 1`
        // spaces, at least 2) — a row with only gutter chars and no source text.
        let spacing_present = plain_full
            .lines()
            .any(|l| {
                let t = l.trim();
                (t == "│ │" || t == "│  │" || t == "│   │" || (t.starts_with("| ") && t.ends_with('|')))
                    && !t.contains('x')
            });
        assert!(
            spacing_present,
            "full layout must keep the sub-box spacing row: {plain_full}"
        );

        let compact = GlyphRenderer::with_opts(
            false,
            Some(&crate::diagnostics::DiagOpts {
                compact: true,
                ..Default::default()
            }),
        )
        .render_diagnostic(&diag);
        let plain_compact = strip_ansi(&compact);
        // Compact mode: the sub-box header prefix (`┌─ ` in Unicode, `+- `
        // in ASCII) is dropped — the location line sits directly after the
        // outer box's gutter.  The close row is also dropped.
        assert!(
            plain_compact.contains("<input>"),
            "compact mode still shows the location: {plain_compact}"
        );
        // The sub-box header in compact mode has no box-drawing prefix:
        // the location line is `│   <input>:1:1` (gutter + location),
        // not `│  ┌─ <input>:1:1` (gutter + box prefix + location).
        let location_line = plain_compact
            .lines()
            .find(|l| l.contains("<input>"))
            .expect("location line missing");
        assert!(
            !location_line.contains("┌─") && !location_line.contains("+-"),
            "compact mode must drop the sub-box header prefix: {location_line:?}\n{plain_compact}"
        );
        // The close row (`└─` / `\-`) is also dropped.
        assert!(
            !plain_compact.lines().any(|l| l.contains("└─") || l.contains("\\-")),
            "compact mode must drop the close row:\n{plain_compact}"
        );
        // The source row and the underline survive both layouts.
        assert!(
            plain_compact.contains("let x = 1;"),
            "compact dropped the source: {plain_compact}"
        );
        assert!(
            plain_compact.contains("^^^"),
            "compact dropped the underline: {plain_compact}"
        );
    }

    /// Item 18: a gap between disjoint sections gets a line count in the
    /// fold marker; contiguous sections get a bare `...`.
    #[test]
    fn fold_marker_reports_omitted_line_count() {
        // 60 lines of source; two spans 50 lines apart → a real gap that
        // no amount of context bridging can fold into one section.
        let source = "fn f() {\n".to_string() + &"line\n".repeat(58) + "}\n";
        let mid = source.find("line\n").unwrap() + 50 * 5; // ~50 lines in
        let mid = mid as u32;
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 4))
            .with_additional_span(Span::new(FileId::DUMMY, mid, mid + 4))
            .with_source(&source);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            out.contains("lines omitted"),
            "gap between sections must report the omitted line count: {out}"
        );
    }

    /// Item 19: sections keep one shared line-number column width even when
    /// the file's last line only appears in the final section.
    #[test]
    fn line_number_width_is_unified_across_sections() {
        // Span 1 is early (1-digit line numbers), span 2 lands past line
        // 10 (2-digit) — the first section's gutter must still be 2 wide.
        let source = "a\n".repeat(15);
        let end = source.len() as u32 - 1;
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_additional_span(Span::new(FileId::DUMMY, end, end + 1))
            .with_source(&source);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        let plain = strip_ansi(&out);
        let lines: Vec<&str> = plain.lines().collect();
        let plain_full = plain.clone();
        // The two spans are 14 lines apart with 2 context lines: the
        // sections stay disjoint and get a counted fold marker between
        // them — that marker is also the layout change's canary.
        assert!(
            lines.iter().any(|l| l.contains("lines omitted")),
            "expected two sections joined by a counted fold marker:\n{plain_full}"
        );
        // Find the source rows for line 1 (section 1) and line 15
        // (section 2).  The row gutter is `│` + 2-wide line-number
        // column + ` ` + `│`, so the digit sits at a fixed offset from
        // the row's left edge — the offsets must agree across sections.
        let row1: &str = lines
            .iter()
            .find(|l| l.contains("| a"))
            .copied()
            .unwrap_or("");
        let row15: &str = lines
            .iter()
            .rev()
            .find(|l| l.contains("| a"))
            .copied()
            .unwrap_or("");
        assert!(!row1.is_empty(), "line-1 source row missing:\n{plain_full}");
        assert!(!row15.is_empty(), "line-15 source row missing:\n{plain_full}");
        // Both rows are right-aligned in the same 2-wide column: the
        // source content begins at the same display column in both
        // sections (a narrower first-section gutter would shift it left).
        let col1 = row1.find("| a").unwrap();
        let col15 = row15.find("| a").unwrap();
        assert_eq!(
            col1, col15,
            "line-number columns drift between sections: {row1:?} vs {row15:?}\n{plain_full}"
        );
    }

    /// Item 17: a single non-multiline label renders its message inline
    /// (no vertical-bar connector row); two labels on the same row keep
    /// the connector layout.
    #[test]
    fn single_label_message_is_inline() {
        let source = "let x = 1;\n";
        let diag = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_label(Span::new(FileId::DUMMY, 0, 3), "the type of this expression")
            .with_source(source);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        let plain = strip_ansi(&out);
        // Inline form: the message text sits on the row directly below the
        // underline, with no `|` connector between them.
        assert!(
            plain.contains("the type of this expression"),
            "message missing: {plain}"
        );
        let msg_line = plain
            .lines()
            .find(|l| l.contains("the type of this expression"))
            .unwrap();
        // Inline → `|  | the …` (gutter then message).  A connector row
        // would show `|  | | the …` — a space-pipe-space run.
        assert!(
            !msg_line.contains("| |"),
            "single label must be inline, not on a connector row: {msg_line:?}\n{plain}"
        );

        // Two labels on the same row: the connector layout is preserved.
        let diag2 = Diagnostic::error("boom")
            .with_code_str("E030")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_label(Span::new(FileId::DUMMY, 0, 3), "first")
            .with_label(Span::new(FileId::DUMMY, 3, 3), "second")
            .with_source(source);
        let out2 = strip_ansi(&GlyphRenderer::new(false).render_diagnostic(&diag2));
        let second_line = out2
            .lines()
            .find(|l| l.contains("second"))
            .expect("second label message missing");
        // Two labels on the same row → the vertical-bar connector layout:
        // the message is on its own line, preceded by a `|` connector
        // column (a space-pipe run inside the gutter).
        assert!(
            second_line.trim_start().starts_with('|'),
            "multi-label rows keep the connector: {second_line:?}\n{out2}"
        );
    }

    #[test]
    fn underline_lanes_collapse_identical_annotations() {
        // The primary span and a Primary label on the same span yield two
        // identical `(start, len, '^')` annotations.  They must share one
        // lane instead of stacking two identical `^^^` rows; a third,
        // genuinely overlapping annotation still opens a second lane.
        let underlines = [
            (4usize, 6usize, '^', "", None),
            (4usize, 6usize, '^', "same span", Some(0)),
            (5usize, 10usize, '^', "overlap", Some(1)),
        ];
        let mut lanes = UnderlineLanes::new();
        let count = lanes.assign(&underlines);
        assert_eq!(count, 2, "identical annotations must collapse to one lane");
    }
}
