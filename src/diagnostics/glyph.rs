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
use crate::diagnostics::Applicability;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::DisplayArg;
use crate::diagnostics::LineIndex;
use crate::diagnostics::Suggestion;
use crate::diagnostics::label::{AnnotationKind, Label, SourcePos};
use crate::diagnostics::level::DiagnosticLevel;
use crate::diagnostics::render::RenderCtx;
use crate::diagnostics::renderers::fmt::{push_spaces, push_uint, push_uint_padded};
use crate::diagnostics::term::path_to_file_url;
use crate::smallvec::{SmallVec, smallvec};
use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering as AtomicOrdering};
use swrite::{SWrite, swrite};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

// ── Box-drawing characters ──────────────────────────────────────

/// Display width of the box-drawing verticals `v` / `sub_v`: `│` U+2502
/// (UNICODE) and `|` U+007C (ASCII) are each exactly one terminal cell.
/// Hoisted as a named constant so the render-path call sites stay
/// self-documenting and the invariant has a single source instead of
/// scattered magic `1`s that would silently drift if the charset changed.
const BOX_VERT_WIDTH: usize = 1;

struct BoxChars {
    tl: &'static str,     // top-left corner
    bl: &'static str,     // bottom-left corner
    h: &'static str,      // horizontal line
    v: &'static str,      // vertical line
    sub_tl: &'static str, // sub-box top-left
    sub_bl: &'static str, // sub-box bottom-left
    sub_v: &'static str,  // sub-box vertical
    sub_h: &'static str,  // sub-box horizontal
    /// `sub_tl` + `sub_h` pre-concatenated: the sub-box header prefix (`┌─`
    /// / `+-`) is a compile-time constant, so the per-section location header
    /// borrows it instead of `format!`-ing a fresh 2-char `String` every
    /// section.
    sub_tl_h: &'static str,
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
    sub_tl_h: "┌─",
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
    sub_tl_h: "+-",
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
const ANSI_RESET: &str = "\x1b[0m";

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
    /// The reset that *pairs* with [`Self::highlight_bg`].  256-colour /
    /// truecolour spans end with `\x1b[49m` (default background); a 16-colour
    /// span is SGR 7 (reverse video), which SGR 49 does **not** clear — the
    /// correct reset is SGR 27 (`\x1b[27m`).  Using the background reset on
    /// 16-colour terminals leaked reverse video across the rest of the row
    /// and every later row drawn with the same escape.
    highlight_reset: &'static str,
}

impl Styles {
    const fn new(use_color: bool, color_depth: super::term::ColorDepth) -> Self {
        if use_color {
            let (highlight_bg, highlight_reset) = match color_depth {
                super::term::ColorDepth::TwoFiftySix | super::term::ColorDepth::TrueColor => {
                    ("\x1b[48;5;236m", "\x1b[49m")
                }
                // Reverse video pairs with SGR 27, not the background reset.
                super::term::ColorDepth::Sixteen => ("\x1b[7m", "\x1b[27m"),
            };
            Styles {
                use_color: true,
                reset: ANSI_RESET,
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
                highlight_reset,
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
                highlight_reset: "",
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

/// What a source section actually drew. `drew_primary` means the primary
/// span's start line was inside the drawn window — the only case in which the
/// location is visible and the caller may suppress its fallback note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SectionOutcome {
    drew_primary: bool,
}

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
    quiet: bool,
    no_wrap: bool,
}

/// Per-section scratch buffers, reused across all sections of a render
/// instead of reallocated per section.  Threaded through
/// [`GlyphRenderer::write_source_section`] so a diagnostic with K primary-span
/// clusters pays ~13 allocations once, not ~13·K.
///
/// The `String` fields (`connector`, `hl_buf`) are
/// `mem::take`-n into owned locals at the top of `write_source_section` and
/// restored on exit (including the `source.is_empty()` early return), so
/// format macros can use them by value.  The remaining fields are used via
/// `&mut` references — their `clear()`/`push()` methods preserve capacity.
pub(crate) struct SectionScratch {
    // ── Already hoisted (SmallVec, inline for ≤4) ─────────────
    pub ranges: SmallVec<[(usize, usize); 4]>,
    pub extra_lines: SmallVec<[(usize, usize); 4]>,
    pub all_label_lines: SmallVec<[(usize, usize); 4]>,
    // ── Row-loop scratch (reused across rows within a section) ─
    // `row_buf` is used by the HTML renderer; the glyph path appends rows
    // directly to its output and no longer copies through it.
    pub row_buf: String,
    /// Pre-built per-row gutter fragments (`│ <num> ` head/tail), cleared and
    /// refilled once per section instead of allocating twice per section.
    pub gutter_head: String,
    pub gutter_tail: String,
    pub connector: String,
    pub lanes: UnderlineLanes,
    pub multiline_flags: SmallVec<[(usize, &'static str); 4]>,
    /// Display column of each label's start row, captured from the start-row
    /// underline record so middle connector rows do not re-walk the line.
    pub label_start_cols: SmallVec<[Option<usize>; 8]>,
    pub ranges_buf: Vec<(usize, usize)>,
    pub display_buf: Vec<(usize, usize)>,
    pub order_buf: Vec<usize>,
    pub done_buf: Vec<bool>,
    pub active_buf: BinaryHeap<Reverse<(usize, usize)>>,
    /// Indent string for the source section's line-number gutter — cleared and
    /// refilled once per section, capacity reused across sections instead of a
    /// fresh `" ".repeat(line_num_width + 1)` allocation per section.
    pub indent: String,
    /// Per-foreign-file suggestion line ranges — cleared and resized to
    /// `diag.suggestions.len()` per diagnostic; capacity survives across the
    /// batch.  `Vec` (not `SmallVec`) because it is indexed by suggestion
    /// number and must cover every suggestion even when most are `None`.
    pub suggestion_lines_buf: Vec<Option<(usize, usize)>>,
    /// Scratch for the highlighter's `_into` entry points where the
    /// highlighted text must be assembled *before* it can be pushed between
    /// gutter pieces (the connector re-colouring pass scans the highlighted
    /// text for reset escapes).  `mem::take`-n per section, restored on exit.
    pub hl_buf: String,
    /// `(start_line, end_line)` per anchor label, cleared and refilled per
    /// diagnostic; capacity reused across the batch.
    pub section_label_lines: SmallVec<[(usize, usize); 4]>,
    /// Index into `merged_labels` per anchor label, same reuse pattern.
    pub section_label_sources: SmallVec<[usize; 4]>,
    /// Start-line-sorted permutation of `section_label_lines`, shared across
    /// every section of one diagnostic; capacity reused across the batch.
    pub section_label_order: SmallVec<[usize; 4]>,
    /// Precomputed `span_lines` per suggestion whose span belongs to the
    /// anchor file (`None` for foreign files); capacity reused.
    pub anchor_suggestion_lines: SmallVec<[Option<(usize, usize)>; 4]>,
    /// Reusable [`LabelWindow`] scratch, threaded through
    /// [`GlyphRenderer::write_source_section`] so a diagnostic with K source
    /// sections pays one set of `SmallVec` / `BinaryHeap` capacities, not K.
    /// `LabelWindow::new` takes it by value and `LabelWindow::into_scratch`
    /// hands it back after the section renders.
    pub label_window: LabelWindowScratch,
    /// Per-row underline records, `(display_col, width, char, label_idx)`.
    /// The message payload was split out (resolved at use sites from the
    /// caller's `all_labels` slice via `label_idx`) exactly so this buffer —
    /// and `meta_buf` below — carries no borrow into the per-section label
    /// list and can be reused across the batch.
    pub underlines_buf: SmallVec<[(usize, usize, char, Option<usize>); 8]>,
    /// Per-row annotation metadata, `(underline_char, label_idx)`; see
    /// [`underlines_buf`](Self::underlines_buf).
    pub meta_buf: SmallVec<[(char, Option<usize>); 8]>,
}

impl SectionScratch {
    pub(crate) fn new() -> Self {
        Self {
            ranges: SmallVec::new(),
            extra_lines: SmallVec::new(),
            all_label_lines: SmallVec::new(),
            row_buf: String::new(),
            gutter_head: String::new(),
            gutter_tail: String::new(),
            connector: String::new(),
            lanes: UnderlineLanes::new(),
            multiline_flags: SmallVec::new(),
            label_start_cols: SmallVec::new(),
            ranges_buf: Vec::new(),
            display_buf: Vec::new(),
            order_buf: Vec::new(),
            done_buf: Vec::new(),
            active_buf: BinaryHeap::new(),
            indent: String::new(),
            hl_buf: String::new(),
            suggestion_lines_buf: Vec::new(),
            section_label_lines: SmallVec::new(),
            section_label_sources: SmallVec::new(),
            section_label_order: SmallVec::new(),
            anchor_suggestion_lines: SmallVec::new(),
            label_window: LabelWindowScratch::new(),
            underlines_buf: SmallVec::new(),
            meta_buf: SmallVec::new(),
        }
    }
}

/// Re-apply `color` after every `\x1b[22m\x1b[39m` pair injected by the
/// code highlighter, writing straight into `out`.  Shared by the inline
/// and connector label-message paths so the two cannot drift.
fn push_recologored(out: &mut String, text: &str, color: &str) {
    const PAT: &str = "\x1b[22m\x1b[39m";
    let mut rest = text;
    while let Some(p) = rest.find(PAT) {
        out.push_str(&rest[..p]);
        out.push_str(PAT);
        out.push_str(color);
        rest = &rest[p + PAT.len()..];
    }
    out.push_str(rest);
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

    /// Whether this renderer emits ANSI colour codes.
    pub(crate) fn use_color(&self) -> bool {
        self.use_color
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
        let (context_lines, max_width, compact, quiet, no_wrap) = match opts {
            Some(o) => (o.context_lines, o.max_width, o.compact, o.quiet, o.no_wrap),
            None => (
                fallback.context_lines,
                fallback.max_width,
                fallback.compact,
                false,
                false,
            ),
        };
        // Clamp, don't panic: `max_width` can come from a user CLI flag
        // (`--width 15`, or an absurd `--width 1e19`) or a mis-detecting
        // terminal, and a diagnostics renderer must never abort the compiler
        // over its own width.  The *upper* clamp matters too: past
        // `usize::MAX / 4` the fitness-class arithmetic would need widening,
        // and `intrinsic_badness`'s `as f64` loses precision — no real
        // terminal is wider than this bound.
        const MAX_RENDER_WIDTH: usize = 8 * super::DEFAULT_MAX_WIDTH;
        let max_width = max_width.max(Self::MIN_WIDTH).min(MAX_RENDER_WIDTH);
        // `context_lines` is a user CLI knob too — clamp it so the
        // `sp_last + context_lines` window arithmetic cannot overflow.
        let context_lines = context_lines.min(Self::MAX_CONTEXT_LINES);
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
        // A user CLI `--tab-width 0` would divide by zero in
        // `grapheme_width_cached`'s `tab_width - (col % tab_width)`; clamp to
        // 1 the same way `max_width` is clamped to `MIN_WIDTH`.  One cell per
        // tab is degenerate but sound.
        let tab_width = match opts {
            Some(o) => o.tab_width.max(1),
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
            quiet,
            no_wrap,
        }
    }

    /// Visible width of the `│  = <label>: ` gutter that precedes child /
    /// note / help messages, for a given level. The box char and fixed
    /// separators are folded into `prefix_base`; only the label width varies.
    /// `visible_width`, not `.len()`: `label()` is localized (ru/zh labels are
    /// multi-byte), and byte length ≠ display cells — using bytes inflated
    /// the gutter and misaligned wrapped continuation lines in non-English
    /// locales.
    fn prefix_cols(&self, level: DiagnosticLevel) -> usize {
        self.prefix_base + level_label_widths()[level.severity() as usize]
    }

    /// Upper bound for `context_lines`.  A CLI `--context 1e19` parses as
    /// `usize::MAX`; the window computations add it to a line number, which
    /// would overflow (panic in debug, wrap in release → wrong window).  No
    /// real render wants more than this many context rows.
    const MAX_CONTEXT_LINES: usize = 1 << 16;

    pub fn with_context_lines(mut self, n: usize) -> Self {
        self.context_lines = n.min(Self::MAX_CONTEXT_LINES);
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
    /// open/reset pair and no SGR run is ever cut in half.  The [`SmallVec`]
    /// return keeps the common 1–2-line case inline (no heap allocation).
    ///
    /// (A caller-supplied-buffer `wrap_msg_into` was tried and removed: the
    /// drop check on `SmallVec<Cow<'a>>` pins the buffer's scope to each
    /// wrapped message's lifetime — the per-message sources here are
    /// loop-locals — so cross-message reuse is impossible without turning
    /// every segment owned.  The inline `SmallVec` already keeps the common
    /// case allocation-free.)
    fn wrap_msg<'a>(&self, msg: &'a str, prefix_cols: usize) -> SmallVec<[Cow<'a, str>; 4]> {
        if self.no_wrap {
            // `split_terminator`, not `split`: `wrap_optimal` treats a trailing
            // newline as a terminator, so `"a\n"` is one line either way.
            // With `split` the no-wrap path emitted a phantom blank gutter row,
            // i.e. toggling `--no-wrap` changed the line *structure*, not just
            // the wrapping.
            return msg.split_terminator('\n').map(Cow::Borrowed).collect();
        }
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
        let mut out = String::new();
        self.render_diagnostic_into(diag, idx, None, &mut out);
        out
    }

    /// Like [`Self::render_diagnostic_with_index`], but appends into an
    /// existing buffer.
    ///
    /// The human renderer owns the diagnostic-body buffer and appends the
    /// call chain to it, so writing through avoids one `String` per
    /// diagnostic.  `cx` is the per-batch render context: when present, its
    /// line-index cache is reused for foreign-file sections (each distinct
    /// foreign source is scanned at most once per batch instead of once per
    /// diagnostic that references it); `None` falls back to a fresh
    /// [`LineIndex::new`] per foreign file.
    pub(crate) fn render_diagnostic_into<'src>(
        &self,
        diag: &Diagnostic<'src>,
        idx: Option<&LineIndex<'_>>,
        cx: Option<&RenderCtx<'_>>,
        out: &mut String,
    ) {
        // Rendered output is dominated by the context windows (a handful of
        // source rows), not the whole file, so derive a hint from the source
        // size but cap it — a 1 MB file must not reserve tens of KB of buffer.
        let hint = diag
            .source
            .as_ref()
            .map_or(256, |s| (s.len() / 8).clamp(256, 8 * 1024));
        out.reserve(hint);

        // Per-suggestion "rendered as a diff block" flags, written back by
        // `write_source_section` so the fallback loop below can skip them.
        let mut suggestion_rendered: SmallVec<[bool; 4]> =
            SmallVec::from_elem(false, diag.suggestions.len());

        // ┌─ error[E003]: message
        self.write_header(out, diag);

        if let Some(source) = diag.source {
            // Borrowed view of the primary labels plus a small local stash for
            // the synthetic ones built from related errors.  The old code
            // cloned every `Label` into a `Vec<Label>`, which allocates a
            // `String` for each `Cow::Owned` message once per rendered
            // diagnostic; the renderer only reads them, so references suffice.
            let mut synthetic: SmallVec<[Label<'src>; 2]> = SmallVec::new();
            for rel in &diag.related_errors {
                if let Some(rel_span) = rel.span {
                    let label_text = match rel.code.as_ref() {
                        Some(code) => {
                            // Manual pushes, not `format!`: skips the
                            // `core::fmt` machinery per related error with a
                            // code (common in merged multi-error diagnostics).
                            let mut s =
                                String::with_capacity(rel.message.len() + code.code().len() + 4);
                            s.push_str(&rel.message);
                            s.push_str(" [");
                            s.push_str(code.code());
                            s.push(']');
                            s.into()
                        }
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
            // A merged group almost always has ≤2 real primaries, so inline
            // storage avoids a heap allocation per section.
            let mut extras_labels: SmallVec<[Label<'src>; 2]> = SmallVec::new();
            // Per-section scratch buffers, reused across all sections of this
            // render (anchor-file and foreign) instead of reallocated per
            // section.  When a `RenderCtx` is supplied, its `scratch` cell
            // amortises the 13 buffers across the whole batch (grown once on
            // the first section, reused by every later diagnostic); the no-ctx
            // path (`render_diagnostic` / `render_diagnostic_with_index`)
            // falls back to a fresh local.  See [`SectionScratch`] for the
            // field layout.
            let mut local_scratch: SectionScratch;
            let mut scratch_guard = cx.map(|c| c.scratch.borrow_mut());
            let section_scratch: &mut SectionScratch = match scratch_guard.as_deref_mut() {
                Some(s) => s,
                None => {
                    local_scratch = SectionScratch::new();
                    &mut local_scratch
                }
            };
            // Hoist the pure-value scratch fields that `write_source_section`
            // receives by `&` while we also pass `&mut section_scratch`.
            // `mem::take` moves the (already-allocated) container into a local
            // and leaves an empty `Default` in the scratch; the local's
            // capacity is moved back at the end of this `if let Some(source)`
            // block.  No allocation happens on take or restore.
            //
            // `synthetic` / `merged_labels` / `section_labels` are NOT moved
            // here: `merged_labels` borrows `synthetic`, so co-locating them
            // in one struct would be self-referential (a field borrowing a
            // sibling), which safe Rust cannot express; and `section_labels`
            // borrows `synthetic`, so `&section_labels` and
            // `&mut section_scratch` would alias the same struct.  Both are
            // small `SmallVec`s (≤2 / ≤4 inline), so the common case has no
            // heap allocation to amortise anyway.
            // `suggestion_lines_buf` is taken inside the foreign gate below —
            // it is only read on the multi-file path, so the common
            // single-file case skips the take/restore entirely.
            let mut section_label_lines = std::mem::take(&mut section_scratch.section_label_lines);
            let mut section_label_sources =
                std::mem::take(&mut section_scratch.section_label_sources);
            let mut section_label_order = std::mem::take(&mut section_scratch.section_label_order);
            let mut anchor_suggestion_lines =
                std::mem::take(&mut section_scratch.anchor_suggestion_lines);
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
            let sorted_spans = diag.spans.primary_ref();
            // Defense in depth: `primary_ref` sorts only on the *dirty* path,
            // so a producer that mutated the primary list into an unsorted
            // state without setting the dirty flag would silently corrupt the
            // merge predicate, fold counts and `group[0]` selection in release
            // builds.  Sorting a (typically ≤2-element) slice is cheap; do it
            // whenever the borrowed view is not already ordered.
            let sorted_spans: Cow<'_, [Span]> = if sorted_spans.windows(2).all(|w| w[0] <= w[1]) {
                sorted_spans
            } else {
                debug_assert!(false, "MultiSpan::primary must be sorted by (file, start)");
                let mut out: SmallVec<[Span; 2]> = sorted_spans.iter().copied().collect();
                out.sort_unstable();
                out.dedup();
                Cow::Owned(out.into_vec())
            };

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
            // ── Multi-file guard ───────────────────────────────────
            // Source sections can only locate spans inside `diag.source`;
            // the emitter contract is that the source belongs to the first
            // primary span's file, so that file is the anchor.  Line/col
            // numbers of spans from other files would be meaningless
            // against this buffer, so only anchor-file spans and labels
            // render in sections; the rest stay visible as byte-range
            // notes after the sections.
            // Every coordinate consumer below is gated on `span.file == anchor`: labels
            // (755/776), `additionalSpans` (766), related-error synthetic labels (796)
            // and the underlines (809).  That is what keeps a foreign-file span — whose
            // offsets are valid only in *its* buffer — from drawing markers on this
            // excerpt; `json.rs` documents the same hazard for its coordinates.  The
            // audit that asked for this note found the gate present at all four sites,
            // so it is stated here as the invariant to preserve rather than re-derived.
            let anchor_file = sorted_spans.first().map(|sp| sp.file);
            // Each merged label's (start_line, end_line), converted once
            // per render and shared by every source section below.
            // Foreign-file labels are skipped here: their `span_lines`
            // result is never read (the filter below drops them, and the
            // foreign-file section loop recomputes its own `f_label_lines`
            // against that file's index), so computing them was pure
            // O(#foreign · log n) waste on multi-file diagnostics.
            let merged_label_lines: SmallVec<[(usize, usize); 4]> = merged_labels
                .iter()
                .map(|lbl| {
                    if anchor_file.is_none_or(|anchor| lbl.span.file == anchor) {
                        idx.span_lines(lbl.span)
                    } else {
                        (0, 0) // filtered out below; never read
                    }
                })
                .collect();

            let foreign_spans: SmallVec<[Span; 2]> = sorted_spans
                .iter()
                .copied()
                .filter(|sp| anchor_file.is_some_and(|anchor| sp.file != anchor))
                .collect();
            // Anchor-file view of `merged_labels`; each kept label also
            // remembers its index into `label_rendered`, so a section can
            // mark exactly the label it rendered even when earlier labels
            // were filtered out.
            let mut section_labels: SmallVec<[&Label<'src>; 4]> = SmallVec::new();
            section_label_lines.clear();
            section_label_sources.clear();
            for (i, lbl) in merged_labels.iter().enumerate() {
                if anchor_file.is_none_or(|anchor| lbl.span.file == anchor) {
                    section_labels.push(lbl);
                    section_label_lines.push(merged_label_lines[i]);
                    section_label_sources.push(i);
                }
            }
            // Start-line-sorted permutation of the anchor labels, built once
            // and shared by every section so `LabelWindow` merges the
            // per-section extras into a pre-sorted order instead of re-sorting
            // the whole union per section — removing the `K·L·log L` term
            // from a diagnostic with K sections and L shared labels.
            section_label_order.clear();
            section_label_order.extend(0..section_label_lines.len());
            section_label_order.sort_by_key(|&i| section_label_lines[i].0);
            // Per-suggestion `(sug_first, sug_last)` against the anchor file's
            // index, precomputed once so each anchor-file section's suggestion
            // loop is a cheap range test instead of an `O(log n)` `span_lines`
            // per section per suggestion — removing the `K·S·log n` term.
            anchor_suggestion_lines.clear();
            anchor_suggestion_lines.extend(diag.suggestions.iter().map(|s| match s.span {
                Some(span) if anchor_file.is_none_or(|a| span.file == a) => {
                    Some(idx.span_lines(span))
                }
                _ => None,
            }));

            // Build (line_range, span) pairs for merging. The line count
            // comes from the shared index (`str::lines`-equivalent) so the
            // whole file is never materialized as a `Vec<&str>` just to take
            // its length.
            let last_text_line = idx.text_line_count().saturating_sub(1);
            let ranges: SmallVec<[((usize, usize), SourcePos, Span); 2]> = sorted_spans
                .iter()
                .filter(|sp| anchor_file.is_none_or(|anchor| sp.file == anchor))
                .map(|&sp| {
                    let start_pos = idx.to_linecol(sp.start_as_usize());
                    let first = start_pos.line.saturating_sub(self.context_lines);
                    // The window must end at the span's *last covered
                    // line* plus `context_lines` — not at the exclusive
                    // end offset's line plus one.  `span_lines` applies
                    // the "end on a line start belongs to the previous
                    // line" rule, so a span ending exactly at a newline
                    // no longer drags one phantom row of context in
                    // (with `context_lines = 0` the window previously
                    // still showed one line below the span).
                    let (_, sp_last) = idx.span_lines(sp);
                    let last = std::cmp::min(sp_last + self.context_lines, last_text_line);
                    ((first, last), start_pos, sp)
                })
                .collect();

            // Merge overlapping ranges. `ranges` is the upper bound on the
            // merged count, so reserve it once instead of regrowing.
            // Each group keeps its real spans (order inherited from
            // `sorted_spans`); the byte-union is derived at use time for the
            // location header and line range only.  The inner span list is a
            // `SmallVec<[Span; 2]>`: a merged group almost always has ≤2
            // spans, so the common case stays inline.
            let mut merged: SmallVec<[((usize, usize), SourcePos, SmallVec<[Span; 2]>); 2]> =
                SmallVec::with_capacity(ranges.len());
            for (range, pos, sp) in &ranges {
                if let Some(last) = merged.last_mut() {
                    // Merge only same-file ranges: line numbers collide
                    // across files, so overlap in (first,last) means
                    // nothing when `sp.file` differs.
                    if last.2[0].file == sp.file && range.0 <= last.0.1.saturating_add(1) {
                        last.0.1 = last.0.1.max(range.1);
                        last.2.push(*sp);
                        continue;
                    }
                }
                merged.push((*range, *pos, smallvec![*sp]));
            }

            // Global line-number width: the largest line number any section
            // can show is the file's last text line, so the width is shared
            // by every section (previously each section sized its own
            // column from its local window, and wide context lines drifted
            // the gutters of neighbouring sections out of alignment).
            let line_num_width = idx.text_line_count().checked_ilog10().unwrap_or(0) as usize + 1;

            // Per-label "rendered inline somewhere" flags, indexed like
            // `merged_labels` (and therefore `diag.labels`), set by the
            // sections as they emit each label's underline or message.  A
            // line-range test is not enough: a label whose line lies inside
            // a rendered range can still be dropped by the per-line byte
            // intersection (e.g. a point on the phantom line after a
            // trailing newline, which no section ever draws), and would then
            // never be shown at all.
            let mut label_rendered: SmallVec<[bool; 8]> =
                SmallVec::from_elem(false, merged_labels.len());
            // Hoist the per-file hyperlink invariants out of the section
            // loop: `use_color`, `supports_hyperlinks`, and the filename
            // test depend only on the renderer config and `diag.file_name`,
            // none of which varies per section.  `canonicalize` is cached in
            // `RenderCtx` so a batch of M diagnostics over one file pays one
            // syscall instead of M (matching `chain.rs`'s hoist).
            // **Capability, not palette.**  OSC 8 is zero-width and orthogonal to
            // SGR: a `--color=never` run on a hyperlink-capable terminal used to
            // lose every clickable link because the click target was gated on
            // colour.  `chain.rs` already decoupled this; the two surfaces must
            // answer the same question the same way.
            let hyperlink_enabled = super::term::supports_hyperlinks();
            let anchor_fname: &str = &diag.file_name;
            let anchor_file_url: Option<Arc<str>> =
                if hyperlink_enabled && super::is_real_file_name(anchor_fname) {
                    match cx {
                        Some(cx) => Some(cx.file_url(anchor_fname)),
                        None => Some(path_to_file_url(std::path::Path::new(anchor_fname))),
                    }
                } else {
                    None
                };
            // Hoist filename sanitization out of the per-section loop: the
            // anchor name is invariant across all K sections, so compute it
            // once instead of K times inside `write_source_section`.
            let anchor_sanitized = super::sanitize_filename(anchor_fname);
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
            for (i, (_, start_pos, group)) in merged.iter().enumerate() {
                // The group's earliest span is the section primary (caret +
                // location header); the other real primaries are passed as
                // extra `^` annotations, and `write_source_section` extends
                // its window to cover them.  The section therefore shows the
                // same line range the old byte-union did, but intermediate
                // lines between the real spans get no caret.
                extras_labels.clear();
                extras_labels.extend(group[1..].iter().map(|&sp| Label::new(sp, "")));
                // Lines hidden between the previous section's window end and this
                // section's window start. `None` (first section) draws no marker;
                // `Some(n)` always has `n >= 1` — contiguous sections are merged by
                // the predicate above, so the old "0 draws a bare `...`" state was
                // unreachable.
                let folded_lines = i
                    .checked_sub(1)
                    .and_then(|prev_i| merged.get(prev_i))
                    .map(|prev| merged[i].0.0.saturating_sub(prev.0.1).saturating_sub(1));
                let anchor_outcome = self.write_source_section(
                    out,
                    source,
                    idx,
                    group[0],
                    *start_pos,
                    anchor_file_url.as_deref(),
                    &section_labels,
                    &section_label_lines,
                    &section_label_order,
                    &section_label_sources,
                    &extras_labels,
                    section_scratch,
                    anchor_sanitized.as_ref(),
                    diag.level,
                    folded_lines,
                    // Only the first section repeats the file name...
                    i == 0,
                    // ...but every anchor section's `:line:col` stays a link.
                    true,
                    &mut label_rendered,
                    line_num_width,
                    &diag.suggestions,
                    &anchor_suggestion_lines,
                    &mut suggestion_rendered,
                    cx,
                );
                // An empty anchor source returns before drawing the
                // `file:line:col` header.  Keep the primary location visible
                // with the no-source fallback instead of dropping it.
                if i == 0 && !anchor_outcome.drew_primary {
                    self.write_note_line(out, &format!("at {}", group[0]));
                }
            }

            // ── Pre-group foreign-file data ───────────────────────
            // Partition spans/labels/suggestions by file once so each
            // foreign file's section does not re-scan the full
            // `sorted_spans` / `merged_labels` / `suggestions` lists.
            // Turns the foreign loop from O(F·(S+L+U)) into O(S+L+U+F).
            // The whole partition is dead work when there are no foreign
            // sources (the overwhelmingly common single-file case), so it is
            // gated below together with the foreign-section loop itself.
            // `LiteMap` (a sorted flat vector) instead of the hash maps:
            // the key set is a handful of `FileId`s, so O(log n) binary
            // search with no hash-table shell is cheaper than Fx's SipHash
            // setup, and the flat storage is cache-friendlier for the F
            // foreign-file loop that reads every key once.
            let mut foreign_rendered_files: litemap::LiteMap<FileId, ()> = litemap::LiteMap::new();
            if let Some(foreign) = diag.foreign_sources.as_deref() {
                let mut suggestion_lines_buf =
                    std::mem::take(&mut section_scratch.suggestion_lines_buf);
                let mut foreign_spans_by_file: litemap::LiteMap<FileId, SmallVec<[Span; 2]>> =
                    litemap::LiteMap::new();
                for &sp in sorted_spans.iter() {
                    if Some(sp.file) != anchor_file {
                        foreign_spans_by_file
                            .entry(sp.file)
                            .or_insert_with(SmallVec::new)
                            .push(sp);
                    }
                }
                let mut foreign_labels_by_file: litemap::LiteMap<
                    FileId,
                    (SmallVec<[&Label<'src>; 4]>, SmallVec<[usize; 4]>),
                > = litemap::LiteMap::new();
                for (i, lbl) in merged_labels.iter().enumerate() {
                    if Some(lbl.span.file) != anchor_file {
                        let entry = foreign_labels_by_file
                            .entry(lbl.span.file)
                            .or_insert_with(|| (SmallVec::new(), SmallVec::new()));
                        entry.0.push(lbl);
                        entry.1.push(i);
                    }
                }
                let mut foreign_suggestions_by_file: litemap::LiteMap<
                    FileId,
                    SmallVec<[(usize, Span); 4]>,
                > = litemap::LiteMap::new();
                for (i, sug) in diag.suggestions.iter().enumerate() {
                    if let Some(span) = sug.span {
                        if Some(span.file) != anchor_file {
                            foreign_suggestions_by_file
                                .entry(span.file)
                                .or_insert_with(SmallVec::new)
                                .push((i, span));
                        }
                    }
                }
                // Reusable buffer for per-foreign-file suggestion line ranges.
                // Entries are set for the current file and reset to None after the
                // call, so no O(S) work happens per foreign file.  Capacity is
                // taken from `section_scratch` and restored at the end of this
                // block, so a batch of diagnostics sharing a suggestion count pays
                // one allocation, not one per diagnostic.
                suggestion_lines_buf.clear();
                suggestion_lines_buf.resize(diag.suggestions.len(), None);

                {
                    for (fid, fname, fsource) in foreign.iter() {
                        if Some(*fid) == anchor_file {
                            continue;
                        }
                        let f_primaries: &[Span] = foreign_spans_by_file
                            .get(fid)
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        let (f_labels, f_label_sources): (&[&Label<'src>], &[usize]) =
                            foreign_labels_by_file
                                .get(fid)
                                .map(|(l, s)| (l.as_slice(), s.as_slice()))
                                .unwrap_or((&[], &[]));
                        if f_primaries.is_empty() && f_labels.is_empty() {
                            continue;
                        }
                        let Some(section_primary) = f_primaries
                            .first()
                            .copied()
                            .or_else(|| f_labels.first().map(|l| l.span))
                        else {
                            continue;
                        };
                        let f_idx = match cx {
                            Some(cx) => cx.line_index(fsource),
                            None => LineIndex::new(fsource),
                        };
                        let f_start_pos = f_idx.to_linecol(section_primary.start_as_usize());
                        let f_fname: &str = fname.as_ref();
                        let f_sanitized = super::sanitize_filename(f_fname);
                        let f_file_url: Option<Arc<str>> =
                            if hyperlink_enabled && super::is_real_file_name(f_fname) {
                                match cx {
                                    Some(cx) => Some(cx.file_url(f_fname)),
                                    None => Some(path_to_file_url(std::path::Path::new(f_fname))),
                                }
                            } else {
                                None
                            };
                        let f_label_lines: SmallVec<[(usize, usize); 4]> =
                            f_labels.iter().map(|l| f_idx.span_lines(l.span)).collect();
                        // Populate the reusable suggestion-lines buffer only for
                        // suggestions belonging to this foreign file; reset after
                        // the call so the next file sees None for these entries.
                        let mut set_indices: SmallVec<[usize; 4]> = SmallVec::new();
                        if let Some(list) = foreign_suggestions_by_file.get(fid) {
                            for &(i, span) in list {
                                suggestion_lines_buf[i] = Some(f_idx.span_lines(span));
                                set_indices.push(i);
                            }
                        }
                        let extras: SmallVec<[Label<'src>; 2]> = f_primaries
                            .iter()
                            .skip(1)
                            .map(|sp| Label::new(*sp, ""))
                            .collect();
                        let f_line_num_width =
                            f_idx.text_line_count().checked_ilog10().unwrap_or(0) as usize + 1;
                        let section_rendered = self.write_source_section(
                            out,
                            fsource,
                            &f_idx,
                            section_primary,
                            f_start_pos,
                            f_file_url.as_deref(),
                            f_labels,
                            &f_label_lines,
                            &[],
                            f_label_sources,
                            &extras,
                            section_scratch,
                            f_sanitized.as_ref(),
                            diag.level,
                            None,
                            true,
                            true,
                            &mut label_rendered,
                            f_line_num_width,
                            &diag.suggestions,
                            &suggestion_lines_buf,
                            &mut suggestion_rendered,
                            cx,
                        );
                        for i in set_indices {
                            suggestion_lines_buf[i] = None;
                        }
                        // Only a section that actually drew the primary span
                        // counts as "rendered": an empty foreign source or
                        // a primary past EOF must fall through to the
                        // location-note fallbacks below, or its spans and labels
                        // would be silently suppressed.
                        if section_rendered.drew_primary {
                            foreign_rendered_files.insert(*fid, ());
                        }
                    }
                }
                // Restore the buffer while it is still in scope: both the take
                // and the restore live inside the foreign gate.
                section_scratch.suggestion_lines_buf = suggestion_lines_buf;
            } // foreign_sources gate
            section_scratch.section_label_lines = section_label_lines;
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
            debug_assert!(
                label_rendered.len() >= diag.labels.len(),
                "label_rendered bookkeeping drift: {} < {}",
                label_rendered.len(),
                diag.labels.len()
            );
            for (i, lbl) in diag.labels.iter().enumerate() {
                // Fail *visible*: an out-of-bookkeeping index (should be
                // impossible) must render the note rather than silently drop
                // the label.
                if label_rendered.get(i).copied().unwrap_or(false) {
                    continue;
                }
                // Multi-file guard: a label from another file cannot be
                // located in this buffer; keep it visible via its byte
                // range instead of a bogus line:col.  A missing anchor (no
                // primary span) means we do *not* know the attached source
                // belongs to any label's file, so treat every label as
                // foreign rather than resolving its offsets against a buffer
                // that may be unrelated.
                let foreign = anchor_file.is_none_or(|anchor| lbl.span.file != anchor);
                let pos = (!foreign).then(|| idx.to_linecol(lbl.span.start_as_usize()));
                // One canonical note path.  A locatable label gets
                // `line:col`; a foreign/unlocatable one gets its byte range so
                // it stays identifiable.  `write_note_line_with` owns the
                // gutter, wrapping, note color, and continuation padding.
                let suffix = match pos {
                    Some(pos) => {
                        // Same batch-scoped table as the header: a diagnostic with
                        // N labelled notes would otherwise walk the line prefix N
                        // times.
                        let display_col = match cx {
                            Some(ctx) => ctx.display_col_in_line(
                                idx.line(pos.line),
                                pos.line,
                                pos.col,
                                self.tab_width,
                            ),
                            // Memoized checkpoints instead of a fresh grapheme walk per
                            // header: this is the O(prefix) cost
                            // `perf_header_column_walk_on_long_line` measures.
                            None => idx.display_col(pos.line, pos.col, self.tab_width),
                        };
                        format!("{}:{}", pos.line + 1, display_col + 1)
                    }
                    None => lbl.span.to_string(),
                };
                self.write_note_line_with(out, lbl.message.as_ref(), Some(&suffix));
            }

            // Multi-file guard: primary spans from other files cannot be
            // drawn in this buffer either; surface each as a location-only
            // note so the diagnostic's other sites stay visible.  Prefix the
            // byte range with the filename when it is a real file (not the
            // `<input>` placeholder) so the site stays identifiable.
            let fname = diag.file_name();
            let has_fname = !fname.is_empty() && fname != "<input>";
            for sp in &foreign_spans {
                if foreign_rendered_files.contains_key(&sp.file) {
                    continue;
                }
                // A foreign span must be labelled with *its own* file name,
                // never the anchor's — the anchor name would send the user
                // to the wrong file.  `file_name_for` resolves the owner's
                // name; only prefix when it is a real filename.
                let loc = match diag.file_name_for(sp.file) {
                    Some(f) if !f.is_empty() && f != "<input>" => format!("{f} {sp}"),
                    _ => sp.to_string(),
                };
                // Route through the wrapping note writer so a long location
                // cannot overflow `max_width` on the no-source path, and the
                // continuation gutter matches every other note.
                self.write_note_line(out, &format!("at {loc}"));
            }
            // Restore the taken scratch fields so their capacity survives
            // into the next diagnostic's render.  `suggestion_lines_buf` was
            // restored inside the foreign gate (its take is gated too), and
            // `section_label_lines` was restored immediately after that gate.
            section_scratch.section_label_sources = section_label_sources;
            section_scratch.section_label_order = section_label_order;
            section_scratch.anchor_suggestion_lines = anchor_suggestion_lines;
        } else {
            // No source attached: surface the byte ranges of the primary
            // spans so the diagnostic's location is not silently lost.
            // Prefix with the span's own filename when it is a real file —
            // with no source there is no anchor buffer, but the spans can
            // still belong to different files, and each must keep its name.
            for sp in diag.spans.primary_ref().iter() {
                let loc = match diag.file_name_for(sp.file) {
                    Some(f) if !f.is_empty() && f != "<input>" => format!("{f} {sp}"),
                    _ => sp.to_string(),
                };
                // Route through the wrapping note writer so a long location
                // cannot overflow `max_width` on the no-source path, and the
                // continuation gutter matches every other note.
                self.write_note_line(out, &format!("at {loc}"));
            }
            // Labels must not vanish when no source is attached — a
            // "previous definition here" annotation is the whole point of
            // some diagnostics.  Surface each as `= note: <message> at
            // <span>`, using the same sanitizer and gutter conventions as
            // the source-attached fallback loop.  Empty messages carry
            // nothing to say (the merged-group extras use them) — skip.
            // No dedupe against related errors is needed here: with no
            // source attached, related errors render nowhere else in this
            // branch, so a label can never be shown twice.
            for lbl in diag.labels.iter() {
                if lbl.message.is_empty() {
                    continue;
                }
                let sanitized = super::sanitize_for_terminal_multiline(&lbl.message);
                // Per-label file name: with no source attached, labels can
                // come from several files, and each must keep its own name.
                let loc = match diag.file_name_for(lbl.span.file) {
                    Some(f) if !f.is_empty() && f != "<input>" => format!("{f} {}", lbl.span),
                    _ => lbl.span.to_string(),
                };
                // Wrap the whole `message at location` through the shared
                // note writer (localized label + pad-only continuations) so a
                // long label cannot break the width contract on this path.
                self.write_note_line(out, &format!("{} at {}", sanitized, loc));
            }
        }

        // Children (sub-diagnostics: notes, help)
        if !self.quiet {
            for child in &diag.children {
                self.write_child(out, child);
            }
        }

        // Suggestions: `help:` for machine-applicable, `note:` otherwise
        // (with a `(maybe incorrect)` / `(has placeholders)` tag via
        // `Suggestion::Display`).  Suggestions already rendered as a diff
        // block inside a source section are skipped here.
        if !self.quiet {
            for (i, suggestion) in diag.suggestions.iter().enumerate() {
                if suggestion_rendered[i] {
                    continue;
                }
                if matches!(
                    suggestion.style,
                    crate::diagnostics::SuggestionStyle::CompletelyHidden
                ) {
                    continue;
                }
                let text = suggestion.to_string();
                match suggestion.applicability {
                    Applicability::MachineApplicable => {
                        self.write_help_line(out, &text);
                    }
                    _ => {
                        self.write_note_line(out, &text);
                    }
                }
            }
        }

        // Help
        if !self.quiet
            && let Some(ref help) = diag.help
        {
            self.write_help_line(out, help);
        }
    }

    /// Render the summary line (e.g. "aborting due to N previous errors").
    pub fn render_summary(&self, error_count: usize, warning_count: usize) -> String {
        if self.quiet {
            return String::new();
        }
        if error_count == 0 && warning_count == 0 {
            return String::new();
        }

        let mut out = String::with_capacity(64);
        let accent = if error_count > 0 {
            self.s.error
        } else {
            self.s.warning
        };
        swrite!(
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
                swrite!(
                    out,
                    "{}",
                    crate::diagnostics::i18n::fluent::summary_warnings(DisplayArg(w))
                );
            }
            (e, 0) => {
                swrite!(
                    out,
                    "{}",
                    crate::diagnostics::i18n::fluent::summary_errors(DisplayArg(e))
                );
            }
            (e, w) => {
                swrite!(
                    out,
                    "{}; ",
                    crate::diagnostics::i18n::fluent::summary_mixed_errors(DisplayArg(e))
                );
                swrite!(
                    out,
                    "{}",
                    crate::diagnostics::i18n::fluent::summary_warnings(DisplayArg(w))
                );
            }
        }
        swrite!(out, "{}", self.s.reset);
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
        swrite!(
            s,
            " {}",
            crate::diagnostics::i18n::fluent::summary_files(DisplayArg(file_count))
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
        let fallback_code = std::sync::OnceLock::new();
        // `SmallVec<[&ErrCode; 2]>`: the overwhelmingly common case has a
        // single code (or none → the `?` fallback), so it stays inline and
        // avoids a heap allocation per rendered diagnostic.  The related-error
        // branch spills to the heap only when ≥3 distinct codes appear.
        let all_codes: SmallVec<[&crate::diagnostics::ErrCode; 2]> = if diag
            .related_errors
            .is_empty()
        {
            match diag.code.as_ref() {
                Some(c) => smallvec![c],
                None => smallvec![
                    fallback_code.get_or_init(|| crate::diagnostics::ErrCode::unknown("?"))
                ],
            }
        } else {
            let mut codes: SmallVec<[&crate::diagnostics::ErrCode; 2]> = SmallVec::new();
            if let Some(c) = diag.code.as_ref() {
                codes.push(c);
            }
            // Dedup by code *string* without the O(R²) `codes.iter().any`
            // scan: R ≤ 4 stays on an inline linear scan (the common
            // one-or-two-related-errors shape pays no hash setup), and a
            // diagnostic aggregating many related errors spills to a
            // hash set — O(R) total instead of O(R²) string compares.
            let mut seen: SmallVec<[&str; 4]> = SmallVec::new();
            if let Some(c) = diag.code.as_ref() {
                seen.push(c.canonical());
            }
            let mut spill: Option<rustc_hash::FxHashSet<&str>> = None;
            for rel in &diag.related_errors {
                let Some(c) = rel.code.as_ref() else {
                    continue;
                };
                let code = c.canonical();
                let dup = match &mut spill {
                    Some(s) => !s.insert(code),
                    None => seen.iter().any(|k| *k == code),
                };
                if !dup {
                    codes.push(c);
                    // Stop maintaining `seen` once the hash set has
                    // spilled: from here the set alone answers the
                    // dup test, and pushing into the (full) inline
                    // buffer for every remaining unique code is pure
                    // waste.
                    if spill.is_none() {
                        seen.push(code);
                        if seen.len() == 4 {
                            spill = Some(seen.iter().copied().collect());
                        }
                    }
                }
            }
            if codes.is_empty() {
                codes.push(fallback_code.get_or_init(|| crate::diagnostics::ErrCode::unknown("?")));
            }
            codes
        };
        // Sanitize each code once into an inline buffer (≤2 codes stay on
        // the stack), so the visible width and the comma-joined display form
        // can be derived without re-sanitizing and without building the
        // joined `String` when the hyperlink branch will discard it.
        let sanitized_codes: SmallVec<[Cow<'_, str>; 2]> = all_codes
            .iter()
            .map(|c| {
                if c.is_known() {
                    Cow::Borrowed(c.code())
                } else {
                    super::sanitize_for_terminal(c.code())
                }
            })
            .collect();
        // Visible width of the comma-joined codes, computed from parts
        // (sanitization can inject U+FFFD, so each piece goes through
        // `visible_width`) without materialising the joined `String`.
        let all_codes_width = {
            let mut w = 0usize;
            for (i, s) in sanitized_codes.iter().enumerate() {
                if i > 0 {
                    w += 1; // comma
                }
                w += visible_width(s);
            }
            w
        };
        // Visible width of the fixed `┌─ error[E030]: ` prefix, computed
        // directly from its parts (layout: `{tl}{h} {level}[{codes}]: `)
        // instead of building the string only to measure it.  Box chars are
        // one cell each; the level label is ASCII; `all_codes_width` carries
        // the codes' contribution because sanitization can inject U+FFFD.
        //
        // With related errors the primary is item `1.`; its continuation
        // lines reserve the same gutter but leave the number off so it does
        // not collide with the related items' `2.`, `3.`, ….
        let numbered = !diag.related_errors.is_empty();
        let header_cols = self.bc.tl.chars().count() // ┌
            + self.bc.h.chars().count()              // ─
            + 1                                      // space
            + visible_width(level_label.as_ref())    // error — display cells, not bytes (localized labels are multi-byte)
            + 1                                      // [
            + all_codes_width                        // E030
            + "]: ".len()                            // ]:
            + if numbered { 3 } else { 0 };
        // Continuation gutter for wrapped header segments: the `│` replaces
        // the `┌`, and the spaces fill the rest of the prefix so the segment
        // starts at the same column.  `header_cols` already includes the
        // 3-column number slot when numbered, so the no-number prefix width
        // is `header_cols - (if numbered { 3 } else { 0 })`.
        let prefix_no_num = header_cols - usize::from(numbered) * 3;
        let cont_number = if numbered { "   " } else { "" };
        // The continuation reserves the same 3-column numbered slot as
        // the first line; pin the invariant so a future literal edit fails
        // loudly instead of silently shifting wrapped header messages.
        debug_assert_eq!(
            prefix_no_num + cont_number.len(),
            header_cols,
            "numbered-header continuation must occupy the full number slot"
        );
        // OSC 8 hyperlink form of the code list, used only when the terminal
        // supports hyperlinks and color is on — each code becomes its own
        // `url_ansi` link so Ctrl+click opens the matching `--explain` page.
        // The codes are written straight into `out` at the render site below;
        // no intermediate comma-joined `String` (nor a per-diagnostic
        // `url_ansi().to_string()`) is built.  `all_codes_width` already
        // measured the visible form from `sanitized_codes`.
        let use_hyperlink_codes = self.s.use_color
            && super::term::supports_hyperlinks()
            && crate::diagnostics::error_code::is_explain_server_running();
        let msg_src: Cow<'_, str> = if diag.message.trim().is_empty() {
            // Empty message: fall back to the code's title when a code
            // exists, otherwise a neutral placeholder.  Without this
            // `wrap_msg` returns a single empty segment and the header
            // ends in a bare trailing colon.
            diag.code
                .as_ref()
                .map(|c| Cow::Borrowed(c.title()))
                .unwrap_or_else(|| crate::diagnostics::i18n::fluent::unknown_error())
        } else {
            Cow::Borrowed(&diag.message)
        };
        let sanitized_msg = super::sanitize_for_terminal_multiline(&msg_src);
        for (i, seg) in self
            .wrap_msg(&sanitized_msg, header_cols)
            .iter()
            .enumerate()
        {
            if i == 0 {
                // First line: the full `┌─ error[...]: ` prefix.
                out.push_str(self.s.dim);
                out.push_str(self.bc.tl);
                out.push_str(self.bc.h);
                out.push_str(self.s.reset);
                out.push(' ');
                out.push_str(level_color);
                out.push_str(self.s.bold);
                out.push_str(level_label);
                out.push_str(self.s.reset);
                out.push('[');
                out.push_str(self.s.bold);
                if use_hyperlink_codes {
                    for (i, c) in all_codes.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        // Write form: no owned wrapper for an uncacheable spelling.
                        // `String`'s `fmt::Write` never fails.
                        let _ = c.write_ansi_hyperlink(out);
                    }
                } else {
                    for (i, s) in sanitized_codes.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        out.push_str(s.as_ref());
                    }
                }
                out.push_str(self.s.reset);
                out.push_str("]: ");
                let lead = if numbered { "1. " } else { "" };
                out.push_str(lead);
                if !highlight_code_into(seg, self.s.use_color, out) {
                    out.push_str(seg.as_ref());
                }
                out.push('\n');
            } else {
                // Continuation: `│` + padding so the segment starts at the
                // same column as the first line's message.  `push_spaces`
                // writes straight into `out` — the wrap path allocates no
                // throwaway pad `String` at all.
                out.push_str(self.s.dim);
                out.push_str(self.bc.v);
                out.push_str(self.s.reset);
                push_spaces(out, prefix_no_num - 1);
                out.push_str(cont_number);
                if !highlight_code_into(seg, self.s.use_color, out) {
                    out.push_str(seg.as_ref());
                }
                out.push('\n');
            }
        }
        // List related errors as numbered items under the header.
        // `indent` is only used inside this loop, so defer its allocation to
        // avoid a per-diagnostic `String` + `memset` when there are no related
        // errors — the overwhelmingly common case (one primary diagnostic).
        if !diag.related_errors.is_empty() {
            // Calculate the indentation for related error items so they align
            // with the "1." on the header line:
            //   ┌─ error[E019,E030]: 1. duplicate definition of `i`
            //   │                    2. type mismatch
            //
            // `header_cols` is invariant per diagnostic, so the indent width
            // is hoisted once; the spaces are pushed straight into `out` per
            // row instead of materializing a `" ".repeat` String that is
            // measured and copied then dropped.
            // Align under the header's `1.`, which starts at `prefix_no_num`
            // (`header_cols` minus the number slot) — not at `header_cols -
            // 1`, which put `2.` under the message text *after* `1. ` and
            // shifted the whole list three columns right.
            let indent_cols = prefix_no_num.saturating_sub(1);
            for (i, rel) in diag.related_errors.iter().enumerate() {
                let num = i + 2;
                // Width from parts: `│` (1 cell in both charsets) + the
                // indent (ASCII spaces) + `<digits>. ` — one `dec_digits`
                // call and three adds, versus a `format!` allocating a
                // `String` per related error that was immediately measured
                // and dropped.
                let num_width = crate::diagnostics::renderers::fmt::dec_digits(num as u64);
                let item_cols = BOX_VERT_WIDTH + indent_cols + num_width + 2;
                let rel_msg = super::sanitize_for_terminal_multiline(&rel.message);
                for (j, seg) in self.wrap_msg(&rel_msg, item_cols).iter().enumerate() {
                    // Gutter pieces written straight into `out`; the continuation
                    // lines carry no number so they stay clear of the related
                    // items' `2.`, `3.`, ….
                    out.push_str(self.s.dim);
                    out.push_str(self.bc.v);
                    out.push_str(self.s.reset);
                    push_spaces(out, indent_cols);
                    if j == 0 {
                        push_uint(out, num as u64);
                        out.push_str(". ");
                    } else {
                        // Continuation reserves the same `<num>. ` slot the
                        // first line used: `num_width` cells plus `". "`.
                        // The old hardcoded 3 spaces shifted the wrapped
                        // text one column left once `num` reached 10.
                        push_spaces(out, num_width + 2);
                    }
                    if !highlight_code_with_backticks_into(seg, self.s.use_color, out) {
                        out.push_str(seg.as_ref());
                    }
                    out.push('\n');
                }
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
        // Precomputed `to_linecol` of `primary_span`'s start/end — the caller
        // already computes these for the merge window, so threading them in
        // avoids two O(log n) lookups per section.
        start_pos: SourcePos,
        // Pre-canonicalized `file://` URL for `filename`, or `None` when
        // hyperlinks are disabled / `filename` is a synthetic placeholder.
        // Threaded in so a batch of M diagnostics over one file pays one
        // `canonicalize` (cached in `RenderCtx`) instead of M.
        file_url: Option<&str>,
        labels: &[&'a Label<'src>],
        label_lines: &[(usize, usize)],
        // Start-line-sorted permutation of `label_lines` (the anchor labels),
        // precomputed once per diagnostic and shared by every section so
        // `LabelWindow` can merge the per-section extras into a pre-sorted
        // order instead of re-sorting the whole union per section.  Pass `&[]`
        // when no pre-sort is available (the foreign-file path).
        label_order: &[usize],
        label_sources: &[usize],
        extras: &'a [Label<'src>],
        // Per-section scratch buffers, reused across the render's sections
        // instead of reallocated (see [`SectionScratch`]).  `all_labels`
        // cannot be threaded: it holds references into per-section `extras`,
        // which do not outlive a section call — so `underlines_buf` and
        // `meta_buf` (which hold `&'a str` from label messages) stay as
        // per-section locals below.
        scratch: &mut SectionScratch,
        filename_display: &str,
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
        // Whether the location header carries the OSC 8 hyperlink.  Always
        // `true` for a section of the anchor file, including the later
        // sections of a multi-span diagnostic: only the *duplicated* file
        // name is suppressed there, the `:line:col` text is still a clickable
        // jump target.
        allow_hyperlink: bool,
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
        // Precomputed `(sug_first, sug_last)` for each suggestion whose span
        // belongs to this section's file (`None` for suggestions in other
        // files), computed once per file per diagnostic so the per-section
        // loop is a cheap range test instead of an `O(log n)` `span_lines`
        // lookup per section per suggestion.
        suggestion_lines: &[Option<(usize, usize)>],
        suggestion_rendered: &mut [bool],
        // Batch-scoped ctx, used for the one column conversion that used to cost a
        // full grapheme walk per diagnostic (the location header).  `None` on the
        // paths that render without a ctx, which fall back to the direct walk.
        render_ctx: Option<&RenderCtx<'_>>,
    ) -> SectionOutcome {
        // Empty source has no rows to draw. `text_line_count()` reports 1
        // for `""` (str::lines-style), so guard on the source itself rather
        // than materializing every line just to test for emptiness.
        // `drew_primary` distinguishes "drew context rows" from "drew the
        // primary span the caller asked about": a foreign primary past EOF
        // draws only context, and treating that as shown would silently
        // swallow the span.
        if source.is_empty() {
            return SectionOutcome {
                drew_primary: false,
            };
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
        scratch.extra_lines.clear();
        scratch
            .extra_lines
            .extend(extras.iter().map(|l| idx.span_lines(l.span)));
        let extras_lines: &[(usize, usize)] = &scratch.extra_lines;

        // `all_labels` is rebuilt per section: it holds references into
        // `extras`, which are per-section data and cannot be stored in a
        // caller-owned scratch across sections.  A section almost always
        // has ≤4 annotations, so inline storage avoids a heap allocation.
        let mut all_labels: SmallVec<[&'a Label<'src>; 4]> =
            SmallVec::with_capacity(labels.len() + extras.len());
        all_labels.extend(labels.iter().copied());
        all_labels.extend(extras.iter());

        scratch.all_label_lines.clear();
        scratch.all_label_lines.extend_from_slice(label_lines);
        scratch.all_label_lines.extend_from_slice(extras_lines);
        let all_label_lines: &[(usize, usize)] = &scratch.all_label_lines;

        // Determine the line range to display (with context)
        let first_line = start_pos.line.saturating_sub(self.context_lines);
        let text_last = idx.text_line_count().saturating_sub(1);
        // `end_pos.line` is the line of the span's *exclusive* end offset;
        // the last line the span actually covers is `span_lines(..).1`
        // (an end sitting on a line start belongs to the previous line).
        // Adding `context_lines` — without the extra +1 — keeps the shown
        // window at exactly `context_lines` rows below the covered lines.
        let mut last_line = std::cmp::min(
            idx.span_lines(primary_span).1 + self.context_lines,
            text_last,
        );
        // Extras come after `primary_span` (spans are sorted by start), so
        // only the window's end needs extending.  `extra_end` is already the
        // last covered line of the extra span, so no +1 here either.
        for &(_, extra_end) in extras_lines {
            last_line = last_line.max(std::cmp::min(extra_end + self.context_lines, text_last));
        }

        // Width of the line number column: the max of this section's local
        // need and the render-wide column, so every section's gutter stays
        // aligned (computed by the caller from the file's last text line).
        let local_line_num_width = (last_line + 1).checked_ilog10().unwrap_or(0) as usize + 1;
        let line_num_width = std::cmp::max(local_line_num_width, global_line_num_width);
        // Indent for lines that don't have a line number (location header, spacing, explanation, fold marker).
        // Reuse `scratch.indent`'s capacity across sections instead of a fresh
        // `" ".repeat(line_num_width + 1)` allocation per section.
        scratch.indent.clear();
        push_spaces(&mut scratch.indent, line_num_width + 1);
        let indent = scratch.indent.as_str();
        // The blank line-number gutter is emitted as `{:>width$}` on the diff
        // rows below (width = `line_num_width`), so no per-section `String`
        // is built for a fill only the suggestion block uses.

        // ── Fold marker ────────────────────────────────────────────
        // Drawn with this section's own gutter (`│` + indent) so it lines
        // up with the location header directly below it.  Emitted before
        // the header so the reader sees the omission *above* the section
        // it precedes.  The marker text is ASCII (`...`) in both the
        // Unicode and ASCII box-character modes; when lines were hidden
        // the gap count makes the omission concrete
        // (`... (37 lines omitted)`).
        if let Some(gap) = folded_lines {
            // The marker text is shared with the HTML renderer (correct
            // singular/plural, one source of truth).
            out.push_str(self.s.dim);
            out.push_str(self.bc.v);
            out.push_str(&indent);
            crate::diagnostics::renderers::fmt::push_fold_marker(out, gap);
            out.push_str(self.s.reset);
            out.push('\n');
        }

        // Location header: ┌─ input:1:1  (with OSC 8 hyperlink when filename is a real path)
        //
        // `filename_display` is already stripped of control characters by the
        // caller (the anchor name is sanitized once per diagnostic, each
        // foreign name once per file), so the terminal cannot forge links,
        // change the title, or clear the screen via the displayed path.  The
        // URL is separately percent-encoded below.
        //
        // Assembled straight into `out` between the SGR runs — two
        // `format!`-built `String`s per section that were copied into `out`
        // and dropped is exactly the pattern the rest of this file
        // (`write_header`, the gutter, `chain.rs`) eliminated with
        // `push_uint`/`push_spaces`.
        let sub_box_prefix: &str = if self.compact { "" } else { self.bc.sub_tl_h };
        out.push_str(self.s.dim);
        out.push_str(self.bc.v);
        out.push_str(&indent);
        out.push_str(sub_box_prefix);
        out.push_str(self.s.reset);
        out.push(' ');
        out.push_str(self.s.help);
        // OSC 8 hyperlink — Ctrl+click opens the file in the editor.  The
        // URL is pre-canonicalized and percent-encoded by the caller (cached
        // per batch in `RenderCtx`), so control characters in a filename
        // cannot terminate the sequence early or inject further terminal
        // escapes.
        let hyperlink = matches!(file_url, Some(_) if allow_hyperlink);
        if hyperlink && let Some(url) = file_url {
            out.push_str("\x1b]8;;");
            out.push_str(url);
            out.push_str("\x1b\\");
        }
        if show_filename && !filename_display.is_empty() {
            out.push_str(filename_display);
            out.push(':');
        }
        let header_display_col = match render_ctx {
            // One table per `(line, tab width)` per batch instead of a fresh
            // grapheme walk per diagnostic (measured at ~55 ms per call on a
            // 200 KB line, against ~0.4 ms for a whole diagnostic's render).
            Some(cx) => cx.display_col_in_line(
                idx.line(start_pos.line),
                start_pos.line,
                start_pos.col,
                self.tab_width,
            ),
            None => idx.display_col(start_pos.line, start_pos.col, self.tab_width),
        };
        push_uint(out, start_pos.line as u64 + 1);
        out.push(':');
        push_uint(out, header_display_col as u64 + 1);
        if hyperlink {
            out.push_str("\x1b]8;;\x1b\\");
        }
        out.push_str(self.s.reset);
        out.push('\n');
        // Spacing line (skipped in compact mode)
        if !self.compact {
            out.push_str(self.s.dim);
            out.push_str(self.bc.v);
            out.push_str(&indent);
            out.push_str(self.bc.sub_v);
            out.push_str(self.s.reset);
            out.push('\n');
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
        scratch.ranges.clear();
        scratch.ranges.reserve(all_labels.len() + 1);
        scratch.ranges.push(idx.span_lines(primary_span));
        scratch.ranges.extend_from_slice(all_label_lines);
        let label_line_ranges: &[(usize, usize)] = &scratch.ranges;
        let label_start_cols = &mut scratch.label_start_cols;
        label_start_cols.clear();
        label_start_cols.resize(all_labels.len(), None);
        // A label's start column is resolved once, but several labels can start
        // on the *same* line (a wide table row, a one-line file with many
        // diagnostics): one `LineColumns` table per line replaces a full
        // grapheme walk per label.
        let mut line_cols = LineColumnsCache::default();

        // Display-column-aware source line rendering loop.  The `String`
        // scratch buffers are `mem::take`-n into owned locals so format macros
        // can use them by value; they are restored into `scratch` at function
        // exit.  The remaining scratch fields are used via `&mut` references —
        // their `clear()`/`push()` methods preserve capacity across sections.
        // (`rendered` is no longer taken: source rows render straight into
        // `out` in append form, so there is no per-row scratch copy.)
        // Underline lane allocator — owns the interval-graph colouring scratch
        // (heap, order, column buffer) so the row loop reuses it across lines.
        let lanes = &mut scratch.lanes;
        // `multiline_flags` holds only `(display_col, part)` — the two fields
        // read back below — so no per-line `String` is cloned out of
        // `underlines_buf`, and the flags outlive that reused buffer.
        // A row rarely has >4 multiline labels, so inline storage avoids a
        // per-row heap allocation.
        let multiline_flags = &mut scratch.multiline_flags;
        let mut connector = std::mem::take(&mut scratch.connector);
        // `underlines_buf` and `meta_buf` used to hold `&'a str` from
        // per-section label messages and could not be threaded across
        // sections.  The payload is now split — messages resolve at use
        // sites from `all_labels` via `label_idx` — so both live in
        // `SectionScratch` like every other buffer and their capacity
        // survives across the batch.
        let mut underlines_buf = std::mem::take(&mut scratch.underlines_buf);
        // Highlighter scratch: the `_into` highlighter entry points need a
        // buffer to assemble into when the highlighted text must be scanned
        // (connector re-colouring) or when gutter pieces interleave the
        // pushes.  Capacity reused across the batch.
        let mut hl_buf = std::mem::take(&mut scratch.hl_buf);
        let ranges_buf = &mut scratch.ranges_buf;
        let mut meta_buf = std::mem::take(&mut scratch.meta_buf);
        let display_buf = &mut scratch.display_buf;
        let order_buf = &mut scratch.order_buf;
        let done_buf = &mut scratch.done_buf;
        let active_buf = &mut scratch.active_buf;
        // Sliding window over `all_labels`, ordered by start line once here;
        // each row then only visits the labels that can cover it, not all
        // of them.  `label_order` pre-sorts the anchor-label prefix (shared
        // across sections); only the per-section extras are merged in here.
        let window_scratch = std::mem::take(&mut scratch.label_window);
        let mut label_window = LabelWindow::new(all_label_lines, label_order, window_scratch);
        // Row-invariant: the primary-highlight escapes depend only on
        // `use_color` and `color_depth`, so resolve them once instead of per
        // source row.
        let highlight_on = self.s.highlight_bg;
        // Depth-aware reset: `\x1b[49m` clears a background colour but NOT
        // SGR 7 (reverse video) — the 16-colour `highlight_bg` pairs with
        // `\x1b[27m` (see `Styles::highlight_reset`).
        let highlight_off = if self.s.use_color {
            self.s.highlight_reset
        } else {
            ""
        };
        let mut gutter_head = std::mem::take(&mut scratch.gutter_head);
        gutter_head.clear();
        gutter_head.push_str(self.s.dim);
        gutter_head.push_str(self.bc.v);
        gutter_head.push_str(self.s.reset);
        gutter_head.push_str(self.s.dim);
        let mut gutter_tail = std::mem::take(&mut scratch.gutter_tail);
        gutter_tail.clear();
        gutter_tail.push_str(self.s.reset);
        gutter_tail.push(' ');
        gutter_tail.push_str(self.s.dim);
        gutter_tail.push_str(self.bc.sub_v);
        gutter_tail.push_str(self.s.reset);
        gutter_tail.push(' ');

        let primary_end_line = idx.to_linecol(primary_span.end_as_usize()).line;
        for line_idx in first_line..=last_line {
            // O(1) slice via the shared index instead of indexing a
            // whole-file `Vec<&str>` that was collected only for this.
            let line = idx.line(line_idx);
            // A suggestion-covered row used to be excluded here by a
            // `sug_covered` table.  That term was redundant: a suggestion
            // contributes its diff block separately and never adds source-row
            // ranges, so the highlighted path emits byte-identical output for
            // such a row.  `suggestion_covered_row_renders_like_a_plain_row`
            // pins the equivalence, and the per-row table plus its O(S·rows)
            // fill are gone.
            // Fold this row into the label window *before* choosing the
            // render path.  Advancing is idempotent in `line_idx`, so the
            // annotated path's own `advance` (inside
            // `compute_line_annotation_ranges`) stays a no-op — but hoisting
            // the call means no row can ever be rendered against a stale
            // window, and the old `is_idle_for` guard (which only existed to
            // prove the skipped advance was a no-op on context rows) is gone.
            // `is_idle_for` is load-bearing for the *branch*, not only for the
            // `advance`: a row outside the span that nevertheless carries a
            // starting/active label must take the annotated path, or its
            // labels and markers are dropped entirely (that regression showed up
            // as "one caret row for two spans" and a missing inline marker).
            let context_only = (line_idx < start_pos.line || line_idx > primary_end_line)
                && label_window.is_idle_for(line_idx);
            if !context_only {
                label_window.advance(line_idx);
            }
            // One grapheme walk converts this row's byte ranges to display
            // columns, measures the line's width, and emits the highlighted
            // source text — replacing the three separate passes over the
            // line the row used to run.  Append form: the gutter is pushed
            // first and the render writes *straight into* `out`, so a plain
            // row is no longer copied twice (render scratch → out).
            // Gutter: `│ <num> │ `
            out.push_str(&gutter_head);
            push_uint_padded(out, (line_idx + 1) as u64, line_num_width);
            out.push_str(&gutter_tail);
            let display_width = if context_only {
                render_line_merged_with(
                    LineRenderArgs {
                        line,
                        ranges: &[],
                        meta: &[],
                        out: &mut underlines_buf,
                        rendered: out,
                        highlight_on,
                        highlight_off,
                        display_buf,
                        order_buf,
                        done_buf,
                        active_buf,
                        tab_width: self.tab_width,
                        content_start_col: BOX_VERT_WIDTH + line_num_width + 1 + BOX_VERT_WIDTH + 1,
                        append: true,
                    },
                    |out, s| out.push_str(s),
                )
            } else {
                compute_line_underlines_and_render(
                    primary_span,
                    &all_labels,
                    &mut label_window,
                    source,
                    line,
                    line_idx,
                    starts,
                    &mut underlines_buf,
                    out,
                    highlight_on,
                    highlight_off,
                    ranges_buf,
                    &mut meta_buf,
                    display_buf,
                    order_buf,
                    done_buf,
                    active_buf,
                    self.tab_width,
                    // Gutter width: `│ <num> │ ` — the terminal column the
                    // source text starts at, so tab stops advance from the
                    // absolute cursor column.
                    BOX_VERT_WIDTH + line_num_width + 1 + BOX_VERT_WIDTH + 1,
                    true,
                )
            };
            let underlines = &underlines_buf;

            // Line number + source
            out.push('\n');

            // ── Detect multi-line labels ───────────────────────────
            // Keyed by label index, not message text: two labels can carry
            // the same message while covering different lines, and a
            // message-keyed map would merge them. `label_line_ranges` is
            // indexed like `labels` offset by +1 (index 0 is the
            // per-section primary).
            // Fused pass: the `multiline_flags` loop and the `needs_connector`
            // block both walk `underlines` as pure functions of
            // `label_line_ranges[i + 1]`, so a single pass accumulates both
            // (plus `visible_msgs`/`force_connector` for the connector test).
            // The synthetic primary is the only entry with `label_idx == None`
            // and it carries an empty message, so skipping `None` here matches
            // the `msg.is_empty()` gates the two original passes used.
            multiline_flags.clear();
            let mut visible_msgs = 0usize;
            let mut force_connector = false;
            for &(col, _ulen, ch, label_idx) in underlines {
                let Some(i) = label_idx else {
                    continue;
                };
                let (start, end) = label_line_ranges[i + 1];
                let msg_empty = all_labels[i].message.is_empty();
                // A label counts as shown once something for it is actually
                // emitted: its message (message loop below, on the final
                // covered line) or — for a single-line or message-less label
                // — this underline row alone.  Multi-line labels are marked
                // only from the message loop: an underline-only tail (end
                // line outside the shown window) still deserves the fallback
                // note rather than being counted as fully shown.
                if (msg_empty || start == end)
                    && let Some(&src_idx) = label_sources.get(i)
                    && let Some(flag) = label_rendered.get_mut(src_idx)
                {
                    *flag = true;
                }
                if line_idx == start {
                    label_start_cols[i] = Some(col);
                }
                if msg_empty {
                    continue;
                }
                let part = if line_idx == start {
                    "start"
                } else if line_idx == end {
                    "end"
                } else {
                    "middle"
                };
                multiline_flags.push((col, part));
                // `needs_connector` contribution: a multi-line NON-FINAL label
                // is not shown on this row, so it must not influence the
                // layout of the messages that are.
                if end != line_idx {
                    continue;
                }
                visible_msgs += 1;
                if ch == ' ' || start != end {
                    force_connector = true;
                }
            }
            let needs_connector = force_connector || visible_msgs > 1;

            // The shared range policy suppresses middle-line underlines, so
            // those rows no longer appear in `underlines`.  Derive their
            // connector geometry from the label span directly: a middle row
            // gets a bare `|` at the label's start column (start/end rows
            // keep their underline segments).
            if !label_window.active().is_empty() {
                for &i in label_window.active() {
                    let (start, end) = label_line_ranges[i + 1];
                    if line_idx <= start || line_idx >= end || all_labels[i].message.is_empty() {
                        continue;
                    }
                    let span_start = all_labels[i].span.start_as_usize();
                    // A middle row's `line` is not the label's start line;
                    // clamping the span into it would always yield column 0.
                    // Measure from the label's start line so the connector
                    // sits under the opening caret.
                    let start_line_idx = label_line_ranges[i + 1].0;
                    let start_line = idx.line(start_line_idx);
                    let start_line_start = starts.get(start_line_idx).map_or(0, |&s| s as usize);
                    let start_off = span_start
                        .saturating_sub(start_line_start)
                        .min(start_line.len());
                    // The fallback is row-invariant (it depends only on the
                    // label's start line, which is above the window), so cache
                    // it in the per-section scratch: without this it is
                    // recomputed once per middle row of the same label.
                    let col = match label_start_cols[i] {
                        Some(c) => c,
                        None => {
                            let c = line_cols.display(
                                start_line_idx,
                                start_line,
                                self.tab_width,
                                start_off,
                            );
                            label_start_cols[i] = Some(c);
                            c
                        }
                    };
                    multiline_flags.push((col, "middle"));
                }
            }

            let annotation_style = Self::level_style(level);
            // Per-character colours: `^` uses the diagnostic's level colour,
            // `~` (secondary) uses the note colour (bright green — the
            // palette's secondary-underline colour; the `Style::Note`/`Green`
            // entry in `Styles`, not the blue `help` style), ` ` (help) uses
            // the help colour, `-` and blank padding use dim.
            let level_color = self.s.get(annotation_style);
            let note_color = self.s.get(Style::Note);
            let dim_c = self.s.dim;
            let help_color = self.s.help;
            let connector_color = level_color;
            let connector_reset = self.s.reset;
            // ── Underline lines (one row per overlap depth) ──────────
            if !underlines.is_empty() {
                let lane_count = lanes.assign(underlines);
                for lane_idx in 0..lane_count {
                    // Emit the gutter first, then append the underline
                    // cells straight into `out`; roll the gutter back if the
                    // lane turned out invisible (all entries clipped).
                    let mark = out.len();
                    out.push_str(self.s.dim);
                    out.push_str(self.bc.v);
                    out.push_str(&indent);
                    out.push_str(self.bc.sub_v);
                    out.push_str(self.s.reset);
                    out.push(' ');
                    let visible = lanes.row_append(
                        underlines,
                        lane_idx,
                        display_width,
                        out,
                        |ch| match ch {
                            '^' => level_color,
                            '~' => note_color,
                            ' ' => help_color,
                            _ => dim_c,
                        },
                        self.s.reset,
                    );
                    if visible {
                        out.push('\n');
                    } else {
                        out.truncate(mark);
                    }
                }

                // `needs_connector` was computed by the fused pass above the
                // lane block — `force_connector || visible_msgs > 1`.  A
                // connector row is needed when a message would otherwise be
                // visually ambiguous: a help-style annotation, a multi-line
                // label, or more than one message sharing the row.
                for (col, _ulen, ch, label_idx) in underlines
                    .iter()
                    .copied()
                    // The payload carries only the label index; the message
                    // resolves from the section's own `all_labels` (the
                    // synthetic primary has `None` and an empty message).
                    .filter(|(_, _, _, li)| li.is_some_and(|i| !all_labels[i].message.is_empty()))
                {
                    let msg: &str = match label_idx {
                        Some(i) => all_labels[i].message.as_ref(),
                        None => "",
                    };
                    // For a multi-line label, render its message only on the
                    // final covered line; earlier lines get the connector
                    // below, so the same text is not repeated on every row.
                    let is_multiline_nonfinal =
                        label_idx.is_some_and(|i| label_line_ranges[i + 1].1 != line_idx);
                    if is_multiline_nonfinal {
                        continue;
                    }
                    if let Some(i) = label_idx
                        && let Some(&src_idx) = label_sources.get(i)
                        && let Some(flag) = label_rendered.get_mut(src_idx)
                    {
                        // This message is the label's rendering; block the
                        // fallback from emitting a duplicate note.
                        *flag = true;
                    }
                    // `col` is a display column; pad with spaces to align.
                    connector.clear();
                    connector.reserve(col + 1);
                    push_spaces(&mut connector, col);
                    connector.push('|');
                    // Visible width of the `│<indent>│ <spaces>| ` prefix the
                    // message text follows.  Summed from its parts (the same
                    // arithmetic `header_cols` uses) instead of building the
                    // string only to measure it — one `format!` per label
                    // otherwise.  A help label prepends `help: ` on its first
                    // line only, so reserve that gutter too.
                    // A help label prepends the localized `help: ` gutter on its first
                    // line only, so reserve its display width — not the ASCII
                    // literal's 6 cells (`ru` was over-reserving).
                    let help_label = crate::diagnostics::DiagnosticLevel::Help.label();
                    let help_cols = visible_width(help_label.as_ref()) + 2;
                    let prefix_cols = BOX_VERT_WIDTH
                        + indent.len() // ASCII spaces
                        + BOX_VERT_WIDTH
                        + 2 // the spaces around the connector
                        + connector.len() // ASCII spaces + `|`
                        + if ch == ' ' { help_cols } else { 0 };
                    if !needs_connector {
                        // Inline: `^^^^ message` on the underline row's
                        // gutter — no separate connector line.  The
                        // underline rows were already emitted above, so
                        // the message follows them directly, indented to
                        // the source-column position.
                        let sanitized_msg = super::sanitize_for_terminal_multiline(msg);
                        // Gutter width: `│` + indent + `│` + one space,
                        // then the message aligns under the source text.
                        let msg_prefix = BOX_VERT_WIDTH + indent.len() + BOX_VERT_WIDTH + 1;
                        let segments = self.wrap_msg(&sanitized_msg, msg_prefix);
                        for seg in segments.iter() {
                            // The whole segment is emitted with one indent prefix and
                            // one SGR on/off pair, and the tab expander ran at
                            // `start_col = indent_cols`; an embedded hard newline
                            // would break all three at once (indent once, SGR leaking
                            // across rows, wrong tab columns) with no other assertion
                            // firing.  `wrap_segments_never_contain_hard_newlines`
                            // is the differential; this is the cheap local tripwire.
                            debug_assert!(
                                !seg.contains('\n'),
                                "wrap segment carries a hard newline: {seg:?}"
                            );
                            // Highlight straight into the section scratch, then
                            // copy once — the gutter pieces below interleave, so
                            // the highlighted text must be assembled first.
                            let highlighted = {
                                hl_buf.clear();
                                if highlight_code_with_backticks_into(
                                    seg,
                                    self.s.use_color,
                                    &mut hl_buf,
                                ) {
                                    hl_buf.as_str()
                                } else {
                                    seg.as_ref()
                                }
                            };
                            out.push_str(self.s.dim);
                            out.push_str(self.bc.v);
                            out.push_str(self.s.reset);
                            out.push_str(&indent);
                            out.push_str(self.s.dim);
                            out.push_str(self.bc.sub_v);
                            out.push_str(self.s.reset);
                            out.push(' ');
                            out.push_str(level_color);
                            push_recologored(out, highlighted.as_ref(), level_color);
                            out.push_str(self.s.reset);
                            out.push('\n');
                        }
                        continue;
                    }
                    // Connector line, emitted once; each wrapped segment then
                    // repeats the connector so the label reads continuously.
                    out.push_str(self.s.dim);
                    out.push_str(self.bc.v);
                    out.push_str(self.s.reset);
                    out.push_str(&indent);
                    out.push_str(self.s.dim);
                    out.push_str(self.bc.sub_v);
                    out.push_str(self.s.reset);
                    out.push(' ');
                    out.push_str(connector_color);
                    out.push_str(&connector);
                    out.push_str(connector_reset);
                    out.push('\n');
                    let sanitized_msg = super::sanitize_for_terminal_multiline(msg);
                    let segments = self.wrap_msg(&sanitized_msg, prefix_cols);
                    for (i, seg) in segments.iter().enumerate() {
                        // Fixed gutter first, written straight into `out`
                        // (reusing its capacity); the segment text follows.
                        out.push_str(self.s.dim);
                        out.push_str(self.bc.v);
                        out.push_str(self.s.reset);
                        out.push_str(&indent);
                        out.push_str(self.s.dim);
                        out.push_str(self.bc.sub_v);
                        out.push_str(self.s.reset);
                        out.push(' ');
                        out.push_str(connector_color);
                        out.push_str(&connector);
                        out.push(' ');
                        out.push_str(connector_reset);
                        if ch == ' ' {
                            if i == 0 {
                                out.push_str(self.s.help);
                                out.push_str(help_label.as_ref());
                                out.push(':');
                                out.push_str(self.s.reset);
                                out.push(' ');
                            } else {
                                // Continuation lines must reserve the same
                                // `help: ` gutter the first line used, or the
                                // wrapped text starts left of the first line's
                                // message.
                                out.push_str(self.s.help);
                                push_spaces(out, help_cols);
                                out.push_str(self.s.reset);
                            }
                            out.push_str(self.s.help);
                            out.push_str(seg.as_ref());
                            out.push_str(self.s.reset);
                            out.push('\n');
                        } else {
                            let highlighted = {
                                hl_buf.clear();
                                if highlight_code_with_backticks_into(
                                    seg,
                                    self.s.use_color,
                                    &mut hl_buf,
                                ) {
                                    hl_buf.as_str()
                                } else {
                                    seg.as_ref()
                                }
                            };
                            // Re-apply the connector color after the reset
                            // pair `highlight` injects.  The pieces are written
                            // straight into `out`; the old `replace` allocated
                            // a second `String` per wrapped segment.  The
                            // opening color and the trailing reset stay
                            // outside the loop so the byte layout matches.
                            out.push_str(connector_color);
                            push_recologored(out, highlighted.as_ref(), connector_color);
                            out.push_str(connector_reset);
                            out.push('\n');
                        }
                    }
                }
            }

            // ── Multi-line label connectors ─────────────────────────
            // Only *middle* rows get a bare connector row here: on the
            // final covered line the message loop above already emits the
            // connector + message row, so drawing an `end` connector again
            // produced a stray duplicate `|` row.  ("start" stays skipped —
            // the label's first line carries its underline instead.)
            for (col, part) in multiline_flags.iter() {
                if *part == "middle" {
                    connector.clear();
                    connector.reserve(*col + 1);
                    push_spaces(&mut connector, *col);
                    connector.push('|');
                    out.push_str(self.s.dim);
                    out.push_str(self.bc.v);
                    out.push_str(self.s.reset);
                    out.push_str(&indent);
                    out.push_str(self.s.dim);
                    out.push_str(self.bc.sub_v);
                    out.push_str(self.s.reset);
                    out.push(' ');
                    out.push_str(connector_color);
                    out.push_str(&connector);
                    out.push_str(connector_reset);
                    out.push('\n');
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
        // A caller that under-populates `suggestion_lines` would silently
        // skip diff rendering for the tail suggestions (they still fall back
        // to text, so nothing crashes — the drift would be invisible).
        debug_assert_eq!(suggestions.len(), suggestion_lines.len());
        for (i, suggestion) in suggestions.iter().enumerate() {
            if self.quiet {
                continue;
            }
            let (Some(span), Some(replacement)) = (suggestion.span, &suggestion.replacement) else {
                continue;
            };
            if matches!(
                suggestion.style,
                crate::diagnostics::SuggestionStyle::CompletelyHidden
            ) {
                suggestion_rendered[i] = true;
                continue;
            }
            if matches!(
                suggestion.style,
                crate::diagnostics::SuggestionStyle::HideCodeAlways
            ) {
                // Hide the diff, but leave `suggestion_rendered[i]` false so
                // the text-only fallback still emits the help/note message.
                continue;
            }
            // `suggestion_lines[i]` is precomputed for this section's file
            // (None when the suggestion's span belongs to a different file),
            // so the per-section loop is a cheap range test — no `O(log n)`
            // `span_lines` lookup per section per suggestion, and a foreign
            // suggestion is skipped without re-examining its span.
            let Some(&(sug_first, sug_last)) = suggestion_lines.get(i).and_then(Option::as_ref)
            else {
                continue;
            };
            // Single-line spans only: a span that crosses lines would
            // silently drop the intermediate lines from the `-`/`+` diff,
            // producing semantically wrong output.  The text-only fallback
            // remains the correct path for those.
            if sug_first != sug_last {
                continue;
            }
            // A single-line span can still carry a replacement *containing*
            // a newline (e.g. "def\n"); splicing that into `new_line`
            // produces a malformed multi-line row under the single-line
            // diff layout, so fall back to the text-only path too.
            if replacement.contains('\n') || replacement.contains('\r') {
                continue;
            }
            if sug_first < first_line || sug_first > last_line {
                continue;
            }
            let old_line = idx.line(sug_first);
            // `starts` has one entry per line; `sug_first` comes from the
            // same index, so the lookup must be in range (a mismatch would
            // panic below).
            debug_assert!(
                sug_first < starts.len(),
                "suggestion line {sug_first} out of start-table range ({})",
                starts.len()
            );
            let line_start = starts[sug_first] as usize;
            let span_start_in_line = span
                .start_as_usize()
                .saturating_sub(line_start)
                .min(old_line.len());
            let span_end_in_line = span
                .end_as_usize()
                .saturating_sub(line_start)
                .min(old_line.len());
            // Defensive: a malformed span with `end < start` would make the
            // slice below panic; bail to the text-only fallback instead.
            if span_end_in_line < span_start_in_line
                || !old_line.is_char_boundary(span_start_in_line)
                || !old_line.is_char_boundary(span_end_in_line)
            {
                continue;
            }
            let mut new_line = String::with_capacity(
                old_line.len() + replacement.len() - (span_end_in_line - span_start_in_line),
            );
            new_line.push_str(&old_line[..span_start_in_line]);
            new_line.push_str(replacement);
            new_line.push_str(&old_line[span_end_in_line..]);
            // Raw source and replacement text are external input: route
            // them through the same sanitizer the label-message loop uses
            // so control characters (ESC, C1) and bidi overrides cannot
            // be injected into the terminal via the diff block.
            let old_display = super::sanitize_for_terminal_multiline(old_line);
            let new_display = super::sanitize_for_terminal_multiline(&new_line);

            let applicability = suggestion.applicability;
            let glyph = match applicability {
                // ASCII markers, not `√`: `√` is East-Asian-Ambiguous width, so
                // CJK terminals render it two cells wide and shift the `-`/`+`
                // payload out of alignment with the source rows above.
                crate::diagnostics::Applicability::MachineApplicable => ">",
                crate::diagnostics::Applicability::MaybeIncorrect => "?",
                _ => "",
            };
            let label: &'static str = match applicability {
                crate::diagnostics::Applicability::MachineApplicable => {
                    DiagnosticLevel::Help.label()
                }
                _ => DiagnosticLevel::Note.label(),
            };
            let label_color = match applicability {
                crate::diagnostics::Applicability::MachineApplicable => self.s.help,
                _ => self.s.note,
            };
            // The glyph slot always occupies one gutter column: an empty
            // glyph (`Unspecified` / `HasPlaceholders`) emits a literal
            // space so the `-`/`+` diff rows keep the same alignment as
            // glyph-bearing suggestions instead of shifting one column left.
            // Borrowed when the glyph is empty (the common no-style case) —
            // the old `" ".to_string()` paid a heap round-trip for one space.
            let glyph_pad: Cow<'_, str> = if glyph.is_empty() {
                Cow::Borrowed(" ")
            } else {
                Cow::Owned(format!(
                    "{dim}{glyph}{reset}",
                    dim = self.s.dim,
                    reset = self.s.reset
                ))
            };
            // `Unspecified` / `HasPlaceholders` render no glyph at all, but
            // the pad still occupies 1 column in the gutter (a blank cell),
            // so measure the *actual* rendered width instead of the glyph's
            // intrinsic width (which is 0 in that case).
            let glyph_rendered_width = if glyph.is_empty() {
                1
            } else {
                visible_width(glyph)
            };
            // The sanitizer keeps tabs as layout intent, but the main source
            // rows expand them via the row printer — the diff rows must do
            // the same, measured from the display column where the `- `/
            // `+ ` payload actually starts (gutter `│` + line number + space
            // + sub-bar `│` + space + glyph cell + the 2-column diff marker).
            let diff_text_col =
                BOX_VERT_WIDTH + line_num_width + 1 + BOX_VERT_WIDTH + 1 + glyph_rendered_width + 2;
            let old_display =
                expand_tabs_from_col(old_display.as_ref(), self.tab_width, diff_text_col);
            let new_display =
                expand_tabs_from_col(new_display.as_ref(), self.tab_width, diff_text_col);

            let sanitized_msg = super::sanitize_for_terminal_multiline(&suggestion.message);
            let first_prefix = BOX_VERT_WIDTH
                + indent.len()
                + BOX_VERT_WIDTH
                + 1
                + glyph_rendered_width
                + visible_width(label)
                + 2;
            let wrapped = self.wrap_msg(&sanitized_msg, first_prefix);
            let mut wrapped_iter = wrapped.iter();
            let first_wrapped = wrapped_iter
                .next()
                .unwrap_or(&std::borrow::Cow::Borrowed(""));

            // Blank gutter separator line (no line number — matches the
            // label-connector convention of not carrying a number on
            // non-source rows).
            out.push_str(self.s.dim);
            out.push_str(self.bc.v);
            out.push_str(self.s.reset);
            out.push_str(self.s.dim);
            push_spaces(out, line_num_width);
            out.push_str(self.s.reset);
            out.push(' ');
            out.push_str(self.s.dim);
            out.push_str(self.bc.sub_v);
            out.push_str(self.s.reset);
            out.push('\n');
            // Message line: `│ 12  message` — same gutter as the diff rows
            // below, whose markers are the ASCII `>` / `?` (never `√`:
            // East-Asian-Ambiguous, two cells wide in CJK terminals).
            // Gutter: `│` + `12` (line_num_width) + ` ` + `│` = 4 visible
            // columns when width is 2; the `+`/`-` rows below share the
            // same gutter so the two diff rows align.
            out.push_str(self.s.dim);
            out.push_str(self.bc.v);
            out.push_str(self.s.reset);
            out.push_str(self.s.dim);
            push_uint_padded(out, (sug_first + 1) as u64, line_num_width);
            out.push_str(self.s.reset);
            out.push(' ');
            out.push_str(self.s.dim);
            out.push_str(self.bc.sub_v);
            out.push_str(self.s.reset);
            out.push(' ');
            out.push_str(&glyph_pad);
            out.push_str(label_color);
            out.push_str(label);
            out.push(':');
            out.push_str(self.s.reset);
            out.push(' ');
            out.push_str(first_wrapped.as_ref());
            out.push('\n');
            for line in wrapped_iter {
                out.push_str(self.s.dim);
                out.push_str(self.bc.v);
                out.push_str(self.s.reset);
                out.push_str(self.s.dim);
                push_spaces(out, line_num_width);
                out.push_str(self.s.reset);
                out.push(' ');
                out.push_str(self.s.dim);
                out.push_str(self.bc.sub_v);
                out.push_str(self.s.reset);
                out.push(' ');
                out.push_str(&glyph_pad);
                // Align continuation lines under the *message*, not under the
                // glyph: the first line emits `glyph label: message`, so the
                // continuations skip the same `label: ` gutter (one cell for
                // the glyph pad + `label` + ": ").
                push_spaces(out, visible_width(label) + 2);
                out.push_str(line.as_ref());
                out.push('\n');
            }

            // `-` diff line: line number + old source text (sanitized).
            out.push_str(self.s.dim);
            out.push_str(self.bc.v);
            out.push_str(self.s.reset);
            out.push_str(self.s.dim);
            push_uint_padded(out, (sug_first + 1) as u64, line_num_width);
            out.push_str(self.s.reset);
            out.push(' ');
            out.push_str(self.s.dim);
            out.push_str(self.bc.sub_v);
            out.push_str(self.s.reset);
            out.push(' ');
            out.push_str(&glyph_pad);
            out.push_str(self.s.diff_del);
            out.push_str("- ");
            out.push_str(old_display.as_ref());
            out.push_str(self.s.reset);
            out.push('\n');
            // `+` diff line: line number + new source text (sanitized).
            // Carries the same line number as the `-` line (an inline
            // replacement, not an insertion), so the two rows share the
            // exact gutter prefix: `│` + line_num_width + ` ` + `│` = 4
            // visible columns when width is 2.  No stray leading space.
            out.push_str(self.s.dim);
            out.push_str(self.bc.v);
            out.push_str(self.s.reset);
            out.push_str(self.s.dim);
            push_uint_padded(out, (sug_first + 1) as u64, line_num_width);
            out.push_str(self.s.reset);
            out.push(' ');
            out.push_str(self.s.dim);
            out.push_str(self.bc.sub_v);
            out.push_str(self.s.reset);
            out.push(' ');
            out.push_str(&glyph_pad);
            out.push_str(self.s.diff_add);
            out.push_str("+ ");
            out.push_str(new_display.as_ref());
            out.push_str(self.s.reset);
            out.push('\n');

            suggestion_rendered[i] = true;
        }

        if !self.compact {
            out.push_str(self.s.dim);
            out.push_str(self.bc.v);
            out.push_str(&indent);
            out.push_str(self.bc.sub_bl);
            out.push_str(self.bc.sub_h);
            out.push_str(self.s.reset);
            out.push('\n');
        }

        // Restore the `mem::take`-n `String` scratch buffers so the next
        // section reuses their capacity.
        scratch.connector = connector;
        scratch.gutter_head = gutter_head;
        scratch.gutter_tail = gutter_tail;
        scratch.label_window = label_window.into_scratch();
        scratch.underlines_buf = underlines_buf;
        scratch.hl_buf = hl_buf;
        scratch.meta_buf = meta_buf;
        SectionOutcome {
            // The primary is visible only when its start line is inside the
            // drawn window; an EOF primary with only context rows must not
            // suppress the fallback location note.
            drew_primary: (first_line..=last_line).contains(&start_pos.line),
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
        cum: &[usize],
        plain: &str,
        seg: &'a str,
        search_from: usize,
    ) -> (Cow<'a, str>, usize) {
        let Some(rel) = plain.get(search_from..).and_then(|rest| rest.find(seg)) else {
            // Whitespace normalization broke the mapping: emit unstyled and
            // invalidate the offset cursor so later segments cannot match an
            // earlier duplicate occurrence.
            return (Cow::Borrowed(seg), plain.len().max(search_from));
        };
        let start = search_from + rel;
        let end = start + seg.len();
        // Cumulative byte offsets of `parts` are built once per message by
        // the caller (`write_child`) and passed in: this function runs once
        // per *wrapped segment*, so building the table here — despite the
        // comment above claiming otherwise — kept the per-message cost at
        // O(P·S), exactly what it said it removed.  With the table hoisted,
        // the per-message cost is O(P + S·log P + Σ overlap).
        let acc = cum
            .last()
            .map_or(0, |&off| off + parts[cum.len() - 1].content.len());
        let mut rendered = String::with_capacity(seg.len() + 16);
        let mut first = cum.partition_point(|&off| off < start);
        // A part may *straddle* `start` (its offset is below `start` but its
        // end is past it) — step back one so the binary search finds the
        // first part that *overlaps*, not the first that begins inside.
        if first > 0 && cum[first - 1] + parts[first - 1].content.len() > start {
            first -= 1;
        }
        let mut pos = cum.get(first).copied().unwrap_or(acc);
        for part in &parts[first..] {
            let pstart = pos;
            let pend = pos + part.content.len();
            pos = pend;
            if pstart >= end {
                // Parts are laid out in order: nothing after this one can
                // overlap the segment.
                break;
            }
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
        let prefix: &'static str = child.level.label();
        let prefix_cols = self.prefix_cols(child.level);
        // Wrap the plain message; when a styled version is present, map each
        // wrapped segment back onto its parts so highlights survive per line.
        // `plain_message` borrows the stored plain text when the child has
        // one; a styled-only child derives it here, at the render boundary,
        // so a note that is never emitted never allocates it.
        // User text written between SGR runs: sanitize before wrapping and
        // before styled mapping.  A control char becomes U+FFFD, which
        // changes *byte* lengths (+2 per C0/DEL, +1 per C1), so the styled
        // parts are sanitized in lockstep whenever the plain text changed —
        // otherwise the `search_from` byte offsets into their concatenation
        // drift and `styled_segment` can slice mid-character.
        let message = child.plain_message();
        let sanitized = super::sanitize_for_terminal_multiline(message.as_ref());
        let plain = sanitized.as_ref();
        // `plain` is the sanitized concatenation of the raw parts.  Do NOT
        // infer "sanitization was a no-op" from equal lengths: replacements
        // grow C0/DEL by +2 and C1 by +1, while removing a bidi override
        // shrinks by −3, so a balanced payload (+6 −6) keeps the total length
        // *and* keeps the control bytes.  Sanitize every part unconditionally;
        // sanitization is a per-char map, so it commutes with concatenation
        // and `cum` over the sanitized parts stays aligned with `plain`.  The
        // zero-allocation fast path is recovered by checking the resulting
        // `Cow`s (a part is `Borrowed` only when it was unchanged).
        let sanitized_parts: SmallVec<[crate::diagnostics::StringPart<'_>; 8]>;
        // A `styled_message: Some(vec![])` alongside a non-empty plain
        // message is a malformed-but-constructible child: styled mapping
        // would emit nothing for every segment and the text would vanish,
        // so an empty parts list falls back to the unstyled path.
        let styled_parts: Option<&[crate::diagnostics::StringPart<'_>]> = match child.styled_message
        {
            Some(ref styled) if !styled.0.is_empty() => {
                sanitized_parts = styled
                    .0
                    .iter()
                    .map(|p| crate::diagnostics::StringPart {
                        content: super::sanitize_for_terminal_multiline(&p.content),
                        style: p.style,
                    })
                    .collect();
                let unchanged = sanitized_parts
                    .iter()
                    .all(|p| matches!(p.content, Cow::Borrowed(_)));
                if unchanged {
                    Some(&styled.0)
                } else {
                    Some(&sanitized_parts)
                }
            }
            Some(_) => None, // empty styled parts: unstyled fallback above
            None => None,
        };
        let mut search_from = 0usize;
        // Cumulative byte offsets of `styled_parts` — built once per message
        // here, not once per wrapped segment inside `styled_segment` (which
        // ran S times per message and kept the cost at O(P·S) despite its
        // comment claiming O(P + S·log P)).
        let mut cum: SmallVec<[usize; 8]> = SmallVec::new();
        if let Some(parts) = styled_parts {
            let mut acc = 0usize;
            for p in parts {
                cum.push(acc);
                acc += p.content.len();
            }
        }
        for (i, seg) in self.wrap_msg(plain, prefix_cols).iter().enumerate() {
            // Keep the wrapped segment as a `Cow`: the unstyled path (no
            // `styled_message`) then borrows straight through, and the styled
            // fallback inside `styled_segment` borrows too — neither needs the
            // old `into_owned()`/`to_string()` heap allocation.
            let msg: Cow<'_, str> = match styled_parts {
                Some(parts) => {
                    let (rendered, next) =
                        self.styled_segment(parts, &cum, plain, seg.as_ref(), search_from);
                    search_from = next;
                    rendered
                }
                None => seg.clone(),
            };
            if i == 0 {
                out.push_str(self.s.dim);
                out.push_str(self.bc.v);
                out.push_str(self.s.reset);
                out.push_str("  ");
                out.push_str(self.s.get(Self::level_style(child.level)));
                out.push_str("= ");
                out.push_str(self.s.bold);
                out.push_str(prefix.as_ref());
                out.push_str(self.s.reset);
                out.push_str(": ");
                out.push_str(msg.as_ref());
                out.push('\n');
            } else {
                // Continuation: `│` + pad aligning the segment to column
                // `prefix_cols` (where the first line's message started).
                // `push_spaces` writes straight into `out` — a one-line child
                // never wraps, and a wrapped child allocates no throwaway pad.
                //
                // Deliberately *unstyled*, matching this arm's first line: only
                // the `= <prefix>` part is coloured above, so the message body is
                // the terminal's default colour on both lines.  (The note/help
                // writers do colour their body, which is why their continuations
                // carry the SGR run and this one must not.)
                out.push_str(self.s.dim);
                out.push_str(self.bc.v);
                out.push_str(self.s.reset);
                push_spaces(out, prefix_cols - 1);
                out.push_str(msg.as_ref());
                out.push('\n');
            }
        }
    }

    fn write_note_line(&self, out: &mut String, msg: &str) {
        self.write_note_line_with(out, msg, None);
    }

    /// Canonical note emitter.  `suffix` is appended to the final wrapped
    /// segment as ` at <suffix>`; the fallback-label loop and the foreign-span
    /// loop both route through this so coloring, continuation padding, and
    /// locator placement cannot diverge.
    fn write_note_line_with(&self, out: &mut String, msg: &str, suffix: Option<&str>) {
        let prefix_cols = self.prefix_cols(DiagnosticLevel::Note);
        let sanitized = super::sanitize_for_terminal_multiline(msg);
        let level_label = DiagnosticLevel::Note.label();
        let suffix_cols = suffix.map_or(0, |s| 4 + visible_width(s));
        let avail_full = self.max_width.saturating_sub(prefix_cols).max(1);
        let mut segments = self.wrap_msg(&sanitized, prefix_cols);
        // Wrap the full measure first; only when the locator suffix would
        // overflow the last line does the reserve enter the layout.
        //
        // The reserve is applied by re-wrapping the *whole* text at
        // `avail_full` with `last_reserve = suffix_cols`, not by re-wrapping the
        // overflowing tail at the narrowed measure.  The two-phase composition Φ
        // = `wrap(M) then wrap(narrow, tail)` is unboundedly worse on line count
        // (it can never pull words from earlier lines onto the reserved line),
        // which is exactly the defect `wrap_optimal_reserved` was written to
        // remove; `research/locator-wrap/` has the proof and the witness family.
        // Byte output is unchanged whenever the suffix already fits (the gate
        // below is false), so only the previously-suboptimal path moves.
        if !self.no_wrap
            && suffix_cols > 0
            && let Some(last_seg) = segments.last()
            && visible_width_with_tab(last_seg.as_ref(), self.tab_width) + suffix_cols > avail_full
        {
            segments = super::wrap::wrap_optimal_reserved(
                &sanitized,
                avail_full,
                self.tab_width,
                suffix_cols,
            );
        }
        let last = segments.len().saturating_sub(1);
        for (i, seg) in segments.iter().enumerate() {
            if i == 0 {
                out.push_str(self.s.dim);
                out.push_str(self.bc.v);
                out.push_str(self.s.reset);
                out.push_str("  ");
                out.push_str(self.s.note);
                out.push_str("= ");
                out.push_str(self.s.bold);
                out.push_str(level_label);
                out.push_str(self.s.reset);
                out.push_str(": ");
                out.push_str(self.s.note);
                out.push_str(seg.as_ref());
                out.push_str(self.s.reset);
            } else {
                // Built only on the wrap path — a one-line note never wraps.
                // The continuation repeats the first line's SGR run: without it a
                // wrapped note rendered coloured on line 1 and default-coloured
                // from line 2 on (`wrap_msg`'s contract is a full open/reset pair
                // per segment).
                out.push_str(self.s.dim);
                out.push_str(self.bc.v);
                out.push_str(self.s.reset);
                push_spaces(out, prefix_cols - 1);
                out.push_str(self.s.note);
                out.push_str(seg.as_ref());
                out.push_str(self.s.reset);
            }
            if i == last
                && let Some(suffix) = suffix
            {
                // Inside the coloured run: the locator belongs to the note.
                out.push_str(self.s.note);
                out.push_str(" at ");
                out.push_str(suffix);
                out.push_str(self.s.reset);
            }
            out.push('\n');
        }
    }

    fn write_help_line(&self, out: &mut String, msg: &str) {
        let prefix_cols = self.prefix_cols(DiagnosticLevel::Help);
        let sanitized = super::sanitize_for_terminal_multiline(msg);
        let level_label = DiagnosticLevel::Help.label();
        for (i, seg) in self.wrap_msg(&sanitized, prefix_cols).iter().enumerate() {
            if i == 0 {
                out.push_str(self.s.dim);
                out.push_str(self.bc.v);
                out.push_str(self.s.reset);
                out.push_str("  ");
                out.push_str(self.s.help);
                out.push_str("= ");
                out.push_str(self.s.bold);
                out.push_str(level_label);
                out.push_str(self.s.reset);
                out.push_str(": ");
                out.push_str(self.s.help);
                out.push_str(seg.as_ref());
                out.push_str(self.s.reset);
                out.push('\n');
            } else {
                // Built only on the wrap path — a one-line help never wraps.
                // Same per-segment SGR contract as line 1 (see `wrap_msg`).
                out.push_str(self.s.dim);
                out.push_str(self.bc.v);
                out.push_str(self.s.reset);
                push_spaces(out, prefix_cols - 1);
                out.push_str(self.s.help);
                out.push_str(seg.as_ref());
                out.push_str(self.s.reset);
                out.push('\n');
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
    memchr::memchr_iter(b'\n', &source.as_bytes()[..clamped]).count()
}

// ── Helper: sliding window over a section's labels ─────────────

/// Sliding-window selector over a section's labels, driven by the
/// monotonically advancing rendered line index.
///
/// The old per-row scan used to test every label against every row
/// (O(lines × labels)); here the labels are sorted once by `start_line`, and
/// as the row index advances each label enters the window at its start line
/// and leaves it after its end line, so a row only touches the labels that
/// can actually cover it — O(labels·log labels + lines + active·log active).
///
/// `active` is a sorted `SmallVec<[usize; 8]>` keyed by original label index,
/// so the per-row output iterates in **original label order** (message/connector
/// order is observable and snapshot-pinned) while expiry removals are `O(active)`
/// shifts — faster than `BTreeSet`'s `O(log active)` for the common `active ≤ 8`
/// case, which stays inline with zero heap allocation.  Expiry is driven by a
/// min-heap of `(end_line, idx)`: `advance` pops only the entries that actually
/// ended, so a section with L short-span labels interleaved with L long-span
/// ones — which used to trigger L `retain`s of `O(L)` (`O(L²)`) — is now
/// `O(L log L)`.
///
/// `label_lines` is the per-label `(start_line, end_line)` slice (indexed
/// exactly like the label slice) and must outlive the window.
pub(crate) struct LabelWindow<'a> {
    label_lines: &'a [(usize, usize)],
    scratch: LabelWindowScratch,
    /// Index into `scratch.order` of the first label not yet folded in.
    cursor: usize,
    /// Debug-only: the last row folded in.  `advance` is *contractually*
    /// idempotent for the same row — the HTML renderer calls it again after a
    /// fast-skip — and that only holds if the fold never walks backwards.
    #[cfg(debug_assertions)]
    last_advanced: Option<usize>,
}

/// Reusable scratch state for a [`LabelWindow`].
///
/// Hoisted into [`SectionScratch`] so a diagnostic with K source sections
/// pays one set of `SmallVec` / `BinaryHeap` capacities instead of K:
/// [`LabelWindow::new`] takes the scratch by value, and
/// [`LabelWindow::into_scratch`] hands it back after the section renders.
///
/// The three per-row scratch vectors (`new_labels`, `expired`, `merged`) and
/// the `order` / `active` vectors are all reused across `advance` calls within
/// one window *and* across the K windows of one diagnostic.  `expiry` — the
/// one always-heap-allocated field — is where the section-to-section win is
/// largest.
pub(crate) struct LabelWindowScratch {
    /// Label indices sorted by `start_line` (stable: ties keep original
    /// order).  Built once per section by [`LabelWindow::new`]; when the
    /// caller pre-sorts the anchor labels (shared across a diagnostic's K
    /// sections), only the per-section extras are sorted and merged into this
    /// order — see [`LabelWindow::new`].
    ///
    /// `SmallVec<[usize; 16]>`: labels-per-diagnostic rarely exceeds 4, so the
    /// common case stays inline.
    order: SmallVec<[usize; 16]>,
    /// Labels covering the current row, keyed by original index so iteration
    /// yields the emission order.  Kept sorted by index via binary-search
    /// insert/remove — `O(active)` shifts, but `active ≤ 8` stays inline.
    active: SmallVec<[usize; 8]>,
    /// Min-heap of `(end_line, idx)` for the labels in `active`.  Each label
    /// is pushed once when it enters and popped once when its `end_line` falls
    /// below the current row, so `advance` touches only the expired labels.
    expiry: BinaryHeap<Reverse<(usize, usize)>>,
    // ── Per-`advance` scratch, hoisted so capacity survives across rows ──
    /// Labels entering the window on the current row.  Inline for ≤8 (the
    /// common case); capacity is reused across `advance` calls.
    new_labels: SmallVec<[usize; 8]>,
    /// Labels leaving the window on the current row.  Inline for ≤8.
    expired: SmallVec<[usize; 8]>,
    /// Merge buffer for the burst path (`new_labels.len() > 1 || expired.len() > 1`).
    /// `active` and `merged` are swapped at the end of a burst so the buffer's
    /// capacity is reused on the next burst instead of a fresh
    /// `SmallVec::with_capacity`.
    merged: SmallVec<[usize; 8]>,
}

impl LabelWindowScratch {
    pub(crate) fn new() -> Self {
        Self {
            order: SmallVec::new(),
            active: SmallVec::new(),
            expiry: BinaryHeap::new(),
            new_labels: SmallVec::new(),
            expired: SmallVec::new(),
            merged: SmallVec::new(),
        }
    }
}

impl Default for LabelWindowScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> LabelWindow<'a> {
    /// Build the start-line-sorted `order` for `label_lines`, reusing
    /// `scratch`'s buffers (and taking ownership of it for the window's
    /// lifetime).
    ///
    /// `sorted_prefix` is a start-line-sorted permutation of
    /// `label_lines[..sorted_prefix.len()]` (the anchor labels, pre-sorted
    /// once per diagnostic and shared by every section).  The remaining
    /// `label_lines[sorted_prefix.len()..]` (the per-section extras) are
    /// sorted here and merged with `sorted_prefix` in `O(L + E)`, so a
    /// diagnostic with K sections and L shared labels pays `O(L log L)` once
    /// plus `O(K · (L + E))` for the merges, instead of re-sorting the whole
    /// `O(K · (L + E) log(L + E))` union per section.  Pass `&[]` to sort
    /// everything (no pre-sort available, e.g. the HTML renderer).
    pub(crate) fn new(
        label_lines: &'a [(usize, usize)],
        sorted_prefix: &[usize],
        mut scratch: LabelWindowScratch,
    ) -> Self {
        // Reset every field the previous section left behind.  `clear`
        // preserves capacity — the point of threading a scratch through the
        // caller's `SectionScratch`.
        scratch.order.clear();
        scratch.active.clear();
        scratch.expiry.clear();
        scratch.new_labels.clear();
        scratch.expired.clear();
        scratch.merged.clear();

        let n = label_lines.len();
        let order = &mut scratch.order;
        if sorted_prefix.is_empty() {
            order.extend(0..n);
            order.sort_by_key(|&i| label_lines[i].0);
        } else {
            let prefix_len = sorted_prefix.len();
            debug_assert!(
                prefix_len <= n,
                "pre-sorted prefix ({prefix_len}) exceeds the label count ({n})"
            );
            if prefix_len > n {
                // Unreachable in a correct caller (asserted above): copy the
                // usable part through instead of indexing `label_lines` out of
                // range, which is what the merge path would do.
                order.extend_from_slice(&sorted_prefix[..n]);
            } else {
                // `prefix_len == n` (every label pre-sorted) routes through
                // here too: `extras` is then empty and the merge degenerates
                // to the same copy, so the two cases cannot drift apart.
                let mut extras: SmallVec<[usize; 16]> = (prefix_len..n).collect();
                extras.sort_by_key(|&i| label_lines[i].0);
                let mut a = 0usize;
                let mut b = 0usize;
                while a < prefix_len && b < extras.len() {
                    if label_lines[sorted_prefix[a]].0 <= label_lines[extras[b]].0 {
                        order.push(sorted_prefix[a]);
                        a += 1;
                    } else {
                        order.push(extras[b]);
                        b += 1;
                    }
                }
                order.extend_from_slice(&sorted_prefix[a..]);
                order.extend_from_slice(&extras[b..]);
            }
        }

        Self {
            label_lines,
            scratch,
            cursor: 0,
            #[cfg(debug_assertions)]
            last_advanced: None,
        }
    }

    /// Consume the window and return its scratch buffers so the next
    /// section of the same render can reuse their capacity.
    pub(crate) fn into_scratch(self) -> LabelWindowScratch {
        self.scratch
    }

    /// Fold `line_idx` into the window. Rows must be visited in ascending
    /// order (the renderers iterate `first_line..=last_line`).
    /// `true` when no label can touch this row: nothing is active and the next
    /// label starts *after* it.  Used to keep such a row out of the annotated
    /// path (skipping `advance` alone is not enough — the branch also decides
    /// whether connectors/markers are drawn).
    fn is_idle_for(&self, line: usize) -> bool {
        self.scratch.active.is_empty()
            && self
                .scratch
                .order
                .get(self.cursor)
                .is_none_or(|&i| self.label_lines[i].0 > line)
    }

    pub(crate) fn advance(&mut self, line_idx: usize) {
        // Re-advancing the same row is a no-op by construction (the cursor is
        // already past it), and a *backwards* advance would silently mis-fold
        // labels; the HTML fast-skip relies on the former, so pin the shape that
        // makes both true.
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                self.last_advanced.is_none_or(|prev| prev <= line_idx),
                "LabelWindow::advance went backwards: {:?} -> {line_idx}",
                self.last_advanced
            );
            self.last_advanced = Some(line_idx);
        }
        let lines = self.label_lines;
        let scratch = &mut self.scratch;

        // Hoisted scratch — `take` out so the merge loop can borrow `scratch.active`
        // while we write `merged`, then restore at the end so capacity survives.
        // `clear()` handles any items left from the previous `advance` (the
        // fast path returns the locals with 0–1 items still present).
        let mut new_labels = std::mem::take(&mut scratch.new_labels);
        new_labels.clear();
        let mut expired = std::mem::take(&mut scratch.expired);
        expired.clear();

        // Phase 1: fold in labels whose start line has arrived.
        let mut first_start: Option<usize> = None;
        let mut multi_start = false;
        while self.cursor < scratch.order.len() {
            let i = scratch.order[self.cursor];
            let start = lines[i].0;
            if start > line_idx {
                break;
            }
            match first_start {
                None => first_start = Some(start),
                Some(s) if s != start => multi_start = true,
                _ => {}
            }
            new_labels.push(i);
            scratch.expiry.push(Reverse((lines[i].1, i)));
            self.cursor += 1;
        }

        // Phase 2: collect expired.
        while let Some(&Reverse((end, i))) = scratch.expiry.peek() {
            if end >= line_idx {
                break;
            }
            scratch.expiry.pop();
            expired.push(i);
        }

        // Fast path — the overwhelmingly common case (b ≤ 1, e ≤ 1).
        // Identical to the old per-item code, so no regression on the hot
        // path.
        if new_labels.len() <= 1 && expired.len() <= 1 {
            if let Some(&i) = new_labels.first() {
                scratch
                    .active
                    .insert(scratch.active.partition_point(|&x| x < i), i);
            }
            if let Some(&i) = expired.first() {
                if let Ok(pos) = scratch.active.binary_search(&i) {
                    scratch.active.remove(pos);
                }
            }
            scratch.new_labels = new_labels;
            scratch.expired = expired;
            return;
        }

        // Mixed small case — at most one arrival plus one or two removals
        // (the common "a few labels over a handful of rows" shape between
        // the single-item fast path and a real burst).  `active` is
        // index-sorted, so this is one binary-search insert plus at most two
        // binary-searched `remove`s — no `reserve`, no full merge loop over
        // the whole active set.
        if new_labels.len() <= 1 && expired.len() <= 2 {
            if let Some(&i) = new_labels.first() {
                scratch
                    .active
                    .insert(scratch.active.partition_point(|&x| x < i), i);
            }
            // `expired` comes off the end-line heap, so index order is
            // arbitrary — sort the (≤ 2) pair before removing.
            if expired.len() == 2 && expired[0] > expired[1] {
                expired.swap(0, 1);
            }
            for &i in &expired {
                if let Ok(pos) = scratch.active.binary_search(&i) {
                    scratch.active.remove(pos);
                }
            }
            scratch.new_labels = new_labels;
            scratch.expired = expired;
            return;
        }

        // Burst: batch merge-and-compact.  `active` is sorted by index.
        // Same-start batches are already index-sorted (stable `sort_by_key`
        // over `0..n`); cross-start batches need a sort.
        if multi_start && new_labels.len() > 1 {
            new_labels.sort_unstable();
        }
        if expired.len() > 1 {
            expired.sort_unstable();
        }

        let mut merged = std::mem::take(&mut scratch.merged);
        merged.clear();
        merged.reserve(scratch.active.len() + new_labels.len());
        let (mut ai, mut ni, mut ei) = (0, 0, 0);
        loop {
            let next = match (scratch.active.get(ai), new_labels.get(ni)) {
                (Some(&a), Some(&n)) => {
                    if a <= n {
                        ai += 1;
                        a
                    } else {
                        ni += 1;
                        n
                    }
                }
                (Some(&a), None) => {
                    ai += 1;
                    a
                }
                (None, Some(&n)) => {
                    ni += 1;
                    n
                }
                (None, None) => break,
            };
            while ei < expired.len() && expired[ei] < next {
                ei += 1;
            }
            if ei < expired.len() && expired[ei] == next {
                ei += 1;
                continue;
            }
            merged.push(next);
        }
        // Swap so `scratch.active` gets the freshly merged content and
        // `merged` retains the old `active`'s capacity for the next burst.
        core::mem::swap(&mut scratch.active, &mut merged);
        scratch.merged = merged;
        scratch.new_labels = new_labels;
        scratch.expired = expired;
    }

    /// Label indices covering the current row, in original order.
    pub(crate) fn active(&self) -> &[usize] {
        &self.scratch.active
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
///
/// The meta payload carries only `(underline_char, label_idx)`: the message
/// is always `labels[label_idx].message` (the synthetic primary has no
/// label and an empty message), so callers resolve it at use sites from
/// their own `labels` slice.  Dropping the `&str` from the payload is what
/// frees `meta_buf` (and the `underlines` buffer derived from it) from the
/// per-section borrow — they can then live in [`SectionScratch`] and be
/// reused across the batch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_line_annotation_ranges<'a, 'src>(
    primary: Span,
    labels: &[&'a Label<'src>],
    window: &mut LabelWindow<'_>,
    source: &str,
    line: &str,
    line_idx: usize,
    line_starts: &[u32],
    ranges_buf: &mut Vec<(usize, usize)>,
    meta_buf: &mut SmallVec<[(char, Option<usize>); 8]>,
    // `false` when the caller already advanced `window` to `line_idx` for this
    // row (the HTML fast-skip branch peeks `active()` first); avoids a
    // redundant expiry-heap pop / merge on those rows.
    advance: bool,
) {
    ranges_buf.clear();
    meta_buf.clear();

    let line_start = line_starts
        .get(line_idx)
        .map_or(source.len(), |&s| s as usize);
    let line_end = line_start + line.len();

    // Narrow the candidate labels to those whose line range covers this row.
    if advance {
        window.advance(line_idx);
    }

    // The synthetic primary's message is empty (`""`), matching the old
    // `Label { message: String::new(), kind: Primary, .. }` placeholder.
    let primary_ann = (
        primary,
        AnnotationKind::Primary
            .underline_char()
            .expect("the primary annotation always underlines"),
        None,
    );
    // `Help` has no underline glyph (`underline_char` is `None`) — it renders as
    // a note line — so it must contribute no underline range at all.  Filtering
    // *here* keeps `ranges_buf` and `meta_buf` aligned: the two pushes below are
    // parallel, so a skipped range has to skip its glyph too.  Before `Option`
    // this path pushed a run of spaces, i.e. invisible corruption.
    let label_anns = window.active().iter().filter_map(|&i| {
        let l = labels[i];
        l.underline_char().map(|c| (l.span, c, Some(i)))
    });
    for (span, underline_char, label_idx) in std::iter::once(primary_ann).chain(label_anns) {
        let span_start = span.start_as_usize();
        let span_end = span.end_as_usize();

        // Does this annotation's span overlap this line? A zero-width
        // annotation (an insertion point) belongs to the line whose
        // `[line_start, line_end]` range contains the point — including
        // both ends of an empty line.
        let intersects = if span_start == span_end {
            // Include this line's terminator bytes (CRLF included) but not
            // the first byte of the next line.
            let zero_end = line_starts
                .get(line_idx + 1)
                .map_or(source.len().saturating_sub(1), |&s| {
                    (s as usize).saturating_sub(1)
                })
                .max(line_end);
            span_start >= line_start && span_start <= zero_end
        } else {
            span_end > line_start && span_start < line_end
        };
        if !intersects {
            continue;
        }

        let multi_line = span_start < line_start || span_end > line_end;
        // Multi-line label policy (shared with HTML via this function):
        // start line underlines its start segment, end line underlines its
        // end segment, and middle lines draw *only* the connector row.  The
        // previous full-line `^^^` on middle lines duplicated the connector
        // and burned underline lanes without adding information.
        // Synthetic extra primaries (merged groups) carry an empty message and
        // must behave like the section primary on every covered row; only
        // real user labels get the endpoint-only multi-line policy.
        let synthetic_extra = label_idx.is_some_and(|i| labels[i].message.is_empty());
        if label_idx.is_some() && multi_line && !synthetic_extra {
            let is_start_line = span_start >= line_start;
            let is_end_line = span_end <= line_end;
            if !is_start_line && !is_end_line {
                continue;
            }
            let start_offset = if is_start_line {
                span_start - line_start
            } else {
                0
            };
            let end_offset = if is_end_line {
                span_end - line_start
            } else {
                line.len()
            };
            ranges_buf.push((start_offset.min(line.len()), end_offset.min(line.len())));
            meta_buf.push((underline_char, label_idx));
            continue;
        }

        let start_offset = span_start.saturating_sub(line_start);
        let end_offset = if span_end < line_end {
            span_end - line_start
        } else {
            line.len()
        };
        ranges_buf.push((start_offset, end_offset));
        meta_buf.push((underline_char, label_idx));
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
fn merged_primary_ranges(
    ranges: &[(usize, usize)],
    meta: &[(char, Option<usize>)],
    len: usize,
) -> SmallVec<[(usize, usize); 4]> {
    // The overwhelmingly common case is 0 or 1 primary interval (the
    // synthetic primary).  Count first instead of paying `collect`'s
    // iterator machinery — closure call, `enumerate`, `SmallVec` growth —
    // once per rendered source row; only multi-primary rows fall through to
    // the sort + merge pass.
    let mut first: Option<(usize, usize)> = None;
    for (i, &(start, end)) in ranges.iter().enumerate() {
        if meta[i].0 == '^' && start <= len {
            match first {
                None => first = Some((start, end.min(len))),
                // Second interval: leave the buffer empty and let the
                // original iterator below rebuild it in order.
                Some(_) => {
                    let mut sorted: SmallVec<[(usize, usize); 4]> = ranges
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &(start, end))| {
                            (meta[i].0 == '^' && start <= len).then(|| (start, end.min(len)))
                        })
                        .collect();
                    sorted.sort_unstable();
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
                    return merged;
                }
            }
        }
    }
    match first {
        None => SmallVec::new(),
        Some(p) => smallvec![p],
    }
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
/// Exact byte-level predicate for the render path's sanitize gate: does
/// `line` contain a control character other than tab (C0 \ {0x09}, DEL,
/// C1 via `C2 80..9F`) or a bidi override/isolate?
///
/// Semantics match `line.bytes/chars().any(|c| (c.is_control() && c != '\t')
/// || is_bidi_override(c))` precisely, but every stage is a byte scan: the
/// C0/DEL stage is SWAR with a scalar rescan of dirty 8-byte chunks (the
/// masks are existentially exact but not position-exact, so they only skip
/// clean chunks), C1 uses `memchr(0xC2)` plus a one-byte follow check, and
/// the bidi stage reuses the exact [`has_bidi_override_bytes`]
/// (`E2 80 AA..AE` / `E2 81 A6..A9`).  No stage decodes characters, and
/// there are no false positives — lines containing `°`, `§`, `±`, emoji, or
/// math symbols (the common European/technical case) now return `false`
/// without a single char decode.
fn line_needs_sanitize(line: &str) -> bool {
    let bytes = line.as_bytes();

    // 1) C0 (excluding tab/newline) and DEL: reuse the shared scanner rather
    // than maintaining a second, drift-prone copy of the same SWAR masks.
    let scan = super::scan_terminal_control_flags_multiline(bytes);
    if scan.dirty || scan.has_c1 {
        return true;
    }

    // 4) Bidi overrides / isolates — exact byte predicate from `mod.rs`, gated on
    // the `0xE2` flag the control scan already computed (every override is a
    // leading-`0xE2` sequence), so a Cyrillic/CJK line skips the `memchr` pass.
    scan.has_e2 && super::has_bidi_override_bytes(bytes)
}

/// Emit bytes `[a, b)` of a printable-ASCII+tab line, splitting at tabs:
/// non-tab runs go through `emit` verbatim, each tab expands to `push_spaces`
/// with its precomputed stop width.  Part of `render_line_merged_with`'s
/// tab-aware fast path; `tabs` (`(byte offset, stop width)` pairs) comes from
/// the caller's one-pass tab scan and is sorted by offset.
fn emit_tab_run<F: FnMut(&mut String, &str)>(
    rendered: &mut String,
    line: &str,
    a: usize,
    b: usize,
    tabs: &[(usize, usize)],
    emit: &mut F,
) {
    let mut a = a;
    loop {
        let k = tabs.partition_point(|&(t, _)| t < a);
        if k < tabs.len() && tabs[k].0 < b {
            let (t, w) = tabs[k];
            if t > a {
                emit(rendered, &line[a..t]);
            }
            push_spaces(rendered, w);
            a = t + 1;
        } else {
            if b > a {
                emit(rendered, &line[a..b]);
            }
            return;
        }
    }
}

pub(crate) struct LineRenderArgs<'a, 'b> {
    pub line: &'a str,
    pub ranges: &'a [(usize, usize)],
    pub meta: &'a [(char, Option<usize>)],
    pub out: &'b mut SmallVec<[(usize, usize, char, Option<usize>); 8]>,
    pub rendered: &'b mut String,
    pub highlight_on: &'a str,
    pub highlight_off: &'a str,
    pub display_buf: &'b mut Vec<(usize, usize)>,
    pub order_buf: &'b mut Vec<usize>,
    pub done_buf: &'b mut Vec<bool>,
    pub active_buf: &'b mut BinaryHeap<Reverse<(usize, usize)>>,
    pub tab_width: usize,
    pub content_start_col: usize,
    pub append: bool,
}

pub(crate) fn render_line_merged_with<F: FnMut(&mut String, &str)>(
    args: LineRenderArgs<'_, '_>,
    mut emit: F,
) -> usize {
    let LineRenderArgs {
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
        content_start_col,
        append,
    } = args;
    if !append {
        rendered.clear();
    }
    display_buf.clear();
    out.clear();

    let len = line.len();

    // Pre-filter and fuse `^` ranges so the per-position highlight check is
    // one cursor sweep.  The byte-overlap formula (`gend > b1 && byte_idx <
    // b2`) covers *all* the cases the display-column check did — including
    // zero-width ranges strictly inside a multi-byte grapheme, which
    // `byte_range_to_display` expands to the whole cluster.
    let primary_ranges = if highlight_on.is_empty() && highlight_off.is_empty() {
        SmallVec::new()
    } else {
        merged_primary_ranges(ranges, meta, len)
    };

    // ── ASCII fast path: byte offset ≡ display column ──
    // One classification pass feeds both byte-level branches below; testing
    // `is_printable_ascii` and then `is_printable_ascii_or_tab` scanned a
    // tab-indented row twice.
    let ascii_class = crate::diagnostics::ascii::classify_ascii(line.as_bytes());
    if ascii_class == crate::diagnostics::ascii::AsciiClass::Printable {
        // `col`/`width` are a closed-form function of the range here (byte
        // offset ≡ display column, nothing else writes `display_buf` before
        // it is read), so the annotation records are computed straight into
        // `out` at the point of consumption — no scratch round-trip.  The
        // `display_buf` scratch remains for the non-ASCII grapheme walk
        // below, which materialises intermediate `(start, end)` states.
        let fill_out = |out: &mut SmallVec<[(usize, usize, char, Option<usize>); 8]>,
                        ranges: &[(usize, usize)]| {
            out.reserve(ranges.len());
            for (&(start, end), &(ch, label_idx)) in ranges.iter().zip(meta.iter()) {
                let end_c = end.min(len);
                let (col, width) = if end > start && start < len {
                    (start, end_c - start)
                } else {
                    (end_c, 0)
                };
                // Insertion points (`width == 0`) get a one-cell marker.
                out.push((col, width.max(1), ch, label_idx));
            }
        };

        // No primary annotations: the interval walk below would emit one
        // unhighlighted segment, so skip it and emit once — the common case
        // for rows with only secondary or note labels.
        if primary_ranges.is_empty() {
            emit(rendered, line);
            fill_out(out, ranges);
            return len;
        }

        // O(intervals) walk instead of an O(len) per-byte sweep.  The old
        // loop called `primary_at` once per byte regardless of how many
        // primary intervals existed — a 500 KB line with one `^` interval paid
        // ~500 K cursor probes to discover one interval.  `primary_ranges` is
        // already sorted, disjoint (adjacent/overlapping runs fused) and
        // `len`-clipped, so we emit slices directly: unhighlighted gaps between
        // intervals and the highlighted intervals themselves, toggling the
        // SGR run only at a real primary boundary.  Total emission cost is
        // O(intervals) slice copies of total length N, not N branchy probes.
        let mut in_primary = false;
        let mut cursor = 0usize;
        for &(ps, pe) in primary_ranges.iter() {
            let pe = pe.min(len);
            if pe <= cursor {
                continue; // zero-width or already passed
            }
            let ps = ps.max(cursor);
            if ps > cursor {
                // unhighlighted run before this interval
                if in_primary {
                    rendered.push_str(highlight_off);
                    in_primary = false;
                }
                emit(rendered, &line[cursor..ps]);
            }
            if ps < pe {
                // the highlighted interval itself
                if !in_primary {
                    rendered.push_str(highlight_on);
                    in_primary = true;
                }
                emit(rendered, &line[ps..pe]);
            }
            cursor = pe;
        }
        if cursor < len {
            // tail after the last interval
            if in_primary {
                rendered.push_str(highlight_off);
                in_primary = false;
            }
            emit(rendered, &line[cursor..]);
        }
        if in_primary {
            rendered.push_str(highlight_off);
        }

        fill_out(out, ranges);
        return len;
    }

    // ── ASCII + tabs: byte-level fast path with tab expansion ──
    // Source code is routinely tab-indented, and the pure-printable gate
    // above rejects tab (`0x09 < 0x20`), pushing every tab-indented row into
    // the grapheme walk.  Here the row stays byte-level: non-tab runs copy
    // through `emit` verbatim, each tab expands to its next stop, and the
    // annotation columns come from a byte→display-column map built from the
    // tab list (`disp(b) = b + extra_before(b)`, O(ntabs) per endpoint) —
    // O(len) with one compare per byte plus O(intervals) work, versus the
    // branch-heavy per-grapheme walk below.
    if ascii_class == crate::diagnostics::ascii::AsciiClass::PrintableOrTab {
        // Tab expansion is column-dependent: a tab's stop is computed from
        // the *absolute* cursor column (`content_start_col` + display col of
        // the tab), exactly as `grapheme_width_cached` does for the walk.
        // Guard the clamp here too so `% tab_width` cannot divide by zero.
        let tab_width = tab_width.max(1);
        // Per-row tab scan.  Its two scratch arrays live in the caller's
        // reused buffers instead of fresh per-row `SmallVec`s: `display_buf`
        // holds `(byte offset, stop width)` and `order_buf` the cumulative
        // extra width after each tab.  Neither is touched anywhere else on
        // this byte-level path (both belong to the grapheme walk below), so a
        // row with more than eight tabs no longer heap-allocates per row.
        let tabs: &mut Vec<(usize, usize)> = &mut *display_buf;
        let tab_extra: &mut Vec<usize> = &mut *order_buf;
        tabs.clear();
        tab_extra.clear();
        let mut extra = 0usize; // extra_before of the tab being scanned
        for i in memchr::memchr_iter(b'\t', line.as_bytes()) {
            let w = tab_width - (content_start_col + i + extra) % tab_width;
            extra += w - 1;
            tabs.push((i, w));
            tab_extra.push(extra); // extra *after* this tab
        }
        // Read-only views for the walks below.
        let tabs: &[(usize, usize)] = &tabs[..];
        let tab_extra: &[usize] = &tab_extra[..];
        // Display column of byte offset `b` (≤ len): byte index plus the
        // cumulative expansion of every tab strictly before `b`.
        let disp = |b: usize| -> usize {
            let k = tabs.partition_point(|&(t, _)| t < b);
            b + if k > 0 { tab_extra[k - 1] } else { 0 }
        };

        // Annotation records with display columns (byte ≢ display col when
        // tabs are present — the same fill the grapheme walk produces).
        out.reserve(ranges.len());
        for (&(start, end), &(ch, label_idx)) in ranges.iter().zip(meta.iter()) {
            let end_c = end.min(len);
            let (col_r, width) = if end > start && start < len {
                let s = disp(start);
                (s, disp(end_c) - s)
            } else {
                (disp(end_c), 0)
            };
            // Insertion points (`width == 0`) get a one-cell marker.
            out.push((col_r, width.max(1), ch, label_idx));
        }

        // Primary intervals: same O(intervals) walk as the pure-ASCII path,
        // but each run splits at its tabs.  A row with only secondary/note
        // annotations has no primary interval at all: emit it in one run and
        // skip the empty interval walk entirely (the common shape for
        // "previous definition here" rows in tab-indented source).
        if primary_ranges.is_empty() {
            emit_tab_run(rendered, line, 0, len, tabs, &mut emit);
            return disp(len);
        }
        let mut in_primary = false;
        let mut cursor = 0usize;
        for &(ps, pe) in primary_ranges.iter() {
            let pe = pe.min(len);
            if pe <= cursor {
                continue; // zero-width or already passed
            }
            let ps = ps.max(cursor);
            if ps > cursor {
                if in_primary {
                    rendered.push_str(highlight_off);
                    in_primary = false;
                }
                emit_tab_run(rendered, line, cursor, ps, tabs, &mut emit);
            }
            if ps < pe {
                if !in_primary {
                    rendered.push_str(highlight_on);
                    in_primary = true;
                }
                emit_tab_run(rendered, line, ps, pe, tabs, &mut emit);
            }
            cursor = pe;
        }
        if cursor < len {
            if in_primary {
                rendered.push_str(highlight_off);
                in_primary = false;
            }
            emit_tab_run(rendered, line, cursor, len, tabs, &mut emit);
        }
        if in_primary {
            rendered.push_str(highlight_off);
        }
        return disp(len);
    }

    // ── Non-ASCII, annotation-free fast path ──
    // A line with no annotations at all (a CJK context row, or a non-ASCII
    // source row with no label here) needs only sanitize + emit + width per
    // grapheme; the `single`/`small_k`/heap branch selection and the
    // per-grapheme expiry/activation checks below are pure no-ops on this
    // shape.  The emission arm is byte-identical to the full walk's arm (4),
    // so the two cannot diverge.
    if ranges.is_empty() && primary_ranges.is_empty() {
        let needs_sanitize = line_needs_sanitize(line);
        let mut col = 0usize;
        if !needs_sanitize {
            // Mixed ASCII/CJK fast path: the grapheme iterator still defines
            // every cluster boundary (so a combining mark, ZWJ emoji or keycap
            // sequence is never split), but consecutive printable single-byte
            // ASCII graphemes — whose width is exactly one cell — are batched
            // into one `emit` call instead of one call + one width-cache probe
            // per grapheme.  This is the common i18n shape: CJK prose with
            // embedded ASCII identifiers/punctuation.
            let mut byte_idx = 0usize;
            let mut run_start = 0usize;
            let mut run_end = 0usize;
            WIDTH_CACHE.with(|cell| {
                let mut cache = cell.borrow_mut();
                for g in line.graphemes(true) {
                    let glen = g.len();
                    let plain_ascii = glen == 1 && (0x20..0x7f).contains(&g.as_bytes()[0]);
                    if plain_ascii {
                        if run_end != byte_idx {
                            if run_end > run_start {
                                emit(rendered, &line[run_start..run_end]);
                            }
                            run_start = byte_idx;
                        }
                        run_end = byte_idx + 1;
                        col += 1;
                    } else {
                        if run_end > run_start {
                            emit(rendered, &line[run_start..run_end]);
                        }
                        run_start = byte_idx + glen;
                        run_end = run_start;
                        let w = grapheme_width_cached(
                            g,
                            content_start_col + col,
                            tab_width,
                            &mut cache,
                        );
                        if g == "\t" {
                            push_spaces(rendered, w);
                        } else {
                            emit(rendered, g);
                        }
                        col += w;
                    }
                    byte_idx += glen;
                }
            });
            if run_end > run_start {
                emit(rendered, &line[run_start..run_end]);
            }
            return col;
        }
        WIDTH_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            for g in line.graphemes(true) {
                let w = grapheme_width_cached(g, content_start_col + col, tab_width, &mut cache);
                if g == "\t" {
                    push_spaces(rendered, w);
                } else if !needs_sanitize {
                    emit(rendered, g);
                } else if g.chars().any(|c| c.is_control()) {
                    rendered.push('\u{FFFD}');
                } else if g.chars().all(super::is_bidi_override) {
                    // zero-width: no `col` advance, matching the full walk
                } else {
                    emit(rendered, g);
                }
                col += w;
            }
        });
        return col;
    }

    // ── Non-ASCII: single grapheme walk doing all three jobs ──
    // Hoisted before the scratch setup: for the dominant single-range row the
    // `clear` + `extend` + `sort_by_key` below are pure overhead (sorting
    // `[0]` is a function call plus a closure invocation on a path whose
    // other work is a handful of compares).
    let single = ranges.len() == 1;
    // k ≤ 3 stays off the `BinaryHeap`: a sorted-array scan over ≤ 3 active
    // ranges is a few compares per grapheme vs the heap's push/pop sifts.
    // k = 4 is the inflection point — the linear expiry scan's k(k−1)/2
    // comparisons start matching the heap there — so k ≥ 4 stays on it.
    let small_k = ranges.len() <= 3;
    if !single {
        order_buf.clear();
        order_buf.extend(0..ranges.len());
        if !order_buf.is_sorted_by_key(|&i| (ranges[i].0, ranges[i].1)) {
            order_buf.sort_by_key(|&i| (ranges[i].0, ranges[i].1));
        }
    }

    display_buf.resize(ranges.len(), (0, 0));
    done_buf.clear();
    done_buf.resize(ranges.len(), false);
    active_buf.clear();

    // Single-range fast path: when exactly one annotation covers this row
    // (the common case — primary only, no labels), track its activation in
    // two scalars instead of pushing/popping a `Reverse<(usize, usize)>`
    // through `active_buf`.  The heap's sift is more instructions than a
    // direct compare at k == 1; k == 0 (no annotation) and k >= 2 stay on
    // the heap path below.
    let mut single_active = false;
    let mut single_end = 0usize;
    // k ∈ {2, 3} fast path: active ranges live in a fixed array of
    // `(end_byte, range_idx)`, kept sorted by end.  Expiry is one forward
    // compaction scan, activation one sorted insert — both O(k) ≤ 3
    // compares, and both produce exactly the writes the heap path makes
    // (close order is immaterial: every close records the same `col`).
    let mut active_arr = [(0usize, 0usize); 3];
    let mut n_active = 0usize;

    let mut col = 0usize;
    let mut byte_idx = 0usize;
    let mut next = 0usize;
    let mut in_primary = false;
    let mut primary_cursor = 0usize;

    // Hoisted sanitization probe: if the line has no non-tab control and no
    // bidi-override character, every non-tab grapheme goes straight to `emit`
    // and we skip the per-grapheme `chars().any`/`chars().all` checks.  The
    // dominant i18n case (pure CJK / combining-mark-heavy lines) is uniformly
    // free of both, so this halves the per-grapheme branch traffic there.
    //
    // Fully byte-level and *exact* (see `line_needs_sanitize`): the old
    // two-stage filter (coarse byte precheck → per-char rescan) forced an
    // O(chars) char-decode walk on any line containing `°`, `§`, an emoji, or
    // any other math/symbol character — `0xC2` leads the whole Latin-1
    // supplement and `0xE2` leads every U+2xxx/emoji block, not just C1
    // controls and bidi overrides.  This predicate's true-positive set
    // (C0 \ {tab}, DEL, C1, bidi) is identical, but it answers with SWAR +
    // `memchr` byte scans only — no char decode, no false positives.
    let needs_sanitize = line_needs_sanitize(line);

    WIDTH_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        for g in line.graphemes(true) {
            let glen = g.len();
            let gend = byte_idx + glen;

            // (1) Close active ranges ending at or before this grapheme's start.
            // The heap is keyed by range end, so the earliest-ending active range
            // is always on top: expiring is a peek + pops, not a linear scan.
            // The single-range fast path replaces the heap peek/pop with one
            // scalar compare — the heap's `Reverse` sift is more instructions
            // than a direct test at k == 1.
            if single {
                if single_active && single_end <= byte_idx {
                    display_buf[0].1 = col;
                    done_buf[0] = true;
                    single_active = false;
                }
            } else if small_k {
                // Array expiry: one compaction scan over ≤ 3 entries.
                let mut w = 0usize;
                for r in 0..n_active {
                    let (end, i) = active_arr[r];
                    if end <= byte_idx {
                        display_buf[i].1 = col;
                        done_buf[i] = true;
                    } else {
                        active_arr[w] = (end, i);
                        w += 1;
                    }
                }
                n_active = w;
            } else {
                while let Some(&Reverse((end, i))) = active_buf.peek() {
                    if end > byte_idx {
                        break;
                    }
                    active_buf.pop();
                    display_buf[i].1 = col;
                    done_buf[i] = true;
                }
            }

            // (2) Activate ranges starting before this grapheme's end.
            // `next == 0` is the single-range analogue of the heap path's
            // `next < order_buf.len()` cursor guard: once the sole range has
            // been considered (activated or marked done), `next` is 1 and it
            // is never revisited, even after the range closes and
            // `single_active` returns to false.
            if single {
                if next == 0 && ranges[0].0 < gend {
                    next = 1;
                    if ranges[0].1 <= byte_idx {
                        display_buf[0] = (col, col);
                        done_buf[0] = true;
                    } else {
                        display_buf[0].0 = col;
                        single_end = ranges[0].1;
                        single_active = true;
                    }
                }
            } else if small_k {
                while next < order_buf.len() && ranges[order_buf[next]].0 < gend {
                    let i = order_buf[next];
                    next += 1;
                    if ranges[i].1 <= byte_idx {
                        display_buf[i] = (col, col);
                        done_buf[i] = true;
                    } else {
                        display_buf[i].0 = col;
                        // Sorted insert by end byte keeps expiry order
                        // identical to the heap's (both are total orders on
                        // the end key); with ≤ 3 entries the shift is a
                        // couple of moves.
                        debug_assert!(
                            n_active < active_arr.len(),
                            "small_k active array overfull: ranges.len() must stay ≤ {}",
                            active_arr.len(),
                        );
                        let mut p = n_active;
                        while p > 0 && active_arr[p - 1].0 > ranges[i].1 {
                            active_arr[p] = active_arr[p - 1];
                            p -= 1;
                        }
                        active_arr[p] = (ranges[i].1, i);
                        n_active += 1;
                    }
                }
            } else {
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
            }

            // (3) Primary highlight — byte overlap via the merged cursor: skip
            // intervals that ended at or before this grapheme, then test the
            // first one that can still overlap.  A row with no primary span
            // (only `~`/`-` annotations) skips the whole cursor probe.
            if !primary_ranges.is_empty() {
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
            }

            // (4) Append the sanitized grapheme.
            // Width is measured once per grapheme and reused by both the
            // emit arm and the column advance — a tab previously paid a
            // second `grapheme_width_cached` call for the identical answer.
            // Tab stops advance from the absolute cursor column: only tabs
            // read the current column (every other grapheme's width is
            // fixed), so adding the gutter offset here is exact and leaves
            // the recorded annotation columns content-relative.
            let w = grapheme_width_cached(g, content_start_col + col, tab_width, &mut cache);
            if g == "\t" {
                push_spaces(rendered, w);
            } else if !needs_sanitize {
                // Fast arm: the hoisted scan proved no control (other than tab)
                // and no bidi-override char exists in the line, so this grapheme
                // needs neither FFFD replacement nor bidi removal.
                emit(rendered, g);
            } else if g.chars().any(|c| c.is_control()) {
                // Source rows are untrusted text written between SGR runs: a
                // control grapheme becomes U+FFFD (tab was expanded above) and
                // the highlight escapes below are added by this function only.
                rendered.push('\u{FFFD}');
            } else if g.chars().all(super::is_bidi_override) {
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

            col += w;
            byte_idx = gend;
        }
    });
    if in_primary {
        rendered.push_str(highlight_off);
    }

    // Close remaining active ranges.  The single-range fast path tracks
    // activation in `single_active` instead of the heap, so a range still
    // active at end-of-line must be closed here the same way the heap loop
    // closes its entries — otherwise the "never activated" fallback below
    // would reset it to `(col, col)` and drop its width.
    if single {
        if single_active {
            display_buf[0].1 = col;
            done_buf[0] = true;
        }
    } else if small_k {
        for r in 0..n_active {
            let (_, i) = active_arr[r];
            display_buf[i].1 = col;
            done_buf[i] = true;
        }
    } else {
        for &Reverse((_, i)) in active_buf.iter() {
            display_buf[i].1 = col;
            done_buf[i] = true;
        }
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
    for (&(col, width), &(ch, label_idx)) in display_buf.iter().zip(meta.iter()) {
        // Insertion points (`width == 0`) get a one-cell marker.
        out.push((col, width.max(1), ch, label_idx));
    }

    col
}

/// Terminal-facing wrapper over [`render_line_merged_with`]: source text is
/// appended verbatim (the terminal escape codes a row needs are the highlight
/// runs the core writes itself, nothing else).
#[allow(clippy::too_many_arguments)]
fn render_line_merged(
    line: &str,
    ranges: &[(usize, usize)],
    meta: &[(char, Option<usize>)],
    out: &mut SmallVec<[(usize, usize, char, Option<usize>); 8]>,
    rendered: &mut String,
    highlight_on: &str,
    highlight_off: &str,
    display_buf: &mut Vec<(usize, usize)>,
    order_buf: &mut Vec<usize>,
    done_buf: &mut Vec<bool>,
    active_buf: &mut BinaryHeap<Reverse<(usize, usize)>>,
    tab_width: usize,
    content_start_col: usize,
    append: bool,
) -> usize {
    render_line_merged_with(
        LineRenderArgs {
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
            content_start_col,
            append,
        },
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
    line_starts: &[u32],
    out: &mut SmallVec<[(usize, usize, char, Option<usize>); 8]>,
    rendered: &mut String,
    highlight_on: &str,
    highlight_off: &str,
    ranges_buf: &mut Vec<(usize, usize)>,
    meta_buf: &mut SmallVec<[(char, Option<usize>); 8]>,
    display_buf: &mut Vec<(usize, usize)>,
    order_buf: &mut Vec<usize>,
    done_buf: &mut Vec<bool>,
    active_buf: &mut BinaryHeap<Reverse<(usize, usize)>>,
    tab_width: usize,
    content_start_col: usize,
    append: bool,
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
        true,
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
        content_start_col,
        append,
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

/// Precompute the byte offset of the start of every line in one O(n) pass.
///
/// Callers that render several lines should build this once and index it,
/// instead of re-scanning from the start of the file for each line (which
/// made source rendering O(lines × file_size)).
///
/// Thin alias for [`LineIndex::new`], which owns the scan; kept because
/// the line-start shape is the only part this helper's callers need.
pub(crate) fn line_starts(source: &str) -> Vec<u32> {
    LineIndex::new(source).starts().to_vec()
}

/// Fill `primary_cols` (resized to `display_width`) with `true` for every
/// display column covered by a `^` underline.  Used only by the tests as the
/// oracle for the fused renderer's byte-overlap highlight rule.
#[cfg(test)]
pub(crate) fn compute_primary_cols(
    underlines: &[(usize, usize, char, Option<usize>)],
    display_width: usize,
    primary_cols: &mut Vec<bool>,
) {
    primary_cols.clear();
    primary_cols.resize(display_width, false);
    for (col, ulen, underline_char, _label_idx) in underlines {
        if *underline_char == '^' {
            let end = (*col + *ulen).min(display_width);
            for cell in primary_cols.iter_mut().take(end).skip(*col) {
                *cell = true;
            }
        }
    }
}

/// One state of [`exact_component_assign`]'s sweep.
struct SweepState {
    /// Lane of every currently active interval, sorted by interval index — the
    /// part of the assignment that constrains future colour choices.
    key: SmallVec<[(usize, usize); 4]>,
    /// Minimum `Σ w·lane` accumulated by any assignment that reaches this key.
    cost: usize,
    /// Full lane-per-interval assignment that attained `cost`, returned to the
    /// caller so the row's lanes can be rebuilt.
    assign: SmallVec<[usize; 8]>,
}

/// Merge `(key, cost, assign)` into `states`, keeping the cheapest assignment per
/// key (ties keep the incumbent, so the sweep stays deterministic).
fn merge_sweep_state(
    states: &mut SmallVec<[SweepState; 8]>,
    key: SmallVec<[(usize, usize); 4]>,
    cost: usize,
    assign: SmallVec<[usize; 8]>,
) {
    if let Some(slot) = states.iter_mut().find(|s| s.key == key) {
        if cost < slot.cost {
            slot.cost = cost;
            slot.assign = assign;
        }
        return;
    }
    states.push(SweepState { key, cost, assign });
}

/// Exact IGMRC optimum for one connected component with `χ ≤ 3`, via the sweep
/// DP of `research/underline-lanes/no-distributive-lattice.md` §9.
///
/// `ivs` holds the component's `(start, end, weight)`; `lanes` the lane indices it
/// occupies (its greedy `χ`).  Returns the lane of every interval under a
/// minimum-`Σ w·lane` **proper** colouring, or `None` when the tier does not
/// apply (`χ > 3`, a degenerate interval, or no colouring found).
///
/// Why the state is sufficient: the active set at any coordinate is a clique
/// (Helly for intervals), so the lanes already taken by active intervals are
/// exactly the colours a starting interval may not reuse.  Merging equal active
/// assignments keeps the state count at `≤ χ! = 6`, giving `O(n · χ! · χ)`.
///
/// Ends are processed before starts at the same coordinate because lanes are
/// half-open `[start, end)`: annotations that merely touch may share a lane.
fn exact_component_assign(
    ivs: &[(usize, usize, usize)],
    lanes: &[usize],
) -> Option<SmallVec<[usize; 8]>> {
    const UNASSIGNED: usize = usize::MAX;
    if lanes.len() > 3 || lanes.is_empty() || ivs.is_empty() {
        return None;
    }
    if ivs.iter().any(|&(s, e, _)| s >= e) {
        // Zero-width annotations are promoted to one-cell markers upstream; a
        // degenerate span here means the caller's geometry is not what this tier
        // assumes, so let the relabelling path handle it.
        return None;
    }
    // (coordinate, kind, index) with kind 0 = end, 1 = start.
    let mut events: SmallVec<[(usize, u8, usize); 16]> = SmallVec::new();
    for (i, &(start, end, _)) in ivs.iter().enumerate() {
        events.push((end, 0, i));
        events.push((start, 1, i));
    }
    events.sort_unstable_by_key(|&(coord, kind, i)| (coord, kind, i));

    let mut assign: SmallVec<[usize; 8]> = SmallVec::new();
    assign.resize(ivs.len(), UNASSIGNED);
    let mut states: SmallVec<[SweepState; 8]> = SmallVec::new();
    states.push(SweepState {
        key: SmallVec::new(),
        cost: 0,
        assign,
    });

    for &(_, kind, i) in events.iter() {
        let mut next: SmallVec<[SweepState; 8]> = SmallVec::new();
        for state in states.iter() {
            if kind == 1 {
                let w = ivs[i].2;
                for &lane in lanes {
                    if state.key.iter().any(|&(_, taken)| taken == lane) {
                        continue;
                    }
                    let mut key = state.key.clone();
                    key.push((i, lane));
                    key.sort_unstable_by_key(|&(idx, _)| idx);
                    let mut next_assign = state.assign.clone();
                    next_assign[i] = lane;
                    merge_sweep_state(&mut next, key, state.cost + w * lane, next_assign);
                }
            } else {
                let mut key = state.key.clone();
                key.retain(|entry| entry.0 != i);
                merge_sweep_state(&mut next, key, state.cost, state.assign.clone());
            }
        }
        states = next;
    }
    let best = states.into_iter().min_by_key(|s| s.cost)?;
    if best.assign.iter().any(|&lane| lane == UNASSIGNED) {
        return None;
    }
    Some(best.assign)
}

/// Advance `p` to the next lexicographic permutation, returning `false` when
/// it already is the last one (descending).  Drives the exhaustive
/// per-component relabel search in [`UnderlineLanes::assign`], where `p` is at
/// most five entries long (120 candidates).
fn next_permutation(p: &mut [usize]) -> bool {
    if p.len() < 2 {
        return false;
    }
    // Rightmost position whose successor is larger: the suffix after it is
    // descending (the last arrangement of that tail).
    let mut i = p.len() - 1;
    while i > 0 && p[i - 1] >= p[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let pivot = i - 1;
    // Smallest entry in the descending tail that still exceeds the pivot.
    let mut j = p.len() - 1;
    while p[j] <= p[pivot] {
        j -= 1;
    }
    p.swap(pivot, j);
    // The tail was descending, so reversing it yields the next arrangement.
    p[i..].reverse();
    true
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
    lanes: Vec<SmallVec<[usize; 4]>>,
    /// Min-heap of `(end_col, lane)` driving the greedy sweep.
    active: BinaryHeap<Reverse<(usize, usize)>>,
    /// Linear active set for rows with ≤4 underlines; a heap's sift cost
    /// dominates at that size.
    small_active: SmallVec<[(usize, usize); 4]>,
    /// Reused `(weight, position)` sort scratch for the component relabel.
    weights: SmallVec<[(usize, usize); 8]>,
    /// Start-sorted index scratch.
    order: Vec<usize>,
    /// `(start, end, underline index, lane)` for every entry on the used
    /// lanes, sorted by `(start, end, index)`: the relabel's component
    /// decomposition sweeps exactly this list.
    entries: SmallVec<[(usize, usize, usize, usize); 8]>,
    /// Target lane per `entries` position, written by the component relabel
    /// and consumed by the rebuild that follows it.
    target: SmallVec<[usize; 8]>,
    /// `(lane, component weight)` for the component currently being relabelled.
    comp_weights: SmallVec<[(usize, usize); 8]>,
    /// Ascending lane indices the current component occupies — the set of
    /// indices any permutation of that component must fill.
    comp_lanes: SmallVec<[usize; 8]>,
    /// Permutation scratch: `perm[p]` is the rank in `comp_lanes` given to
    /// `comp_weights[p]`.
    perm: SmallVec<[usize; 8]>,
    /// Lanes used by the last `assign`.
    count: usize,
}

impl UnderlineLanes {
    pub(crate) fn new() -> Self {
        Self {
            lanes: Vec::new(),
            active: BinaryHeap::new(),
            small_active: SmallVec::new(),
            weights: SmallVec::new(),
            order: Vec::new(),
            entries: SmallVec::new(),
            target: SmallVec::new(),
            comp_weights: SmallVec::new(),
            comp_lanes: SmallVec::new(),
            perm: SmallVec::new(),
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
    pub(crate) fn assign(&mut self, underlines: &[(usize, usize, char, Option<usize>)]) -> usize {
        let Self {
            lanes,
            active,
            small_active,
            weights,
            order,
            entries,
            target,
            comp_weights,
            comp_lanes,
            perm,
            count,
        } = self;
        active.clear();
        small_active.clear();
        weights.clear();
        entries.clear();
        target.clear();
        comp_weights.clear();
        comp_lanes.clear();
        perm.clear();
        for lane in lanes.iter_mut() {
            lane.clear();
        }
        if underlines.is_empty() {
            *count = 0;
            return 0;
        }
        // Overwhelmingly common row: a single underline (one caret, no
        // labels).  The interval-graph machinery below would produce exactly
        // lane 0 = [0] and skip the relabel (n == 1), so short-circuit it.
        if underlines.len() == 1 {
            if lanes.is_empty() {
                lanes.push(SmallVec::new());
            }
            lanes[0].push(0);
            *count = 1;
            return 1;
        }
        order.clear();
        order.extend(0..underlines.len());
        // Unstable: the dedup below keys on `(start, ulen)` contiguity (which
        // any sort preserves for equal keys) and a `chars_seen` membership
        // set, so the relative order of equal-key entries does not matter.
        // `sort_unstable_by_key` avoids the merge buffer a stable sort pays.
        //
        // The common row — the primary span first, then labels in source
        // order — is already `(start, end)`-ordered, so an O(n) sortedness
        // probe skips the sort entirely.
        if !order.is_sorted_by_key(|&i| {
            let (col, ulen, ..) = underlines[i];
            (col, col + ulen)
        }) {
            order.sort_unstable_by_key(|&i| {
                let (col, ulen, ..) = underlines[i];
                (col, col + ulen)
            });
        }

        let mut n = 0usize;
        let small = underlines.len() <= 4;
        // Collapse duplicate annotations to a single lane entry: a later
        // `(start, ulen, ch)` twin would only stack an identical underline
        // row.  `order` is sorted by `(start, end)`, and `(start, end)`
        // fixes `ulen`, so annotations sharing `(start, ulen)` are
        // contiguous. Dedup only needs the chars seen within
        // the current span group, not across the whole row — a growing
        // seen-set would make label-heavy lines quadratic. The entry stays
        // in `underlines`, so its message (if any) is still emitted by the
        // caller's separate message loop.
        let mut last_span: Option<(usize, usize)> = None;
        // `underline_char()` yields exactly four characters, so the per-span
        // seen-set is a u8 bitmask instead of a SmallVec membership scan.
        let mut chars_seen: u8 = 0;
        for &i in order.iter() {
            let (col, ulen, ch, ..) = underlines[i];
            let start = col;
            let end = col + ulen;
            let span = (start, ulen);
            if last_span != Some(span) {
                last_span = Some(span);
                chars_seen = 0;
            }
            // The dedup bitmask relies on `underline_char()` yielding exactly
            // these four.  A fifth character would silently share bit 3 with
            // `' '` (help) and dedup wrongly; assert the cross-module
            // invariant instead of trusting it.
            debug_assert!(
                matches!(ch, '^' | '~' | '-' | ' '),
                "unexpected underline char {ch:?}"
            );
            let bit = 1u8
                << match ch {
                    '^' => 0,
                    '~' => 1,
                    '-' => 2,
                    _ => 3,
                };
            if chars_seen & bit != 0 {
                continue;
            }
            chars_seen |= bit;
            let lane = if small {
                let earliest = small_active
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, pair)| (pair.0, pair.1))
                    .map(|(pos, pair)| (pos, pair.0, pair.1));
                match earliest {
                    Some((pos, end, lane)) if end <= start => {
                        small_active.swap_remove(pos);
                        lane
                    }
                    _ => {
                        let idx = n;
                        n += 1;
                        idx
                    }
                }
            } else {
                match active.peek() {
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
                }
            };
            if lane >= lanes.len() {
                lanes.push(SmallVec::new());
            }
            lanes[lane].push(i);
            if small {
                small_active.push((end, lane));
            } else {
                active.push(Reverse((end, lane)));
            }
        }

        // ── Why only relabel, not search all optimal colourings ─────
        // Full-space WDS optimisation cannot ride a distributive-lattice
        // structure: the optimal-colouring set of an interval graph is
        // not a lattice in general, and for K₃ ⊔ K₃ admits no canonical
        // lattice order at all (witnesses plus the C = ⊔_D C_D
        // decomposition in
        // research/underline-lanes/no-distributive-lattice.md).
        // Exact IGMRC is NP-hard in general (exact-χ padding reduction)
        // and FPT only via the O(n · k! · k) sweep DP; it is polynomial
        // on the fully saturated subclass (every maximal clique of size
        // χ), and the clique LP is not value-exact, so no LP/flow engine
        // replaces the DP here. That machinery is not worth it for the
        // n ≤ ~50, χ ≤ ~5 rows. Below we only relabel the greedy's
        // single colouring, component by component — exact over that
        // colouring's own lane permutations, but still a heuristic over
        // colourings, not an exact optimum:
        // e.g. [(0,2),(0,2),(3,5)] with w = (1,2,1) keeps cost 6 against
        // an optimum of 5 (research/igmrc/README.md, §2 of the
        // underline-lanes note).
        // ── Relabel lanes per overlap component ──────────────────────
        // The greedy numbers lanes in opening order, which has nothing to do
        // with the visual weight each lane carries.  Under the
        // weighted-depth-sum objective — minimise Σ width(u) · lane(u) — a
        // heavier lane should get a smaller row index (closer to the source
        // line).
        //
        // The row's intervals decompose into the connected components of
        // their overlap graph, found by one sweep over the start-sorted
        // entries: a component runs on while the next start does not exceed
        // the running maximum end.  Distinct components are pairwise disjoint
        // in columns, so permuting the lane indices a single component
        // occupies keeps the colouring proper (each lane's new content is a
        // union of mutually disjoint pieces), and the objective is separable
        // over components — so each component can be relabelled on its own.
        // Every global lane permutation is representable per component, so
        // this strictly dominates the previous global sort: two disjoint
        // overlapping pairs with per-component weights (10, 1) and (1, 10)
        // share lanes 0/1, cost 11 under the greedy numbering and 11 under a
        // global sort (the global lane weights tie, so it changes nothing),
        // but 2 once each pair is sorted within its own component.  For a
        // component using two lanes there are exactly two proper
        // 2-colourings, so that swap is the exact optimum.
        if n > 1 {
            for (lane_idx, lane) in lanes.iter().enumerate().take(n) {
                for &i in lane.iter() {
                    let (col, ulen, ..) = underlines[i];
                    entries.push((col, col + ulen, i, lane_idx));
                }
            }
            entries.sort_unstable_by_key(|&(start, end, i, _)| (start, end, i));
            target.resize(entries.len(), 0);

            let mut comp_start = 0usize;
            while comp_start < entries.len() {
                // Ascending start order, so components are relabelled in a
                // deterministic order.
                let mut comp_end = comp_start + 1;
                let mut max_end = entries[comp_start].1;
                while comp_end < entries.len() && entries[comp_end].0 <= max_end {
                    max_end = max_end.max(entries[comp_end].1);
                    comp_end += 1;
                }
                // The weight this component contributes to each lane it uses.
                comp_weights.clear();
                for &(_, _, i, lane_idx) in &entries[comp_start..comp_end] {
                    let (_, ulen, _, label_idx) = underlines[i];
                    // Width dominates; a labelled underline also carries a
                    // rendered message, giving it slightly more visual
                    // weight than a bare synthetic one.
                    let w = ulen + usize::from(label_idx.is_some());
                    match comp_weights.iter_mut().find(|(used, _)| *used == lane_idx) {
                        Some(entry) => entry.1 += w,
                        None => comp_weights.push((lane_idx, w)),
                    }
                }
                let k = comp_weights.len();
                // Ascending lane order: with this alignment the identity
                // permutation reproduces the greedy's current numbering, so
                // "keep the current labelling on a tie" is a no-op.
                comp_weights.sort_unstable_by_key(|&(lane_idx, _)| lane_idx);
                // The lane indices the component must fill: any relabel is a
                // bijection of this set onto itself.
                comp_lanes.clear();
                comp_lanes.extend(comp_weights.iter().map(|&(lane_idx, _)| lane_idx));
                comp_lanes.sort_unstable();
                // Exact tier for χ ≤ 3.  The permutation search below only
                // renumbers the greedy colouring, while a component can have a
                // *different* proper colouring with strictly lower WDS.  The
                // sweep DP explores all of them, so for the shape that dominates
                // real rows (caret + one or two secondary lanes) this is the exact
                // IGMRC optimum rather than the best relabelling.
                if comp_lanes.len() <= 3 {
                    let mut ivs: SmallVec<[(usize, usize, usize); 8]> = SmallVec::new();
                    for pos in comp_start..comp_end {
                        let (start, end, i, _) = entries[pos];
                        let (_, ulen, _, label_idx) = underlines[i];
                        ivs.push((start, end, ulen + usize::from(label_idx.is_some())));
                    }
                    if let Some(assigned) = exact_component_assign(&ivs, &comp_lanes) {
                        for (pos, &lane) in (comp_start..comp_end).zip(assigned.iter()) {
                            target[pos] = lane;
                        }
                        comp_start = comp_end;
                        continue;
                    }
                }
                // Identity = the greedy's own numbering, which is also what a
                // tie keeps.
                let mut best_perm = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
                let mut best: usize = (0..k).map(|p| comp_weights[p].1 * comp_lanes[p]).sum();
                if k > 1 && k <= best_perm.len() {
                    // Both branches write through `perm`; seed it once.
                    perm.clear();
                    perm.extend(0..k);
                    if k <= 5 {
                        // Exhaustive: k! ≤ 120 at k = 5, still trivial per
                        // row, and it needs no proof beyond "try them all".
                        loop {
                            let cost: usize = (0..k)
                                .map(|p| comp_weights[p].1 * comp_lanes[perm[p]])
                                .sum();
                            if cost < best {
                                best = cost;
                                best_perm[..k].copy_from_slice(&perm);
                            }
                            if !next_permutation(perm) {
                                break;
                            }
                        }
                    } else {
                        // Rearrangement inequality: the heaviest lane wants
                        // the smallest index.  This is the old global
                        // permutation, restricted to the component.
                        perm.clear();
                        perm.extend(0..k);
                        weights.clear();
                        weights.extend((0..k).map(|p| (comp_weights[p].1, p)));
                        weights.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                        for (rank, &(_, p)) in weights.iter().enumerate() {
                            perm[p] = rank;
                        }
                        let cost: usize = (0..k)
                            .map(|p| comp_weights[p].1 * comp_lanes[perm[p]])
                            .sum();
                        if cost < best {
                            best_perm[..k].copy_from_slice(&perm);
                        }
                    }
                }
                // Map every entry of this component to its new lane.
                for pos in comp_start..comp_end {
                    let lane_idx = entries[pos].3;
                    let slot = comp_weights.iter().position(|&(used, _)| used == lane_idx);
                    debug_assert!(
                        slot.is_some(),
                        "lane {lane_idx} missing from its own component"
                    );
                    // `best_perm[slot]` is a position in `comp_lanes`, whose
                    // entries are the lane indices this component occupies —
                    // NOT a lane index itself.  Treating it as one remapped a
                    // component on lanes {1,2} onto {0,1}, colliding with
                    // whatever already lived there (two overlapping carets on one
                    // lane).
                    let new_lane = slot.map_or(lane_idx, |slot| {
                        best_perm
                            .get(slot)
                            .and_then(|&p| comp_lanes.get(p).copied())
                            .unwrap_or(lane_idx)
                    });
                    target[pos] = new_lane;
                }
                comp_start = comp_end;
            }

            // Rebuild the used lanes from the computed targets.  A component
            // only permutes the lane indices it occupies, so the used set is
            // unchanged and every lane in `0..n` stays non-empty; clearing
            // then refilling preserves each lane's capacity.  Entries are
            // visited in start order, so every lane stays column-sorted.
            for lane in lanes.iter_mut().take(n) {
                lane.clear();
            }
            for (pos, &(_, _, i, _)) in entries.iter().enumerate() {
                lanes[target[pos]].push(i);
            }
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
    /// Padding cells inherit the colour of the current annotation, so a
    /// lane's runs read as one contiguous row rather than alternating
    /// colour gaps.
    pub(crate) fn row<'a, F: FnMut(char) -> &'a str>(
        &mut self,
        underlines: &[(usize, usize, char, Option<usize>)],
        index: usize,
        display_width: usize,
        out: &mut String,
        color_for: F,
        reset: &'a str,
    ) -> bool {
        // Callers that own only a scratch row buffer (the HTML renderer) get
        // the historical clear-then-write contract; the glyph renderer appends
        // straight to its output via `row_append`.
        out.clear();
        self.row_append(underlines, index, display_width, out, color_for, reset)
    }

    /// Append lane `index`'s underline cells to `out` without clearing it.
    ///
    /// Returns `false` when every entry was clipped away; the caller must then
    /// truncate any gutter it already emitted back to its saved mark.
    pub(crate) fn row_append<'a, F: FnMut(char) -> &'a str>(
        &mut self,
        underlines: &[(usize, usize, char, Option<usize>)],
        index: usize,
        display_width: usize,
        out: &mut String,
        mut color_for: F,
        reset: &'a str,
    ) -> bool {
        let lane = &self.lanes[index];
        // Append straight into the caller's buffer (no per-row scratch
        // copy). The caller saves a mark, emits the gutter, and truncates
        // back to it when this returns `false`.
        // First column not yet emitted.
        let mut cursor = 0usize;
        // `' '` is a real underline character (help-level annotations), so
        // blankness must be tracked separately from the buffer contents.
        let mut visible = false;
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
            out.push_str(color);
            push_spaces(out, start - cursor);
            out.extend(std::iter::repeat_n(ch, end - start));
            out.push_str(reset);
            if ch != ' ' {
                visible = true;
            }
            cursor = end;
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
        // Defensive clamp: `GlyphRenderer::with_opts` and `wrap_optimal`
        // clamp, but this helper is `pub(crate)` and called directly by
        // `expand_tabs_from_col` / test oracles with a caller-supplied width.
        // A modulo by zero would panic; a zero-width tab column is not a
        // meaningful concept anyway.
        let tab_width = tab_width.max(1);
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
    // 2-byte UTF-8 with lead below U+0300 (0xCC) is Latin-1 supplement /
    // IPA/extensions: no combining marks in this range, and C1 controls are
    // already rendered as a single U+FFFD cell.  U+0300+ is 2-byte too, so
    // gate on the lead byte, not the length alone.
    if g.len() == 2 && g.as_bytes()[0] < 0xCC {
        return 1;
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

/// [`grapheme_width`] with an explicit cache handle, so a loop over many
/// graphemes can hoist the `WIDTH_CACHE` TLS access + `RefCell` borrow above
/// the loop instead of paying it per grapheme.  The tab / single-byte fast
/// paths skip the cache, so ASCII clusters are still one range check.
#[inline]
pub(crate) fn grapheme_width_cached(
    g: &str,
    current_col: usize,
    tab_width: usize,
    cache: &mut WidthCache,
) -> usize {
    if g == "\t" {
        let tab_width = tab_width.max(1);
        return tab_width - (current_col % tab_width);
    }
    if g.len() <= 1 {
        return match g.as_bytes().first() {
            Some(_) => 1,
            None => 0,
        };
    }
    // 2-byte UTF-8 with lead below U+0300 (0xCC) is Latin-1 supplement /
    // IPA/extensions: no combining marks in this range, and C1 controls are
    // already rendered as a single U+FFFD cell.  U+0300+ is 2-byte too, so
    // gate on the lead byte, not the length alone.
    if g.len() == 2 && g.as_bytes()[0] < 0xCC {
        return 1;
    }
    if g.len() > WIDTH_CACHE_MAX_KEY {
        return grapheme_width_uncached(g);
    }
    if let Some(w) = cache.get(g) {
        return w;
    }
    let w = grapheme_width_uncached(g);
    cache.put(g, w);
    w
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
    // One fused pass replaces the three separate scans (`it.all(is_ri)`,
    // `contains(ZWJ)`, `chars().any(is_control)`): a single-char BMP cluster —
    // the dominant non-ASCII shape — exits after one iteration; a ZWJ family
    // emoji (up to 7 chars) after ≤7.
    //
    // Regional Indicator pairs: unicode-width sums each RI (2 each),
    // but terminals render a flag (🇺🇸) as 2 cells — correct here.
    let mut first_ri = false;
    let mut all_ri = true;
    let mut has_zwj = false;
    let mut has_ctl = false;
    for (i, c) in g.chars().enumerate() {
        let ri = ('\u{1F1E6}'..='\u{1F1FF}').contains(&c);
        if i == 0 {
            first_ri = ri;
        }
        all_ri &= ri;
        has_zwj |= c == '\u{200D}';
        has_ctl |= c.is_control();
    }
    if first_ri && all_ri {
        return 2;
    }
    // Non-RGI ZWJ joins (e.g. 👨‍👩): unicode-width already returns 2 for
    // well-formed RGI sequences; for malformed left-over spans cap at 2.
    if has_zwj {
        w = w.min(2);
    }
    // A cluster containing a control char prints as a single U+FFFD (one
    // cell); `unicode-width` credits controls with zero, so floor the width.
    if has_ctl {
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
pub(crate) struct WidthCache {
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
    pub(crate) static WIDTH_CACHE: RefCell<WidthCache> = const { RefCell::new(WidthCache::new()) };
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
    if crate::diagnostics::ascii::is_printable_ascii(line.as_bytes()) {
        return line.len();
    }
    let mut w = 0usize;
    WIDTH_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        for g in line.graphemes(true) {
            w += grapheme_width_cached(g, w, tab_width, &mut cache);
        }
    });
    w
}

/// Expand tabs in `s` to spaces at `tab_width`-column stops, where the text
/// begins at (zero-based) display column `start_col`.  The diff block's
/// `-`/`+` rows use this so a tab-indented source line aligns with the main
/// source rows, which expand tabs in the row printer (`render_line_merged`):
/// the tab width there depends on the *display* column, and the diff rows
/// start several gutter cells further right.
///
/// Non-tab graphemes are copied verbatim (the caller has already run
/// `sanitize_for_terminal_multiline`, so no other control can remain);
/// column arithmetic uses [`grapheme_width`], matching the row printer.
pub(crate) fn expand_tabs_from_col<'s>(
    s: &'s str,
    tab_width: usize,
    start_col: usize,
) -> Cow<'s, str> {
    if !s.contains('\t') {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    let mut col = start_col;
    for g in s.graphemes(true) {
        let w = grapheme_width(g, col, tab_width);
        if g == "\t" {
            push_spaces(&mut out, w);
        } else {
            out.push_str(g);
        }
        col += w;
    }
    Cow::Owned(out)
}

/// Visible width of each [`DiagnosticLevel`]'s localized `label()`, cached per
/// language.  Labels are Fluent strings, so the widths are language-dependent;
/// keying the table by [`Lang::index`] keeps a mid-process language switch from
/// serving a stale width.  `severity()` and `label()` use the same 0..6
/// variant-index mapping in `level.rs`, so the lookup is one array read
/// instead of re-walking the label's graphemes on every child/note/help line.
fn level_label_widths() -> &'static [usize; 6] {
    use crate::diagnostics::i18n::{Lang, N_LANGS};
    static WIDTHS: [std::sync::OnceLock<[usize; 6]>; N_LANGS] =
        [const { std::sync::OnceLock::new() }; N_LANGS];
    let lang = Lang::current().index();
    WIDTHS[lang].get_or_init(|| {
        [
            visible_width(DiagnosticLevel::Info.label()),
            visible_width(DiagnosticLevel::Note.label()),
            visible_width(DiagnosticLevel::Help.label()),
            visible_width(DiagnosticLevel::Warning.label()),
            visible_width(DiagnosticLevel::Error.label()),
            visible_width(DiagnosticLevel::Fatal.label()),
        ]
    })
}

/// Visible display width of `s`, skipping CSI (`\x1b[…m`) and OSC
/// (`\x1b]…\x1b\\`) escape sequences.  Reuses [`grapheme_width`] for the
/// per-cluster arithmetic (tabs, CJK, ZWJ) already used everywhere else.
pub(crate) fn visible_width(s: &str) -> usize {
    visible_width_with_tab(s, DEFAULT_TAB_WIDTH)
}

/// Byte length of the escape sequence starting at `bytes[0]` (which must be
/// `0x1b`), matching the skip semantics `visible_width_with_tab` had inline.
/// A lone ESC (not CSI/OSC) returns 1 so the caller measures it as a
/// one-cell grapheme, exactly as before.
fn escape_len(bytes: &[u8]) -> usize {
    if bytes.len() < 2 {
        return 1;
    }
    match bytes[1] {
        b'[' => {
            let mut j = 2;
            while j < bytes.len() && !(0x40..=0x7e).contains(&bytes[j]) {
                j += 1;
            }
            (j + 1).min(bytes.len())
        }
        b']' => {
            let mut j = 2;
            while j < bytes.len() && bytes[j] != 0x07 {
                if bytes[j] == 0x1b && j + 1 < bytes.len() && bytes[j + 1] == b'\\' {
                    return (j + 2).min(bytes.len());
                }
                j += 1;
            }
            (j + 1).min(bytes.len())
        }
        _ => 1,
    }
}

pub(crate) fn visible_width_with_tab(s: &str, tab_width: usize) -> usize {
    let bytes = s.as_bytes();
    // Fast path: every byte printable ASCII. ESC is 0x1b (a control char,
    // outside this range), so a string that reaches this branch contains no
    // CSI/OSC sequences to skip, no tabs to expand, and no multi-byte
    // clusters — one cell per byte, i.e. `s.len()`.
    if crate::diagnostics::ascii::is_printable_ascii(bytes) {
        return s.len();
    }
    let mut w = 0usize;
    let mut rest = s;
    // Jump from ESC to ESC instead of walking the escape's bytes as
    // individual graphemes. `escape_len` always returns a length that ends on
    // a char boundary (CSI final byte and OSC BEL/ST are all ASCII), so the
    // slice below cannot panic.
    WIDTH_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        while let Some(esc) = memchr::memchr(0x1b, rest.as_bytes()) {
            // Plain-text prefix before the escape: measure grapheme by grapheme.
            let (text, tail) = rest.split_at(esc);
            if crate::diagnostics::ascii::is_printable_ascii(text.as_bytes()) {
                w += text.len();
            } else {
                for g in text.graphemes(true) {
                    w += grapheme_width_cached(g, w, tab_width, &mut cache);
                }
            }
            let skip = escape_len(tail.as_bytes());
            if skip == 1 {
                // Lone ESC (no CSI/OSC introducer): measured as one U+FFFD cell,
                // matching the old per-grapheme path.
                w += grapheme_width_cached("\x1b", w, tab_width, &mut cache);
            }
            rest = &tail[skip..];
        }
        if crate::diagnostics::ascii::is_printable_ascii(rest.as_bytes()) {
            w += rest.len();
        } else {
            for g in rest.graphemes(true) {
                w += grapheme_width_cached(g, w, tab_width, &mut cache);
            }
        }
    });
    w
}

/// Convert a byte range `[start_offset, end_offset)` within `line`
/// into `(start_display_col, display_width)`.
///
/// Span boundaries falling inside a grapheme cluster are expanded to the
/// cluster boundary so an emoji (or combining sequence) is never half-
/// underlined.
///
/// Display column of `byte_offset` within `line` (production helper for
/// multi-line connector geometry).  Expands to the containing grapheme's
/// start, matching `byte_range_to_display`'s start column.
/// Precomputed byte→display-column table for one line.
///
/// The per-query helper used to re-walk the line's grapheme clusters from byte
/// zero on *every* call, so a diagnostic with `N` labels on one long line paid
/// `N` full cluster walks.  Building the table once and binary-searching it makes
/// each query `O(log clusters)` on a table that is already cache-hot, and lets a
/// caller consult several offsets of the same line for a single walk.
pub(crate) struct LineColumns {
    /// `(B_k, C_k)` for `k = 0..=n`: `B_k` is the byte end of the first `k`
    /// clusters (`B_0 = 0`, strictly increasing) and `C_k` the display column
    /// after them.
    bounds: Vec<(usize, usize)>,
}

impl LineColumns {
    pub(crate) fn new(line: &str, tab_width: usize) -> Self {
        // `grapheme_width` already clamps a zero tab width; normalise here too so
        // the table cannot disagree with that rule for a caller-supplied 0.
        let tab_width = tab_width.max(1);
        // A cluster is at least one byte, so `len + 1` is an exact upper bound —
        // no reallocation, and for pure ASCII this is the only allocation.
        let mut bounds = Vec::with_capacity(line.len() + 1);
        bounds.push((0, 0));
        let mut col = 0usize;
        let mut byte = 0usize;
        for g in line.graphemes(true) {
            col += grapheme_width(g, col, tab_width);
            byte += g.len();
            bounds.push((byte, col));
        }
        Self { bounds }
    }

    /// Display column of `byte_offset`, i.e. `C_{max{k : B_k <= byte_offset}}`.
    ///
    /// An offset inside a cluster resolves to that cluster's *start* column, and
    /// an offset past the end to the line's full width — exactly the grapheme
    /// expansion the previous loop implemented, now without the walk.
    pub(crate) fn display(&self, byte_offset: usize) -> usize {
        let k = self.bounds.partition_point(|&(b, _)| b <= byte_offset);
        // `bounds[0] == (0, 0)` and `0 <= byte_offset`, so `k >= 1`.
        self.bounds[k - 1].1
    }
}

/// One-entry memo over [`LineColumns`].
///
/// Span streams are sorted, so consecutive queries overwhelmingly hit the same
/// line; a single slot removes the repeated walk with no map and no per-line
/// allocation retained after the line changes.
#[derive(Default)]
pub(crate) struct LineColumnsCache(Option<(usize, LineColumns)>);

impl LineColumnsCache {
    pub(crate) fn display(
        &mut self,
        line_idx: usize,
        line: &str,
        tab_width: usize,
        byte_offset: usize,
    ) -> usize {
        if let Some((cached_idx, cols)) = &self.0
            && *cached_idx == line_idx
        {
            return cols.display(byte_offset);
        }
        let cols = LineColumns::new(line, tab_width);
        let col = cols.display(byte_offset);
        self.0 = Some((line_idx, cols));
        col
    }
}

/// Sparse display-column checkpoints for one line at one tab width.
///
/// The last O(prefix) per-span cost in the renderers: `byte_offset_to_display` walks the
/// line's graphemes from the start for **every** query, and the harness's
/// `perf_header_column_walk_on_long_line` measures exactly that on a long line.  A line
/// that is pure ASCII and tab-free needs no table at all (display column == byte offset);
/// otherwise this records `(byte_off, display_col)` at every tab and every non-ASCII
/// cluster, so a query is `partition_point` plus a walk bounded by the distance to the
/// nearest checkpoint.
///
/// Keyed by `tab_width` because tab expansion is: the table is per `(line, tab_width)`,
/// built lazily by the caller (`LineIndex::display_columns`).
#[derive(Debug)]
pub(crate) struct DisplayColumns {
    /// `(byte offset, display column before that cluster)`, ascending.
    checkpoints: Vec<(u32, u32)>,
    /// `true` when the line is pure ASCII and tab-free — the query is the byte offset.
    trivial: bool,
}

impl DisplayColumns {
    /// Build the checkpoints for one line, or record it as trivial.
    ///
    /// One grapheme walk — the same one `byte_offset_to_display` would do per query.
    pub(crate) fn new(line: &str, tab_width: usize) -> Self {
        let mut checkpoints: Vec<(u32, u32)> = Vec::new();
        let mut col = 0usize;
        let mut byte = 0usize;
        let mut trivial = line.is_ascii() && !line.as_bytes().contains(&b'\t');
        if !trivial {
            for g in line.graphemes(true) {
                // A tab or a non-ASCII cluster changes the byte↔column relationship from
                // this offset onward, so it must be a checkpoint.
                if g == "\t" || !g.is_ascii() {
                    // `u32` holds any realistic line; saturate rather than wrap so a
                    // pathological line degrades to "checkpoint at the end" (correct, just
                    // useless) instead of wrapping to a small column.
                    checkpoints.push((
                        byte.min(u32::MAX as usize) as u32,
                        col.min(u32::MAX as usize) as u32,
                    ));
                }
                col += grapheme_width(g, col, tab_width);
                byte += g.len();
            }
        }
        Self {
            checkpoints,
            trivial,
        }
    }

    /// Display column of `byte_offset` within the line this table was built for.
    pub(crate) fn col(&self, line: &str, byte_offset: usize, tab_width: usize) -> usize {
        if self.trivial {
            return byte_offset;
        }
        let at = self
            .checkpoints
            .partition_point(|&(b, _)| (b as usize) <= byte_offset);
        let (mut byte, mut col) = match at {
            0 => (0usize, 0usize),
            n => {
                let (b, c) = self.checkpoints[n - 1];
                (b as usize, c as usize)
            }
        };
        if byte >= line.len() {
            return col;
        }
        for g in line[byte..].graphemes(true) {
            if byte + g.len() > byte_offset {
                break;
            }
            col += grapheme_width(g, col, tab_width);
            byte += g.len();
        }
        col
    }
}

#[cfg(test)]
mod display_columns_tests {
    use super::{DisplayColumns, byte_offset_to_display, byte_range_to_display};

    /// The checkpoint table must answer *exactly* what the scalar walk answers.
    ///
    /// Corpus: pure ASCII (trivial path), tabs in every position, wide CJK, a combining
    /// mark (a cluster wider than one char but one column), and a mixed line.  Every byte
    /// boundary is checked at every tab width the renderer can be configured with, because
    /// the table is per `(line, tab_width)` and tab expansion is where a checkpoint scheme
    /// goes wrong.
    #[test]
    fn display_columns_matches_the_scalar_walk_at_every_boundary() {
        let corpus = [
            "",
            "plain ascii line",
            "\tindented",
            "a\tb\tc",
            "\t\t\t\t",
            "宽度",
            "a宽b",
            "\t宽\t字",
            "e\u{301}closed", // combining acute: one cluster, one column
            "e\u{301}\tx",
            "  \t  \t  mixed 宽 \t tail  ",
        ];
        for line in corpus {
            for tab_width in [1usize, 2, 4, 8] {
                let table = DisplayColumns::new(line, tab_width);
                for off in 0..=line.len() {
                    // The scalar answer is only defined at a cluster boundary; compare
                    // there, which is where the renderers query (`pos.col` comes from the
                    // line index and is a char boundary).
                    if !line.is_char_boundary(off) {
                        continue;
                    }
                    let want = byte_offset_to_display(line, off, tab_width);
                    assert_eq!(
                        table.col(line, off, tab_width),
                        want,
                        "line {line:?} tab_width {tab_width} offset {off}"
                    );
                }
                // The table must agree with the range oracle too, so a caller that mixes
                // the two cannot see different columns.
                let (start, _) = byte_range_to_display(line, 0, line.len(), tab_width);
                assert_eq!(start, 0, "line {line:?} tab_width {tab_width}");
            }
        }
    }
}

/// Single-shot variant of [`LineColumns::display`].
///
/// Deliberately allocation-free: for one offset a grapheme walk beats building a
/// bounds table, so this stays the helper for callers with a *single* query.
/// Callers that resolve several offsets on the same line should hold a
/// [`LineColumnsCache`] instead — that is the case the table exists for.
pub(crate) fn byte_offset_to_display(line: &str, byte_offset: usize, tab_width: usize) -> usize {
    let mut col = 0usize;
    let mut byte = 0usize;
    for g in line.graphemes(true) {
        // Stop at the cluster that *contains* `byte_offset`: its start column is
        // the answer (a boundary inside a cluster expands outwards).
        if byte + g.len() > byte_offset {
            break;
        }
        col += grapheme_width(g, col, tab_width);
        byte += g.len();
    }
    col
}

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
pub(crate) fn byte_ranges_to_display(
    line: &str,
    ranges: &[(usize, usize)],
    tab_width: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(ranges.len());
    let mut order = Vec::with_capacity(ranges.len());
    let mut done = Vec::with_capacity(ranges.len());
    let mut active: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    byte_ranges_to_display_into(
        line,
        ranges,
        &mut out,
        &mut order,
        &mut done,
        &mut active,
        tab_width,
    );
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
    if crate::diagnostics::ascii::is_printable_ascii(line.as_bytes()) {
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
    // No runtime `debug_assert_eq!(hit, KEYWORDS.contains(&word))` here: it
    // is a linear scan over ~40 keywords on every identifier in debug
    // builds, and table consistency is already pinned by the
    // `keyword_dispatch_matches_contains` test.
    hit
}

/// Posita built-in type names to highlight in bright blue.
const TYPES: &[&str] = &[
    "Int", "UInt", "Float", "Bool", "Char", "Byte", "USize", "Str", "Unit", "Never", "String",
];

/// Groups by first byte to match [`TYPES`]; the
/// `type_dispatch_matches_contains` test asserts they stay in sync.
#[inline]
fn is_type(word: &str) -> bool {
    let Some(&first) = word.as_bytes().first() else {
        return false;
    };
    let hit = match first {
        b'B' => matches!(word, "Bool" | "Byte"),
        b'C' => matches!(word, "Char"),
        b'F' => matches!(word, "Float"),
        b'I' => matches!(word, "Int"),
        b'N' => matches!(word, "Never"),
        b'S' => matches!(word, "Str" | "String"),
        b'U' => matches!(word, "UInt" | "USize" | "Unit"),
        _ => false,
    };
    // Same reasoning as `is_keyword`: `type_dispatch_matches_contains` pins
    // table consistency; a per-call linear scan buys nothing.
    hit
}

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

/// [`highlight_code`] writing directly into the caller's buffer — no
/// intermediate `String` per highlighted line.  Returns `true` when a
/// highlight was applied (the caller then uses `out`'s tail); `false` means
/// nothing was written and the caller should push `line` verbatim.
///
/// Precondition: `line` must already be sanitized for terminal output (see
/// [`crate::diagnostics::sanitize_for_terminal`]) — the highlighter copies
/// every non-token byte through verbatim, so an ESC or other C0/DEL/C1 byte
/// in the input would reach the terminal unsanitized.  Tabs and newlines are
/// allowed (the sanitizers keep them as layout bytes).
pub fn highlight_code_into(line: &str, use_color: bool, out: &mut String) -> bool {
    if !use_color {
        return false;
    }
    highlight_code_impl_into(line, true, false, out)
}

/// [`highlight_code_with_backticks`] writing directly into the caller's
/// buffer (see [`highlight_code_into`] for the contract).
pub(crate) fn highlight_code_with_backticks_into(
    line: &str,
    use_color: bool,
    out: &mut String,
) -> bool {
    if !use_color {
        return false;
    }
    highlight_code_impl_into(line, true, true, out)
}

fn highlight_code_impl<'a>(
    line: &'a str,
    use_color: bool,
    add_backtick_quotes: bool,
) -> Cow<'a, str> {
    if !use_color {
        return Cow::Borrowed(line);
    }
    let mut out = String::new();
    if highlight_code_impl_into(line, true, add_backtick_quotes, &mut out) {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(line)
    }
}

// ── ASCII classification table for the highlighter ─────────────
//
// The highlighter only branches on six ASCII byte shapes; every other
// byte — digits, whitespace, punctuation, brackets — is copied through
// verbatim.  `CLASS` maps each ASCII byte to the state-machine entry it
// triggers, so the scanner's hot loop is one load, one table lookup and
// one `match`: no `next_char` call and no `char` construction on ASCII
// input.  Non-ASCII bytes are still decoded whole through `next_char`.
//
// The table is built at compile time (`const fn`), so it costs one
// `[u8; 128]` in `.rodata` and zero init work at startup.

const PASS: u8 = 0;
const IDENT: u8 = 1;
const STRING: u8 = 2;
const CHAR: u8 = 3;
const OPER: u8 = 4;
const TICK: u8 = 5;
const NONASCII: u8 = 6;

const CLASS: [u8; 256] = build_class();

/// `[A-Za-z0-9_]` continuation table for the IDENT scanner: one load per
/// byte instead of `is_ascii_alphanumeric() || == b'_'`.
const IDENT_CONT: [bool; 256] = build_ident_cont();

const fn build_ident_cont() -> [bool; 256] {
    let mut table = [false; 256];
    let mut b = 0usize;
    while b < 256 {
        table[b] = (b >= b'0' as usize && b <= b'9' as usize)
            || (b >= b'A' as usize && b <= b'Z' as usize)
            || (b >= b'a' as usize && b <= b'z' as usize)
            || b == b'_' as usize;
        b += 1;
    }
    table
}

/// Bitmap of the `PASS` class over `0..0x80`, built from [`CLASS`] so the
/// two can never drift.  Bytes ≥ 0x80 are always `NONASCII`, never `PASS`,
/// so a single `u128` covers the whole PASS set.
///
/// The PASS run scan tests this bitmap (`(PASS_MASK >> b) & 1`) instead of
/// loading `CLASS[b]`: one shift+mask on a register value versus a 256-byte
/// table load that LLVM cannot vectorise through a gather.  The differential
/// test `pass_mask_matches_class` pins the two against each other.
const PASS_MASK: u128 = build_pass_mask();

const fn build_pass_mask() -> u128 {
    let mut m = 0u128;
    let mut b = 0usize;
    while b < 0x80 {
        if CLASS[b] == PASS {
            m |= 1u128 << b;
        }
        b += 1;
    }
    m
}

const fn build_class() -> [u8; 256] {
    let mut t = [PASS; 256];
    let mut i = 0;
    while i < 128 {
        t[i] = match i as u8 {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => IDENT,
            b'"' => STRING,
            b'\'' => CHAR,
            b'&' | b'*' => OPER,
            b'`' => TICK,
            _ => PASS,
        };
        i += 1;
    }
    while i < 256 {
        t[i] = NONASCII;
        i += 1;
    }
    t
}

/// Per-byte "equals `target`" SWAR mark for `x` (bit 7 of each result byte).
/// Derived from the `haszero` idiom on `x ^ target`: a byte is marked when
/// it is exactly `target`, with a conservative *superset* under borrow
/// chaining — the scalar fallback reclassifies any false positive, so the
/// run scan stays exact.
#[inline]
const fn eq_high_mask(x: u64, target: u64) -> u64 {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGHS: u64 = 0x8080_8080_8080_8080;
    let d = x ^ target.wrapping_mul(ONES);
    d.wrapping_sub(ONES) & !d & HIGHS
}

/// Length of the leading run of `PASS`-class bytes in `bytes` (bytes that
/// the highlighter copies verbatim: everything except identifiers' letters
/// and `_`, `"`, `'`, `&`, `*`, `` ` `` and non-ASCII).
///
/// SWAR fast path: 8 bytes per iteration.  A chunk is clean when
/// * no byte has the high bit set (non-ASCII is never PASS), and
/// * no byte folds (`b | 0x20`) into the letter range `0x61..=0x7A` — the
///   fold maps both cases (`a-z`, `A-Z`) there — and
/// * no byte folds into one of the special bytes: `_`/DEL → `0x7F`
///   (both `0x5F | 0x20` and `0x7F | 0x20` are `0x7F`; DEL is PASS but
///   marking it dirty is only conservative), `"` → `0x22`, `&` → `0x26`,
///   `'` → `0x27`, `*` → `0x2A`, `` ` `` → `0x60`.
///
/// Every SWAR predicate is a conservative superset (borrow chaining can
/// only add false positives, never skip a real member — see the
/// differential test), so a dirty chunk falls back to the exact scalar
/// bitmap loop, which re-decides from the chunk start.
fn pass_run_len(bytes: &[u8]) -> usize {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGHS: u64 = 0x8080_8080_8080_8080;
    const FOLD: u64 = 0x2020_2020_2020_2020;
    let len = bytes.len();
    let mut i = 0usize;
    while i + 8 <= len {
        let chunk = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        if chunk & HIGHS != 0 {
            break;
        }
        let t = chunk | FOLD;
        // Bytes in [0x60, 0x7A] minus bytes in [0x00, 0x60) — the letter
        // band with a deliberately conservative lower edge (`0x60` instead
        // of `0x61`) so borrow chaining can never exclude a real letter;
        // `` ` `` (0x60) and `{`/`[`/`\\` (0x7B on a borrow edge) may spill
        // in as false positives and cost one scalar pass.
        let lt_60 = t.wrapping_sub(0x60 * ONES) & HIGHS;
        let lt_7b = t.wrapping_sub(0x7b * ONES) & HIGHS;
        let letter_band = lt_7b & !lt_60;
        let special = eq_high_mask(t, 0x7f)
            | eq_high_mask(t, 0x22)
            | eq_high_mask(t, 0x26)
            | eq_high_mask(t, 0x27)
            | eq_high_mask(t, 0x2a)
            | eq_high_mask(t, 0x60);
        if letter_band != 0 || special != 0 {
            break;
        }
        i += 8;
    }
    // Scalar fallback: the dirty chunk's tail and the run's end.  Bitmap
    // test, not a `CLASS[b]` gather: bytes ≥ 0x80 are never PASS, so one
    // `u128` shift+mask answers without touching the 256-byte table.
    while i < len {
        let b = bytes[i];
        if b >= 0x80 || (PASS_MASK >> b) & 1 == 0 {
            break;
        }
        i += 1;
    }
    i
}

/// The scanner proper: appends the highlighted form of `line` to `out` and
/// returns whether any highlight was emitted.  When `false`, nothing was
/// appended and the caller copies `line` verbatim (the Cow wrappers then
/// return `Cow::Borrowed(line)` with zero allocation).
///
/// `line` must already be sanitized (see [`highlight_code_into`]): the
/// pass-through arms copy bytes verbatim, so a terminal control byte would
/// survive into `out`.
fn highlight_code_impl_into(
    line: &str,
    use_color: bool,
    add_backtick_quotes: bool,
    out: &mut String,
) -> bool {
    debug_assert!(use_color, "callers gate on use_color before this point");
    // Assert the sanitize-first precondition above in debug builds only: one
    // SWAR pass (cheaper than the scanner's own loop), no release cost.
    // `\t`/`\n` are excluded from the dirty set — the sanitizers keep them as
    // layout bytes and the PASS arm copies them through.
    debug_assert!(
        {
            let scan = super::scan_terminal_control_flags_multiline(line.as_bytes());
            !scan.dirty && !scan.has_c1
        },
        "highlight input must be sanitized first, got {line:?}"
    );
    // No separate "line has no ASCII letter" pre-scan: every rewrite arm
    // already funnels through `has_highlight`, so a line that triggers no
    // highlight (punctuation-only, digit-only, plain prose — the common
    // cases) leaves `out` untouched with zero allocation.  The old pre-scan
    // was a second full pass over every highlighted line for an answer the
    // main loop already computes.

    // Reset only foreground color and bold — preserve background color
    // so that callers can wrap the output in a background color block.
    let reset = "\x1b[22m\x1b[39m";
    let bold = "\x1b[1m";
    let hl = "\x1b[94m";

    // Lazy materialization: `out` stays empty until the first highlight is
    // proven necessary.  `flushed` tracks how many leading bytes of `line`
    // are already incorporated into `out` (flushed as plain text or pushed
    // as escaped highlight content); pass-through arms just advance `i` and
    // leave `out`/`flushed` untouched.  A line that triggers no highlight —
    // the common case for plain prose labels — leaves `out` empty (the
    // caller then copies `line` verbatim / returns `Cow::Borrowed`) with
    // zero allocation instead of eagerly copying the whole line.
    let mut flushed = 0usize;
    let mut has_highlight = false;
    let mut i = 0;
    let bytes = line.as_bytes();
    let len = bytes.len();
    // When `add_backtick_quotes` is true, type names are wrapped in
    // backticks (`` `Int<32>` ``) so that downstream consumers like
    // `write_header` can render them with consistent styling.
    let backtick = if add_backtick_quotes { "`" } else { "" };

    // Flush the not-yet-emitted plain prefix `line[flushed..end)` into
    // `out` and advance `flushed`.  Used at each highlight site to copy any
    // verbatim bytes that accumulated between the last highlight and this
    // one.  `out`, `line`, and `flushed` are the function locals above.
    macro_rules! flush_to {
        ($end:expr) => {{
            out.push_str(&line[flushed..$end]);
            flushed = $end;
        }};
    }

    while i < len {
        let b = bytes[i];

        match CLASS[b as usize] {
            // ── Verbatim run ────────────────────────────────────
            // Digits, whitespace, punctuation, brackets: copied in one
            // `push_str` per run instead of one `push` per byte.  The run
            // scan (`pass_run_len`) advances 8 bytes per iteration via SWAR
            // masks, falling back to the scalar bitmap loop on a dirty
            // chunk.  Runs of PASS bytes
            // are the dominant shape in real source (` = `, ` + `,
            // `(x, y)`), so this is the hottest arm.
            PASS => {
                i += 1;
                i += pass_run_len(&bytes[i..]);
            }

            // ── String literal ──────────────────────────────────
            // SIMD-scan for the next `"` (`memchr`) instead of a
            // per-byte `bytes[i] != b'"'` walk.  An escaped quote
            // (`\"`) is a candidate that gets rejected by counting the
            // backslashes immediately preceding it: an even count means
            // the quote is the closing one.
            STRING => {
                i += 1;
                while i < len {
                    match memchr::memchr(b'"', &bytes[i..]) {
                        Some(off) => {
                            let q = i + off;
                            let mut k = q;
                            while k > i && bytes[k - 1] == b'\\' {
                                k -= 1;
                            }
                            i = q + 1;
                            if (q - k) % 2 == 0 {
                                break;
                            }
                        }
                        None => {
                            // Unterminated string: consume to end of line.
                            i = len;
                            break;
                        }
                    }
                }
            }

            // ── Char literal / lifetime ─────────────────────────
            // A `'` is a *lifetime* prefix when the byte after it is an
            // identifier start AND the byte after *that* is not a closing
            // quote: `'a'` is a two-char char literal, `'a` (and `'static`)
            // is a lifetime.  Feeding a lifetime into the char-literal scan
            // below made it hunt for the next `'` and pair the lifetime's
            // apostrophe with a *later* one — in `def f<'a>(x: &'a Int)` the
            // scanner swallowed everything between the two lifetimes.
            // `'\n'`, `'\''` etc. start with `\`, so they fall through to
            // the literal scan unchanged; an unterminated char literal
            // still consumes to end of line.
            CHAR => {
                let lifetime = i + 1 < len
                    && (bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'_')
                    && (i + 2 >= len || bytes[i + 2] != b'\'');
                if lifetime {
                    // Copy the lifetime verbatim: advance past the
                    // identifier (the same shape as the IDENT scan) without
                    // scanning for a closing quote and without highlighting.
                    i += 1;
                    while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                        i += 1;
                    }
                } else {
                    // Symmetric to the string scan: `memchr` for the closing
                    // `'`, with escaped `'\''` rejected the same way.
                    i += 1;
                    while i < len {
                        match memchr::memchr(b'\'', &bytes[i..]) {
                            Some(off) => {
                                let q = i + off;
                                let mut k = q;
                                while k > i && bytes[k - 1] == b'\\' {
                                    k -= 1;
                                }
                                i = q + 1;
                                if (q - k) % 2 == 0 {
                                    break;
                                }
                            }
                            None => {
                                i = len;
                                break;
                            }
                        }
                    }
                }
            }

            // ── Identifier ──────────────────────────────────────
            // `[A-Za-z_][A-Za-z0-9_]*`, then check for a built-in type
            // (highlighted, possibly with a `<...>`/`(...)`/`[...]`
            // suffix) or a keyword (bold).  Anything else is copied
            // verbatim.  The inner scan keeps the `is_ascii_alphanumeric
            // || == '_'` shape — digits are legal *inside* an
            // identifier even though they are PASS at token start.
            IDENT => {
                let start = i;
                i += 1;
                while i < len && IDENT_CONT[bytes[i] as usize] {
                    i += 1;
                }
                let word = &line[start..i];
                if is_type(word) {
                    let type_end = skip_brackets(bytes, i);
                    flush_to!(start);
                    out.push_str(hl);
                    out.push_str(backtick);
                    out.push_str(&line[start..type_end]);
                    out.push_str(backtick);
                    out.push_str(reset);
                    flushed = type_end;
                    i = type_end;
                    has_highlight = true;
                } else if is_keyword(word) {
                    flush_to!(start);
                    out.push_str(bold);
                    out.push_str(word);
                    out.push_str(reset);
                    flushed = i;
                    has_highlight = true;
                }
            }

            // ── `&` / `*` prefix ────────────────────────────────
            // The operator is highlighted only when it is immediately
            // followed by a built-in type name — `&Int<32>` becomes one
            // highlighted span; `&x` and a lone `*` are copied verbatim.
            OPER => {
                if i + 1 < len && (bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'_') {
                    let op_pos = i;
                    i += 1;
                    let start = i;
                    while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                        i += 1;
                    }
                    let word = &line[start..i];
                    if is_type(word) {
                        let type_end = skip_brackets(bytes, i);
                        flush_to!(op_pos);
                        out.push_str(hl);
                        out.push_str(backtick);
                        out.push(b as char);
                        out.push_str(&line[start..type_end]);
                        out.push_str(backtick);
                        out.push_str(reset);
                        flushed = type_end;
                        i = type_end;
                        has_highlight = true;
                    } else {
                        // Not a type (`&mut x`, `*const p`): rewind to the
                        // identifier start so the IDENT arm re-examines the
                        // word — `mut`/`const` are keywords and must still
                        // highlight.  IDENT always advances past the word,
                        // so this cannot loop.
                        i = start;
                    }
                } else {
                    i += 1;
                }
            }

            // ── Backtick-quoted type name ──────────────────────
            // `` `Int<32>` `` highlights as one span when the content
            // (up to the first `<`/`(`/`[`, with any leading `&`/`*`
            // stripped) is a built-in type; otherwise it is copied
            // verbatim.  An unterminated backtick copies the remainder
            // and ends the pass.
            TICK => {
                let start = i;
                i += 1;
                let content_start = i;
                while i < len && bytes[i] != b'`' {
                    i += 1;
                }
                if i >= len {
                    // Unterminated backtick: drop to the tail flush,
                    // which emits the remainder verbatim (this token
                    // adds no highlight, so if nothing earlier
                    // highlighted either, the whole line is borrowed).
                    break;
                }
                let content = &line[content_start..i];
                // Check the content *before* any bracket suffix, e.g.
                // `Int<32>` → check `Int`; `&Str` → check `Str`.
                let type_name = content
                    .split(&['<', '(', '['][..])
                    .next()
                    .unwrap_or(content);
                let type_name = type_name.trim_start_matches(['&', '*']);
                if is_type(type_name) {
                    flush_to!(start);
                    out.push_str(hl);
                    out.push_str(&line[start..=i]);
                    out.push_str(reset);
                    flushed = i + 1;
                    has_highlight = true;
                }
                i += 1;
            }

            // ── Non-ASCII ────────────────────────────────────────
            // Decode the whole UTF-8 scalar (never split a multi-byte
            // char).  The resulting `char` cannot match any of the ASCII
            // token shapes, so it is emitted as-is; only `i` advances.
            // `NONASCII` implies `b0 >= 0x80`, and `i` is contractually on
            // a char boundary, so the scalar length is derivable from the
            // lead byte alone — no `#[cold]` `next_char_slow` call per
            // CJK char on a highlighted line.
            NONASCII => {
                let b0 = bytes[i];
                let n = if b0 < 0xE0 {
                    2
                } else if b0 < 0xF0 {
                    3
                } else {
                    4
                };
                i += n;
            }

            // `CLASS` only holds the seven tags above; the compiler knows
            // this, so this arm is unreachable for any byte the table
            // can produce.  Keep it as a loud failure for a future table
            // edit that introduces a new class, rather than a silent
            // fall-through that copies nothing.
            _ => unreachable!("CLASS table and match arms out of sync"),
        }
    }

    if !has_highlight {
        return false;
    }
    out.push_str(&line[flushed..]);
    true
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::diagnostics::Diagnostic;
    use crate::diagnostics::ErrCode;

    #[test]
    fn test_basic_error_render() {
        let source = "def a(x: Bool)\n  ensures @s > 1\n  {\n    set j = \"0xFFFF\";\n    set i = j + 1;\n    return @s @r i;\n  }\ndef main(){}";
        let diag = Diagnostic::error(
            "trait solver error: no trait implementation found for `Ord` on type `Int`",
        )
        .with_code_str(ErrCode::E030)
        .with_span(Span::new(FileId::DUMMY, 0, 106))
        .with_source(source);
        let renderer = GlyphRenderer::new(false);
        let output = renderer.render_diagnostic(&diag);
        println!("{}", output);
        assert!(output.contains("error[E030]"));
        assert!(output.contains("trait solver error"));
        assert!(output.contains("<input>"));
    }

    /// `grapheme_width` is `pub(crate)` and reached directly by
    /// `expand_tabs_from_col` / test oracles, so a caller-supplied
    /// `tab_width == 0` must not divide by zero.
    /// `LineColumns` must be byte-for-byte equivalent to the per-query walk it
    /// replaces — including mid-cluster offsets, offsets past the end, tab
    /// expansion and the zero-tab-width clamp.
    #[test]
    fn line_columns_matches_the_single_query_walk() {
        for line in [
            "",
            "a",
            "abc",
            "\t\t",
            "a\tb\tc",
            "日本語テキスト",
            "a日b",
            "e\u{301}x",
            "👩‍👩‍👧‍👦y",
            "\ta\t日\tb",
            "  \t  x",
        ] {
            for tab_width in [1usize, 2, 4, 8] {
                let cols = LineColumns::new(line, tab_width);
                for off in 0..=line.len() + 2 {
                    assert_eq!(
                        cols.display(off),
                        byte_offset_to_display(line, off, tab_width),
                        "line {line:?} tab {tab_width} off {off}"
                    );
                }
            }
            // A zero tab width follows `grapheme_width`'s clamp.
            assert_eq!(
                LineColumns::new(line, 0).display(line.len()),
                byte_offset_to_display(line, line.len(), 1),
                "zero tab width must clamp like grapheme_width: {line:?}"
            );
        }
    }

    /// The one-slot cache must answer every offset of a line exactly like a fresh
    /// table, and must not serve a stale line.
    #[test]
    fn line_columns_cache_matches_fresh_tables() {
        let mut cache = LineColumnsCache::default();
        let lines = ["aaa\tb", "日x\tc", "z"];
        for (idx, line) in lines.iter().enumerate() {
            for off in 0..=line.len() {
                let want = LineColumns::new(line, 4).display(off);
                assert_eq!(cache.display(idx, line, 4, off), want, "{idx} {off}");
            }
        }
        // Revisit line 0 after others: the slot must have been replaced.
        for off in 0..=lines[0].len() {
            let want = LineColumns::new(lines[0], 4).display(off);
            assert_eq!(cache.display(0, lines[0], 4, off), want, "revisit {off}");
        }
    }

    #[test]
    fn grapheme_width_clamps_zero_tab_width() {
        assert_eq!(grapheme_width("\t", 5, 0), 1);
        assert_eq!(grapheme_width_cached("\t", 5, 0, &mut WidthCache::new()), 1);
    }

    /// `render_summary_full` must be byte-identical to `render_summary` for
    /// `file_count <= 1` and append the file suffix only for a multi-file
    /// batch.
    #[test]
    fn render_summary_full_appends_file_suffix_only_for_multiple_files() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::En);
        let r = GlyphRenderer::new(false);
        for (e, w) in [(0, 0), (1, 0), (2, 1)] {
            assert_eq!(r.render_summary_full(e, w, 0), r.render_summary(e, w));
            assert_eq!(r.render_summary_full(e, w, 1), r.render_summary(e, w));
        }
        let multi = r.render_summary_full(2, 0, 3);
        assert!(multi.contains("3"), "file count missing: {multi:?}");
        assert!(
            multi.len() > r.render_summary(2, 0).len(),
            "multi-file summary did not append a suffix: {multi:?}"
        );
        // No diagnostics: no summary at any file count.
        assert!(r.render_summary_full(0, 0, 5).is_empty());
    }

    #[test]
    fn test_keyword_suggestion_format() {
        let source = "dem main(){}";
        let diag = Diagnostic::error("unexpected identifier `dem` at top level")
            .with_code_str(ErrCode::E003)
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

    /// `is_type` dispatches by first byte into a small `matches!` table;
    /// every unhandled first byte — must agree with `TYPES::contains`.
    #[test]
    fn type_dispatch_matches_contains() {
        let mut cases: Vec<String> = TYPES.iter().map(|k| (*k).to_string()).collect();
        cases.extend(
            [
                "", "x", "Int2", "INT", "ints", "Stringg", "Unit_", "USizeX", "zzz", "αβ",
            ]
            .iter()
            .map(|s| (*s).to_string()),
        );
        for b in b'A'..=b'Z' {
            cases.push(format!("{}zzz", b as char));
        }

        for word in &cases {
            assert_eq!(
                is_type(word),
                TYPES.contains(&word.as_str()),
                "is_type disagrees with TYPES::contains for {word:?}"
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
        assert_eq!(
            byte_range_to_display("a\x07b", 2, 3, DEFAULT_TAB_WIDTH),
            (2, 1)
        );
    }

    #[test]
    fn grapheme_widths_match_terminal() {
        assert_eq!(grapheme_width("a", 0, DEFAULT_TAB_WIDTH), 1);
        assert_eq!(grapheme_width("中", 0, DEFAULT_TAB_WIDTH), 2);
        assert_eq!(grapheme_width("🚀", 0, DEFAULT_TAB_WIDTH), 2); // previously missed (outside 1F300–1F64F)
        assert_eq!(grapheme_width("e\u{0301}", 0, DEFAULT_TAB_WIDTH), 1); // combining acute = 0 width
        assert_eq!(
            grapheme_width("👨\u{200D}👩\u{200D}👧\u{200D}👦", 0, DEFAULT_TAB_WIDTH),
            2
        ); // ZWJ family
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
                assert_eq!(
                    grapheme_width(&s, 1, DEFAULT_TAB_WIDTH),
                    3,
                    "tab from col 1"
                );
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
    fn byte_offset_display_uses_containing_cluster_start() {
        let line = "e\u{0301}f"; // graphemes: "e\u{0301}" (3 bytes), "f" (1 byte)
        assert_eq!(byte_offset_to_display(line, 0, DEFAULT_TAB_WIDTH), 0);
        assert_eq!(
            byte_offset_to_display(line, 1, DEFAULT_TAB_WIDTH),
            0,
            "offset inside the combining cluster must snap to its start"
        );
        assert_eq!(
            byte_offset_to_display(line, 2, DEFAULT_TAB_WIDTH),
            0,
            "the combining mark is still inside the first cluster"
        );
        assert_eq!(byte_offset_to_display(line, 3, DEFAULT_TAB_WIDTH), 1);
        assert_eq!(byte_offset_to_display(line, 4, DEFAULT_TAB_WIDTH), 2);
        assert_eq!(byte_offset_to_display(line, 5, DEFAULT_TAB_WIDTH), 2);
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
        meta: &[(char, Option<usize>)],
        highlight_on: &str,
        highlight_off: &str,
    ) -> (usize, String, Vec<(usize, usize, char, Option<usize>)>) {
        let display_buf = byte_ranges_to_display(line, ranges, DEFAULT_TAB_WIDTH);
        let display_width = line_display_width(line, DEFAULT_TAB_WIDTH);
        let raw_underlines: Vec<(usize, usize, char, Option<usize>)> = display_buf
            .iter()
            .zip(meta.iter())
            .map(|(&(col, width), &(ch, label_idx))| (col, width, ch, label_idx))
            .collect();
        // Primary marks come from the raw widths, matching the fused
        // renderer's byte-overlap rule: a point on a cluster boundary
        // highlights nothing, while a point inside a cluster has already
        // expanded to the whole cluster in the conversion above.
        let mut primary_cols = Vec::new();
        compute_primary_cols(&raw_underlines, display_width, &mut primary_cols);
        // Insertion-point policy (mirrors the fused pipeline's underline
        // rows): zero-width annotations reach lane assignment one cell wide.
        let underlines: Vec<(usize, usize, char, Option<usize>)> = raw_underlines
            .iter()
            .map(|&(col, width, ch, label_idx)| (col, width.max(1), ch, label_idx))
            .collect();

        let mut rendered = String::new();
        let mut in_primary = false;
        let mut disp_col = 0usize;
        if crate::diagnostics::ascii::is_printable_ascii(line.as_bytes()) {
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
                } else if g.chars().all(crate::diagnostics::is_bidi_override) {
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
            let meta: Vec<(char, Option<usize>)> = ranges
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i % 2 == 0 {
                        ('^', None)
                    } else {
                        ('~', Some(i))
                    }
                })
                .collect();
            let (want_width, want_rendered, want_out) =
                legacy_render(line, ranges, &meta, highlight_on, highlight_off);

            let mut out: SmallVec<[(usize, usize, char, Option<usize>); 8]> = SmallVec::new();
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
                0, // test oracle: content starts at column 0
                false,
            );

            assert_eq!(got_width, want_width, "width for line={line:?}");
            assert_eq!(rendered, want_rendered, "render for line={line:?}");
            assert_eq!(
                out.as_slice(),
                want_out.as_slice(),
                "underlines for line={line:?}"
            );
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
            .with_code_str(ErrCode::E030)
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

    /// `sug_covered` makes a suggestion-covered row take the highlighted path
    /// instead of the `context_only` plain path.  The claim under test is that
    /// both paths emit the *same* bytes for such a row (suggestions add their
    /// diff block separately and never contribute source-row ranges), i.e. the
    /// term is redundant.  This pins the equivalence directly: the second
    /// source row renders identically whether or not a suggestion covers it.
    #[test]
    fn suggestion_covered_row_renders_like_a_plain_row() {
        let src = "aaa\nbbb\n";
        let covered = Diagnostic::error("boom")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_span_suggestion(
                Span::new(FileId::DUMMY, 4, 7),
                "ccc",
                "replace `bbb` with `ccc`",
                Applicability::MachineApplicable,
            )
            .with_source(src);
        // Same diagnostic, suggestion moved onto the first row: the second row
        // is then *not* suggestion-covered.
        let not_covered = Diagnostic::error("boom")
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_span_suggestion(
                Span::new(FileId::DUMMY, 0, 3),
                "zzz",
                "replace `aaa` with `zzz`",
                Applicability::MachineApplicable,
            )
            .with_source(src);
        let row = |out: &str, needle: &str| -> String {
            out.lines()
                .find(|l| l.contains(needle) && !l.contains("- ") && !l.contains("+ "))
                .unwrap_or_else(|| panic!("row {needle:?} missing in {out:?}"))
                .to_string()
        };
        let a = GlyphRenderer::new(false).render_diagnostic(&covered);
        let b = GlyphRenderer::new(false).render_diagnostic(&not_covered);
        assert_eq!(
            row(&a, "bbb"),
            row(&b, "bbb"),
            "a suggestion-covered row must render exactly like a plain row:\n{a}\n---\n{b}"
        );
        // And the covered render still emits its diff block.
        assert!(a.contains("+ ccc"), "diff block lost: {a}");
    }

    #[test]
    fn span_suggestion_renders_diff() {
        let src = "dem main(){}\n";
        let diag = Diagnostic::error("unexpected identifier `dem` at top level")
            .with_code_str(ErrCode::E003)
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
        let minus = lines.iter().find(|l| l.contains("- dem main(){}")).unwrap();
        let plus = lines.iter().find(|l| l.contains("+ def main(){}")).unwrap();
        let prefix_before = |l: &str, marker: char| -> String {
            l.find(marker).map(|i| &l[..i]).unwrap_or(l).to_string()
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
            .with_code_str(ErrCode::E003)
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
            .with_code_str(ErrCode::E003)
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
            span: Some(Span {
                file: FileId::DUMMY,
                start: 5,
                end: 2,
            }), // end < start
            replacement: Some("def".into()),
        };
        let diag = Diagnostic::error("unexpected identifier `dem`")
            .with_code_str(ErrCode::E003)
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
            .with_code_str(ErrCode::E003)
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
            .with_code_str(ErrCode::E030)
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
            .with_code_str(ErrCode::E030)
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
        assert_eq!(starts, vec![0u32, 2, 5, 6]);
        for (i, &s) in starts.iter().enumerate() {
            assert_eq!(line_start_byte(src, i), s as usize, "mismatch at line {i}");
        }
    }

    #[test]
    fn osc8_url_is_percent_encoded() {
        let src = "def x() {}\n";
        let name = "a b\x1b]8;;evil\x07.ps";
        // The URL encoder must neutralise control bytes and spaces whether or
        // not the terminal probe enables hyperlinks (the probe is now
        // stderr-tty-gated, so a bare test process legitimately disables them).
        let url = crate::diagnostics::term::path_to_file_url(std::path::Path::new(name));
        assert!(
            !url.contains('\x1b') && !url.contains('\x07'),
            "control char leaked into file URL: {url:?}"
        );
        assert!(url.contains("%20"), "space not encoded: {url:?}");

        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_source(src)
            .with_file_name(name);
        // Pinned on: a bare test process reports no hyperlink support, so this branch used
        // to be skipped exactly when it was the only thing testing the URL *inside* OSC 8.
        // The capability is a test input now (`with_hyperlinks_override`), and the render
        // must happen inside the pin — rendering outside it produced no hyperlink at all
        // and made the new "links were emitted" assertion fail in every terminal.
        crate::diagnostics::term::with_hyperlinks_override(true, || {
            let out = GlyphRenderer::new(true).render_diagnostic(&diag);
            assert!(
                !out.contains("\x1b]8;;evil"),
                "control char leaked into OSC 8 URL: {out:?}"
            );
            assert!(out.contains("%20"), "space not encoded: {out:?}");
            assert!(
                out.contains("\x1b]8;;"),
                "hyperlinks were not emitted: {out:?}"
            );
        });
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
            assert_eq!(
                line_display_width(s, DEFAULT_TAB_WIDTH),
                s.len(),
                "line_display_width fast path: {s:?}"
            );
            assert_eq!(visible_width(s), s.len(), "visible_width fast path: {s:?}");
        }
        // Fall-through cases: not all bytes printable ASCII.
        assert_eq!(line_display_width("\t", DEFAULT_TAB_WIDTH), 4); // tab → TAB_WIDTH at col 0
        assert_eq!(line_display_width("\x7f", DEFAULT_TAB_WIDTH), 1); // DEL → one U+FFFD cell
        assert_eq!(line_display_width("中", DEFAULT_TAB_WIDTH), 2); // CJK → 2 cells
        assert_eq!(visible_width("\t"), 4);
        assert_eq!(visible_width("\x7f"), 1);
    }

    /// The PASS bitmap must agree with the `CLASS` table on every byte, so
    /// the run-scan's shift+mask can never classify differently than the
    /// dispatcher's table load.
    #[test]
    fn pass_mask_matches_class() {
        for b in 0..=255u8 {
            let table_pass = CLASS[b as usize] == PASS;
            let mask_pass = b < 0x80 && (PASS_MASK >> b) & 1 == 1;
            assert_eq!(table_pass, mask_pass, "byte {b:#04x}");
        }
    }

    #[test]
    fn ident_continuation_table_matches_predicate() {
        for b in 0..=255u8 {
            let expected = b.is_ascii_alphanumeric() || b == b'_';
            assert_eq!(IDENT_CONT[b as usize], expected, "byte {b:#04x}");
        }
    }

    /// Non-ASCII characters fall through the token scanner's `next_char`
    /// decoder; every UTF-8 length class (2/3/4 bytes) must survive verbatim.
    #[test]
    fn highlight_code_preserves_multibyte_chars() {
        for s in ["é", "中", "🙂", "a中b", "é中🙂", "xé中🙂y"] {
            assert_eq!(highlight_code(s, true), s, "mangled {s:?}");
        }
    }

    /// The ASCII fast paths (batch PASS copy, SIMD string/char scan) must not
    /// change the rendered bytes: string contents, char literals, types,
    /// keywords, and verbatim runs all reach the same output as the old
    /// per-byte scanner.
    #[test]
    fn ascii_fast_paths_match_scanner_output() {
        // String with an escaped quote, char literal, type, `&Type`,
        // backtick-quoted type, keyword, and a verbatim run in between.
        let line = r#"set x = "a\"b" + 'q' + Int<32> + &Str + `Bool`;"#;
        let out = highlight_code(line, true);

        // String content untouched, escaped quote preserved verbatim.
        assert!(out.contains(r#""a\"b""#), "string mangled: {out:?}");
        // Char literal untouched.
        assert!(out.contains("'q'"), "char mangled: {out:?}");
        // Type with its bracket suffix highlighted as one span.
        assert!(
            out.contains("\x1b[94mInt<32>\x1b[22m\x1b[39m"),
            "type: {out:?}"
        );
        // `&Type` highlighted as one span (operator included).
        assert!(
            out.contains("\x1b[94m&Str\x1b[22m\x1b[39m"),
            "&type: {out:?}"
        );
        // Backtick-quoted type highlighted.
        assert!(
            out.contains("\x1b[94m`Bool`\x1b[22m\x1b[39m"),
            "tick: {out:?}"
        );
        // Keyword bold.
        assert!(
            out.contains("\x1b[1mset\x1b[22m\x1b[39m"),
            "keyword: {out:?}"
        );
    }

    /// Batch PASS copy must reproduce a run of digits, spaces and punctuation
    /// byte for byte — every non-token run between two interesting tokens.
    #[test]
    fn verbatim_run_is_copied_whole() {
        // `Int<32>` is the only highlighted token; everything after it is a
        // PASS run except the `x` identifier and the bare `*` operator.
        let line = "Int<32> + 42 - (x) * 7";
        let out = highlight_code(line, true);
        assert!(
            out.contains(" + 42 - (x) * 7"),
            "verbatim run mangled: {out:?}"
        );
    }

    /// A label message longer than `max_width` wraps, and every continuation
    /// line repeats the `|` connector so the annotation reads continuously.
    #[test]
    fn long_label_message_wraps_with_repeated_connector() {
        let src = "set i = 1;\n";
        let long = "this label repeats the connector over and over again to force a wrap";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
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

    /// A help message longer than `max_width` wraps, and each continuation
    /// line must place the `│` bar **before** the padding — the bar stays in
    /// the gutter and the text aligns under the first line's message.
    #[test]
    fn note_locator_suffix_respects_max_width() {
        let opts = crate::diagnostics::DiagOpts {
            max_width: 40,
            ..crate::diagnostics::DiagOpts::default()
        };
        let renderer = GlyphRenderer::with_opts(false, Some(&opts));
        let mut out = String::new();
        renderer.write_note_line_with(
            &mut out,
            "this is a very long note message that must wrap before its locator",
            Some("f.ps:12:3"),
        );
        for line in out.lines() {
            assert!(
                visible_width(line) <= 40,
                "note line exceeds max width: {line:?}"
            );
        }
        assert!(out.lines().last().unwrap().contains(" at f.ps:12:3"));
    }

    /// The note's locator reserve must be *reserved-optimal*, not the two-phase
    /// composition Φ (wrap at the full measure, then re-wrap the overflowing
    /// tail narrow): Φ cannot pull words from earlier lines onto the reserved
    /// line, so it can spend a whole extra line.  Line count is compared against
    /// `wrap_optimal_reserved`, with Φ as an upper bound.
    #[test]
    fn note_locator_wrap_is_reserved_optimal() {
        let opts = crate::diagnostics::DiagOpts {
            max_width: 30,
            ..crate::diagnostics::DiagOpts::default()
        };
        let renderer = GlyphRenderer::with_opts(false, Some(&opts));
        let prefix_cols = renderer.prefix_cols(DiagnosticLevel::Note);
        let avail_full = renderer.max_width.saturating_sub(prefix_cols).max(1);
        let suffix = "f.ps:12:3";
        let suffix_cols = 4 + visible_width(suffix);
        let narrow = avail_full.saturating_sub(suffix_cols).max(1);
        let mut improved = false;
        for msg in [
            "alpha bravo charlie delta echo foxtrot",
            "aaa bbb ccc ddd eee fff ggg hhh iii jjj",
            "one two three four five six seven eight nine ten",
            "aa bb cc dd ee ff gg hh ii jj kk ll",
        ] {
            let sanitized = crate::diagnostics::sanitize_for_terminal_multiline(msg);
            let mut out = String::new();
            renderer.write_note_line_with(&mut out, msg, Some(suffix));
            let lines = out.lines().count();
            let optimal = crate::diagnostics::wrap::wrap_optimal_reserved(
                &sanitized,
                avail_full,
                renderer.tab_width,
                suffix_cols,
            )
            .len();
            assert_eq!(
                lines, optimal,
                "note wrap is not the reserved-optimal layout: {msg:?}"
            );
            let full =
                crate::diagnostics::wrap::wrap_optimal(&sanitized, avail_full, renderer.tab_width);
            let phi = match full.last() {
                Some(last)
                    if visible_width_with_tab(last.as_ref(), renderer.tab_width) + suffix_cols
                        > avail_full =>
                {
                    full.len() - 1
                        + crate::diagnostics::wrap::wrap_optimal(
                            last.as_ref(),
                            narrow,
                            renderer.tab_width,
                        )
                        .len()
                }
                _ => full.len(),
            };
            assert!(
                lines <= phi,
                "regressed against two-phase Φ: {msg:?} -> {lines} > {phi}"
            );
            improved |= lines < phi;
        }
        assert!(
            improved,
            "corpus never exercises the case Φ gets wrong; the oracle is vacuous"
        );
    }

    /// Regression: `write_help_line` (and `write_note_line`) once pushed the
    /// pad before the bar, shifting the bar right by `prefix_cols - 1`
    /// columns and mis-aligning every continuation line.
    #[test]
    fn long_help_message_wraps_with_aligned_continuation() {
        let src = "set i = 1;\n";
        let long = "this help repeats the connector over and over again to force a wrap";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 5))
            .with_help(long)
            .with_source(src);
        let opts = crate::diagnostics::DiagOpts {
            max_width: 40,
            use_unicode: false,
            ..Default::default()
        };
        let out = GlyphRenderer::with_opts(false, Some(&opts)).render_diagnostic(&diag);
        let mut wrapped_lines = 0;
        for line in out.lines() {
            // Words unique to the help text: each continuation must start
            // with the bar itself — no leading whitespace before the gutter.
            if line.contains("this") || line.contains("over") || line.contains("wrap") {
                wrapped_lines += 1;
                assert!(
                    line.starts_with('|'),
                    "help continuation lost the gutter: {line:?}\n{out}"
                );
            }
        }
        assert!(
            wrapped_lines >= 2,
            "expected the help to wrap across lines, got:\n{out}"
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
                .with_code_str(ErrCode::E030)
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
            .with_code_str(ErrCode::E030)
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
        let rendered: Vec<&str> = out.lines().filter(|line| line.contains("dup")).collect();
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
                .with_code_str(ErrCode::E030)
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
            .with_code_str(ErrCode::E030)
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
            .with_code_str(ErrCode::E030)
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

    /// 12 labels on one line exceed the SmallVec inline capacity (8) and
    /// exercise the spill-to-heap path; all labels must still render.
    #[test]
    fn many_labels_on_one_line_spill_smallvec_cleanly() {
        let src = "x = 1;\n";
        let mut diag = Diagnostic::error("many")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 1));
        for i in 0..12 {
            diag = diag.with_label(Span::new(FileId::DUMMY, 0, 1), format!("L{i}"));
        }
        let out = GlyphRenderer::new(false).render_diagnostic(&diag.with_source(src));
        for i in 0..12 {
            assert!(out.contains(&format!("L{i}")), "L{i} missing:\n{out}");
        }
    }

    /// Randomised differential test: `advance` line-by-line and compare
    /// `active()` against a from-scratch filter (the set definition).
    #[test]
    fn label_window_batch_matches_reference() {
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };
        for _ in 0..300 {
            let n = 1 + (next() % 30) as usize;
            let ranges: Vec<(usize, usize)> = (0..n)
                .map(|_| {
                    let s = (next() % 50) as usize;
                    (s, s + (next() % 20) as usize)
                })
                .collect();
            let mut win = LabelWindow::new(&ranges, &[], LabelWindowScratch::new());
            for line in 0..80usize {
                win.advance(line);
                let want: Vec<usize> = (0..n)
                    .filter(|&i| ranges[i].0 <= line && ranges[i].1 >= line)
                    .collect();
                assert_eq!(
                    win.active(),
                    want.as_slice(),
                    "line={line} ranges={ranges:?}"
                );
            }
        }
    }

    /// 100 labels starting on the same line: the burst path must keep
    /// original index order, then clear all at once after the end line.
    #[test]
    fn label_window_same_line_burst_keeps_index_order() {
        let ranges: Vec<(usize, usize)> = (0..100).map(|_| (10, 20)).collect();
        let mut win = LabelWindow::new(&ranges, &[], LabelWindowScratch::new());
        win.advance(10);
        let want: Vec<usize> = (0..100).collect();
        assert_eq!(win.active(), want.as_slice());
        win.advance(21);
        assert!(win.active().is_empty());
    }

    /// zero-width annotations render as one-cell markers at
    /// the point itself — line start, line middle, and line end (including
    /// the final character position of a file without a trailing newline).
    #[test]
    fn zero_width_annotations_render_one_cell_markers() {
        let src = "aaa\nbbb\nccc";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
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
            .with_code_str(ErrCode::E030)
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
            .with_code_str(ErrCode::E030)
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
            message: None,
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
            .with_code_str(ErrCode::E030)
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
            .with_code_str(ErrCode::E030)
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
            .with_code_str(ErrCode::E030)
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
            .with_code_str(ErrCode::E030)
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

    // ── Layout tests (compact mode, fold counts, width unify, inline labels) ──

    /// Compact mode drops the sub-box header prefix, the spacing
    /// row, and the close row; the full layout keeps all three.
    #[test]
    fn compact_mode_drops_box_furniture() {
        let source = "let x = 1;\n";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
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
        let spacing_present = plain_full.lines().any(|l| {
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
            !plain_compact
                .lines()
                .any(|l| l.contains("└─") || l.contains("\\-")),
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

    /// A gap between disjoint sections gets a line count in the
    /// fold marker; contiguous sections get a bare `...`.
    #[test]
    fn fold_marker_reports_omitted_line_count() {
        // 60 lines of source; two spans 50 lines apart → a real gap that
        // no amount of context bridging can fold into one section.
        let source = "fn f() {\n".to_string() + &"line\n".repeat(58) + "}\n";
        let mid = source.find("line\n").unwrap() + 50 * 5; // ~50 lines in
        let mid = mid as u32;
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 4))
            .with_additional_span(Span::new(FileId::DUMMY, mid, mid + 4))
            .with_source(&source);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            out.contains("lines omitted"),
            "gap between sections must report the omitted line count: {out}"
        );
    }

    /// Sections keep one shared line-number column width even when
    /// the file's last line only appears in the final section.
    #[test]
    fn line_number_width_is_unified_across_sections() {
        // Span 1 is early (1-digit line numbers), span 2 lands past line
        // 10 (2-digit) — the first section's gutter must still be 2 wide.
        let source = "a\n".repeat(15);
        let end = source.len() as u32 - 1;
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
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
        assert!(
            !row15.is_empty(),
            "line-15 source row missing:\n{plain_full}"
        );
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

    /// A single non-multiline label renders its message inline
    /// (no vertical-bar connector row); two labels on the same row keep
    /// the connector layout.
    #[test]
    fn single_label_message_is_inline() {
        let source = "let x = 1;\n";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_label(
                Span::new(FileId::DUMMY, 0, 3),
                "the type of this expression",
            )
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
            .with_code_str(ErrCode::E030)
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
            (4usize, 6usize, '^', None),
            (4usize, 6usize, '^', Some(0usize)),
            (5usize, 10usize, '^', Some(1usize)),
        ];
        let mut lanes = UnderlineLanes::new();
        let count = lanes.assign(&underlines);
        assert_eq!(count, 2, "identical annotations must collapse to one lane");
    }

    #[test]
    fn underline_lanes_collapse_non_adjacent_identical() {
        // A different-char annotation on the same span interleaves the two
        // `^` twins in `order` (sorted by `(start, end)`, stable), so the
        // dedup must not rely on adjacency. Without the seen-set the two
        // `^` entries stack as a redundant third lane.
        let underlines = [
            (4usize, 6usize, '^', None),
            (4usize, 6usize, '~', Some(0usize)),
            (4usize, 6usize, '^', Some(1usize)),
        ];
        let mut lanes = UnderlineLanes::new();
        let count = lanes.assign(&underlines);
        assert_eq!(
            count, 2,
            "non-adjacent identical annotations must collapse; the `~` keeps its own lane"
        );
    }

    // ── Lane relabelling per overlap component ──

    /// Per-lane visual weight used by the weighted-depth-sum objective:
    /// width dominates, a labelled underline also carries a rendered message.
    fn lane_weight(underlines: &[(usize, usize, char, Option<usize>)], i: usize) -> usize {
        underlines[i].1 + usize::from(underlines[i].3.is_some())
    }

    /// Weighted depth sum (WDS) of an assignment: Σ over lanes of
    /// `lane weight × lane index`.
    fn wds(
        lanes: &UnderlineLanes,
        underlines: &[(usize, usize, char, Option<usize>)],
        n: usize,
    ) -> usize {
        (0..n)
            .map(|l| {
                let w: usize = lanes.lanes[l]
                    .iter()
                    .map(|&i| lane_weight(underlines, i))
                    .sum();
                w * l
            })
            .sum()
    }

    /// Overlap components of an assignment, each as `(lane, component weight)`
    /// pairs in ascending lane order.
    ///
    /// Recomputed here from the finished lanes rather than shared with
    /// `UnderlineLanes::assign`, so the exhaustive oracle below stays an
    /// independent check: the components are the runs of the start-sorted
    /// entries that keep overlapping (`next.start <= running max end`).
    fn assignment_components(
        lanes: &UnderlineLanes,
        underlines: &[(usize, usize, char, Option<usize>)],
        n: usize,
    ) -> Vec<Vec<(usize, usize)>> {
        // `(start, end, lane, index)` of every entry on a used lane.
        let mut items: Vec<(usize, usize, usize, usize)> = Vec::new();
        for l in 0..n {
            for &i in lanes.lanes[l].iter() {
                let (col, ulen, ..) = underlines[i];
                items.push((col, col + ulen, l, i));
            }
        }
        items.sort_unstable_by_key(|&(start, end, lane, i)| (start, end, lane, i));

        let mut out: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut start = 0usize;
        while start < items.len() {
            let mut end = start + 1;
            let mut max_end = items[start].1;
            while end < items.len() && items[end].0 <= max_end {
                max_end = max_end.max(items[end].1);
                end += 1;
            }
            let mut weights: Vec<(usize, usize)> = Vec::new();
            for &(_, _, lane, i) in &items[start..end] {
                let w = lane_weight(underlines, i);
                match weights.iter_mut().find(|(used, _)| *used == lane) {
                    Some(entry) => entry.1 += w,
                    None => weights.push((lane, w)),
                }
            }
            weights.sort_unstable();
            out.push(weights);
            start = end;
        }
        out
    }

    /// Minimum WDS over *all* per-component relabelings of an assignment:
    /// every bijection of each component's lane-index set is tried.  Written
    /// with its own recursion (not the production `next_permutation`) so a
    /// bug in either enumerator cannot hide the other.
    fn brute_force_component_min(
        lanes: &UnderlineLanes,
        underlines: &[(usize, usize, char, Option<usize>)],
        n: usize,
    ) -> usize {
        fn rec(
            k: usize,
            cur: &mut Vec<usize>,
            used: &mut [bool],
            visit: &mut impl FnMut(&[usize]),
        ) {
            if cur.len() == k {
                visit(cur);
                return;
            }
            for i in 0..k {
                if !used[i] {
                    used[i] = true;
                    cur.push(i);
                    rec(k, cur, used, visit);
                    cur.pop();
                    used[i] = false;
                }
            }
        }

        let mut total = 0usize;
        for comp in assignment_components(lanes, underlines, n) {
            let lane_set: Vec<usize> = comp.iter().map(|&(lane, _)| lane).collect();
            let weight: Vec<usize> = comp.iter().map(|&(_, w)| w).collect();
            let k = lane_set.len();
            let mut best = usize::MAX;
            let mut visit = |perm: &[usize]| {
                let cost: usize = (0..k).map(|p| weight[p] * lane_set[perm[p]]).sum();
                best = best.min(cost);
            };
            let mut cur = Vec::with_capacity(k);
            let mut used = vec![false; k];
            rec(k, &mut cur, &mut used, &mut visit);
            total += best;
        }
        total
    }

    /// The greedy numbers lanes in opening order; the weighted-depth-sum
    /// objective wants the *heaviest* lane at index 0.  One wide primary
    /// plus many short disjoint labels that all overlap it stack the labels
    /// into a second lane whose total weight easily outweighs the primary —
    /// so the relabel must move that heavier lane to index 0.
    #[test]
    fn lanes_are_relabelled_by_descending_weight() {
        // The greedy opens lanes in `(start, end)` order, so the primary (which
        // sorts first) opens lane 0; every label lies *inside* the primary, so
        // each one is forced into lane 1.  Nineteen labels weigh
        // 19 × (1 + 1) = 38 against the primary's 20, so the relabel must move
        // the label lane to index 0.  The previous numbers (primary 1000 vs
        // 240, labels outside it) produced the identity permutation, so the
        // assertion held even with the relabel code deleted.
        let mut underlines: Vec<(usize, usize, char, Option<usize>)> =
            vec![(0usize, 20usize, '^', None)]; // primary, weight 20
        for k in 0..19usize {
            underlines.push((1 + k, 1, '~', Some(k)));
        }

        let mut lanes = UnderlineLanes::new();
        let count = lanes.assign(&underlines);
        assert_eq!(count, 2, "primary plus interior labels fits in two lanes");

        let weight = |l: usize| -> usize {
            lanes.lanes[l]
                .iter()
                .map(|&i| underlines[i].1 + usize::from(underlines[i].3.is_some()))
                .sum()
        };
        let (w0, w1) = (weight(0), weight(1));
        assert!(
            w0 >= w1,
            "lane 0 ({w0}) must be at least as heavy as lane 1 ({w1})"
        );
        assert!(
            underlines[lanes.lanes[0][0]].3.is_some(),
            "the label lane is the heavier one, so the relabel must put it at index 0: {:?}",
            lanes.lanes
        );

        // Second case: the primary is already the heaviest lane, so the
        // permutation is the identity.  Pinning both directions keeps a relabel
        // that always swaps from passing the assertion above.
        let mut underlines: Vec<(usize, usize, char, Option<usize>)> =
            vec![(0usize, 100usize, '^', None)]; // primary, weight 100
        for k in 0..19usize {
            underlines.push((1 + k, 1, '~', Some(k)));
        }
        let mut lanes = UnderlineLanes::new();
        assert_eq!(lanes.assign(&underlines), 2);
        assert_eq!(
            lanes.lanes[0].first().copied(),
            Some(0),
            "the primary is the heaviest lane and must stay at index 0: {:?}",
            lanes.lanes
        );
    }

    /// After `assign`, each overlap component's own lanes are non-increasing in
    /// the component's visual weight (the rearrangement-inequality orientation
    /// for the fixed interval→lane assignment), and the assignment itself stays
    /// well-formed: every underline appears exactly once across lanes.
    ///
    /// The claim is per *component*, not global: disjoint components reuse the
    /// same lane indices, so a global monotonicity assertion would contradict
    /// the per-component optimum (the witness in
    /// `lane_relabel_per_component_matches_exhaustive_search` leaves global
    /// lane weights (20, 2) — still sorted — but a row whose components are
    /// lighter-left/heavier-right legitimately is not).
    #[test]
    fn relabel_keeps_component_lane_weights_non_increasing_and_assignment_wellformed() {
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };
        for _ in 0..200 {
            let count = 2 + (next() % 8) as usize;
            let mut underlines: Vec<(usize, usize, char, Option<usize>)> = Vec::new();
            for i in 0..count {
                let start = (next() % 60) as usize;
                let width = 1 + (next() % 20) as usize;
                underlines.push((start, width, '^', if i % 3 == 0 { None } else { Some(i) }));
            }
            underlines.sort_by_key(|&(c, w, _, _)| (c, c + w));

            let mut lanes = UnderlineLanes::new();
            let n = lanes.assign(&underlines);

            // Within each component, weight is non-increasing with the lane
            // index: the heaviest lane sits closest to the source line.
            for comp in assignment_components(&lanes, &underlines, n) {
                let mut prev = usize::MAX;
                for &(lane, w) in comp.iter() {
                    assert!(
                        w <= prev,
                        "component lane {lane} weight {w} exceeds the previous {prev}: {:?}",
                        comp
                    );
                    prev = w;
                }
            }

            // Well-formed: every *distinct* annotation appears exactly once —
            // `assign` collapses identical `(start, ulen, char)` twins by
            // design (they render as one fused row), so compare against the
            // deduped input, not the raw count.
            let mut distinct: Vec<(usize, usize, char)> =
                underlines.iter().map(|&(c, w, ch, _)| (c, w, ch)).collect();
            distinct.sort_unstable();
            distinct.dedup();
            let mut seen: SmallVec<[usize; 16]> = SmallVec::new();
            for l in 0..n {
                seen.extend(lanes.lanes[l].iter().copied());
            }
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(
                seen.len(),
                distinct.len(),
                "assignment lost distinct entries"
            );
        }
    }

    /// Exactness pin.
    ///
    /// For a row whose every component has `χ ≤ 3`, the sweep DP must return the
    /// minimum `Σ w·lane` over **all proper colourings**, not merely over
    /// relabellings of the greedy's — that distinction is invisible to
    /// `lane_relabel_per_component_matches_exhaustive_search`, which searches
    /// bijections only.  The oracle here re-derives the components with its own
    /// union-find, enumerates every colouring of each component into the lane set
    /// that component actually occupies, keeps the proper ones, and takes the
    /// per-component minimum (components are column-disjoint, so the objective
    /// separates).
    #[test]
    fn exact_tier_matches_brute_force_over_all_proper_colourings() {
        let mut seed: u64 = 0x00C2_5EED_0BAD_1234;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        for _ in 0..300 {
            let count = 2 + (next() % 5) as usize; // 2..=6
            let mut underlines: Vec<(usize, usize, char, Option<usize>)> = Vec::new();
            for i in 0..count {
                let start = (next() % 20) as usize;
                let width = 1 + (next() % 5) as usize;
                let ch = ['^', '~', '-'][i % 3];
                underlines.push((start, width, ch, if i % 4 == 0 { None } else { Some(i) }));
            }
            underlines.sort_by_key(|&(c, w, _, _)| (c, c + w));

            let mut lanes = UnderlineLanes::new();
            let n = lanes.assign(&underlines);
            if n == 0 || n > 3 {
                continue; // the tier only covers χ ≤ 3
            }
            let mut lane_of = vec![usize::MAX; underlines.len()];
            for (l, members) in lanes.lanes.iter().enumerate().take(n) {
                for &i in members {
                    lane_of[i] = l;
                }
            }
            assert!(
                lane_of.iter().all(|&l| l != usize::MAX),
                "every interval must sit in a lane"
            );

            fn root(parent: &mut [usize], mut x: usize) -> usize {
                while parent[x] != x {
                    parent[x] = parent[parent[x]];
                    x = parent[x];
                }
                x
            }
            let mut parent: Vec<usize> = (0..underlines.len()).collect();
            for i in 0..underlines.len() {
                for j in (i + 1)..underlines.len() {
                    let (si, wi, ..) = underlines[i];
                    let (sj, wj, ..) = underlines[j];
                    if si < sj + wj && sj < si + wi {
                        let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                        parent[a] = b;
                    }
                }
            }
            let mut expected = 0usize;
            let mut seen = vec![false; underlines.len()];
            for i in 0..underlines.len() {
                if seen[i] {
                    continue;
                }
                let r = root(&mut parent, i);
                let members: Vec<usize> = (0..underlines.len())
                    .filter(|&j| root(&mut parent, j) == r)
                    .collect();
                for &m in &members {
                    seen[m] = true;
                }
                let mut lane_set: Vec<usize> = members.iter().map(|&m| lane_of[m]).collect();
                lane_set.sort_unstable();
                lane_set.dedup();
                let k = lane_set.len();
                let mut best = usize::MAX;
                let mut code = vec![0usize; members.len()];
                loop {
                    let mut proper = true;
                    for a in 0..members.len() {
                        for b in (a + 1)..members.len() {
                            let (ia, ib) = (members[a], members[b]);
                            let (sa, wa, ..) = underlines[ia];
                            let (sb, wb, ..) = underlines[ib];
                            if sa < sb + wb
                                && sb < sa + wa
                                && lane_set[code[a]] == lane_set[code[b]]
                            {
                                proper = false;
                            }
                        }
                    }
                    if proper {
                        let cost: usize = members
                            .iter()
                            .enumerate()
                            .map(|(pos, &m)| lane_weight(&underlines, m) * lane_set[code[pos]])
                            .sum();
                        best = best.min(cost);
                    }
                    let mut k2 = 0;
                    while k2 < members.len() {
                        code[k2] += 1;
                        if code[k2] < k {
                            break;
                        }
                        code[k2] = 0;
                        k2 += 1;
                    }
                    if k2 == members.len() {
                        break;
                    }
                }
                expected += best;
            }
            assert_eq!(
                wds(&lanes, &underlines, n),
                expected,
                "χ≤3 tier must be exact: {underlines:?}"
            );
        }
    }

    /// The per-component relabel must reach the exact minimum of the
    /// weighted-depth-sum objective over all per-component labelings, checked
    /// against an independent brute force on small random interval families,
    /// plus a hand-verifiable witness.
    ///
    /// (The `#[test]` here had slid onto the function above, so this pin never ran;
    /// `every_test_module_fn_is_collected` in `mod.rs` now catches the family.)
    #[test]
    fn lane_relabel_per_component_matches_exhaustive_search() {
        // (a) Deterministic LCG (no `rand` crate): ≤ 6 intervals over 24
        // columns, so a row tops out at a handful of lanes.
        let mut seed: u64 = 0x0BAD_F00D_1234_5678;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        for _ in 0..300 {
            let count = 2 + (next() % 5) as usize; // 2..=6
            let mut underlines: Vec<(usize, usize, char, Option<usize>)> = Vec::new();
            for i in 0..count {
                let start = (next() % 24) as usize;
                let width = 1 + (next() % 6) as usize;
                // Rotating the underline char keeps most entries distinct
                // through `assign`'s duplicate pass; a collapse would only
                // shrink the row, never invalidate the comparison.
                let ch = ['^', '~', '-'][i % 3];
                underlines.push((start, width, ch, if i % 4 == 0 { None } else { Some(i) }));
            }
            underlines.sort_by_key(|&(c, w, _, _)| (c, c + w));

            // (b) The assignment, as `assign` produced it.
            let mut lanes = UnderlineLanes::new();
            let n = lanes.assign(&underlines);
            assert!(n <= count, "χ cannot exceed the interval count");

            // (c) Against the brute-force minimum over per-component
            // relabelings.
            let got = wds(&lanes, &underlines, n);
            let want = brute_force_component_min(&lanes, &underlines, n);
            assert_eq!(
                got, want,
                "WDS {got} != per-component optimum {want} for {underlines:?} (lanes: {:?})",
                lanes.lanes
            );
        }

        // (d) Witness, verifiable by hand: two *disjoint* overlapping pairs on
        // one row with component weights (10, 1) and (1, 10).  The greedy
        // reuses lane 1 for the second component's first interval (lane 1's
        // end is the smallest and already past it) and lane 0 for its second,
        // so both components span lanes 0/1 and the global lane weights are
        // 11 / 11 — a global descending-weight sort ties and changes nothing,
        // WDS 11.  Relabelling each pair within itself costs 1 + 1 = 2.
        let witness: Vec<(usize, usize, char, Option<usize>)> = vec![
            (0, 10, '^', None),  // component A, weight 10 → lane 0
            (2, 1, '~', None),   // component A, weight 1  → lane 1
            (11, 10, '^', None), // component B, weight 10 → lane 1 (reused)
            (12, 1, '~', None),  // component B, weight 1  → lane 0 (reused)
        ];
        let mut lanes = UnderlineLanes::new();
        assert_eq!(lanes.assign(&witness), 2, "both pairs fit in two lanes");
        assert_eq!(
            wds(&lanes, &witness, 2),
            2,
            "per-component relabel must reach WDS 2, lanes: {:?}",
            lanes.lanes
        );
        // Both components end up sorted on their own: the heavy entries share
        // lane 0.
        assert!(
            lanes.lanes[0].iter().all(|&i| witness[i].1 == 10),
            "lane 0 must hold both heavy entries: {:?}",
            lanes.lanes
        );
    }

    // ── Regression tests: context window, diff block, no-source labels ──

    /// `context_lines = 0` must show exactly the covered lines — the
    /// old `end_line + 1 + context_lines` window dragged one line below the
    /// span in even at zero context.
    #[test]
    fn zero_context_shows_no_line_below_the_span() {
        let src = "one\ntwo\nthree\nfour\nfive\n";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 8, 13)) // `three` on line 2
            .with_source(src);
        let opts = crate::diagnostics::DiagOpts {
            context_lines: 0,
            ..Default::default()
        };
        let out = GlyphRenderer::with_opts(false, Some(&opts)).render_diagnostic(&diag);
        assert!(out.contains("three"), "covered line missing:\n{out}");
        assert!(
            !out.contains("four") && !out.contains("five"),
            "zero context must not show lines below the span:\n{out}"
        );
        assert!(
            !out.contains("two"),
            "context above must honour the window:\n{out}"
        );
    }

    /// `context_lines = 2` shows exactly two lines below the covered
    /// lines — not three.
    #[test]
    fn context_lines_two_shows_exactly_two_lines_below() {
        let src = "one\ntwo\nthree\nfour\nfive\nsix\nseven\n";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 8, 13)) // `three` on line 2
            .with_source(src);
        let opts = crate::diagnostics::DiagOpts {
            context_lines: 2,
            ..Default::default()
        };
        let out = GlyphRenderer::with_opts(false, Some(&opts)).render_diagnostic(&diag);
        for want in ["three", "four", "five"] {
            assert!(
                out.contains(want),
                "{want} (within 2 context) missing:\n{out}"
            );
        }
        assert!(
            !out.contains("six"),
            "context_lines=2 must not show a third line below:\n{out}"
        );
    }

    /// A span whose end sits exactly on a line start does not drag the
    /// following line into the window (`span_lines` already attributes the
    /// end to the previous line).
    #[test]
    fn span_ending_on_line_start_adds_no_phantom_row() {
        let src = "aaa\nbbb\nccc\n";
        // (0, 4) covers `aaa` plus the newline: last covered line is 0.
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 4))
            .with_source(src);
        let opts = crate::diagnostics::DiagOpts {
            context_lines: 0,
            ..Default::default()
        };
        let out = GlyphRenderer::with_opts(false, Some(&opts)).render_diagnostic(&diag);
        assert!(out.contains("aaa"), "covered line missing:\n{out}");
        assert!(
            !out.contains("bbb"),
            "an end on a line start must not pull the next line in:\n{out}"
        );
    }

    /// On the final covered line of a multi-line label, the message loop
    /// owns the connector + message row; the connector loop must not draw a
    /// second, stray `|` row for the `end` part.
    #[test]
    fn multiline_label_end_line_has_no_duplicate_connector() {
        let src = "aaa\nbbb\nccc\nddd\n";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 1))
            .with_label(Span::new(FileId::DUMMY, 0, 7), "spans two lines")
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        let lines: Vec<&str> = out.lines().collect();
        let msg_pos = lines
            .iter()
            .position(|l| l.contains("spans two lines"))
            .expect("label message missing");
        // Between the message row and the next source row ("ccc") no bare
        // connector row (gutter + `|`, no text) may appear — that was the
        // stray duplicate `end` connector.
        for line in &lines[msg_pos + 1..] {
            if line.contains("ccc") {
                break;
            }
            let bare = !line.chars().any(|c| c.is_alphanumeric()) && line.trim_end().ends_with('|');
            assert!(
                !bare,
                "stray duplicate connector after the label message:\n{out}"
            );
        }
        // The message itself is still rendered exactly once.
        assert_eq!(out.matches("spans two lines").count(), 1, "{out}");
    }

    #[test]
    fn multiline_label_connector_uses_label_start_column() {
        let src = "    aaa\n    bbb\n    ccc\n";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 4, 7))
            .with_label(Span::new(FileId::DUMMY, 4, 23), "spans three lines")
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        let lines: Vec<&str> = out.lines().collect();
        let caret_col = lines
            .iter()
            .find(|l| l.contains('^'))
            .and_then(|l| l.find('^'))
            .expect("caret row missing");
        let bbb_row = lines
            .iter()
            .position(|l| l.contains("bbb"))
            .expect("middle source row missing");
        let connector = lines[bbb_row + 1..]
            .iter()
            .find(|l| l.trim_end().ends_with('|') && !l.chars().any(|c| c.is_alphabetic()))
            .expect("middle connector row missing");
        assert_eq!(
            connector.rfind('|').unwrap(),
            caret_col,
            "connector must sit under the label's opening caret:\n{out}"
        );
    }

    /// Wrapped structural-suggestion continuations align under the
    /// message, not under the glyph.
    #[test]
    fn wrapped_suggestion_continuations_align_under_message() {
        let src = "set i = j;\n";
        let long = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam".repeat(2);
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_span_suggestion(
                Span::new(FileId::DUMMY, 0, 3),
                "def",
                &long,
                Applicability::MachineApplicable,
            )
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        let lines: Vec<&str> = out.lines().collect();
        let first = lines
            .iter()
            .position(|l| l.contains("help: "))
            .expect("suggestion message missing");
        // Column of the message text on the first line ("help" of `help:`).
        let help_col = lines[first].find("help").unwrap();
        let cont = &lines[first + 1];
        assert!(
            !cont.contains("set i"),
            "message did not wrap; test inconclusive:\n{out}"
        );
        // First alphabetic character of the continuation = its text start;
        // it must sit under the *message* column, i.e. the `help:` label's
        // column plus the `label: ` gutter (`label.len() + 2`), not back at
        // the glyph column and not past the message column.
        let text_col = cont
            .char_indices()
            .find(|(_, c)| c.is_alphabetic())
            .map(|(i, _)| i)
            .expect("continuation has no text");
        assert_eq!(
            text_col,
            help_col + "help".len() + 2,
            "wrapped continuation not aligned under the message:\n{out}"
        );
    }

    /// A single-line span whose replacement contains a newline must
    /// fall back to the text-only suggestion path — splicing it into the
    /// `-`/`+` rows would produce a malformed multi-line diff row.
    #[test]
    fn single_line_span_with_multiline_replacement_falls_back_to_text() {
        let src = "dem main(){}\n";
        let diag = Diagnostic::error("unexpected identifier `dem`")
            .with_code_str(ErrCode::E003)
            .with_span(Span::new(FileId::DUMMY, 0, 3))
            .with_span_suggestion(
                Span::new(FileId::DUMMY, 0, 3),
                "def\nmain(){}",
                "replace `dem` with a two-line form",
                Applicability::MachineApplicable,
            )
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            !out.contains("- dem main(){}") && !out.contains("+ def main(){}"),
            "multi-line replacement must not splice into a diff row:\n{out}"
        );
        assert!(
            out.contains("replace `dem` with a two-line form"),
            "text fallback missing:\n{out}"
        );
    }

    /// A tab-indented source line in the diff block expands tabs like
    /// the main source rows do — no raw `\t` may reach the diff rows.
    #[test]
    fn diff_block_lines_expand_tabs() {
        let src = "\tset i = j;\n";
        let diag = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 1, 4))
            .with_span_suggestion(
                Span::new(FileId::DUMMY, 1, 4),
                "let",
                "replace `set` with `let`",
                Applicability::MachineApplicable,
            )
            .with_source(src);
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            out.contains("- \tset") == false,
            "raw tab in diff row:\n{out:?}"
        );
        let minus = out
            .lines()
            .find(|l| l.contains("- ") && l.contains("set i"));
        let minus = minus.expect("diff - row missing");
        assert!(
            !minus.contains('\t'),
            "diff row kept a raw tab: {minus:?}\n{out}"
        );
        // Tabs expand to the next stop from the row's *absolute* terminal
        // column: the main source row's payload starts after the
        // `│ <num> │ ` gutter (1+1+1+1+1 = 5 cells → 4 − 5%4 = 3 spaces),
        // while the diff row's payload starts after gutter + glyph + the
        // 2-column `- `/`+ ` marker (8 cells → 4 spaces).  Both are
        // stop-aligned in absolute terms (5+3 = 8 = 8+4 mod 4 = 0) — the
        // payloads legitimately differ, exactly as a real terminal renders
        // them.  Compare display columns (char counts — the plain renderer
        // emits no ANSI and the `>`/`?` applicability marker is one cell), not
        // byte offsets.
        let main_row = out
            .lines()
            .find(|l| l.contains("set i = j;") && !l.contains("- "))
            .unwrap();
        let gutter_end = main_row.match_indices('|').nth(1).map(|(i, _)| i).unwrap();
        let main_payload = &main_row[gutter_end + 2..]; // skip '|' + gutter space
        let m_ws = main_payload.len() - main_payload.trim_start().len();
        let d_gutter_end = minus.match_indices('|').nth(1).map(|(i, _)| i).unwrap();
        let d_rest = &minus[d_gutter_end + 2..]; // ">- <expanded payload>"
        let d_payload = &d_rest[d_rest.find("- ").expect("diff marker missing") + 2..];
        let d_ws = d_payload.len() - d_payload.trim_start().len();
        // Absolute columns: main content starts at 5 (`│ 1 │ `), the diff
        // payload at 8 (gutter 5 + glyph 1 + `- ` 2); DEFAULT_TAB_WIDTH = 4.
        assert_eq!(m_ws, 3, "main row tab must stop-align from col 5:\n{out}");
        assert_eq!(d_ws, 4, "diff row tab must stop-align from col 8:\n{out}");
    }

    /// Labels are not dropped when no source is attached — each label
    /// message stays visible as an `= note: <msg> at <span>` line.
    #[test]
    fn labels_render_without_source() {
        let diag = Diagnostic::error("duplicate definition")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 40, 44))
            .with_label(Span::new(FileId::DUMMY, 10, 14), "previous definition here");
        let out = GlyphRenderer::new(false).render_diagnostic(&diag);
        assert!(
            out.contains("previous definition here"),
            "label dropped with no source attached:\n{out}"
        );
        assert!(out.contains("at 10..14"), "label locator missing:\n{out}");
        assert_eq!(
            out.matches("previous definition here").count(),
            1,
            "label duplicated:\n{out}"
        );
    }

    /// A lifetime must not enter the char-literal scan — the scanner
    /// used to pair the lifetime's apostrophe with a later one and swallow
    /// the tokens between (here, the `set` keyword).
    #[test]
    fn lifetime_is_not_scanned_as_char_literal() {
        let out = highlight_code("m = 'a; set x't", true);
        assert!(
            out.contains("\x1b[1mset\x1b[22m\x1b[39m"),
            "keyword swallowed by a lifetime's apostrophe: {out:?}"
        );
        // Two lifetimes: nothing between them may be swallowed either.
        let out = highlight_code("f<'a>(&x: Int, y: &'b Int)", true);
        assert!(
            out.contains("\x1b[94mInt\x1b[22m\x1b[39m"),
            "type lost between lifetimes: {out:?}"
        );
    }

    /// Real char literals keep working — plain, escaped, and the
    /// quote-as-char corner.
    #[test]
    fn char_literals_still_scan_correctly() {
        for line in ["c = 'x' + 'y'", r"c = '\n' + '\''", "c = 'x'"] {
            let out = highlight_code(line, true);
            // The literal text is preserved byte for byte (no highlight in
            // these char-only lines — but nothing may be consumed wrongly
            // either).
            let plain: String = strip_ansi(&out);
            assert_eq!(plain, line, "char literal mangled: {line:?} → {out:?}");
        }
        // A char literal adjacent to a lifetime does not confuse either.
        let out = highlight_code("p = 'a' && q = &'b Int", true);
        assert!(
            out.contains("\x1b[94mInt\x1b[22m\x1b[39m"),
            "type after char-literal + lifetime lost: {out:?}"
        );
    }

    /// Scalar reference for the PASS run scan (the pre-SWAR loop).
    fn pass_run_len_scalar(bytes: &[u8]) -> usize {
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if b >= 0x80 || (PASS_MASK >> b) & 1 == 0 {
                break;
            }
            i += 1;
        }
        i
    }

    /// Differential test: the SWAR run scan must advance exactly as far
    /// as the scalar bitmap loop — over all 256 byte values (at every
    /// substring offset) and over deterministic pseudo-random byte strings
    /// (at every alignment, exercising unaligned 8-byte chunk loads).
    #[test]
    fn pass_run_len_swar_matches_scalar() {
        // Every byte value, alone and in adversarial combinations, at every
        // start offset.
        let mut all: Vec<u8> = (0..=255).collect();
        all.extend([
            b'a', b'A', b'_', b'"', b'\'', b'&', b'*', b'`', 0x7f, 0x80, 0xff,
        ]);
        for start in 0..all.len() {
            for end in start..=all.len() {
                let window = &all[start..end];
                assert_eq!(
                    pass_run_len(window),
                    pass_run_len_scalar(window),
                    "bytes {start}..{end} of the full table"
                );
            }
        }
        // Deterministic xorshift byte strings, all alignments.
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..300 {
            let n = (next() % 96) as usize + 1;
            let buf: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            for s in 0..buf.len() {
                assert_eq!(
                    pass_run_len(&buf[s..]),
                    pass_run_len_scalar(&buf[s..]),
                    "buf {buf:?} from offset {s}"
                );
            }
        }
    }
}
