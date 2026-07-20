//! Custom source-code renderer for diagnostics.
//!
//! Produces Rust-inspired output with `┌─└─│ ^^^` box-drawing,
//! replacing the previous miette-based renderer.
//!
//! Format:
//! ```text
//! ┌─ error[E003]: unexpected identifier `dem` at top level
//! │
//! │  ┌─ input:1:1
//! │  │
//! │1 │ dem main(){
//! │  │ ^^^
//! │  │
//! │  └─ only declarations are allowed at the top level.
//! │     expected: def | set | type | trait | impl | …
//! │     actual:   dem
//! │
//! │  = note: `dem` is not a keyword. Did you mean `def`?
//! │
//! └─ aborting due to previous error.
//! ```

use crate::ast::Span;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::label::{AnnotationKind, Label, SourcePos};
use crate::diagnostics::level::DiagnosticLevel;
use std::fmt::Write;

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
    Warning,
    Note,
    Help,
    Bold,
    Dim,
    BrightRed,
    Cyan,
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
    red_bold: &'static str,
    cyan: &'static str,
    blue: &'static str,
    magenta: &'static str,
    green: &'static str,
    yellow: &'static str,
}

impl Styles {
    const fn new(use_color: bool) -> Self {
        if use_color {
            Styles {
                use_color: true,
                reset: "\x1b[0m",
                bold: "\x1b[1m",
                dim: "\x1b[2m",
                red_bold: "\x1b[31;1m",
                cyan: "\x1b[36m",
                blue: "\x1b[34m",
                magenta: "\x1b[35m",
                green: "\x1b[32m",
                yellow: "\x1b[33m",
            }
        } else {
            Styles {
                use_color: false,
                reset: "",
                bold: "",
                dim: "",
                red_bold: "",
                cyan: "",
                blue: "",
                magenta: "",
                green: "",
                yellow: "",
            }
        }
    }

    fn get(&self, style: Style) -> &'static str {
        if !self.use_color {
            return "";
        }
        match style {
            Style::Error => "\x1b[31;1m",
            Style::Warning => "\x1b[33;1m",
            Style::Note => "\x1b[34m",
            Style::Help => "\x1b[36m",
            Style::Bold => self.bold,
            Style::Dim => self.dim,
            Style::BrightRed => self.red_bold,
            Style::Cyan => self.cyan,
            Style::Blue => self.blue,
            Style::Green => self.green,
            Style::Yellow => self.yellow,
        }
    }
}

// ── Box-drawing characters ──────────────────────────────────────

// ── GlyphRenderer ───────────────────────────────────────────────

/// Renders diagnostics in the Rust-inspired box-drawing format.
pub struct GlyphRenderer {
    use_color: bool,
    context_lines: usize,
    bc: &'static BoxChars,
    s: Styles,
}

impl GlyphRenderer {
    pub fn new(use_color: bool) -> Self {
        GlyphRenderer {
            use_color,
            context_lines: 2,
            bc: if use_color { &UNICODE } else { &ASCII },
            s: Styles::new(use_color),
        }
    }

    pub fn with_context_lines(mut self, n: usize) -> Self {
        self.context_lines = n;
        self
    }

    // ── Public API ──────────────────────────────────────────────

    /// Render a single diagnostic to its full formatted string.
    /// Returns an empty string if the diagnostic has no source text.
    pub fn render_diagnostic(&self, diag: &Diagnostic) -> String {
        let mut out = String::new();

        // ┌─ error[E003]: message
        self.write_header(&mut out, diag);

        if let Some(ref source) = diag.source {
            // Merge labels from related_errors once, used for all primary spans.
            let mut merged_labels = diag.labels.clone();
            for rel in &diag.related_errors {
                if let Some(rel_span) = rel.span {
                    let label_text = if let Some(ref code) = rel.code {
                        format!("{} [{}]", rel.message, code.code())
                    } else {
                        rel.message.clone()
                    };
                    merged_labels.push(Label::secondary(rel_span, label_text));
                }
            }
            // Render a source section for EACH primary span, so that
            // multi-location diagnostics (e.g. duplicate definition + original)
            // show both locations with their own source context.
            for &primary_span in &diag.spans.primary {
                self.write_source_section(&mut out, source, primary_span, &merged_labels, "<input>");
            }
        }

        // ── Fallback: render labels that were not shown in source context ──
        // The write_source_section method only renders labels whose spans
        // intersect the displayed line range.  Labels whose spans are far
        // away (e.g. a "previous definition here" on a different function)
        // would be silently dropped.  This fallback loop ensures they are
        // still visible, matching the old emitters' behaviour.
        if let Some(ref source) = diag.source {
            if let Some(span) = diag.spans.first() {
                let start_pos = byte_to_linecol(source, span.start);
                let end_pos = byte_to_linecol(source, span.end);
                let lines: Vec<&str> = source.lines().collect();
                let first_line = start_pos.line.saturating_sub(self.context_lines);
                let last_line = std::cmp::min(
                    end_pos.line + 1 + self.context_lines,
                    lines.len().saturating_sub(1),
                );
                for lbl in &diag.labels {
                    let lbl_line = span_line(lbl.span, source);
                    if lbl_line < first_line || lbl_line > last_line {
                        let _ = writeln!(
                            out,
                            "{v}  {sub_h}{sub_h}{sub_h} {ch} {span}: {msg}",
                            v = self.bc.v,
                            sub_h = self.bc.sub_h,
                            ch = lbl.underline_char(),
                            span = lbl.span,
                            msg = lbl.message,
                        );
                    }
                }
            }
        }

        // Children (sub-diagnostics: notes, help)
        for child in &diag.children {
            self.write_child(&mut out, child);
        }

        // Suggestions (as = note: ...)
        for suggestion in &diag.suggestions {
            self.write_note_line(&mut out, &suggestion);
        }

        // Help
        if let Some(ref help) = diag.help {
            self.write_help_line(&mut out, help);
        }

        out
    }

    /// Render the summary line (e.g. "aborting due to N previous errors").
    pub fn render_summary(&self, error_count: usize, warning_count: usize) -> String {
        let mut out = String::new();
        let msg = match (error_count, warning_count) {
            (0, 0) => return out,
            (1, 0) => "aborting due to previous error".to_string(),
            (n, 0) => format!("aborting due to {} previous errors", n),
            (0, 1) => "1 warning emitted".to_string(),
            (0, n) => format!("{} warnings emitted", n),
            (e, w) => format!(
                "aborting due to {} previous error{}; {} warning{} emitted",
                e,
                if e == 1 { "" } else { "s" },
                w,
                if w == 1 { "" } else { "s" },
            ),
        };
        let _ = write!(
            out,
            "{}{}{}{} {}{}{}",
            self.s.dim, self.bc.bl, self.bc.h, self.s.reset, self.s.red_bold, msg, self.s.reset,
        );
        out
    }

    // ── Header ──────────────────────────────────────────────────

    fn write_header(&self, out: &mut String, diag: &Diagnostic) {
        let level_label = diag.level.label();
        let level_color = diag.level.ansi_color();
        let code_str = diag
            .code
            .as_ref()
            .map(|c| c.code())
            .unwrap_or("?");
        // Combine primary error code with related error codes, e.g. "E019,E030".
        let all_codes: String = if diag.related_errors.is_empty() {
            code_str.to_string()
        } else {
            let mut codes = vec![code_str];
            for rel in &diag.related_errors {
                if let Some(c) = rel.code.as_ref() {
                    let c = c.code();
                    if !codes.contains(&c) {
                        codes.push(c);
                    }
                }
            }
            codes.join(",")
        };
        let _ = writeln!(
            out,
            "{dim}{tl}{h}{reset} {level_color}{bold}{level}{reset}[{bold}{all_codes}{reset}]: {msg_prefix}",
            dim = self.s.dim,
            tl = self.bc.tl,
            h = self.bc.h,
            reset = self.s.reset,
            level_color = level_color,
            bold = self.s.bold,
            level = level_label,
            all_codes = all_codes,
            msg_prefix = if diag.related_errors.is_empty() {
                diag.message.clone()
            } else {
                format!("1. {}", diag.message)
            },
        );
        // Calculate the indentation for related error items so they align
        // with the "1." on the header line:
        //   ┌─ error[E019,E030]: 1. duplicate definition of `i`
        //   │                    2. type mismatch
        let indent = " ".repeat(level_label.len() + all_codes.len() + 6); // "level[code]: " + 1
        // List related errors as numbered items under the header.
        for (i, rel) in diag.related_errors.iter().enumerate() {
            let _ = writeln!(
                out,
                "{v}{indent}{num}. {msg}",
                v = self.bc.v,
                indent = indent,
                num = i + 2,
                msg = rel.message,
            );
        }
    }

    // ── Source section ───────────────────────────────────────────

    fn write_source_section(
        &self,
        out: &mut String,
        source: &str,
        primary_span: Span,
        labels: &[Label],
        filename: &str,
    ) {
        // Collect all annotated spans: primary + labels
        let mut all_labels: Vec<Label> = Vec::with_capacity(labels.len() + 1);
        all_labels.push(Label {
            span: primary_span,
            message: String::new(),
            kind: AnnotationKind::Primary,
        });
        all_labels.extend(labels.iter().cloned());

        let lines: Vec<&str> = source.lines().collect();
        if lines.is_empty() {
            return;
        }

        let start_pos = byte_to_linecol(source, primary_span.start);
        let end_pos = byte_to_linecol(source, primary_span.end);

        // Determine the line range to display (with context)
        let first_line = start_pos.line.saturating_sub(self.context_lines);
        let last_line = std::cmp::min(
            end_pos.line + 1 + self.context_lines,
            lines.len().saturating_sub(1),
        );

        // Width of the line number column
        let line_num_width = format!("{}", last_line + 1).len();
        // Indent for lines that don't have a line number (location header, spacing, explanation)
        let indent = " ".repeat(line_num_width + 1);

        // Location header: ┌─ input:1:1
        let _ = writeln!(
            out,
            "{dim}{v}{indent}{sub_tl}{sub_h}{reset} {cyan}{filename}:{line}:{col}{reset}",
            dim = self.s.dim,
            v = self.bc.v,
            indent = indent,
            sub_tl = self.bc.sub_tl,
            sub_h = self.bc.sub_h,
            reset = self.s.reset,
            cyan = self.s.cyan,
            filename = filename,
            line = start_pos.line + 1,
            col = start_pos.col + 1,
        );
        // Spacing line
        let _ = writeln!(
            out,
            "{dim}{v}{indent}{sub_v}{reset}",
            dim = self.s.dim,
            v = self.bc.v,
            indent = indent,
            sub_v = self.bc.sub_v,
            reset = self.s.reset
        );

        // Render each line in the range
        for line_idx in first_line..=last_line {
            let line = lines[line_idx];

            // Compute underlines for this line from all labels
            // (runs BEFORE source line rendering so we can use the
            // primary span info to add background highlighting).
            let underlines = compute_line_underlines(&all_labels, source, line, line_idx);

            // ── Compute primary span columns for background highlighting ──
            // Build a set of columns that are covered by a primary underline
            // (underline_char == '^').  These columns will get a subtle
            // background color to make the error location stand out.
            let mut primary_cols: Vec<bool> = vec![false; line.len()];
            for (col, ulen, underline_char, _msg) in &underlines {
                if *underline_char == '^' {
                    let end = std::cmp::min(col + ulen, line.len());
                    for i in *col..end {
                        if i < primary_cols.len() {
                            primary_cols[i] = true;
                        }
                    }
                }
            }

            // Apply background color to primary span columns.
            // Split the line into segments: normal / highlighted / normal.
            let code_bg = "\x1b[48;5;236m";
            let reset_fg = "\x1b[22m\x1b[39m"; // reset bold+fg only, preserve bg
            let full_reset = "\x1b[0m";
            let mut rendered = String::with_capacity(line.len() + 64);
            let mut i = 0;
            while i < line.len() {
                if primary_cols[i] {
                    // Start of a primary span segment
                    rendered.push_str(code_bg);
                    let start = i;
                    while i < line.len() && primary_cols[i] {
                        i += 1;
                    }
                    // Apply syntax highlighting within the primary span
                    let segment = &line[start..i];
                    rendered.push_str(&highlight_code(segment, self.s.use_color));
                    rendered.push_str(reset_fg);
                    rendered.push_str(full_reset); // reset background
                } else {
                    let start = i;
                    while i < line.len() && !primary_cols[i] {
                        i += 1;
                    }
                    let segment = &line[start..i];
                    rendered.push_str(&highlight_code(segment, self.s.use_color));
                }
            }

            // Line number + source
            let _ = writeln!(
                out,
                "{dim}{v}{reset}{cyan}{line_num:>width$}{reset} {dim}{sub_v}{reset} {rendered}",
                dim = self.s.dim,
                v = self.bc.v,
                reset = self.s.reset,
                cyan = self.s.cyan,
                line_num = line_idx + 1,
                width = line_num_width,
                sub_v = self.bc.sub_v,
                rendered = rendered,
            );

            // ── Detect multi-line labels ──
            // For labels that span multiple lines, determine if this line is
            // the first, middle, or last line of the annotation, so we can
            // render `_` connectors and `|` inline marks accordingly.
            // Collect labels that are multi-line and overlap this line.
            let mut multiline_flags: Vec<(usize, usize, char, &str, &str)> = Vec::new();
            for (col, ulen, underline_char, msg) in &underlines {
                // Find the original label in all_labels that matches this underline.
                for lbl in &all_labels {
                    let label_start_line = span_line(lbl.span, source);
                    let label_end_line = span_line(Span::new(lbl.span.end, lbl.span.end), source);
                    if label_start_line != label_end_line {
                        // This label spans multiple lines.
                        // Check if this underline corresponds to this label.
                        if *col == (lbl.span.start.saturating_sub(line_start_byte(source, line_idx)))
                            || lbl.message.as_str() == *msg
                        {
                            let part = if line_idx == label_start_line {
                                "start"
                            } else if line_idx == label_end_line {
                                "end"
                            } else {
                                "middle"
                            };
                            multiline_flags.push((*col, *ulen, *underline_char, msg, part));
                            break;
                        }
                    }
                }
            }

            // ── Render annotation line ──
            // Combine all underlines into a single line, like rustc's `- ^ -`.
            // Sorted by column so overlapping labels are handled correctly.
            if !underlines.is_empty() {
                let spaces = " ".repeat(line_num_width + 1);
                // Build the combined annotation: merge overlapping segments
                // (primary `^` takes precedence over secondary `~` / `-`).
                let mut combined: Vec<char> = vec![' '; line.len()];
                // Priority: Primary > Secondary > Note (lower number = higher priority)
                fn priority(c: char) -> u8 {
                    match c {
                        '^' => 0,
                        '~' => 1,
                        '-' => 2,
                        _ => 3,
                    }
                }
                for (col, ulen, underline_char, _msg) in &underlines {
                    let end = std::cmp::min(col + ulen, line.len());
                    for i in *col..end {
                        if priority(*underline_char) < priority(combined[i]) {
                            combined[i] = *underline_char;
                        }
                    }
                }
                // Trim trailing spaces from the combined annotation
                let combined_str: String = combined.iter().collect();
                let trimmed = combined_str.trim().to_string();
                // Only render if there are actual annotations
                if !trimmed.is_empty() {
                    // Compute the column offset for the first non-space character
                    let first_col = combined.iter().position(|c| *c != ' ').unwrap_or(0);
                    let padded = " ".repeat(first_col) + &trimmed;
                    let _ = writeln!(
                        out,
                        "{dim}{v}{spaces}{sub_v}{reset} {padded}",
                        dim = self.s.dim,
                        v = self.bc.v,
                        spaces = spaces,
                        sub_v = self.bc.sub_v,
                        reset = self.s.reset,
                        padded = padded,
                    );
                }

                // ── Render label message lines ──
                // For labels with messages, show them below the annotation line
                // connected by `|`, like rustc's secondary label style.
                let msg_labels: Vec<&(usize, usize, char, &str)> = underlines
                    .iter()
                    .filter(|(_, _, _, msg)| !msg.is_empty())
                    .collect();
                // Group messages by column (if multiple labels have messages at
                // different positions, show each on its own line).
                for (col, _ulen, ch, msg) in &msg_labels {
                    let msg_spaces = " ".repeat(line_num_width + 1);
                    let mut connector = String::new();
                    for _ in 0..*col {
                        connector.push(' ');
                    }
                    // AnnotationKind::Help labels have no underline char (space);
                    // render them as `| help: message` instead of `| message`,
                    // with the `help:` prefix in cyan (matching write_help_line).
                    let display_msg = if *ch == ' ' {
                        format!(
                            "{cyan}help:{reset} {msg}",
                            cyan = self.s.cyan,
                            reset = self.s.reset,
                            msg = msg
                        )
                    } else {
                        (*msg).to_string()
                    };
                    connector.push('|');
                    let _ = writeln!(
                        out,
                        "{v}{msg_spaces}{sub_v} {connector}",
                        v = self.bc.v,
                        msg_spaces = msg_spaces,
                        sub_v = self.bc.sub_v,
                        connector = connector,
                    );
                    let _ = writeln!(
                        out,
                        "{v}{msg_spaces}{sub_v} {connector} {display_msg}",
                        v = self.bc.v,
                        msg_spaces = msg_spaces,
                        sub_v = self.bc.sub_v,
                        connector = connector,
                        display_msg = display_msg,
                    );
                }
            }
        }

        // Explanation box
        // We'll put the primary span's message (if any) as the explanation
        // Otherwise, use the diagnostic level label
        let explanation = all_labels
            .iter()
            .find(|l| matches!(l.kind, AnnotationKind::Primary) && !l.message.is_empty())
            .map(|l| l.message.as_str())
            .unwrap_or("");

        let _ = writeln!(
            out,
            "{dim}{v}{indent}{sub_bl}{sub_h}{reset} {explanation}",
            dim = self.s.dim,
            v = self.bc.v,
            indent = indent,
            sub_bl = self.bc.sub_bl,
            sub_h = self.bc.sub_h,
            reset = self.s.reset,
            explanation = explanation,
        );
    }

    // ── Children ─────────────────────────────────────────────────

    fn write_child(&self, out: &mut String, child: &crate::diagnostics::Subdiag) {
        let prefix = match child.level {
            DiagnosticLevel::Note => "note",
            DiagnosticLevel::Help => "help",
            _ => child.level.label(),
        };
        let _ = writeln!(
            out,
            "{v}  {prefix_color}= {bold}{prefix}{reset}: {msg}",
            v = self.bc.v,
            prefix_color = child.level.ansi_color(),
            bold = self.s.bold,
            prefix = prefix,
            reset = self.s.reset,
            msg = child.message,
        );
    }

    fn write_note_line(&self, out: &mut String, msg: &str) {
        let _ = writeln!(
            out,
            "{v}  {cyan}= {bold}note{reset}: {msg}",
            v = self.bc.v,
            cyan = self.s.cyan,
            bold = self.s.bold,
            reset = self.s.reset,
            msg = msg,
        );
    }

    fn write_help_line(&self, out: &mut String, msg: &str) {
        let _ = writeln!(
            out,
            "{v}  {cyan}= {bold}help{reset}: {msg}",
            v = self.bc.v,
            cyan = self.s.cyan,
            bold = self.s.bold,
            reset = self.s.reset,
            msg = msg,
        );
    }
}

// ── Helper: byte offset → line:col ──────────────────────────────

pub(crate) fn byte_to_linecol(source: &str, byte_offset: usize) -> SourcePos {
    let len = source.len();
    let clamped = std::cmp::min(byte_offset, len);
    let prefix = &source[..clamped];
    let line = prefix.matches('\n').count();
    let start_of_line = prefix.rfind('\n').map(|i| i + 1).unwrap_or(0);
    SourcePos {
        line,
        col: clamped - start_of_line,
    }
}

// ── Helper: get the 0-based line number for a span's start ───────

/// Returns the 0‑based line index of the start of a span, or `None` if
/// the byte offset is out of bounds.
pub(crate) fn span_line(span: Span, source: &str) -> usize {
    let clamped = std::cmp::min(span.start, source.len());
    source[..clamped].matches('\n').count()
}

// ── Helper: compute underlines for a single line from all labels ──

/// Returns a list of `(col, len, underline_char, message)` for all labels
/// that cover the given line index.
pub(crate) fn compute_line_underlines<'a>(
    labels: &'a [Label],
    source: &str,
    line: &str,
    line_idx: usize,
) -> Vec<(usize, usize, char, &'a str)> {
    let line_start = line_start_byte(source, line_idx);
    let line_end = line_start + line.len();
    let mut result = Vec::new();

    for lbl in labels {
        let span_start = lbl.span.start;
        let span_end = lbl.span.end;

        // Does this label's span overlap this line?
        if span_end <= line_start || span_start >= line_end {
            continue;
        }

        let col = if span_start > line_start {
            span_start - line_start
        } else {
            0
        };

        let ulen = if span_end < line_end {
            span_end.saturating_sub(line_start + col)
        } else {
            line_end.saturating_sub(line_start + col)
        };

        result.push((col, ulen, lbl.underline_char(), lbl.message.as_str()));
    }

    result
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

// ── Syntax highlighting ─────────────────────────────────────────

/// Posita keywords to highlight in bold.
const KEYWORDS: &[&str] = &[
    "def", "set", "let", "return", "if", "else", "while", "for", "loop",
    "break", "leave", "continue", "true", "false", "import", "type", "trait",
    "impl", "ensures", "requires", "invariant", "decreases", "match", "with",
    "struct", "enum", "pub", "mut", "ref", "comptime", "extern", "edition",
    "constraint", "where", "in", "is", "as", "and", "or", "not", "fn",
];

/// Posita built-in type names to highlight in cyan.
const TYPES: &[&str] = &[
    "Int", "UInt", "Float", "Bool", "Char", "Byte", "USize", "Str",
    "Unit", "Never", "String",
];

/// Simple syntax highlighter that wraps select tokens in ANSI color codes.
/// Only keywords (bold) and types (cyan) are highlighted — everything else
/// stays in the default terminal color for a clean, readable output.
/// When `use_color` is false, the input is returned unchanged.
pub fn highlight_code(line: &str, use_color: bool) -> String {
    if !use_color {
        return line.to_string();
    }

    // Reset only foreground color and bold — preserve background color
    // so that callers can wrap the output in a background color block.
    let reset = "\x1b[22m\x1b[39m";
    let bold = "\x1b[1m";
    let cyan = "\x1b[36m";

    let mut out = String::with_capacity(line.len() + 32);
    let mut i = 0;
    let bytes = line.as_bytes();
    let len = bytes.len();

    while i < len {
        // Skip string literals entirely — no highlighting inside strings.
        if bytes[i] == b'"' {
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < len { i += 1; }
                i += 1;
            }
            if i < len { i += 1; }
            out.push_str(&line[start..i]);
            continue;
        }
        if bytes[i] == b'\'' {
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'\'' {
                if bytes[i] == b'\\' && i + 1 < len { i += 1; }
                i += 1;
            }
            if i < len { i += 1; }
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
                let _ = std::fmt::write(
                    &mut out,
                    format_args!("{cyan}{}{reset}", word),
                );
                continue;
            }
            if KEYWORDS.contains(&word) {
                let _ = std::fmt::write(
                    &mut out,
                    format_args!("{bold}{}{reset}", word),
                );
                continue;
            }
            out.push_str(word);
            continue;
        }

        // Everything else: pass through as-is.
        out.push(bytes[i] as char);
        i += 1;
    }

    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
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
        .with_span(Span::new(0, 106))
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
            .with_span(Span::new(0, 3))
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
        let renderer = GlyphRenderer::new(false);
        let s = renderer.render_summary(1, 0);
        assert!(s.contains("aborting due to previous error"));
        let s = renderer.render_summary(3, 0);
        assert!(s.contains("aborting due to 3 previous errors"));
    }
}
