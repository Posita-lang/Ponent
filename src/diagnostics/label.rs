use crate::ast::Span;

/// Kind of annotation underline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationKind {
    /// Solid underline `^^^` — the primary error location.
    Primary,
    /// Tilde underline `~~~` — a secondary related location.
    Secondary,
    /// Dashed underline `---` — a note-level location.
    Note,
}

/// A labeled annotation pointing to a span of source code.
#[derive(Debug, Clone)]
pub struct Label {
    pub span: Span,
    pub message: String,
    pub kind: AnnotationKind,
}

impl Label {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Primary,
        }
    }

    pub fn secondary(span: Span, message: impl Into<String>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Secondary,
        }
    }

    pub fn note(span: Span, message: impl Into<String>) -> Self {
        Label {
            span,
            message: message.into(),
            kind: AnnotationKind::Note,
        }
    }

    pub fn with_kind(mut self, kind: AnnotationKind) -> Self {
        self.kind = kind;
        self
    }

    /// The underline character for this annotation kind.
    pub fn underline_char(&self) -> char {
        match self.kind {
            AnnotationKind::Primary => '^',
            AnnotationKind::Secondary => '~',
            AnnotationKind::Note => '-',
        }
    }
}

// ── Byte-offset → line:column conversion ──────────────────────

/// A (line, column) position in source code.
/// `line` and `col` are 0-based internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePos {
    /// 0-based line index.
    pub line: usize,
    /// 0-based byte offset from the start of the line.
    pub col: usize,
}

/// Convert a byte-offset span into a 0-based (line, col) pair.
fn byte_to_linecol(source: &str, byte_offset: usize) -> SourcePos {
    let len = source.len();
    let clamped = std::cmp::min(byte_offset, len);
    let prefix = &source[..clamped];
    // Count newlines — each one starts a new line
    let line = prefix.matches('\n').count();
    // Find the byte just after the last newline (== start of our line)
    let start_of_line = prefix.rfind('\n').map(|i| i + 1).unwrap_or(0);
    SourcePos {
        line,
        col: clamped - start_of_line,
    }
}

/// The byte offset of the start of a given 0-based line.
fn line_start_byte(source: &str, target_line: usize) -> usize {
    let mut line = 0usize;
    for (i, ch) in source.char_indices() {
        if line == target_line {
            return i;
        }
        if ch == '\n' {
            line += 1;
        }
    }
    source.len()
}

/// All lines of a source text, split with their start byte offsets.
struct LineTable {
    lines: Vec<(usize, String)>, // (1-based line number, content)
    offsets: Vec<usize>,         // byte offset of each 0-based line's start
}

impl LineTable {
    fn build(source: &str) -> Self {
        let mut offsets = Vec::new();
        let mut strings = Vec::new();
        offsets.push(0);
        let mut line_start = 0;
        for (i, ch) in source.char_indices() {
            if ch == '\n' {
                strings.push(source[line_start..i].to_string());
                line_start = i + 1;
                offsets.push(line_start);
            }
        }
        // Last line (may be empty if source ends with \n)
        if line_start <= source.len() {
            strings.push(source[line_start..].to_string());
        }
        let numbered: Vec<(usize, String)> = strings
            .into_iter()
            .enumerate()
            .map(|(i, s)| (i + 1, s))
            .collect();
        LineTable {
            lines: numbered,
            offsets,
        }
    }

    fn byte_to_line(&self, byte_offset: usize) -> Option<usize> {
        // Binary search for the line that contains this byte
        let idx = match self.offsets.binary_search(&byte_offset) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        if idx < self.lines.len() {
            Some(self.lines[idx].0) // 1-based line number
        } else {
            None
        }
    }

    fn line_count(&self) -> usize {
        self.lines.len()
    }

    fn get_line(&self, line_1based: usize) -> Option<&str> {
        if line_1based == 0 || line_1based > self.lines.len() {
            return None;
        }
        Some(self.lines[line_1based - 1].1.as_str())
    }
}

// ── Source context extraction ─────────────────────────────────

/// Renders source code snippets with `^`-underline annotations.
///
/// Format (Rust-inspired):
///   --> file.ps:5:12
///    |
///  5 |     return x + 1;
///    |            ^ expected Bool, found Int<32>
///
/// Secondary labels use `~~~` and notes use `---`.
pub struct SourceContext {
    table: LineTable,
    source: String,
    filename: String,
    ctx_lines: usize,
}

impl SourceContext {
    /// Build source context over the entire source.
    /// `context_lines` controls how many lines of surrounding context
    /// are shown around each annotated line.
    pub fn new(
        source: &str,
        _span: Span,
        filename: &str,
        context_lines: usize,
    ) -> Self {
        // `_span` is reserved for future use (scoping the context to a
        // specific region); currently the entire source is always loaded.
        SourceContext {
            table: LineTable::build(source),
            source: source.to_string(),
            filename: filename.to_string(),
            ctx_lines: context_lines,
        }
    }

    /// Collect the set of 1-based line numbers that should be rendered,
    /// given a primary span and secondary labels.
    fn collect_lines(&self, span: Span, labels: &[Label]) -> Vec<usize> {
        let mut lines_set: std::collections::BTreeSet<usize> =
            std::collections::BTreeSet::new();

        // Primary span lines
        let start_lc = byte_to_linecol(&self.source, span.start);
        let end_lc = byte_to_linecol(&self.source, span.end);
        for l in start_lc.line..=end_lc.line {
            lines_set.insert(l + 1);
        }

        // Secondary labels — include ALL lines each label covers
        for lbl in labels {
            let lc = byte_to_linecol(&self.source, lbl.span.start);
            let lc_end = byte_to_linecol(&self.source, lbl.span.end);
            for l in lc.line..=lc_end.line {
                lines_set.insert(l + 1);
            }
        }

        // Add context lines around each annotated line
        let all_annotated: Vec<usize> = lines_set.iter().copied().collect();
        let total = self.table.line_count();
        for &l in &all_annotated {
            let l0 = if l > self.ctx_lines { l - self.ctx_lines } else { 1 };
            let l1 = std::cmp::min(l + self.ctx_lines, total);
            for ll in l0..=l1 {
                lines_set.insert(ll);
            }
        }

        lines_set.into_iter().collect()
    }

    /// Render the source context with annotations.
    ///
    /// `primary_span` is the main error location, `labels` are additional
    /// annotations (secondary/note).  Both are rendered as underlines.
    pub fn render(
        &self,
        primary_span: Span,
        labels: &[Label],
        use_color: bool,
    ) -> String {
        let lines_to_render = self.collect_lines(primary_span, labels);
        if lines_to_render.is_empty() {
            return String::new();
        }

        let mut out = String::new();

        // Determine annotation style per line: build the set of lines
        // that should show underlines.
        let primary_lc = byte_to_linecol(&self.source, primary_span.start);
        let primary_end_lc = byte_to_linecol(&self.source, primary_span.end);

        // Group labels by line — include ALL lines each label covers
        let mut labels_by_line: std::collections::BTreeMap<usize, Vec<&Label>> =
            std::collections::BTreeMap::new();
        for lbl in labels {
            let lc = byte_to_linecol(&self.source, lbl.span.start);
            let lc_end = byte_to_linecol(&self.source, lbl.span.end);
            for line in (lc.line + 1)..=(lc_end.line + 1) {
                labels_by_line
                    .entry(line)
                    .or_default()
                    .push(lbl);
            }
        }

        let reset = "\x1b[0m";
        let blue = "\x1b[34m";
        let cyan = "\x1b[36m";

        for &line_1based in &lines_to_render {
            let line_str = match self.table.get_line(line_1based) {
                Some(s) => s,
                None => continue,
            };

            // ── Line number + content ──────────────────────
            let line_prefix = format!("{:>4} | ", line_1based);
            if use_color {
                out.push_str(&format!("{}{}{}", blue, line_prefix, reset));
            } else {
                out.push_str(&line_prefix);
            }
            out.push_str(line_str);
            out.push('\n');

            // ── Annotation underline ───────────────────────
            let is_primary_line = line_1based >= primary_lc.line + 1
                && line_1based <= primary_end_lc.line + 1;
            let line_labels = labels_by_line.get(&line_1based);

            if is_primary_line || line_labels.is_some() {
                let annot_prefix = if use_color {
                    format!("{}{}{}", blue, "     | ", reset)
                } else {
                    "     | ".to_string()
                };
                out.push_str(&annot_prefix);

                // Build the annotation line character by character
                // Start with all spaces
                let mut chars: Vec<char> = std::iter::repeat(' ')
                    .take(line_str.len().max(1))
                    .collect();

                // Primary span underline on its line
                if is_primary_line {
                    // Calculate column on this specific line
                    let lc_start = if line_1based == primary_lc.line + 1 {
                        primary_lc.col
                    } else {
                        0
                    };
                    let lc_end = if line_1based == primary_end_lc.line + 1 {
                        primary_end_lc.col
                    } else {
                        line_str.len()
                    };
                    let start = std::cmp::min(lc_start, line_str.len());
                    let end = std::cmp::min(lc_end, line_str.len());
                    for c in chars.iter_mut().take(end).skip(start) {
                        *c = '^';
                    }
                }

                // Secondary labels on this line
                if let Some(lbls) = line_labels {
                    for lbl in lbls {
                        let lc = byte_to_linecol(&self.source, lbl.span.start);
                        let lc_end = byte_to_linecol(&self.source, lbl.span.end);
                        let ch = lbl.underline_char();
                        // For multi-line labels, compute the underline range
                        // on this specific line:
                        //   - first line:    lc.col .. (end of line if span
                        //                     continues, else lc_end.col)
                        //   - middle lines:  0 .. end of line
                        //   - last line:     0 .. lc_end.col
                        let start_col = if line_1based == lc.line + 1 {
                            lc.col
                        } else {
                            0
                        };
                        let end_col = if line_1based == lc_end.line + 1 {
                            lc_end.col
                        } else {
                            line_str.len()
                        };
                        let start = std::cmp::min(start_col, line_str.len());
                        let end = std::cmp::min(end_col, line_str.len());
                        let end = std::cmp::max(end, start + 1);
                        for c in chars.iter_mut().take(end).skip(start) {
                            *c = ch;
                        }
                    }
                }

                // Trim trailing spaces
                let trimmed: String = chars.iter().collect();
                let trimmed = trimmed.trim_end().to_string();
                out.push_str(&trimmed);
                out.push('\n');

                // ── Label messages — only on the label's first line ──
                if let Some(lbls) = line_labels {
                    for lbl in lbls {
                        if !lbl.message.is_empty()
                            && line_1based
                                == byte_to_linecol(&self.source, lbl.span.start).line + 1
                        {
                            let msg_prefix = if use_color {
                                format!("{}{}{}", blue, "     | ", reset)
                            } else {
                                "     | ".to_string()
                            };
                            out.push_str(&msg_prefix);
                            if use_color {
                                out.push_str(&format!(
                                    "{}= {}{}\n",
                                    cyan, lbl.message, reset
                                ));
                            } else {
                                out.push_str(&format!("= {}\n", lbl.message));
                            }
                        }
                    }
                }
            }
        }

        out
    }
}

/// Convert byte offset to human-readable "line:col" string (1-based).
pub fn byte_offset_to_string(source: &str, byte_offset: usize) -> String {
    let pos = byte_to_linecol(source, byte_offset);
    format!("{}:{}", pos.line + 1, pos.col + 1)
}
