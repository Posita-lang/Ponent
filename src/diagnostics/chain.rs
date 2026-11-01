use std::borrow::Cow;
use std::fmt::Write;

use compact_str::{CompactString, format_compact};

use crate::ast::{FileId, Span};
use crate::diagnostics::i18n::{cached_label, fluent};

use super::LineIndex;
use super::term::path_to_file_url;

/// Indentation for a chain entry at `depth`.  Depths 0–3 (the overwhelmingly
/// common case) borrow a `'static` slice instead of allocating; deeper nesting
/// falls back to an owned `" ".repeat(..)`.
fn indent_cow(depth: usize) -> Cow<'static, str> {
    const INDENTS: &[&str] = &["    ", "        ", "            ", "                "];
    match INDENTS.get(depth) {
        Some(s) => Cow::Borrowed(*s),
        None => Cow::Owned(" ".repeat(4 + depth * 4)),
    }
}

/// A single entry in a call/reference chain (Zig-style `referenced by`).
#[derive(Debug, Clone)]
pub struct ChainEntry<'src> {
    pub kind: ChainEntryKind,
    /// Source location of this entry, or `None` when the caller has no
    /// position information.  Using `Option` avoids the `(0, 0)` sentinel,
    /// which could not be distinguished from a real zero-length span at the
    /// start of a file.
    pub span: Option<Span>,
    pub message: Cow<'src, str>,
    /// Nested entries below this one.  `None` for the common flat-chain
    /// case: `Option<Box<[T]>>` is 16 bytes (the slice pointer's niche is
    /// the discriminant), versus 24 for an always-empty `Vec`.
    pub children: Option<Box<[CallChain<'src>]>>,
}

impl<'src> ChainEntry<'src> {
    pub fn new(
        kind: ChainEntryKind,
        span: Option<Span>,
        message: impl Into<Cow<'src, str>>,
    ) -> Self {
        ChainEntry {
            kind,
            span,
            message: message.into(),
            children: None,
        }
    }

    pub fn with_children(mut self, children: Vec<CallChain<'src>>) -> Self {
        self.children = (!children.is_empty()).then(|| children.into_boxed_slice());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainEntryKind {
    /// Direct function call.
    Call,
    /// Inlined call.
    Inlined,
    /// Reference to a declaration.
    Reference,
    /// Defines a symbol.
    Defines,
}

impl ChainEntryKind {
    /// Human-readable label for this entry kind, resolved through Fluent so
    /// it follows the active [`Lang`](super::i18n::Lang).
    pub fn as_str(&self) -> Cow<'static, str> {
        cached_label! {
            n = 4,
            variant = *self as usize,
            init = [
                fluent::chain_call().into_owned(),
                fluent::chain_inlined().into_owned(),
                fluent::chain_reference().into_owned(),
                fluent::chain_defines().into_owned(),
            ],
        }
    }
}

/// A chain of locations tracing how the compiler reached an error.
/// Inspired by Zig's `referenced by` output.
///
/// Example output:
///   referenced by:
///       main: file.ps:8:12
///       callMain: std/start.zig:698:59
#[derive(Debug, Clone)]
pub struct CallChain<'src> {
    entries: Vec<ChainEntry<'src>>,
}

impl<'src> CallChain<'src> {
    pub fn new() -> Self {
        CallChain {
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, entry: ChainEntry<'src>) {
        self.entries.push(entry);
    }

    pub fn push_call(span: Span, message: impl Into<Cow<'src, str>>) -> Self {
        CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), message)],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[ChainEntry<'src>] {
        &self.entries
    }

    /// Render the call chain into a multi-line string.
    ///
    /// When `source` is provided, entries carrying a span are rendered with
    /// their `file:line:col` location (matching the example above); otherwise
    /// the entry's [`ChainEntryKind`] label is shown instead.
    pub fn render(&self, use_color: bool, source: Option<&str>, file_name: &str) -> String {
        self.render_with_index(
            use_color,
            source,
            file_name,
            None,
            super::DEFAULT_MAX_WIDTH,
            super::glyph::DEFAULT_TAB_WIDTH,
        )
    }

    /// Like [`Self::render`], but reuses a caller-provided line index for
    /// `source` (the emitter's per-batch cache) instead of rescanning the file
    /// once per entry via `byte_to_linecol`, and wraps long entry messages to
    /// `max_width` display columns using `tab_width`-column tab stops.
    pub(crate) fn render_with_index(
        &self,
        use_color: bool,
        source: Option<&str>,
        file_name: &str,
        idx: Option<&LineIndex<'_>>,
        max_width: usize,
        tab_width: usize,
    ) -> String {
        if self.entries.is_empty() {
            return String::new();
        }
        let mut out = String::with_capacity(64 + self.entries.len() * 48);
        self.render_into(
            use_color, source, file_name, idx, max_width, tab_width, None, &mut out,
        );
        out
    }

    /// Render the chain into an existing buffer.
    ///
    /// The human renderer appends the chain to the diagnostic body it already
    /// owns, so writing straight into that buffer avoids the per-chain
    /// `String` + copy that [`Self::render_with_index`] would otherwise do.
    pub(crate) fn render_into(
        &self,
        use_color: bool,
        source: Option<&str>,
        file_name: &str,
        idx: Option<&LineIndex<'_>>,
        max_width: usize,
        tab_width: usize,
        cx: Option<&super::RenderCtx<'_>>,
        out: &mut String,
    ) {
        if self.entries.is_empty() {
            return;
        }
        // Build the line index once for the whole chain when the caller did
        // not supply one (the public `render` path).  Every entry then pays
        // O(log n) instead of rescanning the source per entry.
        let owned_index;
        let idx = if let (Some(src), None) = (source, idx) {
            owned_index = LineIndex::new(src);
            Some(&owned_index)
        } else {
            idx
        };
        let header = "referenced by:";
        if use_color {
            let _ = writeln!(out, "\x1b[36m{}\x1b[0m", header);
        } else {
            let _ = writeln!(out, "{header}");
        }
        // Hoist the per-file invariants out of the recursion: `sanitize`,
        // `supports_hyperlinks`, and `canonicalize` depend only on
        // `file_name` and the process environment, neither of which changes
        // across recursion levels — only `depth` varies.  Computing them
        // once turns an N-level chain from N syscalls + N env probes into 1.
        let sanitized = super::sanitize_filename(file_name);
        let hyperlink_enabled =
            use_color && super::term::supports_hyperlinks() && super::is_real_file_name(file_name);
        let file_url: Option<std::sync::Arc<str>> = if hyperlink_enabled {
            match cx {
                Some(cx) => Some(cx.file_url(file_name)),
                None => Some(std::sync::Arc::from(
                    path_to_file_url(std::path::Path::new(file_name)).as_str(),
                )),
            }
        } else {
            None
        };
        self.render_entries(
            use_color,
            source,
            &sanitized,
            hyperlink_enabled,
            file_url.as_deref(),
            idx,
            max_width,
            tab_width,
            0,
            out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn render_entries(
        &self,
        use_color: bool,
        source: Option<&str>,
        file_name: &str,
        hyperlink_enabled: bool,
        file_url: Option<&str>,
        idx: Option<&LineIndex<'_>>,
        max_width: usize,
        tab_width: usize,
        depth: usize,
        out: &mut String,
    ) {
        let indent_str = indent_cow(depth);
        let color_on = if use_color { "\x1b[34m" } else { "" };
        let color_off = if use_color { "\x1b[0m" } else { "" };

        for entry in &self.entries {
            let message = super::sanitize_for_terminal(&entry.message);
            let suffix: CompactString = match (source, entry.span) {
                (Some(src), Some(span)) => {
                    // `render_into` supplies an index whenever `source` is
                    // present, so the fallback is only reachable through a
                    // caller that bypasses it; it builds one rather than
                    // rescanning the file per entry.
                    let (line, col) = match idx {
                        Some(idx) => idx.linecol_display(span.start_as_usize()),
                        None => LineIndex::new(src).linecol_display(span.start_as_usize()),
                    };
                    if file_name.is_empty() {
                        format_compact!(": {}:{}", line + 1, col + 1)
                    } else {
                        let location_str = format_compact!(
                            "{file}:{line}:{col}",
                            file = file_name,
                            line = line + 1,
                            col = col + 1,
                        );
                        if let Some(file_url) = file_url {
                            format_compact!(
                                ": \x1b]8;;{file_url}\x1b\\{location_str}\x1b]8;;\x1b\\"
                            )
                        } else {
                            format_compact!(": {location_str}")
                        }
                    }
                }
                _ => format_compact!(": {}", entry.kind.as_str()),
            };
            let suffix_cols = super::glyph::visible_width(&suffix);
            let avail = max_width
                .saturating_sub(indent_str.len())
                .saturating_sub(suffix_cols)
                .max(1);
            let segs = super::wrap::wrap_optimal(&message, avail, tab_width);
            let last = segs.len() - 1;
            for (i, seg) in segs.iter().enumerate() {
                out.push_str(&indent_str);
                out.push_str(color_on);
                out.push_str(seg.as_ref());
                out.push_str(color_off);
                if i == last {
                    out.push_str(&suffix);
                }
                out.push('\n');
            }
            if let Some(children) = &entry.children {
                for child in children.iter() {
                    child.render_entries(
                        use_color,
                        source,
                        file_name,
                        hyperlink_enabled,
                        file_url,
                        idx,
                        max_width,
                        tab_width,
                        depth + 1,
                        out,
                    );
                }
            }
        }
    }
}

impl<'src> Default for CallChain<'src> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    #[test]
    fn test_render_includes_location_with_source() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 0, 3)),
            "main",
        ));
        let out = chain.render(false, Some("def main() {}"), "file.ps");
        assert!(out.contains("main: file.ps:1:1"), "out: {out}");

        // Without source, fall back to the kind label.
        let out = chain.render(false, None, "");
        assert!(out.contains("called from here"), "out: {out}");
    }

    #[test]
    fn zero_length_span_at_offset_zero_still_renders_location() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 0, 0)),
            "main",
        ));
        let out = chain.render(false, Some("def main() {}"), "file.ps");
        assert!(out.contains("file.ps:1:1"), "expected location, got: {out}");
        assert!(
            !out.contains("called from here"),
            "fell back to kind: {out}"
        );
    }

    #[test]
    fn no_span_falls_back_to_kind_label() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(ChainEntryKind::Call, None, "main"));
        let out = chain.render(false, Some("def main() {}"), "file.ps");
        assert!(out.contains("called from here"), "out: {out}");
    }

    /// `message` and `file_name` are external text written between SGR colour
    /// codes; a control character must not survive to inject terminal escapes.
    #[test]
    fn message_and_file_name_are_sanitized() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 0, 3)),
            "evil\x1b]8;;x\x07",
        ));

        // Source + file_name path: the header open/close (2 ESC) and the
        // message colour open/close (2 ESC) are always present; the OSC 8
        // hyperlink open/ST/close/ST (4 ESC) appears only when the terminal
        // advertises support, so the total tracks `supports_hyperlinks()`
        // rather than assuming a fixed count.
        let out = chain.render(true, Some("def main() {}"), "f\x1b]8;;y\x07.ps");
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced: {out:?}"
        );
        let hyperlink_esc = if crate::diagnostics::term::supports_hyperlinks() {
            4
        } else {
            0
        };
        assert_eq!(
            out.matches('\x1b').count(),
            4 + hyperlink_esc,
            "extra ESC in {out:?}"
        );

        // Kind-label branch (no source) must sanitize as well.
        let out = chain.render(true, None, "");
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert_eq!(out.matches('\x1b').count(), 4, "extra ESC in {out:?}");
    }

    /// A message longer than `max_width` wraps, and the `: location` suffix
    /// stays on the last wrapped line.
    #[test]
    fn long_message_wraps_with_suffix_on_last_line() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 0, 3)),
            "this is a very long function name that definitely exceeds the width",
        ));
        // max_width=40 → indent 4 + suffix ~12 → ~24 cols for the message.
        let out = chain.render_with_index(false, Some("def main() {}"), "f.ps", None, 40, 4);
        let lines: Vec<&str> = out.lines().collect();
        // First line is the header "referenced by:".
        assert_eq!(lines[0], "referenced by:");
        // The last entry line must contain the location suffix.
        let last = lines.last().unwrap();
        assert!(
            last.contains(": f.ps:1:1"),
            "suffix missing on last line: {out}"
        );
        // No single entry line should exceed max_width.
        for line in &lines[1..] {
            assert!(line.len() <= 50, "line too wide (max_width=40): {line:?}");
        }
        // The message must wrap into more than one line.
        assert!(lines.len() > 2, "message did not wrap: {out}");
    }

    #[test]
    fn nested_chain_renders_with_indented_children() {
        let leaf = CallChain {
            entries: vec![ChainEntry::new(
                ChainEntryKind::Call,
                Some(Span::new(FileId::DUMMY, 0, 3)),
                "leaf",
            )],
        };
        let parent = ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 0, 3)),
            "parent",
        )
        .with_children(vec![leaf]);
        let chain = CallChain {
            entries: vec![parent],
        };
        let out = chain.render(false, Some("def main() {}"), "f.ps");
        let parent_line = out.lines().find(|l| l.contains("parent")).unwrap();
        let leaf_line = out.lines().find(|l| l.contains("leaf")).unwrap();
        assert_eq!(
            parent_line.len() - parent_line.trim_start().len(),
            4,
            "parent must be at depth 0 (4 columns)"
        );
        assert_eq!(
            leaf_line.len() - leaf_line.trim_start().len(),
            8,
            "child must be indented one more level (8 columns): {out}"
        );
    }
}
