use crate::ast::FileId;
use super::LineIndex;
use crate::ast::Span;
use std::borrow::Cow;
use std::fmt::Write;

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
        match self {
            ChainEntryKind::Call => crate::tr!("chain-call"),
            ChainEntryKind::Inlined => crate::tr!("chain-inlined"),
            ChainEntryKind::Reference => crate::tr!("chain-reference"),
            ChainEntryKind::Defines => crate::tr!("chain-defines"),
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
            entries: vec![ChainEntry {
                kind: ChainEntryKind::Call,
                span: Some(span),
                message: message.into(),
            }],
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
        let header = "referenced by:";
        if use_color {
            let _ = writeln!(out, "\x1b[36m{}\x1b[0m", header);
        } else {
            let _ = writeln!(out, "{header}");
        }

        // `message` and `file_name` are external text: a control character
        // would terminate the SGR colour run early and let the rest of the
        // string act as terminal commands (forged links, title changes, …).
        // Sanitize before interpolating into the escape sequences below.
        // Keep the raw path for the `file://` hyperlink (percent-encoded
        // separately); the display copy is sanitized.
        let file_name_raw = file_name;
        let file_name = super::sanitize_filename(file_name);
        let hyperlink_enabled =
            use_color && super::term::supports_hyperlinks() && !file_name_raw.is_empty();

        // The OSC 8 hyperlink target is the same path for every entry, so
        // canonicalize it once here instead of issuing a syscall per chain
        // element.  `None` when this diagnostic's location is not linked.
        let file_url: Option<String> = if hyperlink_enabled
            && !file_name.is_empty()
            && file_name_raw != "<input>"
        {
            let abs_path = std::path::Path::new(file_name_raw)
                .canonicalize()
                .unwrap_or_else(|_| std::path::PathBuf::from(file_name_raw));
            Some(format!(
                "file://{}",
                super::glyph::encode_file_url_path(&abs_path.to_string_lossy())
            ))
        } else {
            None
        };

        for entry in &self.entries {
            let message = super::sanitize_for_terminal(&entry.message);

            // Build the suffix (`: location` or `: kind_label`) that follows
            // the message on the last wrapped line.
            let suffix: String = match (source, entry.span) {
                (Some(src), Some(span)) => {
                    let pos = match idx {
                        Some(idx) => idx.to_linecol(span.start_as_usize()),
                        None => super::byte_to_linecol(src, span.start_as_usize()),
                    };
                    if file_name.is_empty() {
                        format!(": {}:{}", pos.line + 1, pos.col + 1)
                    } else {
                        let location_str = format!(
                            "{file}:{line}:{col}",
                            file = file_name,
                            line = pos.line + 1,
                            col = pos.col + 1,
                        );
                        if let Some(file_url) = &file_url {
                            format!(": \x1b]8;;{file_url}\x1b\\{location_str}\x1b]8;;\x1b\\")
                        } else {
                            format!(": {location_str}")
                        }
                    }
                }
                _ => format!(": {}", entry.kind.as_str()),
            };

            // Available width for the message: total minus the 4-column indent
            // minus the suffix's visible width.  Clamped to 1 so wrapping
            // always makes forward progress.
            let suffix_cols = super::glyph::visible_width(&suffix);
            let avail = max_width.saturating_sub(4).saturating_sub(suffix_cols).max(1);
            let segs = super::wrap::wrap_optimal(&message, avail, tab_width);

            let color_on = if use_color { "\x1b[34m" } else { "" };
            let color_off = if use_color { "\x1b[0m" } else { "" };

            for (i, seg) in segs.iter().enumerate() {
                if i == segs.len() - 1 {
                    let _ = writeln!(out, "    {color_on}{seg}{color_off}{suffix}");
                } else {
                    let _ = writeln!(out, "    {color_on}{seg}{color_off}");
                }
            }
        }

        out
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
    use crate::ast::Span;

    #[test]
    fn test_render_includes_location_with_source() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry {
            kind: ChainEntryKind::Call,
            span: Some(Span::new(FileId::DUMMY, 0, 3)),
            message: "main".into(),
        });
        let out = chain.render(false, Some("def main() {}"), "file.ps");
        assert!(out.contains("main: file.ps:1:1"), "out: {out}");

        // Without source, fall back to the kind label.
        let out = chain.render(false, None, "");
        assert!(out.contains("called from here"), "out: {out}");
    }

    #[test]
    fn zero_length_span_at_offset_zero_still_renders_location() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry {
            kind: ChainEntryKind::Call,
            span: Some(Span::new(FileId::DUMMY, 0, 0)),
            message: "main".into(),
        });
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
        chain.push(ChainEntry {
            kind: ChainEntryKind::Call,
            span: None,
            message: "main".into(),
        });
        let out = chain.render(false, Some("def main() {}"), "file.ps");
        assert!(out.contains("called from here"), "out: {out}");
    }

    /// `message` and `file_name` are external text written between SGR colour
    /// codes; a control character must not survive to inject terminal escapes.
    #[test]
    fn message_and_file_name_are_sanitized() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry {
            kind: ChainEntryKind::Call,
            span: Some(Span::new(FileId::DUMMY, 0, 3)),
            message: "evil\x1b]8;;x\x07".into(),
        });

        // Source + file_name path: the header open/close (2 ESC), the
        // message colour open/close (2 ESC), and the OSC 8 hyperlink
        // open/ST/close/ST (4 ESC) are the only escapes allowed.
        let out = chain.render(true, Some("def main() {}"), "f\x1b]8;;y\x07.ps");
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced: {out:?}"
        );
        assert_eq!(out.matches('\x1b').count(), 8, "extra ESC in {out:?}");

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
        chain.push(ChainEntry {
            kind: ChainEntryKind::Call,
            span: Some(Span::new(FileId::DUMMY, 0, 3)),
            message: "this is a very long function name that definitely exceeds the width".into(),
        });
        // max_width=40 → indent 4 + suffix ~12 → ~24 cols for the message.
        let out = chain.render_with_index(
            false,
            Some("def main() {}"),
            "f.ps",
            None,
            40,
            4,
        );
        let lines: Vec<&str> = out.lines().collect();
        // First line is the header "referenced by:".
        assert_eq!(lines[0], "referenced by:");
        // The last entry line must contain the location suffix.
        let last = lines.last().unwrap();
        assert!(last.contains(": f.ps:1:1"), "suffix missing on last line: {out}");
        // No single entry line should exceed max_width.
        for line in &lines[1..] {
            assert!(
                line.len() <= 50,
                "line too wide (max_width=40): {line:?}"
            );
        }
        // The message must wrap into more than one line.
        assert!(lines.len() > 2, "message did not wrap: {out}");
    }
}
