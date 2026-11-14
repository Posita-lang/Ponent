use std::borrow::Cow;
use std::fmt::Write;

use super::glyph::visible_width;
use super::renderers::fmt::{dec_digits, push_uint};

use crate::ast::{FileId, Span};
use crate::smallvec::SmallVec;
use crate::diagnostics::i18n::{cached_label_static, fluent};

use super::LineIndex;
use super::term::path_to_file_url;

/// Maximum chain nesting depth rendered/flattened.
///
/// One module-level constant: `any_spanned` (hyperlink gating),
/// `flattened_entry_count` (buffer pre-sizing) and `render_entries` (the
/// actual cap) must agree, or a drift silently disables links for the
/// rendered tail or under-reserves.
const MAX_CHAIN_DEPTH: usize = 64;
/// Upper bounds for the reserve-estimate walk.  Rendering itself is still
/// bounded by `MAX_CHAIN_DEPTH`/the caller, but a hostile *wide* chain (many
/// entries at shallow depth) must not make `String::with_capacity` reserve
/// megabytes before emitting a truncated `...`.
const MAX_RESERVE_ENTRIES: usize = 64 * 1024;
const MAX_RESERVE_MESSAGE_BYTES: usize = 1 << 20;

/// Suffix for one chain entry, stored as pieces so the display width is
/// computable from parts (no `visible_width` scan over the OSC-8-escaped
/// whole) and the bytes can be written straight into the output buffer (no
/// per-entry `CompactString` for `location_str` / `suffix`).
enum ChainSuffix<'a> {
    /// `: L:C` — no file name to show.
    LineCol { line: usize, col: usize },
    /// `: file:L:C`, optionally hyperlinked via an OSC 8 `file_url`.
    FileLineCol {
        /// **Invariant: already sanitized** by `render_into`
        /// ([`super::sanitize_filename`]).  This writer emits the field raw,
        /// so a future constructor must not pass an unsanitized name.
        file: &'a str,
        line: usize,
        col: usize,
        url: Option<&'a str>,
    },
    /// `: kind` — no span to locate.  `&'static str` (not `Cow`) because
    /// [`ChainEntryKind::as_static_str`] returns a borrow into the cached
    /// `OnceLock` table, so there is no allocation to own and no `Cow`
    /// discriminant to carry.
    Kind { kind: &'static str },
}

impl<'a> ChainSuffix<'a> {
    /// Display width in terminal cells.  OSC 8 escapes contribute zero, so
    /// the hyperlinked and plain `FileLineCol` cases share one formula.
    ///
    /// `file_cols` is the precomputed `visible_width(file)`, hoisted by the
    /// caller: every entry of a chain points at the same file, so measuring
    /// it per entry would rescan the name N times.
    fn cols(&self, file_cols: usize) -> usize {
        match self {
            ChainSuffix::LineCol { line, col } => {
                2 + dec_digits((*line).saturating_add(1) as u64)
                    + 1
                    + dec_digits((*col).saturating_add(1) as u64)
            }
            ChainSuffix::FileLineCol { line, col, .. } => {
                2 + file_cols
                    + 1
                    + dec_digits((*line).saturating_add(1) as u64)
                    + 1
                    + dec_digits((*col).saturating_add(1) as u64)
            }
            ChainSuffix::Kind { kind } => 2 + visible_width(kind),
        }
    }

    fn write_into(&self, out: &mut String) {
        match self {
            ChainSuffix::LineCol { line, col } => {
                out.push_str(": ");
                push_uint(out, (*line).saturating_add(1) as u64);
                out.push(':');
                push_uint(out, (*col).saturating_add(1) as u64);
            }
            ChainSuffix::FileLineCol { file, line, col, url } => {
                if let Some(url) = url {
                    // `render_into` already rejected URLs containing ESC or
                    // the ST terminator (an unconditional check, see C1);
                    // this writer only emits a validated URL.
                    out.push_str(": \x1b]8;;");
                    out.push_str(url);
                    out.push_str("\x1b\\");
                    out.push_str(file);
                    out.push(':');
                    push_uint(out, (*line).saturating_add(1) as u64);
                    out.push(':');
                    push_uint(out, (*col).saturating_add(1) as u64);
                    out.push_str("\x1b]8;;\x1b\\");
                } else {
                    out.push_str(": ");
                    out.push_str(file);
                    out.push(':');
                    push_uint(out, (*line).saturating_add(1) as u64);
                    out.push(':');
                    push_uint(out, (*col).saturating_add(1) as u64);
                }
            }
            ChainSuffix::Kind { kind } => {
                out.push_str(": ");
                out.push_str(kind);
            }
        }
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
        Cow::Borrowed(self.as_static_str())
    }

    /// Same label as [`as_str`](Self::as_str) but as a raw `&'static str`,
    /// skipping the `Cow` wrapper.  The underlying `cached_label_static!`
    /// table is the sole cache; `as_str` delegates here so only one
    /// `OnceLock<[String; N]>` is instantiated per `(type, lang)`.
    pub fn as_static_str(&self) -> &'static str {
        // Explicit variant → index mapping: `*self as usize` would silently
        // swap labels if the enum were ever reordered; a match arms the
        // reorder into a non-exhaustive compile error instead.
        let idx = match self {
            ChainEntryKind::Call => 0,
            ChainEntryKind::Inlined => 1,
            ChainEntryKind::Reference => 2,
            ChainEntryKind::Defines => 3,
        };
        cached_label_static! {
            n = 4,
            variant = idx,
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

#[derive(Clone, Copy)]
struct ChainFacts {
    any_spanned: bool,
    entry_count: usize,
    message_bytes: usize,
}

/// Read-only per-chain invariants threaded through the recursive entry walk.
///
/// Computed once in [`CallChain::render_into`] and passed by reference so
/// `render_entries` does not grow a parameter every time the renderer gains a
/// dimension.  Every field is `Copy`, so the body destructures a copy and keeps
/// using the plain names.
#[derive(Clone, Copy)]
struct RenderEnv<'a> {
    use_color: bool,
    source: Option<&'a str>,
    file_name: &'a str,
    file_url: Option<&'a str>,
    idx: Option<&'a LineIndex<'a>>,
    max_width: usize,
    tab_width: usize,
    file_cols: usize,
    anchor: Option<FileId>,
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

    /// Whether any entry (at any depth) carries a span.  Used to gate the
    /// `file://` URL computation so a kind-only chain pays no `canonicalize`.
    ///
    /// Iterative with an explicit worklist and the same depth bound as
    /// [`Self::render_entries`]: the old recursive version ran *first* and
    /// could overflow the stack on a hostile deep chain before the render
    /// cap ever applied.
    /// One traversal for both facts: whether any reachable entry has a span (the
/// hyperlink gate) and how many entries the renderer will emit (the reserve
/// estimate).  Two identical DFS walks previously paid the tree twice.
fn scan(&self) -> ChainFacts {
    let mut any_spanned = false;
    let mut total = 0usize;
    let mut message_bytes = 0usize;
    let mut stack: SmallVec<[(&CallChain<'src>, usize); 8]> = SmallVec::new();
    stack.push((self, 0));
    while let Some((chain, depth)) = stack.pop() {
        if depth > MAX_CHAIN_DEPTH {
            continue;
        }
        total = total
            .saturating_add(chain.entries.len())
            .min(MAX_RESERVE_ENTRIES);
        for entry in &chain.entries {
            any_spanned |= entry.span.is_some();
            message_bytes = message_bytes
                .saturating_add(entry.message.len())
                .min(MAX_RESERVE_MESSAGE_BYTES);
            if let Some(children) = &entry.children {
                for child in children {
                    stack.push((child, depth + 1));
                }
            }
        }
    }
    ChainFacts {
        any_spanned,
        entry_count: total,
        message_bytes,
    }
}

/// Flattened entry count (all depths), capped like the renderer.  Used to
    /// reserve the output buffer up front so a nested chain does not pay
    /// repeated reallocation.  Test-only: production sizing uses `scan()`'s
    /// `ChainFacts` directly.
    #[cfg(test)]
    fn flattened_entry_count(&self) -> usize {
        self.scan().entry_count
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
        // Reserve for the frame plus the per-entry text.  Hyperlinked entries also
// carry an OSC 8 URL (40–150+ bytes) that the old 48-byte estimate ignored,
// forcing a reallocation on exactly the coloured path that needs the cache
// most; 64 covers the common spanned entry without over-reserving.
        // #3: one scan for both the reserve estimate and the hyperlink gate;
        // the renderer used to walk the tree twice per render.
        let facts = self.scan();
        let mut out = String::with_capacity(64 + facts.message_bytes + facts.entry_count * 32);
        self.render_into_with_facts(
            use_color,
            source,
            file_name,
            idx,
            max_width,
            tab_width,
            None,
            None,
            ChainFacts {
                any_spanned: facts.any_spanned,
                entry_count: facts.entry_count,
                message_bytes: facts.message_bytes,
            },
            &mut out,
        );
        out
    }

    /// Render the chain into an existing buffer.
    ///
    /// The human renderer appends the chain to the diagnostic body it already
    /// owns, so writing straight into that buffer avoids the per-chain
    /// `String` + copy that [`Self::render_with_index`] would otherwise do.
    ///
    /// # Same-file assumption
    ///
    /// This renderer threads a single anchor (`file_name`, `source`, and the
    /// optional [`LineIndex`]) through the whole chain and computes the
    /// `(line, col)` of **every** entry span against that one source buffer.
    /// A chain is emitted by the checker for one diagnostic in one file, so
    /// every entry span is an offset into that same file in practice; an
    /// entry whose span belonged to a *different* file would be located
    /// against the anchor's line index and render bogus coordinates.
    /// Foreign-file entries are not filtered out because the anchor's file
    /// identity arrives here only as `file_name: &str` — the [`Span`]s
    /// carry [`FileId`]s, but no `FileId` for the anchor is threaded in, so
    /// there is nothing to compare against without an API change.  Callers
    /// must only hand chains whose spans belong to `source`'s file; an
    /// entry without a span falls back to its [`ChainEntryKind`] label and
    /// is unaffected.
    pub(crate) fn render_into(
        &self,
        use_color: bool,
        source: Option<&str>,
        file_name: &str,
        idx: Option<&LineIndex<'_>>,
        max_width: usize,
        tab_width: usize,
        cx: Option<&super::RenderCtx<'_>>,
        anchor: Option<FileId>,
        out: &mut String,
    ) {
        let facts = self.scan();
        self.render_into_with_facts(
            use_color,
            source,
            file_name,
            idx,
            max_width,
            tab_width,
            cx,
            anchor,
            ChainFacts {
                any_spanned: facts.any_spanned,
                entry_count: facts.entry_count,
                message_bytes: facts.message_bytes,
            },
            out,
        );
    }

    /// Shared body of [`Self::render_into`], taking the already-computed
    /// [`ChainFacts`] so callers that know them avoid a second DFS.
    #[allow(clippy::too_many_arguments)]
    fn render_into_with_facts(
        &self,
        use_color: bool,
        source: Option<&str>,
        file_name: &str,
        idx: Option<&LineIndex<'_>>,
        max_width: usize,
        tab_width: usize,
        cx: Option<&super::RenderCtx<'_>>,
        anchor: Option<FileId>,
        facts: ChainFacts,
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
        let header = fluent::chain_header();
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
        // The URL is consumed only by a spanned entry; a kind-only chain would
        // otherwise pay a `canonicalize` syscall chain for output that never
        // uses it.  Gate on whether any entry anywhere carries a span.
        let hyperlink_enabled = use_color
            && super::term::supports_hyperlinks()
            && super::is_real_file_name(file_name)
            && facts.any_spanned;
        let file_url: Option<std::sync::Arc<str>> = if hyperlink_enabled {
            let candidate = match cx {
                Some(cx) => cx.file_url(file_name),
                None => std::sync::Arc::from(
                    path_to_file_url(std::path::Path::new(file_name)).as_str(),
                ),
            };
            // C1: enforce the OSC 8 invariant unconditionally (not with a
            // debug_assert that disappears in release).  ESC and the ST
            // terminator would close the hyperlink early, but BEL (0x07) is
            // also widely accepted as an OSC terminator and any other C0/DEL
            // byte can inject or garble terminal state — so reject *every*
            // control byte.  A custom `RenderCtx::file_url` can violate the
            // upstream contracts, so degrade to plain text here.  Validating
            // once per chain also removes a per-entry byte scan.
            if candidate.chars().any(char::is_control) {
                None
            } else {
                Some(candidate)
            }
        } else {
            None
        };
        // `file_cols` is invariant across the whole chain — every entry
        // points at the same file, and `visible_width` reads only
        // `file_name` — so compute it once here instead of once per
        // recursion level (an N-level chain used to rescan the name N×).
        let file_cols = visible_width(&sanitized);
        let capped = std::cell::Cell::new(false);
        self.render_entries(
            &RenderEnv {
                use_color,
                source,
                file_name: &sanitized,
                file_url: file_url.as_deref(),
                idx,
                max_width,
                tab_width,
                file_cols,
                anchor,
            },
            0,
            out,
            &capped,
        );
    }

    fn render_entries(
        &self,
        env: &RenderEnv<'_>,
        depth: usize,
        out: &mut String,
        capped: &std::cell::Cell<bool>,
    ) {
        let RenderEnv {
            use_color,
            source,
            file_name,
            file_url,
            idx,
            max_width,
            tab_width,
            file_cols,
            anchor,
        } = *env;
        // Bound the recursion: a pathological/fuzzer-generated chain thousands
        // of levels deep must not overflow the stack.  Real chains are shallow.
        if depth > MAX_CHAIN_DEPTH {
            // One marker per render, not one per truncated branch: a
            // branching hostile chain must not grow output with its fanout.
            if !capped.replace(true) {
                super::renderers::fmt::push_spaces(out, 4 + depth * 4);
                out.push_str("...\n");
            }
            return;
        }
        // Indent width only — the spaces are pushed straight into `out` per
        // row, so a deep chain (depth ≥ 8) no longer allocates a fresh
        // `" ".repeat(..)` per entry (the old `indent_cow` was O(depth²)
        // allocation on pathological chains).
        let indent_cols = 4 + depth * 4;
        let color_on = if use_color { "\x1b[34m" } else { "" };
        let color_off = if use_color { "\x1b[0m" } else { "" };

        for entry in &self.entries {
            // Multiline sanitizer, not the plain one: a call-chain entry may
            // carry layout-intended `\n`/`\t`, and `wrap_optimal` below
            // supports hard newlines — the plain variant destroyed them into
            // U+FFFD before wrapping ever saw them.
            let message = super::sanitize_for_terminal_multiline(&entry.message);
            let suffix = match (source, entry.span, idx) {
                (Some(_), Some(span), Some(idx))
                    if anchor.map_or(true, |a| span.file == a) =>
                {
                    // `render_into` guarantees `source.is_some() => idx.is_some()`,
                    // so this arm covers every located entry; a caller that
                    // bypasses `render_into` with a source but no index falls
                    // through to the kind label rather than rescanning the file
                    // per entry.
                    let (line, col) = idx.linecol_display(span.start_as_usize());
                    if file_name.is_empty() {
                        ChainSuffix::LineCol { line, col }
                    } else {
                        ChainSuffix::FileLineCol {
                            file: file_name,
                            line,
                            col,
                            url: file_url,
                        }
                    }
                }
                _ => ChainSuffix::Kind {
                    kind: entry.kind.as_static_str(),
                },
            };
            let suffix_cols = suffix.cols(file_cols);
            // Wrap to the full measure first (the suffix only rides the last
            // line).  Only if that last line plus the suffix would overflow do
            // we re-wrap the final segment narrower — using the reduced width
            // for *every* line would waste 12–20 columns on each non-final
            // line of a long message.
            let avail_full = max_width.saturating_sub(indent_cols).max(1);
            // C3: dominant-case fast path — a short printable-ASCII message
            // that fits on one line needs no `wrap_optimal` and no `Vec`.
            // The ASCII gate is load-bearing: it makes the multiline
            // sanitizer the identity (one display cell per byte) and excludes
            // `\t`/`\n`, so byte length == display width.
            if super::ascii::is_printable_ascii(message.as_bytes())
                && !message.is_empty()
                // The `is_printable_ascii` gate above proves the message has
                // no tab/newline and no byte outside 0x20..=0x7E, so the only
                // possible whitespace is `' '`.
                // #1: the slow `wrap_optimal` trims edge spaces; a short
                // `"  foo  "` must render identically, so the fast path
                // declines when an edge space is present (byte checks are
                // exact for the printable-ASCII gate above).
                && !message.starts_with(' ')
                && !message.ends_with(' ')
                && message.len() + indent_cols + suffix_cols <= max_width
            {
                super::renderers::fmt::push_spaces(out, indent_cols);
                out.push_str(color_on);
                out.push_str(message.as_ref());
                out.push_str(color_off);
                suffix.write_into(out);
                out.push('\n');
            } else {
                // Wrap to the full measure first (the suffix only rides the
                // last line).  Only if that last line plus the suffix would
                // overflow do we re-wrap the final segment narrower — using
                // the reduced width for *every* line would waste 12–20
                // columns on each non-final line of a long message.
                // `last_text` is declared before `segs` so the re-wrapped
                // borrowed segments remain valid for the writes below (C4:
                // one allocation instead of one per segment).
                let last_text: String;
                let mut segs = super::wrap::wrap_optimal(&message, avail_full, tab_width);
                let narrow = avail_full.saturating_sub(suffix_cols).max(1);
                if narrow < avail_full
                    && let Some(last_seg) = segs.last()
                    && super::glyph::visible_width_with_tab(last_seg.as_ref(), tab_width) + suffix_cols
                        > avail_full
                {
                    last_text = segs.pop().unwrap().into_owned();
                    let reparsed = super::wrap::wrap_optimal(&last_text, narrow, tab_width);
                    segs.extend(reparsed);
                }
                // `wrap_optimal` always yields ≥ 1 segment for non-empty
                // input, but a message that sanitizes to whitespace yields
                // exactly one empty segment (`""`); `saturating_sub` keeps the
                // suffix logic correct (and debug builds panic-free) even if
                // an empty list ever escapes.
                let last = segs.len().saturating_sub(1);
                for (i, seg) in segs.iter().enumerate() {
                    super::renderers::fmt::push_spaces(out, indent_cols);
                    out.push_str(color_on);
                    out.push_str(seg.as_ref());
                    out.push_str(color_off);
                    if i == last {
                        suffix.write_into(out);
                    }
                    out.push('\n');
                }
            }
            if let Some(children) = &entry.children {
                for child in children.iter() {
                    child.render_entries(env, depth + 1, out, capped);
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

    /// C6: a span from a different file is not an offset into the anchor
    /// source; it must render its kind label, never the anchor's coordinates.
    #[test]
    fn foreign_file_span_falls_back_to_kind_label() {
        let foreign = FileId::from_raw(FileId::FIRST_REAL);
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(foreign, 0, 3)),
            "from another file",
        ));
        let src = "def main() {}";
        let idx = LineIndex::new(src);
        let mut out = String::new();
        chain.render_into(
            false,
            Some(src),
            "file.ps",
            Some(&idx),
            100,
            4,
            None,
            Some(FileId::DUMMY),
            &mut out,
        );
        assert!(
            out.contains("called from here"),
            "expected kind label, got: {out}"
        );
        assert!(
            !out.contains("file.ps:1:1"),
            "foreign span rendered anchor coordinates: {out}"
        );
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
    fn deep_chain_is_capped_consistently() {
        // Build 70 nested levels: the renderer/flatteners cap at
        // `MAX_CHAIN_DEPTH` (64) and must agree, emit exactly one `...`
        // marker, and never panic.
        let span = Span::new(FileId::DUMMY, 0, 3);
        let mut inner = CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), "x")],
        };
        // Single-character messages: at depth ≥ 64 the indent exceeds the
        // measure, and a multi-character word would be split one grapheme per
        // line (correct, but it makes the line-count cap assertion useless).
        for _ in 0..70 {
            let entry = ChainEntry::new(ChainEntryKind::Call, Some(span), "x")
                .with_children(vec![inner]);
            inner = CallChain {
                entries: vec![entry],
            };
        }
        let out = inner.render(false, Some("def main() {}"), "f.ps");
        assert_eq!(
            out.matches("...").count(),
            1,
            "exactly one cap marker expected: {out}"
        );
        assert!(out.lines().count() <= MAX_CHAIN_DEPTH + 4, "cap not applied");
        // The counting traversal must not under-reserve past the cap either:
        // `flattened_entry_count` stops at the same depth as the renderer.
        assert!(inner.flattened_entry_count() >= MAX_CHAIN_DEPTH);
    }

    #[test]
    fn branching_deep_chain_emits_one_marker() {
        let span = Span::new(FileId::DUMMY, 0, 3);
        let leaf = || CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), "x")],
        };
        // Two identical linear branches that both cross the cap.  The old
        // recursion emitted one `...` per branch; the contract is one marker
        // per render.
        let deep = || {
            let mut chain = leaf();
            for _ in 0..=MAX_CHAIN_DEPTH {
                chain = CallChain {
                    entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), "x")
                        .with_children(vec![chain])],
                };
            }
            chain
        };
        let branch = CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), "root")
                .with_children(vec![deep(), deep()])],
        };
        let out = branch.render(false, Some("def main() {}"), "f.ps");
        assert_eq!(
            out.matches("...").count(),
            1,
            "branching cap must emit exactly one marker: {out}"
        );
        assert!(
            out.lines().count() <= 2 * MAX_CHAIN_DEPTH + 8,
            "branching cap grew output: {out}"
        );
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

    /// C7: the width formula (`cols`) and the writer (`write_into`) are
    /// hand-synchronised duplicates; pin them together so a future edit to
    /// one cannot silently desynchronise wrapping from rendering.
    #[test]
    fn suffix_cols_matches_rendered_visible_width() {
        fn strip_osc8(s: &str) -> String {
            let mut out = String::new();
            let mut rest = s;
            while let Some(p) = rest.find("\x1b]8;;") {
                out.push_str(&rest[..p]);
                rest = &rest[p..];
                match rest.find("\x1b\\") {
                    Some(e) => rest = &rest[e + 2..],
                    None => break,
                }
            }
            out.push_str(rest);
            out
        }
        for file in ["a.rs", "src/very/long/name.rs", ""] {
            for line in [0usize, 9, 99, 1234] {
                for col in [0usize, 4, 42, 999] {
                    for url in [None, Some("file:///tmp/a.rs")] {
                        let s = ChainSuffix::FileLineCol {
                            file,
                            line,
                            col,
                            url,
                        };
                        let mut out = String::new();
                        s.write_into(&mut out);
                        let visible = strip_osc8(&out);
                        assert_eq!(
                            visible_width(&visible),
                            s.cols(visible_width(file)),
                            "suffix {out:?}"
                        );
                    }
                    let s = ChainSuffix::LineCol { line, col };
                    let mut out = String::new();
                    s.write_into(&mut out);
                    assert_eq!(
                        visible_width(&out),
                        s.cols(0),
                        "line-col suffix {out:?}"
                    );
                }
            }
        }
        for kind in [ChainEntryKind::Call, ChainEntryKind::Inlined] {
            let s = ChainSuffix::Kind {
                kind: kind.as_static_str(),
            };
            let mut out = String::new();
            s.write_into(&mut out);
            assert_eq!(visible_width(&out), s.cols(0));
        }
    }
}
