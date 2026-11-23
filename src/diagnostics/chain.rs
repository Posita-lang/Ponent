use std::borrow::Cow;
use std::fmt::Write;

use super::glyph::visible_width;
use super::renderers::fmt::{dec_digits, push_uint};

use crate::ast::{FileId, Span};
use crate::diagnostics::i18n::{cached_label_static, fluent};
use crate::smallvec::SmallVec;

use super::LineIndex;
use super::term::path_to_file_url;

/// Maximum rendered depth *offset*: frames at depth 0..=64 are emitted
/// (65 nesting levels).  One module-level constant: `ChainFacts::spanned_files`
/// / `ChainFacts::has_span_in` (hyperlink gating), `scan()` (buffer pre-sizing
/// — `flattened_entry_count` is test-only) and `render_entries` (the actual
/// cap) must agree, or a drift silently disables links for the rendered tail
/// or under-reserves.
const MAX_CHAIN_DEPTH: usize = 64;
/// Upper bounds for the reserve-estimate walk.  Rendering itself is capped by
/// *both* this constant and `MAX_CHAIN_DEPTH` (`render_entries` stops at
/// `MAX_RESERVE_ENTRIES` rendered entries and emits a single `...`), so the
/// scan and the renderer must agree on it exactly as they must on
/// `MAX_CHAIN_DEPTH`; retuning one without the other desynchronises the reserve
/// from the output.  The point of the cap is that a hostile *wide* chain (many
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
    Kind { kind: &'static str, width: usize },
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
            ChainSuffix::Kind { width, .. } => 2 + width,
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
            ChainSuffix::FileLineCol {
                file,
                line,
                col,
                url,
            } => {
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
            ChainSuffix::Kind { kind, .. } => {
                out.push_str(": ");
                out.push_str(kind);
            }
        }
    }
}

/// A single entry in a call/reference chain (Zig-style `referenced by`).
///
/// `Debug` is hand-written ([below](ChainEntry)) so it cannot recurse through
/// `children`; only `Clone` is derived.
pub struct ChainEntry<'src> {
    pub(crate) kind: ChainEntryKind,
    /// Source location of this entry, or `None` when the caller has no
    /// position information.  Using `Option` avoids the `(0, 0)` sentinel,
    /// which could not be distinguished from a real zero-length span at the
    /// start of a file.
    pub(crate) span: Option<Span>,
    pub(crate) message: Cow<'src, str>,
    /// Nested entries below this one.  `None` for the common flat-chain
    /// case: `Option<Box<[T]>>` is 16 bytes (the slice pointer's niche is
    /// the discriminant), versus 24 for an always-empty `Vec`.
    pub(crate) children: Option<Box<[CallChain<'src>]>>,
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

impl<'src> ChainEntry<'src> {
    /// Copy every field except `children`, which the caller attaches.
    ///
    /// The worklist in `CallChain::clone` builds each node shallowly and then
    /// attaches the already-built child chains, which is what keeps cloning a
    /// hostile 10^5-deep chain off the stack (the same reason `Drop` and `Debug`
    /// are hand-written here).
    fn clone_shallow(&self) -> Self {
        ChainEntry {
            kind: self.kind,
            span: self.span,
            message: self.message.clone(),
            children: None,
        }
    }
}

impl<'src> Clone for ChainEntry<'src> {
    fn clone(&self) -> Self {
        // Children go through `CallChain::clone`, which is iterative, so this
        // stays bounded no matter how deep the subtree is.
        let mut out = self.clone_shallow();
        if let Some(children) = &self.children {
            out.children = Some(children.iter().map(CallChain::clone).collect());
        }
        out
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
    const fn index(self) -> usize {
        match self {
            ChainEntryKind::Call => 0,
            ChainEntryKind::Inlined => 1,
            ChainEntryKind::Reference => 2,
            ChainEntryKind::Defines => 3,
        }
    }

    pub fn as_static_str(&self) -> &'static str {
        let idx = self.index();
        cached_label_static! {
            n = 4,
            variant = idx,
            slugs = ["chain-call", "chain-inlined", "chain-reference", "chain-defines"],
        }
    }

    /// Display width of [`Self::as_static_str`] in the **current** language.
    /// Computing it at the suffix construction site (rather than caching a
    /// lang-unaware `[usize; 4]`) makes `width` and `as_static_str` observe the
    /// same language, so `cols()` cannot under/over-count after `Lang::set`.
    ///
    /// Test-only: `render_into_with_facts` builds its own `kind_labels`
    /// table, so production code never calls this.
    #[cfg(test)]
    pub fn label_width(&self) -> usize {
        visible_width(self.as_static_str())
    }
}

/// A chain of locations tracing how the compiler reached an error.
/// Inspired by Zig's `referenced by` output.
///
/// Example output:
///   referenced by:
///       main: file.ps:8:12
///       callMain: std/start.zig:698:59
pub struct CallChain<'src> {
    entries: Vec<ChainEntry<'src>>,
    /// Per-chain memo for [`Self::scan`]; invalidated by [`Self::push`].
    facts: std::sync::OnceLock<ChainFacts>,
}

impl<'src> Clone for CallChain<'src> {
    /// Iterative deep clone.
    ///
    /// `#[derive(Clone)]` here was the last *structurally recursive* path on a
    /// chain: `Clone` is reachable through the public `with_children`, and a
    /// hostile deep chain would overflow the stack through clone glue exactly as
    /// it did through drop glue before `Drop` was hand-written.  Two passes:
    ///
    /// 1. preorder worklist — build every node shallowly, recording for each
    ///    entry the arena indices of its child chains;
    /// 2. reverse index order — attach children, which is safe because every
    ///    child's index is greater than its parent's and its own children were
    ///    therefore already attached.
    ///
    /// Cost is O(nodes + edges) heap, no recursion, no `unsafe`.
    fn clone(&self) -> Self {
        let mut arena: Vec<Option<CallChain<'src>>> = vec![None];
        // Per node: (entry index, child node indices).
        let mut kids: Vec<Vec<(usize, Vec<usize>)>> = vec![Vec::new()];
        let mut stack: Vec<(&CallChain<'src>, usize)> = vec![(self, 0)];
        while let Some((chain, idx)) = stack.pop() {
            arena[idx] = Some(CallChain {
                entries: chain
                    .entries
                    .iter()
                    .map(ChainEntry::clone_shallow)
                    .collect(),
                facts: std::sync::OnceLock::new(),
            });
            for (entry_idx, entry) in chain.entries.iter().enumerate() {
                let Some(children) = &entry.children else {
                    continue;
                };
                let mut ids = Vec::with_capacity(children.len());
                for child in children.iter() {
                    let child_idx = arena.len();
                    arena.push(None);
                    kids.push(Vec::new());
                    ids.push(child_idx);
                    stack.push((child, child_idx));
                }
                kids[idx].push((entry_idx, ids));
            }
        }
        for node in (0..arena.len()).rev() {
            for (entry_idx, ids) in std::mem::take(&mut kids[node]) {
                let built: Vec<CallChain<'src>> = ids
                    .into_iter()
                    .map(|child_idx| {
                        arena[child_idx]
                            .take()
                            .expect("child chain is built before its parent is attached")
                    })
                    .collect();
                let chain = arena[node].as_mut().expect("node built in pass 1");
                chain.entries[entry_idx].children =
                    (!built.is_empty()).then(|| built.into_boxed_slice());
            }
        }
        arena.swap_remove(0).expect("root chain built in pass 1")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChainFacts {
    /// Up to two distinct files that carry a rendered entry span, pre-order.
    ///
    /// Two cover the overwhelming case (the anchor plus one foreign frame) at
    /// no allocation and without giving up `Copy`; any further distinct file
    /// sets [`Self::spanned_overflow`], which keeps the gate conservative
    /// rather than wrong.  This replaces a single `any_spanned: bool`, which
    /// over-approximated: the line index handed to `render_entries` is the
    /// *anchor* file's, so a chain whose only spanned entries pointed at other
    /// files enabled hyperlink construction and paid an O(source) index build
    /// no entry could use.
    spanned_files: [Option<FileId>; 2],
    /// More than two distinct spanned files were seen; every anchor is then
    /// conservatively treated as spanned.
    spanned_overflow: bool,
    /// `FileId` of the first spanned entry in pre-order, within the same depth
    /// and entry caps the renderer applies.  Folded in here so the anchor
    /// derivation costs no second traversal *and* cannot name an entry the
    /// renderer would never emit (the old `first_spanned_file` walk was
    /// uncapped, so on a hostile chain it could disagree with the render).
    first_file: Option<FileId>,
    entry_count: usize,
    message_bytes: usize,
    indent_bytes: usize,
}

impl ChainFacts {
    /// Whether any rendered entry's span lives in `anchor` (or, with
    /// `anchor == None`, in any file at all — the legacy same-file assumption).
    fn has_span_in(&self, anchor: Option<FileId>) -> bool {
        let any = self.spanned_files[0].is_some();
        if self.spanned_overflow {
            return any;
        }
        match anchor {
            Some(file) => self.spanned_files.iter().flatten().any(|f| *f == file),
            None => any,
        }
    }
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
    /// `(label, display width)` for each [`ChainEntryKind`], resolved once per
    /// render.  The language is fixed for the whole call, so re-resolving the
    /// Fluent label and rescanning its width per entry was pure repetition —
    /// the same consistency argument `label_width` already makes.
    kind_labels: [(&'static str, usize); 4],
}

/// Iterative teardown.  The derived drop glue recurses one stack frame per
/// nesting level, so a hostile 100k-deep chain (trivially built iteratively
/// with `with_children`) would overflow the stack on scope exit — precisely
/// where the render caps claim to defend.  Move every nested `entries` vector
/// onto one worklist and leave each child chain empty before its own glue
/// runs, so the compiler's field-drop never walks a nested chain.
///
/// `Clone` remains the derived, structurally recursive implementation: it is
/// only reachable by an explicit clone of a hostile chain (not by any drop
/// path), and an iterative tree-clone would need a bottom-up rebuild for no
/// additional safety on the paths this type actually takes.  Callers that
/// duplicate untrusted chains must bound their depth (the same
/// `MAX_CHAIN_DEPTH` contract the renderer enforces).
impl Drop for CallChain<'_> {
    fn drop(&mut self) {
        let mut work: Vec<ChainEntry<'_>> = std::mem::take(&mut self.entries);
        while let Some(mut entry) = work.pop() {
            if let Some(children) = entry.children.take() {
                for mut child in Vec::from(children) {
                    // Move the child's entries onto the shared worklist; the
                    // child is then dropped with an empty `entries`, so its
                    // own `Drop` invocation cannot recurse.
                    work.append(&mut child.entries);
                }
            }
        }
    }
}

/// How many top-level entries [`Debug`] prints before eliding the rest.
///
/// `Debug` output is bounded for the same reason rendering is: a hostile or
/// fuzzer-built chain must not turn a `format!("{:?}")` into megabytes of text.
const DEBUG_ENTRIES: usize = 8;

/// Depth-limited `Debug` (the derive would recurse one frame per nesting
/// level).
///
/// The iterative [`Drop`] above exists because a 100k-deep chain overflows the
/// stack on teardown; a derived `Debug` would reopen the same hazard through
/// every `unwrap` panic message, `assert_eq!` failure and tracing field on a
/// chain.  Nested chains are summarised by their entry count instead of being
/// printed, so this implementation is O(top-level entries) regardless of depth.
impl std::fmt::Debug for CallChain<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CallChain {{ entries: {}, head: [", self.entries.len())?;
        for (i, entry) in self.entries.iter().take(DEBUG_ENTRIES).enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{entry:?}")?;
        }
        if self.entries.len() > DEBUG_ENTRIES {
            write!(f, ", … {} more", self.entries.len() - DEBUG_ENTRIES)?;
        }
        f.write_str("] }")
    }
}

/// See [`CallChain`]'s `Debug`: `children` is a count, never a recursive walk,
/// and the message is length-capped (see [`DebugMessage`]).
impl std::fmt::Debug for ChainEntry<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainEntry")
            .field("kind", &self.kind)
            .field("span", &self.span)
            .field("message", &DebugMessage(self.message.as_ref()))
            .field(
                "nested_chains",
                &self.children.as_ref().map_or(0, |c| c.len()),
            )
            .finish()
    }
}

/// Characters of a chain message printed by `Debug` before eliding the rest.
///
/// `DEBUG_ENTRIES` bounds the number of entries; this bounds their *size*.  A
/// hostile or fuzzer-built chain can carry a multi-megabyte message, and `Debug`
/// is reachable from every `unwrap` panic, `assert_eq!` failure and tracing
/// field — the very hazard the depth limit exists for, on the other axis.
const DEBUG_MESSAGE_CHARS: usize = 64;

struct DebugMessage<'a>(&'a str);

impl std::fmt::Debug for DebugMessage<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.0;
        match s.char_indices().nth(DEBUG_MESSAGE_CHARS) {
            // Short enough: quote it exactly (escaping is the `str` impl's job).
            None => std::fmt::Debug::fmt(s, f),
            Some((cut, _)) => write!(f, "{:?}… (+{} bytes)", &s[..cut], s.len() - cut),
        }
    }
}

/// Percent-encode every byte `>= 0x80` so an OSC 8 payload stays a valid URL.
///
/// [`path_to_file_url`](super::term::path_to_file_url) already emits ASCII (it
/// encodes non-ASCII path bytes itself), so this only fires for a custom
/// [`RenderCtx::file_url`](super::RenderCtx::file_url) that hands back raw
/// UTF-8.  Dropping the hyperlink there loses the link for a legitimate
/// non-ASCII filename; encoding keeps it.
fn percent_encode_high_bytes(s: &str) -> String {
    // `super::HEX_UPPER`, not a private copy: the same table already lives in
    // `diagnostics` and a second home is exactly how the two drift.
    // Exact capacity: every byte >= 0x80 becomes three bytes (`%XX`).  The
    // sibling `encode_url_path_segment` uses the same worst-case reasoning;
    // reserving `len + 8` reallocated ~log2(n) times on a fully non-ASCII URL.
    let high = s.bytes().filter(|b| *b >= 0x80).count();
    let mut out = String::with_capacity(s.len() + 2 * high);
    for &b in s.as_bytes() {
        if b >= 0x80 {
            out.push('%');
            out.push(super::HEX_UPPER[(b >> 4) as usize] as char);
            out.push(super::HEX_UPPER[(b & 0x0F) as usize] as char);
        } else {
            // ASCII only: `as char` is the identity on this branch.
            out.push(b as char);
        }
    }
    out
}

/// Everything a call-chain render needs besides the chain itself.
///
/// Replaces nine positional parameters.  Two pairs shared a type
/// (`source`/`file_name` and `max_width`/`tab_width`), so a swapped pair
/// compiled cleanly and produced subtly wrong output — a transposition hazard,
/// not a style nit.  Named fields make every call site auditable.
#[derive(Clone, Copy)]
pub(crate) struct ChainRenderCtx<'a, 'src> {
    pub(crate) use_color: bool,
    pub(crate) source: Option<&'a str>,
    pub(crate) file_name: &'a str,
    pub(crate) idx: Option<&'a LineIndex<'src>>,
    pub(crate) max_width: usize,
    pub(crate) tab_width: usize,
    pub(crate) cx: Option<&'a super::RenderCtx<'src>>,
    pub(crate) anchor: Option<FileId>,
}

impl<'src> CallChain<'src> {
    pub fn new() -> Self {
        CallChain {
            entries: Vec::new(),
            facts: std::sync::OnceLock::new(),
        }
    }

    pub fn push(&mut self, entry: ChainEntry<'src>) {
        self.entries.push(entry);
        self.facts = std::sync::OnceLock::new();
    }

    pub fn push_call(span: Span, message: impl Into<Cow<'src, str>>) -> Self {
        CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), message)],
            facts: std::sync::OnceLock::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[ChainEntry<'src>] {
        &self.entries
    }

    /// One traversal for both facts: whether any reachable entry has a span
    /// (the hyperlink gate) and how many entries the renderer will emit (the
    /// reserve estimate).  Two identical DFS walks previously paid the tree
    /// twice.
    ///
    /// Iterative with an explicit worklist and the same depth bound as
    /// [`Self::render_entries`]: the old recursive version ran *first* and
    /// could overflow the stack on a hostile deep chain before the render
    /// cap ever applied.
    fn scan(&self) -> ChainFacts {
        // A value computed by this very call cannot be stale; only a
        // *pre-existing* cache entry can be.  Recomputing in both cases paid
        // double facts cost in debug builds for half the coverage.
        #[cfg(debug_assertions)]
        let had_cached = self.facts.get().is_some();
        let facts = *self.facts.get_or_init(|| self.compute_facts());
        // Memo canary: the soundness of this cache rests on three unstated
        // preconditions (`entries` stays private, `children` is immutable, and
        // `ChainEntry`'s public fields cannot be mutated after `push`).  If a
        // future mutation path is added without invalidating the memo,
        // `any_spanned` would silently gate hyperlinks wrongly and the reserve
        // estimate would go stale; recompute in debug builds so the first
        // render fails loudly instead.
        #[cfg(debug_assertions)]
        if had_cached {
            let fresh = self.compute_facts();
            debug_assert_eq!(
                facts, fresh,
                "stale chain facts: a mutation path bypassed CallChain::push"
            );
        }
        facts
    }

    fn compute_facts(&self) -> ChainFacts {
        let mut spanned_files: [Option<FileId>; 2] = [None; 2];
        let mut spanned_overflow = false;
        let mut first_file: Option<FileId> = None;
        let mut message_bytes = 0usize;
        let mut indent_bytes = 0usize;
        let mut visited = 0usize;
        // Frame = (chain, depth, next child index).  Processing one entry,
        // then its children, then the remaining siblings reproduces
        // `render_entries`'s pre-order exactly — the old LIFO push of all
        // children visited the last sibling's subtree first, so the cap set
        // (and therefore `any_spanned` / the hyperlink gate) could differ
        // from the rendered set at the truncation boundary.
        let mut stack: SmallVec<[(&CallChain<'src>, usize, usize); 8]> = SmallVec::new();
        stack.push((self, 0, 0));
        while let Some((chain, depth, next)) = stack.pop() {
            if depth > MAX_CHAIN_DEPTH || next >= chain.entries.len() {
                continue;
            }
            if visited >= MAX_RESERVE_ENTRIES {
                break;
            }
            let entry = &chain.entries[next];
            // Siblings are processed after this entry's entire subtree.
            if next + 1 < chain.entries.len() {
                stack.push((chain, depth, next + 1));
            }
            visited += 1;
            let indent = 4 + depth * 4;
            indent_bytes = indent_bytes
                .saturating_add(indent)
                .min(MAX_RESERVE_MESSAGE_BYTES);
            // The renderer pushes `indent` spaces on EVERY row, not once per
            // entry: a wrapped or multiline message renders `1 + newlines`
            // (plus wrapped) rows, so the old per-entry term under-reserved by
            // the whole indent-per-row factor — a 1 MiB message at a 40-column
            // measure reserved ~8 bytes for ~200 KB of actual spaces.  Hard
            // newlines are counted exactly (memchr is SIMD); wrapped rows are
            // estimated from message length at ~64 bytes/row, which is
            // render-parameter independent (a memo precondition) and monotone.
            let newlines = memchr::memchr_iter(b'\n', entry.message.as_bytes()).count();
            let extra_rows = newlines.saturating_add(entry.message.len() / 64);
            if extra_rows > 0 {
                indent_bytes = indent_bytes
                    .saturating_add(indent.saturating_mul(extra_rows))
                    .min(MAX_RESERVE_MESSAGE_BYTES);
            }
            if let Some(span) = entry.span {
                if !spanned_files.iter().flatten().any(|f| *f == span.file) {
                    match spanned_files.iter_mut().find(|slot| slot.is_none()) {
                        Some(slot) => *slot = Some(span.file),
                        // Third distinct file: stop tracking exactly and let
                        // `has_span_in` stay conservative.
                        None => spanned_overflow = true,
                    }
                }
                // Pre-order: the first spanned entry encountered is the anchor.
                first_file = first_file.or(Some(span.file));
            }
            message_bytes = message_bytes
                .saturating_add(entry.message.len())
                .min(MAX_RESERVE_MESSAGE_BYTES);
            if let Some(children) = &entry.children {
                for child in children.iter().rev() {
                    stack.push((child, depth + 1, 0));
                }
            }
        }
        // The renderer consults `has_span_in(facts.first_file)` to decide whether
        // it needs an index, so `first_file` must be one of the tracked files
        // unless the tracking overflowed.  Implicit until now; the debug canary
        // costs nothing in release.
        debug_assert!(
            first_file.is_none()
                || spanned_overflow
                || spanned_files[0] == first_file
                || spanned_files[1] == first_file,
            "first_file must be among the tracked (anchor-or-foreign) files"
        );
        ChainFacts {
            spanned_files,
            spanned_overflow,
            first_file,
            entry_count: visited,
            message_bytes,
            indent_bytes,
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
    ///
    /// # Same-file assumption
    ///
    /// Every spanned entry must belong to `source`'s file; foreign-file spans
    /// would be located against the wrong line index.  Use
    /// [`Self::render_into`] with an explicit anchor for mixed chains.
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
    ///
    /// Inherits [`Self::render`]'s same-file assumption.
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
        // Reserve for the frame plus the per-entry text.  Hyperlinked entries
        // also carry an OSC 8 URL (40–150+ bytes); `message_bytes` is a lower
        // bound (control bytes expand to U+FFFD, tabs to spaces), so 64 per
        // entry is a deliberate over-estimate to avoid reallocation on the
        // coloured path.
        let facts = self.scan();
        let mut out = String::with_capacity(
            64 + facts.message_bytes + facts.indent_bytes + facts.entry_count * 64,
        );
        self.render_into_with_facts(
            use_color,
            source,
            file_name,
            idx,
            max_width,
            tab_width,
            None,
            // Derive the anchor instead of assuming every span belongs to
            // `file_name`: a foreign span then falls back to its kind label
            // rather than printing this file's name with another file's
            // line/column (the legacy `anchor = None` hazard).  `facts` already
            // carries the pre-order first spanned file, computed with the same
            // caps as the renderer, so the anchor costs no second walk and can
            // never name an entry the render truncates away.
            facts.first_file,
            facts,
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
    /// See [`ChainRenderCtx`] for the parameter bundle (and why it is a struct).
    pub(crate) fn render_into(&self, ctx: &ChainRenderCtx<'_, '_>, out: &mut String) {
        // Destructured immediately: the body keeps its short local names while
        // callers get named fields and cannot transpose `source`/`file_name` or
        // `max_width`/`tab_width`.
        let ChainRenderCtx {
            use_color,
            source,
            file_name,
            idx,
            max_width,
            tab_width,
            cx,
            anchor,
        } = *ctx;
        self.render_into_parts(
            use_color, source, file_name, idx, max_width, tab_width, cx, anchor, out,
        )
    }

    /// # Same-file assumption
    ///
    /// A chain is emitted for one diagnostic in one file, so every entry span is
    /// normally an offset into that same file.  `anchor = Some(file)` makes
    /// `render_entries` filter an entry whose `span.file` differs, falling
    /// back to the kind label instead of rendering bogus coordinates against
    /// the anchor's line index.  `anchor = None` retains the legacy
    /// same-file assumption; `render`/`render_with_index` pass `None`.
    /// An entry without a span always falls back to its [`ChainEntryKind`]
    /// label and is unaffected.
    fn render_into_parts(
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
        // An empty chain renders nothing, so skip the facts DFS (whose debug
        // canary recomputes it) entirely.
        if self.entries.is_empty() {
            return;
        }
        let facts = self.scan();
        self.render_into_with_facts(
            use_color, source, file_name, idx, max_width, tab_width, cx, anchor, facts, out,
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
        // Reserve from the facts walk (its whole purpose): `message_bytes`,
        // `indent_bytes` and a per-entry allowance for the suffix/gutter.  A
        // no-op when capacity already suffices, so small chains pay nothing, and
        // a large chain stops growing organically through many reallocations.
        out.reserve(
            64 + facts.message_bytes + facts.indent_bytes + facts.entry_count.saturating_mul(64),
        );
        if self.entries.is_empty() {
            return;
        }
        // Build the line index once for the whole chain when the caller did
        // not supply one (the public `render` path).  Every entry then pays
        // O(log n) instead of rescanning the source per entry.
        let owned_index;
        // The index is read only by the `FileLineCol` suffix arm, and only for
        // an entry whose span lives in the *anchor* file (the index is the
        // anchor's source).  With no such entry the O(source) scan is pure waste
        // — the same argument that already gates `sanitize_filename`,
        // `path_to_file_url` and `file_cols` below.
        let idx = if let (Some(src), None) = (source, idx) {
            if facts.has_span_in(anchor) {
                owned_index = LineIndex::new(src);
                Some(&owned_index)
            } else {
                None
            }
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
        //
        // A chain with no *located* entry never renders `file_name` (every
        // suffix falls back to its kind label), so the sanitize allocation and
        // the grapheme-width scan are dead there.
        let located = source.is_some() && facts.has_span_in(anchor);
        let sanitized = if located {
            super::sanitize_filename(file_name)
        } else {
            Cow::Borrowed(file_name)
        };
        // The URL is consumed only by a spanned entry; a kind-only chain would
        // otherwise pay a `canonicalize` syscall chain for output that never
        // uses it.  Gate on whether any entry anywhere carries a span.
        // The URL is consumed only by a spanned entry whose suffix renders
        // `FileLineCol`; with no source every entry falls back to its kind
        // label, so paying `path_to_file_url` (canonicalize syscalls) plus the
        // byte-validation scan there is pure waste.
        // OSC 8 is zero-width and independent of SGR: a `--color=never` run on a
        // hyperlink-capable terminal used to lose every clickable location
        // because the click target was gated on the palette.  Capability probing
        // is `supports_hyperlinks()`; colour is a separate question.
        let hyperlink_enabled = source.is_some()
            && super::term::supports_hyperlinks()
            && super::is_real_file_name(file_name)
            && facts.has_span_in(anchor);
        let file_url: Option<std::sync::Arc<str>> = if hyperlink_enabled {
            let candidate = match cx {
                Some(cx) => cx.file_url(file_name),
                None => path_to_file_url(std::path::Path::new(file_name)),
            };
            // C1: enforce the OSC 8 invariant unconditionally (not with a
            // debug_assert that disappears in release).  ESC and the ST
            // terminator would close the hyperlink early, but BEL (0x07) is
            // also widely accepted as an OSC terminator and any other C0/DEL
            // byte can inject or garble terminal state — so reject *every*
            // control byte.  A custom `RenderCtx::file_url` can violate the
            // upstream contracts, so degrade to plain text here.  Validating
            // once per chain also removes a per-entry byte scan.
            if candidate.bytes().any(|b| b.is_ascii_control()) {
                None
            } else if candidate.is_ascii() {
                Some(candidate)
            } else {
                // High bytes are not dangerous, only non-conforming: terminals
                // accept raw UTF-8 in OSC 8, but a URL must be ASCII (RFC
                // 3986).  Encoding keeps the link for a non-ASCII filename
                // where the old blanket rejection silently dropped it, and the
                // payload is zero-width so `cols()` is unaffected.  One
                // allocation per chain, only on this path.
                Some(std::sync::Arc::from(
                    percent_encode_high_bytes(&candidate).as_str(),
                ))
            }
        } else {
            None
        };
        // `file_cols` is invariant across the whole chain — every entry
        // points at the same file, and `visible_width` reads only
        // `file_name` — so compute it once here instead of once per
        // recursion level (an N-level chain used to rescan the name N×).
        // Only a `FileLineCol` suffix reads it; a kind-only chain leaves the
        // scan out entirely (`ChainSuffix::Kind` carries its own width).
        let file_cols = if located {
            visible_width(&sanitized)
        } else {
            0
        };
        let capped = std::cell::Cell::new(false);
        let rendered = std::cell::Cell::new(0usize);
        // Resolve the four kind labels once: the language cannot change inside
        // one render, and `label_width` would otherwise re-resolve the Fluent
        // label and rescan its graphemes for every entry.
        let kind_labels = [
            ChainEntryKind::Call,
            ChainEntryKind::Inlined,
            ChainEntryKind::Reference,
            ChainEntryKind::Defines,
        ]
        .map(|k| {
            let label = k.as_static_str();
            (label, super::glyph::visible_width(label))
        });
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
                kind_labels,
            },
            0,
            out,
            &capped,
            &rendered,
        );
    }

    /// Capped variant for the fast-path guard: `None` means the measured
    /// width already exceeds `cap`, so the caller must take the wrap path.
    fn ascii_tabbed_width_capped(
        s: &str,
        start_col: usize,
        tab_width: usize,
        cap: usize,
    ) -> Option<usize> {
        let tab_width = tab_width.max(1);
        let mut col = start_col;
        for &b in s.as_bytes() {
            if b == b'\t' {
                col += tab_width - (col % tab_width);
            } else {
                col += 1;
            }
            if col.saturating_sub(start_col) > cap {
                return None;
            }
        }
        Some(col - start_col)
    }

    fn expand_message_tabs<'a>(
        message: &'a str,
        tab_width: usize,
        start_col: usize,
    ) -> Cow<'a, str> {
        let tab_width = tab_width.max(1);
        if start_col % tab_width == 0 || !message.contains('\t') {
            return Cow::Borrowed(message);
        }
        // Each tab expands to at most `tab_width - 1` spaces *instead of* its
        // single byte, so `len + tabs * (tab_width - 1)` is an exact upper
        // bound.  The old `len + 8` reallocated on every tab-dense message —
        // precisely the input this function exists for.  `memchr` is SIMD.
        let tabs = memchr::memchr_iter(b'\t', message.as_bytes()).count();
        let extra = tabs.saturating_mul(tab_width - 1);
        let mut out = String::with_capacity(message.len() + extra);
        for (i, line) in message.split('\n').enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(super::glyph::expand_tabs_from_col(line, tab_width, start_col).as_ref());
        }
        Cow::Owned(out)
    }

    fn render_entries(
        &self,
        env: &RenderEnv<'_>,
        depth: usize,
        out: &mut String,
        capped: &std::cell::Cell<bool>,
        rendered: &std::cell::Cell<usize>,
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
            kind_labels,
        } = *env;
        // Bound the recursion: a pathological/fuzzer-generated chain thousands
        // of levels deep must not overflow the stack.  Real chains are shallow.
        if depth > MAX_CHAIN_DEPTH {
            // One marker per render, not one per truncated branch: a
            // branching hostile chain must not grow output with its fanout.
            if !capped.replace(true) {
                // At the first over-cap depth `4 + depth * 4` is already
                // ~264 columns, so an unclamped marker row is emitted far past
                // the terminal width.  Keep it inside the measure instead: the
                // marker stands for the truncated tail, it does not need the
                // truncated branch's (unbounded) indent.
                let marker_indent = (4 + depth * 4).min(max_width.saturating_sub(4));
                super::renderers::fmt::push_spaces(out, marker_indent);
                out.push_str("...\n");
            }
            return;
        }
        // Indent width only — the spaces are pushed straight into `out` per
        // row, so a deep chain (depth ≥ 8) no longer allocates a fresh
        // `" ".repeat(..)` per entry (the old `indent_cow` was O(depth²)
        // allocation on pathological chains).
        let indent_cols = 4 + depth * 4;
        let tab_width = tab_width.max(1);
        let tab_aligned = indent_cols % tab_width == 0;
        let avail_full = max_width.saturating_sub(indent_cols).max(1);
        let color_on = if use_color { "\x1b[34m" } else { "" };
        let color_off = if use_color { "\x1b[0m" } else { "" };

        for entry in &self.entries {
            if rendered.get() >= MAX_RESERVE_ENTRIES {
                if !capped.replace(true) {
                    // Same clamp as the depth marker: `indent_cols` grows with
                    // depth, and the entry-cap branch can be reached at any
                    // depth, so an unclamped marker row would overrun the
                    // terminal width on a 40-column measure.
                    let marker_indent = indent_cols.min(max_width.saturating_sub(4));
                    super::renderers::fmt::push_spaces(out, marker_indent);
                    out.push_str("...\n");
                }
                return;
            }
            rendered.set(rendered.get() + 1);
            // Test-only mirror of the renderer's visit count, so a differential
            // test can assert `ChainFacts::entry_count == rendered entries`
            // (the reserve estimate and the renderer are independent walks).
            // Same gate as the helper's module (`chain::tests`): under the
            // concurrency checker the module is compiled out, so a bare
            // `cfg(test)` here referenced a module that does not exist.
            #[cfg(all(test, not(feature = "shuttle")))]
            tests::note_rendered_entry();
            // One ASCII classification serves both sanitizer selection and the
            // fast-path gate: printable/tab-only input is untouched by the
            // multiline sanitizer, so scanning it twice was dead work.
            let class = super::ascii::classify_ascii(entry.message.as_bytes());
            let sanitized: Cow<'_, str> = if class == super::ascii::AsciiClass::Other {
                // Multiline sanitizer, not the plain one: a call-chain entry
                // may carry layout-intended `\n`/`\t`, and `wrap_optimal`
                // below supports hard newlines — the plain variant destroyed
                // them into U+FFFD before wrapping ever saw them.
                super::sanitize_for_terminal_multiline(entry.message.as_ref())
            } else {
                Cow::Borrowed(entry.message.as_ref())
            };
            // `wrap_optimal` measures tabs from column 0.  The entry is
            // actually printed after `indent_cols`; when that is not a tab
            // stop, expand tabs with the real starting column so wrapping
            // cannot under-estimate the rendered width.
            let message: Cow<'_, str> =
                if tab_aligned || class == super::ascii::AsciiClass::Printable {
                    sanitized
                } else {
                    Self::expand_message_tabs(sanitized.as_ref(), tab_width, indent_cols)
                };
            let suffix = match (source, entry.span, idx) {
                (Some(_), Some(span), Some(idx)) if anchor.map_or(true, |a| span.file == a) => {
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
                _ => {
                    // One array read replaces a Fluent label resolution plus a
                    // grapheme-width rescan per entry.
                    let (kind, width) = kind_labels[entry.kind.index()];
                    ChainSuffix::Kind { kind, width }
                }
            };
            let suffix_cols = suffix.cols(file_cols);
            // The suffix is indivisible and rides the last line only, so that
            // line's budget is `avail_full - suffix_cols` while every other
            // line keeps the full measure; the wrap below is the minimal-line
            // single pass (`research/locator-wrap/`, solution.md §2.3).
            // C3: dominant-case fast path — a short printable-ASCII message
            // that fits on one line needs no `wrap_optimal` and no `Vec`.
            // `PrintableOrTab` extends the same path to tab-indented single
            // lines; tabs are measured from the real indent column.
            let fast_width = match class {
                super::ascii::AsciiClass::Printable => Some(message.len()),
                super::ascii::AsciiClass::PrintableOrTab => Self::ascii_tabbed_width_capped(
                    message.as_ref(),
                    indent_cols,
                    tab_width,
                    max_width.saturating_sub(indent_cols.saturating_add(suffix_cols)),
                ),
                // A layout-intended newline means the message is not single-line,
                // so it must go through `wrap_optimal`, which honours hard
                // newlines.  (This arm exists only because the class is now
                // exact: previously a newline classified as `Other` and fell
                // through the same way.)
                super::ascii::AsciiClass::PrintableOrTabNewline
                | super::ascii::AsciiClass::Other => None,
            };
            if fast_width.is_some_and(|width| {
                // `wrap_optimal` trims edge whitespace; decline both space
                // and tab edges so the fast path cannot diverge.  An empty
                // message is safe on the fast path (indent + suffix only).
                !message.starts_with(' ')
                    && !message.ends_with(' ')
                    && !message.starts_with('\t')
                    && !message.ends_with('\t')
                    && width + indent_cols + suffix_cols <= max_width
            }) {
                super::renderers::fmt::push_spaces(out, indent_cols);
                out.push_str(color_on);
                out.push_str(message.as_ref());
                out.push_str(color_off);
                suffix.write_into(out);
                out.push('\n');
            } else {
                // The locator is indivisible and rides the **last** line only,
                // so that line's body budget is `avail_full - suffix_cols`
                // while every other line keeps `avail_full`.
                //
                // Wrapping twice — full measure first, then re-wrapping the
                // overflowing tail at the narrowed measure — is the two-phase Φ
                // of `research/locator-wrap/`.  Both phases are optimal for
                // their own budget and the composition is still unboundedly
                // bad: `avail_full = 2M+1`, `suffix_cols = 2M`, `M` unit words
                // ⇒ `M` lines where 2 suffice (Fact 2, ratio `M/2 → ∞`).
                //
                // `wrap_optimal_reserved` runs the theory's single pass: the
                // last line is the *longest feasible suffix*
                // (`j* = min{ j : width(w_j..w_n) ≤ avail_full - suffix_cols }`)
                // and the prefix is wrapped at the full measure, giving the
                // minimum line count `L(j* - 1) + 1` (solution.md §2.3) with the
                // DP resolving the secondary objective.  Non-final lines
                // therefore never pay for the suffix — the property the old
                // second pass preserved only by accident — and the borrow
                // ordering that pass needed is gone with it.
                let segs = super::wrap::wrap_optimal_reserved(
                    &message,
                    avail_full,
                    tab_width,
                    suffix_cols,
                );
                // The reserved path always yields ≥ 1 segment (an empty message
                // yields one empty segment), so `saturating_sub` cannot hide a
                // dropped row here.
                //
                // Infeasible entries — `suffix_cols >= avail_full`, or a last
                // word wider than the narrowed budget — take policy P1 of
                // solution.md §4: wrap at the full measure and let the locator
                // overflow that row.  It is the minimum-row-count policy.
                // Carrying the suffix on its own continuation row (P2) doubles
                // the row count of a deep chain, the cascade the row budget
                // rejects; truncating it (P3) would break the click target and
                // flicker as the line/column digits change.  The width invariant
                // therefore holds for every entry whose
                // `suffix_cols < avail_full`, which is the regime the contract
                // test pins.
                let last = segs.len().saturating_sub(1);
                for (i, seg) in segs.iter().enumerate() {
                    // Contract: a wrap segment never contains a hard newline.
                    // Three things here depend on it — the indent is pushed once
                    // per segment, `color_on`/`color_off` wrap the whole segment
                    // (an embedded `\n` would leak SGR state across rows), and
                    // `expand_message_tabs` expanded every source line at
                    // `indent_cols`, which is only right if each segment starts
                    // at the indent column.  Silently violated, all three break
                    // at once and rows overrun `max_width`.
                    debug_assert!(
                        !seg.contains('\n'),
                        "wrap segment contains a hard newline: {seg:?}"
                    );
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
                    child.render_entries(env, depth + 1, out, capped, rendered);
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

    thread_local! {
        /// Mirrors `render_entries`' visit count through the hook above.
        static RENDERED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    pub(super) fn note_rendered_entry() {
        RENDERED.with(|c| c.set(c.get() + 1));
    }

    fn take_rendered() -> usize {
        RENDERED.with(|c| c.replace(0))
    }

    /// 1.2's pin: a wrap segment never carries a hard newline, which is what the
    /// indent-once push, the SGR wrapping and the tab pre-expansion all assume.
    /// `Clone` is reachable through the public `with_children`, so a hostile deep
    /// chain must clone without touching the stack — the same reason `Drop` and
    /// `Debug` are hand-written here.  A recursive derive overflowed at a few
    /// thousand levels.
    #[test]
    fn deep_chain_clones_iteratively() {
        let span = Span::new(FileId::DUMMY, 0, 1);
        // Build bottom-up so the *source* is deep too (its `Drop` is iterative).
        let mut chain = CallChain::push_call(span, "leaf");
        for _ in 0..20_000 {
            let entry = ChainEntry::new(ChainEntryKind::Call, Some(span), "frame")
                .with_children(vec![chain]);
            let mut next = CallChain::push_call(span, "frame");
            next.entries[0] = entry;
            chain = next;
        }

        let mut cloned = chain.clone();
        // Same structure: the chain flattens to the same entry count.
        assert_eq!(
            cloned.flattened_entry_count(),
            chain.flattened_entry_count()
        );
        // And the clone is independent (mutating it must not touch the original).
        cloned.entries[0].message = "changed".into();
        assert_ne!(cloned.entries[0].message, chain.entries[0].message);
    }

    /// A branching chain keeps its shape through the worklist clone.
    #[test]
    fn branching_chain_clone_preserves_shape() {
        let span = Span::new(FileId::DUMMY, 0, 1);
        let leaf = |name: &'static str| CallChain::push_call(span, name);
        let mut a = leaf("a");
        a.entries[0] = a.entries[0].clone().with_children(vec![leaf("a1")]);
        let mut b = leaf("b");
        b.entries[0] = b.entries[0].clone();
        let mut root = CallChain::push_call(span, "root");
        root.entries[0] = root.entries[0]
            .clone()
            .with_children(vec![a.clone(), b.clone()]);
        let cloned = root.clone();
        let kids = cloned.entries[0]
            .children
            .as_ref()
            .expect("children cloned");
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].entries[0].message, "a");
        assert_eq!(kids[1].entries[0].message, "b");
        assert!(
            kids[0].entries[0].children.is_some(),
            "the grandchild survived the two-pass clone"
        );
    }

    #[test]
    fn wrap_segments_never_contain_hard_newlines() {
        let corpus = [
            "a\nb",
            "one\ntwo\nthree",
            "line one is long enough to need wrapping at a narrow measure\nshort",
            "\n\n\n",
            "trailing\n",
            "\nleading",
            "word word word word word word\nword word word word word word word word",
            "tab\there\nand\tthere",
        ];
        for msg in corpus {
            for width in [10usize, 20, 33, 80] {
                for tw in [1usize, 4] {
                    let segs = crate::diagnostics::wrap::wrap_optimal_reserved(msg, width, tw, 0);
                    for seg in &segs {
                        assert!(
                            !seg.contains('\n'),
                            "segment {seg:?} of {msg:?} at width {width} carries a newline"
                        );
                    }
                }
            }
        }
    }

    /// 1.3's pin: the capped-width scan and the tab expander implement one
    /// tab-stop rule in two places; this is the differential that keeps them one.
    #[test]
    fn tab_width_formula_matches_the_expander() {
        for tw in [1usize, 2, 4, 8] {
            for start_col in 0..(3 * tw).max(1) {
                let mut buf = String::new();
                for pos in 0..32usize {
                    buf.clear();
                    buf.extend(std::iter::repeat_n('x', pos));
                    buf.push('\t');
                    buf.extend(std::iter::repeat_n('y', 3));
                    let expanded =
                        crate::diagnostics::glyph::expand_tabs_from_col(&buf, tw, start_col);
                    let want = crate::diagnostics::glyph::visible_width(&expanded);
                    // A cap large enough that the scan cannot clamp.
                    let got =
                        CallChain::ascii_tabbed_width_capped(&buf, start_col, tw, usize::MAX / 2)
                            .expect("no clamp can occur with a huge cap");
                    assert_eq!(got, want, "tw={tw} start={start_col} pos={pos} buf={buf:?}");
                }
            }
        }
    }

    #[test]
    fn empty_message_keeps_entry_indent_and_suffix() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(ChainEntryKind::Call, None, ""));
        let out = chain.render(false, None, "");
        let entry_line = out.lines().nth(1).expect("entry line");
        assert!(entry_line.starts_with("    "), "out: {out}");
        assert!(entry_line.contains("called from here"), "out: {out}");
    }

    #[test]
    fn tab_fast_path_matches_wrap_for_tabbed_message() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(ChainEntryKind::Call, None, "a\tb"));
        let out = chain.render_with_index(false, None, "", None, 40, 4);
        assert!(out.lines().nth(1).unwrap().contains("a\tb"), "out: {out}");
        let wrapped = crate::diagnostics::wrap::wrap_optimal("a\tb", 36, 4);
        assert_eq!(wrapped.as_slice(), [Cow::Borrowed("a\tb")]);
    }

    #[test]
    fn internal_double_spaces_match_wrap_fast_path() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(ChainEntryKind::Call, None, "a  b"));
        let out = chain.render_with_index(false, None, "", None, 40, 4);
        assert!(out.lines().nth(1).unwrap().contains("a  b"), "out: {out}");
        let wrapped = crate::diagnostics::wrap::wrap_optimal("a  b", 36, 4);
        assert_eq!(wrapped.as_slice(), [Cow::Borrowed("a  b")]);
    }

    #[test]
    fn edge_tabs_decline_fast_path_without_changing_text() {
        for msg in ["\tfoo", "foo\t"] {
            let mut chain = CallChain::new();
            chain.push(ChainEntry::new(ChainEntryKind::Call, None, msg));
            let out = chain.render_with_index(false, None, "", None, 40, 4);
            let line = out.lines().nth(1).expect("entry line");
            // The wrap path trims edge tabs; the fast path must not emit them
            // raw.  Pin the actual post-wrap text instead of the guard shape.
            assert!(!line.starts_with('\t'), "edge tab leaked: {out:?}");
            assert!(!line.trim_end().ends_with('\t'), "edge tab leaked: {out:?}");
        }
    }

    #[test]
    fn empty_message_fast_path_matches_wrap() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(ChainEntryKind::Call, None, ""));
        let out = chain.render_with_index(false, None, "", None, 40, 4);
        assert_eq!(out.lines().nth(1), Some("    : called from here"));
    }

    /// The locator is indivisible and rides the last row only, so the reserve
    /// must narrow that row and no other (`research/locator-wrap/`): wrapping
    /// at the reduced measure would waste the suffix width on every row.
    #[test]
    fn locator_reserve_narrows_only_the_last_row() {
        // 24 two-letter words: 24*3-1 = 71 columns of body.
        let msg = vec!["aa"; 24].join(" ");
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(ChainEntryKind::Call, None, msg.as_str()));
        let out = chain.render_with_index(false, None, "", None, 40, 4);
        let rows: Vec<&str> = out.lines().skip(1).collect();
        for row in &rows {
            assert!(row.len() <= 40, "row exceeds width: {row:?} in {out:?}");
        }
        // `avail_full` is 36 and `: called from here` is 18 columns, so a row
        // wrapped at the reduced measure would stop at column 22; the first
        // row must use the full measure instead.
        assert!(rows[0].len() > 22, "first row was narrowed: {rows:?}");
    }

    #[test]
    fn edge_spaces_decline_fast_path_without_changing_text() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(ChainEntryKind::Call, None, "  foo  "));
        let out = chain.render_with_index(false, None, "", None, 40, 4);
        let line = out.lines().nth(1).expect("entry line");
        assert!(line.trim_start().starts_with("foo"), "out: {out}");
    }

    #[test]
    fn custom_tab_width_accounts_for_indent_column() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(ChainEntryKind::Call, None, "abcdefg\tX"));
        let out = chain.render_with_index(false, None, "", None, 30, 8);
        for line in out.lines().skip(1) {
            assert!(
                crate::diagnostics::glyph::visible_width_with_tab(line, 8) <= 30,
                "line exceeds width: {line:?}"
            );
        }
    }

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
            &ChainRenderCtx {
                use_color: false,
                source: Some(src),
                file_name: "file.ps",
                idx: Some(&idx),
                max_width: 100,
                tab_width: 4,
                cx: None,
                anchor: Some(FileId::DUMMY),
            },
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
            assert!(
                crate::diagnostics::glyph::visible_width(line) <= 40,
                "line exceeds max_width: {line:?}"
            );
        }
        // The message must wrap into more than one line.
        assert!(lines.len() > 2, "message did not wrap: {out}");
    }

    #[test]
    fn multiline_message_preserves_newlines_and_suffixes_last_line() {
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 0, 3)),
            "line one\n\tline two",
        ));
        let out = chain.render_with_index(false, Some("def main() {}"), "f.ps", None, 40, 4);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines.iter().any(|l| l.contains("line one")), "out: {out}");
        assert!(lines.iter().any(|l| l.contains("line two")), "out: {out}");
        assert!(
            lines.last().unwrap().contains(": f.ps:1:1"),
            "suffix missing from last line: {out}"
        );
    }

    /// Hostile-depth teardown: `Drop` must not recurse.  10k levels is far
    /// past any real chain and would overflow the default test stack with the
    /// derived (structurally recursive) glue — the exact class the render caps
    /// claim to defend.
    #[test]
    fn deep_chain_drops_without_recursion() {
        let span = Span::new(FileId::DUMMY, 0, 1);
        let mut chain = CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), "x")],
            facts: std::sync::OnceLock::new(),
        };
        for _ in 0..10_000 {
            let entry =
                ChainEntry::new(ChainEntryKind::Call, Some(span), "x").with_children(vec![chain]);
            chain = CallChain {
                entries: vec![entry],
                facts: std::sync::OnceLock::new(),
            };
        }
        drop(chain);
    }

    /// A push after a render must invalidate the memo: `entry_count` and
    /// `any_spanned` live in a `OnceLock`, and a stale value would under-reserve
    /// or silently gate hyperlinks wrongly.
    #[test]
    fn push_invalidates_the_facts_memo() {
        let span = Span::new(FileId::DUMMY, 0, 1);
        let mut chain = CallChain::new();
        chain.push(ChainEntry::new(ChainEntryKind::Call, Some(span), "first"));
        let out = chain.render(false, Some("abc\n"), "t.pon");
        assert!(out.contains("first"), "out: {out}");
        assert_eq!(chain.scan().entry_count, 1);
        chain.push(ChainEntry::new(ChainEntryKind::Call, Some(span), "second"));
        let out = chain.render(false, Some("abc\n"), "t.pon");
        assert!(out.contains("second"), "out: {out}");
        assert_eq!(chain.scan().entry_count, 2);
    }

    /// Hostile *wide* chains mirror the deep-chain tests on the other axis: the
    /// reserve stays bounded and truncation emits exactly one marker.
    #[test]
    fn hostile_wide_chain_is_capped_with_one_marker() {
        let span = Span::new(FileId::DUMMY, 0, 1);
        let mut chain = CallChain::new();
        for i in 0..(MAX_RESERVE_ENTRIES + 1) {
            chain.push(ChainEntry::new(
                ChainEntryKind::Call,
                Some(span),
                format!("m{i}"),
            ));
        }
        assert!(chain.scan().entry_count <= MAX_RESERVE_ENTRIES);
        let out = chain.render(false, None, "");
        assert_eq!(out.matches("...").count(), 1, "one truncation marker");
        assert!(out.len() < (8 << 20), "bounded output, got {}", out.len());
    }

    /// The reserve estimate (`compute_facts`) and the renderer walk the tree
    /// independently; they must agree on how many entries survive the caps, or
    /// the pre-sizing silently desynchronises from the output.
    #[test]
    fn facts_entry_count_matches_rendered_entries() {
        let span = Span::new(FileId::DUMMY, 0, 1);
        for &(width, depth) in &[(1usize, 0usize), (3, 0), (5, 1), (7, 2), (16, 3), (40, 5)] {
            let mut chain = CallChain::new();
            for w in 0..width {
                let mut node = CallChain::new();
                node.push(ChainEntry::new(
                    ChainEntryKind::Call,
                    Some(span),
                    format!("m{w}"),
                ));
                for _ in 0..depth {
                    let mut parent = CallChain::new();
                    parent.push(
                        ChainEntry::new(ChainEntryKind::Call, Some(span), "n")
                            .with_children(vec![node]),
                    );
                    node = parent;
                }
                chain.entries.extend(node.entries.drain(..));
            }
            take_rendered();
            let _ = chain.render(false, None, "");
            let rendered = take_rendered();
            assert_eq!(
                chain.scan().entry_count,
                rendered,
                "width {width} depth {depth}"
            );
        }
    }

    /// The derived `Debug` recursed one frame per nesting level — the same
    /// hazard the iterative `Drop` above removes, but reachable through any
    /// `unwrap` panic message or `assert_eq!` on a chain.
    #[test]
    fn debug_of_a_deep_chain_is_bounded() {
        let span = Span::new(FileId::DUMMY, 0, 1);
        let mut chain = CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), "x")],
            facts: std::sync::OnceLock::new(),
        };
        for _ in 0..10_000 {
            let entry =
                ChainEntry::new(ChainEntryKind::Call, Some(span), "x").with_children(vec![chain]);
            chain = CallChain {
                entries: vec![entry],
                facts: std::sync::OnceLock::new(),
            };
        }
        let rendered = format!("{chain:?}");
        assert!(
            rendered.len() < 4_096,
            "Debug output grew with depth: {} bytes",
            rendered.len()
        );
        assert!(rendered.contains("nested_chains: 1"), "{rendered}");
        drop(chain);

        // Size axis: one entry carrying a multi-megabyte message must not turn
        // `format!("{chain:?}")` into megabytes of text (reachable from every
        // `unwrap` panic and `assert_eq!` failure).
        let huge = "m".repeat(4 * 1024 * 1024);
        let chain = CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), &huge)],
            facts: std::sync::OnceLock::new(),
        };
        let rendered = format!("{chain:?}");
        assert!(
            rendered.len() < 4_096,
            "Debug output grew with message size: {} bytes",
            rendered.len()
        );
        assert!(rendered.contains("+4194240 bytes"), "{rendered}");
    }

    /// A custom `RenderCtx::file_url` can hand back raw UTF-8; the OSC 8
    /// payload must still be an ASCII URL, so those bytes are percent-encoded
    /// rather than dropping the hyperlink.
    #[test]
    fn non_ascii_url_bytes_are_percent_encoded() {
        assert_eq!(
            percent_encode_high_bytes("file:///tmp/café.rs"),
            "file:///tmp/caf%C3%A9.rs"
        );
        assert_eq!(
            percent_encode_high_bytes("file:///tmp/a.rs"),
            "file:///tmp/a.rs"
        );
    }

    #[test]
    fn deep_chain_is_capped_consistently() {
        // Build 70 nested levels: the renderer/flatteners cap at
        // `MAX_CHAIN_DEPTH` (64) and must agree, emit exactly one `...`
        // marker, and never panic.
        let span = Span::new(FileId::DUMMY, 0, 3);
        let mut inner = CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), "x")],
            facts: std::sync::OnceLock::new(),
        };
        // Single-character messages: at depth ≥ 64 the indent exceeds the
        // measure, and a multi-character word would be split one grapheme per
        // line (correct, but it makes the line-count cap assertion useless).
        for _ in 0..70 {
            let entry =
                ChainEntry::new(ChainEntryKind::Call, Some(span), "x").with_children(vec![inner]);
            inner = CallChain {
                entries: vec![entry],
                facts: std::sync::OnceLock::new(),
            };
        }
        let out = inner.render(false, Some("def main() {}"), "f.ps");
        assert_eq!(
            out.matches("...").count(),
            1,
            "exactly one cap marker expected: {out}"
        );
        assert!(
            out.lines().count() <= MAX_CHAIN_DEPTH + 4,
            "cap not applied"
        );
        // The counting traversal must not under-reserve past the cap either:
        // `flattened_entry_count` stops at the same depth as the renderer.
        assert!(inner.flattened_entry_count() >= MAX_CHAIN_DEPTH);
    }

    #[test]
    fn branching_deep_chain_emits_one_marker() {
        let span = Span::new(FileId::DUMMY, 0, 3);
        let leaf = || CallChain {
            entries: vec![ChainEntry::new(ChainEntryKind::Call, Some(span), "x")],
            facts: std::sync::OnceLock::new(),
        };
        // Two identical linear branches that both cross the cap.  The old
        // recursion emitted one `...` per branch; the contract is one marker
        // per render.
        let deep = || {
            let mut chain = leaf();
            for _ in 0..=MAX_CHAIN_DEPTH {
                chain = CallChain {
                    entries: vec![
                        ChainEntry::new(ChainEntryKind::Call, Some(span), "x")
                            .with_children(vec![chain]),
                    ],
                    facts: std::sync::OnceLock::new(),
                };
            }
            chain
        };
        let branch = CallChain {
            entries: vec![
                ChainEntry::new(ChainEntryKind::Call, Some(span), "root")
                    .with_children(vec![deep(), deep()]),
            ],
            facts: std::sync::OnceLock::new(),
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
            facts: std::sync::OnceLock::new(),
        };
        let parent = ChainEntry::new(
            ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 0, 3)),
            "parent",
        )
        .with_children(vec![leaf]);
        let chain = CallChain {
            entries: vec![parent],
            facts: std::sync::OnceLock::new(),
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
                    assert_eq!(visible_width(&out), s.cols(0), "line-col suffix {out:?}");
                }
            }
        }
        for kind in [ChainEntryKind::Call, ChainEntryKind::Inlined] {
            let s = ChainSuffix::Kind {
                kind: kind.as_static_str(),
                width: kind.label_width(),
            };
            let mut out = String::new();
            s.write_into(&mut out);
            assert_eq!(visible_width(&out), s.cols(0));
        }
    }
}
